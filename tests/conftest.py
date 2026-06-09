"""Shared pytest fixtures for the tldr test suite."""
import contextlib
import glob
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tldr.model_server.server import _PARENT_PID_ENV, _pid_alive


# ---------------------------------------------------------------------------
# Shared test helpers — used by multiple test files
# ---------------------------------------------------------------------------

_FAKE_MODEL_DIM = 4


def make_fake_model() -> MagicMock:
    """Deterministic fake embedder: L2-normalised np.ones vectors.

    Returns a MagicMock that encodes any text sequence as normalized
    all-ones vectors with dimension _FAKE_MODEL_DIM. Used by tests
    that need a model but do not depend on embedding semantics.
    """
    mock_model = MagicMock()

    def fake_encode(texts, batch_size=128, normalize_embeddings=True,
                    show_progress_bar=False):
        n = len(texts) if isinstance(texts, list) else 1
        vecs = np.ones((n, _FAKE_MODEL_DIM), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode.side_effect = fake_encode
    return mock_model


def read_semantic_metadata(project_root: Path) -> dict:
    """Read semantic index metadata.json from a project's cache.

    Args:
        project_root: Root directory of the project (must have .tldr/cache/ already indexed).

    Returns:
        Parsed metadata dict with 'units' and other index metadata.

    Raises:
        AssertionError if metadata.json is missing.
    """
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing: {meta_path}"
    return json.loads(meta_path.read_text())


def find_units_by_name(meta: dict, name: str) -> list:
    """Return all units whose bare 'name' field matches."""
    return [u for u in meta["units"] if u.get("name") == name]


def validate_incr_full_equivalence(meta_incr: dict, meta_full: dict) -> list:
    """Return per-unit divergence messages (empty list == equal).

    Keys by (file, name) pairs to handle multi-file repos where unit names may
    collide across files. For shared units, sorted(calls) and sorted(called_by)
    must match exactly.
    """
    def _key(u: dict) -> tuple:
        return (u.get("file", ""), u.get("name", ""))

    incr_by_key = {_key(u): u for u in meta_incr["units"]}
    full_by_key = {_key(u): u for u in meta_full["units"]}
    problems = []

    only_incr = set(incr_by_key) - set(full_by_key)
    only_full = set(full_by_key) - set(incr_by_key)
    if only_incr:
        problems.append(f"Units only in incremental: {sorted(str(k) for k in only_incr)}")
    if only_full:
        problems.append(f"Units only in --full:      {sorted(str(k) for k in only_full)}")

    for k in set(incr_by_key) & set(full_by_key):
        u_i = incr_by_key[k]
        u_f = full_by_key[k]
        i_calls = sorted(u_i.get("calls") or [])
        f_calls = sorted(u_f.get("calls") or [])
        i_cb = sorted(u_i.get("called_by") or [])
        f_cb = sorted(u_f.get("called_by") or [])
        if i_calls != f_calls:
            problems.append(
                f"{k}.calls: incremental={i_calls!r} != full={f_calls!r}"
            )
        if i_cb != f_cb:
            problems.append(
                f"{k}.called_by: incremental={i_cb!r} != full={f_cb!r}"
            )

    return problems


@pytest.fixture
def write_temp_file():
    """Factory fixture to write a file to a temporary path.

    Returns a callable with signature ``(path: Path, content: str) -> Path``
    that creates parent directories as needed, writes ``content``, and returns
    the path so callers can use the result directly.
    """
    def _write(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    return _write


# ---------------------------------------------------------------------------
# E2E marker: opt-in only (--run-e2e) to avoid slow real-model tests in CI
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run E2E tests that load the real embedding model (slow, fan-loud).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests using the real embedding model (opt-in via --run-e2e).",
    )
    # Keep pytest's tmp_path short. On macOS the default basetemp lives under
    # /private/var/folders/<...>/pytest-of-<user>/pytest-N/<test-name>/ which,
    # combined with a socket filename, exceeds the AF_UNIX sun_path limit
    # (~104 bytes) and makes socket.bind() in the model-server / embedding-backend
    # tests fail with "AF_UNIX path too long". Relocating basetemp to a short
    # path under the system temp dir only moves temp files — no test behaviour
    # changes — and lets those Unix-socket tests bind successfully.
    # Session-START sweep: reap any orphan test model server left behind by a
    # PRIOR session that exited abnormally (SIGKILL / crash / outer loop), whose
    # pytest_sessionfinish reaper never ran. This runs BEFORE we claim our own
    # isolated socket below, so it can never reap the socket THIS session owns
    # (its filename carries the current pid). Without it, one detached orphan
    # model server + socket trio accumulates per abnormal run.
    _sweep_leaked_test_servers()

    # Route the whole session off the real per-user model-server socket. Set the
    # env BEFORE any test runs (ensure.py reads it at call time) so that any test
    # touching the real ensure_server()/spawn path binds a throwaway socket
    # instead of the production one. setdefault: respect a deliberate outer
    # override (and then we never reap it — see pytest_sessionfinish).
    global _OWNED_MODEL_SERVER_SOCKET
    if "TLDR_MODEL_SERVER_SOCKET" not in os.environ:
        sock = _isolated_model_server_socket()
        os.environ["TLDR_MODEL_SERVER_SOCKET"] = sock
        _OWNED_MODEL_SERVER_SOCKET = sock
        # Parent-death watchdog wiring: inject only when WE own the socket — an
        # explicit opt-in for the conftest-managed test-server path, not a global
        # pollution of every spawned process. Any server spawned via ensure_server()
        # inherits os.environ and will receive this env var, starting a daemon
        # watchdog that self-exits when THIS pytest process dies — so no orphan
        # persists in the window between an abnormal exit and the next sweep.
        # Keyed on the SPECIFIC pid, never PPID==1: the production shared server
        # (env unset) never starts the watchdog and never self-exits.
        os.environ[_PARENT_PID_ENV] = str(os.getpid())

    # Auto-reap exited children so a self-exited test model server (the watchdog
    # path above) does NOT linger as an un-waited zombie. The watchdog reacts to
    # the death of the SPECIFIC parent PID it was told, which in a test is a
    # short-lived sentinel — but the server's OS parent stays the (still-alive)
    # pytest process, so without auto-reaping its os._exit would leave a zombie
    # that os.kill(pid, 0) still reports as alive, masking the prompt self-exit.
    # SIGCHLD=SIG_IGN makes the kernel reap such children immediately, so a
    # self-exited server's PID becomes truly gone (ESRCH). Python's subprocess is
    # explicitly resilient to this (Popen.wait()/poll() treat the resulting
    # ECHILD as a clean exit), so existing subprocess-driven tests still work.
    if hasattr(signal, "SIGCHLD"):
        try:
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        except (ValueError, OSError):
            # Not on the main thread (e.g. nested runner) — skip; the
            # session-start sweep still reaps any orphan on the next run.
            pass

    if config.option.basetemp is None:
        # Prefer the genuinely short "/tmp" over macOS's long
        # /private/var/folders/<...>/T base (already ~48 bytes, which alone
        # blows the AF_UNIX budget once a pytest test-name dir + socket name
        # are appended). Fall back to the system temp dir if /tmp is unusable.
        candidate = "/tmp" if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK) else tempfile.gettempdir()
        short_base = os.path.join(candidate, f"tldr-pt-{os.getuid() if hasattr(os, 'getuid') else os.getpid()}")
        os.makedirs(short_base, exist_ok=True)
        config.option.basetemp = short_base


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e", default=False):
        return  # opted in — run all e2e tests normally
    skip_e2e = pytest.mark.skip(reason="opt-in: pass --run-e2e to run E2E tests")
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip_e2e)


# ---------------------------------------------------------------------------
# Session-wide model-server socket isolation + reap
# ---------------------------------------------------------------------------
#
# The shared embedding model server binds a per-USER canonical Unix socket
# (tldr/daemon/ensure.py:_model_server_socket_path -> tldr-model-server-<uid>.sock).
# Several tests exercise the real ensure_server()/spawn path. Without isolation
# they spawn a real ~1 GB model server on the PRODUCTION per-user socket, and
# because spawned servers detach into their own session (start_new_session=True),
# per-test teardown cannot reliably reap them — leaked servers then accumulate
# across full-suite runs. Pinning TLDR_MODEL_SERVER_SOCKET to a throwaway
# per-session path for the WHOLE pytest session routes every such spawn off the
# real socket; the session-finish reaper guarantees nothing survives the run.
#
# We only ever touch a socket we created: if the environment already carries a
# TLDR_MODEL_SERVER_SOCKET (deliberate override / outer runner), we leave it and
# its server alone.

_OWNED_MODEL_SERVER_SOCKET: "str | None" = None


def _socket_candidate_path() -> "tuple[str, int]":
    """Return (base_dir, uid) for socket paths.

    Prefers the genuinely short ``/tmp`` over macOS's long per-user temp dir so
    the resulting path stays within the AF_UNIX ``sun_path`` budget.
    """
    candidate = "/tmp" if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK) else tempfile.gettempdir()
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return candidate, uid


def _isolated_model_server_socket() -> str:
    """A short, per-session socket path off the real per-user canonical path.

    Kept short (prefer ``/tmp``) so ``socket.bind`` stays within the AF_UNIX
    ``sun_path`` budget — see the basetemp note in ``pytest_configure``.
    """
    candidate, uid = _socket_candidate_path()
    return os.path.join(candidate, f"tldr-test-msrv-{uid}-{os.getpid()}.sock")


def _reap_zombie_child(pid: int) -> bool:
    """Non-blocking ``waitpid`` to clear ``pid`` if it is a child of this process.

    A model server we (or a sibling spawn within the SAME pytest process) launched
    becomes a zombie once it dies, because its parent is still alive and has not
    yet reaped it. While zombied, ``os.kill(pid, 0)`` still SUCCEEDS — the PID
    lingers in the table — so naive liveness checks misread a dead orphan as alive.
    Reaping it here clears the zombie so liveness checks report it dead.

    For a TRUE cross-session orphan (its spawning session is gone) the process has
    been re-parented to init, which reaps it for us — ``waitpid`` then raises
    ``ECHILD`` and we treat that as "not our child, nothing to reap". Never raises.

    Returns True if the child was reaped (now truly gone), False otherwise.
    """
    if not hasattr(os, "waitpid"):
        return False
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return False
    return reaped == pid


def _reap_model_server(socket_path: str) -> None:
    """Best-effort terminate any model server on ``socket_path`` and clean files.

    Reads ``{socket}.pid`` and SIGTERM → 5s grace → SIGKILL the process.
    Zombie-reaps via non-blocking waitpid() to clear stalled children so
    they do not linger as unkillable PIDs. Then unlinks the socket, pid
    sidecar, and lock. (The server's own _reap_orphan() does not zombie-reap;
    this conftest reaper is the cleanup path that must handle them.)
    Never raises.
    """
    pid_path = socket_path + ".pid"
    pid: int | None = None
    try:
        pid = int(Path(pid_path).read_text().strip())
    except (OSError, ValueError):
        pass  # already gone / not ours

    if pid is not None and hasattr(os, "kill"):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            alive = False  # already gone / not ours
        else:
            # COUPLING: this 5.0s SIGTERM→SIGKILL window must remain strictly
            # GREATER than tldr/model_server/server.py _DRAIN_TIMEOUT (3.0s) so
            # the server can finish draining its embed queue before we escalate
            # to SIGKILL. If this window is ever reduced below _DRAIN_TIMEOUT
            # the server will be SIGKILLed mid-Metal-op, wedging the GPU.
            deadline = time.time() + 5.0
            alive = True
            while time.time() < deadline:
                # Reap a child zombie first: a server that exited gracefully under
                # SIGTERM is a zombie (os.kill(pid,0) still succeeds) until we
                # waitpid it — without this we would burn the full 5s grace.
                if _reap_zombie_child(pid):
                    alive = False
                    break
                try:
                    os.kill(pid, 0)
                except OSError:
                    alive = False
                    break
                time.sleep(0.1)
            if alive:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
                # Clear the resulting zombie if the process is our child, so the
                # PID stops resolving as alive to os.kill(pid, 0).
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    if _reap_zombie_child(pid):
                        break
                    time.sleep(0.05)

    for path in (socket_path, pid_path, socket_path + ".lock"):
        with contextlib.suppress(OSError):
            os.unlink(path)


def _test_model_server_socket_glob() -> str:
    """Glob pattern matching every per-session isolated test model-server socket.

    Mirrors ``_isolated_model_server_socket`` but with the owning-pytest PID
    replaced by a ``*`` wildcard so a session can discover sockets left behind
    by ANY prior session — including ones killed abnormally (SIGKILL / crash /
    loop) whose ``pytest_sessionfinish`` never ran to reap them.
    """
    candidate, uid = _socket_candidate_path()
    return os.path.join(candidate, f"tldr-test-msrv-{uid}-*.sock")


def _owning_pytest_pid_from_socket(socket_path: str) -> "int | None":
    """Extract the owning-pytest PID encoded in an isolated socket's basename.

    The basename is ``tldr-test-msrv-<uid>-<pytest-pid>.sock``; return the
    trailing ``<pytest-pid>`` as an int, or ``None`` if the name does not parse.
    """
    name = os.path.basename(socket_path)
    if not name.endswith(".sock"):
        return None
    stem = name[: -len(".sock")]
    last = stem.rsplit("-", 1)[-1]
    try:
        return int(last)
    except ValueError:
        return None


def _sweep_leaked_test_servers() -> None:
    """Session-START sweep: reap orphan test model servers from dead prior sessions.

    The per-session reaper (:func:`pytest_sessionfinish`) only fires on a clean
    pytest exit, so an abnormally-killed session (SIGKILL / crash / outer loop)
    leaves its detached model server alive plus a stale ``tldr-test-msrv-<uid>-
    <pytest-pid>.sock`` socket trio that nothing ever cleans up — one orphan
    accumulates per abnormal run.

    Run at session start (from :func:`pytest_configure`), this sweep globs every
    ``tldr-test-msrv-<uid>-*.sock`` and reaps each whose owning pytest PID (the
    one embedded in the socket filename) is no longer THIS process and is not a
    live process — i.e. the session that spawned it is gone. It never touches the
    socket THIS session owns (its filename carries the current pid), so it is
    safe to run before/alongside our own server. Reaping reuses
    :func:`_reap_model_server` (SIGTERM → 5s grace → SIGKILL, then unlink the
    socket, ``.pid`` and ``.lock``). Never raises.
    """
    my_pid = os.getpid()
    for socket_path in glob.glob(_test_model_server_socket_glob()):
        owner_pid = _owning_pytest_pid_from_socket(socket_path)
        # Leave the socket THIS session owns alone — that is reaped at finish.
        if owner_pid == my_pid:
            continue
        # An orphan: its owning pytest session is dead (or the name is
        # unparseable, which means no live session can claim it either way).
        if owner_pid is not None and _pid_alive(owner_pid):
            # The owning pytest session is still running — not our orphan.
            continue
        _reap_model_server(socket_path)


def pytest_sessionfinish(session, exitstatus):
    """Reap any model server this session spawned on its isolated socket."""
    if _OWNED_MODEL_SERVER_SOCKET:
        _reap_model_server(_OWNED_MODEL_SERVER_SOCKET)
