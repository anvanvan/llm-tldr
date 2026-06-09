"""Regression test: test-model-server parent-death watchdog — Bug A strengthening.

ROOT CAUSE RECAP (Bug A / F-1, confirmed):
    ensure_server() spawns ``python -m tldr.model_server`` with
    ``start_new_session=True``, detaching the child as a session leader the
    kernel will never reap. On an ABNORMAL pytest exit (SIGKILL / crash / outer
    loop) the only reaper — ``_reap_model_server()`` from ``pytest_sessionfinish``
    (tests/conftest.py) — never runs, so the test model server survives as an
    orphan holding its socket (and, once it has embedded, GPU memory).

LANDED FIX (session-START sweep):
    ``_sweep_leaked_test_servers()`` in tests/conftest.py reaps a prior session's
    orphan on the NEXT pytest run. But that is a NEXT-run cleanup: the orphan
    STILL survives in the window immediately after the parent dies, before any
    next pytest run — exactly the window in which it can hold the GPU and wedge
    Metal (PRIMARY / EDGE-1 / EDGE-5 live-verification FAILED here).

NEW BEHAVIOR UNDER TEST (the RED -> GREEN seam — NOT implemented yet):
    A TEST model server, told the PID of its spawning parent via the env var
    ``TLDR_MODEL_SERVER_PARENT_PID``, runs a daemon watchdog thread that polls
    ``os.kill(parent_pid, 0)`` (ESRCH => parent dead) every ~1-2s and, on parent
    death, unlinks its own socket/.pid/.lock and ``os._exit()``s WITHOUT running
    MPS teardown (preserving the commit-60bce6d no-MPS-teardown-on-exit
    invariant). The result: NO orphan persists between runs — the server
    self-exits PROMPTLY (<= ~3-4s) when its spawning pytest process dies.

    Intended GREEN wiring (for this test to target — do NOT implement here):
      - tests/conftest.py (pytest_configure) sets
        ``TLDR_MODEL_SERVER_PARENT_PID = str(os.getpid())`` alongside the existing
        ``TLDR_MODEL_SERVER_SOCKET`` it already sets; the test server inherits it.
      - tldr/model_server/server.py (or __main__.py): at startup, IF that env is
        set, start the watchdog thread described above.

CRITICAL SAFETY CONSTRAINT (designed into this test):
    The watchdog keys on the SPECIFIC original parent PID from the env var —
    NEVER on ``PPID == 1``. The LEGIT shared production server runs with PPID 1
    BY DESIGN (intentionally detached to outlive its spawning daemon — the
    shared-server architecture, commits 0dcc6d2 / 219b29c8). A "exit when
    PPID==1" watchdog would instantly kill the production shared server. The
    second test below GUARDS that regression: a server started WITHOUT
    ``TLDR_MODEL_SERVER_PARENT_PID`` (the shared-server config), then reparented
    to PID 1, must NOT self-exit.

WHAT THIS TEST ASSERTS (deterministic, CI-safe, no GPU / model load required —
the server is never sent an embed, so the BAAI model is never loaded):
    1. Spawn a real, killable SENTINEL parent process.
    2. Spawn a test model server on an isolated /tmp/tldr-test-msrv-<uid>-<pid>.sock
       with TLDR_MODEL_SERVER_PARENT_PID pointing at the sentinel. Confirm alive.
    3. SIGKILL the sentinel.
    4. Assert the server SELF-EXITS within a bounded window (<= 5s).
       On CURRENT code (no watchdog) the server stays alive => RED for the
       correct reason: the parent-death watchdog is absent.

RED rationale:
    Current code has no parent-death watchdog (verified: zero hits for
    TLDR_MODEL_SERVER_PARENT_PID / watchdog / parent_pid in tldr/ and tests/).
    A server spawned with start_new_session=True is reparented to init when its
    sentinel dies and survives indefinitely => the self-exit assertion FAILS.

CI-SAFETY / SELF-CLEANING:
    - Isolated per-test socket named after THIS test process's pid, off the
      production per-user socket.
    - The server is spawned WITHOUT an embed request, so no MPS/GPU model loads.
    - The finally block sends a LITERAL SIGKILL to every PID this test spawned
      (server AND sentinel) by their actual Popen pids and unlinks all socket
      artifacts — never leaving a real orphan regardless of pass/fail.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UID = os.getuid() if hasattr(os, "getuid") else os.getpid()


def _watchdog_sock_path() -> str:
    # Distinct suffix so it can never collide with the session-owned socket
    # (tldr-test-msrv-<uid>-<pytest-pid>.sock) or the sweep test's socket.
    return f"/tmp/tldr-test-msrv-{_UID}-watchdog-{os.getpid()}.sock"


def _spawn_sentinel() -> subprocess.Popen:
    """A real, killable sentinel process standing in for the spawning pytest.

    It simply sleeps; the watchdog's job is to notice when THIS pid dies.
    """
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _spawn_server_on(sock_path: str, parent_pid: int) -> subprocess.Popen:
    """Spawn a real model server on *sock_path*, telling it its parent's PID.

    Mirrors the exact spawn path in ensure_server(): start_new_session=True so
    the child is a session leader that survives its (now detached) parent's
    death — which is precisely why a watchdog, not the kernel, must reap it.
    The TLDR_MODEL_SERVER_PARENT_PID env is what the GREEN watchdog reads.
    """
    env = dict(os.environ)
    env["TLDR_MODEL_SERVER_SOCKET"] = sock_path
    env["TLDR_MODEL_SERVER_PARENT_PID"] = str(parent_pid)
    # Long idle so the server does not auto-exit before the test observes it.
    env["TLDR_MODEL_SERVER_IDLE_SECS"] = "600"
    return subprocess.Popen(
        [sys.executable, "-m", "tldr.model_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def _wait_sock(path: str, timeout: float = 12.0) -> bool:
    """Poll until the socket file exists (server has bound it) or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _hard_kill(pid: "int | None", grace: float = 2.0) -> None:
    """Literal SIGKILL a pid and wait briefly for it to disappear. Never raises."""
    if pid is None or not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return
    deadline = time.time() + grace
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------

class TestParentDeathWatchdog:
    """Bug A strengthening: a test server self-exits promptly when its parent dies.

    RED: current code has no parent-death watchdog, so a detached test server
    spawned with start_new_session=True survives its sentinel parent's death
    indefinitely => the self-exit assertion FAILS for the correct reason.
    GREEN: the to-be-added watchdog (keyed on TLDR_MODEL_SERVER_PARENT_PID)
    unlinks its socket trio and os._exit()s within a bounded window.
    """

    def test_test_server_self_exits_when_spawning_parent_dies(self):
        """Killing the spawning parent must make the test server self-exit <= 5s.

        Steps
        -----
        1. Spawn a real sentinel parent (a sleeping subprocess).
        2. Spawn a test model server on an isolated socket with
           TLDR_MODEL_SERVER_PARENT_PID = sentinel.pid. Confirm it bound the
           socket and is alive.
        3. SIGKILL the sentinel (direct literal kill).
        4. Assert the server PID becomes dead within 5s — the watchdog noticed
           the parent's death and self-exited. On CURRENT code there is no
           watchdog, so the server stays alive => RED.
        5. Assert the server unlinked its own socket on the self-exit path.

        Self-cleaning: the finally block literally SIGKILLs both the server and
        the sentinel by their actual Popen pids and unlinks all socket
        artifacts, regardless of pass/fail.
        """
        sock_path = _watchdog_sock_path()
        pid_path = sock_path + ".pid"
        lock_path = sock_path + ".lock"

        sentinel: "subprocess.Popen | None" = None
        server_proc: "subprocess.Popen | None" = None
        sentinel_pid: "int | None" = None
        server_pid: "int | None" = None

        try:
            # Step 1 — real, killable sentinel standing in for the spawning pytest.
            sentinel = _spawn_sentinel()
            sentinel_pid = sentinel.pid
            assert _pid_alive(sentinel_pid), (
                f"Sentinel parent PID {sentinel_pid} failed to start. "
                "Cannot stage the watchdog scenario."
            )

            # Step 2 — spawn the test server bound to the sentinel as its parent.
            server_proc = _spawn_server_on(sock_path, sentinel_pid)
            server_pid = server_proc.pid

            assert _wait_sock(sock_path, timeout=12.0), (
                f"Test server (PID {server_pid}) never bound {sock_path} within "
                "12s. Cannot stage the watchdog scenario."
            )
            assert _pid_alive(server_pid), (
                f"Test server PID {server_pid} died before the sentinel was "
                "killed. Test precondition failed."
            )

            # Step 3 — kill the spawning parent (direct literal SIGKILL). This is
            # the event the watchdog must detect (os.kill(parent_pid, 0) -> ESRCH).
            os.kill(sentinel_pid, signal.SIGKILL)
            # Reap the sentinel zombie so its PID truly frees up — the watchdog
            # polls os.kill(parent_pid, 0), which must raise ESRCH once reaped.
            try:
                sentinel.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            assert not _pid_alive(sentinel_pid), (
                f"Sentinel parent PID {sentinel_pid} is still alive after "
                "SIGKILL+reap; cannot drive the watchdog."
            )

            # Step 4 — the server must self-exit within a bounded window.
            # GREEN: watchdog polls every ~1-2s and os._exit()s on parent death.
            # CURRENT (RED): no watchdog => the detached server survives forever.
            window = 5.0
            deadline = time.time() + window
            server_gone = False
            while time.time() < deadline:
                if not _pid_alive(server_pid):
                    server_gone = True
                    break
                time.sleep(0.1)

            assert server_gone, (
                f"Test server (PID {server_pid}) is STILL ALIVE {window}s after "
                f"its spawning parent (sentinel PID {sentinel_pid}) was killed. "
                "A TEST model server given TLDR_MODEL_SERVER_PARENT_PID must run a "
                "watchdog that polls os.kill(parent_pid, 0) and, on the parent's "
                "death (ESRCH), self-exits (os._exit, NO MPS teardown) so no "
                "orphan persists between runs holding the GPU. The watchdog is "
                f"absent on current code. Socket: {sock_path}"
            )

            # Step 5 — the self-exit path must unlink the server's own socket so
            # the next run finds no stale socket trio.
            assert not os.path.exists(sock_path), (
                f"Test server self-exited but left its socket {sock_path} behind. "
                "The watchdog's self-exit path must unlink its socket/.pid/.lock."
            )

        finally:
            # Unconditional cleanup — never leave a real orphan behind. Kill by
            # the ACTUAL spawned pids (server AND sentinel), then unlink artifacts.
            _hard_kill(server_pid)
            _hard_kill(sentinel_pid)

            if server_proc is not None and server_proc.poll() is None:
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if sentinel is not None and sentinel.poll() is None:
                try:
                    sentinel.kill()
                    sentinel.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass

            for p in (sock_path, pid_path, lock_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def test_server_without_parent_pid_env_does_not_self_exit(self):
        """A server started WITHOUT TLDR_MODEL_SERVER_PARENT_PID must NOT self-exit.

        SAFETY INVARIANT: the legit shared production server runs detached at
        PPID 1 by design and does NOT set TLDR_MODEL_SERVER_PARENT_PID. If the
        watchdog engaged without that env var it would kill the production server.

        This test guards that regression: spawn a server without the env var,
        observe it stays alive across a 3s window even after its spawning process
        (this test) is still running (simulating the "reparented to PID 1" case),
        and assert it does NOT self-exit.

        Self-cleaning: the finally block literally SIGKILLs the spawned server
        by its actual Popen pid and unlinks all socket artifacts.
        """
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        sock_path = f"/tmp/tldr-test-msrv-{uid}-no-watchdog-{os.getpid()}.sock"
        pid_path = sock_path + ".pid"
        lock_path = sock_path + ".lock"

        server_proc: "subprocess.Popen | None" = None
        server_pid: "int | None" = None

        try:
            # Spawn the server WITHOUT TLDR_MODEL_SERVER_PARENT_PID — mirrors the
            # shared production server configuration.
            env = dict(os.environ)
            env["TLDR_MODEL_SERVER_SOCKET"] = sock_path
            env["TLDR_MODEL_SERVER_IDLE_SECS"] = "600"
            # Explicitly ensure the watchdog env is NOT present.
            env.pop("TLDR_MODEL_SERVER_PARENT_PID", None)

            server_proc = subprocess.Popen(
                [sys.executable, "-m", "tldr.model_server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            server_pid = server_proc.pid

            assert _wait_sock(sock_path, timeout=12.0), (
                f"Server (PID {server_pid}) never bound {sock_path} within 12s. "
                "Cannot verify the no-watchdog invariant."
            )
            assert _pid_alive(server_pid), (
                f"Server PID {server_pid} died before the observation window. "
                "Test precondition failed."
            )

            # Observe across a 3s window — the server must stay alive throughout.
            # A stray watchdog would fire within ~1-2s and kill it, causing this
            # assertion to fail. The production shared server (PPID 1, no env var)
            # must never be killed by a watchdog.
            window = 3.0
            deadline = time.time() + window
            died_early = False
            while time.time() < deadline:
                if not _pid_alive(server_pid):
                    died_early = True
                    break
                time.sleep(0.1)

            assert not died_early, (
                f"Server (PID {server_pid}) self-exited within {window}s even "
                "though TLDR_MODEL_SERVER_PARENT_PID was NOT set. A watchdog "
                "must ONLY engage when that env var is explicitly provided. "
                "The shared production server (PPID 1, no env var) must never "
                "be killed by a stray watchdog. "
                f"Socket: {sock_path}"
            )

        finally:
            _hard_kill(server_pid)

            if server_proc is not None and server_proc.poll() is None:
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass

            for p in (sock_path, pid_path, lock_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
