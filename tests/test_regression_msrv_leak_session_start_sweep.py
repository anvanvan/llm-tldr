"""Regression test: test-model-server leak — Bug A (F-1 confirmed).

ROOT CAUSE (verified, 95% confidence):
    ensure_server() spawns `python -m tldr.model_server` with
    start_new_session=True, making it a detached session leader the kernel will
    never reap. The SOLE cleanup path is _reap_model_server() called from
    pytest_sessionfinish (tests/conftest.py:245-248), which does NOT fire on
    abnormal pytest exit (SIGKILL / crash / loop).  Per-PID socket naming
    (conftest.py:204 → tldr-test-msrv-<uid>-<pytest-pid>.sock) structurally
    disables _superseded() self-eviction (server.py:285) because the orphan's
    socket is never rebound.  Net: one orphaned server + socket trio per abnormal
    run, accumulating indefinitely.

FIX SEAM (not implemented yet — GREEN phase):
    A session-start sweep invoked from pytest_configure (or an early
    pytest_sessionstart hook) that globs /tmp/tldr-test-msrv-<uid>-*.sock and
    reaps any whose .pid sidecar names a dead/non-tldr process.  The sweep runs
    at session START so it executes even when the PRIOR session was SIGKILL-ed.

WHAT THIS TEST ASSERTS:
    The to-be-added sweep function (_sweep_leaked_test_servers, or equivalent
    called from pytest_configure) reaps a pre-staged orphan server.

    Test flow
    ---------
    1. Spawn a real model-server bound to a fake "prior-session" socket path
       (tldr-test-msrv-<uid>-<fake-dead-pid>.sock with a .pid sidecar).
    2. Verify the server is alive and the socket exists.
    3. Simulate a dead parent by using a clearly-dead PID in the socket name
       (we own the sidecar .pid file, so it names the actual server PID — the
       session sweep must check whether the BASENAME pid is dead or unrelated,
       but the discriminating check is that the sidecar PID is a test-msrv
       process and the SOCKET-NAME pid is different from any current pytest).
       More precisely: the sweep logic the fix introduces must look at whether
       the server's owning session (the pytest pid embedded in the socket name)
       is still alive. We simulate this by naming the socket after a known-dead
       PID (1 itself is init, which is not our owning pytest, so any sweep that
       checks "is the socket-name PID our current pytest?" will reap it; or any
       sweep that checks "is the socket-name PID alive AND equal to os.getpid()?"
       will also reap it).
    4. Call the (to-be-added) sweep helper from tests/conftest.py.
       Currently no such function exists → AttributeError / ImportError → RED.
    5. Assert: server is dead, socket is gone.

    The test is SELF-CLEANING regardless of pass/fail: the finally block sends
    a direct SIGKILL to the spawned PID and unlinks all artifacts.

RED rationale:
    On current code, no sweep function exists in tests/conftest.py.  Calling it
    raises AttributeError.  The test will fail for the CORRECT reason: the
    feature (session-start sweep) is absent.

CI-SAFETY:
    - Uses an isolated socket named after a clearly-dead PID (PID 1 is always
      init/launchd — never a pytest process — so the sweep will classify it as
      orphaned regardless of the exact sweep predicate used).
    - start_new_session=True mirrors the real ensure_server() spawn: if the test
      itself crashes, no orphan leaks onto the canonical socket.
    - The finally block kills by ACTUAL SERVER PID from the spawned Popen, not
      from the .pid sidecar, so cleanup works even if the sidecar is wrong.
    - Server is spawned WITHOUT requesting an embed, so no MPS/GPU model loads.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UID = os.getuid() if hasattr(os, "getuid") else os.getpid()

# A fake "prior-session" pytest PID that is guaranteed to be dead on any real
# system: PID 1 is always init / launchd, never a pytest process.  The sweep
# must classify this socket as orphaned (its owning pytest session is gone).
_FAKE_DEAD_PYTEST_PID = 1


def _orphan_sock_path() -> str:
    return f"/tmp/tldr-test-msrv-{_UID}-{_FAKE_DEAD_PYTEST_PID}.sock"


def _spawn_server_on(sock_path: str) -> subprocess.Popen:
    """Spawn a real model server on *sock_path* (no model loaded — no embed sent).

    Mirrors the exact spawn path in ensure_server(): start_new_session=True so
    the child is a session leader that survives its parent's death.
    """
    env = dict(os.environ)
    env["TLDR_MODEL_SERVER_SOCKET"] = sock_path
    # Long idle so the server does not auto-exit before the test checks it.
    env["TLDR_MODEL_SERVER_IDLE_SECS"] = "120"
    return subprocess.Popen(
        [sys.executable, "-m", "tldr.model_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def _wait_sock(path: str, timeout: float = 12.0) -> bool:
    """Poll until socket file exists (server has bound it) or timeout."""
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


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------

class TestSessionStartSweepReapsOrphans:
    """Bug A regression: the session-start sweep must reap orphaned test servers.

    RED: no sweep function exists in tests/conftest.py right now.
    Calling it raises AttributeError → test FAILS for the correct reason.
    GREEN: after the fix, calling the sweep reaps the pre-staged orphan.
    """

    def test_sweep_reaps_orphan_from_dead_prior_session(self):
        """A test server whose session-owning PID is dead must be reaped by the sweep.

        Steps
        -----
        1. Stage an orphan: spawn a real model server bound to a prior-session
           socket /tmp/tldr-test-msrv-<uid>-1.sock (PID 1 is the fake "dead
           pytest" that owns this session's socket).
        2. Write a .pid sidecar naming the server's actual PID.
        3. Verify the server is alive and the socket exists.
        4. Call tests.conftest._sweep_leaked_test_servers() — the to-be-added
           session-start sweep.  On CURRENT code: AttributeError → RED.
        5. Assert: server PID is dead AND socket file is gone.

        Self-cleaning: the finally block sends a literal SIGKILL to the real
        server PID whether or not the test passed, then unlinks all artifacts.
        """
        sock_path = _orphan_sock_path()
        pid_path = sock_path + ".pid"
        lock_path = sock_path + ".lock"

        server_proc: "subprocess.Popen | None" = None
        server_pid: "int | None" = None

        try:
            # ------------------------------------------------------------------
            # Step 1 — spawn the orphan server
            # ------------------------------------------------------------------
            server_proc = _spawn_server_on(sock_path)
            server_pid = server_proc.pid

            # Step 2 — write the .pid sidecar (normally written by __main__.py)
            # We write it manually here because __main__.py may not yet write it
            # (that is a separate B6 feature); the sweep must be able to read it.
            # If __main__.py already writes it, the overwrite is idempotent.
            Path(pid_path).write_text(str(server_pid))

            # Step 3 — verify the orphan is staged correctly
            sock_present = _wait_sock(sock_path, timeout=12.0)
            assert sock_present, (
                f"Orphan server (PID {server_pid}) never bound {sock_path} within 12s. "
                "Cannot stage the leak scenario."
            )
            assert _pid_alive(server_pid), (
                f"Orphan server PID {server_pid} is already dead before the sweep. "
                "Test precondition failed."
            )

            # Step 4 — call the to-be-added sweep function
            #
            # The fix adds _sweep_leaked_test_servers() to tests/conftest.py and
            # calls it from pytest_configure.  We import conftest dynamically so
            # the test does not have a hard compile-time dependency, and we get a
            # clear AttributeError pointing exactly at the missing feature.
            #
            # RED: on current code this line raises AttributeError because
            # _sweep_leaked_test_servers does not exist in conftest.
            import tests.conftest as conftest_module
            if not hasattr(conftest_module, "_sweep_leaked_test_servers"):
                raise AttributeError(
                    "tests.conftest._sweep_leaked_test_servers does not exist. "
                    "The session-start sweep (Bug A fix) has not been implemented. "
                    "RED: the orphan server from a prior abnormal session will "
                    "never be reaped by the next session's pytest_configure — "
                    "one orphaned model-server process+socket accumulates per "
                    "abnormal run. "
                    "FIX SEAM: add _sweep_leaked_test_servers() to tests/conftest.py "
                    "that globs /tmp/tldr-test-msrv-<uid>-*.sock and reaps any "
                    "whose owning pytest PID (from the filename) is no longer alive, "
                    "then call it from pytest_configure (before _OWNED_MODEL_SERVER_SOCKET "
                    "is set so the sweep runs even if the current session does not own a socket)."
                )
            conftest_module._sweep_leaked_test_servers()

            # Step 5 — assert the orphan is reaped
            # Give the sweep up to 6s to reap (SIGTERM + 5s grace + SIGKILL).
            deadline = time.time() + 6.0
            server_gone = False
            while time.time() < deadline:
                if not _pid_alive(server_pid):
                    server_gone = True
                    break
                time.sleep(0.1)

            assert server_gone, (
                f"Orphan server (PID {server_pid}) is still alive {6.0}s after "
                "_sweep_leaked_test_servers() was called. "
                "The sweep must SIGTERM (with SIGKILL fallback) any server whose "
                "owning pytest PID is dead. Socket: {sock_path}"
            )
            assert not os.path.exists(sock_path), (
                f"Orphan socket {sock_path} still exists after sweep. "
                "The sweep must unlink the socket file."
            )

        finally:
            # ------------------------------------------------------------------
            # Unconditional cleanup — never leave a real orphan behind.
            # Kill by the actual spawned PID (not from the sidecar) so cleanup
            # works regardless of sidecar content.
            # ------------------------------------------------------------------
            if server_pid is not None and _pid_alive(server_pid):
                try:
                    os.kill(server_pid, signal.SIGKILL)
                except OSError:
                    pass
                # Wait up to 2s for the process to die
                deadline = time.time() + 2.0
                while time.time() < deadline and _pid_alive(server_pid):
                    time.sleep(0.05)

            # Also reap via proc.wait() if we still hold the Popen handle
            if server_proc is not None and server_proc.poll() is None:
                try:
                    server_proc.kill()
                except OSError:
                    pass
                try:
                    server_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass

            # Unlink all artifacts
            for p in [sock_path, pid_path, lock_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
