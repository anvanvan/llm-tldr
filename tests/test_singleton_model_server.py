"""Failing tests (RED phase) for the singleton model-server feature.

Behaviors under test (architecture.md):
  B1 — Singleton invariant: N concurrent ensure_server() callers produce
       exactly ONE server process (flock serializes; only one Popen fires).
  B2 — No-orphan-on-rebind: ModelServer.run() probes the socket before
       os.unlink; sends SIGTERM to the PID in {socket}.pid; waits up to 5s;
       SIGKILL fallback; proceeds only after orphan is gone.
  B3 — SIGTERM-graceful shutdown: a running server exits within bounded time
       on SIGTERM without SIGKILL and removes its socket.
  B4 — Idle-unload on ALL run() exit paths: maybe_unload() is called from
       the finally block of ModelServer.run(), not only the socket.timeout
       branch.
  B5 — Measured MPS footprint: maybe_unload() calls torch.mps.empty_cache()
       (guarded skip when no GPU; fake model for speed).
  B6 — PID sidecar: __main__ writes {socket}.pid on startup and removes it
       on exit.
  B7 — Test isolation / env routing: _model_server_socket_path() in ensure.py
       respects TLDR_MODEL_SERVER_SOCKET; ensure_server() and spawned server
       route to the same isolated socket.
  B8 — Daemon routing: thin per-project daemons use get_server_backed_default
       which calls ensure_server() → one shared model server, no second spawn.

All tests MUST FAIL on current code (RED phase) and pass after the minimal
implementation. Every test exercises NEW behavior that does not exist yet.

Design invariants:
- Every test that spawns a real process uses a unique tmpfile socket via
  TLDR_MODEL_SERVER_SOCKET and reaps the process in teardown (SIGKILL fallback)
  so no live server leaks into the canonical per-user socket.
- Bounded polling; max 12s per live test.
- Short idle window via TLDR_MODEL_SERVER_IDLE_SECS where needed.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_ping(path: str, timeout: float = 12.0) -> bool:
    """Poll until ping_server(path) returns True or timeout."""
    from tldr.daemon.ensure import ping_server
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ping_server(path):
            return True
        time.sleep(0.1)
    return False


def _reap(proc: subprocess.Popen) -> None:
    """Best-effort teardown: SIGTERM then SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _spawn_server(sock_path: str, idle_secs: int = 30) -> subprocess.Popen:
    """Spawn a real model server bound to *sock_path*."""
    env = dict(os.environ)
    env["TLDR_MODEL_SERVER_SOCKET"] = sock_path
    env["TLDR_MODEL_SERVER_IDLE_SECS"] = str(idle_secs)
    return subprocess.Popen(
        [sys.executable, "-m", "tldr.model_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )


# ===========================================================================
# B7 — env routing: _model_server_socket_path respects TLDR_MODEL_SERVER_SOCKET
# ===========================================================================

class TestEnvRouting:
    """B7: ensure._model_server_socket_path() must return the env override.

    RED: current _model_server_socket_path() always computes the uid-based path
    and ignores TLDR_MODEL_SERVER_SOCKET entirely (ensure.py lines 212-215).
    """

    def test_socket_path_returns_env_override_when_set(self, monkeypatch):
        """_model_server_socket_path() must return the TLDR_MODEL_SERVER_SOCKET value.

        RED: current code ignores the env var and returns the uid-based path.
        """
        from tldr.daemon import ensure

        sentinel = "/tmp/my-isolated-test-server.sock"
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", sentinel)

        result = ensure._model_server_socket_path()

        assert result == sentinel, (
            f"_model_server_socket_path() must return the env override "
            f"'{sentinel}' when TLDR_MODEL_SERVER_SOCKET is set. "
            f"Got: {result!r}. "
            "RED: current code ignores TLDR_MODEL_SERVER_SOCKET in ensure.py."
        )

    def test_ensure_server_routes_to_env_socket_not_canonical(self, monkeypatch, tmp_path):
        """ensure_server() must ping and return the env-override socket, not canonical.

        RED: _model_server_socket_path() ignores the env var → ensure_server()
        always pings the uid-based canonical path even when the env is set.
        """
        from tldr.daemon import ensure

        isolated = str(tmp_path / "isolated.sock")
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        pinged_paths: list[str] = []

        def tracking_ping(path: str) -> bool:
            pinged_paths.append(path)
            return True  # fast-path returns immediately

        with patch.object(ensure, "ping_server", side_effect=tracking_ping), \
             patch("subprocess.Popen", return_value=MagicMock()):
            result = ensure.ensure_server(timeout=2.0)

        assert all(p == isolated for p in pinged_paths), (
            f"ensure_server() must ping only the env socket '{isolated}'. "
            f"Pinged paths: {pinged_paths}. "
            "RED: current code pings the uid-based path regardless of env var."
        )
        assert result == isolated, (
            f"ensure_server() must return the env socket path '{isolated}'. "
            f"Got: {result!r}."
        )

    def test_ensure_server_spawn_uses_env_socket(self, monkeypatch, tmp_path):
        """When server absent, ensure_server() must spawn with the env socket path.

        The env var must propagate into the spawned process's environment AND
        into the poll loop (ping against the isolated socket, not canonical).

        RED: current _model_server_socket_path() ignores env var → Popen is
        called and the poll loop pings the canonical uid-based socket. The
        spawned server binds to the wrong path so isolation is broken.
        """
        from tldr.daemon import ensure

        isolated = str(tmp_path / "spawn-env.sock")
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        spawned_envs: list[str] = []
        poll_pings: list[str] = []
        ping_call_n = [0]

        def controlled_ping(path: str) -> bool:
            ping_call_n[0] += 1
            if ping_call_n[0] == 1:
                return False  # fast-path: server not up
            if ping_call_n[0] == 2:
                return False  # re-check under lock: still not up
            # Poll iteration: record what path is being polled, then succeed
            poll_pings.append(path)
            return True

        def tracking_popen(cmd, *args, **kwargs):
            env = kwargs.get("env") or {}
            spawned_envs.append(env.get("TLDR_MODEL_SERVER_SOCKET", "MISSING"))
            return MagicMock()

        with patch.object(ensure, "ping_server", side_effect=controlled_ping), \
             patch("subprocess.Popen", side_effect=tracking_popen), \
             patch("time.sleep", side_effect=lambda t: None):
            ensure.ensure_server(timeout=5.0)

        # Popen must be called with the env socket in its environment
        assert spawned_envs == [isolated], (
            f"Popen must be launched with TLDR_MODEL_SERVER_SOCKET={isolated!r}. "
            f"Got: {spawned_envs}. "
            "RED: current code computes a uid-based path ignoring env var."
        )
        # Poll loop must ping the isolated socket, not the canonical one
        assert all(p == isolated for p in poll_pings), (
            f"Poll loop must ping only the env socket '{isolated}'. "
            f"Got poll_pings={poll_pings}. "
            "RED: current code polls the canonical uid-based path."
        )


# ===========================================================================
# B6 — PID sidecar: {socket}.pid written on startup, removed on exit
# ===========================================================================

class TestPidSidecar:
    """B6: __main__.main() must write {socket}.pid before serving and remove it on exit.

    RED: current __main__.py has no pid-file logic whatsoever.
    """

    def test_pid_file_written_at_startup(self, tmp_path):
        """The model server must create {socket_path}.pid when it starts.

        RED: current __main__.py never writes a .pid sidecar.
        """
        sock_path = str(tmp_path / "pid-test.sock")
        pid_path = sock_path + ".pid"

        proc = _spawn_server(sock_path, idle_secs=30)
        try:
            _wait_for_ping(sock_path, timeout=12.0)

            assert os.path.exists(pid_path), (
                f"Expected {pid_path} to exist after server startup. "
                "RED: current __main__.py never writes a .pid sidecar."
            )
            written_pid = int(Path(pid_path).read_text().strip())
            assert written_pid == proc.pid, (
                f"PID file must contain the server's PID {proc.pid}, "
                f"got {written_pid}."
            )
        finally:
            _reap(proc)
            for p in [sock_path, pid_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def test_pid_file_removed_after_clean_shutdown(self, tmp_path):
        """The .pid file must be removed when the server exits via SIGTERM.

        RED: current __main__.py never writes nor removes a .pid file.
        """
        sock_path = str(tmp_path / "pid-clean.sock")
        pid_path = sock_path + ".pid"

        proc = _spawn_server(sock_path, idle_secs=30)
        try:
            ready = _wait_for_ping(sock_path, timeout=12.0)
            assert ready, "Server never became pingable."

            assert os.path.exists(pid_path), (
                "PID file must exist after startup. RED: __main__.py has no pid write."
            )

            proc.terminate()
            proc.wait(timeout=8)

            assert not os.path.exists(pid_path), (
                f"{pid_path} must be removed after SIGTERM shutdown. "
                "RED: __main__.py has no pid-file cleanup."
            )
        finally:
            _reap(proc)
            for p in [sock_path, pid_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def test_pid_file_contains_integer_pid(self, tmp_path):
        """The .pid file content must be a valid positive integer.

        RED: current __main__.py has no pid-file write.
        """
        sock_path = str(tmp_path / "pid-int.sock")
        pid_path = sock_path + ".pid"

        proc = _spawn_server(sock_path, idle_secs=30)
        try:
            _wait_for_ping(sock_path, timeout=12.0)

            assert os.path.exists(pid_path), (
                "PID file must exist. RED: __main__.py has no pid-file write."
            )
            content = Path(pid_path).read_text().strip()
            try:
                pid = int(content)
            except ValueError:
                pytest.fail(
                    f"PID file content must be an integer, got: {content!r}."
                )
            assert pid > 0, f"PID must be a positive integer, got {pid}."
        finally:
            _reap(proc)
            for p in [sock_path, pid_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


# ===========================================================================
# B3 — SIGTERM-graceful shutdown: exits within bounded time, removes socket
#      AND removes .pid sidecar (chains B3 + B6)
# ===========================================================================

class TestSigtermGracefulShutdown:
    """B3 + B6 combined: SIGTERM causes clean exit AND full artifact cleanup.

    RED for B3 component: a server that properly handles SIGTERM must also write
    and remove the .pid file (B6). The .pid file removal is the NEW behavior —
    it cannot pass today because __main__.py has no pid-file logic.

    We combine the behaviors so each test exercises a gap that is strictly absent
    from current code, even though SIGTERM signal delivery itself already works.
    """

    def test_sigterm_causes_clean_exit_and_pid_file_removed(self, tmp_path):
        """Server exits within 8s on SIGTERM AND the .pid file is removed.

        The SIGTERM-handler-calls-shutdown() path already works. What is NEW
        (and therefore RED) is the .pid file removal on exit, which requires
        B6 to be implemented in __main__.py.

        RED: .pid file is never written, so the assertion on its removal fails.
        """
        sock_path = str(tmp_path / "sigterm-test.sock")
        pid_path = sock_path + ".pid"

        proc = _spawn_server(sock_path, idle_secs=30)
        try:
            ready = _wait_for_ping(sock_path, timeout=12.0)
            assert ready, "Server never became pingable."

            # The .pid file must exist before we terminate (B6 write)
            assert os.path.exists(pid_path), (
                f"PID sidecar {pid_path} must exist BEFORE shutdown. "
                "RED: __main__.py never writes a .pid file."
            )

            proc.terminate()  # SIGTERM
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pytest.fail(
                    "Server did not exit within 8s after SIGTERM (B3 violation)."
                )

            time.sleep(0.2)  # brief flush for filesystem
            assert not os.path.exists(pid_path), (
                f"{pid_path} must be removed on clean shutdown (B6 cleanup). "
                "RED: __main__.py has no pid-file removal logic."
            )
        finally:
            _reap(proc)
            for p in [sock_path, pid_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

    def test_sigterm_removes_socket_and_pid_file(self, tmp_path):
        """After SIGTERM, BOTH the socket file AND the .pid file must be gone.

        Socket removal comes from run()'s finally block (already present).
        PID file removal is NEW: requires __main__.py B6 cleanup code.

        RED: .pid file is never written → assertion on its absence fails.
        """
        sock_path = str(tmp_path / "sigterm-both.sock")
        pid_path = sock_path + ".pid"

        proc = _spawn_server(sock_path, idle_secs=30)
        try:
            ready = _wait_for_ping(sock_path, timeout=12.0)
            assert ready, "Server never became pingable."

            # Both files must exist while running
            assert os.path.exists(pid_path), (
                "PID sidecar must exist while server is running. "
                "RED: __main__.py has no pid-file write."
            )

            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                pytest.fail("Server hung on SIGTERM.")

            time.sleep(0.2)
            assert not os.path.exists(sock_path), (
                f"Socket {sock_path} must be removed after SIGTERM + exit."
            )
            assert not os.path.exists(pid_path), (
                f"PID sidecar {pid_path} must be removed after SIGTERM + exit. "
                "RED: __main__.py has no pid-file cleanup."
            )
        finally:
            _reap(proc)
            for p in [sock_path, pid_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


# ===========================================================================
# B4 — exit path must NOT run in-process GPU teardown (it wedges Metal)
# ===========================================================================

class TestNoGpuTeardownOnExit:
    """run()'s finally must NOT call maybe_unload()/the model unloader.

    The model is freed only on the idle-accept tick (server alive, no in-flight
    GPU work). On a real exit (clean shutdown, SIGTERM/orphan reap, OSError) we
    must NOT run torch.mps.empty_cache()/model unload: it frees nothing the OS
    won't reclaim on process death, and worse it can deadlock inside the Apple
    GPU driver (AGXMetalG16X / IOGPU) when terminated mid-Metal-work — leaving a
    wedged orphan that pins ~9 GB of MPS memory and resists SIGKILL.
    """

    def test_no_unload_on_oserror_exit(self, monkeypatch):
        """run() exiting via OSError must NOT call maybe_unload() (no idle tick)."""
        from tldr.model_server.server import ModelServer

        maybe_unload_calls: list[bool] = []

        class FakeLifecycle:
            def maybe_unload(self) -> bool:
                maybe_unload_calls.append(True)
                return False

            def touch(self) -> None:
                pass

        class FakeSrv:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise OSError("simulated socket error")

            def close(self) -> None:
                pass

        server = ModelServer(socket_path="/tmp/does-not-exist-b4.sock")
        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())  # type: ignore[assignment]
        monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv())
        monkeypatch.setattr("os.unlink", lambda *a, **k: None)

        server.run()

        assert maybe_unload_calls == [], (
            "run()'s exit path must NOT call maybe_unload(): the OSError exit "
            "fires no idle tick, so any call would be the (removed) finally-block "
            f"GPU teardown that wedges Metal. Got {len(maybe_unload_calls)} call(s)."
        )

    def test_idle_tick_unloads_but_finally_does_not(self, monkeypatch):
        """Only the idle-accept tick calls maybe_unload(); the finally adds no 2nd call."""
        from tldr.model_server.server import ModelServer

        maybe_unload_calls: list[bool] = []

        class FakeLifecycle:
            def maybe_unload(self) -> bool:
                maybe_unload_calls.append(True)
                server._stop.set()  # stop after first idle tick
                return False

            def touch(self) -> None:
                pass

        class FakeSrv:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise socket.timeout()

            def close(self) -> None:
                pass

        server = ModelServer(socket_path="/tmp/does-not-exist-b4-shutdown.sock")
        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())  # type: ignore[assignment]
        monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv())
        monkeypatch.setattr("os.unlink", lambda *a, **k: None)

        server.run()

        # Exactly ONE call — the idle tick. The finally block must not add another.
        assert len(maybe_unload_calls) == 1, (
            f"maybe_unload() must be called ONLY from the idle-accept tick, not "
            f"from run()'s finally. Got {len(maybe_unload_calls)} call(s)."
        )

    def test_run_source_has_no_unload_in_finally_block(self):
        """ModelServer.run()'s finally block must NOT contain maybe_unload()/unloader.

        Structural guard against re-introducing the Metal-wedging GPU teardown
        on the exit path.
        """
        import inspect
        from tldr.model_server.server import ModelServer

        src = inspect.getsource(ModelServer.run)
        lines = src.split("\n")

        in_finally = False
        finally_indent = None
        offending: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if stripped == "finally:":
                in_finally = True
                finally_indent = indent
                continue
            if in_finally:
                # Leaving the finally block when indentation returns to/under it.
                if finally_indent is not None and indent <= finally_indent:
                    in_finally = False
                elif "maybe_unload" in stripped or "_model_unloader" in stripped:
                    offending.append(stripped)

        assert not offending, (
            "ModelServer.run()'s finally block must NOT call maybe_unload() or the "
            "model unloader — in-process GPU teardown on the exit path deadlocks "
            f"the Metal driver. Offending line(s): {offending}"
        )


# ===========================================================================
# B2 — No-orphan-on-rebind: probe-before-unlink + SIGTERM + SIGKILL fallback
# ===========================================================================

class TestOrphanReap:
    """B2: ModelServer.run() must probe the socket before os.unlink.

    RED: current run() unconditionally calls os.unlink with no connect probe.
    """

    def test_live_peer_receives_sigterm_before_unlink(self, tmp_path, monkeypatch):
        """When a live socket exists, SIGTERM must be sent to the PID in .pid file.

        RED: current run() does os.unlink unconditionally — os.kill never called.
        """
        from tldr.model_server.server import ModelServer

        sock_path = str(tmp_path / "orphan-reap.sock")
        pid_path = sock_path + ".pid"

        orphan_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        orphan_srv.bind(sock_path)
        orphan_srv.listen(1)
        orphan_pid = os.getpid()
        Path(pid_path).write_text(str(orphan_pid))

        kill_calls: list[tuple[int, int]] = []

        def recording_kill(pid: int, sig: int) -> None:
            kill_calls.append((pid, sig))
            if sig == 0 and any(s == signal.SIGTERM for _, s in kill_calls):
                raise ProcessLookupError("gone after SIGTERM")

        monkeypatch.setattr(os, "kill", recording_kill)
        monkeypatch.setattr(os, "unlink", lambda p: None)

        server = ModelServer(socket_path=sock_path)

        class FakeLifecycle:
            def maybe_unload(self) -> bool:
                return False

            def touch(self) -> None:
                pass

        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())  # type: ignore[assignment]

        class FakeSrv2:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise OSError("exit immediately")

            def close(self) -> None:
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv2())

        try:
            server.run()
        finally:
            orphan_srv.close()

        sigterm_sent = any(sig == signal.SIGTERM for _, sig in kill_calls)
        assert sigterm_sent, (
            f"ModelServer.run() must send SIGTERM to the orphan PID before binding. "
            f"os.kill calls recorded: {kill_calls}. "
            "RED: current run() has no probe-before-unlink — os.kill never called."
        )

    def test_dead_pid_in_pid_file_skips_sigterm(self, tmp_path, monkeypatch):
        """If the PID in .pid is dead, the liveness probe fires but no SIGTERM is sent.

        The B2 path requires: connect probe → read .pid → os.kill(pid, 0) →
        ProcessLookupError → skip SIGTERM. We assert os.kill(pid, 0) WAS called
        (proving the probe code ran) but SIGTERM was NOT sent.

        RED: current run() never calls os.kill at all — probe code doesn't exist.
        """
        from tldr.model_server.server import ModelServer

        sock_path = str(tmp_path / "dead-orphan.sock")
        pid_path = sock_path + ".pid"

        orphan_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        orphan_srv.bind(sock_path)
        orphan_srv.listen(1)

        dead_pid = 2147483647
        Path(pid_path).write_text(str(dead_pid))

        kill_calls: list[tuple[int, int]] = []

        def fake_kill(pid: int, sig: int) -> None:
            kill_calls.append((pid, sig))
            if sig == 0:
                raise ProcessLookupError(f"PID {pid} is dead")

        monkeypatch.setattr(os, "kill", fake_kill)
        monkeypatch.setattr(os, "unlink", lambda p: None)

        server = ModelServer(socket_path=sock_path)

        class FakeLifecycle:
            def maybe_unload(self) -> bool:
                return False

            def touch(self) -> None:
                pass

        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())  # type: ignore[assignment]

        class FakeSrv3:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise OSError("exit immediately")

            def close(self) -> None:
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv3())

        try:
            server.run()
        finally:
            orphan_srv.close()

        # Must call os.kill(pid, 0) as part of the B2 liveness probe
        kill_with_sig0 = [c for c in kill_calls if c[1] == 0]
        assert len(kill_with_sig0) >= 1, (
            f"B2 must call os.kill(pid, 0) to probe liveness. "
            f"Got kill_calls={kill_calls}. "
            "RED: current run() has no probe-before-unlink — os.kill never called."
        )
        # But SIGTERM must NOT be sent for a dead PID
        sigterm_calls = [c for c in kill_calls if c[1] == signal.SIGTERM]
        assert sigterm_calls == [], (
            f"SIGTERM must NOT be sent when PID {dead_pid} is dead. "
            f"Got: {sigterm_calls}."
        )

    def test_missing_pid_file_skips_sigterm_and_proceeds(self, tmp_path, monkeypatch):
        """If the .pid file is missing, skip SIGTERM and proceed with unlink.

        We verify the .pid read was attempted (the B2 probe ran) by trapping
        open() calls on the pid_path. No SIGTERM must be sent.

        RED: current run() never opens the .pid file — probe code doesn't exist.
        """
        from tldr.model_server.server import ModelServer

        sock_path = str(tmp_path / "missing-pid.sock")
        pid_path = sock_path + ".pid"
        # deliberately no .pid file

        orphan_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        orphan_srv.bind(sock_path)
        orphan_srv.listen(1)

        pid_file_open_attempts: list[str] = []
        sigterm_calls: list[int] = []
        real_open = open

        def tracking_open(path: object, *args: object, **kwargs: object) -> object:
            if str(path) == pid_path:
                pid_file_open_attempts.append(str(path))
                raise FileNotFoundError(f"No such file: {path}")
            return real_open(path, *args, **kwargs)  # type: ignore[call-overload]

        def fake_kill(pid: int, sig: int) -> None:
            if sig == signal.SIGTERM:
                sigterm_calls.append(pid)

        monkeypatch.setattr(os, "kill", fake_kill)
        monkeypatch.setattr(os, "unlink", lambda p: None)
        monkeypatch.setattr("builtins.open", tracking_open)

        server = ModelServer(socket_path=sock_path)

        class FakeLifecycle:
            def maybe_unload(self) -> bool:
                return False

            def touch(self) -> None:
                pass

        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())  # type: ignore[assignment]

        class FakeSrv4:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise OSError("exit immediately")

            def close(self) -> None:
                pass

        monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv4())

        try:
            server.run()
        finally:
            orphan_srv.close()

        assert len(pid_file_open_attempts) >= 1, (
            f"B2 must attempt to read the .pid file after a live-socket probe. "
            f"Got pid_file_open_attempts={pid_file_open_attempts}. "
            "RED: current run() has no probe-before-unlink — pid file never read."
        )
        assert sigterm_calls == [], (
            "No SIGTERM must be sent when .pid file is missing. "
            f"Got: {sigterm_calls}."
        )

    def test_run_source_has_probe_before_unlink(self):
        """ModelServer.run() source must contain a connect probe before os.unlink.

        Structural fast-fail: probe existence check.

        RED: current run() starts with `try: os.unlink(...)` — no connect probe.
        """
        import inspect
        from tldr.model_server.server import ModelServer

        src = inspect.getsource(ModelServer.run)

        unlink_pos = src.find("os.unlink")
        connect_pos = src.find(".connect(")
        if connect_pos == -1:
            connect_pos = src.find("connect(self.socket_path")

        assert connect_pos != -1, (
            "ModelServer.run() must contain a socket.connect probe "
            "to detect a live peer before calling os.unlink. "
            "RED: current run() has no connect probe — it unconditionally unlinks."
        )
        assert connect_pos < unlink_pos, (
            "The connect probe must appear BEFORE os.unlink in run(). "
            f"connect_pos={connect_pos}, unlink_pos={unlink_pos}."
        )

    def test_orphan_surviving_grace_window_gets_sigkill(self, tmp_path, monkeypatch):
        """If the orphan survives the 5s SIGTERM grace window, _reap_orphan must SIGKILL.

        We make os.kill(pid, 0) always report the orphan still ALIVE so the
        busy-poll never exits early, and fast-forward time.monotonic() past the
        5s deadline (with time.sleep stubbed) so no real waiting occurs. The
        SIGKILL fallback (signal 9) must then be sent to the orphan pid.
        """
        from tldr.model_server.server import ModelServer

        sock_path = str(tmp_path / "sigkill-fallback.sock")
        pid_path = sock_path + ".pid"

        orphan_pid = 4242
        Path(pid_path).write_text(str(orphan_pid))

        kill_calls: list[tuple[int, int]] = []

        def always_alive_kill(pid: int, sig: int) -> None:
            kill_calls.append((pid, sig))
            # sig 0 = liveness probe: never raise → orphan "stays alive" forever,
            # so the busy-poll runs until the deadline elapses.

        monkeypatch.setattr(os, "kill", always_alive_kill)

        server = ModelServer(socket_path=sock_path)

        # Fast-forward monotonic time so the 5s grace window elapses with no real
        # waiting. Use a monotonically INCREASING fake clock (advances 100s per
        # read) so the deadline (start + 5.0) is always crossed on the next loop
        # check, regardless of how many time.monotonic() calls _reap_orphan makes.
        # Patched only around _reap_orphan so ModelServer.__init__'s own
        # time.monotonic() use (lifecycle deadline) is unaffected.
        clock = {"t": 1000.0}

        def fake_monotonic() -> float:
            clock["t"] += 100.0
            return clock["t"]

        monkeypatch.setattr(time, "monotonic", fake_monotonic)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        server._reap_orphan()

        sigkill_calls = [c for c in kill_calls if c[1] == signal.SIGKILL]
        assert sigkill_calls == [(orphan_pid, signal.SIGKILL)], (
            "_reap_orphan() must SIGKILL the orphan pid when it survives the "
            f"SIGTERM grace window. os.kill calls recorded: {kill_calls}."
        )
        # A SIGTERM must have been attempted first.
        assert any(c == (orphan_pid, signal.SIGTERM) for c in kill_calls), (
            f"SIGTERM must precede the SIGKILL fallback. Got: {kill_calls}."
        )


# ===========================================================================
# B1 — Singleton invariant: N concurrent ensure_server() callers → one Popen
# ===========================================================================

class TestSingletonInvariant:
    """B1: concurrent ensure_server() callers must produce exactly ONE server.

    The env-routing fix (B7) is a prerequisite for test isolation. B1's flock
    logic is already correct; these tests target the combination of correct
    flock behavior AND correct env routing — both are required for a complete
    singleton guarantee in isolated test environments.
    """

    def test_concurrent_callers_produce_single_popen_with_env_routing(
        self, monkeypatch, tmp_path
    ):
        """Three concurrent ensure_server() calls must spawn Popen exactly ONCE
        and all pings must go to the env-override socket, not the canonical path.

        The second half (env socket) is RED on current code: _model_server_socket_path()
        ignores TLDR_MODEL_SERVER_SOCKET, so all pings go to the canonical path
        even though the env var is set → the assertion on ping destinations fails.

        RED: all pings go to the canonical uid-based socket, not 'isolated'.
        """
        from tldr.daemon import ensure

        isolated = str(tmp_path / "singleton.sock")
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        popen_count = [0]
        popen_lock = threading.Lock()
        all_ping_paths: list[str] = []
        paths_lock = threading.Lock()

        ping_state = {"ready": False}

        def controlled_ping(path: str) -> bool:
            with paths_lock:
                all_ping_paths.append(path)
            with popen_lock:
                return ping_state["ready"]

        def counting_popen(*args: object, **kwargs: object) -> MagicMock:
            with popen_lock:
                popen_count[0] += 1
                ping_state["ready"] = True
            return MagicMock()

        errors: list[Exception] = []

        def call_ensure() -> None:
            try:
                ensure.ensure_server(timeout=5.0)
            except Exception as exc:
                errors.append(exc)

        # Apply all patches at test level, not inside threads, to prevent
        # mock leakage into subsequent tests when the full suite runs.
        with patch.object(ensure, "ping_server", side_effect=controlled_ping), \
             patch("subprocess.Popen", side_effect=counting_popen), \
             patch("time.sleep", side_effect=lambda t: None):
            threads = [threading.Thread(target=call_ensure) for _ in range(3)]
            for t in threads:
                t.start()
            # Join ALL worker threads to completion BEFORE the patch context
            # exits. If a thread were still inside ensure_server() when the
            # patches are torn down, it would run under a restored
            # subprocess.Popen and could leak a Popen MagicMock into later
            # same-process tests (non-deterministic full suite). Fail loudly if
            # any thread does not finish so the leak surfaces here, not later.
            for t in threads:
                t.join(timeout=30.0)
            unfinished = [t for t in threads if t.is_alive()]
            assert not unfinished, (
                f"{len(unfinished)} worker thread(s) did not finish within the "
                "join window; a thread escaping the patch context would leak a "
                "Popen mock into later tests."
            )

        assert not errors, f"ensure_server() raised: {errors}"
        assert popen_count[0] == 1, (
            f"ensure_server() from 3 concurrent threads must spawn exactly ONE Popen. "
            f"Got {popen_count[0]}."
        )
        # All pings must go to the env socket, not the canonical uid-based path
        non_isolated = [p for p in all_ping_paths if p != isolated]
        assert non_isolated == [], (
            f"All pings must go to the env socket '{isolated}'. "
            f"Found pings to non-isolated paths: {non_isolated}. "
            "RED: _model_server_socket_path() ignores TLDR_MODEL_SERVER_SOCKET — "
            "all pings go to the canonical uid-based socket instead."
        )

    def test_ensure_server_without_env_routing_uses_wrong_socket(
        self, monkeypatch, tmp_path
    ):
        """ensure_server() must consult ONLY the env socket when the env var is set.

        Documents the G-4 gap precisely: without B7, ensure_server() pings and
        would spawn at the canonical uid-based path, breaking test isolation.

        RED: pings go to the canonical socket, not the isolated env socket.
        """
        from tldr.daemon import ensure

        isolated = str(tmp_path / "isolation-test.sock")
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        ping_calls: list[str] = []

        def tracking_ping(path: str) -> bool:
            ping_calls.append(path)
            return path == isolated  # True only for the correct env socket

        with patch.object(ensure, "ping_server", side_effect=tracking_ping), \
             patch("subprocess.Popen", return_value=MagicMock()), \
             patch("time.sleep", side_effect=lambda t: None):
            try:
                ensure.ensure_server(timeout=2.0)
            except RuntimeError:
                pass  # timeout if canonical socket never answers — expected

        non_isolated = [p for p in ping_calls if p != isolated]
        assert non_isolated == [], (
            f"ensure_server() must ping only the env socket '{isolated}'. "
            f"Found pings to OTHER paths: {non_isolated}. "
            "RED: _model_server_socket_path() ignores TLDR_MODEL_SERVER_SOCKET."
        )


# ===========================================================================
# B5 — MPS footprint: the exit path must NOT clear the cache in-process
# ===========================================================================

class TestMpsFootprint:
    """B5: run()'s exit path must NOT invoke the model unloader / empty_cache.

    Pair to TestNoGpuTeardownOnExit: the GPU teardown on death is both useless
    (the OS reclaims the GPU context on process exit) and dangerous (it can
    deadlock the Metal driver mid-work). Memory is freed only on the idle tick.
    """

    def test_run_finally_does_not_trigger_unloader(self, monkeypatch):
        """run()'s finally must NOT call maybe_unload()/the unloader on OSError exit.

        OSError exits the accept loop immediately with no idle tick, so any
        maybe_unload() call could only come from the (removed) finally teardown.
        """
        from tldr.model_server.server import ModelServer

        unloader_calls: list[bool] = []

        class FakeLifecycle:
            def maybe_unload(self) -> bool:
                unloader_calls.append(True)
                return True

            def touch(self) -> None:
                pass

        class FakeSrv:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise OSError("exit immediately for B5 test")

            def close(self) -> None:
                pass

        server = ModelServer(socket_path="/tmp/b5-finally-test.sock")
        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())  # type: ignore[assignment]
        monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv())
        monkeypatch.setattr("os.unlink", lambda *a, **k: None)

        server.run()

        assert unloader_calls == [], (
            "run()'s exit path must NOT call maybe_unload() (→ empty_cache): "
            "in-process GPU teardown on death wedges Metal and frees nothing the "
            f"OS won't reclaim. Got {len(unloader_calls)} call(s)."
        )

    def test_run_oserror_exit_does_not_clear_semantic_model(self, monkeypatch):
        """run() exit via OSError must NOT clear semantic's module-level model cache.

        The previous design ran semantic.unload_model() from run()'s finally on
        every exit; that path is removed (it deadlocked Metal mid-work). A dying
        process drops its own memory anyway, so the in-process cache need not be
        cleared — and the exit path must not touch the GPU. sem._model therefore
        survives a run() exit.
        """
        import tldr.semantic as sem
        from tldr.model_server.server import ModelServer

        sentinel = object()
        sem._model = sentinel  # type: ignore[assignment]
        sem._model_name = "fake/model"  # type: ignore[assignment]
        sem._model_device = "cpu"  # type: ignore[assignment]

        class FakeSrv:
            def bind(self, *a: object) -> None:
                pass

            def listen(self, *a: object) -> None:
                pass

            def settimeout(self, *a: object) -> None:
                pass

            def accept(self) -> None:
                raise OSError("exit immediately for B5 chain test")

            def close(self) -> None:
                pass

        try:
            server = ModelServer(
                socket_path="/tmp/b5-chain-test.sock",
                idle_seconds=0,  # deadline immediately elapsed
            )
            monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv())
            monkeypatch.setattr("os.unlink", lambda *a, **k: None)

            server.run()

            assert sem._model is sentinel, (  # type: ignore[comparison-overlap]
                "run()'s exit path must NOT run semantic.unload_model(): the GPU "
                "teardown on death is removed (it wedged Metal). sem._model should "
                f"survive the run() exit. Got {sem._model!r}."
            )
        finally:
            sem._model = None  # type: ignore[assignment]
            sem._model_name = None  # type: ignore[assignment]
            sem._model_device = None  # type: ignore[assignment]


# ===========================================================================
# B8 — Daemon routing: thin daemons route through ONE shared server
# ===========================================================================

class TestDaemonRouting:
    """B8: per-project daemons must route through the single shared model server.

    The critical NEW behavior: ensure_server() must respect TLDR_MODEL_SERVER_SOCKET
    (B7) so that daemon tests using an isolated socket actually route there.
    Without B7, daemon routing is broken in test environments even if the
    production code path exists.
    """

    def test_get_server_backed_default_routes_to_env_socket(self, monkeypatch, tmp_path):
        """get_server_backed_default(force=True) must route to the env-override socket.

        Without B7, ensure_server() uses the canonical uid-based socket path
        even when TLDR_MODEL_SERVER_SOCKET is set → daemon test isolation breaks.

        RED: ensure_server() inside get_server_backed_default will call
        _model_server_socket_path() which ignores the env var → the returned
        socket_path is the canonical one, not the isolated env socket.
        The assertion on returned socket path fails.
        """
        from tldr.embedding_backend import get_server_backed_default
        from tldr.daemon import ensure

        isolated = str(tmp_path / "daemon-env-socket.sock")
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        returned_sockets: list[str] = []

        real_ensure_server = ensure.ensure_server

        def tracking_ensure_server(timeout: float = 10.0) -> str:
            # Call through to the real ensure_server so _model_server_socket_path
            # is exercised — this is where B7 must kick in.
            # We mock ping to return True immediately so no real server spawns.
            with patch.object(ensure, "ping_server", return_value=True):
                path = real_ensure_server(timeout=timeout)
            returned_sockets.append(path)
            return path

        with patch.object(ensure, "ensure_server", side_effect=tracking_ensure_server), \
             patch("tldr.model_server.transport.connect_unix", side_effect=OSError("no conn")):
            get_server_backed_default(force=True)

        assert returned_sockets == [isolated], (
            f"get_server_backed_default(force=True) must route through the env socket "
            f"'{isolated}'. ensure_server() returned: {returned_sockets}. "
            "RED: _model_server_socket_path() ignores TLDR_MODEL_SERVER_SOCKET → "
            "ensure_server() returns the canonical uid-based path instead."
        )

    def test_two_daemon_calls_both_use_env_socket(self, monkeypatch, tmp_path):
        """Two get_server_backed_default(force=True) calls must both route to env socket.

        RED: without B7, both calls use the canonical uid-based path even with
        the env var set → the isolated test socket is never used → test suite
        would collide with real running model servers.
        """
        from tldr.embedding_backend import get_server_backed_default
        from tldr.daemon import ensure

        isolated = str(tmp_path / "two-daemons.sock")
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        returned_sockets: list[str] = []

        real_ensure_server = ensure.ensure_server

        def tracking_ensure_server(timeout: float = 10.0) -> str:
            with patch.object(ensure, "ping_server", return_value=True):
                path = real_ensure_server(timeout=timeout)
            returned_sockets.append(path)
            return path

        with patch.object(ensure, "ensure_server", side_effect=tracking_ensure_server), \
             patch("tldr.model_server.transport.connect_unix", side_effect=OSError("no conn")):
            get_server_backed_default(force=True)
            get_server_backed_default(force=True)

        assert len(returned_sockets) == 2, (
            f"ensure_server() must be called for each get_server_backed_default call. "
            f"Got {len(returned_sockets)} calls."
        )
        wrong = [s for s in returned_sockets if s != isolated]
        assert wrong == [], (
            f"Both daemon calls must use the env socket '{isolated}'. "
            f"Non-env paths returned: {wrong}. "
            "RED: _model_server_socket_path() ignores TLDR_MODEL_SERVER_SOCKET."
        )

    def test_daemon_routing_uses_env_socket_when_set(self, monkeypatch):
        """_model_server_socket_path() must return env socket for daemon isolation.

        This is the root cause of B8 test-isolation failure: without this fix,
        any daemon test that sets TLDR_MODEL_SERVER_SOCKET still connects to the
        canonical socket, potentially interacting with a real running server.

        RED: current _model_server_socket_path() always returns the uid-based path.
        """
        from tldr.daemon import ensure

        isolated = "/tmp/daemon-routing-isolated-test.sock"
        monkeypatch.setenv("TLDR_MODEL_SERVER_SOCKET", isolated)

        result_path = ensure._model_server_socket_path()

        assert result_path == isolated, (
            f"With TLDR_MODEL_SERVER_SOCKET={isolated}, _model_server_socket_path() "
            f"must return the isolated socket. "
            f"Got: {result_path!r}. "
            "RED: current _model_server_socket_path() ignores the env var in ensure.py."
        )


# ===========================================================================
# EDGE-1 — No-orphan-on-rebind END-TO-END: second direct server reaps the
# orphan, NOT itself (integration test against real __main__ entrypoint)
# ===========================================================================

class TestOrphanReapIntegration:
    """End-to-end integration test for the orphan-reap race (EDGE-1).

    The bug (confirmed by live verification): in __main__.py, the PID sidecar
    is written (os.getpid() → {socket}.pid) BEFORE server.run() is called.
    When server2 starts, it overwrites the sidecar with its OWN pid. Then
    server.run()'s probe-before-unlink fires _reap_orphan(), reads the sidecar
    (now server2's own pid), and SIGTERMs ITSELF — so server2 suicides and the
    orphan (server1) survives.

    This test drives the REAL __main__ entrypoint end-to-end and asserts the
    CORRECT behaviour: after server2 starts, server1 (the orphan) is reaped and
    server2 (the newcomer) survives.

    RED: on current code, server2 self-terminates so server1 survives → Step 3
    (server1 pid is gone) and Step 5 (survivor is server2) both fail.
    """

    # Timeout constants kept short to fail fast in CI.
    _START_TIMEOUT = 12.0    # seconds to wait for a server to become pingable
    _REAP_TIMEOUT = 6.0      # seconds to wait for orphan to disappear
    _IDLE_SECS = 60          # idle window — long enough to not self-terminate

    def test_second_direct_server_reaps_orphan_not_self(self, tmp_path):
        """A second directly-started server must reap server1 (orphan) and survive.

        Steps:
          1. Start server1 on an isolated socket; wait until pingable.
          2. Start server2 on the SAME isolated socket.
          3. Within _REAP_TIMEOUT, assert server1's PID is GONE (orphan reaped).
          4. Assert exactly one server process is bound to the isolated socket.
          5. Assert the SURVIVOR is server2 (the newcomer), NOT server1.

        RED: current __main__.py writes the sidecar before run(), so server2
        reads its own pid and SIGTERMs itself → server1 lives → assertions in
        steps 3 and 5 fail.
        """
        import tempfile

        # Isolated socket — never touches the canonical per-user socket.
        iso_sock = tempfile.mktemp(suffix=".sock", prefix="tldr-orphan-reap-test-")
        pid_path = iso_sock + ".pid"
        server1: subprocess.Popen | None = None
        server2: subprocess.Popen | None = None

        try:
            # ------------------------------------------------------------------
            # Step 1: Start server1 and wait until it is pingable.
            # ------------------------------------------------------------------
            server1 = _spawn_server(iso_sock, idle_secs=self._IDLE_SECS)
            server1_pid = server1.pid

            pingable = _wait_for_ping(iso_sock, timeout=self._START_TIMEOUT)
            assert pingable, (
                f"server1 (pid={server1_pid}) never became pingable on {iso_sock} "
                f"within {self._START_TIMEOUT}s. Cannot proceed with orphan-reap test."
            )

            # The sidecar must contain server1's pid at this point (B6 prerequisite).
            # If B6 is also broken, we document it but don't fail here — the real
            # assertion is about which process survives after server2 starts.
            if os.path.exists(pid_path):
                sidecar_before = Path(pid_path).read_text().strip()
            else:
                sidecar_before = "<missing>"

            # ------------------------------------------------------------------
            # Step 2: Start server2 against the SAME isolated socket.
            # ------------------------------------------------------------------
            server2 = _spawn_server(iso_sock, idle_secs=self._IDLE_SECS)
            server2_pid = server2.pid

            # ------------------------------------------------------------------
            # Step 3: Within _REAP_TIMEOUT, server1 (orphan) must disappear.
            # The orphan reap in server2's run() should SIGTERM server1.
            # ------------------------------------------------------------------
            deadline = time.time() + self._REAP_TIMEOUT
            server1_gone = False
            while time.time() < deadline:
                try:
                    os.kill(server1_pid, 0)
                    # process still alive — keep polling
                except ProcessLookupError:
                    server1_gone = True
                    break
                except OSError:
                    server1_gone = True
                    break
                time.sleep(0.15)

            # Also check via poll() in case the process is a zombie.
            if not server1_gone:
                rc = server1.poll()
                if rc is not None:
                    server1_gone = True

            assert server1_gone, (
                f"server1 (pid={server1_pid}, the ORPHAN) must be reaped by server2 "
                f"within {self._REAP_TIMEOUT}s. It is still alive. "
                f"Sidecar before server2 started: {sidecar_before!r}. "
                f"Current sidecar: {Path(pid_path).read_text().strip() if os.path.exists(pid_path) else '<missing>'}. "
                "RED: the PID sidecar is written before server.run() in __main__.py; "
                "server2 overwrites the sidecar with its OWN pid, then _reap_orphan() "
                "reads server2's pid and SIGTERMs server2 itself (self-suicide). "
                "server1 survives as the orphan. Fix: write the sidecar AFTER "
                "the orphan-reap probe in run(), not before run() returns."
            )

            # ------------------------------------------------------------------
            # Step 4: Exactly one server must be bound to the isolated socket.
            # Use the .pid sidecar (survivor should have updated it) AND process
            # liveness checks. We deliberately do NOT rely on the sidecar alone
            # for survivor identity (the sidecar is the broken part).
            # ------------------------------------------------------------------

            # Give server2 a moment to fully bind and write its sidecar.
            time.sleep(0.3)

            # server2 must be alive.
            server2_alive = False
            try:
                os.kill(server2_pid, 0)
                server2_alive = True
            except (ProcessLookupError, OSError):
                pass
            if not server2_alive:
                rc = server2.poll()
                server2_alive = (rc is None)

            # ------------------------------------------------------------------
            # Step 5: The SURVIVOR is server2, NOT server1.
            # ------------------------------------------------------------------
            assert server2_alive, (
                f"server2 (pid={server2_pid}, the NEW server) must be the survivor "
                f"after the orphan reap. It is DEAD. "
                f"server1 (pid={server1_pid}) still alive: "
                f"{_pid_alive(server1_pid)}. "
                f"Sidecar content: {Path(pid_path).read_text().strip() if os.path.exists(pid_path) else '<missing>'}. "
                "RED: server2 self-terminated (it SIGTERMed its own pid because "
                "__main__.py wrote the sidecar with server2's pid BEFORE run() "
                "called _reap_orphan(), so _reap_orphan() read server2's own pid "
                "and killed server2). server1 survives as the unintended orphan."
            )

        finally:
            # Teardown: SIGKILL any surviving processes so nothing leaks onto the
            # canonical socket.
            for proc in [server1, server2]:
                if proc is not None:
                    _reap(proc)
            # Clean up socket artifacts.
            for p in [iso_sock, pid_path]:
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a live process."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False
