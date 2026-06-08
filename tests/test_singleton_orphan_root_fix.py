"""Root-cause regression tests for runaway model-server orphans.

Three distinct root causes let multiple ~GB model servers accumulate on the
canonical per-user socket (observed repeatedly in production):

1. UNRESPONSIVE SERVER — the accept loop dispatched each connection
   synchronously, so during a slow embed the single thread stopped calling
   accept(). The small ``listen()`` backlog then filled and further connect()s
   were refused.

2. BUSY != DEAD — both ``ensure_server``'s ping and the new server's reap-probe
   used a bare connect(). A full backlog (a *live but busy* server) refuses the
   connect, which is misread as "server dead": ensure_server spawns a redundant
   server, and that server's reap-probe also fails so it SKIPS the reap — it
   unlinks the busy server's socket and rebinds WITHOUT killing it. The busy
   server becomes an orphan that was never even signalled.

3. NO SELF-EVICTION — an orphaned server (its socket stolen + sidecar
   overwritten) never noticed and ran forever holding the model.

These tests pin the fixes: a responsive accept loop, reap-by-sidecar-PID (not
just the connect probe), and self-eviction when superseded — plus the matching
guard that a superseded server must NOT delete the new owner's socket/sidecar.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from tldr.model_server import transport
from tldr.model_server.server import ModelServer


def _spawn_live_pid() -> subprocess.Popen:
    """A real, signalable child process to stand in for a foreign server PID."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def _wait(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ===========================================================================
# Root cause #2 — reap by sidecar PID, not just the connect probe
# ===========================================================================

class TestReapBySidecarPid:
    """A live server whose socket is unreachable (busy/backlog-full) must still
    be reaped via its PID sidecar before a new server steals the socket."""

    def test_live_sidecar_owner_is_reaped_even_when_connect_probe_fails(
        self, tmp_path, monkeypatch
    ):
        sock_path = str(tmp_path / "ms.sock")
        # A live foreign "server" recorded in the sidecar, but NOTHING is
        # listening on the socket path → the connect probe will fail (mimics a
        # busy server whose backlog is full: connect refused).
        victim = _spawn_live_pid()
        try:
            with open(sock_path + ".pid", "w") as fh:
                fh.write(str(victim.pid))

            # FakeSrv stands in for the accept-loop socket so run() exits
            # immediately after the startup reap. The reap PROBE uses the real
            # captured socket (_raw_socket), which connects to the (absent)
            # sock_path and fails → live_peer=False. Reap must still fire from
            # the live-sidecar-PID branch.
            class FakeSrv:
                def bind(self, *a):
                    pass

                def listen(self, *a):
                    pass

                def settimeout(self, *a):
                    pass

                def accept(self):
                    raise OSError("exit accept loop immediately")

                def close(self):
                    pass

            server = ModelServer(socket_path=sock_path)
            monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv())

            server.run()

            assert _wait(lambda: victim.poll() is not None, timeout=8.0), (
                "the live sidecar-owner process must be reaped (SIGTERM/SIGKILL) "
                "even though the connect probe failed — a busy server with a full "
                "backlog refuses connect but is very much alive."
            )
        finally:
            if victim.poll() is None:
                victim.kill()
            victim.wait()


# ===========================================================================
# Root cause #3 — self-eviction when superseded (and don't nuke the new owner)
# ===========================================================================

class TestSelfEviction:
    """A server whose sidecar is overwritten by another live PID must exit, and
    must NOT delete the new owner's socket file or sidecar on the way out."""

    def _fake_lifecycle(self):
        class FakeLifecycle:
            def maybe_unload(self):
                return False

            def touch(self):
                pass

            def get_model(self):  # never called (no embed in this test)
                raise AssertionError("model must not load")

        return FakeLifecycle()

    def test_superseded_server_exits_and_preserves_new_owner_files(
        self, tmp_path, monkeypatch
    ):
        sock_path = str(tmp_path / "ms.sock")
        server = ModelServer(socket_path=sock_path)
        monkeypatch.setattr(server, "_lifecycle", self._fake_lifecycle())

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        try:
            # Wait until it owns the socket + wrote its sidecar (== our pid,
            # since the server runs in-process here).
            assert _wait(lambda: os.path.exists(sock_path + ".pid")), "sidecar never written"
            assert os.path.exists(sock_path), "socket never bound"

            # Simulate a NEW server taking over: overwrite the sidecar with a
            # live foreign PID. (In production the new server also rebinds the
            # socket; here we keep the socket file so we can assert the orphan
            # leaves it intact.)
            newowner = _spawn_live_pid()
            try:
                with open(sock_path + ".pid", "w") as fh:
                    fh.write(str(newowner.pid))

                assert _wait(lambda: not t.is_alive(), timeout=3.0), (
                    "a superseded server (its sidecar now names another live PID) "
                    "must self-evict and exit within a couple of accept ticks."
                )
                # It must NOT have deleted the new owner's sidecar or socket.
                assert os.path.exists(sock_path + ".pid"), (
                    "self-evicting orphan deleted the NEW owner's sidecar"
                )
                assert os.path.exists(sock_path), (
                    "self-evicting orphan deleted the NEW owner's socket file"
                )
                with open(sock_path + ".pid") as fh:
                    assert fh.read().strip() == str(newowner.pid), (
                        "the new owner's sidecar PID was clobbered"
                    )
            finally:
                newowner.kill()
                newowner.wait()
        finally:
            server.shutdown()
            t.join(timeout=3.0)


# ===========================================================================
# Root cause #1 — the accept loop stays responsive during a slow embed
# ===========================================================================

class TestResponsiveDuringEmbed:
    """While the single GPU worker is busy with a slow embed, the server must
    still accept and answer other connections (e.g. ping). Otherwise the
    backlog fills, connect() is refused, and ensure_server false-spawns."""

    def test_ping_answered_while_embed_in_flight(self, tmp_path, monkeypatch):
        sock_path = str(tmp_path / "ms.sock")

        release = threading.Event()
        embed_started = threading.Event()

        class BlockingModel:
            def encode(self, texts, normalize_embeddings=True, **kw):
                embed_started.set()
                # Hold the GPU worker hostage until the test releases it.
                if not release.wait(timeout=10):
                    raise AssertionError("embed never released")
                return np.ones((len(texts), 4), dtype=np.float32)

        class FakeLifecycle:
            def maybe_unload(self):
                return False

            def touch(self):
                pass

            def get_model(self):
                return BlockingModel()

        server = ModelServer(socket_path=sock_path)
        monkeypatch.setattr(server, "_lifecycle", FakeLifecycle())

        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        try:
            assert _wait(lambda: os.path.exists(sock_path)), "socket never bound"

            # Client 1: fire a slow embed that occupies the single GPU worker.
            def slow_embed():
                s = transport.connect_unix(sock_path)
                s.settimeout(10)
                transport.send_message(s, {"cmd": "embed", "texts": ["a", "b"]})
                transport.recv_message(s)
                s.close()

            e = threading.Thread(target=slow_embed, daemon=True)
            e.start()
            assert embed_started.wait(timeout=5), "embed never started"

            # Client 2: with the worker blocked, a ping must STILL be answered
            # promptly — proving accept() is not blocked by the in-flight embed.
            t0 = time.time()
            s2 = transport.connect_unix(sock_path)
            s2.settimeout(3)
            transport.send_message(s2, {"cmd": "ping"})
            resp = transport.recv_message(s2)
            s2.close()
            elapsed = time.time() - t0

            assert resp.get("status") == "ok", f"ping not answered: {resp}"
            assert elapsed < 3.0, (
                f"ping took {elapsed:.1f}s while an embed was in flight — the "
                "accept loop is blocked (root cause #1 not fixed)."
            )
        finally:
            release.set()
            server.shutdown()
            t.join(timeout=3.0)
