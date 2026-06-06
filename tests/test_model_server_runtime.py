"""Runtime regression tests for the model-server embed path + idle unload.

These cover two real bugs the live Verify floor caught that mocks missed:

1. ServerClientBackend let connect_unix's short 2s *connect* timeout leak onto
   the embed recv, so any embed slower than 2s (every first model load, and most
   real batch encodes) failed with socket.timeout → ServerUnavailableError →
   the delegated index build failed (`index.faiss` never written).
2. ModelServer.run()'s accept loop never called lifecycle.maybe_unload(), so the
   rolling model auto-unload never fired.
"""

from __future__ import annotations

import socket

import numpy as np

from tldr.embedding_backend import ServerClientBackend
from tldr.model_server.server import ModelServer


def test_encode_resets_socket_timeout_for_slow_embed(monkeypatch):
    """encode() must override the short connect timeout with a generous one
    BEFORE recv, so a slow (>2s) server response does not trip socket.timeout."""
    recorded_timeouts = []
    response = {"status": "ok", "vectors": [[0.1, 0.2, 0.3]], "request_id": "x"}

    class FakeSock:
        def settimeout(self, t):
            recorded_timeouts.append(t)

        def close(self):
            pass

    fake = FakeSock()
    # Stub the transport so we exercise encode()'s timeout handling, not real IO.
    import tldr.model_server.transport as transport

    monkeypatch.setattr(transport, "connect_unix", lambda path, timeout=2.0: fake)
    monkeypatch.setattr(transport, "send_message", lambda sock, msg: None)
    monkeypatch.setattr(transport, "recv_message", lambda sock: response)

    backend = ServerClientBackend("/tmp/whatever.sock", embed_timeout=600.0)
    out = backend.encode(["hello"])

    assert isinstance(out, np.ndarray)
    # The generous embed timeout must have been applied (the fix); a leaked 2.0
    # connect timeout alone would be a regression.
    assert 600.0 in recorded_timeouts, (
        f"encode must reset the socket timeout to the generous embed timeout; "
        f"recorded settimeout calls: {recorded_timeouts}"
    )


def test_run_calls_maybe_unload_on_idle_tick(monkeypatch):
    """ModelServer.run()'s accept-timeout branch must call lifecycle.maybe_unload."""
    server = ModelServer(socket_path="/tmp/does-not-matter.sock")

    calls = []

    class FakeLifecycle:
        def maybe_unload(self):
            calls.append(True)
            server._stop.set()  # exit the loop after one idle tick
            return False

    server._lifecycle = FakeLifecycle()

    class FakeSrv:
        def bind(self, *a):
            pass

        def listen(self, *a):
            pass

        def settimeout(self, *a):
            pass

        def accept(self):
            raise socket.timeout()

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *a, **k: FakeSrv())
    # Avoid touching the filesystem for the (non-existent) socket path.
    monkeypatch.setattr("os.unlink", lambda *a, **k: None)

    server.run()

    assert calls == [True], "run() must call maybe_unload() on an idle accept timeout"
