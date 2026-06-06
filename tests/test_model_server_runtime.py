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


def test_maybe_unload_invokes_unloader_to_free_real_model():
    """maybe_unload must call the model_unloader so the cached model is actually
    freed (dropping the lifecycle's own reference alone does not reclaim memory:
    tldr.semantic.get_model caches the model in a module-level global)."""
    from tldr.model_server.lifecycle import ModelServerLifecycle

    unload_calls = []
    lc = ModelServerLifecycle(
        idle_seconds=0,  # deadline already elapsed → eligible to unload
        model_factory=lambda: object(),
        model_unloader=lambda: unload_calls.append(True),
    )
    lc.get_model()  # load
    assert lc.is_loaded
    unloaded = lc.maybe_unload()
    assert unloaded is True
    assert not lc.is_loaded
    assert unload_calls == [True], "maybe_unload must invoke the model_unloader"


def test_maybe_unload_returns_false_when_unloader_raises_but_still_drops_ref():
    """A failing unloader must NOT propagate out of the idle accept-loop tick, but
    maybe_unload should report the incomplete device-memory reclaim by returning
    False. The lifecycle reference is still dropped so the model reloads lazily."""
    from tldr.model_server.lifecycle import ModelServerLifecycle

    def boom():
        raise RuntimeError("device free failed")

    lc = ModelServerLifecycle(
        idle_seconds=0,  # deadline already elapsed → eligible to unload
        model_factory=lambda: object(),
        model_unloader=boom,
    )
    lc.get_model()  # load
    assert lc.is_loaded
    # Must not raise even though the unloader does.
    unloaded = lc.maybe_unload()
    assert unloaded is False, "a failing unloader must report False (incomplete reclaim)"
    assert not lc.is_loaded, "the lifecycle reference must still be dropped on unloader failure"


def test_semantic_unload_model_clears_module_cache():
    """semantic.unload_model() drops the module-level model cache so the ~1 GB is
    released (next get_model reloads). This is what the server's idle-unload needs."""
    import tldr.semantic as sem

    sentinel = object()
    sem._model = sentinel
    sem._model_name = "fake/model"
    sem._model_device = "metal"
    was_loaded = sem.unload_model()
    assert was_loaded is True
    assert sem._model is None and sem._model_name is None and sem._model_device is None
    # idempotent: a second call reports nothing was loaded
    assert sem.unload_model() is False


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
