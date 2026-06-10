"""
Tests for tldr/model_server/ subpackage: EmbedQueue, ModelServerLifecycle,
ModelServerTransport, and ModelServer accept loop.

All tests are RED on HEAD because:
  1. tldr/model_server/ does not exist yet (no __init__.py, queue.py,
     lifecycle.py, transport.py, server.py)
  2. None of the public APIs (EmbedQueue, ModelServerLifecycle,
     ModelServerTransport, ModelServer) exist

Test strategy:
- transport: socket.socketpair() loopback — no mocks needed
- queue: fake callable returning np.ones((n,4)); threading.Event latches
  to observe overlap; no ML
- lifecycle: fake model factory + monkeypatched time.monotonic; zero ML
- server: real Unix socket in tmp_path with a fake embed callable
"""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 4


def _make_fake_model() -> MagicMock:
    """Return a MagicMock whose .encode() returns L2-normalised dim-4 vectors."""
    mock = MagicMock()

    def _encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False):
        n = len(texts) if isinstance(texts, (list, tuple)) else 1
        vecs = np.ones((n, _DIM), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock.encode.side_effect = _encode
    return mock


# ===========================================================================
# SECTION 1 — ModelServerTransport (tldr/model_server/transport.py)
# ===========================================================================


class TestTransportFraming:
    """JSON-newline framing send/recv round-trips over a socket pair.

    RED: tldr.model_server.transport does not exist — ImportError expected.
    """

    def test_send_recv_round_trip_identical_dict(self):
        """send_message + recv_message on a socketpair returns the original dict.

        RED reason: tldr.model_server.transport does not exist.
        """
        from tldr.model_server.transport import send_message, recv_message  # noqa: F401

        a, b = socket.socketpair()
        try:
            payload = {"cmd": "embed", "texts": ["hello", "world"], "request_id": "test-1"}
            send_message(a, payload)
            received = recv_message(b)
        finally:
            a.close()
            b.close()

        assert received == payload, (
            f"recv_message must return the exact dict sent by send_message. "
            f"Expected {payload!r}, got {received!r}."
        )

    def test_send_recv_roundtrip_nested_types(self):
        """Round-trip of a dict with int, float, list, and None values.

        RED reason: tldr.model_server.transport does not exist.
        """
        from tldr.model_server.transport import send_message, recv_message  # noqa: F401

        a, b = socket.socketpair()
        try:
            payload = {
                "status": "ok",
                "vectors": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "count": 2,
                "error": None,
            }
            send_message(a, payload)
            received = recv_message(b)
        finally:
            a.close()
            b.close()

        assert received == payload, (
            f"Nested-type round-trip failed. Expected {payload!r}, got {received!r}."
        )

    def test_connect_unix_raises_on_nonexistent_socket(self, tmp_path: Path):
        """connect_unix raises OSError/FileNotFoundError for a missing socket path.

        RED reason: tldr.model_server.transport does not exist.
        """
        from tldr.model_server.transport import connect_unix  # noqa: F401

        missing = str(tmp_path / "no_such.sock")
        with pytest.raises((FileNotFoundError, ConnectionRefusedError, OSError)):
            connect_unix(missing, timeout=0.5)


# ===========================================================================
# SECTION 2 — EmbedQueue (tldr/model_server/queue.py)
# ===========================================================================


class TestEmbedQueueSingleJobInvariant:
    """EmbedQueue must run at most one embed job at a time.

    RED: tldr.model_server.queue does not exist — ImportError expected.
    """

    def test_only_one_job_runs_concurrently(self):
        """Submit 3 concurrent jobs; assert at most one runs at a time.

        Uses a threading.Event latch inside the fake embed to block the first
        job until we can observe overlap count.

        RED reason: tldr.model_server.queue.EmbedQueue does not exist.
        """
        from tldr.model_server.queue import EmbedQueue  # noqa: F401

        overlap_counter = [0]  # mutable int to track concurrency
        max_overlap = [0]
        lock = threading.Lock()
        unblock = threading.Event()

        def fake_embed(texts: list[str]) -> np.ndarray:
            with lock:
                overlap_counter[0] += 1
                if overlap_counter[0] > max_overlap[0]:
                    max_overlap[0] = overlap_counter[0]
            # Block first call until we release
            unblock.wait(timeout=5.0)
            with lock:
                overlap_counter[0] -= 1
            return np.ones((len(texts), _DIM), dtype=np.float32)

        q = EmbedQueue(embed_fn=fake_embed)

        # Submit 3 jobs from separate threads
        futures: list[Future] = []
        for i in range(3):
            f = q.submit([f"text-{i}"])
            futures.append(f)

        # Give threads a moment to pile up, then unblock all
        time.sleep(0.05)
        unblock.set()

        # Collect all results (with timeout)
        for f in futures:
            result = f.result(timeout=5.0)
            assert result.shape == (1, _DIM), f"Expected shape (1, {_DIM}), got {result.shape}"

        assert max_overlap[0] == 1, (
            f"EmbedQueue must serialize jobs (max_overlap must be 1). "
            f"Got max_overlap={max_overlap[0]}. "
            f"RED: EmbedQueue does not exist; would be ImportError."
        )

    def test_futures_all_resolve_to_ndarray(self):
        """All submitted futures resolve to np.ndarray with correct shape.

        RED reason: tldr.model_server.queue.EmbedQueue does not exist.
        """
        from tldr.model_server.queue import EmbedQueue  # noqa: F401

        def fake_embed(texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), _DIM), dtype=np.float32)

        q = EmbedQueue(embed_fn=fake_embed)
        texts_list = [["a"], ["b", "c"], ["d", "e", "f"]]
        futures = [q.submit(t) for t in texts_list]
        for i, (f, texts) in enumerate(zip(futures, texts_list)):
            result = f.result(timeout=5.0)
            assert isinstance(result, np.ndarray), (
                f"Future {i} must resolve to np.ndarray, got {type(result)}"
            )
            assert result.shape == (len(texts), _DIM), (
                f"Future {i}: expected shape ({len(texts)}, {_DIM}), got {result.shape}"
            )

    def test_shutdown_drains_queue_cleanly(self):
        """EmbedQueue.shutdown() drains pending jobs and stops the run-loop.

        RED reason: tldr.model_server.queue.EmbedQueue does not exist.
        """
        from tldr.model_server.queue import EmbedQueue  # noqa: F401

        def fast_embed(texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), _DIM), dtype=np.float32)

        q = EmbedQueue(embed_fn=fast_embed)
        f = q.submit(["hello"])
        q.shutdown()
        # After shutdown, the already-submitted future must still resolve
        result = f.result(timeout=5.0)
        assert result is not None, "shutdown() must not discard already-submitted jobs"


class TestEmbedQueueDropsCancelledJobs:
    """EmbedQueue must skip queued jobs whose per-job cancel Event is set.

    Regression for the embed orphan-drain bug: when all clients mass-die, the
    single worker drains the entire queued backlog on the GPU for dead readers.
    The fix threads a ``cancelled: threading.Event | None`` token through
    ``submit()`` and guards it at dequeue, *before* ``embed_fn``: a job whose
    Event is set before it is dequeued is resolved with an exception and NEVER
    embedded.

    RED on current code: ``submit()`` accepts only ``texts`` (no ``cancelled``
    arg) and ``_run_loop`` runs every job unconditionally — so this either
    raises TypeError on the extra arg or the worker ignores the token and runs
    all N jobs (counter == N, not 1).
    """

    def test_cancelled_queued_jobs_are_skipped_not_embedded(self):
        """A backlog whose jobs are cancelled before dequeue runs only the in-flight job.

        Arrange: an EmbedQueue with a slow stub embed_fn that signals when the
        in-flight job has started, then blocks until released, counting every
        real call. Submit N jobs, each with its own threading.Event cancel token.
        Act: let only job 1 enter embed_fn, set the Events of jobs 2..N
        out-of-band (before the worker dequeues them), then release job 1.
        Assert: embed_fn ran exactly once (in-flight only); the N-1 cancelled
        futures raise (CancelledError-equivalent), NOT a successful ndarray.

        RED reason: current submit() has no `cancelled` param and _run_loop
        honors no liveness token — orphaned jobs all run (counter == N).
        """
        from tldr.model_server.queue import EmbedQueue  # noqa: F401

        n_jobs = 10

        embed_calls = [0]
        call_lock = threading.Lock()
        in_flight_started = threading.Event()  # signalled once job 1 is inside embed_fn
        release = threading.Event()            # holds job 1 in flight until set

        def slow_embed(texts: list[str]) -> np.ndarray:
            with call_lock:
                embed_calls[0] += 1
            in_flight_started.set()
            # Keep the first (in-flight) job inside embed_fn long enough for us
            # to cancel the rest of the backlog out-of-band before dequeue.
            release.wait(timeout=5.0)
            return np.ones((len(texts), _DIM), dtype=np.float32)

        q = EmbedQueue(embed_fn=slow_embed)

        cancel_events = [threading.Event() for _ in range(n_jobs)]
        futures: list[Future] = []
        for i in range(n_jobs):
            f = q.submit([f"text-{i}"], cancelled=cancel_events[i])
            futures.append(f)

        # Wait until job 1 is actually inside embed_fn (deterministic latch).
        assert in_flight_started.wait(timeout=5.0), (
            "in-flight job never entered embed_fn — worker did not start"
        )

        # Originating clients die: cancel every still-queued job (2..N) BEFORE
        # the worker dequeues them. Job 1 is already in flight and uncancellable.
        for ev in cancel_events[1:]:
            ev.set()

        # Release the in-flight job so the worker can drain the rest of the queue.
        release.set()

        # The in-flight job (job 1) resolves to a real ndarray.
        head = futures[0].result(timeout=5.0)
        assert isinstance(head, np.ndarray), (
            f"In-flight job must resolve to an ndarray, got {type(head)}"
        )

        # Every cancelled job (2..N) must raise — NOT return a successful result.
        for i, f in enumerate(futures[1:], start=1):
            with pytest.raises(BaseException) as excinfo:
                f.result(timeout=5.0)
            assert "Cancel" in type(excinfo.value).__name__, (
                f"Cancelled job {i} must raise a CancelledError-equivalent, "
                f"got {type(excinfo.value).__name__}: {excinfo.value!r}"
            )

        # The decisive assertion: embed_fn ran ONLY for the in-flight job.
        assert embed_calls[0] == 1, (
            f"EmbedQueue must skip cancelled queued jobs at dequeue: embed_fn "
            f"should run exactly once (the in-flight job), but ran "
            f"{embed_calls[0]} times — orphaned jobs were drained on the GPU."
        )

        q.shutdown()


# ===========================================================================
# SECTION 3 — ModelServerLifecycle (tldr/model_server/lifecycle.py)
# ===========================================================================


class TestModelServerLifecycle:
    """ModelServerLifecycle: lazy load, idle unload, rolling deadline.

    RED: tldr.model_server.lifecycle does not exist — ImportError expected.
    """

    def test_model_loads_lazily_on_first_get_model(self):
        """get_model() loads the model; is_loaded becomes True after first call.

        RED reason: tldr.model_server.lifecycle.ModelServerLifecycle does not exist.
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        fake = _make_fake_model()
        load_count = [0]

        def fake_factory():
            load_count[0] += 1
            return fake

        lifecycle = ModelServerLifecycle(idle_seconds=60, model_factory=fake_factory)
        assert not lifecycle.is_loaded, "Model must NOT be loaded before first get_model()"

        model = lifecycle.get_model()
        assert model is fake, "get_model() must return the model from the factory"
        assert lifecycle.is_loaded, "is_loaded must be True after get_model()"
        assert load_count[0] == 1, "Factory must be called exactly once on first get_model()"

    def test_maybe_unload_returns_false_before_idle_deadline(self):
        """maybe_unload() returns False when idle deadline has not elapsed.

        RED reason: tldr.model_server.lifecycle.ModelServerLifecycle does not exist.
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        fake = _make_fake_model()
        lifecycle = ModelServerLifecycle(idle_seconds=3600, model_factory=lambda: fake)
        lifecycle.get_model()  # load
        lifecycle.touch()

        # Freeze monotonic at t=0, then advance only 100s (< 3600s idle)
        t = [1000.0]

        def fake_monotonic():
            return t[0]

        with patch("time.monotonic", side_effect=fake_monotonic):
            lifecycle.touch()   # sets deadline to t[0] + 3600
            t[0] = 1100.0       # advance 100s — still within idle window
            result = lifecycle.maybe_unload()

        assert result is False, (
            f"maybe_unload() must return False when deadline not elapsed, got {result!r}"
        )
        assert lifecycle.is_loaded, "Model must remain loaded when idle deadline not elapsed"

    def test_maybe_unload_returns_true_after_idle_window(self):
        """maybe_unload() returns True and unloads model after idle window expires.

        RED reason: tldr.model_server.lifecycle.ModelServerLifecycle does not exist.
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        fake = _make_fake_model()
        lifecycle = ModelServerLifecycle(idle_seconds=30, model_factory=lambda: fake)
        lifecycle.get_model()   # load

        t = [5000.0]

        def fake_monotonic():
            return t[0]

        with patch("time.monotonic", side_effect=fake_monotonic):
            lifecycle.touch()    # sets deadline to 5000 + 30
            t[0] = 5031.0        # advance 31s — past idle window
            result = lifecycle.maybe_unload()

        assert result is True, (
            f"maybe_unload() must return True when idle deadline elapsed, got {result!r}"
        )
        assert not lifecycle.is_loaded, "Model must be unloaded after idle window expires"

    def test_touch_extends_rolling_idle_deadline(self):
        """Each touch() call extends the idle deadline (rolling window).

        Scenario:
          - idle_seconds = 30
          - t=0: touch() → deadline=30
          - t=20: touch() → deadline=50  (rolling extension)
          - t=40: maybe_unload() → False (within extended deadline)
          - t=51: maybe_unload() → True

        RED reason: tldr.model_server.lifecycle.ModelServerLifecycle does not exist.
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        fake = _make_fake_model()
        lifecycle = ModelServerLifecycle(idle_seconds=30, model_factory=lambda: fake)
        lifecycle.get_model()  # load

        t = [0.0]

        def fake_monotonic():
            return t[0]

        with patch("time.monotonic", side_effect=fake_monotonic):
            lifecycle.touch()       # deadline = 0 + 30 = 30
            t[0] = 20.0
            lifecycle.touch()       # deadline = 20 + 30 = 50  (rolling)
            t[0] = 40.0
            result_before = lifecycle.maybe_unload()   # 40 < 50 → False
            t[0] = 51.0
            result_after = lifecycle.maybe_unload()    # 51 > 50 → True

        assert result_before is False, (
            "maybe_unload() must return False at t=40 when rolling deadline is t=50"
        )
        assert result_after is True, (
            "maybe_unload() must return True at t=51 when rolling deadline was t=50"
        )

    def test_get_model_reloads_after_unload(self):
        """get_model() reloads the model after unload; factory called a second time.

        RED reason: tldr.model_server.lifecycle.ModelServerLifecycle does not exist.
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        fake = _make_fake_model()
        load_count = [0]

        def factory():
            load_count[0] += 1
            return fake

        lifecycle = ModelServerLifecycle(idle_seconds=1, model_factory=factory)
        lifecycle.get_model()   # load #1

        t = [0.0]

        def fake_monotonic():
            return t[0]

        with patch("time.monotonic", side_effect=fake_monotonic):
            lifecycle.touch()
            t[0] = 2.0
            lifecycle.maybe_unload()   # unloads

        assert not lifecycle.is_loaded, "Must be unloaded after idle"
        lifecycle.get_model()          # load #2 — reloads
        assert lifecycle.is_loaded, "get_model() must reload after unload"
        assert load_count[0] == 2, (
            f"Factory must be called again on reload. Expected 2, got {load_count[0]}"
        )

    def test_concurrent_get_model_and_maybe_unload_no_crash(self):
        """Concurrent get_model()/maybe_unload() must not crash; model reloads.

        The RLock in ModelServerLifecycle serializes load/unload so an idle
        unload racing with an incoming request cannot corrupt the model slot.
        We force every maybe_unload() to be eligible (deadline already past via
        an injected monotonic clock) so the race window is maximally exercised,
        and assert that get_model() always returns a usable model and the
        process never raises.

        The idle window is exercised via an injected clock (no real sleeping).
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        load_count = [0]
        count_lock = threading.Lock()

        def factory():
            with count_lock:
                load_count[0] += 1
            return _make_fake_model()

        lifecycle = ModelServerLifecycle(idle_seconds=0, model_factory=factory)

        errors: list[BaseException] = []
        stop = threading.Event()

        # A strictly-monotonic fake clock: every read advances by a tick, so
        # any maybe_unload() that runs after a get_model() (which set the
        # deadline to now + idle_seconds == now) sees now > deadline and
        # GENUINELY unloads — and the next get_model() reloads. This exercises
        # the documented unload/reload race deterministically with no real sleep.
        clock = [0.0]
        clock_lock = threading.Lock()

        def fake_monotonic() -> float:
            with clock_lock:
                clock[0] += 1.0
                return clock[0]

        unload_count = [0]
        unload_lock = threading.Lock()

        real_maybe_unload = lifecycle.maybe_unload

        def counting_maybe_unload() -> bool:
            unloaded = real_maybe_unload()
            if unloaded:
                with unload_lock:
                    unload_count[0] += 1
            return unloaded

        with patch("time.monotonic", side_effect=fake_monotonic):

            def getter():
                try:
                    while not stop.is_set():
                        model = lifecycle.get_model()
                        # get_model must always return a usable model object
                        assert model is not None
                        _ = model.encode(["x"])
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            def unloader():
                try:
                    while not stop.is_set():
                        counting_maybe_unload()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=getter) for _ in range(4)]
            threads += [threading.Thread(target=unloader) for _ in range(4)]
            for th in threads:
                th.start()
            # Bounded busy wait (no real sleep semantics needed): let the
            # threads churn through enough iterations to force unload+reload.
            deadline = time.time() + 2.0
            while time.time() < deadline and load_count[0] < 2:
                pass
            stop.set()
            for th in threads:
                th.join(timeout=5.0)

        assert not errors, f"Concurrent get_model/maybe_unload raised: {errors!r}"
        # maybe_unload() must have GENUINELY unloaded a loaded model at least
        # once (the documented idle-unload path actually fired, not just a
        # no-op thread-safety check).
        assert unload_count[0] >= 1, (
            "maybe_unload() must actually unload a loaded model at least once"
        )
        # The model must have reloaded after an unload (proving get_model
        # recovers when _model is None under contention).
        assert load_count[0] >= 2, (
            "Model must load AND reload under contention (unload then reload)"
        )

    def test_concurrent_get_model_and_maybe_unload_is_threadsafe(self):
        """Concurrent get_model() and maybe_unload() from many threads raise no exception.

        Exercises the threading.RLock that guards get_model/touch/maybe_unload
        (architecture edge case: "Server idle unload races with incoming embed").

        16 threads (8 getters + 8 unloaders) each run 50 bounded iterations.
        idle_seconds=0 so maybe_unload() is always eligible; the lifecycle
        oscillates between loaded and unloaded under contention.
        After the storm, get_model() must return a non-None model.
        No real sleeping — idle window kept at 0.

        RED reason: tldr.model_server.lifecycle.ModelServerLifecycle does not exist.
        """
        from tldr.model_server.lifecycle import ModelServerLifecycle  # noqa: F401

        fake = _make_fake_model()
        load_count = [0]
        count_lock = threading.Lock()

        def factory():
            with count_lock:
                load_count[0] += 1
            return fake

        lifecycle = ModelServerLifecycle(idle_seconds=0, model_factory=factory)

        thread_errors: list[Exception] = []
        err_lock = threading.Lock()

        def getter(iterations: int) -> None:
            try:
                for _ in range(iterations):
                    model = lifecycle.get_model()
                    assert model is not None, "get_model() must never return None"
                    lifecycle.touch()
            except Exception as exc:
                with err_lock:
                    thread_errors.append(exc)

        def unloader(iterations: int) -> None:
            try:
                for _ in range(iterations):
                    lifecycle.maybe_unload()
            except Exception as exc:
                with err_lock:
                    thread_errors.append(exc)

        threads = (
            [threading.Thread(target=getter, args=(50,)) for _ in range(8)]
            + [threading.Thread(target=unloader, args=(50,)) for _ in range(8)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not thread_errors, (
            f"Concurrent get_model()/maybe_unload() raised exceptions: "
            f"{[str(e) for e in thread_errors]}"
        )

        # After the storm, get_model() must return the live model (reloads if needed)
        final_model = lifecycle.get_model()
        assert final_model is not None, (
            "get_model() must return a valid (non-None) model after concurrent storm"
        )


# ===========================================================================
# SECTION 4 — ModelServer accept loop (tldr/model_server/server.py)
# ===========================================================================


class TestModelServerAcceptLoop:
    """ModelServer binds a Unix socket, accepts connections, dispatches to EmbedQueue.

    RED: tldr.model_server.server does not exist — ImportError expected.
    """

    def test_server_accepts_connection_and_processes_embed_request(self, tmp_path: Path):
        """ModelServer in a thread accepts a connection and returns an embed response.

        Uses a fake EmbedQueue (mock) and transport framing.
        The server must accept the connection and write a JSON response containing
        a 'vectors' key with the embedded result.

        RED reason: tldr.model_server.server.ModelServer does not exist.
        """
        from tldr.model_server.server import ModelServer  # noqa: F401
        from tldr.model_server.transport import send_message, recv_message, connect_unix  # noqa: F401

        sock_path = str(tmp_path / "model_server_test.sock")

        fake_vecs = np.ones((2, _DIM), dtype=np.float32)

        # Fake embed queue that returns immediately
        class FakeEmbedQueue:
            def submit(self, texts, cancelled=None):
                f: Future = Future()
                f.set_result(fake_vecs[:len(texts)])
                return f

            def shutdown(self):
                pass

        server = ModelServer(socket_path=sock_path, idle_seconds=3600)
        server._embed_queue = FakeEmbedQueue()  # inject fake queue

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # Wait for socket to appear
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if Path(sock_path).exists():
                break
            time.sleep(0.05)
        assert Path(sock_path).exists(), f"Server socket {sock_path} did not appear"

        # Connect and send embed request
        client_sock = connect_unix(sock_path, timeout=2.0)
        try:
            req = {"cmd": "embed", "texts": ["hello", "world"], "request_id": "r1"}
            send_message(client_sock, req)
            response = recv_message(client_sock)
        finally:
            client_sock.close()
            server.shutdown()

        assert "vectors" in response, (
            f"Server response must contain 'vectors' key. Got keys: {list(response.keys())}"
        )

    def test_server_shutdown_unblocks_run(self, tmp_path: Path):
        """shutdown() causes run() to exit cleanly (no hang).

        RED reason: tldr.model_server.server.ModelServer does not exist.
        """
        from tldr.model_server.server import ModelServer  # noqa: F401

        sock_path = str(tmp_path / "shutdown_test.sock")
        server = ModelServer(socket_path=sock_path, idle_seconds=3600)

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

        # Wait for socket to appear
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if Path(sock_path).exists():
                break
            time.sleep(0.05)

        server.shutdown()
        server_thread.join(timeout=5.0)

        assert not server_thread.is_alive(), (
            "server.shutdown() must cause run() to exit; thread still alive after 5s"
        )


# ===========================================================================
# SECTION 5 — Server socket-EOF cancellation wiring (A-4)
# ===========================================================================


class TestServerSocketEOFCancels:
    """Full end-to-end wiring: socket EOF → cancelled.set() → queue-skip.

    Uses a REAL ModelServer bound to a temp Unix socket with its embed leaf
    stubbed to a slow function so queued jobs accumulate. Forcibly closing
    client sockets before responses are read must cause the worker to STOP
    running queued jobs — the stub counter freezes at ~1 (in-flight only),
    not at the full backlog count.
    """

    def test_socket_eof_cancels_queued_embed_jobs(self, tmp_path: Path):
        """Closing client sockets halts orphaned queued jobs via the cancel wiring.

        Arrange: a real ModelServer whose ``_embed`` leaf is stubbed to a slow
        function (0.15 s/job) that increments a shared counter and gates the
        first job on a latch so we can guarantee the backlog is queued before
        any client dies.

        Act: submit N_JOBS embed requests from N_JOBS real socket clients (one
        per connection), wait until the first job is in flight, then forcibly
        close ALL client sockets without reading responses.

        Assert: within a generous bound the counter stops at ~1 (in-flight job
        only). The decisive assertion is counter <= 2 after disconnect: even if
        one extra job slips through the race window, the rest of the backlog must
        be skipped, not drained.
        """
        from tldr.model_server.server import ModelServer
        from tldr.model_server.transport import send_message, connect_unix

        N_JOBS = 12
        JOB_SLEEP = 0.15

        sock_path = str(tmp_path / "eof_cancel_test.sock")

        # --- Shared state for the stub ---
        embed_calls = [0]
        call_lock = threading.Lock()
        in_flight_started = threading.Event()  # set once job 1 is inside stub
        release_in_flight = threading.Event()  # released after clients are killed

        def slow_embed(texts):
            with call_lock:
                embed_calls[0] += 1
            in_flight_started.set()
            # Block the first (in-flight) job until we've closed all clients.
            release_in_flight.wait(timeout=10.0)
            time.sleep(JOB_SLEEP)
            return np.zeros((len(texts), _DIM), dtype=np.float32)

        # Patch the _embed class method BEFORE instantiation so the EmbedQueue
        # (created in __init__ with embed_fn=self._embed) captures the stub.
        # Mirror the reproduce_orphan_drain.py approach exactly.
        original_embed = ModelServer._embed  # type: ignore[attr-defined]
        ModelServer._embed = lambda self, texts: slow_embed(texts)  # type: ignore[method-assign]
        try:
            server = ModelServer(socket_path=sock_path, idle_seconds=3600)
        finally:
            ModelServer._embed = original_embed  # type: ignore[method-assign]

        client_sockets: list = []
        server_thread = threading.Thread(target=server.run, daemon=True,
                                         name="eof-cancel-test-server")
        server_thread.start()

        try:
            # Wait for socket to appear.
            deadline = time.time() + 5.0
            while time.time() < deadline:
                if Path(sock_path).exists():
                    break
                time.sleep(0.02)
            assert Path(sock_path).exists(), (
                f"Server socket {sock_path} did not appear within 5 s"
            )

            # Submit N_JOBS embed requests — one socket per request, none read.
            for i in range(N_JOBS):
                s = connect_unix(sock_path, timeout=5.0)
                req = {"cmd": "embed", "texts": [f"text-{i}"], "request_id": i}
                send_message(s, req)
                client_sockets.append(s)

            # Wait until the first job is inside the stub (queue is backlogged).
            assert in_flight_started.wait(timeout=10.0), (
                "In-flight job never entered stub — worker did not start"
            )

            # Snapshot the counter right before we kill clients.
            with call_lock:
                calls_before_kill = embed_calls[0]

            # Kill all client sockets — this is the EOF event that should
            # trigger cancelled.set() in _await_embed for every queued job.
            for s in client_sockets:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()
            client_sockets.clear()

            # Release the in-flight job so the worker can drain the queue.
            release_in_flight.set()

            # Wait for the worker to drain whatever remains (bounded).
            # A correct implementation stops after ~1-2 jobs; a broken one
            # runs all N_JOBS. Give it 3× a full-backlog run to be safe.
            time.sleep(N_JOBS * JOB_SLEEP * 0.4)

            with call_lock:
                final_calls = embed_calls[0]

        finally:
            server.shutdown()
            for s in client_sockets:
                try:
                    s.close()
                except OSError:
                    pass

        # The decisive assertion: most queued jobs were skipped.
        # Allow up to 2 to account for timing races (1 in-flight + 1 that may
        # have dequeued before the EOF was observed by _await_embed).
        assert final_calls <= 2, (
            f"Socket-EOF cancellation must halt queued jobs: expected at most 2 "
            f"embed calls (in-flight + possible race), got {final_calls}. "
            f"Before kill: {calls_before_kill}. "
            f"The socket-EOF → cancelled.set() → queue-skip wiring is broken."
        )
