"""EmbedQueue: thread-safe single-job FIFO queue.

Guarantees exactly one embed job runs at a time across all callers.
``submit()`` returns a ``concurrent.futures.Future`` resolving to the
embed function's result; a single background worker thread drains the
queue one job at a time, preserving submission order.
"""

from __future__ import annotations

import queue as _queue
import threading
from concurrent.futures import CancelledError, Future
from typing import Callable, List, Optional, Protocol, runtime_checkable

import numpy as np

_SHUTDOWN = object()


@runtime_checkable
class EmbedQueueProtocol(Protocol):
    """Structural protocol for an embed queue seam.

    Both the real :class:`EmbedQueue` and test doubles satisfy this
    protocol, allowing ``ModelServer._embed_queue`` to be typed as
    ``EmbedQueueProtocol`` so test fakes are assignable without a
    ``type: ignore``.
    """

    def submit(
        self,
        texts: List[str],
        cancelled: Optional[threading.Event] = None,
    ) -> Future:
        """Enqueue texts; return a Future resolving to the embed result."""
        ...

    def shutdown(self) -> None:
        """Drain queued jobs and stop the worker."""
        ...


class EmbedQueue:
    """Single-worker FIFO queue serializing embed jobs.

    Args:
        embed_fn: Callable taking ``list[str]`` and returning ``np.ndarray``.
    """

    def __init__(self, embed_fn: Callable[[List[str]], "np.ndarray"]) -> None:
        self._embed_fn = embed_fn
        self._queue: "_queue.Queue" = _queue.Queue()
        self._shutdown = False
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def submit(
        self,
        texts: List[str],
        cancelled: "Optional[threading.Event]" = None,
    ) -> "Future":
        """Enqueue an embed job; return a Future resolving to its ndarray result.

        Args:
            texts: The strings to embed.
            cancelled: Optional per-job liveness token. When set *before* the
                worker dequeues this job, the job is skipped (its Future is
                resolved with ``CancelledError``) and ``embed_fn`` is never
                called — tying the job's GPU work to its originating client's
                liveness. ``None`` (the default) means "never cancel", i.e. the
                pre-existing fire-and-forget behavior.
        """
        future: "Future" = Future()
        self._queue.put((texts, future, cancelled))
        return future

    def _run_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._queue.task_done()
                return
            texts, future, cancelled = item
            try:
                # Liveness guard: skip orphaned jobs whose originating client
                # died before dequeue. Checked AFTER dequeue, BEFORE embed_fn —
                # the in-flight job is uncancellable, but not-yet-started jobs
                # never touch the GPU. No MPS teardown here (60bce6d invariant).
                if cancelled is not None and cancelled.is_set():
                    future.set_exception(
                        CancelledError("embed job cancelled before dequeue")
                    )
                    continue
                result = self._embed_fn(texts)
                future.set_result(result)
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            finally:
                self._queue.task_done()

    def shutdown(self) -> None:
        """Drain already-submitted jobs, then stop the worker thread."""
        if self._shutdown:
            return
        self._shutdown = True
        self._queue.put(_SHUTDOWN)
        self._worker.join()
