"""EmbedQueue: thread-safe single-job FIFO queue.

Guarantees exactly one embed job runs at a time across all callers.
``submit()`` returns a ``concurrent.futures.Future`` resolving to the
embed function's result; a single background worker thread drains the
queue one job at a time, preserving submission order.
"""

from __future__ import annotations

import queue as _queue
import threading
from concurrent.futures import Future
from typing import Callable, List

import numpy as np

_SHUTDOWN = object()


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

    def submit(self, texts: List[str]) -> "Future":
        """Enqueue an embed job; return a Future resolving to its ndarray result."""
        future: "Future" = Future()
        self._queue.put((texts, future))
        return future

    def _run_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._queue.task_done()
                return
            texts, future = item
            try:
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
