"""ModelServerLifecycle: lazy model load + rolling idle unload.

API: ``get_model()`` (lazy load), ``touch()`` (extend rolling deadline),
``maybe_unload()`` (unload if idle window elapsed), ``is_loaded`` property.

There is intentionally NO ``load_model`` / ``unload_model`` method — load is
implicit in ``get_model`` and unload is driven by ``maybe_unload``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional


def _default_model_factory() -> Any:
    """Load the real embedding model via the existing in-process path."""
    from tldr.semantic import get_model

    return get_model()


def _default_model_unloader() -> None:
    """Free the real model from the in-process cache + device memory.

    ``get_model`` caches the model in module-level globals, so dropping the
    lifecycle's own reference is not enough to reclaim memory — delegate to
    :func:`tldr.semantic.unload_model`, which clears that cache and empties the
    GPU/MPS allocator.
    """
    from tldr.semantic import unload_model

    unload_model()


class ModelServerLifecycle:
    """Track the embedding model and its rolling idle deadline.

    Args:
        idle_seconds: Idle window length; ``maybe_unload`` unloads once the
            monotonic clock passes the rolling deadline.
        model_factory: Zero-arg callable returning the model (testable seam).
        model_unloader: Zero-arg callable that frees the model from the
            underlying cache + device memory on unload (testable seam). Defaults
            to clearing tldr.semantic's module-level model cache; dropping the
            lifecycle's own reference alone would NOT reclaim the ~1 GB.
    """

    def __init__(
        self,
        idle_seconds: int = 1800,
        model_factory: Optional[Callable[[], Any]] = None,
        model_unloader: Optional[Callable[[], None]] = None,
    ) -> None:
        self._idle_seconds = idle_seconds
        self._model_factory = model_factory or _default_model_factory
        self._model_unloader = model_unloader or _default_model_unloader
        self._model: Any = None
        self._deadline: float = time.monotonic() + idle_seconds
        self._lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    def get_model(self) -> Any:
        """Return the model, loading it lazily on first use (or after unload)."""
        with self._lock:
            if self._model is None:
                self._model = self._model_factory()
            self._deadline = time.monotonic() + self._idle_seconds
            return self._model

    def touch(self) -> None:
        """Extend the rolling idle deadline to now + idle_seconds."""
        with self._lock:
            self._deadline = time.monotonic() + self._idle_seconds

    def maybe_unload(self) -> bool:
        """Unload the model if the idle deadline has elapsed.

        Drops the lifecycle's model reference and invokes the injected unloader
        to free the underlying cache + device memory. Never raises: a failing
        unloader is caught.

        Returns True only when a loaded model was unloaded AND the unloader
        completed cleanly (device memory reclaimed). If the unloader raises, the
        lifecycle reference is still dropped (so the model reloads lazily on next
        use), but this returns False to signal the underlying free did not
        complete — the caller can log/track an incomplete reclaim.
        """
        with self._lock:
            if self._model is None:
                return False
            if time.monotonic() > self._deadline:
                self._model = None
                # Drop the lifecycle reference AND free the underlying cached
                # model + device memory. A failing unloader must never propagate
                # out of the idle accept-loop tick — catch it and report the
                # incomplete reclaim via a False return.
                try:
                    self._model_unloader()
                except Exception:
                    return False
                return True
            return False
