"""EmbeddingBackend abstraction.

The only thing that needs to cross a process boundary for a shared model
server is the raw float-array output of ``model.encode(texts)``. This module
expresses that seam as a one-method :class:`EmbeddingBackend` protocol with two
implementations:

- :class:`InProcessBackend` — delegates to the existing in-process
  ``tldr.semantic.get_model().encode`` path (zero behaviour change).
- :class:`ServerClientBackend` — talks to a running model server over a
  JSON-newline Unix socket using :mod:`tldr.model_server.transport`.

:func:`get_default_backend` returns a :class:`ServerClientBackend` when the
server socket is reachable, else silently falls back to
:class:`InProcessBackend`.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from typing import Optional, Protocol, runtime_checkable

import numpy as np


class ServerUnavailableError(RuntimeError):
    """Raised when a :class:`ServerClientBackend` cannot reach the model server."""


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Abstract embedding computation seam.

    Implementations hide whether the model runs in-process or via server IPC.
    """

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:  # pyright: ignore[reportReturnType]
        """Return an ``(len(texts), dim)`` float32 matrix of embeddings.

        This is a :class:`typing.Protocol` method: it declares the structural
        contract that concrete backends (:class:`InProcessBackend`,
        :class:`ServerClientBackend`) must satisfy. The docstring above is the
        entire method body — a Protocol method has no runtime implementation,
        and concrete implementers supply the real ``encode``.
        """


class InProcessBackend:
    """Embed via the existing in-process model (``tldr.semantic.get_model``).

    Identical to today's path — the model is loaded lazily on first
    :meth:`encode` and outputs are L2-normalized.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        # Imported lazily so tests can monkeypatch tldr.semantic.get_model and
        # so importing this module never pulls in heavy ML dependencies.
        from tldr.semantic import get_model

        model = get_model(self.model_name, device=self.device)
        result = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(result, dtype=np.float32)


class ServerClientBackend:
    """Embed via IPC to a running model server over a Unix socket."""

    # Generous round-trip timeout for an embed request. The FIRST embed after a
    # server start (or after an idle unload) triggers a lazy model load that can
    # take tens of seconds; subsequent batch encodes also routinely exceed the
    # short *connect* timeout. We must NOT let connect_unix's 2s connect timeout
    # leak onto the recv — that would make every real embed fail spuriously.
    _EMBED_TIMEOUT = 600.0

    def __init__(self, server_socket: str, embed_timeout: float = _EMBED_TIMEOUT) -> None:
        self.server_socket = server_socket
        self.embed_timeout = embed_timeout

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        from tldr.model_server import transport

        request_id = uuid.uuid4().hex
        try:
            sock = transport.connect_unix(self.server_socket)
        except OSError as exc:
            raise ServerUnavailableError(
                f"model server socket unreachable: {self.server_socket}"
            ) from exc

        # connect_unix leaves a short (2s) connect timeout on the socket; reset
        # it to the generous embed timeout so a slow first-load / large batch does
        # not trip socket.timeout mid-request.
        sock.settimeout(self.embed_timeout)

        try:
            transport.send_message(
                sock,
                {
                    "cmd": "embed",
                    "texts": list(texts),
                    "batch_size": batch_size,
                    "request_id": request_id,
                },
            )
            resp = transport.recv_message(sock)
        except OSError as exc:
            raise ServerUnavailableError(
                f"model server I/O failed: {self.server_socket}"
            ) from exc
        finally:
            # Best-effort close; a socket already torn down by the peer is
            # fine to ignore — we have the response (or are unwinding).
            with contextlib.suppress(OSError):
                sock.close()

        if resp.get("status") != "ok":
            raise ServerUnavailableError(
                f"model server error: {resp.get('error', resp.get('status'))}"
            )
        return np.asarray(resp.get("vectors", []), dtype=np.float32)


def get_default_backend(server_socket: Optional[str] = None) -> EmbeddingBackend:
    """Return a server-backed backend if reachable, else in-process (silent).

    Never raises: a missing/refusing socket falls back to
    :class:`InProcessBackend` transparently. Subsequent calls to
    :meth:`~EmbeddingBackend.encode` on the returned backend may raise
    :exc:`ServerUnavailableError` if the server socket is closed later.
    """
    if server_socket:
        # Socket missing or refusing: silently fall through to the in-process
        # backend (documented zero-behaviour-change fallback).
        with contextlib.suppress(OSError):
            from tldr.model_server import transport

            sock = transport.connect_unix(server_socket)
            sock.close()
            return ServerClientBackend(server_socket=server_socket)
    return InProcessBackend()


def get_server_backed_default(force: bool = False) -> EmbeddingBackend:
    """Ensure the shared model server is up and return a backend that uses it.

    Delegation to the shared server is opt-in:

    - ``force=True`` — the in-daemon embedding path, which always delegates by
      design (a per-project daemon never holds its own model copy).
    - else ``TLDR_USE_MODEL_SERVER`` env set — used for the daemon's reindex
      subprocess (``python -m tldr.cli semantic index``) so it delegates too.
    - otherwise — return :class:`InProcessBackend` WITHOUT starting a server.

    Never raises and never leaves the caller without a backend: if the server
    cannot be started or reached, it falls back silently to in-process embedding
    (the documented zero-behaviour-change fallback). This keeps the ordinary
    (non-daemon) path and the test suite from ever spawning a model server.
    """
    if not force and not os.environ.get("TLDR_USE_MODEL_SERVER"):
        return InProcessBackend()
    try:
        from tldr.daemon.ensure import ensure_server

        socket_path = ensure_server()
    except Exception:
        return InProcessBackend()
    return get_default_backend(socket_path)
