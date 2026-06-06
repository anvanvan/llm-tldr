"""tldr.model_server: shared embedding model server subpackage.

Exposes the Unix-socket model server (:class:`ModelServer`) plus its
independently testable parts: :class:`EmbedQueue`, :class:`ModelServerLifecycle`,
and the JSON-newline transport helpers.
"""

from __future__ import annotations

from .lifecycle import ModelServerLifecycle
from .queue import EmbedQueue
from .server import ModelServer, default_socket_path
from .transport import connect_unix, recv_message, send_message

__all__ = [
    "ModelServer",
    "default_socket_path",
    "EmbedQueue",
    "ModelServerLifecycle",
    "send_message",
    "recv_message",
    "connect_unix",
]
