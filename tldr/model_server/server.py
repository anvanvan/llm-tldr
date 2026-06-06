"""ModelServer: Unix-socket accept loop dispatching embed jobs.

Binds a Unix-domain socket, accepts client connections, reads JSON-newline
``embed`` requests, dispatches them through a single-job :class:`EmbedQueue`
backed by :class:`ModelServerLifecycle`, and writes back the resulting
vectors. ``shutdown()`` cleanly unblocks ``run()``.
"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
from typing import Any, List

import numpy as np

from .lifecycle import ModelServerLifecycle
from .queue import EmbedQueue
from .transport import recv_message, send_message

_ACCEPT_TIMEOUT = 0.2


def default_socket_path() -> str:
    """Return the conventional model-server socket path for this user.

    Uses the platform temp dir (matching ``tldr.daemon.ensure`` and the
    per-project daemon convention in ``tldr.daemon.startup``) so the path the
    server binds and the path ``ensure_server`` pings can never drift. On macOS
    ``tempfile.gettempdir()`` is the per-user ``/var/folders/.../T`` dir, not
    ``/tmp``.
    """
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return os.path.join(tempfile.gettempdir(), f"tldr-model-server-{uid}.sock")


class ModelServer:
    """Accept loop on a Unix socket; serialize embed jobs through EmbedQueue.

    Args:
        socket_path: Path to bind the Unix-domain socket on.
        idle_seconds: Idle window for the model lifecycle (rolling unload).
    """

    def __init__(self, socket_path: str, idle_seconds: int = 1800) -> None:
        self.socket_path = socket_path
        self._idle_seconds = idle_seconds
        self._lifecycle = ModelServerLifecycle(idle_seconds=idle_seconds)
        self._embed_queue = EmbedQueue(embed_fn=self._embed)
        self._stop = threading.Event()
        self._server_sock: socket.socket | None = None

    def _embed(self, texts: List[str]) -> "np.ndarray":
        model = self._lifecycle.get_model()
        self._lifecycle.touch()
        return model.encode(texts, normalize_embeddings=True)

    def run(self) -> None:
        """Bind the socket and serve until ``shutdown()`` is called."""
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._bind_socket(srv)
        srv.listen(8)
        srv.settimeout(_ACCEPT_TIMEOUT)
        self._server_sock = srv

        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    # Idle tick: enforce the rolling model-unload window. The
                    # accept timeout (_ACCEPT_TIMEOUT) is our heartbeat — when no
                    # client connects, check whether the model has been idle past
                    # its deadline and unload it to free GPU/memory.
                    self._lifecycle.maybe_unload()
                    continue
                except OSError:
                    break
                self._handle_connection(conn)
        finally:
            srv.close()
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass

    def _bind_socket(self, srv: socket.socket) -> None:
        """Bind ``srv`` to ``self.socket_path``, tolerating long paths.

        Unix ``sun_path`` is capped (~104 bytes on macOS). When the full path
        exceeds that limit, chdir into the socket's directory and bind by the
        basename, then restore the working directory. The resulting socket file
        still lives at the full path, so clients connect normally.
        """
        try:
            srv.bind(self.socket_path)
            return
        except OSError:
            pass
        directory = os.path.dirname(self.socket_path) or "."
        name = os.path.basename(self.socket_path)
        prev_cwd = os.getcwd()
        try:
            os.chdir(directory)
            srv.bind(name)
        finally:
            os.chdir(prev_cwd)

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            req = recv_message(conn)
            response = self._dispatch(req)
            send_message(conn, response)
        except Exception as exc:  # noqa: BLE001
            try:
                send_message(conn, {"status": "error", "error": str(exc)})
            except OSError:
                pass
        finally:
            conn.close()

    def _dispatch(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "embed":
            texts = req.get("texts", [])
            future = self._embed_queue.submit(texts)
            vecs = future.result()
            arr = np.asarray(vecs)
            return {
                "status": "ok",
                "vectors": arr.tolist(),
                "request_id": req.get("request_id"),
            }
        if cmd == "ping":
            return {"status": "ok"}
        return {"status": "error", "error": f"unknown cmd: {cmd!r}"}

    def shutdown(self) -> None:
        """Signal the accept loop to stop and tear down the queue."""
        self._stop.set()
        try:
            self._embed_queue.shutdown()
        except Exception:  # noqa: BLE001
            pass
