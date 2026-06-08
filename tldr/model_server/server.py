"""ModelServer: Unix-socket accept loop dispatching embed jobs.

Binds a Unix-domain socket, accepts client connections, reads JSON-newline
``embed`` requests, dispatches them through a single-job :class:`EmbedQueue`
backed by :class:`ModelServerLifecycle`, and writes back the resulting
vectors. ``shutdown()`` cleanly unblocks ``run()``.
"""

from __future__ import annotations

import os
import signal
import socket
import tempfile
import threading
import time
from typing import List

import numpy as np

from .lifecycle import ModelServerLifecycle
from .queue import EmbedQueue
from .transport import recv_message, send_message

_ACCEPT_TIMEOUT = 0.2

# Real socket constructor captured at import time. The orphan-reap connect
# probe MUST use the genuine OS socket even if a test monkeypatches
# ``socket.socket`` (the accept-loop seam), so liveness detection of a real
# orphan is never confused by a fake server stand-in.
_raw_socket = socket.socket


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
        # Orphan reap: probe the socket before unlinking it. If a live peer
        # answers the connect, an old server still owns the path — reap it (read
        # its PID sidecar, SIGTERM, bounded wait, SIGKILL fallback) so exactly
        # one server survives instead of two silently fighting over the socket.
        probe = _raw_socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        live_peer = False
        try:
            probe.connect(self.socket_path)
            live_peer = True
        except OSError:
            live_peer = False
        finally:
            probe.close()
        if live_peer:
            self._reap_orphan()

        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._bind_socket(srv)
        srv.listen(8)
        srv.settimeout(_ACCEPT_TIMEOUT)
        self._server_sock = srv

        # PID sidecar: write OUR pid to {socket}.pid only AFTER reaping any
        # orphan and rebinding the socket. Writing it here (not before run())
        # is what makes orphan reap correct: when a second server starts, its
        # _reap_orphan() above reads the EXISTING sidecar — still the orphan's
        # pid — and SIGTERMs the orphan, never itself. Only once we own the
        # socket do we claim the sidecar.
        self._write_pid_sidecar()

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
            self._remove_pid_sidecar()
            # Free the model + device memory on EVERY exit path (clean
            # shutdown, SIGTERM, OSError/orphan), not only the idle-tick
            # branch. maybe_unload() honors the rolling deadline; the explicit
            # unloader call below guarantees torch.mps.empty_cache() runs even
            # when no idle window elapsed so a dying server never strands ~1 GB.
            try:
                self._lifecycle.maybe_unload()
            except Exception:  # noqa: BLE001
                pass
            unloader = getattr(self._lifecycle, "_model_unloader", None)
            if callable(unloader):
                try:
                    unloader()
                except Exception:  # noqa: BLE001
                    pass

    def _reap_orphan(self) -> None:
        """Reap the live server currently holding ``self.socket_path``.

        Called only after a connect probe in ``run()`` confirmed a live peer.
        Read ``{socket_path}.pid``, confirm the process is alive
        (``os.kill(pid, 0)``), SIGTERM it, poll up to 5 s for a clean exit, then
        SIGKILL as a fallback. Missing/dead PID is treated as "already gone" and
        we proceed to rebind.
        """
        pid = self._read_pid_sidecar()
        if pid is None:
            return

        try:
            os.kill(pid, 0)
        except (ProcessLookupError, OSError):
            # PID already dead (or not signalable) — nothing to reap.
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, OSError):
                return  # exited cleanly
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.01, remaining))

        # Still alive after the grace window — force it down.
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    def _pid_sidecar_path(self) -> str:
        """Return the path to the PID sidecar file."""
        return self.socket_path + ".pid"

    def _write_pid_sidecar(self) -> None:
        """Claim ``{socket_path}.pid`` with our own pid after binding.

        Best-effort: a failure to write the sidecar (e.g. read-only dir) must
        never crash a server that has already bound the socket successfully.
        """
        try:
            with open(self._pid_sidecar_path(), "w") as fh:
                fh.write(str(os.getpid()))
        except OSError:
            pass

    def _remove_pid_sidecar(self) -> None:
        """Remove ``{socket_path}.pid`` on exit (best-effort)."""
        try:
            os.unlink(self._pid_sidecar_path())
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _read_pid_sidecar(self) -> int | None:
        """Read the orphan's PID from ``{socket_path}.pid``; None if absent."""
        try:
            with open(self._pid_sidecar_path()) as fh:
                content = fh.read().strip()
        except (FileNotFoundError, OSError):
            return None
        try:
            return int(content)
        except ValueError:
            return None

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
