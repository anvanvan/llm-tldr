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
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

from .lifecycle import ModelServerLifecycle
from .queue import EmbedQueue
from .transport import recv_message, send_message

_ACCEPT_TIMEOUT = 0.2

# Backlog for the listen socket. Must be comfortably larger than the number of
# connections that can pile up while one embed is in flight; an undersized
# backlog (the old listen(8)) gets connect()s REFUSED under load, which callers
# misread as "server dead" and respond to by spawning a redundant server.
_LISTEN_BACKLOG = 128

# Max concurrent connection-handler threads. The GPU itself stays serial (every
# embed funnels through the single-worker EmbedQueue); these threads only keep
# the accept loop draining and answer cheap requests (ping) immediately so the
# server never goes deaf during a slow embed.
_MAX_CONN_WORKERS = 16


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live, signalable process. Never raises."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

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
        # Orphan reap: an old server still holding this path must be killed
        # before we steal it, otherwise it lingers forever pinning ~GBs.
        #
        # We decide to reap on EITHER signal:
        #   (a) the connect probe succeeds — a server is accepting on the path; OR
        #   (b) the PID sidecar names a live process other than us.
        #
        # (b) is essential: a *busy* server (single embed in flight, backlog
        # full) REFUSES the connect probe, so (a) alone is False and the old code
        # skipped the reap — it then unlinked the busy server's socket and
        # rebound WITHOUT killing it, manufacturing an un-signalled orphan. The
        # sidecar PID is the reliable liveness signal a full backlog can't mask.
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

        sidecar_pid = self._read_pid_sidecar()
        live_sidecar_owner = (
            sidecar_pid is not None
            and sidecar_pid != os.getpid()
            and _pid_alive(sidecar_pid)
        )
        if live_peer or live_sidecar_owner:
            self._reap_orphan()

        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._bind_socket(srv)
        srv.listen(_LISTEN_BACKLOG)
        srv.settimeout(_ACCEPT_TIMEOUT)
        self._server_sock = srv

        # PID sidecar: write OUR pid to {socket}.pid only AFTER reaping any
        # orphan and rebinding the socket. Writing it here (not before run())
        # is what makes orphan reap correct: when a second server starts, its
        # _reap_orphan() above reads the EXISTING sidecar — still the orphan's
        # pid — and SIGTERMs the orphan, never itself. Only once we own the
        # socket do we claim the sidecar.
        self._write_pid_sidecar()

        # Each accepted connection is handled on a pool thread so the accept
        # loop NEVER blocks on a slow embed. The GPU stays serial (every embed
        # funnels through the single-worker EmbedQueue); the threads only keep
        # accept() draining and answer cheap requests (ping) immediately. A deaf
        # accept loop was the upstream cause of orphan storms: backlog fills →
        # connect refused → callers think the server died → spawn a duplicate.
        executor = ThreadPoolExecutor(
            max_workers=_MAX_CONN_WORKERS, thread_name_prefix="tldr-ms-conn"
        )
        try:
            while not self._stop.is_set():
                # Self-eviction: if the sidecar now names another live process,
                # a newer server has taken over the socket — we are an orphan.
                # Exit immediately instead of lingering forever holding the
                # model. This is the backstop that makes orphans structurally
                # impossible to persist, even if our reap signal never landed.
                if self._superseded():
                    break
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
                executor.submit(self._handle_connection, conn)
        finally:
            executor.shutdown(wait=False)
            srv.close()
            # Clean up the socket + sidecar ONLY if we still own them. A
            # superseded orphan must NOT unlink the path (now the new owner's
            # socket) nor delete the new owner's sidecar — doing so would make
            # the live server unreachable and trigger yet another respawn.
            if self._read_pid_sidecar() == os.getpid():
                try:
                    os.unlink(self.socket_path)
                except FileNotFoundError:
                    pass
                self._remove_pid_sidecar()
            # Deliberately DO NOT run torch.mps.empty_cache() / model unload on
            # the exit path. A dying process has all of its GPU/unified memory
            # reclaimed by the OS, so an in-process teardown frees nothing extra
            # — but it can WEDGE: when a server is terminated (e.g. by the orphan
            # reap) while Metal command buffers are in flight, the unload's GPU
            # buffer dealloc deadlocks inside the Apple GPU driver
            # (AGXMetalG16X / IOGPU). Observed in production: reaped servers stuck
            # in that dealloc, unkillable-promptly, each pinning ~9 GB of MPS
            # memory — the opposite of the intended cleanup. Idle-time unload
            # still happens on the accept-timeout tick above (server stays alive,
            # no in-flight GPU work). On real termination we just exit fast and
            # let the kernel reclaim the GPU context (see __main__: os._exit).

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

    def _superseded(self) -> bool:
        """True if another live process now owns the socket per the sidecar.

        After we bind we write the sidecar with our own PID. If a newer server
        later rebinds the socket it overwrites the sidecar with ITS pid; reading
        a different, live PID here means we have been superseded and are now an
        orphan — we should exit. A missing or stale (dead) sidecar PID is NOT
        treated as superseded (we keep serving until a real replacement exists).
        """
        pid = self._read_pid_sidecar()
        return pid is not None and pid != os.getpid() and _pid_alive(pid)

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
