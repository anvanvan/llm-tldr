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

# Env var naming the PID of the process that spawned this server. Set ONLY for
# test model servers (by tests/conftest.py pytest_configure). When present, the
# server runs a parent-death watchdog that self-exits when that specific PID dies
# — so a test server orphaned by an abnormal pytest exit cannot persist between
# runs holding the GPU. The legit shared PRODUCTION server (commits 0dcc6d2 /
# 219b29c8) is detached to PPID 1 BY DESIGN and does NOT set this env, so it never
# starts the watchdog and never self-exits. The watchdog keys on this SPECIFIC
# pid, NEVER on PPID==1, precisely to avoid killing the production server.
_PARENT_PID_ENV = "TLDR_MODEL_SERVER_PARENT_PID"

# How often the parent-death watchdog polls os.kill(parent_pid, 0). Kept short so
# self-exit lands well inside the test's 5s window once the parent dies.
_WATCHDOG_POLL_SECS = 1.0

# Bound on the graceful-shutdown queue drain. Must be strictly LESS than the
# conftest SIGTERM→SIGKILL window (5s) so a wedged in-flight Metal op can never
# stall shutdown past that window and provoke a mid-op SIGKILL (which wedges the
# GPU). A healthy embed drains in well under a second; this only bounds the
# pathological wedged case, after which __main__'s os._exit reclaims the context.
#
# COUPLING: _DRAIN_TIMEOUT (3.0s) < conftest _reap_model_server SIGTERM→SIGKILL
# window (5.0s, tests/conftest.py:_reap_model_server deadline = time.time()+5.0).
# If that window is ever reduced below _DRAIN_TIMEOUT the safety margin vanishes —
# keep _DRAIN_TIMEOUT strictly below whatever grace period conftest gives before
# escalating to SIGKILL.
_DRAIN_TIMEOUT = 3.0


def _pid_alive(pid: int) -> bool:
    """True if ``pid`` is a live, signalable process. Never raises."""
    try:
        os.kill(pid, 0)
    except OSError:
        # Intentionally treat EPERM as dead: on macOS, os.kill(1, 0) raises
        # EPERM (not ESRCH) for init/launchd — a non-signalable process is
        # "dead enough" for the sweep and watchdog intent (we cannot reap it
        # or rely on it as a parent), so False is the correct answer here.
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
        # Lazy import: tldr.daemon.__init__ eagerly imports core.py, which in
        # turn imports _pid_alive from THIS module — importing SocketSidecarOwner
        # at module top would form a partially-initialised circular import. By
        # the time a ModelServer is instantiated both modules are fully loaded.
        from tldr.daemon.socket_sidecar import SocketSidecarOwner

        self.socket_path = socket_path
        self._sidecar = SocketSidecarOwner(socket_path, suffix=".pid")
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

        # Parent-death watchdog: a TEST server (told its spawning pytest's PID via
        # TLDR_MODEL_SERVER_PARENT_PID) must self-exit promptly when that parent
        # dies, so no orphan persists between runs holding the GPU. Started only
        # when the env names a parent — the production shared server (PPID 1 by
        # design, env unset) never starts it and never self-exits.
        self._start_parent_death_watchdog()

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

    def _start_parent_death_watchdog(self) -> None:
        """Start a daemon thread that self-exits when the spawning parent dies.

        Reads the spawning process's PID from ``TLDR_MODEL_SERVER_PARENT_PID``.
        When that env is UNSET — the production shared server, which is detached
        to PPID 1 by design — this is a no-op: no watchdog, no self-exit. When it
        names a PID, a daemon thread polls ``os.kill(parent_pid, 0)`` every
        ``_WATCHDOG_POLL_SECS``; the first ``ProcessLookupError`` (ESRCH) means the
        parent is gone, so we unlink our own socket trio and ``os._exit(0)``
        WITHOUT any MPS teardown — preserving the commit-60bce6d invariant (an
        in-process GPU teardown can wedge Metal; the kernel reclaims the context
        on process death).

        The watchdog keys on this SPECIFIC parent PID, NEVER on ``PPID == 1`` — a
        "exit when PPID==1" check would instantly kill the legit production server.
        """
        raw = os.environ.get(_PARENT_PID_ENV)
        if not raw:
            return
        try:
            parent_pid = int(raw)
        except ValueError:
            return
        # A zero/negative or self PID is meaningless as a parent — never watch it.
        if parent_pid <= 0 or parent_pid == os.getpid():
            return

        thread = threading.Thread(
            target=self._watch_parent,
            args=(parent_pid,),
            name="tldr-ms-parent-watchdog",
            daemon=True,
        )
        thread.start()

    def _watch_parent(self, parent_pid: int) -> None:
        """Poll the spawning parent and self-exit on its death. Never returns."""
        while not self._stop.is_set():
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                # Parent is gone — reap ourselves so no orphan persists.
                self._self_evict_on_parent_death()
                return  # os._exit is called above; return prevents infinite spin
                        # if os._exit is ever mocked in tests.
            except OSError:
                # Permission error or similar: parent still exists (a dead PID
                # raises ESRCH, not EPERM), so keep watching.
                pass
            time.sleep(_WATCHDOG_POLL_SECS)

    def _self_evict_on_parent_death(self) -> None:
        """Unlink our socket trio and hard-exit — NO MPS teardown (60bce6d)."""
        for suffix in ("", ".pid", ".lock"):
            try:
                os.unlink(self.socket_path + suffix)
            except OSError:
                pass
        # os._exit: skip interpreter shutdown so torch/Metal destructors and
        # atexit handlers never run — an in-process GPU teardown can wedge the
        # Apple GPU driver. The OS reclaims the GPU/unified context on death.
        os._exit(0)

    def _pid_sidecar_path(self) -> str:
        """Return the path to the PID sidecar file (``{socket_path}.pid``)."""
        return self._sidecar.path()

    def _write_pid_sidecar(self) -> None:
        """Claim ``{socket_path}.pid`` with our own pid after binding.

        Delegates to :class:`SocketSidecarOwner`, whose atomic ``os.replace``
        write closes the empty-file race a plain truncate-then-write leaves
        open (a concurrent reader seeing an empty sidecar reads None → "no live
        owner" → steals the socket → duplicate server, the exact storm this
        guards against). Best-effort: a write failure (e.g. read-only dir) must
        never crash a server that has already bound the socket successfully.
        """
        self._sidecar.write_pid()

    def _remove_pid_sidecar(self) -> None:
        """Remove ``{socket_path}.pid`` on exit (best-effort)."""
        self._sidecar.remove_pid()

    def _read_pid_sidecar(self) -> int | None:
        """Read the orphan's PID from ``{socket_path}.pid``; None if absent."""
        return self._sidecar.read_pid()

    def _superseded(self) -> bool:
        """True if another live process now owns the socket per the sidecar.

        After we bind we write the sidecar with our own PID. If a newer server
        later rebinds the socket it overwrites the sidecar with ITS pid; reading
        a different, live PID here means we have been superseded and are now an
        orphan — we should exit. A missing or stale (dead) sidecar PID is NOT
        treated as superseded (we keep serving until a real replacement exists).

        Delegates to :meth:`SocketSidecarOwner.is_superseded`.
        """
        return self._sidecar.is_superseded()

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
        """Signal the accept loop to stop and tear down the queue — bounded.

        Sets ``_stop`` so the accept loop exits, then drains the embed queue
        with a BOUNDED wait. The bound is the Bug B (Metal-wedge) fix: a normal
        in-flight embed drains in well under a second, but a wedged Metal command
        buffer can make the embed worker's ``join()`` block forever. Without a
        bound, the SIGTERM handler that calls ``shutdown()`` would itself hang,
        the conftest 5s SIGTERM→SIGKILL window would elapse, and the server would
        be SIGKILLed mid-Metal-op — the exact condition that wedges the GPU.

        We cap the drain strictly inside that window (``_DRAIN_TIMEOUT`` ≈ 3s).
        If the queue does not drain in time we stop waiting and return, letting
        the caller (``__main__``: ``os._exit``) terminate the process WITHOUT any
        in-process MPS teardown — preserving the 60bce6d invariant. The kernel
        reclaims the GPU/unified-memory context on process death.
        """
        self._stop.set()

        drainer = threading.Thread(
            target=self._embed_queue.shutdown,
            name="tldr-ms-drain",
            daemon=True,
        )
        drainer.start()
        drainer.join(_DRAIN_TIMEOUT)
        # If `drainer` is still alive here, the in-flight embed is wedged; we do
        # NOT block on it. The daemon thread is abandoned and the process exits
        # via os._exit in __main__, which the OS-level GPU teardown follows.
