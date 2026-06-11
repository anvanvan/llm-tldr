"""On-demand ensure-up helpers for the TLDR daemon and shared model server.

This module centralises the "make sure it's running, start it if not" logic
that was previously inlined in :mod:`tldr.mcp_server`. It exposes:

- :func:`ensure_daemon` — idempotent start of the per-project daemon.
- :func:`ensure_server` — idempotent start of the shared model server.
- :func:`ping_server` — non-raising reachability check for the model server.

Both ensure helpers use a fast ping path first and only acquire a file lock
and spawn a subprocess when the target is absent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Conditional import for file locking
if os.name == "nt":
    import msvcrt
else:
    import fcntl


# ---------------------------------------------------------------------------
# Shared file-locking helpers
# ---------------------------------------------------------------------------


def _acquire_exclusive_lock(lock_file, lock_path: str, timeout: float = 10.0) -> None:
    """Acquire an exclusive file lock, with platform-specific handling.

    Args:
        lock_file: Open file object to lock.
        lock_path: Path to the lock file (for error messages).
        timeout: Lock acquisition timeout in seconds (Windows only).

    Raises:
        RuntimeError: If lock cannot be acquired within timeout.
    """
    if os.name == "nt":
        lock_start = time.time()
        while True:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as e:
                if time.time() - lock_start > timeout:
                    raise RuntimeError(
                        f"Timeout acquiring lock on {lock_path} after {timeout}s"
                    ) from e
                time.sleep(0.1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)


def _release_exclusive_lock(lock_file) -> None:
    """Release an exclusive file lock acquired by _acquire_exclusive_lock.

    Best-effort: failures are suppressed as the lock is released on file close.
    """
    if os.name == "nt":
        with contextlib.suppress(OSError):
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Daemon ensure-up (extracted from mcp_server._ensure_daemon)
# ---------------------------------------------------------------------------


def _anchor_project(project: str) -> str:
    """Resolve a project path to the smart project root.

    Routes the raw ``--path``/``--project`` arg through
    :func:`tldr.semantic._find_project_root` so a daemon started for a deep
    subdir of a marker-bearing project (``.git``, ``.svn``, ``.tldr`` cache, …)
    anchors at the project root — exactly where the in-process indexer anchors —
    instead of fragmenting into a fresh per-subdir ``.tldr`` cache.

    Imported lazily to avoid pulling the heavy :mod:`tldr.semantic` module (and
    its transitive ML deps) into the ensure-up fast path / import graph.
    """
    from ..semantic import _find_project_root

    return str(_find_project_root(Path(project)))


def _get_socket_path(project: str) -> Path:
    """Compute the daemon socket path for a project."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{hash_val}.sock"


def _get_lock_path(project: str) -> Path:
    """Compute the daemon startup lock path for a project."""
    hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
    tmp_dir = tempfile.gettempdir()
    return Path(tmp_dir) / f"tldr-{hash_val}.lock"


def _get_connection_info(project: str) -> tuple[str, int | None]:
    """Return (address, port); port is None for Unix sockets."""
    if sys.platform == "win32":
        hash_val = hashlib.md5(str(Path(project).resolve()).encode()).hexdigest()[:8]
        port = 49152 + (int(hash_val, 16) % 10000)
        return ("127.0.0.1", port)
    socket_path = _get_socket_path(project)
    return (str(socket_path), None)


def _send_raw(project: str, command: dict) -> dict:
    """Send a single command to the daemon socket and return the response."""
    addr, port = _get_connection_info(project)

    sock = None
    try:
        if port is not None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((addr, port))
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(addr)

        sock.sendall(json.dumps(command).encode() + b"\n")

        buf = bytearray()
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
        line, _, _ = bytes(buf).partition(b"\n")
        return json.loads(line.decode("utf-8"))
    finally:
        if sock:
            sock.close()


def _ping_daemon(project: str) -> bool:
    """Return True if the daemon is alive and responding to ``ping``."""
    addr, port = _get_connection_info(project)

    # On Unix, a missing socket file means no daemon.
    if port is None and not Path(addr).exists():
        return False

    try:
        result = _send_raw(project, {"cmd": "ping"})
        return result.get("status") == "ok"
    except Exception:
        return False


def ensure_daemon(project: str, timeout: float = 10.0) -> None:
    """Ensure the per-project daemon is running, starting it if needed.

    Uses a fast ping path first; only acquires a flock and spawns a subprocess
    when the daemon is absent. The lock prevents concurrent callers from
    double-spawning.
    """
    # Anchor the raw path at the smart project root BEFORE deriving any
    # socket/lock identity or the spawn argv, so every daemon entry point
    # (ensure/start/stop/status/query/notify) agrees on a single root and the
    # daemon never fragments into a per-subdir index. See _anchor_project.
    project = _anchor_project(project)

    # Fast path: daemon already running (no lock needed).
    if _ping_daemon(project):
        return

    lock_path = _get_lock_path(project)

    lock_path.touch(exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            _acquire_exclusive_lock(lock_file, str(lock_path), timeout=10.0)

            # Re-check under the lock (another process may have started it).
            if _ping_daemon(project):
                return

            # NOTE: deliberately do NOT unlink the socket here. A live-but-busy
            # daemon (accept loop briefly saturated, backlog full) fails the ping
            # above yet is alive; unlinking its socket and spawning would orphan
            # it. The spawned daemon's _create_server_socket reaps a live holder
            # via its PID sidecar, and handles a genuinely stale socket itself
            # (connect-refused → unlink → rebind). Mirrors ensure_server, which
            # never unlinks.

            subprocess.Popen(
                [
                    sys.executable, "-m", "tldr.cli",
                    "daemon", "start", "--project", project,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for the daemon to become ready.
            start = time.time()
            while time.time() - start < timeout:
                if _ping_daemon(project):
                    return
                time.sleep(0.1)

            raise RuntimeError(f"Failed to start TLDR daemon for {project}")
        finally:
            _release_exclusive_lock(lock_file)


# ---------------------------------------------------------------------------
# Shared model-server ensure-up
# ---------------------------------------------------------------------------


def _model_server_socket_path() -> str:
    """Per-user socket path for the shared model server.

    Respects ``TLDR_MODEL_SERVER_SOCKET``: when set, that value is the canonical
    socket path so that callers (``ensure_server``) and the spawned server agree
    on a single location. This is what lets tests route to an isolated socket and
    keeps the real per-user socket untouched (arch-review G-4).
    """
    env_override = os.environ.get("TLDR_MODEL_SERVER_SOCKET")
    if env_override:
        return env_override
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return os.path.join(tempfile.gettempdir(), f"tldr-model-server-{uid}.sock")


def _model_server_lock_path() -> str:
    """Per-user lock path for the shared model server startup.

    Mirrors ``_model_server_socket_path``: when ``TLDR_MODEL_SERVER_SOCKET`` is
    set, derive the lock path from it (``<sock>.lock``) so isolated tests do not
    serialize on the production canonical lock (arch-review A-2).
    """
    env_override = os.environ.get("TLDR_MODEL_SERVER_SOCKET")
    if env_override:
        return env_override + ".lock"
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return os.path.join(tempfile.gettempdir(), f"tldr-model-server-{uid}.lock")


def ping_server(socket_path: str) -> bool:
    """Return True if the model server at ``socket_path`` accepts a connection.

    Uses non-blocking connect to avoid waiting on unreachable sockets.
    Never raises: a missing or refusing socket returns False.
    """
    if not os.path.exists(socket_path):
        return False
    sock = None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.setblocking(False)
        try:
            sock.connect(socket_path)
        except BlockingIOError:
            # EAGAIN on non-blocking connect means the socket accepted the
            # connection request (in progress). This is success for a reachable
            # Unix domain socket.
            return True
        except OSError:
            return False
        return True
    finally:
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


def ensure_server(timeout: float = 10.0) -> str:
    """Ensure the shared model server is running; return its socket path.

    Fast path: ping first and return immediately if reachable. Otherwise
    acquire a flock and spawn ``python -m tldr.model_server``, then poll
    :func:`ping_server` until ready (or timeout).
    """
    socket_path = _model_server_socket_path()

    # Fast path: server already up.
    if ping_server(socket_path):
        return socket_path

    lock_path = _model_server_lock_path()
    Path(lock_path).touch(exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            _acquire_exclusive_lock(lock_file, str(lock_path), timeout=10.0)

            # Re-check under the lock.
            if ping_server(socket_path):
                return socket_path

            # Propagate the resolved socket path so the spawned server binds to
            # exactly the path this caller polls (honours TLDR_MODEL_SERVER_SOCKET).
            spawn_env = os.environ.copy()
            spawn_env["TLDR_MODEL_SERVER_SOCKET"] = socket_path
            subprocess.Popen(
                [sys.executable, "-m", "tldr.model_server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=spawn_env,
            )

            start = time.time()
            while time.time() - start < timeout:
                if ping_server(socket_path):
                    return socket_path
                time.sleep(0.1)

            raise RuntimeError("Failed to start TLDR model server")
        finally:
            _release_exclusive_lock(lock_file)
