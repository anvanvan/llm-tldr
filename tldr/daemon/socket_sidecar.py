"""Shared PID-sidecar ownership utilities for Unix-socket servers.

A PID sidecar is a small text file placed beside a Unix socket file (e.g.
``/path/to/socket.pid``) that names the PID of the process currently owning
the socket.  Two independent servers — the TLDR daemon and the model server —
use this pattern for orphan-storm defence:

* On startup: read the existing sidecar, probe liveness, SIGTERM + SIGKILL if
  the previous owner is still alive before rebinding the socket.
* After binding: atomically write our own PID to the sidecar.
* While running: poll the sidecar; if another PID appears (because a newer
  server has rebound) we self-evict rather than become an orphan.
* On exit: remove the sidecar (best-effort).

:class:`SocketSidecarOwner` encapsulates all of this logic so it can be reused
without copy-paste duplication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class SocketSidecarOwner:
    """Manage a PID sidecar file alongside a Unix-socket path.

    Args:
        socket_path: Path to the Unix socket file.  The sidecar is placed at
            ``str(socket_path) + suffix``.
        suffix: File-name suffix for the sidecar (default ``".pid"``).
    """

    def __init__(self, socket_path: str | Path, suffix: str = ".pid") -> None:
        self._socket_path = str(socket_path)
        self._suffix = suffix

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def path(self) -> str:
        """Return the absolute path to the sidecar file."""
        return self._socket_path + self._suffix

    # ------------------------------------------------------------------
    # Read / write / remove
    # ------------------------------------------------------------------

    def write_pid(self) -> None:
        """Atomically claim the sidecar with the current process PID.

        Uses ``os.replace`` (rename) so a concurrent reader never sees a
        partially-written or empty file — it either reads the old value or the
        new one, never a truncated intermediate.  Failure is silently ignored
        so a server that has already bound the socket is not crashed by a
        read-only directory or other transient filesystem error.
        """
        sidecar_path = self.path()
        tmp = f"{sidecar_path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as fh:
                fh.write(str(os.getpid()))
            os.replace(tmp, sidecar_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def read_pid(self) -> Optional[int]:
        """Return the PID stored in the sidecar, or ``None`` if absent/corrupt."""
        try:
            with open(self.path()) as fh:
                content = fh.read().strip()
        except (FileNotFoundError, OSError):
            return None
        try:
            return int(content)
        except ValueError:
            return None

    def remove_pid(self) -> None:
        """Remove the sidecar file on exit (best-effort; never raises)."""
        try:
            os.unlink(self.path())
        except (FileNotFoundError, OSError):
            pass

    # ------------------------------------------------------------------
    # Liveness check
    # ------------------------------------------------------------------

    def is_superseded(self) -> bool:
        """Return ``True`` if another *live* process now owns this sidecar.

        A newer server that rebinds the socket overwrites the sidecar with its
        own PID.  Reading a different, live PID here means the current process
        has been superseded and should self-evict.  A missing or stale (dead)
        PID is **not** treated as superseded — we keep serving until a real
        replacement is confirmed.
        """
        from tldr.model_server.server import _pid_alive  # avoid top-level circular

        pid = self.read_pid()
        return pid is not None and pid != os.getpid() and _pid_alive(pid)
