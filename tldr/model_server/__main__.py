"""Runnable entry point for the shared model server.

Launched by :func:`tldr.daemon.ensure.ensure_server` via
``python -m tldr.model_server``. Binds the Unix socket at the SAME path
``ensure_server`` pings (so the two never drift), then serves until killed.

Configuration via environment:

- ``TLDR_MODEL_SERVER_SOCKET`` — override the socket path (defaults to the
  shared per-user path from :func:`tldr.daemon.ensure._model_server_socket_path`).
- ``TLDR_MODEL_SERVER_IDLE_SECS`` — rolling idle window in seconds before the
  model is auto-unloaded (default 1800 = 30 min). Each embed request extends it.

The model is loaded lazily on the first embed request, so simply starting the
server (and answering ``ping``) costs no GPU memory.
"""

from __future__ import annotations

import os
import signal
import sys

from .server import ModelServer

_DEFAULT_IDLE_SECONDS = 1800


def _resolve_socket_path() -> str:
    """Socket path to bind — identical to what ``ensure_server`` pings."""
    override = os.environ.get("TLDR_MODEL_SERVER_SOCKET")
    if override:
        return override
    # Single source of truth: reuse the helper ensure_server itself uses so the
    # spawned server always binds exactly where the caller looks for it.
    from tldr.daemon.ensure import _model_server_socket_path

    return _model_server_socket_path()


def _resolve_idle_seconds() -> int:
    raw = os.environ.get("TLDR_MODEL_SERVER_IDLE_SECS")
    if not raw:
        return _DEFAULT_IDLE_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_IDLE_SECONDS
    return value if value > 0 else _DEFAULT_IDLE_SECONDS


def main(argv: list[str] | None = None) -> int:
    _ = argv  # no positional args today; reserved for future flags
    socket_path = _resolve_socket_path()
    idle_seconds = _resolve_idle_seconds()

    server = ModelServer(socket_path=socket_path, idle_seconds=idle_seconds)

    def _graceful_stop(_signum: int, _frame: object) -> None:
        server.shutdown()

    # Terminate cleanly on the signals ensure_server / the OS will send.
    signal.signal(signal.SIGTERM, _graceful_stop)
    with_sigint = True
    try:
        signal.signal(signal.SIGINT, _graceful_stop)
    except (ValueError, OSError):
        # Not on the main thread (e.g. embedded in tests) — skip SIGINT.
        with_sigint = False
    _ = with_sigint

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
