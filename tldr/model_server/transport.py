"""Reusable JSON-newline framing over Unix sockets.

Mirrors the framing used by ``tldr/daemon/startup.py:query_daemon`` —
each message is a single JSON object followed by a ``\\n`` delimiter.
"""

from __future__ import annotations

import json
import os
import socket

# Generous upper bound on a single inbound message. Real embed batches are a
# few MB at most; this cap exists only to keep a malformed/hostile peer from
# driving recv_message into unbounded memory growth before it sees a newline.
MAX_MESSAGE_BYTES = 512 * 1024 * 1024


def send_message(sock: socket.socket, msg: dict) -> None:
    """Serialize ``msg`` as JSON and write it to ``sock`` with a newline frame."""
    data = json.dumps(msg).encode("utf-8") + b"\n"
    sock.sendall(data)


def recv_message(sock: socket.socket) -> dict:
    """Read one newline-terminated JSON message from ``sock`` and decode it.

    Reads until a newline is seen so that messages larger than a single
    recv buffer are reassembled correctly.
    """
    buf = bytearray()
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_MESSAGE_BYTES:
            raise ValueError(
                f"recv_message: message exceeded {MAX_MESSAGE_BYTES} bytes "
                "without a newline delimiter"
            )
    line, _, _ = bytes(buf).partition(b"\n")
    return json.loads(line.decode("utf-8"))


def connect_unix(path: str, timeout: float = 2.0) -> socket.socket:
    """Connect to a Unix-domain socket at ``path``.

    Raises ``FileNotFoundError`` / ``ConnectionRefusedError`` / ``OSError``
    when the socket does not exist or refuses the connection.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        try:
            sock.connect(path)
        except OSError as first_err:
            # Unix sun_path is length-capped (~104 bytes on macOS). For an
            # over-long but existing path, retry by chdir-ing to its directory
            # and connecting via the basename.
            if isinstance(first_err, (FileNotFoundError, ConnectionRefusedError)):
                raise
            directory = os.path.dirname(path) or "."
            name = os.path.basename(path)
            prev_cwd = os.getcwd()
            try:
                os.chdir(directory)
                sock.connect(name)
            finally:
                os.chdir(prev_cwd)
    except OSError:
        sock.close()
        raise
    return sock
