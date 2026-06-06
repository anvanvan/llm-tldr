"""Regression tests for the ``python -m tldr.model_server`` entry point.

These cover two real defects the mocked unit tests missed:

1. ``tldr/model_server/`` had no ``__main__.py``, so the ``python -m
   tldr.model_server`` command ``ensure_server`` spawns could never launch.
2. ``server.default_socket_path()`` hardcoded ``/tmp/`` while
   ``ensure._model_server_socket_path()`` uses ``tempfile.gettempdir()`` — on
   macOS these differ, so ``ensure_server`` would ping a path the server never
   binds and fall back to in-process forever.

The live smoke test only exercises bind + ``ping`` (the model loads lazily on
the first embed), so it costs no GPU memory.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import pytest

from tldr.daemon import ensure
from tldr.model_server.server import default_socket_path


def test_model_server_has_runnable_entry_point():
    """``python -m tldr.model_server`` must resolve to a __main__ module."""
    spec = importlib.util.find_spec("tldr.model_server.__main__")
    assert spec is not None, (
        "tldr/model_server/__main__.py is missing — `python -m tldr.model_server` "
        "(spawned by ensure_server) cannot launch."
    )


def test_default_socket_path_matches_ensure_helper():
    """The server's default bind path must equal what ensure_server pings."""
    assert default_socket_path() == ensure._model_server_socket_path(), (
        "server.default_socket_path() and ensure._model_server_socket_path() "
        "have drifted — ensure_server would never find the server it spawned."
    )


def test_entrypoint_starts_binds_and_pings_then_shuts_down():
    """Live: spawn the entry point, confirm the socket binds + answers ping.

    Uses TLDR_MODEL_SERVER_SOCKET to bind a throwaway path; ping does not load
    the model, so this is cheap and GPU-free.
    """
    tmpdir = tempfile.mkdtemp(prefix="tldr-ms-entry-")
    sock_path = os.path.join(tmpdir, "model-server.sock")
    env = dict(os.environ)
    env["TLDR_MODEL_SERVER_SOCKET"] = sock_path
    env["TLDR_MODEL_SERVER_IDLE_SECS"] = "60"

    proc = subprocess.Popen(
        [sys.executable, "-m", "tldr.model_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        deadline = time.time() + 15.0
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                pytest.fail(
                    f"model server exited early (code={proc.returncode}): {stderr}"
                )
            if ensure.ping_server(sock_path):
                ready = True
                break
            time.sleep(0.1)
        assert ready, f"server never became pingable at {sock_path}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        with __import__("contextlib").suppress(OSError):
            os.unlink(sock_path)
        with __import__("contextlib").suppress(OSError):
            os.rmdir(tmpdir)
