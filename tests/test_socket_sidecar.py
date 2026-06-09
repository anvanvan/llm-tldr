"""Unit tests for :class:`tldr.daemon.socket_sidecar.SocketSidecarOwner`.

The sidecar is the orphan-storm-defence primitive shared by the TLDR daemon
(``core.py``) and the model server (``server.py``): a small ``{socket}.pid``
(or ``.owner``) file naming the PID currently owning a Unix socket. These tests
exercise the class in isolation — no real socket, no GPU, no subprocess server.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tldr.daemon.socket_sidecar import SocketSidecarOwner


def test_path_uses_default_pid_suffix(tmp_path):
    sock = str(tmp_path / "srv.sock")
    assert SocketSidecarOwner(sock).path() == sock + ".pid"


def test_path_honours_custom_suffix(tmp_path):
    sock = str(tmp_path / "srv.sock")
    assert SocketSidecarOwner(sock, suffix=".owner").path() == sock + ".owner"


def test_write_then_read_roundtrips_own_pid(tmp_path):
    s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
    s.write_pid()
    assert s.read_pid() == os.getpid()
    assert os.path.exists(s.path())


def test_write_is_atomic_no_tmp_left_behind(tmp_path):
    """The atomic os.replace write must not leave a ``*.tmp`` staging file."""
    s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
    s.write_pid()
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"atomic write left staging files: {leftovers}"


def test_read_pid_absent_returns_none(tmp_path):
    s = SocketSidecarOwner(str(tmp_path / "nope.sock"))
    assert s.read_pid() is None


def test_read_pid_corrupt_content_returns_none(tmp_path):
    s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
    with open(s.path(), "w") as fh:
        fh.write("not-a-pid")
    assert s.read_pid() is None


def test_remove_pid_deletes_and_is_idempotent(tmp_path):
    s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
    s.write_pid()
    s.remove_pid()
    assert not os.path.exists(s.path())
    # Best-effort: removing an already-absent sidecar must never raise.
    s.remove_pid()


def test_not_superseded_when_sidecar_absent(tmp_path):
    assert SocketSidecarOwner(str(tmp_path / "nope.sock")).is_superseded() is False


def test_not_superseded_by_our_own_pid(tmp_path):
    s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
    s.write_pid()
    assert s.is_superseded() is False


def test_not_superseded_by_a_dead_pid(tmp_path):
    """A stale sidecar naming a dead process must NOT trigger self-eviction."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()  # reap it so the PID is definitively dead
    s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
    with open(s.path(), "w") as fh:
        fh.write(str(proc.pid))
    assert s.is_superseded() is False


def test_superseded_by_a_different_live_pid(tmp_path):
    """A sidecar naming a DIFFERENT, live process means we were superseded."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        # sanity: a real, live, other PID
        assert proc.pid != os.getpid()
        s = SocketSidecarOwner(str(tmp_path / "srv.sock"))
        with open(s.path(), "w") as fh:
            fh.write(str(proc.pid))
        assert s.is_superseded() is True
    finally:
        proc.kill()
        proc.wait()


def test_write_pid_best_effort_never_raises_on_bad_dir(tmp_path):
    """A write into a non-existent directory is swallowed, not raised."""
    s = SocketSidecarOwner(str(tmp_path / "no_such_dir" / "srv.sock"))
    s.write_pid()  # must not raise
    assert s.read_pid() is None
