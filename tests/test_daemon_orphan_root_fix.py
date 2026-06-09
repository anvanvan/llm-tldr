"""Root-cause regression tests for runaway TLDR *daemon* orphans.

The per-project daemon had the identical orphan-storm bug the model server fixed
in 423841c, never ported: a single-threaded accept loop went deaf during a slow
command, the small ``listen()`` backlog filled, ``ensure_daemon``'s ping was
refused/timed-out, the live-but-busy daemon was misread as dead, and a redundant
daemon was spawned that stole the socket — leaving the old one as an orphan that
never self-evicted. Observed in production: 9 daemons on one project socket.

These pin the ported fixes:
  1. RESPONSIVE accept — ping answered even while a heavy command holds the
     command lock (the threaded accept loop never goes deaf).
  2. REAP BY SIDECAR PID — a live daemon whose socket is unreachable (busy /
     backlog full) is still reaped via its PID sidecar before a new daemon binds.
  3. SELF-EVICTION — a superseded daemon exits and must NOT delete the new
     owner's socket/sidecar on the way out.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time

from tldr.daemon.core import TLDRDaemon, _DAEMON_LISTEN_BACKLOG


def _spawn_live_pid() -> subprocess.Popen:
    """A real, signalable child process to stand in for a foreign daemon PID."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def _wait(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _make_daemon(tmp_path) -> TLDRDaemon:
    """Construct a daemon and point its socket at an isolated tmp path."""
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    daemon = TLDRDaemon(project)
    daemon.socket_path = tmp_path / "d.sock"
    return daemon


def test_listen_backlog_is_large():
    """A tiny backlog is what let connect()s get refused under load → false-dead
    → duplicate spawn. The ported fix uses a comfortably large backlog."""
    assert _DAEMON_LISTEN_BACKLOG >= 128


# ===========================================================================
# Root cause #2 — reap by sidecar PID, not just the connect probe
# ===========================================================================

class TestReapBySidecarPid:
    def test_live_sidecar_owner_is_reaped_even_when_connect_probe_fails(self, tmp_path):
        daemon = _make_daemon(tmp_path)
        # A live foreign "daemon" recorded in the sidecar, but NOTHING listens on
        # the socket → the connect probe fails (mimics a busy daemon whose
        # backlog is full: connect refused). Reap must still fire from the
        # live-sidecar-PID branch.
        victim = _spawn_live_pid()
        try:
            with open(daemon._socket_owner_path(), "w") as fh:
                fh.write(str(victim.pid))

            daemon._reap_orphan_holder()

            assert _wait(lambda: victim.poll() is not None, timeout=8.0), (
                "the live sidecar-owner must be reaped even though the connect "
                "probe failed — a busy daemon with a full backlog refuses connect "
                "but is very much alive."
            )
        finally:
            if victim.poll() is None:
                victim.kill()
            victim.wait()

    def test_dead_sidecar_pid_is_not_signalled(self, tmp_path):
        """A stale sidecar naming a dead PID must be a no-op (nothing to reap)."""
        daemon = _make_daemon(tmp_path)
        victim = _spawn_live_pid()
        victim.kill()
        victim.wait()
        with open(daemon._socket_owner_path(), "w") as fh:
            fh.write(str(victim.pid))
        # Must not raise; simply returns (no live holder, no listener).
        daemon._reap_orphan_holder()


# ===========================================================================
# Root cause #3 — self-eviction when superseded (and don't nuke the new owner)
# ===========================================================================

class TestSelfEviction:
    def test_superseded_true_for_live_foreign_owner(self, tmp_path):
        daemon = _make_daemon(tmp_path)
        newowner = _spawn_live_pid()
        try:
            with open(daemon._socket_owner_path(), "w") as fh:
                fh.write(str(newowner.pid))
            assert daemon._superseded() is True
        finally:
            newowner.kill()
            newowner.wait()

    def test_own_pid_is_not_superseded(self, tmp_path):
        daemon = _make_daemon(tmp_path)
        daemon._write_socket_owner()  # writes our own pid
        assert daemon._superseded() is False

    def test_superseded_cleanup_preserves_new_owner_files(self, tmp_path):
        """A superseded orphan's _cleanup_socket must NOT delete the socket file
        or sidecar now owned by the live new daemon."""
        daemon = _make_daemon(tmp_path)
        # Create a socket file the new owner "owns".
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(daemon.socket_path))
        try:
            newowner = _spawn_live_pid()
            try:
                with open(daemon._socket_owner_path(), "w") as fh:
                    fh.write(str(newowner.pid))

                daemon._cleanup_socket()

                assert daemon.socket_path.exists(), (
                    "superseded orphan deleted the NEW owner's socket file"
                )
                assert os.path.exists(daemon._socket_owner_path()), (
                    "superseded orphan deleted the NEW owner's sidecar"
                )
                with open(daemon._socket_owner_path()) as fh:
                    assert fh.read().strip() == str(newowner.pid)
            finally:
                newowner.kill()
                newowner.wait()
        finally:
            srv.close()


# ===========================================================================
# Root cause #1 — accept loop stays responsive during a slow command
# ===========================================================================

class TestResponsiveDuringCommand:
    def test_ping_answered_while_command_lock_held(self, tmp_path):
        """While a heavy command holds the command lock, a ping on another
        connection must still be answered — proving ping bypasses the lock and
        the daemon never goes deaf (root cause #1)."""
        daemon = _make_daemon(tmp_path)

        a, b = socket.socketpair()
        try:
            b.sendall(json.dumps({"cmd": "ping"}).encode() + b"\n")

            # Simulate a heavy command in flight by holding the lock.
            daemon._command_lock.acquire()
            try:
                t = threading.Thread(target=daemon._serve_connection, args=(a,), daemon=True)
                t.start()
                t.join(timeout=3.0)
                assert not t.is_alive(), (
                    "ping handler blocked on the command lock — a deaf accept "
                    "loop is exactly the orphan-storm root cause."
                )
                resp = json.loads(b.recv(4096).decode())
                assert resp == {"status": "ok"}
            finally:
                daemon._command_lock.release()
        finally:
            b.close()

    def test_nonping_command_serializes_on_lock(self, tmp_path, monkeypatch):
        """A non-ping command must acquire the command lock (preserving the
        previous one-at-a-time semantics)."""
        daemon = _make_daemon(tmp_path)
        ran = threading.Event()

        def fake_handle(command):
            ran.set()
            return {"status": "ok", "echo": command.get("cmd")}

        monkeypatch.setattr(daemon, "handle_command", fake_handle)

        a, b = socket.socketpair()
        try:
            b.sendall(json.dumps({"cmd": "status"}).encode() + b"\n")
            # Hold the lock → the command must NOT run until we release.
            daemon._command_lock.acquire()
            t = threading.Thread(target=daemon._serve_connection, args=(a,), daemon=True)
            t.start()
            assert not ran.wait(timeout=0.5), (
                "a stateful command ran while the command lock was held — "
                "serialization is broken"
            )
            daemon._command_lock.release()
            assert ran.wait(timeout=3.0), "command never ran after lock release"
            resp = json.loads(b.recv(4096).decode())
            assert resp["status"] == "ok"
            t.join(timeout=3.0)
        finally:
            # _serve_connection already closes `a`; closing again is harmless.
            with contextlib.suppress(OSError):
                a.close()
            b.close()


# ===========================================================================
# Review fixes — backpressure (#4), cleanup-on-missing-sidecar (#6),
# atomic sidecar publish (#3)
# ===========================================================================

class TestBackpressureReservesPingCapacity:
    def test_overload_sheds_busy_but_ping_still_answered(self, tmp_path):
        """When all command slots are taken (a slow command holding the lock),
        a further command is shed fast with 'busy' — but a ping is still answered
        lock-free, so the daemon never goes deaf (no false-dead respawn)."""
        daemon = _make_daemon(tmp_path)
        # Drain every command slot so the next command can't acquire one.
        from tldr.daemon.core import _DAEMON_MAX_INFLIGHT_COMMANDS
        for _ in range(_DAEMON_MAX_INFLIGHT_COMMANDS):
            assert daemon._command_slots.acquire(blocking=False)

        # A non-ping command must be shed with 'busy' (not block, not pile up).
        a, b = socket.socketpair()
        try:
            b.sendall(json.dumps({"cmd": "search", "pattern": "x"}).encode() + b"\n")
            t = threading.Thread(target=daemon._serve_connection, args=(a,), daemon=True)
            t.start()
            t.join(timeout=3.0)
            assert not t.is_alive(), "shed path blocked instead of returning busy"
            assert json.loads(b.recv(4096).decode())["status"] == "busy"
        finally:
            with contextlib.suppress(OSError):
                a.close()
            b.close()

        # ...and a ping is still answered even with every command slot taken.
        c, d = socket.socketpair()
        try:
            d.sendall(json.dumps({"cmd": "ping"}).encode() + b"\n")
            t = threading.Thread(target=daemon._serve_connection, args=(c,), daemon=True)
            t.start()
            t.join(timeout=3.0)
            assert not t.is_alive(), "ping blocked while command slots exhausted"
            assert json.loads(d.recv(4096).decode()) == {"status": "ok"}
        finally:
            with contextlib.suppress(OSError):
                c.close()
            d.close()

    def test_command_slot_released_after_handling(self, tmp_path, monkeypatch):
        """A handled command must release its slot so capacity is not leaked."""
        daemon = _make_daemon(tmp_path)
        monkeypatch.setattr(daemon, "handle_command", lambda cmd: {"status": "ok"})
        a, b = socket.socketpair()
        try:
            b.sendall(json.dumps({"cmd": "status"}).encode() + b"\n")
            t = threading.Thread(target=daemon._serve_connection, args=(a,), daemon=True)
            t.start()
            t.join(timeout=3.0)
            assert json.loads(b.recv(4096).decode())["status"] == "ok"
            # All slots should be free again.
            from tldr.daemon.core import _DAEMON_MAX_INFLIGHT_COMMANDS
            got = sum(1 for _ in range(_DAEMON_MAX_INFLIGHT_COMMANDS)
                      if daemon._command_slots.acquire(blocking=False))
            assert got == _DAEMON_MAX_INFLIGHT_COMMANDS, "a command slot was leaked"
        finally:
            with contextlib.suppress(OSError):
                a.close()
            b.close()


class TestCleanupSocketOwnershipGuard:
    def test_missing_sidecar_still_cleans_our_socket(self, tmp_path):
        """If our owner-sidecar write failed (None), shutdown must still unlink
        our socket rather than leak it."""
        daemon = _make_daemon(tmp_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(daemon.socket_path))
        srv.close()
        assert daemon.socket_path.exists()
        # No .owner sidecar exists (simulates a failed _write_socket_owner).
        assert daemon._read_socket_owner() is None
        daemon._cleanup_socket()
        assert not daemon.socket_path.exists(), "missing-sidecar socket was leaked"

    def test_foreign_live_owner_socket_preserved(self, tmp_path):
        daemon = _make_daemon(tmp_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(daemon.socket_path))
        try:
            newowner = _spawn_live_pid()
            try:
                with open(daemon._socket_owner_path(), "w") as fh:
                    fh.write(str(newowner.pid))
                daemon._cleanup_socket()
                assert daemon.socket_path.exists(), "deleted a live owner's socket"
            finally:
                newowner.kill(); newowner.wait()
        finally:
            srv.close()


class TestAtomicSidecarWrite:
    def test_write_then_read_roundtrips_and_leaves_no_tmp(self, tmp_path):
        daemon = _make_daemon(tmp_path)
        daemon._write_socket_owner()
        assert daemon._read_socket_owner() == os.getpid()
        # No leftover temp file from the atomic publish.
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == [], f"atomic write left temp files: {leftovers}"
