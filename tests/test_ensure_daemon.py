"""
Tests for tldr.daemon.ensure module (RED phase — module does not exist yet).

This module tests:
1. ensure_daemon() fast-path: pings first, only spawns if absent
2. ensure_daemon() slow-path: acquires flock + spawns Popen + waits for ready
3. mcp_server delegates to the shared ensure_daemon (no private duplicate)

All tests are RED on HEAD because:
  - tldr/daemon/ensure.py does not exist (ImportError)
  - tldr.mcp_server still has its own inline _ensure_daemon body

Test strategy: monkeypatch subprocess.Popen + socket connectivity; no real
process spawn, no GPU, no network.
"""

from __future__ import annotations

import importlib
import socket
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: minimal project fixture
# ---------------------------------------------------------------------------

def _tiny_project(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir(exist_ok=True)
    return tmp_path


# ===========================================================================
# TEST 1: ensure_daemon module exists and exposes ensure_daemon(project)
# ===========================================================================

class TestEnsureDaemonModuleInterface:
    """tldr.daemon.ensure must export ensure_daemon(project, timeout=10.0).

    RED: ImportError — tldr.daemon.ensure does not exist.
    """

    def test_module_importable(self):
        """tldr.daemon.ensure must be importable.

        RED: ModuleNotFoundError: No module named 'tldr.daemon.ensure'
        """
        from tldr.daemon import ensure  # noqa: F401 — just checking importability

    def test_ensure_daemon_callable_exported(self):
        """ensure_daemon must be a callable exported from tldr.daemon.ensure.

        RED: ImportError — module doesn't exist yet.
        """
        from tldr.daemon.ensure import ensure_daemon
        assert callable(ensure_daemon), "ensure_daemon must be callable"

    def test_ensure_daemon_accepts_project_and_timeout(self):
        """ensure_daemon(project, timeout=10.0) must accept both args without error
        when ping fast-paths immediately (monkeypatched to succeed).

        RED: ImportError — module doesn't exist.
        """
        from tldr.daemon.ensure import ensure_daemon

        # Monkeypatch: pretend the daemon is already running
        with patch("tldr.daemon.ensure._ping_daemon", return_value=True):
            # Should not raise, return None
            result = ensure_daemon("/tmp/fake-project", timeout=10.0)
        assert result is None, "ensure_daemon must return None"


# ===========================================================================
# TEST 2: ensure_daemon fast path — pings first, skips spawn
# ===========================================================================

class TestEnsureDaemonFastPath:
    """When _ping_daemon returns True, ensure_daemon must return immediately
    without acquiring any lock or spawning any subprocess.

    RED: ImportError — tldr.daemon.ensure does not exist.
    """

    def test_fast_path_returns_without_popen(self, tmp_path: Path):
        """If ping succeeds, subprocess.Popen must NOT be called.

        RED: ImportError on ensure module.
        """
        from tldr.daemon.ensure import ensure_daemon

        project = str(_tiny_project(tmp_path))
        popen_calls = []

        with patch("tldr.daemon.ensure._ping_daemon", return_value=True), \
             patch("subprocess.Popen", side_effect=lambda *a, **kw: popen_calls.append((a, kw)) or MagicMock()):
            ensure_daemon(project, timeout=5.0)

        assert popen_calls == [], (
            "ensure_daemon fast path must NOT call subprocess.Popen when daemon is alive. "
            "RED: module does not exist."
        )

    def test_fast_path_does_not_acquire_flock(self, tmp_path: Path):
        """If ping succeeds, no flock must be acquired (open() on lock file + fcntl.flock).

        RED: ImportError on ensure module.
        """
        from tldr.daemon.ensure import ensure_daemon

        project = str(_tiny_project(tmp_path))
        flock_calls = []

        import fcntl as _fcntl

        def track_flock(fd, op):
            flock_calls.append(op)

        with patch("tldr.daemon.ensure._ping_daemon", return_value=True), \
             patch("fcntl.flock", side_effect=track_flock):
            ensure_daemon(project, timeout=5.0)

        assert flock_calls == [], (
            "ensure_daemon fast path must NOT acquire flock when daemon is alive. "
            "RED: module does not exist."
        )


# ===========================================================================
# TEST 3: ensure_daemon slow path — spawns when daemon absent
# ===========================================================================

class TestEnsureDaemonSlowPath:
    """When _ping_daemon returns False initially then True after spawn,
    ensure_daemon must: acquire flock, re-ping under lock, spawn Popen, wait.

    RED: ImportError — tldr.daemon.ensure does not exist.
    """

    def test_slow_path_calls_popen_when_daemon_absent(self, tmp_path: Path):
        """When ping fails, ensure_daemon must call subprocess.Popen to spawn the daemon.

        RED: ImportError on ensure module.
        """
        from tldr.daemon.ensure import ensure_daemon

        project = str(_tiny_project(tmp_path))
        popen_called = []

        mock_proc = MagicMock()

        # ping: False (not running), then False (under lock, still not running),
        # then True (after spawn — wait loop succeeds immediately)
        ping_side_effects = [False, False, True]
        ping_iter = iter(ping_side_effects)

        def fake_ping(p):
            try:
                return next(ping_iter)
            except StopIteration:
                return True

        def fake_popen(*args, **kwargs):
            popen_called.append((args, kwargs))
            return mock_proc

        with patch("tldr.daemon.ensure._ping_daemon", side_effect=fake_ping), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("time.sleep"), \
             patch("fcntl.flock"):
            ensure_daemon(project, timeout=5.0)

        assert len(popen_called) == 1, (
            "ensure_daemon slow path must call subprocess.Popen exactly once. "
            f"Got {len(popen_called)} calls. "
            "RED: module does not exist."
        )

    def test_slow_path_popen_launches_daemon_cli(self, tmp_path: Path):
        """Popen must be called with the daemon start CLI command.

        RED: ImportError on ensure module.
        """
        from tldr.daemon.ensure import ensure_daemon

        project = str(_tiny_project(tmp_path))
        popen_args_captured = []

        # ping: initially False, then True after spawn
        ping_values = [False, False, True]
        ping_iter = iter(ping_values)

        def fake_ping(p):
            try:
                return next(ping_iter)
            except StopIteration:
                return True

        def fake_popen(cmd, *args, **kwargs):
            popen_args_captured.append(cmd)
            return MagicMock()

        with patch("tldr.daemon.ensure._ping_daemon", side_effect=fake_ping), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("time.sleep"), \
             patch("fcntl.flock"):
            ensure_daemon(project, timeout=5.0)

        assert len(popen_args_captured) == 1
        cmd = popen_args_captured[0]
        # Must launch the tldr CLI daemon start command
        assert "daemon" in cmd or any("daemon" in str(a) for a in cmd), (
            f"Popen cmd must include 'daemon', got: {cmd!r}. "
            "RED: module does not exist."
        )
        assert "start" in cmd or any("start" in str(a) for a in cmd), (
            f"Popen cmd must include 'start', got: {cmd!r}. "
            "RED: module does not exist."
        )

    def test_slow_path_waits_for_ready_via_ping_loop(self, tmp_path: Path):
        """After spawning, ensure_daemon must poll ping until ready (or timeout).

        Asserts that _ping_daemon is called more than once when the first
        post-spawn ping fails but subsequent ones succeed.

        RED: ImportError on ensure module.
        """
        from tldr.daemon.ensure import ensure_daemon

        project = str(_tiny_project(tmp_path))
        ping_call_count = []

        # pre-lock: False; under-lock: False; wait-loop: False, False, True
        ping_sequence = [False, False, False, False, True]
        ping_iter = iter(ping_sequence)

        def counting_ping(p):
            result = next(ping_iter, True)
            ping_call_count.append(result)
            return result

        with patch("tldr.daemon.ensure._ping_daemon", side_effect=counting_ping), \
             patch("subprocess.Popen", return_value=MagicMock()), \
             patch("time.sleep"), \
             patch("fcntl.flock"):
            ensure_daemon(project, timeout=5.0)

        assert len(ping_call_count) >= 3, (
            f"ensure_daemon must poll _ping_daemon multiple times waiting for ready. "
            f"Only {len(ping_call_count)} calls. "
            "RED: module does not exist."
        )


# ===========================================================================
# TEST 4: mcp_server delegates to shared ensure_daemon — no private duplicate
# ===========================================================================

class TestMcpServerDelegatesToSharedEnsureDaemon:
    """tldr.mcp_server must import ensure_daemon from tldr.daemon.ensure
    rather than defining its own inline _ensure_daemon.

    After the refactor:
      - tldr.mcp_server must NOT define a function named _ensure_daemon
      - tldr.mcp_server._ensure_daemon (if present as a reference) must be
        the same object as tldr.daemon.ensure.ensure_daemon

    RED: tldr.daemon.ensure does not exist; mcp_server still has its own
    inline _ensure_daemon body.
    """

    def test_mcp_server_imports_ensure_daemon_from_shared_module(self):
        """mcp_server must reference ensure_daemon from tldr.daemon.ensure,
        not define its own private copy.

        RED reason (two layers):
          1. tldr.daemon.ensure doesn't exist → ImportError on the ensure module
          2. Even if it existed: mcp_server._ensure_daemon is still inline today
        """
        # First, ensure the shared module exists (will fail RED until implemented)
        from tldr.daemon.ensure import ensure_daemon as shared_ensure_daemon  # RED

        import tldr.mcp_server as mcp_mod

        # The mcp_server module must NOT have a self-contained _ensure_daemon body.
        # It should delegate; we detect this by checking that the mcp_server's
        # _ensure_daemon (or its internal _send_command) ultimately calls the
        # shared ensure_daemon.
        #
        # Strongest form: verify mcp_server does NOT define _ensure_daemon as its
        # own function object distinct from the shared one.
        mcp_ensure = getattr(mcp_mod, "_ensure_daemon", None)

        # After refactor: _ensure_daemon in mcp_server should either:
        #   (a) not exist as a standalone function (fully removed), or
        #   (b) be the same object as the shared ensure_daemon
        if mcp_ensure is not None:
            assert mcp_ensure is shared_ensure_daemon, (
                "mcp_server._ensure_daemon must be the shared ensure_daemon from "
                "tldr.daemon.ensure, not a private duplicate. "
                "RED: mcp_server still has its own inline body today."
            )

    def test_mcp_server_does_not_have_standalone_inline_ensure_body(self):
        """mcp_server must not contain its own flock/Popen/_ensure_daemon body.

        We detect this by inspecting the source of mcp_server._ensure_daemon:
        if it still contains the inline flock logic, it hasn't been refactored.

        RED: mcp_server._ensure_daemon today is defined inline with its own
        subprocess.Popen + fcntl.flock body.
        """
        import inspect
        import tldr.mcp_server as mcp_mod

        # After refactor, _ensure_daemon should be removed from mcp_server's
        # module-level namespace OR be a thin import alias.
        # Check that the module source no longer defines the function body inline.
        mcp_source = inspect.getsource(mcp_mod)

        # The inline body always contains 'fcntl.flock' and the Popen spawn.
        # After the refactor, these should NOT appear in mcp_server.py's own
        # _ensure_daemon definition.
        #
        # Strategy: look for the combination of _ensure_daemon definition + flock
        # This is a heuristic but catches the most common un-refactored case.
        has_inline_flock_in_ensure = (
            "def _ensure_daemon" in mcp_source
            and "fcntl.flock" in mcp_source
        )
        assert not has_inline_flock_in_ensure, (
            "tldr/mcp_server.py still defines _ensure_daemon inline with fcntl.flock. "
            "After the refactor, mcp_server.py must import ensure_daemon from "
            "tldr.daemon.ensure and not duplicate the flock/Popen logic. "
            "RED: the inline body is still present today."
        )
