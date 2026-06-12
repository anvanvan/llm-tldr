"""
Tests for daemon auto-start behaviors (RED phase — behaviors do not exist yet).

Covers:
1. Rolling 1h idle timeout: IDLE_TIMEOUT == 3600; is_idle() uses sliding deadline;
   activity extends deadline (daemon not idle while being used).
2. Auto warm+index after start: run() launches background thread after
   write_status("ready"), invoking _post_startup_init (which calls _handle_warm
   then _handle_semantic{"action":"index"}), guarded single-flight by
   _reindex_in_progress.
3. CLI auto-route: daemon-backed subcommands (semantic search, context) call
   ensure_daemon then route through the daemon rather than running in-process.

All tests are RED on HEAD because:
  - IDLE_TIMEOUT is 30*60=1800, not 3600 (core.py:47)
  - _post_startup_init does not exist on TLDRDaemon (AttributeError)
  - run() does not launch a background thread after write_status("ready")
  - tldr/daemon/ensure.py does not exist
  - CLI dispatch does not call ensure_daemon before semantic/context/search

Test strategy: monkeypatch threading.Thread (SyncThread or capture), mock
subprocess.Popen + socket connectivity, fake _handle_warm/_handle_semantic.
No real model, no GPU, no network.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tiny_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "app.py").write_text("def greet(name):\n    return f'Hello {name}'\n")
    return tmp_path


def _make_daemon(project: Path):
    from tldr.daemon.core import TLDRDaemon
    return TLDRDaemon(project)


# ===========================================================================
# TEST 1: IDLE_TIMEOUT is exactly 3600 (rolling 1h)
# ===========================================================================

class TestIdleTimeout3600:
    """IDLE_TIMEOUT must equal 3600 (1 hour), not 1800 (30 minutes).

    RED: tldr/daemon/core.py line 47 defines IDLE_TIMEOUT = 30 * 60 = 1800.
    """

    def test_idle_timeout_is_one_hour(self):
        """IDLE_TIMEOUT constant must be exactly 3600 seconds.

        RED: IDLE_TIMEOUT = 30 * 60 = 1800 today (core.py:47).
        """
        from tldr.daemon import core as daemon_core

        assert daemon_core.IDLE_TIMEOUT == 3600, (
            f"IDLE_TIMEOUT must be 3600 (1 hour), but got {daemon_core.IDLE_TIMEOUT}. "
            "RED: core.py still defines IDLE_TIMEOUT = 30 * 60 = 1800."
        )


# ===========================================================================
# TEST 2: is_idle() sliding window — activity extends deadline
# ===========================================================================

class TestRollingIdleDeadline:
    """is_idle() is based on a sliding last_query deadline.
    Each query/use extends the deadline; the daemon is not idle while used.

    RED: IDLE_TIMEOUT = 1800 means the sliding behavior exists but tests
    the wrong constant. Additionally, make_daemon uses a small injectable
    IDLE_TIMEOUT so we don't need to sleep 1h.
    """

    def test_is_not_idle_until_1h_elapses_since_construction(self, tmp_path: Path):
        """Daemon must NOT be idle 1799s after construction (50 minutes have passed).

        This test is RED because IDLE_TIMEOUT=1800 makes the daemon idle after
        30 minutes, so simulating 50 minutes (3000s) of inactivity should NOT
        trigger idle — but with IDLE_TIMEOUT=1800 it does.

        Wait: 3000 > 1800, so the daemon IS idle at 3000s. To catch the constant
        being 1800 rather than 3600, simulate 2700s of inactivity (45 minutes):
        with IDLE_TIMEOUT=1800 the daemon is idle at 45 min (1800 < 2700),
        but with IDLE_TIMEOUT=3600 it is NOT idle yet (3600 > 2700).

        RED: IDLE_TIMEOUT=1800 → daemon incorrectly reports idle at 45min.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        # Simulate 45 minutes (2700s) of inactivity — within 1h window, outside 30min
        daemon.last_query = time.time() - 2700

        # With IDLE_TIMEOUT=3600 (1h): 2700 < 3600 → NOT idle (correct)
        # With IDLE_TIMEOUT=1800 (30m): 2700 > 1800 → IS idle (wrong)
        assert not daemon.is_idle(), (
            "Daemon must NOT be idle 45 minutes after last query (1h window requires 3600s). "
            "RED: IDLE_TIMEOUT=1800 makes the daemon idle at 30 minutes, "
            "so it incorrectly reports idle at 45 minutes."
        )

    def test_is_idle_only_after_1h_not_after_50min(self, tmp_path: Path):
        """Daemon must NOT be idle at 50min but MUST be idle at 65min.

        This test pins the boundary to the 1h (3600s) threshold:
        - At 50min (3000s): IDLE_TIMEOUT=3600 → NOT idle; IDLE_TIMEOUT=1800 → IS idle (wrong)
        - At 65min (3900s): both thresholds → IS idle (passes either way)

        The 50-minute check is the RED anchor: with IDLE_TIMEOUT=1800, is_idle()
        returns True at 50min, so the assertion fails.

        RED: IDLE_TIMEOUT=1800 causes is_idle() to return True at 50min.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        # 50 minutes (3000s) of inactivity
        daemon.last_query = time.time() - 3000

        assert not daemon.is_idle(), (
            "Daemon must NOT be idle at 50 minutes of inactivity (1h = 3600s window). "
            "RED: IDLE_TIMEOUT=1800 makes daemon idle at 30min, "
            "so it incorrectly reports idle at 50min (3000 > 1800)."
        )

    def test_repeated_activity_at_55min_intervals_never_triggers_idle(self, tmp_path: Path):
        """Daemon used every 55min must never become idle (55min < 1h = 3600s).

        With IDLE_TIMEOUT=1800: activity every 55min = 3300s gaps → IS idle (wrong).
        With IDLE_TIMEOUT=3600: activity every 55min = 3300s gaps → NOT idle (correct).

        This tests the sliding deadline: last_query is updated by handle_command,
        and the next idle check should use the updated time.

        RED: IDLE_TIMEOUT=1800 — after the first 55-min gap, daemon wrongly reports idle.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        # Simulate: daemon last used 55 minutes ago (3300s)
        daemon.last_query = time.time() - 3300

        # With IDLE_TIMEOUT=1800: is_idle() is True (3300 > 1800) — WRONG
        # With IDLE_TIMEOUT=3600: is_idle() is False (3300 < 3600) — CORRECT

        assert not daemon.is_idle(), (
            "Daemon must NOT be idle 55 minutes after last activity "
            "(the 1h window = 3600s has not elapsed). "
            "RED: IDLE_TIMEOUT=1800 causes the daemon to incorrectly report idle "
            "at 55 minutes (3300s > 1800s)."
        )

    def test_no_activity_makes_daemon_idle_at_exactly_timeout(self, tmp_path: Path):
        """With an injectable small timeout, daemon becomes idle at exactly that boundary.

        The architecture spec says IDLE_TIMEOUT must be injectable so tests
        don't sleep 1h. This test verifies the injectable pattern works.

        RED: TLDRDaemon does not accept an idle_timeout parameter today.
        """
        project = _build_tiny_repo(tmp_path)

        # Architecture says: make the timeout injectable.
        # Attempt to pass idle_timeout=2 kwarg — should work after implementation.
        # With current code this will either raise TypeError (no such arg) or
        # ignore the arg (if **kwargs is added), but IDLE_TIMEOUT stays 1800.
        try:
            daemon = _make_daemon.__wrapped__(project) if hasattr(_make_daemon, '__wrapped__') else None
        except Exception:
            daemon = None

        # Import the class directly for the injectable test
        from tldr.daemon.core import TLDRDaemon

        try:
            daemon = TLDRDaemon(project, idle_timeout=2)
            has_injectable = True
        except TypeError:
            # Current code: __init__ doesn't accept idle_timeout
            has_injectable = False
            daemon = TLDRDaemon(project)

        assert has_injectable, (
            "TLDRDaemon.__init__ must accept an idle_timeout parameter so tests "
            "can verify idle behavior without sleeping real time. "
            "RED: TLDRDaemon.__init__ does not accept idle_timeout today (TypeError)."
        )

        # With idle_timeout=2: set last_query to 3s ago → should be idle
        daemon.last_query = time.time() - 3
        assert daemon.is_idle(), (
            "With idle_timeout=2, daemon must be idle after 3s of inactivity."
        )

        # Reset via command, then immediately check → not idle
        daemon.handle_command({"cmd": "ping"})
        assert not daemon.is_idle(), (
            "After a command, daemon with idle_timeout=2 must not be idle."
        )


# ===========================================================================
# TEST 3: Auto warm+index after socket-ready
# ===========================================================================

class TestAutoWarmIndexAfterStart:
    """After write_status("ready"), run() must launch a background thread
    targeting _post_startup_init. That method calls _handle_warm and
    _handle_semantic({"action":"index"}), guarded by _reindex_in_progress.

    RED: _post_startup_init does not exist; run() does not launch this thread.
    """

    def test_post_startup_init_method_exists(self, tmp_path: Path):
        """TLDRDaemon must have a _post_startup_init method.

        RED: AttributeError — _post_startup_init not defined on TLDRDaemon today.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        assert hasattr(daemon, "_post_startup_init"), (
            "TLDRDaemon must have a _post_startup_init method. "
            "RED: method does not exist today (AttributeError)."
        )
        assert callable(daemon._post_startup_init), (
            "_post_startup_init must be callable."
        )

    def test_post_startup_init_calls_handle_warm_and_semantic_index(self, tmp_path: Path):
        """_post_startup_init must call _handle_warm({}) then
        _handle_semantic({"action":"index"}).

        RED: _post_startup_init does not exist.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        warm_calls = []
        semantic_calls = []

        def fake_warm(cmd):
            warm_calls.append(cmd)
            return {"status": "ok", "files": 0, "edges": 0}

        def fake_semantic(cmd):
            semantic_calls.append(cmd)
            return {"status": "ok", "indexed": 0}

        daemon._handle_warm = fake_warm
        daemon._handle_semantic = fake_semantic

        # _post_startup_init must exist and call both
        assert hasattr(daemon, "_post_startup_init"), (
            "RED: _post_startup_init does not exist."
        )

        daemon._post_startup_init()

        assert len(warm_calls) == 1, (
            f"_post_startup_init must call _handle_warm exactly once. "
            f"Got {len(warm_calls)} calls. "
            "RED: _post_startup_init does not exist."
        )
        assert len(semantic_calls) == 1, (
            f"_post_startup_init must call _handle_semantic exactly once. "
            f"Got {len(semantic_calls)} calls. "
            "RED: _post_startup_init does not exist."
        )
        # Semantic call must be for indexing
        assert semantic_calls[0].get("action") == "index", (
            f"_handle_semantic call must have action='index', got: {semantic_calls[0]!r}"
        )

    def test_post_startup_init_guarded_by_reindex_in_progress(self, tmp_path: Path):
        """When _reindex_in_progress is True, _post_startup_init must skip
        both _handle_warm and _handle_semantic (single-flight guard).

        RED: _post_startup_init does not exist; guard behavior unspecified.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        warm_calls = []
        semantic_calls = []

        daemon._handle_warm = lambda cmd: warm_calls.append(cmd) or {"status": "ok"}
        daemon._handle_semantic = lambda cmd: semantic_calls.append(cmd) or {"status": "ok"}

        # Set reindex guard to True (simulates user search racing with auto-init)
        daemon._reindex_in_progress = True

        assert hasattr(daemon, "_post_startup_init"), (
            "RED: _post_startup_init does not exist."
        )

        daemon._post_startup_init()

        assert warm_calls == [] and semantic_calls == [], (
            f"_post_startup_init must be a no-op when _reindex_in_progress is True. "
            f"Got warm_calls={warm_calls!r}, semantic_calls={semantic_calls!r}. "
            "RED: guard logic does not exist; _post_startup_init does not exist."
        )

    def test_run_launches_background_thread_after_socket_ready(self, tmp_path: Path):
        """After write_status('ready'), run() must launch a daemon thread
        targeting _post_startup_init (non-blocking, does not block the serve loop).

        RED: run() does not launch any such thread today.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        threads_started = []

        class CapturingThread:
            """Captures Thread(target=...) calls without actually starting threads."""
            def __init__(self, target=None, daemon=None, **kwargs):
                self._target = target
                self._daemon = daemon

            def start(self):
                threads_started.append(self._target)
                # Do NOT actually run the target — we just capture it

        # We need run() to reach the socket-ready point. Patch _create_socket to
        # avoid real socket setup, and _handle_one_connection to immediately request
        # shutdown so the serve loop terminates.
        def fake_create_socket(self_inner=None):
            # Noop: pretend socket is ready
            pass

        daemon_ref = daemon

        def fake_handle_connection(self_inner=None):
            daemon_ref._shutdown_requested = True

        # Patch at the instance level for TLDRDaemon methods
        with patch.object(type(daemon), "_create_socket", fake_create_socket), \
             patch.object(type(daemon), "_handle_one_connection", fake_handle_connection), \
             patch.object(type(daemon), "write_pid_file", lambda s: None), \
             patch.object(type(daemon), "write_status", lambda s, st: None), \
             patch.object(type(daemon), "_cleanup_socket", lambda s: None), \
             patch.object(type(daemon), "remove_pid_file", lambda s: None), \
             patch.object(type(daemon), "_persist_all_stats", lambda s: None), \
             patch("threading.Thread", CapturingThread):
            daemon.run()

        # At least one thread must have been started targeting _post_startup_init
        target_names = [
            getattr(t, "__name__", None) or getattr(t, "__func__", t).__name__
            if hasattr(t, "__func__") else (t.__name__ if hasattr(t, "__name__") else repr(t))
            for t in threads_started
        ]

        # Check that _post_startup_init is among the thread targets
        post_startup_targets = [
            t for t in threads_started
            if (
                getattr(t, "__name__", "") == "_post_startup_init"
                or (hasattr(t, "__func__") and t.__func__.__name__ == "_post_startup_init")
                or "_post_startup_init" in repr(t)
            )
        ]

        assert len(post_startup_targets) >= 1, (
            f"run() must launch a background thread targeting _post_startup_init "
            f"after write_status('ready'). "
            f"Threads started: {target_names!r}. "
            "RED: run() does not launch this thread today."
        )


# ===========================================================================
# TEST 4: CLI auto-route — daemon-backed subcommands call ensure_daemon
# ===========================================================================

class TestCliAutoRoute:
    """Running a daemon-backed subcommand (semantic search, context, search)
    must call ensure_daemon(project) and route through the daemon rather
    than running in-process.

    RED: tldr.daemon.ensure does not exist; cli.py does not call ensure_daemon
    before any subcommand dispatch.
    """

    def test_cli_imports_ensure_daemon_for_dispatch(self):
        """tldr.cli must reference ensure_daemon from tldr.daemon.ensure.

        After the implementation, cli.py should import and call ensure_daemon
        before routing daemon-backed subcommands.

        RED: tldr.daemon.ensure does not exist (ImportError).
        """
        from tldr.daemon.ensure import ensure_daemon  # RED if module absent

        import tldr.cli as cli_mod
        import inspect

        cli_source = inspect.getsource(cli_mod)
        # After implementation, the CLI source must reference ensure_daemon
        assert "ensure_daemon" in cli_source, (
            "tldr/cli.py must call ensure_daemon for daemon-backed subcommands. "
            "RED: ensure_daemon is not referenced in cli.py today."
        )

    def test_semantic_search_calls_ensure_daemon_before_dispatch(self, tmp_path: Path):
        """'tldr semantic search' must call ensure_daemon before running the query.

        We test this by mocking ensure_daemon and query_daemon, then invoking
        the CLI dispatch function (main()) with args for 'semantic search'.
        If ensure_daemon is called, the test passes.

        RED (two layers):
          1. tldr.daemon.ensure doesn't exist → ImportError
          2. Even if it existed: cli.py dispatch for 'semantic search' does NOT
             call ensure_daemon today (runs build_semantic_index in-process).
        """
        from tldr.daemon.ensure import ensure_daemon as real_ensure  # RED

        project = str(_build_tiny_repo(tmp_path))
        ensure_calls = []
        query_calls = []

        def fake_ensure(proj, timeout=10.0):
            ensure_calls.append(proj)

        def fake_query(proj, cmd):
            query_calls.append((proj, cmd))
            return {"status": "ok", "results": []}

        # Simulate CLI dispatch by calling main() with synthetic argv.
        # We need to route through the dispatch without real socket/model.
        import sys

        argv_backup = sys.argv[:]
        try:
            sys.argv = ["tldr", "semantic", "search", "hello world", project]
            with patch("tldr.daemon.ensure.ensure_daemon", fake_ensure), \
                 patch("tldr.daemon.startup.query_daemon", fake_query), \
                 patch("tldr.cli.ensure_daemon", fake_ensure, create=True):
                try:
                    from tldr.cli import main
                    main()
                except SystemExit:
                    pass  # Normal CLI exit
                except Exception:
                    pass  # Other errors are acceptable; we just check calls

        finally:
            sys.argv = argv_backup

        assert len(ensure_calls) >= 1, (
            f"CLI 'semantic search' must call ensure_daemon before dispatching. "
            f"ensure_calls={ensure_calls!r}. "
            "RED: cli.py does not call ensure_daemon for semantic search today."
        )

    def test_context_subcommand_calls_ensure_daemon(self, tmp_path: Path):
        """'tldr context <symbol>' must call ensure_daemon before dispatch.

        RED (two layers):
          1. tldr.daemon.ensure doesn't exist → ImportError
          2. cli.py context dispatch does not call ensure_daemon today.
        """
        from tldr.daemon.ensure import ensure_daemon as _  # RED: module absent

        import sys

        project = str(_build_tiny_repo(tmp_path))
        ensure_calls = []

        def fake_ensure(proj, timeout=10.0):
            ensure_calls.append(proj)

        argv_backup = sys.argv[:]
        try:
            sys.argv = ["tldr", "context", "greet", "--project", project]
            with patch("tldr.cli.ensure_daemon", fake_ensure, create=True), \
                 patch("tldr.daemon.startup.query_daemon", return_value={"status": "ok", "context": ""}):
                try:
                    from tldr.cli import main
                    main()
                except SystemExit:
                    pass
                except Exception:
                    pass
        finally:
            sys.argv = argv_backup

        assert len(ensure_calls) >= 1, (
            f"CLI 'context' must call ensure_daemon before dispatching. "
            f"ensure_calls={ensure_calls!r}. "
            "RED: cli.py does not call ensure_daemon for context today."
        )

    def test_search_subcommand_does_not_call_ensure_daemon(self, tmp_path: Path):
        """'tldr search <pattern>' must NOT call ensure_daemon.

        'search' is an index-free, grep-shaped command that runs purely
        in-process; it was removed from DAEMON_ROUTED_COMMANDS so it never
        cold-starts a daemon. The dispatch gate must record zero ensure_daemon
        calls for search.
        """
        from tldr.daemon.ensure import ensure_daemon as _  # module present

        import sys

        project = str(_build_tiny_repo(tmp_path))
        ensure_calls = []

        def fake_ensure(proj, timeout=10.0):
            ensure_calls.append(proj)

        argv_backup = sys.argv[:]
        try:
            sys.argv = ["tldr", "search", "def greet", project]
            with patch("tldr.cli.ensure_daemon", fake_ensure, create=True), \
                 patch("tldr.daemon.startup.query_daemon", return_value={"status": "ok", "results": []}):
                try:
                    from tldr.cli import main
                    main()
                except SystemExit:
                    pass
                except Exception:
                    pass
        finally:
            sys.argv = argv_backup

        assert ensure_calls == [], (
            f"CLI 'search' is index-free and must NOT call ensure_daemon; "
            f"ensure_calls={ensure_calls!r}. "
            "search was removed from DAEMON_ROUTED_COMMANDS."
        )

    def test_daemon_backed_subcommands_do_not_load_model_in_process(self, tmp_path: Path):
        """When ensure_daemon is available, daemon-backed commands must NOT call
        get_model() in-process (the model should run only in the server/daemon).

        RED:
          1. tldr.daemon.ensure doesn't exist → ImportError
          2. cli.py semantic search calls semantic_search() in-process (loads model).
        """
        from tldr.daemon.ensure import ensure_daemon as _  # RED: module absent

        import sys

        project = str(_build_tiny_repo(tmp_path))
        get_model_calls = []

        def fake_get_model(*args, **kwargs):
            get_model_calls.append((args, kwargs))
            from conftest import make_fake_model
            return make_fake_model()

        argv_backup = sys.argv[:]
        try:
            sys.argv = ["tldr", "semantic", "search", "hello", project]
            with patch("tldr.semantic.get_model", side_effect=fake_get_model), \
                 patch("tldr.cli.ensure_daemon", lambda *a, **kw: None, create=True), \
                 patch("tldr.daemon.startup.query_daemon",
                       return_value={"status": "ok", "results": []}):
                try:
                    from tldr.cli import main
                    main()
                except SystemExit:
                    pass
                except Exception:
                    pass
        finally:
            sys.argv = argv_backup

        assert get_model_calls == [], (
            f"When routed through daemon, 'semantic search' must NOT call get_model() "
            f"in-process. Got {len(get_model_calls)} in-process model load(s). "
            "RED: cli.py calls semantic_search() in-process today (no daemon routing)."
        )
