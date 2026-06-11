"""Daemon-side notify-storm robustness — Layer-1 spec tests.

Feature: split the notify path into an O(1) lock-held ingress (`_handle_notify`
accumulates into `_dirty_files`) and a deferred, debounced, cooldown-gated
scheduler (`_schedule_reindex` → timer → `_drain_pending_reindex`) that arms at
most one timer per burst and folds a large pending set into one bulk reindex.

These tests verify the 6 DoD (Definition of Done) behaviors via injectable
timer/clock seams:
  - `__init__` keyword-only `timer_factory` / `clock` params
  - `_notify_debounce_secs` / `_reindex_cooldown_secs` config knobs
  - `_reindex_sched_lock`, `_reindex_timer`, `_last_reindex_fire_at` fields
  - `_schedule_reindex`, `_drain_pending_reindex`, `_cooldown_remaining`,
    `_cancel_reindex_timer` methods
  - `_handle_notify` calling `_schedule_reindex()` instead of
    `_trigger_background_reindex()` inline

PURE UNIT: every test constructs `TLDRDaemon(tmp_path)` directly — no socket
bind, no real subprocess, no model server, no GPU. Timing is driven through the
injected `timer_factory`/`clock` seams (delay collapses to inline calls; the
clock is a settable fake). No test calls `time.sleep`, spawns a wall-clock timer,
or runs a real subprocess. The single FOLD test (DoD-3) lets the real
`_trigger_background_reindex` run but mocks `subprocess.run` and joins the worker
thread deterministically.

The `TLDRDaemon` is imported lazily inside each test (or a helper called at test
time), never at module level, so a missing symbol yields a FAIL, not a
collection ERROR.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Isolation fixture (per tests/test_daemon_root_svn_anchor.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_root_resolution(monkeypatch, tmp_path):
    """Isolate _find_project_root from the ambient environment.

    - CLAUDE_PROJECT_DIR short-circuits root resolution → delete it.
    - A stray ``.tldr`` ABOVE the tmp project would let the walk-up anchor
      there (the marker-pollution trap) → assert there is none.
    """
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    cur = tmp_path.resolve()
    while cur != cur.parent:
        assert not (cur / ".tldr").exists(), f"stray .tldr ancestor: {cur / '.tldr'}"
        cur = cur.parent


# ---------------------------------------------------------------------------
# Test doubles for the injectable timing seams
# ---------------------------------------------------------------------------

class _ImmediateTimerHandle:
    """A _TimerHandle whose ``.start()`` fires the callback synchronously.

    Records the delay it was armed with so tests can assert the scheduled value
    (debounce / cooldown-remaining). ``.cancel()`` is a no-op once fired.
    """

    def __init__(self, delay, fn, record):
        self.delay = delay
        self._fn = fn
        self._record = record
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True
        self._record["last_delay"] = self.delay
        self._record["fire_count"] = self._record.get("fire_count", 0) + 1
        self._fn()

    def cancel(self):
        self.cancelled = True


def _immediate_timer_factory(record):
    """Return a timer factory that fires the callback inline on ``.start()``.

    ``record`` is a dict the factory writes the most-recent delay / handle into
    so the test can inspect the armed delay and the live handle.
    """

    def factory(delay, fn):
        handle = _ImmediateTimerHandle(delay, fn, record)
        record["last_handle"] = handle
        return handle

    return factory


class _DeferredTimerHandle:
    """A _TimerHandle that records the callback WITHOUT firing it.

    ``.start()`` only marks the timer live; the test fires it manually via
    ``.fire()``. This models a real armed (pending) timer for debounce tests:
    re-arming cancels the prior handle and creates a new one, so the surviving
    handle's callback is the only one that runs when the test fires it.
    """

    def __init__(self, delay, fn, record):
        self.delay = delay
        self._fn = fn
        self._record = record
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True
        self._record["last_delay"] = self.delay

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self._fn()


def _deferred_timer_factory(record):
    """Return a timer factory that records callbacks without auto-firing them."""

    def factory(delay, fn):
        handle = _DeferredTimerHandle(delay, fn, record)
        record.setdefault("handles", []).append(handle)
        record["last_handle"] = handle
        return handle

    return factory


class _FakeClock:
    """A settable monotonic clock; ``__call__`` returns the current virtual time."""

    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------
# Repo / daemon helpers (mirror tests/test_batch_thin_notify_daemon.py)
# ---------------------------------------------------------------------------

def _build_tiny_repo(tmp_path: Path) -> Path:
    """Create a minimal project directory with a .git anchor."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "a.py").write_text("def a(): pass\n")
    return tmp_path


def _make_daemon(project: Path, *, timer_factory=None, clock=None):
    """Construct a TLDRDaemon with optional injected timing seams.

    Imported lazily so a not-yet-implemented ``timer_factory``/``clock`` kwarg
    surfaces as a test FAILURE (TypeError on the keyword), never a module-level
    collection error.
    """
    from tldr.daemon.core import TLDRDaemon

    kwargs = {}
    if timer_factory is not None:
        kwargs["timer_factory"] = timer_factory
    if clock is not None:
        kwargs["clock"] = clock
    return TLDRDaemon(project, **kwargs)


def _set_threshold(daemon, n: int) -> None:
    """Override auto_reindex_threshold (the sole arming gate)."""
    daemon._semantic_config = {**daemon._semantic_config, "auto_reindex_threshold": n}


# ===========================================================================
# DoD-1 DEBOUNCE → B1: N rapid notifies coalesce to exactly ONE reindex fire.
# ===========================================================================

class TestDoD1Debounce:
    """A burst of N notifies across several calls must arm exactly ONE surviving
    timer; firing it triggers exactly ONE reindex. This is the COALESCING
    assertion (distinct from DoD-3's fold proof): `_trigger_background_reindex`
    is MOCKED — we only count fires.
    """

    def test_burst_arms_single_surviving_timer_firing_once_triggers_one_reindex(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_deferred_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        # debounce > 0 so the deferred timer does not fire inline.
        daemon._notify_debounce_secs = 2.0
        _set_threshold(daemon, 3)

        mock_reindex = MagicMock()
        daemon._trigger_background_reindex = mock_reindex

        # Five notifies across several calls, each beyond threshold → each
        # re-arms the scheduler. Debounce coalesces: only the LAST timer survives.
        for batch in (["/p/a.py", "/p/b.py", "/p/c.py"],
                      ["/p/d.py"], ["/p/e.py"], ["/p/f.py"]):
            daemon._handle_notify({"cmd": "notify", "files": batch})

        handles = record["handles"]
        live = [h for h in handles if not h.cancelled]
        assert len(live) == 1, (
            f"Expected exactly ONE surviving (un-cancelled) timer after the "
            f"burst, got {len(live)} live of {len(handles)} armed. Debounce must "
            f"cancel each prior pending timer on re-arm."
        )

        # No reindex has fired yet — it is deferred to the timer.
        assert mock_reindex.call_count == 0, (
            f"Reindex must not fire before the debounce timer does, got "
            f"call_count={mock_reindex.call_count}."
        )

        # Fire the one surviving timer → exactly one reindex.
        record["last_handle"].fire()
        assert mock_reindex.call_count == 1, (
            f"Burst of 5 notifies must coalesce to exactly ONE reindex fire, got "
            f"call_count={mock_reindex.call_count}."
        )


# ===========================================================================
# DoD-2 COOLDOWN → B2/B8: a burst within cooldown is deferred to the boundary.
# ===========================================================================

class TestDoD2Cooldown:
    """After a fire records `_last_reindex_fire_at`, a fresh burst within the
    cooldown window is pushed out to the cooldown boundary: the armed delay
    equals the remaining cooldown, and `_cooldown_remaining()` decays to 0 as
    the clock advances.
    """

    def test_burst_within_cooldown_arms_delay_equal_to_remaining_then_decays(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        clock = _FakeClock(0.0)
        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_immediate_timer_factory(record),
            clock=clock,
        )
        daemon._notify_debounce_secs = 1.0
        daemon._reindex_cooldown_secs = 30.0
        _set_threshold(daemon, 1)

        # First fire happens at t=0 (immediate factory) → records fire time 0.
        daemon._trigger_background_reindex = MagicMock()
        daemon._handle_notify({"cmd": "notify", "files": ["/p/a.py"]})
        assert daemon._last_reindex_fire_at == 0.0, (
            f"A fire must record _last_reindex_fire_at via the clock (=0.0), got "
            f"{daemon._last_reindex_fire_at!r}."
        )

        # Advance the clock partway into the cooldown (t=10 of 30) and schedule
        # again. Remaining cooldown = 20, which exceeds debounce (1.0), so the
        # armed delay must be the remaining cooldown.
        clock.t = 10.0
        assert daemon._cooldown_remaining() == pytest.approx(20.0), (
            f"At t=10 with cooldown=30 from fire@0, remaining must be 20.0, got "
            f"{daemon._cooldown_remaining()!r}."
        )
        daemon._handle_notify({"cmd": "notify", "files": ["/p/b.py"]})
        assert record["last_delay"] == pytest.approx(20.0), (
            f"A burst within cooldown must arm a delay equal to the remaining "
            f"cooldown (20.0), got armed delay {record.get('last_delay')!r}."
        )

        # Advance past the cooldown boundary → remaining decays to exactly 0.
        clock.t = 100.0
        assert daemon._cooldown_remaining() == 0.0, (
            f"Past the cooldown boundary, _cooldown_remaining must clamp to 0.0, "
            f"got {daemon._cooldown_remaining()!r}."
        )


# ===========================================================================
# DoD-3 FOLD → B3: M paths collapse to ONE subprocess carrying the FULL set.
# (Real _trigger_background_reindex; subprocess.run mocked; thread joined.)
# ===========================================================================

class TestDoD3Fold:
    """A pending set of M >> threshold paths must collapse to a SINGLE
    `subprocess.run` whose `--dirty-files` JSON hint contains the FULL path set —
    whether delivered in one call or split across many. This does NOT mock
    `_trigger_background_reindex`; it exercises the real snapshot-spawn and mocks
    only `subprocess.run`, forcing epoch continuity so the hint is emitted.
    """

    def _force_epoch_continuity(self, daemon, monkeypatch):
        """Make epoch_continuous True so --dirty-files is included in the cmd.

        The hint is emitted only when `current_epoch != 0 and
        current_epoch == _watch_start_epoch` (core.py). Pin both to the same
        non-zero value by stubbing `_read_index_epoch`.
        """
        daemon._watch_start_epoch = 12345
        monkeypatch.setattr(daemon, "_read_index_epoch", lambda project: 12345)

    def _run_fold(self, project, paths_calls, threshold, monkeypatch):
        """Drive a fold: deliver paths via paths_calls, return (cmd, dirty_content).

        Returns (argv, dirty_files_set) passed to the (mocked) subprocess.run,
        after joining the spawned do_reindex worker thread deterministically.
        The dirty-files hint file is read WITHIN the mock (before the production
        finally block can unlink it), then returned for the caller to assert on.
        """
        import subprocess

        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_immediate_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        daemon._notify_debounce_secs = 0.0
        daemon._reindex_cooldown_secs = 0.0
        _set_threshold(daemon, threshold)
        self._force_epoch_continuity(daemon, monkeypatch)

        captured: dict = {}
        done = threading.Event()

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = list(cmd)
            # Read the hint file content HERE, inside the mock, while the
            # do_reindex worker still holds dirty_files_path before the finally
            # block can unlink it.
            cmd_list = list(cmd)
            if "--dirty-files" in cmd_list:
                hint_path = cmd_list[cmd_list.index("--dirty-files") + 1]
                try:
                    import json as _json
                    with open(hint_path) as fh:
                        captured["dirty_set"] = set(_json.load(fh))
                except Exception as e:
                    captured["dirty_read_error"] = str(e)
            done.set()
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        # Snapshot live (non-daemon-managed) threads so we can join the worker.
        before = set(threading.enumerate())
        for call in paths_calls:
            daemon._handle_notify({"cmd": "notify", "files": call})

        # do_reindex runs in a real Thread; wait (bounded) for the mocked run.
        fired = done.wait(timeout=5.0)
        assert fired, (
            "The real _trigger_background_reindex never reached subprocess.run "
            "within 5s — the fold path did not run (RED: scheduler/seam missing)."
        )
        for t in set(threading.enumerate()) - before:
            t.join(timeout=5.0)

        return captured.get("cmd"), captured.get("dirty_set")

    def _full_set_from_result(self, cmd, dirty_set):
        assert cmd is not None, "subprocess.run was never invoked"
        assert "--dirty-files" in cmd, (
            f"--dirty-files hint missing from cmd (epoch continuity not forced?): "
            f"{cmd!r}"
        )
        assert dirty_set is not None, (
            "dirty_set was not captured inside the mock (hint file unreadable?)"
        )
        return dirty_set

    def test_single_call_of_50_folds_to_one_subprocess_with_full_set(self, tmp_path, monkeypatch):
        project = _build_tiny_repo(tmp_path)
        paths = [f"/p/f{i}.py" for i in range(50)]
        cmd, dirty_set = self._run_fold(project, [paths], threshold=5, monkeypatch=monkeypatch)
        full = self._full_set_from_result(cmd, dirty_set)
        assert full == set(paths), (
            f"The fold must snapshot the FULL 50-path set into one bulk job; "
            f"got {len(full)} paths in the --dirty-files hint."
        )

    def test_fold_collapses_to_exactly_one_subprocess_invocation(self, tmp_path, monkeypatch):
        """50 paths in one call → exactly ONE subprocess.run (never N spawns)."""
        import subprocess

        project = _build_tiny_repo(tmp_path)
        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_immediate_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        daemon._notify_debounce_secs = 0.0
        daemon._reindex_cooldown_secs = 0.0
        _set_threshold(daemon, 5)
        self._force_epoch_continuity(daemon, monkeypatch)

        calls = {"n": 0}
        done = threading.Event()

        def fake_run(cmd, *args, **kwargs):
            calls["n"] += 1
            done.set()
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        before = set(threading.enumerate())
        paths = [f"/p/f{i}.py" for i in range(50)]
        daemon._handle_notify({"cmd": "notify", "files": paths})

        assert done.wait(timeout=5.0), (
            "subprocess.run never invoked within 5s (RED: fold path missing)."
        )
        for t in set(threading.enumerate()) - before:
            t.join(timeout=5.0)

        assert calls["n"] == 1, (
            f"50 paths must collapse to exactly ONE subprocess invocation, got "
            f"{calls['n']}."
        )

    def test_split_across_ten_calls_folds_to_one_subprocess_with_full_set(self, tmp_path, monkeypatch):
        project = _build_tiny_repo(tmp_path)
        paths = [f"/p/f{i}.py" for i in range(50)]
        calls = [paths[i:i + 5] for i in range(0, 50, 5)]  # 10 calls of 5
        assert len(calls) == 10 and all(len(c) == 5 for c in calls)
        cmd, dirty_set = self._run_fold(project, calls, threshold=5, monkeypatch=monkeypatch)
        full = self._full_set_from_result(cmd, dirty_set)
        assert full == set(paths), (
            f"Paths split across 10 calls must still fold into ONE bulk job "
            f"carrying all 50 paths; got {len(full)} in the hint."
        )


# ===========================================================================
# DoD-4 NON-BLOCKING → B4: ack returns synchronously; heavy reindex deferred.
# ===========================================================================

class TestDoD4NonBlocking:
    """With debounce > 0, `_handle_notify` returns its ack dict synchronously and
    does NOT invoke the heavy reindex inline — it is armed for later.
    """

    def test_handle_notify_returns_ack_without_inline_reindex_when_debounced(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_deferred_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        daemon._notify_debounce_secs = 2.0
        _set_threshold(daemon, 3)

        mock_reindex = MagicMock()
        daemon._trigger_background_reindex = mock_reindex

        response = daemon._handle_notify(
            {"cmd": "notify", "files": ["/p/a.py", "/p/b.py", "/p/c.py"]}
        )

        assert response.get("status") == "ok", (
            f"Ack must be status=='ok', got {response!r}."
        )
        assert response.get("reindex_triggered") is True, (
            f"A threshold-crossing batch must report reindex_triggered True "
            f"(meaning scheduled), got {response!r}."
        )
        assert mock_reindex.call_count == 0, (
            f"With debounce>0 the heavy reindex must NOT run inline in "
            f"_handle_notify; it fires later when the timer does. Got "
            f"call_count={mock_reindex.call_count}."
        )


# ===========================================================================
# DoD-5 NO REGRESSION → B5/B7: dedup + single-flight preserved.
# ===========================================================================

class TestDoD5NoRegression:
    """Dedup (same path across two calls counts once) and single-flight
    (`_reindex_in_progress` blocks a second fire) survive the rewire. Each
    single-flight assertion uses a fresh daemon / resets the flag, since the
    side_effect that sets the flag never clears it.
    """

    def test_same_path_across_two_calls_counts_once(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_deferred_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        daemon._notify_debounce_secs = 2.0
        _set_threshold(daemon, 99)  # never arm; we only test dedup accounting
        daemon._trigger_background_reindex = MagicMock()

        daemon._handle_notify({"cmd": "notify", "files": ["/p/dup.py"]})
        daemon._handle_notify({"cmd": "notify", "files": ["/p/dup.py"]})

        assert daemon._dirty_count == 1, (
            f"Same path notified across two calls must count once (existing "
            f"is_new guard), got dirty_count={daemon._dirty_count}."
        )

    def test_single_flight_blocks_second_fire_via_in_progress_flag(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_immediate_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        daemon._notify_debounce_secs = 0.0
        daemon._reindex_cooldown_secs = 0.0
        _set_threshold(daemon, 2)

        mock_reindex = MagicMock()
        # Mirror the real method: a fire sets the in-progress flag (and never
        # clears it here, so a fresh daemon is required to see a second fire).
        mock_reindex.side_effect = lambda: setattr(daemon, "_reindex_in_progress", True)
        daemon._trigger_background_reindex = mock_reindex

        # First burst fires → flag set. Second burst must be suppressed.
        daemon._handle_notify({"cmd": "notify", "files": ["/p/a.py", "/p/b.py"]})
        daemon._handle_notify({"cmd": "notify", "files": ["/p/c.py", "/p/d.py"]})

        assert mock_reindex.call_count == 1, (
            f"Single-flight: with _reindex_in_progress set after the first fire, "
            f"the second burst must not fire again. Got "
            f"call_count={mock_reindex.call_count}."
        )


# ===========================================================================
# DoD-6 CONFIG → B6: knobs read via _load_semantic_config; 0 → immediate fire.
# ===========================================================================

class TestDoD6Config:
    """The two knobs default correctly and load from `.tldr/config.json`; with
    `notify_debounce_secs=0` and `reindex_cooldown_secs=0` a threshold batch
    arms a zero-delay timer that fires immediately.
    """

    def test_knobs_default_when_no_config_present(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        assert daemon._notify_debounce_secs == pytest.approx(2.0), (
            f"Default notify_debounce_secs must be 2.0, got "
            f"{daemon._notify_debounce_secs!r}."
        )
        assert daemon._reindex_cooldown_secs == pytest.approx(30.0), (
            f"Default reindex_cooldown_secs must be 30.0, got "
            f"{daemon._reindex_cooldown_secs!r}."
        )

    def test_config_file_overrides_knobs_and_zero_delay_fires_immediately(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        tldr_dir = project / ".tldr"
        tldr_dir.mkdir(exist_ok=True)
        (tldr_dir / "config.json").write_text(json.dumps({
            "semantic": {
                "notify_debounce_secs": 0,
                "reindex_cooldown_secs": 0,
                "auto_reindex_threshold": 3,
            }
        }))

        record: dict = {}
        daemon = _make_daemon(
            project,
            timer_factory=_immediate_timer_factory(record),
            clock=_FakeClock(0.0),
        )
        assert daemon._notify_debounce_secs == 0, (
            f"notify_debounce_secs must load 0 from config, got "
            f"{daemon._notify_debounce_secs!r}."
        )
        assert daemon._reindex_cooldown_secs == 0, (
            f"reindex_cooldown_secs must load 0 from config, got "
            f"{daemon._reindex_cooldown_secs!r}."
        )

        mock_reindex = MagicMock()
        daemon._trigger_background_reindex = mock_reindex

        daemon._handle_notify({"cmd": "notify", "files": ["/p/a.py", "/p/b.py", "/p/c.py"]})

        assert record.get("last_delay") == 0, (
            f"With both knobs 0 the armed delay must be 0 (immediate), got "
            f"{record.get('last_delay')!r}."
        )
        assert mock_reindex.call_count == 1, (
            f"A zero-delay immediate timer must fire the reindex once inline, got "
            f"call_count={mock_reindex.call_count}."
        )


# ===========================================================================
# B9: cancel idempotency + production timer is a daemon thread (M3).
# ===========================================================================

class TestB9CancelAndDaemonTimer:
    """`_cancel_reindex_timer` is idempotent (cancels once, clears the field),
    and the production timer-factory builds a daemon `threading.Timer`.
    """

    def test_cancel_reindex_timer_is_idempotent(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        fake_timer = MagicMock()
        daemon._reindex_timer = fake_timer
        daemon._cancel_reindex_timer()
        daemon._cancel_reindex_timer()

        assert fake_timer.cancel.call_count == 1, (
            f"cancel() must be invoked exactly once across two idempotent "
            f"_cancel_reindex_timer calls, got {fake_timer.cancel.call_count}."
        )
        assert daemon._reindex_timer is None, (
            f"_reindex_timer must be None after cancel, got "
            f"{daemon._reindex_timer!r}."
        )

    def test_production_timer_factory_builds_daemon_thread(self, tmp_path):
        project = _build_tiny_repo(tmp_path)
        # Default factory (timer_factory=None resolves to the production wrapper).
        daemon = _make_daemon(project)

        timer = daemon._timer_factory(60.0, lambda: None)
        try:
            assert isinstance(timer, threading.Timer), (
                f"Production factory must build a threading.Timer, got {type(timer)!r}."
            )
            assert timer.daemon is True, (
                "Production timer must be a daemon thread (M3) so an abrupt exit "
                "never leaves a zombie timer blocking interpreter shutdown."
            )
        finally:
            timer.cancel()

    def test_production_timer_factory_is_async_not_synchronous(self, tmp_path):
        """Production timer must NOT fire the callback inline on .start().

        This test fails against the old _DaemonSyncTimer (which fires synchronously)
        and passes after the MUST_FIX 1 fix (real threading.Timer with async delay).
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        fired = threading.Event()
        # Use a 5-second delay — far longer than the assertions below take.
        timer = daemon._timer_factory(5.0, lambda: fired.set())
        try:
            assert isinstance(timer, threading.Timer), (
                f"Production factory must build a threading.Timer, got {type(timer)!r}."
            )
            assert timer.daemon is True, (
                "Production timer must be a daemon thread (M3)."
            )

            timer.start()

            # Immediately after .start(), the callback must NOT have fired yet —
            # this is the key assertion that fails on the old synchronous design.
            assert not fired.is_set(), (
                "Production timer fired the callback synchronously on .start() — "
                "it must be async (threading.Timer waits the full delay)."
            )
            assert timer.is_alive(), (
                "Production timer thread must be alive right after .start() "
                "(waiting for the 5s delay to elapse)."
            )
        finally:
            # Cancel so the 5s timer never actually fires; keeps the test fast.
            timer.cancel()
