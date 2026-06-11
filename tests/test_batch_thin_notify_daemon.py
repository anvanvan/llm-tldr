"""
Tests for _handle_notify batch wire-protocol behaviors (RED phase).

Feature: tldr/daemon/core.py _handle_notify must accept BOTH:
  - new batch wire:  {"cmd": "notify", "files": ["/p1", "/p2", ...]}
  - legacy wire:     {"cmd": "notify", "file": "/p"}

Behaviors under test:
  D1. BATCH wire:     {"files": [f1, f2, f3]} adds 3 dirty files, dirty_count +3
  D2. BACK-COMPAT:    {"file": f1} still adds 1 dirty file / dirty_count +1
  D3. BATCH DEDUP/THRESHOLD: batch reaching threshold triggers reindex exactly
      once; duplicate files in a batch do NOT double-count dirty_count

RED reason for D1 and D3 (current code):
  _handle_notify (core.py line 841-875) only reads command.get("file").
  The "files" key is completely ignored. Therefore:
    - D1: _handle_notify({"files": [...3 paths...]}) returns
          {"status": "error", "message": "Missing required parameter: file"}
          and dirty_count stays 0.
    - D3: batch with len(files) >= threshold never triggers the reindex path
          because dirty_count is never incremented from a batch message.

D2 note: the legacy single-"file" path IS handled by current code for
  dirty-tracking, but the post-change normalized response must also include
  "files_received": 1. On current code that key is absent → RED.
  A single test asserts both dirty_count +1 AND files_received == 1.

Test strategy: construct TLDRDaemon(tmp_path) without a real socket
  (following tests/test_daemon_epoch.py pattern); monkeypatch
  _trigger_background_reindex to avoid spawning subprocesses;
  call _handle_notify directly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


class _ImmediateTimerHandle:
    """Test double: fires the callback synchronously on .start()."""

    def __init__(self, delay, fn):
        self.delay = delay
        self._fn = fn
        self.cancelled = False

    def start(self):
        self._fn()

    def cancel(self):
        self.cancelled = True


def _immediate_timer_factory():
    """Return a timer factory that fires the callback inline on .start()."""
    def factory(delay, fn):
        return _ImmediateTimerHandle(delay, fn)
    return factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tiny_repo(tmp_path: Path) -> Path:
    """Create a minimal project directory with a .git anchor."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "a.py").write_text("def a(): pass\n")
    return tmp_path


def _make_daemon(project: Path, *, timer_factory=None):
    """Construct a TLDRDaemon pointing at *project* with no live socket."""
    from tldr.daemon.core import TLDRDaemon
    kwargs = {}
    if timer_factory is not None:
        kwargs["timer_factory"] = timer_factory
    return TLDRDaemon(project, **kwargs)


def _patch_reindex(daemon) -> MagicMock:
    """Replace _trigger_background_reindex with a no-op MagicMock.

    Prevents the daemon from spawning real subprocesses during tests.
    Returns the mock so callers can assert call_count.
    """
    mock = MagicMock()
    daemon._trigger_background_reindex = mock
    return mock


# ---------------------------------------------------------------------------
# D1. BATCH wire: {"files": [f1, f2, f3]} adds 3 dirty files, dirty_count +3
# ---------------------------------------------------------------------------

class TestBatchWireD1:
    """D1: A single {"cmd":"notify","files":[f1,f2,f3]} message must add all
    three files to _dirty_files and increment _dirty_count by 3.

    RED on current code: the "files" key is ignored; dirty_count stays 0
    and the handler returns status=="error" (missing "file" key).
    """

    def test_batch_three_files_increments_dirty_count_by_three(self, tmp_path):
        """_handle_notify with files=[f1,f2,f3] → dirty_count == 3.

        RED: current code ignores the "files" key → dirty_count stays 0.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        paths = ["/proj/a.py", "/proj/b.py", "/proj/c.py"]
        response = daemon._handle_notify({"cmd": "notify", "files": paths})

        # After a successful batch notify, dirty_count must equal len(paths).
        assert daemon._dirty_count == 3, (
            f"Expected dirty_count==3 after batch of 3 files, "
            f"got {daemon._dirty_count}. "
            f"RED: current _handle_notify ignores 'files' key → dirty_count stays 0."
        )

    def test_batch_three_files_are_all_in_dirty_files_set(self, tmp_path):
        """_handle_notify with files=[f1,f2,f3] → all three paths in _dirty_files.

        RED: current code ignores the "files" key → _dirty_files remains empty.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        paths = ["/proj/a.py", "/proj/b.py", "/proj/c.py"]
        daemon._handle_notify({"cmd": "notify", "files": paths})

        for p in paths:
            assert p in daemon._dirty_files, (
                f"Expected {p!r} in _dirty_files after batch notify. "
                f"RED: current code ignores 'files' key → set is empty."
            )

    def test_batch_response_status_is_ok(self, tmp_path):
        """_handle_notify with a valid "files" list must return status=="ok".

        RED: current code returns status=="error" (missing file key).
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        response = daemon._handle_notify({"cmd": "notify", "files": ["/proj/x.py"]})

        assert response.get("status") == "ok", (
            f"Expected response status=='ok', got {response!r}. "
            f"RED: current code returns status=='error' when 'file' key absent."
        )

    def test_batch_response_includes_files_received(self, tmp_path):
        """Response from a valid batch notify must include 'files_received' == 3.

        RED: current code returns an error dict with no 'files_received' key,
        and even if it succeeded, the current success path has no such key.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        paths = ["/proj/a.py", "/proj/b.py", "/proj/c.py"]
        response = daemon._handle_notify({"cmd": "notify", "files": paths})

        assert "files_received" in response, (
            f"Expected 'files_received' in response, got {response!r}. "
            f"RED: current code does not include 'files_received'."
        )
        assert response["files_received"] == 3, (
            f"Expected files_received==3, got {response.get('files_received')!r}."
        )


# ---------------------------------------------------------------------------
# D2. BACK-COMPAT wire: {"file": f1} still adds 1 dirty file / dirty_count +1
#     AND the post-change normalized response includes "files_received": 1
# ---------------------------------------------------------------------------

class TestBackCompatWireD2:
    """D2: the legacy {"cmd":"notify","file":"/p"} wire message must still
    add exactly 1 dirty file, increment dirty_count by 1, AND return a
    response that includes "files_received": 1 (the normalized key that the
    batch path also returns).

    RED on current code: the response today is
      {"status":"ok","dirty_count":1,"threshold":20,"reindex_triggered":False}
    — it contains NO "files_received" key.  The assertion on files_received
    makes every test in this class FAIL on current code for the right reason.
    """

    def test_single_file_legacy_wire_dirty_count_and_files_received(self, tmp_path):
        """Legacy {"file": "/x"} → dirty_count==1, path in _dirty_files,
        AND response contains files_received==1.

        The dirty_count/set assertions guard back-compat; the files_received
        assertion is RED on current code (key absent from response today).
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        response = daemon._handle_notify({"cmd": "notify", "file": "/proj/x.py"})

        # Back-compat: single file still tracked
        assert daemon._dirty_count == 1, (
            f"Expected dirty_count==1 after single-file notify, "
            f"got {daemon._dirty_count}."
        )
        assert "/proj/x.py" in daemon._dirty_files, (
            "Expected '/proj/x.py' in _dirty_files after single-file notify."
        )
        assert response.get("status") == "ok", (
            f"Expected status=='ok' for single-file notify, got {response!r}."
        )

        # Post-change normalization: single-file path is wrapped in a list
        # internally and reported via the same files_received key as batches.
        # RED: current response has no 'files_received' key.
        assert "files_received" in response, (
            f"Expected 'files_received' in single-file notify response, "
            f"got {response!r}. "
            f"RED: current code does not include 'files_received' in its response."
        )
        assert response["files_received"] == 1, (
            f"Expected files_received==1 for single-file notify, "
            f"got {response.get('files_received')!r}."
        )


# ---------------------------------------------------------------------------
# D3. BATCH DEDUP / THRESHOLD
# ---------------------------------------------------------------------------

class TestBatchDedupAndThresholdD3:
    """D3: batch dedup and threshold-triggered reindex behaviors.

    (a) Duplicate paths within a single batch MUST NOT double-count dirty_count.
    (b) A batch whose NEW-file count raises dirty_count to/above the threshold
        must trigger _trigger_background_reindex exactly once (single-flight).

    RED on current code: "files" key ignored → dirty_count never incremented
    from batch messages → reindex never triggered by batch → both (a) and (b) fail.
    """

    def test_duplicate_files_in_batch_not_double_counted(self, tmp_path):
        """files=[f1, f1, f2] → dirty_count == 2 (f1 counted once only).

        RED: current code ignores 'files' → dirty_count stays 0.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        # f1 appears twice in the batch
        paths = ["/proj/a.py", "/proj/a.py", "/proj/b.py"]
        daemon._handle_notify({"cmd": "notify", "files": paths})

        assert daemon._dirty_count == 2, (
            f"Expected dirty_count==2 (f1 deduped), got {daemon._dirty_count}. "
            f"RED: current code ignores 'files' key → dirty_count stays 0."
        )
        assert len(daemon._dirty_files) == 2, (
            f"Expected 2 paths in _dirty_files (deduped), "
            f"got {len(daemon._dirty_files)}."
        )

    def test_batch_reaching_threshold_triggers_reindex_once(self, tmp_path):
        """A batch of N NEW files where N >= threshold fires reindex exactly once.

        We set auto_reindex_threshold=3 and send a batch of 3 distinct files.
        Expected: _trigger_background_reindex called exactly 1 time.

        Uses an injected immediate timer factory so the scheduler fires the
        callback synchronously (no real timer thread), exercising the async
        contract correctly in a unit test context.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project, timer_factory=_immediate_timer_factory())
        daemon._notify_debounce_secs = 0.0
        daemon._reindex_cooldown_secs = 0.0
        mock_reindex = _patch_reindex(daemon)

        # Override threshold to 3 so a 3-file batch triggers it
        daemon._semantic_config = {**daemon._semantic_config, "auto_reindex_threshold": 3}

        paths = ["/proj/a.py", "/proj/b.py", "/proj/c.py"]
        daemon._handle_notify({"cmd": "notify", "files": paths})

        assert mock_reindex.call_count == 1, (
            f"Expected _trigger_background_reindex called exactly once, "
            f"got {mock_reindex.call_count}. "
            f"Scheduler must arm a timer that fires the reindex."
        )

    def test_batch_below_threshold_does_not_trigger_reindex(self, tmp_path):
        """A batch of N files where N < threshold must NOT trigger reindex.

        threshold=5, batch=3 files → dirty_count=3 < 5 → no reindex.
        This exercises that the threshold guard works correctly after the fix.

        RED: current code ignores 'files', so dirty_count=0 and the guard
        is never evaluated → mock is not called (accidentally green).
        We assert dirty_count==3 to confirm the files key was actually processed.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        mock_reindex = _patch_reindex(daemon)

        daemon._semantic_config = {**daemon._semantic_config, "auto_reindex_threshold": 5}

        paths = ["/proj/a.py", "/proj/b.py", "/proj/c.py"]
        daemon._handle_notify({"cmd": "notify", "files": paths})

        # dirty_count must be 3 (proving files key WAS processed)
        assert daemon._dirty_count == 3, (
            f"Expected dirty_count==3 after batch of 3 below threshold, "
            f"got {daemon._dirty_count}. "
            f"RED: current code ignores 'files' → dirty_count stays 0."
        )
        assert mock_reindex.call_count == 0, (
            f"Expected no reindex when batch below threshold, "
            f"got call_count={mock_reindex.call_count}."
        )

    def test_batch_reaching_threshold_reindex_single_flight(self, tmp_path):
        """Sending TWO batches that each breach the threshold triggers reindex
        only on the FIRST (single-flight: _reindex_in_progress blocks the second).

        threshold=2; batch1=[a,b] → triggers reindex → _reindex_in_progress=True;
        batch2=[c,d] → guard blocks; mock called only once total.

        Uses an injected immediate timer factory so the scheduler fires the
        callback synchronously in the unit-test context.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project, timer_factory=_immediate_timer_factory())
        daemon._notify_debounce_secs = 0.0
        daemon._reindex_cooldown_secs = 0.0
        mock_reindex = _patch_reindex(daemon)

        # When _trigger_background_reindex is called, it normally sets the flag;
        # we need to simulate that the flag gets set (since we mocked the method).
        def _side_effect():
            daemon._reindex_in_progress = True

        mock_reindex.side_effect = _side_effect

        daemon._semantic_config = {**daemon._semantic_config, "auto_reindex_threshold": 2}

        # First batch: 2 new files → threshold reached → reindex triggered
        daemon._handle_notify({"cmd": "notify", "files": ["/proj/a.py", "/proj/b.py"]})
        # Second batch: 2 more files → flag is set → reindex NOT triggered again
        daemon._handle_notify({"cmd": "notify", "files": ["/proj/c.py", "/proj/d.py"]})

        assert mock_reindex.call_count == 1, (
            f"Expected single-flight: reindex triggered exactly once, "
            f"got call_count={mock_reindex.call_count}. "
            f"Scheduler must arm timer → fires reindex; second burst suppressed by "
            f"in-progress flag."
        )

    def test_empty_files_list_returns_error(self, tmp_path):
        """{"files": []} (empty list) must return status=="error".

        Architecture spec: 'files' is empty list → treat same as missing → error.
        RED: current code returns error for a different reason (no "file" key),
        but the error MESSAGE would differ — current message is
        "Missing required parameter: file", post-change it must be
        "Missing required parameter: file or files".
        We assert the new error message to make the test RED on current code.
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        response = daemon._handle_notify({"cmd": "notify", "files": []})

        assert response.get("status") == "error", (
            f"Expected status=='error' for empty files list, got {response!r}."
        )
        # Post-change error message must mention BOTH "file" and "files"
        msg = response.get("message", "")
        assert "file or files" in msg, (
            f"Expected error message to contain 'file or files', got {msg!r}. "
            f"RED: current code says 'Missing required parameter: file' only."
        )

    def test_missing_both_keys_returns_error_with_combined_message(self, tmp_path):
        """{"cmd":"notify"} with neither "file" nor "files" key → error mentioning both.

        RED: current code returns "Missing required parameter: file" (missing
        "or files" in the message). Post-change it must say "file or files".
        """
        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)
        _patch_reindex(daemon)

        response = daemon._handle_notify({"cmd": "notify"})

        assert response.get("status") == "error", (
            f"Expected status=='error' for missing-keys command, got {response!r}."
        )
        msg = response.get("message", "")
        assert "file or files" in msg, (
            f"Expected error message 'Missing required parameter: file or files', "
            f"got {msg!r}. "
            f"RED: current code says 'Missing required parameter: file' only."
        )
