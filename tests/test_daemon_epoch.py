"""
Tests for DaemonEpoch and transactional _dirty_files clear (DAEMON-EPOCH feature).

Feature: daemon/core.py gains:
  - index_epoch written to metadata.json by persist() (incremental_indexer.py)
  - _read_index_epoch(project) -> int  (catches OSError/JSONDecodeError/KeyError -> 0)
  - _watch_start_epoch field initialized at daemon init via _read_index_epoch
  - Transactional subtract: _dirty_files -= files_this_run (replacing .clear())
  - Epoch continuity check: trust dirty-hint ONLY when watch_start_epoch == index_epoch

All tests are RED on HEAD 792eff8 because:
  1. index_epoch field is absent from metadata.json (persist() does not write it)
  2. _read_index_epoch does not exist in TLDRDaemon
  3. _watch_start_epoch does not exist on TLDRDaemon
  4. _dirty_files.clear() (core.py:893) loses files added during reindex
  5. No epoch continuity check exists before --dirty-files is added to cmd

Test strategy: construct TLDRDaemon pointing at a tmp_path project (no socket,
no subprocess); monkeypatch subprocess.run / threading.Thread where needed;
test _read_index_epoch directly once it exists as a free function or method.
"""

from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Embedding dimension used by the fake model (matches existing test convention)
# ---------------------------------------------------------------------------
_DIM = 4


def _make_fake_model() -> MagicMock:
    """Return a MagicMock whose encode() returns np.ones-normalised dim-4 vectors."""
    mock_model = MagicMock()

    def fake_encode(texts, batch_size=128, normalize_embeddings=True,
                    show_progress_bar=False):
        n = len(texts) if isinstance(texts, list) else 1
        vecs = np.ones((n, _DIM), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode.side_effect = fake_encode
    return mock_model


def _build_tiny_repo(tmp_path: Path) -> Path:
    """Create a minimal Python project with .git anchor."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "file_a.py").write_text(
        "def foo(x):\n    \"\"\"Compute foo.\"\"\"\n    return x + 1\n"
    )
    (tmp_path / "file_b.py").write_text(
        "def bar(y):\n    \"\"\"Compute bar.\"\"\"\n    return y * 2\n"
    )
    return tmp_path


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text())


def _make_daemon(project: Path) -> "TLDRDaemon":
    """Construct a TLDRDaemon without starting a socket or subprocess."""
    from tldr.daemon.core import TLDRDaemon
    daemon = TLDRDaemon(project)
    return daemon


# ===========================================================================
# TEST 1: index_epoch written to metadata.json by persist()
# ===========================================================================

class TestIndexEpochWrittenOnPersist:
    """After build_semantic_index, metadata.json must contain an index_epoch field
    that is a strictly positive integer (int(time.time_ns())-shaped value).

    RED: persist() currently writes {"units":..., "model":..., "dimension":...,
    "count":...} — no index_epoch field.
    """

    def test_index_epoch_present_in_metadata_after_build(self, tmp_path: Path):
        """Build a semantic index; read metadata.json; assert index_epoch present and > 0.

        RED reason: IncrementalIndexer.persist() does not write 'index_epoch' today.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta = _read_metadata(project)
        assert "index_epoch" in meta, (
            "metadata.json must contain 'index_epoch' after persist(). "
            "RED: persist() does not write this field today."
        )
        epoch = meta["index_epoch"]
        assert isinstance(epoch, int), (
            f"index_epoch must be an int (int(time.time_ns())), got {type(epoch)}"
        )
        # time.time_ns() in 2024+ is > 1.7e18; any plausible value is > 0
        assert epoch > 0, f"index_epoch must be strictly positive, got {epoch}"

    def test_second_index_has_strictly_greater_epoch(self, tmp_path: Path):
        """A rebuild produces a strictly greater index_epoch than the prior index.

        RED reason: index_epoch field does not exist, so the comparison is
        vacuously impossible to test — first assertion above will fail.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- First build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        meta_first = _read_metadata(project)
        epoch_first = meta_first.get("index_epoch", None)
        assert epoch_first is not None, "index_epoch missing after first build"
        assert isinstance(epoch_first, int), "index_epoch must be int"

        # Modify a file to force a non-trivial second persist
        (project / "file_b.py").write_text(
            "def bar(y):\n    \"\"\"Modified bar.\"\"\"\n    return y * 3\n"
        )

        # --- Second build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        meta_second = _read_metadata(project)
        epoch_second = meta_second.get("index_epoch", None)
        assert epoch_second is not None, "index_epoch missing after second build"
        assert isinstance(epoch_second, int), "index_epoch must be int"

        assert epoch_second > epoch_first, (
            f"Second build's index_epoch ({epoch_second}) must be STRICTLY "
            f"GREATER than first build's ({epoch_first}). "
            f"RED: index_epoch not written; both are None today."
        )


# ===========================================================================
# TEST 2: _read_index_epoch error containment
# ===========================================================================

class TestReadIndexEpochErrorContainment:
    """_read_index_epoch returns 0 on any read/parse/key error.

    RED: _read_index_epoch does not exist on TLDRDaemon today.
    """

    def test_read_index_epoch_returns_zero_on_missing_metadata(self, tmp_path: Path):
        """_read_index_epoch on a project with no metadata.json returns 0, no exception.

        RED: AttributeError — '_read_index_epoch' is not defined on TLDRDaemon.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        # No build: no .tldr/cache/semantic/metadata.json

        daemon = _make_daemon(project)

        # _read_index_epoch must exist and return 0 for missing file
        assert hasattr(daemon, "_read_index_epoch") or callable(
            getattr(type(daemon), "_read_index_epoch", None)
        ), "TLDRDaemon must have a _read_index_epoch method"

        result = daemon._read_index_epoch(project)
        assert result == 0, (
            f"_read_index_epoch on missing metadata.json must return 0, got {result!r}. "
            f"RED: _read_index_epoch does not exist (AttributeError expected today)."
        )

    def test_read_index_epoch_returns_zero_on_corrupt_json(self, tmp_path: Path):
        """_read_index_epoch on corrupt JSON returns 0, no exception propagates.

        RED: AttributeError — '_read_index_epoch' is not defined on TLDRDaemon.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        # Write deliberately corrupt metadata.json
        cache_dir = project / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True)
        (cache_dir / "metadata.json").write_text("{this is not valid json")

        daemon = _make_daemon(project)
        result = daemon._read_index_epoch(project)
        assert result == 0, (
            f"_read_index_epoch on corrupt JSON must return 0, got {result!r}. "
            f"RED: method does not exist today."
        )

    def test_read_index_epoch_returns_zero_on_missing_field(self, tmp_path: Path):
        """_read_index_epoch on valid JSON lacking 'index_epoch' key returns 0.

        RED: AttributeError — '_read_index_epoch' is not defined on TLDRDaemon.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        cache_dir = project / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True)
        # Valid JSON but missing index_epoch
        (cache_dir / "metadata.json").write_text(
            json.dumps({"units": [], "model": "test", "dimension": 4, "count": 0})
        )

        daemon = _make_daemon(project)
        result = daemon._read_index_epoch(project)
        assert result == 0, (
            f"_read_index_epoch with no 'index_epoch' key must return 0, got {result!r}."
        )

    def test_read_index_epoch_returns_integer_when_present(self, tmp_path: Path):
        """_read_index_epoch returns the stored integer when the field is present.

        RED: AttributeError — '_read_index_epoch' is not defined on TLDRDaemon.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        cache_dir = project / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True)
        expected_epoch = 1717200000123456789
        (cache_dir / "metadata.json").write_text(
            json.dumps({
                "units": [], "model": "test", "dimension": 4, "count": 0,
                "index_epoch": expected_epoch,
            })
        )

        daemon = _make_daemon(project)
        result = daemon._read_index_epoch(project)
        assert result == expected_epoch, (
            f"_read_index_epoch must return {expected_epoch}, got {result!r}."
        )


# ===========================================================================
# TEST 3: Transactional _dirty_files subtract (C preserved when added during run)
# ===========================================================================

class TestTransactionalDirtyFilesSubtract:
    """_dirty_files -= files_this_run preserves files added DURING the reindex.

    The current code (core.py:893) does self._dirty_files.clear() which loses
    any file C that was added to _dirty_files by _handle_notify AFTER
    files_this_run was snapshotted but BEFORE the subprocess returned.

    RED anchor: after mocking the subprocess to "add C during execution", the
    current code's .clear() leaves _dirty_files == set(); the correct transactional
    subtract leaves _dirty_files == {"C"}.
    """

    def test_file_added_during_reindex_preserved_after_success(self, tmp_path: Path):
        """Simulate: files_this_run={A,B}; C added to _dirty_files during run;
        subprocess succeeds; assert _dirty_files == {"C"} (not empty).

        RED reason: core.py:893 calls self._dirty_files.clear() which removes C.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        file_a = str(project / "file_a.py")
        file_b = str(project / "file_b.py")
        file_c = str(project / "file_c.py")

        # Pre-populate _dirty_files with A and B
        daemon._dirty_files.add(file_a)
        daemon._dirty_files.add(file_b)

        # Mock subprocess.run to:
        # 1. Add C to _dirty_files (simulates _handle_notify during the subprocess)
        # 2. Return success (returncode=0)
        def fake_subprocess_run(cmd, **kwargs):
            # Simulate a new change notification arriving during the reindex
            daemon._dirty_files.add(file_c)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        # Run the trigger synchronously by running do_reindex in the same thread.
        # We monkeypatch threading.Thread to run synchronously.
        class SyncThread:
            def __init__(self, target=None, daemon=None, **kwargs):
                self._target = target
            def start(self):
                self._target()

        with patch("subprocess.run", side_effect=fake_subprocess_run), \
             patch("threading.Thread", SyncThread):
            daemon._trigger_background_reindex()

        # After success, ONLY files_this_run (A, B) should be removed.
        # C was added DURING the run; it must survive.
        assert file_c in daemon._dirty_files, (
            f"File C added during reindex must remain in _dirty_files after "
            f"successful reindex. "
            f"RED: self._dirty_files.clear() at core.py:893 removes C. "
            f"Fix: self._dirty_files -= files_this_run (transactional subtract). "
            f"Actual _dirty_files after run: {daemon._dirty_files!r}"
        )
        assert file_a not in daemon._dirty_files, (
            f"File A (in files_this_run) must be removed from _dirty_files after success."
        )
        assert file_b not in daemon._dirty_files, (
            f"File B (in files_this_run) must be removed from _dirty_files after success."
        )

    def test_dirty_files_unchanged_when_subprocess_fails(self, tmp_path: Path):
        """When the subprocess returns non-zero, _dirty_files is left intact.

        RED reason: current .clear() in the 'finally' block removes files even
        on failure. The new code must only subtract on SUCCESS.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        file_a = str(project / "file_a.py")
        file_b = str(project / "file_b.py")

        daemon._dirty_files.add(file_a)
        daemon._dirty_files.add(file_b)

        def fake_subprocess_fail(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "indexer failed"
            return mock_result

        class SyncThread:
            def __init__(self, target=None, daemon=None, **kwargs):
                self._target = target
            def start(self):
                self._target()

        with patch("subprocess.run", side_effect=fake_subprocess_fail), \
             patch("threading.Thread", SyncThread):
            daemon._trigger_background_reindex()

        # On failure, both files must still be in _dirty_files for the next run.
        assert file_a in daemon._dirty_files, (
            f"File A must remain in _dirty_files when subprocess fails. "
            f"Actual: {daemon._dirty_files!r}"
        )
        assert file_b in daemon._dirty_files, (
            f"File B must remain in _dirty_files when subprocess fails. "
            f"Actual: {daemon._dirty_files!r}"
        )


# ===========================================================================
# TEST 4: Epoch continuity — dirty hint omitted when epoch gapped
# ===========================================================================

class TestEpochContinuityCheck:
    """The daemon trusts the dirty-files hint ONLY when epoch is provably continuous.

    epoch_continuous = (_read_index_epoch(project) == self._watch_start_epoch)

    When the epochs do NOT match (cold start, daemon restart, out-of-band index),
    --dirty-files must be OMITTED from the subprocess command, falling back to
    the self-validating hash floor.

    RED: _watch_start_epoch and epoch continuity check do not exist today.
    """

    def test_dirty_files_omitted_from_cmd_when_epoch_mismatch(self, tmp_path: Path):
        """When _watch_start_epoch != current index_epoch, cmd must NOT contain
        '--dirty-files', even if _dirty_files is non-empty.

        RED: _watch_start_epoch does not exist on TLDRDaemon; the epoch continuity
        check is absent; --dirty-files is always included when _dirty_files is non-empty.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)
        daemon = _make_daemon(project)

        # Simulate an out-of-band index: write a metadata.json with a DIFFERENT epoch
        # than what the daemon initialized with (which is 0 for a fresh daemon on a
        # project with no prior index).
        cache_dir = project / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True)
        new_epoch = time.time_ns() + 999_000_000  # definitely different from 0
        (cache_dir / "metadata.json").write_text(
            json.dumps({
                "units": [], "model": "test", "dimension": 4, "count": 0,
                "index_epoch": new_epoch,
            })
        )

        # Daemon was initialized before this write; _watch_start_epoch = 0 (cold default)
        # _read_index_epoch(project) will now return new_epoch != 0 → epoch mismatch
        file_a = str(project / "file_a.py")
        daemon._dirty_files.add(file_a)

        captured_cmds = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        class SyncThread:
            def __init__(self, target=None, daemon=None, **kwargs):
                self._target = target
            def start(self):
                self._target()

        with patch("subprocess.run", side_effect=fake_subprocess_run), \
             patch("threading.Thread", SyncThread):
            daemon._trigger_background_reindex()

        assert len(captured_cmds) == 1, "Expected exactly one subprocess.run call"
        cmd = captured_cmds[0]
        assert "--dirty-files" not in cmd, (
            f"--dirty-files must NOT be in subprocess cmd when epoch is mismatched "
            f"(epoch continuity not proven). "
            f"RED: epoch continuity check absent; --dirty-files always included. "
            f"Cmd was: {cmd!r}"
        )

    def test_dirty_files_included_in_cmd_when_epoch_matches(self, tmp_path: Path):
        """When _watch_start_epoch == current index_epoch, --dirty-files IS included.

        RED: _watch_start_epoch does not exist; epoch check absent.
        """
        from tldr.daemon.core import TLDRDaemon

        project = _build_tiny_repo(tmp_path)

        # Write a metadata.json with epoch = 42 BEFORE constructing the daemon,
        # so that _watch_start_epoch is initialized to 42 at __init__ time.
        cache_dir = project / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True)
        expected_epoch = 42
        (cache_dir / "metadata.json").write_text(
            json.dumps({
                "units": [], "model": "test", "dimension": 4, "count": 0,
                "index_epoch": expected_epoch,
            })
        )

        daemon = _make_daemon(project)

        # Verify _watch_start_epoch was initialized from metadata
        assert hasattr(daemon, "_watch_start_epoch"), (
            "TLDRDaemon must have a _watch_start_epoch field. "
            "RED: field does not exist today."
        )
        assert daemon._watch_start_epoch == expected_epoch, (
            f"_watch_start_epoch must be {expected_epoch} (read from metadata.json at init), "
            f"got {daemon._watch_start_epoch!r}."
        )

        # _read_index_epoch(project) will return 42 == _watch_start_epoch → continuous
        file_a = str(project / "file_a.py")
        daemon._dirty_files.add(file_a)

        captured_cmds = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_cmds.append(list(cmd))
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        class SyncThread:
            def __init__(self, target=None, daemon=None, **kwargs):
                self._target = target
            def start(self):
                self._target()

        with patch("subprocess.run", side_effect=fake_subprocess_run), \
             patch("threading.Thread", SyncThread):
            daemon._trigger_background_reindex()

        assert len(captured_cmds) == 1, "Expected exactly one subprocess.run call"
        cmd = captured_cmds[0]
        assert "--dirty-files" in cmd, (
            f"--dirty-files MUST be in subprocess cmd when epoch is continuous "
            f"(_watch_start_epoch == _read_index_epoch(project)). "
            f"RED: epoch check absent; --dirty-files inclusion not epoch-gated. "
            f"Cmd was: {cmd!r}"
        )
