"""
Failing tests (RED phase) for the self-validating hash floor — the indexing
side of parse-skip-default.

All tests MUST FAIL on HEAD 792eff8 for the RIGHT reason:
  - _derive_dirty_set does not exist in tldr.semantic
  - _enumerate_live_files does not exist in tldr.semantic
  - load_snapshot / save_snapshot / SnapshotEntry do not exist in tldr.patch
  - build_semantic_index (manual/default path, no dirty_files kwarg) does a
    full re-parse every run → reused == 0 after an incremental call

Coverage:
  1. SELF-VALIDATING FLOOR: 1 file edited, no dirty_files kwarg → reused > 0
  2. NO-OP reindex: nothing changed → embedded 0, reused == total
  3. DELETION on manual path: delete a file, no dirty_files → deleted units absent, reused > 0
  4. RENAME: mv a.py b.py, no dirty_files → old units gone, new units present, reused > 0
  5. SNAPSHOT BACK-COMPAT: seed narrow file_hashes.json → no crash, valid index,
     subsequent no-op is incremental
  6. WIDER SNAPSHOT WRITTEN: after index, file_hashes.json has wide schema fields
  7. load_snapshot narrow back-compat: every returned entry is a SnapshotEntry dict
  8. load_snapshot corrupt/missing JSON → returns {}
  9. save_snapshot atomicity: failure before os.replace leaves old file intact
  10. _enumerate_live_files parity: == {u.file for u in units} after full index

Tests use _make_fake_model() + tmp_path mini-repos, no real embedding model.
No @pytest.mark.e2e — these are fast unit/integration tests.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
from conftest import make_fake_model as _make_fake_model


# ---------------------------------------------------------------------------
# Tiny Python repo file contents
# ---------------------------------------------------------------------------

_PY_FILE_A = """\
def foo(x):
    \"\"\"Compute foo of x.\"\"\"
    return x + 1
"""

_PY_FILE_B = """\
def bar(y):
    \"\"\"Compute bar of y.\"\"\"
    return y * 2
"""

_PY_FILE_B_MODIFIED = """\
def bar(y):
    \"\"\"Compute bar of y (modified).\"\"\"
    return y * 3
"""

_PY_FILE_C = """\
def baz():
    \"\"\"Simple baz.\"\"\"
    pass
"""


# ---------------------------------------------------------------------------
# Mini-repo builder helpers
# ---------------------------------------------------------------------------

def _build_two_file_repo(tmp_path: Path) -> Path:
    """Two-file Python project with .git anchor."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "file_a.py").write_text(_PY_FILE_A)
    (tmp_path / "file_b.py").write_text(_PY_FILE_B)
    return tmp_path


def _build_three_file_repo(tmp_path: Path) -> Path:
    """Three-file Python project with .git anchor."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "file_a.py").write_text(_PY_FILE_A)
    (tmp_path / "file_b.py").write_text(_PY_FILE_B)
    (tmp_path / "file_c.py").write_text(_PY_FILE_C)
    return tmp_path


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text())


def _read_file_hashes(project_root: Path) -> dict:
    hashes_path = project_root / ".tldr" / "cache" / "file_hashes.json"
    assert hashes_path.exists(), f"file_hashes.json not found at {hashes_path}"
    return json.loads(hashes_path.read_text())


def _parse_summary_line(stderr_output: str) -> tuple[int, int]:
    """Extract (embedded, reused) from 'Semantic index: embedded N, reused M units ...'."""
    import re
    m = re.search(r"embedded\s+(\d+),\s+reused\s+(\d+)", stderr_output)
    assert m is not None, (
        f"Could not parse embedded/reused counts from stderr: {stderr_output!r}\n"
        f"Expected format: 'Semantic index: embedded N, reused M units ...'"
    )
    return int(m.group(1)), int(m.group(2))


# ===========================================================================
# TEST 1: SELF-VALIDATING FLOOR — 1 file edited, NO dirty_files kwarg
# ===========================================================================

class TestSelfValidatingFloor:
    """Manual/default path (no dirty_files kwarg) skips parsing unchanged files.

    RED today: the manual path always calls _full_extract (re-parses ALL files)
    because dirty_files=None → use_parse_skip=False. The self-validating floor
    (_derive_dirty_set) does not exist yet, so file_a.py is always re-parsed even
    when nothing changed.

    The parse-skip test anchors on whether _process_file_for_extraction is called
    for the UNCHANGED file (file_a.py). With the floor, file_a.py must NOT be
    passed to the worker pool. Without the floor (today), it always is.
    """

    def test_one_file_edit_no_dirty_files_kwarg_skips_parsing_unchanged_file(
        self, tmp_path: Path, monkeypatch
    ):
        """Initial index; edit file_b only; reindex with NO dirty_files kwarg.

        Asserts that file_a.py is NOT re-parsed (not passed to the worker pool):
        the self-validating floor auto-derives dirty_set = {file_b.py} from
        file_hashes.json and passes files_to_parse={file_b.py} to
        extract_units_from_project.

        RED reason: the manual path (dirty_files=None → use_parse_skip=False) always
        calls _full_extract which passes files_to_parse=None → ALL files go to the
        worker pool → file_a.py is re-parsed even though it didn't change.
        _derive_dirty_set does not exist yet.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        # Sequential mode so the same-process mock intercepts worker calls
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Initial full build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            total_units = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert total_units >= 2, f"Expected at least 2 units, got {total_units}"

        # --- Edit only file_b ---
        (project / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # --- Incremental reindex: NO dirty_files kwarg (manual/default path) ---
        call_paths: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                    # NOTE: intentionally NO dirty_files= kwarg — this is the manual path.
                    # The self-validating floor must auto-derive dirty_set = {file_b.py}.
                )

        # file_a.py must NOT have been re-parsed (carried from prior index)
        assert not any("file_a" in p for p in call_paths), (
            f"file_a.py must NOT be re-parsed when only file_b changed "
            f"(self-validating floor auto-derives dirty set from file_hashes.json). "
            f"Paths processed by worker: {call_paths}. "
            f"RED: the manual path (dirty_files=None) calls _full_extract which "
            f"passes files_to_parse=None → ALL files go to the worker pool → "
            f"file_a.py is re-parsed. _derive_dirty_set does not exist yet."
        )
        # file_b.py MUST have been re-parsed (it changed)
        assert any("file_b" in p for p in call_paths), (
            f"file_b.py must have been re-parsed after being edited. "
            f"Paths processed: {call_paths}"
        )


# ===========================================================================
# TEST 2: NO-OP REINDEX — nothing changed → zero files re-parsed
# ===========================================================================

class TestNoOpSelfValidatingFloor:
    """No files changed between two calls → second call re-parses nothing.

    RED today: manual path has no floor → all files re-parsed on every call
    (_full_extract with files_to_parse=None → all files go to worker pool).
    """

    def test_noop_reindex_manual_path_skips_all_file_parsing(
        self, tmp_path: Path, monkeypatch
    ):
        """Initial index; no changes; reindex with NO dirty_files kwarg.

        Asserts: zero files passed to _process_file_for_extraction on second call
        (no file changed → _derive_dirty_set returns (set(), set()) → carry all →
        files_to_parse=set() → empty worker pool).

        RED reason: manual path (dirty_files=None → use_parse_skip=False) always
        calls _full_extract with files_to_parse=None → ALL files go to the worker
        pool even when nothing changed. _derive_dirty_set does not exist yet.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Initial build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            total_units = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert total_units >= 2

        # --- No-op reindex: same files, no changes ---
        call_paths: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                )

        # No file should be re-parsed when nothing changed
        py_files_parsed = [p for p in call_paths if p.endswith(".py")]
        assert len(py_files_parsed) == 0, (
            f"No .py files should be re-parsed on a no-op reindex (nothing changed). "
            f"Files processed: {py_files_parsed}. "
            f"RED: the manual path calls _full_extract with files_to_parse=None → "
            f"ALL files go to worker pool. _derive_dirty_set not implemented → "
            f"no-op detection does not exist."
        )


# ===========================================================================
# TEST 3: DELETION on the manual path — no dirty_files kwarg
# ===========================================================================

class TestDeletionSelfValidatingFloor:
    """Delete a file; reindex with NO dirty_files kwarg.

    Asserts:
      - Deleted file's units absent from metadata
      - file_a.py and file_c.py NOT re-parsed (carried from prior index)

    RED today: manual path has no floor → _full_extract always parses all existing
    files. Also, deletion detection via old_units (G-7) doesn't exist on the manual
    path (deleted file's units may appear or not depending on whether the file scan
    finds them; the key assertion here is that unchanged files are NOT re-parsed).
    """

    def test_deleted_file_units_absent_and_unchanged_files_not_reparsed(
        self, tmp_path: Path, monkeypatch
    ):
        """Build index on 3 files; delete file_b; reindex with no dirty_files kwarg.

        Asserts:
          - 'bar' (from file_b) absent from metadata after reindex
          - file_a.py and file_c.py NOT passed to _process_file_for_extraction
            (carried from prior index via _derive_dirty_set + _unified_extract)

        RED reason: manual path calls _full_extract with files_to_parse=None →
        file_a.py and file_c.py ARE re-parsed even though unchanged.
        _derive_dirty_set (with deletion from old_units) does not exist yet.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_three_file_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            total_before = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert total_before >= 3, f"Expected >=3 units before deletion, got {total_before}"

        # --- Delete file_b ---
        (project / "file_b.py").unlink()

        # --- Reindex with NO dirty_files kwarg ---
        call_paths: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                )

        # Check correctness: bar from deleted file_b must be absent
        meta = _read_metadata(project)
        names_after = {u["name"] for u in meta["units"]}
        assert "bar" not in names_after, (
            f"'bar' from deleted file_b.py must not appear in metadata after "
            f"no-dirty_files reindex. Found names: {sorted(names_after)}"
        )

        # Parse-skip assertion: file_a and file_c must NOT have been re-parsed
        assert not any("file_a" in p for p in call_paths), (
            f"file_a.py must NOT be re-parsed after file_b deletion "
            f"(file_a unchanged → carried from prior index). "
            f"Paths processed: {call_paths}. "
            f"RED: manual path calls _full_extract with files_to_parse=None → "
            f"file_a.py is always re-parsed. _derive_dirty_set not implemented."
        )
        assert not any("file_c" in p for p in call_paths), (
            f"file_c.py must NOT be re-parsed after file_b deletion "
            f"(file_c unchanged → carried from prior index). "
            f"Paths processed: {call_paths}."
        )


# ===========================================================================
# TEST 4: RENAME — mv file_b.py file_b_renamed.py, no dirty_files kwarg
# ===========================================================================

class TestRenameSelfValidatingFloor:
    """Rename a file; reindex with NO dirty_files kwarg.

    Asserts:
      - Units under old path (file_b.py) absent
      - Units under new path (file_b_renamed.py) present
      - file_a.py NOT re-parsed (carried from prior index)

    RED today: _derive_dirty_set not implemented → all files re-parsed on manual path.
    """

    def test_rename_file_units_updated_and_unchanged_file_not_reparsed(
        self, tmp_path: Path, monkeypatch
    ):
        """Build index; rename file_b.py → file_b_renamed.py; reindex no dirty_files.

        Asserts:
          - No unit with file == 'file_b.py' in metadata after rename
          - At least one unit with file == 'file_b_renamed.py' present
          - file_a.py NOT passed to _process_file_for_extraction
            (unchanged → carried from prior index)

        RED reason: _derive_dirty_set not implemented → manual path calls _full_extract
        with files_to_parse=None → ALL files re-parsed including file_a.py.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Rename file_b.py → file_b_renamed.py ---
        (project / "file_b.py").rename(project / "file_b_renamed.py")

        # --- Reindex with NO dirty_files kwarg ---
        call_paths: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                )

        meta = _read_metadata(project)
        files_in_meta = {u["file"] for u in meta["units"]}

        assert "file_b.py" not in files_in_meta, (
            f"Units from renamed file_b.py must not appear after rename. "
            f"Found files: {sorted(files_in_meta)}"
        )
        assert "file_b_renamed.py" in files_in_meta, (
            f"Units from renamed file_b_renamed.py must appear in metadata. "
            f"Found files: {sorted(files_in_meta)}"
        )

        # Parse-skip: file_a.py must NOT have been re-parsed
        assert not any("file_a" in p for p in call_paths), (
            f"file_a.py must NOT be re-parsed after file_b rename "
            f"(unchanged → carried from prior index). "
            f"Paths processed: {call_paths}. "
            f"RED: manual path calls _full_extract with files_to_parse=None → "
            f"file_a.py is always re-parsed. _derive_dirty_set not implemented."
        )


# ===========================================================================
# TEST 5: SNAPSHOT BACK-COMPAT — seed narrow file_hashes.json, then reindex
# ===========================================================================

class TestSnapshotBackCompat:
    """Seed an OLD narrow file_hashes.json ({rel: sha1}) before reindexing.

    Asserts:
      - No crash on first reindex (narrow snapshot handled gracefully)
      - Valid index produced (metadata has units, ntotal matches)
      - Subsequent no-op reindex skips re-parsing (parse-skip active after upgrade)

    RED today: load_snapshot does not exist; the self-validating floor (_derive_dirty_set)
    does not exist; the manual path always calls _full_extract regardless of snapshot.
    The no-op parse-skip assertion fails because all files are re-parsed every call.
    """

    def test_narrow_snapshot_survives_reindex_and_subsequent_noop_skips_parsing(
        self, tmp_path: Path, monkeypatch
    ):
        """Seed narrow file_hashes.json; first reindex (not crash); then no-op reindex.

        The narrow snapshot (no __schema_version__, plain SHA-1 string values) must
        be normalized by load_snapshot. After the first reindex that upgrades the
        snapshot to wide format, a subsequent no-op reindex must skip re-parsing all
        files (parse-skip active: files_to_parse=set() → zero worker calls).

        RED reason: load_snapshot does not exist; the manual path always does
        _full_extract → all files re-parsed regardless of snapshot state.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction
        from tldr.patch import compute_file_hash

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Compute real SHA-1 hashes for the files ---
        sha_a = compute_file_hash(str(project / "file_a.py"))
        sha_b = compute_file_hash(str(project / "file_b.py"))

        # --- Seed a NARROW file_hashes.json (old format, plain string values) ---
        cache_dir = project / ".tldr" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        narrow_snapshot = {
            "file_a.py": sha_a,
            "file_b.py": sha_b,
        }
        (cache_dir / "file_hashes.json").write_text(json.dumps(narrow_snapshot))

        # --- First reindex: must not crash, must produce valid index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            total = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert total >= 2, f"Expected >=2 units after reindex, got {total}"

        meta = _read_metadata(project)
        assert len(meta["units"]) == total, (
            f"metadata.json unit count ({len(meta['units'])}) != "
            f"build return value ({total})"
        )

        # --- No-op second reindex: files unchanged → zero files re-parsed ---
        call_paths: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                )

        py_files_parsed = [p for p in call_paths if p.endswith(".py")]
        assert len(py_files_parsed) == 0, (
            f"After narrow-snapshot seed + first reindex, the second (no-op) "
            f"reindex must skip re-parsing all files (nothing changed). "
            f"Files re-parsed: {py_files_parsed}. "
            f"RED: load_snapshot / _derive_dirty_set do not exist → manual path "
            f"always calls _full_extract → all files re-parsed every call."
        )


# ===========================================================================
# TEST 6: WIDER SNAPSHOT WRITTEN — sha1 + mtime_ns + size + inode
# ===========================================================================

class TestWiderSnapshotWritten:
    """After build_semantic_index, file_hashes.json must have wide schema fields.

    RED today: _compute_current_file_hashes returns {rel: sha1} (narrow);
    persist calls save_file_hash_cache which writes the narrow format.
    The wide fields (mtime_ns, size, inode, __schema_version__=2) are not written.
    """

    def test_file_hashes_json_has_wide_schema_after_index(self, tmp_path: Path):
        """Build index; read file_hashes.json; assert wide schema fields present.

        Asserts:
          - __schema_version__ == 2
          - Each file entry is a dict with keys: sha1, mtime_ns, size, inode
          - mtime_ns and size are positive integers (not sentinel -1 or 0)
          - inode is a positive integer

        RED reason: persist() writes narrow {rel: sha1} format via save_file_hash_cache.
        __schema_version__ does not exist; entry values are plain SHA-1 strings, not dicts.
        """
        from tldr.semantic import build_semantic_index

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        hashes = _read_file_hashes(project)

        assert "__schema_version__" in hashes, (
            f"file_hashes.json must have '__schema_version__' key after index. "
            f"Keys found: {sorted(k for k in hashes if not k.endswith('.py'))}. "
            f"RED: persist() calls save_file_hash_cache which writes the narrow "
            f"format (no __schema_version__ key)."
        )
        assert hashes["__schema_version__"] == 2, (
            f"__schema_version__ must be 2, got {hashes['__schema_version__']!r}"
        )

        for rel_path in ("file_a.py", "file_b.py"):
            entry = hashes.get(rel_path)
            assert isinstance(entry, dict), (
                f"file_hashes.json entry for {rel_path!r} must be a dict "
                f"(wide SnapshotEntry format), got {type(entry).__name__}: {entry!r}. "
                f"RED: current format stores plain SHA-1 string values."
            )
            for field in ("sha1", "mtime_ns", "size", "inode"):
                assert field in entry, (
                    f"Wide snapshot entry for {rel_path!r} missing field '{field}'. "
                    f"Got keys: {sorted(entry.keys())}"
                )
            assert isinstance(entry["sha1"], str) and len(entry["sha1"]) == 40, (
                f"sha1 must be a 40-char hex string, got {entry['sha1']!r}"
            )
            assert entry["mtime_ns"] > 0, (
                f"mtime_ns must be > 0 (real mtime), got {entry['mtime_ns']!r}"
            )
            assert entry["size"] > 0, (
                f"size must be > 0 (real file size), got {entry['size']!r}"
            )
            assert entry["inode"] > 0, (
                f"inode must be > 0 (real inode), got {entry['inode']!r}"
            )

    def test_metadata_json_has_index_epoch_after_index(self, tmp_path: Path):
        """After build_semantic_index, metadata.json must have 'index_epoch' field.

        The index_epoch is an explicit int(time.time_ns()) token written at persist
        time (NOT derived from the file mtime). Used by the daemon to verify epoch
        continuity before trusting the dirty-files hint.

        RED reason: persist() does not write 'index_epoch' to metadata.json today.
        """
        from tldr.semantic import build_semantic_index

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        t_before_ns = time.time_ns()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        t_after_ns = time.time_ns()

        meta = _read_metadata(project)
        assert "index_epoch" in meta, (
            f"metadata.json must have 'index_epoch' key after index. "
            f"Keys found: {sorted(meta.keys())}. "
            f"RED: persist() does not write 'index_epoch' today."
        )
        epoch = meta["index_epoch"]
        assert isinstance(epoch, int), (
            f"index_epoch must be an int, got {type(epoch).__name__}: {epoch!r}"
        )
        assert t_before_ns <= epoch <= t_after_ns, (
            f"index_epoch ({epoch}) must be in the range "
            f"[{t_before_ns}, {t_after_ns}] (nanoseconds since epoch). "
            f"Got epoch that is outside the window of the build_semantic_index call."
        )


# ===========================================================================
# TEST 7: load_snapshot NARROW BACK-COMPAT
# ===========================================================================

class TestLoadSnapshotNarrowBackCompat:
    """load_snapshot must normalize narrow {rel: sha1} entries to SnapshotEntry dicts.

    RED today: load_snapshot does not exist in tldr.patch.
    """

    def test_load_snapshot_narrow_format_returns_snapshot_entry_dicts(self, tmp_path: Path):
        """Write a narrow file_hashes.json; call load_snapshot; assert every value is a dict.

        Asserts:
          - load_snapshot returns a dict (not a tuple / version)
          - Each value is a dict (SnapshotEntry), NEVER a plain string
          - Each SnapshotEntry has: sha1, mtime_ns, size, inode
          - mtime_ns == 0, size == -1, inode == -1 (sentinels for narrow upgrade)

        RED reason: load_snapshot does not exist in tldr.patch.
        """
        # Import the not-yet-existing function — must raise ImportError/AttributeError → RED
        from tldr.patch import load_snapshot  # type: ignore[attr-defined]  # noqa

        # Write a narrow file_hashes.json
        cache_dir = tmp_path / ".tldr" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        narrow = {
            "file_a.py": "aabbcc" + "0" * 34,
            "file_b.py": "ddeeff" + "0" * 34,
        }
        (cache_dir / "file_hashes.json").write_text(json.dumps(narrow))

        entries = load_snapshot(str(tmp_path))

        assert isinstance(entries, dict), (
            f"load_snapshot must return a dict, got {type(entries).__name__}"
        )
        for rel_path, val in entries.items():
            if rel_path.startswith("__"):
                continue  # skip schema_version key if present at this level
            assert isinstance(val, dict), (
                f"load_snapshot must return SnapshotEntry dicts, NEVER raw strings. "
                f"Got {type(val).__name__}: {val!r} for key {rel_path!r}. "
                f"Narrow '{rel_path}: sha1' must be normalized to "
                f"{{sha1: ..., mtime_ns: 0, size: -1, inode: -1}}."
            )
            for field in ("sha1", "mtime_ns", "size", "inode"):
                assert field in val, (
                    f"SnapshotEntry for {rel_path!r} missing field '{field}'. "
                    f"Got keys: {sorted(val.keys())}"
                )
            assert val["mtime_ns"] == 0, (
                f"Narrow-upgraded entry must have sentinel mtime_ns=0, "
                f"got {val['mtime_ns']!r}"
            )
            assert val["size"] == -1, (
                f"Narrow-upgraded entry must have sentinel size=-1, "
                f"got {val['size']!r}"
            )
            assert val["inode"] == -1, (
                f"Narrow-upgraded entry must have sentinel inode=-1, "
                f"got {val['inode']!r}"
            )

    def test_load_snapshot_corrupt_json_returns_empty_dict(self, tmp_path: Path):
        """load_snapshot on corrupt JSON returns {} without raising.

        RED reason: load_snapshot does not exist in tldr.patch.
        """
        from tldr.patch import load_snapshot  # type: ignore[attr-defined]  # noqa

        cache_dir = tmp_path / ".tldr" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "file_hashes.json").write_text("INVALID JSON {{{{")

        entries = load_snapshot(str(tmp_path))
        assert entries == {}, (
            f"load_snapshot on corrupt JSON must return {{}}, got {entries!r}"
        )

    def test_load_snapshot_missing_file_returns_empty_dict(self, tmp_path: Path):
        """load_snapshot when file_hashes.json is absent returns {} without raising.

        RED reason: load_snapshot does not exist in tldr.patch.
        """
        from tldr.patch import load_snapshot  # type: ignore[attr-defined]  # noqa

        # No .tldr/cache/file_hashes.json created
        entries = load_snapshot(str(tmp_path))
        assert entries == {}, (
            f"load_snapshot on missing file must return {{}}, got {entries!r}"
        )

    def test_load_snapshot_wide_format_round_trip(self, tmp_path: Path):
        """Write a wide file_hashes.json; load_snapshot returns the same entries.

        RED reason: load_snapshot does not exist in tldr.patch.
        """
        from tldr.patch import load_snapshot, save_snapshot  # type: ignore[attr-defined]  # noqa

        cache_dir = tmp_path / ".tldr" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        wide_entries = {
            "file_a.py": {
                "sha1": "a" * 40,
                "mtime_ns": 1717200000123456789,
                "size": 420,
                "inode": 12345678,
            },
            "file_b.py": {
                "sha1": "b" * 40,
                "mtime_ns": 1717200001000000000,
                "size": 310,
                "inode": 12345679,
            },
        }
        save_snapshot(str(tmp_path), wide_entries)

        loaded = load_snapshot(str(tmp_path))

        assert loaded == wide_entries, (
            f"load_snapshot must return the same entries written by save_snapshot. "
            f"Written: {wide_entries!r}\nLoaded: {loaded!r}"
        )


# ===========================================================================
# TEST 8: save_snapshot ATOMICITY
# ===========================================================================

class TestSaveSnapshotAtomicity:
    """save_snapshot must write atomically: old file survives a simulated failure.

    RED today: save_snapshot does not exist in tldr.patch.
    """

    def test_save_snapshot_atomic_failure_preserves_old_file(self, tmp_path: Path):
        """Monkeypatch os.replace to raise; assert old file_hashes.json unchanged.

        Simulates a crash between tmpfile write and os.replace. The original
        file_hashes.json must survive intact (false-DIRTY on next run, not false-clean).

        RED reason: save_snapshot does not exist in tldr.patch.
        """
        from tldr.patch import save_snapshot  # type: ignore[attr-defined]  # noqa

        cache_dir = tmp_path / ".tldr" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Write an existing file_hashes.json with known content
        original_entries = {
            "file_a.py": {
                "sha1": "original" + "0" * 32,
                "mtime_ns": 1000000000000000000,
                "size": 100,
                "inode": 99999,
            }
        }
        # Write the original as a wide JSON manually (save_snapshot not yet available
        # from caller's perspective; we write raw so the test is self-contained)
        original_raw = {"__schema_version__": 2, **original_entries}
        original_path = cache_dir / "file_hashes.json"
        original_path.write_text(json.dumps(original_raw))
        original_content = original_path.read_text()

        # New entries that would replace the original
        new_entries = {
            "file_a.py": {
                "sha1": "newvalue" + "0" * 32,
                "mtime_ns": 2000000000000000000,
                "size": 200,
                "inode": 88888,
            }
        }

        # Simulate crash: os.replace raises before committing
        original_os_replace = os.replace

        def failing_replace(src, dst):
            raise OSError("Simulated crash before os.replace")

        with patch("os.replace", side_effect=failing_replace):
            with pytest.raises(OSError):
                save_snapshot(str(tmp_path), new_entries)

        # Original file must be unchanged
        surviving_content = original_path.read_text()
        assert surviving_content == original_content, (
            f"save_snapshot crash before os.replace must leave the original "
            f"file_hashes.json intact. Original:\n{original_content}\n"
            f"Surviving:\n{surviving_content}"
        )


# ===========================================================================
# TEST 9: _enumerate_live_files PARITY with {u.file for u in units}
# ===========================================================================

class TestEnumerateLiveFilesParity:
    """_enumerate_live_files(scan_path) == {u.file for u in units} after full index.

    RED today: _enumerate_live_files does not exist in tldr.semantic.
    """

    def test_enumerate_live_files_matches_indexed_files_python_repo(self, tmp_path: Path):
        """Full index on a Python repo; assert _enumerate_live_files matches indexed files.

        _enumerate_live_files must use the SAME file-discovery as
        extract_units_from_project (code_extensions | NON_CODE_EXTENSIONS union).
        This test proves the live set cannot diverge from the parsed set.

        RED reason: _enumerate_live_files does not exist in tldr.semantic.
        """
        from tldr.semantic import build_semantic_index

        # Import the not-yet-existing function — must raise AttributeError → RED
        from tldr.semantic import _enumerate_live_files  # type: ignore[attr-defined]  # noqa

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta = _read_metadata(project)
        indexed_files = {u["file"] for u in meta["units"]}

        # scan_path for a standard index == project_root
        live_files = _enumerate_live_files(project, lang="python")

        assert isinstance(live_files, set), (
            f"_enumerate_live_files must return a set, got {type(live_files).__name__}"
        )
        assert live_files == indexed_files, (
            f"_enumerate_live_files must return exactly the set of files that were "
            f"indexed. Parity invariant violated.\n"
            f"Live: {sorted(live_files)}\n"
            f"Indexed: {sorted(indexed_files)}\n"
            f"Only in live: {sorted(live_files - indexed_files)}\n"
            f"Only in indexed: {sorted(indexed_files - live_files)}"
        )

    def test_enumerate_live_files_matches_indexed_files_lang_none_repo(self, tmp_path: Path):
        """Full index with lang=None (multi-language); assert _enumerate_live_files parity.

        With lang=None, _enumerate_live_files must use the union of code_extensions
        across ALL languages, matching what extract_units_from_project does.

        RED reason: _enumerate_live_files does not exist in tldr.semantic.
        """
        from tldr.semantic import build_semantic_index
        from tldr.semantic import _enumerate_live_files  # type: ignore[attr-defined]  # noqa

        # Create a project with a Python file and a non-code file (README.md)
        (tmp_path / ".git").mkdir()
        (tmp_path / "file_a.py").write_text(_PY_FILE_A)
        (tmp_path / "README.md").write_text("# Project\n\nThis is a project.\n")

        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(tmp_path), lang=None,
                show_progress=False, respect_ignore=False,
            )

        meta = _read_metadata(tmp_path)
        indexed_files = {u["file"] for u in meta["units"]}

        # scan_path == project_root, lang=None
        live_files = _enumerate_live_files(tmp_path, lang=None)

        assert isinstance(live_files, set), (
            f"_enumerate_live_files must return a set, got {type(live_files).__name__}"
        )
        # The key invariant: no indexed file is outside live, and no live file
        # is absent from indexed (modulo dedup from lang=None multi-emit).
        assert live_files == indexed_files, (
            "_enumerate_live_files(lang=None) must match {u.file for u in units} "
            "after a full lang=None index.\n"
            f"Live: {sorted(live_files)}\n"
            f"Indexed: {sorted(indexed_files)}\n"
            f"Only in live: {sorted(live_files - indexed_files)}\n"
            f"Only in indexed: {sorted(indexed_files - live_files)}"
        )


# ===========================================================================
# TEST B-12: DAEMON-HINT STALE-SHA1 GUARD — changed_set prevents stale sha1
#            reuse on the branch-(a) / daemon-hint path
# ===========================================================================

class TestDaemonHintStaleHashGuard:
    """Regression test for the changed_set guard in _compute_current_file_hashes.

    On the daemon-hint (branch-a) path, _derive_dirty_set returns sha1_map={}
    because no file content is hashed in that branch — the hint is trusted directly.
    The file IS present in old_snapshot with its STALE sha1 from the prior run.
    Without the changed_set guard, case-2 in _compute_current_file_hashes would
    reuse that stale sha1 and persist it. The NEXT reindex would then see a
    snapshot sha1 that matches the file's current content (by accident — the old
    sha1 would equal the new content's sha1 only by coincidence; usually it does
    NOT, so the deriver would correctly re-detect the file as changed, but the
    snapshot entry would be wrong). More critically, if a daemon-hint run persists
    a STALE sha1, the subsequent self-validating-floor run would re-hash the file
    and correctly find a mismatch — but this is one unnecessary extra re-parse.

    The guard (case-2 skipped when rel in changed_set) guarantees that any file in
    the changed set always gets a FRESH sha1 from case-3 (compute_file_hash) or
    from fresh_file_sha1s (the parse-time read-once sha1, which is merged into
    reuse_sha1s before _compute_current_file_hashes is called, making it case-1).

    This test (B-12 per review finding) verifies that guard is present and working.
    It SHOULD PASS on current code (NON-RED / GUARD-test mode).
    """

    def test_daemon_hint_changed_file_gets_fresh_sha1_not_stale_snapshot_sha1(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index; modify file_b; reindex via daemon-hint (dirty_files=[file_b]);
        assert persisted sha1 == sha1 of NEW content (not the stale cold sha1).

        The test then performs a no-op reindex to confirm the snapshot is consistent:
        if the sha1 were stale, the no-op reindex would detect a mismatch (stale
        snapshot sha1 != actual file sha1) and wrongly treat file_b as changed.

        Arrangement:
          - Cold index with _PY_FILE_B content → capture sha1_cold from file_hashes.json
          - Write _PY_FILE_B_MODIFIED (different content → different sha1)
          - Reindex with dirty_files=[abs_path_to_file_b] (branch-a / daemon-hint)
          - Read new sha1 from file_hashes.json → must equal sha1 of _PY_FILE_B_MODIFIED
          - Must NOT equal sha1_cold

        Additionally:
          - Run a second no-op reindex (no content change, no dirty_files hint).
          - The self-validating floor must see snapshot sha1 == actual sha1 (clean).
          - Spy confirms file_b is NOT re-parsed in the no-op run.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction
        from tldr.patch import compute_file_hash

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Step 1: Cold index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            total_cold = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert total_cold >= 2, f"Expected >= 2 units on cold index, got {total_cold}"

        # Capture the sha1 that was persisted for file_b after the cold run.
        hashes_after_cold = _read_file_hashes(project)
        assert "file_b.py" in hashes_after_cold, (
            "file_b.py must be in file_hashes.json after cold index"
        )
        entry_cold = hashes_after_cold["file_b.py"]
        assert isinstance(entry_cold, dict), (
            f"Expected wide SnapshotEntry dict for file_b.py, got {type(entry_cold).__name__}: {entry_cold!r}"
        )
        sha1_cold = entry_cold["sha1"]
        assert isinstance(sha1_cold, str) and len(sha1_cold) == 40, (
            f"sha1_cold must be a 40-char hex string, got {sha1_cold!r}"
        )

        # --- Step 2: Modify file_b's content ---
        file_b_path = project / "file_b.py"
        file_b_path.write_text(_PY_FILE_B_MODIFIED)

        # Compute what the FRESH sha1 of the new content SHOULD be.
        sha1_new_expected = compute_file_hash(str(file_b_path))
        assert sha1_new_expected != sha1_cold, (
            f"Test setup error: modified file_b has the same sha1 as the original "
            f"({sha1_cold!r}). The test requires different content so we can distinguish "
            f"fresh from stale sha1."
        )

        # --- Step 3: Incremental reindex via daemon-hint (branch-a) path ---
        # Passing dirty_files triggers the trust_hint=True branch in _derive_dirty_set
        # → sha1_map={} (no hash computed in branch-a). The changed_set guard in
        # _compute_current_file_hashes is what prevents reuse of old_snapshot sha1.
        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                dirty_files=[str(file_b_path)],
            )

        # --- Step 4: Assert persisted sha1 == sha1 of NEW content ---
        hashes_after_hint = _read_file_hashes(project)
        assert "file_b.py" in hashes_after_hint, (
            "file_b.py must be in file_hashes.json after daemon-hint reindex"
        )
        entry_after_hint = hashes_after_hint["file_b.py"]
        assert isinstance(entry_after_hint, dict), (
            f"Expected wide SnapshotEntry dict for file_b.py after hint reindex, "
            f"got {type(entry_after_hint).__name__}: {entry_after_hint!r}"
        )
        sha1_after_hint = entry_after_hint["sha1"]

        assert sha1_after_hint == sha1_new_expected, (
            f"GUARD FAILED: persisted sha1 for file_b.py after daemon-hint reindex "
            f"must equal the sha1 of the NEW content.\n"
            f"  Expected (sha1 of _PY_FILE_B_MODIFIED): {sha1_new_expected}\n"
            f"  Got (persisted):                         {sha1_after_hint}\n"
            f"  Old (sha1_cold):                         {sha1_cold}\n"
            f"If got == old, the changed_set guard is ABSENT: case-2 in "
            f"_compute_current_file_hashes reused the stale old_snapshot sha1 "
            f"for a file that is in changed_set (the daemon-hint path). "
            f"This would cause the NEXT self-validating-floor run to wrongly "
            f"re-detect file_b as changed (stale sha1 != actual sha1 → re-parse). "
            f"Fix: ensure 'rel not in changed_set' in case-2 of "
            f"_compute_current_file_hashes (B-12 guard)."
        )

        assert sha1_after_hint != sha1_cold, (
            f"Persisted sha1 after hint reindex must NOT equal the cold (stale) sha1. "
            f"sha1_cold={sha1_cold!r}, sha1_after_hint={sha1_after_hint!r}. "
            f"The changed_set guard appears absent."
        )

        # --- Step 5: No-op reindex to confirm snapshot consistency ---
        # If the persisted sha1 is fresh (correct), the floor will see snapshot sha1
        # == actual sha1 → no change detected → file_b NOT re-parsed.
        # If the persisted sha1 were stale, the floor would detect a mismatch
        # (stale != actual) and re-parse file_b unnecessarily.
        call_paths_noop: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths_noop.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model3 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model3):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                    # No dirty_files → self-validating hash floor path (branch-b).
                )

        # The floor must see a clean snapshot (sha1 matches) → zero files re-parsed.
        py_files_reparsed = [p for p in call_paths_noop if p.endswith(".py")]
        assert len(py_files_reparsed) == 0, (
            f"No-op reindex after daemon-hint run must not re-parse any files. "
            f"If file_b was re-parsed, it means the persisted snapshot sha1 was STALE "
            f"(old_snapshot sha1 != actual sha1 → floor re-detected it as changed). "
            f"This is the silent-index-corruption scenario the B-12 guard prevents. "
            f"Files re-parsed in no-op: {py_files_reparsed}"
        )

    def test_compute_current_file_hashes_changed_set_guard_forces_fresh_hash(
        self, tmp_path: Path
    ):
        """Direct unit test of the changed_set guard in _compute_current_file_hashes.

        Constructs a scenario where:
          - old_snapshot has a STALE sha1 for file_b.py
          - deriver_sha1_map={} (daemon-hint / branch-a: no deriver hash)
          - changed_set={'file_b.py'} (file is known-changed)

        Without the guard, case-2 would reuse old_snapshot['file_b.py']['sha1']
        (the stale value). With the guard ('rel not in changed_set'), case-2 is
        skipped and case-3 calls compute_file_hash → returns the ACTUAL sha1.

        Asserts: returned entry sha1 == actual file sha1 (NOT the stale value).
        """
        from tldr.semantic import _compute_current_file_hashes
        from tldr.patch import compute_file_hash
        from unittest.mock import MagicMock

        # Build a minimal file with known content
        file_b = tmp_path / "file_b.py"
        file_b.write_text(_PY_FILE_B_MODIFIED)

        actual_sha1 = compute_file_hash(str(file_b))
        stale_sha1 = "a" * 40  # clearly different from actual

        # Construct a fake EmbeddingUnit-like object that has .file attribute
        fake_unit = MagicMock()
        fake_unit.file = "file_b.py"

        # Seed old_snapshot with STALE sha1 for file_b
        old_snapshot = {
            "file_b.py": {
                "sha1": stale_sha1,
                "mtime_ns": 1000000000,
                "size": 42,
                "inode": 99999,
            }
        }

        # Call with changed_set={'file_b.py'} — the guard must skip case-2
        result = _compute_current_file_hashes(
            units=[fake_unit],
            scan_path=str(tmp_path),
            deriver_sha1_map={},          # branch-a: no deriver hash
            old_snapshot=old_snapshot,    # has stale sha1 for file_b
            changed_set={"file_b.py"},    # file is known-changed
        )

        assert "file_b.py" in result, (
            f"_compute_current_file_hashes must produce an entry for file_b.py. "
            f"Got: {result!r}"
        )
        persisted_sha1 = result["file_b.py"]["sha1"]

        assert persisted_sha1 == actual_sha1, (
            f"changed_set guard FAILED: _compute_current_file_hashes returned "
            f"sha1={persisted_sha1!r} for file_b.py but expected the ACTUAL sha1 "
            f"({actual_sha1!r}).\n"
            f"Stale sha1 in old_snapshot: {stale_sha1!r}\n"
            f"If persisted_sha1 == stale_sha1, the guard is ABSENT: case-2 "
            f"reused the old_snapshot sha1 even though 'file_b.py' is in "
            f"changed_set. Fix: add 'rel not in changed_set' to case-2 condition "
            f"in _compute_current_file_hashes (B-12 guard)."
        )
        assert persisted_sha1 != stale_sha1, (
            f"Persisted sha1 must NOT equal the stale snapshot sha1. "
            f"Guard is absent: case-2 reused the stale value."
        )
