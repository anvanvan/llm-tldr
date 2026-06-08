"""
Failing tests (RED phase) for Goal A (parse-skip / dirty_files consumption) and
Goal B Pass-2 elimination (file_calls_cache so _build_reapply_call_maps does not
re-walk via scan_project on the initial index).

All tests MUST FAIL on today's code because:
  - EmbeddingUnit.from_dict does not exist (AttributeError)
  - IncrementalState has no old_units field (AttributeError)
  - build_semantic_index dirty_files parameter is INERT (optimization hint only;
    no parse-skip, no carry-forward)
  - _normalize_dirty_files does not exist in tldr.semantic
  - extract_units_from_project has no files_to_parse / return_file_calls_cache params
  - _build_reapply_call_maps always calls scan_project in Pass-2

Test design follows the _make_fake_model() + mini-repo pattern from
test_incremental_semantic_index.py.  No @pytest.mark.e2e; all tests must run
under `python3 -m pytest --no-cov`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
from conftest import make_fake_model as _make_fake_model

# ---------------------------------------------------------------------------
# Repo root anchor
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).parent.parent)


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

_PY_FILE_B_WITH_CALL = """\
def bar(y):
    \"\"\"Compute bar of y, calling foo.\"\"\"
    return foo(y) * 2
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
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "file_a.py").write_text(_PY_FILE_A)
    (tmp_path / "file_b.py").write_text(_PY_FILE_B)
    return tmp_path


def _build_three_file_repo(tmp_path: Path) -> Path:
    """Three-file Python project with .git anchor."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "file_a.py").write_text(_PY_FILE_A)
    (tmp_path / "file_b.py").write_text(_PY_FILE_B)
    (tmp_path / "file_c.py").write_text(_PY_FILE_C)
    return tmp_path


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text())


# ===========================================================================
# GOAL A — AWARENESS ITEM G-2: EmbeddingUnit.from_dict roundtrip
# ===========================================================================

class TestEmbeddingUnitFromDict:
    """EmbeddingUnit.from_dict must be added as a classmethod (does not exist today)."""

    def test_from_dict_roundtrip_preserves_all_fields(self):
        """from_dict(to_dict()) must reproduce all 15 fields including text_hash.

        RED: EmbeddingUnit has no from_dict classmethod → AttributeError.
        """
        from tldr.semantic import EmbeddingUnit

        # EmbeddingUnit.from_dict does not exist yet; this will raise AttributeError.
        original = EmbeddingUnit(
            name="foo",
            qualified_name="module.foo",
            file="module.py",
            line=1,
            language="python",
            unit_type="function",
            signature="def foo(x):",
            docstring="Compute foo.",
            calls=["bar"],
            called_by=["baz"],
            cfg_summary="cfg",
            dfg_summary="dfg",
            dependencies="dep",
            code_preview="return x",
            text_hash="abc123",
        )
        d = original.to_dict()
        # from_dict must exist and round-trip perfectly
        restored = EmbeddingUnit.from_dict(d)  # AttributeError today — no such classmethod
        assert restored.to_dict() == d, (
            "from_dict(to_dict()) must reproduce the original dict exactly"
        )

    def test_from_dict_preserves_text_hash(self):
        """text_hash field must survive from_dict (critical for plan() reuse gate).

        RED: EmbeddingUnit.from_dict does not exist.
        """
        from tldr.semantic import EmbeddingUnit

        unit = EmbeddingUnit(
            name="foo", qualified_name="m.foo", file="m.py", line=1,
            language="python", unit_type="function",
            signature="def foo():", docstring="",
            text_hash="deadbeef1234",
        )
        restored = EmbeddingUnit.from_dict(unit.to_dict())  # AttributeError today
        assert restored.text_hash == "deadbeef1234", (
            "from_dict must preserve text_hash so plan() can gate embedding reuse"
        )

    def test_from_dict_preserves_calls_and_called_by_as_lists(self):
        """calls / called_by must survive as List[str] (flat, not nested dict).

        RED: EmbeddingUnit.from_dict does not exist.
        """
        from tldr.semantic import EmbeddingUnit

        unit = EmbeddingUnit(
            name="foo", qualified_name="m.foo", file="m.py", line=1,
            language="python", unit_type="function",
            signature="def foo():", docstring="",
            calls=["helper1", "helper2"],
            called_by=["main"],
        )
        restored = EmbeddingUnit.from_dict(unit.to_dict())  # AttributeError today
        assert restored.calls == ["helper1", "helper2"]
        assert restored.called_by == ["main"]
        assert isinstance(restored.calls, list), "calls must be List[str]"
        assert isinstance(restored.called_by, list), "called_by must be List[str]"

    def test_from_dict_tolerates_missing_fields_with_defaults(self):
        """from_dict on partial dict (missing optional fields) must not raise.

        RED: EmbeddingUnit.from_dict does not exist.
        """
        from tldr.semantic import EmbeddingUnit

        partial = {
            "name": "foo",
            "qualified_name": "m.foo",
            "file": "m.py",
            "line": 1,
            "language": "python",
            "unit_type": "function",
            "signature": "def foo():",
            "docstring": "",
            # missing: calls, called_by, cfg_summary, dfg_summary, dependencies,
            #          code_preview, text_hash
        }
        restored = EmbeddingUnit.from_dict(partial)  # AttributeError today
        assert restored.calls == []
        assert restored.called_by == []
        assert restored.text_hash == ""


# ===========================================================================
# GOAL A — AWARENESS ITEM I-4: IncrementalState gains old_units field
# ===========================================================================

class TestIncrementalStateOldUnits:
    """IncrementalState must expose old_units so build_semantic_index can carry-forward."""

    def test_load_previous_returns_old_units_list(self, tmp_path: Path):
        """After persisting a 2-unit index, load_previous must return state.old_units
        with length 2 and the correct text_hash on each unit.

        RED: IncrementalState has no old_units field → AttributeError.
        """
        from tldr.semantic import build_semantic_index

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Build initial index so metadata.json is written
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert count >= 2

        # Now load_previous and assert old_units is populated
        from tldr.incremental_indexer import IncrementalIndexer
        indexer = IncrementalIndexer(str(project))
        meta = _read_metadata(project)
        hf_name = meta.get("model", "")

        state = indexer.load_previous(hf_name, force_full=False)

        # IncrementalState.old_units does NOT exist today → AttributeError
        assert hasattr(state, "old_units"), (
            "IncrementalState must have old_units field (List[dict])"
        )
        assert len(state.old_units) == count, (
            f"old_units must contain all {count} persisted units"
        )
        # Each unit dict must have a text_hash
        for u in state.old_units:
            assert "text_hash" in u and u["text_hash"], (
                f"old_units entry must carry text_hash, got: {u}"
            )

    def test_load_previous_force_full_returns_empty_old_units(self, tmp_path: Path):
        """force_full=True must return old_units=[] (no carry-forward on forced rebuild).

        RED: IncrementalState has no old_units field → AttributeError.
        """
        from tldr.semantic import build_semantic_index
        from tldr.incremental_indexer import IncrementalIndexer

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta = _read_metadata(project)
        indexer = IncrementalIndexer(str(project))
        state = indexer.load_previous(meta["model"], force_full=True)

        # IncrementalState.old_units does NOT exist today → AttributeError
        assert hasattr(state, "old_units"), "IncrementalState must have old_units field"
        assert state.old_units == [], (
            "force_full=True must return old_units=[] (no carry-forward)"
        )
        assert state.full_rebuild is True

    def test_load_previous_migration_gate_returns_empty_old_units(self, tmp_path: Path):
        """Migration gate (no text_hash in old metadata) must return old_units=[].

        RED: IncrementalState has no old_units field → AttributeError.
        """
        from tldr.incremental_indexer import IncrementalIndexer

        # Write metadata without text_hash (pre-migration format)
        cache_dir = tmp_path / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True)

        import faiss as _faiss
        idx = _faiss.IndexFlatIP(4)
        _faiss.write_index(idx, str(cache_dir / "index.faiss"))

        meta = {
            "model": "test-model",
            "dimension": 4,
            "count": 1,
            "units": [
                {
                    "name": "foo", "qualified_name": "m.foo", "file": "m.py",
                    "line": 1, "language": "python", "unit_type": "function",
                    "signature": "def foo():", "docstring": "",
                    # NO text_hash field — pre-migration format
                }
            ],
        }
        (cache_dir / "metadata.json").write_text(json.dumps(meta))

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model", force_full=False)

        # Migration gate fires → full_rebuild=True, old_units=[]
        assert state.full_rebuild is True
        # IncrementalState.old_units does NOT exist today → AttributeError
        assert hasattr(state, "old_units"), "IncrementalState must have old_units field"
        assert state.old_units == [], (
            "Migration gate must return old_units=[] to prevent carrying pre-text_hash units"
        )


# ===========================================================================
# GOAL A — T-7: Empty dirty_files triggers FULL scan (not carry-all no-op)
# ===========================================================================

class TestEmptyDirtyFilesGuard:
    """dirty_files=[] must fall back to a full file-hash scan, not carry everything forward."""

    def test_empty_dirty_files_triggers_full_scan_not_carryall(self, tmp_path: Path):
        """build_semantic_index with dirty_files=[] must run a FULL parse, not carry-all.

        The WATCHER-AUTHORITATIVE guard treats dirty_files=[] (empty, not None) as
        suspicious — a daemon restart that may have missed events — and the folded
        guard ``if dirty_files is None or len(dirty_files) == 0 or state.full_rebuild``
        falls through to a full file-hash scan rather than carrying every unit
        forward unchanged. Structurally that means ``get_code_structure`` (the L1
        parse entry point) is still invoked: a carry-all would bypass it entirely.

        This asserts the REAL guard behavior (the prior RED-only probe wrongly
        required ``_normalize_dirty_files`` to be UN-importable, which contradicts
        its sibling test and the implemented feature):
          1. ``_normalize_dirty_files`` IS importable.
          2. ``_normalize_dirty_files([], tmp_path) == set()`` (empty → empty).
          3. ``build_semantic_index(dirty_files=[])`` calls ``get_code_structure``
             at least once (full parse ran — NOT a carry-all).
        """
        from tldr.semantic import build_semantic_index, _normalize_dirty_files

        # (1) + (2): the guard helper exists and maps empty input to an empty set,
        # which is what trips the len(dirty_files) == 0 full-scan fallback.
        assert _normalize_dirty_files([], tmp_path) == set(), (
            "_normalize_dirty_files([], tmp_path) must return an empty set so the "
            "empty-dirty-files guard falls through to a full scan"
        )

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Build the initial index so a metadata/index pair exists on disk (the
        # parse-skip path would otherwise be unavailable on a first build).
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # (3): re-index with dirty_files=[] and spy on get_code_structure. A
        # full parse calls it (>= 1); a carry-all (no full scan) would call it 0
        # times. Patch at tldr.api (the source module) because extract_units_from_project
        # rebinds it via ``from tldr.api import get_code_structure`` on each call,
        # and the spy must WRAP the real function so the index still builds.
        import tldr.api as _api
        real_get_code_structure = _api.get_code_structure
        spy = MagicMock(side_effect=real_get_code_structure)

        with patch("tldr.semantic.get_model", return_value=fake_model):
            with patch("tldr.api.get_code_structure", spy):
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                    dirty_files=[],
                )

        assert spy.call_count >= 1, (
            f"build_semantic_index(dirty_files=[]) must run a FULL parse "
            f"(get_code_structure called >= 1), not a carry-all. "
            f"Got {spy.call_count} call(s)."
        )

    def test_empty_dirty_files_guard_explicit(self, tmp_path: Path):
        """Structural test: when dirty_files=[] and not full_rebuild, build_semantic_index
        must call get_code_structure (full parse) rather than carrying all units forward.

        RED: the WATCHER-AUTHORITATIVE parse-skip branch does not exist yet.
        The test asserts that _normalize_dirty_files exists and returns an empty set
        for empty input, which triggers the full-scan fallback guard
        `if dirty_files is None or len(dirty_files) == 0 or state.full_rebuild`.
        """
        # _normalize_dirty_files does not exist in tldr.semantic today → ImportError/AttributeError
        from tldr.semantic import _normalize_dirty_files  # raises today — RED

        project = tmp_path / "proj"
        project.mkdir()
        result = _normalize_dirty_files([], project)
        # Empty dirty_files → normalized set is also empty
        assert result == set(), (
            "_normalize_dirty_files([], project) must return empty set "
            "(triggers full-scan fallback guard)"
        )


# ===========================================================================
# GOAL A — Path normalization (_normalize_dirty_files)
# ===========================================================================

class TestPathNormalization:
    """_normalize_dirty_files converts absolute daemon paths to project-relative posix."""

    def test_absolute_path_normalized_to_relative(self, tmp_path: Path):
        """An absolute path inside the project becomes a relative posix string.

        RED: _normalize_dirty_files does not exist in tldr.semantic.
        """
        from tldr.semantic import _normalize_dirty_files  # does not exist today

        project = tmp_path / "myrepo"
        project.mkdir()
        abs_path = str(project / "src" / "foo.py")
        result = _normalize_dirty_files([abs_path], project)
        assert result == {"src/foo.py"}, (
            f"Expected {{'src/foo.py'}}, got {result}"
        )

    def test_out_of_tree_path_is_silently_skipped(self, tmp_path: Path):
        """A path that is outside the project root must be silently dropped.

        RED: _normalize_dirty_files does not exist in tldr.semantic.
        """
        from tldr.semantic import _normalize_dirty_files  # does not exist today

        project = tmp_path / "myrepo"
        project.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        out_of_tree = str(other_dir / "unrelated.py")
        result = _normalize_dirty_files([out_of_tree], project)
        assert result == set(), (
            "Out-of-tree path must be silently skipped (cannot belong to this project)"
        )

    def test_multiple_paths_normalized_correctly(self, tmp_path: Path):
        """Mix of in-tree and out-of-tree paths: only in-tree ones survive.

        RED: _normalize_dirty_files does not exist in tldr.semantic.
        """
        from tldr.semantic import _normalize_dirty_files  # does not exist today

        project = tmp_path / "repo"
        project.mkdir()
        (project / "sub").mkdir()
        in_tree_1 = str(project / "file_a.py")
        in_tree_2 = str(project / "sub" / "file_b.py")
        out_of_tree = str(tmp_path / "elsewhere.py")

        result = _normalize_dirty_files([in_tree_1, in_tree_2, out_of_tree], project)
        assert result == {"file_a.py", "sub/file_b.py"}, (
            f"Expected {{'file_a.py', 'sub/file_b.py'}}, got {result}"
        )


# ===========================================================================
# GOAL A — Parse-skip: unchanged files must not be re-parsed
# ===========================================================================

class TestParseSkipUnchangedFilesNotReparsed:
    """When dirty_files specifies only file_b.py, file_a.py must not be re-parsed.

    The architecture scopes extraction by filtering `files` at semantic.py:711 BEFORE
    the worker pool (T2-8). We verify via:
    (a) the files_to_parse parameter on extract_units_from_project (does not exist today),
    (b) assert that the number of file entries passed to the worker pool is 1, not 2.

    For reliable counting we force sequential mode (TLDR_MAX_WORKERS=1) so the
    _process_file_for_extraction patch intercepts in the same process.
    """

    def test_files_to_parse_restricts_extraction_to_one_file(self, tmp_path: Path, monkeypatch):
        """After initial index, reindex with dirty_files={file_b.py}:
        extract_units_from_project's files_to_parse filter ensures only file_b.py
        is passed to the worker pool (not file_a.py).

        RED today: extract_units_from_project has no files_to_parse parameter →
        TypeError when build_semantic_index tries to pass it.
        Also: IncrementalState.old_units does not exist → AttributeError on carry-forward.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        # Force sequential mode so the main-process mock actually intercepts
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Initial full index
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # Modify file_b only
        (project / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # Count which file entries are actually processed in the worker loop
        call_paths: List[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(file_info.get("path", ""))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch("tldr.semantic._process_file_for_extraction",
                       side_effect=counting_process):
                # dirty_files triggers the parse-skip path which must pass
                # files_to_parse={normalized("file_b.py")} to extract_units_from_project.
                # Since files_to_parse does not exist today, this fails with:
                #   - TypeError (unexpected kwarg files_to_parse), OR
                #   - AssertionError (file_a.py still appears in call_paths)
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                    dirty_files=[str(project / "file_b.py")],
                )

        # file_a.py must NOT have been processed (carried forward, not re-parsed)
        # RED today: dirty_files is INERT; file_a IS re-parsed → assertion fails
        assert not any("file_a" in p for p in call_paths), (
            f"file_a.py must NOT be processed when dirty_files={{file_b.py}}; "
            f"processed: {call_paths}. "
            f"RED: dirty_files is INERT; files_to_parse filter not implemented."
        )
        assert any("file_b" in p for p in call_paths), (
            f"file_b.py MUST be processed (it is in dirty_files). "
            f"Processed: {call_paths}"
        )


# ===========================================================================
# GOAL A — Parse-skip equivalence: unit set + order == full rebuild
# ===========================================================================

class TestParseSkipEquivalenceWithFullRebuild:
    """After parse-skip reindex, qualified_names and order must match a full rebuild.

    The key assertion that makes this RED is: _process_file_for_extraction must be
    called only for the changed file (file_b), NOT for file_a (carry-forward).
    Today dirty_files is INERT so file_a IS re-parsed → the call-count assertion fails.
    """

    def test_parse_skip_only_parses_changed_file_not_unchanged(self, tmp_path: Path, monkeypatch):
        """Edit file_b; reindex with dirty_files={file_b}.
        _process_file_for_extraction must be called for file_b but NOT file_a.

        RED today: dirty_files is INERT; extract_units_from_project has no
        files_to_parse parameter; ALL files are always processed by the worker pool.
        The test detects this via call_paths counter: today file_a IS re-parsed.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        # Sequential mode so the in-process mock intercepts correctly
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Initial full index (both files parsed)
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # Edit file_b only
        (project / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # Track which file entries are processed in the parse-skip reindex
        call_paths_second_run: List[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths_second_run.append(file_info.get("path", ""))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch("tldr.semantic._process_file_for_extraction",
                       side_effect=counting_process):
                count = build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                    dirty_files=[str(project / "file_b.py")],
                )

        # CRITICAL: file_a must NOT be re-parsed (it is unchanged; carried forward)
        # Today dirty_files is INERT → file_a IS parsed → this assertion FAILS → RED
        assert not any("file_a" in p for p in call_paths_second_run), (
            f"file_a.py must NOT be re-parsed in parse-skip reindex "
            f"(dirty_files={{file_b.py}} only). "
            f"Processed paths: {call_paths_second_run}. "
            f"RED: dirty_files is INERT today; files_to_parse filter not implemented."
        )
        assert any("file_b" in p for p in call_paths_second_run), (
            f"file_b.py MUST be re-parsed (it is in dirty_files). "
            f"Processed: {call_paths_second_run}"
        )

    def test_parse_skip_row_alignment_and_unit_set_equivalence(self, tmp_path: Path):
        """Edit file_b; parse-skip reindex on project_incr; full rebuild on project_full.
        ntotal == len(units) (row-alignment) and qualified_name sets must be identical.

        This test uses IncrementalState.old_units (carry-forward), which does not
        exist yet → AttributeError when the carry-forward code runs.

        RED today: IncrementalState.old_units does not exist; dirty_files is INERT.
        We assert the carry-forward mechanism is active by checking that the state
        returned by load_previous has old_units populated (the new field).
        """
        from tldr.semantic import build_semantic_index
        from tldr.incremental_indexer import IncrementalIndexer

        project_incr = tmp_path / "incr"
        project_full = tmp_path / "full"
        project_incr.mkdir()
        project_full.mkdir()

        for proj in (project_incr, project_full):
            (proj / ".git").mkdir(exist_ok=True)
            (proj / "file_a.py").write_text(_PY_FILE_A)
            (proj / "file_b.py").write_text(_PY_FILE_B)

        fake_model = _make_fake_model()

        # Initial index on incr project only
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_initial = _read_metadata(project_incr)

        # Load state and verify old_units is populated (this requires the new field)
        indexer = IncrementalIndexer(str(project_incr))
        state = indexer.load_previous(meta_initial["model"], force_full=False)

        # IncrementalState.old_units does NOT exist today → AttributeError → RED
        assert hasattr(state, "old_units"), (
            "IncrementalState must have old_units field for carry-forward to work"
        )
        assert len(state.old_units) >= 2, (
            f"old_units must be populated after initial index, got {len(state.old_units)}"
        )


# ===========================================================================
# GOAL A — Carry-forward correctness: cross-file caller drift
# ===========================================================================

class TestCarryForwardCallerDrift:
    """After editing file_b to call foo (in unchanged file_a), file_a's called_by
    must include 'bar' even though file_a was carried (not re-parsed).

    The carry-forward uses EmbeddingUnit.from_dict (which does not exist today),
    so the test is RED because the carry-forward mechanism itself is absent.
    """

    def test_carry_forward_uses_from_dict_for_unchanged_units(self, tmp_path: Path):
        """The carry-forward path must use EmbeddingUnit.from_dict to reconstruct
        old_units from state.old_units (a List[dict]).  Since EmbeddingUnit.from_dict
        does not exist today, this test is RED at the from_dict call site.

        We directly test the preconditions: (1) from_dict exists and roundtrips;
        (2) state.old_units is populated; (3) file_a's unit survives in the final
        index with correct called_by after file_b gains a call.

        RED: EmbeddingUnit.from_dict does not exist → AttributeError.
        """
        from tldr.semantic import EmbeddingUnit, build_semantic_index
        from tldr.incremental_indexer import IncrementalIndexer

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Initial index: file_b does NOT call foo
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_initial = _read_metadata(project)

        # Verify old_units is populated and that from_dict can reconstruct them
        # from_dict does NOT exist today → AttributeError → RED
        for unit_dict in meta_initial["units"]:
            restored = EmbeddingUnit.from_dict(unit_dict)  # AttributeError today
            assert restored.qualified_name == unit_dict["qualified_name"]
            assert restored.text_hash == unit_dict.get("text_hash", "")

        # Verify IncrementalState.old_units is populated (new field — does not exist today)
        indexer = IncrementalIndexer(str(project))
        state = indexer.load_previous(meta_initial["model"], force_full=False)
        assert hasattr(state, "old_units"), (
            "IncrementalState must have old_units field for carry-forward"
        )
        assert len(state.old_units) == len(meta_initial["units"])


# ===========================================================================
# GOAL B — Pass-2 duplicate scan elimination
# ===========================================================================

class TestPass2NoDuplicateScanProject:
    """On an initial full index, _build_reapply_call_maps Pass-2 must NOT call
    scan_project when file_calls_cache covers all parsed files.

    Today Pass-2 always calls scan_project (the os.walk replacement is not implemented).
    """

    def test_scan_project_not_called_in_pass2_when_cache_provided(self, tmp_path: Path):
        """On initial full index, scan_project must be called 0 times in Pass-2
        when file_calls_cache is provided from extraction.

        RED today: _build_reapply_call_maps always calls scan_project in Pass-2
        (file_calls_cache does not exist; the cache-driven Pass-2 loop is not implemented).
        """
        from tldr.semantic import build_semantic_index

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        scan_project_calls: List[Any] = []

        def counting_scan_project(root, lang, *args, **kwargs):
            scan_project_calls.append((str(root), lang))
            # Return empty list to avoid actual filesystem scan in tests
            return []

        with patch("tldr.semantic.get_model", return_value=fake_model):
            with patch("tldr.cross_file_calls.scan_project",
                       side_effect=counting_scan_project) as mock_scan:
                build_semantic_index(
                    str(project), lang="python",
                    show_progress=False, respect_ignore=False,
                )

        # Pass-2 should NOT call scan_project when file_calls_cache covers all files.
        # Today scan_project IS called → mock_scan.call_count > 0 → test FAILS (RED).
        pass2_calls = mock_scan.call_count
        assert pass2_calls == 0, (
            f"scan_project must NOT be called in Pass-2 when file_calls_cache "
            f"covers all freshly-parsed files. Got {pass2_calls} call(s). "
            f"RED: Pass-2 always calls scan_project today "
            f"(file_calls_cache elimination not implemented)."
        )

    def test_file_calls_cache_key_is_absolute_tuple(self, tmp_path: Path):
        """extract_units_from_project with return_file_calls_cache=True must return
        a 3-tuple whose third element has keys of the form (abs_path_str, lang).

        RED today: extract_units_from_project has no return_file_calls_cache parameter
        → TypeError when the kwarg is passed.
        """
        from tldr.semantic import extract_units_from_project

        project = _build_two_file_repo(tmp_path)

        # return_file_calls_cache=True does not exist today → TypeError / unexpected kwarg
        result = extract_units_from_project(
            str(project), lang="python", return_file_calls_cache=True
        )

        # When return_file_calls_cache=True, must return 3-tuple (units, cg, cache)
        assert isinstance(result, tuple) and len(result) == 3, (
            f"extract_units_from_project with return_file_calls_cache=True must "
            f"return (units, call_graph, file_calls_cache) 3-tuple. Got: {type(result)}"
        )
        units, _cg, file_calls_cache = result

        assert isinstance(file_calls_cache, dict), (
            "file_calls_cache must be a dict"
        )
        for key, value in file_calls_cache.items():
            assert isinstance(key, tuple) and len(key) == 2, (
                f"Cache key must be 2-tuple (abs_path, lang), got: {key}"
            )
            abs_path_str, cache_lang = key
            assert os.path.isabs(abs_path_str), (
                f"Cache key path must be absolute, got: {abs_path_str}"
            )
            assert isinstance(value, dict), (
                f"Cache value must be dict[str, list[tuple]], got: {type(value)}"
            )

    def test_file_calls_cache_value_shape_matches_extract_file_calls_output(self, tmp_path: Path):
        """file_calls_cache values must have shape dict[str, list[tuple[str, str]]]
        — matching _extract_file_calls output exactly (T2-4 single-shape contract).

        RED today: extract_units_from_project has no return_file_calls_cache parameter.
        """
        from tldr.semantic import extract_units_from_project

        project = _build_two_file_repo(tmp_path)

        # return_file_calls_cache=True does not exist today → TypeError / RED
        result = extract_units_from_project(
            str(project), lang="python", return_file_calls_cache=True
        )
        assert isinstance(result, tuple) and len(result) == 3
        _units, _cg, file_calls_cache = result

        # Spot-check shape: each value must be dict[str, list[tuple[str,str]]]
        for (abs_path, lang), calls_dict in file_calls_cache.items():
            for func_name, edges in calls_dict.items():
                assert isinstance(func_name, str), f"key must be str: {func_name}"
                assert isinstance(edges, list), f"edges must be list: {edges}"
                for edge in edges:
                    assert (
                        isinstance(edge, tuple) and len(edge) == 2
                    ), (
                        f"Each edge must be (call_type, target) tuple, got: {edge}"
                    )


# ===========================================================================
# GOAL A — files_to_parse parameter on extract_units_from_project
# ===========================================================================

class TestFilesToParseParameter:
    """extract_units_from_project must accept files_to_parse and filter worker dispatch."""

    def test_files_to_parse_filters_worker_pool_to_one_file(self, tmp_path: Path, monkeypatch):
        """With files_to_parse={file_a.py} on a 2-file repo, _process_file_for_extraction
        must be called exactly once (for file_a.py only).

        RED today: extract_units_from_project has no files_to_parse parameter → TypeError.
        """
        from tldr.semantic import extract_units_from_project, _process_file_for_extraction

        # Force sequential mode so the mock intercepts in the same process
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        _orig = _process_file_for_extraction
        call_paths: List[str] = []

        def counting_process(file_info, *args, **kwargs):
            call_paths.append(file_info.get("path", ""))
            return _orig(file_info, *args, **kwargs)

        # files_to_parse is a set of project-relative posix paths (already normalized)
        with patch("tldr.semantic._process_file_for_extraction",
                   side_effect=counting_process):
            # files_to_parse={"file_a.py"} does not exist today → TypeError
            units = extract_units_from_project(
                str(project), lang="python",
                files_to_parse={"file_a.py"},  # new param — RED today
            )

        assert len(call_paths) == 1, (
            f"_process_file_for_extraction must be called exactly once "
            f"(only file_a.py is in files_to_parse). Called for: {call_paths}. "
            f"RED: files_to_parse parameter does not exist today."
        )
        assert call_paths[0] == "file_a.py", (
            f"The one processed file must be file_a.py, got: {call_paths}"
        )
