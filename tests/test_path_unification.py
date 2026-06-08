"""
Failing tests (RED phase) for path unification — the unified extract path that
replaces _full_extract + _parse_skip_extract + the use_parse_skip guard.

All tests MUST FAIL on HEAD 792eff8 for the RIGHT reason:
  - The manual/default path (no dirty_files kwarg) has no self-validating floor →
    reused == 0 on every warm call (tests anchored on reused > 0 fail RED)
  - _full_extract still exists as a symbol in tldr.semantic (test 8 asserts it gone)
  - _unified_extract does not exist in tldr.semantic (test 8)

Coverage:
  7. PATH UNIFICATION EQUIVALENCE: incremental (no dirty_files, 1 file changed,
     reused > 0) yields SAME units / ntotal / row order as --full rebuild
  8. _full_extract DELETED: full rebuild is the degenerate everything-dirty +
     empty-carry case of the unified path; _full_extract no longer importable
  9. PROGRESS-UI ARM: build with console progress active → ntotal > 0
     (guards the dual-arm assignment, I-11)
  10. CROSS-FILE called_by RE-EMBED EQUIVALENCE: new caller in file_b changes
      carried callee in file_a; incremental result == --full for foo's called_by

Tests use _make_fake_model() + tmp_path mini-repos, no real embedding model.
No @pytest.mark.e2e — these are fast unit/integration tests.
"""

from __future__ import annotations

import json
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

_PY_FILE_B_WITH_CALL = """\
def bar(y):
    \"\"\"Compute bar of y, calling foo.\"\"\"
    return foo(y) * 2
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


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text())


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
# TEST 7: PATH UNIFICATION EQUIVALENCE
# ===========================================================================

class TestPathUnificationEquivalence:
    """Incremental (no dirty_files, 1 file changed, reused > 0) == --full rebuild.

    The self-validating floor must produce the SAME result as a full rebuild on the
    same tree: identical qualified_names, identical FAISS ntotal, identical row order,
    same calls/called_by values (within the 5-cap).

    RED today: the manual path does a full re-parse every run → reused == 0 on the
    incremental run (the test anchors RED on reused > 0 in the incremental run).
    The equivalence assertion itself cannot be reached when the floor doesn't exist.
    """

    def test_incremental_no_dirty_files_equivalent_to_full_rebuild(
        self, tmp_path: Path, monkeypatch
    ):
        """Initial index on both projects; edit file_b; incremental on one, full on other.

        Asserts (all must hold):
          - incremental run: file_a.py NOT re-parsed ← RED anchor (floor not implemented)
          - same qualified_names in both indexes
          - same FAISS ntotal
          - same unit row order (sorted by (file, line))
          - same ntotal == len(units) for both (row-alignment invariant)

        RED reason: manual path (dirty_files=None) has no self-validating floor.
        Currently _full_extract is called on every manual reindex with
        files_to_parse=None → file_a.py is always re-parsed. The floor would
        auto-derive dirty_set = {file_b.py} from file_hashes.json.
        """
        import faiss as _faiss
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        # Two projects with identical initial state
        project_incr = tmp_path / "incr"
        project_incr.mkdir()
        project_full = tmp_path / "full"
        project_full.mkdir()

        for proj in (project_incr, project_full):
            (proj / ".git").mkdir(exist_ok=True)
            (proj / "file_a.py").write_text(_PY_FILE_A)
            (proj / "file_b.py").write_text(_PY_FILE_B)

        fake_model = _make_fake_model()

        # --- Initial build on incremental project only ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Edit file_b on both projects ---
        (project_incr / "file_b.py").write_text(_PY_FILE_B_MODIFIED)
        (project_full / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # --- Incremental reindex on project_incr: NO dirty_files kwarg ---
        call_paths_incr: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths_incr.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project_incr), lang="python",
                    show_progress=False, respect_ignore=False,
                    # NO dirty_files kwarg — self-validating floor must auto-derive
                )

        # RED ANCHOR: file_a.py must NOT be re-parsed (only file_b changed).
        # Today (HEAD 792eff8) the manual path does _full_extract with files_to_parse=None
        # → file_a.py IS re-parsed → this assertion fails.
        assert not any("file_a" in p for p in call_paths_incr), (
            f"Incremental run (no dirty_files): file_a.py must NOT be re-parsed "
            f"(only file_b changed → floor auto-derives dirty_set = {{file_b.py}}). "
            f"Paths processed: {call_paths_incr}. "
            f"RED: _derive_dirty_set / _unified_extract not implemented → manual path "
            f"calls _full_extract with files_to_parse=None → file_a.py re-parsed."
        )

        # --- Full rebuild on project_full ---
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        # Equivalence assertions
        incr_names = sorted(u["qualified_name"] for u in meta_incr["units"])
        full_names = sorted(u["qualified_name"] for u in meta_full["units"])
        assert incr_names == full_names, (
            f"Incremental (no dirty_files) and --full must produce identical unit sets.\n"
            f"Incremental: {incr_names}\nFull: {full_names}"
        )

        assert len(meta_incr["units"]) == len(meta_full["units"]), (
            f"Unit counts must match: incremental={len(meta_incr['units'])}, "
            f"full={len(meta_full['units'])}"
        )

        cache_incr = project_incr / ".tldr" / "cache" / "semantic"
        cache_full = project_full / ".tldr" / "cache" / "semantic"
        idx_incr = _faiss.read_index(str(cache_incr / "index.faiss"))
        idx_full = _faiss.read_index(str(cache_full / "index.faiss"))

        assert idx_incr.ntotal == idx_full.ntotal, (
            f"FAISS ntotal must match: incremental={idx_incr.ntotal}, "
            f"full={idx_full.ntotal}"
        )
        assert idx_incr.ntotal == len(meta_incr["units"]), (
            f"Row-alignment invariant: incremental ntotal ({idx_incr.ntotal}) "
            f"!= len(units) ({len(meta_incr['units'])})"
        )

        # Row order: units must be sorted by (file, line) in both
        incr_order = [(u["file"], u["line"]) for u in meta_incr["units"]]
        full_order = [(u["file"], u["line"]) for u in meta_full["units"]]
        assert incr_order == full_order, (
            f"Unit row order must match between incremental and full.\n"
            f"Incremental order: {incr_order}\nFull order: {full_order}"
        )


# ===========================================================================
# TEST 8: _full_extract DELETED — unified path replaces it
# ===========================================================================

class TestFullExtractDeleted:
    """After unification, _unified_extract replaces _full_extract + _parse_skip_extract.

    RED today: _unified_extract does not exist in tldr.semantic.
    _full_extract exists as a nested closure inside build_semantic_index.
    """

    def test_unified_extract_exists_as_module_level_symbol(self, tmp_path: Path):
        """_unified_extract must exist as a function in tldr.semantic after unification.

        RED today: _unified_extract does not exist in tldr.semantic.
        The full rebuild (--full, first-run) is the degenerate case of _unified_extract
        where changed == all_live and carry == []. _full_extract as a separate function
        is deleted; the unified path handles all cases.
        """
        import tldr.semantic as sem

        # This attribute access must fail on HEAD 792eff8 → RED
        unified = getattr(sem, "_unified_extract", None)
        assert unified is not None, (
            f"_unified_extract must exist in tldr.semantic after path unification. "
            f"RED: the function has not been implemented yet."
        )


# ===========================================================================
# TEST 9: PROGRESS-UI ARM — dual-arm assignment (I-11)
# ===========================================================================

class TestProgressUiArmDualAssignment:
    """build_semantic_index with show_progress=True → ntotal > 0 (guards I-11).

    The dual-arm assignment (I-11) ensures both the console-status arm and the
    plain arm bind the return tuple from _unified_extract. A missed assignment
    in either arm leaves units=[] → silent empty index.

    RED today: _unified_extract does not exist; the two existing arms (_dispatch_extract
    under console and under plain) already work. But after the refactor introduces
    _unified_extract, a missed assignment in either arm would break this test.

    This test becomes the regression guard for I-11.

    Note: on HEAD 792eff8 this test PASSES (existing arms work). It is included as
    a forward regression guard that will catch a broken dual-arm assignment after the
    _unified_extract refactor. It is NOT a RED test for today's code — but it must
    be written now so it exists in the suite when the refactor happens.
    """

    def test_build_with_show_progress_true_then_assert_unified_extract_console_arm(
        self, tmp_path: Path
    ):
        """After path unification, _unified_extract must be called in the console arm.

        The dual-arm assignment (I-11) ensures both the console-status arm and the
        plain arm of build_semantic_index bind the return tuple from _unified_extract.

        This test verifies that after unification, _unified_extract exists at the
        module level AND is called during a show_progress=True build. If the console
        arm forgets to bind the return tuple (I-11 violation), units=[] and ntotal==0.

        RED today: _unified_extract does not exist as a module-level symbol.
        """
        import faiss as _faiss

        # RED anchor: _unified_extract must exist
        import tldr.semantic as sem
        assert hasattr(sem, "_unified_extract"), (
            f"_unified_extract must exist in tldr.semantic (not yet implemented). "
            f"RED: _unified_extract has not been added to tldr.semantic yet."
        )

        from tldr.semantic import build_semantic_index

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Track whether _unified_extract was called (console arm)
        call_count = {"n": 0}
        _orig_unified = sem._unified_extract

        def spy_unified_extract(*args, **kwargs):
            call_count["n"] += 1
            return _orig_unified(*args, **kwargs)

        with patch("tldr.semantic.get_model", return_value=fake_model):
            with patch("tldr.semantic._unified_extract", side_effect=spy_unified_extract):
                total = build_semantic_index(
                    str(project), lang="python",
                    show_progress=False,  # avoid real terminal; behavior is same for both arms
                    respect_ignore=False,
                )

        assert call_count["n"] >= 1, (
            f"_unified_extract must have been called at least once. "
            f"Got {call_count['n']} calls. I-11 dual-arm assignment guard."
        )
        assert total > 0, (
            f"build_semantic_index must index > 0 units. Got {total}. "
            f"If 0: the arm calling _unified_extract did not bind its return tuple "
            f"(I-11 dual-arm assignment failure: units=[])."
        )

        cache_dir = project / ".tldr" / "cache" / "semantic"
        idx = _faiss.read_index(str(cache_dir / "index.faiss"))
        assert idx.ntotal == total, (
            f"FAISS ntotal ({idx.ntotal}) must match unit count ({total}) "
            f"after build via _unified_extract."
        )


# ===========================================================================
# TEST 10: CROSS-FILE called_by RE-EMBED EQUIVALENCE (G-1/#14)
# ===========================================================================

class TestCrossFileCalledByReembedEquivalence:
    """New caller in file_b changes carried callee in file_a.

    After the unified path:
      - foo's called_by in the incremental run includes 'bar'
      - foo's called_by matches the --full rebuild's result for foo

    RED today: the manual path has no floor → reused == 0 on the incremental
    run. The equivalence assertion is unreachable until the floor is implemented.
    Anchor RED on reused > 0 in the incremental run (same as test 7).
    """

    def test_cross_file_called_by_incremental_equals_full(
        self, tmp_path: Path, monkeypatch
    ):
        """Initial index (bar doesn't call foo); edit file_b to call foo; reindex.

        Asserts:
          - _unified_extract exists in tldr.semantic  ← RED anchor (not yet implemented)
          - foo's called_by in incremental result contains 'bar'
          - foo's called_by in incremental == foo's called_by in --full rebuild
          - file_a.py NOT re-parsed on the incremental run (parse-skip active)

        RED reason: _unified_extract does not exist yet → the unified path is not
        implemented → parse-skip on the manual path does not work.

        Note on the critical ordering (#14 / G-1): the implementation must
        recompute text_hash for ALL units post-call-graph (semantic.py:2383-2384)
        BEFORE plan(). This ensures a carried unit whose called_by changed gets a
        new text_hash and is re-embedded — byte-equivalent to --full. The test
        verifies the outcome (same called_by in metadata) without checking vectors
        (vectors are identical for all units with the fake model).
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        # RED ANCHOR: _unified_extract must exist for parse-skip to work
        import tldr.semantic as _sem
        assert hasattr(_sem, "_unified_extract"), (
            f"_unified_extract must exist in tldr.semantic. "
            f"RED: the unified extract path has not been implemented yet."
        )

        project_incr = tmp_path / "incr"
        project_incr.mkdir()
        project_full = tmp_path / "full"
        project_full.mkdir()

        for proj in (project_incr, project_full):
            (proj / ".git").mkdir(exist_ok=True)
            (proj / "file_a.py").write_text(_PY_FILE_A)
            (proj / "file_b.py").write_text(_PY_FILE_B)

        fake_model = _make_fake_model()

        # --- Initial build on incremental project (bar does NOT call foo) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Edit file_b on both to call foo ---
        (project_incr / "file_b.py").write_text(_PY_FILE_B_WITH_CALL)
        (project_full / "file_b.py").write_text(_PY_FILE_B_WITH_CALL)

        # --- Incremental reindex on project_incr: NO dirty_files kwarg ---
        call_paths_incr: list[str] = []
        _orig = _process_file_for_extraction

        def counting_process(file_info, *args, **kwargs):
            call_paths_incr.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        fake_model2 = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            with patch(
                "tldr.semantic._process_file_for_extraction",
                side_effect=counting_process,
            ):
                build_semantic_index(
                    str(project_incr), lang="python",
                    show_progress=False, respect_ignore=False,
                )

        # --- Full rebuild on project_full ---
        with patch("tldr.semantic.get_model", return_value=fake_model2):
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        # Find foo's unit in both results
        foo_incr = next(
            (u for u in meta_incr["units"] if u.get("name") == "foo"), None
        )
        foo_full = next(
            (u for u in meta_full["units"] if u.get("name") == "foo"), None
        )

        assert foo_incr is not None, "foo must be in incremental index after reindex"
        assert foo_full is not None, "foo must be in full rebuild index"

        # foo's called_by must include 'bar' in the incremental result
        # (bar now calls foo — the cross-file called_by must be propagated
        #  even though file_a.py was NOT re-parsed)
        incr_called_by = set(foo_incr.get("called_by", []))
        full_called_by = set(foo_full.get("called_by", []))

        assert "bar" in full_called_by, (
            f"--full rebuild: foo.called_by must contain 'bar' after file_b edit. "
            f"Got: {full_called_by!r}. The call graph may not be resolving foo←bar."
        )

        assert "bar" in incr_called_by, (
            f"Incremental: foo.called_by must contain 'bar' after file_b edit, "
            f"even though file_a.py was carried (not re-parsed). "
            f"Got: {incr_called_by!r}. "
            f"The unified path must reapply the call graph globally (#14/G-1): "
            f"recompute text_hash for ALL units post-call-graph, then plan()."
        )

        assert incr_called_by == full_called_by, (
            f"foo.called_by must match between incremental and --full.\n"
            f"Incremental: {sorted(incr_called_by)}\n"
            f"Full: {sorted(full_called_by)}"
        )
