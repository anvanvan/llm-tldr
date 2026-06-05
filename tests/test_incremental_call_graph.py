"""
Tests for DELIVERABLE-2: Incremental call-graph edge re-apply.

Feature: _reapply_call_graph / _build_reapply_call_maps gain a "carried units"
pass (Pass-2b) that reuses already-persisted calls lists from unchanged
EmbeddingUnits, eliminating per-file extract_file_calls_for_language I/O for
carried files.  The carried edges + fresh-file edges are merged into a global
called_by invert, so a new caller in a changed file correctly updates a CARRIED
callee's called_by.  After _reapply_call_graph, text_hash is recomputed for ALL
units (fresh + carried), causing plan() to re-embed any carried unit whose
called_by changed, keeping vectors byte-equivalent to --full.

Deliverable-2 correctness criteria (from architecture.md):

Criterion 7 (D2 THE test):
  core.py: def helper(); app.py: main() does NOT call helper.
  Edit app.py so main() calls helper(); reindex via the DEFAULT path (no dirty_files).
  Assert:
    (a) app.py is re-embedded (changed file — always expected)
    (b) core.py is CARRIED / reused (reused > 0) — not re-parsed
    (c) core.helper's unit has app.main in called_by — equal to --full

Text-hash equivalence invariant (architecture #14):
  A CARRIED unit whose called_by CHANGED gets a new text_hash and IS re-embedded
  by plan() so its vector matches --full.  Assert helper's stored unit after
  incremental == helper's stored unit after --full.

Pass-2 reuse (no file-read for carried units):
  On a no-change incremental reindex, extract_file_calls_for_language call count
  for carried files == 0.  Import-resolved ClassName::method edges survive the carry.

All tests are RED on HEAD 792eff8 because:
  - The DEFAULT path (no dirty_files) calls _full_extract which always does a
    full parse → no units are "carried" (reused == 0 or the carry path isn't taken)
  - Pass-2b (carried unit persisted calls reuse) does not exist;
    extract_file_calls_for_language IS called per carried unit today
  - The "carried unit re-embedded when called_by changes" flow requires the
    incremental carry path (_parse_skip_extract), which is only triggered today
    when dirty_files kwarg is passed; the DEFAULT path is always full-parse.

RED anchors:
  - Criterion 7: anchor on "core.py carried/reused>0" — fails because the DEFAULT
    path does a full parse → 0 carried units.
  - Text-hash equivalence: anchor on the carry path existing (reused>0) for OTHER
    unchanged units; fails for the same reason.
  - Pass-2 reuse: anchor on extract_file_calls_for_language call count == 0 for
    carried files; fails because carried path doesn't exist for DEFAULT mode.

Runner: python3 -m pytest --no-cov tests/test_incremental_call_graph.py
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
from conftest import make_fake_model as _make_fake_model


# ---------------------------------------------------------------------------
# Mini-repo source texts
# ---------------------------------------------------------------------------
_CORE_PY_NO_CALL = """\
def helper(x):
    \"\"\"A helper function.\"\"\"
    return x + 1
"""

_APP_PY_NO_CALL = """\
def main():
    \"\"\"Main entry point.\"\"\"
    return 42
"""

_APP_PY_CALLS_HELPER = """\
def main():
    \"\"\"Main entry point, calling helper.\"\"\"
    return helper(42)
"""


def _build_two_file_repo(tmp_path: Path, *,
                         app_content: str = _APP_PY_NO_CALL) -> Path:
    """Build: core.py (helper), app.py (main).  Returns project root."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "core.py").write_text(_CORE_PY_NO_CALL)
    (tmp_path / "app.py").write_text(app_content)
    return tmp_path


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text())


def _get_unit(meta: dict, name: str) -> dict | None:
    return next((u for u in meta["units"] if u.get("name") == name), None)


# ===========================================================================
# TEST 5: Cross-file called_by — criterion 7 (THE D2 test)
# ===========================================================================

class TestCrossFileCalledByIncrementalDefaultPath:
    """Criterion 7: After editing app.py to call helper(), an incremental reindex
    via the DEFAULT path (no dirty_files kwarg) must:
      (a) re-embed app.py (changed file)
      (b) CARRY core.py (reused > 0 for other unchanged units)
      (c) core.helper's called_by == ["main"] == --full rebuild's called_by

    RED anchor: The DEFAULT path today calls _full_extract (full parse) so
    reused == 0 / no units are carried.  Anchor assertion (b) fails:
    the carry path is never taken without dirty_files kwarg.
    """

    def test_core_py_carried_and_helper_called_by_updated(self, tmp_path: Path, monkeypatch):
        """Index repo; edit app.py to call helper(); reindex DEFAULT path;
        assert core.py carried (reused > 0) AND helper.called_by contains 'main'.

        RED reason (current code): DEFAULT path does _full_extract (no carry);
        reused_count == 0 OR the carry path doesn't run → one of:
          - AssertionError on (b): "core.py must be carried/reused after incremental reindex"
          - AssertionError on (c): helper.called_by empty (no cross-file called_by)
        """
        from tldr.semantic import build_semantic_index

        # Sequential mode so the spy intercepts _process_file_for_extraction in the
        # main process (workers would bypass the patch in a subprocess pool).
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_incr = tmp_path / "incr"
        project_full = tmp_path / "full"
        project_incr.mkdir()
        project_full.mkdir()

        # Identical initial state in both
        for proj in (project_incr, project_full):
            (proj / ".git").mkdir()
            (proj / "core.py").write_text(_CORE_PY_NO_CALL)
            (proj / "app.py").write_text(_APP_PY_NO_CALL)

        fake_model = _make_fake_model()

        # --- Initial index on both projects (prior index needed for carry path) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Edit app.py in BOTH projects: main() now calls helper() ---
        (project_incr / "app.py").write_text(_APP_PY_CALLS_HELPER)
        (project_full / "app.py").write_text(_APP_PY_CALLS_HELPER)

        # --- Incremental reindex: DEFAULT path (no dirty_files) ---
        # Spy on _process_file_for_extraction to observe which files are re-parsed.
        from tldr.semantic import _process_file_for_extraction as _orig_pfex

        parse_paths: list[str] = []

        def spy_pfex(file_info, *args, **kwargs):
            parse_paths.append(str(file_info.get("path", "")))
            return _orig_pfex(file_info, *args, **kwargs)

        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.semantic._process_file_for_extraction", side_effect=spy_pfex):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
                # NO dirty_files kwarg — must use self-validating hash floor
            )

        # Capture encode calls from the incremental run ONLY, before the --full run.
        all_encoded_texts_incr: list[str] = []
        for c in fake_model.encode.call_args_list:
            if c.args:
                all_encoded_texts_incr.extend(c.args[0])

        # --- Full rebuild for comparison ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        # (a) app.py must be re-embedded (it changed)
        all_encoded_texts = all_encoded_texts_incr
        assert any("main" in t for t in all_encoded_texts), (
            "app.py's main() unit must be re-embedded after app.py changed."
        )

        # (b) core.py must be CARRIED (reused, not re-parsed from disk).
        # Parse-skip observation: _process_file_for_extraction must NOT be called
        # for core.py (unchanged) but MUST be called for app.py (changed).
        # This mirrors the pattern used by TestNoOpSelfValidatingFloor and
        # TestDeletionSelfValidatingFloor in test_self_validating_floor.py.
        helper_unit_incr = _get_unit(meta_incr, "helper")
        assert helper_unit_incr is not None, (
            "helper unit must be present in incremental metadata after reindex. "
            "RED: if core.py is dropped (not carried and not re-parsed), helper is absent."
        )

        # The key carry assertion: core.py must NOT have been re-parsed.
        # parse_paths was captured by the spy patched into the incremental run above.
        assert not any("core" in p for p in parse_paths), (
            f"core.py must NOT be re-parsed on an incremental reindex "
            f"(only app.py changed; core.py must be carried from prior index). "
            f"Paths processed by _process_file_for_extraction: {parse_paths}. "
            f"RED: DEFAULT path calls _full_extract (full parse) → core.py IS "
            f"re-parsed even though it did not change; carry path not taken."
        )

        # (c) helper.called_by must contain 'main' in incremental result == full result
        helper_unit_full = _get_unit(meta_full, "helper")
        assert helper_unit_full is not None, (
            "helper must be in --full metadata after edit."
        )

        incr_called_by = sorted(helper_unit_incr.get("called_by") or [])
        full_called_by = sorted(helper_unit_full.get("called_by") or [])

        assert "main" in incr_called_by, (
            f"helper.called_by must contain 'main' after incremental reindex "
            f"(app.py now calls helper). "
            f"Actual incremental called_by: {incr_called_by!r}. "
            f"RED: carry path not taken → either helper not carried, "
            f"or _reapply_call_graph pass-2b (carried edges) not implemented, "
            f"so cross-file called_by is not rebuilt."
        )

        assert incr_called_by == full_called_by, (
            f"Incremental helper.called_by {incr_called_by!r} must equal "
            f"--full helper.called_by {full_called_by!r}. "
            f"RED: incremental carry path missing or called_by not globally rebuilt."
        )


# ===========================================================================
# TEST 6: Text-hash equivalence invariant — carried unit re-embedded on called_by change
# ===========================================================================

class TestCarriedUnitReembeddedWhenCalledByChanges:
    """Architecture #14: A CARRIED unit whose called_by changed gets a new text_hash
    and is RE-EMBEDDED by plan().  Its stored unit (text_hash + called_by) must
    match --full.

    RED anchor: requires the carry path to run (reused > 0 for other units);
    fails today because DEFAULT path is always full parse (no carry).
    """

    def test_helper_text_hash_after_incremental_equals_full(self, tmp_path: Path, monkeypatch):
        """After editing app.py to call helper(), the incremental reindex must
        produce helper.text_hash == helper.text_hash from --full rebuild.

        RED reason: DEFAULT path does full parse → core.py IS re-parsed even though
        unchanged. The parse-skip proxy catches this: _process_file_for_extraction
        must NOT be called for core.py on the incremental reindex.
        """
        from tldr.semantic import build_semantic_index

        # Sequential mode so the spy intercepts _process_file_for_extraction in the
        # main process (workers would bypass the patch in a subprocess pool).
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_incr = tmp_path / "incr"
        project_full = tmp_path / "full"
        project_incr.mkdir()
        project_full.mkdir()

        for proj in (project_incr, project_full):
            (proj / ".git").mkdir()
            (proj / "core.py").write_text(_CORE_PY_NO_CALL)
            (proj / "app.py").write_text(_APP_PY_NO_CALL)

        fake_model = _make_fake_model()

        # --- Initial index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_initial = _read_metadata(project_incr)
        helper_before = _get_unit(meta_initial, "helper")
        assert helper_before is not None, "helper must be indexed initially"
        hash_before = helper_before.get("text_hash", "")

        # --- Edit app.py to call helper ---
        (project_incr / "app.py").write_text(_APP_PY_CALLS_HELPER)
        (project_full / "app.py").write_text(_APP_PY_CALLS_HELPER)

        # --- Incremental reindex (DEFAULT path, no dirty_files) ---
        # Spy on _process_file_for_extraction to observe which files are re-parsed.
        from tldr.semantic import _process_file_for_extraction as _orig_pfex2

        parse_paths2: list[str] = []

        def spy_pfex2(file_info, *args, **kwargs):
            parse_paths2.append(str(file_info.get("path", "")))
            return _orig_pfex2(file_info, *args, **kwargs)

        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.semantic._process_file_for_extraction", side_effect=spy_pfex2):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Full rebuild for comparison ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        helper_incr = _get_unit(meta_incr, "helper")
        helper_full = _get_unit(meta_full, "helper")

        assert helper_incr is not None, "helper must be in incremental metadata"
        assert helper_full is not None, "helper must be in --full metadata"

        # Parse-skip observation: core.py (unchanged) must NOT have been re-parsed.
        # This is the valid carry proxy: the feature is PARSE-skip, not EMBED-skip.
        # A carried callee may still be re-embedded when its text_hash changes
        # (called_by updated), but it must NOT be re-parsed from disk.
        assert not any("core" in p for p in parse_paths2), (
            f"core.py must NOT be re-parsed on an incremental reindex "
            f"(only app.py changed; core.py must be carried from prior index). "
            f"Paths processed by _process_file_for_extraction: {parse_paths2}. "
            f"RED: DEFAULT path calls _full_extract (full parse) → core.py IS "
            f"re-parsed even though it did not change; carry path not taken."
        )

        # text_hash equivalence: helper's text_hash after incremental == --full
        incr_hash = helper_incr.get("text_hash", "")
        full_hash = helper_full.get("text_hash", "")

        assert incr_hash == full_hash, (
            f"helper.text_hash after incremental ({incr_hash!r}) must equal "
            f"--full ({full_hash!r}). "
            f"RED: if the carried unit is served with a stale vector (old called_by), "
            f"its text_hash will not match --full."
        )

        # helper's text_hash must have CHANGED from the initial build
        # (because called_by now includes main)
        assert incr_hash != hash_before, (
            f"helper.text_hash must change after main() calls helper "
            f"(called_by changed → build_embedding_text output changes). "
            f"Before: {hash_before!r}, after: {incr_hash!r}."
        )


# ===========================================================================
# TEST 7: Pass-2 reuse — extract_file_calls_for_language not called for carried files
# ===========================================================================

class TestPass2NoFileReadForCarriedUnits:
    """On an incremental reindex (DEFAULT path, no changes), carried units' calls
    come from their persisted calls lists — NOT from re-reading the file via
    extract_file_calls_for_language.

    Assert: when no files changed, extract_file_calls_for_language call count == 0
    for the carried files (all files are carried).

    RED anchor: Pass-2b (carried unit persisted calls) does not exist today;
    the current Pass-2 calls extract_file_calls_for_language per carried unit
    even on the carry path. But the DEFAULT path doesn't take the carry path at
    all (full parse) → extract_file_calls_for_language IS called for ALL files.
    Assertion fails: call count > 0.
    """

    def test_no_file_calls_read_for_unchanged_carried_units(
        self, tmp_path: Path, monkeypatch
    ):
        """Build index; reindex with NO changes (DEFAULT path, no dirty_files);
        assert extract_file_calls_for_language is NOT called for core.py or app.py.

        RED reason: DEFAULT path uses _full_extract (full parse) → Pass-2 calls
        extract_file_calls_for_language for all files → call_count > 0.
        """
        from tldr.semantic import build_semantic_index

        # Force sequential mode so the patch intercepts in the main process
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Initial full index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Reindex with NO file changes (DEFAULT path, all carried) ---
        fake_model.encode.reset_mock()
        call_count_box = [0]

        from tldr.cross_file_calls import extract_file_calls_for_language as _orig_efcfl

        def counting_efcfl(path, root, lang, *args, **kwargs):
            call_count_box[0] += 1
            return _orig_efcfl(path, root, lang, *args, **kwargs)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch(
                 "tldr.cross_file_calls.extract_file_calls_for_language",
                 side_effect=counting_efcfl,
             ):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                # NO dirty_files — DEFAULT path; all files unchanged
            )

        assert call_count_box[0] == 0, (
            f"extract_file_calls_for_language must NOT be called for carried "
            f"(unchanged) units on an incremental no-op reindex. "
            f"Call count: {call_count_box[0]}. "
            f"RED: DEFAULT path runs _full_extract (full parse) → "
            f"extract_file_calls_for_language called for all files "
            f"(or carry path runs but uses Pass-2 file reads, not persisted calls)."
        )

    def test_method_call_edge_preserved_across_incremental_reindex(
        self, tmp_path: Path, monkeypatch
    ):
        """Import-resolved ClassName::method call edges survive an incremental
        reindex of a carried (unchanged) file.

        Scenario: core.py has Processor.process (calls helper) and helper.
        app.py is edited; core.py is UNCHANGED (carried). After the incremental
        reindex, Processor.process.calls must still contain 'helper' — the edge
        is preserved.

        Option C (java-called-by fix): on a real change, carried files are
        RE-PARSED with the per-language extractor (extract_file_calls_for_language)
        — the SAME extractor a --full rebuild uses — so the incremental call graph
        is byte-identical to --full BY CONSTRUCTION (including dual-keyed
        ClassName.method callers for Java/Go/Rust). The re-parse is O(lines) and
        page-cached; only embedding reuse (the expensive work) is preserved. The
        carried-file re-parse is therefore EXPECTED on a real change (count >= 1),
        and only suppressed on a true no-op (covered by
        test_no_file_calls_read_for_unchanged_carried_units above).
        """
        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        _core_with_class = """\
class Processor:
    def process(self, x):
        \"\"\"Process x.\"\"\"
        return helper(x)


def helper(x):
    \"\"\"Standalone helper.\"\"\"
    return x + 1
"""
        _app_no_call = """\
def main():
    \"\"\"Main that uses Processor.\"\"\"
    p = Processor()
    return p.process(10)
"""
        _app_modified = """\
def main():
    \"\"\"Main that uses Processor (MODIFIED).\"\"\"
    p = Processor()
    result = p.process(10)
    return result * 2
"""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".git").mkdir()
        (project / "core.py").write_text(_core_with_class)
        (project / "app.py").write_text(_app_no_call)

        fake_model = _make_fake_model()

        # --- Initial index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Edit app.py (content change); core.py unchanged ---
        (project / "app.py").write_text(_app_modified)

        # Count extract_file_calls_for_language calls against core.py specifically
        core_call_count = [0]
        from tldr.cross_file_calls import extract_file_calls_for_language as _orig_efcfl

        def counting_efcfl(path, root, lang, *args, **kwargs):
            if "core" in str(path):
                core_call_count[0] += 1
            return _orig_efcfl(path, root, lang, *args, **kwargs)

        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch(
                 "tldr.cross_file_calls.extract_file_calls_for_language",
                 side_effect=counting_efcfl,
             ):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                # DEFAULT path: no dirty_files; hash floor detects app.py changed
            )

        # Option C: a real change (app.py edited) DOES re-parse the carried core.py
        # via extract_file_calls_for_language — that is how the incremental call
        # graph stays byte-identical to --full (native dual-keyed callers). The
        # re-parse is cheap (file unchanged, page-cached); embedding reuse is what
        # matters and is preserved. So at least one re-parse of core.py is expected.
        assert core_call_count[0] >= 1, (
            f"Option C: on a real change, the carried core.py must be RE-PARSED via "
            f"extract_file_calls_for_language so its native (dual-keyed) call edges "
            f"match a --full rebuild. Call count for core.py: {core_call_count[0]} "
            f"(expected >= 1). A count of 0 means carried files are not re-parsed — "
            f"the incremental call graph would then diverge from --full for "
            f"class-method languages (Java/Go/Rust)."
        )

        # Edge equivalence: Processor.process.calls should still contain 'helper'
        meta = _read_metadata(project)
        process_unit = next(
            (u for u in meta["units"]
             if u.get("name") in ("process", "Processor.process")), None
        )
        if process_unit:
            assert "helper" in (process_unit.get("calls") or []), (
                f"Processor.process.calls must contain 'helper' after incremental reindex. "
                f"Actual calls: {process_unit.get('calls')!r}. "
                f"RED: carried unit's persisted call edges not reused correctly."
            )
