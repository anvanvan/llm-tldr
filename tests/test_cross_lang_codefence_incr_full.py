"""
Regression test: incremental index must preserve qualified callers harvested
from cross-language code fences (e.g. Rust code inside a ``.md`` file).

ROOT CAUSE (tldr/semantic.py ``_augment_cache_with_carried`` ~line 1991-2002):
  On the incremental path, ``_augment_cache_with_carried`` keys each carried
  (unchanged) file under ``unit.language`` — e.g. ``'markdown'`` for a
  ``spec.md`` file.  But ``extract_file_calls_for_language(file, 'markdown')``
  returns ``{}`` (markdown is not an import-index language).

  On the ``--full`` path, ``_extract_one_language`` keys every file under
  ``structure_lang`` (the DISPATCH language — ``'rust'`` in a ``--lang rust``
  run), so ``_extract_rust_file_calls(spec.md, ...)`` parses the file's bytes
  with tree-sitter-rust, finds the Rust code fence, and harvests qualified
  caller keys like ``SvnClient.parse_unified_diff``.

  RESULT: qualified callers from code fences in non-source files (e.g. ``.md``)
  are present in the ``--full`` call graph but absent after an incremental
  reindex — the per-unit ``called_by`` / ``calls`` diverge.

FIXTURE:
  A synthetic Rust repo with:
  (a) ``src/callee.rs`` — defines ``target_fn()`` (the CALLEE).
  (b) ``src/caller.rs`` — defines ``struct Caller`` with ``call_target(&self)``
      that calls ``target_fn()``.  Produces qualified caller key
      ``Caller.call_target`` in ``target_fn.called_by``.
  (c) ``spec.md`` — contains a Rust code fence whose fenced code defines
      ``struct SvnClient`` with ``parse_unified_diff(&self)`` calling
      ``target_fn()``.  The Rust extractor harvests qualified caller key
      ``SvnClient.parse_unified_diff`` from this fence.

  After cold-indexing with ``--lang rust``, adding ``src/probe.rs`` (trivial
  new file) forces the incremental path where ``spec.md`` is CARRIED.

ASSERTION THAT FAILS ON HEAD (c3d91c5):
  ``target_fn.called_by`` after incremental reindex does NOT contain
  ``'SvnClient.parse_unified_diff'`` — the qualified caller from the code fence.

  The same assertion PASSES after a ``--full`` rebuild of the same disk state.

  After the fix (keying carried files under the dispatch language, not
  ``unit.language``), the incremental result equals the ``--full`` result.

Runner: python3 -m pytest --no-cov tests/test_cross_lang_codefence_incr_full.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import (
    find_units_by_name,
    make_fake_model,
    read_semantic_metadata,
    validate_incr_full_equivalence,
)


# ---------------------------------------------------------------------------
# Rust source fixtures — minimal but real enough for tree-sitter-rust to parse
# ---------------------------------------------------------------------------

# callee.rs: defines target_fn() — the CALLEE that will accumulate callers
# both from a real .rs file (Caller.call_target) and from a .md code fence
# (SvnClient.parse_unified_diff).
_CALLEE_RS = """\
pub fn target_fn() {
    // The callee: expects Caller.call_target AND SvnClient.parse_unified_diff
    // as callers in both incremental and --full call graphs.
}
"""

# caller.rs: Caller struct with call_target() — a genuine cross-file caller.
# After a --full rebuild this produces the qualified key "Caller.call_target"
# AND the bare key "call_target" in target_fn.called_by.
_CALLER_RS = """\
struct Caller;

impl Caller {
    fn call_target(&self) {
        target_fn();
    }
}
"""

# spec.md: a non-source file containing a Rust CODE FENCE whose fenced code
# has SvnClient::parse_unified_diff() calling target_fn().
#
# When _extract_one_language dispatches with structure_lang='rust', it calls
# _extract_rust_file_calls on spec.md (which passes raw bytes to tree-sitter-
# rust).  The parser DOES find the struct/impl and harvests two caller keys:
#   - "SvnClient.parse_unified_diff"  (qualified — the REGRESSION assertion)
#   - "parse_unified_diff"            (bare alias)
#
# Both keys land in the file_calls_cache under (abs_path_to_spec_md, 'rust').
# On the incremental path, spec.md's EmbeddingUnit has unit.language='markdown'
# so _augment_cache_with_carried calls extract_file_calls_for_language(md, 'markdown')
# which returns {} — the fence callers are lost.
_SPEC_MD = """\
# Spec

Here is example usage:

```rust
struct SvnClient;

impl SvnClient {
    fn parse_unified_diff(&self) {
        target_fn();
    }
}
```
"""

# probe.rs: trivial new file added AFTER the cold index to force an incremental
# reindex.  It calls target_fn() to ensure target_fn's called_by changes
# (confirming the incremental path actually ran), but the key test is that
# spec.md's fence callers are ALSO present in the incremental result.
_PROBE_RS = """\
pub fn probe_fn() {
    target_fn();
}
"""


def _build_rust_repo(root: Path) -> None:
    """Write the initial 3-file Rust repo (callee, caller, spec.md) inside root."""
    (root / ".git").mkdir(exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "callee.rs").write_text(_CALLEE_RS)
    (root / "src" / "caller.rs").write_text(_CALLER_RS)
    (root / "spec.md").write_text(_SPEC_MD)


# ---------------------------------------------------------------------------
# TypeScript source fixtures — a cross-file tsMain -> tsHelper call.  These TS
# files are CARRIED (unchanged) on the incremental run that only adds a Rust
# file, so their call edge tsHelper.called_by == ['tsMain'] is precisely the
# edge the pre-C-1 code DROPS (carried TS files never re-extracted under
# 'typescript' when only Rust changed).
# ---------------------------------------------------------------------------

# ts_helper.ts: defines tsHelper() — the CALLEE that must keep 'tsMain' in its
# called_by after an incremental reindex that changed ONLY a Rust file.
_TS_HELPER = """\
export function tsHelper(): number {
    return 42;
}
"""

# ts_main.ts: tsMain() calls tsHelper() across files — the carried TS edge.
_TS_MAIN = """\
import { tsHelper } from "./ts_helper";

export function tsMain(): number {
    return tsHelper();
}
"""


def _build_dual_lang_repo(root: Path) -> None:
    """Write a dual-language repo: the Rust files PLUS two TypeScript files.

    The TS files carry a cross-file ``tsMain -> tsHelper`` call.  On an
    incremental run that changes only Rust, the TS files are carried and their
    edge must survive (the C-1 regression dropped it).
    """
    _build_rust_repo(root)
    (root / "src" / "ts_helper.ts").write_text(_TS_HELPER)
    (root / "src" / "ts_main.ts").write_text(_TS_MAIN)




# ===========================================================================
# The regression test
# ===========================================================================

class TestCrossLangCodeFenceIncrementalEqualsFullCallGraph:
    """Incremental Rust reindex must preserve qualified callers from .md code fences.

    Bug: _augment_cache_with_carried keys spec.md under unit.language='markdown'
    and calls extract_file_calls_for_language(spec.md, 'markdown') == {}.
    The --full path keys spec.md under structure_lang='rust' and harvests
    'SvnClient.parse_unified_diff' from the code fence.

    RED on HEAD (c3d91c5): incremental target_fn.called_by is missing
    'SvnClient.parse_unified_diff'; --full target_fn.called_by contains it.
    The assertion below FAILS on the unfixed tree.

    PASS after fix: _augment_cache_with_carried keys carried files under
    the dispatch language(s) in the fresh cache, not unit.language, so the
    fence callers are re-extracted and included in the augmented cache.
    """

    def test_codefence_qualified_caller_present_in_incremental_called_by(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index 3-file Rust repo; add probe.rs; run incremental; run --full;
        assert target_fn.called_by contains 'SvnClient.parse_unified_diff' in BOTH.

        RED reason (current tree):
          _augment_cache_with_carried (semantic.py ~line 1991-2002) keys each
          carried file under unit.language.  spec.md's EmbeddingUnit has
          unit.language='markdown', so the augmentation calls
              extract_file_calls_for_language(spec.md, 'markdown') == {}
          and the Rust code fence callers ('SvnClient.parse_unified_diff',
          'parse_unified_diff') never enter the augmented cache.

          Pass-1 (build_project_call_graph with prebuilt_file_calls) and
          Pass-2a (_link_file_calls) both filter to lang='rust' and skip the
          missing (spec.md, 'markdown') entry — the qualified fence callers
          vanish from incremental called_by.

          The --full path keys spec.md under structure_lang='rust' via
          _extract_one_language._accumulate (semantic.py ~line 938), so
          _extract_rust_file_calls(spec.md) finds 'SvnClient.parse_unified_diff'
          and includes it in target_fn.called_by.

          The assertion that 'SvnClient.parse_unified_diff' is present in the
          incremental called_by FAILS with AssertionError on HEAD.
        """
        pytest.importorskip("tree_sitter_rust")

        from tldr.semantic import build_semantic_index

        # Remove stray /private/tmp/.tldr that would hijack project-root detection.
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        # Sequential workers so monkeypatching of model is reliable.
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_incr = tmp_path / "rust_incr"
        project_full = tmp_path / "rust_full"
        project_incr.mkdir()
        project_full.mkdir()

        # Identical initial state: callee.rs + caller.rs + spec.md
        for proj in (project_incr, project_full):
            _build_rust_repo(proj)

        fake_model = make_fake_model()

        # ------------------------------------------------------------------
        # Step 1: Cold index on project_incr (full parse — no prior state).
        #   spec.md is included because get_code_structure(language='rust')
        #   unions NON_CODE_EXTENSIONS, so spec.md is in the file list and
        #   _extract_rust_file_calls parses its Rust code fence.
        # ------------------------------------------------------------------
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="rust",
                show_progress=False, respect_ignore=False,
            )

        # SANITY: cold index (which is a full parse) must already produce
        # 'SvnClient.parse_unified_diff' in target_fn.called_by.
        # If this fails, tree-sitter-rust is unavailable or spec.md is not
        # parsed — the regression cannot be demonstrated.
        meta_cold = read_semantic_metadata(project_incr)
        cold_target_units = find_units_by_name(meta_cold, "target_fn")
        assert cold_target_units, (
            "target_fn not found in cold index — tree-sitter-rust may be "
            "unavailable or the Rust fixture source is not parseable."
        )
        cold_cb = sorted(cold_target_units[0].get("called_by") or [])
        assert "SvnClient.parse_unified_diff" in cold_cb, (
            f"FIXTURE SANITY: 'SvnClient.parse_unified_diff' must appear in "
            f"target_fn.called_by after the cold (full-parse) index.  "
            f"Got: {cold_cb!r}.  "
            f"If this fails, tree-sitter-rust may be unavailable or "
            f"get_code_structure(language='rust') is not including spec.md — "
            f"the regression cannot be demonstrated without this fixture edge."
        )

        # ------------------------------------------------------------------
        # Step 2: Add probe.rs to BOTH repos (same final disk state).
        #   probe.rs calls target_fn(), adding 'probe_fn' to target_fn.called_by.
        #   This proves the incremental path actually ran (target_fn's called_by
        #   must include 'probe_fn' in the incremental result).
        # ------------------------------------------------------------------
        (project_incr / "src" / "probe.rs").write_text(_PROBE_RS)
        (project_full / "src" / "probe.rs").write_text(_PROBE_RS)

        # ------------------------------------------------------------------
        # Step 3: Incremental reindex on project_incr (DEFAULT path, no --full).
        #   callee.rs, caller.rs, spec.md are UNCHANGED → carried from prior
        #   snapshot.  probe.rs is new → freshly parsed.
        #
        #   BUG: _augment_cache_with_carried keys spec.md under 'markdown'
        #   (unit.language) not 'rust' (structure_lang).  So the augmented
        #   cache lacks (spec.md, 'rust') and the Rust fence callers vanish.
        # ------------------------------------------------------------------
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="rust",
                show_progress=False, respect_ignore=False,
                # No full=True — this is the incremental path.
            )

        # ------------------------------------------------------------------
        # Step 4: --full rebuild on project_full (same final tree).
        #   Full path keys spec.md under structure_lang='rust' → harvests
        #   'SvnClient.parse_unified_diff' from the code fence.
        # ------------------------------------------------------------------
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="rust",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = read_semantic_metadata(project_incr)
        meta_full = read_semantic_metadata(project_full)

        # ------------------------------------------------------------------
        # Step 5: FOCUSED assertion — the code-fence qualified caller must be
        # present in BOTH incremental and --full results for target_fn.
        #
        # On HEAD (c3d91c5) this FAILS because:
        #   incremental target_fn.called_by = ['Caller.call_target', 'call_target', 'probe_fn']
        #   --full      target_fn.called_by = ['Caller.call_target', 'SvnClient.parse_unified_diff',
        #                                       'call_target', 'parse_unified_diff', 'probe_fn']
        # ------------------------------------------------------------------
        incr_target_units = find_units_by_name(meta_incr, "target_fn")
        full_target_units = find_units_by_name(meta_full, "target_fn")

        assert incr_target_units, (
            "target_fn not found in incremental metadata — index may have failed."
        )
        assert full_target_units, (
            "target_fn not found in --full metadata — --full index may have failed."
        )

        incr_cb = sorted(incr_target_units[0].get("called_by") or [])
        full_cb = sorted(full_target_units[0].get("called_by") or [])

        # Sanity: --full must have the qualified fence caller (fixture check).
        assert "SvnClient.parse_unified_diff" in full_cb, (
            f"FIXTURE SANITY: --full called_by for target_fn must contain "
            f"'SvnClient.parse_unified_diff'.  Got: {full_cb!r}.  "
            f"The regression can only be demonstrated if --full produces the "
            f"qualified fence caller — check the spec.md fixture."
        )

        # Sanity: incremental must have added probe_fn (confirming it ran).
        assert "probe_fn" in incr_cb, (
            f"FIXTURE SANITY: incremental called_by for target_fn must contain "
            f"'probe_fn' (from the newly added probe.rs).  Got: {incr_cb!r}.  "
            f"If this fails, the incremental path may not have run."
        )

        # THE REGRESSION ASSERTION — this FAILS on the unfixed tree:
        assert "SvnClient.parse_unified_diff" in incr_cb, (
            f"REGRESSION (cross-language code-fence cache-keying bug):\n"
            f"  incremental target_fn.called_by is missing the qualified caller "
            f"'SvnClient.parse_unified_diff' harvested from spec.md's Rust code fence.\n"
            f"  incremental called_by: {incr_cb!r}\n"
            f"  --full      called_by: {full_cb!r}\n"
            f"  missing from incr:     {sorted(set(full_cb) - set(incr_cb))!r}\n\n"
            f"ROOT CAUSE: _augment_cache_with_carried (semantic.py ~line 1991-2002)\n"
            f"keys carried files under unit.language ('markdown' for spec.md).\n"
            f"extract_file_calls_for_language(spec.md, 'markdown') returns {{}}.\n"
            f"The --full path keys spec.md under structure_lang='rust' and calls\n"
            f"_extract_rust_file_calls(spec.md) which parses the Rust code fence\n"
            f"and harvests 'SvnClient.parse_unified_diff' as a caller of target_fn.\n\n"
            f"FIX: in _augment_cache_with_carried, key each carried file under the\n"
            f"dispatch language(s) present in the fresh cache (not unit.language)\n"
            f"so carried non-source files are re-extracted under the correct language."
        )

        # ------------------------------------------------------------------
        # Step 6: Full equivalence across ALL units (belt-and-suspenders).
        # Incremental and --full must agree on EVERY unit's calls and called_by.
        # ------------------------------------------------------------------
        all_problems = validate_incr_full_equivalence(meta_incr, meta_full)
        assert not all_problems, (
            f"Incremental call-graph diverges from --full rebuild "
            f"({len(all_problems)} problem(s)) after adding probe.rs "
            f"(callee.rs, caller.rs, and spec.md are carried):\n"
            + "\n".join(f"  {p}" for p in all_problems[:20])
            + (f"\n  ... and {len(all_problems) - 20} more"
               if len(all_problems) > 20 else "")
            + "\n\nRED: _augment_cache_with_carried keys spec.md under "
            f"unit.language='markdown' instead of dispatch language 'rust', "
            f"so the Rust code fence callers (SvnClient.parse_unified_diff, "
            f"parse_unified_diff) are absent from the incremental augmented cache."
        )


# ===========================================================================
# C-1 regression: multi-language incremental run, only ONE language changed.
# ===========================================================================

class TestMultiLangIncrementalEqualsFullCallGraph:
    """Multi-language (`--lang all`) incremental reindex must preserve the call
    edges of UNCHANGED languages when only ONE language's files changed.

    Bug (pre-C-1): ``_augment_cache_with_carried`` derived ``dispatch_langs``
    ONLY from the fresh ``file_calls_cache`` — which in a ``--lang all`` run
    that changed only Rust contains just ``'rust'``.  Carried ``.ts`` files were
    therefore never re-extracted under ``'typescript'``, so the cross-file edge
    ``tsHelper.called_by == ['tsMain']`` vanished from the incrementally-rebuilt
    graph (rebuilt FRESH from the augmented cache; _reapply_call_graph runs with
    call_graph=None).

    RED on pre-C-1 tree: incremental ``tsHelper.called_by == []`` while
    ``--full`` has ``['tsMain']`` (>=2 equivalence diffs).

    GREEN after C-1: ``dispatch_langs |= {u.language for u in carried_units}``
    re-extracts each carried TS file under ``'typescript'`` so the edge survives
    and incremental == --full for every shared unit.
    """

    def test_carried_typescript_edge_survives_rust_only_incremental(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index a Rust+TS repo under `--lang all`; add only a Rust file;
        run incremental and `--full`; assert tsHelper.called_by contains 'tsMain'
        in BOTH and per-unit calls/called_by are EXACTLY equal.

        This GUARDS the C-1 fix: without the carried-language union, the carried
        TS edge is dropped on the incremental run.
        """
        pytest.importorskip("tree_sitter_rust")
        pytest.importorskip("tree_sitter_typescript")

        from tldr.semantic import build_semantic_index

        # Remove stray /private/tmp/.tldr that would hijack project-root detection.
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        # Sequential workers so monkeypatching of model is reliable.
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_incr = tmp_path / "dual_incr"
        project_full = tmp_path / "dual_full"
        project_incr.mkdir()
        project_full.mkdir()

        # Identical initial state: Rust files + spec.md + ts_helper.ts + ts_main.ts
        for proj in (project_incr, project_full):
            _build_dual_lang_repo(proj)

        fake_model = make_fake_model()

        # ------------------------------------------------------------------
        # Step 1: Cold index on project_incr under `--lang all` (multi-language
        #   dispatch — Rust AND TypeScript). Full parse, no prior state.
        # ------------------------------------------------------------------
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="all",
                show_progress=False, respect_ignore=False,
            )

        # SANITY: cold index must already produce the carried TS edge
        # tsHelper.called_by == ['tsMain'] (a full parse covers all languages).
        meta_cold = read_semantic_metadata(project_incr)
        cold_helper_units = find_units_by_name(meta_cold, "tsHelper")
        assert cold_helper_units, (
            "tsHelper not found in cold index — tree-sitter-typescript may be "
            "unavailable or the TS fixture source is not parseable, or `--lang "
            "all` did not dispatch TypeScript."
        )
        cold_helper_cb = sorted(cold_helper_units[0].get("called_by") or [])
        assert "tsMain" in cold_helper_cb, (
            f"FIXTURE SANITY: 'tsMain' must appear in tsHelper.called_by after "
            f"the cold (full-parse, `--lang all`) index.  Got: {cold_helper_cb!r}."
        )

        # ------------------------------------------------------------------
        # Step 2: Change ONLY Rust — add src/probe.rs to BOTH repos. The TS
        #   files (ts_helper.ts, ts_main.ts) are UNCHANGED → carried.
        # ------------------------------------------------------------------
        (project_incr / "src" / "probe.rs").write_text(_PROBE_RS)
        (project_full / "src" / "probe.rs").write_text(_PROBE_RS)

        # ------------------------------------------------------------------
        # Step 3: Incremental reindex on project_incr under `--lang all`.
        #   probe.rs is fresh; everything else (Rust callee/caller, spec.md,
        #   BOTH .ts files) is carried.  The fresh file_calls_cache holds only
        #   the 'rust' dispatch key — pre-C-1, carried .ts files are NEVER
        #   re-extracted under 'typescript' and tsHelper.called_by goes empty.
        # ------------------------------------------------------------------
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="all",
                show_progress=False, respect_ignore=False,
                # No full=True — this is the incremental path.
            )

        # ------------------------------------------------------------------
        # Step 4: --full rebuild on project_full under `--lang all` (same tree).
        # ------------------------------------------------------------------
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="all",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = read_semantic_metadata(project_incr)
        meta_full = read_semantic_metadata(project_full)

        # ------------------------------------------------------------------
        # Step 5: FOCUSED assertion — the carried TS edge must survive in the
        # incremental result.  THIS is the line that goes RED without C-1.
        # ------------------------------------------------------------------
        incr_helper_units = find_units_by_name(meta_incr, "tsHelper")
        full_helper_units = find_units_by_name(meta_full, "tsHelper")

        assert incr_helper_units, (
            "tsHelper not found in incremental metadata — index may have failed."
        )
        assert full_helper_units, (
            "tsHelper not found in --full metadata — --full index may have failed."
        )

        incr_helper_cb = sorted(incr_helper_units[0].get("called_by") or [])
        full_helper_cb = sorted(full_helper_units[0].get("called_by") or [])

        # Sanity: --full must have the carried TS edge.
        assert "tsMain" in full_helper_cb, (
            f"FIXTURE SANITY: --full tsHelper.called_by must contain 'tsMain'.  "
            f"Got: {full_helper_cb!r}."
        )

        # Sanity: the Rust change must have landed (incremental actually ran).
        incr_target_units = find_units_by_name(meta_incr, "target_fn")
        assert incr_target_units, (
            "target_fn not found in incremental metadata — index may have failed."
        )
        incr_target_cb = sorted(incr_target_units[0].get("called_by") or [])
        assert "probe_fn" in incr_target_cb, (
            f"FIXTURE SANITY: incremental target_fn.called_by must contain "
            f"'probe_fn' (from the newly added probe.rs).  Got: {incr_target_cb!r}."
        )

        # THE REGRESSION ASSERTION — FAILS on the pre-C-1 tree:
        assert "tsMain" in incr_helper_cb, (
            f"REGRESSION (C-1 multi-language carried-edge drop):\n"
            f"  incremental tsHelper.called_by is missing the carried cross-file "
            f"caller 'tsMain' — the TS edge was dropped because only Rust changed.\n"
            f"  incremental called_by: {incr_helper_cb!r}\n"
            f"  --full      called_by: {full_helper_cb!r}\n\n"
            f"ROOT CAUSE: _augment_cache_with_carried derived dispatch_langs ONLY "
            f"from the fresh file_calls_cache ({{'rust'}}), so carried .ts files "
            f"were never re-extracted under 'typescript'.\n"
            f"FIX (C-1): dispatch_langs |= {{u.language for u in carried_units}}."
        )

        # ------------------------------------------------------------------
        # Step 6: Full equivalence across ALL units. Incremental and --full must
        # agree on EVERY shared unit's calls and called_by.
        # ------------------------------------------------------------------
        all_problems = validate_incr_full_equivalence(meta_incr, meta_full)
        assert not all_problems, (
            f"Incremental call-graph diverges from --full rebuild "
            f"({len(all_problems)} problem(s)) after a Rust-only change in a "
            f"multi-language (`--lang all`) repo (TS files carried):\n"
            + "\n".join(f"  {p}" for p in all_problems[:20])
            + (f"\n  ... and {len(all_problems) - 20} more"
               if len(all_problems) > 20 else "")
        )
