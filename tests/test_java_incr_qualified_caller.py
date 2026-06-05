"""
Regression test: Java incremental index must emit QUALIFIED ClassName.method callers.

Root cause (tldr/semantic.py::_augment_cache_with_carried ~line 1976):
    per_file.setdefault(unit.name, [...])

uses ONLY the BARE unit.name as the caller key.  Java's
_extract_java_file_calls dual-keys callers:
    calls_by_func[name] = ...           # bare:     "methodA"
    calls_by_func[full_name] = ...      # qualified: "ClassA.methodA"

So a --full rebuild produces both bare AND qualified caller keys in the
prebuilt_file_calls cache, while the incremental carry path only inserts the
bare key → the resulting called_by for every callee is missing every
ClassName.method entry.

Fix (Option C, in semantic.py::_build_reapply_call_maps): re-parse carried
files via the real per-language _extract_*_file_calls extractors instead of
synthesising bare-only entries — then the native dual-keying produces the same
caller entries as --full BY CONSTRUCTION.

This test:
  1. Builds a 2-file Java repo: ClassA (methodA calls ClassB.methodB) and
     ClassB (methodB).  Cross-file call → ClassA.methodA must appear in
     ClassB.methodB's called_by.
  2. Cold-indexes the repo (full parse, no prior state).
  3. Adds a trivial 3rd file (Probe.java, no cross-file calls) to force an
     incremental reindex.
  4. Runs an incremental reindex (DEFAULT path, no --full).
  5. Runs a --full rebuild on the SAME final disk state.
  6. Asserts that for the cross-called method (methodB), the incremental
     called_by CONTAINS the qualified caller "ClassA.methodA" — same as --full.

RED on the current tree:  incremental called_by for methodB is ['methodA'] (bare
only); --full called_by is ['ClassA.methodA', 'methodA'].  The assertion that
'ClassA.methodA' is present in the incremental result FAILS.

PASS after Option C: re-parsing carried files restores the qualified key.

Runner: python3 -m pytest --no-cov tests/test_java_incr_qualified_caller.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

from conftest import (
    find_units_by_name,
    make_fake_model,
    read_semantic_metadata,
    validate_incr_full_equivalence,
)


# ---------------------------------------------------------------------------
# Java source fixtures — minimal but real enough for tree-sitter to parse
# ---------------------------------------------------------------------------

# ClassA.java: ClassA.methodA() calls b.methodB() on a ClassB instance.
# This cross-file call causes ClassA.methodA (and bare methodA) to appear
# as callers of methodB in the --full call graph.
_CLASS_A_JAVA = """\
package com.example;

public class ClassA {
    public void methodA() {
        ClassB b = new ClassB();
        b.methodB();
    }
}
"""

# ClassB.java: ClassB.methodB() is the CALLEE.  After a --full rebuild its
# called_by should be ['ClassA.methodA', 'methodA'] (both bare + qualified).
_CLASS_B_JAVA = """\
package com.example;

public class ClassB {
    public void methodB() {
        // target method — called by ClassA.methodA
    }
}
"""

# Probe.java: trivial file added AFTER the cold index to trigger an
# incremental reindex.  No cross-file calls so it does not affect the
# ClassA→ClassB edge — it only changes the file set.
_PROBE_JAVA = """\
package com.example;

public class Probe {
    public void probe() {
        // no cross-file calls
    }
}
"""


def _build_java_repo(root: Path) -> None:
    """Write the initial 2-file Java repo and a .git dir inside root."""
    (root / ".git").mkdir(exist_ok=True)
    (root / "ClassA.java").write_text(_CLASS_A_JAVA)
    (root / "ClassB.java").write_text(_CLASS_B_JAVA)




# ===========================================================================
# The regression test
# ===========================================================================

class TestJavaIncrQualifiedCallerInCalledBy:
    """Incremental Java reindex must preserve the qualified ClassName.method
    caller key in called_by — not just the bare method name.

    RED on HEAD: _augment_cache_with_carried inserts carried Java units under
    only the bare unit.name key.  Java's _extract_java_file_calls dual-keys
    (bare + ClassName.method), so the incremental called_by is missing every
    qualified ClassName.method entry that --full produces.

    Specifically for this fixture: ClassB.methodB.called_by after incremental
    reindex is ['methodA'] but after --full is ['ClassA.methodA', 'methodA'].
    The assertion that 'ClassA.methodA' is present in the incremental result FAILS.
    """

    def test_qualified_caller_present_in_incremental_called_by(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index 2-file Java repo; add Probe.java; run incremental; run --full;
        assert ClassB.methodB.called_by contains 'ClassA.methodA' in BOTH results.

        RED reason (current tree):
          _augment_cache_with_carried line 1976:
              per_file.setdefault(unit.name, [('intra', callee) for callee in unit.calls])
          uses ONLY unit.name ('methodA') — the qualified 'ClassA.methodA' key is
          NEVER inserted.  _build_java_call_graph consequently never sees
          ClassA.methodA as a caller of methodB.  The incremental called_by for
          methodB is ['methodA'] while --full's is ['ClassA.methodA', 'methodA'].
          The assertion below (that 'ClassA.methodA' is present in incremental
          called_by) FAILS with an AssertionError naming the missing qualified caller.
        """
        from tldr.semantic import build_semantic_index

        # Remove stray /private/tmp/.tldr that would hijack project-root detection
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        # Sequential workers so monkeypatching of model is reliable
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_incr = tmp_path / "java_incr"
        project_full = tmp_path / "java_full"
        project_incr.mkdir()
        project_full.mkdir()

        # Identical initial state: ClassA.java + ClassB.java
        for proj in (project_incr, project_full):
            _build_java_repo(proj)

        fake_model = make_fake_model()

        # ------------------------------------------------------------------
        # Step 1: Cold index on project_incr (creates the prior snapshot)
        # ------------------------------------------------------------------
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="java",
                show_progress=False, respect_ignore=False,
            )

        # Sanity: cold index must have indexed ClassB.methodB with ClassA.methodA
        # as a caller (the cold path does a full parse, same as --full).
        meta_cold = read_semantic_metadata(project_incr)
        cold_method_b_units = find_units_by_name(meta_cold, "methodB")
        assert cold_method_b_units, (
            "methodB not found in cold index — Java tree-sitter parsing may be "
            "unavailable or the fixture Java source is not parseable."
        )
        cold_cb = sorted(cold_method_b_units[0].get("called_by") or [])
        assert "ClassA.methodA" in cold_cb, (
            f"FIXTURE SANITY: ClassA.methodA must appear in methodB.called_by "
            f"after the cold (full-parse) index.  Got: {cold_cb!r}.  "
            f"If this fails, tree-sitter-java may be unavailable or the Java "
            f"source fixture is not parsed correctly — cannot test the regression."
        )

        # ------------------------------------------------------------------
        # Step 2: Add Probe.java to BOTH repos (same final disk state)
        # ------------------------------------------------------------------
        (project_incr / "Probe.java").write_text(_PROBE_JAVA)
        (project_full / "Probe.java").write_text(_PROBE_JAVA)

        # ------------------------------------------------------------------
        # Step 3: Incremental reindex on project_incr (DEFAULT path, no --full)
        #   ClassA.java and ClassB.java are UNCHANGED → carried from prior snapshot.
        #   Probe.java is new → freshly parsed.
        # ------------------------------------------------------------------
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="java",
                show_progress=False, respect_ignore=False,
                # No full=True — this is the incremental path
            )

        # ------------------------------------------------------------------
        # Step 4: --full rebuild on project_full (same final tree)
        # ------------------------------------------------------------------
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_full), lang="java",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        meta_incr = read_semantic_metadata(project_incr)
        meta_full = read_semantic_metadata(project_full)

        # ------------------------------------------------------------------
        # Step 5: FOCUSED assertion — the qualified caller must be present
        #
        # This is the exact symptom of the bug.  On the current (unfixed) tree:
        #   incremental methodB.called_by = ['methodA']         (bare only)
        #   --full       methodB.called_by = ['ClassA.methodA', 'methodA']
        #
        # The assertion below fails on HEAD because 'ClassA.methodA' is absent
        # from the incremental result.  After Option C it will pass.
        # ------------------------------------------------------------------
        incr_method_b_units = find_units_by_name(meta_incr, "methodB")
        full_method_b_units = find_units_by_name(meta_full, "methodB")

        assert incr_method_b_units, (
            "methodB not found in incremental metadata — index may have failed."
        )
        assert full_method_b_units, (
            "methodB not found in --full metadata — --full index may have failed."
        )

        incr_cb = sorted(incr_method_b_units[0].get("called_by") or [])
        full_cb = sorted(full_method_b_units[0].get("called_by") or [])

        # The --full rebuild must have the qualified caller (sanity check on fixture):
        assert "ClassA.methodA" in full_cb, (
            f"FIXTURE SANITY: --full called_by for methodB must contain "
            f"'ClassA.methodA'.  Got: {full_cb!r}.  "
            f"The regression can only be demonstrated if --full produces the "
            f"qualified caller — check the Java fixture source."
        )

        # THE REGRESSION ASSERTION — this is what FAILS on the unfixed tree:
        assert "ClassA.methodA" in incr_cb, (
            f"REGRESSION: incremental called_by for ClassB.methodB is missing "
            f"the qualified caller 'ClassA.methodA'.\n"
            f"  incremental called_by: {incr_cb!r}\n"
            f"  --full      called_by: {full_cb!r}\n"
            f"  missing from incr:     {sorted(set(full_cb) - set(incr_cb))!r}\n\n"
            f"ROOT CAUSE: _augment_cache_with_carried (semantic.py ~line 1976) "
            f"inserts carried Java units under ONLY the bare unit.name key "
            f"('methodA'), not the qualified 'ClassA.methodA' key.  "
            f"Java's _extract_java_file_calls dual-keys callers (bare + "
            f"ClassName.method), so --full emits both.  The incremental carry "
            f"path emits only the bare form → ClassA.methodA absent from "
            f"incremental called_by.\n\n"
            f"FIX (Option C): re-parse carried files via _extract_java_file_calls "
            f"in _build_reapply_call_maps so the native dual-keying restores both "
            f"caller forms, matching --full BY CONSTRUCTION."
        )

        # ------------------------------------------------------------------
        # Step 6: Full equivalence across ALL units (belt-and-suspenders).
        # Incremental and --full must agree on every unit's calls and called_by.
        # ------------------------------------------------------------------
        all_problems = validate_incr_full_equivalence(meta_incr, meta_full)
        assert not all_problems, (
            f"Incremental call-graph diverges from --full rebuild "
            f"({len(all_problems)} problem(s)) after adding Probe.java "
            f"(ClassA.java and ClassB.java are carried):\n"
            + "\n".join(f"  {p}" for p in all_problems[:20])
            + (f"\n  ... and {len(all_problems) - 20} more"
               if len(all_problems) > 20 else "")
            + "\n\nRED: _augment_cache_with_carried uses bare unit.name only; "
            f"Java dual-keying means qualified ClassName.method callers are "
            f"absent from the incremental path."
        )
