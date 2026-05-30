# pyright: reportMissingImports=false
"""Regression test — Bug 007 (reopened): Swift call-graph has 4 real gaps
in the TREE-SITTER-PRESENT path (tree_sitter_swift==0.7.2 installed).

The earlier narrow fix addressed silent-degradation when tree_sitter_swift
is absent.  These 4 gaps exist REGARDLESS of tree_sitter availability:

GAP 1 — Cross-file caller resolution missing
  get_relevant_context(project, "funcX", language="swift") returns only the
  definition node (1 function).  Cross-file callers from other .swift files
  are absent even though _build_swift_call_graph walks all files.

GAP 2 — .tldr/cache/ not persisted after Swift context invocation
  For Python projects, .tldr/cache/ is created.  For Swift, the directory
  is absent after get_relevant_context completes.

GAP 3 — .swift not auto-detected without explicit language="swift"
  get_relevant_context(...) without language= defaults to "python" and
  returns an error.  Expected: auto-detect the dominant language in the
  project and return the same result as with explicit language="swift".

GAP 4 — extract_file_with_code(file, function=NAME) returns empty
  functions[] when the target function is a method inside a class.
  The filter at api.py only searches top-level functions[], not
  classes[].methods[].  Expected: the named method appears in the
  result with line_number > 0.

This single test function asserts all 4 fixed behaviors via separate
assert statements so pytest's assertion introspection pinpoints each gap.
"""

import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Minimal Swift fixture — two files: definition + cross-file caller
# ---------------------------------------------------------------------------

_FUNC_DEF_SWIFT = textwrap.dedent("""\
    import Foundation

    func funcX() {
        print("funcX called")
    }

    func helperFn() {
        print("helper")
    }
""")

_CALLER_SWIFT = textwrap.dedent("""\
    import Foundation

    func callerA() {
        funcX()
    }

    func callerB() {
        funcX()
    }
""")

_CLASS_WITH_METHOD_SWIFT = textwrap.dedent("""\
    import Foundation

    class VocabStore {
        func recordUsedTerms(_ terms: [String]) {
            print(terms)
        }

        func loadIfNeeded() {
            print("loading")
        }
    }
""")


# ---------------------------------------------------------------------------
# The single regression test function — one function, 4 asserts
# ---------------------------------------------------------------------------


def test_swift_context_full_parity_with_python_call_graph(tmp_path: Path):
    """Bug 007 (tree-sitter-PRESENT path): Swift call-graph must expose callers,
    persist cache, auto-detect language, and return class-method metadata.

    All 4 assert statements represent FIXED behavior.  The test MUST FAIL on
    current unfixed code because all 4 gaps are real.  A correctly-implemented
    fix will make each assertion pass independently.
    """
    # --- Fixture setup ---
    # Two-file project: FuncDef.swift defines funcX(); Caller.swift calls it.
    (tmp_path / "FuncDef.swift").write_text(_FUNC_DEF_SWIFT)
    (tmp_path / "Caller.swift").write_text(_CALLER_SWIFT)
    # Class-method file for Assert 4
    (tmp_path / "VocabStore.swift").write_text(_CLASS_WITH_METHOD_SWIFT)

    from tldr.api import extract_file_with_code, get_relevant_context

    # -----------------------------------------------------------------------
    # ASSERT 1 — Cross-file caller resolution
    # -----------------------------------------------------------------------
    # get_relevant_context must surface at least one caller from Caller.swift.
    # Current behavior: only FuncDef.funcX itself is returned (1 function, no
    # cross-file callers), because the Swift call-graph builds edges but the
    # BFS traversal is not finding callers from the adjacency list.
    ctx_explicit = get_relevant_context(
        project=str(tmp_path),
        entry_point="funcX",
        depth=2,
        language="swift",
    )
    caller_files = [f.file for f in ctx_explicit.functions]
    callers_from_caller_swift = [f for f in caller_files if "Caller" in f]
    assert callers_from_caller_swift, (
        "ASSERT 1 FAILED — cross-file caller resolution: "
        f"get_relevant_context returned {len(ctx_explicit.functions)} function(s) "
        f"with files={caller_files!r}, "
        "but expected at least one entry from Caller.swift. "
        "callerA() and callerB() both call funcX() and should appear as callers."
    )

    # -----------------------------------------------------------------------
    # ASSERT 2 — .tldr/cache/ must be created after Swift context invocation
    # -----------------------------------------------------------------------
    # The cache directory must exist with at least one file after the above
    # get_relevant_context call completes.  Python projects always create it;
    # Swift does not, indicating the build-graph → cache-write path is broken.
    cache_dir = tmp_path / ".tldr" / "cache"
    cache_files = list(cache_dir.iterdir()) if cache_dir.exists() else []
    assert cache_dir.exists() and len(cache_files) >= 1, (
        "ASSERT 2 FAILED — cache persistence: "
        f".tldr/cache/ {'does not exist' if not cache_dir.exists() else 'is empty'}. "
        "After get_relevant_context(language='swift'), the project cache directory "
        "must be created with at least one file (e.g. call_graph.json). "
        f"Cache dir: {cache_dir!s}, files found: {cache_files!r}"
    )

    # -----------------------------------------------------------------------
    # ASSERT 3 — Auto-detect: no explicit language= arg must succeed
    # -----------------------------------------------------------------------
    # Calling get_relevant_context without language= currently defaults to
    # "python" and returns an error ("Function 'funcX' not found in project").
    # The fixed API must auto-detect "swift" from the project's .swift files
    # and return the same non-error result as ASSERT 1.
    ctx_auto = get_relevant_context(
        project=str(tmp_path),
        entry_point="funcX",
        depth=2,
        # NOTE: no language= argument — tests auto-detection
    )
    assert ctx_auto.error is None and len(ctx_auto.functions) >= 1, (
        "ASSERT 3 FAILED — auto language detection: "
        f"get_relevant_context(project, 'funcX', depth=2) without language= "
        f"returned error={ctx_auto.error!r} and "
        f"{len(ctx_auto.functions)} function(s). "
        "Expected: auto-detect 'swift' from .swift files and return >=1 function "
        "with no error (same as with explicit language='swift')."
    )

    # -----------------------------------------------------------------------
    # ASSERT 4 — extract_file_with_code returns class-method metadata
    # -----------------------------------------------------------------------
    # extract_file_with_code(file, function="recordUsedTerms") currently returns
    # functions: [] because recordUsedTerms is a method inside VocabStore class,
    # and the function filter only checks top-level functions[], not
    # classes[].methods[].  The fixed behavior must find the method and return
    # it with line_number > 0.
    class_method_file = str(tmp_path / "VocabStore.swift")
    result = extract_file_with_code(class_method_file, function="recordUsedTerms")
    returned_functions = result.get("functions", [])
    # The fix may surface the method either by:
    #   (a) promoting it into functions[] in the extract result, OR
    #   (b) searching classes[].methods[] — either representation is acceptable
    #       as long as the returned entry has line_number > 0.
    # Check both locations.
    all_returned = returned_functions[:]
    for cls in result.get("classes", []):
        for m in cls.get("methods", []):
            if m.get("name") == "recordUsedTerms":
                all_returned.append(m)
    matching = [
        entry for entry in all_returned
        if entry.get("name") == "recordUsedTerms"
        and (entry.get("line_number") or 0) > 0
    ]
    assert matching, (
        "ASSERT 4 FAILED — extract_file_with_code class-method metadata: "
        f"extract_file_with_code(file, function='recordUsedTerms') returned "
        f"functions={returned_functions!r} and classes={result.get('classes', [])!r}. "
        "Expected: an entry for 'recordUsedTerms' with line_number > 0 in either "
        "functions[] or classes[].methods[]. "
        "recordUsedTerms is defined at line 4 of VocabStore.swift inside class VocabStore."
    )
