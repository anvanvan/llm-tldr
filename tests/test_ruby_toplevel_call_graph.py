"""Regression test: Ruby top-level def + call site yields edges in call graph.

Bug: `_extract_ruby_file_calls.process()` in tldr/cross_file_calls.py gates
`calls_by_func` insertion on `method`/`singleton_method` nodes only.  A
program-scope `call` node (e.g. `greet("alice")` at file top-level) never
opens a caller bucket, so the edge-emission loop emits zero edges and
`build_project_call_graph` returns an empty graph.

Regression guard: given a single .rb file with

    def greet(name); puts "hi #{name}"; end
    greet("alice")

the call graph MUST contain at least one edge with callee "greet" (i.e. the
top-level call site was attributed to a synthetic caller frame or a "__main__"
bucket).  An empty edge list or a missing-callee lookup are the bug symptoms.

Verified RED on dev branch (before fix): graph.edges is [] — assertion fails.

Phase 8 regression gaps (two additional tests below):
  (a) LINE-NUMBER MISSING: The edge is emitted but the impact_analysis
      caller entry has no 'line' field.  The expected shape (per verification.md)
      is `{"callers": [{"file": "main.rb", "line": 5, ...}], ...}`.
  (b) ORPHAN-DEF NOT REGISTERED: A top-level `def orphan_fn` with NO callers
      anywhere still errors with "Function 'orphan_fn' not found in call graph"
      instead of returning a valid result with an empty callers set.
"""

import pytest
from pathlib import Path
from tldr.cross_file_calls import build_project_call_graph
from tldr.analysis import impact_analysis


# ---------------------------------------------------------------------------
# Helpers (consistent with test_remaining_langs_tiers_feature.py style)
# ---------------------------------------------------------------------------

def _callee_edges(graph, callee_name: str) -> list:
    """Return all edges whose dst_func == callee_name."""
    return [e for e in graph.edges if e[3] == callee_name]


# ---------------------------------------------------------------------------
# Regression test (Phase 7 fix — stays GREEN)
# ---------------------------------------------------------------------------

def test_ruby_toplevel_call_produces_edge_for_greet(tmp_path: Path):
    """Top-level `greet("alice")` in a single .rb file must produce a callee
    edge for 'greet' in the Ruby call graph.

    Reproducer from verification.md F1b:
        def greet(name); puts "hi #{name}"; end
        greet("alice")

    Before the fix: build_project_call_graph returns an empty edge list and
    `tldr impact greet` errors with 'Function greet not found in call graph'.
    After the fix: at least one edge with dst_func=='greet' must be present,
    proving the program-scope call site was attributed to a caller bucket.
    """
    pytest.importorskip("tree_sitter_ruby")

    main_rb = tmp_path / "main.rb"
    main_rb.write_text(
        'def greet(name)\n'
        '  puts "hi #{name}"\n'
        'end\n'
        '\n'
        'greet("alice")\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="ruby")

    greet_edges = _callee_edges(graph, "greet")
    assert greet_edges, (
        "Expected at least one edge with callee='greet' for a top-level "
        "Ruby call site `greet(\"alice\")`, but the call graph has no such "
        f"edge. All edges: {graph.edges!r}\n\n"
        "Root cause: _extract_ruby_file_calls.process() at "
        "tldr/cross_file_calls.py:5117-5125 only opens a calls_by_func "
        "bucket for `method`/`singleton_method` nodes; program-scope call "
        "nodes are silently dropped, leaving the edge set empty."
    )


# ---------------------------------------------------------------------------
# Phase 8 regression tests — MUST FAIL on current code
# ---------------------------------------------------------------------------

def test_ruby_toplevel_caller_has_line_number(tmp_path: Path):
    """impact_analysis caller entry for a top-level call site MUST include
    the call-site line number in a 'line' field.

    Phase 8 gap (a): after the Phase 7 fix the edge IS emitted, but the
    caller record returned by impact_analysis has no 'line' key.  The
    verification.md expected shape is:
        {"callers": [{"file": "main.rb", "line": 5, ...}], ...}

    Reproducer:
        main.rb line 1:  def greet(name)
        main.rb line 2:    puts "hi #{name}"
        main.rb line 3:  end
        main.rb line 4:  (blank)
        main.rb line 5:  greet("alice")   <-- call-site is line 5

    After the correct fix: the impact_analysis result for 'greet' must have
    at least one caller entry that contains a 'line' key equal to 5 (the
    call-site line number in the source file).

    Verified RED: impact_analysis returns caller dicts with keys
    {function, file, caller_count, callers, truncated} — no 'line' key.
    """
    pytest.importorskip("tree_sitter_ruby")

    main_rb = tmp_path / "main.rb"
    main_rb.write_text(
        'def greet(name)\n'       # line 1
        '  puts "hi #{name}"\n'   # line 2
        'end\n'                    # line 3
        '\n'                       # line 4
        'greet("alice")\n'         # line 5
    )

    graph = build_project_call_graph(str(tmp_path), language="ruby")
    result = impact_analysis(graph, "greet")

    assert "error" not in result, (
        f"impact_analysis returned an error instead of caller data: {result!r}\n"
        "The Phase 7 edge-emission fix must be present for this test to reach "
        "the line-number assertion."
    )

    # Collect all caller dicts from every target entry (handles nested trees)
    def _collect_callers(node: dict) -> list:
        out = list(node.get("callers", []))
        for c in out:
            out.extend(_collect_callers(c))
        return out

    all_caller_entries = []
    for target_dict in result.get("targets", {}).values():
        all_caller_entries.extend(_collect_callers(target_dict))

    assert all_caller_entries, (
        "Expected at least one caller entry in impact_analysis result for "
        f"'greet', but callers list is empty. Full result: {result!r}"
    )

    missing_line = [c for c in all_caller_entries if "line" not in c]
    assert not missing_line, (
        "Every caller entry must include a 'line' field with the call-site "
        f"line number. Entries missing 'line': {missing_line!r}\n"
        "Expected shape: {\"file\": \"main.rb\", \"line\": 5, ...}"
    )

    # At least one caller entry should report line 5 (the greet("alice") call)
    call_site_line = 5
    lines_found = [c["line"] for c in all_caller_entries if "line" in c]
    assert call_site_line in lines_found, (
        f"Expected caller entry with line={call_site_line} (the `greet(\"alice\")` "
        f"call site in main.rb), but got lines: {lines_found!r}\n"
        "The edge tuple and/or impact_analysis must carry the source line number."
    )


def test_ruby_orphan_toplevel_def_is_registered(tmp_path: Path):
    """A top-level `def orphan_fn` with no callers anywhere must be registered
    in the call graph and return a valid (empty-callers) result, NOT an error.

    Phase 8 gap (b): `orphan_fn` is defined at top-level with no callers and
    no outbound calls to project-defined functions.  It therefore appears in
    `defined` but never in any edge (neither as src_func nor dst_func).
    impact_analysis hits the 'not found' branch and returns:
        {"error": "Function 'orphan_fn' not found in call graph"}

    The correct behaviour: a top-level defined function with no callers should
    be reachable via impact_analysis (or equivalent API) as an entry point that
    simply has zero callers — similar to how a named def that calls others is
    treated as 'callers_only'.  No 'error' key should be present.

    Reproducer:
        def orphan_fn
          42
        end
        # — no call to orphan_fn anywhere

    Verified RED: impact_analysis returns {"error": "Function 'orphan_fn' not
    found in call graph"} because ProjectCallGraph has no node registry — only
    edges — and orphan_fn never participates in any edge.
    """
    pytest.importorskip("tree_sitter_ruby")

    main_rb = tmp_path / "main.rb"
    main_rb.write_text(
        'def orphan_fn\n'
        '  42\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="ruby")
    result = impact_analysis(graph, "orphan_fn")

    assert "error" not in result, (
        "impact_analysis must NOT return an error for a top-level defined "
        "function 'orphan_fn' that simply has no callers.  Got: "
        f"{result!r}\n\n"
        "Root cause: ProjectCallGraph (cross_file_calls.py:144-179) is edge-only "
        "— it has no node registry, so 'orphan_fn' never appears in any edge "
        "and impact_analysis hits the 'not found' branch at analysis.py:160.  "
        "The fix must either register defined functions in the graph or treat "
        "them as zero-caller entry points in impact_analysis."
    )

    # Should be a valid result with zero callers
    assert "targets" in result, (
        f"Expected 'targets' key in result for 'orphan_fn', got: {result!r}"
    )

    all_caller_counts = [
        v.get("caller_count", None)
        for v in result["targets"].values()
    ]
    assert all(c == 0 for c in all_caller_counts), (
        "orphan_fn has no callers; all caller_count values must be 0. "
        f"Got: {all_caller_counts!r}"
    )
