"""RED-phase regression tests for Elixir call-graph edge emission gaps (bug 008).

Current state (before the patch described in architecture.md):
- Single-line `def name(args), do: body` body is NOT traversed — outbound calls
  from that body produce ZERO edges.
- `defmacro`/`defmacrop` definitions are NOT registered.
- Capture `&local_fn/arity` and bare `&local_fn` are NOT collected as edges.
  (Note: `&Mod.fn/arity` IS captured via the default child recursion hitting
   the inner `call` node — it is not a gap on the current branch.)
- No `ELIXIR_ORPHAN_SENTINEL` constant exists; orphan functions return 'not found'.
- `_is_orphan_sentinel("__elixir_orphan__")` returns False (constant missing).

All tests below MUST fail on the current `dev` branch (commit f27064a and forward
until the patch is applied). They will turn GREEN once the patch is applied.

Test naming: test_elixir_<behavior>
Fixture style: inline tmp_path .ex files, build_project_call_graph in-process.
"""

import pytest
from pathlib import Path
from tldr.cross_file_calls import build_project_call_graph
from tldr.analysis import impact_analysis


# ---------------------------------------------------------------------------
# Helpers (consistent with test_remaining_langs_tiers_feature.py style)
# ---------------------------------------------------------------------------

def _callee_edges(graph, callee_name: str) -> list:
    """Return all edges whose dst_func ends with callee_name (handles FQN)."""
    return [e for e in graph.edges if e[3] == callee_name or e[3].endswith(f".{callee_name}")]


def _src_func_edges(graph, src_func_suffix: str) -> list:
    """Return all edges whose src_func ends with src_func_suffix."""
    return [e for e in graph.edges if e[1] == src_func_suffix or e[1].endswith(f".{src_func_suffix}")]


# ---------------------------------------------------------------------------
# Behavior 1: Single-line `def name(args), do: body` — outbound calls from body
# ---------------------------------------------------------------------------

def test_elixir_single_line_def_body_calls_emitted(tmp_path: Path):
    """Single-line `def greet(name), do: Helper.process(name)` must emit an
    outbound edge from Greeter.greet to Helper.process.

    Current gap: `visit_all` in `_extract_elixir_file_calls` only traverses
    `do_block` children for function bodies. For single-line defs the body is
    in `arguments → keywords → pair → value` — NOT in a `do_block`. So
    `calls_by_func["Greeter.greet"]` stays empty and no outbound edges are
    emitted from Greeter.greet.

    Note: Greeter.greet IS registered as a callee (other callers can call it),
    but Greeter.greet's own body calls are silently dropped.

    Verified RED: graph.edges is empty (no process edge, no greet-as-caller edge).
    """
    pytest.importorskip("tree_sitter_elixir")

    helper = tmp_path / "helper.ex"
    helper.write_text(
        'defmodule Helper do\n'
        '  def process(x) do\n'
        '    x\n'
        '  end\n'
        'end\n'
    )

    greeter = tmp_path / "greeter.ex"
    greeter.write_text(
        'defmodule Greeter do\n'
        '  def greet(name), do: Helper.process(name)\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    # Greeter.greet must appear as a src_func (caller) with Helper.process as callee
    greet_caller_edges = _src_func_edges(graph, "greet")
    assert greet_caller_edges, (
        "Expected at least one edge with src_func ending in 'greet' "
        "(Greeter.greet calling Helper.process) for single-line def "
        "`def greet(name), do: Helper.process(name)`. All edges: {graph.edges!r}\n\n"
        "Root cause: `visit_all` in _extract_elixir_file_calls only looks for "
        "`do_block` children (line 4908-4910). Single-line `do:` bodies are in "
        "`arguments → keywords → pair → named_children[-1]` — this subtree is "
        "never traversed, so calls_by_func[\"Greeter.greet\"] stays empty."
    )

    process_edges = _callee_edges(graph, "process")
    assert process_edges, (
        "Expected at least one edge with callee ending in 'process'. "
        f"All edges: {graph.edges!r}"
    )

    greet_calls_process = any(
        e[1].endswith(".greet") and e[3].endswith(".process")
        for e in graph.edges
    )
    assert greet_calls_process, (
        "Expected an edge from Greeter.greet to Helper.process. "
        f"All edges: {graph.edges!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 2: Multi-line `def name(args) do … end` — guard test (GREEN today)
# ---------------------------------------------------------------------------

def test_elixir_multiline_def_emits_edge(tmp_path: Path):
    """Multi-line `def greet(name) do ... end` must produce a cross-file edge.

    This form already works on the current `dev` branch. This test is kept as
    a non-RED guard: any regression from the single-line fix must break this test.
    """
    pytest.importorskip("tree_sitter_elixir")

    greeter = tmp_path / "greeter.ex"
    greeter.write_text(
        'defmodule Greeter do\n'
        '  def greet(name) do\n'
        '    IO.puts(name)\n'
        '  end\n'
        'end\n'
    )

    app = tmp_path / "app.ex"
    app.write_text(
        'defmodule App do\n'
        '  def run do\n'
        '    Greeter.greet("world")\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    greet_callee_edges = _callee_edges(graph, "greet")
    assert greet_callee_edges, (
        "Expected at least one edge with callee ending in 'greet' for multi-line "
        f"`def greet(name) do ... end`. All edges: {graph.edges!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 3: defmacro / defmacrop — registered and callable
# ---------------------------------------------------------------------------

def test_elixir_defmacro_is_registered_and_callable(tmp_path: Path):
    """defmacro definitions must be registered so cross-file invocations emit edges.

    Current gap: `ident in ("def", "defp")` at line 4902 excludes `defmacro`
    and `defmacrop`. Any `defmacro my_macro(...)` definition is silently dropped
    from `defined`, so Pass 2 never emits an edge for calls to it.

    Verified RED: graph.edges contains no edge for my_macro.
    """
    pytest.importorskip("tree_sitter_elixir")

    macros = tmp_path / "my_macros.ex"
    macros.write_text(
        'defmodule MyMacros do\n'
        '  defmacro my_macro(x) do\n'
        '    quote do: IO.puts(unquote(x))\n'
        '  end\n'
        '\n'
        '  defmacrop private_macro(x) do\n'
        '    quote do: x * 2\n'
        '  end\n'
        'end\n'
    )

    caller = tmp_path / "caller.ex"
    caller.write_text(
        'defmodule Caller do\n'
        '  require MyMacros\n'
        '\n'
        '  def run(x) do\n'
        '    MyMacros.my_macro(x)\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    # my_macro must appear as a callee (Caller.run -> MyMacros.my_macro)
    macro_edges = _callee_edges(graph, "my_macro")
    assert macro_edges, (
        "Expected at least one edge with callee ending in 'my_macro' for a "
        f"defmacro definition. All edges: {graph.edges!r}\n\n"
        "Root cause: `ident in (\"def\", \"defp\")` at line 4902 of cross_file_calls.py "
        "does not include 'defmacro' or 'defmacrop', so macro defs are never "
        "registered in `defined` and Pass 2 cannot emit edges for calls to them."
    )

    caller_in_caller_file = any("caller.ex" in e[0] for e in macro_edges)
    assert caller_in_caller_file, (
        f"Expected caller in caller.ex for my_macro, got: {macro_edges!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 4: Pipeline operator `|>` in single-line body — edges emitted
# ---------------------------------------------------------------------------

def test_elixir_pipeline_in_single_line_body_emits_edges(tmp_path: Path):
    """`def shout(name), do: name |> Helper.process() |> IO.puts()` must emit
    an outbound edge from Greeter.shout to Helper.process.

    Per architecture.md T2-8, `|>` in multi-line bodies already works via the
    unconditional child recursion. The gap here is specifically for single-line
    bodies: since visit_all skips the keywords/pair body (same gap as Behavior 1),
    the pipeline inside is never traversed — no outbound calls are collected.

    Verified RED: graph.edges has no edge from shout to process.
    """
    pytest.importorskip("tree_sitter_elixir")

    helper = tmp_path / "helper.ex"
    helper.write_text(
        'defmodule Helper do\n'
        '  def process(x) do\n'
        '    x\n'
        '  end\n'
        'end\n'
    )

    greeter = tmp_path / "greeter.ex"
    greeter.write_text(
        'defmodule Greeter do\n'
        '  def shout(name), do: name |> Helper.process()\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    shout_caller_edges = _src_func_edges(graph, "shout")
    assert shout_caller_edges, (
        "Expected at least one edge with src_func ending in 'shout' "
        "(Greeter.shout calling Helper.process via pipeline) for single-line def "
        "`def shout(name), do: name |> Helper.process()`. All edges: {graph.edges!r}\n\n"
        "Root cause: Same as Behavior 1 — visit_all skips the single-line body. "
        "The |> pipeline is inside the keywords/pair body value, which is never "
        "visited by extract_body_calls. After the Behavior 1 fix (keywords/pair "
        "body traversal), the default child recursion (T2-8) will naturally "
        "traverse the |> binary_operator and find its call children."
    )


# ---------------------------------------------------------------------------
# Behavior 5: Capture `&Mod.fn/arity` — qualified edge emitted
# ---------------------------------------------------------------------------

def test_elixir_capture_qualified_mod_fn_arity_emits_edge(tmp_path: Path):
    """Capture `&Helper.process/1` in a single-line def body must emit a
    qualified edge from Greeter.transform to Helper.process.

    Note: In multi-line bodies, `&Mod.fn/arity` IS already captured via
    default child recursion (the inner `call` node is visited by the existing
    `call` branch with dot_child). The gap here is that single-line bodies are
    not traversed at all (Behavior 1 root cause), so even captures that would
    otherwise work are missed.

    This test specifically tests a capture in a single-line body — distinct from
    a capture in a multi-line body (which already works).

    Verified RED: no edge for process when the capture is in a single-line body.
    """
    pytest.importorskip("tree_sitter_elixir")

    helper = tmp_path / "helper.ex"
    helper.write_text(
        'defmodule Helper do\n'
        '  def process(x) do\n'
        '    x\n'
        '  end\n'
        'end\n'
    )

    greeter = tmp_path / "greeter.ex"
    greeter.write_text(
        'defmodule Greeter do\n'
        '  def transform(list), do: Enum.map(list, &Helper.process/1)\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    # Greeter.transform must have an outbound edge to Helper.process
    transform_caller_edges = _src_func_edges(graph, "transform")
    assert transform_caller_edges, (
        "Expected at least one edge with src_func ending in 'transform' "
        "(Greeter.transform calling Helper.process via capture) for single-line def "
        "`def transform(list), do: Enum.map(list, &Helper.process/1)`. "
        f"All edges: {graph.edges!r}\n\n"
        "Root cause: The single-line body (keywords/pair) is not traversed by "
        "visit_all (same Behavior 1 gap). Once the body traversal fix is applied, "
        "the default child recursion will find the inner call(Helper.process) node "
        "from the &Helper.process/1 capture."
    )


# ---------------------------------------------------------------------------
# Behavior 6: Capture `&local_fn/arity` — local/intra edge emitted
# ---------------------------------------------------------------------------

def test_elixir_capture_local_fn_arity_emits_edge(tmp_path: Path):
    """Capture `&helper/1` where helper is defined in the same module must emit
    an intra-module edge from Processor.transform to Processor.helper.

    Current gap: `extract_body_calls.visit` has no `unary_operator` branch.
    For `&helper/1`, the AST is:
        unary_operator(&) → binary_operator(/) → [identifier(helper), integer(1)]
    There is no inner `call` node, so the existing `call` branch and default
    child recursion find only `identifier` and `integer` — neither produces an edge.

    Verified RED: no edge for helper appears.
    """
    pytest.importorskip("tree_sitter_elixir")

    proc = tmp_path / "processor.ex"
    proc.write_text(
        'defmodule Processor do\n'
        '  def transform(list) do\n'
        '    Enum.map(list, &helper/1)\n'
        '  end\n'
        '\n'
        '  def helper(x) do\n'
        '    x * 2\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    helper_edges = _callee_edges(graph, "helper")
    assert helper_edges, (
        "Expected at least one edge with callee ending in 'helper' for capture "
        "`&helper/1` where helper is defined in the same module. "
        f"All edges: {graph.edges!r}\n\n"
        "Root cause: `extract_body_calls.visit` in cross_file_calls.py has no "
        "`unary_operator` branch. `&helper/1` parses as: "
        "`unary_operator(&) → binary_operator(/) → [identifier(helper), integer(1)]`. "
        "There is no inner `call` node, so the existing `call` branch does not fire "
        "and no ('intra'|'local', 'helper') tuple is appended."
    )

    caller_is_transform = any(e[1].endswith(".transform") for e in helper_edges)
    assert caller_is_transform, (
        f"Expected caller ending in '.transform' for helper capture edge, "
        f"got: {helper_edges!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 7: Capture `&local_fn` (bare, no arity) — local/intra edge emitted
# ---------------------------------------------------------------------------

def test_elixir_capture_bare_local_fn_emits_edge(tmp_path: Path):
    """Capture `&helper` (no arity suffix) must also emit an edge.

    Architecture G2-4 adds the `unary_operator → identifier` (bare capture) form.
    This is distinct from the MFA form `&helper/1` (Behavior 6) which goes through
    `binary_operator`. The AST for `&helper` is:
        unary_operator(&) → identifier(helper)
    No `call` node, no `binary_operator` — only a bare identifier.

    Verified RED: no edge for helper appears.
    """
    pytest.importorskip("tree_sitter_elixir")

    proc = tmp_path / "processor.ex"
    proc.write_text(
        'defmodule Processor do\n'
        '  def transform(list) do\n'
        '    Enum.map(list, &helper)\n'
        '  end\n'
        '\n'
        '  def helper(x) do\n'
        '    x * 2\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    helper_edges = _callee_edges(graph, "helper")
    assert helper_edges, (
        "Expected at least one edge with callee ending in 'helper' for bare capture "
        "`&helper` (no arity). All edges: {graph.edges!r}\n\n"
        "Root cause: architecture G2-4 describes the `unary_operator → identifier` "
        "path for bare captures without `/arity`. The AST is: "
        "`unary_operator(&) → identifier(helper)`. "
        "The existing code has no `unary_operator` branch in `extract_body_calls.visit`, "
        "and the bare `identifier` node is not a `call` node, so no edge is emitted."
    )


# ---------------------------------------------------------------------------
# Behavior 8: Orphan function — appears in graph with caller_count=0
# ---------------------------------------------------------------------------

def test_elixir_orphan_function_has_caller_count_zero(tmp_path: Path):
    """A defined-but-never-called Elixir function must have caller_count=0 via sentinel.

    Current gap: `ELIXIR_ORPHAN_SENTINEL` does not exist. Without the orphan
    emission loop, orphan_fn never appears in any edge and impact_analysis
    returns {'error': "Function 'orphan_fn' not found in call graph"}.

    After the fix: impact_analysis returns {'targets': {..., 'caller_count': 0}}.

    Verified RED: impact_analysis returns an error dict.
    """
    pytest.importorskip("tree_sitter_elixir")

    orphan_file = tmp_path / "orphan.ex"
    orphan_file.write_text(
        'defmodule Orphan do\n'
        '  def orphan_fn do\n'
        '    :ok\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")
    result = impact_analysis(graph, "orphan_fn")

    assert "error" not in result, (
        "impact_analysis must NOT return an error for a defined Elixir function "
        "'orphan_fn' that has no callers. Got: "
        f"{result!r}\n\n"
        "Root cause: `ELIXIR_ORPHAN_SENTINEL` constant is missing from "
        "cross_file_calls.py. Without the orphan emission loop (mirroring the Ruby "
        "sentinel at lines 5374-5425), orphan_fn never appears in any graph edge "
        "and impact_analysis hits the 'not found' branch."
    )

    assert "targets" in result, (
        f"Expected 'targets' key in impact_analysis result for 'orphan_fn', "
        f"got: {result!r}"
    )

    all_caller_counts = [
        v.get("caller_count", None)
        for v in result.get("targets", {}).values()
    ]
    assert all_caller_counts, (
        f"Expected at least one target entry with caller_count, got: {result!r}"
    )
    assert all(c == 0 for c in all_caller_counts), (
        "orphan_fn has no callers; all caller_count values must be 0. "
        f"Got: {all_caller_counts!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 8 (supplemental): ELIXIR_ORPHAN_SENTINEL constant must exist
# ---------------------------------------------------------------------------

def test_elixir_orphan_sentinel_constant_exists():
    """ELIXIR_ORPHAN_SENTINEL must be importable from tldr.cross_file_calls.

    Current state: the constant does not exist — ImportError on import attempt.

    Verified RED: ImportError raised.
    """
    try:
        from tldr.cross_file_calls import ELIXIR_ORPHAN_SENTINEL  # noqa: F401
    except ImportError:
        pytest.fail(
            "ELIXIR_ORPHAN_SENTINEL is not defined in tldr/cross_file_calls.py. "
            "It must be added as a string constant parallel to RUBY_ORPHAN_SENTINEL."
        )

    assert ELIXIR_ORPHAN_SENTINEL == "__elixir_orphan__", (
        f"ELIXIR_ORPHAN_SENTINEL must equal '__elixir_orphan__', got: "
        f"{ELIXIR_ORPHAN_SENTINEL!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 9: Cross-module `Module.fn` call — callee defined as single-line def
# ---------------------------------------------------------------------------

def test_elixir_cross_module_single_line_callee_outbound_calls(tmp_path: Path):
    """A single-line `def hello(name), do: Logger.log(name)` must have its
    outbound call to Logger.log emitted in the graph.

    Note: App.run → Greeter.hello IS already emitted (callee resolution works for
    single-line defs, since _extract_elixir_func_name handles the `arguments → call`
    AST shape). The gap is Greeter.hello's OWN outbound body calls being dropped.

    This test verifies that once Greeter.hello is called by App.run, Greeter.hello
    itself also appears as a CALLER of Logger.log — proving the single-line body
    traversal works end-to-end for the cross-module chain:
        App.run → Greeter.hello → Logger.log

    Verified RED: No Greeter.hello → Logger.log edge exists (body not traversed).
    """
    pytest.importorskip("tree_sitter_elixir")

    logger = tmp_path / "logger.ex"
    logger.write_text(
        'defmodule Logger do\n'
        '  def log(msg) do\n'
        '    IO.puts(msg)\n'
        '  end\n'
        'end\n'
    )

    greeter = tmp_path / "greeter.ex"
    greeter.write_text(
        'defmodule Greeter do\n'
        '  def hello(name), do: Logger.log(name)\n'
        'end\n'
    )

    app = tmp_path / "app.ex"
    app.write_text(
        'defmodule App do\n'
        '  def run do\n'
        '    Greeter.hello("world")\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    # Greeter.hello must appear as a CALLER of Logger.log
    hello_caller_edges = _src_func_edges(graph, "hello")
    assert hello_caller_edges, (
        "Expected at least one edge with src_func ending in 'hello' "
        "(Greeter.hello calling Logger.log) for single-line def. "
        f"All edges: {graph.edges!r}\n\n"
        "Root cause: Greeter.hello's body (`do: Logger.log(name)`) is in "
        "`arguments → keywords → pair → value`, which visit_all never descends into. "
        "So calls_by_func['Greeter.hello'] stays empty and no outbound edge is emitted."
    )

    log_callee_edges = _callee_edges(graph, "log")
    assert log_callee_edges, (
        f"Expected an edge to Logger.log, but got: {graph.edges!r}"
    )

    hello_calls_log = any(
        e[1].endswith(".hello") and e[3].endswith(".log")
        for e in graph.edges
    )
    assert hello_calls_log, (
        "Expected Greeter.hello → Logger.log edge. "
        f"All edges: {graph.edges!r}"
    )


# ---------------------------------------------------------------------------
# Behavior 10: defp private function — single-line body outbound calls visible
# ---------------------------------------------------------------------------

def test_elixir_defp_single_line_body_calls_emitted(tmp_path: Path):
    """Single-line `defp private_helper(x), do: Formatter.format(x)` must emit
    an outbound edge from MyModule.private_helper to Formatter.format.

    Current gap: Same as Behavior 1 — single-line `defp` body is not traversed.
    Note: defp IS registered in `defined` (the existing code handles it); the gap
    is specifically the body traversal for outbound calls.

    Verified RED: No private_helper → Formatter.format edge exists.
    """
    pytest.importorskip("tree_sitter_elixir")

    formatter = tmp_path / "formatter.ex"
    formatter.write_text(
        'defmodule Formatter do\n'
        '  def format(x) do\n'
        '    to_string(x)\n'
        '  end\n'
        'end\n'
    )

    module_file = tmp_path / "my_module.ex"
    module_file.write_text(
        'defmodule MyModule do\n'
        '  def public_fn(x) do\n'
        '    private_helper(x)\n'
        '  end\n'
        '\n'
        '  defp private_helper(x), do: Formatter.format(x)\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    # private_helper must appear as a CALLER of Formatter.format
    helper_caller_edges = _src_func_edges(graph, "private_helper")
    assert helper_caller_edges, (
        "Expected at least one edge with src_func ending in 'private_helper' "
        "(MyModule.private_helper calling Formatter.format) for single-line defp. "
        f"All edges: {graph.edges!r}\n\n"
        "Root cause: `visit_all` in _extract_elixir_file_calls only looks for "
        "`do_block` children. Single-line `defp` bodies are in "
        "`arguments → keywords → pair → named_children[-1]` — never traversed, "
        "so calls_by_func['MyModule.private_helper'] stays empty."
    )

    format_callee_edges = _callee_edges(graph, "format")
    assert format_callee_edges, (
        f"Expected an edge to Formatter.format, but got: {graph.edges!r}"
    )


# ---------------------------------------------------------------------------
# CLI sentinel filter: _is_orphan_sentinel("__elixir_orphan__") returns True
# ---------------------------------------------------------------------------

def test_elixir_is_orphan_sentinel_returns_true_for_elixir_sentinel():
    """_is_orphan_sentinel must return True for '__elixir_orphan__'.

    Current state: _is_orphan_sentinel only checks RUBY_ORPHAN_SENTINEL.
    `_is_orphan_sentinel("__elixir_orphan__")` returns False.

    After the fix: cli.py imports ELIXIR_ORPHAN_SENTINEL and the function
    returns True for both sentinel values.

    Verified RED: returns False.
    """
    from tldr.cli import _is_orphan_sentinel

    result = _is_orphan_sentinel("__elixir_orphan__")
    assert result is True, (
        "_is_orphan_sentinel('__elixir_orphan__') must return True after "
        "ELIXIR_ORPHAN_SENTINEL is added to cross_file_calls.py and cli.py "
        "is updated to check both sentinels. Currently returns: "
        f"{result!r}\n\n"
        "Root cause: cli.py line 46 has `return func_name == _RUBY_ORPHAN_SENTINEL` "
        "which only checks the Ruby sentinel. The Elixir sentinel is not yet imported "
        "or checked."
    )
