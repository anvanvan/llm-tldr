"""RED-phase tests for 10 new call-graph languages.

Each test:
  1. Builds a 2-file tmp_path fixture (greeter.<ext> + app.<ext>).
  2. Calls build_project_call_graph(str(tmp_path), language="<lang>").
  3. Asserts a cross-file edge exists from app to greet.
  4. Asserts unused_fn has no callers (dead function).

All tests MUST FAIL on upstream/main (language elifs not yet in dispatcher).
They will pass once the implementation adds the dispatcher elif + builder.

Test naming: test_<lang>_call_graph_cross_file_edge_and_dead
"""

import pytest
from pathlib import Path
from tldr.cross_file_calls import build_project_call_graph


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _callee_edges(graph, callee_name: str) -> list:
    """Return all edges whose dst_func == callee_name."""
    return [e for e in graph.edges if e[3] == callee_name]


def _caller_edges_for(graph, src_file_stem: str) -> list:
    """Return all edges whose src_file contains src_file_stem."""
    return [e for e in graph.edges if src_file_stem in e[0]]


def _is_dead(graph, func_name: str) -> bool:
    """Return True if func_name appears as callee in no edges."""
    return all(e[3] != func_name for e in graph.edges)


# ---------------------------------------------------------------------------
# 1. C — ext_map only (builder already exists on upstream/main)
# The architecture specifies C needs ext_map additions at api.py:453 and :572.
# api.py:453 ext_map falls back to ".py" for unknown languages, so
# _get_module_exports(project, "greeter", "c") looks for "greeter.py" not "greeter.c"
# and raises ValueError (module not found).  After the fix it finds "greeter.c".
# ---------------------------------------------------------------------------

def test_c_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """C: _get_module_exports routes to .c (not .py) when language='c'.

    On upstream/main the api.py:453 ext_map has no 'c' key, so it falls back
    to '.py' and raises ValueError('Module not found').  After the fix it
    correctly opens 'greeter.c' and returns functions from it.
    """
    pytest.importorskip("tree_sitter_c")

    # Create a .c file that _get_module_exports should find
    greeter_c = tmp_path / "greeter.c"
    greeter_c.write_text(
        '#include <stdio.h>\n'
        'void greet(void) { printf("hello\\n"); }\n'
        'void unused_fn(void) { }\n'
    )

    # Import the private helper (it is reachable from the package)
    from tldr.api import _get_module_exports

    # On upstream/main: ext_map has no "c" key → falls back to ".py" →
    # looks for greeter.py → raises ValueError("Module not found")
    # After fix: ext_map["c"] == ".c" → finds greeter.c → returns context.
    try:
        result = _get_module_exports(tmp_path, "greeter", language="c")
    except ValueError as exc:
        pytest.fail(
            f"_get_module_exports did not find greeter.c for language='c'. "
            f"ext_map at api.py:453 is missing the 'c' key. Error: {exc}"
        )

    # (a) result must contain the greet function
    func_names = [f.name for f in result.functions]
    assert "greet" in func_names, (
        f"Expected 'greet' in functions returned from greeter.c, got: {func_names}"
    )

    # (b) unused_fn must also be present (it is defined in greeter.c)
    assert "unused_fn" in func_names, (
        f"Expected 'unused_fn' in functions returned from greeter.c, got: {func_names}"
    )


# ---------------------------------------------------------------------------
# 2. Elixir — dispatcher elif + ported builder
# ---------------------------------------------------------------------------

def test_elixir_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Elixir: cross-file edge app.ex -> greeter.ex for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_elixir")

    greeter = tmp_path / "greeter.ex"
    greeter.write_text(
        'defmodule Greeter do\n'
        '  def greet do\n'
        '    IO.puts("hello")\n'
        '  end\n'
        '\n'
        '  def unused_fn do\n'
        '    :ok\n'
        '  end\n'
        'end\n'
    )

    app = tmp_path / "app.ex"
    app.write_text(
        'defmodule App do\n'
        '  def run do\n'
        '    Greeter.greet()\n'
        '  end\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="elixir")

    # Elixir emits qualified callees (Module.func), so match by suffix.
    greet_edges = [e for e in graph.edges if e[3] == "greet" or e[3].endswith(".greet")]
    assert greet_edges, (
        "Expected at least one edge with callee='greet' (or Module.greet) for elixir. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.ex" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.ex for greet, got: {greet_edges}"
    )

    assert _is_dead(graph, "unused_fn"), (
        f"unused_fn should have no callers in elixir graph, "
        f"but found: {_callee_edges(graph, 'unused_fn')}"
    )


# ---------------------------------------------------------------------------
# 3. Swift — dispatcher elif + ported builder
# ---------------------------------------------------------------------------

def test_swift_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Swift: cross-file edge app.swift -> greeter.swift for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_swift")

    greeter = tmp_path / "greeter.swift"
    greeter.write_text(
        'func greet() {\n'
        '    print("hello")\n'
        '}\n'
        '\n'
        'func unused_fn() {\n'
        '    // never called\n'
        '}\n'
    )

    app = tmp_path / "app.swift"
    app.write_text(
        'func main_func() {\n'
        '    greet()\n'
        '}\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="swift")

    greet_edges = _callee_edges(graph, "greet")
    assert greet_edges, (
        "Expected at least one edge with callee='greet' for swift. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.swift" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.swift for greet, got: {greet_edges}"
    )

    assert _is_dead(graph, "unused_fn"), (
        f"unused_fn should have no callers in swift graph, "
        f"but found: {_callee_edges(graph, 'unused_fn')}"
    )


# ---------------------------------------------------------------------------
# 4. Ruby — new builder
# ---------------------------------------------------------------------------

def test_ruby_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Ruby: cross-file edge app.rb -> greeter.rb for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_ruby")

    greeter = tmp_path / "greeter.rb"
    greeter.write_text(
        'def greet\n'
        '  puts "hello"\n'
        'end\n'
        '\n'
        'def unused_fn\n'
        '  nil\n'
        'end\n'
    )

    app = tmp_path / "app.rb"
    app.write_text(
        "require_relative 'greeter'\n"
        '\n'
        'def main_func\n'
        '  greet\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="ruby")

    greet_edges = _callee_edges(graph, "greet")
    assert greet_edges, (
        "Expected at least one edge with callee='greet' for ruby. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.rb" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.rb for greet, got: {greet_edges}"
    )

    assert _is_dead(graph, "unused_fn"), (
        f"unused_fn should have no callers in ruby graph, "
        f"but found: {_callee_edges(graph, 'unused_fn')}"
    )


# ---------------------------------------------------------------------------
# 5. Kotlin — new builder
# ---------------------------------------------------------------------------

def test_kotlin_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Kotlin: cross-file edge app.kt -> greeter.kt for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_kotlin")

    greeter = tmp_path / "greeter.kt"
    greeter.write_text(
        'fun greet() {\n'
        '    println("hello")\n'
        '}\n'
        '\n'
        'fun unused_fn() {\n'
        '    // never called\n'
        '}\n'
    )

    app = tmp_path / "app.kt"
    app.write_text(
        'fun main_func() {\n'
        '    greet()\n'
        '}\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="kotlin")

    greet_edges = _callee_edges(graph, "greet")
    assert greet_edges, (
        "Expected at least one edge with callee='greet' for kotlin. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.kt" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.kt for greet, got: {greet_edges}"
    )

    assert _is_dead(graph, "unused_fn"), (
        f"unused_fn should have no callers in kotlin graph, "
        f"but found: {_callee_edges(graph, 'unused_fn')}"
    )


# ---------------------------------------------------------------------------
# 6. C# (csharp) — new builder
# ---------------------------------------------------------------------------

def test_csharp_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """C#: cross-file edge app.cs -> greeter.cs for Greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_c_sharp")

    greeter = tmp_path / "greeter.cs"
    greeter.write_text(
        'public class Greeter {\n'
        '    public static void Greet() {\n'
        '        System.Console.WriteLine("hello");\n'
        '    }\n'
        '    public static void UnusedFn() {\n'
        '        // never called\n'
        '    }\n'
        '}\n'
    )

    app = tmp_path / "app.cs"
    app.write_text(
        'using System;\n'
        '\n'
        'public class App {\n'
        '    public static void MainFunc() {\n'
        '        Greeter.Greet();\n'
        '    }\n'
        '}\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="csharp")

    # C# method names may be stored as-is or lowercased; try both 'Greet' and 'greet'
    greet_edges = _callee_edges(graph, "Greet") or _callee_edges(graph, "greet")
    assert greet_edges, (
        "Expected at least one edge with callee='Greet' (or 'greet') for csharp. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.cs" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.cs for Greet, got: {greet_edges}"
    )

    unused_dead = _is_dead(graph, "UnusedFn") and _is_dead(graph, "unused_fn")
    assert unused_dead, (
        f"UnusedFn should have no callers in csharp graph, "
        f"but found callee edges: {[e for e in graph.edges if 'nused' in e[3].lower()]}"
    )


# ---------------------------------------------------------------------------
# 7. Lua — new builder
# ---------------------------------------------------------------------------

def test_lua_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Lua: cross-file edge app.lua -> greeter.lua for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_lua")

    greeter = tmp_path / "greeter.lua"
    greeter.write_text(
        'local M = {}\n'
        '\n'
        'function M.greet()\n'
        '    print("hello")\n'
        'end\n'
        '\n'
        'function M.unused_fn()\n'
        '    -- never called\n'
        'end\n'
        '\n'
        'return M\n'
    )

    app = tmp_path / "app.lua"
    app.write_text(
        "local greeter = require('greeter')\n"
        '\n'
        'local function main_func()\n'
        '    greeter.greet()\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="lua")

    # greet may be stored as 'greet' or 'M.greet' depending on implementation
    greet_edges = (
        _callee_edges(graph, "greet")
        or _callee_edges(graph, "M.greet")
    )
    assert greet_edges, (
        "Expected at least one edge with callee='greet' (or 'M.greet') for lua. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.lua" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.lua for greet, got: {greet_edges}"
    )

    unused_dead = _is_dead(graph, "unused_fn") and _is_dead(graph, "M.unused_fn")
    assert unused_dead, (
        f"unused_fn / M.unused_fn should have no callers in lua graph, "
        f"but found: {[e for e in graph.edges if 'unused' in e[3]]}"
    )


# ---------------------------------------------------------------------------
# 8. Luau — new builder (duplicate of Lua with .luau ext)
# ---------------------------------------------------------------------------

def test_luau_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Luau: cross-file edge app.luau -> greeter.luau for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_luau")

    greeter = tmp_path / "greeter.luau"
    greeter.write_text(
        'local M = {}\n'
        '\n'
        'function M.greet(): ()\n'
        '    print("hello")\n'
        'end\n'
        '\n'
        'function M.unused_fn(): ()\n'
        '    -- never called\n'
        'end\n'
        '\n'
        'return M\n'
    )

    app = tmp_path / "app.luau"
    app.write_text(
        "local greeter = require('greeter')\n"
        '\n'
        'local function main_func(): ()\n'
        '    greeter.greet()\n'
        'end\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="luau")

    greet_edges = (
        _callee_edges(graph, "greet")
        or _callee_edges(graph, "M.greet")
    )
    assert greet_edges, (
        "Expected at least one edge with callee='greet' (or 'M.greet') for luau. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.luau" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.luau for greet, got: {greet_edges}"
    )

    unused_dead = _is_dead(graph, "unused_fn") and _is_dead(graph, "M.unused_fn")
    assert unused_dead, (
        f"unused_fn / M.unused_fn should have no callers in luau graph, "
        f"but found: {[e for e in graph.edges if 'unused' in e[3]]}"
    )


# ---------------------------------------------------------------------------
# 9. Scala — new builder
# ---------------------------------------------------------------------------

def test_scala_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """Scala: cross-file edge app.scala -> greeter.scala for greet(); unused_fn dead."""
    pytest.importorskip("tree_sitter_scala")

    greeter = tmp_path / "greeter.scala"
    greeter.write_text(
        'object Greeter {\n'
        '  def greet(): Unit = {\n'
        '    println("hello")\n'
        '  }\n'
        '\n'
        '  def unused_fn(): Unit = {\n'
        '    // never called\n'
        '  }\n'
        '}\n'
    )

    app = tmp_path / "app.scala"
    app.write_text(
        'object App {\n'
        '  def main_func(): Unit = {\n'
        '    Greeter.greet()\n'
        '  }\n'
        '}\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="scala")

    # greet may be stored as 'greet' or 'Greeter.greet'
    greet_edges = (
        _callee_edges(graph, "greet")
        or _callee_edges(graph, "Greeter.greet")
    )
    assert greet_edges, (
        "Expected at least one edge with callee='greet' (or 'Greeter.greet') for scala. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.scala" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.scala for greet, got: {greet_edges}"
    )

    unused_dead = (
        _is_dead(graph, "unused_fn")
        and _is_dead(graph, "Greeter.unused_fn")
    )
    assert unused_dead, (
        f"unused_fn should have no callers in scala graph, "
        f"but found: {[e for e in graph.edges if 'unused' in e[3]]}"
    )


# ---------------------------------------------------------------------------
# 10. C++ (cpp) — new builder with namespace canonicalization (T-6)
# ---------------------------------------------------------------------------

def test_cpp_call_graph_cross_file_edge_and_dead(tmp_path: Path):
    """C++: cross-file edge app.cpp -> greeter.cpp for greet(); unused_fn dead.

    Also verifies namespace-qualified call ns::greet is resolved (T-6 fix):
    both bare 'greet' and dot-form 'ns.greet' must be checked as callee.
    """
    pytest.importorskip("tree_sitter_cpp")

    # greeter.hpp — declaration header
    header = tmp_path / "greeter.hpp"
    header.write_text(
        '#pragma once\n'
        'namespace ns {\n'
        '    void greet();\n'
        '    void unused_fn();\n'
        '}\n'
    )

    # greeter.cpp — definitions
    greeter = tmp_path / "greeter.cpp"
    greeter.write_text(
        '#include "greeter.hpp"\n'
        '#include <iostream>\n'
        'namespace ns {\n'
        '    void greet() {\n'
        '        std::cout << "hello" << std::endl;\n'
        '    }\n'
        '    void unused_fn() {\n'
        '        // never called\n'
        '    }\n'
        '}\n'
    )

    # app.cpp — caller
    app = tmp_path / "app.cpp"
    app.write_text(
        '#include "greeter.hpp"\n'
        'void main_func() {\n'
        '    ns::greet();\n'
        '}\n'
    )

    graph = build_project_call_graph(str(tmp_path), language="cpp")

    # T-6: callee may be 'greet' (bare) or 'ns.greet' (dot-canonical)
    greet_edges = (
        _callee_edges(graph, "greet")
        or _callee_edges(graph, "ns.greet")
        or _callee_edges(graph, "ns::greet")
    )
    assert greet_edges, (
        "Expected at least one edge with callee='greet' / 'ns.greet' for cpp. "
        f"All edges: {graph.edges}"
    )

    caller_in_app = any("app.cpp" in e[0] for e in greet_edges)
    assert caller_in_app, (
        f"Expected caller in app.cpp for greet/ns.greet, got: {greet_edges}"
    )

    unused_dead = (
        _is_dead(graph, "unused_fn")
        and _is_dead(graph, "ns.unused_fn")
        and _is_dead(graph, "ns::unused_fn")
    )
    assert unused_dead, (
        f"unused_fn should have no callers in cpp graph, "
        f"but found: {[e for e in graph.edges if 'unused' in e[3]]}"
    )
