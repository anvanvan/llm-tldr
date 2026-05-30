"""Regression coverage for TypeScript export-const / export-default function indexing.

Verifies that `HybridExtractor._extract_ts_function` and `_extract_ts_nodes` correctly handle:
- `export const x = () => {}` and `export const x = function () {}` via variable_declarator name recovery
- `export default function App()` with both "App" and "default" aliases
- `export default () => {}` and `export default function () {}` with "default" alias
- Intra-file call detection for arrow-function-in-const callees via `_collect_ts_definitions` recursion

Run with:
    pytest tests/test_typescript_exports.py -v
"""

from pathlib import Path

import pytest
from tldr.hybrid_extractor import HybridExtractor


FIXTURES = Path(__file__).parent / "fixtures" / "typescript_exports"


@pytest.fixture
def extractor():
    return HybridExtractor()


# ---------------------------------------------------------------------------
# Test 1: export const x = () => {} (arrow function)
# ---------------------------------------------------------------------------

def test_export_const_arrow_function_is_extracted(extractor):
    """export const useEventFormSubmission = (form, opts) => {} must be in functions list.

    Currently FAILS: arrow_function node has no identifier child, _extract_ts_function
    returns None, function silently dropped.
    """
    result = extractor.extract(FIXTURES / "ts_export_arrow.ts")
    func_names = [f.name for f in result.functions]
    assert "useEventFormSubmission" in func_names, (
        f"Expected 'useEventFormSubmission' in functions, got: {func_names}"
    )


def test_export_const_arrow_function_has_correct_params(extractor):
    """useEventFormSubmission must expose params ['form', 'opts'] (after stripping types).

    Currently FAILS: function is dropped entirely, params cannot be checked.
    """
    result = extractor.extract(FIXTURES / "ts_export_arrow.ts")
    func = next((f for f in result.functions if f.name == "useEventFormSubmission"), None)
    assert func is not None, "useEventFormSubmission not found in functions list"
    # Params may include type annotations but the bare names must appear
    params_text = " ".join(func.params)
    assert "form" in params_text, f"Expected 'form' in params, got: {func.params}"
    assert "opts" in params_text, f"Expected 'opts' in params, got: {func.params}"


# ---------------------------------------------------------------------------
# Test 2: export const x = function () {} (function_expression)
# ---------------------------------------------------------------------------

def test_export_const_function_expression_is_extracted(extractor):
    """export const enumerateCachedWindowKeys = function () {} must be in functions list.

    Currently FAILS: function_expression node has no identifier child,
    _extract_ts_function returns None, function silently dropped.
    """
    result = extractor.extract(FIXTURES / "ts_export_arrow.ts")
    func_names = [f.name for f in result.functions]
    assert "enumerateCachedWindowKeys" in func_names, (
        f"Expected 'enumerateCachedWindowKeys' in functions, got: {func_names}"
    )


# ---------------------------------------------------------------------------
# Test 3: export default function App() {} → both "App" and "default" in functions
# ---------------------------------------------------------------------------

def test_export_default_named_function_emits_both_names(extractor):
    """export default function App() must produce BOTH FunctionInfo(name='App') and
    FunctionInfo(name='default').

    Currently FAILS: export_statement is in the catch-all recurse list,
    no post-recursion alias injection happens, so 'default' is never emitted.
    """
    result = extractor.extract(FIXTURES / "ts_export_default.ts")
    func_names = [f.name for f in result.functions]
    assert "App" in func_names, (
        f"Expected 'App' in functions for named default export, got: {func_names}"
    )
    assert "default" in func_names, (
        f"Expected 'default' alias in functions for named default export, got: {func_names}"
    )


def test_export_default_alias_has_same_params_as_named(extractor):
    """The 'default' alias must carry the same params as the named function 'App'.

    Currently FAILS: 'default' alias is not emitted at all.
    """
    result = extractor.extract(FIXTURES / "ts_export_default.ts")
    app = next((f for f in result.functions if f.name == "App"), None)
    default = next((f for f in result.functions if f.name == "default"), None)
    assert default is not None, "FunctionInfo(name='default') not found"
    assert app is not None, "FunctionInfo(name='App') not found"
    assert default.params == app.params, (
        f"'default' alias params {default.params!r} differ from 'App' params {app.params!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: export default function () {} (anonymous) → FunctionInfo(name="default")
# ---------------------------------------------------------------------------

def test_export_default_anonymous_function_emits_default(extractor):
    """export default function () {} (anonymous) must produce FunctionInfo(name='default').

    Currently FAILS: anonymous function_declaration has no identifier, _extract_ts_function
    returns None; no special default path exists.
    """
    result = extractor.extract(FIXTURES / "ts_export_default_anon_fn.ts")
    func_names = [f.name for f in result.functions]
    assert "default" in func_names, (
        f"Expected FunctionInfo(name='default') for anonymous default function, got: {func_names}"
    )


# ---------------------------------------------------------------------------
# Test 5: export default () => {} (anonymous arrow) → FunctionInfo(name="default")
# ---------------------------------------------------------------------------

def test_export_default_anonymous_arrow_emits_default(extractor):
    """export default () => {} must produce FunctionInfo(name='default').

    Currently FAILS: anonymous arrow_function has no identifier and no parent
    variable_declarator; _extract_ts_function returns None; no special default path exists.
    """
    result = extractor.extract(FIXTURES / "ts_export_default_anon_arrow.ts")
    func_names = [f.name for f in result.functions]
    assert "default" in func_names, (
        f"Expected FunctionInfo(name='default') for anonymous default arrow, got: {func_names}"
    )


# ---------------------------------------------------------------------------
# Test 6: object-literal arrow boundary guard
# ---------------------------------------------------------------------------

def test_object_literal_arrow_uses_property_name_not_variable_name(extractor):
    """const handler = { onClick: () => {} } must produce FunctionInfo(name='onClick'),
    NOT FunctionInfo(name='handler').

    'onClick' is expected from _get_pair_property_name (existing logic).
    'handler' must NOT appear — the new _get_variable_declarator_name must stop at
    the 'pair' boundary before climbing to the outer variable_declarator.

    Currently FAILS on the 'handler' absence assertion: the boundary set is not yet
    implemented, so the parent walk would return 'handler' (clobbering 'onClick').
    Note: 'onClick' may also currently be missing since the arrow is dropped — the
    test asserts both conditions.
    """
    result = extractor.extract(FIXTURES / "ts_object_literal_boundary.ts")
    func_names = [f.name for f in result.functions]
    assert "handler" not in func_names, (
        f"'handler' (outer variable name) must NOT appear in functions list; got: {func_names}"
    )
    assert "onClick" in func_names, (
        f"'onClick' (property name) must appear in functions list; got: {func_names}"
    )


# ---------------------------------------------------------------------------
# Test 7: intra-file call edge — arrow-in-const callee detected
# ---------------------------------------------------------------------------

def test_intra_file_arrow_const_call_edge_is_detected(extractor):
    """const outer = () => { inner(); } must produce a call_graph edge outer → inner,
    where inner is defined as const inner = () => {}.

    Currently FAILS: _collect_ts_definitions only handles function_declaration and
    class_declaration; it does not handle lexical_declaration, so 'inner' is never
    added to defined_names; _extract_ts_calls filters out callee 'inner' because it
    is not in defined_names; no edge is emitted.
    """
    result = extractor.extract(FIXTURES / "ts_intra_file_calls.ts")
    calls = result.call_graph.calls
    assert "outer" in calls, (
        f"Expected 'outer' as caller in call_graph, got keys: {list(calls.keys())}"
    )
    assert "inner" in calls.get("outer", []), (
        f"Expected edge outer → inner in call_graph, got outer calls: {calls.get('outer', [])}"
    )


# ---------------------------------------------------------------------------
# Test 8 (bonus): plain function_declaration — no regression
# ---------------------------------------------------------------------------

def test_plain_function_declaration_still_extracted(extractor):
    """function foo(a, b) {} must still produce FunctionInfo(name='foo').

    Regression guard: the changes to _extract_ts_function must not break the existing
    identifier-child scan path for plain function_declaration nodes.

    This test should PASS even before the fix (it tests current working behavior),
    but is included to catch accidental regression during implementation.
    """
    result = extractor.extract(FIXTURES / "ts_plain_function.ts")
    func_names = [f.name for f in result.functions]
    assert "foo" in func_names, (
        f"Expected 'foo' in functions (regression guard), got: {func_names}"
    )
