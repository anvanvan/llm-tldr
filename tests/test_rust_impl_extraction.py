"""Regression coverage for Rust impl-block method extraction.

Verifies that `HybridExtractor._extract_rust_impl` correctly:
  (a) stops name rewriting (no "(Type) name" mangling),
  (b) registers methods under both bare name AND dot-qualified "Type.name",
  (c) parses "impl Trait for Type" using child_by_field_name and uses the
      implementing type only (not the trait).

Run with:
    pytest tests/test_rust_impl_extraction.py -v
"""

from pathlib import Path

from tldr.hybrid_extractor import HybridExtractor

# Fixtures directory for Rust impl test sources.
_FIXTURES = Path(__file__).parent / "fixtures" / "rust_impl"


def _func_names(module_info) -> list[str]:
    """Return all function names from a ModuleInfo."""
    return [f.name for f in module_info.functions]


# ---------------------------------------------------------------------------
# 1. Inherent impl: bare name registered
# ---------------------------------------------------------------------------

def test_inherent_impl_bare_name_in_functions():
    """impl DaemonState { fn update_all } -> bare 'update_all' in functions.

    After the fix, _extract_rust_impl appends a FunctionInfo with name='update_all'
    (no parentheses, no type prefix).  Today it only registers '(DaemonState) update_all',
    so this test MUST FAIL on unfixed code.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "inherent_impl.rs")

    names = _func_names(result)
    assert "update_all" in names, (
        f"Expected bare 'update_all' in functions; got: {names}"
    )


# ---------------------------------------------------------------------------
# 2. Inherent impl: dot-qualified name registered
# ---------------------------------------------------------------------------

def test_inherent_impl_dot_qualified_name_in_functions():
    """impl DaemonState { fn update_all } -> 'DaemonState.update_all' in functions.

    After the fix, _extract_rust_impl also appends a dot-qualified copy so that
    resolve_func_name('DaemonState::update_all') normalises '::' -> '.' and hits
    the key directly.  Today no such key exists, so this test MUST FAIL.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "inherent_impl.rs")

    names = _func_names(result)
    assert "DaemonState.update_all" in names, (
        f"Expected 'DaemonState.update_all' in functions; got: {names}"
    )


# ---------------------------------------------------------------------------
# 3. Old mangled name is absent
# ---------------------------------------------------------------------------

def test_no_mangled_parenthesised_names_in_functions():
    """No FunctionInfo name should start with '(' after the fix.

    Today the code produces '(DaemonState) update_all'.  After the fix this
    mangled form must be gone entirely.  Currently the mangled form IS present,
    so this test MUST FAIL on unfixed code.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "inherent_impl.rs")

    names = _func_names(result)
    mangled = [n for n in names if n.startswith("(")]
    assert mangled == [], (
        f"Expected no parenthesised (mangled) names; found: {mangled}"
    )


# ---------------------------------------------------------------------------
# 4. Trait impl uses implementing type, not trait name
# ---------------------------------------------------------------------------

def test_trait_impl_registers_implementing_type_not_trait():
    """impl Display for ProtocolMismatch { fn fmt } -> 'ProtocolMismatch.fmt' present,
    'Display.fmt' absent, and bare 'fmt' present.

    After the fix, _parse_rust_impl_type uses child_by_field_name('type') for the
    qualified name (ProtocolMismatch) not the trait (Display).  Today the method is
    registered under a mangled form, and certainly not under 'ProtocolMismatch.fmt',
    so this test MUST FAIL.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "trait_impl.rs")

    names = _func_names(result)

    assert "fmt" in names, (
        f"Expected bare 'fmt' in functions; got: {names}"
    )
    assert "ProtocolMismatch.fmt" in names, (
        f"Expected 'ProtocolMismatch.fmt' in functions; got: {names}"
    )
    assert "Display.fmt" not in names, (
        f"'Display.fmt' should NOT be in functions; got: {names}"
    )
    # Also verify no trait-qualified mangled form
    trait_variants = [n for n in names if "Display" in n]
    assert trait_variants == [], (
        f"No name referencing 'Display' should appear; found: {trait_variants}"
    )


# ---------------------------------------------------------------------------
# 5. Multiple methods in one impl block: 4 FunctionInfos
# ---------------------------------------------------------------------------

def test_multi_method_impl_produces_bare_and_qualified_for_each():
    """impl Counter { fn first; fn second } -> 4 FunctionInfos total.

    Expected: bare 'first', 'Counter.first', bare 'second', 'Counter.second'.
    Today only two mangled names are produced, so this test MUST FAIL.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "multi_method_impl.rs")

    names = _func_names(result)

    assert "first" in names, f"Expected bare 'first'; got: {names}"
    assert "Counter.first" in names, f"Expected 'Counter.first'; got: {names}"
    assert "second" in names, f"Expected bare 'second'; got: {names}"
    assert "Counter.second" in names, f"Expected 'Counter.second'; got: {names}"


# ---------------------------------------------------------------------------
# 6. Nested impl inside mod: bare + qualified names present
# ---------------------------------------------------------------------------

def test_nested_mod_impl_registers_bare_and_qualified():
    """mod foo { impl Bar { fn baz } } -> 'baz' and 'Bar.baz' both registered.

    The fix must handle impl blocks found anywhere in the AST (including inside
    mod blocks).  Today only a mangled form would appear.  MUST FAIL on unfixed code.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "nested_mod_impl.rs")

    names = _func_names(result)

    assert "baz" in names, (
        f"Expected bare 'baz' from nested mod impl; got: {names}"
    )
    assert "Bar.baz" in names, (
        f"Expected 'Bar.baz' from nested mod impl; got: {names}"
    )


# ---------------------------------------------------------------------------
# 7. Free function outside impl: no regression
# ---------------------------------------------------------------------------

def test_free_function_outside_impl_unchanged():
    """A plain 'fn standalone()' in the same file must still appear under bare name.

    This is a regression guard: the fix must not break free-function extraction.
    The test will PASS on current code (standalone is correctly extracted today)
    but it verifies no regression once impl handling is fixed.

    NOTE: per RED-phase rules this test is included as a regression guard.
    If it passes today, that is expected — it protects against over-correction.
    """
    extractor = HybridExtractor()
    result = extractor.extract(_FIXTURES / "inherent_impl.rs")

    names = _func_names(result)
    assert "standalone" in names, (
        f"Expected free function 'standalone' in functions; got: {names}"
    )
