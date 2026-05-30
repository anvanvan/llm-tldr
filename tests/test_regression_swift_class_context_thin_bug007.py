# pyright: reportMissingImports=false
"""Regression test — Bug 007 (continued): `tldr context <ClassName> --lang swift`
returned class-shell-only results (no methods enumerated, no callers list) even
though `tldr extract` could see the class's methods.

Root cause:
    `_build_swift_call_graph` in `tldr/cross_file_calls.py` indexed Swift
    function names but never associated methods with their enclosing class.
    The class symbol was therefore an isolated node in the call graph:
      * adjacency[ClassName] = []  →  no callees / methods to walk
      * reverse_adjacency[ClassName] = []  →  no callers (Swift call sites
        use receiver-values like `self` or `model`, not the type name, so
        the receiver pattern alone cannot recover class membership)

Fix:
    `_extract_swift_file_calls` now also returns `methods_by_class`, and
    `_build_swift_call_graph` emits synthetic `(ClassName → method_name)`
    edges. The BFS in `get_relevant_context(ClassName)` then enumerates the
    class's methods at depth ≥ 1, and the existing reverse-adjacency pass
    surfaces cross-file callers of those methods.

Contract pinned here (GREEN after fix):
    For a Swift class with methods AND a call site in a separate file,
    `tldr context <ClassName> --lang swift --depth 2` must return:
      * the class itself as a FunctionContext,
      * ≥ 1 of its methods (qualified as `ClassName.method`), AND
      * ≥ 1 caller of those methods (i.e. a FunctionContext whose `calls`
        list mentions a method name OR whose file lives outside the class
        file).

Skipped if `tree_sitter_swift` is not installed (the call-graph builder
silently degrades to empty results in that case — bug 007 part 1, covered
by `test_regression_swift_context_thin_bug007.py`).
"""

import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Two-file Swift fixture: a class with methods + a separate caller file
# ---------------------------------------------------------------------------

_STORE_SWIFT = textwrap.dedent("""\
    import Foundation

    public final class VocabStore {
        public var terms: [String] = []

        public func recordUsedTerms(_ newTerms: [String]) {
            terms.append(contentsOf: newTerms)
        }

        public func loadIfNeeded() {
            if terms.isEmpty {
                terms = ["hello", "world"]
            }
        }

        public func clear() {
            terms = []
        }
    }
""")

_APP_SWIFT = textwrap.dedent("""\
    import Foundation

    public class App {
        let store = VocabStore()

        public func run() {
            store.loadIfNeeded()
            store.recordUsedTerms(["a", "b"])
        }

        public func reset() {
            store.clear()
        }
    }
""")


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------

# Probe tree_sitter_swift availability at collection time so we can skip
# cleanly when the optional dep is missing (CI without swift grammar etc.).
try:  # pragma: no cover - environment probe
    import tree_sitter_swift  # noqa: F401
    _SWIFT_AVAILABLE = True
except ImportError:
    _SWIFT_AVAILABLE = False


@pytest.mark.skipif(
    not _SWIFT_AVAILABLE,
    reason="tree_sitter_swift not installed; class-context fix requires it",
)
class TestSwiftClassContextEnumeratesMethodsAndCallersBug007:
    """`tldr context <ClassName> --lang swift` must enumerate the class's
    methods AND surface at least one caller. Class-shell-only results are a
    regression of bug 007's class-resolution fix."""

    def test_swift_class_context_lists_methods_and_callers(
        self, tmp_path: Path
    ):
        # Build a minimal two-file Swift project in tmp_path.
        src_dir = tmp_path / "Sources" / "App"
        src_dir.mkdir(parents=True)
        (src_dir / "VocabStore.swift").write_text(_STORE_SWIFT)
        (src_dir / "App.swift").write_text(_APP_SWIFT)

        from tldr.api import get_relevant_context

        ctx = get_relevant_context(
            project=str(tmp_path),
            entry_point="VocabStore",
            depth=2,
            language="swift",
        )

        # Sanity: no error path.
        assert ctx.error is None, f"unexpected error: {ctx.error!r}"

        # Collect names for diagnostics.
        names = [fc.name for fc in ctx.functions]
        diag = f"functions returned ({len(names)}):\n  " + "\n  ".join(names)

        # The class itself must be in the result.
        assert any(fc.name == "VocabStore" for fc in ctx.functions), (
            f"VocabStore class missing from result.\n{diag}"
        )

        # The class result must list its methods either as direct entries
        # (qualified `VocabStore.method`) OR via the class node's `calls`
        # adjacency. We pin the *result* shape — both reflect successful
        # method enumeration.
        method_entries = [
            fc for fc in ctx.functions
            if fc.name.startswith("VocabStore.")
        ]
        class_entry = next(
            (fc for fc in ctx.functions if fc.name == "VocabStore"), None
        )
        class_calls = list(class_entry.calls) if class_entry else []

        # Methods of the fixture class.
        expected_methods = {"recordUsedTerms", "loadIfNeeded", "clear"}
        seen_methods = {
            fc.name.split(".", 1)[1] for fc in method_entries
        } | set(class_calls)

        assert expected_methods & seen_methods, (
            "no methods of VocabStore were enumerated.\n"
            f"expected at least one of: {sorted(expected_methods)}\n"
            f"saw method_entries: {[fc.name for fc in method_entries]}\n"
            f"class_calls list: {class_calls}\n{diag}"
        )

        # At least one caller must be surfaced (from App.swift). A "caller"
        # is any FunctionContext whose `calls` list mentions one of the
        # class's methods.
        caller_methods = expected_methods
        callers = [
            fc for fc in ctx.functions
            if any(call in caller_methods for call in (fc.calls or []))
            and fc.name != "VocabStore"
            and not fc.name.startswith("VocabStore.")
        ]
        assert callers, (
            "no caller of VocabStore's methods was surfaced.\n"
            f"expected ≥ 1 FunctionContext whose `calls` mentions one of "
            f"{sorted(caller_methods)}.\n{diag}"
        )
