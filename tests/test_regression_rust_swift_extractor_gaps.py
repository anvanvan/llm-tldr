"""Regression tests for two distinct extractor gaps (RED phase).

GAP 1 (Rust) — ``hybrid_extractor.py`` ``_extract_rust_impl`` appends impl-block
methods only to ``module_info.functions[]``, never to the owning
``ClassInfo.methods[]``.  As a result ``api.extract_file_with_code(method='Type.x')``
returns an empty ``methods`` list and no ``code`` field.  Secondary sub-gaps:

  * No ``enum_item`` branch in ``_extract_rust_nodes`` → enums produce no ClassInfo
    and their impl methods have no class to link to.
  * impl-before-struct forward refs are not linked by a post-walk linker → if the
    ``impl Foo`` block appears textually BEFORE ``struct Foo`` the ClassInfo still
    ends up with ``methods == []``.

GAP 2 (Swift) — tree-sitter-swift emits an ERROR node wrapping the
``CorrectionPreviewModel`` class declaration, so the class is silently dropped
from extraction results (only its sibling classes extract successfully).
``--method CorrectionPreviewModel.<method>`` returns bare ``{file_path, language}``
with no ``classes`` key.

All tests in this file MUST FAIL on the current (unfixed) code.  They will pass
once the extractor is fixed.

Run:
    python -m pytest tests/test_regression_rust_swift_extractor_gaps.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldr.api import extract_file_with_code
from tldr.hybrid_extractor import HybridExtractor

# ---------------------------------------------------------------------------
# Shared fixture path helpers
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures" / "rust_impl"


# ===========================================================================
# GAP 1 — Rust impl-block methods not linked into ClassInfo.methods[]
# ===========================================================================


class TestRustImplMethodsLinkedToClassInfo:
    """Suite that verifies impl-block methods are associated with the owning
    ClassInfo.  All assertions target ``classes[].methods[]`` — the exact array
    that ``extract_file_with_code`` reads — NOT ``functions[]`` (which the
    existing test suite already covers and which is currently correctly
    populated)."""

    # -----------------------------------------------------------------------
    # R-1  inherent impl → ClassInfo.methods populated
    # -----------------------------------------------------------------------

    def test_inherent_impl_method_appears_in_class_methods(self):
        """impl Foo { fn bar } → Foo ClassInfo.methods contains 'bar'.

        Currently ClassInfo.methods == [] while functions[] has ['bar', 'Foo.bar'].
        This test MUST FAIL on unfixed code.
        """
        extractor = HybridExtractor()
        result = extractor.extract(_FIXTURES / "inherent_impl.rs")

        class_map = {c.name: c for c in result.classes}
        assert "DaemonState" in class_map, (
            f"Expected ClassInfo named 'DaemonState'; got classes: "
            f"{[c.name for c in result.classes]}"
        )
        method_names = [m.name for m in class_map["DaemonState"].methods]
        assert "update_all" in method_names, (
            f"Expected 'update_all' in DaemonState.methods; got: {method_names}"
        )

    def test_inherent_impl_extract_file_with_code_returns_code_field(self, tmp_path):
        """extract_file_with_code(..., method='Foo.bar') returns a 'code' field.

        The api filter reads classes[].methods[], so if methods==[] no code is
        injected.  This test exercises the full api path.
        MUST FAIL on unfixed code.
        """
        src = (
            "pub struct Foo {\n"
            "    x: i32,\n"
            "}\n"
            "\n"
            "impl Foo {\n"
            "    pub fn bar(&self) -> i32 {\n"
            "        self.x\n"
            "    }\n"
            "}\n"
        )
        rs = tmp_path / "simple_impl.rs"
        rs.write_text(src, encoding="utf-8")

        result = extract_file_with_code(str(rs), method="Foo.bar")

        classes = result.get("classes", [])
        assert classes, (
            f"Expected at least one class in result; got result keys: {list(result.keys())}"
        )
        foo_class = next((c for c in classes if c.get("name") == "Foo"), None)
        assert foo_class is not None, (
            f"Expected class 'Foo' in result; got class names: "
            f"{[c.get('name') for c in classes]}"
        )
        methods = foo_class.get("methods", [])
        assert methods, (
            f"Expected non-empty methods list for Foo; got methods={methods!r}"
        )
        bar_method = next((m for m in methods if m.get("name") == "bar"), None)
        assert bar_method is not None, (
            f"Expected method 'bar' in Foo.methods; got: "
            f"{[m.get('name') for m in methods]}"
        )
        assert "code" in bar_method, (
            f"Expected 'code' field in bar method dict; got keys: "
            f"{sorted(bar_method.keys())}"
        )
        assert bar_method["code"], (
            "'code' field must be non-empty string"
        )

    # -----------------------------------------------------------------------
    # R-2  trait impl → methods associated with the implementing type's ClassInfo
    # -----------------------------------------------------------------------

    def test_trait_impl_method_linked_to_implementing_type_class(self, tmp_path):
        """impl Trait for Foo { fn baz } → Foo ClassInfo.methods contains 'baz'.

        The implementing type (Foo) should receive the method, not the trait.
        MUST FAIL on unfixed code (Foo.methods == [] currently).
        """
        src = (
            "pub struct Foo {\n"
            "    x: i32,\n"
            "}\n"
            "\n"
            "pub trait Greet {\n"
            "    fn baz(&self) -> String;\n"
            "}\n"
            "\n"
            "impl Greet for Foo {\n"
            "    fn baz(&self) -> String {\n"
            '        format!("hello {}", self.x)\n'
            "    }\n"
            "}\n"
        )
        rs = tmp_path / "trait_impl.rs"
        rs.write_text(src, encoding="utf-8")

        result = HybridExtractor().extract(str(rs))
        class_map = {c.name: c for c in result.classes}

        assert "Foo" in class_map, (
            f"Expected ClassInfo 'Foo'; got classes: {list(class_map.keys())}"
        )
        method_names = [m.name for m in class_map["Foo"].methods]
        assert "baz" in method_names, (
            f"Expected 'baz' in Foo.methods (trait impl methods must be "
            f"linked to the implementing type's ClassInfo); got: {method_names}"
        )

    # -----------------------------------------------------------------------
    # R-3  enum + impl → enum produces ClassInfo; impl methods linked to it
    # -----------------------------------------------------------------------

    def test_enum_produces_class_info(self):
        """enum Color { … } → a ClassInfo named 'Color' appears in classes.

        Currently _extract_rust_nodes has no enum_item branch, so no ClassInfo
        is created and Color is absent.  MUST FAIL on unfixed code.
        """
        extractor = HybridExtractor()
        result = extractor.extract(_FIXTURES / "enum_impl.rs")

        class_names = [c.name for c in result.classes]
        assert "Color" in class_names, (
            f"Expected ClassInfo 'Color' from enum_item; got classes: {class_names}"
        )

    def test_enum_impl_method_linked_to_enum_class_info(self):
        """impl Color { fn rgb } → Color ClassInfo.methods contains 'rgb'.

        Depends on enum_item branch AND the post-walk linker.
        MUST FAIL on unfixed code.
        """
        extractor = HybridExtractor()
        result = extractor.extract(_FIXTURES / "enum_impl.rs")

        class_map = {c.name: c for c in result.classes}
        assert "Color" in class_map, (
            f"Expected ClassInfo 'Color'; got classes: {list(class_map.keys())}"
        )
        method_names = [m.name for m in class_map["Color"].methods]
        assert "rgb" in method_names, (
            f"Expected 'rgb' in Color.methods; got: {method_names}"
        )

    def test_enum_impl_extract_file_with_code_returns_code_field(self):
        """extract_file_with_code(method='Color.rgb') returns a 'code' field.

        Full api path for enum impl methods.  MUST FAIL on unfixed code.
        """
        fixture = _FIXTURES / "enum_impl.rs"
        result = extract_file_with_code(str(fixture), method="Color.rgb")

        classes = result.get("classes", [])
        assert classes, (
            f"Expected 'classes' key in result; got keys: {list(result.keys())}"
        )
        color_class = next((c for c in classes if c.get("name") == "Color"), None)
        assert color_class is not None, (
            f"Expected class 'Color' in result; got: {[c.get('name') for c in classes]}"
        )
        methods = color_class.get("methods", [])
        assert methods, (
            f"Expected non-empty methods for Color; got: {methods!r}"
        )
        rgb_method = next((m for m in methods if m.get("name") == "rgb"), None)
        assert rgb_method is not None, (
            f"Expected method 'rgb'; got method names: {[m.get('name') for m in methods]}"
        )
        assert "code" in rgb_method, (
            f"Expected 'code' field in rgb dict; got keys: {sorted(rgb_method.keys())}"
        )
        assert rgb_method["code"], "'code' must be non-empty"

    # -----------------------------------------------------------------------
    # R-4  impl-before-struct forward reference → ClassInfo.methods populated
    # -----------------------------------------------------------------------

    def test_forward_ref_impl_before_struct_links_methods(self):
        """impl Foo appears textually before struct Foo → Foo.methods still has 'early'.

        Inline linking at impl-visit time fails here because the ClassInfo for Foo
        does not exist yet.  The fix requires a post-walk linker.
        MUST FAIL on unfixed code.
        """
        extractor = HybridExtractor()
        result = extractor.extract(_FIXTURES / "forward_ref_impl.rs")

        class_map = {c.name: c for c in result.classes}
        assert "Foo" in class_map, (
            f"Expected ClassInfo 'Foo' after post-walk linker resolves forward ref; "
            f"got classes: {list(class_map.keys())}"
        )
        method_names = [m.name for m in class_map["Foo"].methods]
        assert "early" in method_names, (
            f"Expected 'early' in Foo.methods (impl appears before struct); "
            f"got: {method_names}"
        )

    # -----------------------------------------------------------------------
    # R-5  method end_line is sensible (> its start line for multi-line bodies)
    # -----------------------------------------------------------------------

    def test_impl_method_end_line_greater_than_start_line(self, tmp_path):
        """Methods in ClassInfo.methods[] must carry end_line > line_number.

        Validates that the FunctionInfo objects linked into ClassInfo.methods
        retain the end_line populated by _extract_rust_function.
        MUST FAIL on unfixed code (methods is empty, so nothing to assert end_line on).
        """
        src = (
            "pub struct Bar {\n"
            "    value: u64,\n"
            "}\n"
            "\n"
            "impl Bar {\n"
            "    pub fn compute(&self) -> u64 {\n"
            "        self.value * 2\n"
            "    }\n"
            "}\n"
        )
        rs = tmp_path / "bar_end_line.rs"
        rs.write_text(src, encoding="utf-8")

        result = HybridExtractor().extract(str(rs))
        class_map = {c.name: c for c in result.classes}

        assert "Bar" in class_map, (
            f"Expected ClassInfo 'Bar'; got: {list(class_map.keys())}"
        )
        methods = class_map["Bar"].methods
        assert methods, (
            f"Expected non-empty Bar.methods; got: {methods!r}"
        )
        compute = next((m for m in methods if m.name == "compute"), None)
        assert compute is not None, (
            f"Expected method 'compute' in Bar.methods; got: "
            f"{[m.name for m in methods]}"
        )
        assert compute.end_line > compute.line_number, (
            f"Expected end_line > line_number for multi-line method body; "
            f"got line_number={compute.line_number}, end_line={compute.end_line}"
        )

    # -----------------------------------------------------------------------
    # R-6  optional live-repro against aqmanager (skipped if repo absent)
    # -----------------------------------------------------------------------

    @pytest.mark.skipif(
        not Path("/Users/tuan/dev/aqmanager/src/project_watcher.rs").exists(),
        reason="aqmanager repo not present",
    )
    def test_live_project_watcher_start_method_in_class_methods(self):
        """ProjectWatcher.start appears in ClassInfo.methods[] in the real repo.

        Skipped when /Users/tuan/dev/aqmanager is absent.
        MUST FAIL on current code when the file is present.
        """
        live_file = "/Users/tuan/dev/aqmanager/src/project_watcher.rs"
        result = HybridExtractor().extract(live_file)

        class_map = {c.name: c for c in result.classes}
        assert "ProjectWatcher" in class_map, (
            f"Expected ClassInfo 'ProjectWatcher'; got: {list(class_map.keys())}"
        )
        method_names = [m.name for m in class_map["ProjectWatcher"].methods]
        assert "start" in method_names, (
            f"Expected 'start' in ProjectWatcher.methods; got: {method_names}"
        )

    @pytest.mark.skipif(
        not Path("/Users/tuan/dev/aqmanager/src/project_watcher.rs").exists(),
        reason="aqmanager repo not present",
    )
    def test_live_project_watcher_extract_file_with_code_has_code(self):
        """extract_file_with_code(method='ProjectWatcher.start') returns 'code'.

        Skipped when /Users/tuan/dev/aqmanager is absent.
        MUST FAIL on current code when the file is present.
        """
        live_file = "/Users/tuan/dev/aqmanager/src/project_watcher.rs"
        result = extract_file_with_code(live_file, method="ProjectWatcher.start")

        classes = result.get("classes", [])
        assert classes, (
            f"Expected 'classes' key; got result keys: {list(result.keys())}"
        )
        pw = next((c for c in classes if c.get("name") == "ProjectWatcher"), None)
        assert pw is not None, (
            f"Expected class 'ProjectWatcher'; got: {[c.get('name') for c in classes]}"
        )
        methods = pw.get("methods", [])
        assert methods, f"Expected non-empty ProjectWatcher.methods; got: {methods!r}"
        start = next((m for m in methods if m.get("name") == "start"), None)
        assert start is not None, (
            f"Expected method 'start'; got: {[m.get('name') for m in methods]}"
        )
        assert "code" in start, (
            f"Expected 'code' field; got keys: {sorted(start.keys())}"
        )
        assert start["code"], "'code' must be non-empty"


# ===========================================================================
# GAP 2 — Swift tree-sitter ERROR node drops CorrectionPreviewModel
# ===========================================================================

_SWIFT_FILE = Path(
    "/Users/tuan/dev/blitz-dictation/Sources/BlitzDictationCore/Preview/"
    "CorrectionPreviewModel.swift"
)

_swift_present = pytest.mark.skipif(
    not _SWIFT_FILE.exists(),
    reason="blitz-dictation repo not present",
)


class TestSwiftCorrectionPreviewModelExtraction:
    """Suite for GAP 2: CorrectionPreviewModel is silently dropped when
    tree-sitter wraps its declaration in an ERROR node.

    All tests in this class are skipped when the blitz-dictation repo is absent.
    They MUST FAIL on current code when the repo is present.
    """

    @_swift_present
    def test_extraction_returns_five_or_more_classes(self):
        """Extracting CorrectionPreviewModel.swift must yield at least 5 classes.

        The file defines 5 top-level types:
          PersistMode, Pill, PreviewResult, PillColorTag, CorrectionPreviewModel.
        Currently only 4 are extracted (CorrectionPreviewModel is dropped by the
        ERROR node) — so the count is 4, not 5.
        MUST FAIL on unfixed code.
        """
        result = HybridExtractor().extract(str(_SWIFT_FILE))
        class_names = [c.name for c in result.classes]
        assert len(class_names) >= 5, (
            f"Expected at least 5 classes (including CorrectionPreviewModel); "
            f"got {len(class_names)}: {class_names}"
        )

    @_swift_present
    def test_correction_preview_model_class_is_extracted(self):
        """CorrectionPreviewModel must appear as a ClassInfo in the extraction output.

        Currently the class is wrapped in a tree-sitter ERROR node and dropped.
        Its 4 sibling classes (PersistMode, Pill, PreviewResult, PillColorTag) do
        extract successfully but CorrectionPreviewModel does not.
        MUST FAIL on unfixed code.
        """
        result = HybridExtractor().extract(str(_SWIFT_FILE))
        class_names = [c.name for c in result.classes]
        assert "CorrectionPreviewModel" in class_names, (
            f"Expected 'CorrectionPreviewModel' in extracted classes; "
            f"got: {class_names}"
        )

    @_swift_present
    def test_approve_method_linked_to_correction_preview_model(self):
        """CorrectionPreviewModel.methods[] must include 'approve'.

        'approve' is a public method on CorrectionPreviewModel (line ~415).
        If the class is dropped it has no ClassInfo to link to.
        MUST FAIL on unfixed code.
        """
        result = HybridExtractor().extract(str(_SWIFT_FILE))
        class_map = {c.name: c for c in result.classes}
        assert "CorrectionPreviewModel" in class_map, (
            f"CorrectionPreviewModel ClassInfo absent; got: {list(class_map.keys())}"
        )
        method_names = [m.name for m in class_map["CorrectionPreviewModel"].methods]
        assert "approve" in method_names, (
            f"Expected 'approve' in CorrectionPreviewModel.methods; got: {method_names}"
        )

    @_swift_present
    def test_extract_file_with_code_method_approve_returns_code(self):
        """extract_file_with_code(method='CorrectionPreviewModel.approve') returns 'code'.

        Currently returns bare {file_path, language} — no classes key at all.
        MUST FAIL on unfixed code.
        """
        result = extract_file_with_code(
            str(_SWIFT_FILE), method="CorrectionPreviewModel.approve"
        )

        assert "classes" in result, (
            f"Expected 'classes' key in result; got keys: {list(result.keys())}"
        )
        classes = result["classes"]
        assert classes, f"Expected non-empty classes list; got: {classes!r}"

        cpm = next(
            (c for c in classes if c.get("name") == "CorrectionPreviewModel"), None
        )
        assert cpm is not None, (
            f"Expected class 'CorrectionPreviewModel'; got: "
            f"{[c.get('name') for c in classes]}"
        )
        methods = cpm.get("methods", [])
        assert methods, (
            f"Expected non-empty methods list; got: {methods!r}"
        )
        approve = next((m for m in methods if m.get("name") == "approve"), None)
        assert approve is not None, (
            f"Expected method 'approve'; got: {[m.get('name') for m in methods]}"
        )
        assert "code" in approve, (
            f"Expected 'code' field in approve dict; got keys: {sorted(approve.keys())}"
        )
        assert approve["code"], "'code' field must be non-empty"


# ===========================================================================
# GAP 2b — Swift ERROR-recovery sibling sweep must STOP at a type/ERROR boundary
# ===========================================================================


def _swift_extractor_available() -> bool:
    """True if the Swift tree-sitter grammar is loadable by HybridExtractor."""
    try:
        ext = HybridExtractor()
        getter = getattr(ext, "_get_swift_parser", None) or getattr(
            ext, "_get_parser", None
        )
        if getter is None:
            return True  # extractor present; let the test run and self-skip if empty
        try:
            getter("swift") if getter.__code__.co_argcount > 1 else getter()
        except Exception:
            return False
        return True
    except Exception:
        return False


_swift_grammar = pytest.mark.skipif(
    not _swift_extractor_available(),
    reason="Swift tree-sitter grammar not available",
)


class TestSwiftErrorRecoverySiblingSweepBoundary:
    """The ERROR-recovery sibling sweep in ``_recover_swift_error_class`` must
    stop at the next type-declaration or ERROR boundary.  Otherwise it (a)
    over-claims a trailing free function declared after a *later* type, and (b)
    double-attaches the same method to multiple ERROR-wrapped classes.

    These tests assert the boundary-stop invariant at the extractor level: no
    method name may appear in more than one class's ``methods[]``.  They are
    robust whether or not a faithful tree-sitter ERROR is triggered — if the
    synthetic source parses cleanly the invariant still holds.
    """

    @_swift_grammar
    def test_no_method_attached_to_more_than_one_class(self, tmp_path):
        """No method name appears in two different classes' methods[].

        The synthetic source declares two classes with disjoint method names
        plus a trailing free function after a later type.  Even if tree-sitter
        wraps either class in an ERROR node, the sibling sweep must not cross a
        type/ERROR boundary, so a method declared inside one class is never
        also attached to a different class, and the trailing free function is
        never mis-attached.  (Distinct names per class are used deliberately so
        that a duplicate signals a genuine cross-boundary bug, not the benign
        case of two clean classes happening to share a method name.)
        """
        src = (
            "class Alpha {\n"
            "    func alphaA() -> Int { return 1 }\n"
            "    func alphaB() -> Int { return 2 }\n"
            "}\n"
            "\n"
            "class Beta {\n"
            "    func betaA() -> Int { return 3 }\n"
            "    func betaB() -> Int { return 4 }\n"
            "}\n"
            "\n"
            "struct Gamma {\n"
            "    var x: Int = 0\n"
            "}\n"
            "\n"
            "func trailingFreeFunction() -> Int { return 99 }\n"
        )
        swift = tmp_path / "boundary.swift"
        swift.write_text(src, encoding="utf-8")

        result = HybridExtractor().extract(str(swift))

        # Invariant (a): a method may not belong to more than one class.
        owners: dict[str, list[str]] = {}
        for cls in result.classes:
            for m in cls.methods:
                owners.setdefault(m.name, []).append(cls.name)
        duplicated = {
            mname: clss for mname, clss in owners.items() if len(clss) > 1
        }
        assert not duplicated, (
            f"Method(s) attached to more than one class (sweep crossed a "
            f"boundary): {duplicated}"
        )

        # Invariant (b): the trailing free function declared after the later
        # type must not be mis-attached as a recovered-class method.
        all_method_names = {m.name for cls in result.classes for m in cls.methods}
        assert "trailingFreeFunction" not in all_method_names, (
            "Trailing free function after a later type was mis-attached as a "
            f"recovered-class method; class methods: "
            f"{ {c.name: [m.name for m in c.methods] for c in result.classes} }"
        )
