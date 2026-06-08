"""
Regression test: tldr extract --function NAME must include a 'code' field.

Bug: Before the fix, `tldr extract <file> --function NAME` returned only
     metadata (name, line_number, signature, params, return_type, docstring,
     is_async, decorators) without the function's source code.

Fix: The CLI extract subcommand now injects a 'code' field on filtered
     symbols (--function / --method / --class) when the extractor
     populated end_line. Bare extracts stay metadata-only.

This test validates that the filtered extract path correctly returns
non-empty source code spans anchored to the declared line_number.
"""

from pathlib import Path


# A Python file that definitely exists in this repo and has a known function.
# _build_java_call_graph is confirmed present in cross_file_calls.py
# (verified in reproduction.md step 2).
_REPO_ROOT = Path(__file__).parent.parent
_CROSS_FILE_CALLS_PY = _REPO_ROOT / "tldr" / "cross_file_calls.py"


class TestExtractCodeField:
    """`tldr extract <file> --function NAME` must emit a non-empty
    'code' field anchored to the function's line_number."""

    def test_extracted_function_has_code_field_anchored_to_line_number(self):
        """`tldr extract --function NAME` must include 'code' in the matched
        function dict, non-empty, starting at the declared line_number.

        Three behaviors tested in one block:
        1. 'code' key is present in the function dict.
        2. 'code' is non-empty.
        3. First line of 'code' matches the actual source at line_number (anchoring)
           and contains the declaration keyword.
        """
        from tldr.api import extract_file_with_code

        payload = extract_file_with_code(
            str(_CROSS_FILE_CALLS_PY),
            function="_build_java_call_graph",
        )
        functions = payload.get("functions", [])
        assert functions, (
            f"Expected --function filter to return at least one match; "
            f"got functions={functions!r}"
        )

        func = next(
            (f for f in functions if f["name"] == "_build_java_call_graph"),
            None,
        )
        assert func is not None, (
            "_build_java_call_graph not found in filtered extract output"
        )

        # Assertion 1: 'code' key must be present
        assert "code" in func, (
            f"Expected 'code' key in function dict but got only: {sorted(func.keys())}"
        )

        code_text: str = func["code"]

        # Assertion 2: 'code' must be non-empty
        assert code_text, "'code' field must be non-empty"

        # Assertion 3: first line of 'code' must match source at line_number
        source_lines = _CROSS_FILE_CALLS_PY.read_text().splitlines()
        line_idx = func["line_number"] - 1  # convert 1-based -> 0-based
        assert 0 <= line_idx < len(source_lines), (
            f"line_number {func['line_number']} out of range for file"
        )
        expected_first_line = source_lines[line_idx]
        actual_first_line = code_text.splitlines()[0]
        assert actual_first_line == expected_first_line, (
            f"code[0] does not match source line {func['line_number']}:\n"
            f"  expected: {expected_first_line!r}\n"
            f"  got:      {actual_first_line!r}"
        )
        # Verify the declaration keyword is present (e.g. starts with 'def')
        assert actual_first_line.lstrip().startswith("def _build_java_call_graph"), (
            f"First line of 'code' must begin with the function declaration; "
            f"got: {actual_first_line!r}"
        )


class TestFilteredExtractIsCompact:
    """A filtered `tldr extract` is a single-symbol request: it must return a
    compact, code-first dict and must NOT bury the `code` field under the
    module's full import list / call graph.

    Regression: Java files routinely carry 30+ imports (~6 JSON lines each),
    which pushed the injected `code` hundreds of lines down — past any
    `head -60` — forcing the grep+Read fallback the command exists to avoid.
    """

    def test_filtered_extract_omits_imports_and_call_graph(self):
        from tldr.api import extract_file_with_code

        result = extract_file_with_code(
            str(_CROSS_FILE_CALLS_PY), function="_build_java_call_graph"
        )

        # Module-wide noise is dropped on a single-symbol request.
        assert "imports" not in result, (
            "filtered extract must omit module-wide 'imports' "
            f"(got keys: {sorted(result.keys())})"
        )
        assert "call_graph" not in result, (
            "filtered extract must omit module-wide 'call_graph' "
            f"(got keys: {sorted(result.keys())})"
        )
        # Identity + the matched symbol survive.
        assert result.get("file_path")
        assert result.get("language")
        assert result.get("functions"), "matched function must be present"

    def test_code_surfaces_within_first_lines_of_json(self):
        """The `code` field must appear near the top of the serialized JSON,
        not buried under a long import list."""
        import json
        from tldr.api import extract_file_with_code

        result = extract_file_with_code(
            str(_CROSS_FILE_CALLS_PY), function="_build_java_call_graph"
        )
        lines = json.dumps(result, indent=2).splitlines()
        code_idx = next(
            (i for i, ln in enumerate(lines) if '"code"' in ln), None
        )
        assert code_idx is not None, "'code' field missing from serialized output"
        assert code_idx < 60, (
            f"'code' field appears at line {code_idx + 1}; must be within the "
            f"first 60 lines so a `head -60` reveals the body"
        )

    def test_unfiltered_extract_keeps_full_shape(self):
        """Bare (no-filter) extract is unchanged — still carries imports."""
        from tldr.api import extract_file_with_code

        result = extract_file_with_code(str(_CROSS_FILE_CALLS_PY))
        assert "imports" in result, "unfiltered extract must keep full shape"
        assert "call_graph" in result

    def test_method_filter_is_compact_with_code(self):
        """`--method Class.method` shares the compact-dict path: imports /
        call_graph absent, and the method's `code` is injected. This is the
        exact path that triggered the original Java burial."""
        from tldr.api import extract_file_with_code

        ast_extractor_py = _REPO_ROOT / "tldr" / "ast_extractor.py"
        result = extract_file_with_code(
            str(ast_extractor_py), method="ModuleInfo.to_dict"
        )
        assert "imports" not in result, "method extract must omit 'imports'"
        assert "call_graph" not in result, "method extract must omit 'call_graph'"
        classes = result.get("classes")
        assert classes, "matched class must be present"
        methods = classes[0].get("methods")
        assert methods, "matched method must be present"
        assert "code" in methods[0], "matched method must carry 'code'"
        assert methods[0]["code"].lstrip().startswith("def to_dict")

    def test_class_filter_is_compact_with_code(self):
        """`--class Name` shares the compact-dict path: imports / call_graph
        absent, and the class `code` is injected."""
        from tldr.api import extract_file_with_code

        ast_extractor_py = _REPO_ROOT / "tldr" / "ast_extractor.py"
        result = extract_file_with_code(
            str(ast_extractor_py), class_="ModuleInfo"
        )
        assert "imports" not in result, "class extract must omit 'imports'"
        assert "call_graph" not in result, "class extract must omit 'call_graph'"
        classes = result.get("classes")
        assert classes, "matched class must be present"
        assert classes[0].get("name") == "ModuleInfo"
        assert "code" in classes[0], "matched class must carry 'code'"
