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
