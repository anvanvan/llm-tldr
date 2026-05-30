"""
Regression test for bug 001 (per-language coverage):
`tldr extract <file> --function NAME` must include a non-empty `code` field
for every supported language, and its bytes must equal the source span
between `line_number` and `end_line` reported by the extractor.

Languages covered: Rust, Swift, Go, Java, PHP, JavaScript.
Python and TypeScript are covered by `tests/test_extract_code_field.py`.

Each case writes an inline fixture into a tmp_path and asserts:
  1. The function appears under `functions`.
  2. The matched dict has a `code` key.
  3. `code` is non-empty.
  4. `code` equals the source bytes between `line_number` and `end_line`.
"""

from __future__ import annotations

import pytest

from tldr.api import extract_file_with_code


def _write(tmp_path, name: str, body: str):
    """Write `body` to tmp_path/name and return the resulting Path."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _assert_code_field(file_path, function_name: str, language: str):
    """Run extract --function and assert the `code` field matches source bytes."""
    payload = extract_file_with_code(str(file_path), function=function_name)
    functions = payload.get("functions", [])
    assert functions, (
        f"[{language}] expected --function={function_name} to match at least one "
        f"top-level function; got functions={functions!r}"
    )

    func = next((f for f in functions if f["name"] == function_name), None)
    assert func is not None, (
        f"[{language}] function {function_name!r} not present in filtered result; "
        f"functions={[f.get('name') for f in functions]!r}"
    )

    assert "code" in func, (
        f"[{language}] expected 'code' key in function dict but got only: "
        f"{sorted(func.keys())}"
    )

    code_text: str = func["code"]
    assert code_text, f"[{language}] 'code' field must be non-empty"

    ln = func.get("line_number")
    end = func.get("end_line")
    assert ln and end and end >= ln, (
        f"[{language}] expected line_number<=end_line; got line_number={ln} "
        f"end_line={end}"
    )

    source_lines = file_path.read_text(encoding="utf-8").splitlines()
    expected = "\n".join(source_lines[ln - 1 : end])
    assert code_text == expected, (
        f"[{language}] 'code' bytes do not match source span "
        f"({ln}..{end}):\n"
        f"  expected ({len(expected)} bytes): {expected[:200]!r}...\n"
        f"  got      ({len(code_text)} bytes): {code_text[:200]!r}..."
    )


# --------------------------------------------------------------------------- #
# Per-language fixtures + tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("filename", "source", "function_name", "language"),
    [
        (
            "fixture.rs",
            (
                "fn add(a: i32, b: i32) -> i32 {\n"
                "    let sum = a + b;\n"
                "    sum\n"
                "}\n"
                "\n"
                "fn other() -> i32 { 42 }\n"
            ),
            "add",
            "Rust",
        ),
        (
            "fixture.swift",
            (
                "import Foundation\n"
                "\n"
                "func greet(name: String) -> String {\n"
                "    let prefix = \"Hello\"\n"
                "    return \"\\(prefix), \\(name)\"\n"
                "}\n"
                "\n"
                "func other() -> Int { return 7 }\n"
            ),
            "greet",
            "Swift",
        ),
        (
            "fixture.go",
            (
                "package main\n"
                "\n"
                "func Add(a int, b int) int {\n"
                "\tsum := a + b\n"
                "\treturn sum\n"
                "}\n"
                "\n"
                "func main() {}\n"
            ),
            "Add",
            "Go",
        ),
        (
            "Fixture.java",
            (
                "public class Fixture {\n"
                "    public static int add(int a, int b) {\n"
                "        int sum = a + b;\n"
                "        return sum;\n"
                "    }\n"
                "}\n"
            ),
            "add",
            "Java",
        ),
        (
            "fixture.php",
            (
                "<?php\n"
                "\n"
                "function add($a, $b) {\n"
                "    $sum = $a + $b;\n"
                "    return $sum;\n"
                "}\n"
                "\n"
                "function other() { return 7; }\n"
            ),
            "add",
            "PHP",
        ),
        (
            "fixture.js",
            (
                "function add(a, b) {\n"
                "  const sum = a + b;\n"
                "  return sum;\n"
                "}\n"
                "\n"
                "function other() { return 7; }\n"
            ),
            "add",
            "JavaScript",
        ),
    ],
    ids=["rust", "swift", "go", "java", "php", "javascript"],
)
def test_extract_code_field_matches_source_bytes(
    tmp_path, filename, source, function_name, language
):
    """`extract_file_with_code(..., function=NAME)` returns a `code` field
    equal to the source bytes between line_number and end_line.

    The Java extractor emits class methods under both `functions` and
    `classes[].methods`; the top-level `functions` filter path is used
    here for uniformity across languages.
    """
    fixture = _write(tmp_path, filename, source)
    _assert_code_field(fixture, function_name, language)
