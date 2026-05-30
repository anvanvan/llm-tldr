"""Regression test for Item 5 of `.claude/specs/per-language-support.md`.

Tier-1 spec: `tldr context` must accept `c` and `elixir` as `--lang` choices,
alongside the prior 9 languages (python, typescript, javascript, go, rust,
php, swift, java, ruby). Builders `_build_c_call_graph` and
`_build_elixir_call_graph` already exist in tldr/cross_file_calls.py and are
wired into build_project_call_graph — this regression locks the argparse-level
config so the spec gap doesn't reopen.

Asserts both the API contract (SUPPORTED_CONTEXT_EXT_MAP / _resolve_context_languages)
and the argparse choice list exposed via `tldr context --help`.

Runtime caveat (per spec, observed 2026-05-24):
  - C: full call graph resolution works (callers + callees + complexity).
  - Elixir: parses and resolves function-by-name but emits no edges
    (no callers/callees). Test only asserts the name resolves and the
    CLI doesn't exit non-zero — runtime edge emission is out of scope.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from tldr.api import SUPPORTED_CONTEXT_EXT_MAP, SUPPORTED_CONTEXT_LANGUAGES
from tldr.cli import CONTEXT_LANG_CHOICES, _resolve_context_languages


# ---------------------------------------------------------------------------
# API-level: the constants include c and elixir with the expected extensions
# ---------------------------------------------------------------------------


def test_supported_context_ext_map_includes_c() -> None:
    assert "c" in SUPPORTED_CONTEXT_EXT_MAP
    assert SUPPORTED_CONTEXT_EXT_MAP["c"] == {".c", ".h"}


def test_supported_context_ext_map_includes_elixir() -> None:
    assert "elixir" in SUPPORTED_CONTEXT_EXT_MAP
    assert SUPPORTED_CONTEXT_EXT_MAP["elixir"] == {".ex", ".exs"}


def test_supported_context_languages_includes_c_and_elixir() -> None:
    assert "c" in SUPPORTED_CONTEXT_LANGUAGES
    assert "elixir" in SUPPORTED_CONTEXT_LANGUAGES


def test_resolve_context_languages_accepts_c(tmp_path: Path) -> None:
    """_resolve_context_languages('c') must not raise and must return ['c']."""
    (tmp_path / "a.c").write_text("int main(void) { return 0; }\n")
    result = _resolve_context_languages("c", project_path=tmp_path)
    assert result == ["c"]


def test_resolve_context_languages_accepts_elixir(tmp_path: Path) -> None:
    """_resolve_context_languages('elixir') must not raise and return ['elixir']."""
    (tmp_path / "a.ex").write_text("defmodule M do\n  def f, do: :ok\nend\n")
    result = _resolve_context_languages("elixir", project_path=tmp_path)
    assert result == ["elixir"]


# ---------------------------------------------------------------------------
# CLI-level: argparse exposes c and elixir as valid --lang choices
# ---------------------------------------------------------------------------


def test_context_lang_choices_includes_c_and_elixir() -> None:
    assert "c" in CONTEXT_LANG_CHOICES
    assert "elixir" in CONTEXT_LANG_CHOICES
    # Sanity: prior nine still present.
    for lang in (
        "python", "typescript", "javascript", "go", "rust",
        "php", "swift", "java", "ruby",
    ):
        assert lang in CONTEXT_LANG_CHOICES


def test_tldr_context_help_lists_c_and_elixir() -> None:
    """`tldr context --help` exposes c + elixir in the --lang choice set."""
    proc = subprocess.run(
        [sys.executable, "-m", "tldr.cli", "context", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    # Argparse renders choices as a comma-separated set inside braces.
    assert ",c," in combined or "{c," in combined or ",c}" in combined
    assert "elixir" in combined


# ---------------------------------------------------------------------------
# Runtime smoke: tldr context resolves a function in C / Elixir fixtures
# ---------------------------------------------------------------------------


def _run_tldr_context(fn: str, project: Path, lang: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tldr.cli", "context", fn, "--project", str(project), "--lang", lang],
        capture_output=True, text=True, check=False,
    )


def test_tldr_context_resolves_c_function(tmp_path: Path) -> None:
    """Full runtime path: tldr context with --lang c resolves a function."""
    (tmp_path / "a.c").write_text(textwrap.dedent("""\
        #include <stdio.h>

        int add(int a, int b) {
            return a + b;
        }

        int compute(int x, int y) {
            int sum = add(x, y);
            return sum * 2;
        }

        int main(void) {
            return compute(3, 4);
        }
        """))
    proc = _run_tldr_context("compute", tmp_path, "c")
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "compute" in proc.stdout
    # C runtime resolves edges — callees should appear.
    assert "add" in proc.stdout


def test_tldr_context_resolves_elixir_function(tmp_path: Path) -> None:
    """Runtime path: tldr context with --lang elixir resolves a function name.

    Per spec caveat, the elixir builder may emit no edges — we only assert
    the function is named in output and exit is clean. Edge emission is
    explicitly out of scope for this regression.
    """
    (tmp_path / "a.ex").write_text(textwrap.dedent("""\
        defmodule Smoke do
          def add(a, b) do
            a + b
          end

          def compute(x, y) do
            sum = add(x, y)
            sum * 2
          end
        end
        """))
    proc = _run_tldr_context("compute", tmp_path, "elixir")
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "compute" in proc.stdout
