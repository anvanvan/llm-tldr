"""Regression test — Bug 004 (re-surfaced 2026-05-24).

Bug 004 was originally marked Fixed on 2026-05-23 ("Option A — extended coverage")
but the dogfooded `tldr semantic search "install dependencies" --path .` over
/Users/tuan/dev/llm-tldr still returned only `.py` files. Root cause for the
follow-up: the index was already built before the original fix landed, but more
importantly the whole-file EmbeddingUnits for non-code files (.md / .toml /
.txt / .sh / .yaml / .json) had embedding text that started with "Code:" and
carried a generic `language="text"` tag, so BGE embeddings consistently lost to
.py units that lexically mention "dependency" in a CS sense.

The follow-up fix (this commit):

1. `_process_file_for_extraction` now tags non-code units with their real
   language (`markdown` / `toml` / `yaml` / `shell` / ...) from
   `EXTENSION_TO_LANGUAGE` instead of a generic `"text"`.
2. `_process_file_for_extraction` now injects a filename-derived description
   into `docstring` for well-known non-code files (README, pyproject.toml,
   requirements.txt, package.json, build.sh, Cargo.toml, ...). This gives the
   embedding semantic context about install/setup/dependencies that pure
   manifest content lacks.
3. `build_embedding_text` no longer labels non-code content "Code:" — it uses
   "Documentation:" / "Configuration:" / "Shell script:" / "Content:" based on
   the unit's language so the embedding model sees doc/config text framed as
   doc/config text.

This test pins the integration: `build_embedding_text` for a pyproject.toml
unit must mention `install` and `dependencies` (via the filename hint) so the
embedding aligns with natural-language install queries, and must NOT carry the
misleading "Code:" prefix that would bias the embedding toward code-style
queries.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_PY_CONTENT = """\
def placeholder():
    \"\"\"A placeholder function.\"\"\"
    pass
"""

_PYPROJECT_CONTENT = """\
[project]
name = "myproject"
version = "0.1.0"
dependencies = ["requests", "numpy"]
"""

_REQUIREMENTS_CONTENT = """\
requests>=2.25
numpy>=1.20
"""

_README_CONTENT = """\
# My Project

A useful project.

## Install

Run `pip install -r requirements.txt` to install dependencies.
"""


@pytest.fixture
def mixed_fixture(tmp_path: Path) -> Path:
    """Repo with one .py code file and well-known non-code manifests/docs."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "main.py").write_text(_PY_CONTENT)
    (tmp_path / "pyproject.toml").write_text(_PYPROJECT_CONTENT)
    (tmp_path / "requirements.txt").write_text(_REQUIREMENTS_CONTENT)
    (tmp_path / "README.md").write_text(_README_CONTENT)
    return tmp_path


class TestSemanticNonCodeBug004FollowUp:
    """Pins the 2026-05-24 follow-up fix that re-surfaces non-code files for
    natural-language queries like "install dependencies"."""

    def test_non_code_units_carry_real_language_tag(self, mixed_fixture: Path):
        """EmbeddingUnit.language must be the real document type
        (markdown/toml/text) rather than the generic 'text' fallback so query-
        time filters and the embedding text can distinguish doc/config files.
        """
        from tldr.semantic import extract_units_from_project

        units = extract_units_from_project(
            str(mixed_fixture), lang="python", respect_ignore=False,
        )
        by_file = {u.file: u for u in units if u.unit_type == "file"}

        assert "pyproject.toml" in by_file, (
            f"pyproject.toml not indexed; got {list(by_file.keys())!r}"
        )
        assert by_file["pyproject.toml"].language == "toml", (
            f"pyproject.toml unit must be tagged language='toml' "
            f"(got {by_file['pyproject.toml'].language!r}). "
            f"Without the real tag, BGE embeddings cannot distinguish "
            f"manifest text from generic 'text', and the docstring hint cannot "
            f"discriminate config vs documentation."
        )
        assert by_file["README.md"].language == "markdown", (
            f"README.md unit must be tagged language='markdown' "
            f"(got {by_file['README.md'].language!r})."
        )

    def test_well_known_filenames_get_install_hint(self, mixed_fixture: Path):
        """The pyproject.toml, requirements.txt and README.md units must carry
        a docstring that mentions install/dependencies so natural-language
        install queries embed close to these units (the raw manifest content
        alone does not contain 'install').
        """
        from tldr.semantic import extract_units_from_project

        units = extract_units_from_project(
            str(mixed_fixture), lang="python", respect_ignore=False,
        )
        by_file = {u.file: u for u in units if u.unit_type == "file"}

        for fname in ("pyproject.toml", "requirements.txt", "README.md"):
            assert fname in by_file, f"{fname} not indexed"
            ds = by_file[fname].docstring.lower()
            assert "install" in ds or "dependencies" in ds, (
                f"{fname} unit docstring missing install/dependencies hint; "
                f"got: {by_file[fname].docstring!r}. This hint is the load-"
                f"bearing semantic signal for queries like 'install dependencies'."
            )

    def test_build_embedding_text_no_code_prefix_for_non_code(
        self, mixed_fixture: Path
    ):
        """build_embedding_text for a non-code file unit must NOT use the
        'Code:' prefix — that prefix biases BGE embeddings toward code-style
        queries and was the cause of README/pyproject.toml losing to .py
        units that lexically mention 'dependency' in a CS sense.
        """
        from tldr.semantic import build_embedding_text, extract_units_from_project

        units = extract_units_from_project(
            str(mixed_fixture), lang="python", respect_ignore=False,
        )
        toml_unit = next(
            u for u in units if u.unit_type == "file" and u.file == "pyproject.toml"
        )
        text = build_embedding_text(toml_unit)

        assert "\nCode:\n" not in text, (
            "Non-code file embedding text must not be labelled 'Code:' — it "
            "biases the BGE embedding toward code-style queries.\nGot:\n"
            f"{text}"
        )
        # Should still surface the actual content for retrieval
        assert "dependencies" in text.lower(), (
            f"Embedding text for pyproject.toml should include the word "
            f"'dependencies' (from the install hint or the file content):\n{text}"
        )

    def test_build_embedding_text_includes_readme_content(
        self, mixed_fixture: Path
    ):
        """build_embedding_text for a markdown file unit must include the
        actual README content (without the misleading 'Code:' prefix) so the
        BGE embedding aligns with natural-language doc queries.
        """
        from tldr.semantic import build_embedding_text, extract_units_from_project

        units = extract_units_from_project(
            str(mixed_fixture), lang="python", respect_ignore=False,
        )
        md_unit = next(
            u for u in units if u.unit_type == "file" and u.file == "README.md"
        )
        text = build_embedding_text(md_unit)

        # Raw content must surface in the embedding text…
        assert "install" in text.lower(), (
            f"README.md embedding text should contain the README body "
            f"(which mentions 'install'). Got:\n{text}"
        )
        # …and the misleading "Code:" prefix must not be applied to prose.
        assert "\nCode:\n" not in text, (
            f"README.md embedding text must NOT be labelled 'Code:'. "
            f"Got:\n{text}"
        )
