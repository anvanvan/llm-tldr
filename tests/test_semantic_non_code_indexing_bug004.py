"""
Regression test — Bug 004: `tldr semantic search` does not index non-code files.

Root cause (verified in /tmp/claude-bug-fix-semantic-non-code-files/verification.md):

  Gate 1 (api.py): `get_code_structure` at api.py:1798-1914 builds an ext_map that
  contains only code-language suffixes (.py, .rs, .ts, …).  The filter at
  api.py:1881 (`if file_path.suffix not in extensions: continue`) drops every
  non-code file before it reaches the semantic indexer.

  Gate 2 (semantic.py): Even if Gate 1 were fixed, `_process_file_for_extraction`
  at semantic.py:895-1273 iterates only over `file_info["functions"]` and
  `file_info["classes"]` lists.  A non-code file has neither, so zero
  EmbeddingUnits would be emitted for it.

Both gates must be fixed together.  This test exercises the combined end-to-end
symptom: calling `extract_units_from_project` over a fixture directory that
contains .sh and .md files produces ZERO EmbeddingUnits whose `file` attribute
ends in a non-code extension.

Pre-fix: the assertions fail because Gate 2 in _process_file_for_extraction emits
         nothing for files with empty functions/classes lists (Gate 1 is already
         fixed by impl-batch; NON_CODE_EXTENSIONS are now enumerated in api.py).
Post-fix: at least one EmbeddingUnit with a .sh or .md (or .toml) `file` must
          appear in the returned list.

Failure under current HEAD (Gate 1 fixed, Gate 2 still broken):
  extract_units_from_project goes through _process_file_for_extraction for each
  non-code file in structure["files"].  _process_file_for_extraction loops only
  over file_info["functions"] and file_info["classes"] (both empty for non-code
  files), emitting zero units.  The assertions below therefore fail.
"""

import json
import pytest
from pathlib import Path


# Minimal .sh content — two lines, non-trivial.
_SH_CONTENT = """\
#!/usr/bin/env bash
set -euo pipefail
echo "building project..."
make all
"""

# Minimal .md content — non-trivial.
_MD_CONTENT = """\
# Project README

This project does something useful.

## Usage

Run `./build.sh` to compile.
"""

# Minimal .toml content.
_TOML_CONTENT = """\
[package]
name = "myproject"
version = "0.1.0"
"""

# Minimal Python content so the indexer has at least one code file to parse,
# and so _detect_project_languages returns a non-empty set for `lang="all"`.
_PY_CONTENT = """\
def placeholder():
    \"\"\"A placeholder function.\"\"\"
    pass
"""


@pytest.fixture
def mixed_fixture(tmp_path: Path) -> Path:
    """Fixture directory with one .py (code) and .sh + .md + .toml (non-code)."""
    # C-10: anchor _find_project_root to tmp_path so it cannot walk up to the
    # real project root on unusual CI configs (hermeticity).
    (tmp_path / ".git").mkdir()
    (tmp_path / "main.py").write_text(_PY_CONTENT)
    (tmp_path / "build.sh").write_text(_SH_CONTENT)
    (tmp_path / "README.md").write_text(_MD_CONTENT)
    (tmp_path / "config.toml").write_text(_TOML_CONTENT)
    return tmp_path


class TestSemanticNonCodeIndexingBug004:
    """extract_units_from_project must emit EmbeddingUnits for non-code files.

    This class exercises the FULL extraction pipeline used by the CLI
    (`tldr semantic index`), which internally calls:

        build_semantic_index(project_path, lang=lang)
          └─ extract_units_from_project(project_path, lang=lang)
               └─ _process_file_for_extraction(file_info, ...)   ← Gate 2

    The primary regression test drives this path end-to-end: if Gate 2
    (_process_file_for_extraction) silently drops non-code files, the
    assertion fails on current HEAD.
    """

    def test_non_code_files_produce_embedding_units(self, mixed_fixture: Path):
        """extract_units_from_project over a mixed directory must return at least
        one EmbeddingUnit whose `file` ends in a non-code extension (.sh, .md, .toml).

        This IS the same internal function that `tldr semantic index` calls:

            build_semantic_index(path, lang="python")
              └─ extract_units_from_project(path, lang="python")   ← called here
                   └─ _process_file_for_extraction(file_info, ...)

        Gate 1 (api.py NON_CODE_EXTENSIONS) is already fixed by impl-batch:
        get_code_structure now includes non-code files in structure["files"] with
        empty functions/classes lists.

        Pre-fix failure (current HEAD): _process_file_for_extraction loops only
        over functions/classes, emitting nothing for non-code file entries, so
        the returned list has zero non-code units and the assertion fails.

        Post-fix: at least one unit with file ending in '.sh', '.md', or '.toml'.
        """
        from tldr.semantic import extract_units_from_project

        units = extract_units_from_project(
            str(mixed_fixture),
            lang="python",
            respect_ignore=False,
        )

        non_code_extensions = {".sh", ".md", ".toml", ".yaml", ".yml"}
        non_code_units = [
            u for u in units
            if Path(u.file).suffix in non_code_extensions
        ]

        assert len(non_code_units) >= 1, (
            f"Expected at least 1 EmbeddingUnit for non-code files (.sh, .md, .toml) "
            f"but got 0.\n"
            f"All returned units: {[(u.file, u.name) for u in units]!r}\n"
            f"Root cause (Gate 2): semantic.py _process_file_for_extraction iterates "
            f"only over file_info['functions'] and file_info['classes']; for non-code "
            f"files both lists are empty, so zero EmbeddingUnits are emitted. "
            f"Fix: add a 'whole-file unit' emission path in _process_file_for_extraction "
            f"for files with no functions/classes."
        )

    def test_gate2_process_file_for_extraction_non_code_entry(
        self, mixed_fixture: Path
    ):
        """_process_file_for_extraction must emit at least one EmbeddingUnit for
        a non-code file_info entry (functions=[], classes=[]).

        This test pins Gate 2 directly: it calls _process_file_for_extraction with
        exactly the dict that _build_non_code_file_entry produces for a .sh file,
        then asserts that a unit is emitted.  On current HEAD the function returns
        an empty list because it only iterates over functions and classes.

        This is the most precise regression pin for Gate 2: it cannot pass unless
        _process_file_for_extraction handles empty-function-list entries.
        """
        from tldr.semantic import _process_file_for_extraction

        # Exactly the dict produced by api._build_non_code_file_entry("build.sh")
        non_code_file_info = {
            "path": "build.sh",
            "functions": [],
            "classes": [],
            "methods": [],
            "imports": [],
        }

        units = _process_file_for_extraction(
            non_code_file_info,
            str(mixed_fixture),
            "python",   # lang arg — non-code files don't parse as any lang
            {},         # calls_map
            {},         # called_by_map
        )

        assert len(units) >= 1, (
            f"Expected _process_file_for_extraction to emit at least 1 EmbeddingUnit "
            f"for a non-code file entry (functions=[], classes=[]) but got 0.\n"
            f"Fix: detect when both lists are empty and emit a whole-file unit using "
            f"the file's raw content as the embedding text."
        )

        # The emitted unit must reference the correct file
        assert any("build.sh" in u.file for u in units), (
            f"EmbeddingUnit.file must reference 'build.sh'; got: "
            f"{[u.file for u in units]!r}"
        )

    def test_full_pipeline_non_code_units_in_metadata_json(
        self, mixed_fixture: Path, monkeypatch
    ):
        """build_semantic_index — the TRUE CLI entry point — must write non-code
        units into .tldr/cache/semantic/metadata.json.

        This test drives the COMPLETE path:

            build_semantic_index(path, lang="python", show_progress=False)
              └─ extract_units_from_project(path, lang="python")
                   └─ _process_file_for_extraction(file_info, ...)
                        └─ [Gate 2 must emit a unit for non-code files]

        Heavyweight parts (model download, FAISS write) are replaced with
        lightweight stubs so the test stays fast and offline.  The extraction
        pipeline (Gates 1+2) runs with real code.

        Pre-fix failure (current HEAD): units list contains only the .py function
        unit; no non-code unit is present in metadata.json.
        """
        import sys
        import numpy as np
        from unittest.mock import MagicMock, patch

        # Stub the embedding model: encode() returns a fixed-size float32 array
        # sized for however many units are passed.
        def fake_encode(texts, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=False):
            n = len(texts) if isinstance(texts, list) else 1
            return np.ones((n, 4), dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode

        # Stub faiss: write_index is a no-op, IndexFlatIP accepts add() silently.
        # faiss is imported locally inside build_semantic_index, so we patch
        # sys.modules to intercept the `import faiss` statement.
        # The mock must have __spec__ set so that importlib.util.find_spec("faiss")
        # doesn't raise ValueError (datasets.search does this check at import time).
        import importlib.util as _ilu
        mock_faiss_index = MagicMock()
        mock_faiss_index.shape = (0, 4)
        mock_faiss_mod = MagicMock()
        mock_faiss_mod.__spec__ = _ilu.spec_from_loader("faiss", loader=None)
        mock_faiss_mod.IndexFlatIP.return_value = mock_faiss_index
        mock_faiss_mod.write_index.return_value = None

        monkeypatch.setitem(sys.modules, "faiss", mock_faiss_mod)

        with patch("tldr.semantic.get_model", return_value=mock_model):
            from tldr.semantic import build_semantic_index

            count = build_semantic_index(
                str(mixed_fixture),
                lang="python",
                show_progress=False,
                respect_ignore=False,
            )

        # Gate: build_semantic_index must have indexed at least the .py unit;
        # if it returns 0 units we can't make a meaningful assertion about non-code.
        assert count >= 1, (
            f"build_semantic_index returned 0 — pipeline produced no units at all "
            f"(not even the .py function).  Check fixture or pipeline wiring."
        )

        # Inspect the written metadata.json (written before faiss write_index)
        metadata_path = (
            mixed_fixture / ".tldr" / "cache" / "semantic" / "metadata.json"
        )
        assert metadata_path.exists(), (
            f"metadata.json not written to {metadata_path}. "
            f"build_semantic_index may have returned early."
        )

        metadata = json.loads(metadata_path.read_text())
        indexed_paths = [u["file"] for u in metadata.get("units", [])]

        non_code_extensions = {".sh", ".md", ".toml", ".yaml", ".yml"}
        non_code_indexed = [
            p for p in indexed_paths
            if Path(p).suffix in non_code_extensions
        ]

        assert len(non_code_indexed) >= 1, (
            f"metadata.json contains no non-code units.\n"
            f"Indexed paths: {indexed_paths!r}\n"
            f"Expected at least one path ending in .sh, .md, or .toml.\n"
            f"Root cause (Gate 2): _process_file_for_extraction emits nothing for "
            f"entries with empty functions/classes lists. Fix it to emit a whole-file "
            f"EmbeddingUnit when the file has non-code extension."
        )

    def test_code_files_still_indexed_alongside_non_code(self, mixed_fixture: Path):
        """Control: code files must still be indexed when non-code files are present.

        This verifies that the Gate 2 fix does not regress code-file indexing.
        Runs through the SAME extract_units_from_project pipeline as the primary test.
        """
        from tldr.semantic import extract_units_from_project

        units = extract_units_from_project(
            str(mixed_fixture),
            lang="python",
            respect_ignore=False,
        )

        code_units = [u for u in units if Path(u.file).suffix == ".py"]

        assert len(code_units) >= 1, (
            f"Expected at least 1 EmbeddingUnit for .py files but got 0. "
            f"All returned units: {[(u.file, u.name) for u in units]!r}"
        )
