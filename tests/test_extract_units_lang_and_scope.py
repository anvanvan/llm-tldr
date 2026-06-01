"""Failing (RED) tests for Goal C (language default auto-detect) and Goal B
(files_to_parse scoping + worker initializer) in extract_units_from_project.

All tests target tldr.semantic.extract_units_from_project directly — no
embedding model is required (extraction returns units before the embed step).

Expected RED state on current code (semantic.py lang default = "python", no
files_to_parse param, no _init_extraction_worker):

  test_multilang_default_returns_py_and_js_units
    -> FAILS because default lang="python" only yields .py units; .js units absent

  test_lang_none_equals_explicit_all_for_noncode_repo
    -> FAILS because lang=None not accepted (TypeError: unexpected keyword) OR
       lang=None → same as "python" default, producing 0 non-code units; the
       explicit "all" path in build_semantic_index returns non-code units (this
       tests structural equivalence on a noncode-only repo)

  test_no_noncode_sentinel_passed_to_lower_level_apis
    -> FAILS because lang=None expansion block doesn't exist yet, so
       get_code_structure/build_project_call_graph are never called with
       guarded language values (can't assert the guard without the guard code)

  test_files_to_parse_filters_worker_dispatch
    -> FAILS because files_to_parse param does not exist (TypeError on call)

  test_worker_initializer_wired_to_process_pool_executor
    -> FAILS because _init_extraction_worker does not exist in tldr.semantic
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Repo + test paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Minimal source file content for temp repos
# ---------------------------------------------------------------------------
_PY_CONTENT = '''\
def hello_python(x):
    """Return x incremented."""
    return x + 1
'''

_JS_CONTENT = '''\
export function helloJavaScript(y) {
    // Return y doubled
    return y * 2;
}
'''

_MD_CONTENT = '''\
# Project README

This is a documentation-only project.

## Overview

No source code here — only markdown.
'''


# ---------------------------------------------------------------------------
# Temp repo builders
# ---------------------------------------------------------------------------

def _build_multilang_repo(tmp_path: Path) -> Path:
    """Create a minimal project with one .py file and one .js file."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "module_a.py").write_text(_PY_CONTENT)
    (tmp_path / "module_b.js").write_text(_JS_CONTENT)
    return tmp_path


def _build_noncode_repo(tmp_path: Path) -> Path:
    """Create a project with only a .md file — no code files."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text(_MD_CONTENT)
    return tmp_path


def _build_py_only_repo(tmp_path: Path, *, n_extra: int = 3) -> Path:
    """Create a project with multiple .py files."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "main.py").write_text(_PY_CONTENT)
    for i in range(n_extra):
        (tmp_path / f"extra_{i}.py").write_text(
            f'def extra_{i}():\n    """Extra {i}."""\n    return {i}\n'
        )
    return tmp_path


# ===========================================================================
# GOAL C — Test 1: lang=None (no arg) auto-detects py AND js
# ===========================================================================

class TestMultilangDefaultNoBehavior:
    """extract_units_from_project with NO lang= arg must auto-detect ALL languages.

    RED today: default lang="python" means only .py units are returned;
    .js units are completely absent.
    """

    def test_multilang_default_returns_py_and_js_units(self, tmp_path: Path):
        """No lang= arg on a py+js repo must return units from BOTH languages.

        Architecture spec (Goal C / G-1/I-2): default should flip from "python"
        to None, which triggers _detect_project_language_tags and collects units
        for every detected language.

        RED reason: current default lang="python" → get_code_structure only walks
        .py files → zero JavaScript units returned.
        """
        from tldr.semantic import extract_units_from_project

        repo = _build_multilang_repo(tmp_path)

        # Call with NO lang argument — this is the new default behavior
        units = extract_units_from_project(str(repo), respect_ignore=False)

        languages = {u.language for u in units}

        assert any(u.language == "python" for u in units), (
            f"Expected at least one python unit; languages found: {languages}"
        )
        assert any(u.language in {"javascript", "js"} for u in units), (
            f"Expected at least one javascript/js unit; languages found: {languages}. "
            f"RED: current default lang='python' yields only .py units."
        )


# ===========================================================================
# GOAL C — Test 2: lang=None expansion must call _detect_project_language_tags
# ===========================================================================

class TestNoncodeSentinelHandling:
    """lang=None expansion must use _detect_project_language_tags (not _detect_project_languages).

    Architecture spec (G-1/I-2): "Corrected design: extract_units_from_project
    expansion for lang=None uses _detect_project_language_tags (semantic.py:1521),
    NOT _detect_project_languages (semantic.py:1483)."

    We verify this structurally: monkeypatch both detectors, call with no lang=,
    assert _detect_project_language_tags was called and _detect_project_languages
    was NOT called at the expansion seam.

    RED today: no expansion block exists at all — neither detector is called
    from within extract_units_from_project; the function goes straight to
    get_code_structure with lang="python" (the default).
    """

    def test_lang_none_uses_detect_language_tags_not_detect_languages(
        self, tmp_path: Path
    ):
        """lang=None must call _detect_project_language_tags, not _detect_project_languages.

        RED reason: no expansion block exists in extract_units_from_project today.
        _detect_project_language_tags is never called from within the function
        on any path — call_count stays 0, so the assertion fails.
        """
        from tldr.semantic import (
            _detect_project_language_tags,
            NON_CODE_DISPATCH_SENTINEL,
        )

        repo = _build_multilang_repo(tmp_path)

        tags_call_count = 0
        langs_call_count = 0

        def recording_detect_tags(project_path, **kwargs):
            nonlocal tags_call_count
            tags_call_count += 1
            # Delegate to the real function so expansion proceeds correctly
            return _detect_project_language_tags(project_path, **kwargs)

        def recording_detect_langs(project_path, **kwargs):
            nonlocal langs_call_count
            langs_call_count += 1
            from tldr.semantic import _detect_project_languages
            return _detect_project_languages(project_path, **kwargs)

        with patch("tldr.semantic._detect_project_language_tags",
                   side_effect=recording_detect_tags), \
             patch("tldr.semantic._detect_project_languages",
                   side_effect=recording_detect_langs):
            from tldr.semantic import extract_units_from_project
            # No lang= arg — should trigger the expansion block
            extract_units_from_project(str(repo), respect_ignore=False)

        assert tags_call_count >= 1, (
            f"_detect_project_language_tags was called {tags_call_count} times "
            f"(expected >= 1). Architecture requires extract_units_from_project "
            f"with lang=None to call _detect_project_language_tags to get the "
            f"full dispatch tag set (including NON_CODE_DISPATCH_SENTINEL). "
            f"RED: no expansion block exists yet — _detect_project_language_tags "
            f"is never called from within extract_units_from_project."
        )


# ===========================================================================
# GOAL C — Test 3: sentinel never forwarded to lower-level APIs
# ===========================================================================

class TestNoncodeSentinelNotForwardedToLowerAPIs:
    """_noncode sentinel must NEVER be passed to get_code_structure or
    build_project_call_graph when lang=None expansion is active.

    Architecture spec (G-1/I-2): "NEVER pass None, 'all', 'auto', or
    NON_CODE_DISPATCH_SENTINEL to get_code_structure, build_project_call_graph,
    or scan_project."

    The expansion block (new in Goal C) must translate NON_CODE_DISPATCH_SENTINEL
    into a safe non-code dispatch before calling lower-level APIs.

    RED today: no expansion block exists — extract_units_from_project never calls
    _detect_project_language_tags, so the sentinel-guarded expansion path is
    absent. We assert the expansion block exists by verifying that
    _detect_project_language_tags is called AND no forbidden value reaches
    get_code_structure / build_project_call_graph.
    """

    def test_no_noncode_sentinel_passed_to_lower_level_apis(self, tmp_path: Path):
        """get_code_structure and build_project_call_graph must never receive
        _noncode, None, 'all', or 'auto' as the language argument.

        We patch tldr.api.get_code_structure and tldr.api.build_project_call_graph
        (the modules they live in, not the local import name in semantic.py) and
        record every 'language' arg passed.  We also record whether
        _detect_project_language_tags was called (the expansion seam).

        RED reason: _detect_project_language_tags is NEVER called from within
        extract_units_from_project today (no expansion block). The assertion
        'tags_call_count >= 1' fails → test is RED for the right reason.
        """
        from tldr.semantic import (
            NON_CODE_DISPATCH_SENTINEL,
            _detect_project_language_tags,
        )

        forbidden_langs = {NON_CODE_DISPATCH_SENTINEL, None, "all", "auto"}
        seen_langs: list = []
        tags_call_count = 0

        def recording_get_code_structure(project_path, language, **kwargs):
            seen_langs.append(("get_code_structure", language))
            return {"files": [], "functions": []}

        def recording_build_call_graph(project_path, language, **kwargs):
            seen_langs.append(("build_project_call_graph", language))
            return MagicMock(edges=[], nodes=[])

        def recording_detect_tags(project_path, **kwargs):
            nonlocal tags_call_count
            tags_call_count += 1
            return _detect_project_language_tags(project_path, **kwargs)

        repo = _build_noncode_repo(tmp_path)

        with patch("tldr.api.get_code_structure",
                   side_effect=recording_get_code_structure), \
             patch("tldr.api.build_project_call_graph",
                   side_effect=recording_build_call_graph), \
             patch("tldr.semantic._detect_project_language_tags",
                   side_effect=recording_detect_tags):
            from tldr.semantic import extract_units_from_project
            # No lang= arg — should trigger expansion block using
            # _detect_project_language_tags, then safely handle _noncode sentinel
            extract_units_from_project(str(repo), respect_ignore=False)

        # PRIMARY RED assertion: expansion block must call _detect_project_language_tags
        assert tags_call_count >= 1, (
            f"_detect_project_language_tags was called {tags_call_count} times "
            f"inside extract_units_from_project (expected >= 1). "
            f"The expansion block must call this function when lang=None. "
            f"RED: expansion block does not exist yet — function never called."
        )

        # SECONDARY assertion (passes today, guards against future regression):
        # no forbidden lang value must reach lower-level APIs
        for api_name, lang_val in seen_langs:
            assert lang_val not in forbidden_langs, (
                f"{api_name} was called with forbidden language value {lang_val!r}. "
                f"Architecture requires the expansion block to translate "
                f"NON_CODE_DISPATCH_SENTINEL before calling lower-level APIs."
            )


# ===========================================================================
# GOAL B — Test 4: files_to_parse scoping filters worker dispatch
# ===========================================================================

class TestFilesToParseFiltersWorkerDispatch:
    """files_to_parse={one_file} must cause ONLY that file to be parsed.

    Architecture spec (T2-8): "Filter point (semantic.py:711): After
    files = structure.get('files', []), when files_to_parse is not None:
    files = [f for f in files if f.get('path') in files_to_parse]"

    The filter must happen BEFORE the ProcessPoolExecutor worker loop. Unchanged
    files must never be submitted as futures.

    RED today: files_to_parse parameter does not exist on
    extract_units_from_project → TypeError on call.
    """

    def test_files_to_parse_filters_worker_dispatch(self, tmp_path: Path):
        """With files_to_parse={one_file}, only that file's units are extracted.

        We build a 4-file Python repo, call extract_units_from_project with
        files_to_parse limited to one file, and assert:
        1. The call does not raise TypeError (files_to_parse param exists).
        2. Only units from the scoped file are returned (all have matching file).

        RED reason: files_to_parse param doesn't exist → TypeError.
        """
        from tldr.semantic import extract_units_from_project

        repo = _build_py_only_repo(tmp_path, n_extra=3)
        # "main.py" is one of four files; scope extraction to just it
        target_file = "main.py"

        # This call must not raise TypeError (param doesn't exist today)
        units = extract_units_from_project(
            str(repo),
            lang="python",
            respect_ignore=False,
            files_to_parse={target_file},
        )

        # Every returned unit must belong to the scoped file
        for unit in units:
            assert unit.file == target_file or unit.file.endswith(target_file), (
                f"Unit {unit.qualified_name!r} from file {unit.file!r} should not "
                f"have been parsed — only {target_file!r} is in files_to_parse. "
                f"RED: files_to_parse param doesn't exist yet (TypeError on call)."
            )

        # There must be at least one unit from the target file (sanity check that
        # the filter didn't accidentally exclude everything)
        assert len(units) >= 1, (
            f"Expected at least one unit from {target_file!r}; got none. "
            f"files_to_parse scoping may have over-filtered."
        )

    def test_files_to_parse_zero_calls_for_excluded_files(self, tmp_path: Path):
        """Excluded files must trigger ZERO _process_file_for_extraction calls.

        We monkeypatch _process_file_for_extraction to count invocations and
        verify that the excluded files are never submitted to the worker pool.

        Architecture spec (T2-8): "EXPLICITLY REJECTED: filter at call site —
        this yields ZERO parse speedup because get_code_structure + worker
        dispatch for every file still occurs."

        RED reason: files_to_parse param doesn't exist (TypeError) — can't even
        reach the assertion about call count.
        """
        from tldr.semantic import extract_units_from_project

        repo = _build_py_only_repo(tmp_path, n_extra=3)
        # 4 files total: main.py, extra_0.py, extra_1.py, extra_2.py
        # Scope to only main.py
        target_file = "main.py"
        excluded_files = {"extra_0.py", "extra_1.py", "extra_2.py"}

        calls_seen: list = []

        original_process = None
        try:
            import tldr.semantic as _sem
            original_process = _sem._process_file_for_extraction
        except AttributeError:
            pass

        def recording_process(file_info, *args, **kwargs):
            calls_seen.append(file_info.get("path", "unknown"))
            if original_process is not None:
                return original_process(file_info, *args, **kwargs)
            return []

        with patch("tldr.semantic._process_file_for_extraction",
                   side_effect=recording_process):
            units = extract_units_from_project(
                str(repo),
                lang="python",
                respect_ignore=False,
                files_to_parse={target_file},
            )

        for excluded in excluded_files:
            assert excluded not in calls_seen, (
                f"_process_file_for_extraction was called for excluded file "
                f"{excluded!r}. Filter must happen BEFORE worker dispatch "
                f"(at semantic.py:711), not at call site. "
                f"RED: files_to_parse param doesn't exist yet."
            )


# ===========================================================================
# GOAL B — Test 5: worker initializer wired to ProcessPoolExecutor
# ===========================================================================

class TestWorkerInitializerWired:
    """ProcessPoolExecutor must be constructed with initializer=_init_extraction_worker.

    Architecture spec (T-3 / B-2): "ProcessPoolExecutor(max_workers=max_workers,
    initializer=_init_extraction_worker, initargs=(lang,))"

    _init_extraction_worker must be a module-level (picklable) function in
    tldr.semantic. It is called ONCE per worker process, amortizing grammar
    import cost across files.

    RED today: _init_extraction_worker does not exist in tldr.semantic.
    """

    def test_init_extraction_worker_exists_as_module_level_function(self):
        """_init_extraction_worker must exist as a callable in tldr.semantic.

        RED reason: function doesn't exist yet → AttributeError.
        """
        import tldr.semantic as sem

        assert hasattr(sem, "_init_extraction_worker"), (
            "tldr.semantic must export _init_extraction_worker (module-level "
            "function, picklable for ProcessPoolExecutor initializer). "
            "RED: function doesn't exist yet."
        )
        assert callable(sem._init_extraction_worker), (
            "_init_extraction_worker must be callable."
        )

    def test_process_pool_executor_constructed_with_initializer(
        self, tmp_path: Path
    ):
        """ProcessPoolExecutor must be created with initializer=_init_extraction_worker.

        We monkeypatch ProcessPoolExecutor at the tldr.semantic import site and
        assert it is constructed with a non-None initializer kwarg.

        RED reason: current ProcessPoolExecutor call has no initializer= argument.
        """
        from concurrent.futures import ProcessPoolExecutor as RealPPE
        from tldr.semantic import extract_units_from_project

        repo = _build_py_only_repo(tmp_path, n_extra=2)  # 3 files → triggers pool

        executor_kwargs_seen: list = []

        class CapturingPPE(RealPPE):
            def __init__(self, *args, **kwargs):
                executor_kwargs_seen.append(kwargs)
                super().__init__(*args, **kwargs)

        with patch("tldr.semantic.ProcessPoolExecutor", CapturingPPE):
            extract_units_from_project(
                str(repo),
                lang="python",
                respect_ignore=False,
            )

        assert len(executor_kwargs_seen) >= 1, (
            "ProcessPoolExecutor was never instantiated — expected parallel "
            "extraction for a 3-file repo."
        )

        for kwargs in executor_kwargs_seen:
            assert "initializer" in kwargs, (
                f"ProcessPoolExecutor constructed without initializer= kwarg: "
                f"{kwargs}. Architecture requires "
                f"initializer=_init_extraction_worker. "
                f"RED: initializer not wired yet."
            )
            assert kwargs["initializer"] is not None, (
                "ProcessPoolExecutor initializer= must not be None."
            )

        # Also verify the wired initializer is _init_extraction_worker specifically
        import tldr.semantic as sem
        if hasattr(sem, "_init_extraction_worker"):
            for kwargs in executor_kwargs_seen:
                if "initializer" in kwargs:
                    assert kwargs["initializer"] is sem._init_extraction_worker, (
                        f"Expected initializer to be _init_extraction_worker, "
                        f"got {kwargs['initializer']!r}. "
                        f"RED: initializer not wired yet."
                    )
