"""
E2E regression tests for parse-skip, initial-index perf, and multi-language default.

These tests load the REAL embedding model and invoke the CLI via subprocess.
They are OPT-IN only: marked @pytest.mark.e2e and skipped by default.
Pass --run-e2e to exercise them.

Covered scenarios (crystallised from verified demo-scenarios.jsonl):
  MULTI-LANGUAGE CLI DEFAULT  (PRIMARY + EDGE-2):
      `tldr semantic index` with NO --lang covers both .py and .js files.
      THE regression that slipped past unit tests.
  PARSE-SKIP REINDEX  (PRIMARY):
      after editing one file, reindex shows reused N>0 and embedded K < total;
      new symbol searchable, unchanged-file symbol retained.
  NO-OP REINDEX  (EDGE-1):
      nothing changed -> embedded 0, reused N.
  DETERMINISM / FULL REUSE  (EDGE-4):
      moderate synthetic project (~30 files); second reindex with no changes ->
      embedded 0 (guards called_by ordering determinism fix).
  CROSS-FILE CALLER EDGE  (EDGE-3):
      edit only the caller file; `tldr context` shows the callee now has the
      caller in its called_by list even though the callee file was not edited.
  MODERATE-PROJECT COMPLETION  (EDGE-5):
      ~100-file Python project indexes to completion and is searchable.

Runner:
    python3 -m pytest --no-cov -p no:cacheprovider --run-e2e \\
        tests/test_parse_skip_perf_e2e.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants (mirrors test_incremental_semantic_index_e2e.py)
# ---------------------------------------------------------------------------

_REPO_ROOT = str(Path(__file__).parent.parent)

# Exact stderr summary pattern:  Semantic index: embedded M, reused N units (device=D)
_SUMMARY_RE = re.compile(
    r"Semantic index: embedded (\d+), reused (\d+) units \(device=(\w+)\)"
)

# Score delta tolerance for cross-process float nondeterminism
_SCORE_TOL = 0.01


# ---------------------------------------------------------------------------
# Helpers (same contract as test_incremental_semantic_index_e2e.py)
# ---------------------------------------------------------------------------

def _run_index(
    repo: Path,
    *,
    full: bool = False,
    lang: str | None = None,
    device: str | None = None,
    timeout: int = 360,
) -> subprocess.CompletedProcess:
    """Run `tldr semantic index <repo>` optionally with --lang / --full / --device."""
    cmd = [sys.executable, "-m", "tldr.cli", "semantic", "index", str(repo)]
    if lang is not None:
        cmd.extend(["--lang", lang])
    if full:
        cmd.append("--full")
    if device is not None:
        cmd.extend(["--device", device])
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=_REPO_ROOT, timeout=timeout
    )


def _run_search(
    repo: Path, query: str, k: int = 5, expand: bool = False, timeout: int = 120
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "tldr.cli", "semantic", "search",
        query, "--path", str(repo), "--k", str(k),
    ]
    if expand:
        cmd.append("--expand")
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=_REPO_ROOT, timeout=timeout
    )


def _run_context(
    symbol: str, repo: Path, depth: int = 2, timeout: int = 60
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "tldr.cli", "context", symbol,
        "--project", str(repo), "--depth", str(depth),
    ]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=_REPO_ROOT, timeout=timeout
    )


def _parse_summary(stderr: str) -> tuple[int, int, str]:
    """Parse 'Semantic index: embedded M, reused N units (device=D)' from stderr.
    Returns (embedded, reused, device).  Raises AssertionError on no match.
    """
    m = _SUMMARY_RE.search(stderr)
    assert m, (
        f"Expected summary line 'Semantic index: embedded M, reused N units "
        f"(device=D)' in stderr.\nActual stderr:\n{stderr!r}"
    )
    return int(m.group(1)), int(m.group(2)), m.group(3)


def _parse_search_results(stdout: str) -> list[dict]:
    """Parse JSON search output.  Returns list of result dicts."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"semantic search stdout is not valid JSON: {exc}\nstdout: {stdout!r}"
        ) from exc
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("results", [])
    return []


def _ranked_names(results: list[dict]) -> list[str]:
    """Extract qualified_name list in score-descending order."""
    sorted_results = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
    return [r.get("qualified_name", r.get("name", "")) for r in sorted_results]


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True,
                   cwd=_REPO_ROOT)


def _git_commit_all(repo: Path, message: str = "init") -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True,
                   capture_output=True)


# ---------------------------------------------------------------------------
# Test 1: MULTI-LANGUAGE CLI DEFAULT
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestMultiLanguageDefault:
    """
    THE regression: `tldr semantic index` with NO --lang must index all languages.

    Evidence (PRIMARY + EDGE-2):
    - Initial index of py+js project: embedded 9 (PRIMARY) / 2 (EDGE-2), reused 0
    - Search 'add function javascript' returns result from math.js (score ~0.600)
    - Search 'greet function' returns result from greet.py (score ~0.749)
    - EDGE-2 minimal case: serialize -> util.py, deserialize -> util.js
    """

    def test_no_lang_flag_indexes_python_and_javascript(self, tmp_path: Path):
        """tldr semantic index with NO --lang must index .py AND .js files."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        src = repo / "src"
        src.mkdir()

        # Python file
        (src / "greet.py").write_text(
            'def greet(name):\n'
            '    """Return a greeting string."""\n'
            '    return f"hello {name}"\n'
            '\n'
            'def farewell(name):\n'
            '    """Return a farewell string."""\n'
            '    greet(name)\n'
            '    return f"bye {name}"\n'
        )
        # JavaScript file
        (src / "math.js").write_text(
            'function add(a, b) { return a + b; }\n'
            'function multiply(a, b) { return a * b; }\n'
        )
        _git_commit_all(repo)

        # Index with NO --lang (this is the regression: must cover ALL languages)
        res = _run_index(repo, timeout=360)
        assert res.returncode == 0, (
            f"Multi-language index (no --lang) failed.\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
        embedded, reused, _ = _parse_summary(res.stderr)
        assert embedded >= 2, (
            f"Initial index must embed >= 2 units (at least one py + one js). "
            f"Got embedded={embedded}. If 0 were embedded from JS, this is the "
            f"multi-language regression."
        )
        assert reused == 0, (
            f"Cold-start index must reuse 0 units. Got reused={reused}."
        )

        # Python symbol must be findable
        res_py = _run_search(repo, "greet function", k=5, timeout=120)
        assert res_py.returncode == 0, (
            f"Search for Python symbol failed.\nstdout: {res_py.stdout}\n"
            f"stderr: {res_py.stderr}"
        )
        results_py = _parse_search_results(res_py.stdout)
        py_names = _ranked_names(results_py)
        assert any("greet" in n for n in py_names), (
            f"Python symbol 'greet' must appear in search results (src/greet.py). "
            f"Got: {py_names}. Multi-language index may have missed .py files."
        )
        py_files = [r.get("file", "") for r in results_py]
        assert any("greet.py" in f for f in py_files), (
            f"A result from greet.py must appear in search. Files seen: {py_files}"
        )

        # JavaScript symbol must be findable — THIS IS THE KEY REGRESSION CHECK
        res_js = _run_search(repo, "add function javascript", k=5, timeout=120)
        assert res_js.returncode == 0, (
            f"Search for JavaScript symbol failed.\nstdout: {res_js.stdout}\n"
            f"stderr: {res_js.stderr}"
        )
        results_js = _parse_search_results(res_js.stdout)
        js_files = [r.get("file", "") for r in results_js]
        assert any("math.js" in f for f in js_files), (
            f"JavaScript symbol from math.js must appear in search results. "
            f"Files seen: {js_files}. "
            f"This is the multi-language regression: index with no --lang must "
            f"cover both .py and .js files."
        )

    def test_no_lang_flag_minimal_py_js(self, tmp_path: Path):
        """Minimal 2-file (serialize.py + deserialize.js) case — no --lang covers both."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "util.py").write_text(
            'def serialize(obj):\n'
            '    """Convert object to string representation."""\n'
            '    return str(obj)\n'
        )
        (repo / "util.js").write_text(
            'function deserialize(s) { return JSON.parse(s); }\n'
        )
        _git_commit_all(repo)

        res = _run_index(repo, timeout=360)
        assert res.returncode == 0, (
            f"Minimal py+js index (no --lang) failed.\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
        embedded, reused, _ = _parse_summary(res.stderr)
        # Must embed both the Python and the JavaScript unit
        assert embedded >= 2, (
            f"Minimal py+js cold-start must embed >= 2 units. Got embedded={embedded}. "
            f"If embedded=1, JS file was silently skipped — multi-language regression."
        )

        # Search for Python symbol
        res_py = _run_search(repo, "serialize", k=5, timeout=120)
        assert res_py.returncode == 0
        results_py = _parse_search_results(res_py.stdout)
        py_files = [r.get("file", "") for r in results_py]
        assert any("util.py" in f for f in py_files), (
            f"util.py result must appear for 'serialize' search. Files: {py_files}"
        )

        # Search for JavaScript symbol — key check
        res_js = _run_search(repo, "deserialize", k=5, timeout=120)
        assert res_js.returncode == 0
        results_js = _parse_search_results(res_js.stdout)
        js_files = [r.get("file", "") for r in results_js]
        assert any("util.js" in f for f in js_files), (
            f"util.js result must appear for 'deserialize' search. Files: {js_files}. "
            f"Multi-language default regression: no --lang must index .js files."
        )


# ---------------------------------------------------------------------------
# Test 2: PARSE-SKIP REINDEX
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestParseSkipReindex:
    """
    PARSE-SKIP: editing one file reindexes only that file's units.

    Evidence (PRIMARY):
    - Initial: embedded 9, reused 0
    - After appending shout() to greet.py: embedded 1, reused 9 (K=1 << M=9)
    - New symbol 'shout' searchable; unchanged 'multiply' (math.js) still present
    """

    def test_parse_skip_reindexes_only_changed_file(self, tmp_path: Path):
        """After editing one file, reindex embeds K < total and reuses N > 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        src = repo / "src"
        src.mkdir()

        (src / "greet.py").write_text(
            'def greet(name):\n'
            '    """Return greeting string."""\n'
            '    return f"hello {name}"\n'
            '\n'
            'def farewell(name):\n'
            '    """Return farewell string."""\n'
            '    greet(name)\n'
            '    return f"bye {name}"\n'
        )
        (src / "math.js").write_text(
            'function add(a, b) { return a + b; }\n'
            'function multiply(a, b) { return a * b; }\n'
        )
        (src / "config.py").write_text(
            'class Config:\n'
            '    def __init__(self):\n'
            '        self.value = 42\n'
            '    def load(self):\n'
            '        """Load configuration value."""\n'
            '        return self.value\n'
        )
        _git_commit_all(repo)

        # Initial full index
        res_init = _run_index(repo, timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )
        embedded_init, reused_init, _ = _parse_summary(res_init.stderr)
        assert embedded_init >= 3, (
            f"Initial index must embed >= 3 units. Got embedded={embedded_init}."
        )
        assert reused_init == 0, (
            f"Cold-start must reuse 0. Got reused={reused_init}."
        )
        total_units = embedded_init

        # Edit greet.py: append a new function
        with open(src / "greet.py", "a") as f:
            f.write(
                '\ndef shout(name):\n'
                '    """Shout a greeting loudly."""\n'
                '    return greet(name).upper()\n'
            )

        # Reindex after edit
        res_reindex = _run_index(repo, timeout=360)
        assert res_reindex.returncode == 0, (
            f"Reindex after edit failed.\nstdout: {res_reindex.stdout}\n"
            f"stderr: {res_reindex.stderr}"
        )
        embedded_re, reused_re, _ = _parse_summary(res_reindex.stderr)

        # Parse-skip invariant: reused N > 0 (not a full rebuild)
        assert reused_re > 0, (
            f"After editing one file, reindex must reuse N > 0 units "
            f"(unchanged files skipped). Got reused={reused_re}. "
            f"Parse-skip is not active."
        )
        # Embedded count must be less than total (only changed-file units re-embedded)
        total_after = embedded_re + reused_re
        assert embedded_re < total_after, (
            f"After one-file edit, embedded ({embedded_re}) must be < total "
            f"({total_after}). Parse-skip must have skipped unchanged files."
        )

        # New symbol must be searchable
        res_new = _run_search(repo, "shout function", k=5, timeout=120)
        assert res_new.returncode == 0
        results_new = _parse_search_results(res_new.stdout)
        new_names = _ranked_names(results_new)
        assert any("shout" in n for n in new_names), (
            f"New function 'shout' must be searchable after reindex. Got: {new_names}"
        )

        # Unchanged JS symbol must still be present (parse-skip must not evict it)
        res_unchanged = _run_search(repo, "multiply function", k=5, timeout=120)
        assert res_unchanged.returncode == 0
        results_unchanged = _parse_search_results(res_unchanged.stdout)
        unchanged_files = [r.get("file", "") for r in results_unchanged]
        assert any("math.js" in f for f in unchanged_files), (
            f"Unchanged JS symbol 'multiply' (math.js) must still be searchable "
            f"after reindex. Files seen: {unchanged_files}. "
            f"Parse-skip must not evict unchanged-file units from the index."
        )


# ---------------------------------------------------------------------------
# Test 3: NO-OP REINDEX
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestNoOpReindex:
    """
    NO-OP: reindex with nothing changed must produce embedded 0, reused N.

    Evidence (EDGE-1):
    - Setup: alpha + beta in logic.py, initial index
    - No-op reindex: embedded 0, reused 2
    - Both symbols still searchable after no-op
    """

    def test_noop_reindex_embeds_zero_reuses_all(self, tmp_path: Path):
        """Reindex with nothing changed must report embedded 0, reused N."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "logic.py").write_text(
            'def alpha():\n'
            '    """Return the alpha value."""\n'
            '    return 1\n'
            '\n'
            'def beta():\n'
            '    """Call alpha and return result."""\n'
            '    return alpha()\n'
        )
        _git_commit_all(repo)

        # Initial index
        res_init = _run_index(repo, lang="python", timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )
        embedded_init, _, _ = _parse_summary(res_init.stderr)
        assert embedded_init >= 2, (
            f"Initial index must embed >= 2 units (alpha, beta). Got {embedded_init}."
        )

        # No-op reindex: nothing changed
        res_noop = _run_index(repo, lang="python", timeout=360)
        assert res_noop.returncode == 0, (
            f"No-op reindex failed.\nstdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )
        embedded_noop, reused_noop, _ = _parse_summary(res_noop.stderr)

        assert embedded_noop == 0, (
            f"No-op reindex (nothing changed) must embed 0 units. "
            f"Got embedded={embedded_noop}. All units should be reused from cache."
        )
        assert reused_noop >= 2, (
            f"No-op reindex must reuse >= 2 units (alpha and beta). "
            f"Got reused={reused_noop}."
        )

        # Both symbols must still be searchable after no-op reindex
        res_alpha = _run_search(repo, "alpha function", k=5, timeout=120)
        assert res_alpha.returncode == 0
        results_alpha = _parse_search_results(res_alpha.stdout)
        alpha_names = _ranked_names(results_alpha)
        assert any("alpha" in n for n in alpha_names), (
            f"'alpha' must be searchable after no-op reindex. Got: {alpha_names}"
        )

        res_beta = _run_search(repo, "beta function", k=5, timeout=120)
        assert res_beta.returncode == 0
        results_beta = _parse_search_results(res_beta.stdout)
        beta_names = _ranked_names(results_beta)
        assert any("beta" in n for n in beta_names), (
            f"'beta' must be searchable after no-op reindex. Got: {beta_names}"
        )


# ---------------------------------------------------------------------------
# Test 4: DETERMINISM / FULL REUSE (moderate synthetic project)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestDeterminismFullReuse:
    """
    DETERMINISM: moderate synthetic project, second reindex reuses ALL units.

    This guards the called_by ordering determinism fix.  Uses ~30 synthetic
    Python files (not the full llm-tldr repo) to keep it bounded.

    Evidence (EDGE-4, extrapolated to bounded project):
    - Second reindex: embedded 0, reused N (all units)
    """

    def test_second_reindex_reuses_all_units(self, tmp_path: Path):
        """Second reindex with no changes must report embedded 0 (determinism guard)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        src = repo / "src"
        src.mkdir()

        # Create 30 synthetic Python modules with inter-file calls to exercise
        # called_by ordering (the determinism regression surface)
        for i in range(30):
            caller = f"module_{(i + 1) % 30}"
            (src / f"module_{i}.py").write_text(
                f'def func_{i}(x):\n'
                f'    """Compute value for module {i}."""\n'
                f'    return x + {i}\n'
                f'\n'
                f'class Class_{i}:\n'
                f'    def method_{i}(self):\n'
                f'        """Run method {i}."""\n'
                f'        return func_{i}(0)\n'
                f'\n'
                f'def caller_{i}(x):\n'
                f'    """Call into {caller}."""\n'
                f'    from {caller} import func_{(i + 1) % 30}\n'
                f'    return func_{(i + 1) % 30}(x)\n'
            )
        _git_commit_all(repo)

        # First index
        res1 = _run_index(repo, lang="python", timeout=480)
        assert res1.returncode == 0, (
            f"First index failed.\nstdout: {res1.stdout}\nstderr: {res1.stderr}"
        )
        embedded1, reused1, _ = _parse_summary(res1.stderr)
        assert embedded1 >= 30, (
            f"First index must embed >= 30 units (one per module minimum). "
            f"Got embedded={embedded1}."
        )
        assert reused1 == 0, f"First index must reuse 0. Got reused={reused1}."
        total = embedded1

        # Second index immediately — nothing changed
        res2 = _run_index(repo, lang="python", timeout=480)
        assert res2.returncode == 0, (
            f"Second (determinism) reindex failed.\n"
            f"stdout: {res2.stdout}\nstderr: {res2.stderr}"
        )
        embedded2, reused2, _ = _parse_summary(res2.stderr)

        # CORE INVARIANT: all units reused — called_by ordering is deterministic
        assert embedded2 == 0, (
            f"Second reindex with no changes must embed 0 units (determinism). "
            f"Got embedded={embedded2} out of {total} total. "
            f"Non-zero means called_by ordering is non-deterministic (text_hash "
            f"flipping between runs) — the ordering determinism regression."
        )
        assert reused2 == total, (
            f"Second reindex must reuse ALL {total} units. "
            f"Got reused={reused2}."
        )

        # Index must still be searchable after the no-op second reindex
        res_search = _run_search(repo, "compute value module", k=5, timeout=120)
        assert res_search.returncode == 0
        results = _parse_search_results(res_search.stdout)
        assert len(results) >= 1, (
            f"Index must be searchable after determinism reindex. Got 0 results."
        )


# ---------------------------------------------------------------------------
# Test 5: CROSS-FILE CALLER EDGE
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestCrossFileCallerEdge:
    """
    CROSS-FILE CALLER: editing only the caller file correctly updates callee metadata.

    Evidence (EDGE-3):
    - Initial: helper has no callers; main.py does not call helper
    - Edit main.py to import and call helper(); helpers.py untouched
    - Reindex: embedded 2, reused 1
    - `tldr context helper` shows main as caller of helper
    - untouched.py symbol still searchable
    - helpers.py symbol still searchable
    """

    def test_cross_file_caller_appears_after_reindex(self, tmp_path: Path):
        """After editing only the caller, callee must show the caller in its context."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        # helpers.py — will NOT be edited
        (repo / "helpers.py").write_text(
            'def helper():\n'
            '    """Return a helper result."""\n'
            '    return 99\n'
        )
        # main.py — initial state: does NOT call helper
        (repo / "main.py").write_text(
            'def main():\n'
            '    """Entry point."""\n'
            '    return 0\n'
        )
        # untouched.py — never edited; must remain searchable throughout
        (repo / "untouched.py").write_text(
            'def untouched():\n'
            '    """This function is never modified."""\n'
            '    return "still here"\n'
        )
        _git_commit_all(repo)

        # Initial index
        res_init = _run_index(repo, lang="python", timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )

        # Confirm helper has NO callers before the edit (via tldr context)
        res_ctx_before = _run_context("helper", repo, depth=2, timeout=60)
        # context may or may not return exit 0; what matters is 'main' is not listed
        ctx_before_output = res_ctx_before.stdout + res_ctx_before.stderr
        assert "main" not in ctx_before_output.lower() or (
            # allow the word "main" only if it appears as an unrelated symbol
            "main.main" not in ctx_before_output and
            "calls: helper" not in ctx_before_output
        ), (
            f"Before the edit, 'main' must NOT appear as a caller of 'helper'.\n"
            f"Context output: {ctx_before_output!r}"
        )

        # Edit ONLY main.py to call helper; helpers.py bytes unchanged
        (repo / "main.py").write_text(
            'from helpers import helper\n'
            '\n'
            'def main():\n'
            '    """Entry point calling helper."""\n'
            '    return helper()\n'
        )

        # Reindex
        res_reindex = _run_index(repo, lang="python", timeout=360)
        assert res_reindex.returncode == 0, (
            f"Reindex after caller edit failed.\n"
            f"stdout: {res_reindex.stdout}\nstderr: {res_reindex.stderr}"
        )
        embedded_re, reused_re, _ = _parse_summary(res_reindex.stderr)
        # At least main.py (changed) and helper (called_by changed) must be re-embedded
        assert embedded_re >= 1, (
            f"At least 1 unit must be re-embedded after caller edit. "
            f"Got embedded={embedded_re}."
        )

        # tldr context helper must now show main as a caller
        res_ctx_after = _run_context("helper", repo, depth=2, timeout=60)
        ctx_after_output = res_ctx_after.stdout + res_ctx_after.stderr
        assert "main" in ctx_after_output.lower(), (
            f"After reindex, 'main' must appear as a caller of 'helper' in "
            f"`tldr context`. Output:\n{ctx_after_output!r}\n"
            f"Cross-file caller edge must be present even though helpers.py was "
            f"not directly edited."
        )

        # Untouched symbol must still be searchable
        res_untouched = _run_search(repo, "untouched function", k=5, timeout=120)
        assert res_untouched.returncode == 0
        results_untouched = _parse_search_results(res_untouched.stdout)
        untouched_files = [r.get("file", "") for r in results_untouched]
        assert any("untouched.py" in f for f in untouched_files), (
            f"untouched.py symbol must still be searchable after cross-file reindex. "
            f"Files: {untouched_files}"
        )

        # helpers.py must still be searchable
        res_helper = _run_search(repo, "helper function result", k=5, timeout=120)
        assert res_helper.returncode == 0
        results_helper = _parse_search_results(res_helper.stdout)
        helper_files = [r.get("file", "") for r in results_helper]
        assert any("helpers.py" in f for f in helper_files), (
            f"helpers.py symbol must still be searchable after reindex. "
            f"Files: {helper_files}"
        )


# ---------------------------------------------------------------------------
# Test 6: MODERATE-PROJECT COMPLETION
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestModerateProjectCompletion:
    """
    MODERATE-PROJECT: 100-file Python project indexes to completion and is searchable.

    Evidence (EDGE-5):
    - 100 files created; index exit 0; embedded 400, reused 0
    - Search for 'func_50' -> module_50.py (score 0.745)
    - Search for 'Class_99 method' -> module_99.py (score 0.757)
    - Completed in ~13 seconds without hanging
    """

    def test_hundred_file_project_indexes_and_is_searchable(self, tmp_path: Path):
        """100-file Python project must index to completion and be searchable."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        src = repo / "src"
        src.mkdir()

        for i in range(100):
            (src / f"module_{i}.py").write_text(
                f'def func_{i}(x):\n'
                f'    """Compute output for module {i}."""\n'
                f'    return x + {i}\n'
                f'\n'
                f'class Class_{i}:\n'
                f'    def method_{i}(self):\n'
                f'        """Run method for class {i}."""\n'
                f'        return func_{i}(0)\n'
            )
        _git_commit_all(repo)

        # Index must complete without hanging (generous timeout: 600s)
        res = _run_index(repo, lang="python", timeout=600)
        assert res.returncode == 0, (
            f"100-file index failed or timed out.\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
        embedded, reused, _ = _parse_summary(res.stderr)
        assert embedded >= 200, (
            f"100-file index must embed >= 200 units (2 per file minimum). "
            f"Got embedded={embedded}."
        )
        assert reused == 0, (
            f"Cold-start index must reuse 0. Got reused={reused}."
        )

        # Symbol from the middle of the project must be findable
        res_mid = _run_search(repo, "func_50", k=5, timeout=120)
        assert res_mid.returncode == 0
        results_mid = _parse_search_results(res_mid.stdout)
        mid_files = [r.get("file", "") for r in results_mid]
        assert any("module_50.py" in f for f in mid_files), (
            f"Symbol from module_50.py must be searchable. Files: {mid_files}"
        )

        # Symbol from the end of the project must be findable
        res_end = _run_search(repo, "Class_99 method", k=5, timeout=120)
        assert res_end.returncode == 0
        results_end = _parse_search_results(res_end.stdout)
        end_files = [r.get("file", "") for r in results_end]
        assert any("module_99.py" in f for f in end_files), (
            f"Symbol from module_99.py must be searchable. Files: {end_files}"
        )
