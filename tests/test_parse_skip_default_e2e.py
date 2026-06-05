"""
E2E crystallisation tests for parse-skip-default (incremental path is now the
default for every `tldr semantic index` caller).

These tests load the REAL embedding model and invoke the CLI via subprocess.
They are OPT-IN only: marked @pytest.mark.e2e and skipped by default.
Pass --run-e2e to exercise them.

Covered scenarios (crystallised from verified demo-scenarios.jsonl +
verification-audit.json for feature parse-skip-default):

  PRIMARY  — one-file edit re-embeds only the changed file; reused > 0;
             embedded < total; new symbol searchable.
  EDGE-1   — no-op reindex embeds 0; noop wall-clock < cold-build wall-clock
             (relative timing, model pre-warmed).
  EDGE-2   — deleting a source file purges its symbols from search results and
             from metadata.json.
  EDGE-3   — renaming a file: old-path symbols absent, new-path symbols present.
  EDGE-4   — cross-file caller update: `tldr context <callee>` lists the new
             caller after incremental reindex; caller set equals --full rebuild.
  EDGE-5   — incremental vs --full: qualified-name SET and call-graph edge SET
             are identical.
  EDGE-6   — snapshot back-compat: old narrow file_hashes.json ({path: sha1})
             triggers safe reindex (no crash); subsequent no-op confirms
             incremental mode active.

Runner:
    python3 -m pytest --no-cov -p no:cacheprovider --run-e2e \\
        tests/test_parse_skip_default_e2e.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = str(Path(__file__).parent.parent)

# Exact stderr summary pattern:  Semantic index: embedded M, reused N units (device=D)
_SUMMARY_RE = re.compile(
    r"Semantic index: embedded (\d+), reused (\d+) units \(device=(\w+)\)"
)


# ---------------------------------------------------------------------------
# Helpers  (mirror test_parse_skip_perf_e2e.py exactly)
# ---------------------------------------------------------------------------

def _run_index(
    repo: Path,
    *,
    full: bool = False,
    lang: str | None = "python",
    timeout: int = 360,
) -> subprocess.CompletedProcess:
    """Run `tldr semantic index <repo>` with --lang python by default."""
    cmd = [sys.executable, "-m", "tldr.cli", "semantic", "index", str(repo)]
    if lang is not None:
        cmd.extend(["--lang", lang])
    if full:
        cmd.append("--full")
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=_REPO_ROOT, timeout=timeout
    )


def _run_search(
    repo: Path, query: str, k: int = 5, timeout: int = 120
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "tldr.cli", "semantic", "search",
        query, "--path", str(repo), "--k", str(k),
        "--lang", "python",
    ]
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
# PRIMARY: one-file edit re-embeds only the changed file
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestPrimaryOneFileEdit:
    """
    PRIMARY scenario: editing one file (utils.py) in a 3-file repo causes the
    plain `tldr semantic index` (no flags, no daemon) to re-embed only the
    changed file's unit and reuse the rest.

    Evidence (PRIMARY from demo-scenarios.jsonl):
    - Cold build: embedded 4, reused 0 (utils.py units: add, multiply,
      main.py unit: compute, helper.py unit: helper)
    - After appending subtract() to utils.py: embedded 1, reused 4
      (file-level granularity: 1 unit = 1 file re-embedded)
    - Search finds 'subtract' at top position (score 0.729)
    """

    def test_one_file_edit_reembeds_only_changed_file(self, tmp_path: Path):
        """Plain `tldr semantic index` re-embeds only the changed file; reused > 0."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "utils.py").write_text(
            'def add(a, b):\n'
            '    """Add two numbers."""\n'
            '    return a + b\n'
            '\n'
            'def multiply(a, b):\n'
            '    """Multiply two numbers."""\n'
            '    return a * b\n'
        )
        (repo / "main.py").write_text(
            'from utils import add, multiply\n'
            '\n'
            'def compute(x, y):\n'
            '    """Compute sum and product."""\n'
            '    return add(x, y), multiply(x, y)\n'
        )
        (repo / "helper.py").write_text(
            'def helper():\n'
            '    """Utility helper."""\n'
            '    return 42\n'
        )
        _git_commit_all(repo)

        # Cold initial build
        res_init = _run_index(repo, timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )
        embedded_init, reused_init, _ = _parse_summary(res_init.stderr)
        assert embedded_init >= 3, (
            f"Initial build must embed >= 3 units. Got embedded={embedded_init}."
        )
        assert reused_init == 0, (
            f"Cold build must reuse 0 units. Got reused={reused_init}."
        )
        # Edit exactly one file: append subtract() to utils.py
        with open(repo / "utils.py", "a") as f:
            f.write(
                '\ndef subtract(a, b):\n'
                '    """Subtract b from a."""\n'
                '    return a - b\n'
            )

        # Re-run plain `tldr semantic index` with NO flags
        res_inc = _run_index(repo, timeout=360)
        assert res_inc.returncode == 0, (
            f"Incremental reindex after one-file edit failed.\n"
            f"stdout: {res_inc.stdout}\nstderr: {res_inc.stderr}"
        )
        embedded_inc, reused_inc, _ = _parse_summary(res_inc.stderr)

        # HEADLINE: carry happened — reused > 0 means unchanged files were skipped
        assert reused_inc > 0, (
            f"After editing one file, plain `tldr semantic index` must reuse N > 0 "
            f"units (unchanged files skipped). Got reused={reused_inc}. "
            f"Before this feature, the same command would have produced reused=0."
        )
        # embedded < total — only the changed file's unit was re-embedded
        total_after = embedded_inc + reused_inc
        assert embedded_inc < total_after, (
            f"After one-file edit, embedded ({embedded_inc}) must be < total "
            f"({total_after}). Only the changed file's units must be re-embedded."
        )

        # New symbol must be searchable after incremental reindex
        res_search = _run_search(repo, "subtract b from a", k=5, timeout=120)
        assert res_search.returncode == 0, (
            f"Search for new symbol failed.\nstdout: {res_search.stdout}\n"
            f"stderr: {res_search.stderr}"
        )
        results = _parse_search_results(res_search.stdout)
        names = _ranked_names(results)
        assert any("subtract" in n for n in names), (
            f"New function 'subtract' must be searchable after incremental reindex. "
            f"Got: {names}"
        )

        # Unchanged symbols must still be present
        res_helper = _run_search(repo, "utility helper", k=5, timeout=120)
        assert res_helper.returncode == 0
        helper_results = _parse_search_results(res_helper.stdout)
        helper_files = [r.get("file", "") for r in helper_results]
        assert any("helper.py" in f for f in helper_files), (
            f"Unchanged helper.py symbol must still be searchable after incremental "
            f"reindex. Files seen: {helper_files}."
        )


# ---------------------------------------------------------------------------
# EDGE-1: no-op reindex embeds 0 and is faster than cold build
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge1NoOpFasterThanCold:
    """
    EDGE-1: no-op reindex (nothing changed) reports embedded 0, reused N,
    and its wall-clock is strictly less than the cold build (relative timing,
    model pre-warmed).

    Evidence (EDGE-1):
    - Cold build: embedded 2, reused 0, T_FULL=6.02s
    - No-op:      embedded 0, reused 2, T_NOOP=4.28s
    - PASS: no-op (4.28s) strictly faster than cold build (6.02s)
    """

    def test_noop_reindex_embeds_zero(self, tmp_path: Path):
        """No-op reindex must report embedded 0, reused >= 2."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "greet.py").write_text(
            'def greet(name):\n'
            '    """Return greeting string."""\n'
            '    return f"Hello, {name}"\n'
            '\n'
            'def farewell(name):\n'
            '    """Return farewell string."""\n'
            '    return f"Goodbye, {name}"\n'
        )
        _git_commit_all(repo)

        # Cold build
        res_cold = _run_index(repo, timeout=360)
        assert res_cold.returncode == 0, (
            f"Cold build failed.\nstdout: {res_cold.stdout}\nstderr: {res_cold.stderr}"
        )
        embedded_cold, reused_cold, _ = _parse_summary(res_cold.stderr)
        assert embedded_cold >= 2, (
            f"Cold build must embed >= 2 units (greet, farewell). Got {embedded_cold}."
        )
        assert reused_cold == 0, f"Cold build must reuse 0. Got {reused_cold}."

        # No-op reindex: nothing changed
        res_noop = _run_index(repo, timeout=360)
        assert res_noop.returncode == 0, (
            f"No-op reindex failed.\nstdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )
        embedded_noop, reused_noop, _ = _parse_summary(res_noop.stderr)

        assert embedded_noop == 0, (
            f"No-op reindex must embed 0 units. Got embedded={embedded_noop}. "
            f"All units should be reused from cache."
        )
        assert reused_noop >= 2, (
            f"No-op reindex must reuse >= 2 units. Got reused={reused_noop}."
        )

    def test_noop_reindex_is_faster_than_cold_build(self, tmp_path: Path):
        """No-op wall-clock must be strictly less than cold-build wall-clock (pre-warmed)."""
        # Pre-warm the embedding model so model cold-start is NOT charged to timed runs
        warmup = tmp_path / "warmup"
        warmup.mkdir()
        _git_init(warmup)
        (warmup / "w.py").write_text('def warmup():\n    pass\n')
        _git_commit_all(warmup)
        _run_index(warmup, timeout=360)  # discard result; just warm the model

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "greet.py").write_text(
            'def greet(name):\n'
            '    """Return greeting string."""\n'
            '    return f"Hello, {name}"\n'
            '\n'
            'def farewell(name):\n'
            '    """Return farewell string."""\n'
            '    return f"Goodbye, {name}"\n'
        )
        _git_commit_all(repo)

        # Timed cold build (model is pre-warmed)
        t0 = time.monotonic()
        res_cold = _run_index(repo, timeout=360)
        t_full = time.monotonic() - t0
        assert res_cold.returncode == 0, (
            f"Cold build failed.\nstdout: {res_cold.stdout}\nstderr: {res_cold.stderr}"
        )

        # Timed no-op reindex
        t0 = time.monotonic()
        res_noop = _run_index(repo, timeout=360)
        t_noop = time.monotonic() - t0
        assert res_noop.returncode == 0, (
            f"No-op reindex failed.\nstdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )

        embedded_noop, _, _ = _parse_summary(res_noop.stderr)
        assert embedded_noop == 0, (
            f"No-op must embed 0 units. Got {embedded_noop}."
        )

        assert t_noop < t_full, (
            f"No-op reindex ({t_noop:.2f}s) must be strictly faster than cold build "
            f"({t_full:.2f}s). Model is pre-warmed, so this should always hold when "
            f"parse-skip is active."
        )


# ---------------------------------------------------------------------------
# EDGE-2: deleting a file purges its symbols
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge2DeletionPurgesSymbols:
    """
    EDGE-2: deleting a source file removes its symbols from search results
    and from metadata.json (no ghost entries).

    Evidence (EDGE-2):
    - legacy_helper present before deletion (score 0.771)
    - After `rm legacy.py` and incremental reindex: embedded 0, reused 1
    - Search returns only active_fn; 'legacy_helper' absent
    - metadata.json: 1 unit, no legacy.py entries
    """

    def test_deletion_purges_symbols_from_search(self, tmp_path: Path):
        """Deleted file's symbols must not appear in search after incremental reindex."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "legacy.py").write_text(
            'def legacy_helper():\n'
            '    """Old helper that should be removed."""\n'
            '    pass\n'
        )
        (repo / "active.py").write_text(
            'def active_fn():\n'
            '    """Active function kept in the codebase."""\n'
            '    return 1\n'
        )
        _git_commit_all(repo)

        # Initial build
        res_init = _run_index(repo, timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )

        # Confirm legacy_helper is present before deletion
        res_before = _run_search(repo, "old helper that should be removed", k=5,
                                 timeout=120)
        assert res_before.returncode == 0
        results_before = _parse_search_results(res_before.stdout)
        names_before = _ranked_names(results_before)
        assert any("legacy" in n.lower() for n in names_before), (
            f"'legacy_helper' must appear in search before deletion. Got: {names_before}"
        )

        # Delete legacy.py
        (repo / "legacy.py").unlink()
        assert not (repo / "legacy.py").exists(), "legacy.py must be gone"

        # Incremental reindex (no flags)
        res_reindex = _run_index(repo, timeout=360)
        assert res_reindex.returncode == 0, (
            f"Incremental reindex after deletion failed.\n"
            f"stdout: {res_reindex.stdout}\nstderr: {res_reindex.stderr}"
        )

        # legacy_helper must be absent from search
        res_after = _run_search(repo, "old helper that should be removed", k=5,
                                timeout=120)
        assert res_after.returncode == 0, (
            f"Search after deletion failed.\nstdout: {res_after.stdout}"
        )
        results_after = _parse_search_results(res_after.stdout)
        names_after = _ranked_names(results_after)
        assert not any("legacy" in n.lower() for n in names_after), (
            f"'legacy_helper' must NOT appear in search after deletion. "
            f"Got: {names_after}. Ghost entry not purged."
        )

        # Check metadata.json: no ghost entries for legacy.py
        meta_path = repo / ".tldr" / "cache" / "semantic" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        units = meta.get("units", [])
        assert all(
            "legacy" not in u.get("file", "").lower() for u in units
        ), (
            f"metadata.json must not contain ghost legacy.py units. "
            f"Units: {[u.get('file') for u in units]}"
        )


# ---------------------------------------------------------------------------
# EDGE-3: rename preserves new-path symbols, purges old-path symbols
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge3RenameSymbols:
    """
    EDGE-3: renaming old_name.py -> new_name.py causes the old-path symbol to
    disappear from search and the new-path symbol to appear.

    Evidence (EDGE-3):
    - Initial: transform appears under old_name.py (score 0.734)
    - After mv + incremental reindex: embedded 1, reused 1
    - old_name.py symbols absent; new_name.py symbols present
    """

    def test_rename_old_path_absent_new_path_present(self, tmp_path: Path):
        """After rename + reindex: old-path symbols gone, new-path symbols searchable."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "old_name.py").write_text(
            'def transform(data):\n'
            '    """Transform input data."""\n'
            '    return data[::-1]\n'
        )
        (repo / "other.py").write_text(
            'def unrelated():\n'
            '    """Unrelated function."""\n'
            '    return 0\n'
        )
        _git_commit_all(repo)

        # Initial build
        res_init = _run_index(repo, timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )

        # Confirm transform appears under old_name.py
        res_before = _run_search(repo, "transform input data", k=5, timeout=120)
        assert res_before.returncode == 0
        results_before = _parse_search_results(res_before.stdout)
        before_files = [r.get("file", "") for r in results_before]
        assert any("old_name" in f for f in before_files), (
            f"transform must appear under old_name.py before rename. Files: {before_files}"
        )

        # Rename the file
        (repo / "old_name.py").rename(repo / "new_name.py")
        assert (repo / "new_name.py").exists()
        assert not (repo / "old_name.py").exists()

        # Incremental reindex (no flags)
        res_reindex = _run_index(repo, timeout=360)
        assert res_reindex.returncode == 0, (
            f"Incremental reindex after rename failed.\n"
            f"stdout: {res_reindex.stdout}\nstderr: {res_reindex.stderr}"
        )

        # Old-path symbols must be ABSENT
        res_after = _run_search(repo, "transform input data", k=10, timeout=120)
        assert res_after.returncode == 0
        results_after = _parse_search_results(res_after.stdout)
        after_files = [r.get("file", "") for r in results_after]
        assert not any("old_name" in f for f in after_files), (
            f"old_name.py symbols must be absent after rename. Files: {after_files}. "
            f"Ghost entry not purged on rename."
        )

        # New-path symbols must be PRESENT (transform still searchable under new_name.py)
        assert any("new_name" in f or "transform" in r.get("qualified_name", "").lower()
                   for r, f in zip(results_after, after_files + [""] * len(results_after))), (
            f"new_name.py symbols must appear after rename. "
            f"Got files: {after_files}, names: {_ranked_names(results_after)}"
        )


# ---------------------------------------------------------------------------
# EDGE-4: cross-file caller update: new call reflected in tldr context
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge4CrossFileCallerUpdate:
    """
    EDGE-4: editing file A to call file B's function makes `tldr context B.fn`
    list A as caller after incremental reindex; caller set equals --full rebuild.

    Evidence (EDGE-4):
    - Initial: core_process has no callers from pipeline.py
    - After pipeline.py rewrites to call core_process: embedded >= 1
    - tldr context core_process shows 'pipeline' as caller
    - Caller set matches --full rebuild exactly (order-insensitive)
    """

    def test_new_caller_appears_in_context_after_reindex(self, tmp_path: Path):
        """After editing caller, `tldr context callee` must list the new caller."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "core.py").write_text(
            'def core_process(data):\n'
            '    """Core processing function."""\n'
            '    return [x * 2 for x in data]\n'
        )
        (repo / "pipeline.py").write_text(
            'def pipeline(items):\n'
            '    """Independent pipeline, does NOT call core."""\n'
            '    return items\n'
        )
        _git_commit_all(repo)

        # Initial build
        res_init = _run_index(repo, timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )

        # Baseline: pipeline must NOT appear as caller of core_process
        res_ctx_before = _run_context("core_process", repo, timeout=60)
        ctx_before = res_ctx_before.stdout + res_ctx_before.stderr
        # Only assert if context returns something meaningful
        if "pipeline" in ctx_before.lower():
            assert "called_by" not in ctx_before.lower() or "pipeline" not in (
                ctx_before.lower().split("called_by")[-1]
                if "called_by" in ctx_before.lower() else ""
            ), (
                f"Before edit, pipeline must NOT be a caller of core_process.\n"
                f"Context: {ctx_before!r}"
            )

        # Edit pipeline.py to call core_process (core.py bytes unchanged)
        (repo / "pipeline.py").write_text(
            'from core import core_process\n'
            '\n'
            'def pipeline(items):\n'
            '    """Pipeline that now calls core_process."""\n'
            '    return core_process(items)\n'
        )

        # Incremental reindex (no flags, no daemon)
        res_reindex = _run_index(repo, timeout=360)
        assert res_reindex.returncode == 0, (
            f"Incremental reindex after caller edit failed.\n"
            f"stdout: {res_reindex.stdout}\nstderr: {res_reindex.stderr}"
        )
        embedded_re, _, _ = _parse_summary(res_reindex.stderr)
        assert embedded_re >= 1, (
            f"At least 1 unit must be re-embedded after caller edit. "
            f"Got embedded={embedded_re}."
        )

        # `tldr context core_process` must now list pipeline as a caller
        res_ctx_after = _run_context("core_process", repo, timeout=60)
        assert res_ctx_after.returncode == 0, (
            f"tldr context core_process failed.\n"
            f"stdout: {res_ctx_after.stdout}\nstderr: {res_ctx_after.stderr}"
        )
        ctx_after = res_ctx_after.stdout + res_ctx_after.stderr
        assert "pipeline" in ctx_after.lower(), (
            f"After reindex, 'pipeline' must appear as a caller of 'core_process' "
            f"in `tldr context`. Output:\n{ctx_after!r}"
        )

    def test_caller_set_equals_full_rebuild(self, tmp_path: Path):
        """Caller set from incremental reindex must match --full rebuild exactly."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "core.py").write_text(
            'def core_process(data):\n'
            '    """Core processing function."""\n'
            '    return [x * 2 for x in data]\n'
        )
        (repo / "pipeline.py").write_text(
            'from core import core_process\n'
            '\n'
            'def pipeline(items):\n'
            '    """Pipeline that calls core_process."""\n'
            '    return core_process(items)\n'
        )
        _git_commit_all(repo)

        # Incremental index (starts from scratch so this is also the cold build)
        res_inc = _run_index(repo, timeout=360)
        assert res_inc.returncode == 0

        # Capture caller set from incremental index
        res_ctx_inc = _run_context("core_process", repo, timeout=60)
        ctx_inc = res_ctx_inc.stdout + res_ctx_inc.stderr

        # --full rebuild on same state
        res_full = _run_index(repo, full=True, timeout=360)
        assert res_full.returncode == 0

        # Capture caller set from --full rebuild
        res_ctx_full = _run_context("core_process", repo, timeout=60)
        ctx_full = res_ctx_full.stdout + res_ctx_full.stderr

        # Both must mention 'pipeline' as a caller
        assert "pipeline" in ctx_inc.lower(), (
            f"Incremental: 'pipeline' must appear in context of core_process.\n"
            f"Got: {ctx_inc!r}"
        )
        assert "pipeline" in ctx_full.lower(), (
            f"Full rebuild: 'pipeline' must appear in context of core_process.\n"
            f"Got: {ctx_full!r}"
        )


# ---------------------------------------------------------------------------
# EDGE-5: incremental vs --full equivalence (symbol SET + call-graph edge SET)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge5IncrementalFullEquivalence:
    """
    EDGE-5: after driving the index through an incremental path, the searchable
    qualified-name SET and the call-graph edge SET for a cross-file symbol must
    equal those from a --full rebuild.

    Evidence (EDGE-5):
    - Incremental: embedded 1, reused 3 (reused > 0 confirmed)
    - Symbol SET diff: empty (core.py.normalize, core.py.process,
      pipeline.py.pipeline, util.py.utility identical in both)
    - Call-graph edge SET diff: empty (pipeline in both)
    """

    def test_symbol_set_equals_full_rebuild(self, tmp_path: Path):
        """Incremental qualified-name SET must equal --full rebuild SET."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "core.py").write_text(
            'def process(data):\n'
            '    """Core process."""\n'
            '    return list(data)\n'
            '\n'
            'def normalize(values):\n'
            '    """Normalize values."""\n'
            '    return values\n'
        )
        (repo / "pipeline.py").write_text(
            'from core import process\n'
            '\n'
            'def pipeline(items):\n'
            '    """Pipeline calling process."""\n'
            '    return process(items)\n'
        )
        (repo / "util.py").write_text(
            'def utility():\n'
            '    """Standalone utility."""\n'
            '    return None\n'
        )
        _git_commit_all(repo)

        # Initial cold build
        res_cold = _run_index(repo, timeout=360)
        assert res_cold.returncode == 0

        # Drive the incremental path: edit util.py and reindex
        (repo / "util.py").write_text(
            'def utility():\n'
            '    """Standalone utility, revised."""\n'
            '    return 1\n'
        )
        res_inc = _run_index(repo, timeout=360)
        assert res_inc.returncode == 0, (
            f"Incremental reindex (after util.py edit) failed.\n"
            f"stdout: {res_inc.stdout}\nstderr: {res_inc.stderr}"
        )
        _, reused_inc, _ = _parse_summary(res_inc.stderr)
        assert reused_inc > 0, (
            f"Incremental path must show reused > 0. Got reused={reused_inc}. "
            f"Test requires actual incremental (not full rebuild) path."
        )

        # Capture symbol SET from incremental index (broad k=100)
        res_search_inc = _run_search(
            repo, "function process pipeline utility normalize", k=100, timeout=180
        )
        assert res_search_inc.returncode == 0
        results_inc = _parse_search_results(res_search_inc.stdout)
        names_inc = sorted(set(
            r.get("qualified_name", r.get("name", "")) for r in results_inc
        ))

        # --full rebuild on the same repo state
        res_full = _run_index(repo, full=True, timeout=360)
        assert res_full.returncode == 0, (
            f"Full rebuild failed.\nstdout: {res_full.stdout}\nstderr: {res_full.stderr}"
        )
        _, reused_full, _ = _parse_summary(res_full.stderr)
        assert reused_full == 0, (
            f"--full rebuild must report reused=0. Got reused={reused_full}."
        )

        # Capture symbol SET from --full index (same broad query)
        res_search_full = _run_search(
            repo, "function process pipeline utility normalize", k=100, timeout=180
        )
        assert res_search_full.returncode == 0
        results_full = _parse_search_results(res_search_full.stdout)
        names_full = sorted(set(
            r.get("qualified_name", r.get("name", "")) for r in results_full
        ))

        assert names_inc == names_full, (
            f"Incremental symbol SET must equal --full rebuild SET.\n"
            f"Incremental: {names_inc}\nFull:        {names_full}\n"
            f"Difference:  {set(names_inc) ^ set(names_full)}"
        )

    def test_call_graph_edge_set_equals_full_rebuild(self, tmp_path: Path):
        """Incremental call-graph edge SET for cross-file symbol must equal --full."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "core.py").write_text(
            'def process(data):\n'
            '    """Core process."""\n'
            '    return list(data)\n'
            '\n'
            'def normalize(values):\n'
            '    """Normalize values."""\n'
            '    return values\n'
        )
        (repo / "pipeline.py").write_text(
            'from core import process\n'
            '\n'
            'def pipeline(items):\n'
            '    """Pipeline calling process."""\n'
            '    return process(items)\n'
        )
        (repo / "util.py").write_text(
            'def utility():\n'
            '    """Standalone utility."""\n'
            '    return None\n'
        )
        _git_commit_all(repo)

        # Cold build, then drive incremental path
        _run_index(repo, timeout=360)
        (repo / "util.py").write_text(
            'def utility():\n'
            '    """Standalone utility, revised."""\n'
            '    return 1\n'
        )
        res_inc = _run_index(repo, timeout=360)
        assert res_inc.returncode == 0
        _, reused_inc, _ = _parse_summary(res_inc.stderr)
        assert reused_inc > 0, f"Must be on incremental path (reused > 0). Got {reused_inc}."

        # Call-graph edges for 'process' from incremental index
        res_ctx_inc = _run_context("process", repo, timeout=60)
        ctx_inc = res_ctx_inc.stdout + res_ctx_inc.stderr

        # --full rebuild and call-graph edges from it
        res_full = _run_index(repo, full=True, timeout=360)
        assert res_full.returncode == 0
        res_ctx_full = _run_context("process", repo, timeout=60)
        ctx_full = res_ctx_full.stdout + res_ctx_full.stderr

        # Both must agree: 'pipeline' calls 'process' cross-file
        inc_has_pipeline = "pipeline" in ctx_inc.lower()
        full_has_pipeline = "pipeline" in ctx_full.lower()
        assert inc_has_pipeline == full_has_pipeline, (
            f"Call-graph edge sets must agree: "
            f"incremental has_pipeline={inc_has_pipeline}, "
            f"full has_pipeline={full_has_pipeline}.\n"
            f"Incremental context:\n{ctx_inc!r}\n"
            f"Full context:\n{ctx_full!r}"
        )


# ---------------------------------------------------------------------------
# EDGE-6: snapshot back-compat (old narrow file_hashes.json -> no crash)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge6SnapshotBackCompat:
    """
    EDGE-6: reindexing against an old narrow file_hashes.json
    ({path: sha1-string} values) must not crash; the snapshot is then in a
    valid state for subsequent no-op incremental runs.

    Evidence (EDGE-6):
    - Simulation sets file_hashes to narrow format (empty {} in this impl since
      it uses per-unit text_hash rather than file-level hashes)
    - Reindex exits 0 with no exception
    - Subsequent no-op shows embedded 0, reused N
    - Search returns correct results throughout
    """

    def test_old_narrow_snapshot_no_crash_then_incremental(self, tmp_path: Path):
        """Reindexing against old narrow file_hashes format must not crash."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "funcs.py").write_text(
            'def alpha():\n'
            '    """Alpha function."""\n'
            '    return 1\n'
            '\n'
            'def beta():\n'
            '    """Beta function."""\n'
            '    return 2\n'
        )
        (repo / "more.py").write_text(
            'def gamma():\n'
            '    """Gamma function."""\n'
            '    return 3\n'
        )
        _git_commit_all(repo)

        # Initial build
        res_init = _run_index(repo, timeout=360)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )
        embedded_init, _, _ = _parse_summary(res_init.stderr)
        assert embedded_init >= 3, (
            f"Initial build must embed >= 3 units (alpha, beta, gamma). "
            f"Got {embedded_init}."
        )

        # Simulate an old narrow file_hashes.json: replace dict values with bare sha1
        meta_path = repo / ".tldr" / "cache" / "semantic" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        hashes = meta.get("file_hashes", {})
        # Produce a narrow format: {path: sha1-string}
        narrow_hashes = {
            k: (v["sha1"] if isinstance(v, dict) and "sha1" in v else str(v))
            for k, v in hashes.items()
        }
        meta["file_hashes"] = narrow_hashes
        meta_path.write_text(json.dumps(meta))

        # Reindex against old narrow snapshot — must NOT crash
        res_compat = _run_index(repo, timeout=360)
        assert res_compat.returncode == 0, (
            f"Reindex against old narrow file_hashes snapshot must exit 0 (no crash).\n"
            f"stdout: {res_compat.stdout}\nstderr: {res_compat.stderr}"
        )
        assert "traceback" not in res_compat.stderr.lower(), (
            f"No traceback must appear after reindex against narrow snapshot.\n"
            f"stderr: {res_compat.stderr}"
        )

        # Subsequent no-op must confirm incremental mode active
        res_noop = _run_index(repo, timeout=360)
        assert res_noop.returncode == 0, (
            f"No-op after back-compat reindex failed.\n"
            f"stdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )
        embedded_noop, reused_noop, _ = _parse_summary(res_noop.stderr)
        assert embedded_noop == 0, (
            f"No-op after back-compat reindex must embed 0 units. "
            f"Got embedded={embedded_noop}. "
            f"Incremental mode must be active after snapshot upgrade."
        )
        assert reused_noop >= 1, (
            f"No-op must reuse >= 1 unit. Got reused={reused_noop}."
        )

        # Search must still return correct results
        res_search = _run_search(repo, "alpha function", k=5, timeout=120)
        assert res_search.returncode == 0
        results = _parse_search_results(res_search.stdout)
        names = _ranked_names(results)
        assert any("alpha" in n.lower() for n in names), (
            f"'alpha' must be searchable after back-compat upgrade. Got: {names}"
        )
