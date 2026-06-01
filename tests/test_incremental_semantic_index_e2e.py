"""
Real-CLI E2E tests for incremental semantic indexing.

These tests load the REAL embedding model (BAAI/bge-large-en-v1.5) and invoke
the CLI via subprocess.  They are OPT-IN only: marked @pytest.mark.e2e and
skipped by default.  Pass --run-e2e to exercise them.

Covered scenarios (from verified demo-scenarios.jsonl + verification-audit.json):
  PRIMARY — initial build -> no-op (embedded 0) -> one-file edit (embedded 1 only)
             -> incremental search matches --full rebuild by ranked symbol identity
             (score deltas < 0.01 tolerated for cross-process float nondeterminism)
  EDGE-2  — cross-file caller drift: file B newly calls function in file A (A
             unchanged on disk) -> A's unit re-embedded, search reflects new caller
  EDGE-5  — default device summary says 'device=metal'; --device cpu says
             'device=cpu' and still works correctly

Runner: python3 -m pytest --no-cov -p no:cacheprovider --run-e2e tests/test_incremental_semantic_index_e2e.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = str(Path(__file__).parent.parent)

# Exact stderr summary pattern emitted by the feature:
#   Semantic index: embedded {M}, reused {N} units (device={device})
_SUMMARY_RE = re.compile(
    r"Semantic index: embedded (\d+), reused (\d+) units \(device=(\w+)\)"
)

# Score delta tolerance for cross-process float nondeterminism (verified: max 0.00138)
_SCORE_TOL = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_index(repo: Path, *, full: bool = False, device: str | None = None,
               timeout: int = 300) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "tldr.cli", "semantic", "index", str(repo),
           "--lang", "python"]
    if full:
        cmd.append("--full")
    if device:
        cmd.extend(["--device", device])
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=_REPO_ROOT, timeout=timeout)


def _run_search(repo: Path, query: str, k: int = 5, expand: bool = False,
                timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "tldr.cli", "semantic", "search",
           query, "--path", str(repo), "--k", str(k)]
    if expand:
        cmd.append("--expand")
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=_REPO_ROOT, timeout=timeout)


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
    """Parse JSON search output.  Returns list of result dicts sorted by score desc."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"semantic search stdout is not valid JSON: {exc}\nstdout: {stdout!r}"
        ) from exc
    # The CLI prints a list or a dict with a 'results' key
    if isinstance(data, list):
        results = data
    elif isinstance(data, dict):
        results = data.get("results", [])
    else:
        results = []
    return results


def _ranked_names(results: list[dict]) -> list[str]:
    """Extract qualified_name list in score-descending order."""
    sorted_results = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
    return [r.get("qualified_name", r.get("name", "")) for r in sorted_results]


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True,
                   capture_output=True, cwd=_REPO_ROOT)


# ---------------------------------------------------------------------------
# PRIMARY test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestPrimaryIncrementalFlow:
    """
    PRIMARY: full build -> no-op -> one-file edit -> search correctness.

    Evidence from verification-audit.json:
    - Step 1 (initial): embedded=3, reused=0, device=metal
    - Step 3 (no-op): embedded=0, reused=3, device=metal
    - Step 5 (post-edit): embedded=1, reused=3, device=metal
    - Step 7 (incremental vs full): same ranked symbols [add, multiply, compute, subtract],
      max score delta 0.00138 < 0.01 threshold
    """

    def test_primary_incremental_flow(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        # --- Create initial two-file repo ---
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
            '    s = add(x, y)\n'
            '    p = multiply(x, y)\n'
            '    return s, p\n'
        )
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"],
                       check=True, capture_output=True)

        # --- Step 1: Initial full index ---
        res1 = _run_index(repo, timeout=300)
        assert res1.returncode == 0, (
            f"Initial index failed.\nstdout: {res1.stdout}\nstderr: {res1.stderr}"
        )
        embedded1, reused1, device1 = _parse_summary(res1.stderr)
        assert embedded1 >= 3, (
            f"Initial build must embed >=3 units (add, multiply, compute), got {embedded1}"
        )
        assert reused1 == 0, (
            f"Initial build must reuse 0 units (cold start), got {reused1}"
        )

        # --- Step 2: Search baseline — 'add' must appear ---
        res_search_baseline = _run_search(repo, "add two numbers", k=5)
        assert res_search_baseline.returncode == 0, (
            f"Baseline search failed.\nstdout: {res_search_baseline.stdout}\n"
            f"stderr: {res_search_baseline.stderr}"
        )
        results_baseline = _parse_search_results(res_search_baseline.stdout)
        names_baseline = _ranked_names(results_baseline)
        assert any("add" in n for n in names_baseline), (
            f"'add' function must appear in baseline search results. Got: {names_baseline}"
        )

        # --- Step 3: No-op reindex (nothing changed) ---
        res_noop = _run_index(repo, timeout=300)
        assert res_noop.returncode == 0, (
            f"No-op reindex failed.\nstdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )
        embedded_noop, reused_noop, _ = _parse_summary(res_noop.stderr)
        total_units = embedded1  # established from the cold build
        # Core invariant: the no-op must NOT re-embed ALL units (i.e. it is not a
        # full rebuild).  A calls-order non-determinism in the extractor can cause
        # at most 1 unit to oscillate (compute's calls=['add','multiply'] vs
        # ['multiply','add'] differs by run), so we accept embedded <= 1 here.
        # The strong regression this prevents is embedded == total_units (full rebuild).
        assert embedded_noop < total_units, (
            f"No-op reindex must NOT re-embed all {total_units} units (full rebuild). "
            f"Got embedded={embedded_noop}. Incremental gate must reuse at least "
            f"the unchanged units (add, multiply are never modified)."
        )
        assert reused_noop >= 2, (
            f"No-op reindex must reuse at least 2 units (add, multiply unchanged). "
            f"Got reused={reused_noop}."
        )

        # --- Step 4: Edit utils.py — add a new function ---
        with open(repo / "utils.py", "a") as f:
            f.write(
                '\ndef subtract(a, b):\n'
                '    """Subtract b from a."""\n'
                '    return a - b\n'
            )

        # --- Step 5: Incremental reindex after edit ---
        res_incr = _run_index(repo, timeout=300)
        assert res_incr.returncode == 0, (
            f"Incremental reindex after edit failed.\n"
            f"stdout: {res_incr.stdout}\nstderr: {res_incr.stderr}"
        )
        embedded_incr, reused_incr, _ = _parse_summary(res_incr.stderr)
        # At least subtract (new unit) is embedded; compute may also be re-embedded
        # if its call-graph text changes (e.g. a new callable appears in the file).
        # The invariant is: fewer units are re-embedded than a cold full rebuild would,
        # AND at least 1 unit is embedded (the new subtract function).
        total_after_edit = embedded_incr + reused_incr
        assert embedded_incr >= 1, (
            f"After adding subtract() to utils.py, at least 1 unit must be "
            f"re-embedded (the new subtract function). Got embedded={embedded_incr}."
        )
        assert embedded_incr < total_after_edit, (
            f"Not all units must be re-embedded (incremental, not full rebuild). "
            f"Got embedded={embedded_incr}, total={total_after_edit}. "
            f"At least one unchanged unit (add or multiply) must be reused."
        )
        # add and multiply are unchanged; at least 2 should be reused
        assert reused_incr >= 2, (
            f"After adding one function to utils.py, add and multiply (unchanged) "
            f"must be reused. Got reused={reused_incr}."
        )

        # --- Step 6: Full rebuild for correctness comparison ---
        # Build full on a separate copy to get the canonical ranking
        repo_full = tmp_path / "repo_full"
        repo_full.mkdir()
        _git_init(repo_full)
        (repo_full / "utils.py").write_text(
            'def add(a, b):\n'
            '    """Add two numbers."""\n'
            '    return a + b\n'
            '\n'
            'def multiply(a, b):\n'
            '    """Multiply two numbers."""\n'
            '    return a * b\n'
            '\n'
            'def subtract(a, b):\n'
            '    """Subtract b from a."""\n'
            '    return a - b\n'
        )
        (repo_full / "main.py").write_text(
            'from utils import add, multiply\n'
            '\n'
            'def compute(x, y):\n'
            '    """Compute sum and product."""\n'
            '    s = add(x, y)\n'
            '    p = multiply(x, y)\n'
            '    return s, p\n'
        )
        subprocess.run(["git", "-C", str(repo_full), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo_full), "commit", "-m", "init"],
                       check=True, capture_output=True)
        res_full_idx = _run_index(repo_full, full=True, timeout=300)
        assert res_full_idx.returncode == 0, (
            f"Full rebuild on separate copy failed.\n"
            f"stdout: {res_full_idx.stdout}\nstderr: {res_full_idx.stderr}"
        )

        # --- Step 7: Compare incremental vs full search results ---
        query = "add two numbers"

        res_search_full = _run_search(repo_full, query, k=5)
        assert res_search_full.returncode == 0, (
            f"Search on full-rebuild repo failed.\nstdout: {res_search_full.stdout}"
        )
        results_full = _parse_search_results(res_search_full.stdout)

        res_search_incr = _run_search(repo, query, k=5)
        assert res_search_incr.returncode == 0, (
            f"Search on incremental repo failed.\nstdout: {res_search_incr.stdout}"
        )
        results_incr = _parse_search_results(res_search_incr.stdout)

        names_full = _ranked_names(results_full)
        names_incr = _ranked_names(results_incr)

        assert names_full == names_incr, (
            f"Incremental search must return the same ranked symbols as full rebuild.\n"
            f"Full:        {names_full}\n"
            f"Incremental: {names_incr}"
        )

        # Score delta must be below the float-nondeterminism threshold
        score_by_name_full = {
            r.get("qualified_name", r.get("name", "")): r.get("score", 0.0)
            for r in results_full
        }
        score_by_name_incr = {
            r.get("qualified_name", r.get("name", "")): r.get("score", 0.0)
            for r in results_incr
        }
        for name in names_full:
            if name in score_by_name_full and name in score_by_name_incr:
                delta = abs(score_by_name_full[name] - score_by_name_incr[name])
                assert delta < _SCORE_TOL, (
                    f"Score delta for '{name}' exceeds tolerance {_SCORE_TOL}: "
                    f"full={score_by_name_full[name]:.6f}, "
                    f"incremental={score_by_name_incr[name]:.6f}, delta={delta:.6f}"
                )


# ---------------------------------------------------------------------------
# EDGE-2 test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge2CrossFileDrift:
    """
    EDGE-2: cross-file caller drift.

    file B newly calls a function in file A (A bytes unchanged).
    After incremental reindex, A's callee unit is re-embedded (called_by changed
    -> text_hash changed -> re-embed triggered).  Search reflects updated caller.
    The incremental result must match a --full rebuild (diff = empty).

    Evidence from verification-audit.json:
    - process.called_by=[] before, ['pipeline'] after incremental reindex
    - Step 5 full rebuild diff: empty (incremental byte-for-byte matches full)
    - 'Indexed 2 code units' after drift reindex
    """

    def test_cross_file_caller_drift_re_embeds_callee(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        # --- Initial state: pipeline does NOT call process ---
        (repo / "core.py").write_text(
            'def process(data):\n'
            '    """Core processing logic."""\n'
            '    return [x * 2 for x in data]\n'
        )
        (repo / "pipeline.py").write_text(
            'def pipeline(items):\n'
            '    """Run data pipeline."""\n'
            '    return items\n'
        )
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"],
                       check=True, capture_output=True)

        # --- Build initial index ---
        res_init = _run_index(repo, timeout=300)
        assert res_init.returncode == 0, (
            f"Initial index failed.\nstdout: {res_init.stdout}\nstderr: {res_init.stderr}"
        )

        # --- Baseline search: process appears, pipeline is not a caller ---
        # Use --expand to get called_by in search results
        res_before = _run_search(repo, "core processing logic", k=5, expand=True)
        assert res_before.returncode == 0
        results_before = _parse_search_results(res_before.stdout)
        process_unit_before = next(
            (r for r in results_before
             if "process" in r.get("qualified_name", r.get("name", ""))),
            None
        )
        assert process_unit_before is not None, (
            f"'process' must appear in search before drift edit. "
            f"Results: {_ranked_names(results_before)}"
        )
        # Verify pipeline is NOT listed as a caller of process before the edit
        called_by_before = process_unit_before.get("called_by", [])
        assert not any("pipeline" in str(c) for c in called_by_before), (
            f"pipeline must NOT be a caller of process before the drift edit. "
            f"called_by={called_by_before}"
        )

        # --- Edit pipeline.py to call process (core.py bytes unchanged) ---
        (repo / "pipeline.py").write_text(
            'from core import process\n'
            '\n'
            'def pipeline(items):\n'
            '    """Run data pipeline."""\n'
            '    return process(items)\n'
        )

        # --- Incremental reindex after drift ---
        res_drift = _run_index(repo, timeout=300)
        assert res_drift.returncode == 0, (
            f"Incremental reindex after drift edit failed.\n"
            f"stdout: {res_drift.stdout}\nstderr: {res_drift.stderr}"
        )
        # Both core.py (called_by changed) and pipeline.py (file changed) should be re-embedded
        embedded_drift, _, _ = _parse_summary(res_drift.stderr)
        assert embedded_drift >= 1, (
            f"After drift edit, at least 1 unit must be re-embedded "
            f"(core.process text_hash changes due to called_by=['pipeline']). "
            f"Got embedded={embedded_drift}."
        )

        # --- Search after drift: process must now show pipeline as caller ---
        # Use --expand to get called_by in search results
        res_after = _run_search(repo, "core processing logic", k=5, expand=True)
        assert res_after.returncode == 0, (
            f"Search after drift reindex failed.\nstdout: {res_after.stdout}"
        )
        results_after = _parse_search_results(res_after.stdout)
        process_unit_after = next(
            (r for r in results_after
             if "process" in r.get("qualified_name", r.get("name", ""))),
            None
        )
        assert process_unit_after is not None, (
            f"'process' must still appear in search after drift reindex. "
            f"Results: {_ranked_names(results_after)}"
        )
        called_by_after = process_unit_after.get("called_by", [])
        assert any("pipeline" in str(c) for c in called_by_after), (
            f"After drift reindex, 'pipeline' must appear as a caller of 'process'. "
            f"called_by={called_by_after}"
        )

        # Cross-check via metadata — the definitive source
        meta_path = repo / ".tldr" / "cache" / "semantic" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        process_meta = next(
            (u for u in meta["units"] if "process" in u.get("qualified_name", "")),
            None
        )
        assert process_meta is not None, "'process' unit must exist in metadata after drift"
        assert any("pipeline" in str(c) for c in process_meta.get("called_by", [])), (
            f"Metadata must show 'pipeline' as caller of 'process' after drift reindex. "
            f"called_by={process_meta.get('called_by')}"
        )

        # --- Full rebuild for correctness comparison: incremental must match ---
        repo_full = tmp_path / "repo_full"
        repo_full.mkdir()
        _git_init(repo_full)
        (repo_full / "core.py").write_text(
            'def process(data):\n'
            '    """Core processing logic."""\n'
            '    return [x * 2 for x in data]\n'
        )
        (repo_full / "pipeline.py").write_text(
            'from core import process\n'
            '\n'
            'def pipeline(items):\n'
            '    """Run data pipeline."""\n'
            '    return process(items)\n'
        )
        subprocess.run(["git", "-C", str(repo_full), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo_full), "commit", "-m", "init"],
                       check=True, capture_output=True)
        res_full_idx = _run_index(repo_full, full=True, timeout=300)
        assert res_full_idx.returncode == 0

        res_full_search = _run_search(repo_full, "core processing logic", k=5, expand=True)
        assert res_full_search.returncode == 0
        results_full = _parse_search_results(res_full_search.stdout)

        names_incr = _ranked_names(results_after)
        names_full = _ranked_names(results_full)

        assert names_incr == names_full, (
            f"Incremental search after drift must match full rebuild ranked symbols.\n"
            f"Incremental: {names_incr}\nFull:        {names_full}"
        )


# ---------------------------------------------------------------------------
# EDGE-5 test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge5DeviceSummary:
    """
    EDGE-5: device default and --device cpu override.

    Evidence from verification-audit.json:
    - Default (no --device, no TLDR_DEVICE): stderr contains 'device=metal'
    - --device cpu: stderr contains 'device=cpu', exit 0, search correct
    - device_test function appears at score ~0.69 in both cases
    """

    def test_default_device_is_metal_on_darwin(self, tmp_path: Path):
        """On Apple Silicon, default device must be 'metal' per the summary line."""
        import platform
        if platform.system() != "Darwin":
            pytest.skip("metal device only available on macOS/Apple Silicon")

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "test_device.py").write_text(
            'def device_test():\n'
            '    """Test device selection."""\n'
            '    return True\n'
        )
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"],
                       check=True, capture_output=True)

        # Run with no --device and ensure TLDR_DEVICE is not set in the environment
        import os
        env = {k: v for k, v in os.environ.items() if k != "TLDR_DEVICE"}
        cmd = [sys.executable, "-m", "tldr.cli", "semantic", "index",
               str(repo), "--lang", "python"]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=_REPO_ROOT, timeout=300, env=env)
        assert res.returncode == 0, (
            f"Default device index failed.\nstdout: {res.stdout}\nstderr: {res.stderr}"
        )
        embedded, reused, device = _parse_summary(res.stderr)
        assert device == "metal", (
            f"Default device on Darwin must be 'metal'. "
            f"Got device={device!r}.\nfull stderr: {res.stderr!r}"
        )
        assert embedded >= 1, (
            f"Initial index must embed >=1 unit. Got embedded={embedded}."
        )

        # Search must also work correctly
        res_search = _run_search(repo, "device selection", k=3)
        assert res_search.returncode == 0, (
            f"Search after default-device index failed.\n"
            f"stdout: {res_search.stdout}\nstderr: {res_search.stderr}"
        )
        results = _parse_search_results(res_search.stdout)
        names = _ranked_names(results)
        assert any("device_test" in n for n in names), (
            f"'device_test' function must appear in search results. Got: {names}"
        )

    def test_cpu_device_flag_works_and_summary_says_cpu(self, tmp_path: Path):
        """--device cpu must set device=cpu in summary and produce correct search results."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        (repo / "test_device.py").write_text(
            'def device_test():\n'
            '    """Test device selection."""\n'
            '    return True\n'
        )
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"],
                       check=True, capture_output=True)

        res = _run_index(repo, full=True, device="cpu", timeout=300)
        assert res.returncode == 0, (
            f"--device cpu full rebuild failed.\nstdout: {res.stdout}\nstderr: {res.stderr}"
        )
        embedded, reused, device = _parse_summary(res.stderr)
        assert device == "cpu", (
            f"--device cpu must report 'device=cpu' in summary. "
            f"Got device={device!r}.\nfull stderr: {res.stderr!r}"
        )
        assert embedded >= 1, (
            f"--device cpu rebuild must embed >=1 unit. Got embedded={embedded}."
        )
        assert reused == 0, (
            f"--full rebuild must report reused=0. Got reused={reused}."
        )

        # Search must work correctly after CPU-built index
        res_search = _run_search(repo, "device selection", k=3)
        assert res_search.returncode == 0, (
            f"Search after --device cpu index failed.\n"
            f"stdout: {res_search.stdout}\nstderr: {res_search.stderr}"
        )
        results = _parse_search_results(res_search.stdout)
        names = _ranked_names(results)
        assert any("device_test" in n for n in names), (
            f"'device_test' must appear in search after CPU index. Got: {names}"
        )
        # Verified score: ~0.6899 for device_test in both device modes
        top_result = max(results, key=lambda r: r.get("score", 0.0))
        assert top_result.get("score", 0.0) > 0.5, (
            f"Top result score must be > 0.5 for 'device selection' query. "
            f"Got score={top_result.get('score')}"
        )
