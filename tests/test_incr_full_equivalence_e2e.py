"""
E2E crystallisation tests for incremental call-graph equivalence (parse-skip-default).

These tests load the REAL embedding model and invoke the CLI via subprocess.
They are OPT-IN only: marked @pytest.mark.e2e and skipped by default.
Pass --run-e2e to exercise them.

Covered scenarios (crystallised from verified demo-scenarios.jsonl +
verification-audit.json for feature parse-skip-default):

  PRIMARY  — Multi-file OOP repo (Animal/Dog/Cat/Trainer/Zoo classes, cross-file
             constructor calls like ``d = Dog()`` in CARRIED methods, >5 callers for
             one callee). After cold-index -> add-reporter.py -> incremental reindex,
             per-unit ``calls`` AND ``called_by`` read directly from metadata.json
             are EXACTLY EQUAL to a --full rebuild for EVERY unit. This is the
             regression-catcher the prior e2e suite missed: it tested tldr context
             text output for one symbol, not exact metadata equality across all units.

  EDGE-2S5 — No-op reindex embeds 0, reuses N (S-5 observable). A subsequent
             edit of one file causes re-embed >= 1 (confirms S-5 early-exit does
             NOT prevent detecting real changes). Distinct from existing
             TestEdge1NoOpFasterThanCold: that test does not verify a post-no-op
             edit is still caught.

  EDGE-4   — ``--full`` writes a valid __schema_version__=2 snapshot at the
             canonical path (.tldr/cache/file_hashes.json) with sha1/mtime_ns/size/
             inode fields for every source file. A no-op incremental immediately
             after --full reports embedded=0, reused=N. Distinct from existing
             TestEdge6SnapshotBackCompat: that test SIMULATES an old narrow format;
             this test verifies the --full path produces the correct wide format
             FROM SCRATCH (schema version, field names, file coverage).

  EDGE-5   — One callee with >5 callers: after adding an 8th caller file and
             running incremental reindex, ``hub.called_by`` is capped to 5 in
             BOTH the incremental and --full metadata. The two 5-entry sets are
             IDENTICAL (same order). Not covered anywhere in the existing e2e suite.

Deduplication notes (these are covered in existing suites and NOT re-added here):
  - EDGE-1 (proportional churn)  : covered by TestPrimaryOneFileEdit + primary flow
  - EDGE-3 (dropped-edge/zero-caller callee): covered by TestEdge4CrossFileCallerUpdate
  - EDGE-6 (snapshot back-compat, narrow format): covered by TestEdge6SnapshotBackCompat
  - No-op embeds-0 + timing     : covered by TestEdge1NoOpFasterThanCold

Runner:
    python3 -m pytest --no-cov -p no:cacheprovider --run-e2e \\
        tests/test_incr_full_equivalence_e2e.py
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

# Exact stderr summary pattern:  Semantic index: embedded M, reused N units (device=D)
_SUMMARY_RE = re.compile(
    r"Semantic index: embedded (\d+), reused (\d+) units \(device=(\w+)\)"
)


# ---------------------------------------------------------------------------
# Helpers — mirror test_parse_skip_default_e2e.py exactly
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


def _read_metadata(repo: Path) -> dict:
    """Read and parse .tldr/cache/semantic/metadata.json inside repo."""
    meta_path = repo / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing at {meta_path}"
    return json.loads(meta_path.read_text())


def _validate_per_unit_equality(meta_incr: dict, meta_full: dict) -> list[str]:
    """Return a list of divergence messages (empty = all equal).

    For every unit present in BOTH indexes, checks that the sorted ``calls``
    list AND the sorted ``called_by`` list are IDENTICAL (exact set equality).
    Presence mismatches (unit only in one index) are also reported.
    """
    incr_by_name = {u["name"]: u for u in meta_incr.get("units", [])}
    full_by_name = {u["name"]: u for u in meta_full.get("units", [])}
    problems: list[str] = []

    only_incr = set(incr_by_name) - set(full_by_name)
    only_full = set(full_by_name) - set(incr_by_name)
    if only_incr:
        problems.append(f"Units only in incremental (absent in --full): {sorted(only_incr)}")
    if only_full:
        problems.append(f"Units only in --full (absent in incremental): {sorted(only_full)}")

    for name in sorted(set(incr_by_name) & set(full_by_name)):
        u_i = incr_by_name[name]
        u_f = full_by_name[name]
        i_calls = sorted(u_i.get("calls") or [])
        f_calls = sorted(u_f.get("calls") or [])
        i_cb = sorted(u_i.get("called_by") or [])
        f_cb = sorted(u_f.get("called_by") or [])
        if i_calls != f_calls:
            problems.append(
                f"{name}.calls: incr={i_calls!r} != full={f_calls!r}"
            )
        if i_cb != f_cb:
            problems.append(
                f"{name}.called_by: incr={i_cb!r} != full={f_cb!r}"
            )

    return problems


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True,
                   cwd=_REPO_ROOT)


def _git_commit_all(repo: Path, message: str = "init") -> None:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True,
                   capture_output=True)


# ---------------------------------------------------------------------------
# PRIMARY: Multi-file OOP with constructor calls — per-unit metadata equality
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestPrimaryMultiFileOopIncrFullEquivalence:
    """
    PRIMARY scenario: cold-index a multi-file OOP repo (Animal/Dog/Cat/Trainer/Zoo),
    add reporter.py as the ONE new file, run incremental reindex (DEFAULT path),
    then run --full rebuild on the same state. Assert that per-unit ``calls`` AND
    ``called_by`` from metadata.json are EXACTLY EQUAL for EVERY unit.

    This is the regression-catcher the prior e2e suite missed: existing
    TestEdge5IncrementalFullEquivalence only checks via tldr context text output
    for a single symbol — not exact metadata JSON equality across ALL units. The
    constructor-edge omission (Dog/Cat/Trainer absent from carried methods' calls
    lists) was invisible to the old tests.

    Evidence (PRIMARY from demo-scenarios.jsonl):
    - Cold build: embedded 21, reused 0 (21 units, N > 20)
    - Incremental after adding reporter.py: embedded 8, reused 16 (carry path active)
    - Full rebuild: embedded 24, reused 0
    - Step 8 assertion: PASS — incremental calls/called_by == --full for all 18 units
    """

    def test_per_unit_calls_and_called_by_exactly_equal_after_incremental(
        self, tmp_path: Path
    ):
        """Per-unit calls + called_by from metadata.json must match --full for every unit.

        The fixture includes cross-file constructor calls (d = Dog(); c = Cat()) in
        CARRIED methods (Trainer.demo_speak, Zoo.daily_show, Zoo.feeding_time) to
        specifically catch the constructor-edge omission that was the root cause of
        the ~80% divergence on HEAD 792eff8.
        """
        import shutil
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        # Two independent repos: one driven through incremental, one through --full.
        repo_incr = tmp_path / "oop_incr"
        repo_full = tmp_path / "oop_full"
        repo_incr.mkdir()
        repo_full.mkdir()

        # --- Build the identical initial 4-file OOP repo in BOTH repos ---
        initial_files = {
            "models.py": (
                "class Animal:\n"
                "    def speak(self):\n"
                "        \"\"\"Base speak.\"\"\"\n"
                "        return 'sound'\n"
                "    def move(self):\n"
                "        \"\"\"Base move.\"\"\"\n"
                "        return 'move'\n"
                "    def breathe(self):\n"
                "        \"\"\"Breathe.\"\"\"\n"
                "        return 'breath'\n"
                "\n"
                "class Dog(Animal):\n"
                "    def speak(self):\n"
                "        \"\"\"Dog speak.\"\"\"\n"
                "        return 'woof'\n"
                "    def fetch(self):\n"
                "        \"\"\"Dog fetch.\"\"\"\n"
                "        return self.speak()\n"
                "\n"
                "class Cat(Animal):\n"
                "    def speak(self):\n"
                "        \"\"\"Cat speak.\"\"\"\n"
                "        return 'meow'\n"
                "    def purr(self):\n"
                "        \"\"\"Cat purr calls speak.\"\"\"\n"
                "        return self.speak()\n"
            ),
            "trainer.py": (
                "from models import Dog, Cat, Animal\n"
                "\n"
                "class Trainer:\n"
                "    def train_dog(self, dog):\n"
                "        \"\"\"Train a dog to speak.\"\"\"\n"
                "        return dog.speak()\n"
                "    def train_cat(self, cat):\n"
                "        \"\"\"Train a cat to speak.\"\"\"\n"
                "        return cat.speak()\n"
                "    def train_animal(self, animal):\n"
                "        \"\"\"Generic animal training.\"\"\"\n"
                "        return animal.speak()\n"
                "    def demo_speak(self):\n"
                "        \"\"\"Demo all speak variants.\"\"\"\n"
                "        d = Dog()\n"
                "        c = Cat()\n"
                "        return d.speak(), c.speak()\n"
            ),
            "zoo.py": (
                "from models import Dog, Cat, Animal\n"
                "from trainer import Trainer\n"
                "\n"
                "class Zoo:\n"
                "    def daily_show(self):\n"
                "        \"\"\"Run daily show calling speak on all animals.\"\"\"\n"
                "        d = Dog()\n"
                "        c = Cat()\n"
                "        t = Trainer()\n"
                "        return d.speak(), c.speak(), t.train_dog(d)\n"
                "    def feeding_time(self):\n"
                "        \"\"\"Feeding time — calls move on animals.\"\"\"\n"
                "        d = Dog()\n"
                "        c = Cat()\n"
                "        return d.move(), c.move()\n"
                "    def report(self):\n"
                "        \"\"\"Zoo report.\"\"\"\n"
                "        return 'report'\n"
            ),
            "utils.py": (
                "def log_event(msg):\n"
                "    \"\"\"Log an event to stdout.\"\"\"\n"
                "    print(msg)\n"
                "\n"
                "def format_output(value):\n"
                "    \"\"\"Format output for display.\"\"\"\n"
                "    return str(value)\n"
            ),
        }

        reporter_py = (
            "from models import Dog, Cat, Animal\n"
            "from trainer import Trainer\n"
            "from zoo import Zoo\n"
            "\n"
            "class Reporter:\n"
            "    def full_report(self):\n"
            "        \"\"\"Full report: calls speak on all animals.\"\"\"\n"
            "        d = Dog()\n"
            "        c = Cat()\n"
            "        t = Trainer()\n"
            "        z = Zoo()\n"
            "        return d.speak(), c.speak(), t.demo_speak(), z.daily_show()\n"
            "    def summary(self):\n"
            "        \"\"\"Summary report.\"\"\"\n"
            "        return format(self.full_report())\n"
        )

        for repo in (repo_incr, repo_full):
            _git_init(repo)
            for fname, content in initial_files.items():
                (repo / fname).write_text(content)
            _git_commit_all(repo)

        # --- Cold index (only repo_incr; repo_full has no prior index) ---
        res_cold = _run_index(repo_incr, timeout=360)
        assert res_cold.returncode == 0, (
            f"Cold index failed.\nstdout: {res_cold.stdout}\nstderr: {res_cold.stderr}"
        )
        embedded_cold, reused_cold, _ = _parse_summary(res_cold.stderr)
        assert embedded_cold > 10, (
            f"Cold build must embed > 10 units from the 4-file OOP repo. "
            f"Got embedded={embedded_cold}."
        )
        assert reused_cold == 0, (
            f"Cold build must reuse 0 units. Got reused={reused_cold}."
        )

        # --- Add reporter.py to BOTH repos (identical final state) ---
        (repo_incr / "reporter.py").write_text(reporter_py)
        (repo_full / "reporter.py").write_text(reporter_py)

        # --- Incremental reindex: DEFAULT path (no --full, no dirty_files) ---
        res_incr = _run_index(repo_incr, timeout=360)
        assert res_incr.returncode == 0, (
            f"Incremental reindex after adding reporter.py failed.\n"
            f"stdout: {res_incr.stdout}\nstderr: {res_incr.stderr}"
        )
        embedded_incr, reused_incr, _ = _parse_summary(res_incr.stderr)
        # CARRY PATH ACTIVE: reused > 0 means models/trainer/zoo are carried.
        assert reused_incr > 0, (
            f"After adding reporter.py, incremental reindex must reuse N > 0 units "
            f"(models/trainer/zoo are unchanged — carry path must be active). "
            f"Got reused={reused_incr}. "
            f"If reused=0 the incremental path did not activate and this test is void."
        )

        # --- Full rebuild on repo_full (same final state: 5 files) ---
        res_full = _run_index(repo_full, full=True, timeout=360)
        assert res_full.returncode == 0, (
            f"Full rebuild failed.\nstdout: {res_full.stdout}\nstderr: {res_full.stderr}"
        )
        embedded_full, reused_full, _ = _parse_summary(res_full.stderr)
        assert reused_full == 0, (
            f"--full rebuild must report reused=0. Got reused={reused_full}."
        )

        # --- Read per-unit metadata from both indexes ---
        meta_incr = _read_metadata(repo_incr)
        meta_full = _read_metadata(repo_full)

        assert len(meta_incr.get("units", [])) > 0, "Incremental metadata has no units"
        assert len(meta_full.get("units", [])) > 0, "Full metadata has no units"

        # --- Core assertion: per-unit calls + called_by exact equality ---
        problems = _validate_per_unit_equality(meta_incr, meta_full)
        assert not problems, (
            f"Incremental call-graph diverges from --full rebuild "
            f"for {len(problems)} unit(s) after adding reporter.py "
            f"(incremental reused={reused_incr} carried units):\n"
            + "\n".join(f"  {p}" for p in problems[:30])
            + (f"\n  ... and {len(problems) - 30} more" if len(problems) > 30 else "")
            + "\n\nThis test catches the constructor-edge omission: methods in "
            f"CARRIED files (trainer.py, zoo.py) reference cross-file class "
            f"constructors (Dog(), Cat(), Trainer()). The --full path resolves "
            f"these via import resolution and includes them in `calls`. "
            f"The incremental path must produce identical results."
        )


# ---------------------------------------------------------------------------
# EDGE-2 S-5: no-op embeds 0; subsequent edit still caught (S-5 does not block)
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge2S5NoOpAndSubsequentEdit:
    """
    EDGE-2 (S-5): A no-op reindex reports embedded=0, reused=N (S-5 observable).
    A subsequent single-file edit causes embedded >= 1 on the next reindex,
    confirming the S-5 early-exit does NOT prevent detection of real changes.

    The existing TestEdge1NoOpFasterThanCold only verifies embedded=0 on no-op.
    It does NOT verify that a subsequent edit is still caught after the no-op
    (i.e., that S-5 doesn't permanently lock out change detection). This test
    adds that second-step assertion.

    Evidence (EDGE-2 from demo-scenarios.jsonl):
    - Cold build: embedded 3, reused 0 (exit 0)
    - No-op: embedded 0, reused 3, compute_file_hash calls=0 (S-5 confirmed)
    - After changing one file: embedded >= 1, reused >= 1 (change detected)
    """

    def test_noop_then_edit_is_detected(self, tmp_path: Path):
        """No-op embeds 0; subsequent single-file edit re-embeds at least 1 unit."""
        import shutil
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "module_a.py").write_text(
            "def func_a():\n"
            "    \"\"\"First standalone function.\"\"\"\n"
            "    return 1\n"
        )
        (repo / "module_b.py").write_text(
            "def func_b():\n"
            "    \"\"\"Second standalone function.\"\"\"\n"
            "    return 2\n"
        )
        (repo / "module_c.py").write_text(
            "def func_c():\n"
            "    \"\"\"Third standalone function.\"\"\"\n"
            "    return 3\n"
        )
        _git_commit_all(repo)

        # --- Cold index ---
        res_cold = _run_index(repo, timeout=360)
        assert res_cold.returncode == 0, (
            f"Cold build failed.\nstdout: {res_cold.stdout}\nstderr: {res_cold.stderr}"
        )
        embedded_cold, reused_cold, _ = _parse_summary(res_cold.stderr)
        assert embedded_cold >= 3, (
            f"Cold build must embed >= 3 units. Got embedded={embedded_cold}."
        )
        assert reused_cold == 0, f"Cold build must reuse 0. Got reused={reused_cold}."

        # --- No-op reindex: NOTHING changed ---
        res_noop = _run_index(repo, timeout=360)
        assert res_noop.returncode == 0, (
            f"No-op reindex failed.\nstdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )
        embedded_noop, reused_noop, _ = _parse_summary(res_noop.stderr)
        assert embedded_noop == 0, (
            f"No-op reindex must embed 0 units (S-5: all files unchanged, "
            f"stat fast-path skips re-hash). Got embedded={embedded_noop}. "
            f"Verify the S-5 optimization is active."
        )
        assert reused_noop >= 3, (
            f"No-op reindex must reuse >= 3 units. Got reused={reused_noop}."
        )

        # --- Edit ONE file: update func_b to return a different value ---
        (repo / "module_b.py").write_text(
            "def func_b():\n"
            "    \"\"\"Second standalone function, updated implementation.\"\"\"\n"
            "    return 42\n"
        )

        # --- Incremental reindex after edit: must detect the change ---
        res_edit = _run_index(repo, timeout=360)
        assert res_edit.returncode == 0, (
            f"Incremental reindex after edit failed.\n"
            f"stdout: {res_edit.stdout}\nstderr: {res_edit.stderr}"
        )
        embedded_edit, reused_edit, _ = _parse_summary(res_edit.stderr)

        # S-5 must NOT permanently suppress change detection.
        assert embedded_edit >= 1, (
            f"After editing module_b.py, at least 1 unit must be re-embedded. "
            f"Got embedded={embedded_edit}. "
            f"The S-5 no-hash-on-no-op optimization must NOT prevent detection "
            f"of real content changes on the subsequent run."
        )
        # Unchanged modules must still be reused.
        assert reused_edit >= 1, (
            f"After editing one file, unchanged files must still be reused. "
            f"Got reused={reused_edit}."
        )


# ---------------------------------------------------------------------------
# EDGE-4: --full writes __schema_version__=2 snapshot; no-op confirms incremental
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge4FullSnapshotSchemaAndNoOp:
    """
    EDGE-4: --full rebuild writes a valid __schema_version__=2 snapshot at the
    canonical path (.tldr/cache/file_hashes.json) with sha1/mtime_ns/size/inode
    fields for every source file. A no-op incremental immediately after --full
    reports embedded=0, reused=N.

    Distinct from existing TestEdge6SnapshotBackCompat: that test SIMULATES an
    old narrow format written by the user; this test verifies the --full path
    produces the correct WIDE format from scratch (schema version, all fields,
    file coverage).

    Evidence (EDGE-4 from demo-scenarios.jsonl):
    - --full build: embedded 2, reused 0
    - Snapshot at .tldr/cache/file_hashes.json: __schema_version__=2,
      entries for pipeline.py and proc.py each with sha1/mtime_ns/size/inode
    - No-op incremental after --full: embedded=0, reused=2
    """

    def test_full_rebuild_writes_valid_wide_snapshot(self, tmp_path: Path):
        """--full must write __schema_version__=2 snapshot with required fields."""
        import shutil
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "pipeline.py").write_text(
            "def run_pipeline(items):\n"
            "    \"\"\"Run the main processing pipeline.\"\"\"\n"
            "    return [x * 2 for x in items]\n"
        )
        (repo / "proc.py").write_text(
            "def process(data):\n"
            "    \"\"\"Process raw data.\"\"\"\n"
            "    return list(data)\n"
        )
        _git_commit_all(repo)

        # --- Full rebuild ---
        res_full = _run_index(repo, full=True, timeout=360)
        assert res_full.returncode == 0, (
            f"--full rebuild failed.\nstdout: {res_full.stdout}\nstderr: {res_full.stderr}"
        )
        embedded_full, reused_full, _ = _parse_summary(res_full.stderr)
        assert embedded_full >= 2, (
            f"--full build must embed >= 2 units. Got embedded={embedded_full}."
        )
        assert reused_full == 0, (
            f"--full build must reuse 0 units. Got reused={reused_full}."
        )

        # --- Verify the snapshot at .tldr/cache/file_hashes.json ---
        snapshot_path = repo / ".tldr" / "cache" / "file_hashes.json"
        assert snapshot_path.exists(), (
            f"--full build must write file_hashes.json at {snapshot_path}. "
            f"File not found. The wide snapshot (S-5) is required for subsequent "
            f"no-op incremental runs to avoid re-hashing unchanged files."
        )

        snapshot = json.loads(snapshot_path.read_text())

        # Verify __schema_version__ == 2
        schema_version = snapshot.get("__schema_version__")
        assert schema_version == 2, (
            f"file_hashes.json must have __schema_version__=2. "
            f"Got __schema_version__={schema_version!r}. "
            f"The wide snapshot format (V2) is required for the S-5 stat fast-path."
        )

        # Verify both source files are present in the snapshot.
        file_entries = {k: v for k, v in snapshot.items() if k != "__schema_version__"}
        snapshot_files = set(file_entries.keys())
        expected_files = {"pipeline.py", "proc.py"}
        # Keys may be absolute paths; check that expected basenames appear.
        snapshot_basenames = {Path(k).name for k in snapshot_files}
        missing = expected_files - snapshot_basenames
        assert not missing, (
            f"file_hashes.json must include entries for all source files. "
            f"Missing: {sorted(missing)}. "
            f"Snapshot keys (basenames): {sorted(snapshot_basenames)}."
        )

        # Verify each entry has the required wide-format fields.
        required_fields = {"sha1", "mtime_ns", "size", "inode"}
        for key, entry in file_entries.items():
            if Path(key).name not in expected_files:
                continue
            if not isinstance(entry, dict):
                continue
            missing_fields = required_fields - set(entry.keys())
            assert not missing_fields, (
                f"Snapshot entry for {key!r} is missing wide-format fields: "
                f"{sorted(missing_fields)}. "
                f"Entry: {entry!r}. "
                f"All four fields (sha1/mtime_ns/size/inode) are required for the "
                f"S-5 stat fast-path to skip re-hashing unchanged files."
            )
            assert isinstance(entry["sha1"], str) and len(entry["sha1"]) == 40, (
                f"sha1 for {key!r} must be a 40-char hex string. "
                f"Got: {entry['sha1']!r}"
            )
            assert isinstance(entry["mtime_ns"], int), (
                f"mtime_ns for {key!r} must be an int. Got: {entry['mtime_ns']!r}"
            )
            assert isinstance(entry["size"], int) and entry["size"] >= 0, (
                f"size for {key!r} must be a non-negative int. Got: {entry['size']!r}"
            )
            assert isinstance(entry["inode"], int) and entry["inode"] > 0, (
                f"inode for {key!r} must be a positive int. Got: {entry['inode']!r}"
            )

    def test_noop_incremental_after_full_embeds_zero(self, tmp_path: Path):
        """No-op incremental immediately after --full must report embedded=0, reused=N."""
        import shutil
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        (repo / "pipeline.py").write_text(
            "def run_pipeline(items):\n"
            "    \"\"\"Run the main processing pipeline.\"\"\"\n"
            "    return [x * 2 for x in items]\n"
        )
        (repo / "proc.py").write_text(
            "def process(data):\n"
            "    \"\"\"Process raw data.\"\"\"\n"
            "    return list(data)\n"
        )
        _git_commit_all(repo)

        # --- Full rebuild ---
        res_full = _run_index(repo, full=True, timeout=360)
        assert res_full.returncode == 0, (
            f"--full rebuild failed.\nstdout: {res_full.stdout}\nstderr: {res_full.stderr}"
        )

        # --- No-op incremental immediately after --full (nothing changed) ---
        res_noop = _run_index(repo, timeout=360)
        assert res_noop.returncode == 0, (
            f"No-op incremental after --full failed.\n"
            f"stdout: {res_noop.stdout}\nstderr: {res_noop.stderr}"
        )
        embedded_noop, reused_noop, _ = _parse_summary(res_noop.stderr)

        assert embedded_noop == 0, (
            f"No-op incremental after --full must report embedded=0. "
            f"Got embedded={embedded_noop}. "
            f"The --full path must write a valid wide snapshot so the subsequent "
            f"incremental run can reuse all units without re-embedding."
        )
        assert reused_noop >= 2, (
            f"No-op incremental after --full must reuse >= 2 units. "
            f"Got reused={reused_noop}."
        )


# ---------------------------------------------------------------------------
# EDGE-5: 5-cap applied identically in incremental and --full
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestEdge55CapParityIncrementalVsFull:
    """
    EDGE-5: A callee function with >5 callers has its ``called_by`` capped to
    5 entries in BOTH incremental and --full metadata, and the two 5-entry sets
    are IDENTICAL (same names, same order).

    This confirms _stable_call_list (or equivalent cap logic) is applied
    consistently on both code paths. Not covered anywhere in the existing e2e suite.

    Evidence (EDGE-5 from demo-scenarios.jsonl):
    - 7 initial caller files (c1..c7) each calling hub(); cold build: embedded 8
    - Add c8.py (8th caller); incremental reindex: embedded 1, reused 8
    - Full rebuild: embedded 9, reused 0
    - Both: hub.called_by = ['caller_1','caller_2','caller_3','caller_4','caller_5']
      (capped to 5, identical sets)
    """

    def test_5cap_applied_identically_in_incremental_and_full(
        self, tmp_path: Path
    ):
        """hub() with >5 callers: called_by capped to 5, identical in incr and --full."""
        import shutil
        shutil.rmtree("/private/tmp/.tldr", ignore_errors=True)

        repo_incr = tmp_path / "cap_incr"
        repo_full = tmp_path / "cap_full"
        repo_incr.mkdir()
        repo_full.mkdir()

        # hub.py: the callee with >5 callers.
        hub_py = (
            "def hub():\n"
            "    \"\"\"Central hub function called by many callers.\"\"\"\n"
            "    return True\n"
        )

        # 7 initial caller files.
        caller_files_initial = {
            f"caller_{i}.py": (
                f"from hub import hub\n"
                f"\n"
                f"def caller_{i}():\n"
                f"    \"\"\"Caller number {i}.\"\"\"\n"
                f"    return hub()\n"
            )
            for i in range(1, 8)
        }

        # c8.py: the new caller added before the incremental reindex.
        c8_py = (
            "from hub import hub\n"
            "\n"
            "def caller_8():\n"
            "    \"\"\"Caller number 8.\"\"\"\n"
            "    return hub()\n"
        )

        # --- Set up BOTH repos with the initial 8-file state (hub + 7 callers) ---
        for repo in (repo_incr, repo_full):
            _git_init(repo)
            (repo / "hub.py").write_text(hub_py)
            for fname, content in caller_files_initial.items():
                (repo / fname).write_text(content)
            _git_commit_all(repo)

        # --- Cold index repo_incr (7 caller files + hub) ---
        res_cold = _run_index(repo_incr, timeout=360)
        assert res_cold.returncode == 0, (
            f"Cold index failed.\nstdout: {res_cold.stdout}\nstderr: {res_cold.stderr}"
        )
        embedded_cold, reused_cold, _ = _parse_summary(res_cold.stderr)
        assert embedded_cold >= 8, (
            f"Cold build must embed >= 8 units (hub + 7 callers). "
            f"Got embedded={embedded_cold}."
        )
        assert reused_cold == 0, f"Cold build must reuse 0. Got reused={reused_cold}."

        # Verify the cold build already caps hub.called_by at 5 (sanity check).
        meta_cold = _read_metadata(repo_incr)
        hub_cold = next(
            (u for u in meta_cold["units"] if u.get("name") == "hub"), None
        )
        if hub_cold is not None:
            cold_cb = hub_cold.get("called_by") or []
            assert len(cold_cb) <= 5, (
                f"Cold build hub.called_by must be capped at 5. "
                f"Got {len(cold_cb)} entries: {cold_cb!r}"
            )

        # --- Add c8.py to BOTH repos (identical final state: hub + 8 callers) ---
        (repo_incr / "caller_8.py").write_text(c8_py)
        (repo_full / "caller_8.py").write_text(c8_py)

        # --- Incremental reindex: DEFAULT path ---
        res_incr = _run_index(repo_incr, timeout=360)
        assert res_incr.returncode == 0, (
            f"Incremental reindex after adding caller_8.py failed.\n"
            f"stdout: {res_incr.stdout}\nstderr: {res_incr.stderr}"
        )
        embedded_incr, reused_incr, _ = _parse_summary(res_incr.stderr)
        # Carry path active: the 7 original caller files are unchanged.
        assert reused_incr > 0, (
            f"After adding caller_8.py, incremental must reuse N > 0 units "
            f"(original 7 callers are unchanged). Got reused={reused_incr}."
        )

        # --- Full rebuild on repo_full (same final state: hub + 8 callers) ---
        res_full = _run_index(repo_full, full=True, timeout=360)
        assert res_full.returncode == 0, (
            f"Full rebuild failed.\nstdout: {res_full.stdout}\nstderr: {res_full.stderr}"
        )
        embedded_full_rebuild, reused_full_rebuild, _ = _parse_summary(res_full.stderr)
        assert reused_full_rebuild == 0, (
            f"--full rebuild must report reused=0. Got reused={reused_full_rebuild}."
        )

        # --- Read hub.called_by from BOTH indexes ---
        meta_incr = _read_metadata(repo_incr)
        meta_full = _read_metadata(repo_full)

        hub_incr = next(
            (u for u in meta_incr["units"] if u.get("name") == "hub"), None
        )
        hub_full = next(
            (u for u in meta_full["units"] if u.get("name") == "hub"), None
        )

        assert hub_incr is not None, (
            "hub unit not found in incremental metadata. "
            "Check that hub.py's 'hub' function was indexed."
        )
        assert hub_full is not None, (
            "hub unit not found in --full metadata. "
            "Check that hub.py's 'hub' function was indexed."
        )

        cb_incr = hub_incr.get("called_by") or []
        cb_full = hub_full.get("called_by") or []

        # Both must be capped at 5.
        assert len(cb_incr) <= 5, (
            f"Incremental hub.called_by must be capped at 5. "
            f"Got {len(cb_incr)} entries: {cb_incr!r}. "
            f"8 callers exist; _stable_call_list must cap at 5."
        )
        assert len(cb_full) <= 5, (
            f"Full rebuild hub.called_by must be capped at 5. "
            f"Got {len(cb_full)} entries: {cb_full!r}. "
            f"8 callers exist; _stable_call_list must cap at 5."
        )

        # Both caps must be at the same size (5) and contain the SAME entries.
        assert len(cb_incr) == len(cb_full), (
            f"Incremental and --full hub.called_by must have the same cap size. "
            f"Incremental: {len(cb_incr)} entries {cb_incr!r}\n"
            f"Full:        {len(cb_full)} entries {cb_full!r}"
        )
        assert sorted(cb_incr) == sorted(cb_full), (
            f"Incremental and --full hub.called_by must contain the same entries "
            f"(order-insensitive). "
            f"Incremental: {sorted(cb_incr)!r}\n"
            f"Full:        {sorted(cb_full)!r}\n"
            f"Difference:  {set(cb_incr) ^ set(cb_full)!r}. "
            f"The 5-cap must be applied identically on both code paths."
        )
