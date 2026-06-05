"""
Behavior (2) — S-5 NO-REHASH: a no-op reindex does NOT re-hash unchanged files.

Cold-index a few-file repo. Then run a no-op reindex with compute_file_hash
spied/counted. Assert the spy count == 0 for unchanged files on the no-op.

The spy patches BOTH `tldr.patch.compute_file_hash` AND — if it exists as a
local binding — `tldr.semantic.compute_file_hash`, to be robust to import style.
A baseline assertion (cold build calls compute_file_hash >= N) proves the spy
actually intercepts calls.

RED reason (HEAD): _compute_current_file_hashes re-hashes every file on every
run (persist-time re-hash loop, no deriver_sha1_map or old_snapshot reuse).
On a no-op reindex with 2 files, count == 2 when it should be 0.

Runner: python3 -m pytest --no-cov tests/test_s5_no_rehash.py
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
from conftest import make_fake_model as _make_fake_model


# ---------------------------------------------------------------------------
# Mini-repo builder
# ---------------------------------------------------------------------------

def _build_two_file_repo(tmp_path: Path) -> Path:
    """Two-file Python project with .git anchor."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "alpha.py").write_text(
        "def alpha_func(x):\n    \"\"\"Alpha function.\"\"\"\n    return x + 1\n"
    )
    (tmp_path / "beta.py").write_text(
        "def beta_func(y):\n    \"\"\"Beta function.\"\"\"\n    return y * 2\n"
    )
    return tmp_path


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing: {meta_path}"
    return json.loads(meta_path.read_text())


# ===========================================================================
# TEST 1: compute_file_hash call count == 0 on no-op reindex
# ===========================================================================

class TestS5NoRehashOnNoOp:
    """S-5: On a no-op reindex (nothing changed), compute_file_hash must NOT
    be called for any unchanged file.

    The fix (S-5) threads the deriver's sha1_map + old_snapshot through to
    _compute_current_file_hashes so it can reuse already-known SHA-1s.
    On a no-op: all files are in old_snapshot (from the prior cold build) →
    case-2 returns the cached sha1 → compute_file_hash call count == 0.

    RED on HEAD: _compute_current_file_hashes calls compute_file_hash for every
    file on every run (no deriver_sha1_map / old_snapshot parameter exists yet).
    On a 2-file no-op reindex, spy count == 2 instead of 0.
    """

    def test_noop_reindex_does_not_rehash_unchanged_files(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index 2-file repo; no-op reindex; assert spy count == 0.

        Dual-patch: both `tldr.patch.compute_file_hash` and
        `tldr.semantic.compute_file_hash` (if it exists as a local binding
        at module load time) are patched so the spy intercepts regardless of
        how the function is imported inside the callee.

        The spy always delegates to the real hash function so the index remains
        correct — we only COUNT calls, never corrupt the hash.

        RED reason: _compute_current_file_hashes (semantic.py) calls
        `from tldr.patch import compute_file_hash` then invokes it for EVERY
        unit's file (line ~2200). On a no-op, there are 2 files → 2 calls.
        S-5 reduces this to 0 by reusing old_snapshot sha1 for unchanged files.
        """
        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # Import the real hash function once so the spy can delegate to it.
        from tldr.patch import compute_file_hash as _real_hash

        # --- Cold build: establish baseline ---
        # The spy must fire >= 2 times during the cold build to prove it is
        # actually intercepting calls (baseline validity check).
        cold_count = [0]

        def cold_spy(path):
            cold_count[0] += 1
            return _real_hash(path)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.patch.compute_file_hash", side_effect=cold_spy):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert cold_count[0] >= 2, (
            f"Spy baseline INVALID: compute_file_hash must be called >= 2 times "
            f"during the cold build (one per file), got {cold_count[0]}. "
            f"The spy is not intercepting correctly — check the patch target."
        )

        # --- No-op reindex: same files, NO changes ---
        noop_count = [0]

        def noop_spy(path):
            noop_count[0] += 1
            return _real_hash(path)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.patch.compute_file_hash", side_effect=noop_spy):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert noop_count[0] == 0, (
            f"S-5 FAIL: compute_file_hash must NOT be called on a no-op reindex "
            f"(all files unchanged → all in old_snapshot → sha1 reused from case-2). "
            f"Call count: {noop_count[0]} (expected 0). "
            f"RED: _compute_current_file_hashes does not yet accept "
            f"deriver_sha1_map / old_snapshot → calls compute_file_hash for every "
            f"file on every persist, including no-op runs."
        )

    def test_incremental_reindex_does_not_rehash_unchanged_files(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index; edit ONE file; incremental reindex; unchanged file hash count == 0.

        After an incremental edit, the changed file's sha1 comes from the deriver
        (branch-b sha1_map), and the unchanged file's sha1 comes from old_snapshot.
        Neither calls compute_file_hash again at persist time.

        Expected behavior:
          - Cold build: compute_file_hash called >= 2 (one per file, case-3).
          - Incremental after edit: compute_file_hash called == 0
            (changed file sha1 in deriver_sha1_map → case-1;
             unchanged file sha1 in old_snapshot → case-2).

        RED reason: _compute_current_file_hashes has no case-1 (deriver_sha1_map)
        or case-2 (old_snapshot) — falls to case-3 for every file every run.
        Call count == 2 on the incremental run instead of 0.
        """
        from tldr.semantic import build_semantic_index

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        from tldr.patch import compute_file_hash as _real_hash

        cold_count = [0]

        def cold_spy(path):
            cold_count[0] += 1
            return _real_hash(path)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.patch.compute_file_hash", side_effect=cold_spy):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert cold_count[0] >= 2, (
            f"Spy baseline INVALID: cold build must call compute_file_hash >= 2. "
            f"Got {cold_count[0]}. Patch target may be wrong."
        )

        # Edit ONE file
        (project / "beta.py").write_text(
            "def beta_func(y):\n    \"\"\"Beta (modified).\"\"\"\n    return y * 3\n"
        )

        incr_count = [0]

        def incr_spy(path):
            incr_count[0] += 1
            return _real_hash(path)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.patch.compute_file_hash", side_effect=incr_spy):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert incr_count[0] == 0, (
            f"S-5 FAIL: after editing beta.py, the incremental reindex persist must "
            f"NOT call compute_file_hash for any file "
            f"(changed file sha1 in deriver_sha1_map; unchanged file sha1 in "
            f"old_snapshot — neither needs a fresh hash). "
            f"Call count: {incr_count[0]} (expected 0). "
            f"RED: _compute_current_file_hashes does not yet have the 3-case lookup; "
            f"it calls compute_file_hash for every file on every persist run."
        )
