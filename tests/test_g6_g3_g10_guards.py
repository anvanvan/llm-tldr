"""
Behaviors (3), (4), (5):

(3) G-6 FALL-THROUGH GUARD: when the daemon-hint (dirty_files_list) is
    non-empty but ALL entries normalize to out-of-scope (not in the live tree),
    the code must FALL THROUGH to the hash floor (branch b), NOT silently
    carry-all as a no-op.  A real change that exists in the live tree must be
    detected even when the hint misses it.

    FAILS on HEAD: branch (a) fires whenever hint is non-empty + trust_hint=True,
    returning changed = _normalize_dirty_files(hint, scan_path) = set().
    An empty changed set means carry-all → the real change is silently skipped.

(4) G-3 GITIGNORE PARITY: _enumerate_live_files must NOT include files that
    are gitignored (excluded by `should_ignore`), because the actual index path
    filters them via `should_ignore`.  A gitignored file absent from the snapshot
    appears as a "new changed" file on every reindex run (false-dirty).

    FAILS on HEAD: _enumerate_live_files uses `load_ignore_patterns` + `match_file`
    (tldrignore only), which does NOT check gitignore.  A gitignored file that is
    not in `.tldrignore` stays in the live set and causes false-dirty every run.

(5) G-10 SUBDIR SNAPSHOT-KEY CONSISTENCY: indexing a subdirectory, editing a
    file in it, then reindexing should produce reused > 0 (snapshot keys are
    scan_path-relative, consistent between runs).  This is a REGRESSION GUARD —
    it currently passes on HEAD and must NOT be removed.

All tests avoid @pytest.mark.e2e and use _make_fake_model().
Stray /private/tmp/.tldr guard: all test repos use .git-anchored tmp_path dirs
to prevent _find_project_root from climbing up to /private/tmp/.tldr.

Runner: python3 -m pytest --no-cov tests/test_g6_g3_g10_guards.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
from conftest import make_fake_model as _make_fake_model


# ---------------------------------------------------------------------------
# Mini-repo helpers
# ---------------------------------------------------------------------------

def _build_two_file_repo(tmp_path: Path) -> Path:
    """Two-file Python project with .git anchor."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "file_a.py").write_text(
        "def foo(x):\n    \"\"\"Function foo.\"\"\"\n    return x + 1\n"
    )
    (tmp_path / "file_b.py").write_text(
        "def bar(y):\n    \"\"\"Function bar.\"\"\"\n    return y * 2\n"
    )
    return tmp_path


def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing: {meta_path}"
    return json.loads(meta_path.read_text())


# ===========================================================================
# BEHAVIOR (3): G-6 Fall-through guard
# ===========================================================================

class TestG6FallThroughGuard:
    """G-6: When a non-empty hint normalizes to an empty changed set (all
    entries out-of-scope), the code must fall through to the hash floor and
    detect real changes.

    Scenario:
      1. Cold-index a 2-file repo.
      2. Edit file_a.py (creates a real change).
      3. Call build_semantic_index with dirty_files=[out_of_scope_path] where
         out_of_scope_path is an absolute path NOT under the project root.
         This is non-empty → trust_hint=True → branch (a) fires TODAY.
         _normalize_dirty_files([out_of_scope]) resolves to set() (path not
         under scan_path → ValueError → filtered out).
         Branch (a) returns changed=set() → carry-all → file_a.py NOT re-parsed.
      4. Assert: file_a.py IS re-parsed after the reindex (the hash floor detected
         the real change).

    RED reason: branch (a) does NOT fall through when changed is empty — it
    returns (set(), deleted) immediately, carrying all files.  The guard that
    should activate the hash floor on empty changed-after-normalize does not exist.
    """

    def test_out_of_scope_hint_falls_through_to_hash_floor(
        self, tmp_path: Path, monkeypatch
    ):
        """Edit file_a; call with out-of-scope hint; assert file_a is re-parsed.

        The out-of-scope hint normalizes to changed=set().  Without the fall-through
        guard, the code carries all files and file_a is NOT re-parsed even though
        it was edited.  The assert on re-embedding proves the fall-through works.

        RED reason: branch (a) returns (set(), set()) immediately when
        changed == set() after normalize — no fall-through to the hash floor.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction as _orig

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Cold build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # Capture text_hash of foo BEFORE the edit
        meta_before = _read_metadata(project)
        foo_before = next(
            (u for u in meta_before["units"] if u.get("name") == "foo"), None
        )
        assert foo_before is not None, "foo must be indexed initially"
        hash_before = foo_before.get("text_hash", "")

        # --- Edit file_a.py (real change) ---
        (project / "file_a.py").write_text(
            "def foo(x):\n    \"\"\"Function foo CHANGED.\"\"\"\n    return x + 100\n"
        )

        # Construct an absolute path that is OUT-OF-SCOPE for this project
        # (a path under a completely different root that cannot be made relative
        # to project).
        out_of_scope_path = str(tmp_path.parent / "completely_other_dir" / "some_file.py")
        assert not str(out_of_scope_path).startswith(str(project)), (
            "out_of_scope_path must not be under project root for this test to be valid"
        )

        # --- Reindex with hint that normalizes to empty ---
        parse_paths: list[str] = []

        def spy_pfex(file_info, *args, **kwargs):
            parse_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.semantic._process_file_for_extraction", side_effect=spy_pfex):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                dirty_files=[out_of_scope_path],  # hint: non-empty but all out-of-scope
            )

        # The fall-through guard must have activated the hash floor, which detects
        # the real change to file_a.py.  Assert file_a.py WAS re-parsed.
        assert any("file_a" in p for p in parse_paths), (
            f"G-6 FALL-THROUGH FAIL: file_a.py was edited but NOT re-parsed after "
            f"reindex with an out-of-scope hint. "
            f"The hint normalized to changed=set(); without the fall-through guard, "
            f"branch (a) carries all files and skips the hash floor. "
            f"Paths processed: {parse_paths}. "
            f"RED: branch (a) in _derive_dirty_set must fall through to branch (b) "
            f"when changed == set() after _normalize_dirty_files — currently it "
            f"returns (set(), deleted) immediately, silently carrying all files."
        )

        # Also verify the real change IS reflected in the index
        meta_after = _read_metadata(project)
        foo_after = next(
            (u for u in meta_after["units"] if u.get("name") == "foo"), None
        )
        assert foo_after is not None, "foo must still be in metadata after reindex"
        hash_after = foo_after.get("text_hash", "")

        assert hash_after != hash_before, (
            f"foo.text_hash must change after editing file_a.py. "
            f"Before: {hash_before!r}, after: {hash_after!r}. "
            f"The real change was not detected (carry-all silently ignored the edit). "
            f"RED: G-6 fall-through guard not implemented."
        )

    def test_in_scope_hint_still_processes_changed_file(
        self, tmp_path: Path, monkeypatch
    ):
        """Control: a valid in-scope hint correctly identifies the changed file.

        This verifies the hint path works for a valid path — if this passes but
        the out-of-scope test fails, the fall-through guard is the specific gap.

        Note: this test DOES depend on the hint path working correctly (branch a
        for a valid in-scope path). It may pass or fail independently on HEAD.
        It is included as a regression guard for the happy-path hint.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction as _orig

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project = _build_two_file_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        (project / "file_a.py").write_text(
            "def foo(x):\n    \"\"\"Function foo CHANGED.\"\"\"\n    return x + 100\n"
        )

        # Valid in-scope hint: absolute path to file_a.py
        in_scope_path = str(project / "file_a.py")

        parse_paths: list[str] = []

        def spy_pfex(file_info, *args, **kwargs):
            parse_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.semantic._process_file_for_extraction", side_effect=spy_pfex):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                dirty_files=[in_scope_path],
            )

        assert any("file_a" in p for p in parse_paths), (
            f"Control: a valid in-scope hint must cause file_a.py to be re-parsed. "
            f"Paths: {parse_paths}"
        )


# ===========================================================================
# BEHAVIOR (4): G-3 Gitignore parity — _enumerate_live_files
# ===========================================================================

class TestG3GitignoreParity:
    """G-3: _enumerate_live_files must exclude files that are gitignored,
    exactly as extract_units_from_project does via should_ignore.

    A gitignored file NOT in the live set cannot become false-dirty (it is
    never compared against the snapshot). On HEAD, _enumerate_live_files uses
    `load_ignore_patterns(project).match_file(rel_path)` (tldrignore only) and
    does NOT call `should_ignore`, so gitignored files remain in the live set.

    Scenario: create a real git repo (so `git check-ignore` works); add a .gitignore
    that excludes `generated.py`; cold-index; assert `generated.py` is NOT in the
    live set returned by `_enumerate_live_files`.

    RED reason: `_enumerate_live_files` only calls `ignore_spec.match_file`
    (tldrignore), never `should_ignore` (gitignore). Gitignored files stay in
    the live set.
    """

    def test_gitignored_file_absent_from_live_set(self, tmp_path: Path):
        """_enumerate_live_files must exclude a gitignored file.

        Creates a real git repo so that `git check-ignore` recognizes
        `.gitignore` patterns. Then calls `_enumerate_live_files` directly and
        asserts the gitignored file is absent.

        RED reason: `_enumerate_live_files` does not call `should_ignore` — it
        only applies `.tldrignore` patterns via `load_ignore_patterns().match_file()`.
        A `.gitignore`-only pattern has no effect on the live set. The gitignored
        file appears in the live set, causing a false-dirty on every reindex.
        """
        from tldr.semantic import _enumerate_live_files

        # Create a real git repo so git check-ignore is functional.
        # Use subprocess.run to initialise git — a bare .git dir is NOT
        # enough for git check-ignore to recognise patterns.
        result = subprocess.run(
            ["git", "init", str(tmp_path)],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            pytest.skip("git init failed — git not available in this environment")

        subprocess.run(
            ["git", "config", "user.email", "test@tldr.test"],
            cwd=str(tmp_path), capture_output=True, timeout=5,
        )
        subprocess.run(
            ["git", "config", "user.name", "TLDRTest"],
            cwd=str(tmp_path), capture_output=True, timeout=5,
        )

        # Project files
        (tmp_path / "module.py").write_text(
            "def real_func():\n    \"\"\"A real function.\"\"\"\n    return 1\n"
        )
        (tmp_path / "generated.py").write_text(
            "def generated_func():\n    \"\"\"Generated code.\"\"\"\n    return 0\n"
        )

        # .gitignore excludes generated.py
        (tmp_path / ".gitignore").write_text("generated.py\n")

        # Commit everything so git check-ignore can resolve patterns.
        subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=str(tmp_path), capture_output=True, timeout=10,
        )

        # Verify git check-ignore works for this setup (diagnostic skip if broken).
        check_result = subprocess.run(
            ["git", "check-ignore", "-q", "generated.py"],
            cwd=str(tmp_path), capture_output=True, timeout=5,
        )
        if check_result.returncode != 0:
            pytest.skip(
                "git check-ignore does not recognise generated.py as ignored — "
                "git setup may not be complete in this environment"
            )

        # Now test _enumerate_live_files with respect_ignore=True.
        live = _enumerate_live_files(tmp_path, lang="python", respect_ignore=True)

        assert isinstance(live, set), (
            f"_enumerate_live_files must return a set, got {type(live).__name__}"
        )

        assert "generated.py" not in live, (
            f"G-3 FAIL: `generated.py` is gitignored but appears in the live set. "
            f"Live files: {sorted(live)}. "
            f"_enumerate_live_files must exclude gitignored files exactly as "
            f"extract_units_from_project does via should_ignore. "
            f"RED: _enumerate_live_files uses load_ignore_patterns().match_file() "
            f"(tldrignore only) — it does NOT call should_ignore and therefore "
            f"ignores .gitignore patterns."
        )

        assert "module.py" in live, (
            f"module.py (not gitignored) must be in the live set. "
            f"Live: {sorted(live)}"
        )

    def test_false_dirty_caused_by_gitignored_file(self, tmp_path: Path, monkeypatch):
        """After cold-index, _derive_dirty_set must NOT report a gitignored file
        as dirty on a no-op reindex.

        A gitignored file is NEVER in the snapshot (it was never indexed), so
        if it IS in the live set, it appears as 'new' (not in snapshot) → dirty.
        This test asserts the dirty set is empty on a true no-op.

        Note: because extract_units_from_project filters gitignored files via
        should_ignore, the gitignored file never actually causes a re-embed
        (the parse step filters it). The test therefore focuses on the
        _enumerate_live_files / _derive_dirty_set seam, not the end embedding.

        RED reason: _enumerate_live_files includes the gitignored file in the
        live set → _derive_dirty_set returns {\"generated.py\"} as changed even
        on a no-op → this IS a false-dirty (the file changes `changed` set,
        triggers a parse attempt, and the parse is filtered — but wastefully).
        """
        from tldr.semantic import _enumerate_live_files

        result = subprocess.run(
            ["git", "init", str(tmp_path)],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            pytest.skip("git init failed — git not available")

        subprocess.run(
            ["git", "config", "user.email", "test@tldr.test"],
            cwd=str(tmp_path), capture_output=True, timeout=5,
        )
        subprocess.run(
            ["git", "config", "user.name", "TLDRTest"],
            cwd=str(tmp_path), capture_output=True, timeout=5,
        )

        (tmp_path / "module.py").write_text(
            "def real_func():\n    \"\"\"A real function.\"\"\"\n    return 1\n"
        )
        (tmp_path / "generated.py").write_text(
            "def gen_func():\n    \"\"\"Generated code.\"\"\"\n    return 0\n"
        )
        (tmp_path / ".gitignore").write_text("generated.py\n")

        subprocess.run(["git", "add", "-A"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=str(tmp_path), capture_output=True, timeout=10,
        )

        check = subprocess.run(
            ["git", "check-ignore", "-q", "generated.py"],
            cwd=str(tmp_path), capture_output=True, timeout=5,
        )
        if check.returncode != 0:
            pytest.skip("git check-ignore not functional in this environment")

        # Cold build with respect_ignore=True
        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")
        from tldr.semantic import build_semantic_index

        fake_model = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(tmp_path), lang="python",
                show_progress=False, respect_ignore=True,
            )

        # Check that generated.py is NOT in the live set after build.
        # This is the root fix: if generated.py is absent from live, it cannot
        # be false-dirty.
        live = _enumerate_live_files(tmp_path, lang="python", respect_ignore=True)

        assert "generated.py" not in live, (
            f"G-3 FAIL: generated.py (gitignored) is still in the live set after "
            f"cold build. This causes it to be 'false-dirty' on every subsequent "
            f"reindex (not in snapshot → appears as new/changed in _derive_dirty_set). "
            f"Live set: {sorted(live)}. "
            f"RED: _enumerate_live_files must apply should_ignore (gitignore) just "
            f"like the index path, but currently only applies tldrignore patterns."
        )


# ===========================================================================
# BEHAVIOR (5): G-10 Subdir snapshot-key consistency (REGRESSION GUARD)
# ===========================================================================

class TestG10SubdirSnapshotKeyConsistency:
    """G-10 REGRESSION GUARD: indexing a subdirectory (not the project root),
    editing a file in it, then reindexing must produce reused > 0 (at least one
    unchanged file is carried).

    Snapshot keys are scan_path-relative (e.g. "main.py"), NOT project_root-
    relative (e.g. "src/main.py"). If the key logic were inconsistent, every
    reindex of a subdir would look like a first-run (no snapshot matches) and
    reused would be 0.

    This test CURRENTLY PASSES on HEAD (snapshot keys are correctly scan_path-
    relative). It is included as a regression guard so that any future refactor
    that accidentally changes key relativity will be caught immediately.

    The test is NOT a RED test; it documents behavior that must be preserved.
    """

    def test_subdir_reindex_reuses_unchanged_file(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index src/ subdir; edit main.py; reindex; assert utils.py not re-parsed.

        Proves snapshot keys are scan_path-relative (consistent across runs when
        scan_path is a subdir of the project root). If keys were project_root-
        relative, the second run would see "main.py" not in snapshot["src/main.py"]
        and treat every file as new → 0 reused.

        This test CURRENTLY PASSES. It is a regression guard.
        NOTE: listed separately in test-plan.md as a pre-passing regression guard.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction as _orig

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        # Create a project: root has .git, subdir src/ has the code.
        # _find_project_root will anchor at root (has .git), but scan_path = root/src.
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        src_dir = project_root / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text(
            "def main_func():\n    \"\"\"Main function.\"\"\"\n    return 1\n"
        )
        (src_dir / "utils.py").write_text(
            "def util_func():\n    \"\"\"Utility function.\"\"\"\n    return 2\n"
        )

        fake_model = _make_fake_model()

        # --- Cold index of src/ subdir ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            total_cold = build_semantic_index(
                str(src_dir), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert total_cold >= 2, f"Expected >= 2 units in cold build, got {total_cold}"

        # --- Edit main.py (one changed file; utils.py unchanged) ---
        (src_dir / "main.py").write_text(
            "def main_func():\n    \"\"\"Main function (updated).\"\"\"\n    return 42\n"
        )

        # --- Reindex src/ subdir ---
        parse_paths: list[str] = []

        def spy_pfex(file_info, *args, **kwargs):
            parse_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.semantic._process_file_for_extraction", side_effect=spy_pfex):
            build_semantic_index(
                str(src_dir), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # utils.py must NOT be re-parsed (unchanged; snapshot key matches).
        # If snapshot keys were project_root-relative ("src/utils.py") instead of
        # scan_path-relative ("utils.py"), the deriver would not find "utils.py"
        # in the snapshot → treats it as new → re-parses it → this assertion fails.
        assert not any("utils" in p for p in parse_paths), (
            f"G-10 REGRESSION: utils.py was re-parsed even though it did not change. "
            f"This means snapshot keys are NOT scan_path-relative — the deriver could "
            f"not match 'utils.py' in the snapshot. "
            f"Paths processed: {parse_paths}. "
            f"NOTE: if this test currently passes on HEAD, it is a regression guard "
            f"(see test-plan.md). If it fails, snapshot-key relativity has regressed."
        )

        # main.py MUST have been re-parsed (it changed).
        assert any("main" in p for p in parse_paths), (
            f"main.py must be re-parsed after being edited. "
            f"Paths: {parse_paths}"
        )

    def test_subdir_noop_reindex_reuses_all_files(
        self, tmp_path: Path, monkeypatch
    ):
        """Cold-index src/ subdir; no-op reindex; assert nothing is re-parsed.

        Confirms full snapshot-key consistency on a no-op: all files are in the
        snapshot, none appear as new/changed.

        This test CURRENTLY PASSES on HEAD. It is a regression guard.
        NOTE: listed separately in test-plan.md as a pre-passing regression guard.
        """
        from tldr.semantic import build_semantic_index, _process_file_for_extraction as _orig

        monkeypatch.setenv("TLDR_MAX_WORKERS", "1")

        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        src_dir = project_root / "src"
        src_dir.mkdir()
        (src_dir / "alpha.py").write_text(
            "def alpha():\n    \"\"\"Alpha.\"\"\"\n    return 1\n"
        )
        (src_dir / "beta.py").write_text(
            "def beta():\n    \"\"\"Beta.\"\"\"\n    return 2\n"
        )

        fake_model = _make_fake_model()

        # Cold build
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(src_dir), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # No-op: same files, no changes
        parse_paths: list[str] = []

        def spy_pfex(file_info, *args, **kwargs):
            parse_paths.append(str(file_info.get("path", "")))
            return _orig(file_info, *args, **kwargs)

        with patch("tldr.semantic.get_model", return_value=fake_model), \
             patch("tldr.semantic._process_file_for_extraction", side_effect=spy_pfex):
            build_semantic_index(
                str(src_dir), lang="python",
                show_progress=False, respect_ignore=False,
            )

        py_files_parsed = [p for p in parse_paths if p.endswith(".py")]
        assert len(py_files_parsed) == 0, (
            f"G-10 REGRESSION: no-op reindex on subdir re-parsed files: "
            f"{py_files_parsed}. "
            f"Snapshot keys are inconsistent between cold build and reindex "
            f"(may have switched from scan_path-relative to project_root-relative). "
            f"NOTE: if this test currently passes, it is a regression guard."
        )
