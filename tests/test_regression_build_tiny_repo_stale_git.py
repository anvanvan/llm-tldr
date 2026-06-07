"""
Regression test for: flaky FileExistsError in _build_tiny_repo on recycled tmp dirs.

Root cause (verified F-1):
  Prior to this fix, tests/test_incremental_semantic_index.py:91 called
  (tmp_path / ".git").mkdir() with implicit exist_ok=False. This has been
  corrected to mkdir(exist_ok=True). When conftest.py pins basetemp to a flat
  /tmp/tldr-pt-{uid}/ (no per-session subdir), stale numbered tmp dirs from prior
  pytest runs are re-used non-empty. A stale .git from a prior run causes mkdir to
  raise FileExistsError at test setup, before any indexing logic runs.

This test makes the failure DETERMINISTIC: it explicitly pre-creates a stale .git
inside a fresh temp dir, then calls the real _build_tiny_repo helper. On unfixed
code (exist_ok=False) this raises FileExistsError (RED). After the fix
(exist_ok=True or shutil.rmtree before mkdir) this succeeds (GREEN).

Both helpers are exercised: _build_tiny_repo and _build_tiny_repo_three_files.

See:
  tests/test_incremental_semantic_index.py:85-94, 97-103 (the helpers)
  tests/conftest.py:147-150 (flat basetemp enabler)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import the real helpers from the test module under diagnosis.
# We use a session fixture so the import only happens when tests actually run,
# not during test collection.
#
# The helpers are module-level functions (_build_tiny_repo, _build_tiny_repo_three_files)
# in tests/test_incremental_semantic_index.py. Since conftest adds `tests/` to
# sys.path (rootdir is the repo root), we import via the spec.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def import_helpers():
    """Lazy import of helpers from test module under diagnosis.

    Loaded only when a test actually requests these helpers,
    not during test collection.

    Returns (build_tiny_repo, build_tiny_repo_three_files).
    Raises ImportError / AttributeError clearly if the helpers are renamed or removed.
    """
    cached_key = "test_incremental_semantic_index"
    if cached_key in sys.modules:
        mod = sys.modules[cached_key]
    else:
        mod_path = Path(__file__).parent / "test_incremental_semantic_index.py"
        spec = importlib.util.spec_from_file_location(cached_key, mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[cached_key] = mod

    build_fn = getattr(mod, "_build_tiny_repo")
    build_three_fn = getattr(mod, "_build_tiny_repo_three_files")
    return build_fn, build_three_fn


@pytest.fixture(autouse=True)
def setup_stale_git(tmp_path: Path):
    """Auto-create stale .git in each test to simulate recycled basetemp state."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    yield


# ===========================================================================
# Regression tests — RED on unfixed code, GREEN after fix
# ===========================================================================


class TestBuildTinyRepoToleratesPreexistingGitDir:
    """Deterministic reproduction of flaky FileExistsError on recycled tmp dirs.

    Each test pre-creates a stale .git inside a fresh tmp directory to simulate
    the recycled-basetemp state that caused ~42% failures in CI (reproduction.md
    Step 5).

    Prior to the fix: _build_tiny_repo() called (tmp_path / ".git").mkdir()
    with implicit exist_ok=False → would raise FileExistsError when .git already exists.

    After the fix: mkdir(exist_ok=True) makes the call safe regardless of existing state.
    """

    def test_build_tiny_repo_tolerates_preexisting_git_dir(
        self, tmp_path: Path, import_helpers
    ):
        """_build_tiny_repo must NOT raise FileExistsError when .git already exists.

        Regression: flaky FileExistsError at test_incremental_semantic_index.py:91
        when pytest recycles a numbered tmp dir that retains a stale .git from a
        prior run. This test makes the condition deterministic by pre-creating .git
        before calling the helper.

        RED (unfixed):  FileExistsError: [Errno 17] File exists: '<tmp>/.git'
        GREEN (fixed):  no exception; helper completes and returns tmp_path
        """
        build_fn, _ = import_helpers

        # Act + Assert: calling the helper must NOT raise FileExistsError
        # .git is pre-created by setup_stale_git fixture.
        # On unfixed code (exist_ok=False), this is where FileExistsError fires.
        result = build_fn(tmp_path)

        # If we reach here the fix is in place
        assert result == tmp_path, "Helper must return tmp_path unchanged"
        assert (tmp_path / "file_a.py").exists(), "file_a.py must be created by helper"
        assert (tmp_path / "file_b.py").exists(), "file_b.py must be created by helper"

    def test_build_tiny_repo_three_files_tolerates_preexisting_git_dir(
        self, tmp_path: Path, import_helpers
    ):
        """_build_tiny_repo_three_files must NOT raise FileExistsError when .git already exists.

        Same root cause as test_build_tiny_repo_tolerates_preexisting_git_dir but
        exercises the sibling helper at test_incremental_semantic_index.py:97-103, which
        has the identical mkdir(exist_ok=True) call.

        RED (unfixed):  FileExistsError: [Errno 17] File exists: '<tmp>/.git'
        GREEN (fixed):  no exception; all three source files are created
        """
        _, build_three_fn = import_helpers

        # Act + Assert
        # .git is pre-created by setup_stale_git fixture.
        result = build_three_fn(tmp_path)

        assert result == tmp_path, "Helper must return tmp_path unchanged"
        assert (tmp_path / "file_a.py").exists(), "file_a.py must be created"
        assert (tmp_path / "file_b.py").exists(), "file_b.py must be created"
        assert (tmp_path / "file_c.py").exists(), "file_c.py must be created"

    def test_build_tiny_repo_with_git_false_never_raises_on_preexisting_git(
        self, tmp_path: Path, import_helpers
    ):
        """_build_tiny_repo(with_git=False) must never raise even if .git exists.

        This is a control-path test: when with_git=False, _build_tiny_repo skips
        the mkdir entirely, so the stale-.git condition can never affect it.
        Documenting this as a test makes the fix boundary explicit — we only need
        to fix the with_git=True code path.
        """
        build_fn, _ = import_helpers

        # with_git=False: should NOT touch .git at all
        # .git is pre-created by setup_stale_git fixture.
        result = build_fn(tmp_path, with_git=False)

        assert result == tmp_path
        assert (tmp_path / "file_a.py").exists()
        assert (tmp_path / "file_b.py").exists()
