"""Regression tests for project-root resolution + .tldrignore creation.

A stray ``.tldr`` cache directory in a subdirectory used to shadow the real
project root: ``.tldr`` was listed in ``PROJECT_ROOT_MARKERS`` alongside the
strong VCS/build markers, and ``_find_project_root`` returned the *closest*
ancestor with *any* marker. So once any subdir-scoped index (or a tldr query
run with ``--path <subdir>``) created ``<subdir>/.tldr``, every later resolution
from under that subdir stopped there instead of climbing to the repo's
``pyproject.toml`` / ``.git`` — fragmenting the index AND scattering boilerplate
``.tldrignore`` files (which, unlike ``.tldr/``, are NOT gitignored) into the
source tree.

Fix (two layers):
1. ``_find_project_root`` gives STRONG markers (.git/pyproject.toml/package.json/
   Cargo.toml/go.mod) precedence — a full upward pass — and only falls back to
   the ``.tldr`` cache marker when no strong marker exists anywhere up the tree.
2. ``ensure_tldrignore`` refuses to create a nested ``.tldrignore`` when an
   ancestor (up to the strong project root) already owns one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_claude_project_dir(monkeypatch):
    # _find_project_root short-circuits on CLAUDE_PROJECT_DIR; isolate from it.
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)


def test_find_project_root_ignores_stray_tldr_in_subdir(tmp_path):
    """A stray .tldr cache in a subdir must NOT shadow the repo's real root."""
    from tldr.semantic import _find_project_root

    repo = tmp_path / "repo"
    sub = repo / "pkg" / "sub"
    sub.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    # Simulate a prior subdir-scoped index leaving a sticky cache marker.
    (repo / "pkg" / ".tldr").mkdir()

    # From deep inside, and from the stray-.tldr dir itself, resolve to the repo.
    assert _find_project_root(sub) == repo.resolve()
    assert _find_project_root(repo / "pkg") == repo.resolve()


def test_find_project_root_strong_marker_beats_closer_tldr(tmp_path):
    """Even when a subdir's .tldr is CLOSER than the strong marker above it,
    the strong marker wins (precedence, not proximity, for .tldr)."""
    from tldr.semantic import _find_project_root

    repo = tmp_path / "repo"
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    (repo / ".git").mkdir()
    (sub / ".tldr").mkdir()  # closest marker is the stray cache

    assert _find_project_root(sub) == repo.resolve()


def test_find_project_root_falls_back_to_tldr_when_no_strong_marker(tmp_path):
    """Standalone non-VCS project: with no strong marker anywhere up-tree, reuse
    the existing .tldr cache root instead of fragmenting per subdir."""
    from tldr.semantic import _find_project_root

    proj = tmp_path / "standalone"
    sub = proj / "sub"
    sub.mkdir(parents=True)
    (proj / ".tldr").mkdir()

    assert _find_project_root(sub) == proj.resolve()


def test_ensure_tldrignore_skips_when_ancestor_has_one(tmp_path):
    """Never scatter a nested .tldrignore when an ancestor already owns one."""
    from tldr.tldrignore import ensure_tldrignore

    root = tmp_path / "repo"
    sub = root / "pkg"
    sub.mkdir(parents=True)
    (root / "pyproject.toml").write_text("x")
    (root / ".tldrignore").write_text("# root ignore\n")

    created, msg = ensure_tldrignore(sub)
    assert created is False
    assert not (sub / ".tldrignore").exists(), "must not create a nested .tldrignore"


def test_ensure_tldrignore_creates_at_clean_root(tmp_path):
    """A real root with no ancestor .tldrignore still gets one created."""
    from tldr.tldrignore import ensure_tldrignore

    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("x")

    created, msg = ensure_tldrignore(root)
    assert created is True
    assert (root / ".tldrignore").exists()
