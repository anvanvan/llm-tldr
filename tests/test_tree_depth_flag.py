"""RED tests: `get_file_tree(max_depth=)` + `tree --depth/--max-depth` parity.

Covers architecture behavior 11:
  - `tree X --depth 2` and `tree X --max-depth 2` produce identical output,
    limited to 2 levels below root
  - `--depth 0` -> root node only
  - no flag -> unlimited (today's output, unchanged — control invariant)

Semantics under test (architecture section 3): ``max_depth=N`` lists entries
down to N levels below root; a dir AT the cutoff appears with ``children: []``
(under default ``extensions=None``); ``None`` -> unlimited, byte-identical to
the no-kwarg call.

The api-level tests import `get_file_tree` lazily inside the test body; pre-
implementation they FAIL with TypeError (unexpected keyword 'max_depth').
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _make_tree(root: Path) -> Path:
    """3-level fixture: top.py / lvl1/mid.py / lvl1/lvl2/deep.py / lvl1/lvl2/lvl3/deepest.py."""
    (root / ".git").mkdir(parents=True)
    (root / "top.py").write_text("top = 1\n")
    (root / "lvl1").mkdir()
    (root / "lvl1" / "mid.py").write_text("mid = 1\n")
    (root / "lvl1" / "lvl2").mkdir()
    (root / "lvl1" / "lvl2" / "deep.py").write_text("deep = 1\n")
    (root / "lvl1" / "lvl2" / "lvl3").mkdir()
    (root / "lvl1" / "lvl2" / "lvl3" / "deepest.py").write_text("deepest = 1\n")
    return root


def _find(node: dict, name: str):
    """Depth-first lookup of a node by name in a get_file_tree result."""
    if node.get("name") == name:
        return node
    for child in node.get("children", []):
        found = _find(child, name)
        if found is not None:
            return found
    return None


def _names(node: dict) -> set[str]:
    return {child["name"] for child in node["children"]}


class TestGetFileTreeMaxDepthUnit:
    """api.get_file_tree(max_depth=...) — unit level on a tmp_path tree."""

    def test_max_depth_zero_returns_root_only(self, tmp_path):
        from tldr.api import get_file_tree  # lazy

        root = _make_tree(tmp_path / "treeproj")
        result = get_file_tree(root, max_depth=0)
        assert result == {"name": "treeproj", "type": "dir", "children": []}

    def test_max_depth_one_lists_first_level_with_empty_dir_children(self, tmp_path):
        from tldr.api import get_file_tree  # lazy

        root = _make_tree(tmp_path / "treeproj")
        result = get_file_tree(root, max_depth=1)
        assert _names(result) == {"top.py", "lvl1"}
        lvl1 = _find(result, "lvl1")
        assert lvl1["children"] == [], (
            f"dir at the cutoff must appear with empty children, got {lvl1}"
        )

    def test_max_depth_two_cuts_below_second_level(self, tmp_path):
        from tldr.api import get_file_tree  # lazy

        root = _make_tree(tmp_path / "treeproj")
        result = get_file_tree(root, max_depth=2)
        lvl1 = _find(result, "lvl1")
        assert _names(lvl1) == {"mid.py", "lvl2"}
        lvl2 = _find(result, "lvl2")
        assert lvl2["children"] == [], lvl2
        assert _find(result, "deep.py") is None
        assert _find(result, "lvl3") is None

    def test_max_depth_none_is_unlimited_and_identical_to_default(self, tmp_path):
        from tldr.api import get_file_tree  # lazy

        root = _make_tree(tmp_path / "treeproj")
        assert get_file_tree(root, max_depth=None) == get_file_tree(root)
        assert _find(get_file_tree(root, max_depth=None), "deepest.py") is not None


@pytest.fixture(scope="module")
def tree_proj(tmp_path_factory) -> Path:
    return _make_tree(tmp_path_factory.mktemp("treecli") / "treeproj")


def _run_tree(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tldr.cli", "tree", *args],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=120,
    )


class TestTreeDepthFlagCli:
    """`tldr tree` CLI: --depth / --max-depth parity (behavior 11)."""

    def test_depth_and_max_depth_flags_are_identical(self, tree_proj):
        res_depth = _run_tree([str(tree_proj), "--depth", "2"])
        res_max_depth = _run_tree([str(tree_proj), "--max-depth", "2"])
        assert res_depth.returncode == 0, (
            f"--depth failed: {res_depth.returncode}, stderr: {res_depth.stderr!r}"
        )
        assert res_max_depth.returncode == 0, (
            f"--max-depth failed: {res_max_depth.returncode}, "
            f"stderr: {res_max_depth.stderr!r}"
        )
        assert res_depth.stdout == res_max_depth.stdout

        tree = json.loads(res_depth.stdout)
        lvl2 = _find(tree, "lvl2")
        assert lvl2 is not None and lvl2["children"] == [], lvl2
        assert _find(tree, "deep.py") is None
        assert _find(tree, "deepest.py") is None

    def test_depth_zero_prints_root_only(self, tree_proj):
        result = _run_tree([str(tree_proj), "--depth", "0"])
        assert result.returncode == 0, (
            f"exit {result.returncode}, stderr: {result.stderr!r}"
        )
        assert json.loads(result.stdout) == {
            "name": "treeproj",
            "type": "dir",
            "children": [],
        }

    def test_no_flag_stays_unlimited(self, tree_proj):
        # Control invariant: today's flagless output is unchanged — the whole
        # tree, down to the deepest file. Passes today.
        result = _run_tree([str(tree_proj)])
        assert result.returncode == 0, (
            f"exit {result.returncode}, stderr: {result.stderr!r}"
        )
        tree = json.loads(result.stdout)
        assert _find(tree, "deepest.py") is not None
