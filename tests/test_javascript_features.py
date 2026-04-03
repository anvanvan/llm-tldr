"""Regression tests for JavaScript call-graph and context support."""

from pathlib import Path

import pytest

from tldr.api import get_relevant_context
from tldr.cross_file_calls import TREE_SITTER_AVAILABLE, build_function_index, build_project_call_graph


pytestmark = pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter-typescript not available")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_call_graph_dispatch_indexes_javascript_symbols(tmp_path: Path) -> None:
    _write(
        tmp_path / "esm.js",
        """
export function esHelper(value) {
  return value.toUpperCase();
}
""".strip(),
    )
    _write(
        tmp_path / "cjs.js",
        """
exports.helper = function (value) {
  return value.toLowerCase();
};
""".strip(),
    )

    index = build_function_index(tmp_path, language="javascript")

    assert ("esm", "esHelper") in index
    assert ("cjs", "helper") in index
    assert "esm/esHelper" in index
    assert "cjs/helper" in index


def test_calls_builds_edges_for_javascript_esm(tmp_path: Path) -> None:
    _write(
        tmp_path / "helper.js",
        """
export function helper(input) {
  return input + "!";
}
""".strip(),
    )
    _write(
        tmp_path / "main.js",
        """
import { helper } from "./helper.js";

export function main() {
  return helper("ok");
}
""".strip(),
    )

    graph = build_project_call_graph(tmp_path, language="javascript", use_workspace_config=False)
    assert ("main.js", "main", "helper.js", "helper") in graph.edges


@pytest.mark.parametrize("module_ext", [".js", ".cjs"])
def test_calls_builds_edges_for_javascript_commonjs_destructure(tmp_path: Path, module_ext: str) -> None:
    _write(
        tmp_path / f"helper{module_ext}",
        """
exports.helper = function (input) {
  return input + "-done";
};
""".strip(),
    )
    _write(
        tmp_path / "main.js",
        """
const { helper } = require("./helper");

function main() {
  return helper("work");
}
""".strip(),
    )

    graph = build_project_call_graph(tmp_path, language="javascript", use_workspace_config=False)
    assert ("main.js", "main", f"helper{module_ext}", "helper") in graph.edges


@pytest.mark.parametrize("module_ext", [".js", ".cjs"])
def test_calls_builds_edges_for_javascript_commonjs_namespace(tmp_path: Path, module_ext: str) -> None:
    _write(
        tmp_path / f"helper{module_ext}",
        """
module.exports.helper = function (input) {
  return input + "-done";
};
""".strip(),
    )
    _write(
        tmp_path / "main.js",
        """
const helper = require("./helper");

function main() {
  return helper.helper("work");
}
""".strip(),
    )

    graph = build_project_call_graph(tmp_path, language="javascript", use_workspace_config=False)
    assert ("main.js", "main", f"helper{module_ext}", "helper") in graph.edges


@pytest.mark.parametrize("module_ext", [".js", ".cjs"])
def test_calls_builds_edges_for_javascript_commonjs_default_export(tmp_path: Path, module_ext: str) -> None:
    _write(
        tmp_path / f"worker{module_ext}",
        """
module.exports = function doThing(input) {
  return input + "-done";
};
""".strip(),
    )
    _write(
        tmp_path / "main.js",
        """
const work = require("./worker");

function run() {
  return work("job");
}
""".strip(),
    )

    graph = build_project_call_graph(tmp_path, language="javascript", use_workspace_config=False)
    assert ("main.js", "run", f"worker{module_ext}", "doThing") in graph.edges
    assert ("main.js", "run", "main.js", "work") not in graph.edges


@pytest.mark.parametrize("module_ext", [".js", ".cjs"])
def test_calls_commonjs_default_export_does_not_resolve_to_local_symbol(tmp_path: Path, module_ext: str) -> None:
    _write(
        tmp_path / f"worker{module_ext}",
        """
function internal(input) {
  return input + "-i";
}

module.exports = function (input) {
  return internal(input);
};
""".strip(),
    )
    _write(
        tmp_path / "main.js",
        """
const work = require("./worker");

function run() {
  return work("job");
}
""".strip(),
    )

    graph = build_project_call_graph(tmp_path, language="javascript", use_workspace_config=False)
    assert ("main.js", "run", f"worker{module_ext}", "default") in graph.edges
    assert ("main.js", "run", f"worker{module_ext}", "internal") not in graph.edges


def test_context_includes_javascript_callee_and_cfg_metrics(tmp_path: Path) -> None:
    _write(
        tmp_path / "helper.js",
        """
export function helper(value) {
  return value + "!";
}
""".strip(),
    )
    _write(
        tmp_path / "main.js",
        """
import { helper } from "./helper.js";

export function main(flag) {
  if (flag) {
    return helper("yes");
  }
  return helper("no");
}
""".strip(),
    )

    context = get_relevant_context(tmp_path, "main", depth=2, language="javascript")
    names = {func.name for func in context.functions}

    assert any(name == "main" or name.endswith(".main") for name in names)
    assert any(name == "helper" or name.endswith(".helper") for name in names)

    main_ctx = next(func for func in context.functions if func.name == "main" or func.name.endswith(".main"))
    assert main_ctx.blocks is not None
    assert main_ctx.cyclomatic is not None


def test_context_module_query_resolves_mjs(tmp_path: Path) -> None:
    _write(
        tmp_path / "pkg" / "util.mjs",
        """
export function helper(value) {
  return value + 1;
}
""".strip(),
    )

    context = get_relevant_context(tmp_path, "pkg/util", depth=1, language="javascript")
    names = {func.name for func in context.functions}
    assert "helper" in names
