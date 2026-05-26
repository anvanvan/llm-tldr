"""Codebase analysis tools built on TLDR's call graph.

Provides:
- Impact analysis: Find all callers of a function (reverse call graph)
- Dead code detection: Find unreachable functions
- Architecture extraction: Detect layers from call patterns

These operate on the call graph from cross_file_calls.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .api import _strip_namespace_qualifier, build_project_call_graph

if TYPE_CHECKING:
    from .cross_file_calls import ProjectCallGraph


@dataclass
class FunctionRef:
    """A function reference in the codebase."""

    file: str
    name: str

    def __hash__(self):
        return hash((self.file, self.name))

    def __eq__(self, other):
        if not isinstance(other, FunctionRef):
            return False
        return self.file == other.file and self.name == other.name

    def __repr__(self):
        return f"{self.file}:{self.name}"


def build_reverse_graph(
    edges: Iterable[tuple[str, str, str, str]],
) -> dict[FunctionRef, list[FunctionRef]]:
    """Build reverse call graph: who calls each function?

    Args:
        edges: Iterable of (from_file, from_func, to_file, to_func) tuples

    Returns:
        Dict mapping callee -> list of callers
    """
    reverse = defaultdict(list)
    for from_file, from_func, to_file, to_func in edges:
        callee = FunctionRef(file=to_file, name=to_func)
        caller = FunctionRef(file=from_file, name=from_func)
        reverse[callee].append(caller)
    return reverse


def build_forward_graph(
    edges: Iterable[tuple[str, str, str, str]],
) -> dict[FunctionRef, list[FunctionRef]]:
    """Build forward call graph: what does each function call?

    Args:
        edges: Iterable of (from_file, from_func, to_file, to_func) tuples

    Returns:
        Dict mapping caller -> list of callees
    """
    forward = defaultdict(list)
    for from_file, from_func, to_file, to_func in edges:
        caller = FunctionRef(file=from_file, name=from_func)
        callee = FunctionRef(file=to_file, name=to_func)
        forward[caller].append(callee)
    return forward


def impact_analysis(
    call_graph: "ProjectCallGraph",
    target_func: str,
    max_depth: int = 3,
    target_file: str | None = None,
    language: str | None = None,
    _reverse: dict | None = None,
) -> dict:
    """Find all callers of a function, up to max_depth levels.

    This is the reverse call graph - useful for understanding
    what code would be affected by changing a function.

    Args:
        call_graph: ProjectCallGraph from cross_file_calls
        target_func: Function name to find callers of
        max_depth: How deep to traverse callers
        target_file: Optional file filter
        _reverse: Optional pre-built reverse graph; if provided, skips
            calling build_reverse_graph (optimization for repeated calls)

    Returns:
        Dict with 'targets' (tree of callers) and 'total_targets' count
    """
    edges = call_graph.edges
    reverse = _reverse if _reverse is not None else build_reverse_graph(edges)

    # Find target function(s) as callees (functions being called)
    # For PHP, names may be ClassName::method — match both qualified and bare names
    is_php = language == "php"
    norm_target_func = target_func.replace(".", "::") if is_php else target_func

    # Precompute distinct candidate names in the graph and the set of full names
    # that share each bare suffix. Used to gate the qualified/bare fallbacks so
    # they only fire when there's exactly one unambiguous candidate — preventing
    # cross-module false positives in qualified-name languages (Elixir, Ruby,
    # Java, Go-methods) where `Mod1.run` and `Mod2.run` must not collide.
    _all_names: set[str] = set()
    for _ff, _fn, _tf, _tn in edges:
        _all_names.add(_fn)
        _all_names.add(_tn)
    _bare_to_full: dict[str, set[str]] = defaultdict(set)
    for _name in _all_names:
        _norm = _name.replace(".", "::") if is_php else _name
        _bare = _strip_namespace_qualifier(_norm)
        if _bare:
            _bare_to_full[_bare].add(_norm)
        else:
            # Already bare — record itself under its own key
            _bare_to_full[_norm].add(_norm)

    def _matches_target(func_name: str) -> bool:
        if is_php:
            # Normalize '.' to '::' so "Class.method" matches "Class::method"
            norm_func = func_name.replace(".", "::")
        else:
            norm_func = func_name
        if norm_func == norm_target_func:
            return True
        # Use centralized utility for extracting bare suffix from qualified names
        norm_func_bare = _strip_namespace_qualifier(norm_func)
        norm_target_bare = _strip_namespace_qualifier(norm_target_func)
        # Bare suffix used to look up unique-candidate counts: prefer the
        # qualified side's bare suffix; otherwise the already-bare side itself.
        lookup_bare = norm_target_bare or norm_func_bare or norm_target_func
        # Number of distinct full names in the graph sharing this bare suffix.
        # If >1, accepting any qualified/bare fallback would conflate them.
        # PHP retains its historical behavior (fallback always allowed) — its
        # call graph stores both qualified and bare under disambiguating
        # ClassName::method form and the test suite depends on permissive
        # matching for PHP class methods.
        candidate_count = len(_bare_to_full.get(lookup_bare, ()))
        unique_candidate = is_php or candidate_count <= 1
        if norm_func_bare and norm_func_bare == norm_target_func:
            if unique_candidate:
                return True
        if norm_target_bare and norm_target_bare == norm_func:
            if unique_candidate:
                return True
        if norm_func_bare and norm_target_bare and norm_func_bare == norm_target_bare:
            # Strict: both sides qualified — require the namespace prefix to
            # match exactly. This blocks `Mod1.run` from matching `Mod2.run`.
            if is_php or norm_func == norm_target_func:
                return True
        return False

    all_callees = set()
    for from_file, from_func, to_file, to_func in edges:
        callee = FunctionRef(file=to_file, name=to_func)
        if _matches_target(callee.name):
            if target_file is None or target_file in callee.file:
                all_callees.add(callee)

    targets = list(all_callees)

    if not targets:
        # Function not found as callee - check if it exists as a caller
        # (function calls others but is never called itself = entry point)
        callers_only = set()
        for from_file, from_func, to_file, to_func in edges:
            if _matches_target(from_func):
                if target_file is None or target_file in from_file:
                    callers_only.add(FunctionRef(file=from_file, name=from_func))

        if callers_only:
            # Function exists in graph but has no callers - return entry point info
            return {
                "targets": {
                    str(ref): {
                        "function": ref.name,
                        "file": ref.file,
                        "caller_count": 0,
                        "callers": [],
                        "truncated": False,
                        "note": "Entry point - never called by other code in graph",
                    }
                    for ref in callers_only
                },
                "total_targets": len(callers_only),
            }

        return {"error": f"Function '{target_func}' not found in call graph"}

    results = {}
    for target in targets:
        tree = _build_caller_tree(target, reverse, max_depth, set(), call_graph)
        results[str(target)] = tree

    return {"targets": results, "total_targets": len(targets)}


def _build_caller_tree(
    func: FunctionRef,
    reverse: dict[FunctionRef, list[FunctionRef]],
    depth: int,
    visited: set,
    call_graph: "ProjectCallGraph | None" = None,
) -> dict:
    """Recursively build caller tree.

    Args:
        call_graph: Optional ProjectCallGraph to extract per-edge line numbers.
            If provided, enables surfacing call-site source lines in the result.
    """
    callers = reverse.get(func, [])

    # Base case: truncate at depth 0 or if we've seen this node
    if depth <= 0 or func in visited:
        return {
            "function": func.name,
            "file": func.file,
            "caller_count": len(callers),
            "callers": [],
            "truncated": True,
        }

    visited.add(func)

    tree = {
        "function": func.name,
        "file": func.file,
        "caller_count": len(callers),
        "callers": [],
        "truncated": False,
    }

    # If the call_graph exposes per-edge line numbers (some builders capture
    # the call-site source line; others don't), surface that as a ``line`` key
    # on each caller subtree. Backward-compatible: builders without line info
    # return None from ``lines_for_edge`` and we omit the key.
    lookup_line = getattr(call_graph, "lines_for_edge", None) if call_graph is not None else None

    for caller in callers:
        subtree = _build_caller_tree(caller, reverse, depth - 1, visited.copy(), call_graph)
        if lookup_line is not None:
            line = lookup_line((caller.file, caller.name, func.file, func.name))
            if line is not None:
                subtree["line"] = line
        tree["callers"].append(subtree)

    return tree


def dead_code_analysis(
    call_graph: "ProjectCallGraph",
    all_functions: list[dict],
    entry_points: list[str] | None = None,
) -> dict:
    """Find functions that are never called (excluding entry points).

    Args:
        call_graph: ProjectCallGraph from cross_file_calls
        all_functions: List of {file, name} dicts from structure analysis
        entry_points: Additional entry point patterns to exclude

    Returns:
        Dict with dead_functions, by_file, totals, and percentage
    """
    edges = call_graph.edges
    entry_points = entry_points or []

    # Synthetic orphan-sentinel edges (Ruby/Elixir) carry no semantic call: the
    # `to_func` is a placeholder (e.g. "__ruby_orphan__", "__elixir_orphan__")
    # emitted by the builder to keep orphan defs visible to impact_analysis. If
    # we treated such edges as real, every truly-orphan function (no incoming
    # AND no outgoing edges) would falsely look like a caller (it "calls" the
    # sentinel) and be classified as an entry-point root at the final
    # `if func in callers: continue` check below. Filter them out up front so
    # genuine orphans surface as dead. Imported lazily inside the function to
    # avoid a top-level circular import with cross_file_calls.
    from .cross_file_calls import is_orphan_sentinel as _is_synthetic_orphan_edge

    # Build set of all called functions
    called = set()
    for _, _, to_file, to_func in edges:
        if _is_synthetic_orphan_edge(to_func):
            continue
        called.add(FunctionRef(file=to_file, name=to_func))

    # Build set of all callers (these are "alive" by definition).
    # Exclude synthetic orphan-sentinel edges — their `from_func` is exactly
    # the orphan we want to surface as dead; counting it as a caller would
    # incorrectly classify it as an entry-point root (see policy gap fix).
    callers = set()
    for from_file, from_func, _, to_func in edges:
        if _is_synthetic_orphan_edge(to_func):
            continue
        callers.add(FunctionRef(file=from_file, name=from_func))

    # Common entry point patterns. The Ruby builder used to emit "<top-level>"
    # as its synthetic top-level caller bucket, but it now emits "__main__"
    # instead, so the legacy "<top-level>" pattern can no longer match anything
    # in the graph and was dropped.
    entry_patterns = [
        "main",
        "__main__",
        "cli",
        "app",
        "run",
        "start",
        "test_",
        "pytest_",
        "setup",
        "teardown",
    ] + entry_points

    # Find dead functions
    dead = []
    for func_info in all_functions:
        func = FunctionRef(file=func_info["file"], name=func_info["name"])

        # Skip dunder methods (always — they are framework-managed regardless
        # of edge counts).
        if func.name.startswith("__") and func.name.endswith("__"):
            continue

        # True-orphan override: a function with NO incoming edges (not in
        # `called`) AND NO outgoing real edges (not in `callers`, where
        # synthetic orphan-sentinel edges have already been excluded above)
        # is genuinely unreachable AND unproductive. Surface it as dead
        # *before* the entry-point heuristics — those heuristics are intended
        # to spare framework entry points like `main`, `cli.run`, `setup`,
        # which have outgoing calls. They must NOT also spare a function
        # whose only "evidence of life" is a name/file substring match.
        is_orphan = func not in called and func not in callers
        if is_orphan:
            dead.append(func)
            continue

        # Skip if it's called
        if func in called:
            continue

        # Skip if it's an entry point pattern
        is_entry = any(
            pattern in func.name or pattern in func.file for pattern in entry_patterns
        )
        if is_entry:
            continue

        # Skip if it calls something (it's a root/entry)
        if func in callers:
            continue

        dead.append(func)

    # Group by file
    by_file = defaultdict(list)
    for func in dead:
        by_file[func.file].append(func.name)

    total_funcs = len(all_functions)
    return {
        "dead_functions": [{"file": f.file, "function": f.name} for f in dead],
        "by_file": dict(by_file),
        "total_dead": len(dead),
        "total_functions": total_funcs,
        "dead_percentage": round(len(dead) / max(total_funcs, 1) * 100, 1),
    }


def architecture_analysis(call_graph: "ProjectCallGraph") -> dict:
    """Detect architectural layers from call patterns.

    Heuristics:
    - Functions that call but are not called = entry layer
    - Functions that are called but don't call = leaf layer
    - Analyze directory structure for layer hints
    - Detect circular dependencies

    Args:
        call_graph: ProjectCallGraph from cross_file_calls

    Returns:
        Dict with layer info, directory analysis, and circular deps
    """
    # B-1/S-10: filter synthetic orphan-sentinel edges (Ruby/Elixir) before
    # computing dir_stats / forward / reverse so they don't pollute layer stats.
    # Same filter dead_code_analysis applies for the same reason.
    from .cross_file_calls import is_orphan_sentinel as _is_synthetic_orphan_edge

    edges = [
        e for e in call_graph.edges
        if not _is_synthetic_orphan_edge(e[1]) and not _is_synthetic_orphan_edge(e[3])
    ]
    forward = build_forward_graph(edges)
    reverse = build_reverse_graph(edges)

    # Categorize functions
    entry_layer = []  # Call others but not called
    leaf_layer = []  # Called but don't call others
    middle_layer = []  # Both call and are called

    all_in_graph = set(forward.keys()) | set(reverse.keys())

    for func in all_in_graph:
        calls_others = func in forward and len(forward[func]) > 0
        is_called = func in reverse and len(reverse[func]) > 0

        if calls_others and not is_called:
            entry_layer.append(func)
        elif is_called and not calls_others:
            leaf_layer.append(func)
        elif calls_others and is_called:
            middle_layer.append(func)

    # Analyze directory patterns
    dir_stats = defaultdict(lambda: {"calls_out": 0, "calls_in": 0, "functions": []})

    for func in all_in_graph:
        dir_name = str(Path(func.file).parent) if "/" in func.file else "."
        dir_stats[dir_name]["functions"].append(func.name)

    for from_file, _, to_file, _ in edges:
        from_dir = str(Path(from_file).parent) if "/" in from_file else "."
        to_dir = str(Path(to_file).parent) if "/" in to_file else "."

        if from_dir != to_dir:
            dir_stats[from_dir]["calls_out"] += 1
            dir_stats[to_dir]["calls_in"] += 1

    # Detect circular dependencies
    circular = []
    seen_pairs = set()
    for from_file, _, to_file, _ in edges:
        pair = (from_file, to_file)
        reverse_pair = (to_file, from_file)
        if reverse_pair in seen_pairs and pair not in seen_pairs:
            circular.append({"a": from_file, "b": to_file})
        seen_pairs.add(pair)

    # Infer layers from directory call ratios
    layer_inference = []
    for dir_name, stats in sorted(dir_stats.items()):
        ratio = stats["calls_out"] / max(stats["calls_in"], 1)
        if ratio > 2:
            layer = "HIGH (entry/controller)"
        elif ratio < 0.5:
            layer = "LOW (utility/data)"
        else:
            layer = "MIDDLE (service)"

        layer_inference.append(
            {
                "directory": dir_name,
                "calls_out": stats["calls_out"],
                "calls_in": stats["calls_in"],
                "inferred_layer": layer,
                "function_count": len(stats["functions"]),
            }
        )

    return {
        "entry_layer": [{"file": f.file, "function": f.name} for f in entry_layer[:20]],
        "leaf_layer": [{"file": f.file, "function": f.name} for f in leaf_layer[:20]],
        "middle_layer_count": len(middle_layer),
        "directory_layers": layer_inference,
        "circular_dependencies": circular,
        "summary": {
            "entry_count": len(entry_layer),
            "leaf_count": len(leaf_layer),
            "middle_count": len(middle_layer),
            "circular_count": len(circular),
        },
    }


# Convenience functions that take path instead of CallGraph
def analyze_impact(
    path: str,
    target_func: str,
    max_depth: int = 3,
    target_file: str | None = None,
    language: str = "python",
) -> dict:
    """Convenience wrapper that builds call graph from path.

    Args:
        path: Project path to analyze
        target_func: Function name to find callers of
        max_depth: How deep to traverse callers
        target_file: Optional file filter
        language: Source language

    Returns:
        Impact analysis results
    """
    call_graph = build_project_call_graph(path, language=language)
    return impact_analysis(call_graph, target_func, max_depth, target_file, language=language)


def analyze_dead_code(
    path: str,
    entry_points: list[str] | None = None,
    language: str = "python",
) -> dict:
    """Convenience wrapper that builds call graph from path.

    Args:
        path: Project path to analyze
        entry_points: Additional entry point patterns
        language: Source language

    Returns:
        Dead code analysis results
    """
    from .api import get_code_structure

    call_graph = build_project_call_graph(path, language=language)
    structure = get_code_structure(path, language=language, max_results=1000)

    # Build function list from structure
    all_functions = []
    for file_info in structure.get("files", []):
        file_path = file_info.get("path", "")
        for func_name in file_info.get("functions", []):
            all_functions.append({"file": file_path, "name": func_name})

    return dead_code_analysis(call_graph, all_functions, entry_points)


def analyze_architecture(path: str, language: str = "python") -> dict:
    """Convenience wrapper that builds call graph from path.

    Args:
        path: Project path to analyze
        language: Source language

    Returns:
        Architecture analysis results
    """
    call_graph = build_project_call_graph(path, language=language)
    return architecture_analysis(call_graph)
