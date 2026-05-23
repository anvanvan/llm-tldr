"""
TLDR Unified API - Token-efficient code context for LLMs.

Usage:
    from tldr.api import get_relevant_context

    context = get_relevant_context(
        project="/path/to/project",
        entry_point="ClassName.method_name",  # or "function_name"
        depth=2,
        language="python"
    )

    # Returns LLM-ready string with call graph, signatures, complexity
"""

import json as _json
import logging as _logging
import os as _os
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional

from .ast_extractor import (
    CallGraphInfo,  # Re-exported for API consumers
    ClassInfo,  # Re-exported for API consumers
    FunctionInfo,
    ImportInfo,  # Re-exported for API consumers
    extract_file as _extract_file_impl,
)


_logger = _logging.getLogger(__name__)


# Languages supported by get_relevant_context (ext_map used for file scanning)
SUPPORTED_CONTEXT_EXT_MAP: dict[str, set[str]] = {
    "python": {".py"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "go": {".go"},
    "rust": {".rs"},
    "php": {".php"},
    "swift": {".swift"},
    "java": {".java"},
}
SUPPORTED_CONTEXT_LANGUAGES: frozenset[str] = frozenset(SUPPORTED_CONTEXT_EXT_MAP.keys())

# Authoritative extension map for all supported languages (used by get_module,
# get_relevant_context, and scan_project_files to avoid duplication).
_EXT_MAP_ALL_LANGUAGES: dict[str, set[str]] = {
    "python": {".py"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "go": {".go"},
    "rust": {".rs"},
    "php": {".php"},
    "java": {".java"},
    "c": {".c", ".h"},
    "elixir": {".ex", ".exs"},
    "swift": {".swift"},
    "ruby": {".rb"},
    "kotlin": {".kt", ".kts"},
    "csharp": {".cs"},
    "lua": {".lua"},
    "luau": {".luau"},
    "scala": {".scala", ".sc"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"},
}

# Bug 004: non-code suffixes that the semantic indexer must include so that
# build/config/doc files (.sh scripts, pyproject.toml, .yaml workflows, etc.)
# appear in semantic search results. Kept in module scope so other call sites
# (e.g. semantic._process_file_for_extraction) can share the same set.
NON_CODE_EXTENSIONS: set[str] = {
    ".sh", ".zsh", ".bash",
    ".toml", ".yaml", ".yml", ".json",
    ".md", ".rst", ".txt",
}

from .cfg_extractor import (
    CFGBlock,  # Re-exported for type hints
    CFGEdge,  # Re-exported for type hints
    CFGInfo,
    extract_c_cfg,
    extract_cpp_cfg,
    extract_csharp_cfg,
    extract_elixir_cfg,
    extract_go_cfg,
    extract_java_cfg,
    extract_kotlin_cfg,
    extract_lua_cfg,
    extract_luau_cfg,
    extract_php_cfg,
    extract_python_cfg,
    extract_ruby_cfg,
    extract_rust_cfg,
    extract_scala_cfg,
    extract_swift_cfg,
    extract_typescript_cfg,
)
from .dedup import ContentHashedIndex  # P5 #21: Content-hash deduplication
from .cross_file_calls import (
    build_project_call_graph,
)
from .cross_file_calls import (
    build_function_index as _build_function_index,
)
from .cross_file_calls import (
    parse_go_imports as _parse_go_imports,
)
from .cross_file_calls import (
    parse_imports as _parse_imports,
)
from .cross_file_calls import (
    parse_rust_imports as _parse_rust_imports,
)
from .cross_file_calls import (
    parse_ts_imports as _parse_ts_imports,
)
from .cross_file_calls import (
    parse_java_imports as _parse_java_imports,
)
from .cross_file_calls import (
    parse_c_imports as _parse_c_imports,
)
from .cross_file_calls import (
    parse_cpp_imports as _parse_cpp_imports,
)
from .cross_file_calls import (
    parse_ruby_imports as _parse_ruby_imports,
)
from .cross_file_calls import (
    parse_kotlin_imports as _parse_kotlin_imports,
)
from .cross_file_calls import (
    parse_scala_imports as _parse_scala_imports,
)
from .cross_file_calls import (
    parse_php_imports as _parse_php_imports,
)
from .cross_file_calls import (
    parse_swift_imports as _parse_swift_imports,
)
from .cross_file_calls import (
    parse_csharp_imports as _parse_csharp_imports,
)
from .cross_file_calls import (
    parse_lua_imports as _parse_lua_imports,
)
from .cross_file_calls import (
    parse_luau_imports as _parse_luau_imports,
)
from .cross_file_calls import (
    parse_elixir_imports as _parse_elixir_imports,
)
from .cross_file_calls import (
    scan_project as _scan_project,
)
from .dfg_extractor import (
    DFGInfo,
    extract_c_dfg,
    extract_cpp_dfg,
    extract_csharp_dfg,
    extract_elixir_dfg,
    extract_go_dfg,
    extract_java_dfg,
    extract_kotlin_dfg,
    extract_lua_dfg,
    extract_luau_dfg,
    extract_php_dfg,
    extract_python_dfg,
    extract_ruby_dfg,
    extract_rust_dfg,
    extract_scala_dfg,
    extract_swift_dfg,
    extract_typescript_dfg,
)
from .hybrid_extractor import (
    HybridExtractor,
    extract_directory,  # Re-exported for API
)
from .pdg_extractor import (
    PDGInfo,
    extract_c_pdg,
    extract_cpp_pdg,
    extract_csharp_pdg,
    extract_go_pdg,
    extract_pdg,
    extract_python_pdg,
    extract_ruby_pdg,
    extract_rust_pdg,
    extract_typescript_pdg,
)

# Explicit exports for public API (single authoritative list)
__all__ = [
    # Dataclasses from ast_extractor (re-exported for API consumers)
    "CallGraphInfo",
    "ClassInfo",
    "FunctionInfo",
    "ImportInfo",
    # Main API functions
    "get_relevant_context",
    "get_relevant_context_multi",
    "query",
    "FunctionContext",
    "RelevantContext",
    "get_imports",
    "get_intra_file_calls",
    "extract_file",
    "extract_file_with_code",
    # Layer 3: CFG types and functions
    "CFGBlock",
    "CFGEdge",
    "get_cfg_context",
    "get_cfg_blocks",
    "get_cfg_edges",
    # Layer 4: DFG functions
    "get_dfg_context",
    # Layer 5: PDG functions
    "get_pdg_context",
    "get_slice",
    # Cross-file functions
    "build_project_call_graph",
    "scan_project_files",
    "build_function_index",
    # Project navigation functions
    "get_file_tree",
    "search",
    "Selection",
    "get_code_structure",
    # Content-hash deduplication
    "ContentHashedIndex",
    # Language support constants
    "SUPPORTED_CONTEXT_LANGUAGES",
    "SUPPORTED_CONTEXT_EXT_MAP",
    # Security exceptions
    "PathTraversalError",
]


def _serialize_call_graph_to_cache(
    cache_file: Path, call_graph, languages: list, timestamp: float | None = None
) -> None:
    """Serialize a ProjectCallGraph to JSON cache format.

    Args:
        cache_file: Path to write cache to.
        call_graph: ProjectCallGraph instance with an ``edges`` iterable.
        languages: List of language strings to store in the cache.
        timestamp: Unix timestamp (defaults to current time).
    """
    cache_data = {
        "edges": [
            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
            for e in call_graph.edges
        ],
        "languages": languages,
        "timestamp": timestamp if timestamp is not None else _time.time(),
    }
    cache_file.write_text(_json.dumps(cache_data, indent=2))


# =============================================================================
# Security: Path Containment Validation
# =============================================================================


class PathTraversalError(ValueError):
    """Raised when a path attempts to escape its container via directory traversal.

    This is a security error indicating an attempted path traversal attack
    (e.g., using ../../../etc/passwd to escape the project directory).
    """
    pass


def _validate_path_containment(file_path: str, base_path: str | None = None) -> Path:
    """Validate that file_path doesn't escape base_path via traversal.

    Detects directory traversal attacks (../..) and symlink escapes.

    Args:
        file_path: The path to validate
        base_path: Optional container directory. If None, detects traversal
                   patterns that escape the apparent starting directory.

    Returns:
        Resolved Path object

    Raises:
        PathTraversalError: If path contains traversal or escapes base
        ValueError: If path is empty or whitespace-only
    """
    # Reject empty or whitespace-only paths
    if not file_path or not file_path.strip():
        raise ValueError("Path cannot be empty or whitespace-only")

    # Check for null bytes (path truncation attack)
    if "\x00" in file_path:
        raise ValueError("Path contains null byte")

    # Resolve the path (follows symlinks, normalizes ..)
    try:
        resolved = Path(file_path).resolve()
    except OSError as e:
        # Handle paths that are too long or have invalid characters
        raise ValueError(f"Invalid path: {e}")

    # Check for traversal patterns in original path
    if ".." in file_path:
        if base_path:
            # Explicit base path provided - enforce containment
            base = Path(base_path).resolve()
            try:
                if not resolved.is_relative_to(base):
                    raise PathTraversalError(
                        f"Path '{file_path}' escapes base directory '{base_path}' via traversal"
                    )
            except ValueError:
                raise PathTraversalError(
                    f"Path '{file_path}' escapes base directory '{base_path}'"
                )
        else:
            # No explicit base path - detect suspicious traversal patterns
            # A path like "/tmp/project/../outside/file.py" is suspicious because
            # the ".." effectively escapes from "project" into a sibling directory
            #
            # Strategy: Find directory components that are "entered" then immediately
            # "exited" via .. - this indicates intentional escape
            path_obj = Path(file_path)
            parts = list(path_obj.parts)

            # Look for pattern: <dir>/.. which indicates entering then leaving a directory
            # This is almost always indicative of traversal attack
            i = 0
            while i < len(parts) - 1:
                current = parts[i]
                next_part = parts[i + 1]

                # Skip root components like "/" or "C:\"
                if current in ("/", "\\") or (len(current) == 2 and current[1] == ":"):
                    i += 1
                    continue

                # If we have a real directory name followed by "..", that's traversal
                if current not in (".", "..") and next_part == "..":
                    raise PathTraversalError(
                        f"Path '{file_path}' contains directory traversal pattern '{current}/..'"
                    )
                i += 1

    # Check symlink targets if the path exists
    # Wrap filesystem operations in try/except to handle mocked/broken stat
    try:
        path_exists = resolved.exists()
    except (OSError, TypeError):
        # stat might be mocked or broken - skip symlink checks
        path_exists = False

    if path_exists:
        # Check if the resolved path is a symlink (readlink on the original)
        original_path = Path(file_path)
        try:
            is_symlink = original_path.is_symlink()
        except (OSError, TypeError):
            # lstat might be mocked or broken
            is_symlink = False

        if is_symlink:
            try:
                target = original_path.readlink()
                # Resolve the target relative to the symlink's parent
                abs_target = (original_path.parent / target).resolve()

                if base_path:
                    base = Path(base_path).resolve()
                    if not abs_target.is_relative_to(base):
                        raise PathTraversalError(
                            f"Symlink '{file_path}' points outside base directory '{base_path}'"
                        )
                else:
                    # No base path - check if symlink target escapes the symlink's directory
                    symlink_parent = original_path.parent.resolve()
                    if not abs_target.is_relative_to(symlink_parent):
                        raise PathTraversalError(
                            f"Symlink '{file_path}' points outside its containing directory"
                        )
            except OSError:
                # Can't read symlink - might be broken, let normal file ops handle it
                pass

    return resolved


def _resolve_source(source_or_path: str) -> tuple[str, str | None]:
    """
    Resolve source code from either source string or file path.

    Auto-detects whether the input is:
    1. A file path (exists on disk) -> reads and returns contents
    2. Source code string -> returns as-is

    Args:
        source_or_path: Either source code string or path to a file

    Returns:
        Tuple of (source_code, file_path_or_none)
        - If path: (file_contents, path)
        - If source: (source, None)

    Raises:
        PathTraversalError: If path contains directory traversal patterns
        ValueError: If path is empty or contains invalid characters
    """
    # Check if it looks like a file path and exists
    if len(source_or_path) < 500:  # Paths are typically short
        try:
            # Security: Validate path before accessing
            # PathTraversalError must propagate - don't catch it
            _validate_path_containment(source_or_path)

            path = Path(source_or_path)
            if path.exists() and path.is_file():
                return path.read_text(encoding="utf-8"), str(path)
        except PathTraversalError:
            # Security error - must propagate
            raise
        except (OSError, ValueError):
            # Long strings can cause OSError on path.exists()
            # ValueError from empty/whitespace paths - treat as source code
            pass

    # Treat as source code
    return source_or_path, None


@dataclass
class FunctionContext:
    """Context for a single function."""
    name: str
    file: str
    line: int
    signature: str
    docstring: str | None = None
    calls: list[str] = field(default_factory=list)
    blocks: int | None = None  # CFG blocks count
    cyclomatic: int | None = None  # Cyclomatic complexity


@dataclass
class RelevantContext:
    """The full context returned by get_relevant_context."""
    entry_point: str
    depth: int
    functions: list[FunctionContext] = field(default_factory=list)
    error: str | None = None

    def to_llm_string(self) -> str:
        """Format for LLM injection."""
        if self.error:
            return f"Error: {self.error}"

        lines = [
            f"## Code Context: {self.entry_point} (depth={self.depth})",
            ""
        ]

        for i, func in enumerate(self.functions):
            # Indentation based on call depth
            indent = "  " * min(i, self.depth)

            # Function header
            short_file = Path(func.file).name if func.file else "?"
            lines.append(f"{indent}📍 {func.name} ({short_file}:{func.line})")
            lines.append(f"{indent}   {func.signature}")

            # Docstring (truncated)
            if func.docstring:
                doc = func.docstring.split('\n')[0][:80]
                lines.append(f"{indent}   # {doc}")

            # Complexity
            if func.blocks is not None:
                complexity_marker = "🔥" if func.cyclomatic and func.cyclomatic > 10 else ""
                lines.append(f"{indent}   ⚡ complexity: {func.cyclomatic or '?'} ({func.blocks} blocks) {complexity_marker}")

            # Calls
            if func.calls:
                calls_str = ", ".join(func.calls[:5])
                if len(func.calls) > 5:
                    calls_str += f" (+{len(func.calls)-5} more)"
                lines.append(f"{indent}   → calls: {calls_str}")

            lines.append("")

        # Footer with stats
        result = "\n".join(lines)
        token_estimate = len(result) // 4
        return result + f"\n---\n📊 {len(self.functions)} functions | ~{token_estimate} tokens"


def _get_module_exports(
    project: Path,
    module_path: str,
    language: str = "python",
    include_docstrings: bool = True
) -> "RelevantContext":
    """Get all exports from a module path.

    Args:
        project: Project root path
        module_path: Module path like "providers/anthropic" or "multimodal/video/processor"
        language: Language for extension mapping
        include_docstrings: Whether to include docstrings

    Returns:
        RelevantContext with all functions/classes from the module
    """
    extensions = _EXT_MAP_ALL_LANGUAGES.get(language, {".py"})

    # Try to find the module file
    # module_path "providers/anthropic" -> providers/anthropic.py
    module_file = None
    for ext in extensions:
        candidate = project / f"{module_path}{ext}"
        if candidate.exists():
            module_file = candidate
            break

    if module_file is None:
        # Try as directory with __init__.py (Python package)
        init_file = project / module_path / "__init__.py"
        if init_file.exists():
            module_file = init_file
        else:
            tried = ", ".join(str(project / f"{module_path}{e}") for e in extensions)
            raise ValueError(f"Module not found: {module_path} (tried {tried} and {init_file})")

    # Extract all functions and classes from the module
    extractor = HybridExtractor()
    try:
        module_info = extractor.extract(str(module_file))
    except Exception as e:
        raise ValueError(f"Failed to parse module {module_path}: {e}")

    functions: list[FunctionContext] = []

    # Add all functions
    for func in module_info.functions:
        ctx = FunctionContext(
            name=func.name,
            signature=f"def {func.name}({', '.join(func.params)}) -> {func.return_type or 'None'}",
            file=str(module_file),
            line=func.line_number,
            docstring=func.docstring if include_docstrings else None,
            calls=[],
        )
        functions.append(ctx)

    # Add all classes (as constructors/callables)
    for cls in module_info.classes:
        ctx = FunctionContext(
            name=cls.name,
            signature=f"class {cls.name}",
            file=str(module_file),
            line=cls.line_number,
            docstring=cls.docstring if include_docstrings else None,
            calls=[m.name for m in cls.methods],
        )
        functions.append(ctx)

        # Also add class methods
        for method in cls.methods:
            method_ctx = FunctionContext(
                name=f"{cls.name}.{method.name}",
                signature=f"def {method.name}({', '.join(method.params)}) -> {method.return_type or 'None'}",
                file=str(module_file),
                line=method.line_number,
                docstring=method.docstring if include_docstrings else None,
                calls=[],
            )
            functions.append(method_ctx)

    return RelevantContext(
        entry_point=module_path,
        depth=0,
        functions=functions
    )


def get_relevant_context(
    project: str | Path,
    entry_point: str,
    depth: int = 2,
    language: str = "python",
    include_docstrings: bool = True
) -> RelevantContext:
    """
    Get token-efficient context for an LLM starting from an entry point.

    Args:
        project: Path to project root
        entry_point: Function/method name (e.g., "Client.stream") or module path (e.g., "providers/anthropic")
        depth: How deep to traverse the call graph
        language: python, typescript, javascript, go, rust, php, java, swift, c, cpp, ruby, kotlin, elixir, csharp, lua, luau, or scala
        include_docstrings: Whether to include function docstrings

    Returns:
        RelevantContext with functions reachable from entry_point
    """
    project = Path(project)

    ext_map = _EXT_MAP_ALL_LANGUAGES

    # Single-pass tree walk: detect available languages AND collect all
    # non-hidden source files grouped by suffix.  A second rglob for the
    # signature-indexing loop below is then unnecessary (O(n) → O(n)).
    available_langs: set = set()
    files_by_ext: dict[str, list[Path]] = {}
    for fp in project.rglob("*"):
        try:
            rel = fp.relative_to(project)
            if any(p.startswith(".") for p in rel.parts):
                continue
        except ValueError:
            pass  # file outside project root — keep it
        ext = fp.suffix
        for lang, exts in ext_map.items():
            if ext in exts:
                available_langs.add(lang)
                break
        if ext:
            files_by_ext.setdefault(ext, []).append(fp)

    # R-7: track whether `language` was auto-detected (i.e. substituted because
    # the caller's value wasn't present in the project). Cache writes below are
    # gated on this flag so that an auto-detect that picked the wrong language
    # in a polyglot repo doesn't poison the on-disk call_graph.json for a later
    # explicit-lang caller.
    language_auto_detected = False
    if language not in available_langs:
        for candidate in (
            "python", "typescript", "javascript", "go", "rust", "java",
            "php", "swift", "c", "cpp", "csharp", "kotlin", "scala",
            "ruby", "elixir", "lua", "luau",
        ):
            if candidate == language:
                continue
            if candidate in available_langs:
                language = candidate
                language_auto_detected = True
                break

    # Module query mode: path with / and no . (e.g., "providers/anthropic")
    if "/" in entry_point and "." not in entry_point:
        return _get_module_exports(project, entry_point, language, include_docstrings)

    # NOTE: Removed module-file shortcut that conflicted with function lookup.
    # If entry_point="main" matched "main.ts", it would return module exports
    # instead of doing BFS call graph traversal. Use explicit path syntax
    # (e.g., "main/" or with extension) for module exports.

    # Build cross-file call graph
    call_graph = build_project_call_graph(str(project), language=language)

    # Persist call graph to .tldr/cache/ so subsequent invocations (and CLI
    # commands that look for cached graphs) see the same data the API just
    # computed. This mirrors the cache-write behaviour of cli._get_or_build_graph.
    # Skip the write when the cached file is recent (< 1 hour) to avoid
    # repeated serialization on rapid-fire queries in the same session.
    # R-7: skip the write entirely when the language was auto-detected — the
    # auto-detect heuristic can pick the wrong language in polyglot repos, and
    # we don't want to overwrite a cache built from an explicit-lang invocation.
    if not language_auto_detected:
        try:
            cache_dir = project / ".tldr" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "call_graph.json"
            _write_cache = True
            if cache_file.exists():
                cache_age = _time.time() - _os.path.getmtime(cache_file)
                if cache_age < 21600:  # skip write if cache is < 6 hours old
                    # C-5: the 6-hour skip is per-language. A prior invocation
                    # with a different --lang must not block the current
                    # language from refreshing the cache.
                    try:
                        cached_payload = _json.loads(cache_file.read_text())
                        cached_languages = cached_payload.get("languages") or []
                    except (OSError, _json.JSONDecodeError, ValueError):
                        cached_languages = []
                    if language in cached_languages:
                        _write_cache = False
            if _write_cache:
                _serialize_call_graph_to_cache(cache_file, call_graph, [language])
        except (OSError, _json.JSONDecodeError) as exc:
            # Best-effort cache write; never block context resolution, but
            # surface the failure so users can diagnose disk/permission issues.
            _logger.warning(
                "call_graph.json cache write failed for %s (%s: %s)",
                project, type(exc).__name__, exc,
            )

    # Index all signatures
    extractor = HybridExtractor()
    signatures: dict[str, tuple[str, FunctionInfo]] = {}  # func_name -> (file, info)

    extensions = ext_map.get(language, {".py"})

    # Also cache file sources for CFG extraction
    file_sources: dict[str, str] = {}

    # Reuse the pre-collected file list from the single-pass walk above instead
    # of doing a second project.rglob("*") traversal.
    candidate_paths: list[Path] = []
    for ext in extensions:
        candidate_paths.extend(files_by_ext.get(ext, []))

    for file_path in candidate_paths:
        try:
            source = file_path.read_text()
            file_sources[str(file_path)] = source

            info = extractor.extract(str(file_path))
            for func in info.functions:
                # Primary key: module.function (e.g., "claude_spawn.spawn_agent")
                module_name = file_path.stem  # "claude_spawn" from "claude_spawn.py"
                qualified_key = f"{module_name}.{func.name}"
                signatures[qualified_key] = (str(file_path), func)

                # Also store unqualified for backward compat (first wins)
                if func.name not in signatures:
                    signatures[func.name] = (str(file_path), func)
            for cls in info.classes:
                # Index class itself as callable (dataclasses, constructors)
                # Create a pseudo-FunctionInfo for the class
                class_as_func = FunctionInfo(
                    name=cls.name,
                    params=[],  # Could extract __init__ params if needed
                    return_type=cls.name,
                    docstring=cls.docstring,
                    line_number=cls.line_number,
                    language=language,
                    is_class_wrapper=(language == "swift"),
                )
                signatures[cls.name] = (str(file_path), class_as_func)

                for method in cls.methods:
                    # Store as ClassName.method
                    key = f"{cls.name}.{method.name}"
                    signatures[key] = (str(file_path), method)
                    # Also store ClassName::method alias (PHP convention)
                    if language == "php":
                        cc_key = f"{cls.name}::{method.name}"
                        signatures[cc_key] = (str(file_path), method)
                    # Also store just method name (for call graph join)
                    # Only if not already taken by a standalone function
                    if method.name not in signatures:
                        signatures[method.name] = (str(file_path), method)
        except Exception:
            pass  # Skip files that fail to parse

    # CFG extractor based on language
    cfg_extractors = {
        "python": extract_python_cfg,
        "typescript": extract_typescript_cfg,
        "javascript": extract_typescript_cfg,
        "go": extract_go_cfg,
        "rust": extract_rust_cfg,
        "java": extract_java_cfg,
        "c": extract_c_cfg,
        "php": extract_php_cfg,
        "kotlin": extract_kotlin_cfg,
        "swift": extract_swift_cfg,
        "csharp": extract_csharp_cfg,
        "scala": extract_scala_cfg,
        "lua": extract_lua_cfg,
        "luau": extract_luau_cfg,
        "elixir": extract_elixir_cfg,
        "cpp": extract_cpp_cfg,
    }
    cfg_extractor_fn = cfg_extractors.get(language, extract_python_cfg)

    # Build forward adjacency list from call graph edges.
    # Edge format: (caller_file, caller_func, callee_file, callee_func)
    # The reverse adjacency is deferred until after BFS so that it is built
    # only for the targets we actually query (memory savings for large graphs).
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in call_graph.edges:
        caller_file, caller_func, callee_file, callee_func = edge
        adjacency[caller_func].append(callee_func)

    # BFS from entry point up to depth
    visited = set()
    queue = [(entry_point, 0)]
    result_functions = []

    # Helper to resolve function name to signature (handles qualified/unqualified)
    def resolve_func_name(name: str) -> list[tuple[str, tuple[str, FunctionInfo]]]:
        """Resolve function name, returning all matches for ambiguous names."""
        # Normalize PHP :: to . for qualified lookup (adv-2)
        if "::" in name:
            name = name.replace("::", ".")
        # If qualified (has dot), do direct lookup
        if "." in name:
            if name in signatures:
                return [(name, signatures[name])]
            return []

        # Unqualified name - find all qualified matches
        matches = [(k, v) for k, v in signatures.items()
                   if k.endswith(f".{name}")]

        if matches:
            # Return all matches (could be 1 or more)
            return matches
        elif name in signatures:
            # Fall back to direct unqualified lookup
            return [(name, signatures[name])]

        return []

    while queue:
        func_name, current_depth = queue.pop(0)

        if func_name in visited or current_depth > depth:
            continue
        visited.add(func_name)

        # Get signature info if available (may return multiple matches)
        resolved_list = resolve_func_name(func_name)
        if resolved_list:
            for resolved_name, (file_path, func_info) in resolved_list:
                # Skip if we already processed this qualified name
                if resolved_name in visited and resolved_name != func_name:
                    continue
                visited.add(resolved_name)

                # Try to get CFG complexity
                blocks = None
                cyclomatic = None
                # Use the actual function name from func_info for CFG lookup
                cfg_func_name = func_info.name
                if file_path in file_sources:
                    try:
                        cfg = cfg_extractor_fn(file_sources[file_path], cfg_func_name)
                        if cfg and cfg.blocks:
                            blocks = len(cfg.blocks)
                            cyclomatic = cfg.cyclomatic_complexity
                    except Exception:
                        pass  # CFG extraction failed, skip

                ctx = FunctionContext(
                    name=resolved_name,  # Use qualified name for clarity
                    file=file_path,
                    line=func_info.line_number,
                    signature=func_info.signature(),
                    docstring=func_info.docstring if include_docstrings else None,
                    calls=adjacency.get(func_info.name, []),  # Use unqualified for adjacency lookup
                    blocks=blocks,
                    cyclomatic=cyclomatic
                )
                result_functions.append(ctx)

                # Queue callees from this function
                for callee in adjacency.get(func_info.name, []):
                    if callee not in visited and current_depth < depth:
                        queue.append((callee, current_depth + 1))
        else:
            # Entry point not found — return error instead of phantom stub
            if func_name == entry_point and current_depth == 0:
                return RelevantContext(
                    entry_point=entry_point,
                    depth=depth,
                    error=f"Function '{entry_point}' not found in project"
                )

            # Callee not found in signatures during BFS, still include stub
            ctx = FunctionContext(
                name=func_name,
                file="?",
                line=0,
                signature=f"def {func_name}(...)",
                calls=adjacency.get(func_name, [])
            )
            result_functions.append(ctx)

            # Queue callees
            for callee in adjacency.get(func_name, []):
                if callee not in visited and current_depth < depth:
                    queue.append((callee, current_depth + 1))

    # Cross-file caller resolution: surface direct callers of the entry point
    # (and of any matching qualified variant). The forward BFS above only
    # walks callees; without this pass, an entry point that is *called from*
    # other files would appear as an isolated node.
    if depth >= 1:
        # set[(file_path_str, func_name_str)] — element-type contract is a
        # 2-tuple of strings; never add a bare str. Normalises any Path that
        # may leak from upstream call-graph edges via str() at the add site.
        caller_names_seen: set[tuple[str, str]] = {
            (str(ctx.file), ctx.name) for ctx in result_functions
        }
        caller_targets: set[str] = {entry_point}
        # Also include any qualified variants of entry_point we already resolved
        for ctx in list(result_functions):
            tail = ctx.name.rsplit(".", 1)[-1]
            caller_targets.add(tail)
            caller_targets.add(ctx.name)

        # Build reverse adjacency on-demand: only index edges whose callee is
        # one of the targets we will query, reducing memory for large graphs.
        reverse_adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for edge in call_graph.edges:
            _, caller_func, _, callee_func = edge
            if callee_func in caller_targets:
                reverse_adjacency[callee_func].append((edge[0], caller_func))

        for target in list(caller_targets):
            for caller_file, caller_func in reverse_adjacency.get(target, []):
                # Skip self-edges (caller == target)
                if caller_func == target:
                    continue
                # Try to resolve a richer signature/line number from the index
                caller_resolved = resolve_func_name(caller_func)
                if caller_resolved:
                    for resolved_name, (file_path, func_info) in caller_resolved:
                        file_path_str = str(file_path)
                        key = (file_path_str, resolved_name)
                        if key in caller_names_seen:
                            continue
                        caller_names_seen.add(key)
                        result_functions.append(FunctionContext(
                            name=resolved_name,
                            file=file_path_str,
                            line=func_info.line_number,
                            signature=func_info.signature(),
                            docstring=func_info.docstring if include_docstrings else None,
                            calls=adjacency.get(func_info.name, []),
                        ))
                else:
                    # Synthesise a minimal entry from the edge metadata when
                    # the caller wasn't found in the signature index.
                    abs_caller_file = (
                        str(project / caller_file)
                        if not Path(caller_file).is_absolute()
                        else str(caller_file)
                    )
                    key = (abs_caller_file, caller_func)
                    if key in caller_names_seen:
                        continue
                    caller_names_seen.add(key)
                    result_functions.append(FunctionContext(
                        name=caller_func,
                        file=abs_caller_file,
                        line=0,
                        signature=f"def {caller_func}(...)",
                        calls=[target],
                    ))

    return RelevantContext(
        entry_point=entry_point,
        depth=depth,
        functions=result_functions
    )


def get_relevant_context_multi(
    project: str | Path,
    entry_point: str,
    depth: int = 2,
    languages: list[str] | tuple[str, ...] = ("python",),
    include_docstrings: bool = True,
) -> RelevantContext:
    """Probe each language in order; return first non-error hit.

    First-hit-wins — in polyglot projects where multiple languages define
    the same name (e.g., both Python and TypeScript have ``process_data``),
    returns the first language that resolves. Probe order is caller-controlled
    via the ``languages`` parameter.

    Note: each per-language probe delegates to :func:`get_relevant_context`,
    which performs its own internal auto-detect when the requested language
    has no files in the project. As a result, a force-probed language can be
    silently overridden by the auto-detected one (e.g., Swift probe in a
    Python-only project will resolve via Python). Callers needing strict
    per-language semantics should pre-filter ``languages`` to those actually
    present in the project.

    On all-miss, returns a :class:`RelevantContext` whose ``error`` lists the
    languages probed (in input order). For an empty ``languages`` argument,
    returns immediately with a clean "no supported languages probed" error
    (no malformed ``(probed: )`` output).

    Args:
        project: Path to project root.
        entry_point: Function/method name (e.g., ``"ClassName.method"`` or
            ``"function_name"``).
        depth: How deep to traverse the call graph.
        languages: Ordered iterable of languages to probe. First non-error
            hit wins. Defaults to ``("python",)``.
        include_docstrings: Whether to include function docstrings in results.

    Returns:
        :class:`RelevantContext` from the first language that resolves, or
        an error-only ``RelevantContext`` if every language misses.
    """
    if not languages:
        return RelevantContext(
            entry_point=entry_point,
            depth=depth,
            error=(
                f"Function '{entry_point}' not found in project "
                f"(no supported languages probed)"
            ),
        )

    for lang in languages:
        ctx = get_relevant_context(
            project,
            entry_point,
            depth=depth,
            language=lang,
            include_docstrings=include_docstrings,
        )
        if not ctx.error:
            return ctx  # first hit wins

    probed = ", ".join(languages)
    return RelevantContext(
        entry_point=entry_point,
        depth=depth,
        error=(
            f"Function '{entry_point}' not found in project "
            f"(probed: {probed})"
        ),
    )


def get_dfg_context(
    source_or_path: str,
    function_name: str,
    language: str = "python"
) -> dict:
    """
    Get data flow analysis for a function.

    Extracts variable references (definitions, updates, uses) and
    def-use chains (dataflow edges) for the specified function.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        Dict with:
          - function: function name
          - refs: list of variable references (name, type, line, column)
          - edges: list of def-use edges (var, def_line, use_line, def, use)
          - variables: list of variable names found
    """
    source_code, _ = _resolve_source(source_or_path)

    # Select extractor based on language
    dfg_extractors = {
        "python": extract_python_dfg,
        "typescript": extract_typescript_dfg,
        "javascript": extract_typescript_dfg,  # JS uses TS extractor
        "go": extract_go_dfg,
        "rust": extract_rust_dfg,
        "java": extract_java_dfg,
        "c": extract_c_dfg,
        "cpp": extract_cpp_dfg,
        "ruby": extract_ruby_dfg,
        "php": extract_php_dfg,
        "kotlin": extract_kotlin_dfg,
        "swift": extract_swift_dfg,
        "csharp": extract_csharp_dfg,
        "scala": extract_scala_dfg,
        "lua": extract_lua_dfg,
        "luau": extract_luau_dfg,
        "elixir": extract_elixir_dfg,
    }

    # Default to Python for unknown languages
    extractor_fn = dfg_extractors.get(language, extract_python_dfg)

    try:
        dfg_info: DFGInfo = extractor_fn(source_code, function_name)
        return dfg_info.to_dict()
    except Exception:
        # Return empty DFG on extraction failure
        return {
            "function": function_name,
            "refs": [],
            "edges": [],
            "variables": []
        }


# =============================================================================
# CFG API Functions (Layer 3)
# =============================================================================


def get_cfg_context(
    source_or_path: str,
    function_name: str,
    language: str = "python"
) -> dict:
    """
    Get control flow graph context for a function.

    Extracts basic blocks, control flow edges, and complexity metrics
    for the specified function.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        Dict with:
          - function: function name
          - blocks: list of basic block dicts (id, type, lines, calls)
          - edges: list of edge dicts (from, to, type, condition)
          - entry_block: entry block ID
          - exit_blocks: list of exit block IDs
          - cyclomatic_complexity: cyclomatic complexity metric
          - nested_functions: dict of nested function CFGs (if any)
    """
    source_code, _ = _resolve_source(source_or_path)

    cfg_extractors = {
        "python": extract_python_cfg,
        "typescript": extract_typescript_cfg,
        "javascript": extract_typescript_cfg,
        "go": extract_go_cfg,
        "rust": extract_rust_cfg,
        "java": extract_java_cfg,
        "c": extract_c_cfg,
        "cpp": extract_cpp_cfg,
        "ruby": extract_ruby_cfg,
        "php": extract_php_cfg,
        "swift": extract_swift_cfg,
        "csharp": extract_csharp_cfg,
        "lua": extract_lua_cfg,
        "luau": extract_luau_cfg,
        "elixir": extract_elixir_cfg,
    }

    extractor_fn = cfg_extractors.get(language, extract_python_cfg)

    try:
        cfg_info: CFGInfo = extractor_fn(source_code, function_name)
        if cfg_info is None:
            return {
                "function": function_name,
                "blocks": [],
                "edges": [],
                "entry_block": 0,
                "exit_blocks": [],
                "cyclomatic_complexity": 0,
            }
        return cfg_info.to_dict()
    except Exception:
        # Return empty CFG on extraction failure
        return {
            "function": function_name,
            "blocks": [],
            "edges": [],
            "entry_block": 0,
            "exit_blocks": [],
            "cyclomatic_complexity": 0,
        }


def get_cfg_blocks(
    source_or_path: str,
    function_name: str,
    language: str = "python"
) -> list[dict]:
    """
    Get CFG basic blocks for a function.

    Basic blocks are sequences of statements with no internal branches.
    Control enters only at the first statement and leaves only at the last.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        List of block dicts, each containing:
          - id: block identifier
          - type: block type (entry, branch, loop_header, return, exit, body)
          - lines: [start_line, end_line]
          - calls: list of function calls in this block (if any)

        Returns empty list if function not found.
    """
    cfg = get_cfg_context(source_or_path, function_name, language)
    return cfg.get("blocks", [])


def get_cfg_edges(
    source_or_path: str,
    function_name: str,
    language: str = "python"
) -> list[dict]:
    """
    Get CFG control flow edges for a function.

    Edges represent possible control flow transitions between basic blocks.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of function to analyze
        language: python, typescript, go, or rust (defaults to python)

    Returns:
        List of edge dicts, each containing:
          - from: source block ID
          - to: target block ID
          - type: edge type (true, false, unconditional, back_edge, break, continue)
          - condition: human-readable condition (for conditional edges)

        Returns empty list if function not found.
    """
    cfg = get_cfg_context(source_or_path, function_name, language)
    return cfg.get("edges", [])


def query(
    project: str | Path,
    query: str,
    depth: int = 2,
    language: str = "python"
) -> str:
    """
    Convenience function that returns LLM-ready string directly.

    Args:
        project: Path to project root
        query: Function or method name to start from
        depth: Call graph traversal depth
        language: Programming language

    Returns:
        Formatted string ready for LLM context injection
    """
    ctx = get_relevant_context(project, query, depth, language)
    return ctx.to_llm_string()


# =============================================================================
# PDG API Functions (Layer 5)
# =============================================================================

def get_pdg_context(
    source_or_path: str,
    function_name: str,
    language: str = "python"
) -> dict | None:
    """
    Get program dependence graph context for a function.

    Provides control and data dependencies unified in a single graph,
    useful for understanding code impact and program slicing.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of the function to analyze
        language: One of "python", "typescript", "javascript", "go", "rust", "java", "c"

    Returns:
        Dict with PDG summary including:
        - function: Function name
        - nodes: Number of PDG nodes
        - edges: List of edge dicts with type and label
        - control_edges: Count of control dependency edges
        - data_edges: Count of data dependency edges
        - complexity: Cyclomatic complexity from CFG
        - variables: List of tracked variable names

        Returns None if function not found or extraction fails.

    Raises:
        ValueError: If language is not supported

    Example:
        >>> code = "def add(a, b):\\n    c = a + b\\n    return c"
        >>> ctx = get_pdg_context(code, "add")
        >>> ctx["function"]
        'add'
        >>> ctx["complexity"]
        1
    """
    source_code, _ = _resolve_source(source_or_path)
    pdg = extract_pdg(source_code, function_name, language)
    if pdg is None:
        return None

    return pdg.to_compact_dict()


def get_slice(
    source_or_path: str,
    function_name: str,
    line: int,
    direction: str = "backward",
    variable: str | None = None,
    language: str = "python"
) -> set[int]:
    """
    Get program slice - lines affecting or affected by a given line.

    Program slicing identifies which parts of code are relevant to
    a specific computation, useful for debugging and understanding
    code dependencies.

    Args:
        source_or_path: Source code string OR path to file (auto-detected)
        function_name: Name of the function to analyze
        line: Line number to slice from
        direction: "backward" (what affects this line) or
                   "forward" (what this line affects)
        variable: Optional specific variable to trace (traces all if None)
        language: One of "python", "typescript", "javascript", "go", "rust", "java", "c"

    Returns:
        Set of line numbers in the slice. Empty set if function not found
        or line is invalid.

    Raises:
        ValueError: If direction is not "backward" or "forward"
        ValueError: If language is not supported

    Example:
        >>> code = '''
        ... def compute(x):
        ...     a = x + 1
        ...     b = a * 2
        ...     return b
        ... '''
        >>> get_slice(code, "compute", line=5, direction="backward")
        {3, 4, 5}  # Lines that affect the return
    """
    if direction not in ("backward", "forward"):
        raise ValueError(
            f"Invalid direction '{direction}'. Must be 'backward' or 'forward'."
        )

    source_code, _ = _resolve_source(source_or_path)
    pdg = extract_pdg(source_code, function_name, language)
    if pdg is None:
        return set()

    if direction == "backward":
        return pdg.backward_slice(line, variable)
    else:
        return pdg.forward_slice(line, variable)


# ==============================================================================
# Layer 2: Cross-File Call Graph Functions
# ==============================================================================


def scan_project_files(
    root: str,
    language: str = "python",
    respect_ignore: bool = True,
) -> list[str]:
    """
    Find all source files in project for given language.

    Args:
        root: Project root directory path
        language: "python", "typescript", "go", or "rust"
        respect_ignore: If True, respect .tldrignore patterns (default True)

    Returns:
        List of absolute paths to source files

    Example:
        >>> files = scan_project_files("/path/to/project", "python")
        >>> print(files)
        ['/path/to/project/main.py', '/path/to/project/utils/helper.py']
    """
    return _scan_project(root, language, respect_ignore=respect_ignore)


def get_imports(file_path: str, language: str = "python") -> list[dict]:
    """
    Parse imports from a source file.

    Args:
        file_path: Path to source file
        language: "python", "typescript", "go", or "rust"

    Returns:
        List of import info dicts. Structure varies by language:
        - Python: {module, names, is_from, alias/aliases}
        - TypeScript: {module, names, is_default, aliases}
        - Go: {module, alias}
        - Rust: {module, names, is_mod}

    Example:
        >>> imports = get_imports("/path/to/file.py", "python")
        >>> print(imports)
        [{'module': 'os', 'names': [], 'is_from': False, 'alias': None},
         {'module': 'pathlib', 'names': ['Path'], 'is_from': True, 'aliases': {}}]
    """
    if language == "python":
        return _parse_imports(file_path)
    elif language == "typescript" or language == "javascript":
        return _parse_ts_imports(file_path)
    elif language == "go":
        return _parse_go_imports(file_path)
    elif language == "rust":
        return _parse_rust_imports(file_path)
    elif language == "java":
        return _parse_java_imports(file_path)
    elif language == "c":
        return _parse_c_imports(file_path)
    elif language == "cpp":
        return _parse_cpp_imports(file_path)
    elif language == "ruby":
        return _parse_ruby_imports(file_path)
    elif language == "php":
        return _parse_php_imports(file_path)
    elif language == "kotlin":
        return _parse_kotlin_imports(file_path)
    elif language == "swift":
        return _parse_swift_imports(file_path)
    elif language == "csharp":
        return _parse_csharp_imports(file_path)
    elif language == "scala":
        return _parse_scala_imports(file_path)
    elif language == "lua":
        return _parse_lua_imports(file_path)
    elif language == "luau":
        return _parse_luau_imports(file_path)
    elif language == "elixir":
        # parse_elixir_imports returns dict keyed by defmodule; flatten for API
        scoped = _parse_elixir_imports(file_path)
        flat = list(chain.from_iterable(scoped.values()))
        return flat
    else:
        raise ValueError(f"Unsupported language: {language}")


def build_function_index(root: str, language: str = "python") -> dict:
    """
    Build index mapping (module, func) -> file_path for all functions.

    Args:
        root: Project root directory path
        language: "python", "typescript", "go", or "rust"

    Returns:
        Dict mapping (module_name, func_name) tuples and "module.func" strings
        to relative file paths

    Example:
        >>> index = build_function_index("/path/to/project", "python")
        >>> print(index[("utils", "helper")])
        'utils.py'
        >>> print(index["utils.helper"])
        'utils.py'
    """
    return _build_function_index(root, language)


# =============================================================================
# Layer 1 (AST) API Functions
# =============================================================================


def get_intra_file_calls(file_path: str) -> dict:
    """
    Get call graph within a single file.

    Extracts function call relationships showing which functions
    call which other functions within the same file.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Dict with two keys:
        - calls: dict mapping caller -> list of callees
        - called_by: dict mapping callee -> list of callers

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file cannot be parsed

    Example:
        >>> cg = get_intra_file_calls("/path/to/file.py")
        >>> cg["calls"]["main"]  # Functions called by main
        ['helper', 'process']
        >>> cg["called_by"]["helper"]  # Functions that call helper
        ['main']
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    module_info = _extract_file_impl(file_path)
    return {
        "calls": dict(module_info.call_graph.calls),
        "called_by": dict(module_info.call_graph.called_by),
    }


def _load_module_info(file_path: str, base_path: str | None = None):
    """Validate the path and extract module info — shared preamble for
    :func:`extract_file` and :func:`extract_file_with_code`.

    Returns a ``(path, module_info)`` tuple. Raises ``FileNotFoundError``
    if the path does not exist and ``PathTraversalError`` / ``ValueError``
    via :func:`_validate_path_containment`.
    """
    # Security: Validate path containment
    _validate_path_containment(file_path, base_path)

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    module_info = _extract_file_impl(file_path)
    return path, module_info


def extract_file(file_path: str, base_path: str | None = None) -> dict:
    """
    Extract code structure from any supported file.

    Generic file extractor that returns complete module information
    including imports, functions, classes, and call graph.

    Args:
        file_path: Path to the file to analyze
        base_path: Optional base directory for path containment validation.
                   If provided, file_path must resolve within base_path.

    Returns:
        Dict containing:
        - file_path: Path to the analyzed file
        - language: Detected language (e.g., "python")
        - docstring: Module-level docstring if present
        - imports: List of import dicts
        - functions: List of function dicts with signatures (each includes
          an ``end_line`` key giving the symbol's last source line, or 0
          when the underlying extractor cannot determine it)
        - classes: List of class dicts with methods (also includes
          ``end_line`` on each class and method)
        - call_graph: Dict with calls and called_by relationships

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file type is not supported or path is invalid
        PathTraversalError: If path escapes base_path via traversal

    Example:
        >>> info = extract_file("/path/to/module.py")
        >>> print(info["functions"][0]["signature"])
        'def my_function(x: int) -> str'
    """
    _, module_info = _load_module_info(file_path, base_path)
    return module_info.to_dict()


def _inject_method_code_spans(classes: list, method_end: dict, _span) -> None:
    """Inject 'code' spans on all methods in the given filtered classes."""
    for c in classes:
        for m in c.get("methods", []):
            m_end = method_end.get(
                (c.get("name"), m.get("name"), m.get("line_number")), 0
            )
            m_code = _span(m.get("line_number") or 0, m_end)
            if m_code is not None:
                m["code"] = m_code


def extract_file_with_code(
    file_path: str,
    function: str | None = None,
    method: str | None = None,
    class_: str | None = None,
    base_path: str | None = None,
) -> dict:
    """
    Extract code structure and inject a 'code' (source span) field on
    symbols matched by ``function`` / ``method`` / ``class_`` filters.

    Same return shape as :func:`extract_file`, plus a ``code`` key on
    each matched function / method / class dict whose extractor populated
    ``end_line``. Bare extraction (no filter) is left unchanged — callers
    that want metadata only should keep using :func:`extract_file`.

    Args:
        file_path: Path to the file to analyze
        function: Name of a top-level function to filter to (and enrich)
        method: ``Class.method`` selector to filter to (and enrich)
        class_: Class name to filter to (enriches the class and its methods)
        base_path: Optional base directory for path containment validation

    Returns:
        Dict with the same shape as :func:`extract_file` (including the
        ``end_line`` key on every function / method / class dict), plus a
        ``code`` field on matches whose ``end_line`` is known.
    """
    path, module_info = _load_module_info(file_path, base_path)
    result = module_info.to_dict()

    if not (function or method or class_):
        return result

    # Build {(name, line_number): end_line} maps from the dataclasses
    # so we can look up end_line without exposing it in to_dict().
    func_end = {(f.name, f.line_number): f.end_line for f in module_info.functions}
    class_end = {(c.name, c.line_number): c.end_line for c in module_info.classes}
    method_end = {
        (c.name, m.name, m.line_number): m.end_line
        for c in module_info.classes
        for m in c.methods
    }

    try:
        source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, UnicodeError):
        source_lines = None

    def _span(start: int, end: int) -> str | None:
        if source_lines is None or start <= 0 or end <= 0 or end < start:
            return None
        lo = start - 1
        hi = min(end, len(source_lines))
        if lo >= len(source_lines):
            return None
        return "\n".join(source_lines[lo:hi])

    # Apply class filter
    if class_:
        result["classes"] = [
            c for c in result.get("classes", [])
            if c.get("name") == class_
        ]
    elif method:
        parts = method.split(".", 1)
        if len(parts) == 2:
            class_name, method_name = parts
            filtered = []
            for c in result.get("classes", []):
                if c.get("name") == class_name:
                    c_copy = dict(c)
                    c_copy["methods"] = [
                        m for m in c.get("methods", [])
                        if m.get("name") == method_name
                    ]
                    filtered.append(c_copy)
            result["classes"] = filtered
        else:
            result["classes"] = []
    elif function:
        # When the filter is `function=NAME` and NAME matches no top-level
        # function, fall back to searching classes[].methods[] so the named
        # symbol still surfaces. The containing class is retained with the
        # method list narrowed to just the matching entries.
        top_level_match = any(
            f.get("name") == function
            for f in result.get("functions", [])
        )
        if top_level_match:
            result["classes"] = []
        else:
            filtered = []
            for c in result.get("classes", []):
                matching_methods = [
                    m for m in c.get("methods", [])
                    if m.get("name") == function
                ]
                if matching_methods:
                    c_copy = dict(c)
                    c_copy["methods"] = matching_methods
                    filtered.append(c_copy)
            result["classes"] = filtered
    else:
        result["classes"] = []

    # Apply function filter
    if function:
        result["functions"] = [
            f for f in result.get("functions", [])
            if f.get("name") == function
        ]
    elif class_ or method:
        # Class or method filter implies the caller wants only the named
        # class/method; clear top-level functions for a minimal result.
        # (CLI bare-extract path does not call this function and is unaffected.)
        result["functions"] = []

    # Inject 'code' on filtered matches whose extractor knows end_line.
    if function:
        for f in result.get("functions", []):
            end = func_end.get((f.get("name"), f.get("line_number")), 0)
            code = _span(f.get("line_number") or 0, end)
            if code is not None:
                f["code"] = code
        # If `function` fell back to a class method, inject method code spans too.
        _inject_method_code_spans(result.get("classes", []), method_end, _span)
    if class_:
        for c in result.get("classes", []):
            end = class_end.get((c.get("name"), c.get("line_number")), 0)
            code = _span(c.get("line_number") or 0, end)
            if code is not None:
                c["code"] = code
        _inject_method_code_spans(result.get("classes", []), method_end, _span)
    elif method:
        _inject_method_code_spans(result.get("classes", []), method_end, _span)

    return result


# =============================================================================
# Project Navigation Functions
# =============================================================================


def get_file_tree(
    root: str | Path,
    extensions: set[str] | None = None,
    exclude_hidden: bool = True,
    ignore_spec=None,
) -> dict:
    """
    Get file tree structure for a project.

    Args:
        root: Root directory to scan
        extensions: Optional set of extensions to include (e.g., {".py", ".ts"})
        exclude_hidden: If True, exclude hidden files/directories (default True)
        ignore_spec: Optional pathspec.PathSpec for gitignore-style patterns

    Returns:
        Dict with tree structure:
        {
            "name": "project",
            "type": "dir",
            "children": [
                {"name": "src", "type": "dir", "children": [...]},
                {"name": "main.py", "type": "file", "path": "src/main.py"}
            ]
        }

    Raises:
        PathTraversalError: If root path contains directory traversal patterns
    """
    # Security: Validate path containment
    _validate_path_containment(str(root))

    root = Path(root)

    def scan_dir(path: Path) -> dict:
        result = {"name": path.name, "type": "dir", "children": []}

        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return result

        for item in items:
            # Skip hidden files/dirs
            if exclude_hidden and item.name.startswith("."):
                continue

            # Get relative path for ignore matching
            try:
                rel_path = str(item.relative_to(root))
            except ValueError:
                rel_path = item.name

            if item.is_dir():
                # Check if directory should be ignored
                if ignore_spec and ignore_spec.match_file(rel_path + "/"):
                    continue
                child = scan_dir(item)
                # Only include non-empty directories
                if child["children"] or extensions is None:
                    result["children"].append(child)
            elif item.is_file():
                # Check if file should be ignored
                if ignore_spec and ignore_spec.match_file(rel_path):
                    continue
                if extensions is None or item.suffix in extensions:
                    result["children"].append(
                        {
                            "name": item.name,
                            "type": "file",
                            "path": rel_path,
                        }
                    )

        return result

    return scan_dir(root)


def search(
    pattern: str,
    root: str | Path,
    extensions: set[str] | None = None,
    context_lines: int = 0,
    max_results: int = 100,
    max_files: int = 10000,
    ignore_spec=None,
) -> list[dict]:
    """
    Search files for a regex pattern.

    Args:
        pattern: Regex pattern to search for
        root: Directory to search recursively, OR a single file to search.
              When ``root`` is a file, ``extensions`` is ignored (explicit file
              beats filter) and ignore_spec / hidden-dir / SKIP_DIRS gating are
              skipped. The ``file`` field in each result is the file's basename.
        extensions: Optional set of extensions to filter (e.g., {".py"}).
            Ignored when ``root`` is a file path.
        context_lines: Number of context lines to include (default 0)
        max_results: Maximum matches to return (default 100, 0 = unlimited)
        max_files: Maximum files to scan (default 10000, 0 = unlimited)
        ignore_spec: Optional pathspec.PathSpec for gitignore-style patterns

    Returns:
        List of matches:
        [
            {"file": "src/main.py", "line": 10, "content": "def hello():"},
            ...
        ]

    Raises:
        PathTraversalError: If root path contains directory traversal patterns
    """
    # Security: Validate path containment
    _validate_path_containment(str(root))

    import re

    # Fallback directories to skip if no ignore_spec provided
    SKIP_DIRS = {
        "node_modules", "__pycache__", ".git", ".svn", ".hg",
        "dist", "build", ".next", ".nuxt", "coverage", ".tox",
        "venv", ".venv", "env", ".env", "vendor", ".cache",
    }

    results = []
    root = Path(root)
    compiled = re.compile(pattern)
    files_scanned = 0

    # Single-file mode: user named an explicit file path. Skip rglob, ignore
    # gating, and the extensions filter — explicit beats filter.
    if root.is_file():
        try:
            content = root.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                if compiled.search(line):
                    match = {
                        "file": root.name,
                        "line": i,
                        "content": line.strip(),
                    }
                    if context_lines > 0:
                        start = max(0, i - 1 - context_lines)
                        end = min(len(lines), i + context_lines)
                        match["context"] = lines[start:end]
                    results.append(match)
                    if max_results > 0 and len(results) >= max_results:
                        return results
        except OSError:
            pass
        return results

    for file_path in root.rglob("*"):
        # Check file limit
        if max_files > 0 and files_scanned >= max_files:
            break

        if not file_path.is_file():
            continue

        # Get relative path for filtering
        try:
            rel_path = file_path.relative_to(root)
            rel_path_str = str(rel_path)
            parts = rel_path.parts
        except ValueError:
            continue

        # Use ignore_spec if provided, otherwise fall back to hardcoded SKIP_DIRS
        if ignore_spec:
            if ignore_spec.match_file(rel_path_str):
                continue
        else:
            # Fallback: skip hidden files and junk directories
            if any(part.startswith(".") for part in parts):
                continue
            if any(part in SKIP_DIRS for part in parts):
                continue

        # Filter by extension
        if extensions and file_path.suffix not in extensions:
            continue

        files_scanned += 1

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()

            for i, line in enumerate(lines, 1):
                if compiled.search(line):
                    match = {
                        "file": str(file_path.relative_to(root)),
                        "line": i,
                        "content": line.strip(),
                    }

                    # Add context if requested
                    if context_lines > 0:
                        start = max(0, i - 1 - context_lines)
                        end = min(len(lines), i + context_lines)
                        match["context"] = lines[start:end]

                    results.append(match)

                    # Check result limit
                    if max_results > 0 and len(results) >= max_results:
                        return results
        except (OSError, UnicodeDecodeError):
            pass

    return results


class Selection:
    """
    Manage file selection state for batch operations.

    Usage:
        sel = Selection()
        sel.add("src/main.py", "src/utils.py")
        sel.remove("src/utils.py")

        for f in sel.files:
            info = extract_file(f)
    """

    def __init__(self):
        self._selected: set[str] = set()

    def add(self, *paths: str) -> "Selection":
        """Add paths to selection."""
        self._selected.update(paths)
        return self

    def remove(self, *paths: str) -> "Selection":
        """Remove paths from selection."""
        self._selected -= set(paths)
        return self

    def clear(self) -> "Selection":
        """Clear all selection."""
        self._selected.clear()
        return self

    def set(self, *paths: str) -> "Selection":
        """Replace entire selection with new paths."""
        self._selected = set(paths)
        return self

    @property
    def files(self) -> list[str]:
        """Return selected files as sorted list."""
        return sorted(self._selected)

    def __contains__(self, path: str) -> bool:
        """Check if path is selected."""
        return path in self._selected

    def __len__(self) -> int:
        """Return number of selected files."""
        return len(self._selected)


# NON_CODE_EXTENSIONS moved to module top (near _EXT_MAP_ALL_LANGUAGES) per
# review C-9. Identity preserved — `from tldr.api import NON_CODE_EXTENSIONS`
# call sites (e.g. semantic._process_file_for_extraction) are unchanged.


def _build_file_entry(info_dict: dict, path: str) -> dict:
    """Build a file entry dict from an extracted info dict and a path label."""
    functions = [f["name"] for f in info_dict.get("functions", [])]
    methods = []
    for cls in info_dict.get("classes", []):
        for method in cls.get("methods", []):
            method_name = method.get("name", "")
            if method_name:
                methods.append(method_name)
                functions.append(method_name)  # Also in functions for discoverability
    return {
        "path": path,
        "functions": functions,
        "classes": [c["name"] for c in info_dict.get("classes", [])],
        "methods": methods,
        "imports": info_dict.get("imports", []),
    }


def _build_empty_file_entry_skeleton(path: str) -> dict:
    """Build the base file entry structure with empty lists."""
    return {
        "path": path,
        "functions": [],
        "classes": [],
        "methods": [],
        "imports": [],
    }


def _build_non_code_file_entry(path: str) -> dict:
    """Build a file entry dict for a non-code file (.sh, .md, .toml, ...).

    Non-code files have no functions, classes, or imports to extract; downstream
    semantic emission (Gate 2) treats this empty entry as a signal to produce
    one whole-file EmbeddingUnit.
    """
    return _build_empty_file_entry_skeleton(path)


def get_code_structure(
    root: str | Path,
    language: str = "python",
    max_results: int = 100,
    ignore_spec=None,
) -> dict:
    """
    Get code structure (codemaps) for all code and non-code files in a project.

    Args:
        root: Root directory to analyze
        language: Language to analyze ("python", "typescript", "javascript", "go", "rust")
        max_results: Maximum number of files to analyze (default 100)
        ignore_spec: Optional pathspec.PathSpec for gitignore-style patterns

    Returns:
        Dict with codemap structure. Code files include functions/classes/imports.
        Non-code files (.sh, .md, .toml, .yaml, .yml, .json, .rst, .txt) appear
        with empty functions/classes/methods/imports lists, and are processed
        whole-file by the semantic indexer:
        {
            "root": "/path/to/project",
            "files": [
                {
                    "path": "src/main.py",
                    "functions": ["main", "helper"],
                    "classes": ["MyClass"],
                    "imports": ["os", "sys"]
                },
                {
                    "path": "build.sh",
                    "functions": [],
                    "classes": [],
                    "methods": [],
                    "imports": []
                },
                ...
            ]
        }
    """
    root = Path(root)

    # Get extension map for language
    code_extensions = _EXT_MAP_ALL_LANGUAGES.get(language, {".py"})
    # Bug 004 fix (Gate 1): non-code build/config/doc files are also indexed so
    # that downstream semantic search can rank them. They are recognized for
    # every code language; the matching emission path in semantic.py
    # (Gate 2) produces a single whole-file EmbeddingUnit per non-code file.
    extensions = code_extensions | NON_CODE_EXTENSIONS

    result = {"root": str(root), "language": language, "files": []}

    # Handle single-file input: rglob("*") on a file returns empty iterator.
    # Use the file's basename so downstream consumers see a meaningful path in
    # metadata.json (qualified_name, display). Downstream extractors that need
    # to read the file reconstruct full_path via project.parent / file_path
    # when project itself is a file (see _process_file_for_extraction). Storing
    # "." here previously caused IsADirectoryError on platforms where the
    # resolved path collapses to the parent directory (bug-004 R-5).
    if root.is_file():
        file_path = root.name
        if root.suffix in NON_CODE_EXTENSIONS:
            # Non-code file: emit a minimal entry without invoking the AST extractor.
            result["files"].append(_build_non_code_file_entry(file_path))
        elif root.suffix in code_extensions:
            try:
                info = _extract_file_impl(str(root))
                result["files"].append(_build_file_entry(info.to_dict(), file_path))
            except Exception:
                pass
        return result

    # R-4 (Bug 004 awareness): non-code files (.sh/.md/.toml/...) now share the
    # `max_results` budget with code files. In doc-heavy repos at the default
    # cap of 100 this can starve Python/TS entries — raise `--max` if needed.
    count = 0
    for file_path in root.rglob("*"):
        if count >= max_results:
            break

        if not file_path.is_file():
            continue

        if file_path.suffix not in extensions:
            continue

        # Skip hidden files (only check relative path, not parent directories)
        try:
            rel_path = file_path.relative_to(root)
            if any(part.startswith(".") for part in rel_path.parts):
                continue
        except ValueError:
            continue

        if ignore_spec and ignore_spec.match_file(rel_path):
            continue

        rel_path_str = str(rel_path)

        if file_path.suffix in NON_CODE_EXTENSIONS:
            # Non-code file: emit a minimal entry without invoking the AST extractor.
            # _extract_file_impl has no extractor for these suffixes and would fail
            # silently, dropping the file. Semantic.py emits a whole-file unit instead.
            result["files"].append(_build_non_code_file_entry(rel_path_str))
            count += 1
            continue

        try:
            info = _extract_file_impl(str(file_path))
            result["files"].append(
                _build_file_entry(info.to_dict(), rel_path_str)
            )
            count += 1
        except Exception:
            # Skip files that can't be parsed
            pass

    return result


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tldr.api <project_path> <entry_point> [depth] [language]")
        print("Example: python -m tldr.api /path/to/project build_project_call_graph 2 python")
        sys.exit(1)

    project_path = sys.argv[1]
    entry = sys.argv[2]
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    lang = sys.argv[4] if len(sys.argv) > 4 else "python"

    print(query(project_path, entry, depth, lang))
