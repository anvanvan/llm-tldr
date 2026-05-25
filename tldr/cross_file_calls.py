"""
Cross-file call graph resolution.

Builds a project-wide call graph that resolves function calls across files
by analyzing import statements and matching call sites to definitions.

Supports: Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby, PHP,
Swift, Kotlin, C#, Scala, Lua, Luau, and Elixir.

Key functions:
- scan_project(root, language) - find all source files in a project
- parse_imports(file, language) - extract import statements from a file
- build_function_index(root, language) - map {module.func: file_path} for all functions
- resolve_calls(file, index, language) - match call sites to definitions
- build_project_call_graph(root, language) - orchestrate all to build complete graph
"""

import ast
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from tldr.workspace import WorkspaceConfig, load_workspace_config, filter_paths

# Tree-sitter support for TypeScript
try:
    import tree_sitter
    import tree_sitter_typescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Tree-sitter support for Go
TREE_SITTER_GO_AVAILABLE = False
try:
    import tree_sitter_go
    TREE_SITTER_GO_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Rust
TREE_SITTER_RUST_AVAILABLE = False
try:
    import tree_sitter_rust
    TREE_SITTER_RUST_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Java
TREE_SITTER_JAVA_AVAILABLE = False
try:
    import tree_sitter_java
    TREE_SITTER_JAVA_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for C
TREE_SITTER_C_AVAILABLE = False
try:
    import tree_sitter_c
    TREE_SITTER_C_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Ruby
TREE_SITTER_RUBY_AVAILABLE = False
try:
    import tree_sitter_ruby
    TREE_SITTER_RUBY_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for PHP
TREE_SITTER_PHP_AVAILABLE = False
try:
    import tree_sitter_php
    TREE_SITTER_PHP_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for C++
TREE_SITTER_CPP_AVAILABLE = False
try:
    import tree_sitter_cpp
    TREE_SITTER_CPP_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Kotlin
TREE_SITTER_KOTLIN_AVAILABLE = False
try:
    import tree_sitter_kotlin
    TREE_SITTER_KOTLIN_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Swift
TREE_SITTER_SWIFT_AVAILABLE = False
try:
    import tree_sitter_swift
    TREE_SITTER_SWIFT_AVAILABLE = True
except ImportError:
    pass

def _warn_swift_unavailable_once() -> None:
    """Emit a Python warning when tree_sitter_swift is missing.

    Swift call-graph and import extraction silently degrade to empty results
    when the optional ``tree_sitter_swift`` dependency is not installed.
    This helper surfaces that degradation as a ``RuntimeWarning`` so callers
    and end users get an actionable diagnostic.

    Implementation note: relies on Python's default warning-filter
    de-duplication (``"once"`` per message+category+module) to deliver a
    single emission per process under normal use, while still re-emitting
    under ``warnings.simplefilter("always")`` (used by tests). A module-level
    flag would block the second emission under "always" and break diagnostics.
    """
    warnings.warn(
        "tree_sitter_swift is not installed; Swift import and call-graph "
        "extraction will return empty results. Install the optional "
        "'tree_sitter_swift' package to enable Swift support.",
        RuntimeWarning,
        stacklevel=2,
    )

# Tree-sitter support for C#
TREE_SITTER_CSHARP_AVAILABLE = False
try:
    import tree_sitter_c_sharp
    TREE_SITTER_CSHARP_AVAILABLE = True
except ImportError:
    pass

TREE_SITTER_SCALA_AVAILABLE = False
try:
    import tree_sitter_scala
    TREE_SITTER_SCALA_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Lua
TREE_SITTER_LUA_AVAILABLE = False
try:
    import tree_sitter_lua
    TREE_SITTER_LUA_AVAILABLE = True
except ImportError:
    pass

# Tree-sitter support for Elixir
TREE_SITTER_ELIXIR_AVAILABLE = False
try:
    import tree_sitter_elixir
    TREE_SITTER_ELIXIR_AVAILABLE = True
except ImportError:
    pass


# Languages with a full _build_*_call_graph implementation in build_project_call_graph.
CALL_GRAPH_LANGUAGES: frozenset[str] = frozenset(
    {"python", "typescript", "go", "rust", "java", "c", "php"}
)

# Synthetic sentinel used by the Ruby call-graph builder to mark "orphan"
# top-level defs (defined functions with no real callers or callees) so they
# appear as a `from_func` in the edge set without polluting real symbols.
# Filtered out at the CLI layer before user-facing output. Shared with
# tldr/cli.py to avoid string-literal divergence.
RUBY_ORPHAN_SENTINEL = "__ruby_orphan__"


@dataclass
class ProjectCallGraph:
    """Cross-file call graph with edges as (src_file, src_func, dst_file, dst_func).

    Optionally carries a side-table mapping each edge to a 1-indexed source line
    number of the call site (the line in src_file where src_func calls dst_func).
    Populated only by builders that capture line info (currently the Ruby
    builder for top-level call sites — bug 006). Other builders that
    do not record lines leave the table empty; ``lines_for_edge`` then returns
    ``None`` for those edges and downstream code (analysis._build_caller_tree)
    omits the ``line`` key. This keeps the change backward-compatible.
    """

    _edges: set[tuple[str, str, str, str]] = field(default_factory=set)
    _edge_lines: dict[tuple[str, str, str, str], int] = field(default_factory=dict)

    def add_edge(self, src_file: str, src_func: str, dst_file: str, dst_func: str):
        """Add a call edge from src_file:src_func to dst_file:dst_func."""
        self._edges.add((src_file, src_func, dst_file, dst_func))

    def add_edge_with_line(
        self,
        src_file: str,
        src_func: str,
        dst_file: str,
        dst_func: str,
        line: int | None,
    ):
        """Add a call edge and record the 1-indexed source line of the call site.

        ``line`` may be ``None`` (no line info available); in that case behaves
        like ``add_edge``. First recorded line wins for any given edge tuple.
        """
        edge = (src_file, src_func, dst_file, dst_func)
        self._edges.add(edge)
        if line is not None:
            self._edge_lines.setdefault(edge, line)

    @property
    def edges(self) -> set[tuple[str, str, str, str]]:
        """Return all edges as a set of tuples."""
        return self._edges

    def lines_for_edge(
        self, edge: tuple[str, str, str, str]
    ) -> int | None:
        """Return the recorded 1-indexed call-site line for an edge, or None."""
        return self._edge_lines.get(edge)

    def __contains__(self, edge: tuple[str, str, str, str]) -> bool:
        """Check if an edge exists in the graph."""
        return edge in self._edges


def _get_ts_parser(language: str = "typescript"):
    """Get or create a tree-sitter TypeScript-family parser."""
    if not TREE_SITTER_AVAILABLE:
        raise RuntimeError("tree-sitter-typescript not available")

    # JavaScript frequently includes JSX; use TSX grammar for that branch.
    if language == "javascript":
        ts_lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
    else:
        ts_lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    parser = tree_sitter.Parser(ts_lang)
    return parser


def _get_rust_parser():
    """Get or create a tree-sitter Rust parser."""
    if not TREE_SITTER_RUST_AVAILABLE:
        raise RuntimeError("tree-sitter-rust not available")

    rust_lang = tree_sitter.Language(tree_sitter_rust.language())
    parser = tree_sitter.Parser(rust_lang)
    return parser


def _get_go_parser():
    """Get or create a tree-sitter Go parser."""
    if not TREE_SITTER_GO_AVAILABLE:
        raise RuntimeError("tree-sitter-go not available")

    go_lang = tree_sitter.Language(tree_sitter_go.language())
    parser = tree_sitter.Parser(go_lang)
    return parser


def _get_java_parser():
    """Get or create a tree-sitter Java parser."""
    if not TREE_SITTER_JAVA_AVAILABLE:
        raise RuntimeError("tree-sitter-java not available")

    java_lang = tree_sitter.Language(tree_sitter_java.language())
    parser = tree_sitter.Parser(java_lang)
    return parser


def _get_c_parser():
    """Get or create a tree-sitter C parser."""
    if not TREE_SITTER_C_AVAILABLE:
        raise RuntimeError("tree-sitter-c not available")

    c_lang = tree_sitter.Language(tree_sitter_c.language())
    parser = tree_sitter.Parser(c_lang)
    return parser


def _get_ruby_parser():
    """Get or create a tree-sitter Ruby parser."""
    if not TREE_SITTER_RUBY_AVAILABLE:
        raise RuntimeError("tree-sitter-ruby not available")

    ruby_lang = tree_sitter.Language(tree_sitter_ruby.language())
    parser = tree_sitter.Parser(ruby_lang)
    return parser


def _get_php_parser():
    """Get or create a tree-sitter PHP parser."""
    if not TREE_SITTER_PHP_AVAILABLE:
        raise RuntimeError("tree-sitter-php not available")

    php_lang = tree_sitter.Language(tree_sitter_php.language_php())
    parser = tree_sitter.Parser(php_lang)
    return parser


def _get_cpp_parser():
    """Get or create a tree-sitter C++ parser."""
    if not TREE_SITTER_CPP_AVAILABLE:
        raise RuntimeError("tree-sitter-cpp not available")

    cpp_lang = tree_sitter.Language(tree_sitter_cpp.language())
    parser = tree_sitter.Parser(cpp_lang)
    return parser


def _get_kotlin_parser():
    """Get or create a tree-sitter Kotlin parser."""
    if not TREE_SITTER_KOTLIN_AVAILABLE:
        raise RuntimeError("tree-sitter-kotlin not available")

    kotlin_lang = tree_sitter.Language(tree_sitter_kotlin.language())
    parser = tree_sitter.Parser(kotlin_lang)
    return parser


def _get_swift_parser():
    """Get or create a tree-sitter Swift parser."""
    if not TREE_SITTER_SWIFT_AVAILABLE:
        raise RuntimeError("tree-sitter-swift not available")

    swift_lang = tree_sitter.Language(tree_sitter_swift.language())
    parser = tree_sitter.Parser(swift_lang)
    return parser


def _get_csharp_parser():
    """Get or create a tree-sitter C# parser."""
    if not TREE_SITTER_CSHARP_AVAILABLE:
        raise RuntimeError("tree-sitter-c-sharp not available")

    csharp_lang = tree_sitter.Language(tree_sitter_c_sharp.language())
    parser = tree_sitter.Parser(csharp_lang)
    return parser


def _get_scala_parser():
    """Get or create a tree-sitter Scala parser."""
    if not TREE_SITTER_SCALA_AVAILABLE:
        raise RuntimeError("tree-sitter-scala not available")

    scala_lang = tree_sitter.Language(tree_sitter_scala.language())
    parser = tree_sitter.Parser(scala_lang)
    return parser


def scan_project(
    root: str | Path,
    language: str = "python",
    workspace_config: Optional[WorkspaceConfig] = None,
    respect_ignore: bool = True,
) -> list[str]:
    """
    Find all source files in the project for the given language.

    Args:
        root: Project root directory
        language: "python", "typescript", "go", or "rust"
        workspace_config: Optional WorkspaceConfig for monorepo scoping.
                         If provided, filters files by activePackages and excludePatterns.
        respect_ignore: If True, respect .tldrignore patterns (default True)

    Returns:
        List of absolute paths to source files
    """
    from .tldrignore import (
        load_ignore_patterns, should_ignore,
        batch_gitignored, is_git_repo, _has_negation_for_file,
    )

    root = Path(root).resolve()
    files = []

    # Load ignore patterns if respecting .tldrignore
    ignore_spec = load_ignore_patterns(root) if respect_ignore else None

    # Cache git repo check to avoid calling is_git_repo on every os.walk iteration
    _is_git = is_git_repo(str(root)) if respect_ignore else False

    if language == "python":
        extensions = {'.py'}
    elif language == "typescript":
        extensions = {'.ts', '.tsx'}
    elif language == "javascript":
        extensions = {'.js', '.jsx', '.mjs', '.cjs'}
    elif language == "go":
        extensions = {'.go'}
    elif language == "rust":
        extensions = {'.rs'}
    elif language == "java":
        extensions = {'.java'}
    elif language == "c":
        extensions = {'.c', '.h'}
    elif language == "cpp":
        extensions = {'.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx'}
    elif language == "ruby":
        extensions = {'.rb'}
    elif language == "php":
        extensions = {'.php'}
    elif language == "kotlin":
        extensions = {'.kt', '.kts'}
    elif language == "swift":
        extensions = {'.swift'}
    elif language == "csharp":
        extensions = {'.cs'}
    elif language == "scala":
        extensions = {'.scala', '.sc'}
    elif language == "lua":
        extensions = {'.lua'}
    elif language == "luau":
        extensions = {'.luau'}
    elif language == "elixir":
        extensions = {'.ex', '.exs'}
    else:
        raise ValueError(f"Unsupported language: {language}")

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip ignored directories (modifying dirnames in-place prunes os.walk)
        # use_gitignore=False avoids spawning a subprocess per directory;
        # gitignore is checked in a single batch call after file collection
        if respect_ignore and ignore_spec:
            rel_dir = os.path.relpath(dirpath, root)
            # Check if current directory should be ignored
            if rel_dir != '.' and should_ignore(
                rel_dir + '/', root, ignore_spec, use_gitignore=False
            ):
                dirnames.clear()  # Don't descend into ignored directories
                continue
            # Filter subdirectories
            dirnames[:] = [
                d for d in dirnames
                if not should_ignore(
                    os.path.join(rel_dir, d) + '/', root, ignore_spec,
                    use_gitignore=False,
                )
            ]

        # Batch-check gitignored directories so os.walk doesn't descend into
        # them (e.g. .venv/, node_modules/).  Without this, we'd collect
        # thousands of files only to discard them at the file-level batch check.
        if respect_ignore and _is_git and dirnames:
            dir_paths = [Path(os.path.join(dirpath, d)) for d in dirnames]
            git_ignored_dirs = batch_gitignored(dir_paths, root)
            if git_ignored_dirs:
                pruned = []
                for d in dirnames:
                    rel_d = os.path.relpath(os.path.join(dirpath, d), root)
                    if rel_d not in git_ignored_dirs or (
                        ignore_spec and _has_negation_for_file(ignore_spec, rel_d)
                    ):
                        pruned.append(d)
                dirnames[:] = pruned

        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(dirpath, filename)
                # Check individual file against .tldrignore patterns only
                if respect_ignore and ignore_spec:
                    rel_path = os.path.relpath(file_path, root)
                    if should_ignore(
                        rel_path, root, ignore_spec, use_gitignore=False
                    ):
                        continue
                files.append(file_path)

    # Batch-check gitignore in a single subprocess call (instead of per-file).
    # Use batch_gitignored directly rather than filter_files, because
    # filter_files' gitignore pass doesn't preserve .tldrignore negation (!)
    # patterns — files explicitly un-ignored by .tldrignore must stay even
    # when gitignored.
    if respect_ignore and files and _is_git:
        gitignored = batch_gitignored([Path(f) for f in files], root)
        if gitignored:
            kept = []
            for f in files:
                rel = os.path.relpath(f, root)
                if rel not in gitignored:
                    kept.append(f)
                elif ignore_spec and _has_negation_for_file(ignore_spec, rel):
                    # .tldrignore negation overrides gitignore
                    kept.append(f)
            files = kept

    # Apply workspace config filtering if provided
    if workspace_config is not None:
        # Convert absolute paths to relative for filtering, then back to absolute
        rel_files = [os.path.relpath(f, root) for f in files]
        filtered_rel = filter_paths(rel_files, workspace_config)
        files = [os.path.join(root, f) for f in filtered_rel]

    return files


def parse_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Python file.

    Args:
        file_path: Path to Python file

    Returns:
        List of import info dicts with keys: module, names, is_from, aliases
    """
    file_path = Path(file_path)
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return []

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'module': alias.name,
                    'names': [],
                    'is_from': False,
                    'alias': alias.asname,
                })
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = []
                aliases = {}
                for alias in node.names:
                    names.append(alias.name)
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                imports.append({
                    'module': node.module,
                    'names': names,
                    'is_from': True,
                    'aliases': aliases,
                })

    return imports


def parse_ts_imports(file_path: str | Path, language: str = "typescript") -> list[dict]:
    """
    Extract import statements from a TypeScript/JavaScript file.

    Args:
        file_path: Path to TypeScript/JavaScript file
        language: "typescript" or "javascript"

    Returns:
        List of import info dicts with keys: module, names, is_default, aliases
    """
    if not TREE_SITTER_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_ts_parser(language)
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []
    seen_imports = set()

    def add_import(import_info: dict):
        if not import_info:
            return
        key = (
            import_info.get("module"),
            tuple(import_info.get("names", [])),
            import_info.get("default"),
            tuple(sorted(import_info.get("aliases", {}).items())),
        )
        if key in seen_imports:
            return
        seen_imports.add(key)
        imports.append(import_info)

    def walk_tree(node):
        if node.type == "import_statement":
            import_info = _parse_ts_import_node(node, source)
            add_import(import_info)
        elif node.type == "variable_declarator":
            for import_info in _parse_ts_require_declarator(node, source):
                add_import(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_ts_import_node(node, source: bytes) -> dict | None:
    """Parse a single TypeScript import statement."""
    module = None
    names = []
    aliases = {}
    default_name = None

    for child in node.children:
        if child.type == "string":
            # Module path - strip quotes
            module = source[child.start_byte:child.end_byte].decode("utf-8").strip("'\"")
        elif child.type == "import_clause":
            for clause_child in child.children:
                if clause_child.type == "identifier":
                    # Default import: import Foo from "module"
                    default_name = source[clause_child.start_byte:clause_child.end_byte].decode("utf-8")
                elif clause_child.type == "named_imports":
                    # Named imports: import { foo, bar as baz } from "module"
                    for named in clause_child.children:
                        if named.type == "import_specifier":
                            orig_name = None
                            alias = None
                            for spec_child in named.children:
                                if spec_child.type == "identifier":
                                    if orig_name is None:
                                        orig_name = source[spec_child.start_byte:spec_child.end_byte].decode("utf-8")
                                    else:
                                        alias = source[spec_child.start_byte:spec_child.end_byte].decode("utf-8")
                            if orig_name:
                                names.append(orig_name)
                                if alias:
                                    aliases[alias] = orig_name
                elif clause_child.type == "namespace_import":
                    # Namespace import: import * as foo from "module"
                    for ns_child in clause_child.children:
                        if ns_child.type == "identifier":
                            alias = source[ns_child.start_byte:ns_child.end_byte].decode("utf-8")
                            aliases[alias] = "*"

    if module:
        return {
            'module': module,
            'names': names,
            'default': default_name,
            'aliases': aliases,
        }
    return None


def _parse_ts_require_declarator(node, source: bytes) -> list[dict]:
    """Parse `const x = require('...')` and `const {x} = require('...')` forms."""
    if node.type != "variable_declarator":
        return []

    lhs = None
    rhs_call = None
    for child in node.children:
        if child.type in ("identifier", "object_pattern"):
            lhs = child
        elif child.type == "call_expression":
            rhs_call = child

    if lhs is None or rhs_call is None:
        return []

    module = _extract_require_module_from_call(rhs_call, source)
    if not module:
        return []

    if lhs.type == "identifier":
        local_name = source[lhs.start_byte:lhs.end_byte].decode("utf-8")
        return [{
            "module": module,
            "names": [],
            "default": local_name,
            "aliases": {local_name: "*"},
        }]

    imports = []
    if lhs.type == "object_pattern":
        names = []
        aliases = {}
        for child in lhs.children:
            if child.type == "shorthand_property_identifier_pattern":
                name = source[child.start_byte:child.end_byte].decode("utf-8")
                names.append(name)
            elif child.type == "pair_pattern":
                orig_name = None
                alias_name = None
                for pair_child in child.children:
                    if pair_child.type in (
                        "identifier",
                        "property_identifier",
                        "shorthand_property_identifier_pattern",
                    ):
                        name = source[pair_child.start_byte:pair_child.end_byte].decode("utf-8")
                        if orig_name is None:
                            orig_name = name
                        else:
                            alias_name = name
                if orig_name:
                    names.append(orig_name)
                    if alias_name and alias_name != orig_name:
                        aliases[alias_name] = orig_name

        if names:
            imports.append({
                "module": module,
                "names": names,
                "default": None,
                "aliases": aliases,
            })

    return imports


def _extract_require_module_from_call(call_node, source: bytes) -> str | None:
    """Extract module path from `require('module')` call expressions."""
    if call_node.type != "call_expression":
        return None

    is_require = False
    args_node = None

    for child in call_node.children:
        if child.type == "identifier":
            fn_name = source[child.start_byte:child.end_byte].decode("utf-8")
            if fn_name == "require":
                is_require = True
        elif child.type == "arguments":
            args_node = child

    if not is_require or args_node is None:
        return None

    for child in args_node.children:
        if child.type == "string":
            return source[child.start_byte:child.end_byte].decode("utf-8").strip("'\"")

    return None


def parse_go_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Go file.

    Args:
        file_path: Path to Go file

    Returns:
        List of import info dicts with keys: module, alias
    """
    if not TREE_SITTER_GO_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_go_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "import_declaration":
            _parse_go_import_node(node, source, imports)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_go_import_node(node, source: bytes, imports: list):
    """Parse Go import declaration - handles both single and grouped imports."""
    for child in node.children:
        if child.type == "import_spec":
            _parse_go_import_spec(child, source, imports)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _parse_go_import_spec(spec, source, imports)


def _parse_go_import_spec(spec_node, source: bytes, imports: list):
    """Parse a single Go import spec (potentially with alias)."""
    alias = None
    module = None

    for child in spec_node.children:
        if child.type == "package_identifier":
            # This is the alias: import alias "path"
            alias = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "interpreted_string_literal":
            # This is the module path
            module = source[child.start_byte:child.end_byte].decode("utf-8").strip('"')

    if module:
        imports.append({
            'module': module,
            'alias': alias,
        })


def parse_rust_imports(file_path: str | Path) -> list[dict]:
    """
    Extract use statements and mod declarations from a Rust file.

    Args:
        file_path: Path to Rust file

    Returns:
        List of import info dicts with keys: module, names, is_mod
    """
    if not TREE_SITTER_RUST_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_rust_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # Use declarations: use crate::utils::helper;
        if node.type == "use_declaration":
            import_info = _parse_rust_use_node(node, source)
            if import_info:
                imports.append(import_info)

        # Mod declarations: mod utils;
        elif node.type == "mod_item":
            # Check if it's a mod declaration (not an inline module)
            has_body = False
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = source[child.start_byte:child.end_byte].decode("utf-8")
                elif child.type == "declaration_list":
                    has_body = True

            if name and not has_body:
                imports.append({
                    'module': name,
                    'names': [],
                    'is_mod': True,
                })

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_rust_use_node(node, source: bytes) -> dict | None:
    """Parse a single Rust use statement."""
    # Get the full use path text
    text = source[node.start_byte:node.end_byte].decode("utf-8")

    # Strip "use " prefix and trailing semicolon
    text = text.replace("use ", "").rstrip(";").strip()

    # Handle pub use
    if text.startswith("pub "):
        text = text[4:].strip()

    # Parse the path to extract module and names
    # Examples:
    #   std::io              -> module="std::io", names=[]
    #   crate::utils::helper -> module="crate::utils", names=["helper"]
    #   self::inner::*       -> module="self::inner", names=["*"]
    #   std::collections::{HashMap, HashSet} -> module="std::collections", names=["HashMap", "HashSet"]

    names = []
    module = text

    # Handle glob imports: use foo::*
    if text.endswith("::*"):
        module = text[:-3]
        names = ["*"]
    # Handle grouped imports: use foo::{bar, baz}
    elif "{" in text:
        brace_start = text.index("{")
        module = text[:brace_start].rstrip("::")
        brace_content = text[brace_start+1:text.rindex("}")]
        names = [n.strip() for n in brace_content.split(",")]
    # Handle simple imports: use foo::bar
    elif "::" in text:
        parts = text.rsplit("::", 1)
        module = parts[0]
        names = [parts[1]]

    return {
        'module': module,
        'names': names,
        'is_mod': False,
    }


def parse_java_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Java file.

    Args:
        file_path: Path to Java file

    Returns:
        List of import info dicts with keys: module, is_static, is_wildcard
    """
    if not TREE_SITTER_JAVA_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_java_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "import_declaration":
            import_info = _parse_java_import_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_java_import_node(node, source: bytes) -> dict | None:
    """Parse a single Java import statement."""
    # Get the full import text
    text = source[node.start_byte:node.end_byte].decode("utf-8")

    # Check for static import
    is_static = "static " in text

    # Check for wildcard import
    is_wildcard = text.rstrip(";").endswith("*")

    # Extract the module path
    # Examples:
    #   import java.util.List;          -> module="java.util.List"
    #   import java.util.*;             -> module="java.util.*"
    #   import static java.lang.Math.PI; -> module="java.lang.Math.PI", is_static=True

    # Find the scoped_identifier or identifier node for the import path
    module = None
    for child in node.children:
        if child.type == "scoped_identifier":
            module = source[child.start_byte:child.end_byte].decode("utf-8")
            break
        elif child.type == "identifier":
            module = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "asterisk":
            # Handle wildcard - module should have been set by scoped_identifier
            if module:
                module = module + ".*"
            is_wildcard = True

    if not module:
        return None

    return {
        'module': module,
        'is_static': is_static,
        'is_wildcard': is_wildcard,
    }


def parse_kotlin_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Kotlin file.

    Args:
        file_path: Path to Kotlin file

    Returns:
        List of import info dicts with keys: module, is_wildcard, alias
    """
    if not TREE_SITTER_KOTLIN_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_kotlin_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # tree-sitter-kotlin uses "import" node type, not "import_header"
        if node.type == "import":
            import_info = _parse_kotlin_import_node(node, source)
            if import_info:
                imports.append(import_info)
            # Don't recurse into import children (they have nested "import" keywords)
            return
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_kotlin_import_node(node, source: bytes) -> dict | None:
    """Parse a single Kotlin import statement."""
    # Get the full import text
    text = source[node.start_byte:node.end_byte].decode("utf-8")

    # Check for wildcard import (ends with .*)
    is_wildcard = ".*" in text or text.rstrip().endswith("*")

    # Check for alias: import foo.bar as baz
    alias = None
    for child in node.children:
        if child.type == "as":
            # The next sibling should be the alias identifier
            idx = list(node.children).index(child)
            if idx + 1 < len(node.children):
                alias_node = node.children[idx + 1]
                if alias_node.type == "identifier":
                    alias = source[alias_node.start_byte:alias_node.end_byte].decode("utf-8")
            break

    # Extract the module path from qualified_identifier
    module = None
    for child in node.children:
        if child.type == "qualified_identifier":
            module = source[child.start_byte:child.end_byte].decode("utf-8")
            break

    # Handle wildcard: if there's a * after qualified_identifier, append it
    if module and is_wildcard and not module.endswith("*"):
        module = module + ".*"

    if not module:
        # Fallback: parse from text
        # Examples:
        #   import kotlin.collections.List     -> module="kotlin.collections.List"
        #   import kotlin.collections.*        -> module="kotlin.collections.*", is_wildcard=True
        #   import kotlin.io.println as print  -> module="kotlin.io.println", alias="print"
        text = text.strip()
        if text.startswith("import "):
            text = text[7:].strip()
        if " as " in text:
            module = text.split(" as ")[0].strip()
        else:
            module = text.rstrip("*").rstrip(".")
            if is_wildcard:
                module = module + ".*"

    if not module:
        return None

    return {
        'module': module,
        'is_wildcard': is_wildcard,
        'alias': alias,
    }


def parse_scala_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Scala file.

    Scala import syntax:
    - import package.Module
    - import package.{A, B, C}  (selective imports)
    - import package._          (wildcard import)
    - import package.Module.{member => alias}  (with rename)

    Args:
        file_path: Path to Scala file

    Returns:
        List of import info dicts with keys: module, is_wildcard, alias
    """
    if not TREE_SITTER_SCALA_AVAILABLE:
        return []

    file_path = Path(file_path)
    if not file_path.exists():
        return []

    try:
        source = file_path.read_bytes()
        parser = _get_scala_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # Scala uses "import_declaration" for import statements
        if node.type == "import_declaration":
            import_infos = _parse_scala_import_node(node, source)
            imports.extend(import_infos)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_scala_import_node(node, source: bytes) -> list[dict]:
    """Parse a single Scala import statement.

    Returns a list because one import statement can have multiple selectors.
    """
    results = []

    # Get the full import text for fallback parsing
    text = source[node.start_byte:node.end_byte].decode("utf-8").strip()

    # Remove "import " prefix
    if text.startswith("import "):
        text = text[7:].strip()

    # Check for selective imports: import scala.util.{Try, Success, Failure}
    if "{" in text:
        # Split into base path and selectors
        base_path = text.split("{")[0].rstrip(".")
        selectors_part = text.split("{")[1].rstrip("}")

        # Parse each selector
        for selector in selectors_part.split(","):
            selector = selector.strip()
            if not selector:
                continue

            # Check for rename: member => alias
            if "=>" in selector:
                parts = selector.split("=>")
                orig = parts[0].strip()
                alias = parts[1].strip()
                if orig != "_":  # Skip hiding imports like {SomeThing => _}
                    full_module = f"{base_path}.{orig}" if base_path else orig
                    results.append({
                        'module': full_module,
                        'is_wildcard': False,
                        'alias': alias if alias != "_" else None,
                    })
            elif selector == "_":
                # Wildcard inside braces: import foo.{_}
                results.append({
                    'module': base_path,
                    'is_wildcard': True,
                    'alias': None,
                })
            else:
                full_module = f"{base_path}.{selector}" if base_path else selector
                results.append({
                    'module': full_module,
                    'is_wildcard': False,
                    'alias': None,
                })
    elif text.endswith("._"):
        # Wildcard import: import scala.collection.mutable._
        base_path = text[:-2]  # Remove ._
        results.append({
            'module': base_path,
            'is_wildcard': True,
            'alias': None,
        })
    else:
        # Simple import: import scala.collection.mutable.ListBuffer
        results.append({
            'module': text,
            'is_wildcard': False,
            'alias': None,
        })

    return results


def parse_c_imports(file_path: str | Path) -> list[dict]:
    """
    Extract #include statements from a C file.

    Args:
        file_path: Path to C file

    Returns:
        List of import info dicts with keys: module, is_system
    """
    if not TREE_SITTER_C_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_c_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "preproc_include":
            import_info = _parse_c_include_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_c_include_node(node, source: bytes) -> dict | None:
    """Parse a single C #include statement."""
    # Get the full include text
    text = source[node.start_byte:node.end_byte].decode("utf-8")

    # Check for system include <...> vs local include "..."
    is_system = "<" in text

    # Extract the module path
    # Examples:
    #   #include <stdio.h>        -> module="stdio.h", is_system=True
    #   #include "utils.h"        -> module="utils.h", is_system=False
    #   #include <sys/types.h>    -> module="sys/types.h", is_system=True

    # Find the string_literal or system_lib_string node for the include path
    module = None
    for child in node.children:
        if child.type == "string_literal":
            # Local include "file.h"
            module_text = source[child.start_byte:child.end_byte].decode("utf-8")
            # Strip quotes
            module = module_text.strip('"')
            is_system = False
            break
        elif child.type == "system_lib_string":
            # System include <file.h>
            module_text = source[child.start_byte:child.end_byte].decode("utf-8")
            # Strip angle brackets
            module = module_text.strip('<>')
            is_system = True
            break

    if not module:
        return None

    return {
        'module': module,
        'is_system': is_system,
    }


def parse_cpp_imports(file_path: str | Path) -> list[dict]:
    """
    Extract #include statements from a C++ file.

    Args:
        file_path: Path to C++ file

    Returns:
        List of import info dicts with keys: module, is_system
    """
    if not TREE_SITTER_CPP_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_cpp_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "preproc_include":
            import_info = _parse_cpp_include_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_cpp_include_node(node, source: bytes) -> dict | None:
    """Parse a single C++ #include statement."""
    # Get the full include text
    text = source[node.start_byte:node.end_byte].decode("utf-8")

    # Check for system include <...> vs local include "..."
    is_system = "<" in text

    # Extract the module path
    module = None
    for child in node.children:
        if child.type == "string_literal":
            # Local include "file.hpp"
            module_text = source[child.start_byte:child.end_byte].decode("utf-8")
            # Strip quotes
            module = module_text.strip('"')
            is_system = False
            break
        elif child.type == "system_lib_string":
            # System include <file.h>
            module_text = source[child.start_byte:child.end_byte].decode("utf-8")
            # Strip angle brackets
            module = module_text.strip('<>')
            is_system = True
            break

    if not module:
        return None

    return {
        'module': module,
        'is_system': is_system,
    }


def parse_ruby_imports(file_path: str | Path) -> list[dict]:
    """
    Extract require statements from a Ruby file.

    Args:
        file_path: Path to Ruby file

    Returns:
        List of import info dicts with keys: module, is_relative
        - require 'json' -> module='json', is_relative=False
        - require_relative 'helper' -> module='helper', is_relative=True
    """
    if not TREE_SITTER_RUBY_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_ruby_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # Ruby imports: require 'module' or require_relative 'module'
        # These are call nodes with method name "require" or "require_relative"
        if node.type == "call":
            import_info = _parse_ruby_require_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_ruby_require_node(node, source: bytes) -> dict | None:
    """Parse a single Ruby require/require_relative statement."""
    # Get the method name
    method_node = node.child_by_field_name("method")
    if not method_node:
        return None

    method_name = source[method_node.start_byte:method_node.end_byte].decode("utf-8")
    if method_name not in ("require", "require_relative"):
        return None

    # Get the arguments
    args_node = node.child_by_field_name("arguments")
    if not args_node:
        return None

    # Find the string argument (first argument)
    module = None
    for child in args_node.children:
        if child.type == "string":
            # Get string content (skip the quotes)
            string_content = child.child_by_field_name("content")
            if string_content:
                module = source[string_content.start_byte:string_content.end_byte].decode("utf-8")
            else:
                # Try to get the text directly and strip quotes
                text = source[child.start_byte:child.end_byte].decode("utf-8")
                # Strip quotes: 'module' or "module"
                module = text.strip("'\"")
            break

    if not module:
        return None

    return {
        'module': module,
        'is_relative': method_name == "require_relative",
    }


def _get_lua_parser():
    """Get or create a tree-sitter Lua parser."""
    if not TREE_SITTER_LUA_AVAILABLE:
        raise RuntimeError("tree-sitter-lua not available")

    lua_lang = tree_sitter.Language(tree_sitter_lua.language())
    parser = tree_sitter.Parser(lua_lang)
    return parser


def parse_lua_imports(file_path: str | Path) -> list[dict]:
    """
    Extract require/dofile/loadfile statements from a Lua file.

    Args:
        file_path: Path to Lua file

    Returns:
        List of import info dicts with keys: module, type
        Types: "require", "dofile", "loadfile"
    """
    if not TREE_SITTER_LUA_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_lua_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # Lua imports are function calls: require("module"), dofile("path"), loadfile("path")
        if node.type == "function_call":
            import_info = _parse_lua_require_node(node, source)
            if import_info:
                imports.append(import_info)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_lua_require_node(node, source: bytes) -> dict | None:
    """Parse a single Lua require/dofile/loadfile call.

    Handles:
    - require("module_name")
    - require "module_name" (parentheses optional for string literal)
    - dofile("path.lua")
    - loadfile("path.lua")
    """
    # Get the function being called
    func_name = None
    arguments = None

    for child in node.children:
        if child.type == "identifier":
            func_name = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "arguments":
            arguments = child
        elif child.type == "string":
            # require "module" syntax (no parentheses)
            arguments = child

    if func_name not in ("require", "dofile", "loadfile"):
        return None

    # Get the module/path argument
    module = None

    if arguments is not None:
        if arguments.type == "string":
            # Direct string (no parentheses case)
            module = _extract_lua_string(arguments, source)
        elif arguments.type == "arguments":
            # Find the first string argument
            for child in arguments.children:
                if child.type == "string":
                    module = _extract_lua_string(child, source)
                    break

    if not module:
        return None

    return {
        'module': module,
        'type': func_name,
    }


def _extract_lua_string(node, source: bytes) -> str | None:
    """Extract string content from a Lua string node."""
    # Lua strings can be:
    # - "double quoted"
    # - 'single quoted'
    # - [[long brackets]]
    text = source[node.start_byte:node.end_byte].decode("utf-8")

    # Strip quotes
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    elif text.startswith("[[") and text.endswith("]]"):
        return text[2:-2]

    return text


# Tree-sitter support for Luau
TREE_SITTER_LUAU_AVAILABLE = False
try:
    import tree_sitter_luau
    TREE_SITTER_LUAU_AVAILABLE = True
except ImportError:
    pass


def _get_luau_parser():
    """Get or create a tree-sitter Luau parser."""
    if not TREE_SITTER_LUAU_AVAILABLE:
        raise RuntimeError("tree-sitter-luau not available")

    luau_lang = tree_sitter.Language(tree_sitter_luau.language())
    parser = tree_sitter.Parser(luau_lang)
    return parser


def parse_luau_imports(file_path: str | Path) -> list[dict]:
    """
    Extract require/GetService statements from a Luau file.

    Args:
        file_path: Path to Luau file

    Returns:
        List of import info dicts with keys: module, type
        Types: "require" (for require calls), "service" (for GetService)
    """
    if not TREE_SITTER_LUAU_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_luau_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # Luau imports are function calls
        if node.type == "function_call":
            import_info = _parse_luau_import_node(node, source)
            if import_info:
                imports.append(import_info)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_luau_import_node(node, source: bytes) -> dict | None:
    """Parse a single Luau require or GetService call.

    Handles:
    - require(script.Utils)
    - require(script.Parent.Module)
    - require("@pkg/json")
    - game:GetService("Players")
    """
    # Check for method call (GetService pattern)
    method_expr = None
    func_name = None
    arguments = None

    for child in node.children:
        if child.type == "method_index_expression":
            method_expr = child
        elif child.type == "identifier":
            func_name = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "arguments":
            arguments = child

    # Handle GetService pattern: game:GetService("ServiceName")
    if method_expr is not None:
        method_name = None
        for child in method_expr.children:
            if child.type == "identifier":
                method_name = source[child.start_byte:child.end_byte].decode("utf-8")

        if method_name == "GetService" and arguments is not None:
            # Extract the service name from arguments
            for arg_child in arguments.children:
                if arg_child.type == "string":
                    service_name = _extract_luau_string(arg_child, source)
                    if service_name:
                        return {
                            'module': service_name,
                            'type': 'service',
                        }
        return None

    # Handle require pattern
    if func_name != "require":
        return None

    if arguments is None:
        return None

    # Get the module argument - can be dot_index_expression or string
    for arg_child in arguments.children:
        if arg_child.type == "dot_index_expression":
            # require(script.Utils) or require(script.Parent.Module)
            module_path = source[arg_child.start_byte:arg_child.end_byte].decode("utf-8")
            return {
                'module': module_path,
                'type': 'require',
            }
        elif arg_child.type == "string":
            # require("@pkg/json")
            module_name = _extract_luau_string(arg_child, source)
            if module_name:
                return {
                    'module': module_name,
                    'type': 'require',
                }
        elif arg_child.type == "identifier":
            # require(ReplicatedStorage.Utils) - first part is identifier
            # Actually this case is for variable reference like require(someVar)
            # We need to handle ReplicatedStorage.Utils which would be dot_index_expression
            module_name = source[arg_child.start_byte:arg_child.end_byte].decode("utf-8")
            return {
                'module': module_name,
                'type': 'require',
            }

    return None


def _extract_luau_string(node, source: bytes) -> str | None:
    """Extract string content from a Luau string node."""
    # Luau strings can have string_content child
    for child in node.children:
        if child.type == "string_content":
            return source[child.start_byte:child.end_byte].decode("utf-8")

    # Fallback: strip quotes manually
    text = source[node.start_byte:node.end_byte].decode("utf-8")
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    elif text.startswith("'") and text.endswith("'"):
        return text[1:-1]

    return text


def parse_elixir_imports(file_path: str | Path) -> dict[str, list[dict]]:
    """
    Extract alias/import/use/require statements from an Elixir file.

    Args:
        file_path: Path to Elixir file

    Returns:
        Dict keyed by defmodule name -> list of import info dicts.
        Each import dict has keys: module, type, as (optional),
        only (optional), except (optional).
        Types: "alias", "import", "use", "require"
    """
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        return {}

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_elixir_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    scoped_imports: dict[str, list[dict]] = {}
    current_module: list[str] = []

    def walk_tree(node):
        # current_module: list[str] stack of FQN names (deepest scope at [-1])
        # Track defmodule scope
        if node.type == "call":
            func_id = None
            for child in node.children:
                if child.type == "identifier":
                    func_id = source[child.start_byte:child.end_byte].decode("utf-8")
                    break

            if func_id == "defmodule":
                mod_name = _extract_elixir_module_name(node, source)
                if mod_name:
                    # Build fully-qualified name from parent context
                    if current_module:
                        fqn = f"{current_module[-1]}.{mod_name}"
                    else:
                        fqn = mod_name
                    current_module.append(fqn)
                    if fqn not in scoped_imports:
                        scoped_imports[fqn] = []
                    for child in node.children:
                        walk_tree(child)
                    current_module.pop()
                else:
                    # mod_name is None — still recurse into children so nested
                    # imports are not missed, but don't fall through to
                    # _parse_elixir_import_node which would misinterpret the
                    # defmodule call as an import statement.
                    for child in node.children:
                        walk_tree(child)
                return

            # Elixir imports are call nodes with specific identifiers
            import_info = _parse_elixir_import_node(node, source)
            if import_info and current_module:
                scoped_imports[current_module[-1]].append(import_info)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return scoped_imports


def _get_elixir_parser():
    """Get or create an Elixir tree-sitter parser."""
    from tree_sitter import Language, Parser
    parser = Parser()
    parser.language = Language(tree_sitter_elixir.language())
    return parser


def _extract_elixir_module_name(call_node, source: bytes) -> str | None:
    """Extract the module name from an Elixir defmodule call node.

    Walks the call node's arguments to find an alias node (e.g. MyApp.Web)
    and returns its text.
    """
    for child in call_node.children:
        if child.type == "arguments":
            for arg_child in child.children:
                if arg_child.is_named and arg_child.type == "alias":
                    return source[arg_child.start_byte:arg_child.end_byte].decode("utf-8")
    return None


def _extract_elixir_func_name(call_node, source: bytes) -> str | None:
    """Extract the function name from an Elixir def/defp call node.

    Handles:
      def func_name(args) do ... end  ->  arguments > call > identifier
      def func_name do ... end        ->  arguments > identifier
      def func_name \\\\ default      ->  arguments > binary_operator > identifier/call
    """
    for child in call_node.children:
        if child.type == "arguments":
            for arg_child in child.children:
                if arg_child.type == "call":
                    for cc in arg_child.children:
                        if cc.type == "identifier":
                            return source[cc.start_byte:cc.end_byte].decode("utf-8")
                elif arg_child.type == "identifier":
                    return source[arg_child.start_byte:arg_child.end_byte].decode("utf-8")
                elif arg_child.type == "binary_operator":
                    for cc in arg_child.children:
                        if cc.type == "identifier":
                            return source[cc.start_byte:cc.end_byte].decode("utf-8")
                        elif cc.type == "call":
                            for ccc in cc.children:
                                if ccc.type == "identifier":
                                    return source[ccc.start_byte:ccc.end_byte].decode("utf-8")
    return None


def _parse_elixir_import_node(node, source: bytes) -> dict | None:
    """Parse a single Elixir import call.

    Handles:
    - alias Module.Name
    - alias Module.Name, as: Alias
    - import Module
    - import Module, only: [...]
    - use Module
    - use Module, opts
    - require Module
    """
    # Get the function being called
    func_name = None
    arguments = None

    for child in node.children:
        if child.type == "identifier":
            func_name = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "arguments":
            arguments = child

    if func_name not in ("alias", "import", "use", "require"):
        return None

    if arguments is None:
        return None

    # Get the module argument (first argument)
    module = None
    alias_name = None
    filter_lists = {}

    for child in arguments.children:
        if child.is_named:
            if child.type == "alias":
                # Module reference like Phoenix.Controller
                module = source[child.start_byte:child.end_byte].decode("utf-8")
            elif child.type == "dot":
                # Qualified module name
                module = source[child.start_byte:child.end_byte].decode("utf-8")
            elif child.type == "keywords":
                # Keyword arguments like "as: AliasName", "only: [...]", "except: [...]"
                for kw_child in child.children:
                    if kw_child.type == "pair":
                        key = None
                        value = None
                        list_items = None
                        for pair_child in kw_child.children:
                            if pair_child.type == "keyword":
                                key = source[pair_child.start_byte:pair_child.end_byte].decode("utf-8").rstrip(": ")
                            elif pair_child.type == "alias":
                                value = source[pair_child.start_byte:pair_child.end_byte].decode("utf-8")
                            elif pair_child.type == "list":
                                # Parse [func: arity, ...] list for only/except
                                list_items = []
                                for list_child in pair_child.children:
                                    if list_child.type == "keywords":
                                        for kw in list_child.children:
                                            if kw.type == "pair":
                                                fname = None
                                                arity = None
                                                for pc in kw.children:
                                                    if pc.type == "keyword":
                                                        fname = source[pc.start_byte:pc.end_byte].decode("utf-8").rstrip(": ")
                                                    elif pc.type == "integer":
                                                        arity = int(source[pc.start_byte:pc.end_byte].decode("utf-8"))
                                                if fname is not None and arity is not None:
                                                    list_items.append((fname, arity))
                        if key == "as" and value:
                            alias_name = value
                        elif key in ("only", "except") and list_items is not None:
                            filter_lists[key] = list_items

    if not module:
        return None

    result = {
        'module': module,
        'type': func_name,
    }
    if alias_name:
        result['as'] = alias_name
    if 'only' in filter_lists:
        result['only'] = filter_lists['only']
    if 'except' in filter_lists:
        result['except'] = filter_lists['except']

    return result


def parse_php_imports(file_path: str | Path) -> list[dict]:
    """
    Extract use/require/include statements from a PHP file.

    Args:
        file_path: Path to PHP file

    Returns:
        List of import info dicts with keys: module, type
        Types: "use", "require", "require_once", "include", "include_once"
    """
    if not TREE_SITTER_PHP_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_php_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        # use statements: use App\Models\User;
        if node.type == "namespace_use_declaration":
            _parse_php_use_node(node, source, imports)
        # require/include statements
        elif node.type in ("include_expression", "include_once_expression",
                           "require_expression", "require_once_expression"):
            import_info = _parse_php_require_include_node(node, source)
            if import_info:
                imports.append(import_info)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_php_use_node(node, source: bytes, imports: list):
    """Parse PHP use declaration(s).

    Handles:
    - Simple: use App\\Models\\User;
    - Grouped: use App\\Models\\{User, Post};
    - Aliased: use App\\Models\\User as UserModel;
    - Function/const: use function array_map;
    """
    # Check if this has a namespace_use_group (grouped imports)
    has_group = any(child.type == "namespace_use_group" for child in node.children)

    if has_group:
        # Grouped imports: use App\Models\{User, Post}
        # Get the prefix from the namespace_name
        prefix = ""
        for child in node.children:
            if child.type == "namespace_name":
                prefix = source[child.start_byte:child.end_byte].decode("utf-8")
                break

        # Parse each group item
        for child in node.children:
            if child.type == "namespace_use_group":
                for group_child in child.children:
                    # In tree-sitter-php, grouped items are namespace_use_clause
                    if group_child.type == "namespace_use_clause":
                        clause_text = source[group_child.start_byte:group_child.end_byte].decode("utf-8").strip()
                        # Handle alias: User as UserModel
                        parts = clause_text.split(" as ")
                        name = parts[0].strip()
                        alias = parts[1].strip() if len(parts) > 1 else None
                        full_module = f"{prefix}\\{name}" if prefix else name
                        import_info = {
                            'module': full_module,
                            'type': 'use',
                        }
                        if alias:
                            import_info['alias'] = alias
                        imports.append(import_info)
    else:
        # Simple imports: use App\Models\User;
        for child in node.children:
            if child.type == "namespace_use_clause":
                clause_text = source[child.start_byte:child.end_byte].decode("utf-8").strip()
                # Handle alias: User as UserModel
                parts = clause_text.split(" as ")
                module = parts[0].strip()
                alias = parts[1].strip() if len(parts) > 1 else None
                import_info = {
                    'module': module,
                    'type': 'use',
                }
                if alias:
                    import_info['alias'] = alias
                imports.append(import_info)


def _parse_php_require_include_node(node, source: bytes) -> dict | None:
    """Parse PHP require/include expression."""
    node_type = node.type

    # Map node type to import type
    type_map = {
        "include_expression": "include",
        "include_once_expression": "include_once",
        "require_expression": "require",
        "require_once_expression": "require_once",
    }
    import_type = type_map.get(node_type, "require")

    # Find the string literal or expression being included
    module = None
    for child in node.children:
        if child.type in ("string", "encapsed_string"):
            module_text = source[child.start_byte:child.end_byte].decode("utf-8")
            # Strip quotes
            module = module_text.strip("'\"")
            break
        elif child.type == "binary_expression":
            # Handle expressions like __DIR__ . '/file.php'
            # Just get the full text for now
            module = source[child.start_byte:child.end_byte].decode("utf-8")
            break

    if not module:
        # Try to get full text after the keyword
        text = source[node.start_byte:node.end_byte].decode("utf-8")
        # Extract path from require 'path' or require('path')
        for pattern in ["require_once", "require", "include_once", "include"]:
            if text.startswith(pattern):
                rest = text[len(pattern):].strip()
                # Remove parentheses and quotes
                rest = rest.strip("();'\" ")
                if rest:
                    module = rest
                break

    if not module:
        return None

    return {
        'module': module,
        'type': import_type,
    }


def parse_swift_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Swift file.

    Args:
        file_path: Path to Swift file

    Returns:
        List of import info dicts with keys: module, kind
        - import Foundation -> module='Foundation', kind=None
        - import struct Foundation.Date -> module='Foundation.Date', kind='struct'
    """
    if not TREE_SITTER_SWIFT_AVAILABLE:
        _warn_swift_unavailable_once()
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_swift_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "import_declaration":
            import_info = _parse_swift_import_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_swift_import_node(node, source: bytes) -> dict | None:
    """Parse a single Swift import statement.

    Swift imports can be:
    - import Foundation
    - import struct Foundation.Date
    - import func Foundation.strcmp
    - import class UIKit.UIView
    - @testable import MyApp
    """
    # Get the full import text
    text = source[node.start_byte:node.end_byte].decode("utf-8").strip()

    # Handle @testable or other attribute imports
    # Remove leading @attribute if present
    if text.startswith("@"):
        # Find the import keyword
        import_idx = text.find("import")
        if import_idx == -1:
            return None
        text = text[import_idx:]

    if not text.startswith("import"):
        return None

    # Remove 'import ' prefix
    rest = text[6:].strip()

    # Check for kind specifier (struct, class, func, enum, etc.)
    kind = None
    kind_specifiers = ["struct", "class", "enum", "protocol", "func", "var", "let", "typealias"]
    for spec in kind_specifiers:
        if rest.startswith(spec + " "):
            kind = spec
            rest = rest[len(spec):].strip()
            break

    # The rest is the module path
    module = rest

    if not module:
        return None

    return {
        'module': module,
        'kind': kind,
    }


def parse_csharp_imports(file_path: str | Path) -> list[dict]:
    """
    Extract using statements from a C# file.

    Args:
        file_path: Path to C# file

    Returns:
        List of import info dicts with keys: module, is_static, alias
        - using System; -> module='System'
        - using static System.Math; -> module='System.Math', is_static=True
        - using Alias = System.Collections; -> module='System.Collections', alias='Alias'
        - global using System; -> module='System', is_global=True
    """
    if not TREE_SITTER_CSHARP_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_csharp_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "using_directive":
            import_info = _parse_csharp_using_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return imports


def _parse_csharp_using_node(node, source: bytes) -> dict | None:
    """Parse a single C# using statement.

    C# using directives can be:
    - using System;
    - using static System.Math;
    - using Alias = System.Collections;
    - global using System;
    """
    # Get the full using text
    text = source[node.start_byte:node.end_byte].decode("utf-8").strip()

    result = {
        'module': None,
        'is_static': False,
        'is_global': False,
        'alias': None,
    }

    # Check for global using
    if text.startswith("global"):
        result['is_global'] = True
        text = text[6:].strip()

    # Check for using static
    if "static" in text.split():
        result['is_static'] = True

    # Look for the qualified name in children
    for child in node.children:
        if child.type == "qualified_name":
            result['module'] = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "identifier":
            # Check if this is an alias (using Alias = ...)
            # or just a simple namespace
            next_sibling = None
            for i, c in enumerate(node.children):
                if c == child and i + 1 < len(node.children):
                    next_sibling = node.children[i + 1]
                    break
            if next_sibling and next_sibling.type == "=":
                result['alias'] = source[child.start_byte:child.end_byte].decode("utf-8")
            elif not result['module']:
                # Simple identifier without qualified name
                result['module'] = source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "name_equals":
            # This handles: using Alias = Something
            alias_node = child.child_by_field_name("name")
            if alias_node:
                result['alias'] = source[alias_node.start_byte:alias_node.end_byte].decode("utf-8")

    if not result['module']:
        return None

    return result


def build_function_index(
    root: str | Path,
    language: str = "python",
    workspace_config: Optional[WorkspaceConfig] = None
) -> dict[tuple[str, str], str]:
    """
    Build an index mapping (module_name, function_name) to file paths.

    Args:
        root: Project root directory
        language: "python" or "typescript"
        workspace_config: Optional WorkspaceConfig for monorepo scoping

    Returns:
        Dict mapping (module, func_name) tuples to relative file paths
    """
    root = Path(root).resolve()
    index = {}

    for src_file in scan_project(root, language, workspace_config):
        src_path = Path(src_file)
        rel_path = src_path.relative_to(root)

        # Derive module name from file path
        # e.g., pkg/core.py -> pkg.core, utils.ts -> utils
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_name = '/'.join(module_parts) if language in ("typescript", "javascript") else '.'.join(module_parts)

        # Also track the simple module name (last component)
        simple_module = rel_path.stem

        if language == "python":
            _index_python_file(src_path, rel_path, module_name, simple_module, index)
        elif language in ("typescript", "javascript"):
            _index_typescript_file(src_path, rel_path, module_name, simple_module, index, language=language)
        elif language == "go":
            _index_go_file(src_path, rel_path, module_name, simple_module, index)
        elif language == "rust":
            _index_rust_file(src_path, rel_path, module_name, simple_module, index)
        elif language == "java":
            _index_java_file(src_path, rel_path, module_name, simple_module, index)
        elif language == "c":
            _index_c_file(src_path, rel_path, module_name, simple_module, index)
        elif language == "php":
            _index_php_file(src_path, rel_path, module_name, simple_module, index)
        elif language == "elixir":
            _index_elixir_file(src_path, rel_path, module_name, simple_module, index)
        # Swift is handled via the self-contained builder path
        # (see ``self_contained_languages`` set in build_project_call_graph),
        # so build_function_index is never invoked for it. The previous
        # ``elif language == "swift": _index_swift_file(...)`` branch called
        # a function that was never defined — a latent NameError if anyone
        # removed Swift from self_contained_languages. Branch deleted.

    return index


def _index_python_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions and classes from a Python file."""
    try:
        source = src_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Map both full and simple module names
            index[(module_name, node.name)] = str(rel_path)
            index[(simple_module, node.name)] = str(rel_path)
            # Also index with string key for convenience
            index[f"{module_name}.{node.name}"] = str(rel_path)
            index[f"{simple_module}.{node.name}"] = str(rel_path)
        elif isinstance(node, ast.ClassDef):
            # Track class definitions too (for instantiation calls)
            index[(module_name, node.name)] = str(rel_path)
            index[(simple_module, node.name)] = str(rel_path)
            index[f"{module_name}.{node.name}"] = str(rel_path)
            index[f"{simple_module}.{node.name}"] = str(rel_path)


def _index_typescript_file(
    src_path: Path,
    rel_path: Path,
    module_name: str,
    simple_module: str,
    index: dict,
    language: str = "typescript",
):
    """Index functions and classes from a TypeScript/JavaScript file."""
    if not TREE_SITTER_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_ts_parser(language)
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}/{name}"] = str(rel_path)
        index[f"{simple_module}/{name}"] = str(rel_path)

    def walk_tree(node):
        # Handle export statements - look inside them
        if node.type == "export_statement":
            for child in node.children:
                walk_tree(child)
            return

        # Function declarations
        if node.type in ("function_declaration", "method_definition"):
            name = _get_ts_node_name(node, source)
            if name:
                add_to_index(name)

        # Arrow functions assigned to variables: const foo = () => {}
        elif node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    name = None
                    has_arrow = False
                    for vc in child.children:
                        if vc.type == "identifier":
                            name = source[vc.start_byte:vc.end_byte].decode("utf-8")
                        elif vc.type == "arrow_function":
                            has_arrow = True
                    if name and has_arrow:
                        add_to_index(name)

        # Class declarations
        elif node.type == "class_declaration":
            name = _get_ts_node_name(node, source)
            if name:
                add_to_index(name)

        # CommonJS exports:
        #   exports.foo = function() {}
        #   module.exports.foo = helper
        #   module.exports = { foo, bar: baz }
        elif node.type == "assignment_expression":
            for export_name in _get_commonjs_export_names(node, source):
                add_to_index(export_name)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_ts_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a TypeScript AST node."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


def _get_commonjs_export_names(assign_node, source: bytes) -> list[str]:
    """Extract exported symbol names from CommonJS assignment expressions."""
    if assign_node.type != "assignment_expression" or len(assign_node.children) < 3:
        return []

    lhs = assign_node.children[0]
    rhs = assign_node.children[-1]

    lhs_text = source[lhs.start_byte:lhs.end_byte].decode("utf-8").replace(" ", "")
    function_like_rhs = {
        "function_expression",
        "arrow_function",
        "identifier",
        "member_expression",
        "class",
        "class_declaration",
    }

    if lhs_text.startswith("exports.") and rhs.type in function_like_rhs:
        export_name = lhs_text.split("exports.", 1)[1]
        return [export_name] if export_name.isidentifier() else []

    if lhs_text.startswith("module.exports.") and rhs.type in function_like_rhs:
        export_name = lhs_text.split("module.exports.", 1)[1]
        return [export_name] if export_name.isidentifier() else []

    if lhs_text == "module.exports" and rhs.type in function_like_rhs:
        export_names = ["default"]
        rhs_name = _get_commonjs_rhs_name(rhs, source)
        if rhs_name and rhs_name not in export_names:
            export_names.append(rhs_name)
        return export_names

    if lhs_text == "module.exports" and rhs.type == "object":
        return _extract_commonjs_object_export_names(rhs, source)

    return []


def _get_commonjs_rhs_name(rhs_node, source: bytes) -> str | None:
    """Extract a stable symbol name from a CommonJS RHS expression when possible."""
    if rhs_node.type == "identifier":
        name = source[rhs_node.start_byte:rhs_node.end_byte].decode("utf-8")
        return name if name.isidentifier() else None

    if rhs_node.type in {"function_expression", "class", "class_declaration"}:
        name = _get_ts_node_name(rhs_node, source)
        return name if name and name.isidentifier() else None

    return None


def _extract_commonjs_object_export_names(object_node, source: bytes) -> list[str]:
    """Extract export names from `module.exports = { ... }` object literals."""
    export_names = []

    for child in object_node.children:
        if child.type == "shorthand_property_identifier":
            name = source[child.start_byte:child.end_byte].decode("utf-8")
            if name.isidentifier():
                export_names.append(name)
        elif child.type == "pair":
            key_name = None
            for pair_child in child.children:
                if pair_child.type in ("identifier", "property_identifier"):
                    key_name = source[pair_child.start_byte:pair_child.end_byte].decode("utf-8")
                    break
                if pair_child.type == "string":
                    key_name = source[pair_child.start_byte:pair_child.end_byte].decode("utf-8").strip("'\"")
                    break
            if key_name and key_name.isidentifier():
                export_names.append(key_name)

    return export_names


def _extract_commonjs_file_exports(file_path: Path, language: str = "javascript") -> set[str]:
    """Extract CommonJS exported names for a file."""
    if not TREE_SITTER_AVAILABLE:
        return set()

    try:
        source = file_path.read_bytes()
        parser = _get_ts_parser(language)
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return set()

    exports = set()

    def walk_tree(node):
        if node.type == "assignment_expression":
            exports.update(_get_commonjs_export_names(node, source))
        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)
    return exports


def _index_go_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions, types, and methods from a Go file."""
    if not TREE_SITTER_GO_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_go_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}/{name}"] = str(rel_path)
        index[f"{simple_module}/{name}"] = str(rel_path)

    def walk_tree(node):
        # Function declarations
        if node.type == "function_declaration":
            name = _get_go_node_name(node, source)
            if name:
                add_to_index(name)

        # Method declarations (function with receiver)
        elif node.type == "method_declaration":
            name = _get_go_node_name(node, source)
            if name:
                add_to_index(name)
                # Also try to get the receiver type for full name
                receiver_type = _get_go_receiver_type(node, source)
                if receiver_type:
                    add_to_index(f"{receiver_type}.{name}")

        # Type declarations (struct, interface)
        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name = _get_go_node_name(child, source)
                    if name:
                        add_to_index(name)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_go_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a Go AST node."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


def _get_go_receiver_type(node, source: bytes) -> str | None:
    """Get the receiver type from a Go method declaration."""
    for child in node.children:
        if child.type == "parameter_list":
            # First parameter list is the receiver
            for param in child.children:
                if param.type == "parameter_declaration":
                    for pc in param.children:
                        if pc.type == "pointer_type":
                            for pt in pc.children:
                                if pt.type == "type_identifier":
                                    return source[pt.start_byte:pt.end_byte].decode("utf-8")
                        elif pc.type == "type_identifier":
                            return source[pc.start_byte:pc.end_byte].decode("utf-8")
            break
    return None


def _index_rust_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions, structs, and impl blocks from a Rust file."""
    if not TREE_SITTER_RUST_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_rust_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}.{name}"] = str(rel_path)
        index[f"{simple_module}.{name}"] = str(rel_path)

    def walk_tree(node):
        # Function definitions
        if node.type == "function_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Struct definitions
        elif node.type == "struct_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Enum definitions
        elif node.type == "enum_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Trait definitions
        elif node.type == "trait_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Impl blocks - index methods
        elif node.type == "impl_item":
            type_name = None
            for child in node.children:
                if child.type == "type_identifier":
                    type_name = source[child.start_byte:child.end_byte].decode("utf-8")
                    break
            # Index methods within impl block
            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        if item.type == "function_item":
                            method_name = _get_rust_node_name(item, source)
                            if method_name:
                                # Index as both bare name and Type::method
                                add_to_index(method_name)
                                if type_name:
                                    add_to_index(f"{type_name}::{method_name}")

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_rust_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a Rust AST node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8")
        elif child.type == "type_identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


def _index_java_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index methods and classes from a Java file."""
    if not TREE_SITTER_JAVA_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_java_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}.{name}"] = str(rel_path)
        index[f"{simple_module}.{name}"] = str(rel_path)

    current_class = None

    def walk_tree(node):
        nonlocal current_class

        # Class declarations
        if node.type == "class_declaration":
            class_name = _get_java_node_name(node, source)
            if class_name:
                add_to_index(class_name)
                old_class = current_class
                current_class = class_name
                # Process class body
                for child in node.children:
                    walk_tree(child)
                current_class = old_class
                return  # Already processed children

        # Interface declarations
        elif node.type == "interface_declaration":
            interface_name = _get_java_node_name(node, source)
            if interface_name:
                add_to_index(interface_name)

        # Method declarations
        elif node.type == "method_declaration":
            name = _get_java_node_name(node, source)
            if name:
                add_to_index(name)
                # Also index as Class.method if we have a class context
                if current_class:
                    add_to_index(f"{current_class}.{name}")

        # Constructor declarations
        elif node.type == "constructor_declaration":
            name = _get_java_node_name(node, source)
            if name:
                add_to_index(name)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_java_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a Java AST node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


def _index_c_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions from a C file."""
    if not TREE_SITTER_C_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_c_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}.{name}"] = str(rel_path)
        index[f"{simple_module}.{name}"] = str(rel_path)

    def walk_tree(node):
        # Function definitions
        if node.type == "function_definition":
            name = _get_c_node_name(node, source)
            if name:
                add_to_index(name)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_c_node_name(node, source: bytes) -> str | None:
    """Get the function name from a C function_definition node."""
    for child in node.children:
        if child.type == "function_declarator":
            for dc in child.children:
                if dc.type == "identifier":
                    return source[dc.start_byte:dc.end_byte].decode("utf-8")
        elif child.type == "pointer_declarator":
            # Pointer return type like int* func()
            for pc in child.children:
                if pc.type == "function_declarator":
                    for dc in pc.children:
                        if dc.type == "identifier":
                            return source[dc.start_byte:dc.end_byte].decode("utf-8")
    return None


def _index_php_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions, classes, and methods from a PHP file."""
    if not TREE_SITTER_PHP_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_php_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}\\{name}"] = str(rel_path)
        index[f"{simple_module}\\{name}"] = str(rel_path)

    current_class = None
    namespace = None

    def walk_tree(node):
        nonlocal current_class, namespace

        # Namespace declaration
        if node.type == "namespace_definition":
            for child in node.children:
                if child.type == "namespace_name":
                    namespace = source[child.start_byte:child.end_byte].decode("utf-8")
                    break
            # Continue processing children
            for child in node.children:
                walk_tree(child)
            return

        # Class declarations
        if node.type == "class_declaration":
            class_name = _get_php_node_name(node, source)
            if class_name:
                add_to_index(class_name)
                if namespace:
                    # Also index with full namespace
                    full_name = f"{namespace}\\{class_name}"
                    index[(namespace, class_name)] = str(rel_path)
                    index[full_name] = str(rel_path)
                old_class = current_class
                current_class = class_name
                # Process class body
                for child in node.children:
                    walk_tree(child)
                current_class = old_class
                return  # Already processed children

        # Interface declarations
        elif node.type == "interface_declaration":
            interface_name = _get_php_node_name(node, source)
            if interface_name:
                add_to_index(interface_name)

        # Trait declarations
        elif node.type == "trait_declaration":
            trait_name = _get_php_node_name(node, source)
            if trait_name:
                add_to_index(trait_name)

        # Method declarations
        elif node.type == "method_declaration":
            name = _get_php_node_name(node, source)
            if name:
                add_to_index(name)
                # Also index as Class.method if we have a class context
                if current_class:
                    add_to_index(f"{current_class}::{name}")
                    index[(current_class, name)] = str(rel_path)

        # Function definitions (top-level)
        elif node.type == "function_definition":
            name = _get_php_node_name(node, source)
            if name:
                add_to_index(name)

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_php_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a PHP AST node."""
    for child in node.children:
        if child.type == "name":
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


def _get_php_class_context(node, source: bytes) -> str | None:
    """Get parent class name from PHP method declaration by walking up the tree."""
    parent = node.parent
    while parent:
        if parent.type == "class_declaration":
            return _get_php_node_name(parent, source)
        parent = parent.parent
    return None


def _index_elixir_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions and modules from an Elixir file.

    Walks the AST to find defmodule declarations and def/defp function definitions.
    Indexes each function under both the file-path-derived module name and the
    Elixir module name (from defmodule), so callers can resolve by either.
    """
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_elixir_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(func_name: str, elixir_module: str | None):
        """Add a function to the index under multiple keys."""
        index[(module_name, func_name)] = str(rel_path)
        index[(simple_module, func_name)] = str(rel_path)
        index[f"{module_name}.{func_name}"] = str(rel_path)
        if elixir_module:
            index[(elixir_module, func_name)] = str(rel_path)
            index[f"{elixir_module}.{func_name}"] = str(rel_path)
            # Also index with the last segment of the module name
            last_segment = elixir_module.rsplit(".", 1)[-1]
            if last_segment != elixir_module:
                index[(last_segment, func_name)] = str(rel_path)

    current_module = None

    def walk_tree(node):
        # current_module: str scalar tracking current FQN
        nonlocal current_module

        if node.type == "call":
            # Check if this is a defmodule, def, or defp
            func_id = None
            for child in node.children:
                if child.type == "identifier":
                    func_id = source[child.start_byte:child.end_byte].decode("utf-8")
                    break

            if func_id == "defmodule":
                elixir_mod = _extract_elixir_module_name(node, source)
                if elixir_mod:
                    # Build fully-qualified name from parent context
                    if current_module:
                        fqn = f"{current_module}.{elixir_mod}"
                    else:
                        fqn = elixir_mod
                    # Index the module itself
                    index[(module_name, fqn)] = str(rel_path)
                    index[(simple_module, fqn)] = str(rel_path)

                    old_module = current_module
                    current_module = fqn
                    # Process the do_block children
                    for child in node.children:
                        if child.type == "do_block":
                            for do_child in child.children:
                                walk_tree(do_child)
                    current_module = old_module
                    return

            elif func_id == "def":
                func_name = _extract_elixir_func_name(node, source)
                if func_name:
                    add_to_index(func_name, current_module)
                return  # Don't recurse into function bodies for indexing
            elif func_id == "defp":
                return  # Private functions — skip cross-file index

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


class CallVisitor(ast.NodeVisitor):
    """AST visitor that extracts function calls and references from a function body."""

    def __init__(self, defined_funcs: set[str] | None = None):
        self.calls: list[str] = []
        self.attr_calls: list[tuple[str, str]] = []  # (obj, method) pairs
        self.refs: list[str] = []  # Function references (higher-order usage)
        self._defined_funcs = defined_funcs or set()
        self._in_call = False  # Track if we're inside a Call node

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # Direct call: func()
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Attribute call: obj.method() or module.func()
            if isinstance(node.func.value, ast.Name):
                self.attr_calls.append((node.func.value.id, node.func.attr))

        # Visit arguments - function references passed as args
        self._in_call = True
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
        self._in_call = False

        # Don't call generic_visit - we handled children manually

    def visit_Name(self, node: ast.Name):
        # Track function references (not calls) when used as values
        # Only track if it matches a known function name
        if node.id in self._defined_funcs and node.id not in self.calls:
            self.refs.append(node.id)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        # Track function references in dict values: {"key": func}
        for value in node.values:
            if isinstance(value, ast.Name) and value.id in self._defined_funcs:
                if value.id not in self.refs:
                    self.refs.append(value.id)
        self.generic_visit(node)

    def visit_List(self, node: ast.List):
        # Track function references in lists: [func1, func2]
        for elt in node.elts:
            if isinstance(elt, ast.Name) and elt.id in self._defined_funcs:
                if elt.id not in self.refs:
                    self.refs.append(elt.id)
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):
        # Track function references in tuples: (func1, func2)
        for elt in node.elts:
            if isinstance(elt, ast.Name) and elt.id in self._defined_funcs:
                if elt.id not in self.refs:
                    self.refs.append(elt.id)
        self.generic_visit(node)


def _extract_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return {}

    calls_by_func = {}

    # Collect all function names defined in this file (for intra-file calls)
    defined_funcs = set()
    defined_classes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined_classes.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visitor = CallVisitor(defined_funcs=defined_funcs)
            visitor.visit(node)

            calls = []
            for call in visitor.calls:
                if call in defined_funcs or call in defined_classes:
                    calls.append(('intra', call))
                else:
                    calls.append(('direct', call))

            for obj, method in visitor.attr_calls:
                calls.append(('attr', f"{obj}.{method}"))

            # Add function references (higher-order usage)
            for ref in visitor.refs:
                if ref in defined_funcs:
                    calls.append(('ref', ref))

            calls_by_func[node.name] = calls

    # Also scan module-level code for function calls and references
    # This catches: COMMANDS = {"key": func}, if __name__ == "__main__", etc.
    module_calls = []
    for node in tree.body:
        # Skip function/class definitions - we handle those above
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Visit module-level statements for function references and calls
        visitor = CallVisitor(defined_funcs=defined_funcs)
        visitor.visit(node)

        # Add intra-file references
        for ref in visitor.refs:
            if ref in defined_funcs:
                module_calls.append(('ref', ref))

        # Add ALL calls (both intra-file and external imports)
        for call in visitor.calls:
            if call in defined_funcs:
                module_calls.append(('intra', call))
            else:
                module_calls.append(('direct', call))  # Could be imported function

    # Add module-level calls from a synthetic "<module>" function
    if module_calls:
        calls_by_func['<module>'] = module_calls

    return calls_by_func


def _extract_ts_file_calls(
    file_path: Path,
    root: Path,
    language: str = "typescript",
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a TypeScript file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    if not TREE_SITTER_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_ts_parser(language)
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function/class names
    def collect_definitions(node):
        if node.type in ("function_declaration", "class_declaration"):
            name = _get_ts_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    var_name = None
                    has_local_callable_initializer = False
                    for vc in child.children:
                        if vc.type == "identifier" and var_name is None:
                            var_name = source[vc.start_byte:vc.end_byte].decode("utf-8")
                        elif vc.type in ("arrow_function", "function_expression", "class", "class_declaration"):
                            has_local_callable_initializer = True
                    if var_name and has_local_callable_initializer:
                        defined_names.add(var_name)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the callee
                for child in node.children:
                    if child.type == "identifier":
                        callee = source[child.start_byte:child.end_byte].decode("utf-8")
                        if callee in defined_names:
                            calls.append(('intra', callee))
                        else:
                            calls.append(('direct', callee))
                        break
                    elif child.type == "member_expression":
                        # obj.method() call
                        obj_name = None
                        obj_is_this = False
                        method_name = None
                        for mc in child.children:
                            if mc.type == "this":
                                obj_is_this = True
                            elif mc.type == "identifier" and obj_name is None:
                                obj_name = source[mc.start_byte:mc.end_byte].decode("utf-8")
                            elif mc.type == "property_identifier":
                                method_name = source[mc.start_byte:mc.end_byte].decode("utf-8")

                        if obj_is_this and method_name:
                            # this.method() - treat as intra-file call to the method
                            calls.append(('intra', method_name))
                        elif obj_name and method_name:
                            calls.append(('attr', f"{obj_name}.{method_name}"))
                        break

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    def process_functions(node):
        # Handle export statements - look inside them
        if node.type == "export_statement":
            for child in node.children:
                process_functions(child)
            return

        if node.type == "function_declaration":
            name = _get_ts_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        elif node.type == "lexical_declaration":
            # Handle arrow functions: const foo = () => {}
            for child in node.children:
                if child.type == "variable_declarator":
                    name = None
                    arrow_node = None
                    for vc in child.children:
                        if vc.type == "identifier":
                            name = source[vc.start_byte:vc.end_byte].decode("utf-8")
                        elif vc.type == "arrow_function":
                            arrow_node = vc
                    if name and arrow_node:
                        calls_by_func[name] = extract_calls_from_func(arrow_node, name)

        elif node.type == "class_declaration":
            class_name = _get_ts_node_name(node, source)
            if class_name:
                # Process methods
                for child in node.children:
                    if child.type == "class_body":
                        for body_child in child.children:
                            if body_child.type == "method_definition":
                                method_name = _get_ts_node_name(body_child, source)
                                if method_name:
                                    full_name = f"{class_name}.{method_name}"
                                    calls_by_func[full_name] = extract_calls_from_func(body_child, full_name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_go_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a Go file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    if not TREE_SITTER_GO_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_go_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function/type names
    def collect_definitions(node):
        if node.type == "function_declaration":
            name = _get_go_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "method_declaration":
            name = _get_go_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name = _get_go_node_name(child, source)
                    if name:
                        defined_names.add(name)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the callee - first child is the function being called
                func_child = node.children[0] if node.children else None
                if func_child:
                    if func_child.type == "identifier":
                        callee = source[func_child.start_byte:func_child.end_byte].decode("utf-8")
                        if callee in defined_names:
                            calls.append(('intra', callee))
                        else:
                            calls.append(('direct', callee))
                    elif func_child.type == "selector_expression":
                        # pkg.Func() or obj.Method() call
                        parts = []
                        for sc in func_child.children:
                            if sc.type == "identifier":
                                parts.append(source[sc.start_byte:sc.end_byte].decode("utf-8"))
                            elif sc.type == "field_identifier":
                                parts.append(source[sc.start_byte:sc.end_byte].decode("utf-8"))
                        if len(parts) >= 2:
                            obj, method = parts[0], parts[-1]
                            # Check if method is defined locally
                            if method in defined_names:
                                calls.append(('intra', method))
                            else:
                                calls.append(('attr', f"{obj}.{method}"))

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    def process_functions(node):
        if node.type == "function_declaration":
            name = _get_go_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        elif node.type == "method_declaration":
            name = _get_go_node_name(node, source)
            receiver_type = _get_go_receiver_type(node, source)
            if name:
                full_name = f"{receiver_type}.{name}" if receiver_type else name
                calls_by_func[full_name] = extract_calls_from_func(node, full_name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_rust_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a Rust file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    if not TREE_SITTER_RUST_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_rust_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function/struct names
    def collect_definitions(node):
        if node.type == "function_item":
            name = _get_rust_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type in ("struct_item", "enum_item", "trait_item"):
            name = _get_rust_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "impl_item":
            # Collect method names from impl blocks
            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        if item.type == "function_item":
                            name = _get_rust_node_name(item, source)
                            if name:
                                defined_names.add(name)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the callee
                for child in node.children:
                    if child.type == "identifier":
                        callee = source[child.start_byte:child.end_byte].decode("utf-8")
                        if callee in defined_names:
                            calls.append(('intra', callee))
                        else:
                            calls.append(('direct', callee))
                        break
                    elif child.type == "scoped_identifier":
                        # Path call: module::func() or Type::method()
                        text = source[child.start_byte:child.end_byte].decode("utf-8")
                        # Get the last segment as the function name
                        if "::" in text:
                            parts = text.rsplit("::", 1)
                            func = parts[1]
                            if func in defined_names:
                                calls.append(('intra', func))
                            else:
                                calls.append(('attr', text))
                        break
                    elif child.type == "field_expression":
                        # Method call: obj.method()
                        method_name = None
                        for fc in child.children:
                            if fc.type == "field_identifier":
                                method_name = source[fc.start_byte:fc.end_byte].decode("utf-8")
                        if method_name:
                            if method_name in defined_names:
                                calls.append(('intra', method_name))
                            else:
                                calls.append(('attr', f"self.{method_name}"))
                        break

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    def process_functions(node):
        if node.type == "function_item":
            name = _get_rust_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        elif node.type == "impl_item":
            type_name = None
            for child in node.children:
                if child.type == "type_identifier":
                    type_name = source[child.start_byte:child.end_byte].decode("utf-8")
                    break

            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        if item.type == "function_item":
                            method_name = _get_rust_node_name(item, source)
                            if method_name:
                                full_name = f"{type_name}.{method_name}" if type_name else method_name
                                calls_by_func[full_name] = extract_calls_from_func(item, full_name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_java_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all method calls from a Java file, grouped by caller method.

    Returns:
        Dict mapping caller method name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    if not TREE_SITTER_JAVA_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_java_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()
    current_class = None

    # First pass: collect all defined method/class names
    def collect_definitions(node):
        nonlocal current_class

        if node.type == "class_declaration":
            class_name = _get_java_node_name(node, source)
            if class_name:
                defined_names.add(class_name)
                old_class = current_class
                current_class = class_name
                for child in node.children:
                    collect_definitions(child)
                current_class = old_class
                return

        elif node.type == "method_declaration":
            name = _get_java_node_name(node, source)
            if name:
                defined_names.add(name)
                if current_class:
                    defined_names.add(f"{current_class}.{name}")

        elif node.type == "constructor_declaration":
            name = _get_java_node_name(node, source)
            if name:
                defined_names.add(name)

        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each method
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "method_invocation":
                # Get the method name and object (if any)
                method_name = None
                object_name = None

                for child in node.children:
                    if child.type == "identifier":
                        # Could be method name or object
                        text = source[child.start_byte:child.end_byte].decode("utf-8")
                        if method_name is None:
                            # First identifier could be object or direct call
                            if object_name is None:
                                method_name = text
                            else:
                                method_name = text
                        else:
                            method_name = text
                    elif child.type in ("field_access", "this"):
                        # Object.method() or this.method()
                        if child.type == "this":
                            object_name = "this"
                        else:
                            object_name = source[child.start_byte:child.end_byte].decode("utf-8")
                    elif child.type == "argument_list":
                        # Skip argument list
                        pass

                # Determine call type
                if method_name:
                    if method_name in defined_names:
                        calls.append(('intra', method_name))
                    elif object_name:
                        calls.append(('attr', f"{object_name}.{method_name}"))
                    else:
                        calls.append(('direct', method_name))

            # Also handle object creation as calls (new ClassName())
            elif node.type == "object_creation_expression":
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = source[child.start_byte:child.end_byte].decode("utf-8")
                        if class_name in defined_names:
                            calls.append(('intra', class_name))
                        else:
                            calls.append(('direct', class_name))
                        break

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    # Third pass: process functions
    current_class = None

    def process_functions(node):
        nonlocal current_class

        if node.type == "class_declaration":
            class_name = _get_java_node_name(node, source)
            if class_name:
                old_class = current_class
                current_class = class_name
                for child in node.children:
                    process_functions(child)
                current_class = old_class
                return

        elif node.type == "method_declaration":
            name = _get_java_node_name(node, source)
            if name:
                full_name = f"{current_class}.{name}" if current_class else name
                calls_by_func[name] = extract_calls_from_func(node, name)
                # Also store with full name
                if current_class:
                    calls_by_func[full_name] = calls_by_func[name]

        elif node.type == "constructor_declaration":
            name = _get_java_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_c_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a C file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct' or 'intra'
    """
    if not TREE_SITTER_C_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_c_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function names
    def collect_definitions(node):
        if node.type == "function_definition":
            name = _get_c_node_name(node, source)
            if name:
                defined_names.add(name)

        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the function name being called
                callee = None
                for child in node.children:
                    if child.type == "identifier":
                        callee = source[child.start_byte:child.end_byte].decode("utf-8")
                        break

                if callee:
                    if callee in defined_names:
                        calls.append(('intra', callee))
                    else:
                        calls.append(('direct', callee))

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    # Third pass: process functions
    def process_functions(node):
        if node.type == "function_definition":
            name = _get_c_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_php_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a PHP file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'static', 'attr', or 'intra'
    """
    if not TREE_SITTER_PHP_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_php_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_funcs = set()
    defined_classes = set()

    # Pass 1: Collect all defined function/class/method names
    def collect_definitions(node):
        if node.type == "function_definition":
            name = _get_php_node_name(node, source)
            if name:
                defined_funcs.add(name)
        elif node.type == "method_declaration":
            name = _get_php_node_name(node, source)
            if name:
                defined_funcs.add(name)
        elif node.type == "class_declaration":
            name = _get_php_node_name(node, source)
            if name:
                defined_classes.add(name)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Pass 2: Extract calls from each function/method
    def extract_calls_from_node(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            # Regular function call: foo()
            if node.type == "function_call_expression":
                # Get the function name or class::method being called
                func_child = node.child_by_field_name("function")
                if func_child:
                    if func_child.type == "name":
                        # Simple function call: foo()
                        callee = source[func_child.start_byte:func_child.end_byte].decode("utf-8")
                        if callee in defined_funcs:
                            calls.append(('intra', callee))
                        else:
                            calls.append(('direct', callee))
                    elif func_child.type == "scoped_call_expression":
                        # Static method call: ClassName::method()
                        scope = func_child.child_by_field_name("scope")
                        name = func_child.child_by_field_name("name")
                        if scope and name:
                            class_name = source[scope.start_byte:scope.end_byte].decode("utf-8")
                            method_name = source[name.start_byte:name.end_byte].decode("utf-8")
                            if class_name in defined_classes:
                                calls.append(('intra', f"{class_name}::{method_name}"))
                            else:
                                calls.append(('static', f"{class_name}::{method_name}"))
                    elif func_child.type == "qualified_name":
                        # Fully qualified call: \App\Service::method()
                        callee = source[func_child.start_byte:func_child.end_byte].decode("utf-8")
                        calls.append(('direct', callee))

            # Member call expression: $obj->method()
            elif node.type == "member_call_expression":
                obj_node = node.child_by_field_name("object")
                name_node = node.child_by_field_name("name")
                if obj_node and name_node:
                    obj_name = source[obj_node.start_byte:obj_node.end_byte].decode("utf-8")
                    method_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
                    # $this->method() is intra-file call to same class method
                    if obj_name == "$this":
                        if method_name in defined_funcs:
                            calls.append(('intra', method_name))
                        else:
                            calls.append(('attr', f"$this->{method_name}"))
                    else:
                        calls.append(('attr', f"{obj_name}->{method_name}"))

            # Top-level static method call: User::find() - scoped_call_expression is standalone
            elif node.type == "scoped_call_expression":
                # Extract class name and method name from children
                names = [c for c in node.children if c.type == "name"]
                if len(names) >= 2:
                    class_name = source[names[0].start_byte:names[0].end_byte].decode("utf-8")
                    method_name = source[names[1].start_byte:names[1].end_byte].decode("utf-8")
                    if class_name in defined_classes:
                        calls.append(('intra', f"{class_name}::{method_name}"))
                    else:
                        calls.append(('static', f"{class_name}::{method_name}"))

            for child in node.children:
                visit_calls(child)

        # Visit the function body
        body_node = func_node.child_by_field_name("body")
        if body_node:
            visit_calls(body_node)

        return calls

    # Pass 3: Visit each function/method definition
    current_class = None

    def process_functions(node):
        nonlocal current_class

        if node.type == "class_declaration":
            class_name = _get_php_node_name(node, source)
            old_class = current_class
            current_class = class_name
            for child in node.children:
                process_functions(child)
            current_class = old_class
            return

        if node.type == "function_definition":
            name = _get_php_node_name(node, source)
            if name:
                calls = extract_calls_from_node(node, name)
                calls_by_func[name] = calls

        elif node.type == "method_declaration":
            name = _get_php_node_name(node, source)
            if name:
                calls = extract_calls_from_node(node, name)
                calls_by_func[name] = calls
                # Also store with class prefix if we have class context
                if current_class:
                    full_name = f"{current_class}::{name}"
                    calls_by_func[full_name] = calls

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def build_project_call_graph(
    root: str | Path,
    language: str = "python",
    use_workspace_config: bool = True
) -> ProjectCallGraph:
    """
    Build a complete project-wide call graph.

    Resolves cross-file calls by:
    1. Scanning all source files for the language
    2. Building a function index
    3. Parsing imports in each file
    4. Matching call sites to definitions

    Args:
        root: Project root directory
        language: "python" or "typescript"
        use_workspace_config: If True, loads .claude/workspace.json to scope
                             indexing to activePackages and excludePatterns.
                             Defaults to True for monorepo support.

    Returns:
        ProjectCallGraph with edges as (src_file, src_func, dst_file, dst_func)
    """
    root = Path(root).resolve()
    graph = ProjectCallGraph()

    # Load workspace config if enabled
    workspace_config = None
    if use_workspace_config:
        workspace_config = load_workspace_config(root)

    # Self-contained builders skip the eager full-project func_index scan.
    self_contained_languages = {
        "elixir", "swift", "ruby", "kotlin", "csharp",
        "lua", "luau", "scala", "cpp",
    }
    if language in self_contained_languages:
        func_index = None
    else:
        func_index = build_function_index(root, language, workspace_config)

    if language == "python":
        _build_python_call_graph(root, graph, func_index, workspace_config)
    elif language in ("typescript", "javascript"):
        _build_typescript_call_graph(root, graph, func_index, workspace_config, language=language)
    elif language == "go":
        _build_go_call_graph(root, graph, func_index, workspace_config)
    elif language == "rust":
        _build_rust_call_graph(root, graph, func_index, workspace_config)
    elif language == "java":
        _build_java_call_graph(root, graph, func_index, workspace_config)
    elif language == "c":
        _build_c_call_graph(root, graph, func_index, workspace_config)
    elif language == "php":
        _build_php_call_graph(root, graph, func_index, workspace_config)
    elif language == "elixir":
        _build_elixir_call_graph(root, graph, workspace_config)
    elif language == "swift":
        _build_swift_call_graph(root, graph, workspace_config)
    elif language == "ruby":
        _build_ruby_call_graph(root, graph, workspace_config)
    elif language == "kotlin":
        _build_kotlin_call_graph(root, graph, workspace_config)
    elif language == "csharp":
        _build_csharp_call_graph(root, graph, workspace_config)
    elif language == "lua":
        _build_lua_call_graph(root, graph, workspace_config)
    elif language == "luau":
        _build_luau_call_graph(root, graph, workspace_config)
    elif language == "scala":
        _build_scala_call_graph(root, graph, workspace_config)
    elif language == "cpp":
        _build_cpp_call_graph(root, graph, workspace_config)

    return graph


def _build_python_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for Python files."""
    for py_file in scan_project(root, "python", workspace_config):
        py_path = Path(py_file)
        rel_path = str(py_path.relative_to(root))

        # Get imports for this file
        imports = parse_imports(py_path)

        # Build import resolution map
        import_map = {}
        module_imports = {}

        for imp in imports:
            if imp['is_from']:
                module = imp['module']
                aliases = imp.get('aliases', {})
                for name in imp['names']:
                    alias = None
                    for alias_name, orig_name in aliases.items():
                        if orig_name == name:
                            alias = alias_name
                            break
                    if alias:
                        import_map[alias] = (module, name)
                    import_map[name] = (module, name)
            else:
                module = imp['module']
                alias = imp.get('alias')
                if alias:
                    module_imports[alias] = module
                else:
                    module_imports[module] = module

        # Get calls from this file
        calls_by_func = _extract_file_calls(py_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)
                elif call_type == 'direct':
                    if call_target in import_map:
                        module, orig_name = import_map[call_target]
                        key = (module.split('.')[-1], orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                        else:
                            key = (module, orig_name)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                elif call_type == 'attr':
                    parts = call_target.split('.', 1)
                    if len(parts) == 2:
                        obj, method = parts
                        if obj in module_imports:
                            module = module_imports[obj]
                            simple_module = module.split('.')[-1]
                            key = (simple_module, method)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, method)
                elif call_type == 'ref':
                    # Function reference (higher-order usage) - intra-file only
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)


def _build_typescript_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    language: str = "typescript",
):
    """Build call graph for TypeScript/JavaScript files."""
    commonjs_exports_cache = {}

    for ts_file in scan_project(root, language, workspace_config):
        ts_path = Path(ts_file)
        rel_path = str(ts_path.relative_to(root))

        # Get imports for this file
        imports = parse_ts_imports(ts_path, language=language)

        # Build import resolution map
        # For TypeScript, imports are relative paths or package names
        import_map = {}  # local_name -> (module_path, original_name)
        default_imports = {}  # local_name -> module_path
        namespace_imports = {}  # local_name -> module_path

        for imp in imports:
            module = imp['module']
            # Resolve relative imports
            if module.startswith('.'):
                # Convert relative path to file path
                module_path = _resolve_ts_import(rel_path, module)
            else:
                module_path = module

            # Named imports: import { foo, bar as baz } from "./module"
            for name in imp.get('names', []):
                import_map[name] = (module_path, name)

            # Handle aliases
            for alias, orig_name in imp.get('aliases', {}).items():
                if orig_name == "*":
                    namespace_imports[alias] = module_path
                else:
                    import_map[alias] = (module_path, orig_name)

            # Default import: import Foo from "./module"
            if imp.get('default'):
                default_imports[imp['default']] = module_path

        # Get calls from this file
        calls_by_func = _extract_ts_file_calls(ts_path, root, language=language)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'direct':
                    if call_target in import_map:
                        module_path, orig_name = import_map[call_target]
                        # Try to find in function index
                        simple_module = Path(module_path).stem
                        key = (simple_module, orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                    elif call_target in default_imports:
                        module_path = default_imports[call_target]
                        simple_module = Path(module_path).stem
                        # Default export often matches the module name or 'default'
                        key = (simple_module, call_target)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, call_target)
                        else:
                            default_key = (simple_module, "default")
                            if default_key in func_index:
                                dst_file = func_index[default_key]
                                if dst_file not in commonjs_exports_cache:
                                    commonjs_exports_cache[dst_file] = _extract_commonjs_file_exports(
                                        root / dst_file,
                                        language=language,
                                    )
                                named_candidates = (
                                    commonjs_exports_cache[dst_file]
                                    - {"default", call_target}
                                )
                                resolved_target = next(iter(named_candidates)) if len(named_candidates) == 1 else "default"
                                graph.add_edge(rel_path, caller_func, dst_file, resolved_target)

                elif call_type == 'attr':
                    parts = call_target.split('.', 1)
                    if len(parts) == 2:
                        obj, method = parts
                        if obj in namespace_imports:
                            module_path = namespace_imports[obj]
                            simple_module = Path(module_path).stem
                            key = (simple_module, method)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, method)


def _resolve_ts_import(from_file: str, import_path: str) -> str:
    """Resolve a relative TypeScript import path to a file path."""
    from_dir = str(Path(from_file).parent)
    if from_dir == '.':
        from_dir = ''

    # Handle ./ and ../
    if import_path.startswith('./'):
        resolved = import_path[2:]
        if from_dir:
            resolved = f"{from_dir}/{resolved}"
    elif import_path.startswith('../'):
        parts = from_dir.split('/') if from_dir else []
        import_parts = import_path.split('/')
        while import_parts and import_parts[0] == '..':
            import_parts.pop(0)
            if parts:
                parts.pop()
        resolved = '/'.join(parts + import_parts)
    else:
        resolved = import_path

    return resolved


def _build_go_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for Go files."""
    for go_file in scan_project(root, "go", workspace_config):
        go_path = Path(go_file)
        rel_path = str(go_path.relative_to(root))

        # Get imports for this file
        imports = parse_go_imports(go_path)

        # Build import resolution map
        # For Go, imports are package paths with optional aliases
        package_imports = {}  # local_name -> package_path

        for imp in imports:
            module = imp['module']
            alias = imp.get('alias')

            # Resolve relative imports (./pkg)
            if module.startswith('./') or module.startswith('../'):
                module_path = _resolve_go_import(rel_path, module)
            else:
                module_path = module

            # Determine the local name (alias or last path component)
            if alias:
                local_name = alias
            else:
                # Use last component of path as package name
                local_name = module.rstrip('/').split('/')[-1]

            package_imports[local_name] = module_path

        # Get calls from this file
        calls_by_func = _extract_go_file_calls(go_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'attr':
                    parts = call_target.split('.', 1)
                    if len(parts) == 2:
                        pkg, func_name = parts
                        if pkg in package_imports:
                            pkg_path = package_imports[pkg]
                            # Try to find in function index
                            # For Go packages, look in all files in the package directory
                            for key, file_path in func_index.items():
                                # Handle both tuple keys (mod, name) and string keys
                                if isinstance(key, tuple) and len(key) == 2:
                                    mod, name = key
                                    if name == func_name:
                                        # Check if this file is in the right package
                                        if pkg_path.lstrip('./') in file_path or mod == pkg:
                                            graph.add_edge(rel_path, caller_func, file_path, func_name)
                                            break


def _resolve_go_import(from_file: str, import_path: str) -> str:
    """Resolve a relative Go import path to a directory path."""
    from_dir = str(Path(from_file).parent)
    if from_dir == '.':
        from_dir = ''

    # Handle ./ and ../
    if import_path.startswith('./'):
        resolved = import_path[2:]
        if from_dir:
            resolved = f"{from_dir}/{resolved}"
    elif import_path.startswith('../'):
        parts = from_dir.split('/') if from_dir else []
        import_parts = import_path.split('/')
        while import_parts and import_parts[0] == '..':
            import_parts.pop(0)
            if parts:
                parts.pop()
        resolved = '/'.join(parts + import_parts)
    else:
        resolved = import_path

    return resolved


def _build_rust_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for Rust files."""
    for rs_file in scan_project(root, "rust", workspace_config):
        rs_path = Path(rs_file)
        rel_path = str(rs_path.relative_to(root))

        # Get imports for this file
        imports = parse_rust_imports(rs_path)

        # Build import resolution map
        # For Rust, use statements map names to modules
        import_map = {}  # local_name -> (module_path, original_name)
        mod_imports = {}  # mod_name -> potential file path

        for imp in imports:
            module = imp['module']
            names = imp['names']

            if imp.get('is_mod'):
                # mod declaration: mod utils; -> maps to utils.rs or utils/mod.rs
                mod_name = module
                # Try to find the file
                parent_dir = rs_path.parent
                mod_file = parent_dir / f"{mod_name}.rs"
                if mod_file.exists():
                    mod_imports[mod_name] = str(mod_file.relative_to(root))
                else:
                    mod_dir_file = parent_dir / mod_name / "mod.rs"
                    if mod_dir_file.exists():
                        mod_imports[mod_name] = str(mod_dir_file.relative_to(root))
            else:
                # use declaration
                # Resolve crate::, self::, super:: prefixes
                resolved_module = _resolve_rust_module(module, rel_path, root)

                for name in names:
                    if name == "*":
                        # Glob import - can't resolve specific names
                        continue
                    import_map[name] = (resolved_module, name)

        # Get calls from this file
        calls_by_func = _extract_rust_file_calls(rs_path, root)

        # Note: edges stored dot-form; display conversion (dot→::) happens at
        # output sites in api.py (RelevantContext.to_llm_string) and cli.py
        # (calls/arch JSON branches) via _rust_display_name. Keep lookup keys
        # dot-form here so resolve_func_name normalization stays consistent.
        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'direct':
                    if call_target in import_map:
                        module_path, orig_name = import_map[call_target]
                        # Try to find in function index
                        simple_module = Path(module_path).stem if module_path else ""
                        key = (simple_module, orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)

                elif call_type == 'attr':
                    # Scoped call like module::func or Type::method
                    if "::" in call_target:
                        parts = call_target.split("::")
                        func_name = parts[-1]
                        module_prefix = parts[0]

                        # Check if it's a mod import
                        if module_prefix in mod_imports:
                            dst_file = mod_imports[module_prefix]
                            simple_module = Path(dst_file).stem
                            key = (simple_module, func_name)
                            if key in func_index:
                                graph.add_edge(rel_path, caller_func, func_index[key], func_name)
                        else:
                            # Try to find in function index by simple name
                            key = (module_prefix, func_name)
                            if key in func_index:
                                graph.add_edge(rel_path, caller_func, func_index[key], func_name)


def _resolve_rust_module(module: str, from_file: str, root: Path) -> str:
    """
    Resolve a Rust module path to a potential file path.

    Handles:
    - crate:: -> project root
    - self:: -> current module
    - super:: -> parent module
    """
    from_path = Path(from_file)
    from_dir = from_path.parent

    if module.startswith("crate::"):
        # crate:: refers to the crate root
        remainder = module[7:]  # Strip "crate::"
        parts = remainder.split("::")
        return "/".join(parts)

    elif module.startswith("self::"):
        # self:: refers to current module
        remainder = module[6:]  # Strip "self::"
        parts = remainder.split("::")
        if from_dir == Path("."):
            return "/".join(parts)
        return str(from_dir / "/".join(parts))

    elif module.startswith("super::"):
        # super:: refers to parent module
        remainder = module[7:]  # Strip "super::"
        parts = remainder.split("::")
        parent = from_dir.parent if from_dir != Path(".") else Path(".")
        return str(parent / "/".join(parts))

    else:
        # External crate or std library - return as is
        return module.replace("::", "/")


def _build_name_index(func_index: dict) -> dict[str, list[tuple[str, tuple]]]:
    """Build a reverse index: function_name -> [(file_path, full_key), ...].

    This avoids O(N) linear scans of func_index for every call site.
    """
    name_index: dict[str, list[tuple[str, tuple]]] = {}
    seen: dict[str, set[str]] = {}
    for key, file_path in func_index.items():
        if isinstance(key, tuple) and len(key) == 2:
            _, name = key
            if name not in name_index:
                name_index[name] = []
                seen[name] = set()
            if file_path not in seen[name]:
                seen[name].add(file_path)
                name_index[name].append((file_path, key))
    return name_index


def _build_java_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for Java files."""
    name_index = _build_name_index(func_index)

    for java_file in scan_project(root, "java", workspace_config):
        java_path = Path(java_file)
        rel_path = str(java_path.relative_to(root))

        # Get imports for this file
        imports = parse_java_imports(java_path)

        # Build import resolution map
        # For Java, imports are fully qualified class names
        import_map = {}  # simple_name -> full_module

        for imp in imports:
            module = imp['module']
            is_wildcard = imp.get('is_wildcard', False)

            if is_wildcard:
                # Wildcard import - can't resolve specific names easily
                # Store the package prefix for later matching
                package = module.rstrip('.*')
                import_map[f"*:{package}"] = package
            else:
                # Get simple name from full import
                # e.g., java.util.List -> List
                simple_name = module.split('.')[-1]
                import_map[simple_name] = module

        # Get calls from this file
        calls_by_func = _extract_java_file_calls(java_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'direct':
                    resolved = False
                    # Check import_map first for a fully qualified name
                    if call_target in import_map:
                        fq_module = import_map[call_target]
                        # Try func_index with the fully qualified class name
                        fq_simple = fq_module.split('.')[-1]
                        key = (fq_simple, call_target)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, call_target)
                            resolved = True
                        else:
                            key = (fq_module, call_target)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, call_target)
                                resolved = True
                    if not resolved and call_target in name_index and len(name_index[call_target]) == 1:
                        target_file, _ = name_index[call_target][0]
                        graph.add_edge(rel_path, caller_func, target_file, call_target)

                elif call_type == 'attr':
                    if '.' in call_target:
                        class_name = call_target.split('.')[0]
                        method_name = call_target.split('.')[-1]
                        resolved = False
                        # Resolve class through import_map
                        if class_name in import_map:
                            fq_module = import_map[class_name]
                            fq_simple = fq_module.split('.')[-1]
                            # Try qualified Class.method key
                            qual_key = (fq_simple, f"{class_name}.{method_name}")
                            if qual_key in func_index:
                                dst_file = func_index[qual_key]
                                graph.add_edge(rel_path, caller_func, dst_file, method_name)
                                resolved = True
                            else:
                                qual_key = (fq_module, f"{class_name}.{method_name}")
                                if qual_key in func_index:
                                    dst_file = func_index[qual_key]
                                    graph.add_edge(rel_path, caller_func, dst_file, method_name)
                                    resolved = True
                        if not resolved and method_name in name_index and len(name_index[method_name]) == 1:
                            target_file, _ = name_index[method_name][0]
                            graph.add_edge(rel_path, caller_func, target_file, method_name)


def _build_c_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for C files."""
    name_index = _build_name_index(func_index)

    for c_file in scan_project(root, "c", workspace_config):
        c_path = Path(c_file)
        rel_path = str(c_path.relative_to(root))

        # Get includes for this file
        includes = parse_c_imports(c_path)

        # Build include resolution map
        # For C, includes are header file paths
        include_map = {}  # header_name -> header_path

        for inc in includes:
            module = inc['module']
            is_system = inc.get('is_system', False)
            header_name = module.split('/')[-1] if '/' in module else module
            include_map[header_name] = module

        # Get calls from this file
        calls_by_func = _extract_c_file_calls(c_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'direct':
                    if call_target in name_index and len(name_index[call_target]) == 1:
                        target_file, _ = name_index[call_target][0]
                        graph.add_edge(rel_path, caller_func, target_file, call_target)


def _find_method_in_index(method_index: dict, method: str, preferred_file: str | None = None) -> str | None:
    """Return the file_path for a func_index entry matching method name, or None.

    Returns None when no candidates exist or when the match is ambiguous
    (multiple candidates and no preferred_file match), to avoid false edges.
    If preferred_file is given, a same-file match is returned preferentially.
    """
    candidates = method_index.get(method, [])
    if not candidates:
        return None
    if preferred_file:
        for _key, fp in candidates:
            if fp == preferred_file:
                return fp
    # Only one candidate — unambiguous, safe to return
    if len(candidates) == 1:
        return candidates[0][1]
    # Multiple candidates and no preferred_file match — ambiguous; skip edge
    return None


def _build_php_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for PHP files."""
    # Pre-build method_name -> [(key, file_path)] for O(1) lookups
    method_index: dict[str, list] = {}
    for key, file_path in func_index.items():
        if isinstance(key, tuple) and len(key) == 2:
            _, name = key
            method_index.setdefault(name, []).append((key, file_path))

    for php_file in scan_project(root, "php", workspace_config):
        php_path = Path(php_file)
        rel_path = str(php_path.relative_to(root))

        # Get imports for this file
        imports = parse_php_imports(php_path)

        # Build import resolution map
        # For PHP: 'User' -> ('App\\Models', 'User')
        import_map = {}  # alias -> (namespace, name)

        for imp in imports:
            if imp.get('type') == 'use':
                module = imp.get('module', '')
                # Parse full module path like "App\Models\User"
                parts = module.split('\\')
                if parts:
                    name = parts[-1]  # Last part is the class/function name
                    namespace = '\\'.join(parts[:-1]) if len(parts) > 1 else ''
                    # Get alias if present
                    alias = imp.get('alias', name)
                    import_map[alias] = (namespace, name)
                    import_map[name] = (namespace, name)

        # Get calls from this file
        calls_by_func = _extract_php_file_calls(php_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == 'intra':
                    # Same file call
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'direct':
                    # Direct function call
                    if call_target in import_map:
                        namespace, orig_name = import_map[call_target]
                        # Try to find in func_index
                        # First try with simple module name
                        simple_module = namespace.split('\\')[-1] if namespace else ''
                        key = (simple_module, orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                        else:
                            # Try with full namespace
                            key = (namespace, orig_name)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                    else:
                        # Try to find directly in func_index
                        dst = _find_method_in_index(method_index, call_target)
                        if dst:
                            graph.add_edge(rel_path, caller_func, dst, call_target)

                elif call_type == 'static':
                    parts = call_target.split('::', 1)
                    if len(parts) == 2:
                        class_name, method = parts
                        if class_name in import_map:
                            namespace, resolved_class = import_map[class_name]
                            # Look for Class::method in index (ambiguity-safe)
                            _candidates = []
                            for key, file_path in func_index.items():
                                if isinstance(key, tuple) and len(key) == 2:
                                    _, name = key
                                    if name == method or name == f"{resolved_class}::{method}":
                                        _candidates.append(file_path)
                            if len(_candidates) == 1:
                                graph.add_edge(rel_path, caller_func, _candidates[0], method)
                            elif _candidates:
                                # Multiple candidates: prefer same-file, else skip
                                if rel_path in _candidates:
                                    graph.add_edge(rel_path, caller_func, rel_path, method)
                        else:
                            key = (class_name, method)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, method)
                            else:
                                # Search in index
                                dst = _find_method_in_index(method_index, method)
                                if dst:
                                    graph.add_edge(rel_path, caller_func, dst, method)

                elif call_type == 'attr':
                    parts = call_target.split('->', 1)
                    if len(parts) == 2:
                        obj, method = parts
                        # For $this->method(), prefer same file (own class), fall back cross-file
                        if obj == "$this":
                            dst = _find_method_in_index(method_index, method, preferred_file=rel_path)
                            if dst:
                                graph.add_edge(rel_path, caller_func, dst, method)
                        else:
                            # Generic object method call - try to find method
                            for key, file_path in func_index.items():
                                if isinstance(key, tuple) and len(key) == 2:
                                    _, name = key
                                    if name == method:
                                        graph.add_edge(rel_path, caller_func, file_path, method)
                                        break


# =============================================================================
# Generic helpers for new-language call-graph builders.
#
# These builders are self-contained: they do NOT rely on build_function_index()
# (which has no _index_<lang>_file for Ruby/Kotlin/C#/Lua/Luau/Scala/C++/Elixir/
# Swift on upstream/main). Instead each builder constructs its own
# global_defs map by re-walking each project file with the language's
# extractor (which records defined_names as a side channel via a closure).
# =============================================================================


def _generic_extract(
    file_path: Path,
    parser,
    def_node_types: set,
    name_child_types: frozenset[str] = frozenset({"identifier", "simple_identifier"}),
    call_node_types: frozenset[str] = frozenset({"call", "call_expression", "function_call"}),
):
    """
    Generic tree-sitter walker: returns (defined_names: set[str],
    calls_by_func: dict[str, list[(call_type, target)]]).

    A function definition is any node whose type is in def_node_types and which
    has a direct or grandchild identifier-like child for its name.
    A call site is a node whose type is in call_node_types; the call target is
    the text of its first identifier-like descendant (with simple normalization
    for receiver.method / receiver::method / receiver:method).
    """
    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return set(), {}

    defined_names: set[str] = set()
    calls_by_func: dict[str, list[tuple[str, str]]] = {}

    def find_name(node):
        # Return text of first direct child whose type is in name_child_types
        for child in node.children:
            if child.type in name_child_types:
                return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    def extract_call_target(call_node):
        # Find first identifier-like text; handle method-receiver shapes.
        # Look for navigation/member/scoped_call structures.
        # Default: the first identifier-like descendant text.
        first_text = None
        for child in call_node.children:
            t = child.type
            if t in name_child_types:
                first_text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                return ("direct", first_text)
            if t in ("field_expression", "scoped_identifier", "qualified_name",
                     "navigation_expression", "member_expression",
                     "method_index_expression", "dot_index_expression",
                     "field_access", "scope_resolution",
                     "member_access_expression"):
                # qualified call obj.method or obj::method
                txt = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                # normalize :: and : separators to .
                norm = txt.replace("::", ".").replace(":", ".")
                # strip any whitespace
                norm = norm.strip()
                if "." in norm:
                    return ("attr", norm)
                return ("direct", norm)
        # Fallback: walk descendants for first identifier-like
        for child in call_node.children:
            for sub in child.children:
                if sub.type in name_child_types:
                    first_text = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                    return ("direct", first_text)
        return None

    def extract_calls_in(body_node) -> list[tuple[str, str]]:
        calls = []

        def visit(node):
            if node.type in call_node_types:
                t = extract_call_target(node)
                if t:
                    calls.append(t)
            for child in node.children:
                # Don't descend into nested definitions — visit_all walks them
                # separately, and attributing their calls to the outer function
                # would double-count.
                if child.type in def_node_types:
                    continue
                visit(child)

        visit(body_node)
        return calls

    # Pass 1: collect all top-level definitions so forward references resolve to "intra".
    def collect_defs(node):
        if node.type in def_node_types:
            n = find_name(node)
            if n:
                defined_names.add(n)
        for child in node.children:
            collect_defs(child)

    collect_defs(tree.root_node)

    # Pass 2: classify calls against the complete defined_names set.
    def visit_all(node):
        if node.type in def_node_types:
            n = find_name(node)
            if n:
                calls = extract_calls_in(node)
                resolved = []
                for ctype, target in calls:
                    if ctype == "direct" and target in defined_names:
                        # Record self-references as intra-file (covers recursion).
                        resolved.append(("intra", target))
                    else:
                        resolved.append((ctype, target))
                calls_by_func[n] = resolved
        for child in node.children:
            visit_all(child)

    visit_all(tree.root_node)
    return defined_names, calls_by_func


def _build_generic_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    language: str,
    parser_factory,
    available_flag: bool,
    def_node_types: set,
    workspace_config: Optional[WorkspaceConfig] = None,
    name_child_types: set = frozenset({"identifier", "simple_identifier"}),
    call_node_types: set = frozenset({"call", "call_expression", "function_call"}),
):
    """Generic builder used by Ruby/Kotlin/C#/Lua/Luau/Scala/C++."""
    if not available_flag:
        return

    parser = parser_factory()

    # Pass 1: walk all files, collect (file, defined_names, calls_by_func)
    per_file: list[tuple[str, set, dict]] = []
    global_defs: dict[str, str] = {}  # bare name -> rel_path

    for src_file in scan_project(root, language, workspace_config):
        src_path = Path(src_file)
        rel_path = str(src_path.relative_to(root))
        defs, calls_by_func = _generic_extract(
            src_path, parser, def_node_types,
            name_child_types=name_child_types,
            call_node_types=call_node_types,
        )
        per_file.append((rel_path, defs, calls_by_func))
        for name in defs:
            global_defs.setdefault(name, rel_path)

    # Pass 2: emit edges
    for rel_path, defs, calls_by_func in per_file:
        for caller_func, calls in calls_by_func.items():
            for call_type, target in calls:
                if call_type == "intra":
                    graph.add_edge(rel_path, caller_func, rel_path, target)
                elif call_type == "direct":
                    if target in defs:
                        graph.add_edge(rel_path, caller_func, rel_path, target)
                    elif target in global_defs:
                        graph.add_edge(rel_path, caller_func, global_defs[target], target)
                elif call_type == "attr":
                    # target is "a.b.c"; try last segment as method name
                    parts = target.split(".")
                    method = parts[-1]
                    if method in defs:
                        graph.add_edge(rel_path, caller_func, rel_path, method)
                    elif method in global_defs:
                        graph.add_edge(rel_path, caller_func, global_defs[method], method)


def _extract_elixir_module_name(call_node, source: bytes):
    """Extract the module name from an Elixir defmodule call node."""
    for child in call_node.children:
        if child.type == "arguments":
            for arg_child in child.children:
                if arg_child.is_named and arg_child.type == "alias":
                    return source[arg_child.start_byte:arg_child.end_byte].decode("utf-8", errors="replace")
    return None


def _extract_elixir_func_name(call_node, source: bytes):
    """Extract the function name from an Elixir def/defp call node."""
    for child in call_node.children:
        if child.type == "arguments":
            for arg_child in child.children:
                if arg_child.type == "call":
                    for cc in arg_child.children:
                        if cc.type == "identifier":
                            return source[cc.start_byte:cc.end_byte].decode("utf-8", errors="replace")
                elif arg_child.type == "identifier":
                    return source[arg_child.start_byte:arg_child.end_byte].decode("utf-8", errors="replace")
                elif arg_child.type == "binary_operator":
                    for cc in arg_child.children:
                        if cc.type == "identifier":
                            return source[cc.start_byte:cc.end_byte].decode("utf-8", errors="replace")
                        if cc.type == "call":
                            for ccc in cc.children:
                                if ccc.type == "identifier":
                                    return source[ccc.start_byte:ccc.end_byte].decode("utf-8", errors="replace")
    return None


# -----------------------------------------------------------------------------
# Elixir builder — simplified self-contained version.
# Handles defmodule + def/defp; resolves Module.func() qualified calls by
# bare name across the project.
# -----------------------------------------------------------------------------

def _extract_elixir_file_calls(file_path: Path, root: Path, parser=None) -> tuple[dict[str, set], dict[str, list[tuple[str, str]]]]:
    """Returns ({module_name: set(defined_funcs)}, {caller_key: [(call_type, target)]}).

    Single-pass implementation: collects definitions and extracts calls in one
    tree walk. Note: calls within a function can only reference functions defined
    earlier in the same module in this pass (forward references within a module
    are classified as 'local' rather than 'intra').
    """
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        return {}, {}

    try:
        source = file_path.read_bytes()
        if parser is None:
            parser = _get_elixir_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}, {}

    defined: dict[str, set] = {}  # module fqn -> set of func names
    calls_by_func: dict[str, list[tuple[str, str]]] = {}

    def call_ident(node):
        for child in node.children:
            if child.type == "identifier":
                return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    def extract_body_calls(body_node, local_defs: set) -> list[tuple[str, str]]:
        calls: list[tuple[str, str]] = []
        skip_keywords = {"def", "defp", "defmodule", "alias", "import", "use", "require",
                        "if", "unless", "case", "cond", "with", "for", "try",
                        "raise", "throw", "quote", "unquote"}

        def visit(node):
            if node.type == "call":
                dot_child = None
                bare_ident = None
                for child in node.children:
                    if child.type == "dot":
                        dot_child = child
                    elif child.type == "identifier" and dot_child is None:
                        bare_ident = child
                if dot_child is not None:
                    alias_text = None
                    fname = None
                    for dc in dot_child.children:
                        if dc.type == "alias":
                            alias_text = source[dc.start_byte:dc.end_byte].decode("utf-8", errors="replace")
                        elif dc.type == "identifier":
                            fname = source[dc.start_byte:dc.end_byte].decode("utf-8", errors="replace")
                    if alias_text and fname:
                        calls.append(("qualified", f"{alias_text}.{fname}"))
                elif bare_ident is not None:
                    fname = source[bare_ident.start_byte:bare_ident.end_byte].decode("utf-8", errors="replace")
                    if fname not in skip_keywords:
                        if fname in local_defs:
                            calls.append(("intra", fname))
                        else:
                            calls.append(("local", fname))
            for child in node.children:
                visit(child)

        visit(body_node)
        return calls

    current_module = None

    def visit_all(node):
        nonlocal current_module
        if node.type == "call":
            ident = call_ident(node)
            if ident == "defmodule":
                modname = _extract_elixir_module_name(node, source)
                if modname:
                    fqn = f"{current_module}.{modname}" if current_module else modname
                    defined.setdefault(fqn, set())
                    old = current_module
                    current_module = fqn
                    for child in node.children:
                        if child.type == "do_block":
                            for dc in child.children:
                                visit_all(dc)
                    current_module = old
                    return
            if ident in ("def", "defp"):
                fname = _extract_elixir_func_name(node, source)
                if fname and current_module:
                    defined.setdefault(current_module, set()).add(fname)
                    key = f"{current_module}.{fname}"
                    local = defined.get(current_module, set())
                    for child in node.children:
                        if child.type == "do_block":
                            calls_by_func.setdefault(key, []).extend(extract_body_calls(child, local))
                    return
        for child in node.children:
            visit_all(child)

    visit_all(tree.root_node)
    return defined, calls_by_func


def _build_elixir_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    """Build call graph for Elixir files (self-contained, no func_index dependency)."""
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        return

    parser = _get_elixir_parser()  # Create once; reuse across all files

    # Pass 1: collect per-file defs + calls, build global module->file map
    per_file: list[tuple[str, dict, dict]] = []
    module_to_file: dict[str, str] = {}  # full module fqn -> rel_path
    module_funcs: dict[str, set] = {}  # full module fqn -> defined funcs

    for ex_file in scan_project(root, "elixir", workspace_config):
        ex_path = Path(ex_file)
        rel_path = str(ex_path.relative_to(root))
        defined, calls_by_func = _extract_elixir_file_calls(ex_path, root, parser)
        per_file.append((rel_path, defined, calls_by_func))
        for mod_fqn, funcs in defined.items():
            module_to_file.setdefault(mod_fqn, rel_path)
            module_funcs.setdefault(mod_fqn, set()).update(funcs)

    # Build alias lookup: last segment of a module name -> full fqn
    last_segment_to_fqn: dict[str, str] = {}
    for fqn in module_to_file:
        last = fqn.rsplit(".", 1)[-1]
        last_segment_to_fqn.setdefault(last, fqn)

    # Pass 2: emit edges
    for rel_path, defined, calls_by_func in per_file:
        for caller_func, calls in calls_by_func.items():
            # Caller's module fqn = caller_func minus last segment
            caller_mod = caller_func.rsplit(".", 1)[0] if "." in caller_func else None
            for call_type, target in calls:
                if call_type == "intra":
                    if caller_mod:
                        graph.add_edge(rel_path, caller_func, rel_path, f"{caller_mod}.{target}")
                    else:
                        graph.add_edge(rel_path, caller_func, rel_path, target)
                elif call_type == "qualified":
                    mod_ref, fname = target.rsplit(".", 1)
                    # Try to resolve mod_ref: direct fqn, or last-segment alias
                    resolved_fqn = None
                    if mod_ref in module_to_file:
                        resolved_fqn = mod_ref
                    elif mod_ref in last_segment_to_fqn:
                        resolved_fqn = last_segment_to_fqn[mod_ref]
                    if resolved_fqn and fname in module_funcs.get(resolved_fqn, set()):
                        dst_file = module_to_file[resolved_fqn]
                        graph.add_edge(rel_path, caller_func, dst_file, f"{resolved_fqn}.{fname}")
                elif call_type == "local":
                    # Bare same-file call: only emit when target is a function
                    # in the caller's module — avoids fake edges for imports/builtins.
                    if caller_mod and target in module_funcs.get(caller_mod, set()):
                        graph.add_edge(rel_path, caller_func, rel_path, f"{caller_mod}.{target}")


# -----------------------------------------------------------------------------
# Swift — simplified builder.
# -----------------------------------------------------------------------------

def _swift_func_name(node, source):
    for child in node.children:
        if child.type == "simple_identifier":
            return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
    return None


def _swift_type_name(node, source):
    """Get the name of a Swift type declaration (class/struct/enum/protocol/extension).

    Extensions name the *extended* type via ``user_type``; other type kinds use
    ``type_identifier``. Returns ``None`` if no name is found.
    """
    for child in node.children:
        if child.type in ("type_identifier", "user_type"):
            return source[child.start_byte:child.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
    return None


# Swift type-declaration node-kind allow-list. Some entries may currently be
# dead under tree-sitter-swift 0.7.2 — that grammar appears to collapse
# ``struct_declaration``, ``enum_declaration``, and ``extension_declaration``
# down to ``class_declaration`` in many positions — but the claim is
# empirically unverified across grammar versions. We intentionally keep all
# five node kinds here so future tree-sitter-swift upgrades that re-emit the
# more specific node types don't silently regress Swift type detection.
# DO NOT trim this list without a grammar-version sweep.
_SWIFT_TYPE_DECL_NODES = (
    "class_declaration",
    "struct_declaration",
    "enum_declaration",
    "protocol_declaration",
    "extension_declaration",
)


def _extract_swift_file_calls(file_path: Path, root: Path):
    """Extract Swift definitions and call sites for the project call graph.

    Returns:
        ``(defined_names, calls_by_func, methods_by_class)`` where:

        * ``defined_names`` — set of every function/method name (unqualified)
          defined in this file. Used by the builder to resolve global call
          targets.
        * ``calls_by_func`` — ``{caller_name: [(ctype, target), ...]}``. Caller
          names are unqualified; ``ctype`` is ``"intra"``, ``"direct"``, or
          ``"attr"``.
        * ``methods_by_class`` — ``{ClassName: [method_name, ...]}``. Bug 007:
          this lets ``_build_swift_call_graph`` emit synthetic ``Class →
          method`` edges so ``tldr context <ClassName>`` walks into the class's
          methods (which then surface their cross-file callers via the
          standard reverse-adjacency pass in ``get_relevant_context``). Without
          this, querying a class returned the class shell only.
    """
    if not TREE_SITTER_SWIFT_AVAILABLE:
        _warn_swift_unavailable_once()
        return set(), {}, {}

    try:
        source = file_path.read_bytes()
        parser = _get_swift_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return set(), {}, {}

    defined: set = set()
    calls_by_func: dict[str, list[tuple[str, str]]] = {}
    methods_by_class: dict[str, list[str]] = {}

    def collect_defs(node):
        if node.type == "function_declaration":
            n = _swift_func_name(node, source)
            if n:
                defined.add(n)
        for child in node.children:
            collect_defs(child)

    collect_defs(tree.root_node)

    def extract_call(call_node):
        for child in call_node.children:
            if child.type == "simple_identifier":
                return ("bare", source[child.start_byte:child.end_byte].decode("utf-8", errors="replace"))
            if child.type == "navigation_expression":
                receiver = None
                method = None
                for sub in child.children:
                    if sub.type == "simple_identifier" and receiver is None:
                        receiver = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                    elif sub.type == "navigation_suffix":
                        for ns in sub.children:
                            if ns.type == "simple_identifier":
                                method = source[ns.start_byte:ns.end_byte].decode("utf-8", errors="replace")
                if method:
                    return ("attr", f"{receiver}.{method}" if receiver else method)
        return None

    def extract_calls_in(func_node) -> list[tuple[str, str]]:
        calls = []

        def visit(node):
            if node.type == "function_declaration" and node is not func_node:
                return
            if node.type == "call_expression":
                r = extract_call(node)
                if r:
                    kind, target = r
                    if kind == "bare":
                        if target in defined:
                            calls.append(("intra", target))
                        else:
                            calls.append(("direct", target))
                    else:
                        calls.append(("attr", target))
            for child in node.children:
                visit(child)

        visit(func_node)
        return calls

    def process(node):
        if node.type == "function_declaration":
            n = _swift_func_name(node, source)
            if n:
                calls_by_func[n] = extract_calls_in(node)
        for child in node.children:
            process(child)

    process(tree.root_node)

    # Bug 007: walk every type declaration and record the methods declared in
    # its body. We look for ``function_declaration`` nodes that live directly
    # under a ``class_body`` / ``enum_class_body`` / ``protocol_body`` child of
    # the type node (matches the structure ``hybrid_extractor._extract_swift_class``
    # already uses). Nested types are picked up by the outer recursion in
    # ``walk_types`` — only the innermost enclosing type owns a method.
    def collect_methods(type_node, class_name: str):
        for child in type_node.children:
            if child.type in ("class_body", "enum_class_body", "protocol_body"):
                for member in child.children:
                    if member.type in ("function_declaration", "protocol_function_declaration"):
                        m = _swift_func_name(member, source)
                        if m:
                            methods_by_class.setdefault(class_name, []).append(m)

    def walk_types(node):
        # Deliberate P2 O(n²) avoidance: we DO NOT recurse into ``class_body``,
        # ``enum_class_body``, ``protocol_body``, or ``function_body`` children
        # here, which means nested types declared INSIDE another type's body
        # are not enumerated and don't get their own ``Type → method`` edges.
        # ``collect_methods`` already walked the body's direct children for
        # methods; recursing again would re-enter that subtree per outer call
        # and rediscover the same methods quadratically on deep nesting.
        # If we later need ``Outer.Inner → method`` edges, factor out a single
        # body-walk pass that yields both methods and nested type decls in one
        # traversal rather than enabling recursion here. Don't ``git blame``
        # for context — this is intentional, not an oversight.
        if node.type in _SWIFT_TYPE_DECL_NODES:
            name = _swift_type_name(node, source)
            if name:
                collect_methods(node, name)
        # Only recurse into children that could contain more types
        for child in node.children:
            if child.type not in ("class_body", "enum_class_body", "protocol_body", "function_body"):
                walk_types(child)

    walk_types(tree.root_node)
    return defined, calls_by_func, methods_by_class


def _build_swift_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    """Build call graph for Swift files.

    Bug 007: after the per-file call extraction pass, emit synthetic
    ``ClassName → method_name`` edges so ``tldr context <ClassName>`` walks
    into the class's methods. The receiver-name in Swift call sites is
    usually a value (``self`` / ``model`` / etc.) rather than the type name,
    so the call-graph builder cannot recover class membership from call
    sites alone — only the type-body lexical structure (captured below in
    ``methods_by_class``) can.
    """
    if not TREE_SITTER_SWIFT_AVAILABLE:
        _warn_swift_unavailable_once()
        return
    per_file: list[tuple[str, set, dict, dict]] = []
    global_defs: dict[str, str] = {}
    for swift_file in scan_project(root, "swift", workspace_config):
        sp = Path(swift_file)
        rel = str(sp.relative_to(root))
        defs, calls, methods_by_class = _extract_swift_file_calls(sp, root)
        per_file.append((rel, defs, calls, methods_by_class))
        for n in defs:
            global_defs.setdefault(n, rel)
    for rel, defs, calls, methods_by_class in per_file:
        for caller, call_list in calls.items():
            for ctype, target in call_list:
                if ctype == "intra":
                    graph.add_edge(rel, caller, rel, target)
                elif ctype == "direct":
                    if target in global_defs:
                        graph.add_edge(rel, caller, global_defs[target], target)
                elif ctype == "attr":
                    method = target.rsplit(".", 1)[-1]
                    if method in defs:
                        graph.add_edge(rel, caller, rel, method)
                    elif method in global_defs:
                        graph.add_edge(rel, caller, global_defs[method], method)
        # Bug 007: emit class → method edges. ``ClassName`` becomes a caller
        # node whose callees are its methods, so the BFS in
        # ``get_relevant_context(ClassName, depth=N)`` enumerates methods at
        # depth 1 and the existing reverse-adjacency pass surfaces their
        # cross-file callers at depth ≥ 1.
        for class_name, method_names in methods_by_class.items():
            for method_name in method_names:
                graph.add_edge(rel, class_name, rel, method_name)


# -----------------------------------------------------------------------------
# Ruby
# -----------------------------------------------------------------------------

def _extract_ruby_file_calls(file_path: Path, parser):
    """Ruby extractor. Bare `foo` inside a method body parses as identifier
    (not a call node) in tree-sitter Ruby — we treat any identifier whose
    text matches a globally-defined method name as a call. `obj.method`
    parses as `call` with a receiver."""
    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return set(), {}

    def get_text(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    defined: set = set()

    def find_method_name(node):
        for child in node.children:
            if child.type == "identifier":
                return get_text(child)
        return None

    def collect_defs(node):
        if node.type in ("method", "singleton_method"):
            n = find_method_name(node)
            if n:
                defined.add(n)
        for child in node.children:
            collect_defs(child)

    collect_defs(tree.root_node)

    def _parse_call_node(node, calls_out: list):
        """Parse a call node and append (ctype, target, line) to calls_out."""
        receiver = None
        method = None
        for child in node.children:
            if child.type == "identifier":
                method = get_text(child)
            elif child.type in ("constant", "self"):
                receiver = get_text(child)
        if method:
            calls_out.append((
                "attr" if receiver else "direct",
                method,
                node.start_point[0] + 1,
            ))

    # Call entries are 3-tuples (ctype, target, line) where ``line`` is the
    # 1-indexed source line of the call site (tree-sitter's start_point.row is
    # 0-indexed, so we add 1).
    calls_by_func: dict[str, list[tuple[str, str, int]]] = {}

    def _collect_identifier_call(node, calls_out: list, exclude_txt=None):
        """Append a bare-identifier call entry if the node passes false-positive filters.

        Shared by visit() (method scope) and visit_toplevel() (module scope).
        """
        txt = get_text(node)
        if exclude_txt and txt == exclude_txt:
            return
        parent = node.parent
        parent_type = parent.type if parent is not None else ""
        is_lhs = (
            parent_type in ("assignment", "operator_assignment")
            and parent.child_by_field_name("left") is node
        ) or parent_type == "left_assignment_list"
        if (
            parent_type not in (
                "method_parameters", "block_parameters", "lambda_parameters",
            )
            and not is_lhs
        ):
            calls_out.append(("direct", txt, node.start_point[0] + 1))

    def extract_calls_in(method_node):
        calls = []
        method_name = find_method_name(method_node)

        def visit(node):
            if node.type in ("method", "singleton_method") and node is not method_node:
                return
            if node.type == "call":
                # call has children: receiver, ".", method (identifier)
                _parse_call_node(node, calls)
                return
            if node.type == "identifier":
                # Bare identifier in a method body — treat as potential call.
                # Builder resolves against global defs; skip the method's own
                # name, parameter names, and assignment LHS (false positives).
                _collect_identifier_call(node, calls, exclude_txt=method_name)
            for child in node.children:
                visit(child)

        # Walk only the body_statement children
        for child in method_node.children:
            if child.type == "body_statement":
                visit(child)
        return calls

    def process(node):
        if node.type in ("method", "singleton_method"):
            n = find_method_name(node)
            if n:
                calls_by_func[n] = extract_calls_in(node)
        for child in node.children:
            process(child)

    process(tree.root_node)

    # Program-scope (top-level) call sites: any `call` / bare-`identifier`
    # that lives directly under `program` (or inside a non-def container at
    # top level — e.g. an `if`/`begin` block) needs a caller bucket too, or
    # the edge-emission loop in `_build_ruby_call_graph` can never attribute
    # it. We synthesise a "__main__" bucket per file holding these targets;
    # nested `method`/`singleton_method`/`class`/`module` subtrees are
    # skipped (they already get their own buckets via `process` above —
    # avoiding the double-counting that commit c4fa922 was about).
    toplevel_calls: list[tuple[str, str, int]] = []

    def visit_toplevel(node):
        if node.type in ("method", "singleton_method", "class", "module"):
            return
        if node.type == "call":
            _parse_call_node(node, toplevel_calls)
            return
        if node.type == "identifier":
            _collect_identifier_call(node, toplevel_calls)
        for child in node.children:
            visit_toplevel(child)

    for child in tree.root_node.children:
        visit_toplevel(child)
    if toplevel_calls:
        calls_by_func["__main__"] = toplevel_calls

    return defined, calls_by_func


def _build_ruby_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    """Build call graph for Ruby files.

    Note (G-9): require_relative path resolution is `current_file.parent /
    resolved_path` (Ruby semantics), but the simplified bare-name resolution
    we use here doesn't require explicit import resolution — method names are
    matched globally across the project.
    """
    if not TREE_SITTER_RUBY_AVAILABLE:
        return
    parser = _get_ruby_parser()
    per_file = []
    global_defs: dict[str, str] = {}
    for rb_file in scan_project(root, "ruby", workspace_config):
        rp = Path(rb_file)
        rel = str(rp.relative_to(root))
        defs, calls = _extract_ruby_file_calls(rp, parser)
        per_file.append((rel, defs, calls))
        for n in defs:
            global_defs.setdefault(n, rel)
    # Track which defined functions participate in any real edge.  Defs that
    # never appear (no callers, no resolvable outbound calls) are "orphan
    # top-level defs" — impact_analysis (which only consults graph.edges)
    # would otherwise return a "not found" error for them.  We emit a
    # synthetic edge per orphan so the def appears as a `from_func` in the
    # edge set; impact_analysis then routes through its `callers_only`
    # branch and returns an entry-point result with zero callers (no error).
    #
    # The synthetic callee `(__ruby_orphan__, __ruby_orphan__)` is a
    # sentinel — distinct from any real symbol — so:
    #   * `_callee_edges(graph, <real_name>)` is unaffected,
    #   * `_is_dead(graph, <real_name>)` (dst_func-based) is unaffected,
    #   * dead_code_analysis sees the orphan as "alive" (correct — it IS
    #     defined at top level and acts as an entry point).
    referenced: set[tuple[str, str]] = set()
    for rel, defs, calls in per_file:
        for caller, clist in calls.items():
            for entry in clist:
                # Tolerate legacy 2-tuples for forward-compat with any caller
                # that monkey-patches this extractor; production code now emits
                # 3-tuples (ctype, target, line).
                if len(entry) == 3:
                    ctype, target, line = entry
                else:
                    ctype, target = entry  # type: ignore[misc]
                    line = None
                if ctype in ("direct", "attr"):
                    if target in defs:
                        graph.add_edge_with_line(rel, caller, rel, target, line)
                        # Mark both caller (if it's a real def) and callee as
                        # referenced so a caller-only def is not double-counted
                        # as an orphan. Caller bucket "__main__" is synthetic
                        # and is never in `defs`, so the `caller in defs` guard
                        # naturally excludes it while still protecting real
                        # entry-point callers from spurious orphan edges.
                        if caller in defs:
                            referenced.add((rel, caller))
                        referenced.add((rel, target))
                    elif target in global_defs:
                        graph.add_edge_with_line(
                            rel, caller, global_defs[target], target, line
                        )
                        if caller in defs:
                            referenced.add((rel, caller))
                        referenced.add((global_defs[target], target))
    # Emit synthetic orphan-marker edges for defined functions that never
    # appeared as either caller or callee of any real edge.
    for rel, defs, _calls in per_file:
        for name in defs:
            if (rel, name) in referenced:
                continue
            graph.add_edge(rel, name, RUBY_ORPHAN_SENTINEL, RUBY_ORPHAN_SENTINEL)


# -----------------------------------------------------------------------------
# Kotlin
# -----------------------------------------------------------------------------

def _build_kotlin_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    _build_generic_call_graph(
        root, graph, "kotlin", _get_kotlin_parser, TREE_SITTER_KOTLIN_AVAILABLE,
        def_node_types={"function_declaration", "secondary_constructor"},
        workspace_config=workspace_config,
        name_child_types={"simple_identifier", "identifier"},
        call_node_types={"call_expression"},
    )


# -----------------------------------------------------------------------------
# C# (csharp)
# -----------------------------------------------------------------------------

def _build_csharp_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    """C# call graph. C# method names are matched by bare name (T-7 awareness:
    partial classes and namespaces produce same-name collisions; bare-name
    matching mirrors Java's approach)."""
    _build_generic_call_graph(
        root, graph, "csharp", _get_csharp_parser, TREE_SITTER_CSHARP_AVAILABLE,
        def_node_types={"method_declaration", "constructor_declaration", "local_function_statement"},
        workspace_config=workspace_config,
        name_child_types={"identifier"},
        call_node_types={"invocation_expression"},
    )


# -----------------------------------------------------------------------------
# Lua  — handles function declarations and table-method calls (`obj:method`).
# Colon calls are normalized to dot form for graph edges.
# -----------------------------------------------------------------------------

def _extract_lua_file_calls(file_path: Path, parser):
    if parser is None:
        return set(), {}
    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return set(), {}

    defined: set = set()
    calls_by_func: dict[str, list[tuple[str, str]]] = {}

    def get_text(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def find_func_name(node):
        # function_declaration / local_function: child is "identifier" or
        # "dot_index_expression" (M.greet) or "method_index_expression"
        # (M:greet). Return last segment.
        for child in node.children:
            if child.type == "identifier":
                return get_text(child)
            if child.type in ("dot_index_expression", "method_index_expression"):
                # last identifier child
                ids = [c for c in child.children if c.type == "identifier"]
                if ids:
                    return get_text(ids[-1])
        return None

    def collect_defs(node):
        if node.type in ("function_declaration", "local_function"):
            n = find_func_name(node)
            if n:
                defined.add(n)
        for child in node.children:
            collect_defs(child)

    collect_defs(tree.root_node)

    def extract_call_target(call_node):
        # call has a single "prefix" child which may be: identifier,
        # dot_index_expression, method_index_expression
        for child in call_node.children:
            if child.type == "identifier":
                return ("direct", get_text(child))
            if child.type in ("dot_index_expression", "method_index_expression"):
                ids = [c for c in child.children if c.type == "identifier"]
                if ids:
                    method = get_text(ids[-1])
                    return ("attr", method)
        return None

    def extract_calls_in(body_node):
        calls = []

        def visit(node):
            if node.type in ("function_declaration", "local_function") and node is not body_node:
                return
            if node.type in ("function_call", "call"):
                r = extract_call_target(node)
                if r:
                    ctype, target = r
                    if ctype == "direct":
                        if target in defined:
                            calls.append(("intra", target))
                        else:
                            calls.append(("direct", target))
                    else:
                        calls.append(("attr", target))
            for child in node.children:
                visit(child)

        visit(body_node)
        return calls

    def process(node):
        if node.type in ("function_declaration", "local_function"):
            n = find_func_name(node)
            if n:
                calls_by_func[n] = extract_calls_in(node)
        for child in node.children:
            process(child)

    process(tree.root_node)
    return defined, calls_by_func


def _build_lua_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    if not TREE_SITTER_LUA_AVAILABLE:
        return
    parser = _get_lua_parser()
    per_file = []
    global_defs: dict[str, str] = {}
    for lua_file in scan_project(root, "lua", workspace_config):
        lp = Path(lua_file)
        rel = str(lp.relative_to(root))
        defs, calls = _extract_lua_file_calls(lp, parser)
        per_file.append((rel, defs, calls))
        for n in defs:
            global_defs.setdefault(n, rel)
    for rel, defs, calls in per_file:
        for caller, clist in calls.items():
            for ctype, target in clist:
                if ctype == "intra":
                    graph.add_edge(rel, caller, rel, target)
                elif ctype == "direct":
                    if target in global_defs:
                        graph.add_edge(rel, caller, global_defs[target], target)
                elif ctype == "attr":
                    method = target  # already last segment
                    if method in defs:
                        graph.add_edge(rel, caller, rel, method)
                    elif method in global_defs:
                        graph.add_edge(rel, caller, global_defs[method], method)


# -----------------------------------------------------------------------------
# Luau — duplicate of Lua per T-3 (horizontal slice; manual propagation).
# Luau tree-sitter grammar is a superset of Lua; node names are typically
# identical. If divergence surfaces, fix here independently of Lua.
# -----------------------------------------------------------------------------

def _build_luau_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    if not TREE_SITTER_LUAU_AVAILABLE:
        return
    parser = _get_luau_parser()
    per_file = []
    global_defs: dict[str, str] = {}
    for lua_file in scan_project(root, "luau", workspace_config):
        lp = Path(lua_file)
        rel = str(lp.relative_to(root))
        defs, calls = _extract_lua_file_calls(lp, parser)  # same extractor
        per_file.append((rel, defs, calls))
        for n in defs:
            global_defs.setdefault(n, rel)
    for rel, defs, calls in per_file:
        for caller, clist in calls.items():
            for ctype, target in clist:
                if ctype == "intra":
                    graph.add_edge(rel, caller, rel, target)
                elif ctype == "direct":
                    if target in global_defs:
                        graph.add_edge(rel, caller, global_defs[target], target)
                elif ctype == "attr":
                    method = target
                    if method in defs:
                        graph.add_edge(rel, caller, rel, method)
                    elif method in global_defs:
                        graph.add_edge(rel, caller, global_defs[method], method)


# -----------------------------------------------------------------------------
# Scala — objects/defs.  apply/implicit limits are name-only; matches Java's
# name-only approach.
# -----------------------------------------------------------------------------

def _build_scala_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    _build_generic_call_graph(
        root, graph, "scala", _get_scala_parser, TREE_SITTER_SCALA_AVAILABLE,
        def_node_types={"function_definition", "function_declaration"},
        workspace_config=workspace_config,
        name_child_types={"identifier"},
        call_node_types={"call_expression"},
    )


# -----------------------------------------------------------------------------
# C++ — T-6: store BOTH bare `func` AND dotted `ns.func` in defined_names;
# normalize `::` to `.` in call targets so the join succeeds either way.
# Namespace resolution is name-only; overloaded symbols may produce multiple
# edges; dot-form canonical with bare-name fallback.
# -----------------------------------------------------------------------------

def _extract_cpp_file_calls(file_path: Path, parser=None):
    if not TREE_SITTER_CPP_AVAILABLE:
        return set(), {}
    try:
        source = file_path.read_bytes()
        if parser is None:
            parser = _get_cpp_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return set(), {}

    defined: set = set()
    calls_by_func: dict[str, list[tuple[str, str]]] = {}

    def get_text(node):
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    current_ns: list[str] = []

    def function_name_from_declarator(declarator):
        """Walk declarator to find function name. Returns (bare_name, qualified_name_or_None)."""
        # Common shapes:
        #   function_declarator -> identifier or qualified_identifier
        node = declarator
        while node is not None:
            if node.type == "identifier":
                return get_text(node), None
            if node.type == "qualified_identifier":
                # contains namespace_identifier and identifier (last segment)
                txt = get_text(node).replace("::", ".")
                last = txt.rsplit(".", 1)[-1]
                return last, txt
            if node.type == "function_declarator":
                # find child declarator
                next_node = None
                for child in node.children:
                    if child.type in ("identifier", "qualified_identifier", "function_declarator"):
                        next_node = child
                        break
                node = next_node
                continue
            # try first named child
            named = [c for c in node.children if c.is_named]
            if not named:
                return None, None
            node = named[0]
        return None, None

    def extract_call_target(call_node):
        for child in call_node.children:
            if child.type == "identifier":
                return ("direct", get_text(child))
            if child.type == "qualified_identifier":
                txt = get_text(child).replace("::", ".")
                return ("attr", txt)
            if child.type == "field_expression":
                txt = get_text(child)
                norm = txt.replace("->", ".")
                return ("attr", norm)
        return None

    def extract_calls_in(body_node):
        calls = []

        def visit(node):
            if node.type == "function_definition" and node is not body_node:
                return
            if node.type == "call_expression":
                r = extract_call_target(node)
                if r:
                    ctype, target = r
                    if ctype == "direct":
                        if target in defined:
                            calls.append(("intra", target))
                        else:
                            calls.append(("direct", target))
                    else:
                        # T-6: target is already dot-normalized
                        calls.append(("attr", target))
            for child in node.children:
                visit(child)

        visit(body_node)
        return calls

    def visit_all(node):
        nonlocal current_ns
        if node.type == "namespace_definition":
            name = None
            for child in node.children:
                if child.type == "namespace_identifier":
                    name = get_text(child)
                    break
            old = current_ns
            if name:
                current_ns = current_ns + [name]
            for child in node.children:
                visit_all(child)
            current_ns = old
            return
        if node.type == "function_definition":
            for child in node.children:
                if child.type == "function_declarator":
                    bare, qualified = function_name_from_declarator(child)
                    if bare:
                        defined.add(bare)
                        if qualified:
                            defined.add(qualified)
                        elif current_ns:
                            defined.add(".".join(current_ns + [bare]))
                        key = qualified if qualified else (".".join(current_ns + [bare]) if current_ns else bare)
                        calls_by_func[key] = extract_calls_in(node)
                    break
        for child in node.children:
            visit_all(child)

    visit_all(tree.root_node)
    return defined, calls_by_func


def _build_cpp_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    workspace_config: Optional[WorkspaceConfig] = None,
):
    if not TREE_SITTER_CPP_AVAILABLE:
        return
    parser = _get_cpp_parser()  # Create once; reuse across all files
    per_file = []
    global_defs: dict[str, str] = {}
    for cpp_file in scan_project(root, "cpp", workspace_config):
        cp = Path(cpp_file)
        rel = str(cp.relative_to(root))
        defs, calls = _extract_cpp_file_calls(cp, parser)
        per_file.append((rel, defs, calls))
        for n in defs:
            global_defs.setdefault(n, rel)
    for rel, defs, calls in per_file:
        for caller, clist in calls.items():
            for ctype, target in clist:
                if ctype == "intra":
                    graph.add_edge(rel, caller, rel, target)
                elif ctype == "direct":
                    # try dot-form match first (T-6), then bare
                    if target in global_defs:
                        graph.add_edge(rel, caller, global_defs[target], target)
                elif ctype == "attr":
                    # target is dot-canonical "ns.func"
                    if target in global_defs:
                        graph.add_edge(rel, caller, global_defs[target], target)
                    else:
                        # bare-name fallback
                        bare = target.rsplit(".", 1)[-1]
                        if bare in global_defs:
                            graph.add_edge(rel, caller, global_defs[bare], bare)
