"""
Cross-file call graph resolution.

Builds a project-wide call graph that resolves function calls across files
by analyzing import statements and matching call sites to definitions.

Supports: Python (.py), TypeScript (.ts, .tsx), Go (.go), and Rust (.rs)

Key functions:
- scan_project(root, language) - find all source files in a project
- parse_imports(file) - extract import statements from a file
- build_function_index(root, language) - map {module.func: file_path} for all functions
- resolve_calls(file, index) - match call sites to definitions
- build_project_call_graph(root, language) - orchestrate all to build complete graph
"""

import ast
import os
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


@dataclass
class ProjectCallGraph:
    """Cross-file call graph with edges as (src_file, src_func, dst_file, dst_func)."""

    _edges: set[tuple[str, str, str, str]] = field(default_factory=set)

    def add_edge(self, src_file: str, src_func: str, dst_file: str, dst_func: str):
        """Add a call edge from src_file:src_func to dst_file:dst_func."""
        self._edges.add((src_file, src_func, dst_file, dst_func))

    @property
    def edges(self) -> set[tuple[str, str, str, str]]:
        """Return all edges as a set of tuples."""
        return self._edges

    def __contains__(self, edge: tuple[str, str, str, str]) -> bool:
        """Check if an edge exists in the graph."""
        return edge in self._edges


def _get_ts_parser():
    """Get or create a tree-sitter TypeScript parser."""
    if not TREE_SITTER_AVAILABLE:
        raise RuntimeError("tree-sitter-typescript not available")

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


def parse_ts_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a TypeScript file.

    Args:
        file_path: Path to TypeScript file

    Returns:
        List of import info dicts with keys: module, names, is_default, aliases
    """
    if not TREE_SITTER_AVAILABLE:
        return []

    file_path = Path(file_path)
    try:
        source = file_path.read_bytes()
        parser = _get_ts_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node):
        if node.type == "import_statement":
            import_info = _parse_ts_import_node(node, source)
            if import_info:
                imports.append(import_info)
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
        module_name = '/'.join(module_parts) if language == "typescript" else '.'.join(module_parts)

        # Also track the simple module name (last component)
        simple_module = rel_path.stem

        if language == "python":
            _index_python_file(src_path, rel_path, module_name, simple_module, index)
        elif language == "typescript":
            _index_typescript_file(src_path, rel_path, module_name, simple_module, index)
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


def _index_typescript_file(src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict):
    """Index functions and classes from a TypeScript file."""
    if not TREE_SITTER_AVAILABLE:
        return

    try:
        source = src_path.read_bytes()
        parser = _get_ts_parser()
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

        for child in node.children:
            walk_tree(child)

    walk_tree(tree.root_node)


def _get_ts_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a TypeScript AST node."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return source[child.start_byte:child.end_byte].decode("utf-8")
    return None


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


def _extract_ts_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
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
        parser = _get_ts_parser()
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
                    for vc in child.children:
                        if vc.type == "identifier":
                            defined_names.add(source[vc.start_byte:vc.end_byte].decode("utf-8"))
                            break
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

    func_index = build_function_index(root, language, workspace_config)

    if language == "python":
        _build_python_call_graph(root, graph, func_index, workspace_config)
    elif language == "typescript":
        _build_typescript_call_graph(root, graph, func_index, workspace_config)
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
        _build_elixir_call_graph(root, graph, func_index, workspace_config)

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
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for TypeScript files."""
    for ts_file in scan_project(root, "typescript", workspace_config):
        ts_path = Path(ts_file)
        rel_path = str(ts_path.relative_to(root))

        # Get imports for this file
        imports = parse_ts_imports(ts_path)

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
        calls_by_func = _extract_ts_file_calls(ts_path, root)

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
                            dst = _find_method_in_index(method_index, method)
                            if dst:
                                graph.add_edge(rel_path, caller_func, dst, method)


def _extract_elixir_file_calls(file_path: Path, root: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from an Elixir file, grouped by caller function.

    Args:
        file_path: Path to the Elixir source file.
        root: Project root (unused — kept for consistent _extract_*_file_calls signature).

    Returns:
        Dict mapping module-qualified caller name (e.g., "MyModule.func") to list of
        (call_type, call_target) tuples. Keys are scoped per defmodule block.
        call_type is one of:
          - 'intra': call to a function defined in the same module; call_target is the bare name.
          - 'local': bare function call not defined in the caller's module; call_target is the name.
          - 'qualified': Module.func() call; call_target is "AliasOrModule.func_name".
    """
    if not TREE_SITTER_ELIXIR_AVAILABLE:
        return {}

    try:
        source = file_path.read_bytes()
        parser = _get_elixir_parser()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    # Module-scoped defined function names: module_name -> set(func_names)
    defined_funcs: dict[str, set[str]] = {}

    # Keywords that are call nodes but NOT actual function calls
    _elixir_keywords = {
        "def", "defp", "defmodule", "defmacro", "defmacrop", "defguard", "defguardp",
        "defstruct", "defprotocol", "defimpl", "defexception", "defdelegate",
        "defoverridable", "defcallback", "defmacrocallback",
        "alias", "import", "use", "require",
        "if", "unless", "case", "cond", "with", "for", "try", "receive",
        "raise", "reraise", "throw", "exit",
        "quote", "unquote", "unquote_splicing",
        "super", "__MODULE__", "__DIR__", "__ENV__", "__CALLER__",
    }

    # Pass 1: Collect all defined function names, scoped by module
    current_module = None

    def collect_definitions(node):
        nonlocal current_module
        if node.type == "call":
            func_id = None
            for child in node.children:
                if child.type == "identifier":
                    func_id = source[child.start_byte:child.end_byte].decode("utf-8")
                    break
            if func_id == "defmodule":
                mod_name = _extract_elixir_module_name(node, source)
                if mod_name:
                    old_module = current_module
                    current_module = f"{current_module}.{mod_name}" if current_module else mod_name
                    if current_module not in defined_funcs:
                        defined_funcs[current_module] = set()
                    for child in node.children:
                        if child.type == "do_block":
                            for do_child in child.children:
                                collect_definitions(do_child)
                    current_module = old_module
                    return
            if func_id in ("def", "defp"):
                fname = _extract_elixir_func_name(node, source)
                if fname and current_module is not None:
                    defined_funcs[current_module].add(fname)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Pass 2: Extract calls from each def/defp body
    def extract_calls_from_body(body_node, module_defined_funcs: set[str]) -> list[tuple[str, str]]:
        """Walk the do_block of a function and extract call sites."""
        calls = []

        def visit(node):
            if node.type == "call":
                # Check for qualified call: call > dot > (alias, identifier)
                dot_child = None
                ident_child = None
                for child in node.children:
                    if child.type == "dot":
                        dot_child = child
                    elif child.type == "identifier" and dot_child is None:
                        ident_child = child

                if dot_child is not None:
                    # Qualified call: Module.func()
                    alias_text = None
                    func_name = None
                    for dc in dot_child.children:
                        if dc.type == "alias":
                            alias_text = source[dc.start_byte:dc.end_byte].decode("utf-8")
                        elif dc.type == "identifier":
                            func_name = source[dc.start_byte:dc.end_byte].decode("utf-8")
                    if alias_text and func_name:
                        calls.append(('qualified', f"{alias_text}.{func_name}"))
                elif ident_child is not None:
                    # Local call: func()
                    func_name = source[ident_child.start_byte:ident_child.end_byte].decode("utf-8")
                    if func_name not in _elixir_keywords:
                        if func_name in module_defined_funcs:
                            calls.append(('intra', func_name))
                        else:
                            calls.append(('local', func_name))

            for child in node.children:
                visit(child)

        visit(body_node)
        return calls

    # Pass 3: Walk the tree to find def/defp and extract their body calls
    current_module_p3 = None

    def process_functions(node):
        nonlocal current_module_p3
        if node.type == "call":
            func_id = None
            for child in node.children:
                if child.type == "identifier":
                    func_id = source[child.start_byte:child.end_byte].decode("utf-8")
                    break

            if func_id == "defmodule":
                mod_name = _extract_elixir_module_name(node, source)
                if mod_name:
                    old_module = current_module_p3
                    current_module_p3 = f"{current_module_p3}.{mod_name}" if current_module_p3 else mod_name
                    for child in node.children:
                        if child.type == "do_block":
                            for do_child in child.children:
                                process_functions(do_child)
                    current_module_p3 = old_module
                    return

            if func_id in ("def", "defp"):
                fname = _extract_elixir_func_name(node, source)
                if fname:
                    # Module-qualify the function key
                    qualified_key = f"{current_module_p3}.{fname}" if current_module_p3 else fname
                    # Get the defined funcs for the current module
                    module_funcs = defined_funcs.get(current_module_p3, set()) if current_module_p3 else set()
                    # Find the do_block
                    for child in node.children:
                        if child.type == "do_block":
                            body_calls = extract_calls_from_body(child, module_funcs)
                            # Merge with existing calls for this function (multiple clauses)
                            if qualified_key in calls_by_func:
                                calls_by_func[qualified_key].extend(body_calls)
                            else:
                                calls_by_func[qualified_key] = body_calls
                    return  # Don't recurse into the function body again

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _build_elixir_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None
):
    """Build call graph for Elixir files."""
    for ex_file in scan_project(root, "elixir", workspace_config):
        ex_path = Path(ex_file)
        rel_path = str(ex_path.relative_to(root))

        # Get imports for this file (scoped by defmodule)
        scoped_imports = parse_elixir_imports(ex_path)

        # Build per-scope alias maps and import sets
        # alias_map_by_scope: {defmodule_name: {short_name: full_module}}
        # import_modules_by_scope: {defmodule_name: {module_name: list[dict]}}
        alias_map_by_scope: dict[str, dict[str, str]] = {}
        import_modules_by_scope: dict[str, dict[str, list[dict]]] = {}

        for scope_name, imports in scoped_imports.items():
            alias_map = {}
            import_mods: dict[str, list[dict]] = {}
            for imp in imports:
                if imp.get('type') == 'alias':
                    full_module = imp['module']
                    if 'as' in imp:
                        alias_map[imp['as']] = full_module
                    else:
                        last_segment = full_module.rsplit('.', 1)[-1]
                        alias_map[last_segment] = full_module
                elif imp.get('type') in ('import', 'use'):
                    # Pre-build frozensets of function names for O(1) filter
                    # lookups instead of O(N) any() scans at call-resolution time.
                    #
                    # NOTE: only/except lists store (name, arity) tuples from
                    # the parser, but we match by name only because the call
                    # extractor (_extract_elixir_file_calls) does not track
                    # call-site argument counts. If arity tracking is added
                    # to the call extractor, the frozensets below can be
                    # replaced with arity-aware lookups.
                    imp = dict(imp)  # shallow copy to avoid mutating shared data
                    only_raw = imp.get('only')
                    except_raw = imp.get('except')
                    imp['only_names'] = (
                        frozenset(fn for fn, _arity in only_raw)
                        if only_raw is not None else None
                    )
                    imp['except_names'] = (
                        frozenset(fn for fn, _arity in except_raw)
                        if except_raw is not None else None
                    )
                    import_mods.setdefault(imp['module'], []).append(imp)
            alias_map_by_scope[scope_name] = alias_map
            import_modules_by_scope[scope_name] = import_mods

        # Extract calls from this file
        calls_by_func = _extract_elixir_file_calls(ex_path, root)

        # Sort scopes longest-first so the first prefix match is the most
        # specific (longest) scope, avoiding the need to scan all scopes and
        # track the longest seen so far.
        sorted_scopes = sorted(alias_map_by_scope.keys(), key=len, reverse=True)

        for caller_func, calls in calls_by_func.items():
            # Determine which defmodule scope the caller belongs to.
            # Iterate longest-prefix first and stop at the first match.
            caller_scope = None
            for scope_name in sorted_scopes:
                if caller_func.startswith(scope_name + ".") or caller_func == scope_name:
                    caller_scope = scope_name
                    break

            alias_map = alias_map_by_scope.get(caller_scope, {}) if caller_scope else {}
            import_mods = import_modules_by_scope.get(caller_scope, {}) if caller_scope else {}

            for call_type, call_target in calls:
                if call_type == 'intra':
                    # Same-file call to a locally defined function
                    # Module-qualify the target to match module-qualified caller keys
                    dot_pos = caller_func.rfind('.')
                    if dot_pos >= 0:
                        module_prefix = caller_func[:dot_pos + 1]
                        graph.add_edge(rel_path, caller_func, rel_path, module_prefix + call_target)
                    else:
                        graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == 'qualified':
                    # call_target is "AliasOrModule.func_name"
                    parts = call_target.rsplit('.', 1)
                    if len(parts) == 2:
                        module_ref, func_name = parts
                        # Resolve alias
                        resolved_module = alias_map.get(module_ref, module_ref)
                        qualified_name = f"{resolved_module}.{func_name}"

                        # Try to find in func_index
                        # No dst_file != rel_path guard — Elixir files can contain
                        # multiple defmodule blocks, so qualified cross-module calls
                        # within the same file are valid edges.
                        # 1. Try (resolved_module, func_name) tuple
                        key = (resolved_module, func_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, qualified_name)
                        # 2. Try the qualified string key
                        elif qualified_name in func_index:
                            dst_file = func_index[qualified_name]
                            graph.add_edge(rel_path, caller_func, dst_file, qualified_name)
                        else:
                            # 3. Try last segment of resolved module
                            last_seg = resolved_module.rsplit('.', 1)[-1]
                            key = (last_seg, func_name)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, qualified_name)

                elif call_type == 'local':
                    # Bare function call — could be from an imported/used module
                    func_name = call_target

                    # Try each imported module (consult only/except filters)
                    resolved = False
                    for imp_module, imp_dicts in import_mods.items():
                        for imp_dict in imp_dicts:
                            # Check only/except filters
                            only_names = imp_dict.get('only_names')
                            except_names = imp_dict.get('except_names')
                            if only_names is not None:
                                # Only allow functions in the only list
                                if func_name not in only_names:
                                    continue
                            if except_names is not None:
                                # Skip functions in the except list
                                if func_name in except_names:
                                    continue

                            key = (imp_module, func_name)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, f"{imp_module}.{func_name}")
                                resolved = True
                                break
                            # Try last segment of module
                            last_seg = imp_module.rsplit('.', 1)[-1]
                            key = (last_seg, func_name)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, f"{imp_module}.{func_name}")
                                resolved = True
                                break
                        if resolved:
                            break
                    # If not resolved through known imports, skip rather than
                    # guessing — a broad name-only scan would create false
                    # positives for common Elixir names (init, handle_call, etc.)
