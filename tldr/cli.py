#!/usr/bin/env python3
"""
TLDR-Code CLI - Token-efficient code analysis for LLMs.

Usage:
    tldr tree [path]                    Show file tree
    tldr structure [path]               Show code structure (codemaps)
    tldr search <pattern> [paths...]    Search files for pattern (grep-shaped: -i, --include, -m)
    tldr extract <file>                 Extract full file info
    tldr context <entry> [<entry> ...]  Get relevant context for LLM
    tldr cfg <file> <function>          Control flow graph
    tldr dfg <file> <function>          Data flow graph
    tldr slice <file> <func> <line>     Program slice
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Fix for Windows: Explicitly import tree-sitter bindings early to prevent
# silent DLL loading failures when running as a console script entry point.
if os.name == 'nt':
    try:
        import tree_sitter
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_typescript
    except ImportError:
        pass

from . import __version__
# Daemon ensure-up: daemon-backed subcommands ensure the per-project daemon (and
# shared model server) are running, then route through the daemon instead of
# loading the embedding model in-process.
from .daemon.ensure import ensure_daemon

# Subcommands that are served by the long-lived daemon. For these we call
# ensure_daemon(project) before dispatch so the in-memory index server handles
# the request (no per-invocation cold start, no in-process model load).
DAEMON_ROUTED_COMMANDS = {
    "search", "context", "extract", "semantic", "warm", "impact", "dead",
    "arch", "calls", "imports", "importers", "structure", "tree",
    "diagnostics", "change_impact",
}
# Dual-gate idiom: api.py uses self.language=='rust' (RelevantContext owns language);
# cli.py uses .endswith('.rs') (edge tuples have no language object).
from .api import SUPPORTED_CONTEXT_LANGUAGES, _serialize_call_graph_to_cache, _rust_display_name
from .cross_file_calls import (
    CALL_GRAPH_LANGUAGES,
    ELIXIR_ORPHAN_SENTINEL,
    RUBY_ORPHAN_SENTINEL,
    is_orphan_sentinel as _is_orphan_sentinel,
)
from .lang_constants import ALL_LANGUAGES, EXTENSION_TO_LANGUAGE, _resolve_device_arg


def _get_subprocess_detach_kwargs():
    """Get platform-specific kwargs for detaching subprocess."""
    import subprocess
    if os.name == 'nt':  # Windows
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    else:  # Unix (Mac/Linux)
        return {'start_new_session': True}

# Canonical list of supported languages for --lang choices (derived from semantic.py)
LANG_CHOICES = ["auto", *ALL_LANGUAGES]
LANG_CHOICES_WITH_ALL = [*LANG_CHOICES, "all"]

# SUPPORTED_CONTEXT_LANGUAGES is now imported from api.py (single source of truth)

# Choices accepted by `tldr context --lang`: 'auto', 'all', or any explicit
# supported language. Lifted to module scope so the argparse subparser stays
# readable and the choices list is reusable for tests/tooling.
CONTEXT_LANG_CHOICES = ["auto", "all", *sorted(SUPPORTED_CONTEXT_LANGUAGES)]


def _includes_to_extensions(includes: list[str] | None) -> set[str] | None:
    """Map ``--include`` glob values onto the extension-filter set.

    Supported shapes (case preserved; suffix matching stays exact):
    ``"*.py"`` -> ``".py"``; ``".py"`` -> ``".py"``; ``"py"`` -> ``".py"``.
    ``None``/empty -> ``None`` (no filter).

    Raises:
        ValueError: for any other glob shape (wildcard not in leading ``*.``
            position, path separators, multiple dots before ``*``).
    """
    if not includes:
        return None
    extensions: set[str] = set()
    for glob in includes:
        suffix = None
        if "/" not in glob and "\\" not in glob:
            if glob.startswith("*."):
                suffix = glob[2:]
            elif glob.startswith("."):
                suffix = glob[1:]
            else:
                suffix = glob
        if not suffix or any(c in suffix for c in "*?[]./\\"):
            raise ValueError(
                f"unsupported --include glob {glob!r}; use '*.EXT' or '.EXT'"
            )
        extensions.add("." + suffix)
    return extensions


def detect_language_from_extension(file_path: str) -> str:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the source file

    Returns:
        Language name (defaults to 'python' if unknown)
    """
    ext = Path(file_path).suffix.lower()
    tag = EXTENSION_TO_LANGUAGE.get(ext, 'python')
    # R-2: Bug-004 expanded EXTENSION_TO_LANGUAGE with non-code tags
    # ('shell', 'markdown', 'toml', ...). Callers feed this result to CFG/DFG/
    # slice extractors which only understand ALL_LANGUAGES members. Filter
    # non-code tags back to 'python' to restore the pre-fix fallback contract.
    if tag not in ALL_LANGUAGES:
        return 'python'
    return tag


def get_cached_languages(project_path: str | Path) -> list[str] | None:
    """Read cached languages from .tldr/languages.json if available.

    Returns the list of cached languages re-sorted with call-graph-supported
    languages first (consistent with _detect_project_languages sort), or None
    if no cache exists or the cache cannot be read.
    """
    lang_cache = Path(project_path) / ".tldr" / "languages.json"
    if lang_cache.exists():
        try:
            data = json.loads(lang_cache.read_text())
            langs = data.get("languages")
            if langs:
                # Re-sort with call-graph-supported languages first
                # to stay consistent with _detect_project_languages sort
                langs = sorted(langs, key=lambda l: (0 if l in CALL_GRAPH_LANGUAGES else 1, l))
            return langs
        except (json.JSONDecodeError, OSError):
            pass
    return None


class NoSupportedContextLanguagesError(Exception):
    """Raised when language detection finds source files but none of the detected
    languages are in SUPPORTED_CONTEXT_LANGUAGES (e.g., a Ruby+Elixir project).
    """

    def __init__(
        self,
        detected: list[str],
        supported: list[str],
        project_path: str | Path | None = None,
    ):
        self.detected = detected
        self.supported = frozenset(supported)
        self.project_path = project_path
        loc = f"'{project_path}'" if project_path is not None else "project"
        super().__init__(
            f"no supported context languages in {loc} "
            f"(found: {', '.join(detected) or '<none>'}; "
            f"supported: {', '.join(supported)})"
        )


def _resolve_context_languages(
    lang_arg: str,
    project_path: str | Path,
    respect_ignore: bool = True,
) -> list[str]:
    """Convert lang_arg + project_path into the ordered list of languages to
    probe for the `context` command.

    Branches:
      - "all"  → sorted(SUPPORTED_CONTEXT_LANGUAGES) unconditionally.
      - "auto" → languages cached for the project (or detected if no cache),
                 filtered to SUPPORTED_CONTEXT_LANGUAGES. If detection found
                 languages but none are supported, raises
                 NoSupportedContextLanguagesError. If truly empty, returns
                 ["python"] (consistent with resolve_language fallback).
      - explicit (e.g., "swift") → [lang_arg].
    """
    if lang_arg == "all":
        return sorted(SUPPORTED_CONTEXT_LANGUAGES)

    if lang_arg == "auto":
        detected = get_cached_languages(project_path) or []
        if not detected:
            from .semantic import _detect_project_languages
            detected = _detect_project_languages(
                Path(project_path), respect_ignore=respect_ignore
            ) or []
        supported = [l for l in detected if l in SUPPORTED_CONTEXT_LANGUAGES]
        if detected and not supported:
            raise NoSupportedContextLanguagesError(
                detected=detected,
                supported=sorted(SUPPORTED_CONTEXT_LANGUAGES),
                project_path=project_path,
            )
        return supported or ["python"]

    # Argparse `choices=` already restricts lang_arg to "all", "auto", or a
    # supported language. The "all" and "auto" branches return above, so by
    # the time we get here lang_arg must be in SUPPORTED_CONTEXT_LANGUAGES.
    if lang_arg not in SUPPORTED_CONTEXT_LANGUAGES:
        raise ValueError(
            f"_resolve_context_languages: unexpected lang_arg={lang_arg!r}; "
            "argparse choices should have rejected this upstream."
        )
    return [lang_arg]


def _show_first_run_tip():
    """Show a one-time tip about Swift support on first run."""
    marker = Path.home() / ".tldr_first_run"
    if marker.exists():
        return

    # Check if Swift is already installed
    try:
        import tree_sitter_swift
        # Swift already works, no tip needed
        marker.touch()
        return
    except ImportError:
        pass

    # Show tip
    import sys
    print("Tip: For Swift support, run: python -m tldr.install_swift", file=sys.stderr)
    print("     (This message appears once)", file=sys.stderr)
    print(file=sys.stderr)

    marker.touch()


def main():
    _show_first_run_tip()
    parser = argparse.ArgumentParser(
        prog="tldr",
        description="Token-efficient code analysis for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Version: %(prog)s """ + __version__ + """

Examples:
    tldr tree src/                      # File tree for src/
    tldr structure . --lang python      # Code structure for Python files
    tldr search "def process" .         # Search for pattern
    tldr extract src/main.py            # Full file analysis
    tldr context main --project .       # LLM context starting from main()
    tldr cfg src/main.py process        # Control flow for process()
    tldr slice src/main.py func 42      # Lines affecting line 42

Ignore Patterns:
    TLDR respects .tldrignore files (gitignore syntax).
    First run creates .tldrignore with sensible defaults.
    Use --ignore PATTERN to add patterns from CLI (repeatable).
    Use --no-ignore to bypass all ignore patterns.

Daemon:
    TLDR runs a per-project daemon for fast repeated queries.
    - Socket: /tmp/tldr-{hash}.sock (hash from project path)
    - Auto-shutdown: 30 minutes idle
    - Memory: ~50-100MB base, +500MB-1GB with semantic search

    Start explicitly:  tldr daemon start
    Check status:      tldr daemon status
    Stop:              tldr daemon stop

Semantic Search:
    First run downloads embedding model (1.3GB default).
    Use --model all-MiniLM-L6-v2 for smaller 80MB model.
    Set TLDR_AUTO_DOWNLOAD=1 to skip download prompts.
        """,
    )

    # Global flags
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--no-ignore",
        action="store_true",
        help="Ignore .tldrignore patterns (include all files)",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        metavar="PATTERN",
        help="Additional ignore patterns (gitignore syntax, can be repeated)",
    )

    # Shell completion support
    try:
        import shtab
        shtab.add_argument_to(parser, ["--print-completion", "-s"])
    except ImportError:
        pass  # shtab is optional

    subparsers = parser.add_subparsers(dest="command", required=True)

    # tldr tree [path]
    tree_p = subparsers.add_parser("tree", help="Show file tree")
    tree_p.add_argument("path", nargs="?", default=".", help="Directory to scan")
    tree_p.add_argument(
        "--ext", nargs="+", help="Filter by extensions (e.g., --ext .py .ts)"
    )
    tree_p.add_argument(
        "--show-hidden", action="store_true", help="Include hidden files"
    )
    tree_p.add_argument(
        "--depth", "--max-depth", dest="max_depth", type=int, default=None,
        help="Max directory depth (default: unlimited)",
    )

    # tldr structure [path]
    struct_p = subparsers.add_parser("structure", help="Show code structure (codemaps)")
    struct_p.add_argument("path", nargs="?", default=".", help="Directory to analyze")
    struct_p.add_argument(
        "--lang",
        default="auto",
        choices=LANG_CHOICES_WITH_ALL,
        help="Language to analyze (auto=use cached, all=detect all)",
    )
    struct_p.add_argument(
        "--max", type=int, default=50, help="Max files to analyze (default: 50)"
    )

    # tldr search <pattern> [paths...]
    search_p = subparsers.add_parser(
        "search",
        help="Search files for pattern",
        description=(
            "Search files for a regex pattern (grep-shaped: -i, multiple "
            "paths, --include, --exclude-dir, -m). Grep/BRE escapes like "
            r"'\|' are auto-normalized to their ERE meaning."
        ),
    )
    search_p.add_argument("pattern", help="Regex pattern to search")
    search_p.add_argument(
        "path", nargs="*", default=["."], help="Directories/files to search"
    )
    search_p.add_argument(
        "-i", "--ignore-case", action="store_true",
        help="Match case-insensitively",
    )
    search_p.add_argument(
        "--include", action="append", metavar="GLOB",
        help='File filter: "*.py" or ".py" (repeatable)',
    )
    search_p.add_argument(
        "--exclude-dir", action="append", metavar="GLOB",
        help="Skip directories whose name matches GLOB (repeatable)",
    )
    search_p.add_argument(
        "-C", "--context", type=int, default=0, help="Context lines around match"
    )
    search_p.add_argument(
        "-m", "--max-count", type=int, default=100, dest="max_count",
        help="Max total results (default: 100, 0=unlimited)",
    )
    search_p.add_argument(
        "--max-files", type=int, default=10000, help="Max files to scan (default: 10000)"
    )

    # tldr extract <file> [--class X] [--function Y] [--method Class.method]
    extract_p = subparsers.add_parser("extract", help="Extract full file info (warns if bare on >5 symbols)")
    extract_p.add_argument("file", help="File to analyze")
    extract_p.add_argument("--class", dest="filter_class", help="Filter to specific class")
    extract_p.add_argument("--function", dest="filter_function", help="Filter to specific function")
    extract_p.add_argument("--method", dest="filter_method", help="Filter to specific method (Class.method)")
    extract_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr context <entry> [<entry> ...]
    ctx_p = subparsers.add_parser("context", help="Get relevant context for LLM")
    ctx_p.add_argument(
        "entry", nargs="+", help="Entry point(s): function_name or Class.method"
    )
    ctx_p.add_argument("--project", default=".", help="Project root directory")
    ctx_p.add_argument("--depth", type=int, default=2, help="Call depth (default: 2)")
    ctx_p.add_argument(
        "--lang",
        default="auto",
        choices=CONTEXT_LANG_CHOICES,
        help="Language for call-graph context (auto=detect languages in project; "
             "all=probe every supported language regardless of detection; "
             "or specify one: python, typescript, javascript, go, rust, php, swift, java, ruby)",
    )

    # tldr cfg <file> <function>
    cfg_p = subparsers.add_parser("cfg", help="Control flow graph")
    cfg_p.add_argument("file", help="Source file")
    cfg_p.add_argument("function", help="Function name")
    cfg_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr dfg <file> <function>
    dfg_p = subparsers.add_parser("dfg", help="Data flow graph")
    dfg_p.add_argument("file", help="Source file")
    dfg_p.add_argument("function", help="Function name")
    dfg_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr slice <file> <function> <line>
    slice_p = subparsers.add_parser("slice", help="Program slice")
    slice_p.add_argument("file", help="Source file")
    slice_p.add_argument("function", help="Function name")
    slice_p.add_argument("line", type=int, help="Line number to slice from")
    slice_p.add_argument(
        "--direction",
        default="backward",
        choices=["backward", "forward"],
        help="Slice direction",
    )
    slice_p.add_argument("--var", help="Variable to track (optional)")
    slice_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr calls <path>
    calls_p = subparsers.add_parser("calls", help="Build cross-file call graph")
    calls_p.add_argument("path", nargs="?", default=".", help="Project root")
    calls_p.add_argument(
        "--lang",
        default="auto",
        choices=CONTEXT_LANG_CHOICES,
        help="Language (auto=cached, all=detect)",
    )

    # tldr impact <func> [path]
    impact_p = subparsers.add_parser(
        "impact", help="Find all callers of a function (reverse call graph)"
    )
    impact_p.add_argument("func", help="Function name to find callers of")
    impact_p.add_argument("path", nargs="?", default=None, help="Project root")
    impact_p.add_argument("--project", dest="project_path", default=".", help="Project root (alternative to positional path)")
    impact_p.add_argument("--depth", type=int, default=3, help="Max depth (default: 3)")
    impact_p.add_argument("--file", help="Filter by file containing this string")
    impact_p.add_argument(
        "--lang",
        default="auto",
        choices=CONTEXT_LANG_CHOICES,
        help="Language (auto=cached, all=detect)",
    )

    # tldr dead [path]
    dead_p = subparsers.add_parser("dead", help="Find unreachable (dead) code")
    dead_p.add_argument("path", nargs="?", default=".", help="Project root")
    dead_p.add_argument(
        "--entry", nargs="*", default=[], help="Additional entry point patterns"
    )
    dead_p.add_argument(
        "--lang",
        default="auto",
        choices=CONTEXT_LANG_CHOICES,
        help="Language (auto=cached, all=detect)",
    )

    # tldr arch [path]
    arch_p = subparsers.add_parser(
        "arch", help="Detect architectural layers from call patterns"
    )
    arch_p.add_argument("path", nargs="?", default=".", help="Project root")
    arch_p.add_argument(
        "--lang",
        default="auto",
        choices=CONTEXT_LANG_CHOICES,
        help="Language (auto=cached, all=detect)",
    )

    # tldr imports <file>
    imports_p = subparsers.add_parser(
        "imports", help="Parse imports from a source file"
    )
    imports_p.add_argument("file", help="Source file to analyze")
    imports_p.add_argument("--lang", default=None, help="Language (auto-detected from extension if not specified)")

    # tldr importers <module> [path]
    importers_p = subparsers.add_parser(
        "importers", help="Find all files that import a module (reverse import lookup)"
    )
    importers_p.add_argument("module", help="Module name to search for importers")
    importers_p.add_argument("path", nargs="?", default=".", help="Project root")
    importers_p.add_argument(
        "--lang",
        default="auto",
        choices=LANG_CHOICES,
        help="Language (auto=detect from project)",
    )

    # tldr change-impact [files...]
    impact_p = subparsers.add_parser(
        "change-impact", help="Find tests affected by changed files"
    )
    impact_p.add_argument(
        "files", nargs="*", help="Files to analyze (default: auto-detect from session/git)"
    )
    impact_p.add_argument(
        "--session", action="store_true", help="Use session-modified files (dirty_flag)"
    )
    impact_p.add_argument(
        "--git", action="store_true", help="Use git diff to find changed files"
    )
    impact_p.add_argument(
        "--git-base", default="HEAD~1", help="Git ref to diff against (default: HEAD~1)"
    )
    impact_p.add_argument(
        "--lang",
        default="auto",
        choices=LANG_CHOICES,
        help="Language (auto=detect from project)",
    )
    impact_p.add_argument(
        "--depth", type=int, default=5, help="Max call graph depth (default: 5)"
    )
    impact_p.add_argument(
        "--run", action="store_true", help="Actually run the affected tests"
    )

    # tldr diagnostics <file|path>
    diag_p = subparsers.add_parser(
        "diagnostics", help="Get type and lint diagnostics"
    )
    diag_p.add_argument("target", help="File or project directory to check")
    diag_p.add_argument(
        "--project", action="store_true", help="Check entire project (default: single file)"
    )
    diag_p.add_argument(
        "--no-lint", action="store_true", help="Skip linter, only run type checker"
    )
    diag_p.add_argument(
        "--format", choices=["json", "text"], default="json", help="Output format"
    )
    diag_p.add_argument("--lang", default=None, help="Override language detection")

    # tldr warm <path>
    warm_p = subparsers.add_parser(
        "warm", help="Pre-build call graph cache for faster queries"
    )
    warm_p.add_argument("path", help="Project root directory")
    warm_p.add_argument(
        "--background", action="store_true", help="Build in background process"
    )
    warm_p.add_argument(
        "--lang",
        default="all",
        choices=LANG_CHOICES_WITH_ALL,
        help="Language (default: auto-detect all)",
    )

    # tldr semantic index <path> / tldr semantic search <query>
    semantic_p = subparsers.add_parser(
        "semantic", help="Semantic code search using embeddings"
    )
    semantic_sub = semantic_p.add_subparsers(dest="action", required=True)

    # tldr semantic index [path]
    index_p = semantic_sub.add_parser("index", help="Build semantic index for project")
    index_p.add_argument("path", nargs="?", default=".", help="Project root")
    index_p.add_argument(
        "--lang",
        default="auto",
        choices=LANG_CHOICES_WITH_ALL,
        help="Language (auto=detect from project, 'all' for multi-language)",
    )
    index_p.add_argument(
        "--model",
        default=None,
        help="Embedding model: bge-large-en-v1.5 (1.3GB, default) or all-MiniLM-L6-v2 (80MB)",
    )
    index_p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "metal"],
        help="Compute device for embedding inference: 'cpu' or 'metal'. "
             "If omitted, falls back to TLDR_DEVICE env var "
             "(default: 'metal' on Apple Silicon, 'cpu' otherwise).",
    )
    index_p.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Force a full rebuild, ignoring all cached state.",
    )
    # --dirty-files is an internal optimization hint (path to a temp file listing
    # changed files) passed by the daemon's background reindex. Registered BEFORE
    # the daemon wiring (I-12) so the subprocess call is never rejected by argparse.
    # Correctness never depends on it: the full call graph + L1 text_hash gate run
    # regardless, so a missing/unreadable file just falls back to a full scan.
    index_p.add_argument(
        "--dirty-files",
        default=None,
        help=argparse.SUPPRESS,
    )

    # tldr semantic search <query>
    search_p = semantic_sub.add_parser("search", help="Search semantically")
    search_p.add_argument("query", help="Natural language query")
    # Optional positional project root (in addition to --path) so
    # `tldr semantic search "query" /path` works like other subcommands.
    search_p.add_argument(
        "path_pos", nargs="?", default=None,
        help="Project root (positional; overrides --path)",
    )
    search_p.add_argument("--path", default=".", help="Project root")
    search_p.add_argument("--k", type=int, default=5, help="Number of results")
    search_p.add_argument("--expand", action="store_true", help="Include call graph expansion")
    search_p.add_argument(
        "--lang",
        default="auto",
        choices=LANG_CHOICES_WITH_ALL,
        help="Language (auto=detect from project)",
    )
    search_p.add_argument(
        "--model",
        default=None,
        help="Embedding model (uses index model if not specified)",
    )
    search_p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "metal"],
        help="Compute device for embedding inference: 'cpu' or 'metal'. "
             "If omitted, falls back to TLDR_DEVICE env var "
             "(default: 'metal' on Apple Silicon, 'cpu' otherwise).",
    )

    # tldr daemon start/stop/status/query
    daemon_p = subparsers.add_parser(
        "daemon", help="Daemon management subcommands"
    )
    daemon_sub = daemon_p.add_subparsers(dest="action", required=True)

    # tldr daemon start [--project PATH]
    daemon_start_p = daemon_sub.add_parser("start", help="Start daemon for project (background)")
    daemon_start_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon stop [--project PATH]
    daemon_stop_p = daemon_sub.add_parser("stop", help="Stop daemon gracefully")
    daemon_stop_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon status [--project PATH]
    daemon_status_p = daemon_sub.add_parser("status", help="Check if daemon running")
    daemon_status_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon query CMD [--project PATH]
    daemon_query_p = daemon_sub.add_parser("query", help="Send raw JSON command to daemon")
    daemon_query_p.add_argument("cmd", help="Command to send (e.g., ping, status, search)")
    daemon_query_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr daemon notify FILES [--project PATH] — accepts one or more file paths
    daemon_notify_p = daemon_sub.add_parser("notify", help="Notify daemon of file change (triggers reindex at threshold)")
    daemon_notify_p.add_argument("files", nargs="+", help="Paths to changed files")
    daemon_notify_p.add_argument("--project", "-p", default=".", help="Project path (default: current directory)")

    # tldr doctor [--install LANG]
    doctor_p = subparsers.add_parser(
        "doctor", help="Check and install diagnostic tools (type checkers, linters)"
    )
    doctor_p.add_argument(
        "--install", metavar="LANG", help="Install missing tools for language (e.g., python, go)"
    )
    doctor_p.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    args = parser.parse_args()

    # `semantic search` accepts an optional positional project root that, when
    # provided, overrides the --path option.
    if getattr(args, "path_pos", None):
        args.path = args.path_pos

    def _routed_project(parsed) -> str | None:
        """Resolve the project path for a daemon-routed subcommand.

        Returns the ANCHORED project root (via ``_find_project_root``) so that
        every daemon consumer — ``ensure_daemon``, ``query_daemon``, daemon
        start/stop/status — all derive their socket hash from the same path.
        Without anchoring here, ``ensure_daemon`` would anchor internally (via
        ``_anchor_project``) and compute socket = md5(anchored root), while
        ``query_daemon`` would use the raw resolved subdir and compute a
        different socket hash → silent mismatch for deep ``--path`` args under
        SVN/non-.git roots.  ``_find_project_root`` is idempotent, so passing
        an already-anchored root is a safe no-op.

        Imported lazily (function-local) to mirror ``ensure.py``'s deferred
        import pattern and avoid pulling heavy ML deps at CLI startup.
        """
        from .semantic import _find_project_root

        for attr in ("project", "path"):
            val = getattr(parsed, attr, None)
            # Multi-path commands warm the daemon for the first path only
            # (warm-up is advisory; search/tree run in-process).
            if isinstance(val, (list, tuple)):
                val = val[0] if val else None
            if val:
                p = Path(val)
                raw = p.parent if p.is_file() else p
                return str(_find_project_root(raw.resolve()))
        file_val = getattr(parsed, "file", None)
        if file_val:
            return str(_find_project_root(Path(file_val).resolve().parent))
        return str(_find_project_root(Path(".").resolve()))

    # Daemon ensure-up: for daemon-backed subcommands, make sure the per-project
    # daemon is running before dispatch. Failures here are non-fatal — dispatch
    # falls through to the existing in-process path.
    if getattr(args, "command", None) in DAEMON_ROUTED_COMMANDS:
        _routed_proj = _routed_project(args)
        if _routed_proj is not None:
            try:
                ensure_daemon(_routed_proj)
            except Exception:
                pass

            # 'semantic search' is embedding-backed: route it through the daemon
            # so the model is loaded only in the daemon/server, never in this CLI
            # process. ('semantic index' is also delegated when the daemon's reindex
            # subprocess sets TLDR_USE_MODEL_SERVER=1, but stays in-process for
            # standalone `tldr semantic index` invocations.)
            if args.command == "semantic" and getattr(args, "action", None) == "search":
                try:
                    from .daemon.startup import query_daemon
                    result = query_daemon(
                        _routed_proj,
                        {
                            "cmd": "semantic",
                            "action": "search",
                            "query": getattr(args, "query", ""),
                            "k": getattr(args, "k", 10),
                            "expand": getattr(args, "expand", False),
                        },
                    )
                    print(json.dumps(result.get("results", result), indent=2))
                    return
                except Exception:
                    # Fall through to the in-process path on any routing failure.
                    pass

    # Import here to avoid slow startup for --help
    from .api import (
        build_project_call_graph,
        extract_file,
        get_cfg_context,
        get_code_structure,
        get_dfg_context,
        get_file_tree,
        get_imports,
        get_relevant_context,
        get_slice,
        scan_project_files,
        search as api_search,
    )
    from .analysis import (
        analyze_architecture,
        analyze_dead_code,
        analyze_impact,
    )
    from .dirty_flag import is_dirty, get_dirty_files, clear_dirty
    from .patch import patch_call_graph
    from .cross_file_calls import ProjectCallGraph

    def _get_or_build_graph(project_path, lang, build_fn):
        """Get cached graph with incremental patches, or build fresh.

        This implements P4 incremental updates:
        1. If no cache exists, do full build
        2. If cache exists but no dirty files, load cache
        3. If cache exists with dirty files, patch incrementally
        """
        import time
        project = Path(project_path).resolve()
        cache_dir = project / ".tldr" / "cache"
        cache_file = cache_dir / "call_graph.json"

        # Check if we have a cached graph
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text())
                
                # Validate cache language compatibility
                cache_langs = cache_data.get("languages", [])
                if cache_langs and lang not in cache_langs and lang != "all":
                    # Cache was built with different languages; rebuild
                    raise ValueError("Cache language mismatch")
                
                # Reconstruct graph from cache
                graph = ProjectCallGraph()
                for e in cache_data.get("edges", []):
                    graph.add_edge(e["from_file"], e["from_func"], e["to_file"], e["to_func"])

                # Check for dirty files
                if is_dirty(project):
                    dirty_files = get_dirty_files(project)
                    # Patch incrementally for each dirty file
                    for rel_file in dirty_files:
                        abs_file = project / rel_file
                        if abs_file.exists():
                            graph = patch_call_graph(graph, str(abs_file), str(project), lang=lang)

                    # Update cache with patched graph
                    _serialize_call_graph_to_cache(
                        cache_file, graph, cache_langs if cache_langs else [lang]
                    )

                    # Clear dirty flag
                    clear_dirty(project)

                return graph
            except (json.JSONDecodeError, KeyError, ValueError):
                # Invalid cache or language mismatch, fall through to fresh build
                pass

        # No cache or invalid cache - do fresh build
        graph = build_fn(project_path, language=lang)

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        _serialize_call_graph_to_cache(cache_file, graph, [lang])

        # Clear any dirty flag since we just rebuilt
        clear_dirty(project)

        return graph

    # Helper to load ignore patterns from .tldrignore + CLI --ignore flags + .gitignore
    def get_ignore_spec(project_path: str | Path):
        """Load ignore patterns, combining .tldrignore, .gitignore, and CLI --ignore flags."""
        if getattr(args, 'no_ignore', False):
            return None

        from .tldrignore import IgnoreSpec

        cli_patterns = getattr(args, 'ignore', None) or []
        return IgnoreSpec(
            project_dir=project_path,
            use_gitignore=True,
            cli_patterns=cli_patterns if cli_patterns else None,
        )

    def _resolve_device(args_device: str | None) -> str | None:
        """Resolve compute device: CLI arg > TLDR_DEVICE env > None (auto-pick).

        Validates TLDR_DEVICE if set. Exits with code 2 on invalid env value.
        Returns 'cpu', 'metal', 'mps' (the last only via a legacy ``TLDR_DEVICE``
        env value — it is not an argparse ``--device`` choice), or None (let
        downstream pick the default).
        """
        device = _resolve_device_arg(args_device)
        if device is None:
            # _resolve_device_arg only accepts recognised values; if TLDR_DEVICE was
            # set but not recognised, it returned None — validate and exit here.
            env_device = os.environ.get("TLDR_DEVICE")
            if env_device and env_device not in ("cpu", "metal"):
                print(
                    f"tldr: error: TLDR_DEVICE: invalid choice: {env_device!r} "
                    f"(choose from 'cpu', 'metal')",
                    file=sys.stderr,
                )
                sys.exit(2)
        return device

    def resolve_language(lang_arg: str, project_path: str | Path) -> str:
        """Resolve 'auto' to actual language (single language for non-context commands).

        Returns 'all' unchanged for multi-lang commands; otherwise picks the first
        language from _resolve_context_languages to maintain backward compatibility.
        """
        if lang_arg == "all":
            return "all"
        respect_ignore = not getattr(args, 'no_ignore', False)
        try:
            languages = _resolve_context_languages(
                lang_arg, Path(project_path).resolve(), respect_ignore=respect_ignore
            )
            return languages[0] if languages else "python"
        except NoSupportedContextLanguagesError:
            # For single-language commands, fall back to python if no supported langs found
            # (different from context command which exits with error)
            print("Warning: no supported languages detected, defaulting to python", file=sys.stderr)
            return "python"

    try:
        if args.command == "tree":
            ext = set(args.ext) if args.ext else None
            ignore_spec = get_ignore_spec(args.path)
            result = get_file_tree(
                args.path, extensions=ext, exclude_hidden=not args.show_hidden,
                ignore_spec=ignore_spec, max_depth=args.max_depth
            )
            print(json.dumps(result, indent=2))

        elif args.command == "structure":
            ignore_spec = get_ignore_spec(args.path)
            project_path = Path(args.path).resolve()

            # Determine language(s) to analyze
            if args.lang == "auto":
                # For single files, detect language from extension
                if project_path.is_file():
                    detected = detect_language_from_extension(str(project_path))
                    languages = [detected]
                else:
                    # Use cached languages, or detect if no cache
                    cached = get_cached_languages(project_path)
                    if cached:
                        languages = cached
                    else:
                        from .semantic import _detect_project_languages
                        respect_ignore = not getattr(args, 'no_ignore', False)
                        languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                        if not languages:
                            languages = ["python"]
            elif args.lang == "all":
                # Detect all languages in project
                from .semantic import _detect_project_languages
                respect_ignore = not getattr(args, 'no_ignore', False)
                languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                if not languages:
                    languages = ["python"]
            else:
                languages = [args.lang]

            # Collect results for all languages
            all_files = []
            for lang in languages:
                result = get_code_structure(
                    args.path, language=lang, max_results=args.max,
                    ignore_spec=ignore_spec
                )
                all_files.extend(result.get("files", []))

            combined_result = {
                "root": str(project_path),
                "languages": languages,
                "files": all_files[:args.max],  # Respect max across all languages
            }
            print(json.dumps(combined_result, indent=2))

        elif args.command == "search":
            paths = args.path or ["."]
            # Validate ALL paths before searching any.
            for p in paths:
                if not Path(p).exists():
                    print(f"Error: path '{p}' not found", file=sys.stderr)
                    sys.exit(1)
            try:
                ext = _includes_to_extensions(args.include)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                sys.exit(2)
            results = []
            for p in paths:
                # -m budget spans paths: each path gets the remainder.
                budget = (
                    0 if args.max_count == 0 else args.max_count - len(results)
                )
                if args.max_count > 0 and budget <= 0:
                    break
                sp = Path(p)
                # When sp is a single file, api_search ignores --include
                # (explicit file beats filter) and ignore_spec is rooted at
                # the parent.
                ignore_spec = get_ignore_spec(
                    str(sp.parent if sp.is_file() else sp)
                )
                hits = api_search(
                    args.pattern, p,
                    extensions=ext,
                    context_lines=args.context,
                    max_results=budget,
                    max_files=args.max_files,
                    ignore_spec=ignore_spec,
                    ignore_case=args.ignore_case,
                    exclude_dirs=args.exclude_dir,
                )
                if len(paths) > 1:
                    # A-1: api.search single-file mode sets "file" to the
                    # file's BASENAME — joining p onto it would yield
                    # "dir/foo.py/foo.py". For a file-typed path argument,
                    # the path argument itself IS the hit's identity; for a
                    # dir-typed path, prefix the root-relative path.
                    for h in hits:
                        h["file"] = (
                            p if sp.is_file() else os.path.join(p, h["file"])
                        )
                results.extend(hits)
            print(json.dumps(results, indent=2))

        elif args.command == "extract":
            # Apply filters if specified
            filter_class = getattr(args, "filter_class", None)
            filter_function = getattr(args, "filter_function", None)
            filter_method = getattr(args, "filter_method", None)

            if filter_class or filter_function or filter_method:
                # Filtered extract: use the helper that also injects a
                # 'code' (source span) field on matched symbols whose
                # extractor populated end_line.
                from .api import extract_file_with_code
                result = extract_file_with_code(
                    args.file,
                    function=filter_function,
                    method=filter_method,
                    class_=filter_class,
                )
            else:
                # Bare extract: metadata only, with warning for large files.
                result = extract_file(args.file)
                n_symbols = (
                    len(result.get("functions", []))
                    + sum(
                        len(c.get("methods", []))
                        for c in result.get("classes", [])
                    )
                    + len(result.get("classes", []))
                )
                if n_symbols > 5:  # Threshold for warning on bare extract with many symbols
                    sys.stderr.write(
                        f"tldr: extract dumped {n_symbols} symbols' metadata from {args.file}.\n"
                        f"      Pass --function NAME / --method Class.method / --class Name to filter.\n"
                    )

            print(json.dumps(result, indent=2))

        elif args.command == "context":
            from .api import get_relevant_context_multi
            project_path = Path(args.project).resolve()
            respect_ignore = not getattr(args, "no_ignore", False)
            try:
                languages = _resolve_context_languages(
                    args.lang, project_path, respect_ignore=respect_ignore,
                )
            except NoSupportedContextLanguagesError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            if len(args.entry) == 1:
                # Single-symbol path: byte-identical to the historical
                # contract (hit -> stdout, miss -> stderr + exit 1). Note the
                # args.entry[0] — passing the list would leak its repr into
                # the miss message.
                ctx = get_relevant_context_multi(
                    project_path,
                    args.entry[0],
                    depth=args.depth,
                    languages=languages,
                )
                if ctx.error:
                    print(ctx.to_llm_string(), file=sys.stderr)
                    sys.exit(1)
                # Output LLM-ready string directly
                print(ctx.to_llm_string())
            else:
                # Batch mode: one block per resolved symbol on stdout (one
                # blank line between blocks); per-symbol misses (with did-you-
                # mean) on stderr. Exit 0 if at least one symbol resolved,
                # 1 if all missed.
                any_resolved = False
                for entry in args.entry:
                    ctx = get_relevant_context_multi(
                        project_path,
                        entry,
                        depth=args.depth,
                        languages=languages,
                    )
                    if ctx.error:
                        print(ctx.to_llm_string(), file=sys.stderr)
                    else:
                        if any_resolved:
                            print()
                        print(ctx.to_llm_string())
                        any_resolved = True
                if not any_resolved:
                    sys.exit(1)

        elif args.command == "cfg":
            lang = args.lang or detect_language_from_extension(args.file)
            result = get_cfg_context(args.file, args.function, language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "dfg":
            lang = args.lang or detect_language_from_extension(args.file)
            result = get_dfg_context(args.file, args.function, language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "slice":
            lang = args.lang or detect_language_from_extension(args.file)
            lines = get_slice(
                args.file,
                args.function,
                args.line,
                direction=args.direction,
                variable=args.var,
                language=lang,
            )
            result = {"lines": sorted(lines), "count": len(lines)}
            print(json.dumps(result, indent=2))

        elif args.command == "calls":
            # Check for cached graph and dirty files for incremental update
            lang = resolve_language(args.lang, args.path)
            graph = _get_or_build_graph(args.path, lang, build_project_call_graph)
            # Filter Ruby/Elixir orphan-sentinel edges (canonical helper
            # `_is_orphan_sentinel` from cross_file_calls) on both e[1]
            # (from_func) and e[3] (to_func) positions before emission.
            filtered_edges = [
                e for e in graph.edges
                if not _is_orphan_sentinel(e[1]) and not _is_orphan_sentinel(e[3])
            ]
            # Rust display: dot→:: for .rs edges.  Per-edge .endswith(".rs")
            # guards handle all lang values uniformly: pure-Rust (all edges
            # convert), mixed (only .rs-side converts), and non-Rust (no edges
            # match the guard, all pass through unchanged).
            formatted_edges = [
                {
                    "from_file": e[0],
                    "from_func": _rust_display_name(e[1]) if e[0].endswith(".rs") else e[1],
                    "to_file": e[2],
                    "to_func": _rust_display_name(e[3]) if e[2].endswith(".rs") else e[3],
                }
                for e in filtered_edges
            ]
            result = {"edges": formatted_edges, "count": len(filtered_edges)}
            print(json.dumps(result, indent=2))

        elif args.command == "impact":
            # Support both positional path and --project flag
            project_root = args.path if args.path else args.project_path
            lang = resolve_language(args.lang, project_root)
            result = analyze_impact(
                project_root,
                args.func,
                max_depth=args.depth,
                target_file=args.file,
                language=lang,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "dead":
            lang = resolve_language(args.lang, args.path)
            result = analyze_dead_code(
                args.path,
                entry_points=args.entry if args.entry else None,
                language=lang,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "arch":
            lang = resolve_language(args.lang, args.path)
            result = analyze_architecture(args.path, language=lang)
            # Rust display: dot→:: in arch function fields for .rs files; skip
            # Ruby/Elixir orphan-sentinel entries (on either `file` or `function`)
            # BEFORE the .rs guard. B-2 defensive: orphans are dead code already
            # surfaced by `tldr dead`, not real entry points — preserve real
            # entry points like main() (caller=0, callee>0) which DO have a
            # real file path and non-sentinel function name.
            for layer_key in ("entry_layer", "leaf_layer"):
                processed = []
                for entry in result.get(layer_key, []):
                    if _is_orphan_sentinel(entry.get("file", "")):
                        continue
                    if _is_orphan_sentinel(entry.get("function", "")):
                        continue
                    if entry.get("file", "").endswith(".rs"):
                        entry["function"] = _rust_display_name(entry["function"])
                    processed.append(entry)
                if layer_key in result:
                    result[layer_key] = processed
            print(json.dumps(result, indent=2))

        elif args.command == "imports":
            file_path = Path(args.file).resolve()
            if not file_path.exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            lang = args.lang or detect_language_from_extension(args.file)
            result = get_imports(str(file_path), language=lang)
            print(json.dumps(result, indent=2))

        elif args.command == "importers":
            # Find all files that import the given module
            project = Path(args.path).resolve()
            if not project.exists():
                print(f"Error: Path not found: {args.path}", file=sys.stderr)
                sys.exit(1)

            # Scan all source files and check their imports
            lang = resolve_language(args.lang, args.path)
            respect_ignore = not getattr(args, 'no_ignore', False)
            files = scan_project_files(str(project), language=lang, respect_ignore=respect_ignore)
            importers = []
            for file_path in files:
                try:
                    imports = get_imports(file_path, language=lang)
                    for imp in imports:
                        module = imp.get("module", "")
                        names = imp.get("names", [])
                        # Check if module matches or if any imported name matches
                        if args.module in module or args.module in names:
                            importers.append({
                                "file": str(Path(file_path).relative_to(project)),
                                "import": imp,
                            })
                except Exception:
                    # Skip files that can't be parsed
                    pass

            print(json.dumps({"module": args.module, "importers": importers}, indent=2))

        elif args.command == "change-impact":
            from .change_impact import analyze_change_impact

            lang = resolve_language(args.lang, ".")
            result = analyze_change_impact(
                project_path=".",
                files=args.files if args.files else None,
                use_session=args.session,
                use_git=args.git,
                git_base=args.git_base,
                language=lang,
                max_depth=args.depth,
            )

            if args.run and result.get("test_command"):
                # Actually run the tests (test_command is a list to avoid shell injection)
                import shlex
                import subprocess as sp
                cmd = result["test_command"]
                print(f"Running: {shlex.join(cmd)}", file=sys.stderr)
                sp.run(cmd)  # No shell=True - safe from injection
            else:
                print(json.dumps(result, indent=2))

        elif args.command == "diagnostics":
            from .diagnostics import (
                get_diagnostics,
                get_project_diagnostics,
                format_diagnostics_for_llm,
            )

            target = Path(args.target).resolve()
            if not target.exists():
                print(f"Error: Target not found: {args.target}", file=sys.stderr)
                sys.exit(1)

            if args.project or target.is_dir():
                diag_lang = resolve_language(args.lang or "auto", str(target))
                result = get_project_diagnostics(
                    str(target),
                    language=diag_lang,
                    include_lint=not args.no_lint,
                )
            else:
                diag_lang = detect_language_from_extension(str(target)) if not args.lang or args.lang == "auto" else args.lang
                result = get_diagnostics(
                    str(target),
                    language=diag_lang,
                    include_lint=not args.no_lint,
                )

            if args.format == "text":
                print(format_diagnostics_for_llm(result))
            else:
                print(json.dumps(result, indent=2))

        elif args.command == "warm":
            import subprocess
            import time

            project_path = Path(args.path).resolve()

            # Validate path exists
            if not project_path.exists():
                print(f"Error: Path not found: {args.path}", file=sys.stderr)
                sys.exit(1)

            if args.background:
                # Spawn background process (cross-platform)
                subprocess.Popen(
                    [sys.executable, "-m", "tldr.cli", "warm", str(project_path), "--lang", "all" if args.lang == "auto" else args.lang],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **_get_subprocess_detach_kwargs(),
                )
                print(f"Background indexing spawned for {project_path}")
            else:
                # Build call graph
                from .cross_file_calls import scan_project, ProjectCallGraph
                from .tldrignore import ensure_tldrignore

                # Ensure .tldrignore exists (create with defaults if not)
                created, msg = ensure_tldrignore(project_path)
                if created:
                    print(msg)

                respect_ignore = not getattr(args, 'no_ignore', False)
                
                # Determine languages to process
                if args.lang in ("auto", "all"):
                    try:
                        from .semantic import _detect_project_languages
                        target_languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                        print(f"Detected languages: {', '.join(target_languages)}")
                    except ImportError:
                        # Fallback if semantic module issue
                        target_languages = ["python", "typescript", "javascript", "go", "rust"]
                else:
                    target_languages = [args.lang]

                all_files = set()
                combined_edges = []
                processed_languages = []
                
                for lang in target_languages:
                    try:
                        # Scan files
                        files = scan_project(project_path, language=lang, respect_ignore=respect_ignore)
                        all_files.update(files)
                        
                        # Build graph
                        graph = build_project_call_graph(project_path, language=lang)
                        combined_edges.extend([
                            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                            for e in graph.edges
                        ])
                        print(f"Processed {lang}: {len(files)} files, {len(graph.edges)} edges")
                        processed_languages.append(lang)
                    except ValueError as e:
                        # Expected for unsupported languages
                        print(f"Warning: {lang}: {e}", file=sys.stderr)
                    except Exception as e:
                        # Unexpected error - show traceback if debug enabled
                        print(f"Warning: Failed to process {lang}: {e}", file=sys.stderr)
                        if os.environ.get("TLDR_DEBUG"):
                            import traceback
                            traceback.print_exc()

                # Create cache directory
                cache_dir = project_path / ".tldr" / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Save cache file
                cache_file = cache_dir / "call_graph.json"
                # Deduplicate edges
                unique_edges = list({(e["from_file"], e["from_func"], e["to_file"], e["to_func"]): e for e in combined_edges}.values())
                
                cache_data = {
                    "edges": unique_edges,
                    "languages": processed_languages if processed_languages else target_languages,
                    "timestamp": time.time(),
                }
                cache_file.write_text(json.dumps(cache_data, indent=2))

                # Also save quick-access language cache for structure/search auto-detect
                lang_cache_file = project_path / ".tldr" / "languages.json"
                lang_cache_file.write_text(json.dumps({
                    "languages": processed_languages if processed_languages else target_languages,
                    "timestamp": time.time(),
                }, indent=2))

                # Print stats
                print(f"Total: Indexed {len(all_files)} files, found {len(unique_edges)} edges")

        elif args.command == "semantic":
            from .semantic import build_semantic_index, semantic_search
            from .embedding_backend import get_server_backed_default

            # Server delegation is opt-in here (env-gated by TLDR_USE_MODEL_SERVER):
            # the daemon's reindex subprocess sets it so this embeds via the shared
            # model server; a standalone `tldr semantic …` stays in-process by
            # default. Falls back silently to in-process if the server is down.
            _semantic_backend = get_server_backed_default()

            if args.action == "index":
                respect_ignore = not getattr(args, 'no_ignore', False)
                # Mirror the `search` action below: "auto"/"all" map to lang=None so
                # build_semantic_index -> extract_units_from_project(lang=None) takes
                # the multi-language expansion path (Goal C: index ALL languages by
                # default). resolve_language("auto") collapses to a SINGLE language,
                # which would silently index only that one language.
                lang = None if args.lang in ("auto", "all") else resolve_language(args.lang, args.path)
                device = _resolve_device(getattr(args, "device", None))

                # --dirty-files arg is an OPTIMIZATION HINT only: read the temp
                # file's JSON list of changed paths if present, and silently fall
                # back to a full scan if it is missing or unreadable (correctness
                # never depends on it — the full call graph + L1 text_hash gate
                # run regardless).
                dirty_files_path = getattr(args, "dirty_files", None)
                changed_files = None
                if dirty_files_path:
                    try:
                        with open(dirty_files_path, "r") as _df:
                            loaded = json.load(_df)
                        if isinstance(loaded, list):
                            changed_files = {str(p) for p in loaded}
                    except (OSError, ValueError):
                        # Missing or unreadable hint file: ignore and full-scan.
                        changed_files = None

                index_kwargs = dict(
                    lang=lang, model=args.model,
                    respect_ignore=respect_ignore, device=device,
                    full=getattr(args, "full", False),
                    backend=_semantic_backend,
                )
                if changed_files is not None:
                    index_kwargs["dirty_files"] = changed_files

                count = build_semantic_index(args.path, **index_kwargs)
                print(f"Indexed {count} code units")

            elif args.action == "search":
                lang = None if args.lang in ("auto", "all") else resolve_language(args.lang, args.path)
                device = _resolve_device(getattr(args, "device", None))
                results = semantic_search(
                    args.path,
                    args.query,
                    k=args.k,
                    expand_graph=args.expand,
                    model=args.model,
                    language=lang,
                    device=device,
                    backend=_semantic_backend,
                )
                print(json.dumps(results, indent=2))

        elif args.command == "doctor":
            import shutil
            import subprocess

            # Tool definitions: language -> (type_checker, linter, install_commands)
            TOOL_INFO = {
                "python": {
                    "type_checker": ("pyright", "pip install pyright  OR  npm install -g pyright"),
                    "linter": ("ruff", "pip install ruff"),
                },
                "typescript": {
                    "type_checker": ("tsc", "npm install -g typescript"),
                    "linter": None,
                },
                "javascript": {
                    "type_checker": None,
                    "linter": ("eslint", "npm install -g eslint"),
                },
                "go": {
                    "type_checker": ("go", "https://go.dev/dl/"),
                    "linter": ("golangci-lint", "brew install golangci-lint  OR  go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest"),
                },
                "rust": {
                    "type_checker": ("cargo", "https://rustup.rs/"),
                    "linter": ("cargo-clippy", "rustup component add clippy"),
                },
                "java": {
                    "type_checker": ("javac", "Install JDK: https://adoptium.net/"),
                    "linter": ("checkstyle", "brew install checkstyle  OR  download from checkstyle.org"),
                },
                "c": {
                    "type_checker": ("gcc", "xcode-select --install  OR  apt install gcc"),
                    "linter": ("cppcheck", "brew install cppcheck  OR  apt install cppcheck"),
                },
                "cpp": {
                    "type_checker": ("g++", "xcode-select --install  OR  apt install g++"),
                    "linter": ("cppcheck", "brew install cppcheck  OR  apt install cppcheck"),
                },
                "ruby": {
                    "type_checker": None,
                    "linter": ("rubocop", "gem install rubocop"),
                },
                "php": {
                    "type_checker": None,
                    "linter": ("phpstan", "composer global require phpstan/phpstan"),
                },
                "kotlin": {
                    "type_checker": ("kotlinc", "brew install kotlin  OR  sdk install kotlin"),
                    "linter": ("ktlint", "brew install ktlint"),
                },
                "swift": {
                    "type_checker": ("swiftc", "xcode-select --install"),
                    "linter": ("swiftlint", "brew install swiftlint"),
                },
                "csharp": {
                    "type_checker": ("dotnet", "https://dotnet.microsoft.com/download"),
                    "linter": None,
                },
                "scala": {
                    "type_checker": ("scalac", "brew install scala  OR  sdk install scala"),
                    "linter": None,
                },
                "elixir": {
                    "type_checker": ("elixir", "brew install elixir  OR  asdf install elixir"),
                    "linter": ("mix", "Included with Elixir"),
                },
                "lua": {
                    "type_checker": None,
                    "linter": ("luacheck", "luarocks install luacheck"),
                },
            }

            # Install commands for --install flag
            INSTALL_COMMANDS = {
                "python": ["pip", "install", "pyright", "ruff"],
                "go": ["go", "install", "github.com/golangci/golangci-lint/cmd/golangci-lint@latest"],
                "rust": ["rustup", "component", "add", "clippy"],
                "ruby": ["gem", "install", "rubocop"],
                "kotlin": ["brew", "install", "kotlin", "ktlint"],
                "swift": ["brew", "install", "swiftlint"],
                "lua": ["luarocks", "install", "luacheck"],
            }

            if args.install:
                lang = args.install.lower()
                if lang not in INSTALL_COMMANDS:
                    print(f"Error: No auto-install available for '{lang}'", file=sys.stderr)
                    print(f"Available: {', '.join(sorted(INSTALL_COMMANDS.keys()))}", file=sys.stderr)
                    sys.exit(1)

                cmd = INSTALL_COMMANDS[lang]
                print(f"Installing tools for {lang}: {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True)
                    print(f"✓ Installed {lang} tools")
                except subprocess.CalledProcessError as e:
                    print(f"✗ Install failed: {e}", file=sys.stderr)
                    sys.exit(1)
                except FileNotFoundError:
                    print(f"✗ Command not found: {cmd[0]}", file=sys.stderr)
                    sys.exit(1)
            else:
                # Check all tools
                results = {}
                for lang, tools in TOOL_INFO.items():
                    lang_result = {"type_checker": None, "linter": None}

                    if tools["type_checker"]:
                        tool_name, install_cmd = tools["type_checker"]
                        path = shutil.which(tool_name)
                        lang_result["type_checker"] = {
                            "name": tool_name,
                            "installed": path is not None,
                            "path": path,
                            "install": install_cmd if not path else None,
                        }

                    if tools["linter"]:
                        tool_name, install_cmd = tools["linter"]
                        path = shutil.which(tool_name)
                        lang_result["linter"] = {
                            "name": tool_name,
                            "installed": path is not None,
                            "path": path,
                            "install": install_cmd if not path else None,
                        }

                    results[lang] = lang_result

                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print("TLDR Diagnostics Check")
                    print("=" * 50)
                    print()

                    missing_count = 0
                    for lang, checks in sorted(results.items()):
                        has_issues = False
                        lines = []

                        tc = checks["type_checker"]
                        if tc:
                            if tc["installed"]:
                                lines.append(f"  ✓ {tc['name']} - {tc['path']}")
                            else:
                                lines.append(f"  ✗ {tc['name']} - not found")
                                lines.append(f"    → {tc['install']}")
                                has_issues = True
                                missing_count += 1

                        linter = checks["linter"]
                        if linter:
                            if linter["installed"]:
                                lines.append(f"  ✓ {linter['name']} - {linter['path']}")
                            else:
                                lines.append(f"  ✗ {linter['name']} - not found")
                                lines.append(f"    → {linter['install']}")
                                has_issues = True
                                missing_count += 1

                        if lines:
                            print(f"{lang.capitalize()}:")
                            for line in lines:
                                print(line)
                            print()

                    if missing_count > 0:
                        print(f"Missing {missing_count} tool(s). Run: tldr doctor --install <lang>")
                    else:
                        print("All diagnostic tools installed!")

        elif args.command == "daemon":
            from .daemon import start_daemon, stop_daemon, query_daemon
            from .semantic import _find_project_root

            # Anchor at the smart project root so start/stop/status/query/notify
            # (each defaulting --project to '.') all target the SAME socket as
            # ensure_daemon — otherwise `daemon stop` misses and orphans the
            # daemon that ensure_daemon started at the anchored root.
            project_path = _find_project_root(Path(args.project))

            if args.action == "start":
                # Ensure .tldr directory exists
                tldr_dir = project_path / ".tldr"
                tldr_dir.mkdir(parents=True, exist_ok=True)
                # Start daemon (will fork to background on Unix)
                start_daemon(project_path, foreground=False)

            elif args.action == "stop":
                if stop_daemon(project_path):
                    print("Daemon stopped")
                else:
                    print("Daemon not running")

            elif args.action == "status":
                try:
                    result = query_daemon(project_path, {"cmd": "status"})
                    print(f"Status: {result.get('status', 'unknown')}")
                    if 'uptime' in result:
                        uptime = int(result['uptime'])
                        mins, secs = divmod(uptime, 60)
                        hours, mins = divmod(mins, 60)
                        print(f"Uptime: {hours}h {mins}m {secs}s")
                except (ConnectionRefusedError, FileNotFoundError):
                    print("Daemon not running")

            elif args.action == "query":
                try:
                    result = query_daemon(project_path, {"cmd": args.cmd})
                    print(json.dumps(result, indent=2))
                except (ConnectionRefusedError, FileNotFoundError):
                    print("Error: Daemon not running", file=sys.stderr)
                    sys.exit(1)

            elif args.action == "notify":
                try:
                    # Deduplicate input paths before resolving to avoid redundant filesystem ops
                    unique_paths = list(dict.fromkeys(args.files))  # preserves order
                    file_paths = [str(Path(f).resolve()) for f in unique_paths]
                    result = query_daemon(project_path, {
                        "cmd": "notify",
                        "files": file_paths
                    })
                    if result.get("status") == "ok":
                        dirty_count = result.get("dirty_count", 0)
                        threshold = result.get("threshold", 20)
                        files_received = result.get("files_received", len(file_paths))
                        if result.get("reindex_triggered"):
                            print(f"Reindex triggered ({dirty_count}/{threshold} files, sent {files_received} file(s))")
                        else:
                            print(f"Tracked {files_received} file(s): {dirty_count}/{threshold} files")
                    else:
                        print(f"Error: {result.get('message', 'Unknown error')}", file=sys.stderr)
                        sys.exit(1)
                except (ConnectionRefusedError, FileNotFoundError):
                    # Daemon not running - silently ignore, file edits shouldn't fail
                    pass

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
