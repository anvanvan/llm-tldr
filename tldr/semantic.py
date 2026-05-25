"""
Semantic search for code using 5-layer embeddings.

Embeds functions/methods using all 5 TLDR analysis layers:
- L1: Signature + docstring
- L2: Top callers + callees (from call graph)
- L3: Control flow summary
- L4: Data flow summary
- L5: Dependencies

Supports multiple embedding backends: sentence-transformers models (BGE, MiniLM)
and MLX-optimized models (Qwen3, Jina) for GPU-accelerated inference on Apple Silicon.
Uses FAISS for fast vector similarity search.
"""

import contextlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger("tldr.semantic")

# Module-level guarded MLX import — names exist as None on non-Apple platforms
# so tests can patch('tldr.semantic.mx', ...) regardless of install state.
try:
    import mlx.core as mx
    import mlx_embeddings
except ImportError:
    mx = None
    mlx_embeddings = None

# Module-level SentenceTransformer import for monkeypatching in tests.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

ALL_LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir"]

from tldr.cross_file_calls import CALL_GRAPH_LANGUAGES  # single source of truth
from tldr.api import NON_CODE_EXTENSIONS  # single source of truth for non-code suffixes
from tldr.dirty_flag import _normalize_file_path  # shared path normalization utility

# Cap embedded text per non-code file. Well above the BGE 512-token truncation
# window but bounded so pathological large config/log files don't bloat the index.
_NON_CODE_PREVIEW_CHARS = 8000

# Bug 004: short, query-aligned hints for well-known non-code files. Injected
# into the embedding text by build_embedding_text so that natural-language
# queries about install / setup / dependencies / CI surface README,
# pyproject.toml, requirements.txt, package.json, build.sh, etc. at the top.
#
# Keep these CONCISE — long "Description:" prose dilutes the BGE signal
# (measured: 0.5873 for verbose hint+Configuration label, 0.7439 for the
# concise "filename: keywords" form below, vs 0.7040 baseline with no hint
# and a misleading "Code:" prefix).
_NON_CODE_FILENAME_HINTS: dict[str, str] = {
    "readme.md": "project readme: overview installation usage",
    "readme.rst": "project readme: overview installation usage",
    "readme.txt": "project readme: overview installation usage",
    "readme": "project readme: overview installation usage",
    "contributing.md": "contributor guide: install dev dependencies run tests",
    "changelog.md": "changelog: release notes version history",
    "license": "license",
    "license.md": "license",
    "license.txt": "license",
    "pyproject.toml": "python project manifest: install dependencies metadata build",
    "setup.py": "python setup script: install dependencies package metadata",
    "setup.cfg": "python setup config: install dependencies package metadata",
    "requirements.txt": "python pip requirements: install dependencies",
    "requirements-dev.txt": "python pip dev requirements: install development dependencies",
    "package.json": "node.js package manifest: install dependencies scripts metadata",
    "package-lock.json": "node.js dependency lockfile",
    "cargo.toml": "rust crate manifest: install dependencies build config",
    "cargo.lock": "rust dependency lockfile",
    "go.mod": "go module manifest: install dependencies module path",
    "gemfile": "ruby bundler manifest: install dependencies",
    "podfile": "cocoapods manifest: install ios macos dependencies",
    "build.sh": "build shell script",
    "install.sh": "install shell script: install dependencies set up project",
    "dockerfile": "container image build instructions",
    "makefile": "gnu make build configuration",
}

# Extensionless build/manifest basenames (lowercased) that should reach the
# non-code embedding path even though their suffix is empty. Kept in sync with
# the matching entries in `_NON_CODE_FILENAME_HINTS` above.
_NON_CODE_EXTENSIONLESS_BASENAMES: frozenset[str] = frozenset(
    {"makefile", "dockerfile", "gemfile", "podfile"}
)

# Extension-to-language map (defined here to avoid circular import with cli.py)
EXTENSION_TO_LANGUAGE = {
    '.java': 'java',
    '.py': 'python',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.hpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hh': 'cpp',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.cs': 'csharp',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
    '.sc': 'scala',
    '.lua': 'lua',
    '.luau': 'luau',
    '.ex': 'elixir',
    '.exs': 'elixir',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.hxx': 'cpp',
    # Bug 004 (Gate 3): non-code extensions mapped to stand-in "language" tags
    # so _detect_project_languages returns a non-empty set for sh-only / doc-only
    # repos. These tags are NOT in ALL_LANGUAGES and are dispatched separately
    # (see NON_CODE_LANGUAGE_TAGS below); get_code_structure falls back to its
    # default code_extensions ({".py"}) and unions in NON_CODE_EXTENSIONS, so the
    # non-code files are still enumerated and reach _process_file_for_extraction
    # Gate 2.
    '.sh': 'shell',
    '.bash': 'shell',
    '.zsh': 'shell',
    '.toml': 'toml',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.json': 'json',
    '.md': 'markdown',
    '.rst': 'rst',
    '.txt': 'text',
}

# Bug 004 (Gate 3): stand-in "language" tags for non-code files. When a project
# has only non-code files (or `--lang all` is requested over such a tree),
# _detect_project_languages returns these tags so build_semantic_index dispatches
# extract_units_from_project at least once and reaches Gate 2.
NON_CODE_LANGUAGE_TAGS: set[str] = {
    "shell", "toml", "yaml", "json", "markdown", "rst", "text",
}

_HF_NOISE_SUPPRESSIONS = {
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}
# Lazy imports for heavy dependencies
_model = None
_model_name = None  # Track which model is loaded
_model_device = None  # Track which device the cached model is on

# Supported models with approximate download sizes
SUPPORTED_MODELS = {
    "bge-large-en-v1.5": {
        "hf_name": "BAAI/bge-large-en-v1.5",
        "size": "1.3GB",
        "dimension": 1024,
        "description": "High quality, recommended for production",
        "backend": "sentence-transformers",
    },
    "bge-base-en-v1.5": {
        "hf_name": "BAAI/bge-base-en-v1.5",
        "size": "440MB",
        "dimension": 768,
        "description": "MIT, ~3x smaller than bge-large, near-identical MTEB",
        "backend": "sentence-transformers",
    },
    "all-MiniLM-L6-v2": {
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "80MB",
        "dimension": 384,
        "description": "Lightweight, good for testing",
        "backend": "sentence-transformers",
    },
    "qwen3-0.6b-mlx": {
        "hf_name": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "size": "0.6GB",
        "dimension": 1024,
        "description": "Qwen3 0.6B 4-bit MLX, fast probe variant",
        "backend": "mlx",
        "mlx_batch": 8,
        "upstream_hf_name": None,
    },
    "qwen3-4b-mlx": {
        "hf_name": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        "size": "2.5GB",
        "dimension": 2560,
        "description": "Qwen3 4B 4-bit MLX, top code quality, batch=4 for 24GB safety",
        "backend": "mlx",
        "mlx_batch": 4,
        "upstream_hf_name": None,
    },
    "jina-v5-mlx": {
        "hf_name": "jinaai/jina-embeddings-v5-text-small-retrieval-mlx",
        "size": "0.6GB",
        "dimension": 768,
        "description": "Jina v5 text-small retrieval MLX, fastest",
        "backend": "mlx",
        "mlx_batch": 8,
        "upstream_hf_name": "jinaai/jina-embeddings-v5-text-small-retrieval",
    },
}

DEFAULT_MODEL = "bge-large-en-v1.5"

# Project root markers - files that indicate a project root
PROJECT_ROOT_MARKERS = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", ".tldr"]


def _find_project_root(start_path: Path) -> Path:
    """Find project root by walking up from start_path.

    Looks for common project markers (.git, pyproject.toml, etc.).
    Also respects CLAUDE_PROJECT_DIR environment variable.

    Args:
        start_path: Path to start searching from.

    Returns:
        Project root path, or start_path if no markers found.
    """
    # Check environment variable first
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        env_path = Path(env_root).resolve()
        if env_path.exists():
            return env_path

    # Walk up looking for project markers
    current = start_path.resolve()
    while current != current.parent:
        for marker in PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent

    # No markers found - use start_path
    return start_path.resolve()


@dataclass
class EmbeddingUnit:
    """A unit (function/method/class/file) for embedding.

    For code units, contains information from all 5 TLDR layers:
    - L1: signature, docstring
    - L2: calls, called_by
    - L3: cfg_summary
    - L4: dfg_summary
    - L5: dependencies

    For non-code files, contains whole-file preview and basic metadata.
    """
    name: str
    qualified_name: str
    file: str
    line: int
    language: str
    unit_type: str  # "function" | "method" | "class" | "file"
    signature: str
    docstring: str
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    cfg_summary: str = ""
    dfg_summary: str = ""
    dependencies: str = ""
    code_preview: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file": self.file,
            "line": self.line,
            "language": self.language,
            "unit_type": self.unit_type,
            "signature": self.signature,
            "docstring": self.docstring,
            "calls": self.calls,
            "called_by": self.called_by,
            "cfg_summary": self.cfg_summary,
            "dfg_summary": self.dfg_summary,
            "dependencies": self.dependencies,
            "code_preview": self.code_preview,
        }


MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Legacy, use SUPPORTED_MODELS


def _model_exists_locally(hf_name: str) -> bool:
    """Check if a model is already downloaded locally."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # Check if model config exists in cache
        result = try_to_load_from_cache(hf_name, "config.json")
        return result is not None
    except Exception:
        return False


def _confirm_download(model_key: str) -> bool:
    """Prompt user to confirm model download. Returns True if confirmed."""
    model_info = SUPPORTED_MODELS.get(model_key, {})
    size = model_info.get("size", "unknown size")
    hf_name = model_info.get("hf_name", model_key)

    # Skip prompt if TLDR_AUTO_DOWNLOAD is set or not a TTY
    if os.environ.get("TLDR_AUTO_DOWNLOAD") == "1":
        return True
    if not sys.stdin.isatty():
        # Non-interactive: warn but proceed
        print(f"⚠️  Downloading {hf_name} ({size})...", file=sys.stderr)
        return True

    print(f"\n⚠️  Semantic search requires embedding model: {hf_name}", file=sys.stderr)
    print(f"   Download size: {size}", file=sys.stderr)
    print(f"   (Set TLDR_AUTO_DOWNLOAD=1 to skip this prompt)\n", file=sys.stderr)

    try:
        response = input("Continue with download? [Y/n] ").strip().lower()
        return response in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


@contextlib.contextmanager
def _suppress_hf_noise() -> Iterator[None]:
    """Suppress HuggingFace/tqdm noise emitted during model weight loading.

    Sets env vars for libraries not yet imported, and calls the programmatic
    progress-bar disable APIs for both `huggingface_hub` and `transformers`
    (the latter is the source of the `Loading weights:` bar emitted by
    `transformers/core_model_loading.py` — env vars alone don't suppress it
    once `tqdm.auto` has been imported). If either import fails, suppression
    gracefully degrades (best-effort).

    TOKENIZERS_PARALLELISM is set defensively: ProcessPoolExecutor forks worker
    processes before get_model() is called, so no active tokenizer pool exists at
    this point. The env var is set preemptively in case a tokenizer is initialised
    inside the context body, not to suppress a warning that is already occurring.
    """
    saved = {key: os.environ.pop(key, None) for key in _HF_NOISE_SUPPRESSIONS}
    _bars_were_disabled = False
    _enable_progress_bars = None
    _tf_bar_was_enabled = False
    _tf_enable_progress_bar = None

    try:
        os.environ.update(_HF_NOISE_SUPPRESSIONS)
        # huggingface_hub.utils.tqdm caches HF_HUB_DISABLE_PROGRESS_BARS at import time,
        # so env vars alone are insufficient once it is imported. Use the programmatic API.
        try:
            from huggingface_hub.utils import (
                are_progress_bars_disabled,
                disable_progress_bars,
                enable_progress_bars,
            )
            _bars_were_disabled = are_progress_bars_disabled()
            if not _bars_were_disabled:
                disable_progress_bars()
                _enable_progress_bars = enable_progress_bars
        except (ImportError, AttributeError):
            pass  # best-effort: if import fails, skip progress-bar suppression
        # transformers has its own tqdm wrapper (`from tqdm.auto import tqdm` cached
        # at import time); HF_HUB_DISABLE_PROGRESS_BARS does not cover it. Disable
        # via the programmatic API and snapshot prior state so we restore correctly.
        try:
            from transformers.utils import logging as _tf_logging
            if _tf_logging.is_progress_bar_enabled():
                _tf_logging.disable_progress_bar()
                _tf_bar_was_enabled = True
                _tf_enable_progress_bar = _tf_logging.enable_progress_bar
        except (ImportError, AttributeError):
            pass
        yield
    finally:
        for key in _HF_NOISE_SUPPRESSIONS:
            val = saved[key]
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        if _enable_progress_bars is not None:
            _enable_progress_bars()
        if _tf_enable_progress_bar is not None and _tf_bar_was_enabled:
            _tf_enable_progress_bar()


def get_model(model_name: Optional[str] = None, *, device: Optional[str] = None):
    """Lazy-load the embedding model (cached).

    Args:
        model_name: Model key from SUPPORTED_MODELS, or None for default.
                   Can also be a full HuggingFace model name.
        device: 'cpu', 'metal', 'mps', or None for default behavior.

    Returns:
        SentenceTransformer model instance.

    Raises:
        ValueError: If model not found or user declines download.
        RuntimeError: If model has no upstream_hf_name for CPU fallback.
    """
    global _model, _model_name, _model_device

    # Default device from TLDR_DEVICE env if caller omitted it — ensures every
    # call site (including no-console embed fallback and daemon paths) honors
    # the requested device instead of letting PyTorch silently pick MPS.
    if device is None:
        env_device = os.environ.get("TLDR_DEVICE")
        if env_device in ("cpu", "metal", "mps"):
            device = env_device

    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Get HuggingFace name
    if model_name in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_name]["hf_name"]
    else:
        # Allow arbitrary HuggingFace model names
        hf_name = model_name

    # Return cached model if same (key includes device)
    if _model is not None and _model_name == hf_name and _model_device == device:
        return _model

    # Check if model needs downloading
    if not _model_exists_locally(hf_name):
        model_key = model_name if model_name in SUPPORTED_MODELS else None
        if model_key and not _confirm_download(model_key):
            raise ValueError(f"Model download declined. Use --model to choose a smaller model.")

    backend = "sentence-transformers"
    if model_name in SUPPORTED_MODELS:
        backend = SUPPORTED_MODELS[model_name].get("backend", "sentence-transformers")
    elif "-mlx" in hf_name.lower() or "/mlx-" in hf_name.lower() or hf_name.startswith("mlx-community/"):
        backend = "mlx"

    with _suppress_hf_noise():
        if backend == "mlx" and device == "cpu":
            # CPU fallback path for MLX-labeled models — requires upstream_hf_name.
            if model_name not in SUPPORTED_MODELS:
                raise RuntimeError(
                    f"Model {model_name!r} cannot run on --device cpu: no upstream_hf_name (non-MLX weights) defined"
                )
            upstream = SUPPORTED_MODELS[model_name].get("upstream_hf_name")
            if upstream is None:
                raise RuntimeError(
                    f"Model {model_name!r} cannot run on --device cpu: no upstream_hf_name (non-MLX weights) defined"
                )
            mlx_batch = SUPPORTED_MODELS[model_name].get("mlx_batch", 8)
            if os.environ.get("TLDR_FORCE_PYTORCH_CPU"):
                _model = SentenceTransformer(upstream, device="cpu", trust_remote_code=True)
            else:
                try:
                    _model = _MLXEmbedder(hf_name, mlx_batch=mlx_batch, device="cpu")
                except RuntimeError:
                    _model = SentenceTransformer(upstream, device="cpu", trust_remote_code=True)
        elif backend == "mlx":
            mlx_batch = 8
            if model_name in SUPPORTED_MODELS:
                mlx_batch = SUPPORTED_MODELS[model_name].get("mlx_batch", 8)
            _model = _MLXEmbedder(hf_name, mlx_batch=mlx_batch, device=device)
        else:
            if device == "cpu":
                _model = SentenceTransformer(hf_name, device="cpu", trust_remote_code=True)
            else:
                _model = SentenceTransformer(hf_name, trust_remote_code=True)
    _model_name = hf_name
    _model_device = device
    return _model


class _MLXEmbedder:
    """Wrapper around mlx-embeddings that mimics SentenceTransformer.encode()."""

    def __init__(self, hf_name: str, mlx_batch: int = 8, *, device: Optional[str] = None):
        if mx is None:
            raise RuntimeError("MLX not available — install mlx and mlx-embeddings")
        if device == "cpu":
            if not hasattr(mx, "cpu"):
                raise RuntimeError("mlx.cpu device not available in this MLX version")
            mx.set_default_device(mx.cpu)
        self._device = device
        self.hf_name = hf_name
        self.mlx_batch = mlx_batch
        self.model, self.tokenizer = mlx_embeddings.load(hf_name)

    def encode(self, texts, batch_size: int = 32,
               normalize_embeddings: bool = True,
               show_progress_bar: bool = False, **_kwargs):
        import numpy as np
        generate = mlx_embeddings.generate

        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        # MLX models (esp. autoregressive Qwen3) need cache clearing — without it,
        # the Metal allocator caches buffers across calls and balloons swap.
        mlx_batch = min(batch_size, self.mlx_batch)

        out_chunks = []
        for i in range(0, len(texts), mlx_batch):
            batch = list(texts[i:i + mlx_batch])
            result = generate(self.model, self.tokenizer, batch)
            embeds = result.text_embeds
            if normalize_embeddings:
                norms = mx.linalg.norm(embeds, axis=1, keepdims=True)
                embeds = mx.divide(embeds, mx.maximum(norms, 1e-12))
            embeds = embeds.astype(mx.float32)
            mx.eval(embeds)
            out_chunks.append(np.asarray(embeds))
            del result, embeds
            if (i // mlx_batch) % 16 == 15:
                if self._device != "cpu":
                    mx.metal.clear_cache()

        if self._device != "cpu":
            mx.metal.clear_cache()
        result_np = np.vstack(out_chunks) if len(out_chunks) > 1 else out_chunks[0]
        return result_np[0] if single else result_np


def build_embedding_text(unit: EmbeddingUnit) -> str:
    """Build rich text for embedding from all 5 layers.

    Creates a single text string containing information from all
    analysis layers, suitable for embedding with a language model.

    Args:
        unit: The EmbeddingUnit containing code analysis.

    Returns:
        A text string combining all layer information.
    """
    parts = []

    # L1: Signature + docstring
    if unit.signature:
        parts.append(f"Signature: {unit.signature}")
    if unit.docstring:
        parts.append(f"Description: {unit.docstring}")

    # L2: Call graph (forward - callees)
    if unit.calls:
        calls_str = ", ".join(unit.calls[:5])  # Top 5
        parts.append(f"Calls: {calls_str}")

    # L2: Call graph (backward - callers)
    if unit.called_by:
        callers_str = ", ".join(unit.called_by[:5])  # Top 5
        parts.append(f"Called by: {callers_str}")

    # L3: Control flow summary
    if unit.cfg_summary:
        parts.append(f"Control flow: {unit.cfg_summary}")

    # L4: Data flow summary
    if unit.dfg_summary:
        parts.append(f"Data flow: {unit.dfg_summary}")

    # L5: Dependencies
    if unit.dependencies:
        parts.append(f"Dependencies: {unit.dependencies}")

    # Code preview (first 10 lines of function body, or whole-file content for
    # non-code units). Bug 004: for file units (.md/.toml/.yaml/.sh/.json/...)
    # the preview is plain documentation/config text — labelling it "Code:"
    # biases the BGE embedding toward code-style queries. Drop the prefix for
    # file units; the docstring-derived hint above already supplies query-
    # aligned keywords ("install dependencies", "build configuration", ...).
    if unit.code_preview:
        if unit.unit_type == "file":
            parts.append(unit.code_preview)
        else:
            parts.append(f"Code:\n{unit.code_preview}")

    # Add name and type for context
    type_str = unit.unit_type if unit.unit_type else "function"
    parts.insert(0, f"{type_str.capitalize()}: {unit.name}")

    # Bug 004 follow-up: the docstring is already appended via the
    # ``Description:`` field above for every unit (including file units), so
    # we no longer also prepend it at position 0 — that double-inserted the
    # hint text for non-code file units. The BGE embedding still picks up the
    # hint via the ``Description:`` field; the leading ``File: <name>`` token
    # carries enough query alignment for filename-keyed queries.

    return "\n".join(parts)


def compute_embedding(text: str, model_name: Optional[str] = None, *, device: Optional[str] = None):
    """Compute embedding vector for text.

    Args:
        text: The text to embed.
        model_name: Model to use (from SUPPORTED_MODELS or HF name).
        device: Compute device ('cpu' or 'metal'). If None, get_model
            falls back to TLDR_DEVICE env or auto-pick.

    Returns:
        numpy array with L2-normalized embedding.
    """
    import numpy as np

    model = get_model(model_name, device=device)

    # BGE models work best with instruction prefix for queries
    # For document embedding, we use text directly
    embedding = model.encode(text, normalize_embeddings=True)

    return np.array(embedding, dtype=np.float32)


def extract_units_from_project(project_path: str, lang: str = "python", respect_ignore: bool = True, progress_callback=None) -> List[EmbeddingUnit]:
    """Extract all functions/methods/classes from a project.

    Uses existing TLDR APIs:
    - tldr.api.get_code_structure() for L1 (signatures)
    - tldr.cross_file_calls for L2 (call graph)
    - CFG/DFG extractors for L3/L4 summaries
    - tldr.api.get_imports for L5 (dependencies)

    Args:
        project_path: Path to project root.
        lang: Programming language ("python", "typescript", "go", "rust").
        respect_ignore: If True, respect .tldrignore patterns (default True).

    Returns:
        List of EmbeddingUnit objects with enriched metadata.
    """
    from tldr.api import get_code_structure, build_project_call_graph, get_imports
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    project = Path(project_path).resolve()
    units = []

    # Load ignore spec before getting structure
    ignore_spec = load_ignore_patterns(project) if respect_ignore else None

    # Get code structure (L1) - use high limit for semantic index
    structure = get_code_structure(str(project), language=lang, max_results=100000, ignore_spec=ignore_spec)

    # Filter ignored files
    if respect_ignore:
        spec = load_ignore_patterns(project)
        structure["files"] = [
            f for f in structure.get("files", [])
            if not should_ignore(project / f.get("path", ""), project, spec)
        ]

    # Build call graph (L2)
    try:
        call_graph = build_project_call_graph(str(project), language=lang)

        # Build call/called_by maps
        calls_map = {}  # func -> [called functions]
        called_by_map = {}  # func -> [calling functions]

        for edge in call_graph.edges:
            src_file, src_func, dst_file, dst_func = edge

            # Forward: src calls dst
            if src_func not in calls_map:
                calls_map[src_func] = []
            calls_map[src_func].append(dst_func)

            # Backward: dst is called by src
            if dst_func not in called_by_map:
                called_by_map[dst_func] = []
            called_by_map[dst_func].append(src_func)
    except Exception:
        # Call graph may not be available for all projects
        calls_map = {}
        called_by_map = {}

    # Process files in parallel for better performance
    files = structure.get("files", [])
    max_workers = int(os.environ.get("TLDR_MAX_WORKERS", os.cpu_count() or 4))

    # Use parallel processing if we have multiple files
    if len(files) > 1 and max_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_file_for_extraction,
                        file_info,
                        str(project),
                        lang,
                        calls_map,
                        called_by_map,
                    ): file_info
                    for file_info in files
                }

                for future in as_completed(futures):
                    file_info = futures[future]
                    try:
                        file_units = future.result(timeout=60)
                        units.extend(file_units)
                        if progress_callback:
                            progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
                    except Exception as e:
                        logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

        except Exception as e:
            logger.warning(f"Parallel extraction failed: {e}, falling back to sequential")
            for file_info in files:
                try:
                    file_units = _process_file_for_extraction(
                        file_info, str(project), lang, calls_map, called_by_map
                    )
                    units.extend(file_units)
                    if progress_callback:
                        progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
                except Exception as fe:
                    logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {fe}")
    else:
        for file_info in files:
            try:
                file_units = _process_file_for_extraction(
                    file_info, str(project), lang, calls_map, called_by_map
                )
                units.extend(file_units)
                if progress_callback:
                    progress_callback(file_info.get('path', 'unknown'), len(units), len(files))
            except Exception as e:
                logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

    return units


def _parse_file_ast(file_path: Path, lang: str) -> dict:
    """Parse file AST to extract line numbers and code previews.

    Returns:
        Dict with structure:
        {
            "functions": {func_name: {"line": int, "code_preview": str}},
            "classes": {class_name: {"line": int}},
            "methods": {"ClassName.method": {"line": int, "code_preview": str}}
        }
    """
    result = {"functions": {}, "classes": {}, "methods": {}}

    if not file_path.exists():
        return result

    try:
        content = file_path.read_text()
        lines = content.split('\n')

        if lang == "python":
            import ast
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Check if this is a method (inside a class)
                    parent_class = None
                    for potential_parent in ast.walk(tree):
                        if isinstance(potential_parent, ast.ClassDef):
                            if node in ast.walk(potential_parent) and node.name != potential_parent.name:
                                # Check if node is a direct child method
                                for item in potential_parent.body:
                                    if item is node:
                                        parent_class = potential_parent.name
                                        break

                    # Extract code preview (first 10 lines of body)
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    body_lines = lines[start_line - 1:min(end_line, start_line + 10) - 1]
                    code_preview = '\n'.join(body_lines[:10])

                    if parent_class:
                        result["methods"][f"{parent_class}.{node.name}"] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                    else:
                        result["functions"][node.name] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }

                elif isinstance(node, ast.ClassDef):
                    result["classes"][node.name] = {"line": node.lineno}

    except Exception:
        # Return empty result on any parsing error
        pass

    return result


def _get_file_dependencies(file_path: Path, lang: str) -> str:
    """Get file-level import dependencies as a string."""
    if not file_path.exists():
        return ""

    try:
        from tldr.api import get_imports
        imports = get_imports(str(file_path), language=lang)

        # Extract module names (limit to first 5 for brevity)
        modules = []
        for imp in imports[:5]:
            module = imp.get("module", "")
            if module:
                modules.append(module)

        return ", ".join(modules) if modules else ""
    except Exception:
        return ""


def _get_cfg_summary(file_path: Path, func_name: str, lang: str) -> str:
    """Get CFG summary (complexity, block count) for a function."""
    if not file_path.exists():
        return ""

    try:
        content = file_path.read_text()

        # Import the appropriate CFG extractor based on language
        from tldr import cfg_extractor

        extractor_map = {
            "python": cfg_extractor.extract_python_cfg,
            "typescript": cfg_extractor.extract_typescript_cfg,
            "javascript": cfg_extractor.extract_typescript_cfg,  # JS uses TS extractor
            "go": cfg_extractor.extract_go_cfg,
            "rust": cfg_extractor.extract_rust_cfg,
            "java": cfg_extractor.extract_java_cfg,
            "c": cfg_extractor.extract_c_cfg,
            "cpp": cfg_extractor.extract_cpp_cfg,
            "php": cfg_extractor.extract_php_cfg,
            "ruby": cfg_extractor.extract_ruby_cfg,
            "swift": cfg_extractor.extract_swift_cfg,
            "csharp": cfg_extractor.extract_csharp_cfg,
            "kotlin": cfg_extractor.extract_kotlin_cfg,
            "scala": cfg_extractor.extract_scala_cfg,
            "lua": cfg_extractor.extract_lua_cfg,
            "luau": cfg_extractor.extract_luau_cfg,
            "elixir": cfg_extractor.extract_elixir_cfg,
        }

        extractor = extractor_map.get(lang)
        if extractor:
            cfg = extractor(content, func_name)
            return f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
    except Exception:
        pass

    return ""


def _get_dfg_summary(file_path: Path, func_name: str, lang: str) -> str:
    """Get DFG summary (variable count, def-use chains) for a function."""
    if not file_path.exists():
        return ""

    try:
        content = file_path.read_text()

        # Import the appropriate DFG extractor based on language
        from tldr import dfg_extractor

        extractor_map = {
            "python": dfg_extractor.extract_python_dfg,
            "typescript": dfg_extractor.extract_typescript_dfg,
            "javascript": dfg_extractor.extract_typescript_dfg,  # JS uses TS extractor
            "go": dfg_extractor.extract_go_dfg,
            "rust": dfg_extractor.extract_rust_dfg,
            "java": dfg_extractor.extract_java_dfg,
            "c": dfg_extractor.extract_c_dfg,
            "cpp": dfg_extractor.extract_cpp_dfg,
            "php": dfg_extractor.extract_php_dfg,
            "ruby": dfg_extractor.extract_ruby_dfg,
            "swift": dfg_extractor.extract_swift_dfg,
            "csharp": dfg_extractor.extract_csharp_dfg,
            "kotlin": dfg_extractor.extract_kotlin_dfg,
            "scala": dfg_extractor.extract_scala_dfg,
            "lua": dfg_extractor.extract_lua_dfg,
            "luau": dfg_extractor.extract_luau_dfg,
            "elixir": dfg_extractor.extract_elixir_dfg,
        }

        extractor = extractor_map.get(lang)
        if extractor:
            dfg = extractor(content, func_name)

            # Count unique variables and def-use chains
            var_names = set()
            for ref in dfg.var_refs:
                var_names.add(ref.name)

            return f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
    except Exception:
        pass

    return ""


def _get_function_signature(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function signature from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    # Build signature from args
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    return f"def {func_name}({', '.join(args)}){returns}"


        # For other languages, return simple signature
        return f"function {func_name}(...)"

    except Exception:
        return None


def _get_function_docstring(file_path: Path, func_name: str, lang: str) -> Optional[str]:
    """Extract function docstring from file."""
    if not file_path.exists():
        return None

    try:
        content = file_path.read_text()

        if lang == "python":
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return ast.get_docstring(node)

        return None

    except Exception:
        return None


def _process_file_for_extraction(
    file_info: Dict[str, Any],
    project_path: str,
    lang: str,
    calls_map: Dict[str, List[str]],
    called_by_map: Dict[str, List[str]],
) -> List[EmbeddingUnit]:
    """Process a single file and extract all units. Top-level for pickling.

    This function reads the file ONCE and extracts all information in a single pass,
    avoiding the O(n*m) file read issue where n=files and m=functions.

    Args:
        file_info: Dict with 'path', 'functions', 'classes' from get_code_structure.
        project_path: Absolute path to project root.
        lang: Programming language.
        calls_map: Map of function name -> list of called functions.
        called_by_map: Map of function name -> list of calling functions.

    Returns:
        List of EmbeddingUnit objects for this file.
    """
    units = []
    project = Path(project_path)
    file_path = file_info.get("path", "")
    # Bug 004 R-5: when project_path is a single file (passed through from
    # build_semantic_index's scan_path), get_code_structure stores root.name as
    # file_path. Reconstruct full_path from the parent in that case; otherwise
    # the standard `project / file_path` directory-relative join.
    if project.is_file():
        full_path = project.parent / file_path
    else:
        full_path = project / file_path

    if not full_path.exists():
        return units

    try:
        # Read file content ONCE.
        # C-7: use utf-8-sig so a UTF-8 BOM (U+FEFF) is stripped instead of
        # polluting the embedding preview / first source line.
        # C-8: best-effort fallback to latin-1 for non-UTF-8 legacy config
        # files (common in older .yaml/.toml/.ini) so we don't silently drop
        # them via the broad except below.
        try:
            content = full_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            content = full_path.read_text(encoding="latin-1")
        lines = content.split('\n')
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return units

    # Gate 2 (Bug 004): non-code files (.sh, .md, .toml, .yaml, .yml, .json,
    # .rst, .txt, .zsh, .bash) have no functions/classes for the AST loop below
    # to emit. Gate 1 in api.py already includes them in structure["files"] via
    # _build_non_code_file_entry (empty functions/classes/methods/imports). Emit
    # one whole-file EmbeddingUnit so they appear in semantic search results.
    # Extensionless build/manifest files (Makefile, Dockerfile, Gemfile,
    # Podfile) have no suffix to match NON_CODE_EXTENSIONS, so the hint dict
    # entries for them would otherwise be unreachable; whitelist their basenames
    # (case-insensitive) so the non-code hint path still picks them up.
    if (
        full_path.suffix in NON_CODE_EXTENSIONS
        or full_path.name.lower() in _NON_CODE_EXTENSIONLESS_BASENAMES
    ):
        basename = full_path.name
        basename_lower = basename.lower()
        preview = content[:_NON_CODE_PREVIEW_CHARS]
        # Tag the unit with its actual document type (markdown/toml/yaml/shell/...)
        # rather than a generic "text" so query-time language filters and the
        # embedding text below can distinguish doc/config/build files from code.
        nc_language = EXTENSION_TO_LANGUAGE.get(full_path.suffix, "text")
        # Bug 004: inject a filename-derived description so well-known files
        # (README, pyproject.toml, requirements.txt, package.json, build.sh, ...)
        # have semantic context beyond their raw content. Without this hint,
        # BGE embeddings of pure config/manifest text often lose to .py units
        # that lexically mention "dependency" in a CS sense.
        nc_docstring = _NON_CODE_FILENAME_HINTS.get(basename_lower, "")
        unit = EmbeddingUnit(
            name=basename,
            qualified_name=file_path,
            file=file_path,
            line=1,
            language=nc_language,
            unit_type="file",
            signature=basename,
            docstring=nc_docstring,
            calls=[],
            called_by=[],
            cfg_summary="",
            dfg_summary="",
            dependencies="",
            code_preview=preview,
        )
        units.append(unit)
        return units

    # Parse AST once for all function info
    ast_info = {"functions": {}, "classes": {}, "methods": {}}
    all_signatures = {}
    all_docstrings = {}

    if lang == "python":
        try:
            import ast
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if this is a method (inside a class)
                    parent_class = None
                    for potential_parent in ast.walk(tree):
                        if isinstance(potential_parent, ast.ClassDef):
                            for item in potential_parent.body:
                                if item is node:
                                    parent_class = potential_parent.name
                                    break

                    # Extract code preview (first 10 lines of body)
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    body_lines = lines[start_line - 1:min(end_line, start_line + 10) - 1]
                    code_preview = '\n'.join(body_lines[:10])

                    # Build signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)
                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"
                    signature = f"def {node.name}({', '.join(args)}){returns}"

                    # Get docstring
                    docstring = ast.get_docstring(node) or ""

                    if parent_class:
                        key = f"{parent_class}.{node.name}"
                        ast_info["methods"][key] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                        all_signatures[key] = signature
                        all_docstrings[key] = docstring
                    else:
                        ast_info["functions"][node.name] = {
                            "line": node.lineno,
                            "code_preview": code_preview
                        }
                        all_signatures[node.name] = signature
                        all_docstrings[node.name] = docstring

                elif isinstance(node, ast.ClassDef):
                    ast_info["classes"][node.name] = {"line": node.lineno}

        except Exception as e:
            logger.debug(f"AST parse failed for {file_path}: {e}")

    elif lang == "php":
        try:
            from tldr.cross_file_calls import _get_php_parser

            parser = _get_php_parser()
            if parser:
                content_bytes = content.encode("utf-8")
                tree = parser.parse(content_bytes)

                # F3: Track active PHP namespace for qualified names
                _php_active_ns = ""

                def _walk_php(node):
                    nonlocal _php_active_ns
                    # F3: Detect namespace_definition and update active namespace
                    if node.type == "namespace_definition":
                        ns_name_node = node.child_by_field_name("name")
                        if ns_name_node:
                            _php_active_ns = content_bytes[ns_name_node.start_byte:ns_name_node.end_byte].decode("utf-8", errors="replace")

                    if node.type in ("function_definition", "function_declaration", "method_declaration"):
                        func_name = None
                        params = []
                        for child in node.children:
                            if child.type == "name":
                                func_name = content_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                            elif child.type == "formal_parameters":
                                for p in child.children:
                                    if p.type in ("simple_parameter", "variadic_parameter"):
                                        params.append(content_bytes[p.start_byte:p.end_byte].decode("utf-8", errors="replace").strip())

                        if func_name:
                            start_line = node.start_point[0] + 1
                            sig = f"function {func_name}({', '.join(params)})"
                            # Walk up to the enclosing class/interface/trait, if any
                            parent_class = None
                            p = node.parent
                            while p and p.type not in ("class_declaration", "interface_declaration", "trait_declaration", "program"):
                                p = p.parent
                            if p and p.type in ("class_declaration", "interface_declaration", "trait_declaration"):
                                name_node = p.child_by_field_name("name")
                                if name_node:
                                    parent_class = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")

                            if parent_class:
                                # F3: Qualify parent class with namespace
                                fq_parent = f"{_php_active_ns}\\{parent_class}" if _php_active_ns else parent_class
                                key = f"{fq_parent}.{func_name}"
                                ast_info["methods"][key] = {"line": start_line, "code_preview": ""}
                                all_signatures[key] = sig
                                all_docstrings[key] = ""
                            else:
                                # F3: Qualify top-level function with namespace
                                fq_func = f"{_php_active_ns}\\{func_name}" if _php_active_ns else func_name
                                ast_info["functions"][fq_func] = {"line": start_line, "code_preview": ""}
                                all_signatures[fq_func] = sig
                                all_docstrings[fq_func] = ""

                    elif node.type in ("class_declaration", "interface_declaration", "trait_declaration"):
                        name_node = node.child_by_field_name("name")
                        if name_node:
                            cls_name = content_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                            # F4: Qualify with namespace and store kind
                            fq_cls = f"{_php_active_ns}\\{cls_name}" if _php_active_ns else cls_name
                            kind = node.type.replace("_declaration", "")  # class, interface, trait
                            ast_info["classes"][fq_cls] = {"line": node.start_point[0] + 1, "kind": kind}

                    for child in node.children:
                        _walk_php(child)

                _walk_php(tree.root_node)
        except Exception as e:
            logger.debug(f"PHP AST parse failed for {file_path}: {e}")

    # Get dependencies (imports) - single call
    dependencies = ""
    try:
        from tldr.api import get_imports
        imports = get_imports(str(full_path), language=lang)
        modules = [imp.get("module", "") for imp in imports[:5] if imp.get("module")]
        dependencies = ", ".join(modules)
    except Exception:
        pass

    # Pre-compute CFG/DFG for all functions at once
    cfg_cache = {}
    dfg_cache = {}

    # Language-to-extractor mapping for CFG/DFG analysis
    def _get_extractors(language: str):
        """Return (cfg_extractor, dfg_extractor) for the given language."""
        if language == "python":
            from tldr.cfg_extractor import extract_python_cfg
            from tldr.dfg_extractor import extract_python_dfg
            return extract_python_cfg, extract_python_dfg
        elif language in ("typescript", "javascript"):
            from tldr.cfg_extractor import extract_typescript_cfg
            from tldr.dfg_extractor import extract_typescript_dfg
            return extract_typescript_cfg, extract_typescript_dfg
        elif language == "php":
            from tldr.cfg_extractor import extract_php_cfg
            from tldr.dfg_extractor import extract_php_dfg
            return extract_php_cfg, extract_php_dfg
        return None, None

    cfg_extractor, dfg_extractor = _get_extractors(lang)

    if cfg_extractor and dfg_extractor:
        # Get all function names we need to process
        all_func_names = list(file_info.get("functions", []))
        for class_info in file_info.get("classes", []):
            if isinstance(class_info, dict):
                all_func_names.extend(class_info.get("methods", []))
            else:
                # F5: Derive methods from ast_info for string-type class_info (PHP path)
                _prefix = f"{class_info}."
                # Also check FQ variants: any key ending with "\ClassName.method"
                _fq_suffix = f"\\{class_info}."
                for mk in ast_info.get("methods", {}):
                    if mk.startswith(_prefix) or _fq_suffix in mk:
                        all_func_names.append(mk.split(".")[-1])

        for func_name in all_func_names:
            try:
                cfg = cfg_extractor(content, func_name)
                cfg_cache[func_name] = f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
            except Exception:
                cfg_cache[func_name] = ""

            try:
                dfg = dfg_extractor(content, func_name)
                var_names = {ref.name for ref in dfg.var_refs}
                dfg_cache[func_name] = f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
            except Exception:
                dfg_cache[func_name] = ""

    # Build suffix -> list[(mkey, mval)] reverse lookup for O(1) method fallback
    _ast_methods = ast_info.get("methods", {})
    _method_by_suffix: dict[str, list] = {}
    for _mk, _mv in _ast_methods.items():
        _dot = _mk.rfind(".")
        if _dot != -1:
            _method_by_suffix.setdefault(_mk[_dot:], []).append((_mk, _mv))  # key is ".methodName"

    # Pre-build bare_name -> fq_key reverse maps for O(1) PHP FQ-name resolution
    _ast_funcs = ast_info.get("functions", {})
    _func_by_bare: dict[str, str] = {}
    for _fk in _ast_funcs:
        _bare = _fk.rsplit("\\", 1)[-1]
        if _bare not in _func_by_bare:
            _func_by_bare[_bare] = _fk

    _ast_classes = ast_info.get("classes", {})
    _cls_by_bare: dict[str, tuple[str, dict]] = {}
    for _ck, _cv in _ast_classes.items():
        _bare = _ck.rsplit("\\", 1)[-1]
        if _bare not in _cls_by_bare:
            _cls_by_bare[_bare] = (_ck, _cv)

    # Process functions
    file_stem = Path(file_path).stem
    for _bare_func in file_info.get("functions", []):
        # F3: Resolve bare PHP function name to FQ name from ast_info
        func_name = _bare_func
        func_info = _ast_funcs.get(func_name) or {}
        if not func_info:
            _fq = _func_by_bare.get(func_name)
            if _fq:
                func_name = _fq
                func_info = _ast_funcs[_fq]
        sig = all_signatures.get(func_name)
        # Fallback: check ast_info["methods"] for "*.func_name" compound keys (O(1))
        if not func_info or not sig:
            hits = _method_by_suffix.get(f".{func_name}")
            if hits:
                # Prefer the hit whose class belongs to the current file
                if len(hits) > 1:
                    file_hits = [h for h in hits if h[0].split(".")[0].rsplit("\\", 1)[-1] == file_stem]
                    if file_hits:
                        hits = file_hits
                mkey, mval = hits[0]
                if not func_info:
                    func_info = mval
                if not sig:
                    sig = all_signatures.get(mkey)
        if not sig:
            kw = "function" if lang == "php" else "def"
            sig = f"{kw} {func_name}(...)"
        unit = EmbeddingUnit(
            name=func_name,
            qualified_name=f"{file_path.replace('/', '.')}.{func_name}",
            file=file_path,
            line=func_info.get("line", 1),
            language=lang,
            unit_type="function",
            signature=sig,
            docstring=all_docstrings.get(func_name, ""),
            calls=(calls_map.get(func_name) or calls_map.get(func_name.rsplit("\\", 1)[-1], []))[:5],
            called_by=(called_by_map.get(func_name) or called_by_map.get(func_name.rsplit("\\", 1)[-1], []))[:5],
            cfg_summary=cfg_cache.get(func_name, ""),
            dfg_summary=dfg_cache.get(func_name, ""),
            dependencies=dependencies,
            code_preview=func_info.get("code_preview", ""),
        )
        units.append(unit)

    # Process classes
    for class_info in file_info.get("classes", []):
        if isinstance(class_info, dict):
            class_name = class_info.get("name", "")
            methods = class_info.get("methods", [])
        else:
            class_name = class_info
            # F3/F4: Resolve bare class name to FQ name from ast_info
            _cls_data = _ast_classes.get(class_name)
            if not _cls_data:
                # O(1) lookup via pre-built bare_name -> (fq_key, data) map
                _cls_hit = _cls_by_bare.get(class_name)
                if _cls_hit:
                    class_name, _cls_data = _cls_hit
            # Derive methods from ast_info["methods"] keys matching "ClassName.method"
            prefix = f"{class_name}."
            methods = [k[len(prefix):] for k in ast_info.get("methods", {}) if k.startswith(prefix)]

        _cls_data_for_line = ast_info.get("classes", {}).get(class_name, {})
        class_line = _cls_data_for_line.get("line", 1) if isinstance(_cls_data_for_line, dict) else 1
        # F4: Use kind from ast_info to distinguish class/interface/trait in signature
        _kind = "class"
        if isinstance(_cls_data_for_line, dict) and "kind" in _cls_data_for_line:
            _kind = _cls_data_for_line["kind"]

        # Add class itself
        unit = EmbeddingUnit(
            name=class_name,
            qualified_name=f"{file_path.replace('/', '.')}.{class_name}",
            file=file_path,
            line=class_line,
            language=lang,
            unit_type="class",
            signature=f"{_kind} {class_name}",
            docstring="",
            calls=[],
            called_by=[],
            cfg_summary="",
            dfg_summary="",
            dependencies=dependencies,
            code_preview="",
        )
        units.append(unit)

        # Add methods
        _sig_kw = "function" if lang == "php" else "def"
        _sig_args = "..." if lang == "php" else "self, ..."
        for method in methods:
            method_key = f"{class_name}.{method}"
            method_info = ast_info.get("methods", {}).get(method_key, {})

            _bare_cls = class_name.rsplit("\\", 1)[-1]
            unit = EmbeddingUnit(
                name=method,
                qualified_name=f"{file_path.replace('/', '.')}.{method_key}",
                file=file_path,
                line=method_info.get("line", 1),
                language=lang,
                unit_type="method",
                signature=all_signatures.get(method_key, f"{_sig_kw} {method}({_sig_args})"),
                docstring=all_docstrings.get(method_key, ""),
                calls=(calls_map.get(f"{_bare_cls}::{method}") or calls_map.get(method, []))[:5],
                called_by=(called_by_map.get(f"{_bare_cls}::{method}") or called_by_map.get(method, []))[:5],
                cfg_summary=cfg_cache.get(method, ""),
                dfg_summary=dfg_cache.get(method, ""),
                dependencies=dependencies,
                code_preview=method_info.get("code_preview", ""),
            )
            units.append(unit)

    return units


def _get_progress_console():
    """Get rich Console if available and TTY, else None."""
    if not sys.stdout.isatty():
        return None
    if os.environ.get("NO_PROGRESS") or os.environ.get("CI"):
        return None
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def _scan_project_language_set(project_path: Path, respect_ignore: bool = True) -> set:
    """Walk the project tree once and return the set of detected language tags.

    Internal helper shared by ``_detect_project_languages`` (code-only contract)
    and ``_detect_project_language_tags`` (broader contract used by the semantic
    indexer). Returned set may include both ``ALL_LANGUAGES`` members and
    ``NON_CODE_LANGUAGE_TAGS`` stand-ins (shell, markdown, toml, ...).
    """
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    found_languages: set = set()
    spec = load_ignore_patterns(project_path) if respect_ignore else None

    for root, dirs, files in os.walk(project_path):
        # Prune common heavy dirs immediately for speed
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '.tldr', 'venv', '.venv', '__pycache__', '.idea', '.vscode', 'env', '.env', 'vendor', 'deps', '_build', 'cover'}]

        for file in files:
             file_path = Path(root) / file

             # Check ignore patterns
             if respect_ignore and should_ignore(file_path, project_path, spec):
                 continue

             ext = file_path.suffix.lower()
             if ext in EXTENSION_TO_LANGUAGE:
                 found_languages.add(EXTENSION_TO_LANGUAGE[ext])

    return found_languages


def _detect_project_languages(project_path: Path, respect_ignore: bool = True) -> List[str]:
    """Scan project files to detect present code languages.

    Returns ONLY languages that are members of ``ALL_LANGUAGES`` (the pre-Bug-004
    contract). Non-code stand-in tags such as ``"shell"``, ``"markdown"``,
    ``"toml"`` etc. are intentionally NOT returned here — callers in cli.py and
    diagnostics.py feed the result into ``resolve_language`` / ``build_project_call_graph``
    / ``_resolve_context_languages`` which only understand ``ALL_LANGUAGES``
    members. Non-code-aware callers (the semantic indexer) should use
    ``_detect_project_language_tags`` instead.

    Args:
        project_path: Path to project root to scan.
        respect_ignore: If True, respect .tldrignore patterns.

    Returns:
        List of detected code languages, call-graph-supported ones first.
    """
    found_languages = _scan_project_language_set(project_path, respect_ignore=respect_ignore)
    code_langs = found_languages & set(ALL_LANGUAGES)

    # Sort code languages with call-graph-supported ones first so resolve_language
    # picks them; this preserves the pre-Bug-004 ordering invariant for code-only
    # projects.
    return sorted(
        code_langs,
        key=lambda l: (0 if l in CALL_GRAPH_LANGUAGES else 1, l),
    )


# Bug 004 (Gate 3) sentinel: when a project contains only non-code files, the
# semantic indexer dispatches a single extraction pass under this tag. It is
# NOT a real language — ``get_code_structure`` falls back to ``{".py"}`` for
# code_extensions and unions in ``NON_CODE_EXTENSIONS``, so only the non-code
# files are enumerated (the directory has no .py files by hypothesis).
NON_CODE_DISPATCH_SENTINEL = "_noncode"


def _detect_project_language_tags(project_path: Path, respect_ignore: bool = True) -> List[str]:
    """Scan project files for the semantic indexer's broader dispatch contract.

    Unlike ``_detect_project_languages`` (which returns only ALL_LANGUAGES
    members), this returns:

      - ``[code_lang_1, code_lang_2, ...]`` when at least one code language is
        present (non-code files are picked up via ``NON_CODE_EXTENSIONS`` union
        inside ``get_code_structure`` during each code-language pass; the
        semantic indexer dedupes by ``qualified_name`` across passes).
      - ``[NON_CODE_DISPATCH_SENTINEL]`` (a single tag) when only non-code files
        are present — avoids the C-4 triple-indexing case where multiple
        non-code language stand-ins (shell, markdown, toml, ...) each triggered
        a redundant dispatch pass.
      - ``[]`` when the project contains no recognized files at all.

    Args:
        project_path: Path to project root to scan.
        respect_ignore: If True, respect .tldrignore patterns.

    Returns:
        Ordered list of dispatch tags for ``build_semantic_index``.
    """
    found_languages = _scan_project_language_set(project_path, respect_ignore=respect_ignore)
    code_langs = found_languages & set(ALL_LANGUAGES)
    non_code_langs = found_languages & NON_CODE_LANGUAGE_TAGS

    sorted_code = sorted(
        code_langs,
        key=lambda l: (0 if l in CALL_GRAPH_LANGUAGES else 1, l),
    )

    if sorted_code:
        return sorted_code
    if non_code_langs:
        # Single sentinel pass — see NON_CODE_DISPATCH_SENTINEL docstring.
        return [NON_CODE_DISPATCH_SENTINEL]
    return []


def build_semantic_index(
    project_path: str,
    lang: str = "python",
    model: Optional[str] = None,
    show_progress: bool = True,
    respect_ignore: bool = True,
    *,
    device: Optional[str] = None,
) -> int:
    """Build and save FAISS index + metadata for a project.

    Creates:
    - .tldr/cache/semantic/index.faiss - Vector index
    - .tldr/cache/semantic/metadata.json - Unit metadata

    Args:
        project_path: Path to project root.
        lang: Programming language.
        model: Model name from SUPPORTED_MODELS or HuggingFace name.
        show_progress: Show progress spinner (default: True).
        respect_ignore: If True, respect .tldrignore patterns (default True).
        device: Compute device ('cpu' or 'metal'). If None, defaults to TLDR_DEVICE
                environment variable. Defaults to 'cpu' if env var is not set.

    Returns:
        Number of indexed units.
    """
    import faiss
    import numpy as np
    from tldr.tldrignore import ensure_tldrignore

    if device is None:
        device = os.environ.get("TLDR_DEVICE") or "cpu"

    console = _get_progress_console() if show_progress else None

    # Resolve paths: scan_path is where to look for code, project_root is where to store cache
    scan_path = Path(project_path).resolve()
    project_root = _find_project_root(scan_path)

    # Ensure .tldrignore exists at project root (create with defaults if not)
    created, message = ensure_tldrignore(project_root)
    if created and console:
        console.print(f"[yellow]{message}[/yellow]")

    # Resolve model name early to get HF name for metadata
    model_key = model if model else DEFAULT_MODEL
    if model_key in SUPPORTED_MODELS:
        hf_name = SUPPORTED_MODELS[model_key]["hf_name"]
    else:
        hf_name = model_key

    # Always store cache at project root, not scan path
    cache_dir = project_root / ".tldr" / "cache" / "semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract all units (respecting .tldrignore) - scan from scan_path, not project_root
    if console:
        with console.status("[bold green]Extracting code units...") as status:
            def update_progress(file_path, units_count, total_files):
                short_path = file_path if len(file_path) < 50 else "..." + file_path[-47:]
                status.update(f"[bold green]Processing {short_path}... ({units_count} units)")

            if lang == "all":
                status.update("[bold green]Scanning project languages...")
                target_languages = _detect_project_language_tags(scan_path, respect_ignore=respect_ignore)
                if not target_languages:
                    console.print("[yellow]No supported languages detected in project[/yellow]")
                    return 0
                if console:
                    console.print(f"[dim]Detected languages: {', '.join(target_languages)}[/dim]")

                units = []
                for lang_name in target_languages:
                    status.update(f"[bold green]Extracting {lang_name} code units...")
                    units.extend(extract_units_from_project(str(scan_path), lang=lang_name, respect_ignore=respect_ignore, progress_callback=update_progress))
            else:
                units = extract_units_from_project(str(scan_path), lang=lang, respect_ignore=respect_ignore, progress_callback=update_progress)
            status.update(f"[bold green]Extracted {len(units)} code units")
    else:
        if lang == "all":
            target_languages = _detect_project_language_tags(scan_path, respect_ignore=respect_ignore)
            if not target_languages:
                return 0
            units = []
            for lang_name in target_languages:
                units.extend(extract_units_from_project(str(scan_path), lang=lang_name, respect_ignore=respect_ignore))
        else:
            units = extract_units_from_project(str(scan_path), lang=lang, respect_ignore=respect_ignore)

    # Bug 004 (C-3): when multiple code languages are dispatched under `--lang all`,
    # each pass unions NON_CODE_EXTENSIONS into get_code_structure, so the same
    # non-code file (e.g. build.sh) gets emitted N times. Dedupe by qualified_name
    # to ensure each file/function appears at most once in the FAISS index.
    if units:
        seen_qn: set = set()
        deduped: List[EmbeddingUnit] = []
        for u in units:
            if u.qualified_name in seen_qn:
                continue
            seen_qn.add(u.qualified_name)
            deduped.append(u)
        units = deduped

    if not units:
        return 0

    import numpy as np

    BATCH_SIZE = 128
    num_units = len(units)
    texts = [build_embedding_text(unit) for unit in units]

    if console:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing embeddings...", total=num_units)

            model_obj = get_model(model, device=device)
            all_embeddings = []

            for i in range(0, num_units, BATCH_SIZE):
                chunk_end = min(i + BATCH_SIZE, num_units)
                chunk_texts = texts[i:chunk_end]

                current_unit = units[i]
                short_path = current_unit.file if len(current_unit.file) < 40 else "..." + current_unit.file[-37:]
                progress.update(task, description=f"[bold green]Embedding {short_path}::{current_unit.name}")

                result = model_obj.encode(
                    chunk_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(np.array(result, dtype=np.float32))

                progress.update(task, completed=chunk_end)

            embeddings_matrix = np.vstack(all_embeddings)
    else:
        model_obj = get_model(model, device=device)
        result = model_obj.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True
        )
        embeddings_matrix = np.array(result, dtype=np.float32)

    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    # Save index
    index_file = cache_dir / "index.faiss"
    faiss.write_index(index, str(index_file))

    # Save metadata with actual model used
    metadata = {
        "units": [u.to_dict() for u in units],
        "model": hf_name,
        "dimension": dimension,
        "count": len(units),
    }
    metadata_file = cache_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))

    if console:
        console.print(f"[bold green]✓[/] Indexed {len(units)} code units")

    return len(units)


def _compute_path_filter(scan_path: Path, project_root: Path) -> Optional[str]:
    """Compute a directory-bounded path filter for semantic search.

    For directories, returns a POSIX-normalized relative path with trailing
    "/" to prevent prefix collisions (e.g., "docs/" vs "docs2/"). For files,
    returns the exact relative path (no trailing slash) so the caller can do
    an exact-match comparison. Returns None if scan_path is the project root
    or does not exist.
    """
    if scan_path == project_root:
        return None
    try:
        rel = scan_path.relative_to(project_root)
    except ValueError:
        return None
    rel_str = rel.as_posix()
    if rel_str and rel_str != ".":
        # C-5: file paths must be exact-matched (no trailing slash), since
        # "scripts/build.sh/" would never prefix-match a stored unit.file.
        if scan_path.is_file():
            return rel_str
        return rel_str.rstrip("/") + "/"
    return None


def semantic_search(
    project_path: str,
    query: str,
    k: int = 5,
    expand_graph: bool = False,
    model: Optional[str] = None,
    language: Optional[str] = None,
    *,
    device: Optional[str] = None,
) -> List[dict]:
    """Search for code units semantically.

    Args:
        project_path: Path to project root.
        query: Natural language query.
        k: Number of results to return.
        expand_graph: If True, include callers/callees in results.
        model: Model to use for query embedding. If None, uses
               the model from the index metadata.
        language: Filter results to this language. None or "all" returns all.
        device: Compute device ('cpu' or 'metal'). If None, defaults to TLDR_DEVICE
                environment variable. Must match the device used to build the index.

    Returns:
        List of result dictionaries with name, file, line, score, etc.
    """
    import faiss
    import numpy as np

    # Handle empty query
    if not query or not query.strip():
        return []

    # Find project root for cache location (matches build_semantic_index behavior)
    scan_path = Path(project_path).resolve()
    project_root = _find_project_root(scan_path)
    cache_dir = project_root / ".tldr" / "cache" / "semantic"

    index_file = cache_dir / "index.faiss"
    metadata_file = cache_dir / "metadata.json"

    # Check index exists
    if not index_file.exists():
        raise FileNotFoundError(f"Semantic index not found at {index_file}. Run build_semantic_index first.")

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_file}. Run build_semantic_index first.")

    # Compute --path filter (relative to project_root). When the user passes
    # `--path docs/`, results must be scoped to files under that subdirectory.
    # When --path resolves to the project root itself (the default `.`), no
    # filter is applied. unit["file"] is stored as a path relative to the
    # project root by extract_units_from_project, so we compare against the
    # relative form of scan_path. Path filter is a string prefix with a
    # trailing "/" appended to avoid prefix collisions (e.g. "docs" matching
    # "docs2/foo.md"). Missing scan_path -> empty results + stderr hint.
    if not scan_path.exists():
        print(
            f"warning: --path {project_path!r} does not exist (resolved to {scan_path}); "
            f"returning zero results.",
            file=sys.stderr,
        )
        return []
    if scan_path != project_root:
        # Validate scan_path is inside the project before computing filter.
        try:
            scan_path.relative_to(project_root)
        except ValueError:
            print(
                f"warning: --path {project_path!r} (resolved {scan_path}) is outside "
                f"project root {project_root}; returning zero results.",
                file=sys.stderr,
            )
            return []
    path_filter = _compute_path_filter(scan_path, project_root)

    # Load index and metadata
    index = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text())
    units = metadata["units"]

    # Use model from metadata if not specified (ensures matching embeddings)
    index_model = metadata.get("model")
    if model is None and index_model:
        model = index_model

    # Embed query (with instruction prefix for BGE)
    query_text = f"Represent this code search query: {query}"
    query_embedding = compute_embedding(query_text, model_name=model, device=device)
    query_embedding = query_embedding.reshape(1, -1)

    # Search -- request more results when filtering (by language and/or path),
    # since post-filtering can drop a large fraction of the FAISS candidates.
    filter_lang = language if language and language != "all" else None
    # Normalize path_filter to posix-style once before the loop so we don't
    # repeat the replace() call for every candidate unit on Windows.
    path_filter_posix = _normalize_file_path(path_filter) if path_filter else None
    # C-5: when scan_path is a file, _compute_path_filter returns the exact
    # rel_str (no trailing slash) and we must do an exact match instead of a
    # prefix match. Detect "file mode" by the absence of a trailing slash.
    path_filter_is_file = (
        path_filter_posix is not None and not path_filter_posix.endswith("/")
    )
    # Adaptive over-fetch: path filtering in large monorepos can drop 95%+ of
    # FAISS candidates (a small subdirectory vs. the full index). Use a higher
    # multiplier when path filtering is active than for language-only filtering.
    if path_filter_posix is not None and filter_lang is not None:
        over_fetch_multiplier = 10  # both filters active: be generous
    elif path_filter_posix is not None:
        over_fetch_multiplier = 6   # path filter alone can be very selective
    elif filter_lang is not None:
        over_fetch_multiplier = 3   # language filter: moderate selectivity
    else:
        over_fetch_multiplier = 1
    search_k = min(k * over_fetch_multiplier, len(units))
    scores, indices = index.search(query_embedding, search_k)

    # Build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(units):
            continue

        unit = units[idx]

        if filter_lang and unit.get("language") != filter_lang:
            continue

        if path_filter_posix is not None:
            unit_file = unit.get("file", "")
            # Normalize stored unit path to posix-style so the comparison works
            # uniformly on Windows-built indices too.
            unit_file_posix = _normalize_file_path(unit_file)
            if path_filter_is_file:
                # Exact-match mode (C-5): --path was a file, not a directory.
                if unit_file_posix != path_filter_posix:
                    continue
            elif not unit_file_posix.startswith(path_filter_posix):
                continue

        result = {
            "name": unit["name"],
            "qualified_name": unit["qualified_name"],
            "file": unit["file"],
            "line": unit["line"],
            "unit_type": unit["unit_type"],
            "signature": unit["signature"],
            "score": float(score),
        }

        # Include graph expansion if requested
        if expand_graph:
            result["calls"] = unit.get("calls", [])
            result["called_by"] = unit.get("called_by", [])
            result["related"] = list(set(unit.get("calls", []) + unit.get("called_by", [])))

        results.append(result)
        if len(results) >= k:
            break

    return results
