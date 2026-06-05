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
from typing import Iterable, List, Literal, Optional, Set, Tuple, Dict, Any, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from tldr.patch import SnapshotEntry
    from tldr.incremental_indexer import IncrementalState

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


def _resolve_default_device() -> str:
    """Return the GPU-first default compute device for this platform.

    ``"metal"`` on Apple Silicon (darwin), ``"cpu"`` everywhere else (e.g. Linux
    CI). This is the platform default used when neither an explicit ``device``
    argument nor ``TLDR_DEVICE`` is set; the env/flag precedence is applied at
    the API boundary (the ``device = device or os.environ.get(...) or
    _resolve_default_device()`` chain) before this is consulted.
    """
    return "metal" if sys.platform == "darwin" else "cpu"


def _resolve_device_arg(device: Optional[str]) -> Optional[str]:
    """Resolve device: explicit arg > TLDR_DEVICE env > None (auto-pick).

    Shared helper used by both ``get_model`` (semantic.py) and ``_resolve_device``
    (cli.py) so the TLDR_DEVICE lookup lives in exactly one place.

    If TLDR_DEVICE is set to a recognised value ('cpu', 'metal', 'mps'), returns
    it. Unrecognised values are silently ignored here — the CLI layer
    (``_resolve_device``) is responsible for user-visible validation and exit.
    Note the asymmetry: 'mps' is accepted from the env (legacy, predates this
    refactor) but is NOT an argparse ``--device`` choice, so ``TLDR_DEVICE=mps``
    works while ``--device mps`` is rejected by the CLI.
    Returns the device string, or None if no device can be determined.
    """
    if device is not None:
        return device
    env_device = os.environ.get("TLDR_DEVICE")
    if env_device in ("cpu", "metal", "mps"):
        return env_device
    return None


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
    # L1 incremental-reindex gate: sha256 of build_embedding_text(self). Empty by
    # default so metadata written before this field loads back without error.
    text_hash: str = ""

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
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingUnit":
        """Reconstruct an EmbeddingUnit from a ``to_dict()`` payload.

        Round-trips ``to_dict()`` exactly (``from_dict(u.to_dict()).to_dict() ==
        u.to_dict()``). Used by the parse-skip carry-forward path to rehydrate
        unchanged units from the persisted metadata WITHOUT re-parsing the file —
        so ``text_hash`` (the plan() reuse gate), ``calls`` and ``called_by``
        (flat ``List[str]`` edge lists) must survive verbatim.

        Every field is read with ``.get(field, default)`` so old metadata written
        before a field existed still loads (forward-compatibility). ``calls`` and
        ``called_by`` are the unit's persisted flat string lists here — NOT the
        richer ``dict[str, list[tuple[str, str]]]`` shape used by file_calls_cache.
        """
        return cls(
            name=d.get("name", ""),
            qualified_name=d.get("qualified_name", ""),
            file=d.get("file", ""),
            line=d.get("line", 0),
            language=d.get("language", ""),
            unit_type=d.get("unit_type", ""),
            signature=d.get("signature", ""),
            docstring=d.get("docstring", ""),
            calls=list(d.get("calls", []) or []),
            called_by=list(d.get("called_by", []) or []),
            cfg_summary=d.get("cfg_summary", ""),
            dfg_summary=d.get("dfg_summary", ""),
            dependencies=d.get("dependencies", ""),
            code_preview=d.get("code_preview", ""),
            text_hash=d.get("text_hash", ""),
        )


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
        device: 'cpu', 'metal', 'mps', or None to read from TLDR_DEVICE env var
                (or let SentenceTransformer auto-pick if not set). Note: this
                function does NOT apply the platform default — that is only done
                at the build_semantic_index/semantic_search API boundaries.

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
    device = _resolve_device_arg(device)

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
                if self._device != "cpu" and hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
                    mx.metal.clear_cache()

        if self._device != "cpu" and hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
        result_np = np.vstack(out_chunks) if len(out_chunks) > 1 else out_chunks[0]
        return result_np[0] if single else result_np


def _stable_call_list(names: List[str], limit: int = 5) -> List[str]:
    """Return a deterministic, de-duplicated, length-capped call list.

    The L2 call-graph edges feeding ``calls`` / ``called_by`` are emitted in a
    hash-seed-dependent order by ``build_project_call_graph`` (dict/set iteration
    over the func_index/registry), so a raw ``list[:5]`` slice picks both a
    different ORDER and a different MEMBERSHIP across separate Python processes.
    That made ``build_embedding_text`` — and therefore each unit's ``text_hash``
    — vary between an index build and a later reindex in a fresh interpreter,
    needlessly re-embedding unchanged units (the EDGE-4 reuse-churn finding).

    Sorting (and de-duplicating) BEFORE the cap makes the kept subset and its
    order independent of the upstream edge-emission order, so the text_hash is
    stable across runs and unchanged units reuse their cached vector.
    """
    if not names:
        return []
    # Fast path: list is short enough that dedup + sort without full set overhead.
    if len(names) <= limit:
        seen: set = set()
        result = []
        for name in names:
            if name not in seen:
                seen.add(name)
                result.append(name)
        # Deduped result is already ≤limit; sort for cross-process determinism.
        return sorted(result)
    # Slow path: full dedup + sort + cap for long lists.
    return sorted(set(names))[:limit]


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


# Process-global set by _init_extraction_worker so each spawned worker records
# the language it was warmed for (diagnostic / future reuse hook). Lives in the
# worker process only; the main process never reads it.
_WORKER_LANG: Optional[str] = None


def _init_extraction_worker(lang: Optional[str]) -> None:
    """ProcessPoolExecutor initializer — runs ONCE per worker process.

    Amortizes the per-task spawn/re-import cost: the worker imports the heavy
    tree-sitter extractor module a single time when the process starts, warming
    Python's module cache for every file the process subsequently handles
    (instead of paying the import on each future). Module-level so it is picklable
    across the macOS ``spawn`` start method.

    ``lang`` is ``None`` only on the multi-language expansion path, which recurses
    into ``extract_units_from_project`` once per concrete tag — each recursive
    call constructs its own pool with a concrete ``lang`` — so a ``None`` here
    simply skips the language-specific pre-import and the worker still processes
    files correctly.
    """
    global _WORKER_LANG
    # Dead write today: no code reads _WORKER_LANG. Kept as a diagnostic /
    # future per-worker reuse hook (see the module-level declaration).
    _WORKER_LANG = lang
    if lang is None or lang == NON_CODE_DISPATCH_SENTINEL:
        # Non-code / unresolved: nothing language-specific to pre-import.
        return
    try:
        # Warm the extractor module cache once for this worker. Best-effort: an
        # import failure must never abort the worker (the per-file try/except in
        # the dispatch loop still isolates real extraction errors).
        import tldr.api  # noqa: F401  (warms get_code_structure's dependency tree)
    except Exception:
        pass


# QLT-1932-1935: @overload signatures so callers narrow the union return type by
# the literal value of return_file_calls_cache — True yields the 3-tuple, the
# default False yields the bare list. Type-annotation only; the single
# implementation below is unchanged at runtime.
@overload
def extract_units_from_project(
    project_path: str,
    lang: Optional[str] = ...,
    respect_ignore: bool = ...,
    progress_callback=...,
    files_to_parse: Optional[Set[str]] = ...,
    *,
    return_file_calls_cache: Literal[True],
    fresh_file_sha1s: Optional[Dict[str, str]] = ...,
) -> "Tuple[List[EmbeddingUnit], Optional[ProjectCallGraph], dict]": ...


@overload
def extract_units_from_project(
    project_path: str,
    lang: Optional[str] = ...,
    respect_ignore: bool = ...,
    progress_callback=...,
    files_to_parse: Optional[Set[str]] = ...,
    return_file_calls_cache: Literal[False] = ...,
    fresh_file_sha1s: Optional[Dict[str, str]] = ...,
) -> "List[EmbeddingUnit]": ...


def extract_units_from_project(
    project_path: str,
    lang: Optional[str] = None,
    respect_ignore: bool = True,
    progress_callback=None,
    files_to_parse: Optional[Set[str]] = None,
    return_file_calls_cache: bool = False,
    fresh_file_sha1s: Optional[Dict[str, str]] = None,
) -> "List[EmbeddingUnit] | Tuple[List[EmbeddingUnit], Optional[ProjectCallGraph], dict]":
    """Extract all functions/methods/classes from a project.

    Uses existing TLDR APIs:
    - tldr.api.get_code_structure() for L1 (signatures)
    - tldr.cross_file_calls for L2 (call graph)
    - CFG/DFG extractors for L3/L4 summaries
    - tldr.api.get_imports for L5 (dependencies)

    Args:
        project_path: Path to project root.
        lang: Programming language ("python", "typescript", "go", "rust"). When
            ``None`` (the default) the project's languages are auto-detected via
            ``_detect_project_language_tags`` and units are MERGED across every
            detected tag — the single authoritative multi-language expansion seam
            (build_semantic_index delegates its ``--lang all`` path here). A
            non-code-only project yields its whole-file units via the
            ``NON_CODE_DISPATCH_SENTINEL`` tag without the sentinel ever reaching
            ``get_code_structure`` / ``build_project_call_graph`` / ``scan_project``.
        respect_ignore: If True, respect .tldrignore patterns (default True).
        files_to_parse: Optional set of project-relative posix paths. When
            provided, the file list from ``get_code_structure`` is filtered to this
            allowlist BEFORE worker dispatch, so excluded (unchanged) files get
            ZERO parse/worker cost. ``None`` (default) parses every file.
        return_file_calls_cache: When True, return a 3-tuple
            ``(units, call_graph_or_None, file_calls_cache)`` where
            ``file_calls_cache`` maps ``(abs_path_str, lang)`` to the
            ``_extract_file_calls`` output (``dict[str, list[tuple[str, str]]]``)
            for each freshly-parsed file. Consumed by ``_build_reapply_call_maps``
            Pass-2 so it can skip the scan_project walk. When False (default — all
            pre-existing callers), the existing return shape is preserved.
        fresh_file_sha1s: Optional mutable ``{rel_posix_path: sha1}`` sink. When
            provided, it is FILLED (not returned) with the raw-bytes SHA-1 of every
            freshly-parsed file, computed from the single read the parser performs
            (S-5 read-once). The persist-time snapshot reuses these instead of
            re-hashing changed files. ``None`` (default) skips collection. This is
            an out-param, not a return value — the 3-tuple return shape is unchanged.

    Returns:
        List of EmbeddingUnit objects with enriched metadata, or a
        (units, call_graph, file_calls_cache) 3-tuple if return_file_calls_cache=True.
    """
    from tldr.api import get_code_structure, build_project_call_graph, get_imports
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    project = Path(project_path).resolve()

    # Load ignore spec before getting structure
    ignore_spec = load_ignore_patterns(project) if respect_ignore else None

    max_workers = int(os.environ.get("TLDR_MAX_WORKERS", os.cpu_count() or 4))

    def _extract_one_language(structure_lang: str):
        """Extract units for ONE concrete language pass.

        ``structure_lang`` is the language handed to ``get_code_structure`` /
        ``build_project_call_graph`` / the worker pool — it is ALWAYS a concrete
        code language (never ``None`` / ``"all"`` / ``"auto"`` / the non-code
        sentinel; the sentinel is mapped to a representative code language by the
        caller so non-code files are still enumerated via the
        ``code_extensions | NON_CODE_EXTENSIONS`` union inside get_code_structure).

        Returns ``(lang_units, call_graph_obj_or_None, file_calls_cache)``.
        """
        lang_units: List[EmbeddingUnit] = []
        file_calls_cache: Dict[Tuple[str, str], Dict[str, List[Tuple[str, str]]]] = {}

        # Get code structure (L1) - use high limit for semantic index
        structure = get_code_structure(str(project), language=structure_lang, max_results=100000, ignore_spec=ignore_spec)

        # Filter ignored files. Reuse the outer ``ignore_spec`` (already loaded
        # once above) instead of re-reading the patterns per language.
        if respect_ignore:
            structure["files"] = [
                f for f in structure.get("files", [])
                if not should_ignore(project / f.get("path", ""), project, ignore_spec)
            ]

        # Build call graph (L2). When the caller wants the file_calls_cache
        # (build_semantic_index's index path), the eager build_project_call_graph
        # is SKIPPED: it would re-walk the project via scan_project (the os.walk the
        # cache exists to eliminate), and its per-unit calls/called_by enrichment is
        # OVERWRITTEN by build_semantic_index's subsequent _reapply_call_graph anyway
        # (which consumes the file_calls_cache). Worker enrichment then uses empty
        # maps; the authoritative edges come from the cache-driven re-apply.
        call_graph_obj = None
        if return_file_calls_cache:
            calls_map = {}
            called_by_map = {}
        else:
            try:
                call_graph = build_project_call_graph(str(project), language=structure_lang)
                calls_map, called_by_map = _build_calls_maps(call_graph)
                call_graph_obj = call_graph
            except Exception:
                # Call graph may not be available for all projects
                calls_map = {}
                called_by_map = {}

        # Process files in parallel for better performance
        files = structure.get("files", [])

        # Parse-skip scoping (T2-8): restrict the worker dispatch to the allowlist
        # BEFORE any future is submitted, so unchanged files cost zero parse/spawn.
        if files_to_parse is not None:
            files = [f for f in files if f.get("path") in files_to_parse]

        # Only ask the worker for file_calls when the caller actually wants the
        # file_calls_cache; otherwise the per-file _extract_file_calls AST parse is
        # pure waste (existing single-language callers that don't need the cache).
        want_calls = return_file_calls_cache

        def _accumulate(file_info, result):
            """Accumulate one worker result (units, or (units, file_calls, sha1))."""
            if want_calls:
                f_units, f_calls, f_sha1 = result
                if f_calls:
                    # Key lang is the DISPATCH language (structure_lang) — the same
                    # value _iter_call_graph_files filters on — NOT unit.language.
                    # On the non-code sentinel path this is "python" by design.
                    key = (str((project / file_info.get("path", "")).resolve()), structure_lang)
                    file_calls_cache[key] = f_calls
                # S-5 read-once: record the freshly-parsed file's raw-bytes sha1
                # (keyed by the project-relative posix path, matching unit.file)
                # so persist reuses it instead of re-hashing. Only collected when
                # the caller passes a fresh_file_sha1s sink.
                if f_sha1 is not None and fresh_file_sha1s is not None:
                    rel_key = Path(file_info.get("path", "")).as_posix()
                    fresh_file_sha1s[rel_key] = f_sha1
            else:
                f_units = result
            lang_units.extend(f_units)
            if progress_callback:
                progress_callback(file_info.get('path', 'unknown'), len(lang_units), len(files))

        # Use parallel processing if we have multiple files
        if len(files) > 1 and max_workers > 1:
            try:
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_extraction_worker,
                    initargs=(structure_lang,),
                ) as executor:
                    futures = {
                        executor.submit(
                            _process_file_for_extraction,
                            file_info,
                            str(project),
                            structure_lang,
                            calls_map,
                            called_by_map,
                            want_calls,
                        ): file_info
                        for file_info in files
                    }

                    for future in as_completed(futures):
                        file_info = futures[future]
                        try:
                            _accumulate(file_info, future.result(timeout=60))
                        except Exception as e:
                            logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

            except Exception as e:
                logger.warning(f"Parallel extraction failed: {e}, falling back to sequential")
                for file_info in files:
                    try:
                        _accumulate(file_info, _process_file_for_extraction(
                            file_info, str(project), structure_lang, calls_map, called_by_map, want_calls
                        ))
                    except Exception as fe:
                        logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {fe}")
        else:
            for file_info in files:
                try:
                    _accumulate(file_info, _process_file_for_extraction(
                        file_info, str(project), structure_lang, calls_map, called_by_map, want_calls
                    ))
                except Exception as e:
                    logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

        return lang_units, call_graph_obj, file_calls_cache

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    units: List[EmbeddingUnit] = []
    file_calls_cache: Dict[Tuple[str, str], Dict[str, List[Tuple[str, str]]]] = {}
    _call_graph_obj = None

    if lang is None or lang == "all":
        # Single authoritative expansion seam: detect the project's dispatch tags
        # (code languages, or the non-code sentinel for doc-only repos) and MERGE
        # units across every tag. Mirrors build_semantic_index's retired per-lang
        # loop. _detect_project_language_tags (NOT _detect_project_languages) is
        # used so non-code-only projects still yield their whole-file units.
        tags = _detect_project_language_tags(project, respect_ignore=respect_ignore)
        for tag in tags:
            # Never hand the sentinel / pseudo-langs to lower-level APIs. The
            # sentinel is a non-code-only project: extract under a representative
            # code language ("python") so get_code_structure unions in
            # NON_CODE_EXTENSIONS and enumerates the doc/config files (there are no
            # .py files by hypothesis, so only the non-code files are emitted).
            structure_lang = "python" if tag == NON_CODE_DISPATCH_SENTINEL else tag
            lang_units, _, lang_cache = _extract_one_language(structure_lang)
            units.extend(lang_units)
            file_calls_cache.update(lang_cache)
        # Multi-language merge cannot be represented by a single graph object.
        _call_graph_obj = None
    else:
        units, _call_graph_obj, file_calls_cache = _extract_one_language(lang)

    if return_file_calls_cache:
        return units, _call_graph_obj, file_calls_cache
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
    return_file_calls: bool = False,
):
    """Process a single file and extract all units. Top-level for pickling.

    This function reads the file ONCE and extracts all information in a single pass,
    avoiding the O(n*m) file read issue where n=files and m=functions.

    Args:
        file_info: Dict with 'path', 'functions', 'classes' from get_code_structure.
        project_path: Absolute path to project root.
        lang: Programming language.
        calls_map: Map of function name -> list of called functions.
        called_by_map: Map of function name -> list of calling functions.
        return_file_calls: When True, return ``(units, file_calls)`` where
            ``file_calls`` is the ``_extract_file_calls`` output for this file
            (``dict[str, list[tuple[str, str]]]``). This feeds the file_calls_cache
            so the call-graph re-apply Pass-2 can skip the scan_project walk. When
            False (default — all pre-existing direct callers), returns just the
            ``List[EmbeddingUnit]`` exactly as before.

    Returns:
        List of EmbeddingUnit objects for this file, or ``(units, file_calls)``
        when ``return_file_calls=True``.
    """
    units = []
    project = Path(project_path)
    file_path = file_info.get("path", "")

    # S-5 read-once: the raw-bytes SHA-1 of this file, computed from the SAME read
    # the parser uses (set below). Threaded back via _ret so the persist-time
    # snapshot reuses it instead of re-hashing — the changed-file path reads each
    # file exactly ONCE. Parity: identical to ``compute_file_hash`` (raw-bytes
    # SHA-1), so the next run's deriver hash-compare stays consistent.
    file_sha1: Optional[str] = None

    def _ret(result_units):
        """Wrap the return value with file_calls (and the read-once sha1) when
        requested.

        file_calls is computed best-effort from the resolved on-disk path using the
        extractor for ``lang`` (so the cache carries correct per-language call data
        for all 8 import-index languages — python, ts/js, go, rust, java, c, php —
        not just Python). It is ``{}`` for self-contained languages / unreadable /
        non-existent files (every ``_extract_*_file_calls`` swallows parse/read
        errors). This is what lets the registry-injected Pass-1 build_project_call_graph
        produce import-resolved edges WITHOUT re-walking via scan_project.

        The third tuple element is the read-once raw-bytes sha1 (``None`` when the
        file could not be read). It is consumed only by ``_accumulate`` in
        ``extract_units_from_project``; the public 3-tuple return is unchanged.
        """
        if not return_file_calls:
            return result_units
        file_calls: Dict[str, List[Tuple[str, str]]] = {}
        try:
            from tldr.cross_file_calls import extract_file_calls_for_language
            file_calls = extract_file_calls_for_language(
                full_path,
                project if not project.is_file() else project.parent,
                lang,
            )
        except Exception:
            file_calls = {}
        return result_units, file_calls, file_sha1
    # Bug 004 R-5: when project_path is a single file (passed through from
    # build_semantic_index's scan_path), get_code_structure stores root.name as
    # file_path. Reconstruct full_path from the parent in that case; otherwise
    # the standard `project / file_path` directory-relative join.
    if project.is_file():
        full_path = project.parent / file_path
    else:
        full_path = project / file_path

    if not full_path.exists():
        return _ret(units)

    try:
        # Read file bytes ONCE; derive both the parse text AND the snapshot sha1
        # from that single read (S-5 read-once — no separate compute_file_hash).
        # C-7: utf-8-sig so a UTF-8 BOM (U+FEFF) is stripped instead of polluting
        # the embedding preview / first source line. C-8: best-effort latin-1
        # fallback for non-UTF-8 legacy config files so we don't silently drop
        # them. The sha1 is over the RAW bytes (matching compute_file_hash).
        raw_bytes = full_path.read_bytes()
        import hashlib as _hashlib
        file_sha1 = _hashlib.sha1(raw_bytes, usedforsecurity=False).hexdigest()
        try:
            content = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1")
        lines = content.split('\n')
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return _ret(units)

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
        return _ret(units)

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
            calls=_stable_call_list(calls_map.get(func_name) or calls_map.get(func_name.rsplit("\\", 1)[-1], [])),
            called_by=_stable_call_list(called_by_map.get(func_name) or called_by_map.get(func_name.rsplit("\\", 1)[-1], [])),
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
                calls=_stable_call_list(calls_map.get(f"{_bare_cls}::{method}") or calls_map.get(method, [])),
                called_by=_stable_call_list(called_by_map.get(f"{_bare_cls}::{method}") or called_by_map.get(method, [])),
                cfg_summary=cfg_cache.get(method, ""),
                dfg_summary=dfg_cache.get(method, ""),
                dependencies=dependencies,
                code_preview=method_info.get("code_preview", ""),
            )
            units.append(unit)

    return _ret(units)


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


def _add_call_edge(
    calls_map: Dict[str, List[str]],
    called_by_map: Dict[str, List[str]],
    src: str,
    dst: str,
) -> None:
    """Record a src->dst call edge into the forward + reverse maps (dedup per key).

    Shared by ``_build_calls_maps`` and ``_build_reapply_call_maps`` so the
    edge-accumulation/dedup logic lives in exactly one place.
    """
    bucket = calls_map.setdefault(src, [])
    if dst not in bucket:
        bucket.append(dst)
    rbucket = called_by_map.setdefault(dst, [])
    if src not in rbucket:
        rbucket.append(src)


def _build_calls_maps(call_graph) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build (calls_map, called_by_map) from a ProjectCallGraph.

    Shared helper used by both ``extract_units_from_project`` and
    ``_build_reapply_call_maps`` so the edge-keying logic stays in one place.

    Returns:
        Tuple of (calls_map: func -> [called functions],
                  called_by_map: func -> [calling functions])
    """
    calls_map: Dict[str, List[str]] = {}
    called_by_map: Dict[str, List[str]] = {}

    for edge in call_graph.edges:
        _src_file, src_func, _dst_file, dst_func = edge
        _add_call_edge(calls_map, called_by_map, src_func, dst_func)

    return calls_map, called_by_map


def _augment_cache_with_carried(
    file_calls_cache: dict,
    carried_units: List["EmbeddingUnit"],
    scan_path: Path,
) -> dict:
    """Return a NEW cache merging ``file_calls_cache`` with RE-PARSED entries for
    carried (parse-skipped, unchanged) files.

    DEL-1 single-point fix: on the incremental path ``file_calls_cache`` holds
    ONLY the freshly-parsed (changed) files, so a naive Pass-1/Pass-2 sees a
    PARTIAL project and diverges from a ``--full`` rebuild (which parses every
    file and therefore has every file in its cache). This helper fills in the
    ``{(abs_path_str, lang): {caller: [(call_type, target)]}}`` entry for each
    carried file so the augmented cache covers EVERY live file — exactly the file
    set (and the file-call data) a ``--full`` rebuild would see.

    Parity with ``--full`` (Option C — re-parse, not synthesize):
      Each carried file is RE-PARSED with the SAME per-language extractor a
      ``--full`` rebuild uses — ``extract_file_calls_for_language`` (the exact
      dispatcher the extraction worker's ``_ret`` calls to populate the fresh
      cache). The carried file is UNCHANGED on disk and page-cached, so the
      tree-sitter / AST re-parse is O(lines) and cheap; the expensive embedding
      reuse is untouched.

      This produces NATIVE, language-correct caller keys BY CONSTRUCTION. The
      previous implementation synthesized a single ``('intra', callee)`` list
      under ONLY the BARE ``unit.name`` key — but class-method languages dual-key
      callers: Java's ``_extract_java_file_calls`` stores each method's calls
      under BOTH the bare ``methodA`` AND the qualified ``ClassA.methodA`` (Go
      ``Receiver.method``, Rust ``Type::method`` similarly). A ``--full`` rebuild
      therefore emits the caller into ``called_by`` under BOTH names, while the
      bare-only synthesis emitted only the bare form → incremental ``called_by``
      diverged from ``--full`` on ~81% of Java units. Re-parsing restores both
      caller forms exactly as ``--full`` does, for all import-index languages.

      The re-parsed dict is the extractor's verbatim output: a changed file's
      ``('intra'|'direct'|'attr'|'ref', target)`` tuples drive Pass-1's
      import-resolved edges AND the cross-file class CONSTRUCTOR edges (callee = a
      bare class name from ``d = Dog()``), and Pass-2a's name-based linking — the
      same way the fresh cache entries do.

    Keying: the cache key is ``(str(scan_path / unit.file), unit.language)`` —
    ``unit.language`` is the dispatch language a ``--full`` rebuild keyed the file
    under (in single-language mode it equals the structure_lang; in multi-language
    mode it is the file's own dispatch language), so the augmented entry is
    interchangeable with a fresh one and ``_iter_call_graph_files`` filters it
    correctly.

    Files already present in ``file_calls_cache`` (the changed files) are NEVER
    re-parsed or overridden — a changed file's FRESH per-caller edges always win.
    Each carried file is re-parsed AT MOST ONCE even when several carried units
    share it. A file whose re-parse yields no calls contributes no entry.

    No-op guard (S-5): when ``file_calls_cache`` is empty there were no fresh
    (changed) files this run — a true no-op reindex — so NO carried file is
    re-parsed (the empty cache is returned unchanged). Only a real change (a
    non-empty fresh cache) triggers the carried-file re-parse.

    The input cache is NOT mutated; a shallow copy plus the re-parsed entries is
    returned.
    """
    augmented = dict(file_calls_cache)
    # No-op guard: an empty fresh cache means nothing changed on disk this run —
    # do not re-parse carried files (no churn for a true no-op reindex).
    if not file_calls_cache:
        return augmented

    from tldr.cross_file_calls import extract_file_calls_for_language

    root = Path(scan_path)
    # Deduplicate by (file, language) BEFORE constructing keys/paths
    unique_files = {}
    for unit in carried_units or []:
        file_lang_key = (unit.file, unit.language)
        if file_lang_key not in unique_files:
            unique_files[file_lang_key] = unit

    for (file_path, language), unit in unique_files.items():
        key = (str(scan_path / file_path), language)
        if key in file_calls_cache:
            # Changed file already in the fresh cache: its freshly-parsed edges
            # are authoritative; never re-parse / override it from a carried unit.
            continue
        file_path_obj = Path(key[0])
        # Re-parse with the EXACT extractor --full uses for this language. Cheap
        # (file unchanged, page-cached); yields native dual-keyed caller entries.
        file_calls = extract_file_calls_for_language(file_path_obj, root, language)
        if file_calls:
            augmented[key] = file_calls
    return augmented


@dataclass
class _ReapplyContext:
    """The cache-context bundle for the 4b call-graph re-apply.

    These four fields are always varied together (the aggressive cache path sets
    all of them; the legacy path leaves them at their defaults). Bundling them
    keeps ``_build_reapply_call_maps`` from carrying four independent optionals
    that are only meaningful in combination.
    """
    call_graph: Optional["ProjectCallGraph"] = None
    file_calls_cache: Optional[dict] = None
    carried_units: Optional[List["EmbeddingUnit"]] = None
    all_units: Optional[List["EmbeddingUnit"]] = None


def _build_reapply_call_maps(
    project_path: str,
    lang: str,
    unit_names: set,
    ctx: Optional[_ReapplyContext] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build (calls_map, called_by_map) for the 4b call-graph re-apply.

    Combines two resolution passes so cross-file caller drift is always caught:

    1. Resolved edges from ``build_project_call_graph`` — these carry proper
       ``ClassName::method`` keys for OOP methods and import-resolved cross-file
       edges, mirroring the keying ``extract_units_from_project`` uses.
    2. A name-based pass over ``_extract_file_calls`` — links any direct/intra/
       ref call whose target name matches a known project function even when the
       callee was not imported (e.g. a sibling module-level helper). Without
       this, a new un-imported caller would not change the callee's text.

    ``unit_names`` is the set of bare names known to be project units; only calls
    whose target is in that set become edges, keeping the maps tight.

    Args:
        ctx: Optional ``_ReapplyContext`` bundling the four cache-context fields
            (always varied together — see the dataclass docstring):

            - ``call_graph``: pre-built ProjectCallGraph (e.g. from
              extract_units_from_project). When provided, skips the Pass 1 rebuild
              to avoid redundant graph computation in single-language mode.
            - ``file_calls_cache``: ``{(abs_path_str, lang): file_calls}`` map
              collected during extraction. When provided, Pass-2 CONSUMES it (plus
              ``carried_units`` for any unchanged files not in it) instead of
              walking the project with ``scan_project`` — eliminating the
              duplicate os.walk on the initial index. ``file_calls`` is the
              ``_extract_file_calls`` output
              (``dict[str, list[tuple[str, str]]]``). It ALSO drives Pass-1's
              aggressive registry path (see ``all_units``).
            - ``carried_units``: carried-forward EmbeddingUnits (parse-skip
              unchanged files). For each, Pass-2 reconstructs the absolute path
              and reads it via ``_extract_file_calls`` directly (cheap — file
              unchanged on disk / in page cache). Only consulted when
              ``file_calls_cache`` is provided.
            - ``all_units``: complete merged unit list (fresh + carried). When
              ``file_calls_cache`` is also provided and no pre-built graph exists,
              Pass-1 builds the func_index from these units via
              ``build_func_index_from_units`` and feeds both it and
              ``file_calls_cache`` into ``build_project_call_graph`` — yielding the
              import-resolved ``ClassName::method`` edges WITHOUT calling
              ``scan_project``.

            ``None`` (the default) keeps the legacy Pass-1 behavior.
    """
    if ctx is None:
        ctx = _ReapplyContext()
    call_graph = ctx.call_graph
    file_calls_cache = ctx.file_calls_cache
    carried_units = ctx.carried_units
    all_units = ctx.all_units

    calls_map: Dict[str, List[str]] = {}
    called_by_map: Dict[str, List[str]] = {}

    # Build the func-index ONCE before the per-language loop so every language
    # iteration reuses the same registry instead of recomputing it O(n*L) times.
    # The registry is language-agnostic (it indexes all units' files), so hoisting
    # it out is safe regardless of how many languages are processed.
    _prebuilt_func_idx = None
    # DEL-1: when a file_calls_cache is supplied it covers ONLY the freshly-parsed
    # (changed) files. Augment it by RE-PARSING the carried (unchanged) files with
    # the SAME per-language extractors --full uses (extract_file_calls_for_language)
    # so BOTH Pass-1 and Pass-2 see EVERY live file — the same file set AND the same
    # native dual-keyed caller entries a --full rebuild produces — making the
    # incremental per-unit calls/called_by BYTE-IDENTICAL to --full for every
    # import-index language. The augmented cache feeds Pass-1's registry build and
    # Pass-2a's name-based linking. Re-parsing (Option C) replaces the former
    # bare-only synthesis from unit.calls, which dropped the qualified
    # ClassName.method caller key for class-method languages (Java/Go/Rust) and so
    # diverged from --full's called_by.
    _augmented_cache = file_calls_cache
    if file_calls_cache is not None:
        _augmented_cache = _augment_cache_with_carried(
            file_calls_cache, carried_units or [], Path(project_path).resolve()
        )
    if file_calls_cache is not None and all_units is not None:
        from tldr.cross_file_calls import build_func_index_from_units
        _prebuilt_func_idx = build_func_index_from_units(all_units, Path(project_path))

    def _add(src: str, dst: str) -> None:
        _add_call_edge(calls_map, called_by_map, src, dst)

    def _link_file_calls(file_calls) -> None:
        """Merge one file's ``_extract_file_calls`` output into the shared maps."""
        for caller_func, edges in file_calls.items():
            for _call_type, target in edges:
                # target may be "obj.method"; take the trailing name.
                bare_target = target.rsplit(".", 1)[-1]
                if bare_target in unit_names:
                    _add(caller_func, bare_target)

    def _apply_one_language(one_lang: str, prebuilt_graph=None) -> None:
        """Run Pass 1 + Pass 2 for a single concrete language, merging into the
        shared calls_map / called_by_map via ``_add`` (which dedupes per key).

        Both passes are best-effort: a failure for one language is logged at
        debug and never aborts the others (important for the lang="all" merge,
        where one odd/unsupported language must not skip the rest).
        """
        root = Path(project_path).resolve()
        # Pass 1: resolved project call graph (method keys, import-resolved edges).
        # Reuse a pre-built graph when available (avoids double build_project_call_graph
        # in single-language mode); fall back to building it here for multi-lang / tests.
        # AGGRESSIVE cache path: when a file_calls_cache is supplied (initial-index /
        # parse-skip fast path) and no graph was pre-built, build_project_call_graph is
        # still run — but with an O(n) registry built from the already-parsed units
        # (build_func_index_from_units) AND the cache fed in as the per-file call data.
        # That keeps the import-resolved ClassName::method edges while NEVER calling
        # scan_project (neither build_function_index nor the per-language walk runs).
        try:
            graph = prebuilt_graph
            if graph is None:
                from tldr.api import build_project_call_graph
                if file_calls_cache is not None and _prebuilt_func_idx is not None:
                    # Reuse the func-index built once outside the loop (O(n) vs O(n*L)).
                    # DEL-1: feed the AUGMENTED cache (changed + carried files) so
                    # import resolution covers every file, matching --full.
                    graph = build_project_call_graph(
                        project_path,
                        language=one_lang,
                        prebuilt_func_index=_prebuilt_func_idx,
                        prebuilt_file_calls=_augmented_cache,
                    )
                elif file_calls_cache is None:
                    graph = build_project_call_graph(project_path, language=one_lang)
                # Defensive no-op (unreachable in production): file_calls_cache set
                # but _prebuilt_func_idx None requires all_units to be None at the
                # call site, yet _reapply_call_graph always passes all_units. If it
                # ever fires, graph stays None and Pass-1 is skipped (Pass-2 still
                # runs); a different text_hash would just trigger a harmless
                # re-embed, never corruption.
            if graph is not None:
                _cmap, _ = _build_calls_maps(graph)
                for src, dsts in _cmap.items():
                    for dst in dsts:
                        _add(src, dst)
        except Exception as e:
            logger.debug("_build_reapply_call_maps pass1 (%s) failed: %s", one_lang, e, exc_info=True)

        # Pass 2: name-based linking for un-imported same-name calls (Python/TS-style
        # direct calls). Best-effort — failures here never break indexing.
        if file_calls_cache is not None:
            # Cache-driven Pass-2 (no scan_project os.walk): the AUGMENTED cache
            # already covers freshly-parsed (changed) files AND carried (unchanged)
            # files — the latter RE-PARSED by ``_augment_cache_with_carried`` via the
            # SAME per-language ``extract_file_calls_for_language`` a --full rebuild
            # uses, so every carried entry is byte-identical to the one --full would
            # produce. Iterating it here makes Pass-2a's name-based linking identical
            # to the --full ``else`` branch below (scan_project + _link_file_calls
            # over all files), so NO separate carried-unit supplement is needed.
            #
            # [T-9/I-6 #12] Carried entries carry the extractor's native
            # ``(call_type, target)`` tuples (including the dual-keyed
            # ClassName.method caller form for Java/Go/Rust); Pass-2a's
            # _link_file_calls reads only ``target`` while Pass-1 USES call_type
            # ('intra' -> unconditional edge) to restore the cross-file constructor
            # (class-name) callees Pass-2a cannot link.
            try:
                for (cache_path, cache_lang), fcalls in _augmented_cache.items():
                    if cache_lang != one_lang:
                        continue
                    _link_file_calls(fcalls)
            except Exception as e:
                logger.debug("_build_reapply_call_maps pass2-cache (%s) failed: %s", one_lang, e, exc_info=True)
        else:
            try:
                from tldr.cross_file_calls import _extract_file_calls, scan_project

                for src in scan_project(root, one_lang, None):
                    spath = Path(src)
                    try:
                        file_calls = _extract_file_calls(spath, root)
                    except Exception:
                        continue
                    _link_file_calls(file_calls)
            except Exception as e:
                logger.debug("_build_reapply_call_maps pass2 (%s) failed: %s", one_lang, e, exc_info=True)

    if lang is None or lang == "all":
        # B-3 / I-7: neither pass understands the "all"/None pseudo-language (Pass 1
        # builds an empty graph; Pass 2's scan_project raises ValueError), so a naive
        # value silently skipped the entire name-based re-apply for multi-language
        # projects. Resolve the project's concrete code languages and run + merge the
        # same two passes per language. _detect_project_languages (NOT the tags
        # variant) is correct here: scan_project raises on the non-code sentinel, and
        # concrete code tags let non-code-only repos produce empty maps and early-exit
        # without crashing. No pre-built graph is reused (single-language path only).
        for detected_lang in _detect_project_languages(Path(project_path)):
            _apply_one_language(detected_lang)
    else:
        _apply_one_language(lang, prebuilt_graph=call_graph)

    return calls_map, called_by_map


def _reapply_call_graph(
    units: List["EmbeddingUnit"],
    project_path: str,
    lang: str,
    call_graph: Optional["ProjectCallGraph"] = None,
    file_calls_cache: Optional[dict] = None,
    carried_units: Optional[List["EmbeddingUnit"]] = None,
) -> None:
    """Re-apply calls/called_by to every unit from the full project call graph.

    Runs over ALL freshly-extracted units (4b). Uses the same key logic as
    ``extract_units_from_project`` — methods look up ``ClassName::method`` first
    then the bare method name; functions look up the bare name. Mutates each
    unit in place. Non-code/file units (no callable name) are left untouched.

    Args:
        call_graph: Optional pre-built ProjectCallGraph. When provided (single-
            language mode), Pass 1 reuses it instead of rebuilding.
        file_calls_cache: Optional ``{(abs_path, lang): file_calls}`` from
            extraction. Threaded into Pass-1 (as the per-file call data for the
            registry-injected build_project_call_graph) AND Pass-2 so BOTH skip the
            scan_project walk while Pass-1 still yields import-resolved edges.
        carried_units: Optional carried-forward units (parse-skip) so Pass-2 can
            link their (unchanged) calls without scan_project.

    The complete ``units`` list is forwarded as ``all_units`` so Pass-1 can build
    the func_index registry from already-parsed units (no scan_project).
    """
    if not units:
        return

    # Bare names of every code unit, used to scope name-based call linking.
    unit_names: set = set()
    for u in units:
        if u.unit_type in ("function", "method"):
            unit_names.add(u.name)

    calls_map, called_by_map = _build_reapply_call_maps(
        project_path, lang, unit_names,
        ctx=_ReapplyContext(
            call_graph=call_graph,
            file_calls_cache=file_calls_cache,
            carried_units=carried_units,
            all_units=units,
        ),
    )
    if not calls_map and not called_by_map:
        return

    for unit in units:
        if unit.unit_type == "method":
            bare = unit.name
            # cls is the component just before the method in the qualified name.
            parts = unit.qualified_name.rsplit(".", 2)
            cls = parts[-2] if len(parts) >= 2 else ""
            bare_cls = cls.rsplit("\\", 1)[-1]
            unit.calls = _stable_call_list(
                calls_map.get(f"{bare_cls}::{bare}") or calls_map.get(bare, [])
            )
            unit.called_by = _stable_call_list(
                called_by_map.get(f"{bare_cls}::{bare}") or called_by_map.get(bare, [])
            )
        elif unit.unit_type == "function":
            bare = unit.name
            unit.calls = _stable_call_list(
                calls_map.get(bare) or calls_map.get(bare.rsplit(".", 1)[-1], [])
            )
            unit.called_by = _stable_call_list(
                called_by_map.get(bare) or called_by_map.get(bare.rsplit(".", 1)[-1], [])
            )


def _normalize_dirty_files(dirty_files: Iterable[str], scan_path: Path) -> Set[str]:
    """Normalize daemon-supplied dirty paths to scan_path-relative posix strings.

    The daemon writes ABSOLUTE file paths to its dirty set; ``unit.file`` is
    always scan_path-relative posix. The parse-skip carry-forward filter compares
    the two, so the dirty paths must be normalized first (G-4/I-10) — otherwise
    every comparison misses and all units are silently carried (stale index).

    Each entry is converted via ``Path(p).relative_to(scan_path).as_posix()``.
    Paths outside ``scan_path`` (which cannot belong to this project's units)
    raise ``ValueError`` on ``relative_to`` and are silently skipped.
    """
    root = Path(scan_path).resolve()
    result: Set[str] = set()
    for p in dirty_files:
        try:
            rel = Path(p).resolve().relative_to(root).as_posix()
        except ValueError:
            # Out-of-tree path: cannot belong to this project's units. Skip.
            continue
        result.add(rel)
    return result


def _compute_current_file_hashes(
    units: List["EmbeddingUnit"],
    scan_path: str,
    deriver_sha1_map: Optional[Dict[str, str]] = None,
    fresh_file_sha1s: Optional[Dict[str, str]] = None,
    old_snapshot: Optional[Dict[str, "SnapshotEntry"]] = None,
    changed_set: "Optional[Set[str]]" = None,
) -> Dict[str, "SnapshotEntry"]:
    """Build the WIDE per-file snapshot entries for the files that produced ``units``.

    Returns a ``{rel_path -> {sha1, mtime_ns, size, inode}}`` map for the next
    run's self-validating floor. Each file is stat'd exactly once here (for
    mtime_ns/size/inode); best-effort — files that vanish between extraction and
    stat are skipped.

    S-5 (no second hash pass): the SHA-1 is sourced WITHOUT re-hashing whenever
    possible, via a four-case lookup keyed by the file's scan_path-relative rel
    path (``unit.file`` — the SAME key the snapshot is written/read under). The
    cases are tried in this order, matching the code below:

      1. ``rel in fresh_file_sha1s`` -> the parser already computed this file's
         raw-bytes sha1 from its single read this run (the primary S-5 read-once
         source). Reuse that sha1.
      2. ``rel in deriver_sha1_map`` -> the deriver already hashed this file in
         the branch-(b) floor (a confirmed-changed file). Reuse that sha1.
      3. ``rel not in changed_set`` AND ``rel in old_snapshot`` -> the file is
         stat-unchanged this run; reuse its sha1 from the previous snapshot.
      4. otherwise -> ``compute_file_hash`` (new files, the daemon-hint path's
         changed files that neither the parser nor the deriver hashed, and the
         first-run / --full path where all reuse sources are empty).

    The ``changed_set`` guard in case 3 is critical for the daemon-hint path: a
    hint-supplied changed file IS present in ``old_snapshot`` with its STALE sha1,
    so reusing it would persist a stale hash and make the file look clean next
    run even though it changed. Forcing such files to case 4 guarantees a fresh
    sha1 for any known-changed file; a stat-unchanged file can never get a stale
    sha1. On a true no-op every live file hits case 3 -> ``compute_file_hash`` is
    called ZERO times.
    """
    from tldr.patch import compute_file_hash

    deriver_sha1_map = deriver_sha1_map or {}
    fresh_file_sha1s = fresh_file_sha1s or {}
    old_snapshot = old_snapshot or {}
    changed_set = changed_set or set()

    root = Path(scan_path)
    entries: Dict[str, "SnapshotEntry"] = {}
    for unit in units:
        rel = unit.file
        if rel in entries:
            continue
        abs_path = root / rel
        try:
            # stat-before-hash TOCTOU: a concurrent write between os.stat and the
            # hash could pair an old mtime_ns with a new sha1. This is covered by
            # the deriver's `st.st_mtime >= index_start_time` re-hash guard, which
            # forces a hash-confirm for any file touched during the index run.
            st = os.stat(str(abs_path))
        except (FileNotFoundError, OSError):
            continue

        # Case 1: fresh parser read (wins over deriver confirm per S-5).
        sha1 = fresh_file_sha1s.get(rel)
        if sha1 is None:
            # Case 2: deriver confirmed-changed sha1.
            sha1 = deriver_sha1_map.get(rel)
        if sha1 is None:
            # Case 3: stat-unchanged file -> reuse the prior snapshot's sha1.
            # Guarded by changed_set so a known-changed file never reuses a stale
            # hash (daemon-hint path: changed file is in old_snapshot w/ old sha1).
            if rel not in changed_set:
                prior = old_snapshot.get(rel)
                if prior is not None:
                    prior_sha1 = prior.get("sha1")
                    if prior_sha1:
                        sha1 = prior_sha1
        if sha1 is None:
            # Case 4: no reusable sha1 -> hash fresh (new / hint-changed / --full).
            try:
                sha1 = compute_file_hash(str(abs_path))
            except (FileNotFoundError, OSError):
                continue

        entries[rel] = {
            "sha1": sha1,
            "mtime_ns": int(st.st_mtime_ns),
            "size": int(st.st_size),
            "inode": int(st.st_ino),
        }
    return entries


def _enumerate_live_files(
    scan_path: "str | Path",
    lang: Optional[str] = None,
    respect_ignore: bool = True,
) -> Set[str]:
    """Enumerate the set of project-relative posix files that WOULD be indexed.

    [G-11/I-5 LOCKED — shared discovery, no divergence] This helper MUST NOT
    rebuild its own extension set. It reuses the EXACT same file-discovery that
    ``get_code_structure`` (and therefore ``extract_units_from_project``) uses:

      - the accepted-extension set is ``code_extensions | NON_CODE_EXTENSIONS``
        where ``code_extensions`` comes from the SAME ``_EXT_MAP_ALL_LANGUAGES``
        map ``get_code_structure`` consumes;
      - for ``lang=None`` (the multi-language default) the union is taken over
        every dispatch tag ``extract_units_from_project`` would resolve via
        ``_detect_project_language_tags`` (sentinel -> "python");
      - the SAME rglob walk, hidden-path skip, and ``ignore_spec.match_file``
        filtering as ``get_code_structure``.

    Returning the walk's relative posix paths (matching ``unit.file`` keys) makes
    it structurally impossible for the live set to diverge from the parsed set —
    proven by the mandatory parity test
    ``_enumerate_live_files(scan_path) == {u.file for u in units}``.
    """
    from tldr.api import _EXT_MAP_ALL_LANGUAGES
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    project = Path(scan_path).resolve()

    # Resolve the dispatch tags EXACTLY as extract_units_from_project does, so the
    # extension union matches the set of get_code_structure passes that run.
    if lang is None or lang == "all":
        tags = _detect_project_language_tags(project, respect_ignore=respect_ignore)
        structure_langs = [
            "python" if t == NON_CODE_DISPATCH_SENTINEL else t for t in tags
        ]
    else:
        structure_langs = [lang]

    # Union of accepted extensions across every dispatch pass (same as the union
    # of the per-pass ``code_extensions | NON_CODE_EXTENSIONS`` in get_code_structure).
    extensions: Set[str] = set(NON_CODE_EXTENSIONS)
    for sl in structure_langs:
        extensions |= _EXT_MAP_ALL_LANGUAGES.get(sl, {".py"})

    ignore_spec = load_ignore_patterns(project) if respect_ignore else None

    live: Set[str] = set()
    if project.is_file():
        if project.suffix in extensions:
            live.add(project.name)
        return live

    for file_path in project.rglob("*"):
        # Skip entire hidden-name components (not just .tldr, but any .xxx)
        try:
            rel_path = file_path.relative_to(project)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_path.parts):
            continue
        # Check extension FIRST (no syscall needed)
        if file_path.suffix not in extensions:
            continue
        if not file_path.is_file():
            continue
        # [G-3] Apply the SAME effective ignore filter as the index path: not just
        # .tldrignore (ignore_spec.match_file) but ALSO .gitignore. The index path
        # filters via should_ignore (get_code_structure's match_file + the
        # should_ignore pass in extract_units_from_project), so a gitignored file
        # is NEVER parsed/indexed. Mirroring should_ignore here keeps the live set
        # equal to the parsed set — a gitignored file absent from the snapshot can
        # no longer appear as a false-dirty "new" file on every reindex.
        if respect_ignore and should_ignore(file_path, project, ignore_spec):
            continue
        live.add(rel_path.as_posix())
    return live


def _derive_dirty_set(
    scan_path: Path,
    project_root: Path,
    state: "IncrementalState",
    dirty_files_hint: Optional[Iterable[str]],
    trust_hint: bool,
    full: bool,
    index_start_time: float,
    lang: Optional[str] = None,
    respect_ignore: bool = True,
    live_files: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Return ``(changed, deleted, sha1_map)`` — the single dirty-set seam that
    unifies all run patterns (manual / aqm / daemon / full).

    ``changed`` / ``deleted`` are scan_path-relative posix sets. ``sha1_map`` is a
    ``{rel_path: sha1}`` map of the freshly-computed SHA-1s for files confirmed
    dirty via the branch-(b) hash floor (S-5): the persist-time hasher reuses
    these instead of re-hashing. It is ``{}`` for the early return and branch (a)
    (those paths compute no hashes here).

    [T-1/T-2 LOCKED] Structure — early return + TWO branches:

      - EARLY RETURN (forced-full / first-run): ``full or state.full_rebuild or
        not state.old_units`` -> ``(all_live, set())``. The degenerate
        empty-snapshot case on the SAME path as manual.
      - Branch (a): daemon hint present AND epoch-continuous -> trust the hint.
      - Branch (b): everything else -> the self-validating hash floor (two-phase
        stat fast-path then SHA-1 confirm).

    [G-7 LOCKED] Deletion source = index metadata (``state.old_units``), NOT the
    snapshot — robust to a narrow/stale/partial snapshot or a crash mid-persist.
    """
    from tldr.patch import load_snapshot, compute_file_hash

    # Reuse pre-computed live set if provided; otherwise enumerate.
    # Use `is not None` guard (NOT `or`) so an empty set is respected correctly.
    live = live_files if live_files is not None else _enumerate_live_files(scan_path, lang=lang, respect_ignore=respect_ignore)

    # EARLY RETURN: forced full / first run — carry nothing.
    if full or state.full_rebuild or not state.old_units:
        return live, set(), {}

    # Deletion = index metadata minus live (G-7). old_units is authoritative.
    indexed = {u.get("file") for u in (state.old_units or [])}
    deleted = {f for f in indexed if f is not None} - live

    # Branch (a): daemon hint AND epoch provably continuous -> trust the hint.
    # The caller materializes the hint to a list before passing it here, so
    # re-iteration is safe.
    #
    # [G-6 FALL-THROUGH] A non-empty hint may normalize to an EMPTY changed set
    # when every hinted path is out-of-scope (not under scan_path → filtered by
    # _normalize_dirty_files). An empty changed set on the trusted-hint path would
    # carry ALL files (no-op) and silently miss a real edit the hint failed to
    # mention. So only SHORT-CIRCUIT on a NON-EMPTY normalized set; an empty one
    # falls through to the branch-(b) hash floor, which independently detects the
    # real change. (No sha1 is computed on the hint path → empty sha1_map.)
    hint = list(dirty_files_hint) if dirty_files_hint is not None else []
    if hint and trust_hint:
        changed = _normalize_dirty_files(hint, scan_path)
        if changed:
            return changed, deleted, {}
        # else: hint normalized to empty (all out-of-scope) — fall through.

    # Branch (b): everything else — self-validating hash floor.
    # load_snapshot is fully normalized to SnapshotEntry dicts; NO isinstance /
    # schema_version guards here (T-2). Narrow->wide sentinels (mtime_ns=0,
    # size=-1, inode=-1) force a hash-confirm on the first post-upgrade run.
    #
    # S-5: a changed file's content is read EXACTLY ONCE here for its sha1, which
    # is recorded in ``sha1_map`` so persist reuses it (case-1) and never re-hashes.
    # A stat-unchanged file is never read here (persist reuses the prior snapshot's
    # sha1 via case-2). On a true no-op every file takes the stat fast-path -> ZERO
    # reads/hashes in this loop AND zero at persist.
    #
    # The floor's one legitimate per-changed-file hash is computed via
    # ``compute_file_hash`` on confirmed-ambiguous files only (size+inode match but
    # timestamp moved). On a true no-op every file takes the stat fast-path -> ZERO
    # hashes here. A first-run / --full build skips this branch (early return) and
    # hashes at persist, so it still exercises ``compute_file_hash`` there.
    # A-1: validate the stored scan_path header against the current scan; a
    # mismatch (or an old header-less snapshot) returns {} so every live file is
    # treated as new and re-hashed — correct, just slower — instead of trusting
    # scan_path-relative keys written under a different scan.
    snapshot = load_snapshot(str(project_root), scan_path=str(scan_path))
    changed = set()
    sha1_map: Dict[str, str] = {}

    for rel_path in live:
        entry = snapshot.get(rel_path)
        if entry is None:
            changed.add(rel_path)  # new file (not yet in snapshot)
            continue
        abs_path = scan_path / rel_path  # only construct for existing entries
        try:
            st = os.stat(str(abs_path))
        except (FileNotFoundError, OSError):
            # File vanished between enumeration and stat — treat as changed so the
            # next parse drops it (it won't appear in fresh units).
            changed.add(rel_path)
            continue
        # Definitive content change — different byte count or replaced inode. A
        # different size CANNOT be the same content, so mark dirty by STAT ALONE,
        # with NO hash here: the parse step reads this file once and supplies its
        # sha1 (threaded via fresh_file_sha1s into _compute_current_file_hashes), so
        # the whole changed-file path reads each file exactly ONCE (S-5 read-once).
        if st.st_size != entry["size"] or st.st_ino != entry["inode"]:
            changed.add(rel_path)
            continue
        # Ambiguous — size+inode match but the timestamp moved (or fell inside the
        # index run's ccache window): a possible touch-without-content-change. One
        # SHA-1 confirm; skip if it still matches. The confirmed sha1 is recorded in
        # sha1_map so persist reuses it (no second hash for that file either).
        ambiguous = (
            st.st_mtime_ns != entry["mtime_ns"]      # timestamp changed
            or st.st_mtime >= index_start_time       # ccache sub-second race guard
        )
        if ambiguous:
            try:
                fresh_sha1 = compute_file_hash(str(abs_path))
                if fresh_sha1 != entry["sha1"]:
                    changed.add(rel_path)
                    sha1_map[rel_path] = fresh_sha1
            except (FileNotFoundError, OSError):
                changed.add(rel_path)
        # else: stat fast-path match -> clean, skip (persist reuses old sha1).
    return changed, deleted, sha1_map


def _unified_extract(
    scan_path: "str | Path",
    extract_lang: Optional[str],
    respect_ignore: bool,
    changed: Set[str],
    deleted: Set[str],
    old_units: List[dict],
    all_live: Set[str],
    progress_cb=None,
) -> "Tuple[List[EmbeddingUnit], dict, List[EmbeddingUnit], Dict[str, str]]":
    """Single extraction path replacing _full_extract + _parse_skip_extract.

    Receives the ``(changed, deleted)`` dirty set from ``_derive_dirty_set``;
    parses ONLY the changed files; carries forward every unchanged, non-deleted
    unit from the persisted metadata (``EmbeddingUnit.from_dict`` — preserving
    text_hash / calls / called_by) so they are never re-parsed or re-embedded.

    Returns ``(merged_units, file_calls_cache, carried_units, fresh_file_sha1s)``
    where ``fresh_file_sha1s`` is the ``{rel: sha1}`` read-once map for the
    freshly-parsed files (S-5) — persist reuses it instead of re-hashing.

    Degenerate full-rebuild case: when ``changed == all_live`` (forced full /
    first run) ``files_to_parse`` is passed as ``None`` (parse every file) and the
    carry filter excludes everything — strictly identical output to the old
    _full_extract (which returned a dead None call_graph that is simply dropped).
    """
    # Full parse when everything is dirty (first-run / --full): pass None so the
    # extractor parses every file (no allowlist) and produces an empty carry.
    full_parse = (all_live is not None) and (changed == all_live)
    files_to_parse = None if full_parse else changed

    # S-5 read-once sink: on an INCREMENTAL parse it is filled with the raw-bytes
    # sha1 of every freshly-parsed (changed) file so persist reuses it instead of
    # re-hashing. On a FULL parse (first run / --full) it is intentionally left
    # EMPTY: there is no prior snapshot to make the run incremental, so persist
    # hashes every file fresh (case-3) — identical to the historical --full
    # snapshot and keeping the floor's hash provenance unambiguous on a cold build.
    fresh_sha1s: Dict[str, str] = {}
    fresh, _cg, fcache = extract_units_from_project(
        str(scan_path), lang=extract_lang, respect_ignore=respect_ignore,
        progress_callback=progress_cb, files_to_parse=files_to_parse,
        return_file_calls_cache=True,
        fresh_file_sha1s=(None if full_parse else fresh_sha1s),
    )

    if full_parse:
        carried: List[EmbeddingUnit] = []
        merged = sorted(fresh, key=lambda u: (u.file, u.line))
        return merged, fcache, carried, {}

    # Carry filter with deletion drop (G-7): a carried unit can never reference a
    # file removed from disk because both ``changed`` and ``deleted`` are computed
    # against the same old_units record.
    carried = [
        EmbeddingUnit.from_dict(u)
        for u in old_units
        if u.get("file") is not None         # drop null-file units (nothing to validate; avoids TypeError in the sort below)
        and u.get("file") not in changed     # unchanged
        and u.get("file") not in deleted     # still exists — MANDATORY
    ]
    # Deterministic merged order (row i <-> units[i]); this single sort is the
    # row-order source. plan/assemble are qualified_name-keyed, so the exact order
    # only needs to be stable.
    merged = sorted(fresh + carried, key=lambda u: (u.file, u.line))
    return merged, fcache, carried, fresh_sha1s


def build_semantic_index(
    project_path: str,
    lang: str = "python",
    model: Optional[str] = None,
    show_progress: bool = True,
    respect_ignore: bool = True,
    *,
    device: Optional[str] = None,
    full: bool = False,
    # dirty_files: Optional hint from the daemon (when epoch-continuous). When
    # present and epoch-continuous, only those files are re-parsed; unchanged files
    # are carried forward from metadata (parse-skip). When None, empty, or
    # epoch-discontinuous, the self-validating hash floor derives the dirty set
    # from file stat + content hashes, enabling incremental indexing on manual paths.
    dirty_files: Optional[Iterable[str]] = None,
) -> int:
    """Build and save FAISS index + metadata for a project.

    Incremental by default: only units whose embedding text changed (gated by a
    per-unit ``text_hash``) are re-embedded; unchanged units reuse their old
    vector via FAISS ``reconstruct_n``. Changed/deleted files are derived from:
    (1) the daemon's dirty_files hint if provided and epoch-continuous, OR (2) the
    self-validating hash floor (file stat + content hashes) for manual and
    epoch-discontinuous paths. Either way the call graph is re-APPLIED to the
    complete (fresh + carried) unit list every run, so cross-file caller drift is
    always reflected. Pass ``full=True`` to force a clean rebuild.

    Creates:
    - .tldr/cache/semantic/index.faiss - Vector index
    - .tldr/cache/semantic/metadata.json - Unit metadata

    Args:
        project_path: Path to project root.
        lang: Programming language.
        model: Model name from SUPPORTED_MODELS or HuggingFace name.
        show_progress: Show progress spinner (default: True).
        respect_ignore: If True, respect .tldrignore patterns (default True).
        device: Compute device ('cpu' or 'metal'). If None, honours TLDR_DEVICE,
                then falls back to the platform default ('metal' on Apple Silicon,
                'cpu' otherwise).
        full: If True, force a full rebuild ignoring all cached vectors/hashes.
        dirty_files: Optional WATCHER-AUTHORITATIVE list of changed file paths
                (absolute, from the daemon). When non-empty and the daemon's epoch
                is provably continuous (indicating unbroken watcher activity), these
                files are re-parsed and unchanged files are carried forward from
                metadata (parse-skip). When absent, empty, or epoch-discontinuous,
                the self-validating hash floor derives the dirty set from file stat
                and content hashes — enabling incremental indexing on manual paths.
                The L1 text_hash gate still decides re-embedding, so a stale hint
                can never corrupt the index — only over- or under-skip parsing, both
                self-correcting on the next full rebuild.

    Returns:
        Number of indexed units.
    """
    import faiss
    import numpy as np
    from tldr.tldrignore import ensure_tldrignore

    if device is None:
        # GPU-first default: honour an explicit TLDR_DEVICE first (so
        # TLDR_DEVICE=cpu still wins), then fall back to the platform default
        # (metal on Apple Silicon, cpu otherwise).
        device = os.environ.get("TLDR_DEVICE") or _resolve_default_device()

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

    from tldr.incremental_indexer import IncrementalIndexer, text_hash

    # Incremental decision: load prior state (vectors + per-unit hashes + the
    # persisted unit dicts for parse-skip carry-forward). --full short-circuits
    # this (no reconstruct_n I/O), forces every unit to encode, and returns
    # old_units=[] so the parse-skip guard below falls through to a full parse.
    # Loaded BEFORE extraction so the parse-skip path can consult full_rebuild /
    # old_units to decide whether to skip parsing unchanged files.
    # A-1: pass scan_path so the wide snapshot is stamped + validated against the
    # scan it was keyed under; a scan_path mismatch (or an old header-less
    # snapshot) falls back to the safe re-hash-all path instead of trusting
    # scan_path-relative keys that cannot match the current scan.
    indexer = IncrementalIndexer(str(project_root), scan_path=str(scan_path))
    state = indexer.load_previous(hf_name, force_full=full)

    # Single expansion seam (I-3): the retired per-language `lang=="all"` loop is
    # replaced by delegating to extract_units_from_project(lang=None), which
    # auto-detects + merges all languages internally. A concrete `lang` is passed
    # through unchanged. The same effective lang drives the call-graph re-apply.
    extract_lang: Optional[str] = None if lang == "all" else lang

    # UNIFIED DIRTY-SET DERIVATION (replaces the use_parse_skip guard — T-1/T-7).
    # ONE seam derives (changed, deleted) for every run pattern (manual / aqm /
    # daemon / full): a daemon dirty-files hint is trusted only when epoch is
    # provably continuous (the daemon already gates this and omits the hint when
    # gapped); otherwise the self-validating hash floor re-derives the change set
    # from the wide snapshot. ``index_start_time`` is captured BEFORE any file read
    # so the floor can force a re-hash of any file touched during this run
    # (sub-second ccache race guard).
    import time as _time
    index_start_time = _time.time()

    # Materialize the hint once (it may be a set/list/None — never a generator
    # from the CLI/daemon, but materialize defensively so both reads below see the
    # same contents).
    dirty_files_list = list(dirty_files) if dirty_files is not None else None

    all_live = _enumerate_live_files(scan_path, lang=extract_lang, respect_ignore=respect_ignore)

    # epoch_continuity is the daemon's responsibility: when the daemon cannot
    # prove continuity it OMITS --dirty-files, so a present dirty_files hint here is
    # already epoch-validated. We therefore trust a non-empty hint.
    changed, deleted, deriver_sha1_map = _derive_dirty_set(
        scan_path=scan_path,
        project_root=project_root,
        state=state,
        dirty_files_hint=dirty_files_list,
        trust_hint=dirty_files_list is not None and len(dirty_files_list) > 0,
        full=full,
        index_start_time=index_start_time,
        lang=extract_lang,
        respect_ignore=respect_ignore,
        live_files=all_live,
    )

    file_calls_cache: Dict[Tuple[str, str], Dict[str, List[Tuple[str, str]]]] = {}
    carried_units: List[EmbeddingUnit] = []
    fresh_file_sha1s: Dict[str, str] = {}

    units = []
    if console:
        with console.status("[bold green]Extracting code units...") as status:
            def update_progress(file_path, units_count, total_files):
                short_path = file_path if len(file_path) < 50 else "..." + file_path[-47:]
                status.update(f"[bold green]Processing {short_path}... ({units_count} units)")

            # I-7: orphaned use_parse_skip UI text re-derived from the dirty-set sizes.
            if len(changed) < len(all_live):
                status.update("[bold green]Extracting changed code units (parse-skip)...")
            # I-11 dual-arm assignment (console arm): bind the return tuple.
            units, file_calls_cache, carried_units, fresh_file_sha1s = _unified_extract(
                scan_path, extract_lang, respect_ignore, changed, deleted,
                state.old_units, all_live, progress_cb=update_progress,
            )
            status.update(f"[bold green]Extracted {len(units)} code units")
    else:
        # I-11 dual-arm assignment (plain arm): bind the return tuple.
        units, file_calls_cache, carried_units, fresh_file_sha1s = _unified_extract(
            scan_path, extract_lang, respect_ignore, changed, deleted,
            state.old_units, all_live, progress_cb=None,
        )

    # Bug 004 (C-3): when multiple code languages are dispatched (lang=None), each
    # pass unions NON_CODE_EXTENSIONS into get_code_structure, so the same non-code
    # file (e.g. build.sh) gets emitted N times. Dedupe by qualified_name to ensure
    # each file/function appears at most once in the FAISS index. (Also collapses
    # any carried/fresh overlap on the parse-skip path.)
    if units:
        # Phantom function-form-of-method collapse: the per-file extraction emits a
        # class method BOTH as a ``method`` unit (qualified_name
        # ``file.Class.method``) AND, for some language paths, as a phantom
        # ``function`` unit (qualified_name ``file.method``) at the SAME (file, line).
        # The two describe one source construct, so the phantom inflates the index
        # and double-counts re-embeds when the (shared) called_by changes. Drop the
        # ``function`` twin when a ``method`` unit covers the same (file, name, line);
        # the ``method`` unit (richer signature/docstring) is authoritative. This is
        # keyed on an EXACT (file, line, name) collision so a genuine module-level
        # function that merely shares a name with a method in another region is never
        # removed.
        _method_sites = {
            (u.file, u.line, u.name)
            for u in units
            if u.unit_type == "method"
        }
        if _method_sites:
            units = [
                u for u in units
                if not (
                    u.unit_type == "function"
                    and (u.file, u.line, u.name) in _method_sites
                )
            ]

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

    # 4b: re-apply calls/called_by to every unit (fresh AND carried) from the
    # project call graph, then fold that into each unit's text_hash. A new caller
    # changes build_embedding_text -> changes the hash -> routes the unit to
    # encode_units. The file_calls_cache (+ carried_units) lets Pass-2 serve every
    # file without re-walking the project via scan_project.
    # call_graph is always None on the unified path: extract_units_from_project is
    # called with return_file_calls_cache=True, which forces call_graph_obj=None
    # (the authoritative edges come from the cache-driven Pass-2 re-apply).
    _reapply_call_graph(
        units, str(scan_path), extract_lang, call_graph=None,
        file_calls_cache=file_calls_cache, carried_units=carried_units,
    )
    # G2-7: recompute over the COMPLETE merged list (fresh + carried) — a carried
    # unit's called_by may have changed (e.g. a newly-parsed file now calls it),
    # so plan() must see its up-to-date text_hash. Do NOT skip carried units here.
    for unit in units:
        unit.text_hash = text_hash(build_embedding_text(unit))

    plan = indexer.plan(units, state)

    BATCH_SIZE = 128

    def _encode_units(units_to_encode):
        """Encode the embedding text for the given units into a float32 matrix."""
        if not units_to_encode:
            # state.old_dimension may be 0 ONLY on a first run, but then every unit
            # is unseen so plan.encode_units is non-empty and this branch is not
            # reached — so the (0, 0) shape that would break IndexFlatIP is
            # unreachable in practice (the empty-units case returns 0 earlier).
            return np.empty((0, state.old_dimension), dtype=np.float32)
        enc_texts = [build_embedding_text(u) for u in units_to_encode]
        n = len(enc_texts)
        model_obj = get_model(model, device=device)
        if console:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Computing embeddings...", total=n)
                chunks = []
                for i in range(0, n, BATCH_SIZE):
                    chunk_end = min(i + BATCH_SIZE, n)
                    current_unit = units_to_encode[i]
                    short_path = (
                        current_unit.file
                        if len(current_unit.file) < 40
                        else "..." + current_unit.file[-37:]
                    )
                    progress.update(
                        task,
                        description=f"[bold green]Embedding {short_path}::{current_unit.name}",
                    )
                    result = model_obj.encode(
                        enc_texts[i:chunk_end],
                        batch_size=BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    chunks.extend(np.array(result, dtype=np.float32))
                    progress.update(task, completed=chunk_end)
                return np.vstack(chunks)
        result = model_obj.encode(
            enc_texts, batch_size=BATCH_SIZE, normalize_embeddings=True
        )
        return np.array(result, dtype=np.float32)

    fresh = _encode_units(plan.encode_units)

    # Post-encode dimension check (I-9/N-1): for custom HF models the dim is not
    # known until the first encode. If it disagrees with the stored dimension,
    # the reused vectors are incomparable — force a full rebuild, re-plan (so all
    # units land in encode_units and reuse_rows is empty), and re-encode.
    if plan.encode_units and state.old_dimension and fresh.shape[1] != state.old_dimension:
        state.full_rebuild = True
        state.old_hashes = {}
        plan = indexer.plan(units, state)
        fresh = _encode_units(plan.encode_units)

    matrix = indexer.assemble(units, plan, state.old_matrix, fresh)
    dimension = matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(matrix)

    # Compute current file hashes (orchestrator hint for the next run); persist
    # writes index.faiss + metadata.json atomically and saves the cache.
    # NOTE: pass scan_path (NOT project_root) — unit.file keys are scan_path-relative,
    # so the floor keys must align with _enumerate_live_files / _derive_dirty_set
    # (which also key off scan_path) for parse-skip to activate on subdir scans.
    # S-5: changed files' sha1s come from the single parse read (fresh_file_sha1s);
    # the deriver's ambiguous-confirm sha1s (sha1_map) are a fallback for files the
    # extractor didn't re-parse. fresh_file_sha1s wins (it reflects the exact bytes
    # just parsed). Together they let persist avoid re-hashing any changed file.
    current_file_hashes = _compute_current_file_hashes(
        units, str(scan_path),
        deriver_sha1_map=deriver_sha1_map,
        fresh_file_sha1s=fresh_file_sha1s,
        old_snapshot=state.old_snapshot,
        changed_set=changed,
    )
    indexer.persist(index, units, hf_name, dimension, current_file_hashes)

    # One-line run summary: how many units were re-embedded this run vs reused
    # via the text_hash gate, plus the active device. Counts come from the FINAL
    # plan, so a --full run (or a post-encode dim-mismatch reset) correctly shows
    # reused 0. Emitted to stderr on every run (independent of show_progress) so
    # direct callers and the CLI both get the feedback.
    embedded_count = len(plan.encode_units)
    reused_count = len(plan.reuse_rows)
    print(
        f"Semantic index: embedded {embedded_count}, reused {reused_count} "
        f"units (device={device})",
        file=sys.stderr,
    )

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
        device: Compute device ('cpu' or 'metal'). If None, honours TLDR_DEVICE,
                then falls back to the platform default ('metal' on Apple Silicon,
                'cpu' otherwise).

    Returns:
        List of result dictionaries with name, file, line, score, etc.
    """
    import faiss
    import numpy as np

    # Handle empty query
    if not query or not query.strip():
        return []

    # Symmetric device default (I-11): resolve the same GPU-first default the
    # index used so query embedding shares the model cache key and the daemon
    # never reloads the model on an index/search device switch.
    if device is None:
        device = os.environ.get("TLDR_DEVICE") or _resolve_default_device()

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
