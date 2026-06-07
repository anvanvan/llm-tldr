"""Torch-free language constants and device resolution.

Single source of truth for ALL_LANGUAGES, EXTENSION_TO_LANGUAGE, and
_resolve_device_arg. This module imports ONLY the standard library so that
short-lived commands (e.g. ``tldr daemon notify``) can import the constants
without transitively pulling in ``sentence_transformers`` / ``torch`` through
``tldr.semantic``.

``tldr.semantic`` re-exports these three names for backward compatibility, so
``from tldr.semantic import ALL_LANGUAGES`` (and the other two) keeps working.
"""

import os
from typing import Optional

ALL_LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "luau", "elixir"]

# Extension-to-language map (single source of truth, imported by both cli.py and
# semantic.py). Includes non-code stand-in tags (Bug 004) used by semantic
# indexing for doc-only / shell-only repos.
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
    # (see NON_CODE_LANGUAGE_TAGS in semantic.py); get_code_structure falls back
    # to its default code_extensions ({".py"}) and unions in NON_CODE_EXTENSIONS,
    # so the non-code files are still enumerated and reach
    # _process_file_for_extraction Gate 2.
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


def _build_language_to_extensions_map() -> dict[str, set[str]]:
    """Build language → extensions map from canonical extension → language mapping.

    Reverses EXTENSION_TO_LANGUAGE to provide the form needed by api.py's
    file-scanning logic, while keeping a single source of truth for all
    language-extension relationships.
    """
    result: dict[str, set[str]] = {}
    for ext, lang in EXTENSION_TO_LANGUAGE.items():
        # Skip non-code stand-in tags (Bug 004) so api.py gets only real languages
        if lang not in ('shell', 'toml', 'yaml', 'json', 'markdown', 'rst', 'text'):
            if lang not in result:
                result[lang] = set()
            result[lang].add(ext)
    return result


# Public constant for api.py to import
LANGUAGE_TO_EXTENSIONS_MAP = _build_language_to_extensions_map()


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
