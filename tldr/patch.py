"""True Incremental Updates (P4) - File-level graph patching.

This module provides O(1) per-file updates to the call graph, instead of
rebuilding the entire graph on every file edit.

Key functions:
- compute_file_hash(file_path) - SHA-1 hash for content-based deduplication
- extract_edges_from_file(file_path, lang) - Extract edges from single file
- patch_call_graph(graph, edited_file, project_root, lang) - Patch graph incrementally
- has_file_changed(file_path, cached_hash) - Check if file content changed

Usage:
    from tldr.patch import patch_call_graph, compute_file_hash, has_file_changed

    # Check if file changed
    if has_file_changed(file_path, cached_hash):
        # Patch the graph for just this file
        graph = patch_call_graph(graph, file_path, project_root, lang="python")
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TypedDict

from tldr.cross_file_calls import (
    ProjectCallGraph,
    _extract_file_calls,
    _extract_ts_file_calls,
    _extract_go_file_calls,
    _extract_rust_file_calls,
)


# ---------------------------------------------------------------------------
# SnapshotStore — wider file-hash snapshot with atomic writes + back-compat.
#
# The wider snapshot stores, per file, enough metadata for the self-validating
# hash floor to fast-path the common no-change case via a stat() comparison
# before falling back to a SHA-1 confirm. All narrow->wide normalization happens
# inside load_snapshot so the deriver never sees a raw string and never branches
# on schema_version.
# ---------------------------------------------------------------------------


class SnapshotEntry(TypedDict):
    """Per-file snapshot metadata for the self-validating hash floor."""
    sha1: str
    mtime_ns: int
    size: int
    inode: int


# SCHEMA_VERSION is a write-only forward-compat format marker: it is stamped into
# every snapshot but is NEVER branched on during load_snapshot normalization (the
# narrow->wide sentinel logic is version-agnostic). It exists so a FUTURE format
# change can detect+migrate old snapshots without guessing the layout.
SCHEMA_VERSION = 2
_SCHEMA_KEY = "__schema_version__"
# A-1: reserved header recording the scan_path the snapshot keys are relative to.
# Snapshot keys are scan_path-relative, but the file lives at project_root; when a
# later run scans a DIFFERENT scan_path under the same project_root every key
# misses. The header lets load_snapshot detect that mismatch (or an OLD snapshot
# with no header) and fall back to the safe re-hash-all path instead of trusting
# keys that cannot match. Like _SCHEMA_KEY it is consumed internally and never
# surfaced to callers.
_SCAN_PATH_KEY = "__scan_path__"


def _snapshot_path(project_root: str) -> Path:
    return Path(project_root) / ".tldr" / "cache" / "file_hashes.json"


def _normalize_scan_path(scan_path: "str | Path") -> str:
    """Resolve a scan_path to a stable absolute posix string for header compare."""
    return Path(scan_path).resolve().as_posix()


def _safe_int(value, default: int = 0) -> int:
    """Coerce ``value`` to int, returning ``default`` on bad input.

    A hand-edited/corrupt snapshot entry like ``{"mtime_ns": "abc"}`` must
    self-heal (the sentinel default forces a re-hash in the deriver) instead of
    raising ``ValueError`` up through the indexing run.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_snapshot(
    project_root: str, scan_path: "str | Path | None" = None
) -> dict[str, "SnapshotEntry"]:
    """Load the file-hash snapshot, returning fully-formed SnapshotEntry dicts.

    Back-compat: a narrow ``{rel_path: sha1}`` value is normalized to a
    SnapshotEntry with sentinel ``mtime_ns=0 / size=-1 / inode=-1`` — these
    sentinels force a hash-confirm on the first post-upgrade run (the deriver's
    stat comparisons always trip) WITHOUT any schema branch in the deriver.

    A corrupt or missing file returns ``{}`` (never raises). The internal
    ``__schema_version__`` / ``__scan_path__`` keys are consumed here and never
    surfaced to callers.

    A-1: when ``scan_path`` is given, the stored ``__scan_path__`` header MUST
    match it. A mismatch — or an OLD snapshot with no header (back-compat) —
    returns ``{}`` so the caller falls back to the safe re-hash-all path instead
    of trusting scan_path-relative keys that cannot match the current scan. When
    ``scan_path`` is None (round-trip / deprecated callers) no validation runs.
    """
    cache_path = _snapshot_path(project_root)
    if not cache_path.exists():
        return {}

    try:
        raw = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return {}

    if not isinstance(raw, dict):
        return {}

    # Pop the schema version + scan_path header (consumed internally only).
    raw.pop(_SCHEMA_KEY, None)
    stored_scan_path = raw.pop(_SCAN_PATH_KEY, None)

    # A-1 validation: only when the caller supplied a scan_path to check against.
    if scan_path is not None:
        if stored_scan_path != _normalize_scan_path(scan_path):
            # Mismatched scan_path OR an old header-less snapshot -> distrust keys,
            # fall back to re-hash-all.
            return {}

    entries: dict[str, SnapshotEntry] = {}
    for rel_path, value in raw.items():
        if isinstance(value, str):
            # Narrow format: plain SHA-1 string -> normalize to sentinel entry.
            entries[rel_path] = {
                "sha1": value,
                "mtime_ns": 0,
                "size": -1,
                "inode": -1,
            }
        elif isinstance(value, dict):
            entries[rel_path] = {
                "sha1": value.get("sha1", ""),
                "mtime_ns": _safe_int(value.get("mtime_ns", 0), 0),
                "size": _safe_int(value.get("size", -1), -1),
                "inode": _safe_int(value.get("inode", -1), -1),
            }
        # Any other (unexpected) value type is ignored.
    return entries


def save_snapshot(
    project_root: str,
    entries: dict[str, "SnapshotEntry"],
    scan_path: "str | Path | None" = None,
) -> None:
    """Atomically write the wider snapshot to file_hashes.json.

    Writes to a tmpfile in the same directory then ``os.replace`` (POSIX
    atomic). A crash before ``os.replace`` leaves the original file intact.

    A-1: when ``scan_path`` is given, a ``__scan_path__`` header is stamped so a
    later ``load_snapshot(project_root, scan_path=...)`` can detect a scan_path
    mismatch and fall back to re-hash-all. Omitted when ``scan_path`` is None
    (round-trip / deprecated callers) — no header is written.
    """
    cache_path = _snapshot_path(project_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {_SCHEMA_KEY: SCHEMA_VERSION}
    if scan_path is not None:
        payload[_SCAN_PATH_KEY] = _normalize_scan_path(scan_path)
    payload.update(entries)

    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    try:
        os.replace(str(tmp_path), str(cache_path))
    except OSError:
        # Clean up the orphaned tmpfile so a failed replace (cross-device,
        # permission) does not leave a stale .json.tmp behind; the original
        # snapshot stays intact (atomicity preserved). Re-raise per the
        # metadata.json persist pattern.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class Edge:
    """Represents a call edge from one function to another.

    Attributes:
        from_file: Source file path (relative to project root)
        from_func: Source function name
        to_file: Target file path (relative to project root)
        to_func: Target function name
    """
    from_file: str
    from_func: str
    to_file: str
    to_func: str

    def to_tuple(self) -> tuple[str, str, str, str]:
        """Convert to the tuple format used by ProjectCallGraph."""
        return (self.from_file, self.from_func, self.to_file, self.to_func)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-1 hash of file content.

    Args:
        file_path: Absolute path to the file

    Returns:
        40-character hex string representing the SHA-1 hash

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(file_path)
    content = path.read_bytes()
    return hashlib.sha1(content, usedforsecurity=False).hexdigest()


def has_file_changed(file_path: str, cached_hash: str) -> bool:
    """Check if file content has changed from cached hash.

    Args:
        file_path: Absolute path to the file
        cached_hash: Previously computed SHA-1 hash

    Returns:
        True if file has changed (or doesn't exist), False otherwise
    """
    try:
        current_hash = compute_file_hash(file_path)
        return current_hash != cached_hash
    except (FileNotFoundError, IOError):
        # Missing or unreadable file is considered "changed"
        return True


def extract_edges_from_file(
    file_path: str,
    lang: str = "python",
    project_root: Optional[str] = None
) -> List[Edge]:
    """Extract call edges from a single source file.

    Args:
        file_path: Absolute path to the source file
        lang: Language - "python", "typescript", "go", or "rust"
        project_root: Optional project root for computing relative paths

    Returns:
        List of Edge objects representing intra-file calls
    """
    path = Path(file_path)

    if project_root:
        root = Path(project_root)
        try:
            rel_path = path.relative_to(root)
            file_name = str(rel_path)
        except ValueError:
            file_name = path.name
    else:
        file_name = path.name

    # Get the appropriate extractor based on language
    if lang == "python":
        extractor = _extract_file_calls
    elif lang == "typescript":
        extractor = _extract_ts_file_calls
    elif lang == "go":
        extractor = _extract_go_file_calls
    elif lang == "rust":
        extractor = _extract_rust_file_calls
    else:
        raise ValueError(f"Unsupported language: {lang}")

    try:
        # Use project root or file's parent as root
        root_path = Path(project_root) if project_root else path.parent
        calls_by_func = extractor(path, root_path)
    except Exception:
        return []

    edges = []
    for caller_func, calls in calls_by_func.items():
        for call_type, call_target in calls:
            # Only include intra-file calls for now
            # Cross-file resolution requires the full function index
            if call_type == 'intra':
                edges.append(Edge(
                    from_file=file_name,
                    from_func=caller_func,
                    to_file=file_name,
                    to_func=call_target
                ))
            elif call_type == 'ref':
                # Function references (e.g., higher-order)
                edges.append(Edge(
                    from_file=file_name,
                    from_func=caller_func,
                    to_file=file_name,
                    to_func=call_target
                ))

    return edges


def patch_call_graph(
    graph: ProjectCallGraph,
    edited_file: str,
    project_root: str,
    lang: str = "python"
) -> ProjectCallGraph:
    """Incrementally update call graph for an edited file.

    This is the core incremental update algorithm:
    1. Remove all edges where from_file == edited_file
    2. Extract new edges from the edited file
    3. Add new edges to the graph
    4. Return the updated graph

    Args:
        graph: Existing ProjectCallGraph to patch
        edited_file: Absolute path to the edited file
        project_root: Project root directory
        lang: Language - "python", "typescript", "go", or "rust"

    Returns:
        Updated ProjectCallGraph (modifies in place and returns same object)
    """
    edited_path = Path(edited_file)
    root_path = Path(project_root)

    # Compute relative path for matching
    try:
        rel_path = str(edited_path.relative_to(root_path))
    except ValueError:
        rel_path = edited_path.name

    # Step 1: Remove all edges FROM the edited file
    edges_to_remove = set()
    for edge in graph.edges:
        src_file, src_func, dst_file, dst_func = edge
        if src_file == rel_path:
            edges_to_remove.add(edge)

    # Remove the edges (modify internal state)
    graph._edges -= edges_to_remove

    # Step 2: Extract new edges from the edited file
    new_edges = extract_edges_from_file(
        str(edited_file),
        lang=lang,
        project_root=project_root
    )

    # Step 3: Add new edges to the graph
    for edge in new_edges:
        graph.add_edge(edge.from_file, edge.from_func, edge.to_file, edge.to_func)

    return graph


def get_file_hash_cache(project_root: str) -> dict[str, str]:
    """DEPRECATED: Use load_snapshot instead.

    This narrow interface is superseded by load_snapshot which returns the
    wider SnapshotEntry format (sha1, mtime_ns, size, inode). Kept for
    back-compat only; all internal callers have migrated to load_snapshot.

    Args:
        project_root: Project root directory

    Returns:
        Dict mapping relative file paths to their SHA-1 hashes
    """
    import warnings
    warnings.warn(
        "get_file_hash_cache is deprecated; use load_snapshot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # A-3: the snapshot on disk is now WIDE (SnapshotEntry dicts). Delegate to
    # load_snapshot and re-narrow to the legacy {rel: sha1} shape so back-compat
    # callers still get the string map they expect (not dict-of-dicts).
    return {rel: entry["sha1"] for rel, entry in load_snapshot(project_root).items()}


def save_file_hash_cache(project_root: str, cache: dict[str, str]) -> None:
    """DEPRECATED: Use save_snapshot instead.

    This narrow interface is superseded by save_snapshot which handles the
    wider SnapshotEntry format (sha1, mtime_ns, size, inode). Kept for
    back-compat only; all internal callers have migrated to save_snapshot.

    Args:
        project_root: Project root directory
        cache: Dict mapping relative file paths to their SHA-1 hashes
    """
    import warnings
    warnings.warn(
        "save_file_hash_cache is deprecated; use save_snapshot instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # A-3: delegate to save_snapshot, wrapping the narrow {rel: sha1} into
    # the wide SnapshotEntry format so the on-disk snapshot is always current.
    from tldr.patch import save_snapshot
    entries = {rel: {"sha1": sha1, "mtime_ns": 0, "size": -1, "inode": -1}
               for rel, sha1 in cache.items()}
    save_snapshot(project_root, entries)


def patch_dirty_files(
    graph: ProjectCallGraph,
    project_root: str,
    dirty_files: list[str],
    lang: str = "python"
) -> ProjectCallGraph:
    """Patch the graph for all dirty files.

    This is the main entry point for incremental updates when the dirty
    flag system reports changed files.

    Args:
        graph: Existing ProjectCallGraph to patch
        project_root: Project root directory
        dirty_files: List of relative file paths that changed
        lang: Language for all files

    Returns:
        Updated ProjectCallGraph
    """
    root_path = Path(project_root)

    for rel_file in dirty_files:
        abs_file = root_path / rel_file
        if abs_file.exists():
            graph = patch_call_graph(graph, str(abs_file), project_root, lang=lang)

    return graph
