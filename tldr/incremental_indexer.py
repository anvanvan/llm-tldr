"""Incremental semantic-index assembly seam.

``IncrementalIndexer`` encapsulates the full incremental-or-full-rebuild
decision for ``build_semantic_index`` so that the orchestrator only has to
extract units, recompute the call graph, and hash the embedding text.

The class has four cohesive stages forming one lifecycle:

  load_previous -> plan -> assemble -> persist

``plan()`` and ``assemble()`` are PURE FUNCTIONS — they take ordinary Python
objects / NumPy arrays in and return them out, with no model, no FAISS I/O,
and no filesystem access. That makes the hash-gate and the row-alignment
invariant testable in milliseconds without mocks.

Only the embedding VECTOR is reused across runs (gated per-unit by
``text_hash``). All units are re-extracted and the full call graph recomputed
every run by the orchestrator, so cross-file caller drift (4b) is handled
automatically: a caller change alters ``build_embedding_text``, which alters
the hash, which routes the unit to ``encode_units``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tldr.patch import SnapshotEntry


def text_hash(text: str) -> str:
    """Return the 64-char SHA-256 hex digest of ``text``.

    This is the per-unit fingerprint that gates embedding reuse. It is computed
    over the exact bytes of ``build_embedding_text(unit)`` (which already folds
    in callers/callees), so any change to a unit's L1-L5 text — including a new
    cross-file caller — produces a different hash and forces a re-embed.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IncrementalState:
    """Typed value returned by ``load_previous``.

    Carries everything the orchestrator needs for the incremental decision.

    ``old_file_hashes`` is I/O state for the orchestrator (the L2 file-hash
    comparison / dirty-files optimization); the pure seams (``plan`` /
    ``assemble``) never touch it.
    """

    old_matrix: np.ndarray            # (N_old, dim) from reconstruct_n; (0, 0) on full_rebuild
    old_hashes: Dict[str, Tuple[int, str]]  # qualified_name -> (row_index, text_hash)
    old_dimension: int                # from old_metadata["dimension"]; 0 on full_rebuild
    full_rebuild: bool
    # I-1: the self-validating floor reads the wide snapshot directly via
    # ``patch.load_snapshot``, so the narrow per-run ``get_file_hash_cache`` read is
    # no longer done in ``load_previous``. The field is retained (default {}) only
    # for backward-compatible construction by callers that still pass it; nothing
    # in semantic.py reads it.
    old_file_hashes: Dict[str, str] = field(default_factory=dict)
    # Persisted unit dicts (metadata["units"]) for the parse-skip carry-forward:
    # build_semantic_index rehydrates unchanged units from these via
    # EmbeddingUnit.from_dict instead of re-parsing them. Empty list on every
    # full-rebuild / migration-gate path so carry-forward is never attempted when
    # the cached vectors are not reusable.
    old_units: List[dict] = field(default_factory=list)
    # S-5: the previous run's WIDE file-hash snapshot ({rel_path -> SnapshotEntry}),
    # loaded ONCE here so persist-time hashing (_compute_current_file_hashes) can
    # reuse a stat-unchanged file's already-known sha1 instead of re-hashing it.
    # ``{}`` on first run / full rebuild (no prior snapshot, or vectors not reusable
    # so every file must be hashed fresh anyway).
    old_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexPlan:
    """Result of ``plan``: which old rows to reuse, which units to re-embed."""

    reuse_rows: Dict[str, int] = field(default_factory=dict)  # qualified_name -> old row index
    encode_units: List = field(default_factory=list)          # units needing a fresh embedding


class IncrementalIndexer:
    """Side-effect-free orchestration helper for incremental semantic indexing.

    The index.faiss + metadata.json live under
    ``<project_root>/.tldr/cache/semantic/``; the file-hash snapshot lives at
    ``<project_root>/.tldr/cache/file_hashes.json`` and is reached through
    ``tldr.patch.load_snapshot`` / ``save_snapshot``.
    """

    def __init__(self, project_root: str, scan_path: "str | None" = None) -> None:
        self.project_root = str(project_root)
        self._semantic_dir = Path(project_root) / ".tldr" / "cache" / "semantic"
        # A-1: the scan_path these snapshot keys are relative to. When provided,
        # the wide snapshot is stamped with a scan_path header on persist and
        # validated against it on load (a mismatch -> safe re-hash-all). None
        # preserves the legacy header-less behavior (no validation).
        self._scan_path = str(scan_path) if scan_path is not None else None

    # ------------------------------------------------------------------
    # load_previous
    # ------------------------------------------------------------------
    def load_previous(self, model_name: str, force_full: bool = False) -> IncrementalState:
        """Read the previous index/metadata and decide whether to rebuild fully.

        On ``force_full=True`` this short-circuits immediately — it does NOT
        read the old index or call ``reconstruct_n`` — and returns an empty
        full-rebuild state (T-8).

        Otherwise it reads index.faiss + metadata.json + file_hashes.json,
        reconstructs the old vectors, and triggers a full rebuild for any of the
        fallback conditions (missing/corrupt index, missing metadata, model
        mismatch, ntotal != len(units) [G-2], missing text_hash migration gate).
        """
        empty_full = IncrementalState(
            old_matrix=np.empty((0, 0), dtype=np.float32),
            old_hashes={},
            old_dimension=0,
            full_rebuild=True,
        )

        if force_full:
            return empty_full

        # I-1: the narrow ``get_file_hash_cache`` read is intentionally NOT done
        # here anymore — the self-validating floor reads the wide snapshot via
        # ``patch.load_snapshot`` instead, so loading the narrow cache into state
        # on every run was a redundant per-run file read with zero readers.

        index_file = self._semantic_dir / "index.faiss"
        metadata_file = self._semantic_dir / "metadata.json"

        # Missing index or metadata -> first run / deleted cache -> full rebuild.
        if not index_file.exists() or not metadata_file.exists():
            return empty_full

        # Metadata must parse.
        try:
            old_metadata = json.loads(metadata_file.read_text())
        except (json.JSONDecodeError, OSError):
            return empty_full

        old_units = old_metadata.get("units", [])

        # Model changed -> embeddings incomparable -> full rebuild.
        if old_metadata.get("model") != model_name:
            return empty_full

        # Migration gate: old metadata predates text_hash -> silent full rebuild
        # (one rebuild, then incremental thereafter). Guarded for empty units.
        if old_units and old_units[0].get("text_hash") is None:
            return empty_full

        # Load the old index; corrupt/unreadable -> full rebuild.
        try:
            import faiss

            old_index = faiss.read_index(str(index_file))
        except (OSError, RuntimeError):
            return empty_full

        # G-2: ntotal vs len(old_units) must agree BEFORE building reuse rows,
        # otherwise assemble() could index out of bounds on a partial write.
        if old_index.ntotal != len(old_units):
            return empty_full

        old_dimension = int(old_metadata.get("dimension", 0))

        # Reconstruct old vectors (IndexFlatIP supports reconstruct_n directly).
        if old_index.ntotal > 0:
            old_matrix = old_index.reconstruct_n(0, old_index.ntotal)
            old_matrix = np.asarray(old_matrix, dtype=np.float32)
        else:
            old_matrix = np.empty((0, old_dimension), dtype=np.float32)

        # Build old_hashes in MATRIX-ROW ORDER (N-2/N-9): dict insertion order ==
        # faiss row order, so plan() can derive a reused unit's row index from
        # the stored tuple's first element.
        old_hashes: Dict[str, Tuple[int, str]] = {}
        for row_idx, unit in enumerate(old_units):
            qname = unit.get("qualified_name")
            if qname is None:
                continue
            old_hashes[qname] = (row_idx, unit.get("text_hash", ""))

        # S-5: load the prior wide snapshot ONCE here (it is keyed by the same
        # scan_path-relative rel paths as unit.file). _compute_current_file_hashes
        # reuses a stat-unchanged file's sha1 from this map instead of re-hashing.
        # Only loaded on this successful non-full-rebuild return; every early
        # return above leaves old_snapshot at its {} default (full rebuild hashes
        # every file fresh anyway, so the reuse map is intentionally empty there).
        from tldr.patch import load_snapshot
        old_snapshot = load_snapshot(self.project_root, scan_path=self._scan_path)

        return IncrementalState(
            old_matrix=old_matrix,
            old_hashes=old_hashes,
            old_dimension=old_dimension,
            full_rebuild=False,
            # Carry the persisted unit dicts so the orchestrator can rehydrate
            # unchanged units (parse-skip) without re-parsing. Only populated on
            # this successful non-full-rebuild return; every early return above
            # leaves old_units at its [] default (no carry-forward).
            old_units=old_units,
            old_snapshot=old_snapshot,
        )

    # ------------------------------------------------------------------
    # plan (PURE)
    # ------------------------------------------------------------------
    def plan(self, new_units: List, state: IncrementalState) -> IndexPlan:
        """Split ``new_units`` into reusable rows and units needing re-embedding.

        PURE FUNCTION — no I/O, no model. For each unit keyed by
        ``qualified_name``: if the old hash exists and matches the unit's
        ``text_hash`` (and we are not in full-rebuild mode), reuse the old row;
        otherwise the unit is queued for encoding. Deleted units (present in
        ``old_hashes`` but absent from ``new_units``) simply never appear in the
        output.
        """
        reuse_rows: Dict[str, int] = {}
        encode_units: List = []

        for unit in new_units:
            qname = unit.qualified_name
            old_entry = None if state.full_rebuild else state.old_hashes.get(qname)
            if old_entry is not None:
                old_row, old_hash = old_entry
                if old_hash == unit.text_hash:
                    reuse_rows[qname] = old_row
                    continue
            encode_units.append(unit)

        return IndexPlan(reuse_rows=reuse_rows, encode_units=encode_units)

    # ------------------------------------------------------------------
    # assemble (PURE)
    # ------------------------------------------------------------------
    def assemble(
        self,
        new_units: List,
        plan: IndexPlan,
        old_matrix: np.ndarray,
        fresh_vectors: np.ndarray,
    ) -> np.ndarray:
        """Build the final matrix in ``new_units`` order (row-alignment invariant).

        PURE FUNCTION. Row ``i`` of the result corresponds to ``new_units[i]`` —
        reused units pull their old row from ``old_matrix`` (via
        ``plan.reuse_rows``), fresh units consume the next row of
        ``fresh_vectors`` in ``plan.encode_units`` order. ``fresh_vectors`` may
        be shape ``(0, dim)`` (no-op reindex), in which case every row comes
        from ``old_matrix``.

        Postcondition: ``result.shape[0] == len(new_units)``; never append-only.
        """
        # Determine the embedding dimension from whichever source has columns.
        dim = 0
        if fresh_vectors.ndim == 2 and fresh_vectors.shape[1] > 0:
            dim = fresh_vectors.shape[1]
        elif old_matrix.ndim == 2 and old_matrix.shape[1] > 0:
            dim = old_matrix.shape[1]

        if not new_units:
            return np.empty((0, dim), dtype=np.float32)

        # Map each unit queued for encoding to its row in fresh_vectors, in the
        # exact order plan.encode_units was built (== the order rows were encoded).
        fresh_index: Dict[str, int] = {}
        for i, unit in enumerate(plan.encode_units):
            fresh_index[unit.qualified_name] = i

        rows: List[np.ndarray] = []
        for unit in new_units:
            qname = unit.qualified_name
            if qname in plan.reuse_rows:
                rows.append(np.asarray(old_matrix[plan.reuse_rows[qname]], dtype=np.float32))
            else:
                rows.append(np.asarray(fresh_vectors[fresh_index[qname]], dtype=np.float32))

        return np.vstack(rows).astype(np.float32)

    # ------------------------------------------------------------------
    # persist
    # ------------------------------------------------------------------
    def persist(
        self,
        index,
        units: List,
        model_name: str,
        dimension: int,
        file_hash_cache: Dict[str, "SnapshotEntry"],
    ) -> None:
        """Write index.faiss + metadata.json + the wide file-hash snapshot.

        The FAISS index is always rewritten in full (never append-only) so the
        on-disk rows stay aligned with metadata after deletions/reorderings.

        [G-4] PERSIST ORDER INVARIANT (hard): the three writes happen in exactly
        this order — ``faiss.write_index`` -> ``metadata.json`` (atomic
        ``os.replace``) -> ``file_hashes.json`` snapshot (atomic ``os.replace``).
        Writing the snapshot LAST guarantees a crash mid-persist leaves the
        system false-DIRTY (next run re-derives + re-indexes — safe), NEVER
        false-clean (a fresh snapshot would otherwise make a stale index look
        current).

        ``file_hash_cache`` carries the WIDE SnapshotEntry dicts
        ``{rel: {sha1, mtime_ns, size, inode}}`` produced by
        ``_compute_current_file_hashes`` (cached sha1 + one os.stat per file — no
        second hash pass).
        """
        import faiss
        import time

        from tldr.patch import save_snapshot

        self._semantic_dir.mkdir(parents=True, exist_ok=True)

        index_file = self._semantic_dir / "index.faiss"
        metadata_file = self._semantic_dir / "metadata.json"

        # (1) FAISS index.
        faiss.write_index(index, str(index_file))

        # (2) metadata.json (atomic) — includes the explicit epoch token (T-4).
        metadata = {
            "units": [u.to_dict() for u in units],
            "model": model_name,
            "dimension": dimension,
            "count": len(units),
            # T-4: explicit continuity token (NOT the file mtime). The daemon
            # compares its _watch_start_epoch against this value to decide whether
            # the dirty-files hint is provably continuous.
            "index_epoch": int(time.time_ns()),
        }
        tmp_metadata_file = metadata_file.with_suffix(".json.tmp")
        tmp_metadata_file.write_text(json.dumps(metadata, indent=2))
        try:
            os.replace(tmp_metadata_file, metadata_file)
        except OSError:
            # Clean up the orphaned tmpfile so a failed replace (cross-device,
            # permission) does not leave a stale .json.tmp behind; the original
            # metadata stays intact (atomicity preserved).
            try:
                tmp_metadata_file.unlink()
            except OSError:
                pass
            raise

        # (3) wide snapshot (atomic) — LAST, so a crash here is false-DIRTY.
        save_snapshot(self.project_root, file_hash_cache, scan_path=self._scan_path)
