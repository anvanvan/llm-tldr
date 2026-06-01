"""
Unit tests for IncrementalIndexer — RED phase.

This file covers PURE-COMPONENT behaviors of the not-yet-existing
tldr/incremental_indexer.py module.  The file COLLECTS cleanly; each test
FAILS at runtime (FAIL, not ERROR) because _imp() raises ImportError inside
the test's call phase.

Behaviors covered (one test per behavior):
  1. text_hash determinism — hash is stable; to_dict() emits 'text_hash'
  2. plan() — unchanged unit goes to reuse_rows
  3. plan() — changed unit (same key, different hash) goes to encode_units
  4. plan() — brand-new unit goes to encode_units
  5. plan() — deleted unit silently drops from output
  6. assemble() — row-alignment invariant (mixed reuse + fresh)
  7. assemble() — reordering rebuilds from new_units order
  8. assemble() — empty encode case (shape-only reuse, no IndexError)
  9. load_previous() — full-rebuild triggers: model mismatch,
     missing text_hash, ntotal != len(old_units), corrupt index, missing index
 10. load_previous() — force_full=True short-circuits all I/O
 11. load_previous() — old_dimension read from metadata (dim-mismatch seam)
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest

# EmbeddingUnit and build_embedding_text exist today — safe to import at module level.
from tldr.semantic import EmbeddingUnit, build_embedding_text
from tldr.incremental_indexer import text_hash


# ---------------------------------------------------------------------------
# Lazy importer — raises ImportError INSIDE each test body (FAIL not ERROR)
# ---------------------------------------------------------------------------

def _imp():
    """Import not-yet-existing names from tldr.incremental_indexer.

    Called at the TOP of each test body that needs these names.  When the
    module does not exist, ImportError is raised during the test's CALL phase,
    so pytest reports it as FAILED (F), not a collection ERROR (E).
    """
    from tldr.incremental_indexer import (  # noqa: PLC0415
        IncrementalIndexer,
        IncrementalState,
        IndexPlan,
    )
    return IncrementalIndexer, IncrementalState, IndexPlan


# ---------------------------------------------------------------------------
# Module-level helpers that do NOT reference not-yet-existing names
# ---------------------------------------------------------------------------

def _make_unit(
    name: str,
    *,
    signature: str = "",
    docstring: str = "",
    calls: List[str] = None,
    called_by: List[str] = None,
    unit_type: str = "function",
) -> EmbeddingUnit:
    """Construct a minimal EmbeddingUnit for tests (no incremental_indexer needed)."""
    return EmbeddingUnit(
        name=name,
        qualified_name=f"module.{name}",
        file="module.py",
        line=1,
        language="python",
        unit_type=unit_type,
        signature=signature or f"def {name}():",
        docstring=docstring or f"Docstring for {name}.",
        calls=calls or [],
        called_by=called_by or [],
    )


def _make_state(IncrementalState, old_hashes=None, old_matrix=None,
                old_dimension=4, full_rebuild=False):
    """Construct an IncrementalState value; caller passes in the class."""
    if old_matrix is None:
        n = len(old_hashes) if old_hashes else 0
        old_matrix = (
            np.zeros((n, old_dimension), dtype=np.float32)
            if n > 0
            else np.empty((0, 0))
        )
    return IncrementalState(
        old_matrix=old_matrix,
        old_hashes=old_hashes or {},
        old_file_hashes={},
        old_dimension=old_dimension,
        full_rebuild=full_rebuild,
    )


# ---------------------------------------------------------------------------
# Test 1 — text_hash determinism
# ---------------------------------------------------------------------------

class TestTextHashDeterminism:
    """text_hash must be deterministic; EmbeddingUnit must expose the field."""

    def test_same_unit_produces_same_hash_twice(self):
        """Hashing build_embedding_text(unit) twice yields the same hash.

        The text_hash field is set on the unit and read back to confirm the
        dataclass field exists (RED: field not yet declared on EmbeddingUnit,
        so unit.text_hash is an ordinary dynamic attribute — the real RED is
        that to_dict() does not emit it, tested below).
        """
        _imp()  # raises ImportError until module exists → FAIL
        unit = _make_unit("foo", signature="def foo(): ...", docstring="Does foo.")
        text = build_embedding_text(unit)
        hash1 = text_hash(text)
        hash2 = text_hash(text)
        assert hash1 == hash2, "hash must be deterministic"
        unit.text_hash = hash1
        assert unit.text_hash == hash1

    def test_different_units_get_different_hashes(self):
        """Two units with different embedding text must produce different hashes."""
        _imp()  # raises ImportError until module exists → FAIL
        unit_a = _make_unit("alpha", docstring="Does alpha.")
        unit_b = _make_unit("beta", docstring="Does something completely different.")
        hash_a = text_hash(build_embedding_text(unit_a))
        hash_b = text_hash(build_embedding_text(unit_b))
        assert hash_a != hash_b, (
            "Different units must produce different hashes; "
            "otherwise the hash gate would suppress re-embedding after changes"
        )

    def test_text_hash_field_present_in_to_dict(self):
        """EmbeddingUnit.to_dict() must include a 'text_hash' key.

        RED: to_dict() currently does not emit this key — AssertionError.
        (This test would still FAIL even after the module is created, until
        EmbeddingUnit.to_dict() is also updated.)
        """
        _imp()  # raises ImportError until module exists → FAIL
        unit = _make_unit("foo")
        unit.text_hash = text_hash(build_embedding_text(unit))
        d = unit.to_dict()
        assert "text_hash" in d, (
            "EmbeddingUnit.to_dict() must emit 'text_hash'; "
            "currently the key is absent from to_dict()"
        )


# ---------------------------------------------------------------------------
# Test 2 — plan() unchanged unit goes to reuse_rows
# ---------------------------------------------------------------------------

class TestPlanUnchangedUnit:
    """plan() must put unchanged units (matching text_hash) into reuse_rows."""

    def test_unchanged_unit_in_reuse_rows_not_encode(self):
        """Given a unit whose text_hash matches old_hashes, plan() must place
        it in reuse_rows, NOT in encode_units.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        unit = _make_unit("foo")
        h = text_hash(build_embedding_text(unit))
        unit.text_hash = h

        state = _make_state(
            IncrementalState,
            old_hashes={"module.foo": (0, h)},
            old_matrix=np.arange(4, dtype=np.float32).reshape(1, 4),
            old_dimension=4,
        )

        indexer = IncrementalIndexer("/fake/root")
        plan = indexer.plan([unit], state)

        assert "module.foo" in plan.reuse_rows, (
            "Unchanged unit must appear in reuse_rows"
        )
        assert plan.reuse_rows["module.foo"] == 0, (
            "reuse_rows must map qualified_name to the old row index"
        )
        assert unit not in plan.encode_units, (
            "Unchanged unit must NOT appear in encode_units"
        )


# ---------------------------------------------------------------------------
# Test 3 — plan() changed unit goes to encode_units
# ---------------------------------------------------------------------------

class TestPlanChangedUnit:
    """plan() must route a unit with a different text_hash to encode_units."""

    def test_changed_unit_in_encode_units_not_reuse(self):
        """A unit with the same key but a different text_hash must go to
        encode_units (not reuse_rows).
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        unit = _make_unit("bar", docstring="New docstring after edit.")
        new_hash = text_hash(build_embedding_text(unit))
        unit.text_hash = new_hash

        old_hash = text_hash("completely different text")
        state = _make_state(
            IncrementalState,
            old_hashes={"module.bar": (0, old_hash)},
            old_matrix=np.arange(4, dtype=np.float32).reshape(1, 4),
            old_dimension=4,
        )

        indexer = IncrementalIndexer("/fake/root")
        plan = indexer.plan([unit], state)

        assert unit in plan.encode_units, (
            "Changed unit (same key, different hash) must be in encode_units"
        )
        assert "module.bar" not in plan.reuse_rows, (
            "Changed unit must NOT be in reuse_rows"
        )


# ---------------------------------------------------------------------------
# Test 4 — plan() brand-new unit goes to encode_units
# ---------------------------------------------------------------------------

class TestPlanNewUnit:
    """plan() must route a brand-new unit (key absent from old_hashes) to encode_units."""

    def test_new_unit_in_encode_units(self):
        """A unit whose qualified_name is absent from old_hashes must land in
        encode_units.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        unit = _make_unit("baz", docstring="Totally new function.")
        unit.text_hash = text_hash(build_embedding_text(unit))

        state = _make_state(IncrementalState, old_hashes={}, old_dimension=4)

        indexer = IncrementalIndexer("/fake/root")
        plan = indexer.plan([unit], state)

        assert unit in plan.encode_units, (
            "New unit (key absent from old_hashes) must be in encode_units"
        )
        assert "module.baz" not in plan.reuse_rows, (
            "New unit must NOT be in reuse_rows"
        )


# ---------------------------------------------------------------------------
# Test 5 — plan() deleted unit drops silently
# ---------------------------------------------------------------------------

class TestPlanDeletedUnit:
    """A unit present in old_hashes but absent from new_units must not appear
    in any plan output."""

    def test_deleted_unit_absent_from_plan(self):
        """Deleted unit: in old_hashes, not in new_units — must not appear in
        reuse_rows or encode_units.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        unit_alive = _make_unit("alive", docstring="Still here.")
        alive_hash = text_hash(build_embedding_text(unit_alive))
        unit_alive.text_hash = alive_hash

        state = _make_state(
            IncrementalState,
            old_hashes={
                "module.alive": (0, alive_hash),
                "module.deleted": (1, "somehash"),
            },
            old_matrix=np.arange(8, dtype=np.float32).reshape(2, 4),
            old_dimension=4,
        )

        indexer = IncrementalIndexer("/fake/root")
        plan = indexer.plan([unit_alive], state)

        assert "module.deleted" not in plan.reuse_rows, (
            "Deleted unit must not appear in reuse_rows"
        )
        deleted_in_encode = any(
            u.qualified_name == "module.deleted" for u in plan.encode_units
        )
        assert not deleted_in_encode, (
            "Deleted unit must not appear in encode_units"
        )
        assert "module.alive" in plan.reuse_rows


# ---------------------------------------------------------------------------
# Test 6 — assemble() row-alignment invariant
# ---------------------------------------------------------------------------

class TestAssembleRowAlignment:
    """assemble() must place each unit's vector at row i == new_units[i]."""

    def test_row_alignment_mixed_reuse_and_fresh(self):
        """Given two units — one reused, one freshly encoded — assemble()
        must return a matrix where row 0 is the reused vector and row 1 is
        the fresh vector; shape[0] == len(new_units).
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        dim = 4
        unit_reused = _make_unit("old_fn")
        unit_reused.text_hash = "oldhash"
        unit_fresh = _make_unit("new_fn")
        unit_fresh.text_hash = "newhash"

        old_vec = np.array([[10.0, 11.0, 12.0, 13.0]], dtype=np.float32)
        old_matrix = old_vec  # shape (1, 4)
        fresh_vec = np.array([[20.0, 21.0, 22.0, 23.0]], dtype=np.float32)

        plan = IndexPlan(
            reuse_rows={"module.old_fn": 0},
            encode_units=[unit_fresh],
        )

        new_units = [unit_reused, unit_fresh]
        indexer = IncrementalIndexer("/fake/root")
        result = indexer.assemble(new_units, plan, old_matrix, fresh_vec)

        assert result.shape[0] == len(new_units), (
            f"result.shape[0] must equal len(new_units)={len(new_units)}, "
            f"got {result.shape[0]}"
        )
        assert result.shape[1] == dim
        np.testing.assert_array_equal(
            result[0], old_vec[0],
            err_msg="Row 0 must be the reused vector from old_matrix[0]",
        )
        np.testing.assert_array_equal(
            result[1], fresh_vec[0],
            err_msg="Row 1 must be the fresh vector for new_fn",
        )


# ---------------------------------------------------------------------------
# Test 7 — assemble() reordering
# ---------------------------------------------------------------------------

class TestAssembleReordering:
    """assemble() must place each unit at its NEW index, not the old index."""

    def test_reordering_rebuilds_from_new_units_order(self):
        """If new_units reverses the order of existing units, each unit's
        vector must appear at its NEW position — proving order is rebuilt
        from new_units, never append-only.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        dim = 4
        unit_a = _make_unit("func_a")
        unit_a.text_hash = "hash_a"
        unit_b = _make_unit("func_b")
        unit_b.text_hash = "hash_b"

        vec_a = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        vec_b = np.array([[5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        old_matrix = np.vstack([vec_a, vec_b])  # shape (2, 4): row0=a, row1=b

        # new_units reverses order: [func_b, func_a]
        new_units = [unit_b, unit_a]

        plan = IndexPlan(
            reuse_rows={
                "module.func_a": 0,  # func_a was at old row 0
                "module.func_b": 1,  # func_b was at old row 1
            },
            encode_units=[],
        )

        fresh_vectors = np.empty((0, dim), dtype=np.float32)

        indexer = IncrementalIndexer("/fake/root")
        result = indexer.assemble(new_units, plan, old_matrix, fresh_vectors)

        assert result.shape == (2, dim)
        # new row 0 must be func_b (was old row 1)
        np.testing.assert_array_equal(
            result[0], vec_b[0],
            err_msg="Row 0 must be func_b's vector (new index 0)",
        )
        # new row 1 must be func_a (was old row 0)
        np.testing.assert_array_equal(
            result[1], vec_a[0],
            err_msg="Row 1 must be func_a's vector (new index 1)",
        )


# ---------------------------------------------------------------------------
# Test 8 — assemble() empty encode case
# ---------------------------------------------------------------------------

class TestAssembleEmptyEncode:
    """assemble() must handle fresh_vectors of shape (0, dim) without error."""

    def test_empty_fresh_vectors_returns_all_from_old_matrix(self):
        """When plan.encode_units is empty, fresh_vectors is np.empty((0, dim)).
        assemble() must return all rows from old_matrix in new order,
        shape (len(new_units), dim), with no IndexError.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        dim = 4
        unit_x = _make_unit("x_fn")
        unit_x.text_hash = "hash_x"
        unit_y = _make_unit("y_fn")
        unit_y.text_hash = "hash_y"

        vec_x = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        vec_y = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        old_matrix = np.vstack([vec_x, vec_y])  # shape (2, 4)

        new_units = [unit_x, unit_y]

        plan = IndexPlan(
            reuse_rows={"module.x_fn": 0, "module.y_fn": 1},
            encode_units=[],
        )

        fresh_vectors = np.empty((0, dim), dtype=np.float32)

        indexer = IncrementalIndexer("/fake/root")
        result = indexer.assemble(new_units, plan, old_matrix, fresh_vectors)

        assert result.shape == (2, dim), (
            f"Expected shape (2, {dim}), got {result.shape}"
        )
        np.testing.assert_array_equal(result[0], vec_x[0])
        np.testing.assert_array_equal(result[1], vec_y[0])


# ---------------------------------------------------------------------------
# Test 9 — load_previous() full-rebuild triggers
# ---------------------------------------------------------------------------

class TestLoadPreviousTriggers:
    """load_previous() must return full_rebuild=True for each trigger condition."""

    def _write_index_and_metadata(
        self,
        semantic_dir: Path,
        *,
        model_name: str = "test-model",
        n_index_rows: int = 2,
        n_metadata_units: int = 2,
        dim: int = 4,
        include_text_hash: bool = True,
    ) -> None:
        """Write a real tiny faiss index + metadata.json under semantic_dir."""
        import faiss

        semantic_dir.mkdir(parents=True, exist_ok=True)

        vecs = np.ones((n_index_rows, dim), dtype=np.float32)
        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)
        faiss.write_index(idx, str(semantic_dir / "index.faiss"))

        units = []
        for i in range(n_metadata_units):
            entry = {
                "name": f"fn_{i}",
                "qualified_name": f"module.fn_{i}",
                "file": "module.py",
                "line": i + 1,
                "language": "python",
                "unit_type": "function",
                "signature": f"def fn_{i}():",
                "docstring": f"Docstring {i}",
                "calls": [],
                "called_by": [],
                "cfg_summary": "",
                "dfg_summary": "",
                "dependencies": "",
                "code_preview": "",
            }
            if include_text_hash:
                entry["text_hash"] = f"fakehash_{i}"
            units.append(entry)

        metadata = {
            "units": units,
            "model": model_name,
            "dimension": dim,
            "count": n_metadata_units,
        }
        (semantic_dir / "metadata.json").write_text(json.dumps(metadata))

    def test_model_mismatch_triggers_full_rebuild(self, tmp_path: Path):
        """load_previous(model_name) where model_name != old metadata model must
        return IncrementalState with full_rebuild=True.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        semantic_dir = tmp_path / ".tldr" / "cache" / "semantic"
        self._write_index_and_metadata(semantic_dir, model_name="old-model")

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("new-model")

        assert state.full_rebuild is True, (
            "Model name mismatch must trigger full_rebuild=True"
        )
        assert state.old_hashes == {}, (
            "full_rebuild state must have empty old_hashes"
        )

    def test_missing_text_hash_in_metadata_triggers_full_rebuild(
        self, tmp_path: Path
    ):
        """load_previous() where old metadata units lack text_hash field must
        return full_rebuild=True (migration gate).
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        semantic_dir = tmp_path / ".tldr" / "cache" / "semantic"
        self._write_index_and_metadata(
            semantic_dir, model_name="test-model", include_text_hash=False
        )

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model")

        assert state.full_rebuild is True, (
            "Missing text_hash in old metadata units must trigger full_rebuild=True "
            "(migration gate for first run after upgrade)"
        )

    def test_ntotal_mismatch_triggers_full_rebuild(self, tmp_path: Path):
        """load_previous() where old_index.ntotal != len(old_units) (G-2) must
        return full_rebuild=True.

        Index has 2 rows; metadata claims 3 units.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        semantic_dir = tmp_path / ".tldr" / "cache" / "semantic"
        self._write_index_and_metadata(
            semantic_dir,
            model_name="test-model",
            n_index_rows=2,
            n_metadata_units=3,  # mismatch: index has 2 rows, metadata claims 3
        )

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model")

        assert state.full_rebuild is True, (
            "ntotal (2) != len(old_units) (3) must trigger full_rebuild=True [G-2]"
        )

    def test_corrupt_index_file_triggers_full_rebuild(self, tmp_path: Path):
        """load_previous() with a corrupt index.faiss must catch the error and
        return full_rebuild=True.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        semantic_dir = tmp_path / ".tldr" / "cache" / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)

        (semantic_dir / "index.faiss").write_bytes(b"not a valid faiss index")

        units = [{
            "name": "fn_0", "qualified_name": "module.fn_0", "file": "module.py",
            "line": 1, "language": "python", "unit_type": "function",
            "signature": "def fn_0():", "docstring": "Docstring 0",
            "calls": [], "called_by": [], "cfg_summary": "", "dfg_summary": "",
            "dependencies": "", "code_preview": "", "text_hash": "somehash",
        }]
        (semantic_dir / "metadata.json").write_text(json.dumps({
            "units": units, "model": "test-model", "dimension": 4, "count": 1,
        }))

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model")

        assert state.full_rebuild is True, (
            "Corrupt index.faiss must be caught and trigger full_rebuild=True"
        )

    def test_missing_index_file_triggers_full_rebuild(self, tmp_path: Path):
        """load_previous() with no index.faiss must return full_rebuild=True
        (first run or file deleted).
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        # No files written at all
        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model")

        assert state.full_rebuild is True, (
            "Missing index.faiss must trigger full_rebuild=True (first run)"
        )


# ---------------------------------------------------------------------------
# Test 10 — load_previous() force_full short-circuits
# ---------------------------------------------------------------------------

class TestLoadPreviousForceFullShortCircuits:
    """load_previous(force_full=True) must skip all I/O and return full_rebuild=True."""

    def test_force_full_returns_empty_state_without_reading_index(
        self, tmp_path: Path, monkeypatch
    ):
        """force_full=True must short-circuit: no faiss.read_index call; state
        has full_rebuild=True, old_dimension=0, old_hashes={}.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        import faiss

        semantic_dir = tmp_path / ".tldr" / "cache" / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)

        vecs = np.ones((2, 4), dtype=np.float32)
        idx = faiss.IndexFlatIP(4)
        idx.add(vecs)
        faiss.write_index(idx, str(semantic_dir / "index.faiss"))

        units = [{
            "name": "fn_0", "qualified_name": "module.fn_0", "file": "module.py",
            "line": 1, "language": "python", "unit_type": "function",
            "signature": "def fn_0():", "docstring": "Docstring 0",
            "calls": [], "called_by": [], "cfg_summary": "", "dfg_summary": "",
            "dependencies": "", "code_preview": "", "text_hash": "fakehash_0",
        }]
        (semantic_dir / "metadata.json").write_text(json.dumps({
            "units": units, "model": "test-model", "dimension": 4, "count": 1,
        }))

        read_index_calls = []
        original_read = faiss.read_index

        def spy_read_index(path):
            read_index_calls.append(path)
            return original_read(path)

        monkeypatch.setattr(faiss, "read_index", spy_read_index)

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model", force_full=True)

        assert state.full_rebuild is True, (
            "force_full=True must produce full_rebuild=True"
        )
        assert state.old_dimension == 0, (
            "force_full=True must return old_dimension=0 (short-circuits metadata read)"
        )
        assert state.old_hashes == {}, (
            "force_full=True must return empty old_hashes"
        )
        assert len(read_index_calls) == 0, (
            "force_full=True must NOT call faiss.read_index — skips all I/O (T-8)"
        )


# ---------------------------------------------------------------------------
# Test 11 — old_dimension readable from state (dim-mismatch seam)
# ---------------------------------------------------------------------------

class TestLoadPreviousOldDimensionFromMetadata:
    """load_previous() must read old_dimension from old_metadata['dimension'].

    This value is required by the orchestrator's post-encode dim-mismatch
    check: if fresh.shape[1] != state.old_dimension, trigger a full rebuild.
    """

    def test_old_dimension_read_from_metadata(self, tmp_path: Path):
        """load_previous() must populate state.old_dimension from
        old_metadata['dimension'] — not hardcode or ignore it.
        """
        IncrementalIndexer, IncrementalState, IndexPlan = _imp()

        import faiss

        dim = 8  # non-default dim to catch any hardcoding to 4
        semantic_dir = tmp_path / ".tldr" / "cache" / "semantic"
        semantic_dir.mkdir(parents=True, exist_ok=True)

        vecs = np.ones((1, dim), dtype=np.float32)
        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)
        faiss.write_index(idx, str(semantic_dir / "index.faiss"))

        units = [{
            "name": "fn_0", "qualified_name": "module.fn_0", "file": "module.py",
            "line": 1, "language": "python", "unit_type": "function",
            "signature": "def fn_0():", "docstring": "Docstring 0",
            "calls": [], "called_by": [], "cfg_summary": "", "dfg_summary": "",
            "dependencies": "", "code_preview": "", "text_hash": "somehash",
        }]
        (semantic_dir / "metadata.json").write_text(json.dumps({
            "units": units, "model": "test-model", "dimension": dim, "count": 1,
        }))

        indexer = IncrementalIndexer(str(tmp_path))
        state = indexer.load_previous("test-model")

        assert state.old_dimension == dim, (
            f"state.old_dimension must equal metadata['dimension']={dim}, "
            f"got {state.old_dimension!r}. "
            "Required for orchestrator post-encode dim-mismatch check "
            "(fresh.shape[1] != state.old_dimension → full rebuild)."
        )
        assert state.full_rebuild is False, (
            "Valid index + matching model + text_hash present must NOT trigger full_rebuild"
        )


# ---------------------------------------------------------------------------
# Test 12 — _build_reapply_call_maps with lang="all" (B-3 regression)
# ---------------------------------------------------------------------------

class TestReapplyCallGraphLangAll:
    """B-3: _build_reapply_call_maps(lang='all') must return populated maps.

    Bug: when lang='all', Pass 1 calls build_project_call_graph(language='all')
    which hits no branch and returns an empty ProjectCallGraph; Pass 2 calls
    scan_project(root, 'all', None) which raises ValueError('Unsupported
    language: all') — swallowed by the except — so both passes yield nothing
    and the function returns empty maps.  _reapply_call_graph then early-returns
    on the 'if not calls_map and not called_by_map' guard, silently skipping
    the entire name-based cross-file drift supplement for multi-language projects.

    The fix will make _build_reapply_call_maps iterate detected code languages
    (via _detect_project_languages) and merge each language's maps when
    lang='all'.  This test is RED until that fix is applied.
    """

    def _make_two_file_project(self, tmp_path: Path) -> Path:
        """Create a minimal Python project where bar() calls foo() across files.

        file_a.py: defines foo()
        file_b.py: defines bar() which calls foo()

        The name-based pass in _build_reapply_call_maps should resolve
        bar -> foo from the same project, so called_by_map['foo'] includes 'bar'.
        """
        (tmp_path / ".git").mkdir(exist_ok=True)

        (tmp_path / "file_a.py").write_text(
            "def foo(x):\n"
            "    \"\"\"Compute foo of x.\"\"\"\n"
            "    return x + 1\n"
        )
        (tmp_path / "file_b.py").write_text(
            "def bar(y):\n"
            "    \"\"\"Compute bar of y, calling foo.\"\"\"\n"
            "    return foo(y) * 2\n"
        )
        return tmp_path

    def test_lang_all_returns_populated_call_maps_for_python_project(
        self, tmp_path: Path
    ):
        """_build_reapply_call_maps(project_path, 'all', unit_names) must return
        a non-empty calls_map or called_by_map when the project contains Python
        files with cross-file calls.

        RED: currently lang='all' causes both Pass 1 (empty graph) and Pass 2
        (ValueError swallowed) to fail silently, returning ({}, {}).  The
        assertion on called_by_map['foo'] therefore fails.

        Expected failure reason: called_by_map is empty ({}), so the key lookup
        raises KeyError or the assertion `'bar' in called_by_map.get('foo', [])`
        is False — proving that multi-language projects silently skip the
        call-graph re-apply.
        """
        from tldr.semantic import _build_reapply_call_maps

        project_path = str(self._make_two_file_project(tmp_path))
        unit_names = {"foo", "bar"}

        calls_map, called_by_map = _build_reapply_call_maps(
            project_path, "all", unit_names
        )

        # With a working fix, the name-based pass (or the per-language pass)
        # must resolve bar -> foo, so called_by_map['foo'] contains 'bar'
        # and/or calls_map['bar'] contains 'foo'.
        assert calls_map or called_by_map, (
            "lang='all' returned empty maps for a Python project with cross-file "
            "calls. Expected _build_reapply_call_maps to detect the project's "
            "Python language and build call edges via the detected-language path. "
            "Bug B-3: both Pass 1 (build_project_call_graph('all') → empty graph) "
            "and Pass 2 (scan_project(root, 'all', None) → ValueError swallowed) "
            "fail silently, leaving the map empty."
        )

        # More specific: bar calls foo, so the edge must appear in at least one map.
        bar_calls_foo = "foo" in calls_map.get("bar", [])
        foo_called_by_bar = "bar" in called_by_map.get("foo", [])
        assert bar_calls_foo or foo_called_by_bar, (
            f"Expected bar->foo edge in calls_map or called_by_map. "
            f"calls_map={calls_map!r}, called_by_map={called_by_map!r}. "
            f"With lang='all', the multi-language path must iterate detected "
            f"languages and merge per-language call maps."
        )
