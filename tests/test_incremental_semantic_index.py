"""
Integration + device + daemon tests for incremental semantic indexing.

Feature: build_semantic_index gains incremental reindex capability via IncrementalIndexer.
         --full flag forces clean rebuild. GPU/Metal is default device on darwin.
         semantic_search resolves device symmetrically. Daemon passes --dirty-files.

All tests are RED until the feature is implemented. They fail because:
  - IncrementalIndexer does not exist yet (tldr/incremental_indexer.py not created)
  - build_semantic_index lacks `full` kwarg
  - _resolve_default_device() does not exist in tldr.semantic
  - semantic_search does not call _resolve_default_device() when device=None
  - CLI lacks --full and --dirty-files args
  - daemon does not write dirty-files temp file

Monkeypatching seam: tldr.semantic.get_model (same as test_semantic_non_code_indexing_bug004.py).
Returns fake deterministic dim-4 L2-normalized vectors; call count is observable.

FAKE FAISS strategy: tests use a real-but-tiny FAISS IndexFlatIP (dim=4, a few units).
This lets reconstruct_n work correctly for vector-reuse assertions while staying fast.
Only faiss.write_index and faiss.read_index are intercepted where needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
# It is a MagicMock whose encode.call_count / call args remain observable.
from conftest import make_fake_model as _make_fake_model

# ---------------------------------------------------------------------------
# Repo root (for CLI subprocess tests)
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).parent.parent)

# ---------------------------------------------------------------------------
# Embedding dimension used by the fake model
# ---------------------------------------------------------------------------
_DIM = 4


# ---------------------------------------------------------------------------
# Tiny repo builder helpers
# ---------------------------------------------------------------------------

_PY_FILE_A = """\
def foo(x):
    \"\"\"Compute foo of x.\"\"\"
    return x + 1
"""

_PY_FILE_B = """\
def bar(y):
    \"\"\"Compute bar of y.\"\"\"
    return y * 2
"""

_PY_FILE_B_WITH_CALL = """\
def bar(y):
    \"\"\"Compute bar of y, calling foo.\"\"\"
    return foo(y) * 2
"""

_PY_FILE_B_MODIFIED = """\
def bar(y):
    \"\"\"Compute bar of y (modified).\"\"\"
    return y * 3
"""

_PY_FILE_C = """\
def baz():
    \"\"\"Simple baz.\"\"\"
    pass
"""


def _build_tiny_repo(tmp_path: Path, *, with_git: bool = True) -> Path:
    """Create a minimal Python project with .git anchor.

    Returns the project root (tmp_path).
    """
    if with_git:
        (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "file_a.py").write_text(_PY_FILE_A)
    (tmp_path / "file_b.py").write_text(_PY_FILE_B)
    return tmp_path


def _build_tiny_repo_three_files(tmp_path: Path) -> Path:
    """Create a repo with three Python files."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "file_a.py").write_text(_PY_FILE_A)
    (tmp_path / "file_b.py").write_text(_PY_FILE_B)
    (tmp_path / "file_c.py").write_text(_PY_FILE_C)
    return tmp_path


# ---------------------------------------------------------------------------
# Helper: read metadata.json from the cache dir
# ---------------------------------------------------------------------------

def _read_metadata(project_root: Path) -> dict:
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"
    return json.loads(meta_path.read_text())


# ===========================================================================
# TEST 1: Initial index then NO-OP reindex re-embeds NOTHING
# ===========================================================================

class TestNoOpReindex:
    """After an initial index with no file changes, a second index call encodes 0 units."""

    def test_noop_reindex_zero_encode_calls(self, tmp_path: Path):
        """Build index on a 2-file repo; reindex with no changes; assert 0 new encode calls.

        Expected RED failure: build_semantic_index does not accept `full` kwarg
        OR IncrementalIndexer does not exist so the incremental path never runs
        and all units are re-embedded on the second call.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- First build: full index ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count1 = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        assert count1 >= 2, f"Expected at least 2 units indexed, got {count1}"

        first_encode_count = fake_model.encode.call_count
        assert first_encode_count >= 1, "First build must call encode at least once"

        # --- Second build: no file changes (incremental should be a no-op) ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count2 = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # CRITICAL ASSERTION: incremental should encode NOTHING when nothing changed
        assert fake_model.encode.call_count == 0, (
            f"Expected 0 encode calls on no-op reindex (all units unchanged), "
            f"got {fake_model.encode.call_count}. "
            f"Reason: IncrementalIndexer.plan() should see all text_hashes match "
            f"the previous run and return encode_units=[]."
        )
        assert count2 == count1, (
            f"Unit count must stay the same on no-op reindex: {count2} != {count1}"
        )


# ===========================================================================
# TEST 2: ONE-FILE-CHANGE reindexes only changed units
# ===========================================================================

class TestOneFileChangeReindex:
    """Editing one file triggers re-embedding only for that file's units."""

    def test_one_file_change_encodes_only_changed_units(self, tmp_path: Path):
        """Build index; edit file_b.py; reindex; assert encode called only for file_b units.

        Expected RED failure: incremental path doesn't exist; build_semantic_index
        re-embeds ALL units regardless of changes.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- First build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )
        total_units = fake_model.encode.call_count
        # total_units is how many encode batches happened; we need total embed count
        # Collect how many texts were encoded in total
        total_texts_first = sum(
            len(c.args[0]) if c.args else 0
            for c in fake_model.encode.call_args_list
        )
        assert total_texts_first >= 2, f"Need at least 2 units from first build, got {total_texts_first}"

        # --- Edit file_b only ---
        (project / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # --- Incremental reindex ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        total_texts_second = sum(
            len(c.args[0]) if c.args else 0
            for c in fake_model.encode.call_args_list
        )

        # Only file_b's units should have been re-embedded (file_a unchanged)
        assert total_texts_second < total_texts_first, (
            f"Expected incremental reindex to encode fewer texts than full build "
            f"(only changed file_b units). "
            f"First build encoded {total_texts_first} texts; second encoded {total_texts_second}. "
            f"Reason: IncrementalIndexer.plan() should detect file_a.foo's text_hash "
            f"is unchanged and skip it."
        )
        assert total_texts_second >= 1, (
            f"Expected at least 1 encode call for file_b's modified unit"
        )


# ===========================================================================
# TEST 3: CORRECTNESS — incremental == full
# ===========================================================================

class TestIncrementalEqualsFullCorrectness:
    """After a file edit, incremental and --full produce identical unit sets."""

    def test_incremental_result_matches_full_rebuild(self, tmp_path: Path):
        """Edit one file; run incremental; run --full on a fresh copy; assert unit sets identical.

        Expected RED failure: build_semantic_index does not accept `full=True` kwarg
        OR the incremental path produces a different unit ordering.
        """
        from tldr.semantic import build_semantic_index

        # Build the project in tmp_path
        project_incr = tmp_path / "incr"
        project_incr.mkdir()
        project_full = tmp_path / "full"
        project_full.mkdir()

        # Identical initial state in both
        for proj in (project_incr, project_full):
            (proj / ".git").mkdir()
            (proj / "file_a.py").write_text(_PY_FILE_A)
            (proj / "file_b.py").write_text(_PY_FILE_B)

        fake_model = _make_fake_model()

        # --- Initial build on incr project ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Edit file_b on both projects ---
        (project_incr / "file_b.py").write_text(_PY_FILE_B_MODIFIED)
        (project_full / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # --- Incremental reindex on project_incr ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_incr = build_semantic_index(
                str(project_incr), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Full rebuild on project_full using --full flag ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_full = build_semantic_index(
                str(project_full), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,  # This kwarg does not exist yet → AttributeError or TypeError → RED
            )

        assert count_incr == count_full, (
            f"Incremental and full rebuild must produce the same unit count: "
            f"{count_incr} != {count_full}"
        )

        meta_incr = _read_metadata(project_incr)
        meta_full = _read_metadata(project_full)

        incr_names = sorted(u["qualified_name"] for u in meta_incr["units"])
        full_names = sorted(u["qualified_name"] for u in meta_full["units"])

        assert incr_names == full_names, (
            f"Incremental and full rebuild must produce identical unit sets.\n"
            f"Incremental: {incr_names}\nFull: {full_names}"
        )


# ===========================================================================
# TEST 4: ROW-ALIGNMENT end-to-end
# ===========================================================================

class TestRowAlignment:
    """After incremental reindex, FAISS ntotal == len(metadata units) and row order matches."""

    def test_row_alignment_after_incremental_reindex(self, tmp_path: Path):
        """Reindex after file edit; assert ntotal == len(metadata units) and spot-check alignment.

        Expected RED failure: the incremental path doesn't reconstruct the FAISS
        index with proper row alignment, OR the feature doesn't exist at all.
        """
        import faiss as _faiss

        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Initial build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Edit file_b ---
        (project / "file_b.py").write_text(_PY_FILE_B_MODIFIED)

        # --- Incremental reindex ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Verify row alignment ---
        cache_dir = project / ".tldr" / "cache" / "semantic"
        index_file = cache_dir / "index.faiss"
        meta = _read_metadata(project)

        assert index_file.exists(), "index.faiss must exist after incremental reindex"

        idx = _faiss.read_index(str(index_file))
        unit_count = len(meta["units"])

        assert idx.ntotal == unit_count, (
            f"FAISS ntotal ({idx.ntotal}) must equal len(metadata['units']) ({unit_count}). "
            f"Row-alignment invariant violated: any incremental writer must assemble "
            f"the matrix in new_units order so row i <-> units[i]."
        )

        # Spot-check: reconstruct all rows, assert shape correct
        matrix = idx.reconstruct_n(0, idx.ntotal)
        assert matrix.shape == (unit_count, _DIM), (
            f"Reconstructed matrix shape {matrix.shape} != ({unit_count}, {_DIM}). "
            f"assemble() must produce shape (len(new_units), dim)."
        )

        # Verify text_hash present on all units
        for i, u in enumerate(meta["units"]):
            assert "text_hash" in u, (
                f"Unit {i} ({u.get('qualified_name', '?')}) missing 'text_hash' field. "
                f"EmbeddingUnit.to_dict() must include text_hash for L1 gate."
            )
            assert u["text_hash"], f"Unit {i} has empty text_hash"


# ===========================================================================
# TEST 5: 4(b) CROSS-FILE DRIFT — call graph changes re-embed called unit
# ===========================================================================

class TestCrossFileDrift:
    """File A unchanged; file B newly calls A's function; A's unit must be re-embedded."""

    def test_cross_file_call_change_re_embeds_callee(self, tmp_path: Path):
        """file_a.py unchanged (foo); file_b.py edited to call foo; assert foo re-embedded.

        Expected RED failure: no incremental path exists yet; the test cannot observe
        the per-unit re-embedding behavior through text_hash change in metadata.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)  # file_a: foo, file_b: bar (no call)
        fake_model = _make_fake_model()

        # --- Initial build: bar does NOT call foo ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_before = _read_metadata(project)
        foo_unit_before = next(
            (u for u in meta_before["units"] if u["name"] == "foo"), None
        )
        assert foo_unit_before is not None, "foo unit must be in metadata after initial build"
        foo_hash_before = foo_unit_before.get("text_hash", "")

        # --- file_b.py now calls foo() — file_a.py bytes UNCHANGED ---
        (project / "file_b.py").write_text(_PY_FILE_B_WITH_CALL)

        # --- Incremental reindex ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_after = _read_metadata(project)
        foo_unit_after = next(
            (u for u in meta_after["units"] if u["name"] == "foo"), None
        )
        assert foo_unit_after is not None, "foo unit must remain in metadata after reindex"
        foo_hash_after = foo_unit_after.get("text_hash", "")

        # foo's embedding text must have changed because called_by now includes bar
        assert foo_hash_after != foo_hash_before, (
            f"foo's text_hash must change after bar starts calling foo. "
            f"Before: {foo_hash_before!r}, after: {foo_hash_after!r}. "
            f"Reason: build_embedding_text(foo) includes 'Called by: bar' after "
            f"the call-graph recompute (4b correctness)."
        )

        # foo must have been re-embedded: its text appeared in at least one encode call
        all_encoded_texts = []
        for c in fake_model.encode.call_args_list:
            if c.args:
                all_encoded_texts.extend(c.args[0])
        assert any("foo" in t for t in all_encoded_texts), (
            f"foo's embedding text must have been passed to model.encode() "
            f"during the incremental reindex. Encoded texts: {all_encoded_texts!r}"
        )


# ===========================================================================
# TEST 6: DELETION — deleted file's symbols absent from metadata/search
# ===========================================================================

class TestDeletion:
    """Deleting a file removes its symbols from the index, with correct row-alignment."""

    def test_deleted_file_symbols_absent_after_reindex(self, tmp_path: Path):
        """Build index; delete file_b.py; incremental reindex; assert bar gone AND
        metadata units all have text_hash (proving the incremental path ran, not a plain full rebuild).

        Expected RED failure: The incremental path (IncrementalIndexer) does not exist.
        The current full-rebuild code does NOT write text_hash to metadata, so the
        assertion that all units have text_hash fails — proving the old code ran instead
        of the new incremental path.
        """
        import faiss as _faiss

        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo_three_files(tmp_path)  # foo, bar, baz
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_before = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert count_before >= 3, f"Expected >=3 units before deletion, got {count_before}"

        # --- Delete file_b.py ---
        (project / "file_b.py").unlink()

        # --- Reindex (should be incremental, not a full rebuild from scratch) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_after = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        meta_after = _read_metadata(project)
        names_after = [u["name"] for u in meta_after["units"]]

        assert "bar" not in names_after, (
            f"'bar' from deleted file_b.py must not appear in metadata after reindex. "
            f"Found names: {names_after}"
        )

        assert count_after < count_before, (
            f"Unit count must decrease after deleting file_b.py: "
            f"{count_after} !< {count_before}"
        )

        # ntotal must match metadata count (row-alignment invariant)
        cache_dir = project / ".tldr" / "cache" / "semantic"
        idx = _faiss.read_index(str(cache_dir / "index.faiss"))
        assert idx.ntotal == len(meta_after["units"]), (
            f"FAISS ntotal ({idx.ntotal}) != metadata unit count ({len(meta_after['units'])}) "
            f"after deletion. Row-alignment invariant violated."
        )

        # CRITICAL: all remaining units must have text_hash — proves incremental path ran
        # (current build_semantic_index does NOT populate text_hash in to_dict())
        for i, u in enumerate(meta_after["units"]):
            assert "text_hash" in u, (
                f"Unit {i} ({u.get('qualified_name', '?')}) missing 'text_hash' field "
                f"after reindex following deletion. "
                f"EmbeddingUnit.to_dict() must include text_hash (added as part of "
                f"the incremental indexer feature). Current to_dict() omits this field."
            )
            assert u["text_hash"], (
                f"Unit {i} ({u.get('qualified_name', '?')}) has empty text_hash"
            )


# ===========================================================================
# TEST 7: MIGRATION — old metadata without text_hash triggers full rebuild
# ===========================================================================

class TestMigration:
    """Old-format metadata (no text_hash) silently triggers a full rebuild once."""

    def test_old_metadata_without_text_hash_triggers_full_rebuild(self, tmp_path: Path):
        """Write old-format metadata.json lacking text_hash; reindex; assert full rebuild,
        no crash, and text_hash present on all units afterward.

        Expected RED failure: IncrementalIndexer.load_previous() does not exist;
        the migration gate logic is not implemented.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)

        # --- Write old-format metadata.json WITHOUT text_hash ---
        cache_dir = project / ".tldr" / "cache" / "semantic"
        cache_dir.mkdir(parents=True, exist_ok=True)

        old_metadata = {
            "units": [
                {
                    "name": "foo",
                    "qualified_name": "file_a.foo",
                    "file": "file_a.py",
                    "line": 1,
                    "language": "python",
                    "unit_type": "function",
                    "signature": "def foo(x):",
                    "docstring": "Compute foo of x.",
                    "calls": [],
                    "called_by": [],
                    "cfg_summary": "",
                    "dfg_summary": "",
                    "dependencies": "",
                    "code_preview": "",
                    # NOTE: no "text_hash" field — simulating pre-upgrade metadata
                }
            ],
            "model": "BAAI/bge-large-en-v1.5",
            "dimension": _DIM,
            "count": 1,
        }
        (cache_dir / "metadata.json").write_text(json.dumps(old_metadata))

        # Write a trivial fake FAISS index file so load_previous finds one
        import faiss as _faiss
        fake_idx = _faiss.IndexFlatIP(_DIM)
        fake_vec = np.ones((1, _DIM), dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec, axis=1, keepdims=True)
        fake_idx.add(fake_vec)
        _faiss.write_index(fake_idx, str(cache_dir / "index.faiss"))

        # --- Reindex: must NOT crash, must trigger full rebuild silently ---
        fake_model = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert count >= 2, (
            f"Migration full rebuild must re-index all units (>= 2), got {count}"
        )

        # All units in new metadata must have text_hash
        meta = _read_metadata(project)
        for i, u in enumerate(meta["units"]):
            assert "text_hash" in u, (
                f"Unit {i} ({u.get('qualified_name', '?')}) missing text_hash after migration rebuild."
            )
            assert u["text_hash"], f"Unit {i} has empty text_hash after migration"

        # --- Second reindex: must be a no-op now (text_hashes present) ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert fake_model.encode.call_count == 0, (
            f"After migration, second reindex must encode 0 units (all hashes match). "
            f"Got {fake_model.encode.call_count} encode calls."
        )


# ===========================================================================
# TEST 8: --full FLAG forces clean rebuild regardless of cache
# ===========================================================================

class TestFullFlag:
    """--full flag forces re-embedding of ALL units even when cache is up-to-date."""

    def test_full_flag_re_embeds_all_units(self, tmp_path: Path):
        """Build index; second call with full=True must encode all units (not 0).

        Expected RED failure: build_semantic_index does not accept `full` kwarg →
        TypeError: build_semantic_index() got an unexpected keyword argument 'full'.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- First build ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_first = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        texts_first = sum(
            len(c.args[0]) if c.args else 0
            for c in fake_model.encode.call_args_list
        )
        assert texts_first >= 2, f"First build must encode >=2 units, got {texts_first}"

        # --- Second build with --full flag: must re-embed ALL units ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_full = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,  # This kwarg does NOT exist yet → RED
            )

        texts_full = sum(
            len(c.args[0]) if c.args else 0
            for c in fake_model.encode.call_args_list
        )

        assert texts_full >= texts_first, (
            f"--full must re-embed at least as many units as the first build "
            f"({texts_first}), got {texts_full}. "
            f"IncrementalIndexer.load_previous(force_full=True) must short-circuit "
            f"and return full_rebuild=True so all units go to encode_units."
        )
        assert texts_full > 0, "--full must never produce 0 encode calls"


# ===========================================================================
# TEST 9: DEVICE DEFAULT — _resolve_default_device() returns metal on darwin
# ===========================================================================

class TestDeviceDefault:
    """_resolve_default_device() returns 'metal' on darwin, 'cpu' otherwise."""

    def test_resolve_default_device_returns_metal_on_darwin(self, monkeypatch):
        """_resolve_default_device() must return 'metal' when sys.platform == 'darwin'.

        Expected RED failure: _resolve_default_device() does not exist in tldr.semantic.
        """
        monkeypatch.setattr("sys.platform", "darwin")

        # This import will fail with AttributeError if function not yet implemented
        from tldr.semantic import _resolve_default_device  # type: ignore[attr-defined]

        result = _resolve_default_device()
        assert result == "metal", (
            f"_resolve_default_device() must return 'metal' on darwin, got {result!r}"
        )

    def test_resolve_default_device_returns_cpu_on_linux(self, monkeypatch):
        """_resolve_default_device() must return 'cpu' when sys.platform == 'linux'.

        Expected RED failure: _resolve_default_device() does not exist in tldr.semantic.
        """
        monkeypatch.setattr("sys.platform", "linux")

        from tldr.semantic import _resolve_default_device  # type: ignore[attr-defined]

        result = _resolve_default_device()
        assert result == "cpu", (
            f"_resolve_default_device() must return 'cpu' on linux, got {result!r}"
        )

    def test_build_semantic_index_uses_metal_device_on_darwin(self, tmp_path, monkeypatch):
        """build_semantic_index with no device arg + no TLDR_DEVICE must call get_model
        with device='metal' on darwin.

        Expected RED failure: build_semantic_index still defaults to 'cpu' (line 1582
        has `or 'cpu'` not `or _resolve_default_device()`).
        """
        import importlib

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.delenv("TLDR_DEVICE", raising=False)

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        captured_device = {}

        def spy_get_model(model_name=None, device=None):
            captured_device["device"] = device
            return fake_model

        from tldr.semantic import build_semantic_index

        with patch("tldr.semantic.get_model", side_effect=spy_get_model):
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert captured_device.get("device") == "metal", (
            f"build_semantic_index must call get_model with device='metal' on darwin "
            f"when no explicit device arg or TLDR_DEVICE env is set. "
            f"Got device={captured_device.get('device')!r}. "
            f"Fix: change semantic.py:1582 from `or 'cpu'` to `or _resolve_default_device()`."
        )

    def test_build_semantic_index_falls_back_to_cpu_when_no_gpu(self, tmp_path, monkeypatch):
        """On non-darwin platform, build_semantic_index uses _resolve_default_device()
        which returns 'cpu', and this must flow through tldr.semantic._resolve_default_device
        (not be hardcoded at the call site).

        Expected RED failure: _resolve_default_device() does not exist in tldr.semantic;
        importing it raises ImportError/AttributeError.
        """
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.delenv("TLDR_DEVICE", raising=False)

        # This must import from the module — fails if function does not exist yet
        from tldr.semantic import _resolve_default_device  # type: ignore[attr-defined]

        # _resolve_default_device() must return 'cpu' on linux
        result = _resolve_default_device()
        assert result == "cpu", (
            f"_resolve_default_device() must return 'cpu' on non-darwin. "
            f"Got {result!r}."
        )

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        captured_device = {}

        def spy_get_model(model_name=None, device=None):
            captured_device["device"] = device
            return fake_model

        from tldr.semantic import build_semantic_index

        with patch("tldr.semantic.get_model", side_effect=spy_get_model):
            # Must not crash even on non-darwin
            build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        assert captured_device.get("device") == "cpu", (
            f"build_semantic_index must fall back to 'cpu' on non-darwin via "
            f"_resolve_default_device(). Got device={captured_device.get('device')!r}."
        )


# ===========================================================================
# TEST 10: SEARCH DEVICE SYMMETRY (I-11)
# ===========================================================================

class TestSearchDeviceSymmetry:
    """semantic_search with device=None must call _resolve_default_device() — same as index."""

    def test_semantic_search_resolves_device_via_default_resolver(self, monkeypatch):
        """semantic_search(device=None) must resolve to 'metal' on darwin via
        _resolve_default_device(), NOT hardcode 'cpu'.

        Expected RED failure: semantic_search does not call _resolve_default_device()
        when device=None; it passes device=None straight to compute_embedding/get_model.
        """
        import faiss as _faiss

        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.delenv("TLDR_DEVICE", raising=False)

        # Build a minimal real index so semantic_search can load it
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            (project_root / ".git").mkdir()
            cache_dir = project_root / ".tldr" / "cache" / "semantic"
            cache_dir.mkdir(parents=True)

            # Write a tiny real index
            idx = _faiss.IndexFlatIP(_DIM)
            vec = np.ones((1, _DIM), dtype=np.float32)
            vec /= np.linalg.norm(vec)
            idx.add(vec)
            _faiss.write_index(idx, str(cache_dir / "index.faiss"))

            meta = {
                "units": [{
                    "name": "foo",
                    "qualified_name": "file_a.foo",
                    "file": "file_a.py",
                    "line": 1,
                    "language": "python",
                    "unit_type": "function",
                    "signature": "def foo(x):",
                    "docstring": "Compute foo.",
                    "calls": [],
                    "called_by": [],
                    "cfg_summary": "",
                    "dfg_summary": "",
                    "dependencies": "",
                    "code_preview": "",
                    "text_hash": "abc123",
                }],
                "model": "BAAI/bge-large-en-v1.5",
                "dimension": _DIM,
                "count": 1,
            }
            (cache_dir / "metadata.json").write_text(json.dumps(meta))

            # Spy on compute_embedding to capture the device used
            captured_device = {}

            def spy_compute_embedding(text, model_name=None, *, device=None, backend=None):
                captured_device["device"] = device
                return np.ones(_DIM, dtype=np.float32)

            from tldr.semantic import semantic_search

            with patch("tldr.semantic.compute_embedding", side_effect=spy_compute_embedding):
                try:
                    semantic_search(str(project_root), "foo query", k=1, device=None)
                except Exception:
                    pass  # We only care about what device was resolved

            # The device passed to compute_embedding must be 'metal', not None or 'cpu'
            assert captured_device.get("device") == "metal", (
                f"semantic_search with device=None must resolve to 'metal' on darwin "
                f"via _resolve_default_device(). "
                f"Got device={captured_device.get('device')!r}. "
                f"Fix: add `device = device or _resolve_default_device()` at the top "
                f"of semantic_search() (I-11 symmetric device)."
            )


# ===========================================================================
# TEST 11: DAEMON --dirty-files plumbing
# ===========================================================================

class TestDaemonDirtyFilesPlumbing:
    """The semantic index CLI accepts --dirty-files; daemon correctness does not depend on it."""

    def test_cli_accepts_dirty_files_arg(self, tmp_path: Path):
        """The `tldr semantic index` CLI must accept --dirty-files <path> without error.

        Expected RED failure: cli.py does not register --dirty-files arg →
        argparse error: unrecognized arguments: --dirty-files ...
        """
        import subprocess

        project = _build_tiny_repo(tmp_path)

        # Write a temp file containing a JSON list of dirty paths
        dirty_file = tmp_path / "dirty.json"
        dirty_file.write_text(json.dumps(["file_b.py"]))

        result = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "semantic", "index", str(project),
                "--dirty-files", str(dirty_file),
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=120,
        )

        # Must not fail with argparse error about unrecognized arguments
        assert "unrecognized arguments" not in result.stderr, (
            f"CLI does not recognize --dirty-files arg. stderr: {result.stderr!r}. "
            f"Fix: add index_p.add_argument('--dirty-files', ...) in cli.py BEFORE "
            f"the daemon subprocess wiring (I-12)."
        )
        assert "error: argument --dirty-files" not in result.stderr, (
            f"argparse rejected --dirty-files. stderr: {result.stderr!r}"
        )

    def test_dirty_files_result_matches_clean_index(self, tmp_path: Path):
        """Index built with --dirty-files matches one built without: correctness never depends on it.

        The dirty-files list is only an optimization hint; the L1 text_hash gate
        and full call graph ensure correctness regardless.

        Expected RED failure: --dirty-files arg not yet accepted by CLI.
        """
        import subprocess

        # Project with --dirty-files
        project_with = tmp_path / "with_dirty"
        project_with.mkdir()
        (project_with / ".git").mkdir()
        (project_with / "file_a.py").write_text(_PY_FILE_A)
        (project_with / "file_b.py").write_text(_PY_FILE_B)

        # Project without --dirty-files
        project_without = tmp_path / "without_dirty"
        project_without.mkdir()
        (project_without / ".git").mkdir()
        (project_without / "file_a.py").write_text(_PY_FILE_A)
        (project_without / "file_b.py").write_text(_PY_FILE_B)

        dirty_file = tmp_path / "dirty.json"
        dirty_file.write_text(json.dumps(["file_b.py"]))

        # Index with --dirty-files
        res_with = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "semantic", "index", str(project_with),
                "--dirty-files", str(dirty_file),
            ],
            capture_output=True, text=True, cwd=_REPO_ROOT, timeout=120,
        )

        # Index without --dirty-files
        res_without = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "semantic", "index", str(project_without),
            ],
            capture_output=True, text=True, cwd=_REPO_ROOT, timeout=120,
        )

        assert res_with.returncode == 0, (
            f"semantic index with --dirty-files failed: {res_with.stderr!r}"
        )
        assert res_without.returncode == 0, (
            f"semantic index without --dirty-files failed: {res_without.stderr!r}"
        )

        # Both should index the same number of units (correctness invariant)
        def _parse_count(stdout: str) -> int:
            for line in stdout.splitlines():
                if "Indexed" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "Indexed" and i + 1 < len(parts):
                            try:
                                return int(parts[i + 1])
                            except ValueError:
                                pass
            return -1

        count_with = _parse_count(res_with.stdout)
        count_without = _parse_count(res_without.stdout)

        assert count_with == count_without, (
            f"Indexing with --dirty-files ({count_with}) must produce the same unit count "
            f"as without --dirty-files ({count_without}). "
            f"Correctness must not depend on dirty-files hint."
        )

    def test_stale_dirty_files_does_not_corrupt_index(self, tmp_path: Path):
        """A --dirty-files pointing to a non-existent file must not corrupt the index.

        If the temp file is missing, build_semantic_index falls back gracefully
        and produces a correct (full L1-gate) index.

        Expected RED failure: --dirty-files arg not accepted or build_semantic_index
        crashes when the path is absent.
        """
        import subprocess

        project = _build_tiny_repo(tmp_path)

        missing_dirty_file = tmp_path / "does_not_exist.json"
        # Do NOT create this file

        result = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "semantic", "index", str(project),
                "--dirty-files", str(missing_dirty_file),
            ],
            capture_output=True, text=True, cwd=_REPO_ROOT, timeout=120,
        )

        # Must not crash (exit 0) — missing dirty-files is a no-op hint failure
        assert result.returncode == 0, (
            f"Missing --dirty-files path must not crash build_semantic_index. "
            f"returncode={result.returncode}, stderr={result.stderr!r}. "
            f"Fix: in build_semantic_index, if dirty_files path is absent, log warning "
            f"and continue with full L2 file-hash scan."
        )


# ===========================================================================
# TEST 12: SUMMARY OUTPUT — embedded/reused counts and active device
# ===========================================================================

class TestIndexSummaryOutput:
    """build_semantic_index emits a one-line summary after every run.

    The summary must report:
    - Number of units EMBEDDED (re-computed from scratch)
    - Number of units REUSED (vector carried forward via text_hash gate)
    - Active device name (e.g. 'metal' or 'cpu')

    These tests are RED because build_semantic_index currently:
    - Returns only an int (no summary dict/object)
    - Prints nothing when show_progress=False
    - Does not track or expose embedded vs reused counts anywhere
    """

    def test_index_summary_reports_embedded_and_reused_counts(
        self, tmp_path: Path, capsys
    ):
        """Initial build then no-op reindex; the no-op run's summary must report
        embedded=0 and a non-zero reused count.

        Asserts on the RETURN VALUE of build_semantic_index: the feature must
        change the return type from int to a result object/dict that carries
        ``embedded_count`` and ``reused_count``.  Alternatively the summary is
        printed to stdout (also captured and checked).

        Expected RED failure:
        - build_semantic_index currently returns int, not a dict/object with
          embedded_count / reused_count fields.
        - No stdout output is produced when show_progress=False.
        So both branches of the assertion fail → RED.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- First build: all units are embedded (no prior index) ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            result_first = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # result_first may be int (current) or a richer object (after feature lands)
        # We'll check the summary via the RETURN VALUE approach (preferred) and
        # also capture stdout in case the implementer chose print-to-stdout.

        # --- No-op second build: nothing changed, all units must be REUSED ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            result_noop = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # The summary must report embedded=0 for the no-op run.
        # Check return-value dict first (preferred seam):
        if isinstance(result_noop, dict):
            assert result_noop.get("embedded_count", -1) == 0, (
                f"No-op reindex summary must report embedded_count=0. "
                f"Got result_noop={result_noop!r}. "
                f"All units had unchanged text_hash → IncrementalIndexer.plan() "
                f"should return encode_units=[], so embedded_count=0."
            )
            assert result_noop.get("reused_count", 0) > 0, (
                f"No-op reindex summary must report reused_count > 0. "
                f"Got result_noop={result_noop!r}. "
                f"The unchanged units were carried forward via vector reuse."
            )
        else:
            # Fall back: assert the summary was printed to stdout/stderr
            # This branch also fails RED because no summary is printed with show_progress=False
            assert "embedded 0" in combined_output.lower() or "embedded=0" in combined_output.lower(), (
                f"No-op reindex summary must contain 'embedded 0' (or 'embedded=0') in output. "
                f"stdout+stderr: {combined_output!r}. "
                f"build_semantic_index must print a one-line summary even when "
                f"show_progress=False, reporting how many units were embedded vs reused."
            )
            assert any(
                phrase in combined_output.lower()
                for phrase in ("reused", "reuse")
            ), (
                f"No-op reindex summary must contain 'reused' count in output. "
                f"stdout+stderr: {combined_output!r}."
            )

    def test_index_summary_reports_active_device(self, tmp_path: Path, capsys):
        """The summary includes the active device name ('metal' on darwin, 'cpu' under --device cpu).

        Expected RED failure:
        - build_semantic_index returns int (no device field in result).
        - No stdout output is produced when show_progress=False.
        Both branches fail → RED.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # Force CPU device so the test is platform-independent
        with patch("tldr.semantic.get_model", return_value=fake_model):
            result = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                device="cpu",
            )

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # Check return-value dict first (preferred seam):
        if isinstance(result, dict):
            device_in_result = result.get("device", "")
            assert device_in_result == "cpu", (
                f"Summary result dict must include device='cpu' when --device cpu was passed. "
                f"Got result={result!r}."
            )
        else:
            # Fall back: assert the device name appears in stdout/stderr summary line
            assert "cpu" in combined_output.lower(), (
                f"Index summary must include the active device name ('cpu') in output. "
                f"stdout+stderr: {combined_output!r}. "
                f"build_semantic_index must emit a summary line naming the device used, "
                f"even when show_progress=False."
            )

    def test_full_rebuild_summary_reports_all_embedded_reused_zero(
        self, tmp_path: Path, capsys
    ):
        """--full rebuild summary must report embedded==total units, reused==0.

        Expected RED failure:
        - build_semantic_index returns int (no embedded_count / reused_count in result).
        - No stdout output produced when show_progress=False.
        Both branches fail → RED.
        """
        from tldr.semantic import build_semantic_index

        project = _build_tiny_repo(tmp_path)
        fake_model = _make_fake_model()

        # --- Initial build so there IS a prior index to ignore ---
        with patch("tldr.semantic.get_model", return_value=fake_model):
            count_first = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
            )

        # --- Full rebuild: ALL units must be embedded, NONE reused ---
        fake_model.encode.reset_mock()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            result_full = build_semantic_index(
                str(project), lang="python",
                show_progress=False, respect_ignore=False,
                full=True,
            )

        captured = capsys.readouterr()
        combined_output = captured.out + captured.err

        # Resolve total unit count
        total_units = (
            result_full.get("total_count", count_first)
            if isinstance(result_full, dict)
            else count_first
        )

        # Check return-value dict (preferred seam):
        if isinstance(result_full, dict):
            assert result_full.get("reused_count", -1) == 0, (
                f"--full rebuild must report reused_count=0 (all vectors recomputed). "
                f"Got result_full={result_full!r}."
            )
            embedded = result_full.get("embedded_count", -1)
            assert embedded == total_units, (
                f"--full rebuild must report embedded_count==total_units ({total_units}). "
                f"Got embedded_count={embedded!r}."
            )
        else:
            # Fall back: assert 'reused 0' appears in summary output
            assert "reused 0" in combined_output.lower() or "reused=0" in combined_output.lower(), (
                f"--full rebuild summary must contain 'reused 0' (or 'reused=0'). "
                f"stdout+stderr: {combined_output!r}. "
                f"With full=True, IncrementalIndexer.load_previous(force_full=True) "
                f"short-circuits and all units land in encode_units → reused=0."
            )
