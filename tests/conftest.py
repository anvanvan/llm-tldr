"""Shared pytest fixtures for the tldr test suite."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared test helpers — used by multiple test files
# ---------------------------------------------------------------------------

_FAKE_MODEL_DIM = 4


def make_fake_model() -> MagicMock:
    """Deterministic fake embedder: L2-normalised np.ones vectors.

    Returns a MagicMock that encodes any text sequence as normalized
    all-ones vectors with dimension _FAKE_MODEL_DIM. Used by tests
    that need a model but do not depend on embedding semantics.
    """
    mock_model = MagicMock()

    def fake_encode(texts, batch_size=128, normalize_embeddings=True,
                    show_progress_bar=False):
        n = len(texts) if isinstance(texts, list) else 1
        vecs = np.ones((n, _FAKE_MODEL_DIM), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_model.encode.side_effect = fake_encode
    return mock_model


def read_semantic_metadata(project_root: Path) -> dict:
    """Read semantic index metadata.json from a project's cache.

    Args:
        project_root: Root directory of the project (must have .tldr/cache/ already indexed).

    Returns:
        Parsed metadata dict with 'units' and other index metadata.

    Raises:
        AssertionError if metadata.json is missing.
    """
    meta_path = project_root / ".tldr" / "cache" / "semantic" / "metadata.json"
    assert meta_path.exists(), f"metadata.json missing: {meta_path}"
    return json.loads(meta_path.read_text())


def find_units_by_name(meta: dict, name: str) -> list:
    """Return all units whose bare 'name' field matches."""
    return [u for u in meta["units"] if u.get("name") == name]


def validate_incr_full_equivalence(meta_incr: dict, meta_full: dict) -> list:
    """Return per-unit divergence messages (empty list == equal).

    Keys by (file, name) pairs to handle multi-file repos where unit names may
    collide across files. For shared units, sorted(calls) and sorted(called_by)
    must match exactly.
    """
    def _key(u: dict) -> tuple:
        return (u.get("file", ""), u.get("name", ""))

    incr_by_key = {_key(u): u for u in meta_incr["units"]}
    full_by_key = {_key(u): u for u in meta_full["units"]}
    problems = []

    only_incr = set(incr_by_key) - set(full_by_key)
    only_full = set(full_by_key) - set(incr_by_key)
    if only_incr:
        problems.append(f"Units only in incremental: {sorted(str(k) for k in only_incr)}")
    if only_full:
        problems.append(f"Units only in --full:      {sorted(str(k) for k in only_full)}")

    for k in set(incr_by_key) & set(full_by_key):
        u_i = incr_by_key[k]
        u_f = full_by_key[k]
        i_calls = sorted(u_i.get("calls") or [])
        f_calls = sorted(u_f.get("calls") or [])
        i_cb = sorted(u_i.get("called_by") or [])
        f_cb = sorted(u_f.get("called_by") or [])
        if i_calls != f_calls:
            problems.append(
                f"{k}.calls: incremental={i_calls!r} != full={f_calls!r}"
            )
        if i_cb != f_cb:
            problems.append(
                f"{k}.called_by: incremental={i_cb!r} != full={f_cb!r}"
            )

    return problems


@pytest.fixture
def write_temp_file():
    """Factory fixture to write a file to a temporary path.

    Returns a callable with signature ``(path: Path, content: str) -> Path``
    that creates parent directories as needed, writes ``content``, and returns
    the path so callers can use the result directly.
    """
    def _write(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path
    return _write


# ---------------------------------------------------------------------------
# E2E marker: opt-in only (--run-e2e) to avoid slow real-model tests in CI
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run E2E tests that load the real embedding model (slow, fan-loud).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests using the real embedding model (opt-in via --run-e2e).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e", default=False):
        return  # opted in — run all e2e tests normally
    skip_e2e = pytest.mark.skip(reason="opt-in: pass --run-e2e to run E2E tests")
    for item in items:
        if item.get_closest_marker("e2e"):
            item.add_marker(skip_e2e)
