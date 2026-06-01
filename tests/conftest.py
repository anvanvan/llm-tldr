"""Shared pytest fixtures for the tldr test suite."""
from pathlib import Path

import pytest


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
