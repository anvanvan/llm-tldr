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
