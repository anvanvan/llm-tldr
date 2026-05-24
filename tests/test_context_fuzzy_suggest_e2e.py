"""Crystallized E2E tests for tldr context fuzzy-suggest + qualified-name fallback.

Converted from 7 live-verified demo scenarios (verification-audit.json: all PASS at 10.1).
Each test invokes the real tldr CLI via subprocess and asserts on exit code + output substrings.
"""

import subprocess
import sys
from pathlib import Path

# Repo root for subprocess invocation (inherits test interpreter)
_REPO_ROOT = Path(__file__).parent.parent


def _run(tmp_path: Path, entry_point: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tldr.cli", "context", entry_point,
         "--project", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


def test_e2e_primary_typo_hint(tmp_path: Path):
    """PRIMARY: one-character typo 'filter_invocie' → 'Did you mean: filter_invoice', exit 1."""
    (tmp_path / "invoice.py").write_text(
        "def filter_invoice(records):\n    return [r for r in records if r.get('type') == 'invoice']\n"
    )
    result = _run(tmp_path, "filter_invocie")
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "Did you mean: filter_invoice" in combined, (
        f"Expected 'Did you mean: filter_invoice' in output, got: {combined!r}"
    )


def test_e2e_edge1_multi_suggestion(tmp_path: Path):
    """EDGE-1: '_Init' close to three names → all three listed after 'Did you mean:', exit 1."""
    (tmp_path / "mod.py").write_text(
        "def init():\n    pass\ndef Init():\n    pass\ndef _init():\n    pass\n"
    )
    result = _run(tmp_path, "_Init")
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "Did you mean:" in combined, f"Expected 'Did you mean:' in output, got: {combined!r}"
    assert "init" in combined and "Init" in combined and "_init" in combined, (
        f"Expected all three candidates in output, got: {combined!r}"
    )


def test_e2e_edge2_unrelated_no_hint(tmp_path: Path):
    """EDGE-2: completely unrelated name 'xyzqwerty' → 'not found', no 'Did you mean', exit 1."""
    (tmp_path / "mod.py").write_text("def filter_invoice(r):\n    pass\n")
    result = _run(tmp_path, "xyzqwerty")
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "not found" in combined, f"Expected 'not found' in output, got: {combined!r}"
    assert "Did you mean" not in combined, (
        f"Expected NO 'Did you mean' for unrelated name, got: {combined!r}"
    )


def test_e2e_edge3_qualified_colon(tmp_path: Path):
    """EDGE-3: 'Calendar::nextDay' resolves bare 'nextDay' with Note, exit 0."""
    (tmp_path / "calendar.py").write_text("def nextDay(d):\n    return d + 1\n")
    result = _run(tmp_path, "Calendar::nextDay")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Expected exit 0 for qualified-colon fallback, got: {combined!r}"
    )
    assert "Note:" in combined, f"Expected 'Note:' in output, got: {combined!r}"
    assert "nextDay" in combined, f"Expected 'nextDay' referenced in output, got: {combined!r}"


def test_e2e_edge4_path_style_chain(tmp_path: Path):
    """EDGE-4: 'providers/anthropic.stream' two-hop chain → Note with 'resolved via', exit 0."""
    (tmp_path / "providers.py").write_text("def stream(client):\n    pass\n")
    result = _run(tmp_path, "providers/anthropic.stream")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Expected exit 0 for path-style chain fallback, got: {combined!r}"
    )
    assert "Note:" in combined, f"Expected 'Note:' in output, got: {combined!r}"
    assert "stream" in combined, f"Expected 'stream' in output, got: {combined!r}"


def test_e2e_edge5_ambiguous_enum(tmp_path: Path):
    """EDGE-5: 'AppState.filterPills' matches two files → Note enumerates both, exit 0."""
    (tmp_path / "a.py").write_text("def filterPills():\n    pass\n")
    (tmp_path / "b.py").write_text("def filterPills():\n    pass\n")
    result = _run(tmp_path, "AppState.filterPills")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Expected exit 0 for ambiguous bare-name fallback, got: {combined!r}"
    )
    assert "Note:" in combined, f"Expected 'Note:' in output, got: {combined!r}"
    assert "filterPills" in combined, f"Expected 'filterPills' in output, got: {combined!r}"
    assert ("2" in combined or "candidate" in combined.lower()), (
        f"Expected '2' or 'candidate' in output for ambiguous match, got: {combined!r}"
    )


def test_e2e_edge6_dot_single_match(tmp_path: Path):
    """EDGE-6: 'AppState.filterPills' with single definition → Note resolves uniquely, exit 0."""
    (tmp_path / "pills.py").write_text(
        "def filterPills(state):\n    return [p for p in state if p.active]\n"
    )
    result = _run(tmp_path, "AppState.filterPills")
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Expected exit 0 for single-match dot-qualified fallback, got: {combined!r}"
    )
    assert "Note:" in combined, f"Expected 'Note:' in output, got: {combined!r}"
    assert "filterPills" in combined, f"Expected 'filterPills' in output, got: {combined!r}"
    # Single match: note must NOT enumerate multiple candidates
    assert "2 candidates" not in combined and "matched 2" not in combined, (
        f"Expected no multi-candidate enumeration for single match, got: {combined!r}"
    )
