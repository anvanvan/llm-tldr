"""
Regression test: tldr extract <file> (no filter) must emit a stderr warning
when symbol count > 5, naming all three filter flags.

Bug 005 (verified in /tmp/claude-bug-fix-bare-extract-warn/verification.md):
    tldr extract tldr/cross_file_calls.py
    -> stdout: 119,491-byte JSON   stderr: (0 bytes)   exit: 0

Root cause: cli.py:699-745 extract handler has zero sys.stderr writes and no
conditional branch on "filter absent + symbol count > threshold".  The expected
warning block was never authored.  Confirmed: 0 hits for stderr/warn inside
the handler range (awk + grep verified in verification.md experiment).

Fix shape: insert a peer `if not (filter_class or filter_function or filter_method):`
block at cli.py:744 (immediately before the final print on line 745) that:
  1. Counts n_symbols via dict-key access on result.get("functions",[]) and
     result.get("classes",[]) with c.get("methods",[]).
  2. If n_symbols > 5, writes a multi-line warning to sys.stderr naming all
     three filter flags: --function, --method, --class.

Pre-fix: test_bare_extract_no_filter_warns_stderr FAILS because stderr is empty.
Post-fix: it passes.

Control invariant: filtered invocations must remain stderr-silent.
"""

import subprocess
import sys
from pathlib import Path

# Repository root — parent of this test file's directory.
_REPO_ROOT = str(Path(__file__).parent.parent)
# Known large file: 213 symbols (verified in reproduction.md), well above threshold 5.
_TARGET_FILE = "tldr/cross_file_calls.py"


class TestBareExtractWarningBug005:
    """tldr extract without a filter flag must warn on stderr when symbol count > 5."""

    def test_bare_extract_no_filter_warns_stderr(self):
        """Bare extract on a large file (>5 symbols) must write a non-empty stderr
        warning that mentions all three filter flags: --function, --method, --class.

        Three behaviors verified:
        1. Exit code is still 0 (warning is advisory, not fatal).
        2. stderr is non-empty.
        3. stderr contains all three flag strings: --function, --method, --class
           (case-sensitive, exact flag names as specified by the feature contract).

        Pre-fix: assertion 2 fails because stderr is 0 bytes
        (no warning code exists in cli.py:699-745, confirmed in verification.md).
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "extract", _TARGET_FILE,
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )

        # Assertion 1: exit code 0 (warning must be advisory, not abort the command)
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}. stderr: {result.stderr!r}"
        )

        # Assertion 2: stderr must be non-empty
        stderr = result.stderr
        assert stderr.strip(), (
            "Expected a non-empty stderr warning about bare extract with > 5 symbols "
            "but got 0 bytes. The fix (cli.py:744 peer block) is missing."
        )

        # Assertion 3: warning must name all three filter flags
        assert "--function" in stderr, (
            f"stderr warning must contain '--function' but got: {stderr!r}"
        )
        assert "--method" in stderr, (
            f"stderr warning must contain '--method' but got: {stderr!r}"
        )
        assert "--class" in stderr, (
            f"stderr warning must contain '--class' but got: {stderr!r}"
        )

    def test_filtered_extract_does_not_warn_stderr(self):
        """Control: a filtered extract (--function) must produce zero stderr output.

        This confirms the fix branch is correctly gated on the absence of all three
        filter flags, and does NOT regress properly-filtered calls.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "extract", _TARGET_FILE,
                "--function", "detect_cross_file_calls",
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )

        assert result.returncode == 0, (
            f"Filtered extract failed with exit {result.returncode}. "
            f"stderr: {result.stderr!r}"
        )

        # Filtered call must remain stderr-silent (no spurious warning)
        assert not result.stderr.strip(), (
            f"Filtered extract must NOT emit a stderr warning, "
            f"but got: {result.stderr!r}"
        )
