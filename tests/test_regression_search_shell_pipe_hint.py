"""
Regression test: tldr search must emit a stderr hint when the pattern contains
the shell-escaped pipe (backslash-pipe) and returns zero results.

Bug (verified in /tmp/claude-bug-fix-search-shell-pipe/verification.md):
    tldr search 'build_project_call_graph\\|build_function_index' tldr/ --ext .py
    -> stdout: []   stderr: (0 bytes)   exit: 0

Root cause: cli.py:697 has no diagnostic branch between the api_search() call
and the json.dumps() print.  No inspection of
    not result and r"\\|" in args.pattern.

Fix shape: insert a branch at cli.py:697 that writes to sys.stderr when
    not result and the pattern contains the two-char sequence backslash-pipe.

Pre-fix this test FAILS because stderr is empty (0 bytes).
Post-fix it passes because stderr contains an ERE / bare-pipe hint.

Control invariant: bare pipe (ERE alternation) with real hits must NOT trigger
the hint, and must still return a non-empty JSON array.
"""

import json
import subprocess
import sys
from pathlib import Path

# Root of the repository — always an ancestor of this test file.
_REPO_ROOT = str(Path(__file__).parent.parent)
# A subdirectory with known Python files so the search path is always valid.
_SEARCH_PATH = str(Path(__file__).parent.parent / "tldr")


class TestSearchShellPipeHint:
    """tldr search backslash-pipe zero-hit path must emit an actionable stderr hint."""

    def test_backslash_pipe_zero_hits_emits_stderr_hint(self):
        """When backslash-pipe in pattern yields zero hits, stderr must contain a hint.

        Three behaviors verified:
        1. Exit code is still 0 (non-error — user gets usable output).
        2. stdout is exactly '[]' (JSON empty array, whitespace-normalised).
        3. stderr is non-empty and contains a recognisable ERE/pipe hint substring.

        Pre-fix: assertion 3 fails because stderr is 0 bytes.
        """
        # Pattern identical to the reproduction command.  Single-quoted in shell
        # preserves the backslash; here in Python the raw string achieves the
        # same: args.pattern == r'build_project_call_graph\|build_function_index'
        pattern = r"build_project_call_graph\|build_function_index"

        result = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "search", pattern, _SEARCH_PATH,
                "--ext", ".py",
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )

        # Assertion 1: exit code 0 (hint is advisory, not fatal)
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}. stderr: {result.stderr!r}"
        )

        # Assertion 2: stdout is the JSON empty array
        stdout_stripped = result.stdout.strip()
        assert stdout_stripped == "[]", (
            f"Expected stdout '[]', got {stdout_stripped!r}"
        )

        # Assertion 3: stderr is non-empty and contains an ERE/pipe hint.
        # The fix must write something like:
        #   "Hint: pattern contains '\\|' which ERE treats as a literal "
        #   "backslash-pipe, not alternation. Use a bare '|' for "
        #   "alternation (e.g. 'foo|bar')."
        # or any substring containing the signal words below.
        stderr = result.stderr
        assert stderr.strip(), (
            "Expected a non-empty stderr hint about \\| / ERE but got 0 bytes. "
            "The fix (cli.py:697 diagnostic branch) is missing."
        )
        hint_lower = stderr.lower()
        has_ere_mention = "ere" in hint_lower or "alternation" in hint_lower
        has_pipe_mention = r"\|" in stderr or "bare '|'" in hint_lower or "bare|" in hint_lower
        assert has_ere_mention or has_pipe_mention, (
            f"stderr is non-empty but does not mention ERE or pipe alternation. "
            f"Got: {stderr!r}"
        )

    def test_bare_pipe_with_hits_does_not_emit_hint(self):
        """Control: bare `|` (ERE alternation) returns real hits and no hint.

        This confirms the fix branch is correctly gated on r'\\|' in pattern
        so it does NOT regress the normal (working) bare-pipe case.
        """
        # Bare pipe — ERE alternation — matches real functions in api.py / cli.py
        pattern = "build_project_call_graph|build_function_index"

        result = subprocess.run(
            [
                sys.executable, "-m", "tldr.cli",
                "search", pattern, _SEARCH_PATH,
                "--ext", ".py",
            ],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )

        assert result.returncode == 0, (
            f"Control search failed with exit {result.returncode}. "
            f"stderr: {result.stderr!r}"
        )

        hits = json.loads(result.stdout)
        assert isinstance(hits, list) and len(hits) > 0, (
            f"Control search expected non-empty hits but got: {result.stdout!r}. "
            f"This likely means the control fixture is stale — one or both of "
            f"'build_project_call_graph' / 'build_function_index' has been "
            f"renamed/removed. Please update the pattern on the line above to "
            f"reference two functions that currently exist in tldr/."
        )

        # stderr must remain empty for the control path
        assert not result.stderr.strip(), (
            f"Control path (bare |) must NOT emit a hint to stderr, "
            f"but got: {result.stderr!r}"
        )
