# pyright: reportMissingImports=false
"""Regression test — Bug 007: `tldr context --lang swift` returns a 1-node
thin result with line_number:0 when `tree_sitter_swift` cannot be imported.

Root cause (two-site latent fragility):

PRIMARY — cross_file_calls.py `_build_swift_call_graph`:
    if not TREE_SITTER_SWIFT_AVAILABLE:
        return
  No logging, no exception, no marker. Graph stays empty.

SECONDARY — hybrid_extractor.py `_extract_pygments`:
    FunctionInfo(name=..., params=..., return_type=None, ...)
  `line_number=` is omitted; ast_extractor.py:36 defaults it to 0.

Combined effect: 1 node returned with line:0, no callers, no callees,
no error field set, no stderr warning — silent useless output.

This RED test pins the FIXED contract (Option C end-to-end):

When TREE_SITTER_SWIFT_AVAILABLE is False, get_relevant_context for a Swift
project MUST NOT silently produce a 1-node line:0 result.  It must satisfy at
least ONE of:
  (a) Set ctx.error to a non-None, non-empty string (failure is reported), OR
  (b) Return functions whose line values are all > 0 (Pygments fallback sets
      line_number correctly even without tree-sitter).

Current (unfixed) code satisfies NEITHER: ctx.error is None AND every
FunctionContext.line == 0.  Therefore the assertion FAILS → test is RED.
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Minimal Swift fixture — two files so callers can exist
# ---------------------------------------------------------------------------

_STORE_SWIFT = textwrap.dedent("""\
    import Foundation

    class VocabStore {
        func recordUsedTerms(_ terms: [String]) {
            print(terms)
        }

        func loadIfNeeded() {
            print("loading")
        }
    }
""")

_APP_SWIFT = textwrap.dedent("""\
    import Foundation

    class App {
        let store = VocabStore()

        func run() {
            store.recordUsedTerms(["hello", "world"])
        }
    }
""")


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------

class TestSwiftContextThinResultBug007:
    """get_relevant_context must not silently produce a 1-node line:0 result
    when TREE_SITTER_SWIFT_AVAILABLE is False."""

    def test_swift_context_no_silent_degradation_when_tree_sitter_unavailable(
        self, tmp_path: Path
    ):
        """Option C end-to-end: monkey-patch both AVAILABLE flags to False and
        assert the result is NOT silently degraded (either error is reported OR
        line_number > 0 for all returned functions).

        RED contract (current code):
          ctx.error is None AND all FunctionContext.line == 0
          → neither condition of the fixed contract is met → assertion FAILS.

        GREEN contract (after fix):
          At least one of:
            (a) ctx.error is not None and len(ctx.error) > 0, OR
            (b) all(f.line > 0 for f in ctx.functions) and len(ctx.functions) >= 1
        """
        # Build a minimal Swift project in tmp_path
        src_dir = tmp_path / "Sources" / "App"
        src_dir.mkdir(parents=True)
        (src_dir / "VocabStore.swift").write_text(_STORE_SWIFT)
        (src_dir / "App.swift").write_text(_APP_SWIFT)

        # Import here so patching is applied inside the call
        from tldr.api import get_relevant_context

        with (
            patch("tldr.cross_file_calls.TREE_SITTER_SWIFT_AVAILABLE", False),
            patch("tldr.hybrid_extractor.TREE_SITTER_SWIFT_AVAILABLE", False),
        ):
            ctx = get_relevant_context(
                project=str(tmp_path),
                entry_point="recordUsedTerms",
                depth=1,
                language="swift",
            )

        # --- Condition (a): error is explicitly reported ---
        error_reported = (
            ctx.error is not None and len(ctx.error.strip()) > 0
        )

        # --- Condition (b): Pygments fallback produced correct line numbers ---
        # If functions were returned, every line must be > 0.
        line_numbers_correct = (
            len(ctx.functions) >= 1
            and all(f.line > 0 for f in ctx.functions)
        )

        # Gather diagnostic string for the failure message
        func_summary = [
            f"  {f.name}  line={f.line}"
            for f in ctx.functions
        ]
        diagnostic = (
            f"ctx.error={ctx.error!r}\n"
            f"ctx.functions ({len(ctx.functions)} total):\n"
            + ("\n".join(func_summary) if func_summary else "  (none)")
        )

        assert error_reported or line_numbers_correct, (
            "Bug 007 silent degradation: get_relevant_context returned a "
            "useless result when TREE_SITTER_SWIFT_AVAILABLE=False with no "
            "user-visible diagnostic.\n"
            "Expected EITHER ctx.error to be set (dep missing reported) OR "
            "all returned FunctionContext.line > 0 (Pygments fallback with "
            "correct line numbers).\n"
            "Neither condition was met:\n"
            f"{diagnostic}"
        )
