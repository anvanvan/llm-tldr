# pyright: reportMissingImports=false
"""Regression test for: tldr imports <swift-file> returns [] silently when
tree_sitter_swift is unavailable.

Root cause: cross_file_calls.py parse_swift_imports() has a guard
    if not TREE_SITTER_SWIFT_AVAILABLE:
        return []
that silently short-circuits without any diagnostic (no warning, no log, no
exception). A user invoking `tldr imports foo.swift` receives an empty array
with no indication that the result is degraded because the parser dependency
is missing.

This regression test pins the FIXED contract: when TREE_SITTER_SWIFT_AVAILABLE
is False and the target file contains visible import statements, parse_swift_imports
MUST emit at least one Python warning (via warnings.warn or any subclass of
Warning) whose message mentions 'swift' or 'tree_sitter' — callers must not
receive a silent empty list.

The test fails on the current (unfixed) code because no warning is emitted.
"""

import warnings
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_SWIFT_SOURCE_WITH_IMPORTS = (
    "import Foundation\n"
    "import SwiftUI\n"
    "import AppKit\n"
    "\n"
    "class MyViewController {}\n"
)


# ---------------------------------------------------------------------------
# Regression test (ONE — covers the full root-cause contract)
# ---------------------------------------------------------------------------

class TestSwiftImportsMissingDepDiagnostic:
    """parse_swift_imports must not silently return [] when tree_sitter_swift
    is unavailable on a file that visibly contains import statements."""

    def test_parse_swift_imports_warns_with_actionable_message_when_dep_unavailable(
        self, tmp_path
    ):
        """When TREE_SITTER_SWIFT_AVAILABLE is False, parse_swift_imports must
        emit at least one Python warning whose text mentions 'swift' or
        'tree_sitter', so the missing-dep failure is observable.

        RED contract (current code): returns [] with no warning → fails.
        FIXED contract: a Warning is issued; its message is actionable (mentions
        'swift' or 'tree_sitter') so callers / users can diagnose the problem.
        """
        from tldr.cross_file_calls import parse_swift_imports

        swift_file = tmp_path / "App.swift"
        swift_file.write_text(_SWIFT_SOURCE_WITH_IMPORTS)

        # Patch the availability flag to simulate the missing-dep environment.
        with patch("tldr.cross_file_calls.TREE_SITTER_SWIFT_AVAILABLE", False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = parse_swift_imports(str(swift_file))

        # Assertion 1: at least one warning must have been emitted.
        assert len(caught) >= 1, (
            "parse_swift_imports silently returned [] when tree_sitter_swift is "
            "unavailable on a Swift file that contains import statements. "
            "Expected at least one warning to be emitted so callers can detect "
            "the degraded result. "
            f"Got: result={result!r}, warnings={[str(w.message) for w in caught]}"
        )

        # Assertion 2: the warning must be actionable — mention the dep or lang.
        messages = " ".join(str(w.message).lower() for w in caught)
        assert "swift" in messages or "tree_sitter" in messages, (
            "A warning was emitted but does not mention 'swift' or 'tree_sitter', "
            "making it unactionable for diagnosing the missing dependency. "
            f"Got warning messages: {[str(w.message) for w in caught]}"
        )

    # ---------------------------------------------------------------------------
    # Control: happy path must still work after the fix
    # ---------------------------------------------------------------------------

    def test_parse_swift_imports_returns_correct_result_when_dep_available(
        self, tmp_path
    ):
        """Baseline / control: when tree_sitter_swift IS available, parse_swift_imports
        must return the correct non-empty list of imports.

        Skipped automatically when tree_sitter_swift is not installed.
        """
        pytest.importorskip("tree_sitter_swift")

        from tldr.cross_file_calls import parse_swift_imports

        swift_file = tmp_path / "App.swift"
        swift_file.write_text(_SWIFT_SOURCE_WITH_IMPORTS)

        result = parse_swift_imports(str(swift_file))

        assert isinstance(result, list), (
            f"Expected list, got {type(result)}"
        )
        assert len(result) >= 1, (
            f"Expected at least 1 import from Swift file with 3 import statements, "
            f"got: {result!r}"
        )
        module_names = [entry.get("module") for entry in result]
        assert "Foundation" in module_names, (
            f"Expected 'Foundation' in parsed imports, got: {module_names}"
        )
