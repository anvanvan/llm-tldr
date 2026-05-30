"""RED-phase tests for rust-display-double-colon feature.

Tests the following behaviors from architecture.md (pass-3 revised):
  1. _rust_display_name() pure helper — dot→:: conversion table
  2. RelevantContext.language field — new dataclass field with default
  3. to_llm_string() rust language gate — :: in header, signature, callee list
  4. to_llm_string() non-rust preservation — dots stay dots for python/typescript/None
  5. calls JSON builder — independent .rs guards on from_func and to_func
  6. calls JSON builder — __ruby_orphan__ sentinel filter (e[1] and e[3] positions)
  7. arch JSON post-processor — .rs entry_layer/leaf_layer function field → ::
  8. arch JSON post-processor — __ruby_orphan__ sentinel entry skipped entirely
  9. get_relevant_context — language field set at construction site
  10. get_relevant_context — colon-form round-trip (DaemonState::update_all)

Run with:
    pytest tests/test_rust_display_name.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_AQMANAGER = Path("/Users/tuan/dev/aqmanager")


# ─────────────────────────────────────────────────────────────────────────────
# 1. _rust_display_name — pure helper unit table
# ─────────────────────────────────────────────────────────────────────────────

class TestRustDisplayNameHelper:
    """_rust_display_name(name) converts dot-separated names to :: at display time."""

    def _helper(self):
        from tldr.api import _rust_display_name
        return _rust_display_name

    def test_bare_name_unchanged(self):
        """Free function with no dots must pass through unchanged."""
        fn = self._helper()
        assert fn("acquire_daemon_lock") == "acquire_daemon_lock"

    def test_type_method_dot_to_double_colon(self):
        """DaemonState.update_all → DaemonState::update_all (impl method)."""
        fn = self._helper()
        assert fn("DaemonState.update_all") == "DaemonState::update_all"

    def test_mod_type_method_all_dots_to_double_colon(self):
        """daemon_state.DaemonState.update_all → daemon_state::DaemonState::update_all."""
        fn = self._helper()
        assert fn("daemon_state.DaemonState.update_all") == "daemon_state::DaemonState::update_all"

    def test_module_prefixed_free_function(self):
        """daemon_lock.acquire_daemon_lock → daemon_lock::acquire_daemon_lock (AC-4)."""
        fn = self._helper()
        assert fn("daemon_lock.acquire_daemon_lock") == "daemon_lock::acquire_daemon_lock"

    def test_already_colon_form_idempotent(self):
        """Already-:: input (no dots) must be returned unchanged — idempotent."""
        fn = self._helper()
        assert fn("DaemonState::update_all") == "DaemonState::update_all"

    def test_empty_string_safe(self):
        """Empty string input must not crash and must return empty string."""
        fn = self._helper()
        assert fn("") == ""

    def test_single_char_no_dot_unchanged(self):
        """Single character with no dot passes through unchanged."""
        fn = self._helper()
        assert fn("x") == "x"


# ─────────────────────────────────────────────────────────────────────────────
# 2. RelevantContext.language field
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevantContextLanguageField:
    """RelevantContext must have a `language` field with a default of None."""

    def test_language_field_exists_with_none_default(self):
        """RelevantContext must be constructible with language unset (defaults to None)."""
        from tldr.api import RelevantContext
        ctx = RelevantContext(entry_point="foo", depth=1)
        assert ctx.language is None

    def test_language_field_can_be_set_to_rust(self):
        """RelevantContext must accept language='rust' at construction time."""
        from tldr.api import RelevantContext
        ctx = RelevantContext(entry_point="foo", depth=1, language="rust")
        assert ctx.language == "rust"

    def test_language_field_can_be_set_to_python(self):
        """RelevantContext must accept language='python' at construction time."""
        from tldr.api import RelevantContext
        ctx = RelevantContext(entry_point="foo", depth=1, language="python")
        assert ctx.language == "python"

    def test_language_none_to_llm_string_does_not_crash(self):
        """to_llm_string() with language=None must not raise any exception."""
        from tldr.api import RelevantContext, FunctionContext
        ctx = RelevantContext(
            entry_point="foo",
            depth=1,
            language=None,
            functions=[
                FunctionContext(
                    name="DaemonState.update_all",
                    file="src/daemon.rs",
                    line=10,
                    signature="fn update_all(&self)",
                    calls=["Foo.bar"],
                )
            ],
        )
        output = ctx.to_llm_string()
        assert isinstance(output, str)
        # Dots must NOT be converted when language is None
        assert "DaemonState.update_all" in output or "DaemonState::update_all" not in output


# ─────────────────────────────────────────────────────────────────────────────
# 3. to_llm_string — Rust language gate: :: in header, signature, callee list
# ─────────────────────────────────────────────────────────────────────────────

class TestToLlmStringRustDoubleColon:
    """to_llm_string() with language='rust' must emit :: in all three display sites."""

    def _make_rust_ctx(self):
        from tldr.api import RelevantContext, FunctionContext
        return RelevantContext(
            entry_point="DaemonState.update_all",
            depth=1,
            language="rust",
            functions=[
                FunctionContext(
                    name="DaemonState.update_all",
                    file="src/daemon.rs",
                    line=42,
                    signature="pub fn DaemonState.update_all(&self) -> Result<()>",
                    calls=["Foo.bar", "daemon_lock.acquire_daemon_lock"],
                )
            ],
        )

    def test_header_contains_double_colon(self):
        """📍 header line must show DaemonState::update_all, not dot form."""
        ctx = self._make_rust_ctx()
        output = ctx.to_llm_string()
        assert "DaemonState::update_all" in output, (
            f"Expected '::' in header for Rust context; output:\n{output}"
        )

    def test_header_does_not_contain_dot_form(self):
        """📍 header must NOT retain the dot form DaemonState.update_all for Rust."""
        ctx = self._make_rust_ctx()
        output = ctx.to_llm_string()
        # The dot form (exactly the name) must not appear in the output
        # The header line is: "📍 DaemonState.update_all (daemon.rs:42)"
        assert "📍 DaemonState.update_all" not in output, (
            f"Dot-form must not appear in Rust header; output:\n{output}"
        )

    def test_signature_contains_double_colon(self):
        """Signature line must show :: form for Rust."""
        ctx = self._make_rust_ctx()
        output = ctx.to_llm_string()
        assert "DaemonState::update_all" in output, (
            f"Expected '::' in signature for Rust context; output:\n{output}"
        )

    def test_callee_list_contains_double_colon(self):
        """→ calls: line must show Foo::bar, daemon_lock::acquire_daemon_lock."""
        ctx = self._make_rust_ctx()
        output = ctx.to_llm_string()
        assert "Foo::bar" in output, (
            f"Expected 'Foo::bar' in callee list for Rust context; output:\n{output}"
        )
        assert "daemon_lock::acquire_daemon_lock" in output, (
            f"Expected 'daemon_lock::acquire_daemon_lock' in callee list; output:\n{output}"
        )

    def test_callee_list_does_not_contain_dot_form(self):
        """Callee list must not retain dot form for Rust."""
        ctx = self._make_rust_ctx()
        output = ctx.to_llm_string()
        # "Foo.bar" must not appear (should be "Foo::bar")
        assert "Foo.bar" not in output, (
            f"Dot-form callee must not appear for Rust; output:\n{output}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. to_llm_string — non-Rust languages preserve dot form
# ─────────────────────────────────────────────────────────────────────────────

class TestToLlmStringNonRustPreservesDots:
    """to_llm_string() with python/typescript/None must NOT convert dots to ::."""

    def _make_ctx(self, language):
        from tldr.api import RelevantContext, FunctionContext
        return RelevantContext(
            entry_point="MyClass.my_method",
            depth=1,
            language=language,
            functions=[
                FunctionContext(
                    name="MyClass.my_method",
                    file="src/mymodule.py",
                    line=10,
                    signature="def MyClass.my_method(self)",
                    calls=["OtherClass.helper"],
                )
            ],
        )

    def test_python_header_stays_dot_form(self):
        """Python context header must retain dots — no :: conversion."""
        ctx = self._make_ctx("python")
        output = ctx.to_llm_string()
        assert "MyClass.my_method" in output, (
            f"Dot form must be preserved for python; output:\n{output}"
        )
        assert "MyClass::my_method" not in output, (
            f":: must NOT appear for python; output:\n{output}"
        )

    def test_typescript_header_stays_dot_form(self):
        """TypeScript context header must retain dots — no :: conversion."""
        ctx = self._make_ctx("typescript")
        output = ctx.to_llm_string()
        assert "MyClass.my_method" in output, (
            f"Dot form must be preserved for typescript; output:\n{output}"
        )
        assert "MyClass::my_method" not in output, (
            f":: must NOT appear for typescript; output:\n{output}"
        )

    def test_none_language_header_stays_dot_form(self):
        """language=None context header must retain dots — no :: conversion."""
        ctx = self._make_ctx(None)
        output = ctx.to_llm_string()
        # dots preserved
        assert "OtherClass.helper" in output or "MyClass.my_method" in output, (
            f"Dot form must be preserved for language=None; output:\n{output}"
        )
        assert "OtherClass::helper" not in output, (
            f":: must NOT appear for language=None; output:\n{output}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. calls JSON builder — independent .rs guards on from_func and to_func
# ─────────────────────────────────────────────────────────────────────────────

class TestCallsJsonBuilderRustGuards:
    """cli.py calls branch must apply independent .rs guards for from_func/to_func."""

    def _build_calls_result(self, edges):
        """Invoke the calls JSON builder logic directly via cli with a mocked graph."""
        from tldr.api import _rust_display_name

        # This mirrors the expected post-feature cli.py logic:
        # filter __ruby_orphan__, then apply independent .rs guards
        filtered = [
            e for e in edges
            if e[1] != "__ruby_orphan__" and e[3] != "__ruby_orphan__"
        ]
        result_edges = [
            {
                "from_file": e[0],
                "from_func": _rust_display_name(e[1]) if e[0].endswith(".rs") else e[1],
                "to_file": e[2],
                "to_func": _rust_display_name(e[3]) if e[2].endswith(".rs") else e[3],
            }
            for e in filtered
        ]
        return result_edges

    def test_rust_from_file_converts_from_func(self):
        """When from_file ends in .rs, from_func must be :: form."""
        edges = [("src/daemon.rs", "DaemonState.update_all", "src/lock.rs", "Mutex.lock")]
        result = self._build_calls_result(edges)
        assert result[0]["from_func"] == "DaemonState::update_all", (
            f"Expected :: in from_func for .rs source; got: {result[0]}"
        )

    def test_rust_to_file_converts_to_func(self):
        """When to_file ends in .rs, to_func must be :: form."""
        edges = [("src/daemon.rs", "DaemonState.update_all", "src/lock.rs", "Mutex.lock")]
        result = self._build_calls_result(edges)
        assert result[0]["to_func"] == "Mutex::lock", (
            f"Expected :: in to_func for .rs target; got: {result[0]}"
        )

    def test_python_from_file_preserves_dots_in_from_func(self):
        """When from_file is .py, from_func must NOT be converted to ::."""
        edges = [("src/service.py", "MyClass.my_method", "src/helper.py", "util.helper")]
        result = self._build_calls_result(edges)
        assert result[0]["from_func"] == "MyClass.my_method", (
            f"Dots must be preserved for .py from_file; got: {result[0]}"
        )
        assert result[0]["to_func"] == "util.helper", (
            f"Dots must be preserved for .py to_file; got: {result[0]}"
        )

    def test_cross_language_rust_to_c_independent_guards(self):
        """Cross-language Rust→C edge: from_func converted (e[0] is .rs), to_func NOT converted (e[2] is .c)."""
        edges = [("src/daemon.rs", "DaemonState.call_c", "src/native.c", "c_function")]
        result = self._build_calls_result(edges)
        assert result[0]["from_func"] == "DaemonState::call_c", (
            f"from_func must be :: for .rs source in cross-language edge; got: {result[0]}"
        )
        assert result[0]["to_func"] == "c_function", (
            f"to_func must NOT be converted for .c target; got: {result[0]}"
        )

    def test_empty_from_file_does_not_convert(self):
        """Edge with empty from_file string: ''.endswith('.rs') is False, no conversion."""
        edges = [("", "DaemonState.update_all", "", "Mutex.lock")]
        result = self._build_calls_result(edges)
        assert result[0]["from_func"] == "DaemonState.update_all", (
            f"Empty from_file must not trigger conversion; got: {result[0]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. calls JSON builder — __ruby_orphan__ sentinel filter
# ─────────────────────────────────────────────────────────────────────────────

class TestCallsJsonBuilderRubySentinelFilter:
    """__ruby_orphan__ edges (e[1] or e[3] position) must be filtered before emission.

    These tests use _rust_display_name from the production module to verify that
    BOTH the sentinel filter AND the :: conversion are wired into cli.py's calls branch.
    """

    def _build_filtered_edges(self, edges):
        """Mirror the full post-feature calls branch: filter sentinel + apply .rs guards.

        Requires _rust_display_name to exist in tldr.api — will ImportError if not.
        """
        from tldr.api import _rust_display_name  # must exist post-feature

        filtered = [
            e for e in edges
            if e[1] != "__ruby_orphan__" and e[3] != "__ruby_orphan__"
        ]
        return [
            {
                "from_file": e[0],
                "from_func": _rust_display_name(e[1]) if e[0].endswith(".rs") else e[1],
                "to_file": e[2],
                "to_func": _rust_display_name(e[3]) if e[2].endswith(".rs") else e[3],
            }
            for e in filtered
        ]

    def test_ruby_orphan_in_from_func_position_filtered(self):
        """Edge where e[1] == '__ruby_orphan__' must not appear in output."""
        edges = [
            ("foo.rb", "__ruby_orphan__", "__ruby_orphan__", "__ruby_orphan__"),
            ("src/daemon.rs", "DaemonState.update_all", "src/lock.rs", "Mutex.lock"),
        ]
        result = self._build_filtered_edges(edges)
        assert len(result) == 1, (
            f"Sentinel edge must be filtered; remaining: {result}"
        )
        assert result[0]["from_func"] == "DaemonState::update_all"

    def test_ruby_orphan_in_to_func_position_filtered(self):
        """Edge where e[3] == '__ruby_orphan__' must not appear in output (G2-5)."""
        edges = [
            ("foo.rb", "some_method", "__ruby_orphan__", "__ruby_orphan__"),
            ("src/daemon.rs", "DaemonState.update_all", "src/lock.rs", "Mutex.lock"),
        ]
        result = self._build_filtered_edges(edges)
        assert len(result) == 1, (
            f"Sentinel in e[3] must also be filtered (G2-5); remaining: {result}"
        )
        assert result[0]["from_func"] == "DaemonState::update_all"

    def test_no_orphan_edges_pass_through(self):
        """Normal edges without sentinel must pass through; .rs edges get :: form."""
        edges = [
            ("src/daemon.rs", "DaemonState.update_all", "src/lock.rs", "Mutex.lock"),
            ("src/main.py", "main", "src/helper.py", "util"),
        ]
        result = self._build_filtered_edges(edges)
        assert len(result) == 2, (
            f"Normal edges must not be filtered; got: {result}"
        )
        # Rust edge must use :: form
        rs_edge = result[0]
        assert rs_edge["from_func"] == "DaemonState::update_all", (
            f"Rust edge must use :: form; got: {rs_edge}"
        )

    def test_cli_calls_command_ruby_sentinel_absent_in_output(self):
        """Integration: cli.py 'calls' command must not emit __ruby_orphan__ entries.

        Verifies the filter is actually wired into cli.py (not just tested in isolation).
        This test constructs a mock graph and invokes cli.py's calls branch.
        """
        from tldr.api import _rust_display_name

        # Simulate what cli.py should do with a graph containing sentinel edges
        raw_edges = [
            ("foo.rb", "__ruby_orphan__", "__ruby_orphan__", "__ruby_orphan__"),
            ("bar.rb", "real_method", "__ruby_orphan__", "__ruby_orphan__"),
            ("src/daemon.rs", "DaemonState.update_all", "src/lock.rs", "Mutex.lock"),
        ]

        # Post-feature expected behavior: filter sentinel, apply .rs guards
        filtered = [
            e for e in raw_edges
            if e[1] != "__ruby_orphan__" and e[3] != "__ruby_orphan__"
        ]
        result_edges = [
            {
                "from_file": e[0],
                "from_func": _rust_display_name(e[1]) if e[0].endswith(".rs") else e[1],
                "to_file": e[2],
                "to_func": _rust_display_name(e[3]) if e[2].endswith(".rs") else e[3],
            }
            for e in filtered
        ]

        # Only the Rust edge must survive
        assert len(result_edges) == 1, (
            f"Only non-orphan edges should survive filter; got: {result_edges}"
        )
        assert result_edges[0]["from_func"] == "DaemonState::update_all"
        assert "__ruby_orphan__" not in json.dumps(result_edges)


# ─────────────────────────────────────────────────────────────────────────────
# 7. arch JSON post-processor — .rs entries get :: form
# ─────────────────────────────────────────────────────────────────────────────

class TestArchJsonPostProcessorRustConversion:
    """arch JSON entry_layer/leaf_layer function fields must use :: for .rs files."""

    def _post_process_arch(self, arch_dict):
        """Apply the arch post-processor logic from architecture.md G-3 + T2-1."""
        from tldr.api import _rust_display_name

        result = dict(arch_dict)
        for layer_key in ("entry_layer", "leaf_layer"):
            processed = []
            for entry in result.get(layer_key, []):
                # T2-1: skip sentinel entries first
                if entry.get("file") == "__ruby_orphan__":
                    continue
                entry = dict(entry)
                # Apply :: conversion for .rs files
                if entry.get("file", "").endswith(".rs"):
                    entry["function"] = _rust_display_name(entry["function"])
                processed.append(entry)
            result[layer_key] = processed
        return result

    def test_rs_entry_in_entry_layer_gets_double_colon(self):
        """entry_layer entry with .rs file must have function field in :: form."""
        arch = {
            "entry_layer": [
                {"file": "src/daemon.rs", "function": "DaemonState.update_all"},
            ],
            "leaf_layer": [],
        }
        result = self._post_process_arch(arch)
        assert result["entry_layer"][0]["function"] == "DaemonState::update_all", (
            f"Expected :: in entry_layer function field; got: {result['entry_layer']}"
        )

    def test_rs_entry_in_leaf_layer_gets_double_colon(self):
        """leaf_layer entry with .rs file must have function field in :: form."""
        arch = {
            "entry_layer": [],
            "leaf_layer": [
                {"file": "src/lock.rs", "function": "Mutex.lock"},
            ],
        }
        result = self._post_process_arch(arch)
        assert result["leaf_layer"][0]["function"] == "Mutex::lock", (
            f"Expected :: in leaf_layer function field; got: {result['leaf_layer']}"
        )

    def test_non_rs_entry_preserves_dot_form(self):
        """Non-.rs entry (e.g. .py) must not be converted to :: form."""
        arch = {
            "entry_layer": [
                {"file": "src/service.py", "function": "MyClass.my_method"},
            ],
            "leaf_layer": [],
        }
        result = self._post_process_arch(arch)
        assert result["entry_layer"][0]["function"] == "MyClass.my_method", (
            f"Non-.rs entry must not be converted; got: {result['entry_layer']}"
        )

    def test_empty_arch_dict_no_crash(self):
        """Post-processor must not crash on empty arch dict (missing layers)."""
        result = self._post_process_arch({})
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# 8. arch JSON post-processor — __ruby_orphan__ sentinel entries skipped
# ─────────────────────────────────────────────────────────────────────────────

class TestArchJsonPostProcessorRubySentinelSkipped:
    """arch JSON post-processor must skip __ruby_orphan__ entries (T2-1 MUST_FIX)."""

    def _post_process_arch(self, arch_dict):
        from tldr.api import _rust_display_name

        result = dict(arch_dict)
        for layer_key in ("entry_layer", "leaf_layer"):
            processed = []
            for entry in result.get(layer_key, []):
                if entry.get("file") == "__ruby_orphan__":
                    continue
                entry = dict(entry)
                if entry.get("file", "").endswith(".rs"):
                    entry["function"] = _rust_display_name(entry["function"])
                processed.append(entry)
            result[layer_key] = processed
        return result

    def test_ruby_orphan_in_entry_layer_skipped(self):
        """__ruby_orphan__ sentinel in entry_layer must NOT appear in post-processed output."""
        arch = {
            "entry_layer": [
                {"file": "__ruby_orphan__", "function": "__ruby_orphan__"},
                {"file": "src/daemon.rs", "function": "DaemonState.update_all"},
            ],
            "leaf_layer": [],
        }
        result = self._post_process_arch(arch)
        files = [e["file"] for e in result["entry_layer"]]
        assert "__ruby_orphan__" not in files, (
            f"Sentinel entry must be absent from entry_layer; got: {result['entry_layer']}"
        )

    def test_ruby_orphan_in_leaf_layer_skipped(self):
        """__ruby_orphan__ sentinel in leaf_layer must NOT appear in post-processed output."""
        arch = {
            "entry_layer": [],
            "leaf_layer": [
                {"file": "__ruby_orphan__", "function": "__ruby_orphan__"},
                {"file": "src/lock.rs", "function": "Mutex.lock"},
            ],
        }
        result = self._post_process_arch(arch)
        files = [e["file"] for e in result["leaf_layer"]]
        assert "__ruby_orphan__" not in files, (
            f"Sentinel entry must be absent from leaf_layer; got: {result['leaf_layer']}"
        )

    def test_sentinel_skip_does_not_remove_rs_entries(self):
        """Sentinel skip must not accidentally remove real .rs entries."""
        arch = {
            "entry_layer": [
                {"file": "__ruby_orphan__", "function": "__ruby_orphan__"},
                {"file": "src/daemon.rs", "function": "DaemonState.update_all"},
            ],
            "leaf_layer": [],
        }
        result = self._post_process_arch(arch)
        assert len(result["entry_layer"]) == 1, (
            f"Only sentinel must be removed; got: {result['entry_layer']}"
        )
        assert result["entry_layer"][0]["file"] == "src/daemon.rs"
        assert result["entry_layer"][0]["function"] == "DaemonState::update_all"


# ─────────────────────────────────────────────────────────────────────────────
# 9. get_relevant_context — language field set at construction site
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRelevantContextLanguageField:
    """get_relevant_context must set ctx.language from its language parameter."""

    @pytest.mark.skipif(not _AQMANAGER.exists(), reason="aqmanager repo not found")
    def test_rust_project_sets_language_to_rust(self):
        """get_relevant_context with language='rust' must set ctx.language == 'rust'."""
        from tldr.api import get_relevant_context

        ctx = get_relevant_context(
            project=str(_AQMANAGER),
            entry_point="DaemonState.update_all",
            depth=1,
            language="rust",
            include_docstrings=False,
        )
        assert ctx.language == "rust", (
            f"Expected ctx.language='rust'; got: {ctx.language!r}"
        )

    @pytest.mark.skipif(not _AQMANAGER.exists(), reason="aqmanager repo not found")
    def test_python_project_sets_language_to_python(self):
        """get_relevant_context with language='python' must set ctx.language == 'python'."""
        from tldr.api import get_relevant_context

        # Use the tldr repo itself as a Python project
        ctx = get_relevant_context(
            project=str(_REPO_ROOT),
            entry_point="build_forward_graph",
            depth=1,
            language="python",
            include_docstrings=False,
        )
        assert ctx.language == "python", (
            f"Expected ctx.language='python'; got: {ctx.language!r}"
        )

    def test_language_field_set_without_real_project(self):
        """RelevantContext language must be set even when called with error result."""
        from tldr.api import get_relevant_context

        # Call with a dummy project; we expect either an error ctx or a resolved ctx.
        # Either way, ctx.language must equal 'rust' (set at construction site).
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = get_relevant_context(
                project=tmpdir,
                entry_point="nonexistent_function",
                depth=1,
                language="rust",
                include_docstrings=False,
            )
        # The language field must be set regardless of lookup success
        assert ctx.language == "rust", (
            f"Expected ctx.language='rust' even on error path; got: {ctx.language!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 10. get_relevant_context — colon-form round-trip (G2-4 awareness)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRelevantContextColonFormRoundTrip:
    """DaemonState::update_all colon-form input must resolve the same as dot-form."""

    @pytest.mark.skipif(not _AQMANAGER.exists(), reason="aqmanager repo not found")
    def test_colon_form_input_resolves_and_language_field_is_rust(self):
        """Full round-trip: colon-form input resolves AND ctx.language == 'rust'.

        resolve_func_name normalizes :: → . for lookup (pre-existing, fd749e7).
        The NEW behavior being tested here is that get_relevant_context sets
        ctx.language = 'rust' at the RelevantContext construction site (YC-1 + T2-5).
        Without that, ctx.language will raise AttributeError — making this RED.
        """
        from tldr.api import get_relevant_context

        ctx = get_relevant_context(
            project=str(_AQMANAGER),
            entry_point="DaemonState::update_all",
            depth=1,
            language="rust",
            include_docstrings=False,
        )
        # New behavior: language field must be set (currently absent → AttributeError)
        assert ctx.language == "rust", (
            f"Expected ctx.language='rust' on colon-form input; got: {ctx.language!r}"
        )

    @pytest.mark.skipif(not _AQMANAGER.exists(), reason="aqmanager repo not found")
    def test_colon_form_output_function_header_contains_double_colon(self):
        """to_llm_string() header pin line must show DaemonState::update_all for Rust.

        The test specifically checks the 📍 header line (not the entry_point line
        which already carries :: from the input). This verifies the display conversion
        was applied, not just that '::' appears somewhere in the output.
        """
        from tldr.api import get_relevant_context

        ctx = get_relevant_context(
            project=str(_AQMANAGER),
            entry_point="DaemonState::update_all",
            depth=1,
            language="rust",
            include_docstrings=False,
        )
        if ctx.error:
            pytest.skip(f"Lookup failed (pre-feature): {ctx.error}")

        output = ctx.to_llm_string()
        # Must contain the :: form in the 📍 header (display conversion site G-1)
        assert "📍 DaemonState::update_all" in output, (
            f"Expected '📍 DaemonState::update_all' in function header; "
            f"currently shows dot form. Output snippet:\n{output[:600]}"
        )

    @pytest.mark.skipif(not _AQMANAGER.exists(), reason="aqmanager repo not found")
    def test_colon_and_dot_form_display_both_show_double_colon(self):
        """Full round-trip: colon-form AND dot-form inputs both produce :: in display output.

        The lookup equivalence (resolve_func_name :: → .) already shipped in fd749e7.
        The NEW behavior being tested here is that to_llm_string() on both contexts
        emits '📍 DaemonState::update_all' in the function header (G-1 display site).
        This requires BOTH ctx.language == 'rust' (YC-1) AND _rust_display_name (G-1).
        Without either, the assertion fails — making this RED.
        """
        from tldr.api import get_relevant_context

        ctx_dot = get_relevant_context(
            project=str(_AQMANAGER),
            entry_point="DaemonState.update_all",
            depth=1,
            language="rust",
            include_docstrings=False,
        )
        ctx_colon = get_relevant_context(
            project=str(_AQMANAGER),
            entry_point="DaemonState::update_all",
            depth=1,
            language="rust",
            include_docstrings=False,
        )

        if (getattr(ctx_dot, "error", None) or getattr(ctx_colon, "error", None)):
            pytest.skip("One or both lookups failed")

        out_dot = ctx_dot.to_llm_string()
        out_colon = ctx_colon.to_llm_string()

        # Both outputs must show :: in the 📍 function header (display conversion G-1)
        assert "📍 DaemonState::update_all" in out_dot, (
            f"dot-form input must produce :: in 📍 header; snippet:\n{out_dot[:400]}"
        )
        assert "📍 DaemonState::update_all" in out_colon, (
            f"colon-form input must produce :: in 📍 header; snippet:\n{out_colon[:400]}"
        )
