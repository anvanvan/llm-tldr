"""Regression tests for PHP extraction pipeline bugs.

F1: Anonymous class `continue` skips constructor arguments.
    _extract_php_calls has a `continue` on object_creation_expression when it
    contains an anonymous_class. This skips the ENTIRE node, including
    constructor arguments. So `new class($this->getValue()) { ... }` loses the
    `$this->getValue()` call.

F3: rename_prefix overwrites existing dict keys.
    `d[new_key] = d.pop(k)` overwrites if new_key already exists. When a
    fully-qualified name already appears as a key, the rename silently drops
    edges.

Run with:
    pytest tests/test_php_extract_bugs.py -v
"""

from pathlib import Path

from tldr.ast_extractor import CallGraphInfo
from tldr.hybrid_extractor import HybridExtractor


# ---------------------------------------------------------------------------
# F1: Anonymous class constructor arguments must be extracted
# ---------------------------------------------------------------------------

class TestAnonymousClassConstructorArgs:
    """Bug F1: constructor args of anonymous classes must not be skipped."""

    def test_getValue_extracted_from_anonymous_class_constructor(self, tmp_path: Path):
        """Calls in anonymous class constructor args should appear in caller's call graph."""
        php_file = tmp_path / "anon_ctor.php"
        php_file.write_text("""\
<?php
class Foo {
    public function bar() {
        $handler = new class($this->getValue()) {
            public function handle() {
                $this->process();
            }
        };
    }
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(str(php_file))

        callees = result.call_graph.calls.get("Foo::bar", [])
        assert "Foo::getValue" in callees, (
            f"getValue() in anonymous class constructor args was not extracted; "
            f"got callees: {callees}"
        )

    def test_anonymous_class_body_calls_not_leaked_to_outer_scope(self, tmp_path: Path):
        """Calls inside anonymous class body must NOT leak into the outer method."""
        php_file = tmp_path / "anon_body.php"
        php_file.write_text("""\
<?php
class Foo {
    public function bar() {
        $handler = new class($this->getValue()) {
            public function handle() {
                $this->process();
            }
        };
    }
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(str(php_file))

        callees = result.call_graph.calls.get("Foo::bar", [])
        assert "Foo::process" not in callees, (
            f"process() from anonymous class body leaked into Foo::bar scope; "
            f"got callees: {callees}"
        )

    def test_anonymous_class_without_constructor_args_no_regression(self, tmp_path: Path):
        """Anonymous class with no constructor args should not break extraction."""
        php_file = tmp_path / "anon_no_args.php"
        php_file.write_text("""\
<?php
class Foo {
    public function bar() {
        $handler = new class {
            public function handle() {
                $this->process();
            }
        };
        $this->doWork();
    }
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(str(php_file))

        callees = result.call_graph.calls.get("Foo::bar", [])
        assert "Foo::doWork" in callees, (
            f"doWork() after anonymous class was not extracted; "
            f"got callees: {callees}"
        )
        assert "Foo::process" not in callees, (
            f"process() from anonymous class body leaked; "
            f"got callees: {callees}"
        )

    def test_multiple_args_in_anonymous_class_constructor(self, tmp_path: Path):
        """Multiple calls in anonymous class constructor should all be extracted."""
        php_file = tmp_path / "anon_multi_args.php"
        php_file.write_text("""\
<?php
namespace App\\Services;
class Handler {
    public function dispatch() {
        $listener = new class($this->getConfig(), $this->getLogger()) {
            public function onEvent() {
                $this->log();
            }
        };
        $this->notify();
    }
}
""")

        extractor = HybridExtractor()
        result = extractor.extract(str(php_file))

        key = "App\\Services\\Handler::dispatch"
        callees = result.call_graph.calls.get(key, [])
        assert "App\\Services\\Handler::getConfig" in callees, (
            f"getConfig missing from callees; got: {callees}"
        )
        assert "App\\Services\\Handler::getLogger" in callees, (
            f"getLogger missing from callees; got: {callees}"
        )
        assert "App\\Services\\Handler::notify" in callees, (
            f"notify missing from callees; got: {callees}"
        )


# ---------------------------------------------------------------------------
# F3: rename_prefix must merge lists, not overwrite
# ---------------------------------------------------------------------------

class TestRenamePrefixMerge:
    """Bug F3: rename_prefix must merge lists when target key already exists."""

    def test_rename_prefix_merges_when_key_already_exists(self):
        """When new_key already exists in called_by, edges must be merged."""
        cg = CallGraphInfo()
        cg.add_call("User::save", "Foo::staticMethod")
        cg.add_call("Other::bar", "App\\Models\\Foo::staticMethod")

        cg.rename_prefix("Foo::", "App\\Models\\Foo::")

        callers = cg.called_by.get("App\\Models\\Foo::staticMethod", [])
        assert "User::save" in callers, (
            f"User::save was lost after rename_prefix merge; got callers: {callers}"
        )
        assert "Other::bar" in callers, (
            f"Other::bar was lost after rename_prefix merge; got callers: {callers}"
        )

    def test_rename_prefix_no_collision_works_normally(self):
        """When no key collision exists, rename_prefix should work as before."""
        cg = CallGraphInfo()
        cg.add_call("User::save", "Foo::bar")

        cg.rename_prefix("Foo::", "App\\Models\\Foo::")

        calls = cg.calls.get("User::save", [])
        assert "App\\Models\\Foo::bar" in calls, (
            f"rename failed; got calls: {calls}"
        )
        assert "Foo::bar" not in calls, (
            f"old key remains after rename; got calls: {calls}"
        )

    def test_rename_prefix_merge_produces_no_duplicates(self):
        """Merged lists must not contain duplicate entries."""
        cg = CallGraphInfo()
        cg.add_call("A", "Foo::m")
        cg.add_call("A", "NS\\Foo::m")

        cg.rename_prefix("Foo::", "NS\\Foo::")

        callers = cg.called_by.get("NS\\Foo::m", [])
        assert callers.count("A") == 1, (
            f"duplicate caller A after merge; got callers: {callers}"
        )
