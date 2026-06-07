"""RED-phase tests for THIN-IMPORT + CLI-BATCH behaviors.

Behaviors under test (per architecture.md):
  B1. Importing tldr.cli in a fresh subprocess must NOT load torch or
      sentence_transformers.
  B2. tldr.lang_constants is torch-free and exposes the three required symbols.
  B3. semantic.py re-exports ALL_LANGUAGES, EXTENSION_TO_LANGUAGE,
      _resolve_device_arg and those objects are identical to the ones from
      tldr.lang_constants.
  B4. The `tldr daemon notify` argparse subparser accepts nargs="+" (multiple
      positional file paths) and stores them as a list.

Each test maps 1-to-1 to a behavior.  All four MUST fail on the current
codebase (RED) because:
  - tldr/lang_constants.py does not yet exist → B2 fails with ModuleNotFoundError
  - cli.py still imports from tldr.semantic (which pulls torch at top level) →
    B1 fails (torch IS in sys.modules)
  - semantic.py does not yet re-export from lang_constants → B3 fails
  - daemon_notify_p.add_argument uses "file" (singular, nargs=None) → B4 fails

NOTE: this file intentionally does NOT test daemon socket/_handle_notify wire
handling; those are covered by the companion architect's test file.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Repository root — every subprocess call uses this as cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).parent.parent)


# ===========================================================================
# B1 — Importing tldr.cli must NOT load torch / sentence_transformers
# ===========================================================================

class TestThinImport:
    """B1: tldr.cli import must not transitively load ML libraries."""

    def test_cli_import_does_not_load_torch(self):
        """After `import tldr.cli`, 'torch' must NOT be in sys.modules.

        This test runs a fresh Python subprocess so the module cache is clean.
        On current code (cli.py line 55 imports from tldr.semantic which
        imports sentence_transformers/torch) this assertion FAILS → RED.

        Architecture contract: once lang_constants.py exists and cli.py
        imports from it instead of from .semantic, torch stays out of the
        module cache on every short-lived `tldr daemon notify` invocation.
        """
        snippet = (
            "import sys; "
            "import tldr.cli; "
            "in_torch = 'torch' in sys.modules; "
            "in_st = 'sentence_transformers' in sys.modules; "
            "print(f'torch:{in_torch} sentence_transformers:{in_st}')"
        )
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"subprocess crashed; stderr: {result.stderr!r}"
        )
        stdout = result.stdout.strip()
        assert "torch:False" in stdout, (
            f"Expected 'torch:False' in subprocess stdout but got: {stdout!r}. "
            "cli.py still imports from tldr.semantic (which pulls torch) at "
            "module scope. Fix: replace line-55 import with "
            "`from .lang_constants import ALL_LANGUAGES, EXTENSION_TO_LANGUAGE, "
            "_resolve_device_arg`."
        )
        assert "sentence_transformers:False" in stdout, (
            f"Expected 'sentence_transformers:False' in subprocess stdout but "
            f"got: {stdout!r}. sentence_transformers is loaded transitively "
            "through tldr.semantic when cli.py is imported."
        )


# ===========================================================================
# B2 — tldr.lang_constants is torch-free and exposes correct symbols
# ===========================================================================

class TestLangConstants:
    """B2: tldr.lang_constants must exist, be torch-free, and expose the
    three required public symbols with correct types and spot-check values.
    """

    def test_lang_constants_module_exists_and_is_importable(self):
        """tldr.lang_constants must be importable without error.

        RED: the module does not exist yet → ModuleNotFoundError.
        """
        import importlib
        # This will raise ModuleNotFoundError on current code → RED ✓
        mod = importlib.import_module("tldr.lang_constants")
        assert mod is not None

    def test_lang_constants_import_does_not_load_torch(self):
        """After `import tldr.lang_constants`, torch must NOT be in sys.modules.

        Uses a clean subprocess so the module cache is pristine.
        RED: module doesn't exist yet → subprocess exits non-zero.
        """
        snippet = (
            "import sys; "
            "import tldr.lang_constants; "
            "in_torch = 'torch' in sys.modules; "
            "in_st = 'sentence_transformers' in sys.modules; "
            "print(f'torch:{in_torch} sentence_transformers:{in_st}')"
        )
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"subprocess failed (module likely missing); stderr: {result.stderr!r}"
        )
        stdout = result.stdout.strip()
        assert "torch:False" in stdout, (
            f"tldr.lang_constants pulled torch into sys.modules: {stdout!r}"
        )
        assert "sentence_transformers:False" in stdout, (
            f"tldr.lang_constants pulled sentence_transformers into sys.modules: "
            f"{stdout!r}"
        )

    def test_lang_constants_exposes_all_languages_nonempty_list(self):
        """ALL_LANGUAGES must be a non-empty list (or list-like collection).

        RED: module missing → ModuleNotFoundError.
        """
        from tldr.lang_constants import ALL_LANGUAGES  # noqa: PLC0415

        assert isinstance(ALL_LANGUAGES, list), (
            f"ALL_LANGUAGES must be a list; got {type(ALL_LANGUAGES)!r}"
        )
        assert len(ALL_LANGUAGES) > 0, "ALL_LANGUAGES must not be empty"
        # Architecture specifies 17 elements — lock that count so a stray
        # deletion is caught immediately.
        assert len(ALL_LANGUAGES) == 17, (
            f"ALL_LANGUAGES must have 17 elements (per architecture.md); "
            f"got {len(ALL_LANGUAGES)}: {ALL_LANGUAGES!r}"
        )

    def test_lang_constants_exposes_extension_to_language_dict(self):
        """EXTENSION_TO_LANGUAGE must be a dict with spot-check entries.

        RED: module missing → ModuleNotFoundError.
        """
        from tldr.lang_constants import EXTENSION_TO_LANGUAGE  # noqa: PLC0415

        assert isinstance(EXTENSION_TO_LANGUAGE, dict), (
            f"EXTENSION_TO_LANGUAGE must be a dict; got {type(EXTENSION_TO_LANGUAGE)!r}"
        )
        assert len(EXTENSION_TO_LANGUAGE) > 0, (
            "EXTENSION_TO_LANGUAGE must not be empty"
        )
        # Spot-check the most common entries
        assert EXTENSION_TO_LANGUAGE.get(".py") == "python", (
            f"Expected EXTENSION_TO_LANGUAGE['.py'] == 'python'; "
            f"got {EXTENSION_TO_LANGUAGE.get('.py')!r}"
        )
        assert EXTENSION_TO_LANGUAGE.get(".ts") == "typescript", (
            f"Expected EXTENSION_TO_LANGUAGE['.ts'] == 'typescript'; "
            f"got {EXTENSION_TO_LANGUAGE.get('.ts')!r}"
        )

    def test_lang_constants_exposes_resolve_device_arg_callable(self):
        """_resolve_device_arg must be a callable.

        RED: module missing → ModuleNotFoundError.
        """
        from tldr.lang_constants import _resolve_device_arg  # noqa: PLC0415

        assert callable(_resolve_device_arg), (
            f"_resolve_device_arg must be callable; got {type(_resolve_device_arg)!r}"
        )


# ===========================================================================
# B3 — semantic.py re-exports the three names from lang_constants (back-compat)
# ===========================================================================

class TestSemanticReexport:
    """B3: `from tldr.semantic import X` still works after the refactor and
    the objects are identical to (or equal to) those in tldr.lang_constants.
    """

    def test_semantic_still_exports_all_languages(self):
        """ALL_LANGUAGES imported from tldr.semantic must equal the one from
        tldr.lang_constants (same content; identity or equality).

        RED: tldr.lang_constants doesn't exist yet → ModuleNotFoundError when
        the back-compat re-export path tries to import from it.
        """
        from tldr.semantic import ALL_LANGUAGES as sem_AL  # noqa: PLC0415
        from tldr.lang_constants import ALL_LANGUAGES as lc_AL  # noqa: PLC0415

        assert sem_AL == lc_AL, (
            "ALL_LANGUAGES from tldr.semantic must equal tldr.lang_constants. "
            f"semantic: {sem_AL!r} vs lang_constants: {lc_AL!r}"
        )

    def test_semantic_still_exports_extension_to_language(self):
        """EXTENSION_TO_LANGUAGE imported from tldr.semantic must equal the
        one from tldr.lang_constants.

        RED: same — lang_constants missing.
        """
        from tldr.semantic import EXTENSION_TO_LANGUAGE as sem_EL  # noqa: PLC0415
        from tldr.lang_constants import EXTENSION_TO_LANGUAGE as lc_EL  # noqa: PLC0415

        assert sem_EL == lc_EL, (
            "EXTENSION_TO_LANGUAGE from tldr.semantic must equal "
            "tldr.lang_constants. "
            f"semantic['.py']={sem_EL.get('.py')!r} vs "
            f"lang_constants['.py']={lc_EL.get('.py')!r}"
        )

    def test_semantic_still_exports_resolve_device_arg_callable(self):
        """_resolve_device_arg imported from tldr.semantic must be callable
        (back-compat for any external caller doing
        `from tldr.semantic import _resolve_device_arg`).

        RED: lang_constants missing → import chain fails.
        """
        from tldr.semantic import _resolve_device_arg as sem_rda  # noqa: PLC0415
        from tldr.lang_constants import _resolve_device_arg as lc_rda  # noqa: PLC0415

        assert callable(sem_rda), (
            "_resolve_device_arg from tldr.semantic must be callable"
        )
        assert callable(lc_rda), (
            "_resolve_device_arg from tldr.lang_constants must be callable"
        )
        # Both must be the same object (re-export, not a copy)
        assert sem_rda is lc_rda, (
            "_resolve_device_arg from tldr.semantic and tldr.lang_constants "
            "must be the identical object (semantic re-exports from lang_constants). "
            "If they differ, back-compat callers may get a different function."
        )


# ===========================================================================
# B4 — CLI argparse: daemon notify accepts nargs="+" (multiple file paths)
# ===========================================================================

class TestDaemonNotifyArgparse:
    """B4: The `tldr daemon notify` subparser must accept multiple positional
    file paths (nargs="+") and store them as a list attribute.
    """

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        """Rebuild the real tldr argparse parser by calling cli.main with
        a patched sys.argv that asks only for --help on a harmless subcommand,
        then extract the parser.

        Strategy: import the real argparse setup by parsing a known-good
        set of args.  Because main() embeds the parser, we have to replicate
        the relevant daemon subparser fragment so our test is not brittle.
        However, we want to test the REAL parser, not a replica.

        We test the real parser by patching sys.argv and calling
        `parser.parse_args()` directly after extracting the parser from
        within main's scope — but that requires refactoring cli.py.

        Instead, we use the subprocess approach: run
        `python -m tldr.cli daemon notify f1 f2 --project .`
        and check exit code != 2 (argparse error).  Exit 2 means argparse
        rejected the args.  We also do a direct argparse test by constructing
        a minimal replica of the daemon notify subparser.
        """
        # Construct a minimal replica matching the architecture spec:
        # daemon_notify_p.add_argument("files", nargs="+", ...)
        # This is what the POST-fix parser must look like.
        # The test asserts that parsing succeeds and produces a list.
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="action")
        notify_p = sub.add_parser("notify")
        # Architecture spec: nargs="+" → attribute name "files"
        notify_p.add_argument("files", nargs="+", help="Paths to changed files")
        notify_p.add_argument("--project", "-p", default=".", help="Project path")
        return p

    @staticmethod
    def _capture_real_parser():
        """Capture the real cli.py ArgumentParser without running main() to
        completion. Uses monkey-patching of parse_args to intercept the parser
        object right before it would attempt to dispatch to a subcommand.

        Returns the captured ArgumentParser instance.
        """
        import argparse as _argparse
        import sys as _sys

        captured = {}
        _real_parse_args = _argparse.ArgumentParser.parse_args

        def _capture_parser(self, args=None, namespace=None):
            captured["parser"] = self
            raise SystemExit(0)

        _argparse.ArgumentParser.parse_args = _capture_parser
        _orig_argv = _sys.argv[:]
        _sys.argv = ["tldr", "--help"]
        try:
            from tldr.cli import main
            main()
        except SystemExit:
            pass
        finally:
            _argparse.ArgumentParser.parse_args = _real_parse_args
            _sys.argv = _orig_argv

        p = captured.get("parser")
        assert p is not None, "Failed to capture parser from tldr.cli.main()"
        return p

    def test_notify_accepts_multiple_files_in_real_cli(self):
        """The real cli.py parser must accept three positional file paths for
        `daemon notify` without raising SystemExit(2).

        On current code, `daemon_notify_p.add_argument("file", ...)` (singular,
        no nargs="+") causes argparse to reject the second positional with
        SystemExit(2).

        RED: parse_args raises SystemExit(2) when 3 files given.
        GREEN (post-fix): parse succeeds and returns a namespace.
        """
        p = self._capture_real_parser()

        try:
            args = p.parse_args(
                ["daemon", "notify", "file1.py", "file2.py", "file3.py", "--project", "."]
            )
        except SystemExit as e:
            raise AssertionError(
                f"Real cli.py parser rejected 'daemon notify f1 f2 f3' with exit {e.code}. "
                "Current code: add_argument('file', ...) is singular — cannot accept 3 paths. "
                "Fix: add_argument('files', nargs='+', ...)"
            ) from e

        # Verify we got the right structure (3-element list)
        assert hasattr(args, "files"), (
            f"Parsed namespace has no 'files' attribute. namespace={vars(args)!r}"
        )
        assert len(args.files) == 3, (
            f"Expected 3 files in args.files but got {args.files!r}"
        )

    def test_notify_real_cli_parsed_namespace_has_files_attribute_as_list(self):
        """The REAL cli.py daemon notify subparser must have a positional action
        with dest='files' and nargs='+' — NOT dest='file' and nargs=None.

        Introspects the real parser's _actions without running to completion.
        On current code: the 'notify' positional is named 'file' (dest='file'),
        nargs=None → this test FAILS (no action with dest='files' and nargs='+').

        After the fix: dest='files', nargs='+' → PASSES.
        """
        p = self._capture_real_parser()

        # Walk parser actions to find daemon subparser
        daemon_sp = None
        for action in p._actions:
            if hasattr(action, "_name_parser_map") and "daemon" in action._name_parser_map:
                daemon_sp = action._name_parser_map["daemon"]
                break
        assert daemon_sp is not None, "daemon subparser not found in real cli.py parser"

        # Walk daemon actions to find notify subparser
        notify_sp = None
        for action in daemon_sp._actions:
            if hasattr(action, "_name_parser_map") and "notify" in action._name_parser_map:
                notify_sp = action._name_parser_map["notify"]
                break
        assert notify_sp is not None, "notify subparser not found in daemon parser"

        # Find positional actions (no option_strings = positional)
        pos_actions = [a for a in notify_sp._actions if not a.option_strings]
        files_action = next(
            (a for a in pos_actions if a.dest == "files"), None
        )
        assert files_action is not None, (
            f"No positional action with dest='files' found in the notify subparser. "
            f"Current positional actions: {[(a.dest, a.nargs) for a in pos_actions]!r}. "
            "Current code uses add_argument('file', ...) which produces dest='file', nargs=None. "
            "Fix: add_argument('files', nargs='+', ...)"
        )
        assert files_action.nargs == "+", (
            f"notify 'files' action must have nargs='+' but got nargs={files_action.nargs!r}. "
            "Fix: add_argument('files', nargs='+', ...)"
        )

    def test_notify_argparse_single_file_yields_one_element_list(self):
        """Single-file `daemon notify f1 --project P` must produce
        args.files == ['f1'] (a one-element list), not a bare string.

        Uses the REAL cli.py parser via _capture_real_parser() — same pattern
        as the sibling two-file test — so this test passes iff and only iff
        cli.py actually uses nargs='+' with dest='files'.
        """
        p = self._capture_real_parser()

        try:
            args = p.parse_args(["daemon", "notify", "f1", "--project", "P"])
        except SystemExit as e:
            raise AssertionError(
                f"Real cli.py parser rejected single-file 'daemon notify f1' with exit {e.code}. "
                "nargs='+' must accept exactly 1 positional argument."
            ) from e

        assert hasattr(args, "files"), (
            f"Parsed namespace has no 'files' attribute. namespace={vars(args)!r}. "
            "Fix: add_argument('files', nargs='+', ...)"
        )
        assert args.files == ["f1"], (
            f"Expected args.files == ['f1'] for single-file case; got {args.files!r}. "
            "nargs='+' must return a one-element list, not a bare string."
        )

    def test_notify_real_cli_argparse_stores_attribute_named_files(self):
        """The real cli.py's daemon notify subparser, when given two file paths,
        must NOT raise SystemExit(2) and must produce args.files=['a.py','b.py'].

        On current code: parse_args(['daemon','notify','a.py','b.py','--project','.'])
        raises SystemExit(2) because the singular 'file' positional is not variadic.

        RED: SystemExit(2) is caught and re-raised as AssertionError.
        """
        p = self._capture_real_parser()

        try:
            args = p.parse_args(["daemon", "notify", "a.py", "b.py", "--project", "."])
        except SystemExit as e:
            raise AssertionError(
                f"Real cli.py parser rejected 'daemon notify a.py b.py' with exit {e.code}. "
                "Current code: add_argument('file', ...) is singular — cannot accept 2 paths. "
                "Fix: add_argument('files', nargs='+', ...)"
            ) from e

        assert hasattr(args, "files"), (
            f"Parsed namespace has no 'files' attribute. "
            f"Namespace: {vars(args)!r}. "
            "Fix: add_argument('files', nargs='+', ...)"
        )
        assert args.files == ["a.py", "b.py"], (
            f"Expected args.files == ['a.py','b.py']; got {args.files!r}"
        )
