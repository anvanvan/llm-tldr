"""RED-phase tests for Swift language auto-detect fix in `tldr context`.

Feature: Swift language auto-detect fix for `tldr context` in mixed-extension projects.

New production code required (does NOT yet exist):
  - tldr.cli._resolve_context_languages  (module-scope function)
  - tldr.cli.NoSupportedContextLanguagesError  (module-scope exception class)
  - tldr.api.get_relevant_context_multi  (new function after get_relevant_context)

All 15 tests MUST FAIL in RED phase:
  - Group A (1-5): AttributeError/ImportError — helpers don't exist yet at module scope
  - Group B (6-8): AttributeError/ImportError — get_relevant_context_multi doesn't exist
  - Group C (9-15): assertion failures — CLI uses old single-language dispatch
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Lazy imports for new symbols (ImportError / AttributeError = correct RED)
# ---------------------------------------------------------------------------

try:
    from tldr.cli import _resolve_context_languages
except (ImportError, AttributeError):
    _resolve_context_languages = None  # type: ignore[assignment]

try:
    from tldr.cli import NoSupportedContextLanguagesError
except (ImportError, AttributeError):
    NoSupportedContextLanguagesError = None  # type: ignore[assignment]

try:
    from tldr.api import get_relevant_context_multi
except (ImportError, AttributeError):
    get_relevant_context_multi = None  # type: ignore[assignment]

# Always importable (existing)
from tldr.api import RelevantContext, SUPPORTED_CONTEXT_LANGUAGES

# Repo root for subprocess tests (cwd)
_REPO_ROOT = Path(__file__).parent.parent


# ===========================================================================
# GROUP A — _resolve_context_languages unit tests (direct import)
# ===========================================================================

class TestResolveContextLanguages:
    """Direct unit tests for _resolve_context_languages (module-scope in cli.py)."""

    def test_resolve_auto_returns_detected_supported_languages(
        self, tmp_path: Path, monkeypatch
    ):
        """auto with cached ["python", "swift"] returns ["python", "swift"] in order.

        get_cached_languages is patched to return the cached list so no real
        project detection happens. Both python and swift are in
        SUPPORTED_CONTEXT_LANGUAGES so both should appear in the result.
        """
        if _resolve_context_languages is None:
            pytest.fail(
                "_resolve_context_languages is not importable from tldr.cli — "
                "production code for Phase 2 (module-scope helper) is missing."
            )

        monkeypatch.setattr(
            "tldr.cli.get_cached_languages",
            lambda project_path: ["python", "swift"],
        )

        result = _resolve_context_languages("auto", tmp_path)

        assert result == ["python", "swift"], (
            f"Expected ['python', 'swift'] for auto with cached python+swift, "
            f"got {result!r}"
        )

    def test_resolve_all_returns_all_supported_unconditionally(
        self, tmp_path: Path, monkeypatch
    ):
        """lang_arg="all" returns sorted(SUPPORTED_CONTEXT_LANGUAGES) regardless of detection.

        Even if the project only has files in a single language, --lang all
        must return every supported context language unconditionally. The
        expected count is derived from SUPPORTED_CONTEXT_LANGUAGES rather
        than hardcoded so this test remains correct as the supported-language
        set grows (e.g. when ruby/c/elixir were added on top of the original 8).
        """
        if _resolve_context_languages is None:
            pytest.fail(
                "_resolve_context_languages is not importable from tldr.cli — "
                "production code for Phase 2 is missing."
            )

        # Patch get_cached_languages to return only one (which may or may
        # not be supported); --lang all must ignore this and return the full
        # supported set regardless.
        monkeypatch.setattr(
            "tldr.cli.get_cached_languages",
            lambda project_path: ["kotlin"],
        )

        result = _resolve_context_languages("all", tmp_path)

        expected = sorted(SUPPORTED_CONTEXT_LANGUAGES)
        assert set(result) == set(SUPPORTED_CONTEXT_LANGUAGES), (
            f"Expected set(SUPPORTED_CONTEXT_LANGUAGES) = {set(expected)!r} "
            f"for lang_arg='all', got {set(result)!r}"
        )
        assert len(result) == len(SUPPORTED_CONTEXT_LANGUAGES), (
            f"Expected exactly len(SUPPORTED_CONTEXT_LANGUAGES) "
            f"= {len(SUPPORTED_CONTEXT_LANGUAGES)} languages, "
            f"got {len(result)}: {result!r}"
        )

    def test_resolve_explicit_lang_returns_single_element_list(
        self, tmp_path: Path, monkeypatch
    ):
        """lang_arg="swift" returns ["swift"] without detection or cache lookup."""
        if _resolve_context_languages is None:
            pytest.fail(
                "_resolve_context_languages is not importable from tldr.cli — "
                "production code for Phase 2 is missing."
            )

        # Patch get_cached_languages to ensure it is NOT called (wrong result would fail)
        monkeypatch.setattr(
            "tldr.cli.get_cached_languages",
            lambda project_path: ["python"],  # should not be consulted
        )

        result = _resolve_context_languages("swift", tmp_path)

        assert result == ["swift"], (
            f"Expected ['swift'] for explicit lang_arg='swift', got {result!r}"
        )

    def test_resolve_auto_raises_when_only_unsupported_detected(
        self, tmp_path: Path, monkeypatch
    ):
        """auto with only kotlin+scala cached raises NoSupportedContextLanguagesError.

        Detection finds languages but none are in SUPPORTED_CONTEXT_LANGUAGES.
        This is the T-8 fix: no silent python fallback for this case. We use
        kotlin+scala because they remain unsupported for context (ruby and
        elixir were added to SUPPORTED_CONTEXT_LANGUAGES alongside c).
        """
        if _resolve_context_languages is None:
            pytest.fail(
                "_resolve_context_languages is not importable from tldr.cli — "
                "production code for Phase 2 is missing."
            )
        if NoSupportedContextLanguagesError is None:
            pytest.fail(
                "NoSupportedContextLanguagesError is not importable from tldr.cli — "
                "production code for Phase 1 is missing."
            )

        # Sanity guard — if a future change makes either language supported,
        # this test must be updated rather than silently passing on a
        # half-true premise.
        assert "kotlin" not in SUPPORTED_CONTEXT_LANGUAGES, (
            "kotlin became supported; pick a different unsupported language"
        )
        assert "scala" not in SUPPORTED_CONTEXT_LANGUAGES, (
            "scala became supported; pick a different unsupported language"
        )

        monkeypatch.setattr(
            "tldr.cli.get_cached_languages",
            lambda project_path: ["kotlin", "scala"],
        )

        with pytest.raises(NoSupportedContextLanguagesError) as exc_info:
            _resolve_context_languages("auto", tmp_path)

        err = exc_info.value
        assert hasattr(err, "detected"), (
            f"NoSupportedContextLanguagesError must have 'detected' attribute, got: {dir(err)}"
        )
        assert set(err.detected) == {"kotlin", "scala"}, (
            f"Expected detected=['kotlin', 'scala'], got {err.detected!r}"
        )
        assert hasattr(err, "supported"), (
            f"NoSupportedContextLanguagesError must have 'supported' attribute, got: {dir(err)}"
        )

    def test_resolve_auto_falls_back_to_python_when_no_languages_detected(
        self, tmp_path: Path, monkeypatch
    ):
        """Truly-empty project (no source files) falls back to ["python"].

        This is the consistent fallback for empty tmp_path fixtures — same as
        the existing resolve_language behavior. Note: this is distinct from the
        case where languages are detected but unsupported (which raises).
        """
        if _resolve_context_languages is None:
            pytest.fail(
                "_resolve_context_languages is not importable from tldr.cli — "
                "production code for Phase 2 is missing."
            )

        # Return None/empty from cache (no source files indexed)
        monkeypatch.setattr(
            "tldr.cli.get_cached_languages",
            lambda project_path: None,
        )
        # Also patch _detect_project_languages to return empty (no files).
        # Patch on tldr.semantic (the actual definition site) since
        # _resolve_context_languages uses a local `from .semantic import ...`,
        # so patching tldr.cli.<name> would be a silent no-op.
        monkeypatch.setattr(
            "tldr.semantic._detect_project_languages",
            lambda project_path, respect_ignore=True: [],
        )

        result = _resolve_context_languages("auto", tmp_path)

        assert result == ["python"], (
            f"Expected ['python'] fallback for empty project, got {result!r}"
        )


# ===========================================================================
# GROUP B — get_relevant_context_multi unit tests (api.py)
# ===========================================================================

class TestGetRelevantContextMulti:
    """Direct unit tests for get_relevant_context_multi (api.py)."""

    @pytest.fixture
    def mixed_project(self, tmp_path: Path) -> Path:
        """Fixture: mixed Python+Swift project with one function each."""
        src_dir = tmp_path / "Sources"
        src_dir.mkdir()
        # Swift file with compositeEmail
        (src_dir / "Email.swift").write_text(
            textwrap.dedent("""\
                func compositeEmail() -> String {
                    return "x"
                }
            """)
        )
        # Python file with make_build
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "build.py").write_text(
            textwrap.dedent("""\
                def make_build():
                    pass
            """)
        )
        return tmp_path

    def test_multi_returns_first_non_error_hit(self, mixed_project: Path):
        """Probing ["python", "swift"] for compositeEmail: Python misses, Swift hits.

        The function returns the first RelevantContext with no error.
        Since compositeEmail is a Swift function, Python probes and misses,
        then Swift hits and the result is returned.
        """
        tree_sitter_swift = pytest.importorskip(
            "tree_sitter_swift",
            reason="tree_sitter_swift not installed — cannot build Swift call graph",
        )

        if get_relevant_context_multi is None:
            pytest.fail(
                "get_relevant_context_multi is not importable from tldr.api — "
                "production code for Phase 3 is missing."
            )

        result = get_relevant_context_multi(
            project=str(mixed_project),
            entry_point="compositeEmail",
            depth=2,
            languages=["python", "swift"],
        )

        assert isinstance(result, RelevantContext), (
            f"Expected RelevantContext, got {type(result)!r}"
        )
        assert result.error is None, (
            f"Expected no error for compositeEmail (Swift hit), got: {result.error!r}"
        )

    def test_multi_returns_combined_miss_error_with_probed_list(
        self, mixed_project: Path
    ):
        """All-miss for doesNotExist: error contains 'not found in project', 'probed:', 'python', 'swift'."""
        if get_relevant_context_multi is None:
            pytest.fail(
                "get_relevant_context_multi is not importable from tldr.api — "
                "production code for Phase 3 is missing."
            )

        result = get_relevant_context_multi(
            project=str(mixed_project),
            entry_point="doesNotExist",
            depth=2,
            languages=["python", "swift"],
        )

        assert isinstance(result, RelevantContext), (
            f"Expected RelevantContext, got {type(result)!r}"
        )
        assert result.error is not None, (
            "Expected error on all-miss for doesNotExist"
        )
        assert "not found in project" in result.error, (
            f"Expected 'not found in project' in error, got: {result.error!r}"
        )
        assert "probed:" in result.error, (
            f"Expected 'probed:' in error message, got: {result.error!r}"
        )
        assert "python" in result.error, (
            f"Expected 'python' listed in probed languages, got: {result.error!r}"
        )
        assert "swift" in result.error, (
            f"Expected 'swift' listed in probed languages, got: {result.error!r}"
        )

    def test_multi_empty_languages_guard_returns_clean_error(
        self, tmp_path: Path
    ):
        """languages=[] returns RelevantContext with clean error (no malformed '(probed: )').

        The defensive guard at the top of get_relevant_context_multi must fire
        immediately for empty languages, returning a clean error message like
        "no supported languages probed".
        """
        if get_relevant_context_multi is None:
            pytest.fail(
                "get_relevant_context_multi is not importable from tldr.api — "
                "production code for Phase 3 is missing."
            )

        result = get_relevant_context_multi(
            project=str(tmp_path),
            entry_point="someFunc",
            depth=2,
            languages=[],
        )

        assert isinstance(result, RelevantContext), (
            f"Expected RelevantContext, got {type(result)!r}"
        )
        assert result.error is not None, (
            "Expected error for empty languages list"
        )
        # Must NOT produce malformed "(probed: )" output
        assert "(probed: )" not in result.error, (
            f"Got malformed '(probed: )' in error — guard missing. Error: {result.error!r}"
        )
        # Must contain a meaningful message about no languages being probed
        assert "no supported languages probed" in result.error or "no languages" in result.error.lower(), (
            f"Expected clean 'no supported languages probed' error, got: {result.error!r}"
        )


# ===========================================================================
# GROUP C — CLI integration tests (subprocess)
# ===========================================================================

def _run_context(
    tmp_path: Path,
    entry: str,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run `tldr context <entry> --project <tmp_path> --depth 2` via subprocess."""
    cmd = [
        sys.executable, "-m", "tldr.cli",
        "context", entry,
        "--project", str(tmp_path),
        "--depth", "2",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


@pytest.fixture
def mixed_fixture(tmp_path: Path) -> Path:
    """Mixed Python+Swift project fixture used by most CLI tests."""
    src_dir = tmp_path / "Sources"
    src_dir.mkdir()
    (src_dir / "Foo.swift").write_text(
        textwrap.dedent("""\
            func compositeEmail() -> String {
                return "x"
            }
        """)
    )
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "build.py").write_text(
        textwrap.dedent("""\
            def make_build():
                pass
        """)
    )
    return tmp_path


def test_context_cli_swift_auto_detect_in_mixed_project(mixed_fixture: Path):
    """DoD 1: tldr context compositeEmail in mixed project (no --lang) exits 0 and shows compositeEmail.

    Current code: resolve_language returns 'python' (first cached hit), then checks
    swift is not in SUPPORTED_CONTEXT_LANGUAGES (swift IS supported, but only python
    is resolved by the old single-pick), so compositeEmail is not found → exit 1.
    After fix: _resolve_context_languages returns ["python","swift"], multi-probe
    finds compositeEmail in Swift → exit 0.
    """
    pytest.importorskip(
        "tree_sitter_swift",
        reason="tree_sitter_swift not installed — cannot run Swift CLI integration test",
    )

    result = _run_context(mixed_fixture, "compositeEmail")

    assert result.returncode == 0, (
        f"Expected exit 0 for compositeEmail in mixed project (no --lang). "
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "compositeEmail" in result.stdout, (
        f"Expected 'compositeEmail' in stdout, got: {result.stdout!r}"
    )


def test_context_cli_python_still_resolves_in_mixed_project(mixed_fixture: Path):
    """DoD 2: tldr context make_build in mixed project (no --lang) exits 0 and shows make_build.

    Python should resolve first (it appears first in the languages list).
    After fix: _resolve_context_languages returns ["python","swift"]; Python hit
    → exit 0, stdout contains make_build.

    RED contract: also verifies that --lang all is an accepted choice (new argparse
    addition required by the fix). Currently 'all' is not in --lang choices → exit 2.
    After fix: exit 0, stdout contains make_build.
    """
    # Primary assertion (may already pass): no --lang probes auto-detected languages
    result_auto = _run_context(mixed_fixture, "make_build")
    # Secondary assertion (currently FAILS): --lang all must also be accepted and resolve
    result_all = _run_context(mixed_fixture, "make_build", extra_args=["--lang", "all"])

    assert result_all.returncode == 0, (
        f"Expected exit 0 for make_build with --lang all (after argparse fix adds 'all' to choices). "
        f"Currently 'all' is not a valid --lang choice. "
        f"stdout: {result_all.stdout!r}\nstderr: {result_all.stderr!r}"
    )
    assert "make_build" in result_all.stdout, (
        f"Expected 'make_build' in stdout with --lang all, got: {result_all.stdout!r}"
    )
    # Also verify the auto-detect path works correctly (must not regress)
    assert result_auto.returncode == 0, (
        f"Expected exit 0 for make_build in mixed project (no --lang). "
        f"stdout: {result_auto.stdout!r}\nstderr: {result_auto.stderr!r}"
    )
    assert "make_build" in result_auto.stdout, (
        f"Expected 'make_build' in stdout (no --lang), got: {result_auto.stdout!r}"
    )


def test_context_cli_unknown_symbol_lists_probed_languages(mixed_fixture: Path):
    """DoD 3: tldr context doesNotExistAnywhere exits 1 and stderr contains 'probed:' + both languages.

    After fix: multi-probe exhausts python+swift, returns error with probed list.
    Dispatcher prints to stderr + exits 1. stderr must name both languages.
    """
    result = _run_context(mixed_fixture, "doesNotExistAnywhere")

    assert result.returncode == 1, (
        f"Expected exit 1 for unknown symbol in mixed project. "
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "probed:" in result.stderr, (
        f"Expected 'probed:' in stderr, got: {result.stderr!r}"
    )
    assert "python" in result.stderr, (
        f"Expected 'python' in probed list in stderr, got: {result.stderr!r}"
    )
    assert "swift" in result.stderr, (
        f"Expected 'swift' in probed list in stderr, got: {result.stderr!r}"
    )


def test_context_cli_swift_only_project_resolves_without_lang(tmp_path: Path):
    """DoD 4: Swift-only project resolves without --lang (no --lang = auto-detect).

    tmp_path has ONLY a .swift file with medianIndex(). With auto-detect,
    _resolve_context_languages returns ["swift"]; Swift build finds medianIndex.

    RED contract: also verifies that when the symbol DOES NOT exist, the error
    message contains "probed:" listing the languages that were probed.
    The current code does NOT include "probed:" in the error — it says
    "Function 'X' not found in project." (no probed list).
    After the fix: error is "Function 'X' not found in project (probed: swift)".
    """
    pytest.importorskip(
        "tree_sitter_swift",
        reason="tree_sitter_swift not installed — cannot run Swift CLI integration test",
    )

    src_dir = tmp_path / "Sources"
    src_dir.mkdir()
    (src_dir / "Util.swift").write_text(
        textwrap.dedent("""\
            func medianIndex() -> Int {
                return 1
            }
        """)
    )

    # Verify that an UNKNOWN symbol in a Swift-only project lists probed languages
    # in the error. This requires get_relevant_context_multi in the dispatch.
    result_unknown = _run_context(tmp_path, "doesNotExistInSwift")

    assert result_unknown.returncode == 1, (
        f"Expected exit 1 for unknown symbol in Swift-only project. "
        f"stdout: {result_unknown.stdout!r}\nstderr: {result_unknown.stderr!r}"
    )
    assert "probed:" in result_unknown.stderr, (
        f"Expected 'probed:' in stderr for unknown symbol in Swift-only project. "
        f"Current code says 'not found in project' without a probed list. "
        f"Got: {result_unknown.stderr!r}"
    )
    assert "swift" in result_unknown.stderr, (
        f"Expected 'swift' in probed list in stderr, got: {result_unknown.stderr!r}"
    )


def test_context_cli_explicit_lang_swift_still_works(mixed_fixture: Path):
    """DoD 5: --lang swift explicit still resolves compositeEmail correctly.

    Regression: explicit --lang swift must still work after refactor.
    _resolve_context_languages("swift", ...) returns ["swift"]; single probe.

    RED contract: also verifies that unknown symbol with --lang swift shows "probed: swift"
    in the error output. The new dispatch path (get_relevant_context_multi) must emit
    this even for single-language explicit probes. Current code does NOT include "probed:"
    in the error message → assertion fails until the new dispatch is wired.
    """
    pytest.importorskip(
        "tree_sitter_swift",
        reason="tree_sitter_swift not installed — cannot run Swift CLI integration test",
    )

    # Verify success path still works (regression guard)
    result_ok = _run_context(mixed_fixture, "compositeEmail", extra_args=["--lang", "swift"])
    assert result_ok.returncode == 0, (
        f"Expected exit 0 for compositeEmail with --lang swift. "
        f"stdout: {result_ok.stdout!r}\nstderr: {result_ok.stderr!r}"
    )
    assert "compositeEmail" in result_ok.stdout, (
        f"Expected 'compositeEmail' in stdout with explicit --lang swift, got: {result_ok.stdout!r}"
    )

    # Verify error path shows "probed: swift" (NEW BEHAVIOR required by dispatch refactor)
    result_miss = _run_context(
        mixed_fixture, "doesNotExistAnywhere", extra_args=["--lang", "swift"]
    )
    assert result_miss.returncode == 1, (
        f"Expected exit 1 for unknown symbol with --lang swift, "
        f"got: {result_miss.returncode}"
    )
    assert "probed:" in result_miss.stderr, (
        f"Expected 'probed:' in stderr for miss with --lang swift. "
        f"Current code: 'Function ... not found in project.' (no probed list). "
        f"After fix: 'Function ... not found in project (probed: swift)'. "
        f"Got: {result_miss.stderr!r}"
    )


def test_context_cli_lang_all_accepted_and_probes(mixed_fixture: Path):
    """DoD 6: --lang all is accepted by argparse and resolves both Swift+Python symbols.

    After fix: 'all' added to --lang choices; _resolve_context_languages("all",...)
    returns every language in SUPPORTED_CONTEXT_LANGUAGES; first hit wins
    for each symbol. This tests two calls: make_build (Python hit) and
    compositeEmail (Swift hit).
    """
    pytest.importorskip(
        "tree_sitter_swift",
        reason="tree_sitter_swift not installed — cannot run Swift CLI integration test",
    )

    # Python symbol with --lang all
    result_py = _run_context(mixed_fixture, "make_build", extra_args=["--lang", "all"])
    assert result_py.returncode == 0, (
        f"Expected exit 0 for make_build with --lang all. "
        f"stdout: {result_py.stdout!r}\nstderr: {result_py.stderr!r}"
    )
    assert "make_build" in result_py.stdout, (
        f"Expected 'make_build' in stdout for --lang all, got: {result_py.stdout!r}"
    )

    # Swift symbol with --lang all
    result_sw = _run_context(mixed_fixture, "compositeEmail", extra_args=["--lang", "all"])
    assert result_sw.returncode == 0, (
        f"Expected exit 0 for compositeEmail with --lang all. "
        f"stdout: {result_sw.stdout!r}\nstderr: {result_sw.stderr!r}"
    )
    assert "compositeEmail" in result_sw.stdout, (
        f"Expected 'compositeEmail' in stdout for --lang all, got: {result_sw.stdout!r}"
    )


def test_context_cli_unsupported_only_project_exits_1_with_diagnostic(tmp_path: Path):
    """T-8: Kotlin-only project exits 1 with stderr diagnostic about no supported context languages.

    _resolve_context_languages raises NoSupportedContextLanguagesError; dispatcher
    catches it and prints a diagnostic like:
      "Error: no supported context languages in '<project>' (found: kotlin; supported: ...)"
    then exits 1. The stderr must contain 'no supported context languages' (or similar).

    Ruby was the original probe language here, but ruby joined
    SUPPORTED_CONTEXT_LANGUAGES (alongside c and elixir) once those builders
    landed. Kotlin is genuinely unsupported, so it still exercises the
    "no supported context languages" diagnostic path.
    """
    # Sanity guard — keep this test exercising the unsupported path.
    from tldr.api import SUPPORTED_CONTEXT_LANGUAGES as _SUPPORTED
    assert "kotlin" not in _SUPPORTED, (
        "kotlin became supported; pick a different unsupported language"
    )

    # Create ONLY a Kotlin file (truly unsupported for context)
    (tmp_path / "App.kt").write_text("fun hello() {}\n")

    result = _run_context(tmp_path, "hello")

    assert result.returncode == 1, (
        f"Expected exit 1 for Ruby-only project (no supported context languages). "
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    # Check for meaningful diagnostic — either the exact phrase or something close
    stderr_lower = result.stderr.lower()
    assert (
        "no supported context languages" in stderr_lower
        or "no supported context language" in stderr_lower
        or ("found:" in result.stderr and "supported" in result.stderr)
    ), (
        f"Expected stderr to contain diagnostic about no supported context languages. "
        f"Got: {result.stderr!r}"
    )


def test_other_commands_lang_auto_unchanged(mixed_fixture: Path):
    """DoD 7: `tldr calls/impact/dead/arch` still accept `--lang auto` and resolve to a single language.

    These commands intentionally still use the legacy single-language dispatch
    (resolve_language picks the first auto-detected language). They should NOT
    have been switched to the new multi-probe path that `tldr context` uses.

    This test runs each of the four commands with --lang auto on the mixed
    fixture and asserts:
      - argparse accepts --lang auto (returncode != 2)
      - the command doesn't crash on the language selection
      - no NoSupportedContextLanguagesError diagnostic appears (mixed_fixture
        has Python which is supported)

    A passing/failing exit code is allowed (these commands may legitimately
    miss the symbol on an empty graph); the contract is purely that --lang auto
    behaves the same as before (i.e., wasn't accidentally regressed when
    `tldr context` was switched to multi-probe).
    """
    base_cmd_prefix = [
        sys.executable, "-m", "tldr.cli",
    ]
    proj = str(mixed_fixture)
    lang_auto = ["--lang", "auto"]

    # These four commands take the project path as a positional argument
    # (not via --project) and they each accept --lang auto.
    invocations = [
        ["calls", proj, *lang_auto],
        ["impact", "make_build", proj, *lang_auto],
        ["dead", proj, *lang_auto],
        ["arch", proj, *lang_auto],
    ]

    for sub in invocations:
        result = subprocess.run(
            base_cmd_prefix + sub,
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        # argparse must accept --lang auto (returncode 2 == argparse rejection)
        assert result.returncode != 2, (
            f"argparse rejected `tldr {' '.join(sub)}` — "
            f"--lang auto must still be accepted for non-context commands. "
            f"stderr: {result.stderr!r}"
        )
        # mixed_fixture has Python (supported); should NOT raise the multi-probe
        # "no supported context languages" diagnostic from the context command.
        assert "no supported context languages" not in result.stderr.lower(), (
            f"`tldr {' '.join(sub)}` unexpectedly hit the context-command "
            f"NoSupportedContextLanguagesError path. stderr: {result.stderr!r}"
        )
