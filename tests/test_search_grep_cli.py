"""RED tests: grep-canonical `tldr search` CLI surface + api.search extensions.

Covers architecture behaviors:
  1. -i / --ignore-case
  2. multi-path positionals with path-order merge and per-path-arg file prefixing
     (file-typed path arg -> the path arg itself; NO basename doubling)
  3. single-path output byte-identity (control invariant)
  4. BRE-escape normalization end-to-end (\\| alternation, \\(..\\) grouping,
     \\d -> ASCII [0-9])
  6. compile fallback to the original pattern (API level)
  7. --include (CLI half)
  8. removed flags: --ext and --max exit 2
  9. -m / --max-count caps TOTAL results across paths; -m 0 unlimited
 10. --exclude-dir (glob, any depth, composes with --no-ignore)
 16. the zero-hit '\\|' stderr hint no longer exists (positive replacement for
     tests/test_regression_search_shell_pipe_hint.py, which GREEN deletes)

Conventions (architecture Testability Plan): subprocess `python -m tldr.cli`,
module-scoped tmp fixture trees with a `.git` anchor and a minimal `.tldrignore`
(".tldr/", ".git/") so daemon-written caches can never pollute results,
returncode asserted first, control tests paired with behavior tests.

Control tests pin TODAY's behavior that must survive the migration — they pass
at RED time by design and guard the GREEN step. Every non-control test fails
pre-implementation (argparse exit 2 for new flags / TypeError for new kwargs /
zero hits for normalization).
"""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

_A_PY = '''\
def alpha_one():
    return 1


APPLE_MARKER = 2
mp_token_a = 1
inc_token_a = 1
value7 = 7
result = call(foo)
xxfooyy = 1
arabic_digit = "num٣"
'''

# .tldrignore replaces the DEFAULT_TEMPLATE entirely, giving the fixture full
# deterministic control: only .tldr/ (daemon cache) and .git/ are ignored on
# the default-ignore path; vendor/, testdata/, extlib/ stay searchable.
_TLDRIGNORE = ".tldr/\n.git/\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def grep_base(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("grepcli")

    proj = base / "proj1"
    (proj / ".git").mkdir(parents=True)
    _write(proj / ".tldrignore", _TLDRIGNORE)
    _write(proj / "a.py", _A_PY)
    _write(proj / "b.ts", "inc_token_ts = 1\n")
    _write(proj / "sub" / "c.py", "mp_token_sub = 1\ninc_token_sub = 1\n")
    _write(proj / "keep.py", "exdir_token_keep = 1\n")
    _write(proj / "vendor" / "v.py", "exdir_token_vendor = 1\n")
    _write(proj / "deep" / "vendor" / "n.py", "exdir_token_deepvendor = 1\n")
    _write(proj / "testdata" / "t.py", "exdir_token_testdata = 1\n")
    _write(proj / "extlib" / "e.py", "exdir_token_extlib = 1\n")
    _write(
        proj / "bulk.py",
        "\n".join(f"bulk_token_{i:03d} = {i}" for i in range(120)) + "\n",
    )

    dirb = base / "dirB"
    (dirb / ".git").mkdir(parents=True)
    _write(dirb / ".tldrignore", _TLDRIGNORE)
    _write(dirb / "d.py", "mp_token_b = 1\n")

    return base


def _run(args: list[str], cwd) -> subprocess.CompletedProcess:
    # conftest.py sets SIGCHLD=SIG_IGN; Popen.wait() then treats ECHILD as
    # exit 0, masking any non-zero returncode.  Save/restore SIG_DFL so that
    # exit-code assertions are reliable regardless of the test runner's state.
    _old = signal.getsignal(signal.SIGCHLD)
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        return subprocess.run(
            [sys.executable, "-m", "tldr.cli", *args],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=120,
        )
    finally:
        signal.signal(signal.SIGCHLD, _old)


def _hits(result: subprocess.CompletedProcess) -> list[dict]:
    assert result.returncode == 0, (
        f"Expected exit 0, got {result.returncode}. stderr: {result.stderr!r}"
    )
    return json.loads(result.stdout)


def _files(result: subprocess.CompletedProcess) -> list[str]:
    return [h["file"] for h in _hits(result)]


class TestIgnoreCaseFlag:
    """Behavior 1: -i/--ignore-case matches case-insensitively."""

    def test_short_i_flag_matches_case_insensitively(self, grep_base):
        result = _run(["search", "apple_marker", "proj1", "-i"], cwd=grep_base)
        files = _files(result)
        assert "a.py" in files, f"-i should match APPLE_MARKER, got {files}"

    def test_long_ignore_case_flag(self, grep_base):
        result = _run(
            ["search", "apple_marker", "proj1", "--ignore-case"], cwd=grep_base
        )
        files = _files(result)
        assert "a.py" in files, f"--ignore-case should match APPLE_MARKER, got {files}"

    def test_without_flag_stays_case_sensitive(self, grep_base):
        # Control invariant: default stays case-sensitive (passes today).
        result = _run(["search", "apple_marker", "proj1"], cwd=grep_base)
        assert _hits(result) == []


class TestMultiPath:
    """Behavior 2 + edge cases: multi-path positionals."""

    def test_two_dir_paths_merge_in_path_order_with_prefixed_files(self, grep_base):
        result = _run(["search", "mp_token", "proj1", "dirB"], cwd=grep_base)
        files = _files(result)
        assert set(files) == {"proj1/a.py", "proj1/sub/c.py", "dirB/d.py"}, files
        # Merge order = CLI path order: every proj1 hit before any dirB hit.
        assert files[-1] == "dirB/d.py", f"dirB hit must come last, got {files}"

    def test_file_and_dir_mix_no_basename_doubling(self, grep_base):
        # A-1: api.search single-file mode returns the BASENAME — the dispatch
        # must use the path argument itself for a file-typed arg, never join.
        result = _run(["search", "mp_token", "proj1/a.py", "dirB"], cwd=grep_base)
        files = _files(result)
        assert "proj1/a.py" in files, files
        assert "proj1/a.py/a.py" not in files, (
            f"basename doubling detected for file-typed path arg: {files}"
        )
        assert "dirB/d.py" in files, files
        assert set(files) == {"proj1/a.py", "dirB/d.py"}, files

    def test_missing_path_among_several_exits_1_naming_it(self, grep_base):
        result = _run(
            ["search", "mp_token", "proj1", "no_such_dir"], cwd=grep_base
        )
        assert result.returncode == 1, (
            f"Expected exit 1 for missing path, got {result.returncode}. "
            f"stderr: {result.stderr!r}"
        )
        assert "no_such_dir" in result.stderr

    def test_single_path_output_shape_unchanged(self, grep_base):
        # Control invariant (behavior 3): single-path output is byte-identical
        # to today — relative 'file' fields, no path-arg prefixing, exactly the
        # {file, line, content} keys, nothing on stderr. Passes today.
        result = _run(["search", "mp_token", "proj1"], cwd=grep_base)
        hits = _hits(result)
        assert {h["file"] for h in hits} == {"a.py", "sub/c.py"}
        for h in hits:
            assert set(h.keys()) == {"file", "line", "content"}, h
        assert not result.stderr.strip(), result.stderr


class TestEscapeNormalizationEndToEnd:
    """Behaviors 4 + 16 (CLI), 5 control: normalization at the compile site."""

    def test_backslash_pipe_alternation_now_hits_with_no_hint(self, grep_base):
        # Positive replacement for the deleted hint regression test: the exact
        # grep-ism that used to 0-hit-and-hint now simply works (behaviors 4+16).
        result = _run(
            ["search", r"alpha_one\|zzz_never_match", "proj1"], cwd=grep_base
        )
        files = _files(result)
        assert "a.py" in files, (
            f"normalized \\| alternation should match alpha_one, got {files}"
        )
        assert not result.stderr.strip(), (
            f"the zero-hit \\| hint branch must be gone, got: {result.stderr!r}"
        )

    def test_bare_pipe_with_hits_still_works_without_hint(self, grep_base):
        # Control invariant (cloned from the deleted test): bare-| alternation
        # keeps working with empty stderr. Passes today.
        result = _run(
            ["search", "alpha_one|zzz_never_match", "proj1"], cwd=grep_base
        )
        assert "a.py" in _files(result)
        assert not result.stderr.strip(), result.stderr

    def test_backslash_parens_group_with_alternation(self, grep_base):
        # \(foo\|bar\) must mean a GROUP with alternation, not the literal
        # text '(foo|bar)' — observable only via the alternation inside.
        result = _run(["search", r"xx\(foo\|bar\)yy", "proj1"], cwd=grep_base)
        files = _files(result)
        assert "a.py" in files, (
            f"normalized group+alternation should match 'xxfooyy', got {files}"
        )

    def test_backslash_d_is_ascii_only_after_normalization(self, grep_base):
        # \d -> [0-9] (ASCII). Python's \d matches the Arabic-Indic digit in
        # the fixture, so a pre-normalization implementation returns 1 hit here
        # — this pins that normalization really happened at the compile site.
        result = _run(["search", r"num\d", "proj1"], cwd=grep_base)
        assert _hits(result) == []

    def test_backslash_d_still_matches_ascii_digit(self, grep_base):
        # Control: \d (-> [0-9]) keeps matching plain ASCII digits.
        result = _run(["search", r"value\d", "proj1"], cwd=grep_base)
        assert "a.py" in _files(result)

    def test_class_wrapped_backslash_d_passes_through(self, grep_base):
        # Control (behavior 5 end-to-end): [\d] is untouched by normalization
        # and still matches a (unicode) digit. A corrupting rewrite to
        # [[0-9]] would return zero hits.
        result = _run(["search", r"num[\d]", "proj1"], cwd=grep_base)
        assert "a.py" in _files(result)


class TestApiSearchCompileSite:
    """Behaviors 1/6/10 at the API level (architecture build step 2)."""

    def test_api_search_ignore_case_kwarg(self, grep_base):
        from tldr.api import search  # lazy

        hits = search("apple_marker", str(grep_base / "proj1"), ignore_case=True)
        assert [h["file"] for h in hits] == ["a.py"]

    def test_api_search_exclude_dirs_kwarg(self, grep_base):
        from tldr.api import search  # lazy

        hits = search(
            "exdir_token", str(grep_base / "proj1"), exclude_dirs=["extlib"]
        )
        files = {h["file"] for h in hits}
        assert "extlib/e.py" not in files, files
        assert "keep.py" in files, files

    def test_api_search_exclude_dirs_glob(self, grep_base):
        from tldr.api import search  # lazy

        hits = search(
            "exdir_token", str(grep_base / "proj1"), exclude_dirs=["test*"]
        )
        files = {h["file"] for h in hits}
        assert "testdata/t.py" not in files, files
        assert "keep.py" in files, files

    def test_api_search_normalizes_at_compile_site(self, grep_base):
        from tldr.api import search  # lazy

        hits = search(r"alpha_one\|zzz_never_match", str(grep_base / "proj1"))
        assert any(h["file"] == "a.py" for h in hits), hits

    def test_api_search_falls_back_to_original_when_normalized_invalid(
        self, grep_base
    ):
        # Control invariant (behavior 6): '\(foo' normalizes to the unbalanced
        # '(foo' which fails to compile -> fall back to the original pattern,
        # which matches the literal text 'call(foo'. Passes today (no
        # normalization yet); post-GREEN it pins the fallback requirement —
        # a normalize-without-fallback implementation raises re.error here.
        from tldr.api import search  # lazy

        hits = search(r"call\(foo", str(grep_base / "proj1"))
        assert any(h["file"] == "a.py" for h in hits), hits

    def test_api_search_propagates_error_when_both_compiles_fail(self, grep_base):
        # Control invariant: an unbalanced pattern with no escapes to rewrite
        # fails both compiles -> today's re.error propagates.
        import re

        from tldr.api import search  # lazy

        with pytest.raises(re.error):
            search("(foo", str(grep_base / "proj1"))


class TestIncludeFlag:
    """Behavior 7 (CLI half): --include replaces --ext."""

    def test_include_dot_form_filters_to_extension(self, grep_base):
        result = _run(
            ["search", "inc_token", "proj1", "--include", ".py"], cwd=grep_base
        )
        assert set(_files(result)) == {"a.py", "sub/c.py"}

    def test_include_star_form(self, grep_base):
        result = _run(
            ["search", "inc_token", "proj1", "--include", "*.ts"], cwd=grep_base
        )
        assert set(_files(result)) == {"b.ts"}

    def test_include_bare_form(self, grep_base):
        result = _run(
            ["search", "inc_token", "proj1", "--include", "py"], cwd=grep_base
        )
        assert set(_files(result)) == {"a.py", "sub/c.py"}

    def test_repeated_include_values_union(self, grep_base):
        result = _run(
            [
                "search", "inc_token", "proj1",
                "--include", ".py", "--include", ".ts",
            ],
            cwd=grep_base,
        )
        assert set(_files(result)) == {"a.py", "sub/c.py", "b.ts"}

    def test_unsupported_include_glob_exits_2_with_actionable_message(
        self, grep_base
    ):
        result = _run(
            ["search", "inc_token", "proj1", "--include", "foo*.py"],
            cwd=grep_base,
        )
        assert result.returncode == 2, (
            f"Expected exit 2, got {result.returncode}. stderr: {result.stderr!r}"
        )
        # Must be the dispatch's own message, not argparse 'unrecognized
        # arguments' (which is what the flag produces pre-implementation).
        assert "unsupported" in result.stderr.lower(), result.stderr

    def test_no_include_searches_all_extensions(self, grep_base):
        # Control invariant: without a filter, .py and .ts both hit.
        result = _run(["search", "inc_token", "proj1"], cwd=grep_base)
        assert set(_files(result)) == {"a.py", "sub/c.py", "b.ts"}


class TestRemovedFlags:
    """Behavior 8: --ext and --max are gone from search (exit 2)."""

    def test_ext_flag_removed_exits_2(self, grep_base):
        result = _run(
            ["search", "inc_token", "proj1", "--ext", ".py"], cwd=grep_base
        )
        assert result.returncode == 2, (
            f"--ext must be an argparse error, got exit {result.returncode}. "
            f"stdout: {result.stdout[:200]!r}"
        )

    def test_max_flag_removed_exits_2(self, grep_base):
        # --max becomes an ambiguous abbreviation of --max-count/--max-files.
        # Assert the exit code only (architecture: not the message wording).
        result = _run(
            ["search", "inc_token", "proj1", "--max", "50"], cwd=grep_base
        )
        assert result.returncode == 2, (
            f"--max must be an argparse error, got exit {result.returncode}. "
            f"stdout: {result.stdout[:200]!r}"
        )


class TestMaxCount:
    """Behavior 9: -m/--max-count caps TOTAL results across all paths."""

    def test_m_caps_total_results(self, grep_base):
        result = _run(["search", "bulk_token", "proj1", "-m", "3"], cwd=grep_base)
        assert len(_hits(result)) == 3

    def test_max_count_long_form(self, grep_base):
        result = _run(
            ["search", "bulk_token", "proj1", "--max-count", "5"], cwd=grep_base
        )
        assert len(_hits(result)) == 5

    def test_m_zero_is_unlimited(self, grep_base):
        # 120 fixture matches > the default cap of 100 — proves 0 lifts it.
        result = _run(["search", "bulk_token", "proj1", "-m", "0"], cwd=grep_base)
        assert len(_hits(result)) == 120

    def test_default_cap_stays_100(self, grep_base):
        # Control invariant: no flag -> 100 results (today's --max default).
        result = _run(["search", "bulk_token", "proj1"], cwd=grep_base)
        assert len(_hits(result)) == 100

    def test_budget_spans_paths(self, grep_base):
        # proj1 has exactly 2 mp_token hits; -m 2 exhausts the budget before
        # dirB, so dirB must contribute nothing.
        result = _run(
            ["search", "mp_token", "proj1", "dirB", "-m", "2"], cwd=grep_base
        )
        assert set(_files(result)) == {"proj1/a.py", "proj1/sub/c.py"}


class TestExcludeDir:
    """Behavior 10: --exclude-dir skips matching dir components at any depth."""

    def test_exclude_dir_skips_named_dir_at_any_depth(self, grep_base):
        result = _run(
            ["search", "exdir_token", "proj1", "--exclude-dir", "vendor"],
            cwd=grep_base,
        )
        files = set(_files(result))
        assert "vendor/v.py" not in files, files
        assert "deep/vendor/n.py" not in files, (
            f"nested vendor dir must also be excluded: {files}"
        )
        assert files == {"keep.py", "testdata/t.py", "extlib/e.py"}, files

    def test_exclude_dir_glob_form(self, grep_base):
        result = _run(
            ["search", "exdir_token", "proj1", "--exclude-dir", "test*"],
            cwd=grep_base,
        )
        files = set(_files(result))
        assert "testdata/t.py" not in files, files
        assert "vendor/v.py" in files, files

    def test_exclude_dir_repeatable(self, grep_base):
        result = _run(
            [
                "search", "exdir_token", "proj1",
                "--exclude-dir", "vendor", "--exclude-dir", "extlib",
            ],
            cwd=grep_base,
        )
        assert set(_files(result)) == {"keep.py", "testdata/t.py"}

    def test_all_dirs_searched_without_flag(self, grep_base):
        # Control invariant: the fixture's .tldrignore only hides .tldr/.git,
        # so without --exclude-dir every exdir_token file hits. Passes today
        # and proves the exclusion tests are not vacuous.
        result = _run(["search", "exdir_token", "proj1"], cwd=grep_base)
        assert set(_files(result)) == {
            "keep.py",
            "vendor/v.py",
            "deep/vendor/n.py",
            "testdata/t.py",
            "extlib/e.py",
        }

    def test_exclude_dir_composes_with_no_ignore(self, grep_base):
        # --no-ignore drops the IgnoreSpec (api falls back to SKIP_DIRS, which
        # hides vendor/) — --exclude-dir is independent and must still work.
        result = _run(
            [
                "--no-ignore", "search", "exdir_token", "proj1",
                "--exclude-dir", "extlib",
            ],
            cwd=grep_base,
        )
        files = set(_files(result))
        assert "extlib/e.py" not in files, files
        assert files == {"keep.py", "testdata/t.py"}, files
