"""RED tests: grep/BRE escape normalization + --include glob mapping (unit half).

Covers architecture behaviors:
  4. ``\\|`` -> alternation, ``\\(``/``\\)`` -> group, ``\\d`` -> ``[0-9]``
  5. escapes inside ``[...]`` classes pass through; ``\\\\`` is the literal
     escape hatch; same-meaning escapes (``\\.``, ``\\b``, ``\\w``, ...) untouched
  7. (unit half) ``--include`` values map onto the extension filter; unsupported
     glob shapes raise ValueError

Component under test (architecture sections 1 and 5):
  - ``tldr.api.normalize_grep_pattern`` (new pure function)
  - ``tldr.cli._includes_to_extensions`` (new module-level pure function)

Imports of the not-yet-implemented symbols are deliberately LAZY (inside the
helpers below) so collection never errors — pre-implementation each test FAILS
with ImportError, which is the correct RED signal.
"""

import re

import pytest


def _normalize(pattern: str) -> str:
    from tldr.api import normalize_grep_pattern  # lazy: RED = ImportError

    return normalize_grep_pattern(pattern)


def _includes(includes):
    from tldr.cli import _includes_to_extensions  # lazy: RED = ImportError

    return _includes_to_extensions(includes)


class TestNormalizeGrepPattern:
    """normalize_grep_pattern: BRE-habit escapes -> the ERE the user meant."""

    # --- behavior 4: rewrites outside a class -----------------------------

    def test_backslash_pipe_becomes_alternation(self):
        assert _normalize(r"foo\|bar") == "foo|bar"

    def test_backslash_parens_become_group(self):
        assert _normalize(r"xx\(foo\|bar\)yy") == "xx(foo|bar)yy"

    def test_backslash_d_becomes_ascii_digit_class(self):
        assert _normalize(r"value\d") == "value[0-9]"

    def test_normalized_alternation_compiles_and_matches(self):
        # Functional check: the rewritten pattern means alternation, not the
        # literal three-char sequence 'foo|bar'.
        compiled = re.compile(_normalize(r"foo\|bar"))
        assert compiled.search("only bar here")
        assert compiled.search("only foo here")
        assert not compiled.search("neither one")

    # --- behavior 5: bracket-class passthrough ----------------------------

    def test_escape_inside_class_passes_through(self):
        # [\d] must NOT become [[0-9]]
        assert _normalize(r"[\d]") == r"[\d]"

    def test_class_wrapped_digit_still_matches_a_digit(self):
        assert re.compile(_normalize(r"num[\d]")).search("num3")

    def test_literal_pipe_class_passes_through(self):
        assert _normalize(r"[|]") == "[|]"

    def test_class_honors_leading_negation_and_literal_first_bracket(self):
        # A literal ']' first in the class (with or without '^') must not be
        # treated as the class terminator — everything stays verbatim.
        assert _normalize(r"[]\d]") == r"[]\d]"
        assert _normalize(r"[^]\d]") == r"[^]\d]"

    def test_rewriting_resumes_after_class_closes(self):
        assert _normalize(r"[\d]\d") == r"[\d][0-9]"

    def test_escaped_open_bracket_does_not_open_a_class(self):
        # \[ is an escape pair (verbatim), so the following \d is OUTSIDE any
        # class and must be rewritten.
        assert _normalize(r"\[\d") == r"\[[0-9]"

    # --- behavior 5: escaped-backslash literal escape hatch ---------------

    def test_escaped_backslash_pair_preserved_as_literal_hatch(self):
        # \\d means "literal backslash, then d" — must NOT become \\[0-9]
        assert _normalize(r"\\d") == r"\\d"
        # and it still matches the literal two-char text '\d'
        assert re.compile(_normalize(r"\\d")).search(r"x\d y")

    def test_escaped_backslash_then_escaped_pipe(self):
        # chars: \ \ \ |  ->  '\\' kept as a pair, then '\|' -> '|'
        assert _normalize("\\\\\\|") == "\\\\|"

    def test_double_backslash_pipe_keeps_bare_alternation(self):
        # chars: \ \ |  ->  literal backslash, then alternation (unchanged text)
        assert _normalize(r"\\|") == r"\\|"

    # --- behavior 5: same-meaning escapes pass through ---------------------

    @pytest.mark.parametrize("esc", [r"\.", r"\b", r"\w", r"\s", r"\+"])
    def test_same_meaning_escapes_pass_through(self, esc):
        pattern = f"a{esc}z"
        assert _normalize(pattern) == pattern

    def test_trailing_lone_backslash_emitted_verbatim(self):
        assert _normalize("foo" + "\\") == "foo" + "\\"

    def test_plain_ere_pattern_unchanged(self):
        pattern = r"foo|bar(baz)[0-9]+\.py"
        assert _normalize(pattern) == pattern


class TestIncludesToExtensions:
    """_includes_to_extensions: --include globs -> extension-filter set."""

    # --- behavior 7 (unit half): supported shapes --------------------------

    def test_star_dot_form(self):
        assert _includes(["*.py"]) == {".py"}

    def test_dot_form(self):
        assert _includes([".py"]) == {".py"}

    def test_bare_form(self):
        assert _includes(["py"]) == {".py"}

    def test_repeated_values_union(self):
        assert _includes(["*.py", ".ts", "go"]) == {".py", ".ts", ".go"}

    def test_case_is_preserved(self):
        # Suffix matching stays exact (as today) — no case folding.
        assert _includes(["*.PY"]) == {".PY"}

    def test_none_and_empty_mean_no_filter(self):
        assert _includes(None) is None
        assert _includes([]) is None

    # --- behavior 7 (unit half): unsupported shapes ------------------------

    def test_wildcard_not_in_leading_position_raises_value_error(self):
        with pytest.raises(ValueError, match="--include"):
            _includes(["foo*.py"])

    def test_path_separator_raises_value_error(self):
        with pytest.raises(ValueError, match="--include"):
            _includes(["src/*.py"])
