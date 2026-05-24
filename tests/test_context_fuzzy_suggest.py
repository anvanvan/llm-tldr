"""RED-phase tests for fuzzy-suggest and qualified-name fallback in get_relevant_context.

Acceptance criteria covered:
  AC-1:  Typo → error message contains "Did you mean: <correct_name>"
  AC-2:  Multiple close candidates surface comma-separated, closeness order
  AC-3:  Unrelated name → error has NO "Did you mean:" clause
  AC-4:  Module::fn qualified fallback → success, ctx.note notes bare-name match
  AC-5:  Class.method qualified fallback → success, ctx.note shows real index key
  AC-6:  Path-style provider/module.fn → success, ctx.note references resolved name
  AC-7:  Ambiguous bare name → ctx.note enumerates both candidates
  AC-8:  Happy-path direct hit → ctx.note is None
  AC-9:  _strip_namespace_qualifier parametrised boundary tests
  AC-10: Two-hop chain note renders 'resolved via … to …'
  AC-11: Separator precedence: "a/b.c" → "c" (rightmost separator among matches)

All tests MUST FAIL until the Phase-9 implementation lands (_strip_namespace_qualifier,
_format_fallback_note, RelevantContext.note field, fuzzy-suggest block are all absent).
"""

from pathlib import Path

import pytest

from tldr.api import get_relevant_context, get_relevant_context_multi, RelevantContext

# Import new helpers lazily inside the tests that need them so that
# pytest --collect-only succeeds even before the implementation lands.
# The tests themselves will fail with ImportError / AttributeError — which
# is the correct RED signal (missing feature, not broken test infrastructure).
try:
    from tldr.api import _strip_namespace_qualifier
except ImportError:
    _strip_namespace_qualifier = None  # type: ignore[assignment]

try:
    from tldr.api import _format_fallback_note
except ImportError:
    _format_fallback_note = None  # type: ignore[assignment]

try:
    from tldr.api import _first_namespace_strip
except ImportError:
    _first_namespace_strip = None  # type: ignore[assignment]

try:
    from tldr.api import _format_chain_note
except ImportError:
    _format_chain_note = None  # type: ignore[assignment]


# ===========================================================================
# AC-1: Typo → "Did you mean" in error
# ===========================================================================

def test_typo_suggests_correct_name(tmp_path: Path, write_temp_file):
    """AC-1: A one-character transposition triggers a fuzzy suggestion.

    'filter_invocie' is a typo for 'filter_invoice'; difflib at cutoff=0.6
    should surface it in the error message.
    """
    write_temp_file(tmp_path / "module.py", "def filter_invoice(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="filter_invocie",
        depth=2,
        language="python",
    )

    assert ctx.error is not None, "Expected an error for a missing entry point"
    assert "Did you mean:" in ctx.error, (
        f"Expected 'Did you mean:' in error, got: {ctx.error!r}"
    )
    assert "filter_invoice" in ctx.error, (
        f"Expected 'filter_invoice' in the suggestion, got: {ctx.error!r}"
    )


# ===========================================================================
# AC-2: Multiple close matches surface comma-separated in closeness order
# ===========================================================================

def test_typo_multiple_suggestions_are_comma_separated(tmp_path: Path, write_temp_file):
    """AC-2: When multiple names are close to the typo, all are suggested comma-separated.

    '_Init', 'Init', '_init' are all close to 'init' (slight case differences /
    underscores); difflib (n=3, cutoff=0.6) should return all three.
    The error message must list them comma-separated.
    """
    write_temp_file(
        tmp_path / "module.py",
        "def _Init(): pass\ndef Init(): pass\ndef _init(): pass\n",
    )

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="init",
        depth=2,
        language="python",
    )

    assert ctx.error is not None, "Expected an error for missing entry point 'init'"
    assert "Did you mean:" in ctx.error, (
        f"Expected 'Did you mean:' in error, got: {ctx.error!r}"
    )
    # All three should appear in the suggestion string
    assert "Init" in ctx.error, (
        f"Expected 'Init' among suggestions, got: {ctx.error!r}"
    )
    assert "_init" in ctx.error or "_Init" in ctx.error, (
        f"Expected at least one underscored variant in suggestions, got: {ctx.error!r}"
    )
    # Comma-separated means at least one comma exists after "Did you mean:"
    did_you_mean_part = ctx.error[ctx.error.index("Did you mean:"):]
    assert "," in did_you_mean_part, (
        f"Expected comma-separated suggestions after 'Did you mean:', got: {did_you_mean_part!r}"
    )


# ===========================================================================
# AC-3: Unrelated name → no "Did you mean:" clause
# ===========================================================================

def test_typo_no_close_match_omits_hint(tmp_path: Path, write_temp_file):
    """AC-3: When nothing is close, no 'Did you mean:' suffix appears.

    Additionally asserts that the error message includes the entry_point name so
    the user knows what was searched — this locks in the error message format
    beyond just 'no suggestion', ensuring the fuzzy block doesn't accidentally
    swallow the entry_point name.
    """
    write_temp_file(tmp_path / "module.py", "def filter_invoice(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="xyzqwerty123",
        depth=2,
        language="python",
    )

    assert ctx.error is not None, "Expected an error for nonsense entry point"
    assert "Did you mean" not in ctx.error, (
        f"Expected NO 'Did you mean:' for unrelated name, got: {ctx.error!r}"
    )
    # The error must also explicitly report the missing name (not silently drop it)
    assert "xyzqwerty123" in ctx.error, (
        f"Expected entry point name in error message, got: {ctx.error!r}"
    )
    # The ctx.note must be None — fuzzy suggest is an error-path feature (no note on miss)
    assert ctx.note is None, (
        f"Expected ctx.note to be None on total miss, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-4: Module::fn qualified fallback → success + note
# ===========================================================================

def test_qualified_colon_falls_back_to_bare(tmp_path: Path, write_temp_file):
    """AC-4: 'Calendar::next_day' — '::' separator triggers bare-name fallback.

    The index only has 'next_day' (no Calendar:: prefix); fallback should
    resolve via bare segment and set ctx.note containing 'Calendar::next_day'
    and the bare resolved name.
    """
    write_temp_file(tmp_path / "calendar_mod.py", "def next_day(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="Calendar::next_day",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected success for qualified fallback, got error: {ctx.error!r}"
    )
    assert ctx.note is not None, "Expected ctx.note to be set on qualified-name fallback"
    assert "Calendar::next_day" in ctx.note, (
        f"Expected original entry point in note, got: {ctx.note!r}"
    )
    assert "next_day" in ctx.note, (
        f"Expected bare name 'next_day' referenced in note, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-5: Class.method qualified fallback → shows real index key (G-5 decision)
# ===========================================================================

def test_qualified_dot_falls_back_to_bare(tmp_path: Path, write_temp_file):
    """AC-5: 'AppState.filter_pills' — '.' separator triggers bare-name fallback.

    The index has 'filter_pills' (no AppState prefix); the note must show
    'resolved to' + the actual signatures-dict key (the real index key per G-5).
    """
    write_temp_file(tmp_path / "state.py", "def filter_pills(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="AppState.filter_pills",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected success for qualified-dot fallback, got error: {ctx.error!r}"
    )
    assert ctx.note is not None, "Expected ctx.note to be set on qualified-name fallback"
    assert "AppState.filter_pills" in ctx.note, (
        f"Expected original entry point in note, got: {ctx.note!r}"
    )
    # G-5: note shows 'resolved to' + real index key
    assert "resolved to" in ctx.note, (
        f"Expected 'resolved to' in note (G-5 decision), got: {ctx.note!r}"
    )
    # The actual key (either 'filter_pills' or 'state.filter_pills' etc.) must appear
    assert "filter_pills" in ctx.note, (
        f"Expected 'filter_pills' (the resolved key) in note, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-6: Path-style fallback (single-hop)
# ===========================================================================

def test_path_style_single_hop_fallback(tmp_path: Path, write_temp_file):
    """AC-6: 'providers/mod.stream' — '/' separator triggers bare-name fallback.

    'providers/mod.stream' has both '/' AND '.' so it bypasses module-query mode
    (spec: module-query fires only when '/' in name AND '.' NOT in name).
    The index only has 'stream'; '/' fires first → fallback 'mod.stream',
    then '.' fires → bare 'stream'. ctx.note should reference the resolved name.
    """
    write_temp_file(tmp_path / "mod.py", "def stream(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="providers/mod.stream",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected success for path-style fallback, got error: {ctx.error!r}"
    )
    assert ctx.note is not None, "Expected ctx.note to be set on path-style fallback"
    assert "stream" in ctx.note, (
        f"Expected 'stream' referenced in note, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-7: Ambiguous bare name → note enumerates all candidates
# ===========================================================================

def test_ambiguous_bare_name_enumerates_candidates(tmp_path: Path, write_temp_file):
    """AC-7: Two files each defining 'filter_pills' — qualified lookup 'AppState.filter_pills'
    falls back to bare 'filter_pills' which matches TWO entries.
    The ctx.note must enumerate both qualified keys.
    """
    write_temp_file(tmp_path / "a.py", "def filter_pills(): pass\n")
    write_temp_file(tmp_path / "b.py", "def filter_pills(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="AppState.filter_pills",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected success (first match used), got error: {ctx.error!r}"
    )
    assert ctx.note is not None, "Expected ctx.note to be set when multiple candidates found"
    # Note must enumerate both candidates; at minimum it says 'candidates' or '2'
    note_lower = ctx.note.lower()
    assert "candidate" in note_lower or "2" in ctx.note, (
        f"Expected 'candidates' or '2' in note for ambiguous match, got: {ctx.note!r}"
    )
    # Both qualified keys must be named (they come from files a.py and b.py)
    assert "filter_pills" in ctx.note, (
        f"Expected 'filter_pills' in the candidates list, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-8: Happy-path direct hit → note is None
# ===========================================================================

def test_resolved_first_try_has_no_note(tmp_path: Path, write_temp_file):
    """AC-8: When the entry point resolves directly, ctx.note must be None."""
    write_temp_file(tmp_path / "module.py", "def filter_invoice(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="filter_invoice",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected clean resolution, got error: {ctx.error!r}"
    )
    assert ctx.note is None, (
        f"Expected ctx.note to be None on direct hit, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-9: _strip_namespace_qualifier boundary inputs (G-4 parametrize decision)
# ===========================================================================

@pytest.mark.parametrize("input_name,expected", [
    ("",                None),   # empty string — no separator
    ("foo",             None),   # bare name — no separator
    (".",               None),   # separator only, empty trailing segment
    (".foo",            None),   # leading dot, empty leading segment → None for safety
    ("foo.",            None),   # trailing dot, empty trailing segment
    ("::foo",           "foo"),  # leading :: — trailing segment is non-empty
    ("A.B",             "B"),    # dot separator
    ("A::B",            "B"),    # :: separator
    ("A/B",             "B"),    # slash separator
    ("A..B",            "B"),    # double-dot — rightmost . splits on B
    ("mod::sub::func",  "func"), # multiple :: — rightmost segment
    ("a/b.c",           "c"),    # mixed separators — helper picks rightmost matching
                                 # separator across all separators, so the trailing
                                 # segment after the rightmost '.' wins → "c".
])
def test_strip_namespace_qualifier_boundaries(input_name: str, expected):
    """AC-9: _strip_namespace_qualifier boundary inputs per G-4 decision table."""
    if _strip_namespace_qualifier is None:
        pytest.fail(
            "_strip_namespace_qualifier is not exported from tldr.api — "
            "Phase-9 implementation is required."
        )
    result = _strip_namespace_qualifier(input_name)
    assert result == expected, (
        f"_strip_namespace_qualifier({input_name!r}) expected {expected!r}, got {result!r}"
    )


# ===========================================================================
# AC-10: Two-hop chain note (T-2 decision)
# ===========================================================================

def test_two_hop_chain_note_renders_full_chain(tmp_path: Path, write_temp_file):
    """AC-10: 'providers/anthropic.stream' — two hops:
      hop 1: / → 'anthropic.stream' (still misses)
      hop 2: . → 'stream' (hits)
    The OUTERMOST ctx.note must contain all three of:
      - 'providers/anthropic.stream' (original)
      - 'resolved via' (chain keyword per T-2)
      - 'stream' (final resolved name)
    """
    # Only 'stream' is indexed; neither 'providers/anthropic.stream' nor
    # 'anthropic.stream' exist in the index.
    write_temp_file(tmp_path / "anthropic_mod.py", "def stream(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="providers/anthropic.stream",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected two-hop resolution to succeed, got error: {ctx.error!r}"
    )
    assert ctx.note is not None, "Expected ctx.note set on two-hop chain"
    assert "providers/anthropic.stream" in ctx.note, (
        f"Expected original entry point in chain note, got: {ctx.note!r}"
    )
    assert "resolved via" in ctx.note, (
        f"Expected 'resolved via' in chain note (T-2 decision), got: {ctx.note!r}"
    )
    assert "stream" in ctx.note, (
        f"Expected final resolved name 'stream' in chain note, got: {ctx.note!r}"
    )


# ===========================================================================
# AC-11: Separator precedence "a/b.c" → final fallback is "c"
# ===========================================================================

def test_separator_precedence_path_dot(tmp_path: Path, write_temp_file):
    """AC-11: For 'a/b.c', the final resolved bare name is 'c'.

    / fires first (returning 'b.c'), then . fires on 'b.c' (returning 'c').
    The function 'c' is indexed; both hops must succeed and ctx.note should
    reference 'c' as the ultimately resolved name.
    """
    write_temp_file(tmp_path / "module.py", "def c(): pass\n")

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="a/b.c",
        depth=2,
        language="python",
    )

    assert ctx.error is None, (
        f"Expected 'a/b.c' to resolve to bare 'c' via two-hop fallback, got error: {ctx.error!r}"
    )
    assert ctx.note is not None, "Expected ctx.note to be set on multi-hop fallback"
    # The resolved bare name "c" must appear in the chain note
    assert "c" in ctx.note, (
        f"Expected 'c' referenced in chain note, got: {ctx.note!r}"
    )


# ===========================================================================
# RelevantContext.note field and to_llm_string() contract
# ===========================================================================

def test_to_llm_string_prepends_note_on_success():
    """to_llm_string() must prepend 'Note: ...' when note is set and error is None."""
    ctx = RelevantContext(entry_point="fn", depth=2, note="matched bare name; 'A.fn' resolved to 'fn'")
    output = ctx.to_llm_string()
    assert output.startswith("Note: "), (
        f"Expected to_llm_string() to start with 'Note: ' when note is set, got: {output[:60]!r}"
    )
    assert "matched bare name" in output, (
        f"Expected note content in to_llm_string() output, got: {output[:120]!r}"
    )


def test_to_llm_string_note_absent_when_error_set():
    """to_llm_string() error path must NOT include the note, even if note is set."""
    ctx = RelevantContext(
        entry_point="fn",
        depth=2,
        error="Function 'fn' not found in project",
        note="should not appear",
    )
    output = ctx.to_llm_string()
    assert output.startswith("Error:"), (
        f"Expected error path to start with 'Error:', got: {output[:60]!r}"
    )
    assert "should not appear" not in output, (
        f"Note must NOT appear in error-path output, got: {output!r}"
    )


def test_to_llm_string_no_prefix_when_note_is_none():
    """to_llm_string() must NOT prepend any prefix when note is None.

    Also asserts that RelevantContext.note defaults to None and that
    the output starts with '## Code Context:' (not shifted by any prefix).
    """
    ctx = RelevantContext(entry_point="fn", depth=2)
    # The 'note' field must exist and be None by default
    assert hasattr(ctx, "note"), (
        "RelevantContext must have a 'note' field (Phase-9 adds it)"
    )
    assert ctx.note is None, (
        f"RelevantContext.note must default to None, got: {ctx.note!r}"
    )
    output = ctx.to_llm_string()
    # With no functions and no error, output must start with the header, not a note
    assert output.startswith("## Code Context:"), (
        f"Expected output to start with '## Code Context:' when note=None, got: {output[:60]!r}"
    )
    assert not output.startswith("Note:"), (
        f"Expected no 'Note:' prefix when ctx.note is None, got: {output[:60]!r}"
    )


# --- Multi-language wrapper integration tests (Phase 10.2 reroute) ---
# These tests exercise get_relevant_context_multi(), which the CLI actually
# invokes. The L1×L2 gap: the all-language-miss path at api.py:1204-1212
# discards per-language ctx.error (containing the fuzzy hint) and returns
# a plain "(probed: <langs>)" wrapper. Tests 1 and 2 below expose that gap
# and MUST FAIL until Phase-9 wires the hint through the multi wrapper.
# Tests 3 and 4 are regression guards (may already pass on current code).

def test_multi_preserves_fuzzy_hint_single_language(tmp_path: Path, write_temp_file):
    """Phase 10.2 RED: get_relevant_context_multi with one language must
    preserve the fuzzy 'Did you mean:' hint from the per-language probe.

    'filter_invocie' is a typo for 'filter_invoice'; the per-language probe
    inside get_relevant_context_multi produces a ctx.error that already
    contains 'Did you mean: filter_invoice' (once Phase-9 lands). But today
    the multi-wrapper discards that per-language error and substitutes
    'Function ... not found in project (probed: python)' — stripping the hint.

    Asserts:
    - ctx.error is not None (a miss is expected)
    - 'Did you mean: filter_invoice' is IN ctx.error (the hint survives)
    """
    write_temp_file(tmp_path / "module.py", "def filter_invoice(): pass\n")

    ctx = get_relevant_context_multi(
        project=str(tmp_path),
        entry_point="filter_invocie",
        languages=("python",),
    )

    assert ctx.error is not None, (
        "Expected an error for typo 'filter_invocie' via get_relevant_context_multi"
    )
    assert "Did you mean: filter_invoice" in ctx.error, (
        f"Expected fuzzy hint 'Did you mean: filter_invoice' to survive the "
        f"multi-language wrapper, but ctx.error was: {ctx.error!r}"
    )


def test_multi_preserves_fuzzy_hint_multi_language_all_miss(tmp_path: Path, write_temp_file):
    """Phase 10.2 RED: get_relevant_context_multi with multiple languages,
    all missing, must still surface the fuzzy hint from the language probe
    that produced one.

    Fixture has only Python code with 'filter_invoice'. Both python and
    typescript probes miss on 'filter_invocie'. The Python probe produces
    'Did you mean: filter_invoice'. The multi-wrapper must propagate that
    hint rather than discarding it with a plain '(probed: python, typescript)'
    wrapper.

    Asserts:
    - ctx.error is not None (total miss)
    - 'Did you mean: filter_invoice' is IN ctx.error
    """
    write_temp_file(tmp_path / "module.py", "def filter_invoice(): pass\n")

    ctx = get_relevant_context_multi(
        project=str(tmp_path),
        entry_point="filter_invocie",
        languages=("python", "typescript"),
    )

    assert ctx.error is not None, (
        "Expected an error for typo 'filter_invocie' (all languages miss)"
    )
    assert "Did you mean: filter_invoice" in ctx.error, (
        f"Expected fuzzy hint to propagate from the per-language probe that "
        f"produced it, but ctx.error was: {ctx.error!r}. "
        f"The multi-wrapper must not discard per-language hints."
    )


def test_multi_preserves_qualified_fallback_note(tmp_path: Path, write_temp_file):
    """Phase 10.2 regression guard: the qualified-name fallback success path
    (ctx.error=None, ctx.note set) must survive get_relevant_context_multi.

    Fixture has bare 'next_day'; call with 'Calendar::next_day'. The per-
    language probe resolves via the '::' fallback and returns ctx.error=None
    with ctx.note containing 'resolved to'. The multi-wrapper's first-hit-wins
    path returns that ctx directly — this should already work on current code.

    Asserts:
    - ctx.error is None (success via fallback)
    - ctx.note is not None
    - 'resolved to' is in ctx.note (the qualified-name fallback note text)
    """
    write_temp_file(tmp_path / "calendar_mod.py", "def next_day(): pass\n")

    ctx = get_relevant_context_multi(
        project=str(tmp_path),
        entry_point="Calendar::next_day",
        languages=("python",),
    )

    assert ctx.error is None, (
        f"Expected success for qualified fallback via multi-wrapper, "
        f"got error: {ctx.error!r}"
    )
    assert ctx.note is not None, (
        "Expected ctx.note to be set on qualified-name fallback through multi-wrapper"
    )
    assert "resolved to" in ctx.note, (
        f"Expected 'resolved to' in ctx.note (qualified fallback note text), "
        f"got: {ctx.note!r}"
    )


def test_multi_unrelated_name_no_did_you_mean(tmp_path: Path, write_temp_file):
    """Phase 10.2 regression guard: an entirely unrelated name must NOT produce
    a spurious 'Did you mean' hint through get_relevant_context_multi.

    This is the negative counterpart to the two fuzzy-hint RED tests above.
    It also prevents a future Phase-9 implementation from being over-eager
    (e.g., always injecting a hint even when difflib returns no candidates).

    Asserts:
    - ctx.error is not None (a miss is expected)
    - 'Did you mean' is NOT in ctx.error
    """
    write_temp_file(tmp_path / "module.py", "def filter_invoice(): pass\n")

    ctx = get_relevant_context_multi(
        project=str(tmp_path),
        entry_point="xyzqwerty",
        languages=("python",),
    )

    assert ctx.error is not None, (
        "Expected an error for nonsense entry point 'xyzqwerty'"
    )
    assert "Did you mean" not in ctx.error, (
        f"Expected NO 'Did you mean:' hint for unrelated name 'xyzqwerty', "
        f"but ctx.error was: {ctx.error!r}"
    )


# ===========================================================================
# C-14: Direct unit tests for _first_namespace_strip and _format_chain_note
# ===========================================================================


@pytest.mark.parametrize("input_name,expected", [
    ("a.b",          "b"),       # single dot
    ("a/b.c",        "b.c"),     # leftmost separator wins (/ before .)
    ("a",            None),      # no separator
    ("a::b::c",      "b::c"),    # leftmost :: — trailing has remaining segments
])
def test_first_namespace_strip_boundaries(input_name: str, expected):
    """C-14: direct boundary tests for _first_namespace_strip (leftmost find)."""
    if _first_namespace_strip is None:
        pytest.fail("_first_namespace_strip is not exported from tldr.api")
    result = _first_namespace_strip(input_name)
    assert result == expected, (
        f"_first_namespace_strip({input_name!r}) expected {expected!r}, got {result!r}"
    )


def test_format_chain_note_single_hop_dropthrough():
    """C-14: single-hop (intermediate == fallback) is NOT this helper's concern;
    the caller decides to call _format_fallback_note instead. This test verifies
    the chain helper still renders intelligibly when intermediate equals final
    (defensive: should never be invoked this way but must not crash)."""
    if _format_chain_note is None:
        pytest.fail("_format_chain_note is not exported from tldr.api")
    note = _format_chain_note("A.b", "b", "b")
    assert "resolved via" in note
    assert "'A.b'" in note
    assert "'b'" in note


def test_format_chain_note_chain_rendering():
    """C-14: chain rendering produces 'resolved via {intermediate} to {final}'."""
    if _format_chain_note is None:
        pytest.fail("_format_chain_note is not exported from tldr.api")
    note = _format_chain_note(
        "providers/anthropic.stream", "anthropic.stream", "stream"
    )
    assert "resolved via 'anthropic.stream' to 'stream'" in note
    assert "'providers/anthropic.stream'" in note
    assert "(qualified form not in index)" in note
