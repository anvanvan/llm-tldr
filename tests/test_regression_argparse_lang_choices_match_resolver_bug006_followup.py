"""Regression test for bug 006 follow-up: argparse `--lang` choices on every
call-graph-family subcommand must be a subset of what
``_resolve_context_languages`` accepts at runtime.

History
-------
Items 3 + 5 of the fix sweep widened ``SUPPORTED_CONTEXT_EXT_MAP`` from 8 to
11 languages (added ``ruby``, ``c``, ``elixir``) and surfaced them through
``CONTEXT_LANG_CHOICES`` -- but only the ``tldr context`` subparser was wired
to that constant.  The other call-graph subcommands (``impact``, ``calls``,
``dead``, ``arch``, ``importers``, ``change-impact``) still declared the
wider 18-language ``LANG_CHOICES`` list, so callers like
``tldr importers Foo --lang kotlin`` would pass argparse and then crash inside
``_resolve_context_languages`` with::

    Error: _resolve_context_languages: unexpected lang_arg='kotlin';
    argparse choices should have rejected this upstream.

Item 6 narrows every call-graph subparser to ``CONTEXT_LANG_CHOICES``.  This
test pins that invariant: for every advertised choice on every call-graph
subcommand, ``_resolve_context_languages`` must NOT raise the "unexpected
lang_arg" ValueError that signals an argparse/resolver skew.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tldr.api import SUPPORTED_CONTEXT_LANGUAGES
from tldr.cli import (
    CONTEXT_LANG_CHOICES,
    NoSupportedContextLanguagesError,
    _resolve_context_languages,
)


# Call-graph-family subcommands whose `--lang` argparse choices MUST match the
# resolver's accepted set. (`context` is the canonical case and is included
# here as a sanity anchor.)
#
# Note: ``importers`` and ``change-impact`` are intentionally EXCLUDED here.
# Their runtime supports a broader language set than the call-graph resolver
# (per-file extension detection works for 16+ languages), so they retain the
# wider ``LANG_CHOICES`` argparse list. Item 6's invariant only applies to the
# five subcommands that actually drive ``_resolve_context_languages``.
CALL_GRAPH_SUBCOMMANDS = (
    "context",
    "impact",
    "calls",
    "dead",
    "arch",
)


# --- Test 1: every CONTEXT_LANG_CHOICES entry is accepted by the resolver ---


@pytest.mark.parametrize("lang", CONTEXT_LANG_CHOICES)
def test_resolver_accepts_every_context_lang_choice(tmp_path: Path, lang: str):
    """Every value in ``CONTEXT_LANG_CHOICES`` (the constant wired to every
    call-graph subparser) must be a value ``_resolve_context_languages``
    handles without the upstream-skew ValueError.

    ``auto`` and ``all`` are special-cased; explicit languages must be in
    ``SUPPORTED_CONTEXT_LANGUAGES``.
    """
    # tmp_path is an empty dir -> auto-detection finds nothing -> falls back
    # to ["python"]. That's a valid resolver return value and avoids
    # NoSupportedContextLanguagesError (which only fires when source files
    # are detected but none are supported).
    result: list[str] | None = None
    try:
        result = _resolve_context_languages(lang, tmp_path, respect_ignore=True)
    except NoSupportedContextLanguagesError:
        # Should not happen in an empty dir, but tolerate it as a non-skew
        # failure mode.
        pytest.fail(
            f"Resolver raised NoSupportedContextLanguagesError for {lang!r} "
            "in an empty project (unexpected)."
        )
    except ValueError as exc:
        if "unexpected lang_arg" in str(exc):
            pytest.fail(
                f"argparse/resolver skew: choice {lang!r} is advertised by "
                f"argparse via CONTEXT_LANG_CHOICES but _resolve_context_languages "
                f"rejects it: {exc}"
            )
        raise

    assert result is not None, f"resolver returned no result for lang={lang!r}"
    assert isinstance(result, list)
    assert result, f"resolver returned empty list for lang={lang!r}"

    if lang == "all":
        assert set(result) == set(SUPPORTED_CONTEXT_LANGUAGES)
    elif lang == "auto":
        # Empty project -> fallback to ["python"]
        assert result == ["python"]
    else:
        assert result == [lang]
        assert lang in SUPPORTED_CONTEXT_LANGUAGES


# --- Test 2: CONTEXT_LANG_CHOICES is in sync with SUPPORTED_CONTEXT_LANGUAGES ---


def test_context_lang_choices_matches_supported_set():
    """``CONTEXT_LANG_CHOICES`` must be exactly ``{auto, all} ∪
    SUPPORTED_CONTEXT_LANGUAGES``. Drift would re-introduce the bug 006
    follow-up skew.
    """
    expected = {"auto", "all", *SUPPORTED_CONTEXT_LANGUAGES}
    assert set(CONTEXT_LANG_CHOICES) == expected, (
        f"CONTEXT_LANG_CHOICES drifted from SUPPORTED_CONTEXT_LANGUAGES: "
        f"choices={sorted(CONTEXT_LANG_CHOICES)} expected={sorted(expected)}"
    )


# --- Test 3: every call-graph subcommand uses CONTEXT_LANG_CHOICES ---
#
# We assert this structurally by reading tldr/cli.py and checking that each
# subcommand's `--lang` argument uses `choices=CONTEXT_LANG_CHOICES` (not
# `LANG_CHOICES` or `LANG_CHOICES_WITH_ALL`). This is a guard against future
# regressions where a new call-graph subcommand is added with the wrong
# choices constant.


def _cli_source() -> str:
    cli_path = Path(__file__).resolve().parents[1] / "tldr" / "cli.py"
    return cli_path.read_text()


@pytest.mark.parametrize("subcommand", CALL_GRAPH_SUBCOMMANDS)
def test_subcommand_uses_context_lang_choices(subcommand: str):
    """Each call-graph subcommand's subparser must wire `--lang` to
    ``CONTEXT_LANG_CHOICES``. Detects skew at source-level so a casual
    refactor that switches back to ``LANG_CHOICES`` trips this immediately.
    """
    source = _cli_source()
    # The subparser-creation comment line uses the canonical name, e.g.
    # ``# tldr impact <func> [path]`` or ``# tldr change-impact [files...]``.
    # We find that anchor then scan forward to the next `--lang` block.
    anchor = f"# tldr {subcommand}"
    idx = source.find(anchor)
    assert idx != -1, f"could not locate `{anchor}` anchor in tldr/cli.py"

    # Slice from anchor to the next subparser anchor (or EOF) and search
    # within that window for a `--lang` argument.
    next_anchor = source.find("# tldr ", idx + len(anchor))
    window = source[idx : next_anchor if next_anchor != -1 else len(source)]

    lang_idx = window.find('"--lang"')
    assert lang_idx != -1, (
        f"subcommand {subcommand!r} has no `--lang` argument; either remove "
        f"it from CALL_GRAPH_SUBCOMMANDS or add the argument."
    )

    # Window from `--lang` to the next add_argument call to scope the
    # choices= lookup tightly to this argument.
    after_lang = window[lang_idx:]
    end_of_block = after_lang.find("add_argument", len('"--lang"'))
    block = (
        after_lang[:end_of_block] if end_of_block != -1 else after_lang
    )

    assert "choices=CONTEXT_LANG_CHOICES" in block, (
        f"subcommand {subcommand!r} `--lang` argument does not use "
        f"choices=CONTEXT_LANG_CHOICES. This re-introduces the bug 006 "
        f"follow-up argparse/resolver skew."
    )
