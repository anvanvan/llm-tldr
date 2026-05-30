# pyright: reportMissingImports=false
"""Regression test — Bug 006: Ruby file-scope `def` not in call graph
AND `--lang ruby` rejected by argparse.

Two-part bug:
  PART A — `tldr context --lang ruby` and `tldr impact --lang ruby` rejected
           because ``SUPPORTED_CONTEXT_LANGUAGES`` did not include "ruby".
           Argparse choices on ``tldr context`` rejected ``ruby``; the
           ``impact`` subcommand reached ``_resolve_context_languages`` and
           raised a "unexpected lang_arg='ruby'" ValueError.

  PART B — File-scope ``def`` (top-level Ruby functions, not nested inside
           ``class``/``module``) must participate in the call graph.  A
           top-level ``def caller_fn`` that bare-calls ``greet`` (another
           top-level def) must result in ``greet`` showing ``caller_fn`` as
           a caller via ``get_relevant_context``.

Fix landed by adding ``"ruby": {".rb"}`` to
``tldr.api.SUPPORTED_CONTEXT_EXT_MAP``.  The existing ``_build_ruby_call_graph``
already walks every ``method``/``singleton_method`` node regardless of nesting,
so file-scope defs were always registered — the only missing piece was the
language gate that argparse and ``_resolve_context_languages`` consult.
"""

import textwrap
from pathlib import Path


_APP_RB = textwrap.dedent("""\
    def greet(name)
      puts "Hello, #{name}!"
    end

    def caller_fn
      greet("World")
    end

    caller_fn
""")


def test_ruby_file_scope_def_call_graph_bug006(tmp_path: Path):
    """Bug 006: top-level Ruby ``def`` must appear in call graph AND
    ``--lang ruby`` must be accepted by argparse / _resolve_context_languages.

    Each assert covers one previously-broken behavior; pytest's introspection
    pinpoints which sub-bug regresses.
    """
    # --- Fixture: file-scope def calling another file-scope def ---
    (tmp_path / "app.rb").write_text(_APP_RB)

    # -----------------------------------------------------------------------
    # ASSERT 1 (Part A) — ``ruby`` must be in SUPPORTED_CONTEXT_LANGUAGES.
    # This is the single source of truth feeding both argparse `choices=` on
    # ``tldr context --lang`` and the ``_resolve_context_languages`` resolver
    # used by ``impact``.
    # -----------------------------------------------------------------------
    from tldr.api import SUPPORTED_CONTEXT_LANGUAGES

    assert "ruby" in SUPPORTED_CONTEXT_LANGUAGES, (
        "ASSERT 1 FAILED (Part A) — `ruby` missing from SUPPORTED_CONTEXT_LANGUAGES. "
        f"Present: {sorted(SUPPORTED_CONTEXT_LANGUAGES)!r}. "
        "Without ruby here, `tldr context --lang ruby` is rejected by argparse "
        "and `tldr impact --lang ruby` raises in _resolve_context_languages."
    )

    # -----------------------------------------------------------------------
    # ASSERT 2 (Part A) — _resolve_context_languages must accept "ruby"
    # without raising and return ["ruby"].
    # -----------------------------------------------------------------------
    from tldr.cli import _resolve_context_languages

    resolved = _resolve_context_languages("ruby", str(tmp_path))
    assert resolved == ["ruby"], (
        "ASSERT 2 FAILED (Part A) — _resolve_context_languages('ruby', ...) "
        f"returned {resolved!r}; expected ['ruby']."
    )

    # -----------------------------------------------------------------------
    # ASSERT 3 (Part A) — argparse `choices=` on `tldr context --lang` must
    # include "ruby" so the CLI accepts it.
    # -----------------------------------------------------------------------
    from tldr.cli import CONTEXT_LANG_CHOICES

    assert "ruby" in CONTEXT_LANG_CHOICES, (
        "ASSERT 3 FAILED (Part A) — `ruby` missing from CONTEXT_LANG_CHOICES. "
        f"Choices: {CONTEXT_LANG_CHOICES!r}. "
        "argparse will continue to reject `tldr context --lang ruby`."
    )

    # -----------------------------------------------------------------------
    # ASSERT 4 (Part B) — Top-level `def caller_fn` calling top-level
    # `def greet` must produce a call-graph edge where caller_fn is a caller
    # of greet, surfaced via get_relevant_context.
    # -----------------------------------------------------------------------
    from tldr.api import get_relevant_context

    ctx = get_relevant_context(
        project=str(tmp_path),
        entry_point="greet",
        depth=2,
        language="ruby",
    )
    func_names = [f.name for f in ctx.functions]
    # Names may be module-qualified (e.g. "app.caller_fn") depending on the
    # display path Ruby's get_relevant_context returns.  Match the trailing
    # segment.
    matched = [n for n in func_names if n.split(".")[-1] == "caller_fn"]
    assert matched, (
        "ASSERT 4 FAILED (Part B) — file-scope def `caller_fn` (which calls "
        "`greet`) did not appear as a caller of `greet`. "
        f"Functions returned: {func_names!r}. "
        "Top-level Ruby defs must register in the call graph just like "
        "defs nested in class/module."
    )
