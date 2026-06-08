"""Regression tests for the Goal-C CLI integration bug and the EDGE-4 reuse
determinism finding.

Two production bugs are guarded here:

FIX 1 (Goal C — `semantic index` CLI entry point), tldr/cli.py:
    `tldr semantic index <py+js project>` with no --lang (default "auto") indexed
    ONLY Python. ``resolve_language("auto")`` collapses to a SINGLE language, so
    ``build_semantic_index`` never reached the multi-language expansion path even
    though the API (``extract_units_from_project(lang=None)``) is correct. The
    fix mirrors the `search` action: "auto"/"all" -> lang=None.

FIX 2 (EDGE-4 — reuse determinism), tldr/semantic.py:
    The L2 call-graph edges feeding each unit's ``calls`` / ``called_by`` are
    emitted in a hash-seed-dependent order by ``build_project_call_graph``. A raw
    ``list[:5]`` slice therefore picked a different ORDER *and* MEMBERSHIP across
    separate Python processes, so a reindex in a fresh interpreter changed every
    affected unit's ``text_hash`` and needlessly re-embedded it. Sorting the call
    lists before the cap (``_stable_call_list``) makes the text_hash stable, so an
    unchanged reindex reuses all vectors.

Design: follows the ``_make_fake_model()`` + mini-repo pattern from
test_parse_skip.py / test_incremental_semantic_index.py. No real embedding model
and no ``@pytest.mark.e2e`` — every test runs under ``python3 -m pytest --no-cov``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

# Shared deterministic fake embedder lives in conftest.py (importable as a
# module — rootdir is on sys.path during the pytest run). De-duped local copy.
from conftest import make_fake_model as _make_fake_model

# ---------------------------------------------------------------------------
# Repo root anchor
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

_PY_CONTENT = '''\
def hello_python(x):
    """Return x incremented."""
    return x + 1
'''

_JS_CONTENT = '''\
export function helloJavaScript(y) {
    // Return y doubled
    return y * 2;
}
'''

# A Python callee called by many sibling functions across several files. The
# fan-in makes ``shared.called_by`` a multi-element list, so any non-deterministic
# edge order surfaces as an unstable, hash-seed-dependent slice.
_PY_SHARED = '''\
def shared(x):
    """Shared helper."""
    return x + 1
'''


def _caller_file(names: List[str]) -> str:
    body = "from shared import shared\n"
    for nm in names:
        body += (
            f"\ndef {nm}(y):\n"
            f'    """Caller {nm}."""\n'
            f"    return shared(y) + {len(nm)}\n"
        )
    return body


# ---------------------------------------------------------------------------
# Mini-repo builders
# ---------------------------------------------------------------------------

def _build_py_js_repo(tmp_path: Path) -> Path:
    """Project with one .py file and one .js file (+ .git anchor)."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "module_a.py").write_text(_PY_CONTENT)
    (tmp_path / "module_b.js").write_text(_JS_CONTENT)
    return tmp_path


def _build_fanin_repo(tmp_path: Path) -> Path:
    """Multi-file Python project: many callers of one shared callee."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "shared.py").write_text(_PY_SHARED)
    (tmp_path / "a.py").write_text(_caller_file(["alpha", "beta", "gamma"]))
    (tmp_path / "b.py").write_text(_caller_file(["delta", "epsilon", "zeta"]))
    (tmp_path / "c.py").write_text(_caller_file(["eta", "theta", "iota"]))
    return tmp_path


def _read_metadata(repo: Path) -> dict:
    meta_file = repo / ".tldr" / "cache" / "semantic" / "metadata.json"
    return json.loads(meta_file.read_text())


def _run_index_capture_summary(repo: Path) -> str:
    """Build the semantic index with the fake model; return the stderr summary line.

    ``build_semantic_index`` always prints ``Semantic index: embedded N, reused M
    units (device=...)`` to stderr — independent of show_progress — so the reuse
    counts are observable without a real model.
    """
    from tldr.semantic import build_semantic_index

    buf = io.StringIO()
    with patch("tldr.semantic.get_model", return_value=_make_fake_model()):
        with contextlib.redirect_stderr(buf):
            build_semantic_index(str(repo), lang="python", show_progress=False)
    summary = [ln for ln in buf.getvalue().splitlines() if "Semantic index:" in ln]
    assert summary, f"expected a 'Semantic index:' summary line; got: {buf.getvalue()!r}"
    return summary[-1]


# ===========================================================================
# FIX 1 — Unit test: CLI `semantic index` lang resolution (guards cli.py:1184)
# ===========================================================================

class TestSemanticIndexLangResolution:
    """`tldr semantic index` must pass lang=None for the default ("auto") and
    for "all", so build_semantic_index takes the multi-language expansion path.

    RED before the fix: cli.py did ``lang = resolve_language(args.lang, args.path)``
    which collapses "auto" to a SINGLE detected language, so only that language
    was indexed (Goal C broken at the CLI entry point).
    """

    def _invoke_cli_index(self, lang_flag: str, repo: Path):
        """Drive ``cli.main()`` for ``semantic index`` with build_semantic_index
        patched, and return the captured call kwargs.

        Patches ``tldr.semantic.build_semantic_index`` (the source binding picked
        up by the in-function ``from .semantic import build_semantic_index``) and
        ``sys.argv`` so the real cli.py:1184 dispatch executes.
        """
        from tldr import cli

        captured: dict = {}

        def fake_build(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return 0

        argv = ["tldr", "semantic", "index", str(repo), "--lang", lang_flag]
        with patch("tldr.semantic.build_semantic_index", side_effect=fake_build) as mock_build:
            with patch.object(sys, "argv", argv):
                cli.main()
        assert mock_build.called, "build_semantic_index was never invoked by the CLI"
        return captured

    def test_index_lang_auto_passes_lang_none(self, tmp_path: Path):
        """Default ``--lang auto`` must reach build_semantic_index as lang=None.

        This is the multi-language trigger: build_semantic_index(lang=None) ->
        extract_units_from_project(lang=None) -> auto-detect + merge ALL languages.
        """
        repo = _build_py_js_repo(tmp_path)
        captured = self._invoke_cli_index("auto", repo)
        assert captured["kwargs"].get("lang") is None, (
            "CLI `semantic index --lang auto` must pass lang=None (multi-language "
            f"expansion), not a single resolved language; got lang="
            f"{captured['kwargs'].get('lang')!r}. Regression of the Goal-C CLI bug."
        )

    def test_index_lang_all_passes_lang_none(self, tmp_path: Path):
        """Explicit ``--lang all`` must also reach build_semantic_index as lang=None."""
        repo = _build_py_js_repo(tmp_path)
        captured = self._invoke_cli_index("all", repo)
        assert captured["kwargs"].get("lang") is None, (
            "CLI `semantic index --lang all` must pass lang=None; got lang="
            f"{captured['kwargs'].get('lang')!r}."
        )

    def test_index_explicit_lang_still_resolves_to_that_language(self, tmp_path: Path):
        """Negative control: an explicit ``--lang python`` must NOT become None.

        Guards against an over-broad fix that forces lang=None for every value —
        explicit single-language selection must keep working.
        """
        repo = _build_py_js_repo(tmp_path)
        captured = self._invoke_cli_index("python", repo)
        assert captured["kwargs"].get("lang") == "python", (
            "CLI `semantic index --lang python` must pass lang='python' (explicit "
            f"single language preserved); got lang={captured['kwargs'].get('lang')!r}."
        )


# ===========================================================================
# FIX 1 — Integration test: end-to-end multi-language index build (no real model)
# ===========================================================================

class TestMultiLanguageIndexBuild:
    """build_semantic_index(lang=None) on a py+js repo must index BOTH languages.

    Guards the full end-to-end path (CLI -> build_semantic_index ->
    extract_units_from_project(lang=None) -> per-language merge) with the fake
    model, so the persisted metadata contains units from Python AND JavaScript.
    """

    def test_lang_none_indexes_python_and_javascript_units(self, tmp_path: Path):
        from tldr.semantic import build_semantic_index

        repo = _build_py_js_repo(tmp_path)

        with patch("tldr.semantic.get_model", return_value=_make_fake_model()):
            count = build_semantic_index(
                str(repo), lang=None, show_progress=False, respect_ignore=False,
            )

        assert count >= 2, f"expected >=2 units across both languages; got {count}"

        meta = _read_metadata(repo)
        languages = {u.get("language") for u in meta["units"]}

        assert any(u.get("language") == "python" for u in meta["units"]), (
            f"expected >=1 python unit in the index; languages found: {languages}"
        )
        assert any(u.get("language") in {"javascript", "js"} for u in meta["units"]), (
            "expected >=1 javascript unit in the index; languages found: "
            f"{languages}. Regression: lang=None must index ALL languages, not "
            "only Python."
        )


# ===========================================================================
# FIX 2 — Determinism regression: unchanged reindex reuses ALL vectors
# ===========================================================================

class TestReindexReuseDeterminism:
    """A second build_semantic_index run on an UNCHANGED repo must reuse every
    vector (embedded 0).

    RED before the fix: ``calls`` / ``called_by`` were sliced ``[:5]`` from a
    hash-seed-dependent call-graph edge order, so the persisted slice (and thus
    each unit's text_hash) was unstable across processes. A reindex in a fresh
    interpreter re-embedded the affected units. The ``_stable_call_list`` sort
    (applied before the cap) makes the slice deterministic, so the text_hash is
    stable and unchanged units reuse their cached vector.
    """

    def test_second_unchanged_reindex_reuses_all_units(self, tmp_path: Path):
        repo = _build_fanin_repo(tmp_path)

        first = _run_index_capture_summary(repo)
        assert "reused 0" in first, (
            f"first build should embed everything (reused 0); got: {first!r}"
        )

        second = _run_index_capture_summary(repo)
        assert "embedded 0" in second, (
            "second reindex of an UNCHANGED repo must reuse ALL vectors "
            f"(embedded 0); got: {second!r}. Regression of EDGE-4 reuse churn "
            "(non-deterministic calls/called_by order leaking into text_hash)."
        )

    def test_persisted_called_by_is_sorted_deterministic(self, tmp_path: Path):
        """The persisted ``called_by`` list must be sorted (and de-duplicated).

        A sorted list is independent of the hash-seed-dependent edge-emission
        order, which is what makes the text_hash stable across processes. This
        pins the determinism invariant directly on the persisted metadata.
        """
        repo = _build_fanin_repo(tmp_path)
        _run_index_capture_summary(repo)

        meta = _read_metadata(repo)
        shared = [u for u in meta["units"] if u.get("name") == "shared"]
        assert shared, "expected a 'shared' unit with multiple callers in the index"

        called_by = shared[0].get("called_by") or []
        assert len(called_by) >= 2, (
            f"expected 'shared' to have multiple callers; got {called_by!r}"
        )
        assert called_by == sorted(set(called_by)), (
            "persisted 'shared.called_by' must be sorted + de-duplicated for a "
            f"stable text_hash across processes; got {called_by!r}"
        )
