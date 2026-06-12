"""RED-phase tests for the two ephemeral-index daemon-routing guards.

Guard 1 (Component A — ``DAEMON_ROUTED_COMMANDS`` shrink in ``tldr/cli.py``):
  index-free commands (search, tree, extract, imports, structure) must dispatch
  WITHOUT calling ``ensure_daemon``; index-using commands (semantic, context,
  impact, warm, change-impact) must still call it.  ``change-impact`` is newly
  routed (the prior set carried a dead ``change_impact`` underscore key that
  argparse's hyphen subcommand name never matched).

Guard 2 (Component B — ``_is_ephemeral_root`` + ``ensure_daemon`` early-skip in
  ``tldr/daemon/ensure.py``): a project root that resolves under (or equals)
  ``/tmp`` / ``/private/tmp`` must NOT spawn a daemon subprocess, unless the
  ``TLDR_INDEX_EPHEMERAL`` env escape hatch is truthy.

These are pure-unit tests: every seam (``tldr.cli.ensure_daemon``,
``ensure_mod.subprocess.Popen``, ``_ping_daemon``, ``query_daemon``) is
monkeypatched so NO GPU model server is ever touched.

RED on HEAD because:
  - search/tree/extract/imports/structure are still IN DAEMON_ROUTED_COMMANDS
    (Guard 1 not applied) → they DO call ensure_daemon.
  - change-impact is registered with the wrong underscore key → it does NOT
    route today.
  - ensure_daemon has no _is_ephemeral_root early-skip → it ALWAYS proceeds to
    Popen, even on a /tmp root.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest


# ---------------------------------------------------------------------------
# Isolation fixture (mandatory): CLAUDE_PROJECT_DIR re-anchors _find_project_root
# away from the explicit /tmp dir, and a stray .tldr ancestor would hijack the
# Pass-2 walk — both would mask the guard defects under test.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_root_resolution(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    # The ephemeral guard relies on a clean /tmp; the escape hatch must not leak
    # in from the ambient environment for the negative (skip) arms.
    monkeypatch.delenv("TLDR_INDEX_EPHEMERAL", raising=False)
    cur = tmp_path.resolve()
    while cur != cur.parent:
        assert not (cur / ".tldr").exists(), f"stray .tldr ancestor: {cur / '.tldr'}"
        cur = cur.parent


def _build_tiny_repo(root):
    (root / ".git").mkdir(exist_ok=True)
    (root / "app.py").write_text("def greet(name):\n    return f'Hello {name}'\n")
    return root


def _run_cli_recording_ensure(argv):
    """Drive ``tldr.cli.main`` with ``ensure_daemon`` replaced by a recorder.

    Returns the list of project args ensure_daemon was called with.  Dispatch
    after the routing gate is allowed to fail (swallowed) — only the gate's
    ensure call (or its absence) is under test.  ``query_daemon`` is patched so
    a routed ``semantic search`` never reaches the GPU model server.
    """
    from unittest.mock import patch

    ensure_calls = []

    def fake_ensure(proj, timeout=10.0):
        ensure_calls.append(proj)

    argv_backup = sys.argv[:]
    try:
        sys.argv = argv
        with patch("tldr.cli.ensure_daemon", fake_ensure, create=True), \
             patch(
                 "tldr.daemon.startup.query_daemon",
                 return_value={"status": "ok", "results": []},
             ):
            from tldr.cli import main
            try:
                main()
            except SystemExit:
                pass
            except Exception:
                pass
    finally:
        sys.argv = argv_backup
    return ensure_calls


# ===========================================================================
# Guard 1 — routing-set membership (Component A): B1, B2, B3
# ===========================================================================

class TestIndexFreeCommandsDoNotRoute:
    """B1 + B2: search/tree/extract/imports/structure never call ensure_daemon."""

    def test_search_does_not_call_ensure_daemon(self, tmp_path):
        # B1
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(["tldr", "search", "greet", project])
        assert calls == [], (
            f"'search' is index-free and must NOT route to ensure_daemon; "
            f"got {calls!r}. RED: search still in DAEMON_ROUTED_COMMANDS."
        )

    def test_tree_does_not_call_ensure_daemon(self, tmp_path):
        # B2
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(["tldr", "tree", project])
        assert calls == [], (
            f"'tree' must NOT route to ensure_daemon; got {calls!r}. "
            "RED: tree still in DAEMON_ROUTED_COMMANDS."
        )

    def test_extract_does_not_call_ensure_daemon(self, tmp_path):
        # B2
        _build_tiny_repo(tmp_path)
        target = str(tmp_path / "app.py")
        calls = _run_cli_recording_ensure(["tldr", "extract", target, "--function", "greet"])
        assert calls == [], (
            f"'extract' must NOT route to ensure_daemon; got {calls!r}. "
            "RED: extract still in DAEMON_ROUTED_COMMANDS."
        )

    def test_imports_does_not_call_ensure_daemon(self, tmp_path):
        # B2
        _build_tiny_repo(tmp_path)
        target = str(tmp_path / "app.py")
        calls = _run_cli_recording_ensure(["tldr", "imports", target])
        assert calls == [], (
            f"'imports' must NOT route to ensure_daemon; got {calls!r}. "
            "RED: imports still in DAEMON_ROUTED_COMMANDS."
        )

    def test_structure_does_not_call_ensure_daemon(self, tmp_path):
        # B2
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(["tldr", "structure", project])
        assert calls == [], (
            f"'structure' must NOT route to ensure_daemon; got {calls!r}. "
            "RED: structure still in DAEMON_ROUTED_COMMANDS."
        )


class TestIndexUsingCommandsStillRoute:
    """B3: semantic/context/impact/warm/change-impact still call ensure_daemon."""

    def test_semantic_search_calls_ensure_daemon(self, tmp_path):
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(
            ["tldr", "semantic", "search", "greet", "--path", project]
        )
        assert len(calls) >= 1, (
            f"'semantic search' must route to ensure_daemon; got {calls!r}."
        )

    def test_context_calls_ensure_daemon(self, tmp_path):
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(
            ["tldr", "context", "greet", "--project", project]
        )
        assert len(calls) >= 1, (
            f"'context' must route to ensure_daemon; got {calls!r}."
        )

    def test_impact_calls_ensure_daemon(self, tmp_path):
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(
            ["tldr", "impact", "greet", "--project", project]
        )
        assert len(calls) >= 1, (
            f"'impact' must route to ensure_daemon; got {calls!r}."
        )

    def test_warm_calls_ensure_daemon(self, tmp_path):
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(["tldr", "warm", project])
        assert len(calls) >= 1, (
            f"'warm' must route to ensure_daemon; got {calls!r}."
        )

    def test_change_impact_calls_ensure_daemon(self, tmp_path):
        # B3 — newly routed via the hyphen-key fix (MUST_FIX A-4).
        project = str(_build_tiny_repo(tmp_path))
        calls = _run_cli_recording_ensure(["tldr", "change-impact", "--project", project])
        assert len(calls) >= 1, (
            f"'change-impact' must route to ensure_daemon; got {calls!r}. "
            "RED: the set carries a dead 'change_impact' underscore key that "
            "the hyphen subcommand name never matches."
        )


# ===========================================================================
# Guard 2 — ensure_daemon ephemeral skip (Component B): B4, B5, B6, B7
# ===========================================================================

class _PopenReached(Exception):
    """Sentinel raised by the fake Popen to prove the spawn branch was reached
    without actually launching a process (and without paying the readiness-wait
    timeout)."""


def _patch_popen_sentinel(monkeypatch):
    """Replace ensure_mod.subprocess.Popen with a sentinel-raiser and force
    _ping_daemon False so the spawn branch is the one reached.

    Returns a 1-element list whose membership flips to True iff Popen is invoked
    (so the B4 skip arm can assert Popen was NOT reached even though the
    sentinel short-circuits before any timeout loop).
    """
    import tldr.daemon.ensure as ensure_mod

    reached = []

    def _fake_popen(*args, **kwargs):
        reached.append((args, kwargs))
        raise _PopenReached()

    monkeypatch.setattr(ensure_mod.subprocess, "Popen", _fake_popen)
    # No live daemon: forces ensure_daemon down to the spawn branch (and the
    # under-lock re-check also misses), so reaching the guard's early-return is
    # the ONLY thing that can keep Popen from firing.
    monkeypatch.setattr(ensure_mod, "_ping_daemon", lambda project: False)
    return reached


class TestEphemeralSkip:

    def test_tmp_root_does_not_spawn_daemon(self, monkeypatch):
        """B4: a real /tmp/<name> git repo root must NOT spawn a daemon.

        With no live daemon, a non-guarded ensure_daemon would reach Popen
        (raising the sentinel). The early-skip is the only thing that prevents
        that — so a clean return AND an unfired sentinel are both required.
        """
        from tldr.daemon.ensure import ensure_daemon

        root = f"/tmp/tldr-ephemeral-guard-test-{os.getpid()}"
        try:
            os.makedirs(root, exist_ok=True)
            os.makedirs(os.path.join(root, ".git"), exist_ok=True)
            reached = _patch_popen_sentinel(monkeypatch)
            # Must return cleanly (no sentinel, no RuntimeError timeout).
            ensure_daemon(root)
            assert reached == [], (
                f"ensure_daemon on an ephemeral /tmp root must NOT reach Popen; "
                f"got {reached!r}. RED: no _is_ephemeral_root early-skip."
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_non_ephemeral_root_spawns_daemon(self, monkeypatch, tmp_path):
        """B5: a genuinely non-/tmp root WITH no live daemon DOES reach Popen.

        On macOS tmp_path is normally /private/var/folders (non-ephemeral); on
        Linux CI (and this fork's conftest basetemp) tmp_path can live under
        /tmp, so force the guard off for THIS normal-path arm via the env hatch.
        (The hatch's own behavior is asserted separately by B6.)
        """
        from tldr.daemon.ensure import ensure_daemon

        (tmp_path / ".git").mkdir(exist_ok=True)
        resolved = str(tmp_path.resolve())
        if resolved == "/tmp" or "/tmp/" in resolved or resolved.startswith("/private/tmp"):
            monkeypatch.setenv("TLDR_INDEX_EPHEMERAL", "1")

        reached = _patch_popen_sentinel(monkeypatch)
        with pytest.raises(_PopenReached):
            ensure_daemon(str(tmp_path))
        assert reached, "Popen must be reached on a non-ephemeral root with no daemon."

    def test_env_escape_hatch_spawns_on_tmp_root(self, monkeypatch):
        """B6: TLDR_INDEX_EPHEMERAL=1 forces a spawn even on a /tmp root."""
        from tldr.daemon.ensure import ensure_daemon

        root = f"/tmp/tldr-ephemeral-hatch-test-{os.getpid()}"
        try:
            os.makedirs(root, exist_ok=True)
            os.makedirs(os.path.join(root, ".git"), exist_ok=True)
            monkeypatch.setenv("TLDR_INDEX_EPHEMERAL", "1")
            reached = _patch_popen_sentinel(monkeypatch)
            with pytest.raises(_PopenReached):
                ensure_daemon(root)
            assert reached, (
                "With TLDR_INDEX_EPHEMERAL=1 the /tmp guard must be bypassed so "
                "Popen is reached."
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


class TestIsEphemeralRootPredicate:
    """B7 + edge cases: _is_ephemeral_root resolves /tmp symlink then tests
    self-or-ancestor membership."""

    def test_tmp_symlink_subdir_is_ephemeral(self):
        # B7: /tmp/<sub> resolves through the macOS /tmp -> /private/tmp symlink.
        from tldr.daemon.ensure import _is_ephemeral_root

        root = f"/tmp/tldr-ephemeral-predicate-{os.getpid()}"
        try:
            os.makedirs(root, exist_ok=True)
            assert _is_ephemeral_root(root) is True, (
                f"{root} resolves under /private/tmp and must be ephemeral."
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_private_tmp_exact_is_ephemeral(self):
        # Root IS the ephemeral base exactly (self-membership arm).
        from tldr.daemon.ensure import _is_ephemeral_root

        assert _is_ephemeral_root("/private/tmp") is True

    def test_home_root_is_not_ephemeral(self):
        from tldr.daemon.ensure import _is_ephemeral_root

        assert _is_ephemeral_root(str(os.path.expanduser("~"))) is False

    def test_escape_hatch_makes_root_non_ephemeral(self, monkeypatch):
        # Escape hatch is checked FIRST: a truthy env var → not ephemeral.
        from tldr.daemon.ensure import _is_ephemeral_root

        monkeypatch.setenv("TLDR_INDEX_EPHEMERAL", "1")
        assert _is_ephemeral_root("/tmp") is False
