"""Wiring tests: thin daemons actually delegate embedding to the shared server.

These lock in the integration the mocked unit tests missed — the seam
(`build_semantic_index`/`semantic_search`/`compute_embedding` accept `backend=`)
existed but nothing connected it, so the server was dead code in the real flow.

Covered:
- `get_server_backed_default` opt-in policy (force / env / silent fallback).
- daemon `_handle_semantic` routes BOTH index and search through a server-backed
  backend.
- the background-reindex subprocess opts into delegation via TLDR_USE_MODEL_SERVER.
- `semantic_search` accepts a `backend` kwarg.
"""

from __future__ import annotations

import inspect
import threading

import pytest


# --------------------------------------------------------------------------- #
# get_server_backed_default policy
# --------------------------------------------------------------------------- #

def test_default_is_inprocess_without_flag_or_force(monkeypatch):
    """No env flag + force=False → in-process, and NO server is started."""
    monkeypatch.delenv("TLDR_USE_MODEL_SERVER", raising=False)
    import tldr.embedding_backend as eb

    def _boom(*a, **k):  # ensure_server must NOT be called
        raise AssertionError("ensure_server should not run without flag/force")

    monkeypatch.setattr("tldr.daemon.ensure.ensure_server", _boom, raising=True)
    backend = eb.get_server_backed_default()
    assert isinstance(backend, eb.InProcessBackend)


def test_force_true_delegates_to_server(monkeypatch):
    """force=True → ensure_server is called and a server-backed backend returned."""
    import tldr.embedding_backend as eb

    calls = []
    monkeypatch.setattr(
        "tldr.daemon.ensure.ensure_server",
        lambda *a, **k: calls.append("ensure") or "/tmp/fake-model-server.sock",
        raising=True,
    )
    sentinel = object()
    monkeypatch.setattr(eb, "get_default_backend", lambda sock: (calls.append(sock), sentinel)[1])
    result = eb.get_server_backed_default(force=True)
    assert "ensure" in calls
    assert "/tmp/fake-model-server.sock" in calls
    assert result is sentinel


def test_env_flag_delegates_without_force(monkeypatch):
    """TLDR_USE_MODEL_SERVER set → delegate even when force=False."""
    import tldr.embedding_backend as eb

    monkeypatch.setenv("TLDR_USE_MODEL_SERVER", "1")
    seen = []
    monkeypatch.setattr(
        "tldr.daemon.ensure.ensure_server",
        lambda *a, **k: "/tmp/s.sock", raising=True,
    )
    monkeypatch.setattr(eb, "get_default_backend", lambda sock: seen.append(sock) or "B")
    assert eb.get_server_backed_default() == "B"
    assert seen == ["/tmp/s.sock"]


def test_silent_fallback_when_server_unstartable(monkeypatch):
    """ensure_server raising → silent fallback to InProcessBackend (never raises)."""
    import tldr.embedding_backend as eb

    monkeypatch.setenv("TLDR_USE_MODEL_SERVER", "1")

    def _raise(*a, **k):
        raise RuntimeError("cannot start server")

    monkeypatch.setattr("tldr.daemon.ensure.ensure_server", _raise, raising=True)
    backend = eb.get_server_backed_default(force=True)
    assert isinstance(backend, eb.InProcessBackend)


# --------------------------------------------------------------------------- #
# daemon _handle_semantic delegates (index + search)
# --------------------------------------------------------------------------- #

@pytest.fixture
def daemon(tmp_path):
    (tmp_path / ".git").mkdir()  # make it a project root
    from tldr.daemon.core import TLDRDaemon
    return TLDRDaemon(tmp_path)


def test_handle_semantic_index_passes_server_backend(daemon, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "tldr.embedding_backend.get_server_backed_default",
        lambda force=False: sentinel,
        raising=True,
    )
    captured = {}
    monkeypatch.setattr(
        "tldr.semantic.build_semantic_index",
        lambda project, **kw: captured.update(kw) or 7,
        raising=True,
    )
    resp = daemon._handle_semantic({"action": "index", "language": "python"})
    assert resp["status"] == "ok"
    assert captured.get("backend") is sentinel, "index must delegate via server backend"


def test_handle_semantic_search_passes_server_backend(daemon, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "tldr.embedding_backend.get_server_backed_default",
        lambda force=False: sentinel,
        raising=True,
    )
    captured = {}
    monkeypatch.setattr(
        "tldr.semantic.semantic_search",
        lambda project, query, **kw: captured.update(kw) or [],
        raising=True,
    )
    resp = daemon._handle_semantic({"action": "search", "query": "foo", "k": 3})
    assert resp["status"] == "ok"
    assert captured.get("backend") is sentinel, "search must delegate via server backend"


def test_handle_semantic_forces_delegation(daemon, monkeypatch):
    """The in-daemon path must call get_server_backed_default with force=True."""
    seen = {}
    monkeypatch.setattr(
        "tldr.embedding_backend.get_server_backed_default",
        lambda force=False: seen.update(force=force) or object(),
        raising=True,
    )
    monkeypatch.setattr("tldr.semantic.build_semantic_index", lambda p, **k: 0, raising=True)
    daemon._handle_semantic({"action": "index"})
    assert seen.get("force") is True


# --------------------------------------------------------------------------- #
# background reindex subprocess opts into delegation
# --------------------------------------------------------------------------- #

def test_background_reindex_subprocess_sets_model_server_env(daemon, monkeypatch):
    """The spawned `semantic index` subprocess must carry TLDR_USE_MODEL_SERVER=1."""
    captured = {}

    class _Result:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _Result()

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run, raising=True)

    # Run the reindex thread synchronously for a deterministic assertion.
    class SyncThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(threading, "Thread", SyncThread, raising=True)
    monkeypatch.setattr(daemon, "_read_index_epoch", lambda *a, **k: 0, raising=False)

    daemon._dirty_files = {"a.py"}
    daemon._reindex_in_progress = False
    daemon._trigger_background_reindex()

    assert captured.get("env") is not None, "subprocess.run must receive an explicit env"
    assert captured["env"].get("TLDR_USE_MODEL_SERVER") == "1"
    # sanity: it really is the semantic index command
    assert "semantic" in captured["cmd"] and "index" in captured["cmd"]


# --------------------------------------------------------------------------- #
# signature plumbing
# --------------------------------------------------------------------------- #

def test_semantic_search_accepts_backend_param():
    from tldr.semantic import semantic_search
    assert "backend" in inspect.signature(semantic_search).parameters
