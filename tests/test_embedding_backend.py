"""
Tests for tldr/embedding_backend.py:
  - EmbeddingBackend protocol
  - ServerClientBackend.encode() via IPC
  - InProcessBackend.encode() via monkeypatched get_model
  - get_default_backend() silent fallback when server unreachable

All tests are RED on HEAD because:
  1. tldr/embedding_backend.py does not exist — ImportError on every test
  2. EmbeddingBackend, ServerClientBackend, InProcessBackend,
     ServerUnavailableError, get_default_backend are not defined anywhere

Test strategy:
- ServerClientBackend: start a minimal in-process loopback responder on a
  Unix socketpair (or real tmp_path socket) that speaks the same JSON-newline
  framing as transport.py; assert encode() returns ndarray of correct shape
- InProcessBackend: monkeypatch tldr.semantic.get_model to return a fake model;
  assert encode() returns ndarray without touching real model weights
- get_default_backend: monkeypatch connection to raise ConnectionRefusedError;
  assert InProcessBackend is returned (silent fallback)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conftest import make_fake_model as _make_fake_model

_DIM = 4


# ---------------------------------------------------------------------------
# Helpers: minimal in-process echo-server for ServerClientBackend tests
# ---------------------------------------------------------------------------

def _start_fake_model_server(sock_path: str, dim: int = _DIM) -> threading.Thread:
    """Bind a Unix socket that handles embed requests and returns fake vectors.

    Protocol: receive JSON-newline {"cmd": "embed", "texts": [...], "request_id": ...}
               respond JSON-newline {"status": "ok", "vectors": [[...], ...], "request_id": ...}
    """

    # Bind synchronously BEFORE starting the serve thread so the socket file
    # exists the moment this function returns. Binding inside the thread created
    # a socket-appearance race that flaked under heavy parallel suite load.
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(sock_path)
    srv.listen(5)
    srv.settimeout(5.0)

    def _serve():
        try:
            while True:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    break
                try:
                    buf = b""
                    while b"\n" not in buf:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    if not buf:
                        continue
                    req = json.loads(buf.strip())
                    texts = req.get("texts", [])
                    n = len(texts)
                    vecs = (np.ones((n, dim), dtype=np.float32) / np.sqrt(dim)).tolist()
                    resp = {
                        "status": "ok",
                        "vectors": vecs,
                        "request_id": req.get("request_id", ""),
                    }
                    conn.sendall(json.dumps(resp).encode() + b"\n")
                except Exception:
                    pass
                finally:
                    conn.close()
        finally:
            srv.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


# ===========================================================================
# SECTION 1 — ServerClientBackend
# ===========================================================================


class TestServerClientBackendEncode:
    """ServerClientBackend.encode() talks to the model server over IPC.

    RED: tldr.embedding_backend does not exist — ImportError expected.
    """

    def test_encode_returns_ndarray_with_correct_shape(self, tmp_path: Path):
        """encode(texts) returns np.ndarray of shape (len(texts), dim).

        Uses an in-process fake server on a tmp_path Unix socket.

        RED reason: tldr.embedding_backend.ServerClientBackend does not exist.
        """
        from tldr.embedding_backend import ServerClientBackend  # noqa: F401

        sock_path = str(tmp_path / "test_server.sock")
        _start_fake_model_server(sock_path, dim=_DIM)

        # Wait for socket to appear
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if Path(sock_path).exists():
                break
            time.sleep(0.02)
        assert Path(sock_path).exists(), "Fake model server socket did not appear"

        backend = ServerClientBackend(server_socket=sock_path)
        texts = ["hello world", "foo bar baz"]
        result = backend.encode(texts)

        assert isinstance(result, np.ndarray), (
            f"encode() must return np.ndarray, got {type(result)}"
        )
        assert result.shape == (len(texts), _DIM), (
            f"encode() shape must be ({len(texts)}, {_DIM}), got {result.shape}"
        )

    def test_encode_raises_server_unavailable_when_socket_missing(self, tmp_path: Path):
        """encode() raises ServerUnavailableError when the server socket doesn't exist.

        RED reason: tldr.embedding_backend.ServerClientBackend does not exist.
        """
        from tldr.embedding_backend import ServerClientBackend, ServerUnavailableError  # noqa: F401

        missing_sock = str(tmp_path / "nonexistent.sock")
        backend = ServerClientBackend(server_socket=missing_sock)

        with pytest.raises(ServerUnavailableError):
            backend.encode(["hello"])

    def test_server_client_backend_implements_protocol(self, tmp_path: Path):
        """ServerClientBackend satisfies the EmbeddingBackend Protocol (has encode method).

        RED reason: tldr.embedding_backend does not exist.
        """
        from tldr.embedding_backend import ServerClientBackend, EmbeddingBackend  # noqa: F401

        sock_path = str(tmp_path / "protocol_test.sock")
        backend = ServerClientBackend(server_socket=sock_path)

        # Protocol structural check: encode must be callable with (texts, batch_size) sig
        assert callable(getattr(backend, "encode", None)), (
            "ServerClientBackend must have a callable .encode() method"
        )
        # Runtime isinstance check with Protocol (requires runtime_checkable)
        try:
            assert isinstance(backend, EmbeddingBackend), (
                "ServerClientBackend must be an instance of EmbeddingBackend Protocol"
            )
        except TypeError:
            # Protocol not runtime_checkable — that's fine; structural check above suffices
            pass


# ===========================================================================
# SECTION 2 — InProcessBackend
# ===========================================================================


class TestInProcessBackendEncode:
    """InProcessBackend.encode() delegates to get_model().encode() (existing path).

    RED: tldr.embedding_backend does not exist — ImportError expected.
    """

    def test_encode_returns_ndarray_from_fake_model(self):
        """encode() returns ndarray from the monkeypatched in-process model.

        RED reason: tldr.embedding_backend.InProcessBackend does not exist.
        """
        from tldr.embedding_backend import InProcessBackend  # noqa: F401

        fake_model = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            backend = InProcessBackend()
            result = backend.encode(["hello", "world", "test"])

        assert isinstance(result, np.ndarray), (
            f"encode() must return np.ndarray, got {type(result)}"
        )
        assert result.shape == (3, _DIM), (
            f"encode() shape must be (3, {_DIM}), got {result.shape}"
        )

    def test_encode_calls_get_model_encode(self):
        """InProcessBackend.encode() calls model.encode() with the texts.

        RED reason: tldr.embedding_backend.InProcessBackend does not exist.
        """
        from tldr.embedding_backend import InProcessBackend  # noqa: F401

        fake_model = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model) as mock_get:
            backend = InProcessBackend()
            texts = ["alpha", "beta"]
            backend.encode(texts)

        # get_model was called (possibly once on construction or on encode)
        assert mock_get.called or fake_model.encode.called, (
            "InProcessBackend.encode() must call tldr.semantic.get_model() "
            "and/or model.encode()"
        )
        # model.encode was called with the texts
        assert fake_model.encode.call_count >= 1, (
            "InProcessBackend.encode() must call model.encode() at least once"
        )
        call_args = fake_model.encode.call_args
        called_texts = call_args[0][0] if call_args[0] else call_args[1].get("texts")
        assert called_texts == texts or list(called_texts) == texts, (
            f"model.encode() must be called with the input texts. "
            f"Expected {texts!r}, got {called_texts!r}"
        )

    def test_in_process_backend_implements_protocol(self):
        """InProcessBackend satisfies the EmbeddingBackend Protocol (has encode method).

        RED reason: tldr.embedding_backend does not exist.
        """
        from tldr.embedding_backend import InProcessBackend, EmbeddingBackend  # noqa: F401

        fake_model = _make_fake_model()
        with patch("tldr.semantic.get_model", return_value=fake_model):
            backend = InProcessBackend()

        assert callable(getattr(backend, "encode", None)), (
            "InProcessBackend must have a callable .encode() method"
        )
        try:
            assert isinstance(backend, EmbeddingBackend)
        except TypeError:
            pass  # not runtime_checkable — structural check above suffices


# ===========================================================================
# SECTION 3 — get_default_backend silent fallback
# ===========================================================================


class TestGetDefaultBackendFallback:
    """get_default_backend() returns InProcessBackend when server is unreachable.

    RED: tldr.embedding_backend does not exist — ImportError expected.
    """

    def test_returns_server_client_backend_when_socket_reachable(self, tmp_path: Path):
        """get_default_backend() returns ServerClientBackend when server is up.

        RED reason: tldr.embedding_backend.get_default_backend does not exist.
        """
        from tldr.embedding_backend import (  # noqa: F401
            get_default_backend, ServerClientBackend,
        )

        sock_path = str(tmp_path / "reachable.sock")
        _start_fake_model_server(sock_path, dim=_DIM)

        deadline = time.time() + 3.0
        while time.time() < deadline:
            if Path(sock_path).exists():
                break
            time.sleep(0.02)
        assert Path(sock_path).exists()

        backend = get_default_backend(server_socket=sock_path)
        assert isinstance(backend, ServerClientBackend), (
            f"get_default_backend() must return ServerClientBackend when socket is reachable. "
            f"Got {type(backend).__name__}"
        )

    def test_returns_inprocess_backend_silently_when_socket_missing(self, tmp_path: Path):
        """get_default_backend() falls back to InProcessBackend silently — no exception.

        RED reason: tldr.embedding_backend.get_default_backend does not exist.
        """
        from tldr.embedding_backend import (  # noqa: F401
            get_default_backend, InProcessBackend,
        )

        missing_sock = str(tmp_path / "no_server_here.sock")

        # Must NOT raise; must silently return InProcessBackend
        backend = get_default_backend(server_socket=missing_sock)
        assert isinstance(backend, InProcessBackend), (
            f"get_default_backend() must silently fall back to InProcessBackend "
            f"when server socket is missing. Got {type(backend).__name__}"
        )

    def test_fallback_backend_still_returns_vectors(self, tmp_path: Path):
        """Fallback InProcessBackend returns vectors transparently — no exception surfaced.

        RED reason: tldr.embedding_backend.get_default_backend does not exist.
        """
        from tldr.embedding_backend import get_default_backend  # noqa: F401

        missing_sock = str(tmp_path / "gone.sock")
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model):
            backend = get_default_backend(server_socket=missing_sock)
            result = backend.encode(["fallback text"])

        assert isinstance(result, np.ndarray), (
            f"Fallback backend must return np.ndarray, got {type(result)}"
        )
        assert result.shape[0] == 1, (
            f"Fallback backend must return 1 vector for 1 text, got shape {result.shape}"
        )
        assert result.ndim == 2, (
            f"Fallback backend must return 2D array, got ndim={result.ndim}"
        )


# ===========================================================================
# SECTION 4 — _ensure_server (tldr/daemon/ensure.py)
# ===========================================================================


class TestEnsureServer:
    """ensure_server() starts the server on demand; idempotent if already up.

    RED: tldr.daemon.ensure does not exist — ImportError expected.
    """

    def test_ensure_server_spawns_popen_when_socket_absent(self, tmp_path: Path):
        """ensure_server() calls subprocess.Popen when no server socket is present.

        Monkeypatches: subprocess.Popen (no real process), ping_server (returns
        False then True to simulate startup).

        RED reason: tldr.daemon.ensure.ensure_server does not exist.
        """
        from tldr.daemon.ensure import ensure_server  # noqa: F401

        popen_calls = []

        class FakePopen:
            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})
            def poll(self):
                return None  # still running

        ping_results = iter([False, False, True])  # fails twice, then succeeds

        with patch("subprocess.Popen", FakePopen), \
             patch("tldr.daemon.ensure.ping_server", side_effect=ping_results):
            ensure_server(timeout=5.0)

        assert len(popen_calls) == 1, (
            f"ensure_server() must spawn exactly one process when socket is absent. "
            f"Got {len(popen_calls)} Popen calls."
        )

    def test_ensure_server_does_not_spawn_when_already_up(self, tmp_path: Path):
        """ensure_server() does NOT call Popen when server is already reachable.

        Monkeypatches ping_server to return True (fast path).

        RED reason: tldr.daemon.ensure.ensure_server does not exist.
        """
        from tldr.daemon.ensure import ensure_server  # noqa: F401

        popen_calls = []

        class FakePopen:
            def __init__(self, *args, **kwargs):
                popen_calls.append({"args": args, "kwargs": kwargs})

        with patch("subprocess.Popen", FakePopen), \
             patch("tldr.daemon.ensure.ping_server", return_value=True):
            ensure_server(timeout=5.0)

        assert len(popen_calls) == 0, (
            f"ensure_server() must NOT spawn a new process when server already responds. "
            f"Got {len(popen_calls)} Popen calls."
        )

    def test_ping_server_returns_false_for_missing_socket(self, tmp_path: Path):
        """ping_server() returns False (not exception) for a missing socket path.

        RED reason: tldr.daemon.ensure does not exist.
        """
        from tldr.daemon.ensure import ping_server  # noqa: F401

        missing = str(tmp_path / "no_server.sock")
        result = ping_server(missing)
        assert result is False, (
            f"ping_server() must return False for a missing socket, got {result!r}"
        )


# ===========================================================================
# SECTION 5 — Semantic backend injection (tldr/semantic.py refactor)
# ===========================================================================


def _build_tiny_py_repo(tmp_path: Path) -> Path:
    """Minimal Python project with .git anchor; used by semantic injection tests."""
    (tmp_path / ".git").mkdir(exist_ok=True)
    (tmp_path / "module.py").write_text(
        "def greet(name):\n    \"\"\"Return a greeting string.\"\"\"\n    return f'hello {name}'\n"
    )
    return tmp_path


class _SentinelBackend:
    """Minimal EmbeddingBackend stand-in that records calls and returns a sentinel array."""

    def __init__(self, dim: int = _DIM):
        self._dim = dim
        self.call_count = 0
        self.last_texts: list[str] = []

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        self.call_count += 1
        self.last_texts = list(texts)
        # Return a distinctive sentinel: all-twos (distinguishable from fake_model's all-ones)
        return np.full((len(texts), self._dim), 2.0, dtype=np.float32)


class TestSemanticBackendInjection:
    """Tests that tldr.semantic.compute_embedding and _encode_units (via
    build_semantic_index) accept an optional backend= kwarg per the architecture plan.

    All tests are RED now because:
      - compute_embedding does NOT accept a backend= kwarg → TypeError on call
      - build_semantic_index does NOT accept a backend= kwarg → TypeError on call
      - The EmbeddingBackend abstraction does not exist yet
    """

    # -----------------------------------------------------------------------
    # Test 1: compute_embedding default path preserves L2-normalisation
    # -----------------------------------------------------------------------

    def test_compute_embedding_default_path_is_l2_normalized(self):
        """compute_embedding() with no backend= arg returns an L2-normalized vector.

        Monkeypatches get_model to a fake whose encode returns a KNOWN
        un-normalized vector [3.0, 0.0, 4.0, 0.0] (L2 norm = 5.0).
        The default in-process path must normalize to norm ≈ 1.0.

        RED reason: compute_embedding does not yet accept a backend= kwarg;
        once the refactor lands the default InProcessBackend must apply
        normalize_embeddings=True so this assertion holds.
        """
        from tldr.semantic import compute_embedding  # noqa: F401

        # Fake model: encode returns a single un-normalized vector with norm=5
        unnormed = np.array([[3.0, 0.0, 4.0, 0.0]], dtype=np.float32)  # norm = 5
        fake_model = MagicMock()
        fake_model.encode.return_value = unnormed

        with patch("tldr.semantic.get_model", return_value=fake_model):
            # Call WITHOUT backend= kwarg — must use the default in-process path
            result = compute_embedding("some test text")

        assert isinstance(result, np.ndarray), (
            f"compute_embedding must return np.ndarray, got {type(result)}"
        )
        result_flat = result.flatten()
        norm = float(np.linalg.norm(result_flat))
        assert abs(norm - 1.0) < 1e-5, (
            f"compute_embedding default path must return L2-normalized vector "
            f"(norm ≈ 1.0). Got norm={norm:.6f}. "
            f"RED: once backend= is added, InProcessBackend must preserve normalize_embeddings=True."
        )

    # -----------------------------------------------------------------------
    # Test 2: compute_embedding uses injected backend instead of get_model
    # -----------------------------------------------------------------------

    def test_compute_embedding_uses_injected_backend(self):
        """compute_embedding("text", backend=...) uses the injected backend.

        Asserts:
        - The returned embedding equals the sentinel backend's output.
        - get_model is NOT called (in-process path bypassed).

        RED reason: compute_embedding does not accept a backend= kwarg → TypeError.
        """
        from tldr.semantic import compute_embedding  # noqa: F401

        sentinel_backend = _SentinelBackend(dim=_DIM)

        with patch("tldr.semantic.get_model") as mock_get_model:
            # Pass the backend= kwarg — must bypass get_model entirely
            result = compute_embedding("inject me", backend=sentinel_backend)

        assert mock_get_model.call_count == 0, (
            f"get_model must NOT be called when backend= is injected. "
            f"Got {mock_get_model.call_count} call(s)."
        )
        assert sentinel_backend.call_count == 1, (
            f"sentinel_backend.encode must be called exactly once. "
            f"Got {sentinel_backend.call_count} call(s)."
        )
        assert isinstance(result, np.ndarray), (
            f"compute_embedding must return np.ndarray, got {type(result)}"
        )
        # Sentinel returns all-twos; verify the output is derived from it
        assert np.all(result >= 1.9), (
            f"Result must come from sentinel backend (all-twos). Got: {result}"
        )

    # -----------------------------------------------------------------------
    # Test 3: build_semantic_index default path uses in-process get_model
    # -----------------------------------------------------------------------

    def test_encode_units_default_path_uses_inprocess(self, tmp_path: Path):
        """build_semantic_index() with NO backend= (None default) drives the in-process else branch.

        Calls build_semantic_index without a backend= kwarg so _encode_units
        takes the `else` branch: get_model() → model_obj.encode(...,
        normalize_embeddings=True, ...). Asserts:
        - get_model WAS called (proves the None/else in-process branch executed).
        - model.encode was called with normalize_embeddings=True (proves the
          in-process path requests normalization from the model).
        """
        from tldr.semantic import build_semantic_index  # noqa: F401

        project = _build_tiny_py_repo(tmp_path)
        fake_model = _make_fake_model()

        with patch("tldr.semantic.get_model", return_value=fake_model) as mock_get:
            # NO backend= kwarg — drives the else/None default branch
            count = build_semantic_index(
                str(project),
                lang="python",
                show_progress=False,
                respect_ignore=False,
            )

        assert mock_get.called, (
            "build_semantic_index with no backend= must call get_model "
            "(the None/else in-process branch must execute)."
        )
        assert count >= 0, f"build_semantic_index must return a non-negative count, got {count}"

        # Prove the else branch passed normalize_embeddings=True to model.encode.
        # _make_fake_model() records all calls on mock_model.encode; inspect kwargs.
        assert fake_model.encode.called, (
            "model.encode must have been called by the in-process else branch."
        )
        for call in fake_model.encode.call_args_list:
            kwargs = call[1] if call[1] else {}
            args = call[0] if call[0] else ()
            # normalize_embeddings may be positional or keyword depending on impl
            norm_kwarg = kwargs.get("normalize_embeddings")
            assert norm_kwarg is True, (
                f"The in-process else branch must call model.encode with "
                f"normalize_embeddings=True to preserve L2 normalization. "
                f"Got call kwargs: {kwargs!r}, args: {args!r}"
            )

    # -----------------------------------------------------------------------
    # Test 4: build_semantic_index uses injected backend instead of get_model
    # -----------------------------------------------------------------------

    def test_encode_units_uses_injected_backend(self, tmp_path: Path):
        """build_semantic_index(..., backend=...) uses the injected backend.

        Asserts:
        - sentinel_backend.encode is called (not get_model).
        - get_model is NOT called.

        RED reason: build_semantic_index does not accept a backend= kwarg → TypeError.
        """
        from tldr.semantic import build_semantic_index  # noqa: F401

        project = _build_tiny_py_repo(tmp_path)
        sentinel_backend = _SentinelBackend(dim=_DIM)

        with patch("tldr.semantic.get_model") as mock_get_model:
            build_semantic_index(
                str(project),
                lang="python",
                show_progress=False,
                respect_ignore=False,
                backend=sentinel_backend,
            )

        assert mock_get_model.call_count == 0, (
            f"get_model must NOT be called when backend= is injected into "
            f"build_semantic_index. Got {mock_get_model.call_count} call(s). "
            f"RED: build_semantic_index does not yet accept backend= kwarg."
        )
        assert sentinel_backend.call_count >= 1, (
            f"sentinel_backend.encode must be called at least once when injected. "
            f"Got {sentinel_backend.call_count} call(s). "
            f"RED: build_semantic_index does not yet accept backend= kwarg."
        )
