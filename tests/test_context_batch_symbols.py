"""RED tests: batch `tldr context foo bar baz` + single-symbol byte-identity.

Covers architecture behaviors:
 12. batch context prints one self-identifying block per resolved symbol to
     stdout, in argument order, separated by one blank line
 13. each unknown symbol in a batch produces a per-symbol stderr line with the
     did-you-mean suggestion; exit 0 when at least one symbol resolved
 14. batch with ALL symbols unknown exits 1
 15. single-symbol output/channels/exit codes are byte-identical to today —
     including the A-3 pin: the miss message names the BARE symbol with no
     Python list-repr artifacts (an accidental `args.entry` list passthrough
     would print "Function '['foo']' not found ...")

Pre-implementation, every multi-symbol invocation fails argparse with
"unrecognized arguments" (exit 2) — the RED signal. The two single-symbol
tests are control invariants pinning today's contract; they pass at RED time
by design and guard the GREEN dispatch rewrite (especially A-3).
"""

import signal
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)

_APP_PY = '''\
def helper_fn():
    """Shared helper used by main_fn."""
    return 1


def other_fn():
    """Second standalone function."""
    return 2


def main_fn():
    """Entry point calling both helpers."""
    return helper_fn() + other_fn()
'''


def _header(name: str) -> str:
    # The existing self-identifying block header (api.py to_llm_string).
    return f"## Code Context: {name} (depth=2)"


@pytest.fixture(scope="module")
def ctx_proj(tmp_path_factory) -> Path:
    proj = tmp_path_factory.mktemp("ctxbatch") / "proj"
    (proj / ".git").mkdir(parents=True)
    (proj / "app.py").write_text(_APP_PY, encoding="utf-8")
    return proj


def _run_context(entries: list[str], proj: Path) -> subprocess.CompletedProcess:
    # conftest.py sets SIGCHLD=SIG_IGN; Popen.wait() then treats ECHILD as
    # exit 0, masking any non-zero returncode.  Save/restore SIG_DFL so that
    # exit-code assertions are reliable regardless of the test runner's state.
    _old = signal.getsignal(signal.SIGCHLD)
    try:
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        return subprocess.run(
            [sys.executable, "-m", "tldr.cli", "context", *entries,
             "--project", str(proj)],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=120,
        )
    finally:
        signal.signal(signal.SIGCHLD, _old)


def _assert_no_list_repr(stream: str) -> None:
    # A-3: an args.entry (list) passthrough renders as "['name']" / '["name"]'.
    assert "['" not in stream and '["' not in stream, (
        f"Python list-repr artifact leaked into output: {stream!r}"
    )


class TestBatchContext:
    """Behaviors 12-14: multi-symbol context."""

    def test_batch_prints_one_block_per_symbol_in_argument_order(self, ctx_proj):
        result = _run_context(["main_fn", "helper_fn", "other_fn"], ctx_proj)
        assert result.returncode == 0, (
            f"exit {result.returncode}, stderr: {result.stderr!r}"
        )
        out = result.stdout
        for name in ("main_fn", "helper_fn", "other_fn"):
            assert _header(name) in out, (
                f"missing block header for {name}. stdout: {out!r}"
            )
        assert (
            out.index(_header("main_fn"))
            < out.index(_header("helper_fn"))
            < out.index(_header("other_fn"))
        ), f"blocks must follow argument order. stdout: {out!r}"

    def test_batch_blocks_separated_by_one_blank_line(self, ctx_proj):
        result = _run_context(["main_fn", "helper_fn"], ctx_proj)
        assert result.returncode == 0, (
            f"exit {result.returncode}, stderr: {result.stderr!r}"
        )
        assert "\n\n" + _header("helper_fn") in result.stdout, (
            f"expected one blank line before the second block. "
            f"stdout: {result.stdout!r}"
        )

    def test_partial_miss_goes_to_stderr_with_did_you_mean_and_exit_0(
        self, ctx_proj
    ):
        result = _run_context(["main_fn", "helper_fnx"], ctx_proj)
        assert result.returncode == 0, (
            f"at least one symbol resolved -> exit 0, got {result.returncode}. "
            f"stderr: {result.stderr!r}"
        )
        assert _header("main_fn") in result.stdout
        assert _header("helper_fnx") not in result.stdout
        assert "helper_fnx" in result.stderr and "not found" in result.stderr, (
            f"per-symbol miss must land on stderr: {result.stderr!r}"
        )
        assert "Did you mean" in result.stderr and "helper_fn" in result.stderr, (
            f"per-symbol did-you-mean missing: {result.stderr!r}"
        )
        _assert_no_list_repr(result.stderr)

    def test_all_symbols_unknown_exits_1(self, ctx_proj):
        result = _run_context(["zzqx_one", "zzqx_two"], ctx_proj)
        assert result.returncode == 1, (
            f"all-miss batch must exit 1, got {result.returncode}. "
            f"stderr: {result.stderr!r}"
        )
        assert "zzqx_one" in result.stderr and "zzqx_two" in result.stderr, (
            f"every miss must produce its own stderr message: {result.stderr!r}"
        )
        _assert_no_list_repr(result.stderr)


class TestSingleSymbolByteIdentity:
    """Behavior 15 (control invariants): N=1 contract is unchanged."""

    def test_single_symbol_hit_unchanged(self, ctx_proj):
        # Control: hit -> stdout starts with the block header, exit 0. Passes
        # today; post-GREEN it pins the args.entry[0] (not args.entry) path.
        result = _run_context(["main_fn"], ctx_proj)
        assert result.returncode == 0, (
            f"exit {result.returncode}, stderr: {result.stderr!r}"
        )
        assert result.stdout.startswith(_header("main_fn")), (
            f"single-symbol stdout must start with the block header. "
            f"stdout: {result.stdout[:120]!r}"
        )
        assert "not found" not in result.stdout
        _assert_no_list_repr(result.stdout)

    def test_single_symbol_miss_unchanged_no_list_repr(self, ctx_proj):
        # Control + A-3: miss -> stderr names the BARE symbol, exit 1, and
        # carries no list-repr artifact ("Function '['main_fnx']' not found"
        # is the classic regression this pins down).
        result = _run_context(["main_fnx"], ctx_proj)
        assert result.returncode == 1, (
            f"exit {result.returncode}, stdout: {result.stdout!r}"
        )
        assert "Function 'main_fnx' not found" in result.stderr, (
            f"miss must name the bare symbol on stderr: {result.stderr!r}"
        )
        _assert_no_list_repr(result.stderr)
        assert _header("main_fnx") not in result.stdout
