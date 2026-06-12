# llm-tldr

Token-efficient code-analysis CLI (`tldr`). Python ≥3.10, setuptools. Package in `tldr/`; the daemon lives in `tldr/daemon/`, the shared embedding model server in `tldr/model_server/`.

## Commands

- Editable install: `pip install -e .`
- Full test suite: `pytest tests/` — run at most once per session (see below)
- Change-scoped tests: `tldr change-impact --git --run`, or `pytest -k <expr>` / `pytest --lf`

## Testing gotchas

- Don't loop the full suite. Tests share one GPU-backed model server, and repeated full runs leak daemons. Run it once, then narrow with change-impact / `-k` / `--lf`.
- Aborted runs leak processes: sweep `/tmp/tldr-test-msrv-*` model-server sockets and detached `tldr-pt-<uid>` e2e daemons under `/private/tmp`. The conftest reaper only fires on a clean pytest sessionfinish.
- A stray `/private/tmp/.tldr` directory hijacks `_find_project_root` for anything indexing under /tmp → silently wrong index. Remove it, and `git init` any temp repo used in index tests.
- `tests/conftest.py` pins SIGCHLD per-module (`_SIGCHLD_AUTOREAP_MODULES`); don't install process-wide SIGCHLD handlers in tests.

## Code gotchas

- When adding a language, grep every live `ext_map` (there are several independent ones — `tldr/api.py`, `tldr/hybrid_extractor.py`); missing one yields a partially-working language.

## Environment gotchas

- Never kill `aqm` processes (launchd-parented daemon or shell-parented TUI) even when they're spawning tldr indexers — they're the user's live tooling. Only reap tldr children that aqm spawned, and check ppid+argv first.
- The Bash sandbox silently drops kill signals to external PIDs (kill returns 0, process survives); cleanup of leaked daemons needs `dangerouslyDisableSandbox`.

## Workflow

- Base branches on `upstream/main` (parcadei/llm-tldr), not `origin/main`.
- Test files stay on this fork — exclude them from PRs sent upstream.
