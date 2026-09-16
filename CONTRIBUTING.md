# Contributing to Verikan

Thanks for taking a look. Verikan is alpha and the internals still move; issues and pull
requests are welcome.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The full local setup — configuration, sign-in, Docker — is in the [README](README.md).

**The test suite is hermetic.** ~640 tests, under a minute, and it needs no API key, no
network and no `.env`: every outbound call is mocked, and a fixture points storage at a
temporary directory so a run never writes into your checkout. If a change you make can only be
tested with a live key or a running service, the test is at the wrong boundary — mock the
boundary, or make it a manual script.

## What CI enforces

| Check | Blocking | Notes |
|---|---|---|
| `pytest` | **Yes** | Must stay green on 3.12 and 3.14 |
| `ruff check src/ tests/ scripts/` | **Yes** | Currently clean — please keep it that way |
| `ruff format --check` | No — advisory | Pre-existing drift across many files |
| `mypy src/` | No — advisory | ~90 pre-existing errors, mostly missing annotations |
| `uv lock --check` | **Yes** | `uv.lock` must match `pyproject.toml` — no backlog, so it's a gate |

The advisory job reports so the debt stays visible, but it does not block your pull request.
Don't add *new* type errors; fixing ones you touch is welcome. **Please don't reformat files
you aren't otherwise changing** — a repo-wide `ruff format` would bury real changes in noise,
which is exactly why it isn't a gate.

The `lockfile` job is blocking for the same reason `ruff check` is: it passes today, so keeping
it passing costs nothing. Neither CI nor the Docker image installs from `uv.lock` — both use pip
— but `uv sync --frozen` is a [documented way to set up a clone](README.md#install-and-run), and
the lock has quietly rotted before, once leaving notebook verification with no Jupyter kernel.
If you change a dependency, run `uv lock` and commit the result alongside the `pyproject.toml`
edit. Editing `[tool.ruff]`, `[tool.mypy]` and the like does not affect the lock.

Pull requests also get an automated adversarial review (see [`.roborev.toml`](.roborev.toml))
that looks for this project's specific failure modes. It's a reviewer, not a gate — a finding
is a prompt to think, not an order.

## Things that will bite you

These are real invariants, not style preferences. Each has caused a production bug.

- **`GraphState` is a `TypedDict`, not a Pydantic model** — LangGraph requires it. Read keys
  with `.get()`; there are no defaults and no validators.
- **There are two graphs.** A deterministic one for Data Commons and an LLM-driven one for
  CKAN/MCP sources, chosen by `data_source`. A change to shared state, confidence or notebook
  generation usually has to be reasoned through on both.
- **`execution_trace` is what becomes the notebook.** Any action that retrieves, computes,
  visualises or cites must append to it, with its code snippet, or the published notebook is
  silently incomplete.
- **Generated notebook cells must be valid Python.** Codegen embeds tool arguments and
  retrieved text, so interpolate through `repr()`. One unparseable cell fails verification for
  the entire notebook.
- **A confidence factor that can't be computed is `None` with a reason** — never `0.0`. A hard
  zero is indistinguishable from a measured-bad score and misleads the user.
- **Route order in `gateway/router.py` matters.** Specific paths must come before wildcards
  like `/{query_id}`.
- **`/query` never surfaces a raw error.** Every failure path returns a friendly answer plus
  suggested follow-up questions.
- **The web UI binds JS to template element ids.** Change both sides together, and use theme
  tokens rather than hardcoded colours — light and dark both have to work.

## Pull requests

Keep them focused, explain the failing scenario a change fixes, and add a test when you fix a
bug. Commit messages here describe *why* in prose rather than following a strict format.

## Reporting security issues

Please don't open a public issue — see [SECURITY.md](SECURITY.md).
