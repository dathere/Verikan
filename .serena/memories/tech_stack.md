# Tech Stack

- Python `>=3.12`; CI matrix 3.12 + 3.14. `target-version = "py312"`, mypy `python_version = "3.12"`. Do not use 3.13+-only syntax.
- **3.11 was dropped deliberately.** `tests/unit/test_query_stream.py::test_cancelled_stream_reader_cancels_worker` deadlocks on 3.11 only: `gateway/query_stream.py::query_events` does `worker.cancel()` then `await worker` inside `anyio.CancelScope(shield=True)` in its `finally`, and on 3.11 that shielded await never completes when the *consumer* task is cancelled. Introduced by `fa50bfb` (PR #7). The suite passes on 3.12 and 3.14. If anyone proposes restoring 3.11, that deadlock is the blocker.
- 3.14 is in the matrix because the Dockerfile ships `python:3.14-slim`. Bump the two together or the deployed interpreter goes untested.
- **A blocking `docker` CI job builds the image and boots it** (`/health` + `/api/v1/health`). It exists because nothing built the image before: dependabot bumped the base to `python:3.14-slim` (#3) while `Dockerfile` still copied `python3.11/site-packages`, so every build — including the documented `docker compose up --build` — failed for six days on a path that no longer existed. The production stage copies site-packages and `src/` separately, so the build succeeding is not sufficient; the smoke test is the part that catches a bad copy.
- FastAPI + uvicorn + Jinja2 templates; Pydantic v2 (`pydantic-settings` for `core/config.py`).
- LangGraph for agent orchestration; `anthropic` SDK directly (no LangChain LLM wrappers).
- pandas / numpy for computation; `nbformat` + `nbclient` + `ipykernel` for notebook generation and execution.
- structlog for logging (`core/logging.py` `get_logger`); tenacity for retries; httpx (async) for all outbound HTTP.
- Optional at runtime, all degrade gracefully when unconfigured: Redis (cache), Pinecone (semantic dataset search), Auth0 (social login), GCS (shared storage), SMTP (admin mail), GitHub (publishing verified notebooks).

## Pins that matter
- **`pinecone>=5.0.0`, never `pinecone-client`.** The code uses the v3+ `Pinecone` class and the integrated-inference search API; the legacy package does not have them. A "fix" that swaps the dependency back breaks import and search.
- `ruff` `line-length = 100`.

## Two dependency manifests coexist — keep both current
- `pip install -e ".[dev]"` (`[project.optional-dependencies] dev`) is the **documented and enforced** path: README, CONTRIBUTING, `.github/workflows/ci.yml` and the `Dockerfile` (`pip install --no-cache-dir .`) all use it. The Dockerfile does not even `COPY uv.lock`.
- `uv.lock` (+ `[dependency-groups] dev`) is a **maintained, supported** second path for `uv sync --frozen`, even though no doc mentions it and no workflow runs it. Do not assume it is dead: commit `4b7a29f` ("Make a fresh clone actually runnable") deliberately regenerated it because it "was stale: it still pinned a removed dependency and was missing nbclient/ipykernel, so `uv sync --frozen` failed and notebook verification would have had no kernel."

**A dependency change must land in both.** After editing `[project.dependencies]` or the dev extra, run `uv lock` and commit the result.

Check locally with `uv lock --check` — read-only, writes neither the lockfile nor `.venv`.

CI enforces it in the **`lockfile` job — blocking**, a third gate alongside `test`'s `pytest` and `ruff check`. Same rationale as `ruff check`: no backlog, so keeping it clean is free.
- Its own job, not a step in `test`: the lock does not vary by Python version (so the 3.12/3.14 matrix would check it twice) and the check needs no project install, making it ~15s instead of a full dep install.
- Changes under `[tool.ruff]`, `[tool.mypy]`, etc. do not invalidate the lock — only dependency metadata does. False positives are unlikely.
- `.roborev.toml` still lists `uv.lock` in `exclude_patterns`, so the adversarial PR reviewer never reads the file; the gate is the only thing watching it.

`uv sync --frozen --extra dev` is documented in README § "Install and run" as a supported alternative to pip, so the lock now has acknowledged users rather than being a maintained-but-invisible artifact — which is what let it rot the first time.

**`--extra dev` is not optional there.** The dev toolchain is in `[project.optional-dependencies]`, which uv does not install by default; `[dependency-groups] dev` (which uv *does* install) holds only `respx`. Verified empirically: plain `uv sync --frozen` yields a runnable app with `ipykernel` and `respx` but **no pytest, ruff or mypy**, so a contributor set up that way cannot run the suite or the lint gate. `pip install -e ".[dev]"` has no such split.

## Not a Python dependency
`qsv` (<https://github.com/dathere/qsv>, same authors) is an external **CLI binary**, not a package — nothing in `pyproject.toml` installs it. `data_layer/qsv_profiling.py` shells out with `asyncio.create_subprocess_exec("qsv", ...)` for three passes behind portal onboarding: `describegpt`, `stats --everything`, `frequency --limit 20`.

There is no `shutil.which` preflight. A missing binary surfaces as a subprocess failure per pass, which is printed and tolerated — the pass returns empty and onboarding continues with degraded metadata rather than crashing. So "the data dictionary came out thin" is a plausible symptom of qsv simply not being installed.
