# Tech Stack

- Python `>=3.11`; CI matrix 3.11 + 3.12. `target-version = "py311"`, mypy `python_version = "3.11"`. Do not use 3.12+-only syntax.
- FastAPI + uvicorn + Jinja2 templates; Pydantic v2 (`pydantic-settings` for `core/config.py`).
- LangGraph for agent orchestration; `anthropic` SDK directly (no LangChain LLM wrappers).
- pandas / numpy for computation; `nbformat` + `nbclient` + `ipykernel` for notebook generation and execution.
- structlog for logging (`core/logging.py` `get_logger`); tenacity for retries; httpx (async) for all outbound HTTP.
- Optional at runtime, all degrade gracefully when unconfigured: Redis (cache), Pinecone (semantic dataset search), Auth0 (social login), GCS (shared storage), SMTP (admin mail), GitHub (publishing verified notebooks).

## Pins that matter
- **`pinecone>=5.0.0`, never `pinecone-client`.** The code uses the v3+ `Pinecone` class and the integrated-inference search API; the legacy package does not have them. A "fix" that swaps the dependency back breaks import and search.
- `ruff` `line-length = 100`.

## Two dependency manifests coexist
- `pyproject.toml` `[project.optional-dependencies] dev` — what README, CONTRIBUTING and CI all install: `pip install -e ".[dev]"`.
- `uv.lock` + `[dependency-groups] dev` — present, but nothing in CI or the docs uses uv.

Treat `pip install -e ".[dev]"` as canonical (it is what CI runs); keep `uv.lock` in mind if a dependency change needs to land in both.

## Not a Python dependency
`qsv` (<https://github.com/dathere/qsv>, same authors) is an external **CLI binary**, not a package — nothing in `pyproject.toml` installs it. `data_layer/qsv_profiling.py` shells out with `asyncio.create_subprocess_exec("qsv", ...)` for three passes behind portal onboarding: `describegpt`, `stats --everything`, `frequency --limit 20`.

There is no `shutil.which` preflight. A missing binary surfaces as a subprocess failure per pass, which is printed and tolerated — the pass returns empty and onboarding continues with degraded metadata rather than crashing. So "the data dictionary came out thin" is a plausible symptom of qsv simply not being installed.
