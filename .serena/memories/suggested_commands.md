# Commands

Assume an activated venv at `.venv` and `pip install -e ".[dev]"` already done.

## Run the app
```bash
./run_web.sh            # the real app; exports PYTHONPATH=$PWD/src, runs python -m data_concierge.ui.web
```
Serves on `$PORT`, default 8501. `run_web.sh` prefers `venv/` over `.venv/` if both exist.

`ModuleNotFoundError: data_concierge` means the editable install is missing — reinstall, or set `PYTHONPATH="$(pwd)/src"`.

Docker: `docker compose up --build` (port 8080). A full local CKAN stack is behind a profile: `docker compose --profile fairstore up` — not needed for normal work.

## Tests
```bash
pytest -q --no-cov      # what CI runs
```
Two traps:
- **Always run from the repo root.** `mcp/registry.py` sets `MCP_CONFIG_DIR = Path.cwd() / "configs"` at module scope, and several tests read it. This is why `.github/workflows/ci.yml` carries a comment forbidding `working-directory`.
- **`addopts` in `pyproject.toml` includes `--cov=src/data_concierge --cov-report=term-missing`**, so a bare `pytest` is slower than CI and prints a coverage table. Pass `--no-cov` to match CI.

The suite is hermetic: no API key, no network, no `.env`. It actively ignores `.env` — `core/config.py` reads that file for every setting, so `tests/conftest.py` sets `DATA_CONCIERGE_ENV_FILE=""` before anything imports the `settings` singleton. Assert shipped defaults via `Settings.model_fields["x"].default`, never the live `settings`, which still reflects the process environment (a shell `FOO=bar pytest` bypasses the conftest guard). `asyncio_mode = "auto"`, so `async def` tests need no marker. Notebook-verification tests really execute notebooks (the venv's own ipykernel), so a full run is slower than a typical unit suite.

## Lint / format / types
```bash
ruff check src/ tests/ scripts/    # blocking in CI; keep clean
ruff format src/ tests/ scripts/   # advisory — see mem:conventions before running repo-wide
mypy src/                          # advisory
```

## Darwin notes
- macOS `sed -i` needs a backup-suffix argument (`sed -i ''`); prefer Serena's edit tools anyway.
- `grep -P` is unavailable in BSD grep; use `grep -E` or ripgrep.
