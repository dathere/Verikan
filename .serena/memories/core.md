# Verikan — Core

## Three names for one thing
- Repo / product: `Verikan`
- Distribution (pyproject `name`): `data-concierge`
- Import package: `data_concierge`, under `src/` (setuptools `where = ["src"]`)

Grep for any of the three when tracing something; docs and code disagree on which is used.

## Entrypoints
- `ui/web.py` — the real app (FastAPI + Jinja2, chat + `/admin` + `/library` + `/docs`). `python -m data_concierge.ui.web`, port from `$PORT`, default 8501.
- `api/main.py` — secondary REST-only app. Not what `run_web.sh`/Docker start. Changes to behaviour usually belong in the web app; check whether the REST app needs the same change.
- `gateway/router.py` — every `/api/v1` endpoint, 4.4k lines, mounted by both apps.

## Storage is a module-level singleton over the repo checkout
- `data_layer/storage.py` ends with `storage = _create_storage()`, evaluated at **import** time.
- `LocalStorage` root is `_PROJECT_ROOT` — the repo checkout itself. Runtime state (`users.json`, `roles.json`, `chats/`, `query_logs/`, `verified_notebooks/`, `*_settings.json`) is written next to the source and gitignored by explicit root-anchored entries in `.gitignore`. Adding one back to the index is a defect.
- Backend choice reads `GCS_BUCKET` / `REQUIRE_SHARED_STORAGE` from live env inside `_create_storage()`, not at module scope, so `monkeypatch.setenv` works. `REQUIRE_SHARED_STORAGE=true` turns a GCS init/probe failure into a startup `RuntimeError` instead of a silent LocalStorage fallback (multi-instance deploys diverge otherwise).
- Because the singleton is built at import, tests cannot re-create it — they mutate `storage.root`. See `mem:task_completion`.

## Invariants
Project-specific invariants that have each caused a production bug are written down in two tracked files and are the authoritative statement — read them rather than a paraphrase:
- `CONTRIBUTING.md` § "Things that will bite you"
- `.roborev.toml` `review_guidelines` (longer; adds agent-log versioning, notebook containment, MCP URL guards, secret/config rules, async-client rules)

`mem:conventions` covers only what those files do *not* say.

## Further memories
- Languages, pins, and the two dependency-manifest systems in play: `mem:tech_stack`
- How to run the app/tests, and the cwd + `--no-cov` traps: `mem:suggested_commands`
- Enum-rendering constraint, lint/format/type policy, what not to reformat: `mem:conventions`
- Exact done-checks and the storage-isolation fixture: `mem:task_completion`
- `GraphState` key list, which `data_source` picks which of the two graphs, `execution_trace` / `agent_log` rules: `mem:agents/core`
- Route-order rule, runtime-state files, evidence/verification/publishing surface: `mem:gateway/core`
