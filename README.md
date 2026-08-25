# Verikan

**The verified Data Concierge** — ask a question about public statistical data in plain
English, get a citation-backed answer *and* the reproducible Jupyter notebook that derives it.

Verikan is a FastAPI + LangGraph application. It classifies your question, finds the right
dataset on an open data portal or federal API, loads and analyses the data, and emits a
self-contained Colab-ready notebook showing exactly how the answer was produced. Every
generated notebook is then **executed** and **adversarially reviewed** before its confidence
score is shown to you.

```
"How many building permits were issued in Pittsburgh in 2024?"
        │
        ├─ finds the dataset, loads it, runs the query
        ├─ writes the answer with citations
        ├─ generates a runnable .ipynb
        └─ executes + reviews that notebook, then scores its own confidence
```

---

## What makes it different

**The notebook is the evidence.** Most AI data tools give you an answer and ask you to trust
it. Verikan publishes the derivation and then checks its own work:

- **Execution** — the generated notebook is run in a contained subprocess. If the code does
  not execute, that is a measured failure, not a footnote.
- **Reconciliation** — the numeric claims in the answer must appear in the notebook's *own*
  output. Re-deriving a figure from published code is much stronger evidence than finding it
  in the transcript the model already had in front of it.
- **Adversarial method review** — a separate model reviews the notebook the way a skeptical
  reviewer would: is anything hardcoded that should be computed? Does the filter select the
  right subset? Does the citation match the data actually queried?

Those three signals merge into a single confidence factor. A factor that *could not* be
measured is reported as unavailable with a reason — never silently substituted with a zero
or a plausible-looking constant.

**Follow-ups edit the notebook.** Ask for a change ("use a different dataset", "that number
looks wrong") and Verikan edits the existing notebook cell by cell rather than starting over,
then puts the revision through the same execution and review pass.

---

## Quick start

### Prerequisites

- **Python 3.11 or newer**
- An **Anthropic API key** for running live queries ([console.anthropic.com](https://console.anthropic.com/))

No other API key is required to get a working local instance. The default data source
(WPRDC — Pittsburgh open data) needs no credentials at all.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/dathere/Verikan.git
cd Verikan
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run

```bash
./run_web.sh
```

or equivalently:

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python -m data_concierge.ui.web
```

The server listens on **http://localhost:8501** (override with the `PORT` env var).

| URL | What it is |
|---|---|
| `http://localhost:8501/` | Chat interface |
| `http://localhost:8501/admin` | Admin panel — submissions, logs, notebook reviews, settings |
| `http://localhost:8501/library` | Verified notebook library |
| `http://localhost:8501/docs` | User guide |
| `http://localhost:8501/api/docs` | FastAPI Swagger UI |

### 5. Log in

A fresh install seeds one local account:

- **Username:** `user`
- **Password:** `datHere@123`, or whatever you set as `USER_PASSWORD` in `.env` before the
  first start

> The seeded password is a local-development convenience. Set `USER_PASSWORD` to something
> of your own before exposing the app to anyone else.

**To get admin access** (the `/admin` panel, notebook review, settings), name your account in
`.env` *before the first run*, then start the app:

```env
ADMIN_USERS=user
USER_PASSWORD=choose-your-own
```

`ADMIN_USERS` accepts a comma-separated list of usernames or emails. It only seeds on first
run — after that, roles live in `roles.json` and are managed from the admin panel.

---

## Configuration

Everything is read from `.env` via Pydantic Settings (`src/data_concierge/core/config.py`).
**Only `ANTHROPIC_API_KEY` matters to get started** — every other integration degrades
gracefully when its key is absent.

### Data sources

| Variable | Enables | Without it |
|---|---|---|
| `ANTHROPIC_API_KEY` | All analysis | UI loads; queries cannot run |
| *(none needed)* | **WPRDC / CKAN open data portals** | — works out of the box |
| `BLS_API_KEY` | Bureau of Labor Statistics | Capped at 25 requests/day instead of 500 |
| `CENSUS_API_KEY` | Census Bureau | Census queries unavailable |
| `BEA_API_KEY` | Bureau of Economic Analysis (GDP) | BEA queries unavailable |
| `FRED_API_KEY` | Federal Reserve Economic Data | FRED queries unavailable |
| `DATA_COMMONS_API_KEY` | Google Data Commons | Data Commons queries unavailable |
| `PINECONE_API_KEY` | Semantic search over CKAN metadata | Falls back to keyword search |

### Optional services

| Variable | Purpose | Without it |
|---|---|---|
| `REDIS_HOST` | Response caching | App runs uncached; connection errors are logged and ignored |
| `AUTH0_*` | Social login | Local username/password login only |
| `GITHUB_TOKEN`, `GITHUB_REPO` | Publishing verified notebooks to a repo | Publishing disabled |
| `SMTP_*` | Admin email notifications | Notifications disabled |

### Behaviour flags

| Variable | Default | Effect |
|---|---|---|
| `NOTEBOOK_VERIFICATION_ENABLED` | `true` | Execute each generated notebook and reconcile its output |
| `NOTEBOOK_REVIEW_ENABLED` | `true` | Adversarial method review of each notebook |
| `FOLLOWUP_LLM_ENABLED` | `true` | Understand chat follow-ups / notebook revision requests |
| `LLM_MODEL` | `claude-sonnet-5` | Main analysis model |

Notebook verification runs generated code. It is contained — a subprocess with a hard
timeout, a minimal environment allowlist that withholds every credential, shell-escape cells
skipped, and an egress guard that blocks loopback, private, link-local and cloud-metadata
addresses from inside the kernel. Set `NOTEBOOK_VERIFICATION_ENABLED=false` to turn it off.

---

## Running with Docker

```bash
cp .env.example .env      # add your ANTHROPIC_API_KEY
docker compose up --build
```

The containerised app is served on **http://localhost:8080**.

### Optional: a local CKAN portal

A full CKAN stack (for developing against a local open data portal) ships behind a profile:

```bash
docker compose --profile fairstore up
```

This starts CKAN, PostgreSQL and Solr alongside the app. It is not needed for normal
development — the public WPRDC portal works without it.

---

## Project layout

```
src/data_concierge/
  ui/web.py                  # Primary entrypoint — FastAPI + Jinja2 web app
  api/main.py                # Secondary REST-only app
  gateway/
    router.py                # All /api/v1 endpoints
    intent_classifier.py     # Regex intent + complexity classification
    notebook_verification.py # Schedules notebook execution + review, merges into confidence
    followup.py              # Chat follow-up classifier (new question vs notebook revision)
    verified_notebooks.py    # Verified notebook library
    evidence.py              # Typed Standards evidence packages
    session.py, chats.py     # Session and chat storage
  agents/
    supervisor.py            # Builds both LangGraph graphs; routes by data source
    llm_agent.py             # LLM-driven agent for CKAN / MCP sources (tool calling)
    query_parser.py          # Entity extraction for the deterministic graph
    data_finder.py           # Data Commons retrieval
    stats_computer.py        # Statistical computation
    viz_builder.py           # Vega-Lite visualisation specs
    citation_builder.py      # Citation formatting
    notebook_generator.py    # Colab-ready .ipynb generation from the execution trace
    notebook_verifier.py     # Executes a notebook, reconciles output vs answer
    notebook_reviewer.py     # Adversarial method review
    notebook_editor.py       # Edits an existing notebook per a chat follow-up
  data_layer/connectors/     # Data Commons, BLS, Census, BEA, FRED, CKAN, Pinecone
  mcp/                       # Model Context Protocol client, registry, connector
  core/
    config.py                # All settings
    confidence.py            # Multi-factor confidence scoring
    models.py                # Pydantic models
configs/                     # Data source registry, MCP server definitions
tests/unit, tests/integration
examples/                    # Sample generated notebooks
```

### How a query flows

Two LangGraph graphs, selected by data source:

- **LLM-driven graph** (CKAN / WPRDC / MCP sources) — Claude drives retrieval with tool
  calling: search datasets, inspect them, load rows, run SQL, then write the answer.
- **Deterministic graph** (Data Commons) — parse entities → route → retrieve → compute →
  visualise → cite → generate notebook.

Both end at notebook generation, confidence scoring, and the async verification + review
pass. Every agent action appends to an `execution_trace`, and that trace is what the notebook
generator turns into runnable cells — which is why the notebook genuinely reproduces the
analysis rather than describing it.

---

## Development

```bash
pytest                                  # run the test suite
ruff check src/ tests/ scripts/         # lint
ruff format src/ tests/ scripts/        # format
mypy src/                               # type check
```

Tests are plain pytest with `asyncio_mode = "auto"` — async tests need no decorator. The
suite uses real classifiers and state objects rather than mocks, and notebook-verification
tests actually execute notebooks in-process, so a full run takes a little longer than a
typical unit suite.

Storage is isolated for tests: a session-scoped fixture redirects the storage backend at a
temporary directory so a test run never writes into your checkout.

### Notes for contributors

- `GraphState` (`agents/state.py`) is a `TypedDict`, not a Pydantic model — LangGraph
  requires this. Read keys with `.get()`.
- Any agent action that retrieves, computes, visualises or cites **must** append to
  `state["execution_trace"]`, or the generated notebook will be silently incomplete.
- Route order in `gateway/router.py` is load-bearing: specific paths must be declared before
  wildcards such as `/{query_id}`.
- `/query` never surfaces a raw exception — every failure path returns a friendly answer plus
  suggested follow-up questions.
- A confidence factor that cannot be computed is `None` with a reason, never `0.0`.
- The web UI is token-based and supports light and dark themes; JavaScript in
  `ui/static/js/` binds to element ids in the Jinja templates, so change both sides together.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'data_concierge'`**
Run `pip install -e ".[dev]"` inside the activated virtual environment, or set
`PYTHONPATH="$(pwd)/src"`.

**Queries fail but the UI loads**
Expected without `ANTHROPIC_API_KEY`. Set it in `.env` and restart.

**Redis connection errors on startup**
Redis is optional and `REDIS_HOST` defaults to `localhost`, so the app keeps trying to reach
one. Either start a local instance (`docker run -p 6379:6379 redis`) or ignore the log lines
— caching is skipped and everything else works.

**BLS returns rate-limit errors**
Register a free key at [bls.gov/developers](https://www.bls.gov/developers/) and set
`BLS_API_KEY` to raise the limit from 25 to 500 requests/day.

**Notebook verification never completes**
It runs as a background task after the answer is returned. On a laptop this takes seconds to
minutes depending on the notebook. Set `NOTEBOOK_VERIFICATION_ENABLED=false` to skip it
during development.

---

## A note on issue references

Comments and docstrings throughout the code cite issue numbers like `(#131)` or `issue #104`.
Those refer to the private tracker this project was developed in before it was open sourced —
they are kept because they mark *why* a piece of code exists, but they do not correspond to
issues in this repository.

---

## License

MIT — see [LICENSE](LICENSE).
