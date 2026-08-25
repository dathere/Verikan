"""Admin-editable prompt/template store for the LLM agent and notebooks.

Lets admins **view and change** the system prompt that ``LLMAnalysisAgent``
(``agents/llm_agent.py``) sends to Claude, plus the boilerplate markdown that
``NotebookGeneratorAgent`` writes into every generated notebook. Five templates
are configurable:

* ``ckan_template`` — used for CKAN / WPRDC / admin-registered open-data
  portals (the default chat path, ``data_source='wprdc'``).
* ``mcp_template`` — used for MCP-backed sources (Census, FBI Crime, …).
* ``notebook_header_template`` — the title / "How to Use" markdown cell at the
  top of every generated notebook.
* ``notebook_results_template`` — the results/answer markdown cell near the
  bottom of every generated notebook.
* ``notebook_review_template`` — the system prompt for the adversarial
  adversarial method review that runs on every generated notebook (#131,
  ``agents/notebook_reviewer.py``). This template plays the role of
  the reviewer's guidelines: edit it to teach the reviewer
  new defect classes.

Each template is a Python ``str.format`` template. The agent fills in a fixed
set of **placeholders** at request time (portal name/URL, org filter, the list
of other portals, etc.); an admin editing the template keeps those ``{name}``
markers to preserve the dynamic behavior. Removing a placeholder simply drops
that dynamic block — it never breaks rendering. Adding an *unknown* placeholder
is rejected on save (and, defensively, falls back to the default at runtime).

Settings persist via the shared storage backend, same pattern as
``landing_page`` / ``github_settings``. Defaults reproduce the previously
hardcoded prompts verbatim, so behavior is unchanged until an admin edits
anything. An empty/blank submission for a template resets it to the default.
"""

from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

# Storage key for the system-prompt overrides (persisted across restarts).
_SYSTEM_PROMPT_KEY = "system_prompt_settings.json"

# Placeholders each template may reference. The agent always supplies all of
# them; unused ones are harmless. Editing a template to reference anything
# outside these sets is rejected on save.
CKAN_PLACEHOLDERS: tuple[str, ...] = (
    "portal_name",
    "portal_url",
    "org_block",
    "other_portals_block",
)
MCP_PLACEHOLDERS: tuple[str, ...] = (
    "portal_name",
    "portal_url",
    "description",
)
NOTEBOOK_HEADER_PLACEHOLDERS: tuple[str, ...] = (
    "query",
    "timestamp",
    "data_source",
    "sources",
)
NOTEBOOK_RESULTS_PLACEHOLDERS: tuple[str, ...] = (
    "answer",
    "confidence_block",
)

# Defaults — originally the prompts hardcoded in
# ``llm_agent._build_system_prompt`` / ``_build_mcp_system_prompt``, since
# extended with the reproducibility rules (#131: every cited number must be
# derived from retrieved data — the review will catch fabrication).
DEFAULT_CKAN_TEMPLATE = """You are an AI Data Concierge — an expert at finding, loading, \
and analyzing open government data to answer user questions.

You have access to the **{portal_name}** portal at {portal_url} as your primary source.
{org_block}{other_portals_block}
## Workflow
1. **Semantic Search FIRST** — call `semantic_search_resources` with a \
natural-language description of what you need. This uses vector search over \
pre-indexed CKAN metadata and understands synonyms (e.g. 'air quality' \
matches 'pollution', 'AQI', 'emissions'). It returns resource IDs directly, \
often skipping the need for steps 2-3.
2. **Keyword Search (fallback)** — if semantic search returns nothing or \
unavailable, use `search_datasets` for keyword search. Try the primary portal \
first, then other registered portals via `portal_id`.
3. **Inspect** (if needed) the dataset with `get_dataset_info` for details \
when semantic search didn't surface a clear resource ID.
4. **Load** a sample with `load_resource_data` (100 rows) to understand the \
schema and column names.
5. **Analyze** using `run_sql_query` for aggregations (COUNT, SUM, AVG, \
GROUP BY, ORDER BY, etc.). Always use double-quoted resource IDs as table \
names and double-quoted column names.
6. **Answer** the user's question with specific numbers, rankings, \
percentages, and cite the dataset by name and portal.

## SQL tips for CKAN DataStore
- Table name = resource_id in double quotes: `SELECT * FROM "uuid-here"`
- Column names with spaces or special chars need double quotes.
- **IMPORTANT: Most numeric columns are stored as TEXT in CKAN DataStore.** \
You MUST use explicit CAST for any numeric operations, comparisons, or WHERE \
filters on numeric-looking columns. Examples:
  - Comparison: `WHERE CAST("column" AS FLOAT) > 0`
  - Arithmetic: `CAST("col_a" AS FLOAT) / CAST("col_b" AS FLOAT) * 100`
  - Sorting: `ORDER BY CAST("column" AS FLOAT) DESC`
  - Aggregation: `SUM(CAST("column" AS FLOAT))`
  - Handle nulls/empty: `WHERE "column" IS NOT NULL AND "column" != ''`
- If a SQL query fails with a type error, fix the query by adding CAST and \
retry — **do not give up**.
- Standard SQL: GROUP BY, ORDER BY, COUNT(*), AVG(), SUM(), LIMIT.
- Keep queries under 30 000 rows for performance.

## Answer guidelines
- Lead with the key finding and specific numbers.
- Mention the dataset name and portal as the source.
- Note any data limitations or caveats.
- **Always try multiple search terms** before concluding data is unavailable. \
For example, if "air quality" returns nothing, also try "pollution", "AQI", \
"emissions", "environmental", etc.
- **Try different portals** if the primary one doesn't have what you need.
- **Provide partial answers** when possible — if you found related data but \
not the exact metric asked about, present what you found and explain the gap.
- **Never say you are unable to answer** without having tried at least 3 \
different search queries across available portals.
- If after exhaustive searching no relevant data exists, say so honestly, \
describe what you searched for, and suggest where the user could look next.

## Reproducibility rules (your work becomes a published notebook)
Every tool call you make is turned into a notebook cell, the notebook is \
re-executed, and an adversarial reviewer checks that your answer's numbers \
are DERIVED by the code — so:
- **Every number you cite must come from a tool result you actually \
retrieved in this conversation.** Prefer computing final figures with a SQL \
aggregate (`run_sql_query`) so the derivation is a single reproducible step.
- **Never estimate, approximate, or fill in a figure from memory.** If a \
needed value could not be retrieved, say exactly that — do not substitute \
an approximation, however plausible.
- If a lookup fails (a resource is missing, a query errors), either retry \
differently until it succeeds or exclude that figure from the answer. Do \
not build comparisons or trends on values whose retrieval failed.
"""

DEFAULT_MCP_TEMPLATE = """You are an AI Data Concierge — an expert at finding and analyzing \
U.S. federal statistical data to answer user questions.

You have access to the **{portal_name}** ({portal_url}) through a set of MCP tools.
{description}

## Workflow
1. **Discover** — if unsure which dataset to use, call `list-datasets` to see \
what's available.
2. **Resolve geography** — if the user names a place (city, county, state), \
call `resolve-geography-fips` to get the correct FIPS code and query syntax.
3. **Check geography** — for unfamiliar datasets, call `fetch-dataset-geography` \
to learn what geographic breakdowns are available.
4. **Fetch data** — call `fetch-aggregate-data` with the right dataset, year, \
variables/group, and geography.
5. **Answer** the user's question with specific numbers and cite the Census \
dataset (name, vintage year, geography level).

## Answer guidelines
- Lead with the key finding and the specific values.
- Always mention the dataset name (e.g. "ACS 1-Year Estimates, 2023") and \
geographic level as the source.
- Note any data limitations (suppressed cells, margins of error, coverage gaps).
- **Always try multiple approaches** before concluding data is unavailable. \
Try different datasets, vintage years, or geographic levels.
- **Provide partial answers** when possible — present what you found even if \
it's not the exact metric requested.
- **Never say you are unable to answer** without having tried at least 3 \
different queries or datasets.
- If after exhaustive searching no data is found, say so, describe what you \
tried, and suggest an alternative dataset or vintage.

## Reproducibility rules (your work becomes a published notebook)
Every tool call you make is turned into a notebook cell, the notebook is \
re-executed, and an adversarial reviewer checks that your answer's numbers \
are DERIVED by the retrieved data — so:
- **Every number you cite must come from a tool result you actually \
retrieved in this conversation.** Never estimate, approximate, or fill in \
a figure from memory, however plausible.
- **When the API already provides a computed figure (a `rates` field, a \
percentage, a margin of error), cite that value as-is.** Do not recompute \
it by hand — hand-derived rates have repeatedly disagreed with the source's \
own figures by an order of magnitude. If you must derive a number the API \
does not provide, state the formula and the exact retrieved inputs it uses.
- **Verify geography resolutions.** If `resolve-geography-fips` returns a \
different place than asked (or nothing), do not proceed with a guessed \
code — retry with a different spelling, or state that the lookup failed.
- If a fetch fails for a year or place, exclude it from tables and trends \
rather than substituting an approximation; mark it explicitly as \
unavailable.
"""

# Default notebook boilerplate — byte-identical to the markdown formerly
# hardcoded in ``notebook_generator._create_title_cell`` /
# ``_create_results_cell``, so generated notebooks are unchanged until an
# admin edits these.
DEFAULT_NOTEBOOK_HEADER_TEMPLATE = """# 🔬 Verikan — Reproducible Analysis

<a href="https://colab.research.google.com/" target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## 📋 Query
> **{query}**

## 📊 Metadata
| Property | Value |
|----------|-------|
| **Generated** | {timestamp} |
| **Data Source** | {data_source} |
| **Sources Used** | {sources} |
| **Notebook Version** | 1.0 (Colab Compatible) |

---

## 📖 How to Use This Notebook

This notebook reproduces the exact analysis performed by **Verikan**.
Follow the steps below to verify, modify, or extend the analysis.

### ✅ Quick Start
1. **Run All Cells**: Click `Runtime` → `Run all` (or press `Ctrl+F9`)
2. **Wait for Setup**: The first cells install dependencies and configure the environment

### 🔧 What You Can Do
| Action | Description |
|--------|-------------|
| **Verify** | Run all cells to confirm the original results |
| **Modify** | Change parameters (dates, locations, filters) and re-run |
| **Extend** | Add your own analysis cells below the results |
| **Export** | Download results as CSV, or save notebook to Drive |

### 📚 Notebook Structure
1. **Setup** - Install dependencies (runs once in Colab)
2. **Configuration** - Import libraries and set up API connections
3. **Data Retrieval** - Fetch data from the data source
4. **Analysis** - Process and analyze the data
5. **Results** - View the final answer and confidence scores
6. **Citations** - Reference sources for your research

---
"""

DEFAULT_NOTEBOOK_RESULTS_TEMPLATE = """# ============================================================
# 📊 RESULTS
# ============================================================

## 💡 Answer (as delivered in chat)

> **{answer}**

*The figures above restate the assistant's answer for convenience — they are
not computed by this cell. The derivation lives in the code cells earlier in
this notebook; re-run them to re-derive every number from the source APIs.*

---

## 🎯 Confidence Assessment

The Data Concierge evaluates the reliability of its answer using multiple factors:

{confidence_block}"""

# System prompt for the adversarial notebook method review (#131, third
# signal — adversarial review applied to generated notebooks). No placeholders:
# the question, answer, and notebook go in the user message at review time.
# NOTE: this is a ``str.format`` template like the others, so literal braces
# would need doubling — keep the guidelines brace-free.
DEFAULT_NOTEBOOK_REVIEW_TEMPLATE = """You are an adversarial reviewer of \
auto-generated Jupyter notebooks. Each notebook was produced by an AI data \
concierge to justify an answer about federal/civic statistical data. Your \
job is to try to REFUTE the notebook: assume the answer is wrong until the \
code convinces you otherwise.

Judge the METHOD only — do not execute anything, and do not penalise style.

Look specifically for these defect classes, worst first:

- Hardcoded answers: a number the answer claims appears as a literal in the \
code or a markdown cell instead of being computed from loaded data. This is \
the main hallucination surface — severity critical.
- Wrong computation: the code aggregates, filters, or joins in a way that \
answers a different question than the one asked (wrong column, wrong subset, \
wrong statistic, wrong time period, wrong geography). Severity critical or \
high depending on how far the result drifts.
- Unsupported claims: the answer asserts something no cell derives at all. \
Severity high.
- Fragile or misleading retrieval: the code queries a dataset or resource \
that plainly does not match what the citation or markdown says it is, or \
relies on a magic resource ID with no provenance. Severity medium.
- Statistical missteps: comparing incompatible units or vintages, averaging \
rates without weights, trend claims from two data points. Severity medium.
- Cosmetic drift: markdown text that contradicts the code around it. \
Severity low.

Do NOT report: missing error handling, style, performance, or the use of \
the portal APIs themselves. For MCP-backed sources, cells that embed the \
verbatim retrieved result and parse it with json.loads are the expected \
provenance mechanism — do not flag the embedding itself; flag figures in \
the answer that no cell's retrieved data supports. The final results \
markdown cell legitimately restates the shipped answer — flag it only when \
its figures are unsupported by the retrieval cells. A notebook with a sound \
method and no defects gets an empty findings list — do not invent findings \
to seem thorough.

Record your verdict with the record_review tool: a one-or-two sentence \
summary plus concrete findings, each with severity, title, detail (what the \
code does vs what the answer claims), and the offending cell index when \
there is one."""
NOTEBOOK_REVIEW_PLACEHOLDERS: tuple[str, ...] = ()

# Registry of every editable template: base key -> (default, placeholders).
# Storage / API field names derive from the base key: ``{base}_template``,
# ``{base}_is_custom``, ``default_{base}_template``, ``{base}_placeholders``.
TEMPLATE_REGISTRY: dict[str, tuple[str, tuple[str, ...]]] = {
    "ckan": (DEFAULT_CKAN_TEMPLATE, CKAN_PLACEHOLDERS),
    "mcp": (DEFAULT_MCP_TEMPLATE, MCP_PLACEHOLDERS),
    "notebook_header": (DEFAULT_NOTEBOOK_HEADER_TEMPLATE, NOTEBOOK_HEADER_PLACEHOLDERS),
    "notebook_results": (DEFAULT_NOTEBOOK_RESULTS_TEMPLATE, NOTEBOOK_RESULTS_PLACEHOLDERS),
    "notebook_review": (DEFAULT_NOTEBOOK_REVIEW_TEMPLATE, NOTEBOOK_REVIEW_PLACEHOLDERS),
}


def _validate_template(template: str, placeholders: tuple[str, ...]) -> None:
    """Raise ``ValueError`` if ``template`` can't render with these placeholders.

    Catches unbalanced braces, positional ``{}`` fields, and references to
    unknown ``{names}`` — anything that would make ``str.format`` throw when
    the agent renders the prompt at request time.
    """
    dummy = dict.fromkeys(placeholders, "")
    try:
        template.format(**dummy)
    except KeyError as e:  # {unknown_placeholder}
        raise ValueError(
            f"Unknown placeholder {e} — allowed placeholders: "
            + ", ".join("{" + p + "}" for p in placeholders)
        ) from e
    except (IndexError, ValueError) as e:  # bare {}/{0}, or stray/unbalanced brace
        raise ValueError(
            "Invalid template: use named placeholders like "
            + ", ".join("{" + p + "}" for p in placeholders)
            + f". Literal braces must be doubled (»{{{{«, »}}}}«). ({e})"
        ) from e


def load_system_prompt_settings() -> dict[str, Any]:
    """Return the effective templates plus the defaults + placeholder metadata.

    Shape (consumed by the admin UI), for every base key in
    ``TEMPLATE_REGISTRY``:
      ``{base}_template`` — effective (custom or default)
      ``{base}_is_custom`` — whether an override is stored
      ``default_{base}_template`` — for "reset"
      ``{base}_placeholders`` — allowed ``{names}``
    A read error degrades gracefully to the defaults.
    """
    try:
        saved = storage.read_json(_SYSTEM_PROMPT_KEY) or {}
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to read system-prompt settings; using defaults", error=str(e))
        saved = {}

    out: dict[str, Any] = {}
    for base, (default, placeholders) in TEMPLATE_REGISTRY.items():
        custom = saved.get(f"{base}_template") or None
        out[f"{base}_template"] = custom or default
        out[f"{base}_is_custom"] = bool(custom)
        out[f"default_{base}_template"] = default
        out[f"{base}_placeholders"] = list(placeholders)
    return out


def _get_template(base: str) -> str:
    """Effective template for a registry base key (custom override or default)."""
    return str(load_system_prompt_settings()[f"{base}_template"])


def get_ckan_template() -> str:
    """Effective CKAN prompt template (custom override or default)."""
    return _get_template("ckan")


def get_mcp_template() -> str:
    """Effective MCP prompt template (custom override or default)."""
    return _get_template("mcp")


def get_notebook_header_template() -> str:
    """Effective notebook title/how-to cell template."""
    return _get_template("notebook_header")


def get_notebook_results_template() -> str:
    """Effective notebook results cell template."""
    return _get_template("notebook_results")


def get_notebook_review_template() -> str:
    """Effective adversarial notebook-review system prompt."""
    return _get_template("notebook_review")


def save_system_prompt_settings(
    ckan_template: str | None = None,
    mcp_template: str | None = None,
    notebook_header_template: str | None = None,
    notebook_results_template: str | None = None,
    notebook_review_template: str | None = None,
) -> dict[str, Any]:
    """Validate + persist template overrides.

    ``None`` leaves a template untouched. A blank/whitespace string resets that
    template to its default (removes the stored override). A non-empty string
    is validated (must render with only the allowed placeholders) and stored.
    Returns the fully-merged settings for echoing back to the UI.
    """
    updates = {
        "ckan": ckan_template,
        "mcp": mcp_template,
        "notebook_header": notebook_header_template,
        "notebook_results": notebook_results_template,
        "notebook_review": notebook_review_template,
    }
    valid_keys = {f"{base}_template" for base in TEMPLATE_REGISTRY}

    try:
        existing = storage.read_json(_SYSTEM_PROMPT_KEY) or {}
    except Exception:  # pragma: no cover - defensive
        existing = {}
    to_store: dict[str, Any] = {k: v for k, v in existing.items() if k in valid_keys}

    for base, value in updates.items():
        if value is None:
            continue
        key = f"{base}_template"
        if not value.strip():
            to_store.pop(key, None)  # reset to default
            continue
        _validate_template(value, TEMPLATE_REGISTRY[base][1])
        to_store[key] = value

    storage.write_json(_SYSTEM_PROMPT_KEY, to_store)
    logger.info("System-prompt settings saved", customized=sorted(to_store.keys()))
    return load_system_prompt_settings()
