"""LLM-driven Analysis Agent.

Replaces the hardcoded pipeline (regex intent → entity extraction → API lookup →
hardcoded stats → template answer) with a Claude-powered agent that uses tool
calling to search, load, analyze data, and generate natural language answers.

All tool calls are tracked in execution_trace for reproducible notebook generation.
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import (
    Citation,
    DataSource,
    QueryIntent,
    QueryTier,
    RetrievedData,
    ToolCallSignals,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQL guard for run_sql_query (issue #101)
# ---------------------------------------------------------------------------
# CKAN's datastore_search_sql endpoint is the real security boundary (it runs
# in a read-only role), but validating LLM-generated SQL here is cheap
# defense-in-depth: it blocks data-modifying statements, stacked statements,
# and unbounded result sets before we ever hit the network.
_SQL_MAX_ROWS = 10_000
_SQL_TIMEOUT_SECONDS = 60.0
# Data-modifying / DDL keywords. Postgres supports data-modifying CTEs
# (``WITH x AS (DELETE ...) SELECT ...``), so a leading SELECT/WITH alone is
# not sufficient — scan the whole statement for these as well.
_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|vacuum|reindex|merge|call|attach|detach|replace)\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    """Raised when an LLM-generated SQL query fails the read-only guard."""


def _validate_select_sql(sql: str, max_rows: int = _SQL_MAX_ROWS) -> str:
    """Validate and lightly normalize an LLM-generated SQL query.

    Enforces (defense-in-depth on top of CKAN's read-only role):
      * statement must be a single SELECT (or WITH ... SELECT) query,
      * no stacked / multiple statements (embedded ``;``),
      * no data-modifying / DDL keywords anywhere in the statement,
      * a row cap — appends ``LIMIT max_rows`` when none is present.

    Returns the (possibly LIMIT-augmented) SQL. Raises
    :class:`SQLValidationError` on rejection.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL query.")

    stripped = sql.strip()
    # Tolerate a single trailing semicolon, then reject any further ones —
    # an embedded ``;`` means a second (potentially write) statement.
    stripped = stripped.rstrip().rstrip(";").rstrip()
    if ";" in stripped:
        raise SQLValidationError("Only a single SQL statement is allowed (no ';' separators).")

    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SQLValidationError("Only read-only SELECT queries are allowed.")

    if _SQL_FORBIDDEN.search(stripped):
        raise SQLValidationError(
            "Only read-only SELECT queries are allowed "
            "(data-modifying / DDL statements are rejected)."
        )

    # Enforce a row cap so a missing LIMIT can't pull an unbounded result.
    if not re.search(r"\blimit\b", lowered):
        stripped = f"{stripped} LIMIT {max_rows}"

    return stripped


try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# ---------------------------------------------------------------------------
# Agent-log evidence capture
# ---------------------------------------------------------------------------
# The agent_log is the raw evidence record behind every published answer and
# notebook. It follows the civic-ai-tools evidence standards
# (docs/publish-evidence.md / typedstandards.org): verbatim content, complete
# token accounting (including cache tokens), per-tool-call source and
# operationType, accurate model attribution, and wall-clock timestamps.
AGENT_LOG_FORMAT_VERSION = 2

# operationType enum from the evidence standard: query | search | catalog |
# metadata | metrics. Unmapped tools fall back to name-based heuristics.
_TOOL_OPERATION_TYPES = {
    "search_datasets": "search",
    "semantic_search_resources": "search",
    "get_dataset_info": "metadata",
    "load_resource_data": "query",
    "run_sql_query": "query",
}


def _utc_now() -> str:
    """ISO 8601 UTC timestamp for agent-log entries."""
    return datetime.now(UTC).isoformat()


def _operation_type_for_tool(tool_name: str) -> str:
    """Classify a tool call per the evidence-standard operationType enum."""
    if tool_name in _TOOL_OPERATION_TYPES:
        return _TOOL_OPERATION_TYPES[tool_name]
    # MCP tools: classify from the unprefixed tool name.
    bare = tool_name.split("__")[-1].lower()
    if "search" in bare or "find" in bare:
        return "search"
    if any(k in bare for k in ("info", "describe", "metadata", "schema", "variable")):
        return "metadata"
    if "list" in bare or "catalog" in bare:
        return "catalog"
    return "query"


def _source_for_tool(tool_name: str, data_source: str) -> str:
    """Identify the connector behind a tool call (toolCalls[].source)."""
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__", 2)
        return f"mcp:{parts[1]}" if len(parts) == 3 else "mcp"
    return f"ckan:{data_source}"


def _sources_from_trace(
    tool_calls_trace: list[dict[str, Any]],
    data_source: str,
    portal_cfg: dict[str, Any],
    portal_url: str,
) -> list[DataSource]:
    """The data sources a run actually touched, in first-use order.

    A CKAN tool call attributes the primary portal; an ``mcp__{server}__*``
    call attributes that server (named from ``_STATIC_PORTAL_CONFIGS`` when
    registered there). Falls back to the nominal portal when no tool ran, so
    citations are never empty.
    """
    sources: list[DataSource] = []
    seen: set[str] = set()

    def add(source_id: str, name: str, url: str, quality: float) -> None:
        if source_id in seen:
            return
        seen.add(source_id)
        sources.append(DataSource(id=source_id, name=name, url=url, quality_score=quality))

    for entry in tool_calls_trace:
        tool_name = str(entry.get("tool_name") or entry.get("action") or "")
        if not tool_name:
            continue
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__", 2)
            server_id = parts[1] if len(parts) == 3 else "mcp"
            cfg = _STATIC_PORTAL_CONFIGS.get(server_id)
            if cfg:
                add(server_id, cfg["name"], cfg["url"], cfg.get("quality_score", 0.85))
            else:
                add(server_id, f"MCP server: {server_id}", "", 0.85)
        else:
            add(
                data_source,
                portal_cfg.get("name", data_source),
                portal_url,
                portal_cfg.get("quality_score", 0.85),
            )

    if not sources:
        add(
            data_source,
            portal_cfg.get("name", data_source),
            portal_url,
            portal_cfg.get("quality_score", 0.85),
        )
    return sources


# ---------------------------------------------------------------------------
# Portal configurations
# ---------------------------------------------------------------------------
# CKAN portals (wprdc, ckan, and any admin-added sites) are loaded from the
# admin-managed registry in ``gateway.ckan_sites``; non-CKAN LLM sources like
# the Census MCP server stay defined here.  ``get_portal_config`` below looks
# at the registry first and falls back to ``_STATIC_PORTAL_CONFIGS``.
_STATIC_PORTAL_CONFIGS: dict[str, dict[str, Any]] = {
    "census-data-api": {
        "url": "https://api.census.gov",
        "name": "U.S. Census Bureau Data API (via MCP)",
        "organization": None,
        "description": (
            "Official U.S. Census Bureau Data API accessed through an MCP "
            "server. Tools: list-datasets, fetch-dataset-geography, "
            "fetch-aggregate-data, resolve-geography-fips, search-data-tables, "
            "and visualize-census. Backed by a local SQLite metadata index for "
            "fast dataset/geography discovery over the ACS, Decennial Census, "
            "and other Census programs."
        ),
        "quality_score": 0.95,
    },
    "fbi-crime-data": {
        "url": "https://cde.ucr.cjis.gov",
        "name": "FBI Crime Data Explorer (via MCP)",
        "organization": None,
        "description": (
            "FBI Crime Data Explorer (UCR/NIBRS) accessed through an MCP "
            "server. Tools cover summarized crime data, NIBRS incidents, "
            "arrests, hate crimes, expanded homicide/property details, police "
            "employment, LEOKA, LESDC, use of force, agency lookups, and "
            "reference codes. Most date parameters use mm-yyyy (e.g. '01-2020')."
        ),
        "quality_score": 0.9,
    },
}


def _all_portal_configs() -> dict[str, dict[str, Any]]:
    """Return the merged view of registered CKAN sites + static LLM sources."""
    # Lazy import avoids a module-load-time circular with the gateway package.
    from data_concierge.gateway import ckan_sites

    configs: dict[str, dict[str, Any]] = {}
    try:
        configs.update(ckan_sites.get_portal_configs())
    except Exception as exc:  # pragma: no cover – storage failures shouldn't kill the agent
        logger.warning("Failed to load CKAN sites registry", error=str(exc))
    configs.update(_STATIC_PORTAL_CONFIGS)
    return configs


# Portal-type resolution
# ----------------------
# Tool dispatch carries a resolved portal *URL* (a ``portal_id`` override is
# turned into a URL before the tool runs), so these recover which access
# mechanism that URL belongs to.  Both fail closed to CKAN: an unregistered or
# unreachable registry must not change how existing portals are queried.
_PORTAL_TYPE_DCAT = "dcat"
_PORTAL_TYPE_CKAN = "ckan"

# Status codes meaning "this endpoint does not work on this portal", as opposed
# to "this particular query was bad". CKAN answers a malformed or failing SQL
# statement with 400/409 and a JSON error — those are worth another attempt
# with different SQL. These are not: the DataStore extension is disabled, or a
# CDN/WAF sits in front of it (WPRDC currently answers 403 with a CloudFront
# error page). Retrying any of them just burns turns and depresses the
# retrieval-success confidence signal.
_SQL_ENDPOINT_DEAD_STATUSES = frozenset({401, 403, 404, 405, 501})

# How long a portal stays marked SQL-less. The agent is a process-lifetime
# singleton, so without an expiry a single transient block would disable SQL
# until the instance recycled; without any memory, every query would re-probe.
_SQL_DISABLED_TTL_SECONDS = 30 * 60

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _summarize_error_body(text: str, limit: int = 200) -> str:
    """Condense an error body for the model.

    A blocked endpoint returns an HTML error page; pasting 500 characters of
    it into the conversation tells the model nothing and wastes context.
    """
    if not text:
        return ""
    stripped = _HTML_TAG_RE.sub(" ", text)
    collapsed = " ".join(stripped.split())
    return collapsed[:limit]


def _normalize_portal_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in (_PORTAL_TYPE_CKAN, _PORTAL_TYPE_DCAT) else _PORTAL_TYPE_CKAN


def _portal_type_for_url(portal_url: str) -> str:
    """Return the registered portal type for ``portal_url`` (CKAN if unknown)."""
    try:
        from data_concierge.gateway import ckan_sites

        return _normalize_portal_type(ckan_sites.portal_type_for_url(portal_url))
    except Exception as exc:  # pragma: no cover - registry failures are non-fatal
        logger.warning("Portal type lookup failed", url=portal_url, error=str(exc))
        return _PORTAL_TYPE_CKAN


def _site_id_for_url(portal_url: str) -> str | None:
    """Return the registry ID of the portal at ``portal_url``, if registered."""
    try:
        from data_concierge.gateway import ckan_sites

        site = ckan_sites.find_site_by_url(portal_url)
        return str(site["id"]) if site and site.get("id") else None
    except Exception:  # pragma: no cover - registry failures are non-fatal
        return None


def _catalog_url_for_url(portal_url: str) -> str | None:
    """Return the explicit DCAT catalog URL registered for ``portal_url``, if any."""
    try:
        from data_concierge.gateway import ckan_sites

        site = ckan_sites.find_site_by_url(portal_url)
        return (site or {}).get("catalog_url") or None
    except Exception:  # pragma: no cover
        return None


# Backwards-compatible alias — some callers still import PORTAL_CONFIGS.
# Implemented as a module-level property via __getattr__.


def __getattr__(name: str) -> Any:  # pragma: no cover - tiny shim
    if name == "PORTAL_CONFIGS":
        return _all_portal_configs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Tool definitions for Claude function calling
# ---------------------------------------------------------------------------
# Reusable ``portal_id`` property — every CKAN tool accepts it so the agent
# can fan out across admin-registered portals within a single conversation.
_PORTAL_ID_PROPERTY = {
    "type": "string",
    "description": (
        "Optional portal ID to target a specific admin-registered CKAN site "
        "(e.g. 'wprdc', 'ckan', or any ID from the list shown in your system "
        "prompt). Defaults to the primary portal for this conversation."
    ),
}

TOOLS = [
    {
        "name": "semantic_search_resources",
        "description": (
            "**Preferred first step** for finding relevant data. Performs "
            "semantic (vector) search over pre-indexed CKAN resource metadata "
            "using natural-language understanding (not just keywords). "
            "Returns resources with relevance scores, AI tags, record counts, "
            "and column counts. Use this BEFORE search_datasets — it understands "
            "synonyms (e.g. 'air quality' ↔ 'pollution' ↔ 'AQI') and concepts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the data you need "
                        "(e.g. 'unemployment by neighborhood', "
                        "'air pollution monitoring stations')"
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10, max 20)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_datasets",
        "description": (
            "Keyword search for datasets on a CKAN open data portal (CKAN's "
            "package_search API). Use this AFTER semantic_search_resources if "
            "you need exact keyword matches or to search a different portal. "
            "Returns dataset titles, descriptions, and resource IDs. "
            "Pass ``portal_id`` to search a different registered CKAN site."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords to find relevant datasets",
                },
                "organization": {
                    "type": "string",
                    "description": "Filter by organization (e.g. 'city-of-pittsburgh')",
                },
                "rows": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10)",
                    "default": 10,
                },
                "portal_id": _PORTAL_ID_PROPERTY,
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_dataset_info",
        "description": (
            "Get detailed information about a specific dataset including "
            "all its resources (CSV files, APIs). Use to find the resource_id "
            "needed for loading data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "The dataset ID or name from search results",
                },
                "portal_id": _PORTAL_ID_PROPERTY,
            },
            "required": ["dataset_id"],
        },
    },
    {
        "name": "load_resource_data",
        "description": (
            "Load rows from a CKAN DataStore resource. Returns field names and "
            "data values. Use after finding a relevant resource_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "resource_id": {
                    "type": "string",
                    "description": "Resource ID (UUID) from dataset info",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of rows to load (default 100, max 500)",
                    "default": 100,
                },
                "filters": {
                    "type": "object",
                    "description": (
                        'Key-value pairs to filter data, e.g. {"NEIGHBORHOOD": "Squirrel Hill"}'
                    ),
                },
                "q": {
                    "type": "string",
                    "description": "Full-text search within the resource data",
                },
                "sort": {
                    "type": "string",
                    "description": "Sort field and direction, e.g. 'date desc'",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to return",
                },
                "portal_id": _PORTAL_ID_PROPERTY,
            },
            "required": ["resource_id"],
        },
    },
    {
        "name": "run_sql_query",
        "description": (
            "Run a SQL query against CKAN DataStore for aggregations, "
            "GROUP BY, JOINs, etc. The table name is the resource_id "
            "in double quotes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "SQL query.  Use resource_id as table name in double "
                        'quotes.  Example: SELECT "REQUEST_TYPE", COUNT(*) as '
                        'cnt FROM "29462525-…" GROUP BY "REQUEST_TYPE" '
                        "ORDER BY cnt DESC LIMIT 10"
                    ),
                },
                "portal_id": _PORTAL_ID_PROPERTY,
            },
            "required": ["sql"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------
class LLMAnalysisAgent(BaseAgent):
    """LLM-driven agent that handles the full query → answer pipeline.

    Uses Claude with tool calling to search CKAN portals, load data,
    run SQL analytics, and generate a cited natural-language answer.
    Every tool call is recorded in ``execution_trace`` so the notebook
    generator can produce a reproducible Colab notebook.
    """

    name = "llm_analyst"
    description = "LLM-driven data analysis agent"

    def __init__(self) -> None:
        super().__init__()
        self._anthropic: Any = None
        self._http_clients: dict[str, httpx.AsyncClient] = {}
        self._dcat_clients: dict[str, Any] = {}  # portal_url -> DCATClient
        # portal_url -> (expires_at, status_code) for portals whose DataStore
        # SQL endpoint answered with an endpoint-level failure.
        self._sql_disabled: dict[str, tuple[float, int]] = {}
        self._pinecone_store: Any = None  # lazy-init on first semantic search
        # Register so the FastAPI lifespan shutdown can drain our per-portal
        # CKAN connection pools (issue #96).
        from data_concierge.data_layer.connectors import register_closeable

        register_closeable(self)

    async def close(self) -> None:
        """Close all per-portal HTTP clients held by this agent."""
        clients = list(self._http_clients.values())
        self._http_clients.clear()
        for client in clients:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

        dcat_clients = list(self._dcat_clients.values())
        self._dcat_clients.clear()
        for dcat_client in dcat_clients:
            try:
                await dcat_client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    def _get_pinecone_store(self) -> Any:
        """Lazy-initialize the Pinecone vector store for semantic search."""
        if self._pinecone_store is None:
            from data_concierge.data_layer.connectors.pinecone_store import (
                PineconeVectorStore,
            )

            self._pinecone_store = PineconeVectorStore()
        return self._pinecone_store

    # -- helpers -----------------------------------------------------------

    def _get_anthropic_client(self) -> Any:
        if self._anthropic is None:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package is not installed")
            api_key = settings.anthropic_api_key.get_secret_value()
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not configured")
            self._anthropic = anthropic.AsyncAnthropic(api_key=api_key)
        return self._anthropic

    async def _get_http_client(self, portal_url: str) -> httpx.AsyncClient:
        if portal_url not in self._http_clients:
            self._http_clients[portal_url] = httpx.AsyncClient(
                base_url=portal_url.rstrip("/"),
                timeout=60.0,
                headers={"Content-Type": "application/json"},
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._http_clients[portal_url]

    @staticmethod
    def get_portal_config(data_source: str) -> dict[str, Any]:
        """Return portal configuration for a data source key.

        Looks up the admin-managed CKAN sites registry first; falls back to
        the static non-CKAN LLM sources (e.g. ``census_mcp``); and finally
        falls back to the WPRDC portal so existing flows keep working if the
        requested source has been removed.
        """
        configs = _all_portal_configs()
        if data_source in configs:
            return configs[data_source]
        # Fallback: prefer WPRDC if still registered, otherwise the first
        # registered CKAN site, otherwise a minimal hardcoded default.
        wprdc = configs.get("wprdc")
        if wprdc:
            return wprdc
        for cfg in configs.values():
            return cfg
        return {
            "url": "https://data.wprdc.org",
            "name": "Western PA Regional Data Center (WPRDC)",
            "organization": "city-of-pittsburgh",
            "description": "Open data portal for Pittsburgh and Western PA.",
            "quality_score": 0.85,
        }

    # -- system prompt -----------------------------------------------------

    def _build_mcp_system_prompt(self, portal_cfg: dict[str, Any]) -> str:
        """Build a system prompt focused on MCP tool usage (no CKAN).

        The template is admin-editable (``gateway/system_prompt``); if a custom
        template fails to render we fall back to the shipped default so a bad
        override can never break analysis.
        """
        from data_concierge.gateway.system_prompt import (
            DEFAULT_MCP_TEMPLATE,
            get_mcp_template,
        )

        fields = {
            "portal_name": portal_cfg["name"],
            "portal_url": portal_cfg["url"],
            "description": portal_cfg.get("description", ""),
        }
        try:
            return get_mcp_template().format(**fields)
        except (KeyError, IndexError, ValueError) as e:
            self.logger.warning("Custom MCP prompt failed to render; using default", error=str(e))
            return DEFAULT_MCP_TEMPLATE.format(**fields)

    def _build_other_portals_block(self, primary_id: str | None) -> str:
        """List the other registered portals the agent can reach via ``portal_id``.

        Each line carries the portal's type, because the two behave differently:
        a CKAN portal answers SQL, a DCAT portal does not.  Without the label
        the model reaches for ``run_sql_query`` on a catalog portal and burns a
        turn on a guaranteed failure.
        """
        ckan_lines: list[str] = []
        dcat_lines: list[str] = []
        try:
            from data_concierge.gateway import ckan_sites

            for site in ckan_sites.list_sites():
                sid = site.get("id")
                if not sid or sid == primary_id:
                    continue
                name = site.get("name", sid)
                descr = (site.get("description") or "").strip()
                line = f"- **{sid}** — {name}: {descr[:200]}"
                if _normalize_portal_type(site.get("portal_type")) == _PORTAL_TYPE_DCAT:
                    dcat_lines.append(line)
                else:
                    ckan_lines.append(line)
        except Exception:
            return ""

        if not ckan_lines and not dcat_lines:
            return ""

        parts = [
            "\n## Other portals available\n"
            "If the primary portal doesn't have relevant data, you may search "
            "these other registered portals by passing the listed `portal_id` "
            "to any tool call.\n"
        ]
        if ckan_lines:
            parts.append(
                "\n**CKAN portals** (full tool set, including `run_sql_query`):\n"
                + "\n".join(ckan_lines)
                + "\n"
            )
        if dcat_lines:
            parts.append(
                "\n**DCAT catalog portals** — use `search_datasets`, "
                "`get_dataset_info`, and `load_resource_data` only. These "
                "publish downloadable files, not a queryable database, so "
                "`run_sql_query` does NOT work on them; load rows and "
                "aggregate in pandas instead. A truncated load is the first "
                "slice of a file, never the whole dataset:\n"
                + "\n".join(dcat_lines)
                + "\n"
            )
        return "".join(parts)

    def _build_dcat_system_prompt(
        self, portal_cfg: dict[str, Any], primary_id: str | None = None
    ) -> str:
        """Build the system prompt for a DCAT-catalog primary portal."""
        from data_concierge.gateway.system_prompt import (
            DEFAULT_DCAT_TEMPLATE,
            get_dcat_template,
        )

        fields = {
            "portal_name": portal_cfg["name"],
            "portal_url": portal_cfg["url"],
            "description": portal_cfg.get("description", ""),
            "other_portals_block": self._build_other_portals_block(primary_id),
        }
        try:
            return get_dcat_template().format(**fields)
        except (KeyError, IndexError, ValueError) as e:
            self.logger.warning(
                "Custom DCAT prompt failed to render; using default", error=str(e)
            )
            return DEFAULT_DCAT_TEMPLATE.format(**fields)

    def _build_system_prompt(
        self, portal_cfg: dict[str, Any], primary_id: str | None = None
    ) -> str:
        # MCP-backed sources use a focused prompt — no CKAN-specific instructions
        if primary_id and primary_id in _STATIC_PORTAL_CONFIGS:
            return self._build_mcp_system_prompt(portal_cfg)

        # A DCAT catalog has no action API and no SQL, so the CKAN prompt would
        # instruct the model to call tools that cannot work there.
        if _normalize_portal_type(portal_cfg.get("portal_type")) == _PORTAL_TYPE_DCAT:
            return self._build_dcat_system_prompt(portal_cfg, primary_id)

        portal_name = portal_cfg["name"]
        portal_url = portal_cfg["url"]
        org = portal_cfg.get("organization")

        org_block = ""
        if org:
            org_block = (
                f"\nThe primary organization is **{org}**. "
                "Always include this as the organization filter when searching.\n"
            )

        # Build a list of all other admin-registered CKAN portals the agent
        # can reach by passing ``portal_id`` into any of the CKAN tools.  This
        # is what lets the agent "check other CKAN sites before answering":
        # when the primary portal has no relevant data, Claude can re-run
        # search_datasets against one of these alternatives.
        other_portals_block = self._build_other_portals_block(primary_id)

        # The CKAN template is admin-editable (``gateway/system_prompt``). The
        # dynamic parts (portal name/URL, org filter, other portals) are passed
        # as placeholders; a custom template that fails to render falls back to
        # the shipped default so a bad override can't break analysis.
        from data_concierge.gateway.system_prompt import (
            DEFAULT_CKAN_TEMPLATE,
            get_ckan_template,
        )

        fields = {
            "portal_name": portal_name,
            "portal_url": portal_url,
            "org_block": org_block,
            "other_portals_block": other_portals_block,
        }
        try:
            return get_ckan_template().format(**fields)
        except (KeyError, IndexError, ValueError) as e:
            self.logger.warning("Custom CKAN prompt failed to render; using default", error=str(e))
            return DEFAULT_CKAN_TEMPLATE.format(**fields)

    # -- tool execution ----------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
        portal_url: str,
        site_id: str | None = None,
    ) -> str:
        # Allow Claude to override the target portal via ``portal_id`` so it
        # can fan out across admin-registered CKAN sites within one query.
        override_id = tool_input.pop("portal_id", None) if isinstance(tool_input, dict) else None
        effective_url = portal_url
        effective_catalog_url: str | None = None
        effective_type = _portal_type_for_url(portal_url)
        # The portal's registry ID, not just its URL — semantic search is scoped
        # by it so one portal's query cannot return another portal's resource
        # IDs (which would 404 when the agent then tried to load them).
        effective_site_id = site_id or _site_id_for_url(portal_url)
        if override_id:
            override_cfg = self.get_portal_config(override_id)
            override_url = override_cfg.get("url") if override_cfg else None
            if override_url:
                effective_url = override_url
                effective_type = _normalize_portal_type(override_cfg.get("portal_type"))
                effective_catalog_url = override_cfg.get("catalog_url")
                effective_site_id = str(override_id)
                self.logger.info(
                    "Portal override",
                    tool=tool_name,
                    requested=override_id,
                    url=override_url,
                    portal_type=effective_type,
                )
        if effective_catalog_url is None:
            effective_catalog_url = _catalog_url_for_url(effective_url)

        try:
            # Semantic search uses Pinecone, not a per-portal HTTP client
            if tool_name == "semantic_search_resources":
                return await self._tool_semantic_search(tool_input, effective_site_id)

            # DCAT portals have no action API: the same three tool names are
            # served from the catalog document instead (see connectors/dcat.py).
            if effective_type == _PORTAL_TYPE_DCAT:
                dcat = self._get_dcat_client(effective_url, effective_catalog_url)
                if tool_name == "search_datasets":
                    return await self._tool_dcat_search(dcat, tool_input)
                if tool_name == "get_dataset_info":
                    return await self._tool_dcat_dataset_info(dcat, tool_input)
                if tool_name == "load_resource_data":
                    return await self._tool_dcat_load(dcat, tool_input)
                if tool_name == "run_sql_query":
                    return (
                        "Error: run_sql_query is not available on a DCAT portal — "
                        "a DCAT catalog publishes downloadable files, not a queryable "
                        "database. Use load_resource_data to pull rows, then filter "
                        "and aggregate them in pandas."
                    )
                return f"Unknown tool: {tool_name}"

            client = await self._get_http_client(effective_url)
            if tool_name == "search_datasets":
                return await self._tool_search(client, tool_input)
            if tool_name == "get_dataset_info":
                return await self._tool_dataset_info(client, tool_input)
            if tool_name == "load_resource_data":
                return await self._tool_load(client, tool_input)
            if tool_name == "run_sql_query":
                return await self._tool_sql(client, tool_input)
            return f"Unknown tool: {tool_name}"
        except httpx.HTTPStatusError as exc:
            return f"HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        except Exception as exc:
            logger.error("Tool error", tool=tool_name, error=str(exc))
            return f"Error: {exc}"

    async def _tool_semantic_search(self, params: dict, site_id: str | None = None) -> str:
        """Semantic search over pre-indexed portal resources via Pinecone.

        Scoped to ``site_id``: the index is shared across portals, and an
        unscoped search hands the model resource IDs from a portal it is not
        querying.
        """
        query = params.get("query", "")
        n_results = min(params.get("n_results", 10), 20)

        if not query:
            return "Error: query parameter is required"

        store = self._get_pinecone_store()
        if not getattr(store, "use_pinecone", False):
            return (
                "Semantic search is unavailable (Pinecone not configured). "
                "Fall back to search_datasets for keyword search."
            )

        try:
            # Pinecone client is synchronous — run in executor to avoid blocking
            import asyncio

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                None, lambda: store.search_resources(query, n_results, site_id=site_id)
            )
        except Exception as exc:
            return f"Semantic search error: {exc}"

        if not results:
            scope = f" indexed for '{site_id}'" if site_id else ""
            return (
                f"No semantic matches for '{query}' among the resources{scope}. "
                "This portal may not be indexed yet — use search_datasets, which "
                "queries the portal's own live search."
            )

        lines = [f"Found {len(results)} semantically matching resources for '{query}':\n"]
        for i, r in enumerate(results, 1):
            score_pct = round(r.get("score", 0) * 100, 1)
            lines.append(f"{i}. **{r.get('dataset_title', 'Unknown')}** ({score_pct}% match)")
            lines.append(f"   Resource: {r.get('resource_name', 'Unnamed')}")
            lines.append(f"   Resource ID: `{r.get('resource_id', '')}`")
            lines.append(f"   Dataset ID: `{r.get('dataset_id', '')}`")
            lines.append(
                f"   Format: {r.get('format', '?')} | "
                f"Rows: {r.get('record_count', 0):,} | "
                f"Cols: {r.get('column_count', 0)}"
            )
            tags = r.get("ai_tags", "")
            if tags:
                lines.append(f"   Tags: {tags[:200]}")
            desc = r.get("description", "")
            if desc:
                lines.append(f"   {desc[:200]}")
            tc = r.get("temporal_coverage")
            if tc:
                lines.append(f"   Coverage: {tc.get('min')} → {tc.get('max')}")
            lines.append("")
        return "\n".join(lines)

    # -- DCAT portal tools --------------------------------------------------
    #
    # These back the same three tool names as their CKAN counterparts, so the
    # model needs no separate schema; only the mechanism differs.

    def _get_dcat_client(self, portal_url: str, catalog_url: str | None = None) -> Any:
        """Return a pooled DCAT client for ``portal_url``.

        Pooled per portal so the parsed catalog (and its in-process cache) is
        shared across every tool call in a run.
        """
        from data_concierge.data_layer.connectors.dcat import DCATClient

        key = portal_url.rstrip("/")
        client = self._dcat_clients.get(key)
        if client is None:
            client = DCATClient(key, catalog_url=catalog_url)
            self._dcat_clients[key] = client
        return client

    async def _tool_dcat_search(self, dcat: Any, params: dict) -> str:
        query = params.get("query", "")
        rows = min(int(params.get("rows", 10) or 10), 20)

        matches = await dcat.search_datasets(query, limit=rows)
        if not matches:
            catalog = await dcat.fetch_catalog()
            return (
                f"No datasets in {dcat.portal_url} match '{query}' "
                f"(catalog has {len(catalog.datasets)} datasets). "
                "Try broader or different keywords."
            )

        lines = [f"Found {len(matches)} datasets matching '{query}':\n"]
        for i, (ds, score) in enumerate(matches, 1):
            lines.append(f"{i}. **{ds.title}**")
            lines.append(f"   Dataset ID: `{ds.id}`  (relevance {score:.2f})")
            if ds.description:
                lines.append(f"   {ds.description[:250]}")
            if ds.keywords:
                lines.append(f"   Keywords: {', '.join(ds.keywords[:10])}")
            tabular = ds.tabular_distributions
            lines.append(
                f"   Modified: {ds.modified[:10] or '?'} | "
                f"Distributions: {len(ds.distributions)} "
                f"({len(tabular)} tabular)"
            )
            lines.append("")
        lines.append(
            "Call get_dataset_info with a Dataset ID to see its distributions, "
            "then load_resource_data to pull rows."
        )
        return "\n".join(lines)

    async def _tool_dcat_dataset_info(self, dcat: Any, params: dict) -> str:
        dataset_id = params.get("dataset_id") or params.get("id") or ""
        ds = await dcat.get_dataset(dataset_id)
        if ds is None:
            return (
                f"Dataset '{dataset_id}' not found in the {dcat.portal_url} catalog. "
                "Use search_datasets first and pass a Dataset ID from its results."
            )

        lines = [f"# {ds.title}", f"Dataset ID: `{ds.id}`"]
        if ds.description:
            lines.append(f"\n{ds.description[:800]}")
        if ds.publisher:
            lines.append(f"\nPublisher: {ds.publisher}")
        lines.append(f"Modified: {ds.modified[:10] or '?'}  Issued: {ds.issued[:10] or '?'}")
        if ds.themes:
            lines.append(f"Themes: {', '.join(ds.themes[:10])}")
        if ds.keywords:
            lines.append(f"Keywords: {', '.join(ds.keywords[:15])}")
        if ds.landing_page:
            lines.append(f"Landing page: {ds.landing_page}")
        if ds.license:
            lines.append(f"License: {ds.license}")

        lines.append(f"\nDistributions ({len(ds.distributions)}):")
        for i, dist in enumerate(ds.distributions, 1):
            label = dist.title or dist.format or dist.media_type or "Unnamed"
            mark = "tabular" if dist.is_tabular else "non-tabular"
            lines.append(f"  {i}. {label} [{mark}]")
            lines.append(f"     Media type: {dist.media_type or '?'}")
            if dist.best_url:
                lines.append(f"     URL: {dist.best_url}")

        if ds.tabular_distributions:
            lines.append(
                f"\nTo load rows: load_resource_data with resource_id=`{ds.id}` "
                "(uses the first tabular distribution), or pass a distribution "
                "URL above as resource_id to pick a specific one."
            )
        else:
            lines.append(
                "\nNo tabular (CSV/TSV) distribution — this dataset cannot be "
                "loaded as rows. Cite it from its metadata or pick another dataset."
            )
        return "\n".join(lines)

    async def _tool_dcat_load(self, dcat: Any, params: dict) -> str:
        """Load rows from a DCAT distribution.

        ``resource_id`` accepts either a dataset ID (the first tabular
        distribution is used) or a distribution URL taken from
        get_dataset_info.
        """
        resource_id = str(params.get("resource_id") or "").strip()
        limit = min(int(params.get("limit", 100) or 100), 1000)
        if not resource_id:
            return "Error: resource_id is required (a dataset ID or a distribution URL)."

        target_url = ""
        dataset_label = resource_id
        if resource_id.lower().startswith(("http://", "https://")):
            target_url = resource_id
        else:
            ds = await dcat.get_dataset(resource_id)
            if ds is None:
                return (
                    f"Dataset '{resource_id}' not found in the {dcat.portal_url} "
                    "catalog. Use search_datasets to find a valid Dataset ID."
                )
            tabular = ds.tabular_distributions
            if not tabular:
                return (
                    f"Dataset '{ds.title}' has no tabular (CSV/TSV) distribution, "
                    "so its rows cannot be loaded."
                )
            target_url = tabular[0].best_url
            dataset_label = ds.title

        result = await dcat.load_distribution(target_url, max_rows=limit)
        rows = result["rows"]
        if not rows:
            return (
                f"No rows returned from {target_url} "
                f"({result['bytes_read']:,} bytes read)."
            )

        lines = [
            f"Dataset: {dataset_label}",
            f"Distribution: {target_url}",
            f"Loaded: {result['row_count']} rows, {len(result['columns'])} columns",
        ]
        # A DCAT file has no row count in its metadata, so the honest statement
        # is how many rows we read and whether we stopped early — never a total.
        if result["truncated"]:
            lines.append(
                f"NOTE: truncated at the {limit}-row read limit — this is the "
                "first slice of the file, not the whole dataset. Any total or "
                "aggregate you report must be qualified accordingly, or "
                "re-loaded with a higher limit."
            )
        else:
            lines.append("Complete: the full distribution fit within the read limit.")
        lines.append(f"Columns ({len(result['columns'])}): {', '.join(result['columns'])}")

        sample = pd.DataFrame(rows[:15])
        lines.append(f"\nSample ({min(15, len(rows))} rows):\n")
        lines.append(sample.to_string(index=False, max_colwidth=40))
        return "\n".join(lines)

    async def _tool_search(self, client: httpx.AsyncClient, params: dict) -> str:
        query = params.get("query", "")
        org = params.get("organization")
        rows = min(params.get("rows", 10), 20)

        search_params: dict[str, Any] = {"q": query, "rows": rows}
        if org:
            # The organization comes from the model and lands in a Solr filter
            # query. Unescaped, a value like `x OR *:*` widens the filter to
            # everything, and Solr's special characters can restructure the
            # clause. Quote it and escape the delimiters (#135).
            safe_org = str(org).replace("\\", "\\\\").replace('"', '\\"')
            search_params["fq"] = f'organization:"{safe_org}"'

        resp = await client.get("/api/3/action/package_search", params=search_params)
        resp.raise_for_status()
        body = resp.json()

        if not body.get("success"):
            return f"Search failed: {body.get('error', 'unknown')}"

        result = body["result"]
        total = result.get("count", 0)
        datasets = result.get("results", [])

        lines = [f"Found {total} datasets matching '{query}'\n"]
        for i, pkg in enumerate(datasets, 1):
            title = pkg.get("title", pkg.get("name", "Untitled"))
            lines.append(f"{i}. **{title}**")
            lines.append(f"   ID: `{pkg.get('name')}`")
            if pkg.get("notes"):
                lines.append(f"   {pkg['notes'][:180].replace(chr(10), ' ')}")
            for res in pkg.get("resources", [])[:5]:
                ds = " [DataStore]" if res.get("datastore_active") else ""
                fmt = res.get("format", "?")
                lines.append(f"   - {res.get('name', 'Unnamed')} ({fmt}){ds} ID: `{res.get('id')}`")
            lines.append("")
        return "\n".join(lines)

    async def _tool_dataset_info(self, client: httpx.AsyncClient, params: dict) -> str:
        dataset_id = params["dataset_id"]
        resp = await client.get("/api/3/action/package_show", params={"id": dataset_id})
        resp.raise_for_status()
        body = resp.json()
        if not body.get("success"):
            return f"Dataset '{dataset_id}' not found."

        pkg = body["result"]
        lines = [f"# {pkg.get('title', pkg.get('name'))}"]
        if pkg.get("notes"):
            lines.append(f"\n{pkg['notes'][:600]}")
        if pkg.get("organization"):
            lines.append(f"\nOrganization: {pkg['organization'].get('title')}")
        lines.append(f"Modified: {pkg.get('metadata_modified', '?')[:10]}")
        tags = [t.get("name") if isinstance(t, dict) else str(t) for t in pkg.get("tags", [])]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")

        resources = pkg.get("resources", [])
        lines.append(f"\nResources ({len(resources)}):")
        for i, res in enumerate(resources, 1):
            ds = "Yes" if res.get("datastore_active") else "No"
            lines.append(f"  {i}. {res.get('name', 'Unnamed')}")
            lines.append(f"     ID: `{res.get('id')}`")
            lines.append(f"     Format: {res.get('format', '?')}  DataStore: {ds}")
            if res.get("description"):
                lines.append(f"     {res['description'][:150]}")
        return "\n".join(lines)

    async def _tool_load(self, client: httpx.AsyncClient, params: dict) -> str:
        resource_id = params["resource_id"]
        limit = min(params.get("limit", 100), 500)

        body: dict[str, Any] = {"resource_id": resource_id, "limit": limit}
        if params.get("offset"):
            body["offset"] = params["offset"]
        if params.get("filters"):
            body["filters"] = params["filters"]
        if params.get("q"):
            body["q"] = params["q"]
        if params.get("sort"):
            body["sort"] = params["sort"]
        if params.get("fields"):
            body["fields"] = params["fields"]

        resp = await client.post("/api/3/action/datastore_search", json=body)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            return f"Load failed: {data.get('error', 'unknown')}"

        result = data["result"]
        records = result.get("records", [])
        total = result.get("total", 0)
        fields = [f.get("id", "?") for f in result.get("fields", []) if f.get("id") != "_id"]

        lines = [
            f"Resource: {resource_id}",
            f"Total records: {total:,}",
            f"Loaded: {len(records)}",
            f"Fields ({len(fields)}): {', '.join(fields)}",
        ]

        if records:
            df = pd.DataFrame(records[:15]).drop(columns=["_id", "_full_text"], errors="ignore")
            lines.append(f"\nSample ({min(15, len(records))} rows):\n")
            lines.append(df.to_string(index=False, max_colwidth=40))
        return "\n".join(lines)

    def _sql_unavailable_status(self, portal_url: str) -> int | None:
        """Status that disabled SQL for this portal, or ``None`` if usable."""
        entry = self._sql_disabled.get(portal_url)
        if not entry:
            return None
        expires_at, status = entry
        if time.time() >= expires_at:
            del self._sql_disabled[portal_url]
            return None
        return status

    def _disable_sql(self, portal_url: str, status: int) -> None:
        self._sql_disabled[portal_url] = (
            time.time() + _SQL_DISABLED_TTL_SECONDS,
            status,
        )
        self.logger.warning(
            "DataStore SQL disabled for portal",
            portal_url=portal_url,
            status=status,
            ttl_seconds=_SQL_DISABLED_TTL_SECONDS,
        )

    # Prefix marking a tool result that is neither a retrieval success nor a
    # retrieval failure: the capability does not exist on this portal. It is
    # excluded from the success-rate accounting entirely — counting it as a
    # success would inflate a user-visible confidence number with a call that
    # fetched nothing, and counting it as a failure would penalise the agent
    # for correctly routing around a portal-side outage.
    TOOL_UNAVAILABLE_PREFIX = "Tool unavailable:"

    @staticmethod
    def _sql_unavailable_message(status: int, detail: str = "") -> str:
        """What the model is told when SQL cannot work on this portal.

        Phrased as a standing fact plus the alternative, because the previous
        message (an HTTP status and a slab of HTML) read like a transient
        failure and the model retried it five times per query.
        """
        suffix = f" Portal said: {detail}" if detail else ""
        return (
            f"{LLMAnalysisAgent.TOOL_UNAVAILABLE_PREFIX} "
            f"run_sql_query is NOT available on this portal — its DataStore SQL "
            f"endpoint returned HTTP {status}, which means the endpoint is "
            f"blocked or not enabled rather than the query being wrong. "
            f"Do NOT call run_sql_query against this portal again; every retry "
            f"will fail the same way. Use load_resource_data instead (it "
            f"accepts filters, q, sort, fields, and limit) and aggregate the "
            f"rows yourself.{suffix}"
        )

    async def _tool_sql(self, client: httpx.AsyncClient, params: dict) -> str:
        portal_url = str(client.base_url).rstrip("/")

        # Already known dead for this portal: answer without a round trip.
        known = self._sql_unavailable_status(portal_url)
        if known is not None:
            return self._sql_unavailable_message(known)

        raw_sql = params["sql"]
        try:
            sql = _validate_select_sql(raw_sql)
        except SQLValidationError as exc:
            return f"SQL rejected: {exc}"

        try:
            resp = await client.post(
                "/api/3/action/datastore_search_sql",
                json={"sql": sql},
                timeout=_SQL_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException:
            return (
                f"SQL error: query exceeded the {_SQL_TIMEOUT_SECONDS:.0f}s "
                "execution timeout. Narrow the query (add filters, "
                "aggregate, or reduce the result set)."
            )
        if resp.status_code in _SQL_ENDPOINT_DEAD_STATUSES:
            # Endpoint-level failure: remember it so the rest of this query —
            # and the next one against the same portal — skips SQL entirely.
            self._disable_sql(portal_url, resp.status_code)
            return self._sql_unavailable_message(
                resp.status_code, _summarize_error_body(resp.text)
            )
        if resp.status_code != 200:
            # Query-level failure (CKAN uses 400/409): a different statement
            # may well succeed, so this stays retryable.
            return (
                f"SQL error (HTTP {resp.status_code}): "
                f"{_summarize_error_body(resp.text, 400)}"
            )

        data = resp.json()
        if not data.get("success"):
            return f"SQL error: {json.dumps(data.get('error', {}))}"

        result = data["result"]
        records = result.get("records", [])
        fields = [f.get("id", "?") for f in result.get("fields", [])]

        lines = [f"SQL: {sql}", f"Rows: {len(records)}", f"Columns: {', '.join(fields)}", ""]
        if records:
            df = pd.DataFrame(records).drop(columns=["_id", "_full_text"], errors="ignore")
            lines.append(df.to_string(index=False, max_colwidth=50))
        return "\n".join(lines)

    # -- code generation for notebooks -------------------------------------

    @staticmethod
    def _dcat_catalog_search_code(
        catalog_url: str, query: str, rows: int, helper: str
    ) -> str:
        """Code that fetches a DCAT catalog and keyword-searches it locally.

        Shared by the ``search_datasets`` cell and the ``semantic_search_resources``
        stand-in so the two cannot drift apart.
        """
        return (
            "import requests\n\n"
            f"CATALOG_URL = {catalog_url!r}\n"
            f"QUERY = {query!r}\n\n"
            "catalog = requests.get(CATALOG_URL, timeout=60).json()\n"
            "datasets = catalog.get('dataset', [])\n"
            "print(f'Catalog: {len(datasets)} datasets')\n\n"
            + helper
            + "\n"
            "terms = [t for t in QUERY.lower().split() if t]\n\n"
            "def score(ds):\n"
            "    title = (ds.get('title') or '').lower()\n"
            "    desc = (ds.get('description') or '').lower()\n"
            "    kws = ' '.join(ds.get('keyword') or []).lower()\n"
            "    return sum(3 * (t in title) + 2 * (t in kws) + (t in desc) for t in terms)\n\n"
            "matches = [d for d in datasets if score(d) > 0]\n"
            "matches.sort(key=score, reverse=True)\n\n"
            f"for ds in matches[:{rows}]:\n"
            "    print(f\"{short_id(ds):14} {ds.get('title', '')}\")\n"
        )

    @staticmethod
    def _code_for_dcat_tool(tool_name: str, tool_input: dict, portal_url: str) -> str:
        """Reproducible code for a DCAT portal call.

        A DCAT catalog is a static document, so the reproduction is: fetch the
        catalog, find the dataset, read the distribution.  The row read uses
        ``pd.read_csv(..., nrows=N)``, which streams and stops the same way the
        agent's own loader does — so the notebook reproduces the same slice
        rather than silently pulling a much larger file.

        The bodies below are templates rather than concatenated fragments: the
        generated code is full of quotes and f-strings, and hand-escaping it
        into Python string literals is how broken cells get shipped.
        """
        catalog_url = _catalog_url_for_url(portal_url) or f"{portal_url}/data.json"

        # Shared helper the generated cells use to recover a portal's short
        # dataset ID from the DCAT identifier URL.
        helper = (
            "def short_id(ds):\n"
            '    """Portal dataset ID = last path segment of the DCAT identifier."""\n'
            "    ident = ds.get('identifier') or ''\n"
            "    return ident.rstrip('/').rsplit('/', 1)[-1]\n"
        )

        if tool_name == "semantic_search_resources":
            # Same problem as the CKAN stand-in: the live tool queries a private
            # Pinecone index the notebook cannot reach. On a DCAT portal the
            # public equivalent is a keyword search of the catalog document —
            # NOT CKAN's package_search, which does not exist here and would
            # 404 on every run.
            query = tool_input.get("query", "")
            try:
                rows = int(tool_input.get("n_results", 10))
            except (TypeError, ValueError):
                rows = 10
            return (
                "# Resource discovery. The concierge used a semantic (vector) search\n"
                "# over pre-indexed resource metadata here; the public equivalent for a\n"
                "# DCAT portal is a keyword search of the catalog document.\n"
                + LLMAnalysisAgent._dcat_catalog_search_code(
                    catalog_url, query, min(rows, 20), helper
                )
            )

        if tool_name == "search_datasets":
            query = tool_input.get("query", "")
            rows = min(int(tool_input.get("rows", 10) or 10), 20)
            return "# Search the DCAT catalog (fetched once, searched locally)\n" + (
                LLMAnalysisAgent._dcat_catalog_search_code(catalog_url, query, rows, helper)
            )

        if tool_name == "get_dataset_info":
            did = tool_input.get("dataset_id", "")
            return (
                "# Inspect one catalog dataset and its distributions\n"
                "import requests\n\n"
                f"CATALOG_URL = {catalog_url!r}\n"
                f"DATASET_ID = {did!r}\n\n"
                "catalog = requests.get(CATALOG_URL, timeout=60).json()\n\n"
                + helper
                + "\n"
                "match = next(\n"
                "    (d for d in catalog.get('dataset', []) if short_id(d) == DATASET_ID),\n"
                "    None,\n"
                ")\n\n"
                "if match is None:\n"
                "    print(f'Dataset {DATASET_ID} not found in catalog')\n"
                "else:\n"
                "    print(match.get('title', ''))\n"
                "    print(match.get('description', '')[:400])\n"
                "    print()\n"
                "    for dist in match.get('distribution', []):\n"
                "        media = dist.get('mediaType') or dist.get('format') or '?'\n"
                "        print(f\"  {media:28} {dist.get('downloadURL', '')}\")\n"
            )

        if tool_name == "load_resource_data":
            rid = str(tool_input.get("resource_id", ""))
            limit = min(int(tool_input.get("limit", 100) or 100), 1000)

            if rid.lower().startswith(("http://", "https://")):
                preamble = (
                    "# Load rows from a DCAT distribution\n"
                    f"CSV_URL = {rid!r}\n\n"
                )
            else:
                # Resolve the dataset's first tabular distribution the same way
                # the agent did, so the notebook is not pinned to a URL the
                # portal may re-issue.
                preamble = (
                    "# Resolve the dataset's first CSV distribution, then load rows\n"
                    "import requests\n\n"
                    f"CATALOG_URL = {catalog_url!r}\n"
                    f"DATASET_ID = {rid!r}\n\n"
                    "catalog = requests.get(CATALOG_URL, timeout=60).json()\n\n"
                    + helper
                    + "\n"
                    "match = next(\n"
                    "    (d for d in catalog.get('dataset', []) if short_id(d) == DATASET_ID),\n"
                    "    None,\n"
                    ")\n"
                    "CSV_URL = next(\n"
                    "    (\n"
                    "        d['downloadURL']\n"
                    "        for d in (match or {}).get('distribution', [])\n"
                    "        if d.get('mediaType') == 'text/csv' and d.get('downloadURL')\n"
                    "    ),\n"
                    "    None,\n"
                    ")\n"
                    "print(f'CSV distribution: {CSV_URL}')\n\n"
                )

            return (
                preamble
                + "import pandas as pd\n\n"
                + "# nrows stops the read at the same slice the agent loaded — a DCAT\n"
                + "# distribution is a static file with no server-side row limit.\n"
                + f"df = pd.read_csv(CSV_URL, nrows={limit})\n\n"
                + "print(f'Loaded {len(df)} rows, {len(df.columns)} columns')\n"
                + "print('Columns: ' + ', '.join(df.columns.tolist()))\n"
                + "df.head(10)\n"
            )

        return f"# {tool_name}: {json.dumps(tool_input)}"

    @staticmethod
    def _code_for_tool(tool_name: str, tool_input: dict, portal_url: str) -> str:
        """Generate reproducible Python code for a tool call."""
        if _portal_type_for_url(portal_url) == _PORTAL_TYPE_DCAT:
            return LLMAnalysisAgent._code_for_dcat_tool(tool_name, tool_input, portal_url)

        if tool_name == "semantic_search_resources":
            q = tool_input.get("query", "")
            try:
                n = int(tool_input.get("n_results", 10))
            except (TypeError, ValueError):
                n = 10
            # The live tool searches a private Pinecone index, which the
            # published notebook cannot reach — importing it made every
            # notebook fail both in Colab and in the verifier's kernel. The
            # reproducible stand-in is the portal's public keyword search:
            # different ranking, same purpose (find candidate resources).
            return (
                "# Resource discovery. The concierge used a semantic (vector) search\n"
                "# over pre-indexed resource metadata here; the public equivalent is\n"
                "# CKAN's keyword search, which reproduces the discovery step without\n"
                "# private infrastructure.\n"
                "import requests\n\n"
                f'params = {{"q": {q!r}, "rows": {n}}}\n'
                f'resp = requests.get("{portal_url}/api/3/action/package_search", params=params)\n'
                'results = resp.json()["result"]["results"]\n\n'
                "for ds in results:\n"
                "    print(ds['title'])\n"
                '    for r in ds.get("resources", []):\n'
                "        print(f\"  - {r['name']} ({r['format']}) ID: {r['id']}\")\n"
            )

        if tool_name == "search_datasets":
            q = tool_input.get("query", "")
            org = tool_input.get("organization")
            # org and rows come from the model and are written into the .ipynb
            # the user later runs, so they cannot be interpolated raw: an org
            # like `x"}\nimport os\nos.system(...)` would inject code. repr the
            # Solr fq value and coerce rows to an int.
            try:
                rows = int(tool_input.get("rows", 10))
            except (TypeError, ValueError):
                rows = 10
            # repr the whole fq value so any quotes/newlines in org are escaped
            # into a proper Python string literal in the generated cell.
            fq = f', "fq": {f"organization:{org}"!r}' if org else ""
            return (
                "# Search for datasets\n"
                "import requests, json\n\n"
                f'params = {{"q": {q!r}, "rows": {rows}{fq}}}\n'
                f'resp = requests.get("{portal_url}/api/3/action/package_search", params=params)\n'
                'results = resp.json()["result"]["results"]\n\n'
                "print(f\"Found {resp.json()['result']['count']} datasets\")\n"
                "for i, ds in enumerate(results, 1):\n"
                "    print(f\"\\n{i}. {ds['title']}\")\n"
                "    print(f\"   ID: {ds['name']}\")\n"
                '    for r in ds.get("resources", []):\n'
                "        print(f\"   - {r['name']} ({r['format']}) ID: {r['id']}\")\n"
            )

        if tool_name == "get_dataset_info":
            did = tool_input["dataset_id"]
            return (
                "# Get dataset details\n"
                f'resp = requests.get("{portal_url}/api/3/action/package_show", '
                f'params={{"id": {did!r}}})\n'
                'ds = resp.json()["result"]\n\n'
                "print(f\"Title: {ds['title']}\")\n"
                "print(f\"Description: {ds.get('notes', 'N/A')[:200]}\")\n"
                'print(f"\\nResources:")\n'
                'for r in ds.get("resources", []):\n'
                '    act = "DataStore" if r.get("datastore_active") else "File"\n'
                "    print(f\"  - {r['name']} ({r['format']}, {act}) ID: {r['id']}\")\n"
            )

        if tool_name == "load_resource_data":
            rid = tool_input["resource_id"]
            limit = tool_input.get("limit", 100)
            parts = [f'"resource_id": {rid!r}', f'"limit": {limit}']
            if tool_input.get("filters"):
                parts.append(f'"filters": {json.dumps(tool_input["filters"])}')
            if tool_input.get("q"):
                parts.append(f'"q": {tool_input["q"]!r}')
            if tool_input.get("sort"):
                parts.append(f'"sort": {tool_input["sort"]!r}')
            params_str = ", ".join(parts)
            return (
                "# Load resource data\n"
                "import pandas as pd\n\n"
                f"params = {{{params_str}}}\n"
                f'resp = requests.post("{portal_url}/api/3/action/datastore_search", json=params)\n'
                'result = resp.json()["result"]\n\n'
                'df = pd.DataFrame(result["records"])\n'
                'df = df.drop(columns=["_id", "_full_text"], errors="ignore")\n\n'
                'print(f"Loaded {len(df)} rows, {len(df.columns)} columns")\n'
                "print(f\"Total available: {result['total']:,}\")\n"
                # Single quotes inside the f-string expression — a backslash
                # there is a SyntaxError in the notebook kernel, and this one
                # line was failing verification for every generated notebook.
                "print(f\"\\nColumns: {', '.join(df.columns.tolist())}\")\n"
                "df.head(10)\n"
            )

        if tool_name == "run_sql_query":
            sql = tool_input["sql"]
            return (
                "# Run SQL analysis query\n"
                f"sql = {sql!r}\n\n"
                f'resp = requests.post("{portal_url}/api/3/action/datastore_search_sql", '
                'json={"sql": sql})\n'
                'result = resp.json()["result"]\n\n'
                'df = pd.DataFrame(result["records"])\n'
                'df = df.drop(columns=["_id", "_full_text"], errors="ignore")\n\n'
                'print(f"Query returned {len(df)} rows")\n'
                "df\n"
            )

        return f"# {tool_name}: {json.dumps(tool_input)}"

    # -- MCP tool integration -----------------------------------------------

    def _get_mcp_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions from connected MCP servers for LLM calling."""
        try:
            from data_concierge.mcp.connector import get_mcp_connector

            connector = get_mcp_connector()
            return connector.get_tool_definitions_for_llm()
        except Exception as exc:
            logger.debug("MCP tools not available", error=str(exc))
            return []

    async def _execute_mcp_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute an MCP tool call from the LLM."""
        from data_concierge.mcp.connector import get_mcp_connector

        connector = get_mcp_connector()
        return await connector.handle_llm_tool_call(tool_name, tool_input)

    def _code_for_mcp_tool(self, tool_name: str, tool_input: dict, result_text: str) -> str:
        """Generate reproducible code for an MCP tool call."""
        try:
            from data_concierge.mcp.connector import get_mcp_connector

            connector = get_mcp_connector()
            parts = tool_name.split("__", 2)
            if len(parts) == 3:
                server_id = parts[1]
                actual_tool = parts[2]
                return connector.generate_reproducible_code(
                    server_id, actual_tool, tool_input, result_text
                )
        except Exception:
            pass
        # repr() keeps this valid Python whatever the input contains — an
        # f-string with multi-line JSON left every line after the first
        # uncommented, a guaranteed SyntaxError in the notebook.
        return f"# MCP tool: {tool_name}\narguments = {tool_input!r}\n"

    # -- LLM call with retry -------------------------------------------------

    async def _call_llm_with_retry(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list,
        messages: list,
        max_retries: int = 3,
        initial_wait: float = 1.0,
        event_log: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> Any:
        """Call the Anthropic API with retry + model fallback for transient errors.

        On 529 (overloaded) or 529-like errors, retries with exponential backoff.
        After exhausting retries on the primary model, falls back to
        ``settings.llm_fallback_model`` for one final attempt.

        Retry and fallback events are appended to ``event_log`` (the agent_log)
        so the evidence record reflects every model interaction, not just the
        ones that succeeded.
        """
        last_exc: Exception | None = None
        wait = initial_wait
        extra: dict[str, Any] = {"tool_choice": tool_choice} if tool_choice else {}

        for attempt in range(max_retries):
            try:
                return await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                    **extra,
                )
            except Exception as exc:
                err_str = str(exc)
                is_transient = "529" in err_str or "overloaded" in err_str.lower()
                if not is_transient:
                    raise
                last_exc = exc
                self.logger.warning(
                    "Transient LLM error, retrying",
                    attempt=attempt + 1,
                    wait_secs=wait,
                    model=model,
                    error=err_str[:200],
                )
                if event_log is not None:
                    event_log.append(
                        {
                            "type": "llm_retry",
                            "timestamp": _utc_now(),
                            "attempt": attempt + 1,
                            "model": model,
                            "error": err_str[:500],
                            "wait_secs": wait,
                        }
                    )
                await asyncio.sleep(wait)
                wait *= 2  # exponential backoff

        # All retries on primary model exhausted — try fallback model once
        fallback = settings.llm_fallback_model
        if fallback and fallback != model:
            self.logger.warning(
                "Primary model exhausted retries, falling back",
                primary=model,
                fallback=fallback,
            )
            if event_log is not None:
                event_log.append(
                    {
                        "type": "model_fallback",
                        "timestamp": _utc_now(),
                        "from_model": model,
                        "to_model": fallback,
                    }
                )
            try:
                response = await client.messages.create(
                    model=fallback,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tools,
                    messages=messages,
                    **extra,
                )
                # Tag so caller can stick with fallback for remaining iterations
                response._used_model = fallback
                return response
            except Exception:
                pass  # fall through to raise the original error

        raise last_exc  # type: ignore[misc]

    # -- main process ------------------------------------------------------

    async def process(self, state: GraphState) -> GraphState:  # noqa: C901
        """Run the LLM-driven analysis pipeline."""
        query = state["query"]
        data_source = state.get("data_source", "wprdc")
        portal_cfg = self.get_portal_config(data_source)
        portal_url = portal_cfg["url"]

        self.logger.info("LLM analysis starting", query=query[:100], data_source=data_source)
        start = time.monotonic()

        tool_calls_trace: list[dict[str, Any]] = []
        # Initialized before the try block so the evidence record survives
        # (and captures) failures anywhere in the pipeline.
        agent_log: list[dict[str, Any]] = []
        iteration = 0

        try:
            client = self._get_anthropic_client()
            system_prompt = self._build_system_prompt(portal_cfg, primary_id=data_source)
            messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

            # For MCP-backed sources, use ONLY MCP tools (no CKAN tools).
            # For CKAN sources, combine CKAN tools with any available MCP tools.
            is_mcp_source = data_source in _STATIC_PORTAL_CONFIGS
            mcp_tools = self._get_mcp_tools()

            if is_mcp_source:
                all_tools = mcp_tools
                self.logger.info(
                    "MCP-only tool set",
                    mcp_tool_count=len(mcp_tools),
                )
            else:
                all_tools = list(TOOLS)
                # A portal whose SQL endpoint is known dead should not be
                # offered run_sql_query at all. The breaker in _tool_sql
                # already answers without a round trip, but a tool the model
                # cannot see is a turn it cannot waste — and the DCAT branch
                # sets the same precedent for portals that never had SQL.
                sql_status = self._sql_unavailable_status(portal_cfg["url"].rstrip("/"))
                if sql_status is not None:
                    all_tools = [t for t in all_tools if t["name"] != "run_sql_query"]
                    system_prompt += (
                        "\n\n## Note on SQL\n"
                        f"This portal's DataStore SQL endpoint is unavailable "
                        f"(HTTP {sql_status}), so `run_sql_query` has been removed "
                        "from your tools. Load rows with `load_resource_data` and "
                        "aggregate them yourself.\n"
                    )
                    self.logger.info(
                        "run_sql_query withheld — portal SQL unavailable",
                        portal_url=portal_cfg["url"],
                        status=sql_status,
                    )
                if mcp_tools:
                    all_tools.extend(mcp_tools)
                    mcp_names = [t["name"] for t in mcp_tools]
                    system_prompt += (
                        "\n\n## Additional MCP Tools\n"
                        "You also have access to tools from connected MCP servers. "
                        "These tools provide access to additional data sources like "
                        "the U.S. Census Bureau. Use them when the user's question "
                        "involves census data, demographics, or population statistics.\n"
                        f"Available MCP tools: {', '.join(mcp_names)}\n"
                    )
                    self.logger.info(
                        "MCP tools appended to CKAN tool set",
                        mcp_tool_count=len(mcp_tools),
                        total_tools=len(all_tools),
                    )

            final_answer: str | None = None
            max_iterations = 12
            current_model = settings.llm_model

            # Record the full context that shaped the analysis (the evidence
            # standard's skillText equivalent): verbatim query, system prompt
            # (with hash for integrity checks), model, and available tools.
            agent_log.append(
                {
                    "type": "session_start",
                    "log_format_version": AGENT_LOG_FORMAT_VERSION,
                    "timestamp": _utc_now(),
                    "query": query,
                    "data_source": data_source,
                    "portal_url": portal_url,
                    "model": current_model,
                    "max_iterations": max_iterations,
                    "tools_available": [t["name"] for t in all_tools],
                    "system_prompt": system_prompt,
                    "system_prompt_sha256": hashlib.sha256(
                        system_prompt.encode("utf-8")
                    ).hexdigest(),
                }
            )

            # Running token totals across all LLM calls (summed once per
            # response, cache tokens included — per the evidence standard).
            total_input_tokens = 0
            total_output_tokens = 0
            total_cache_creation_tokens = 0
            total_cache_read_tokens = 0

            # ── Signal tracking for confidence scoring ───────────────
            successful_tool_calls = 0
            failed_tool_calls = 0
            sql_retry_count = 0
            total_rows_loaded = 0
            resource_ids_seen: set[str] = set()
            max_semantic_score = 0.0
            resource_metadata_modified: str | None = None
            resource_record_counts: list[int] = []
            all_tool_result_texts: list[str] = []
            _prev_sql_errors: set[str] = set()  # track SQL tool_use IDs that errored

            for iteration in range(max_iterations):
                iter_start = time.monotonic()
                response = await self._call_llm_with_retry(
                    client,
                    model=current_model,
                    max_tokens=settings.llm_max_tokens,
                    system=system_prompt,
                    tools=all_tools,
                    messages=messages,
                    event_log=agent_log,
                )
                # If the primary model was overloaded and we fell back,
                # stick with the fallback for remaining iterations.
                if hasattr(response, "_used_model"):
                    current_model = response._used_model
                llm_elapsed_ms = round((time.monotonic() - iter_start) * 1000)

                # Log the LLM response
                response_texts = []
                response_tool_calls = []
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        response_texts.append(block.text)
                    elif block.type == "tool_use":
                        response_tool_calls.append(
                            {
                                "tool": block.name,
                                "input": block.input,
                                "id": block.id,
                            }
                        )

                # Complete token accounting per the evidence standard: cache
                # creation/read tokens are part of the prompt-token cost and
                # must be summed once per response (message id recorded so
                # downstream packaging can dedupe).
                usage = response.usage
                cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens
                total_cache_creation_tokens += cache_creation
                total_cache_read_tokens += cache_read

                agent_log.append(
                    {
                        "type": "llm_response",
                        "timestamp": _utc_now(),
                        "iteration": iteration + 1,
                        "message_id": response.id,
                        "model": current_model,
                        "stop_reason": response.stop_reason,
                        "texts": response_texts,
                        "tool_calls": response_tool_calls,
                        "tokens": {
                            "input": usage.input_tokens,
                            "output": usage.output_tokens,
                            "cache_creation_input": cache_creation,
                            "cache_read_input": cache_read,
                        },
                        "duration_ms": llm_elapsed_ms,
                    }
                )

                # Finished — extract text answer
                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_answer = block.text
                    break

                # Tool use — execute requested tools
                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue

                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        self.logger.info(
                            f"Tool call: {tool_name}",
                            input=str(tool_input)[:200],
                        )

                        tool_start = time.monotonic()

                        # Route to MCP or CKAN tool handler
                        if tool_name.startswith("mcp__"):
                            result_text = await self._execute_mcp_tool(tool_name, tool_input)
                            code = self._code_for_mcp_tool(tool_name, tool_input, result_text)
                        else:
                            result_text = await self._execute_tool(
                                tool_name, tool_input, portal_url
                            )
                            code = self._code_for_tool(tool_name, tool_input, portal_url)

                        tool_elapsed_ms = round((time.monotonic() - tool_start) * 1000)

                        # ── Capture confidence signals ───────────
                        # A capability the portal does not have is neither a
                        # success nor a failure — it is excluded from the
                        # success-rate denominator so the rate keeps measuring
                        # only calls that actually attempted retrieval.
                        is_unavailable = result_text.startswith(
                            self.TOOL_UNAVAILABLE_PREFIX
                        )
                        is_error = not is_unavailable and result_text.startswith(
                            ("Error:", "HTTP ", "SQL error")
                        )
                        if is_unavailable:
                            # Withdraw the tool for the REST of this run, not
                            # just the next one. Telling the model in prose not
                            # to retry did not work — a measured trial had it
                            # call run_sql_query five times after being told the
                            # endpoint was blocked. `tools` is re-read every
                            # iteration, so removing it here actually stops it.
                            before = len(all_tools)
                            all_tools = [
                                t for t in all_tools if t.get("name") != tool_name
                            ]
                            self.logger.info(
                                "Tool unavailable on this portal — excluded from "
                                "retrieval success rate and withdrawn for the "
                                "rest of this run",
                                tool=tool_name,
                                withdrawn=before != len(all_tools),
                            )
                        elif is_error:
                            failed_tool_calls += 1
                            # Detect SQL retries: same resource queried
                            # again after a previous SQL error
                            if tool_name == "run_sql_query":
                                sql_retry_count += 1
                        else:
                            successful_tool_calls += 1

                        all_tool_result_texts.append(result_text[:10000])

                        # Semantic search score
                        if tool_name == "semantic_search_resources" and not is_error:
                            import re as _re

                            pct_matches = _re.findall(r"\((\d+\.?\d*)% match\)", result_text)
                            for pct in pct_matches:
                                score_val = float(pct) / 100.0
                                if score_val > max_semantic_score:
                                    max_semantic_score = score_val

                        # Row/record counts from load_resource_data
                        if tool_name == "load_resource_data" and not is_error:
                            import re as _re

                            total_match = _re.search(r"Total records:\s*([\d,]+)", result_text)
                            if total_match:
                                # CKAN's DataStore reports the resource total.
                                try:
                                    count = int(total_match.group(1).replace(",", ""))
                                    resource_record_counts.append(count)
                                    total_rows_loaded += min(count, tool_input.get("limit", 100))
                                except ValueError:
                                    pass
                            else:
                                # DCAT distributions are files: there is no
                                # server-side total, only what we actually
                                # streamed. Count the real rows read, and treat
                                # the file's size as known ONLY when the read
                                # ran to completion — a truncated read tells us
                                # nothing about the dataset's true size, and
                                # recording a floor as if it were the total
                                # would inflate the source-metadata factor with
                                # a number nobody measured.
                                loaded_match = _re.search(
                                    r"Loaded:\s*([\d,]+)\s+rows", result_text
                                )
                                if loaded_match:
                                    try:
                                        loaded = int(loaded_match.group(1).replace(",", ""))
                                        total_rows_loaded += loaded
                                        if "NOTE: truncated" not in result_text:
                                            resource_record_counts.append(loaded)
                                    except ValueError:
                                        pass
                            rid = tool_input.get("resource_id", "")
                            if rid:
                                resource_ids_seen.add(rid)

                        # SQL queries also touch resources
                        if tool_name == "run_sql_query" and not is_error:
                            import re as _re

                            uuids = _re.findall(
                                r'"([0-9a-f]{8}-[0-9a-f-]{27})"',
                                tool_input.get("sql", ""),
                            )
                            resource_ids_seen.update(uuids)

                        # Resource metadata (from get_dataset_info)
                        if tool_name == "get_dataset_info" and not is_error:
                            import re as _re

                            mod_match = _re.search(r"Modified:\s*(\d{4}-\d{2}-\d{2})", result_text)
                            if mod_match:
                                mod_str = mod_match.group(1)
                                if (
                                    resource_metadata_modified is None
                                    or mod_str > resource_metadata_modified
                                ):
                                    resource_metadata_modified = mod_str

                        tool_calls_trace.append(
                            {
                                "agent": self.name,
                                "action": tool_name,
                                "tool_name": tool_name,
                                "arguments": tool_input,
                                "result_preview": result_text[:800],
                                "code": code,
                            }
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_text[:10000],
                            }
                        )

                        # The logged result is byte-identical to the content
                        # returned to the model above (verbatim capture), with
                        # any truncation made explicit.
                        agent_log.append(
                            {
                                "type": "tool_execution",
                                "timestamp": _utc_now(),
                                "iteration": iteration + 1,
                                "tool": tool_name,
                                "tool_use_id": tool_id,
                                "source": _source_for_tool(tool_name, data_source),
                                "operation_type": _operation_type_for_tool(tool_name),
                                "status": "error" if is_error else "success",
                                "input": tool_input,
                                "result": result_text[:10000],
                                "result_chars": len(result_text),
                                "result_truncated": len(result_text) > 10000,
                                "duration_ms": tool_elapsed_ms,
                            }
                        )

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})
                    continue

                # Any other stop reason — grab text and finish
                for block in response.content:
                    if hasattr(block, "text"):
                        final_answer = block.text
                break

            elapsed = time.monotonic() - start

            answer_from_model = final_answer is not None
            if not final_answer:
                final_answer = (
                    "I wasn't able to complete the analysis for this question. "
                    "It may help to rephrase it — for example:\n\n"
                    "- Name a specific place (a city, county, or state)\n"
                    "- Name a specific time period (a year or range of years)\n"
                    "- Ask about one metric at a time\n\n"
                    'You can also ask *"What datasets are available on this topic?"* '
                    "to see what I can search."
                )

            # -- populate state for downstream nodes -----------------------

            state["answer"] = final_answer
            state["execution_trace"].extend(tool_calls_trace)

            # Store raw agent log for admin review / evidence packaging.
            # `model` is the model that actually served the final response
            # (may differ from the configured primary after a fallback).
            agent_log.append(
                {
                    "type": "summary",
                    "timestamp": _utc_now(),
                    "total_iterations": iteration + 1,
                    "total_tool_calls": len(tool_calls_trace),
                    "total_elapsed_ms": round(elapsed * 1000),
                    "model": current_model,
                    "primary_model": settings.llm_model,
                    "data_source": data_source,
                    "answer_from_model": answer_from_model,
                    # Named token_totals (not "tokens") so per-entry aggregators
                    # that sum every entry's "tokens" dict don't double-count.
                    "token_totals": {
                        "input": total_input_tokens,
                        "output": total_output_tokens,
                        "cache_creation_input": total_cache_creation_tokens,
                        "cache_read_input": total_cache_read_tokens,
                    },
                }
            )
            state["agent_log"] = agent_log

            # Always TIER_2 so notebook is generated
            state["intent"] = QueryIntent.FACTUAL_LOOKUP
            state["tier"] = QueryTier.TIER_2
            state["quick_answer_mode"] = False

            # ── Build real confidence signals ────────────────────────
            tool_signals = ToolCallSignals(
                successful_tool_calls=successful_tool_calls,
                failed_tool_calls=failed_tool_calls,
                sql_retry_count=sql_retry_count,
                total_rows_loaded=total_rows_loaded,
                distinct_resources_used=len(resource_ids_seen),
                iterations_used=iteration + 1,
                max_semantic_score=max_semantic_score,
                resource_metadata_modified=resource_metadata_modified,
                resource_record_counts=resource_record_counts,
            )
            state["tool_call_signals"] = tool_signals
            state["tool_result_texts"] = all_tool_result_texts

            # parse_confidence is no longer fabricated — the signal-based
            # calculator ignores it for LLM-graph queries. Keep a
            # reasonable value for any code that reads it defensively.
            state["parse_confidence"] = 0.0

            # Cite the sources the run ACTUALLY used, derived from the tool
            # trace. Claude fans out across portals and MCP servers within
            # one conversation, so citing only the nominal portal produced
            # false provenance — e.g. a notebook built entirely from Census
            # and FBI MCP calls citing WPRDC as its data source.
            used_sources = _sources_from_trace(
                tool_calls_trace, data_source, portal_cfg, portal_url
            )
            state["citations"] = [
                Citation(
                    source=src,
                    dataset_title=src.name,
                    url=src.url,
                )
                for src in used_sources
            ]

            state["retrieved_data"] = RetrievedData(
                observations=[],
                source_info=used_sources,
                retrieval_method="llm_tool_calling",
                retrieval_score=0.0,  # no longer used for confidence
            )

            state["computed_results"] = {
                "computation_type": "llm_analysis",
                "tool_calls": len(tool_calls_trace),
            }

            self.logger.info(
                "LLM analysis complete",
                tool_calls=len(tool_calls_trace),
                answer_len=len(final_answer),
                elapsed_s=round(elapsed, 1),
            )

        except Exception as exc:
            self.logger.error("LLM analysis failed", error=str(exc))
            state["error"] = f"Analysis error: {exc}"
            # Friendly, non-technical answer — the raw error stays in
            # state["error"], the server logs, and the agent log for admins.
            state["answer"] = (
                "I ran into a problem while analyzing your question and couldn't "
                "finish this one. Please try again in a moment — or try asking it "
                "a different way:\n\n"
                "- Narrow it to a specific place or time period\n"
                "- Ask about one metric at a time\n"
                '- Ask *"What datasets are available on this topic?"* to see '
                "what I can search"
            )
            # Preserve the evidence record up to the failure point.
            agent_log.append(
                {
                    "type": "error",
                    "timestamp": _utc_now(),
                    "iteration": iteration + 1,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
            )
            state["agent_log"] = agent_log

        return state


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_llm_agent: LLMAnalysisAgent | None = None


def get_llm_agent() -> LLMAnalysisAgent:
    """Get or create the LLM analysis agent singleton."""
    global _llm_agent
    if _llm_agent is None:
        _llm_agent = LLMAnalysisAgent()
    return _llm_agent
