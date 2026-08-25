"""Search index over onboarded CKAN data dictionaries.

Loads the index.json produced by ``scripts/onboard_ckan.py`` and provides
keyword-based search across dataset titles, descriptions, column metadata,
tags, and top frequency values.  Results include rich column-level detail
(stats, top values, AI-generated descriptions) so the agent can often
answer metadata questions without loading row-level data.

Usage::

    from data_concierge.data_layer.onboard_index import get_onboarded_index

    index = get_onboarded_index()
    if index.is_available():
        results = index.search("311 complaints pittsburgh", n_results=5)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# qsv describegpt records its full invocation in the AI-generated descriptions,
# and that command carries the OpenRouter API key on --api-key. Those
# descriptions are served over the Fair Store API, rendered in the data
# dictionary UI, and fed into the agent's prompt — so the key would leak three
# ways. Redact provider secrets and the key flag before any description is used.
_SECRET_PATTERNS = [
    re.compile(r"--api[-_]?key(?:\s+|=)\S+", re.IGNORECASE),
    re.compile(r"sk-or-v1-[A-Za-z0-9]+"),      # OpenRouter
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),  # Anthropic
    re.compile(r"sk-[A-Za-z0-9]{20,}"),        # OpenAI-style
]


def _scrub_secrets(text: str) -> str:
    """Redact provider API keys embedded in onboarded description text."""
    if not text:
        return text
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[redacted]", text)
    return text


@dataclass
class _ColumnSummary:
    name: str
    qsv_label: str = ""
    qsv_description: str = ""
    ckan_type: str = ""
    qsv_type: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    top_values: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _SearchEntry:
    dataset_id: str
    dataset_title: str
    dataset_description: str
    organization: str
    dataset_tags: list[str]
    resource_id: str
    resource_name: str
    qsv_description: str
    qsv_tags: list[str]
    columns: list[_ColumnSummary]
    row_count: int
    file_size_bytes: int
    format: str
    local_path: str

    _title_text: str = ""
    _tag_text: str = ""
    _desc_text: str = ""
    _col_text: str = ""
    _topval_text: str = ""

    def build_search_fields(self) -> None:
        self._title_text = f"{self.dataset_title} {self.resource_name}".lower()
        self._tag_text = " ".join(self.dataset_tags + self.qsv_tags).lower()
        self._desc_text = f"{self.dataset_description} {self.qsv_description}".lower()
        col_parts = []
        topval_parts = []
        for c in self.columns:
            col_parts.extend([c.name, c.qsv_label, c.qsv_description])
            for tv in c.top_values:
                topval_parts.append(str(tv.get("value", "")))
        self._col_text = " ".join(col_parts).lower()
        self._topval_text = " ".join(topval_parts).lower()


# Stats we surface per column, mapped from the key qsv actually emits.
# qsv writes the median as "q2_median" (it is the second quartile), so a
# whitelist looking for "median" silently dropped it from every column.
_STAT_KEYS: tuple[tuple[str, str], ...] = (
    ("type", "type"),
    ("min", "min"),
    ("max", "max"),
    ("mean", "mean"),
    ("median", "q2_median"),
    ("stddev", "stddev"),
    ("nullcount", "nullcount"),
    ("cardinality", "cardinality"),
)


def _parse_column(raw: dict[str, Any]) -> _ColumnSummary:
    stats = raw.get("stats", {})
    summary_stats: dict[str, Any] = {}
    for out_key, qsv_key in _STAT_KEYS:
        value = stats.get(qsv_key)
        if value in ("", None):
            # Tolerate a source that already uses the plain name.
            value = stats.get(out_key)
        if value not in ("", None):
            summary_stats[out_key] = value
    return _ColumnSummary(
        name=raw.get("name", ""),
        qsv_label=raw.get("qsv_label", ""),
        qsv_description=_scrub_secrets(raw.get("qsv_description", "")),
        ckan_type=raw.get("ckan_type", ""),
        qsv_type=raw.get("qsv_type", ""),
        stats=summary_stats,
        top_values=raw.get("top_values", [])[:20],
    )


class OnboardedIndex:
    """Keyword search over pre-onboarded CKAN data dictionaries."""

    def __init__(self) -> None:
        self._entries: list[_SearchEntry] = []
        self._by_resource_id: dict[str, _SearchEntry] = {}
        self._loaded = False
        self._site_id: str = ""

    def load(self, site_id: str = "wprdc") -> bool:
        """Load the index via the storage backend (GCS or local).  Returns True on success."""
        self._site_id = site_id
        storage_key = f"ckan_onboard/{site_id}/index.json"

        try:
            data = storage.read_json(storage_key)
            if data is None:
                # Fallback: try local data/ directory directly
                local_path = _PROJECT_ROOT / "data" / "ckan_onboard" / site_id / "index.json"
                if local_path.exists():
                    with open(local_path) as f:
                        data = json.load(f)
                else:
                    logger.info("Onboarded index not found", key=storage_key)
                    return False
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load onboarded index", error=str(exc))
            return False

        self._entries = []
        self._by_resource_id = {}

        for ds in data.get("datasets", []):
            for res in ds.get("resources", []):
                if res.get("status") not in ("ok", None):
                    continue

                columns = [_parse_column(c) for c in res.get("columns", [])]
                entry = _SearchEntry(
                    dataset_id=ds.get("dataset_id", ""),
                    dataset_title=ds.get("dataset_title", ""),
                    dataset_description=_scrub_secrets(ds.get("dataset_description", "")),
                    organization=ds.get("organization", ""),
                    dataset_tags=ds.get("tags", []),
                    resource_id=res.get("resource_id", ""),
                    resource_name=res.get("resource_name", ""),
                    qsv_description=_scrub_secrets(res.get("qsv_description", "") or ""),
                    qsv_tags=res.get("qsv_tags", []) or [],
                    columns=columns,
                    row_count=res.get("row_count", 0) or 0,
                    file_size_bytes=res.get("file_size_bytes", 0) or 0,
                    format=res.get("format", "CSV"),
                    local_path=res.get("local_path", ""),
                )
                entry.build_search_fields()
                self._entries.append(entry)
                if entry.resource_id:
                    self._by_resource_id[entry.resource_id] = entry

        self._loaded = True
        logger.info(
            "Onboarded index loaded",
            site_id=site_id,
            resources=len(self._entries),
        )
        return True

    def is_available(self) -> bool:
        return self._loaded and len(self._entries) > 0

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Search the index by keyword matching.  Returns Pinecone-compatible dicts."""
        if not self.is_available():
            return []

        tokens = query.lower().split()
        if not tokens:
            return []

        scored: list[tuple[float, _SearchEntry]] = []
        for entry in self._entries:
            score = 0.0
            for token in tokens:
                if token in entry._title_text:
                    score += 3.0
                if token in entry._tag_text:
                    score += 2.0
                if token in entry._desc_text:
                    score += 1.5
                if token in entry._col_text:
                    score += 1.0
                if token in entry._topval_text:
                    score += 0.5
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        max_score = scored[0][0] if scored else 1.0
        results = []
        for score, entry in scored[:n_results]:
            results.append({
                "resource_id": entry.resource_id,
                "resource_name": entry.resource_name,
                "dataset_id": entry.dataset_id,
                "dataset_title": entry.dataset_title,
                "description": entry.qsv_description or entry.dataset_description,
                "record_count": entry.row_count,
                "column_count": len(entry.columns),
                "score": score / max_score,
                "source": "onboarded_index",
                "qsv_tags": entry.qsv_tags,
                "dataset_tags": entry.dataset_tags,
                "columns_detail": [
                    {
                        "name": c.name,
                        "label": c.qsv_label,
                        "description": c.qsv_description,
                        "type": c.qsv_type or c.ckan_type,
                        "stats": c.stats,
                        "top_values": c.top_values[:10],
                    }
                    for c in entry.columns
                ],
            })
        return results

    def get_resource_detail(self, resource_id: str) -> dict[str, Any] | None:
        """Look up a single resource by ID with full column metadata."""
        entry = self._by_resource_id.get(resource_id)
        if not entry:
            return None
        return {
            "resource_id": entry.resource_id,
            "resource_name": entry.resource_name,
            "dataset_id": entry.dataset_id,
            "dataset_title": entry.dataset_title,
            "description": entry.qsv_description or entry.dataset_description,
            "record_count": entry.row_count,
            "column_count": len(entry.columns),
            "qsv_tags": entry.qsv_tags,
            "dataset_tags": entry.dataset_tags,
            "columns_detail": [
                {
                    "name": c.name,
                    "label": c.qsv_label,
                    "description": c.qsv_description,
                    "type": c.qsv_type or c.ckan_type,
                    "stats": c.stats,
                    "top_values": c.top_values,
                }
                for c in entry.columns
            ],
        }


_onboarded_index: OnboardedIndex | None = None


def get_onboarded_index(site_id: str = "wprdc") -> OnboardedIndex:
    """Get the singleton OnboardedIndex instance."""
    global _onboarded_index  # noqa: PLW0603
    if _onboarded_index is None:
        _onboarded_index = OnboardedIndex()
        _onboarded_index.load(site_id)
    return _onboarded_index
