"""Turn onboarded resource metadata into Pinecone records and upload them.

The onboarding scripts profile each CSV resource with qsv and write a
``meta.json`` describing its columns.  This module converts those records into
the document shape the agent's ``semantic_search_resources`` tool queries, and
upserts them into the integrated-embedding index.

The contract with the query side is exact: the field names produced here are
the ones ``connectors.pinecone_store._pinecone_search`` asks for in its
``fields`` list and filters on.  A record written with different names embeds
fine and then comes back with empty metadata, which is worse than not being
indexed at all — the agent sees a match it cannot describe or cite.  Change
one side and you must change the other.

``text`` is the field the index embeds.  It is assembled from everything a
person would use to recognise the dataset — title, description, resource name,
AI tags, and the column names, labels, and descriptions — because the agent
searches by concept ("air quality") rather than by column name.
"""

from __future__ import annotations

import re
from typing import Any

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# Column-name signals for the boolean facets the search tool filters on.
#
# These are matched against COLUMN metadata only — never the dataset title or
# description.  A title like "... Health Care Cost Containment Council" would
# otherwise set has_financial on a dataset with no financial column at all,
# and a facet that is true for everything is worse than no facet: it silently
# widens every filtered search instead of narrowing it.
#
# For the same reason the hint lists stay specific. Generic words that appear
# in ordinary column names ("value", "fund", "total") were removed after they
# marked every profiled PA health dataset as financial.
_TEMPORAL_HINTS = (
    "date", "time", "year", "month", "day", "quarter", "period",
    "timestamp", "created", "updated", "issued", "week",
)
_GEOGRAPHIC_HINTS = (
    "county", "state", "city", "zip", "postal", "municipal", "region",
    "district", "address", "latitude", "longitude", "lat", "lon", "geo",
    "tract", "block", "neighborhood", "borough", "ward", "fips", "location",
)
_DEMOGRAPHIC_HINTS = (
    "age", "race", "sex", "gender", "ethnicity", "hispanic", "population",
    "demographic", "household", "income bracket", "education", "marital",
)
_FINANCIAL_HINTS = (
    "amount", "cost", "price", "revenue", "expense", "budget", "salary",
    "wage", "payment", "spend", "dollar", "usd", "income",
)

# Embedded-text budget.
#
# These numbers are measured, not guessed. The first version emitted up to 8000
# chars per record (median 4963, many at the cap) and retrieval collapsed: every
# similarity score landed in 0.17-0.28 and short records became the top hit for
# unrelated queries, because one vector over a long blob averages the topic away
# while generic column names ("Site #; Address; Begin Date") crowd out the
# distinctive words.
#
# Re-benchmarked on 35 labeled queries against the 110-dataset WPRDC corpus,
# tightening to these budgets raised precision@1 from 69% to 89% (hit@3 94%),
# versus 66% for the portal's own keyword search over the same corpus. Variants
# that dropped column data entirely, or kept only corpus-rare column vocabulary,
# scored the same or worse — so the win comes from LENGTH DISCIPLINE, not from
# choosing cleverer column content.
#
# Raising these caps is a retrieval regression unless re-measured — see
# tests/unit/test_pinecone_upload_text.py, which guards these budgets.
MAX_TEXT_CHARS = 800
_MAX_DESCRIPTION_CHARS = 300
_MAX_FIELDS_CHARS = 200
_MAX_VALUES_CHARS = 200
_MAX_FIELD_CHARS = 2000  # metadata fields (description), not the embedded text

# Column names and values this generic carry no topical signal — they appear in
# most datasets, so including them pushes every record toward the same region of
# the embedding space.
_GENERIC_TERMS = frozenset({
    "id", "objectid", "fid", "globalid", "name", "address", "street", "city",
    "state", "zip", "zipcode", "lat", "lon", "latitude", "longitude", "x", "y",
    "geom", "shape", "the_geom", "date", "begin", "end", "created", "updated",
    "modified", "year", "month", "day", "type", "status", "notes", "description",
    "comment", "count", "total", "number", "num", "code", "value", "label",
    "area", "length", "perimeter", "n/a", "null", "none", "unknown", "other",
    "yes", "no", "true", "false",
})

# qsv describegpt sometimes tags a dataset with words describing its SCHEMA
# rather than its subject ("identifier", "float", "boolean"). Measured on the
# WPRDC corpus: 23 of 110 records carry at least one. They describe every
# dataset equally, so they pull records together instead of apart.
_SCHEMA_TAGS = frozenset({
    "identifier", "unique identifier", "metadata", "data type", "data record",
    "float", "integer", "string", "boolean", "categorical", "textual",
    "numerical", "structured data", "record", "field", "column", "table",
    "dataset", "schema", "primary key", "row", "rows", "attribute",
})


def _matches(haystack: str, hints: tuple[str, ...]) -> bool:
    return any(h in haystack for h in hints)


def _column_blob(column: dict[str, Any]) -> str:
    """All searchable text for one column, lowercased."""
    parts = [
        str(column.get("name", "")),
        str(column.get("qsv_label", "")),
        str(column.get("qsv_description", "")),
    ]
    info = column.get("ckan_info")
    if isinstance(info, dict):
        parts.extend(str(v) for v in info.values())
    return " ".join(parts).lower()


def _temporal_range(columns: list[dict[str, Any]]) -> tuple[str, str]:
    """Best-effort min/max over columns that look temporal.

    Uses qsv's own ``stats`` min/max rather than re-parsing values.  Returns
    empty strings when nothing temporal is present — the search tool treats
    those as "no coverage recorded", which is the honest answer.
    """
    mins: list[str] = []
    maxs: list[str] = []
    for col in columns:
        if not _matches(_column_blob(col), _TEMPORAL_HINTS):
            continue
        stats = col.get("stats")
        if not isinstance(stats, dict):
            continue
        lo, hi = str(stats.get("min", "")).strip(), str(stats.get("max", "")).strip()
        # Keep only values that start with a plausible year or date.
        if re.match(r"^(1[89]|20)\d{2}", lo):
            mins.append(lo)
        if re.match(r"^(1[89]|20)\d{2}", hi):
            maxs.append(hi)
    return (min(mins) if mins else "", max(maxs) if maxs else "")


def _is_generic(term: str) -> bool:
    return term.strip().lower() in _GENERIC_TERMS


def build_text(resource: dict[str, Any]) -> str:
    """Assemble the natural-language blob that gets embedded.

    Ordered by signal density and budgeted per section: the topic (title,
    description, tags) comes first and gets most of the room, then a short list
    of non-generic field names, then a few distinctive values. See the budget
    constants above for why the total is small.
    """
    columns = resource.get("columns") or []
    lines: list[str] = []

    # -- topic: what the dataset is about (highest signal, always first) ----
    title = resource.get("dataset_title") or resource.get("resource_name") or ""
    if title:
        lines.append(str(title).strip())

    description = (
        resource.get("qsv_description") or resource.get("dataset_description") or ""
    )
    if description:
        lines.append(str(description).strip()[:_MAX_DESCRIPTION_CHARS])

    # Merge both tag sources rather than preferring one: 84 of 110 WPRDC
    # records have BOTH, and `qsv_tags or dataset_tags` silently discarded the
    # curator's own tags on every one of them.
    merged_tags: list[str] = []
    for source in (resource.get("qsv_tags"), resource.get("dataset_tags")):
        if isinstance(source, list):
            for tag in source:
                text = str(tag).strip()
                if text and text.lower() not in _SCHEMA_TAGS:
                    merged_tags.append(text)
    if merged_tags:
        deduped = list(dict.fromkeys(t.lower() for t in merged_tags))
        lines.append("Topics: " + ", ".join(deduped[:14]))

    # -- structure: field names, minus the ones every dataset has -----------
    names = [
        str(c.get("name", "")).strip()
        for c in columns[:40]
        if str(c.get("name", "")).strip() and not _is_generic(str(c.get("name", "")))
    ]
    if names:
        lines.append("Fields: " + ", ".join(names)[:_MAX_FIELDS_CHARS])

    # -- contents: a few distinctive values, so a dataset is findable by what
    # is *in* it ("potholes" only ever appears as a 311 request-type value).
    values: list[str] = []
    for col in columns[:20]:
        for entry in (col.get("top_values") or [])[:3]:
            val = str(entry.get("value", "")).strip()
            if not val or _is_generic(val) or val.lower() == "(nullcount)":
                continue
            if val.replace(".", "").replace("-", "").isdigit():
                continue
            values.append(val)
    if values:
        lines.append("Values: " + ", ".join(dict.fromkeys(values))[:_MAX_VALUES_CHARS])

    return "\n".join(lines)[:MAX_TEXT_CHARS]


def build_record(resource: dict[str, Any], *, site_id: str) -> dict[str, Any] | None:
    """Build one Pinecone record from an onboarded resource's ``meta.json``.

    Returns ``None`` when the resource has no usable identity or no text worth
    embedding — an empty record would occupy an ID and match nothing.
    """
    resource_id = str(
        resource.get("resource_id") or resource.get("dataset_id") or ""
    ).strip()
    if not resource_id:
        return None

    text = build_text(resource)
    if not text.strip():
        return None

    columns = resource.get("columns") or []
    # Facets describe what COLUMNS the table has, so only column metadata is
    # consulted (see the hint-list comment above).
    haystack = " ".join(_column_blob(c) for c in columns)

    temporal_min, temporal_max = _temporal_range(columns)
    tags = resource.get("qsv_tags") or resource.get("dataset_tags") or []
    description = str(
        resource.get("qsv_description") or resource.get("dataset_description") or ""
    )[:_MAX_FIELD_CHARS]

    return {
        # Namespacing by site keeps two portals' identically-named resources
        # from overwriting each other in a shared index.
        "_id": f"{site_id}:{resource_id}",
        "text": text,
        "resource_id": resource_id,
        "resource_name": str(resource.get("resource_name", "") or "")[:500],
        "dataset_id": str(resource.get("dataset_id", "") or "")[:500],
        "dataset_title": str(resource.get("dataset_title", "") or "")[:500],
        "site_id": site_id,
        "format": str(resource.get("format", "CSV") or "CSV"),
        "record_count": int(resource.get("row_count") or 0),
        "column_count": len(columns),
        "ai_tags": ", ".join(str(t) for t in tags[:30]) if isinstance(tags, list) else "",
        "description": description,
        "has_temporal": _matches(haystack, _TEMPORAL_HINTS),
        "has_geographic": _matches(haystack, _GEOGRAPHIC_HINTS),
        "has_demographic": _matches(haystack, _DEMOGRAPHIC_HINTS),
        "has_financial": _matches(haystack, _FINANCIAL_HINTS),
        "temporal_min": temporal_min,
        "temporal_max": temporal_max,
        "source_url": str(resource.get("source_url", "") or "")[:1000],
    }


def build_records(index_data: dict[str, Any], *, site_id: str) -> list[dict[str, Any]]:
    """Build records for every resource in an onboarding ``index.json``."""
    records: list[dict[str, Any]] = []
    for dataset in index_data.get("datasets", []):
        for resource in dataset.get("resources", []):
            merged = {
                "dataset_id": dataset.get("dataset_id", ""),
                "dataset_title": dataset.get("dataset_title", ""),
                "dataset_description": dataset.get("dataset_description", ""),
                "organization": dataset.get("organization", ""),
                "dataset_tags": dataset.get("tags", []),
                **resource,
            }
            record = build_record(merged, site_id=site_id)
            if record is not None:
                records.append(record)
    return records


def upload_index(
    index_data: dict[str, Any],
    *,
    site_id: str,
    namespace: str | None = None,
    index_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build and upsert records for a whole onboarded portal.

    With ``dry_run`` the records are built and summarised but nothing is sent,
    so an operator can inspect what would be written before touching a live
    index.
    """
    records = build_records(index_data, site_id=site_id)
    summary: dict[str, Any] = {
        "site_id": site_id,
        "records_built": len(records),
        "dry_run": dry_run,
    }
    if not records:
        summary["upserted"] = 0
        summary["errors"] = ["no records built from index (nothing profiled?)"]
        return summary

    if dry_run:
        summary["upserted"] = 0
        summary["sample"] = records[0]
        return summary

    from data_concierge.data_layer.connectors.pinecone_store import PineconeVectorStore

    store = PineconeVectorStore(index_name=index_name, namespace=namespace)
    result = store.upsert_records(records)
    summary.update(result)
    return summary


__all__ = [
    "MAX_TEXT_CHARS",
    "build_record",
    "build_records",
    "build_text",
    "upload_index",
]
