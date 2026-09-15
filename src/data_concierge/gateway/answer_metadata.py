"""Public answer provenance, keeping observation, retrieval and review dates separate.

Only structured data is used here. A year mentioned in a question or an answer,
a notebook's generation time, and an approval timestamp cannot establish which
period the underlying data covers.
"""

import re
from datetime import date, datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field


class EvidenceSource(BaseModel):
    name: str
    url: str
    description: str | None = None


class DataPeriod(BaseModel):
    start: str
    end: str


class AnswerEvidence(BaseModel):
    sources: list[EvidenceSource] = Field(default_factory=list)
    data_period: DataPeriod | None = None
    retrieved_at: str | None = None
    source_updated_at: str | None = None
    verified_at: str | None = None
    verification_status: Literal["reviewed", "pending", "unreviewed", "unknown"] = "unknown"
    original_query: str | None = None


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        # Some legacy models synthesize dates through default factories. Only
        # explicitly stored values count as provenance.
        return value.model_dump(exclude_unset=True)
    return value if isinstance(value, dict) else {}


def _date(value: Any) -> str | None:
    """Accept real ISO dates/timestamps, retaining an observation's granularity."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    value = value.strip()
    try:
        if re.fullmatch(r"\d{4}", value):
            date(int(value), 1, 1)
        elif re.fullmatch(r"\d{4}-\d{2}", value):
            date.fromisoformat(value + "-01")
        elif re.fullmatch(r"\d{4}-Q[1-4]", value):
            date(int(value[:4]), 1, 1)
        else:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value or None


def _sources(*groups: Any) -> list[EvidenceSource]:
    sources: dict[str, EvidenceSource] = {}
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for item in group:
            item = _record(item)
            source = _record(item.get("source"))
            url = item.get("url") or source.get("url")
            if not isinstance(url, str):
                continue
            try:
                parsed = urlsplit(url)
            except ValueError:
                continue
            if parsed.scheme not in {"https", "http"} or not parsed.netloc:
                continue
            if parsed.username or parsed.password:
                continue
            name = item.get("name") or source.get("name")
            if not name and isinstance(item.get("source"), str):
                name = item["source"]
            description = (
                item.get("description") or item.get("dataset_title") or item.get("dataset")
            )
            sources.setdefault(
                url,
                EvidenceSource(
                    name=str(name or parsed.hostname or "Source"),
                    url=url,
                    description=str(description) if description else None,
                ),
            )
    return list(sources.values())


def evidence_from_state(state: dict[str, Any], *, original_query: str) -> AnswerEvidence:
    """Read sources, actual observation dates and source-maintenance signals."""
    retrieved = _record(state.get("retrieved_data"))
    citations = state.get("citations") or []
    observations = retrieved.get("observations") or []
    dates = [
        observation_date
        for observation in observations
        if (observation_date := _date(_record(observation).get("date")))
    ]
    accessed = [
        access_date
        for citation in citations
        if (
            access_date := _date(
                getattr(citation, "access_date", _record(citation).get("access_date"))
            )
        )
    ]
    signals = _record(state.get("tool_call_signals"))
    modified = [
        modified_date
        for value in (signals.get("resource_metadata_modified"), retrieved.get("data_vintage"))
        if (modified_date := _date(value))
    ]
    return AnswerEvidence(
        sources=_sources(
            citations,
            state.get("source_links"),
            retrieved.get("source_info"),
            [_record(observation).get("source") for observation in observations],
        ),
        data_period=DataPeriod(start=min(dates), end=max(dates)) if dates else None,
        retrieved_at=max(accessed) if accessed else None,
        source_updated_at=max(modified) if modified else None,
        verification_status="unreviewed",
        original_query=original_query or None,
    )


def evidence_from_saved(
    value: Any, *, reviewed: bool = False, stored_evidence: Any = None
) -> AnswerEvidence:
    """Read a saved answer without turning its access time into data freshness.

    ``reviewed`` is supplied by a verified-library caller, never taken from
    client-controlled notebook metadata. Imported evidence cannot self-attest
    that an answer has received human review.
    """
    saved = _record(value)
    notebook = _record(saved.get("notebook_json"))
    metadata = _record(_record(notebook.get("metadata")).get("data_concierge"))
    evidence = _record(stored_evidence or metadata.get("evidence"))
    period = _record(evidence.get("data_period"))
    start, end = _date(period.get("start")), _date(period.get("end"))
    observation_date = _date(saved.get("date"))
    if not start or not end or start > end:
        start = end = observation_date
    return AnswerEvidence(
        sources=_sources(
            evidence.get("sources"), saved.get("source_links"), saved.get("citations")
        ),
        data_period=DataPeriod(start=start, end=end) if start and end else None,
        retrieved_at=_date(evidence.get("retrieved_at")),
        source_updated_at=_date(evidence.get("source_updated_at")),
        verified_at=_date(saved.get("verified_at")) if reviewed else None,
        verification_status="reviewed" if reviewed else "unknown",
        original_query=saved.get("query") or metadata.get("query") or None,
    )


def attach_notebook_evidence(notebook: dict[str, Any], evidence: AnswerEvidence) -> None:
    """Keep structured provenance with downloaded and approved notebooks."""
    metadata = notebook.setdefault("metadata", {})
    concierge = metadata.setdefault("data_concierge", {})
    concierge["evidence"] = evidence.model_dump(mode="json")
