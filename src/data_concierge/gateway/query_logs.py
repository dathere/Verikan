"""Per-user query log storage.

Tracks every question a user asks, whether the answer came from a cached/verified
notebook or was freshly generated, plus metadata (confidence, tier, intent, timing).

Stored as a single JSON document under ``query_logs/index.json`` via the unified
storage backend (local or GCS). Capped at ``MAX_LOGS`` most-recent entries to
keep the file small enough to read in one shot for the admin UI.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_LOGS_KEY = "query_logs/index.json"
MAX_LOGS = 2000


def _load() -> list[dict[str, Any]]:
    data = storage.read_json(_LOGS_KEY)
    if not data:
        return []
    logs = data.get("logs", [])
    if isinstance(logs, list):
        return logs
    return []


def _save(logs: list[dict[str, Any]]) -> None:
    # Keep only the most-recent MAX_LOGS entries
    if len(logs) > MAX_LOGS:
        logs = logs[-MAX_LOGS:]
    storage.write_json(_LOGS_KEY, {"logs": logs})


def append_log(
    *,
    user: str,
    auth_type: str,
    query: str,
    source: str,
    query_id: str | None = None,
    confidence: float | None = None,
    confidence_level: str | None = None,
    tier: str | None = None,
    intent: str | None = None,
    processing_time_ms: int | None = None,
    had_notebook: bool = False,
    is_quick_answer: bool = False,
    similarity_score: float | None = None,
    verified_query: str | None = None,
    notebook_url: str | None = None,
    data_source: str | None = None,
    error: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict[str, Any]:
    """Append a single query-log entry and persist.

    ``source`` must be one of: ``generated``, ``verified_cache``, ``revision``
    (a chat follow-up that edited an existing notebook), or ``error``.
    """
    entry: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "user": user,
        "auth_type": auth_type,
        "query": query[:500],
        "source": source,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query_id": query_id,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "tier": tier,
        "intent": intent,
        "processing_time_ms": processing_time_ms,
        "had_notebook": had_notebook,
        "is_quick_answer": is_quick_answer,
        "similarity_score": similarity_score,
        "verified_query": verified_query,
        "notebook_url": notebook_url,
        "data_source": data_source,
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    try:
        logs = _load()
        logs.append(entry)
        _save(logs)
    except Exception as e:
        logger.error("Failed to persist query log", error=str(e), user=user)
    return entry


def list_logs(
    *,
    limit: int = 200,
    user: str | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Return most-recent logs (newest first), optionally filtered."""
    logs = _load()
    # Newest first
    logs = list(reversed(logs))
    if user:
        logs = [x for x in logs if x.get("user") == user]
    if source:
        logs = [x for x in logs if x.get("source") == source]
    return logs[:limit]


def summary() -> dict[str, Any]:
    """Aggregate counts for the admin dashboard."""
    logs = _load()
    total = len(logs)
    by_source: dict[str, int] = {}
    by_user: dict[str, int] = {}
    for x in logs:
        by_source[x.get("source", "unknown")] = by_source.get(x.get("source", "unknown"), 0) + 1
        u = x.get("user") or "unknown"
        by_user[u] = by_user.get(u, 0) + 1
    # Verified cache-hit rate = how often the verified corpus answered instantly
    # (verified_cache) vs. a fresh generation. Errors are excluded from the
    # denominator so a spate of failures doesn't distort the flywheel metric.
    generated = by_source.get("generated", 0)
    cached = by_source.get("verified_cache", 0)
    answerable = generated + cached
    cache_hit_rate = round(cached / answerable * 100) if answerable else None
    return {
        "total": total,
        "by_source": by_source,
        "by_user": by_user,
        "unique_users": len(by_user),
        "cache_hit_rate": cache_hit_rate,
    }
