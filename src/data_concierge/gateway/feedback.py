"""Answer-feedback store.

Captures a lightweight 👍 / 👎 signal (plus an optional note) on individual
assistant answers so admins can see which answers land well and which need
work — the missing quality signal that should drive verified-library curation.

Stored as a single JSON document under ``feedback/index.json`` via the unified
storage backend (local or GCS), same pattern as ``query_logs``. Capped at
``MAX_FEEDBACK`` most-recent entries so the admin UI can read it in one shot.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_FEEDBACK_KEY = "feedback/index.json"
MAX_FEEDBACK = 2000


def _load() -> list[dict[str, Any]]:
    data = storage.read_json(_FEEDBACK_KEY)
    if not data:
        return []
    items = data.get("feedback", [])
    return items if isinstance(items, list) else []


def _save(items: list[dict[str, Any]]) -> None:
    if len(items) > MAX_FEEDBACK:
        items = items[-MAX_FEEDBACK:]
    storage.write_json(_FEEDBACK_KEY, {"feedback": items})


def append_feedback(
    *,
    rating: str,
    query: str,
    user: str,
    auth_type: str,
    answer_preview: str | None = None,
    query_id: str | None = None,
    source: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Append a single feedback entry and persist. ``rating`` is 'up' or 'down'."""
    entry: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "rating": "up" if rating == "up" else "down",
        "query": (query or "")[:500],
        "answer_preview": (answer_preview or "")[:500] or None,
        "user": user,
        "auth_type": auth_type,
        "query_id": query_id,
        "source": source,
        "note": (note or "")[:1000] or None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        items = _load()
        items.append(entry)
        _save(items)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Failed to persist feedback", error=str(e), user=user)
    return entry


def list_feedback(*, limit: int = 200, rating: str | None = None) -> list[dict[str, Any]]:
    """Return most-recent feedback (newest first), optionally filtered by rating."""
    items = list(reversed(_load()))
    if rating in ("up", "down"):
        items = [x for x in items if x.get("rating") == rating]
    return items[:limit]


def summary() -> dict[str, Any]:
    """Aggregate counts for the admin dashboard."""
    items = _load()
    up = sum(1 for x in items if x.get("rating") == "up")
    down = sum(1 for x in items if x.get("rating") == "down")
    total = up + down
    return {
        "total": total,
        "up": up,
        "down": down,
        # Satisfaction = share of positive ratings, rounded to a whole percent.
        "satisfaction_pct": round(up / total * 100) if total else None,
    }
