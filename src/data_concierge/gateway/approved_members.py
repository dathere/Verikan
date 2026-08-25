"""Approved members list for Auth0 social login.

An email must be in this list before the Auth0 flow will create a session
for it. Managed by admins via the admin panel. Persisted through the unified
storage backend so it survives restarts on Cloud Run.

Seeded on first load from the ``APPROVED_MEMBERS`` env var (comma-separated
emails) if the stored list is empty.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_KEY = "approved_members.json"


def _normalize(email: str) -> str:
    return email.strip().lower()


def _load_raw() -> dict[str, Any]:
    data = storage.read_json(_KEY)
    if data and isinstance(data.get("members"), list):
        return data
    # Seed from env var
    seed_raw = os.environ.get("APPROVED_MEMBERS", "")
    seeds = [_normalize(x) for x in seed_raw.split(",") if x.strip()]
    members = [
        {"email": e, "added_by": "env", "added_at": datetime.utcnow().isoformat() + "Z"}
        for e in seeds
    ]
    payload = {"members": members}
    storage.write_json(_KEY, payload)
    return payload


def _save(members: list[dict[str, Any]]) -> None:
    storage.write_json(_KEY, {"members": members})


def list_members() -> list[dict[str, Any]]:
    return list(_load_raw().get("members", []))


def is_approved(email: str) -> bool:
    if not email:
        return False
    target = _normalize(email)
    for m in list_members():
        if _normalize(m.get("email", "")) == target:
            return True
    return False


def add_member(
    email: str,
    added_by: str = "admin",
    display_name: str = "",
) -> dict[str, Any]:
    email = _normalize(email)
    members = list_members()
    for m in members:
        if _normalize(m.get("email", "")) == email:
            # Update display_name if we now know it
            if display_name and not m.get("display_name"):
                m["display_name"] = display_name
                _save(members)
            return m
    entry: dict[str, Any] = {
        "email": email,
        "added_by": added_by,
        "added_at": datetime.utcnow().isoformat() + "Z",
    }
    if display_name:
        entry["display_name"] = display_name
    members.append(entry)
    _save(members)
    logger.info("Approved member added", email=email, added_by=added_by)
    return entry


def remove_member(email: str) -> bool:
    email = _normalize(email)
    members = list_members()
    new_members = [m for m in members if _normalize(m.get("email", "")) != email]
    if len(new_members) == len(members):
        return False
    _save(new_members)
    logger.info("Approved member removed", email=email)
    return True


# =========================================================================
# Pending access requests — recorded when unapproved users try to log in
# =========================================================================

_PENDING_KEY = "pending_requests.json"


def _load_pending() -> list[dict[str, Any]]:
    data = storage.read_json(_PENDING_KEY)
    if data and isinstance(data.get("requests"), list):
        return data["requests"]
    return []


def _save_pending(requests: list[dict[str, Any]]) -> None:
    storage.write_json(_PENDING_KEY, {"requests": requests})


def add_pending_request(email: str, name: str = "") -> dict[str, Any]:
    """Record a pending access request. Updates the timestamp if already present."""
    email = _normalize(email)
    requests = _load_pending()
    for req in requests:
        if _normalize(req.get("email", "")) == email:
            req["requested_at"] = datetime.utcnow().isoformat() + "Z"
            if name:
                req["name"] = name
            _save_pending(requests)
            return req
    entry: dict[str, Any] = {
        "email": email,
        "name": name,
        "requested_at": datetime.utcnow().isoformat() + "Z",
    }
    requests.append(entry)
    _save_pending(requests)
    logger.info("Pending access request recorded", email=email, name=name)
    return entry


def list_pending_requests() -> list[dict[str, Any]]:
    """Return all pending access requests, most recent first."""
    requests = _load_pending()
    # Exclude anyone who has since been approved
    pending = [r for r in requests if not is_approved(r.get("email", ""))]
    pending.sort(key=lambda r: r.get("requested_at", ""), reverse=True)
    return pending


def remove_pending_request(email: str) -> bool:
    """Remove a pending request (after approval or dismissal)."""
    email = _normalize(email)
    requests = _load_pending()
    new_requests = [r for r in requests if _normalize(r.get("email", "")) != email]
    if len(new_requests) == len(requests):
        return False
    _save_pending(new_requests)
    return True
