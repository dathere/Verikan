"""Role-based access control (RBAC) for user authorization.

Stores per-user roles in ``roles.json`` via the unified storage backend.
Currently the only meaningful role is ``admin``; the module is structured
so additional roles can be added later without schema changes.

Seeded on first load from the ``ADMIN_USERS`` env var (comma-separated
usernames or emails) if the stored role map is empty.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_KEY = "roles.json"
ADMIN_ROLE = "admin"
# Optional built-in fallback admin, empty by default: a public checkout grants
# admin to nobody until an operator names one. Set ADMIN_USERS in .env
# (comma-separated usernames or emails) to seed admins on first run. Every use
# below is guarded on this being non-empty — an empty value must never match a
# user id, or an anonymous/blank caller would be treated as an admin.
_DEFAULT_ADMIN = ""


def _normalize(user_id: str) -> str:
    return user_id.strip().lower()


def _load() -> dict[str, Any]:
    """Load roles data. Returns ``{"roles": {user_id: {"roles": [...], ...}}}``."""
    data = storage.read_json(_KEY)
    if data and data.get("roles"):  # non-empty roles dict → already seeded
        return data

    # Seed from env var + default admin on first run (or when roles dict is empty)
    seed_raw = os.environ.get("ADMIN_USERS", "")
    seeds = [_normalize(x) for x in seed_raw.split(",") if x.strip()]
    default_admin = _normalize(_DEFAULT_ADMIN)
    if default_admin and default_admin not in seeds:
        seeds.append(default_admin)
    roles: dict[str, Any] = {}
    now = datetime.utcnow().isoformat() + "Z"
    for user_id in seeds:
        roles[user_id] = {
            "roles": [ADMIN_ROLE],
            "granted_by": "env",
            "granted_at": now,
        }
    payload: dict[str, Any] = {"roles": roles}
    storage.write_json(_KEY, payload)
    return payload


def _save(roles: dict[str, Any]) -> None:
    storage.write_json(_KEY, {"roles": roles})


def get_roles(user_id: str) -> list[str]:
    """Return the list of roles for a user (empty list if none)."""
    user_id = _normalize(user_id)
    entry = _load().get("roles", {}).get(user_id)
    if not entry:
        return []
    return list(entry.get("roles", []))


def has_role(user_id: str, role: str) -> bool:
    return role in get_roles(user_id)


def is_admin(user_id: str) -> bool:
    default_admin = _normalize(_DEFAULT_ADMIN)
    if default_admin and _normalize(user_id) == default_admin:
        return True
    return has_role(user_id, ADMIN_ROLE)


def grant_role(
    user_id: str,
    role: str,
    granted_by: str = "admin",
    display_name: str = "",
) -> dict[str, Any]:
    """Grant a role to a user. Returns the updated entry."""
    user_id = _normalize(user_id)
    all_roles = _load().get("roles", {})
    entry = all_roles.get(user_id, {"roles": []})
    if role not in entry["roles"]:
        entry["roles"].append(role)
        entry["granted_by"] = granted_by
        entry["granted_at"] = datetime.utcnow().isoformat() + "Z"
    if display_name:
        entry["display_name"] = display_name
    all_roles[user_id] = entry
    _save(all_roles)
    logger.info("Role granted", user_id=user_id, role=role, granted_by=granted_by)
    return {"user_id": user_id, **entry}


def revoke_role(user_id: str, role: str) -> bool:
    """Revoke a role from a user. Returns True if the role was removed."""
    user_id = _normalize(user_id)
    all_roles = _load().get("roles", {})
    entry = all_roles.get(user_id)
    if not entry or role not in entry.get("roles", []):
        return False
    entry["roles"].remove(role)
    if not entry["roles"]:
        del all_roles[user_id]
    else:
        all_roles[user_id] = entry
    _save(all_roles)
    logger.info("Role revoked", user_id=user_id, role=role)
    return True


def list_admins() -> list[dict[str, Any]]:
    """Return a list of all users with the admin role."""
    all_roles = _load().get("roles", {})
    admins = []
    seen: set[str] = set()
    for user_id, entry in sorted(all_roles.items()):
        if ADMIN_ROLE in entry.get("roles", []):
            admins.append({
                "user_id": user_id,
                "display_name": entry.get("display_name", ""),
                "granted_by": entry.get("granted_by", "unknown"),
                "granted_at": entry.get("granted_at"),
            })
            seen.add(user_id)
    # Ensure the default admin always appears in the list
    default = _normalize(_DEFAULT_ADMIN)
    if default and default not in seen:
        admins.insert(0, {
            "user_id": default,
            "display_name": "",
            "granted_by": "default",
            "granted_at": None,
        })
    return admins
