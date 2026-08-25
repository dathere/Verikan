"""Admin-managed registry of CKAN open data portals.

Each entry describes a CKAN portal (URL, display name, optional default
``organization`` filter, description, and quality score) that the LLM-driven
analysis agent can query.  The registry is seeded from the defaults that used
to live as a hardcoded dict in ``agents.llm_agent.PORTAL_CONFIGS`` so the agent
keeps working out-of-the-box for WPRDC (Pittsburgh) and the datHere CKAN
portal; admins can add, update, or remove additional portals from the admin
panel and changes take effect on the next query without a restart.

The registry is persisted through the unified storage backend (``ckan_sites.json``)
so it survives restarts on Cloud Run.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_KEY = "ckan_sites.json"

# Default portals — seeded on first load.  Keeping these here (rather than
# importing from ``agents.llm_agent``) avoids a circular import; the agent now
# reads its PORTAL_CONFIGS via this module instead of the other way around.
DEFAULT_SITES: list[dict[str, Any]] = [
    {
        "id": "wprdc",
        "url": "https://data.wprdc.org",
        "name": "Western PA Regional Data Center (WPRDC)",
        "organization": "city-of-pittsburgh",
        "description": (
            "Open data portal for Pittsburgh and Western Pennsylvania. "
            "Contains 126+ datasets from the City of Pittsburgh including "
            "311 requests, crime data, building permits, property assessments, "
            "community center attendance, traffic counts, and more."
        ),
        "quality_score": 0.85,
        "keywords": [
            "pittsburgh", "allegheny", "western pennsylvania", "wprdc",
            "311", "crime", "permits", "property", "community center",
        ],
    },
    {
        "id": "ckan",
        "url": settings.ckan_url,
        "name": "datHere CKAN Portal",
        "organization": None,
        "description": "General CKAN open data portal with diverse datasets.",
        "quality_score": 0.85,
        "keywords": ["open data", "datasets", "csv", "ckan"],
    },
]


def _slugify(raw: str) -> str:
    """Turn a name/URL into a stable lowercase ID (letters, digits, dashes)."""
    slug = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return slug or "ckan-site"


def _normalize_url(url: str) -> str:
    """Trim whitespace and trailing slash from a portal URL."""
    return url.strip().rstrip("/")


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_raw() -> dict[str, Any]:
    """Load the stored sites list.  Seed from DEFAULT_SITES on first run."""
    data = storage.read_json(_KEY)
    if data and isinstance(data.get("sites"), list):
        return data

    seeded: list[dict[str, Any]] = []
    for entry in DEFAULT_SITES:
        seeded.append({
            **entry,
            "url": _normalize_url(entry["url"]),
            "added_by": "default",
            "added_at": _now(),
        })
    payload = {"sites": seeded}
    storage.write_json(_KEY, payload)
    return payload


def _save(sites: list[dict[str, Any]]) -> None:
    storage.write_json(_KEY, {"sites": sites})


def list_sites() -> list[dict[str, Any]]:
    """Return all registered CKAN sites (a copy; safe to mutate)."""
    return [dict(s) for s in _load_raw().get("sites", [])]


def get_site(site_id: str) -> dict[str, Any] | None:
    """Look up a site by its ID.  Returns ``None`` if missing."""
    if not site_id:
        return None
    target = site_id.strip().lower()
    for s in list_sites():
        if str(s.get("id", "")).lower() == target:
            return s
    return None


def add_site(
    *,
    url: str,
    name: str,
    site_id: str | None = None,
    organization: str | None = None,
    description: str = "",
    quality_score: float = 0.85,
    keywords: list[str] | None = None,
    added_by: str = "admin",
) -> dict[str, Any]:
    """Add a new CKAN site.  Auto-generates an ID from the name if not given.

    Raises ``ValueError`` if ``url`` or ``name`` is empty, or if a site with
    the chosen ID already exists.
    """
    url = _normalize_url(url)
    name = name.strip()
    if not url:
        raise ValueError("URL is required")
    if not name:
        raise ValueError("Name is required")

    sites = list_sites()
    base_id = _slugify(site_id) if site_id else _slugify(name)
    chosen_id = base_id
    existing_ids = {str(s.get("id", "")).lower() for s in sites}
    # Ensure uniqueness by appending -2, -3, ... if needed
    suffix = 2
    while chosen_id in existing_ids:
        chosen_id = f"{base_id}-{suffix}"
        suffix += 1

    entry: dict[str, Any] = {
        "id": chosen_id,
        "url": url,
        "name": name,
        "organization": (organization or None),
        "description": description.strip(),
        "quality_score": float(quality_score),
        "keywords": list(keywords or []),
        "added_by": added_by,
        "added_at": _now(),
    }
    sites.append(entry)
    _save(sites)
    logger.info("CKAN site added", site_id=chosen_id, url=url, added_by=added_by)
    return entry


def update_site(site_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Update a site in place.  Returns the updated entry or ``None`` if not found."""
    target = site_id.strip().lower()
    sites = list_sites()
    for i, s in enumerate(sites):
        if str(s.get("id", "")).lower() != target:
            continue
        merged = dict(s)
        for key in ("url", "name", "organization", "description",
                    "quality_score", "keywords"):
            if key in updates and updates[key] is not None:
                if key == "url":
                    merged[key] = _normalize_url(str(updates[key]))
                elif key == "quality_score":
                    merged[key] = float(updates[key])
                else:
                    merged[key] = updates[key]
        merged["updated_at"] = _now()
        sites[i] = merged
        _save(sites)
        logger.info("CKAN site updated", site_id=target)
        return merged
    return None


def remove_site(site_id: str) -> bool:
    """Remove a site.  Returns ``True`` if removed, ``False`` if not found."""
    target = site_id.strip().lower()
    sites = list_sites()
    new_sites = [s for s in sites if str(s.get("id", "")).lower() != target]
    if len(new_sites) == len(sites):
        return False
    _save(new_sites)
    logger.info("CKAN site removed", site_id=target)
    return True


def get_portal_config(site_id: str) -> dict[str, Any] | None:
    """Return the subset of fields the LLM analysis agent needs.

    Shape matches the old ``PORTAL_CONFIGS`` dict entries: ``url``, ``name``,
    ``organization``, ``description``, ``quality_score``.  Returns ``None`` if
    the site isn't registered.
    """
    site = get_site(site_id)
    if not site:
        return None
    return {
        "url": site.get("url", ""),
        "name": site.get("name", ""),
        "organization": site.get("organization") or None,
        "description": site.get("description", ""),
        "quality_score": float(site.get("quality_score", 0.85)),
    }


def get_portal_configs() -> dict[str, dict[str, Any]]:
    """Return ``{site_id: config}`` for every registered site.

    Used by the LLM agent in place of the old hardcoded PORTAL_CONFIGS dict.
    """
    return {s["id"]: get_portal_config(s["id"]) for s in list_sites() if s.get("id")}  # type: ignore[misc]


def list_site_ids() -> list[str]:
    """Return just the IDs of registered sites (used for routing decisions)."""
    return [str(s.get("id", "")) for s in list_sites() if s.get("id")]
