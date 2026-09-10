"""Admin-managed registry of open data portals (CKAN and DCAT).

Each entry describes a portal (URL, display name, optional default
``organization`` filter, description, and quality score) that the LLM-driven
analysis agent can query.  The registry is seeded from the defaults that used
to live as a hardcoded dict in ``agents.llm_agent.PORTAL_CONFIGS`` so the agent
keeps working out-of-the-box for WPRDC (Pittsburgh) and the datHere CKAN
portal; admins can add, update, or remove additional portals from the admin
panel and changes take effect on the next query without a restart.

Two ``portal_type`` values are supported:

``ckan``
    A CKAN portal with a live action API — ``package_search``,
    ``datastore_search_sql``, and friends.  Queries run server-side.

``dcat``
    A portal publishing a DCAT catalog (the ``/data.json`` document required by
    Project Open Data / DCAT-US 1.1, served by Socrata, ArcGIS Hub, data.gov,
    and CKAN's DCAT extension).  There is no query API: the catalog is fetched
    once and searched locally, and row data comes from streaming a
    distribution's download URL.  ``catalog_url`` may be set explicitly;
    otherwise the standard paths are probed.  See
    :mod:`data_concierge.data_layer.connectors.dcat`.

The module keeps its ``ckan_sites`` name — and its ``ckan_sites.json`` storage
key — because both are load-bearing for existing deployments' persisted state
and for the admin API path.

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

PORTAL_TYPE_CKAN = "ckan"
PORTAL_TYPE_DCAT = "dcat"
PORTAL_TYPES = (PORTAL_TYPE_CKAN, PORTAL_TYPE_DCAT)


def normalize_portal_type(raw: Any) -> str:
    """Coerce a portal type to a supported value, defaulting to CKAN.

    Entries persisted before DCAT support existed carry no ``portal_type``,
    so an absent or unrecognised value has to mean ``ckan`` — otherwise every
    pre-existing portal would silently change behaviour on upgrade.
    """
    value = str(raw or "").strip().lower()
    return value if value in PORTAL_TYPES else PORTAL_TYPE_CKAN

# Default portals — seeded on first load.  Keeping these here (rather than
# importing from ``agents.llm_agent``) avoids a circular import; the agent now
# reads its PORTAL_CONFIGS via this module instead of the other way around.
DEFAULT_SITES: list[dict[str, Any]] = [
    {
        "id": "wprdc",
        "portal_type": PORTAL_TYPE_CKAN,
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
        "portal_type": PORTAL_TYPE_CKAN,
        "url": settings.ckan_url,
        "name": "datHere CKAN Portal",
        "organization": None,
        "description": "General CKAN open data portal with diverse datasets.",
        "quality_score": 0.85,
        "keywords": ["open data", "datasets", "csv", "ckan"],
    },
    {
        "id": "data-pa-gov",
        "portal_type": PORTAL_TYPE_DCAT,
        "url": "https://data.pa.gov",
        "catalog_url": "https://data.pa.gov/data.json",
        "name": "Pennsylvania Open Data Portal (data.pa.gov)",
        "organization": None,
        "description": (
            "Commonwealth of Pennsylvania statewide open data portal. "
            "Publishes a DCAT-US catalog of 470+ state agency datasets covering "
            "health and human services (drug overdose deaths, opioid "
            "hospitalizations, COVID-19), education, labor and industry, "
            "transportation (PennDOT), agriculture, environment, corrections, "
            "and state government spending — most with county-level breakdowns."
        ),
        "quality_score": 0.85,
        "keywords": [
            "pennsylvania", "pa", "commonwealth", "statewide", "state agency",
            "opioid", "overdose", "health", "penndot", "transportation",
            "education", "labor", "corrections", "agriculture",
        ],
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


def _seed_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        **entry,
        "url": _normalize_url(entry["url"]),
        "portal_type": normalize_portal_type(entry.get("portal_type")),
        "added_by": "default",
        "added_at": _now(),
    }


def _load_raw() -> dict[str, Any]:
    """Load the stored sites list, seeding defaults that have never been seeded.

    On first run every entry in :data:`DEFAULT_SITES` is written out.  On later
    runs a default is added only if its ID has never been seeded before —
    tracked in ``seeded_defaults`` rather than inferred from the current site
    list, so that a portal an admin deliberately **deleted** is not silently
    resurrected on the next deploy.  Deployments that predate that key are
    backfilled from the IDs they already have, which has the same effect.
    """
    data = storage.read_json(_KEY)

    if data and isinstance(data.get("sites"), list):
        sites = data["sites"]
        existing_ids = {str(s.get("id", "")).lower() for s in sites}
        seeded_ids = data.get("seeded_defaults")
        if not isinstance(seeded_ids, list):
            # Pre-existing install: treat whatever is present as already seeded.
            seeded_ids = sorted(existing_ids)
        seeded_lower = {str(i).lower() for i in seeded_ids}

        added: list[str] = []
        for entry in DEFAULT_SITES:
            entry_id = str(entry.get("id", "")).lower()
            if not entry_id or entry_id in seeded_lower or entry_id in existing_ids:
                continue
            sites.append(_seed_entry(entry))
            seeded_lower.add(entry_id)
            added.append(entry_id)

        if added or not isinstance(data.get("seeded_defaults"), list):
            payload = {"sites": sites, "seeded_defaults": sorted(seeded_lower)}
            storage.write_json(_KEY, payload)
            if added:
                logger.info("Seeded new default portals", site_ids=added)
            return payload
        return data

    seeded = [_seed_entry(entry) for entry in DEFAULT_SITES]
    payload = {
        "sites": seeded,
        "seeded_defaults": sorted(str(e.get("id", "")).lower() for e in DEFAULT_SITES),
    }
    storage.write_json(_KEY, payload)
    return payload


def _save(sites: list[dict[str, Any]]) -> None:
    """Persist the site list, preserving the seeded-defaults ledger.

    Dropping ``seeded_defaults`` here would make the next load re-add every
    default portal an admin had removed.
    """
    existing = storage.read_json(_KEY) or {}
    payload: dict[str, Any] = {"sites": sites}
    if isinstance(existing.get("seeded_defaults"), list):
        payload["seeded_defaults"] = existing["seeded_defaults"]
    storage.write_json(_KEY, payload)


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
    portal_type: str = PORTAL_TYPE_CKAN,
    catalog_url: str | None = None,
) -> dict[str, Any]:
    """Add a new portal.  Auto-generates an ID from the name if not given.

    ``portal_type`` selects the access mechanism — ``ckan`` for a live action
    API, ``dcat`` for a catalog document.  ``catalog_url`` is DCAT-only and
    optional: when omitted the standard catalog paths are probed under ``url``.

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
        "portal_type": normalize_portal_type(portal_type),
        "url": url,
        "catalog_url": _normalize_url(catalog_url) if catalog_url else None,
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
    logger.info(
        "Portal added",
        site_id=chosen_id,
        url=url,
        portal_type=entry["portal_type"],
        added_by=added_by,
    )
    return entry


def update_site(site_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    """Update a site in place.  Returns the updated entry or ``None`` if not found."""
    target = site_id.strip().lower()
    sites = list_sites()
    for i, s in enumerate(sites):
        if str(s.get("id", "")).lower() != target:
            continue
        merged = dict(s)
        for key in ("url", "catalog_url", "name", "organization", "description",
                    "quality_score", "keywords", "portal_type"):
            if key in updates and updates[key] is not None:
                if key in ("url", "catalog_url"):
                    merged[key] = _normalize_url(str(updates[key]))
                elif key == "portal_type":
                    merged[key] = normalize_portal_type(updates[key])
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
        "portal_type": normalize_portal_type(site.get("portal_type")),
        "catalog_url": site.get("catalog_url") or None,
    }


def get_portal_configs() -> dict[str, dict[str, Any]]:
    """Return ``{site_id: config}`` for every registered site.

    Used by the LLM agent in place of the old hardcoded PORTAL_CONFIGS dict.
    """
    return {s["id"]: get_portal_config(s["id"]) for s in list_sites() if s.get("id")}  # type: ignore[misc]


def list_site_ids() -> list[str]:
    """Return just the IDs of registered sites (used for routing decisions)."""
    return [str(s.get("id", "")) for s in list_sites() if s.get("id")]


def get_portal_type(site_id: str) -> str:
    """Return the portal type for ``site_id`` (``ckan`` when unregistered)."""
    site = get_site(site_id)
    return normalize_portal_type(site.get("portal_type")) if site else PORTAL_TYPE_CKAN


def find_site_by_url(url: str) -> dict[str, Any] | None:
    """Look up a registered site by portal URL.

    The agent's tool dispatch carries a resolved portal *URL* rather than an
    ID (a ``portal_id`` override is resolved to a URL before the tool runs),
    so this is how a tool call recovers which portal — and therefore which
    access mechanism — it is talking to.
    """
    target = _normalize_url(url or "").lower()
    if not target:
        return None
    for s in list_sites():
        if _normalize_url(str(s.get("url", ""))).lower() == target:
            return s
    return None


def portal_type_for_url(url: str) -> str:
    """Return the portal type registered for ``url``, defaulting to CKAN."""
    site = find_site_by_url(url)
    return normalize_portal_type(site.get("portal_type")) if site else PORTAL_TYPE_CKAN


def list_sites_by_type(portal_type: str) -> list[dict[str, Any]]:
    """Return every registered site of one portal type."""
    wanted = normalize_portal_type(portal_type)
    return [s for s in list_sites() if normalize_portal_type(s.get("portal_type")) == wanted]
