"""Verikan's link to the Fair Store.

The Fair Store is a CKAN (2.11 + ckanext-hierarchy) that mirrors every
registered portal's catalog, with its organization hierarchy and the qsv
profiles onboarding produced (``scripts/populate_fairstore.py``). This module
is everything Verikan knows about it:

**Connection settings.**  The Fair Store's URL and a sysadmin API token,
seeded from ``FAIRSTORE_URL`` / ``FAIRSTORE_API_KEY`` and overridable from the
admin panel (``fairstore_settings.json`` through the storage backend — the same
pattern as the GitHub settings). The token is never returned to a client.

**The token is bound to the origin it was saved for** (scheme, host and port).
Changing the URL to a different origin without supplying a new token drops the
saved token, and the environment token is used only while the URL is on the
environment URL's origin. A URL edit can therefore never send the sysadmin
token to another server, or another service on the same host. A URL that is
already a registered source portal is refused (:func:`conflicting_portal`).

**Chat source.**  With ``chat_source`` on, the Fair Store is registered in the
portal registry under :data:`PORTAL_ID`, so users can pick it in chat and the
agent can search every mirrored portal at once.

**Status and site settings.**  Live reads through the CKAN action API, and the
CKAN options a sysadmin can change at runtime (site title, about text, logo,
custom CSS) — the Fair Store's own ``/ckan-admin/config`` page.

Mirror runs are started through :mod:`data_concierge.gateway.onboarding_jobs`,
which already runs one supervised job at a time.
"""

from __future__ import annotations

import asyncio
import ipaddress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx

from data_concierge.core.config import settings as app_settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.connectors.ckan import CKANActionError, _action_result
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_KEY = "fairstore_settings.json"

#: Registry ID of the Fair Store when it is offered as a chat source.
PORTAL_ID = "fairstore"

DEFAULT_PORTAL_NAME = "Verikan Fair Store"
DEFAULT_PORTAL_DESCRIPTION = (
    "One catalog that mirrors every registered open data portal (WPRDC, "
    "data.pa.gov, datHere) with each portal's organization hierarchy. Profiled "
    "datasets carry qsv data dictionaries, summary statistics, frequency tables "
    "and AI-written descriptions. Row data stays at each source portal."
)
DEFAULT_QUALITY_SCORE = 0.85

_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_TEXT = 20_000

# A token may travel over plain http only to this machine.
_LOCAL_HOSTS = frozenset({"localhost", "host.docker.internal"})

#: The CKAN options a sysadmin can change at runtime (``config_option_update``),
#: minus ``ckan.site_url`` (changing it breaks every link CKAN generates),
#: ``ckan.theme`` (2.11 ships one theme) and the logo-upload pseudo-options.
SITE_OPTIONS: dict[str, dict[str, str]] = {
    "ckan.site_title": {"label": "Site title", "kind": "text"},
    "ckan.site_description": {"label": "Site description", "kind": "text"},
    "ckan.site_logo": {"label": "Logo URL", "kind": "text"},
    "ckan.site_intro_text": {"label": "Home page intro (Markdown)", "kind": "textarea"},
    "ckan.site_about": {"label": "About page (Markdown)", "kind": "textarea"},
    "ckan.site_custom_css": {"label": "Custom CSS", "kind": "code"},
}


class FairStoreError(RuntimeError):
    """The Fair Store could not be reached, or refused a call."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _origin(url: str) -> str:
    """``scheme://host:port`` — what a saved token is bound to (``""`` if unset)."""
    if not url:
        return ""
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{parsed.host.lower()}:{port}"


def conflicting_portal(url: str) -> str | None:
    """A registered source portal already at this origin, if any.

    The Fair Store must be its own CKAN: pointed at a source portal (say
    data.dathere.com), a mirror run would write every other portal into it.
    """
    from data_concierge.gateway import ckan_sites

    origin = _origin(url)
    if not origin:
        return None
    for site in ckan_sites.list_sites():
        if site.get("managed_by") == "fairstore":
            continue
        if _origin(str(site.get("url") or "")) == origin:
            return str(site.get("id"))
    return None


def normalize_url(raw: str) -> str:
    """Validate and normalise a Fair Store base URL (``""`` stays unset).

    Raises ``ValueError`` for anything that is not a plain http(s) origin plus
    optional path: credentials, a query or fragment, or plain http to a host
    other than this machine (the sysadmin token would travel in clear text).
    """
    url = (raw or "").strip().rstrip("/")
    if not url:
        return ""
    if any(ord(ch) <= 32 or ord(ch) >= 127 for ch in url):
        # urlparse silently drops tabs and newlines, and a non-ASCII host is a
        # different string from the one the token is bound to: use punycode.
        raise ValueError("The Fair Store URL may contain only printable ASCII (use punycode)")
    try:
        httpx.URL(url).port  # noqa: B018 - validates the port as httpx will
    except (httpx.InvalidURL, ValueError) as exc:
        raise ValueError(f"The Fair Store URL is not valid: {exc}") from exc
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("The Fair Store URL must start with https:// and name a host")
    if parsed.username or parsed.password:
        raise ValueError("Put the API token in the token field, not in the URL")
    if parsed.query or parsed.fragment:
        raise ValueError("The Fair Store URL must not carry a query or fragment")
    if parsed.scheme == "http" and not _is_local(parsed.hostname):
        raise ValueError(
            "Use https:// for a Fair Store on another host — the sysadmin token "
            "would otherwise be sent in clear text"
        )
    return url


def _is_local(host: str) -> bool:
    host = host.lower()
    if host in _LOCAL_HOSTS or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _env_url() -> str:
    try:
        return normalize_url(app_settings.fairstore_url)
    except ValueError:
        logger.warning("Ignoring an invalid FAIRSTORE_URL")
        return ""


def _env_token() -> str:
    return app_settings.fairstore_api_key.get_secret_value().strip()


class SettingsUnreadable(FairStoreError):
    """The saved settings exist but could not be read (not the same as none)."""


def _read_saved(*, strict: bool = False) -> dict[str, Any]:
    """The saved settings; ``{}`` when there are none.

    A read failure degrades to ``{}`` for display, but ``strict`` callers — a
    save (which would otherwise overwrite the token) and the chat-source sync
    (which would otherwise delete the portal entry) — get
    :class:`SettingsUnreadable` instead.
    """
    try:
        saved = storage.read_json(_KEY)
    except Exception as exc:  # noqa: BLE001 - degrade to defaults
        if strict:
            raise SettingsUnreadable(f"Fair Store settings could not be read: {exc}") from exc
        logger.warning("Failed to read Fair Store settings", error=str(exc))
        return {}
    return saved if isinstance(saved, dict) else {}


def load_settings(*, strict: bool = False) -> dict[str, Any]:
    """The effective settings, token included — never hand this to a client.

    ``token_source`` is ``"admin"``, ``"environment"`` or ``""``: the saved
    token is used only for the origin (scheme, host and port) it was saved for,
    and the environment's only for the environment URL's origin.
    """
    saved = _read_saved(strict=strict)
    url = saved.get("url") if isinstance(saved.get("url"), str) else None
    url = url if url is not None else _env_url()

    token, source = "", ""
    origin = _origin(url)
    saved_token = str(saved.get("token") or "")
    if saved_token and origin and saved.get("token_origin") == origin:
        token, source = saved_token, "admin"
    elif _env_token() and origin and origin == _origin(_env_url()):
        token, source = _env_token(), "environment"

    mirror_sites = saved.get("mirror_sites")
    return {
        "url": url,
        "url_source": "admin" if "url" in saved else ("environment" if url else ""),
        "token": token,
        "token_source": source,
        "chat_source": bool(saved.get("chat_source", False)),
        "portal_name": str(saved.get("portal_name") or DEFAULT_PORTAL_NAME),
        "portal_description": str(saved.get("portal_description") or DEFAULT_PORTAL_DESCRIPTION),
        "quality_score": float(saved.get("quality_score", DEFAULT_QUALITY_SCORE)),
        "mirror_sites": [str(s) for s in mirror_sites] if isinstance(mirror_sites, list) else [],
        "enrich": bool(saved.get("enrich", True)),
        "updated_at": saved.get("updated_at"),
        "updated_by": saved.get("updated_by"),
    }


def public_settings(current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Settings safe to send to the admin UI: the token becomes set/unset flags.

    ``mirror_sites`` lists the chosen portals still registered;
    ``mirror_sites_missing`` the ones since removed.
    """
    from data_concierge.gateway import ckan_sites

    current = current if current is not None else load_settings()
    token = current.get("token") or ""
    known = set(ckan_sites.list_site_ids())
    chosen = current.get("mirror_sites") or []
    return {
        **{k: v for k, v in current.items() if k != "token"},
        "mirror_sites": [s for s in chosen if s in known],
        "mirror_sites_missing": [s for s in chosen if s not in known],
        "conflicting_portal": conflicting_portal(current.get("url") or ""),
        "token_set": bool(token),
        "token_masked": f"••••{token[-4:]}" if len(token) > 12 else ("••••" if token else ""),
        "configured": bool(current.get("url")),
        "site_options": SITE_OPTIONS,
    }


def save_settings(updates: dict[str, Any], *, updated_by: str = "admin") -> dict[str, Any]:
    """Validate and persist an admin's changes; returns the public settings.

    ``token``: a non-empty string sets it, blank keeps it (the UI never sees
    the current value), and ``clear_token`` removes it. A URL change to a
    different origin (scheme, host or port) without a new token drops the saved
    token — see the module docstring. Raises ``ValueError`` on invalid input,
    including a URL that is already a registered source portal.
    """
    saved = _read_saved(strict=True)
    before = load_settings(strict=True)

    if "url" in updates and updates["url"] is not None:
        saved["url"] = normalize_url(str(updates["url"]))
        clash = conflicting_portal(saved["url"])
        if clash:
            raise ValueError(
                f"That URL is registered as the portal '{clash}' on the Data Portals page. "
                "The Fair Store must be a separate CKAN, or a mirror run would write into "
                f"that portal. (If '{clash}' is this Fair Store, delete that entry and use "
                "'Offer the Fair Store as a data source in chat' instead.)"
            )
    origin = _origin(saved.get("url") if "url" in saved else _env_url())

    new_token = str(updates.get("token") or "").strip()
    if new_token:
        if any(ch.isspace() for ch in new_token) or len(new_token) > 4096:
            raise ValueError("That does not look like a CKAN API token")
        if not origin:
            raise ValueError("Set the Fair Store URL before its API token")
        saved["token"] = new_token
        saved["token_origin"] = origin
    elif updates.get("clear_token"):
        saved.pop("token", None)
        saved.pop("token_origin", None)
    elif saved.get("token") and saved.get("token_origin") != origin:
        saved.pop("token", None)
        saved.pop("token_origin", None)
        logger.info("Fair Store token dropped: the URL moved to another origin", origin=origin)

    if updates.get("chat_source") is not None:
        saved["chat_source"] = bool(updates["chat_source"])
    for key in ("portal_name", "portal_description"):
        if updates.get(key) is not None:
            text = str(updates[key]).strip()[:2000]
            if key == "portal_name" and not text:
                raise ValueError("The chat source needs a name")
            saved[key] = text
    if updates.get("quality_score") is not None:
        score = float(updates["quality_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("Quality score must be between 0 and 1")
        saved["quality_score"] = score
    if updates.get("mirror_sites") is not None:
        from data_concierge.gateway import ckan_sites

        known = set(ckan_sites.list_site_ids())
        sites = [str(s).strip() for s in updates["mirror_sites"] if str(s).strip()]
        unknown = [s for s in sites if s not in known]
        if unknown:
            raise ValueError(f"Unknown portal(s): {', '.join(sorted(unknown))}")
        saved["mirror_sites"] = sorted(set(sites) - {PORTAL_ID})
    if updates.get("enrich") is not None:
        saved["enrich"] = bool(updates["enrich"])

    saved["updated_at"] = _now()
    saved["updated_by"] = updated_by
    storage.write_json(_KEY, saved)

    current = load_settings()
    sync_chat_source(current)
    logger.info(
        "Fair Store settings saved",
        updated_by=updated_by,
        url_changed=before["url"] != current["url"],
        token_changed=before["token"] != current["token"],
        chat_source=current["chat_source"],
    )
    return public_settings(current)


def sync_chat_source(current: dict[str, Any] | None = None) -> None:
    """Make the portal registry's ``fairstore`` entry match the settings.

    Only an entry this module created (``managed_by == "fairstore"``) is ever
    changed or removed, so a portal an admin registered by hand under the same
    ID is left alone.
    """
    from data_concierge.gateway import ckan_sites

    if current is None:
        try:
            current = load_settings(strict=True)
        except SettingsUnreadable as exc:
            # Unreadable is not "turned off": leave the registry as it is.
            logger.warning("Skipping the Fair Store chat-source sync", error=str(exc))
            return
    # Two instances syncing at once can both add the entry; add_site renames
    # the second to fairstore-2, which the Data Portals page cannot delete.
    for site in ckan_sites.list_sites():
        if site.get("managed_by") == "fairstore" and site.get("id") != PORTAL_ID:
            ckan_sites.remove_site(str(site["id"]))
            logger.info("Removed a duplicate Fair Store chat-source entry", site_id=site["id"])
    existing = ckan_sites.get_site(PORTAL_ID)
    managed = bool(existing and existing.get("managed_by") == "fairstore")
    wanted = bool(current.get("chat_source") and current.get("url"))
    if wanted and conflicting_portal(current["url"]):
        # An environment URL is not validated on save; never shadow a portal.
        logger.warning("The Fair Store URL is a registered source portal; not offering it")
        wanted = False

    if existing and not managed:
        if wanted:
            logger.warning(
                "A hand-registered portal already uses the Fair Store's ID; leaving it",
                site_id=PORTAL_ID,
            )
        return
    if not wanted:
        if managed:
            ckan_sites.remove_site(PORTAL_ID)
        return

    fields = {
        "url": current["url"],
        "name": current["portal_name"],
        "description": current["portal_description"],
        "quality_score": current["quality_score"],
        "portal_type": ckan_sites.PORTAL_TYPE_CKAN,
    }
    if managed:
        if any(existing.get(key) != value for key, value in fields.items()):  # type: ignore[union-attr]
            ckan_sites.update_site(PORTAL_ID, fields)
    else:
        ckan_sites.add_site(
            site_id=PORTAL_ID,
            keywords=["fair store", "all portals", "catalog", "data dictionary", "qsv"],
            added_by="fairstore",
            managed_by="fairstore",
            **fields,
        )


# -----------------------------------------------------------------------------
# CKAN calls
# -----------------------------------------------------------------------------


async def _call(
    current: dict[str, Any],
    action: str,
    params: dict[str, Any] | None = None,
    *,
    auth: bool = False,
) -> Any:
    """One CKAN action against the Fair Store (the token only when ``auth``)."""
    url = current.get("url") or ""
    if not url:
        raise FairStoreError("The Fair Store URL is not configured")
    headers = {"Content-Type": "application/json"}
    if auth:
        if not current.get("token"):
            raise FairStoreError("No Fair Store API token is configured")
        headers["Authorization"] = current["token"]
    try:
        async with httpx.AsyncClient(base_url=url, timeout=_TIMEOUT, headers=headers) as client:
            response = await client.post(f"/api/3/action/{action}", json=params or {})
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise FairStoreError(f"{action}: {type(exc).__name__}: {exc}") from exc
    try:
        return _action_result(action, response)
    except CKANActionError as exc:
        raise FairStoreError(str(exc)) from exc


async def _count(current: dict[str, Any], fq: str | None = None) -> int | None:
    params: dict[str, Any] = {"rows": 0, "include_private": True}
    if fq:
        params["fq"] = fq
    try:
        result = await _call(current, "package_search", params, auth=bool(current.get("token")))
    except FairStoreError:
        return None
    return int(result.get("count", 0)) if isinstance(result, dict) else None


async def status(current: dict[str, Any] | None = None) -> dict[str, Any]:
    """A live health and content summary for the admin panel.

    Never raises: an unreachable Fair Store, or a token CKAN refuses, is
    reported in the result so the panel can say what is wrong.
    """
    from data_concierge.gateway import ckan_sites

    current = current if current is not None else load_settings()
    report: dict[str, Any] = {
        "configured": bool(current.get("url")),
        "url": current.get("url") or "",
        "reachable": False,
        "checked_at": _now(),
    }
    if not report["configured"]:
        return report

    try:
        info = await _call(current, "status_show")
    except FairStoreError as exc:
        report["error"] = str(exc)
        return report
    report["reachable"] = True
    report["ckan_version"] = info.get("ckan_version")
    report["site_title"] = info.get("site_title")
    report["extensions"] = sorted(info.get("extensions") or [])

    # A sysadmin-only action tells a valid sysadmin token from any other.
    if current.get("token"):
        try:
            await _call(current, "config_option_list", auth=True)
            report["token"] = "sysadmin"
        except FairStoreError as exc:
            report["token"] = "rejected"
            report["token_error"] = str(exc)
    else:
        report["token"] = "missing"

    sources = [s for s in ckan_sites.list_sites() if s.get("id") and s.get("id") != PORTAL_ID]
    counts = await asyncio.gather(
        _count(current),
        *(_count(current, f'mirror_source_portal:"{s["id"]}"') for s in sources),
    )
    report["datasets"] = counts[0]
    report["by_portal"] = [
        {"site_id": s["id"], "name": s.get("name") or s["id"], "datasets": n}
        for s, n in zip(sources, counts[1:], strict=True)
    ]
    for action, key in (("organization_list", "organizations"), ("group_list", "groups")):
        try:
            report[key] = len(await _call(current, action, {"all_fields": False}))
        except FairStoreError:
            report[key] = None
    return report


def _refuse_conflict(current: dict[str, Any]) -> None:
    clash = conflicting_portal(current.get("url") or "")
    if clash:
        raise FairStoreError(
            f"The Fair Store URL is the registered portal '{clash}'; its site settings "
            "are not edited from here."
        )


async def get_site_config(current: dict[str, Any] | None = None) -> dict[str, Any]:
    """The Fair Store's runtime-editable site options (sysadmin token needed)."""
    current = current if current is not None else load_settings()
    _refuse_conflict(current)
    editable = set(await _call(current, "config_option_list", auth=True) or [])
    keys = [key for key in SITE_OPTIONS if key in editable]
    values = await asyncio.gather(
        *(_call(current, "config_option_show", {"key": key}, auth=True) for key in keys)
    )
    return {
        "options": [
            {**SITE_OPTIONS[key], "key": key, "value": "" if value is None else str(value)}
            for key, value in zip(keys, values, strict=True)
        ]
    }


async def update_site_config(
    values: dict[str, Any], current: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Change the Fair Store's site options; only :data:`SITE_OPTIONS` pass."""
    current = current if current is not None else load_settings()
    _refuse_conflict(current)
    unknown = sorted(set(values) - set(SITE_OPTIONS))
    if unknown:
        raise ValueError(f"Not an editable Fair Store option: {', '.join(unknown)}")
    payload = {}
    for key, value in values.items():
        text = "" if value is None else str(value)
        if len(text) > _MAX_TEXT:
            raise ValueError(f"{SITE_OPTIONS[key]['label']} is longer than {_MAX_TEXT} characters")
        payload[key] = text
    if not payload:
        raise ValueError("Nothing to update")
    await _call(current, "config_option_update", payload, auth=True)
    logger.info("Fair Store site options updated", keys=sorted(payload))
    return await get_site_config(current)


def public_info() -> dict[str, Any]:
    """What any visitor may know: whether there is a Fair Store, and its address."""
    current = load_settings()
    return {
        "configured": bool(current.get("url")),
        "url": current.get("url") or "",
        "name": current.get("portal_name") or DEFAULT_PORTAL_NAME,
        "chat_source": bool(current.get("chat_source") and current.get("url")),
        "portal_id": PORTAL_ID,
    }


__all__ = [
    "DEFAULT_PORTAL_DESCRIPTION",
    "conflicting_portal",
    "DEFAULT_PORTAL_NAME",
    "PORTAL_ID",
    "SITE_OPTIONS",
    "FairStoreError",
    "get_site_config",
    "load_settings",
    "normalize_url",
    "public_info",
    "public_settings",
    "save_settings",
    "status",
    "sync_chat_source",
    "update_site_config",
]
