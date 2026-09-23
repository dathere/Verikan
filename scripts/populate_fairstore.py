#!/usr/bin/env python
"""Mirror source catalogs into the Fair Store.

Both portal types in the registry are mirrored:

``ckan``
    Source organization, group, dataset, and resource names and UUIDs are kept.
``dcat``
    A DCAT-US catalog (``/data.json`` from Socrata, DKAN, ArcGIS Hub, ...) has
    no organization records or UUIDs, so they are derived deterministically:
    publishers become organizations, themes become groups, distributions become
    resources, and every UUID is a UUIDv5 of the source identifier (or the
    source's own UUID, when it publishes one). A Socrata catalog is enriched
    from the Socrata Discovery API, whose owning agency and column list the
    catalog document omits.

A top-level organization represents each source portal, and the portal's
publishers are attached beneath it through ckanext-hierarchy. Every field a
source publishes is kept: fields the Fair Store's default schema has no column
for are carried as extras instead of being silently dropped. Available qsv
profiling metadata from ``data/{ckan,dcat}_onboard/<site>/index.json`` is added
to the corresponding resource without replacing the source description.

A source record that its portal itself mirrored from another portal (it
carries ``mirror_source_portal`` naming that portal) is skipped: mirror the
origin portal directly so provenance names the real source.

The command is read-only unless ``--apply`` is supplied. Before any write it
checks the whole source snapshot for name and UUID collisions. Re-running is
idempotent: existing objects are patched by their preserved source UUIDs, and a
record the source renamed is renamed. Every read fails closed: a portal that
errors mid-read aborts that portal's run rather than being mirrored from a
partial snapshot, which would move the unseen records' datasets to the root.
Records the source has deleted are reported (``withdrawn_at_source``), never
deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from data_concierge.data_layer.connectors.ckan import CKANActionError, CKANClient  # noqa: E402
from data_concierge.data_layer.connectors.dcat import (  # noqa: E402
    DCATClient,
    _as_list,
    _text,
    parse_dataset,
    raw_dataset_nodes,
)
from data_concierge.data_layer.onboard_index import _scrub_secrets  # noqa: E402
from data_concierge.data_layer.qsv_profiling import _extract_qsv_tags  # noqa: E402
from data_concierge.gateway.ckan_sites import (  # noqa: E402
    PORTAL_TYPE_DCAT,
    get_site,
    list_sites,
    normalize_portal_type,
)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_PROVENANCE_KEYS = {
    "portal": "mirror_source_portal",
    "url": "mirror_source_url",
    "id": "mirror_source_id",
}

_ORGANIZATION_FIELDS = (
    "id",
    "name",
    "title",
    "description",
    "image_url",
    "type",
    "state",
    "approval_status",
)
_GROUP_FIELDS = ("id", "name", "title", "description", "image_url", "type", "state")
_PACKAGE_FIELDS = (
    "id",
    "name",
    "title",
    "author",
    "author_email",
    "maintainer",
    "maintainer_email",
    "license_id",
    "notes",
    "url",
    "version",
    "state",
    "type",
    "private",
    "plugin_data",
)
_CLEARABLE_PACKAGE_FIELDS = (
    "author",
    "author_email",
    "maintainer",
    "maintainer_email",
    "license_id",
    "url",
    "version",
)
# Package keys CKAN computes or the mirror sets itself. Anything else a source
# returns — ckanext-scheming fields such as a data steward or temporal
# coverage — is source metadata the Fair Store's default schema would drop, so
# it is carried as an extra.
_PACKAGE_MANAGED = frozenset(
    {
        *_PACKAGE_FIELDS,
        "owner_org",
        "organization",
        "groups",
        "tags",
        "tag_string",
        "extras",
        "resources",
        "license_title",
        "license_url",
        "isopen",
        "num_resources",
        "num_tags",
        "metadata_created",
        "metadata_modified",
        "creator_user_id",
        "relationships_as_object",
        "relationships_as_subject",
        "revision_id",
        "tracking_summary",
    }
)
# Resource keys describing the source CKAN's own bookkeeping rather than the
# resource. ``datastore_*`` keys are excluded separately: the Fair Store holds
# metadata only and must never claim that a DataStore table was copied.
_RESOURCE_MANAGED = frozenset(
    {"package_id", "position", "state", "metadata_modified", "revision_id", "tracking_summary"}
)
# url_type values for which CKAN rewrites ``url`` to the *serving* site's own
# download route. Mirrored as-is, the link would point at a file the Fair Store
# never received, so the source's absolute URL is kept as a plain link.
_LOCAL_URL_TYPES = frozenset({"upload", "datastore"})
# Longest term Solr will index (bytes, UTF-8). CKAN indexes every dataset extra
# as one such term; resource extras are not indexed, so they have no limit.
_SOLR_MAX_TERM_BYTES = 32766

_TAG_VALID_RE = re.compile(r"[\w \-.]{2,100}")
_TAG_INVALID_RE = re.compile(r"[^\w \-.]+")
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")

_SOCRATA_VIEW_RE = re.compile(r"/api/views/([a-z0-9]{4}-[a-z0-9]{4})/?$")
_SOCRATA_DISCOVERY_URL = "https://api.us.socrata.com/api/catalog/v1"
# Socrata domains name their owning-agency field themselves, so match the
# usual spellings; ``attribution`` is the standard fallback.
_SOCRATA_OWNER_KEY_RE = re.compile(r"business[-_ ]?owner|agency|department|publisher", re.I)

_MEDIA_TYPE_FORMATS = {
    "text/csv": "CSV",
    "application/csv": "CSV",
    "text/tab-separated-values": "TSV",
    "application/json": "JSON",
    "application/geo+json": "GeoJSON",
    "application/vnd.geo+json": "GeoJSON",
    "application/xml": "XML",
    "text/xml": "XML",
    "application/rdf+xml": "RDF",
    "application/vnd.google-earth.kml+xml": "KML",
    "application/vnd.google-earth.kmz": "KMZ",
    "application/vnd.ms-excel": "XLS",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
    "application/zip": "ZIP",
    "application/pdf": "PDF",
    "text/html": "HTML",
}

# DCAT publishes licenses as URLs; CKAN's license register uses short IDs.
# Keys are normalized by _license_key. Unknown URLs are kept only in extras.
_LICENSE_IDS = {
    "usa.gov/government-works": "other-pd",
    "usa.gov/publicdomain/label/1.0": "other-pd",
    "creativecommons.org/publicdomain/zero/1.0": "cc-zero",
    "creativecommons.org/publicdomain/mark/1.0": "other-pd",
    "creativecommons.org/licenses/by/4.0": "cc-by",
    "creativecommons.org/licenses/by/3.0": "cc-by",
    "creativecommons.org/licenses/by-sa/4.0": "cc-by-sa",
    "creativecommons.org/licenses/by-sa/3.0": "cc-by-sa",
    "creativecommons.org/licenses/by-nc/4.0": "cc-nc",
    "opendatacommons.org/licenses/pddl/1.0": "odc-pddl",
    "opendatacommons.org/licenses/by/1.0": "odc-by",
    "opendatacommons.org/licenses/odbl/1.0": "odc-odbl",
    "gnu.org/licenses/fdl-1.3": "gfdl",
}


class MirrorError(RuntimeError):
    """Raised when a collision or failed CKAN action makes a mirror unsafe."""


class DeletedInTarget(MirrorError):
    """The object exists in the Fair Store but an admin deleted it there."""


_ATTEMPTS = 3


async def _read(
    client: CKANClient,
    action: str,
    params: dict[str, Any],
    *,
    label: str,
    missing_ok: bool = False,
) -> Any:
    """A read that fails closed; ``None`` for a missing object when ``missing_ok``.

    ``CKANClient.action`` returns ``{}`` for any failure, which a mirror reads
    as "nothing there": one 502 while paging a portal's organizations would
    leave the unseen ones out of the snapshot, and ``--apply`` would then move
    their datasets to the portal root. Transient failures are retried; anything
    else, or a transient failure that persists, aborts the run.
    """
    for attempt in range(_ATTEMPTS):
        try:
            return await client.call(action, params)
        except CKANActionError as exc:
            if missing_ok and exc.not_found:
                return None
            if not exc.transient or attempt == _ATTEMPTS - 1:
                raise MirrorError(f"{action} failed for {label}: {exc}") from exc
        await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


def _slug(text: str, fallback: str) -> str:
    """Return a CKAN-safe slug while retaining the legacy helper API."""
    slug = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) < 2:
        slug = fallback
    return slug[:100]


def _load_index(site: str, index_path: str | Path | None = None) -> dict[str, Any]:
    """Load qsv output when present; mirroring itself does not require it."""
    candidates = (
        [Path(index_path)]
        if index_path
        else [
            _PROJECT_ROOT / "data" / "ckan_onboard" / site / "index.json",
            _PROJECT_ROOT / "data" / "dcat_onboard" / site / "index.json",
            _PROJECT_ROOT / "ckan_onboard" / site / "index.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _column_dictionary(resource: dict[str, Any]) -> list[dict[str, Any]]:
    """Return publishable qsv column metadata with secrets scrubbed."""
    columns: list[dict[str, Any]] = []
    for column in resource.get("columns", []) or []:
        stats = column.get("stats", {}) or {}
        columns.append(
            {
                "name": column.get("name", ""),
                "label": column.get("qsv_label") or column.get("ckan_info", {}).get("label", ""),
                "description": _scrub_secrets(column.get("qsv_description", "")),
                "type": column.get("qsv_type") or column.get("ckan_type", ""),
                "stats": {
                    key: stats.get(key)
                    for key in (
                        "type",
                        "min",
                        "max",
                        "mean",
                        "q2_median",
                        "stddev",
                        "nullcount",
                        "cardinality",
                    )
                    if stats.get(key) not in (None, "")
                },
                "top_values": (column.get("top_values") or [])[:10],
            }
        )
    return columns


def _qsv_by_resource(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(resource["resource_id"]): _vet_profile(resource)
        for dataset in index.get("datasets", []) or []
        for resource in dataset.get("resources", []) or []
        if resource.get("resource_id")
    }


def _raise_csv_field_limit() -> None:
    # Python's csv module refuses fields over 128 KB, which qsv emits for
    # geometry columns.
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))


def _profiled_header(qsv: dict[str, Any]) -> list[str] | None:
    """Column names of the file qsv profiled, while it is still on disk."""
    local_path = str(qsv.get("local_path") or "")
    if not local_path:
        return None
    path = Path(local_path)
    path = path if path.is_absolute() else _PROJECT_ROOT / path
    if not path.is_file():
        return None
    _raise_csv_field_limit()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [name for name in next(csv.reader(handle), []) if name]


def _qsv_fields(path: Path) -> set[str] | None:
    """The columns a qsv stats or frequency CSV reports on."""
    if not path.is_file():
        return None
    _raise_csv_field_limit()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return {row.get("field") or "" for row in csv.DictReader(handle)} - {""}


def _vet_profile(qsv: dict[str, Any]) -> dict[str, Any]:
    """A profile checked against the file it names, ready to publish.

    onboard_*.py writes ``qsv_stats.csv`` and ``qsv_frequency.csv`` once per
    dataset directory, so when a dataset has several CSVs and one profile
    fails, the directory can hold another file's output — two WPRDC datasets
    showed another file's columns and row counts under the profiled file's
    name. Output that does not match the profiled file's header is left out,
    as are the statistics merged from it. Tags are re-read from describegpt's
    own output, whose key is sometimes capitalised.
    """
    if "_vetted" in qsv:
        return qsv
    profile = dict(qsv, _vetted=True)
    header = _profiled_header(qsv)
    directory = _qsv_dir(qsv)
    trusted = str(qsv.get("status") or "") == "ok"

    def describes(name: str, *, exact: bool) -> bool:
        fields = _qsv_fields(directory / name) if directory else None
        if fields is None:
            return False
        if header is None:
            return trusted
        return fields == set(header) if exact else fields <= set(header)

    profile["_stats_ok"] = describes("qsv_stats.csv", exact=True)
    profile["_frequency_ok"] = describes("qsv_frequency.csv", exact=False)
    profile["_header"] = header
    if header is None and not trusted:
        # A failed profile whose file is gone: its merged statistics cannot be
        # checked against the file, so only names, labels and types remain.
        profile["columns"] = [
            {**column, "stats": {}, "top_values": []} for column in qsv.get("columns") or []
        ]
    elif header is not None and not (profile["_stats_ok"] and profile["_frequency_ok"]):
        # Keep the file's own columns and the portal's DataStore fields; drop
        # columns only the mismatched output contributed.
        profile["columns"] = [
            {
                **column,
                "stats": column.get("stats") if profile["_stats_ok"] else {},
                "top_values": column.get("top_values") if profile["_frequency_ok"] else [],
            }
            for column in qsv.get("columns") or []
            if column.get("name") in header or column.get("ckan_info") or column.get("ckan_type")
        ]

    # describegpt's own dictionary names the columns it described; like the
    # stats, it can belong to another file in the same directory.
    describegpt = directory / "qsv_dict.json" if directory else None
    try:
        output = (
            json.loads(describegpt.read_text(encoding="utf-8"))
            if describegpt is not None and describegpt.is_file()
            else None
        )
    except (ValueError, OSError):
        output = None
    described = {
        str(field.get("name"))
        for field in ((output or {}).get("Dictionary") or {}).get("response", {}).get("fields")
        or []
        if isinstance(field, dict) and field.get("name")
    }
    if output is None:
        profile["_describegpt_ok"] = False
    elif header is None or not described:
        profile["_describegpt_ok"] = trusted
    else:
        profile["_describegpt_ok"] = described - {"_id"} <= set(header)
    if profile["_describegpt_ok"]:
        tags = _extract_qsv_tags(output or {})
        if tags:
            profile["qsv_tags"] = tags
    elif output is not None:
        # Another file's AI output: its summary, tags and column labels go too.
        profile["qsv_description"] = ""
        profile["qsv_tags"] = []
        profile["columns"] = [
            {
                key: value
                for key, value in column.items()
                if key not in ("qsv_label", "qsv_description")
            }
            for column in profile.get("columns") or []
        ]
    return profile


def _project(source: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {
        field: source[field] for field in fields if field in source and source[field] is not None
    }


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _extra_value(value: Any) -> str:
    """Dataset extras are strings in CKAN; structured values are kept as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _source_id(row: dict[str, Any]) -> str:
    """The identifier the source itself uses for a record.

    For a CKAN source that is the preserved UUID. A DCAT snapshot records the
    catalog's own identifier under ``_source_id`` because its UUIDs are derived.
    """
    return str(row.get("_source_id") or row.get("id", ""))


def _with_image(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Point an organization or group image at a URL that resolves on the Fair Store.

    An uploaded image is stored as a bare filename that CKAN resolves against
    the serving site's own ``/uploads``, where the mirror never put the file,
    so the source's absolute ``image_display_url`` is used instead.
    """
    image = str(row.get("image_url") or "")
    if image and not image.startswith(("http://", "https://")):
        image = str(row.get("image_display_url") or "")
    if image:
        payload["image_url"] = image
    else:
        payload.pop("image_url", None)
    return payload


def _provenance_extras(
    extras: list[dict[str, Any]] | None,
    *,
    site_id: str,
    source_url: str,
    source_id: str,
    carried: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Preserve source extras and add stable mirror identity fields.

    ``carried`` holds source fields with no column in the target schema; a
    source extra of the same name wins, and the identity fields win over both.
    """
    values = dict(carried or {})
    values.update(
        {
            str(item.get("key")): str(item.get("value", ""))
            for item in (extras or [])
            if item.get("key")
        }
    )
    values.update(
        {
            _PROVENANCE_KEYS["portal"]: site_id,
            _PROVENANCE_KEYS["url"]: source_url,
            _PROVENANCE_KEYS["id"]: source_id,
        }
    )
    return [{"key": key, "value": value} for key, value in values.items()]


def _mirrored_from(row: dict[str, Any]) -> str:
    """Name the portal a source record was itself mirrored from, if any."""
    for item in row.get("extras") or []:
        if isinstance(item, dict) and item.get("key") == _PROVENANCE_KEYS["portal"]:
            return str(item.get("value") or "").strip()
    return ""


def _group_refs(groups: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for group in groups or []:
        name = group.get("name")
        if name:
            ref = {"name": str(name)}
            if group.get("capacity"):
                ref["capacity"] = str(group["capacity"])
            refs.append(ref)
    return refs


def _clean_tag(raw: str) -> str:
    """Return ``raw`` as a valid CKAN tag, or ``""`` when nothing usable is left.

    CKAN accepts 2-100 word characters, spaces, hyphens and dots. Valid tags pass
    through untouched; DCAT keywords such as ``l&i`` are rewritten (``l-i``).
    """
    tag = raw.strip()
    if _TAG_VALID_RE.fullmatch(tag):
        return tag
    tag = re.sub(r"-{2,}", "-", _TAG_INVALID_RE.sub("-", tag)).strip(" -")
    tag = tag[:100].strip()
    return tag if len(tag) >= 2 else ""


def _tag_refs(tags: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for tag in tags or []:
        name = _clean_tag(str(tag.get("name") or ""))
        if not name or name in seen:
            continue
        seen.add(name)
        refs.append(
            {
                "name": name,
                **({"vocabulary_id": tag["vocabulary_id"]} if tag.get("vocabulary_id") else {}),
            }
        )
    return refs


async def _all_packages(
    client: CKANClient, *, label: str = "portal", include_private: bool = False
) -> list[dict[str, Any]]:
    """Every dataset the portal lists; raises rather than return part of them."""
    params: dict[str, Any] = {"q": "*:*", "rows": 1000, "sort": "name asc"}
    if include_private:
        # The Fair Store's own private and draft datasets still hold their
        # UUIDs: left out, they would look new and never be updated.
        params.update(include_private=True, include_drafts=True)
    packages: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    start = 0
    while True:
        page = await _read(client, "package_search", {**params, "start": start}, label=label)
        if not isinstance(page, dict):
            raise MirrorError(f"package_search for {label} returned {type(page).__name__}")
        batch = page.get("results") or []
        count = int(page.get("count") or 0)
        packages.extend(row for row in batch if str(row.get("id")) not in seen_ids)
        seen_ids.update(str(row.get("id")) for row in batch)
        start += len(batch)
        if start >= count:
            return packages
        if not batch:
            raise MirrorError(f"package_search for {label} stopped at {start} of {count} datasets")


async def _all_group_rows(
    client: CKANClient, action_name: str, *, label: str = "portal", include_extras: bool = False
) -> list[dict[str, Any]]:
    """Read every organization or group despite portal-side page-size caps."""
    # Sorted by name, which is unique: CKAN's default (title) order is not a
    # total order, so offset paging over it can skip a row.
    params: dict[str, Any] = {"all_fields": True, "limit": 1000, "sort": "name asc"}
    if include_extras:
        params["include_extras"] = True
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0
    while True:
        result = await _read(client, action_name, {**params, "offset": offset}, label=label)
        if not isinstance(result, list):
            raise MirrorError(f"{action_name} for {label} returned {type(result).__name__}")
        batch = list(result)
        if not batch:
            return rows
        new_rows = [row for row in batch if str(row.get("id")) not in seen_ids]
        if not new_rows:
            return rows
        rows.extend(new_rows)
        seen_ids.update(str(row.get("id")) for row in new_rows)
        offset += len(batch)


def _empty_copies() -> dict[str, Any]:
    return {"organizations": 0, "groups": 0, "datasets": 0, "origins": []}


async def _check_complete(
    client: CKANClient, action_name: str, rows: list[dict[str, Any]], *, label: str
) -> None:
    """Cross-check paged ``all_fields`` rows against the plain name listing.

    The name listing is one unpaged call: WPRDC's CDN answers 403 to a
    names-only listing that carries limit/offset. CKAN returns up to 1,000
    names that way; a portal at that cap is not checked.
    """
    listed = await _read(client, action_name, {"sort": "name asc"}, label=label)
    if not isinstance(listed, list):
        raise MirrorError(f"{action_name} for {label} returned {type(listed).__name__}")
    names = {str(item.get("name") if isinstance(item, dict) else item) for item in listed}
    if len(listed) >= 1000:
        print(f"warning: {action_name} for {label}: too many to cross-check", file=sys.stderr)
        return
    missing = names - {str(row.get("name")) for row in rows}
    if missing:
        raise MirrorError(
            f"{action_name} for {label} paged {len(rows)} of {len(names)} records; "
            f"missing e.g. {sorted(missing)[:5]}"
        )


async def _source_snapshot(
    client: CKANClient,
    *,
    site_id: str | None = None,
    organization: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Read a CKAN source, leaving out records it mirrored from another portal."""
    copies = _empty_copies()
    origins: set[str] = set()

    def is_copy(row: dict[str, Any], kind: str) -> bool:
        origin = _mirrored_from(row)
        if site_id and origin and origin != site_id:
            copies[kind] += 1
            origins.add(origin)
            return True
        return False

    label = f"source {site_id or client.ckan_url}"
    organization_rows = await _all_group_rows(client, "organization_list", label=label)
    await _check_complete(client, "organization_list", organization_rows, label=label)
    organizations: list[dict[str, Any]] = []
    for row in organization_rows:
        if organization and row.get("name") != organization:
            continue
        # The list row lacks the extras (including any mirror provenance) and
        # the parent groups, so it is never a stand-in for the full record.
        full = await _read(
            client,
            "organization_show",
            {
                "id": row["id"],
                "include_datasets": False,
                "include_users": False,
                "include_groups": True,
                "include_tags": True,
            },
            label=f"{label} organization {row.get('name')}",
        )
        if not is_copy(full, "organizations"):
            organizations.append(full)

    # With extras, so a group the source itself mirrored is seen as a copy.
    groups = [
        group
        for group in await _all_group_rows(client, "group_list", label=label, include_extras=True)
        if not is_copy(group, "groups")
    ]

    packages = [
        package
        for package in await _all_packages(client, label=label)
        if not is_copy(package, "datasets")
    ]
    if organization:
        org_ids = {str(item.get("id")) for item in organizations}
        packages = [
            package
            for package in packages
            if (package.get("organization") or {}).get("name") == organization
            or str(package.get("owner_org")) in org_ids
        ]
    if limit is not None:
        packages = packages[:limit]
    used_group_names = {
        str(group.get("name"))
        for package in packages
        for group in package.get("groups", []) or []
        if group.get("name")
    }
    groups = [group for group in groups if group.get("name") in used_group_names]
    copies["origins"] = sorted(origins)
    return {
        "organizations": organizations,
        "groups": groups,
        "packages": packages,
        "copies": copies,
    }


def _dcat_uuid(identifier: str, source_url: str, kind: str) -> str:
    """Stable UUID for a DCAT record: the source's own UUID when it has one."""
    try:
        return str(uuid.UUID(identifier))
    except (ValueError, TypeError):
        pass
    if urlparse(identifier).scheme in ("http", "https"):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identifier))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_url.rstrip('/')}#{kind}/{identifier}"))


def _socrata_id(identifier: str) -> str:
    match = _SOCRATA_VIEW_RE.search(urlparse(identifier or "").path)
    return match.group(1) if match else ""


def _dcat_dataset_name(title: str, identifier: str, dataset_id: str) -> str:
    """A unique, stable CKAN name: the title slug plus the source's short ID.

    CKAN names are site-wide, and portal titles repeat ("Crashes", "Budget"), so
    every name carries the portal's own ID (a Socrata 4x4) or a UUID prefix.
    """
    tail = urlparse(identifier).path.rstrip("/").rsplit("/", 1)[-1] if identifier else ""
    suffix = tail.lower() if re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,19}", tail.lower()) else ""
    suffix = suffix or dataset_id[:8]
    base = _slug(title, "dataset")[: 100 - len(suffix) - 1].rstrip("-")
    return f"{base}-{suffix}"


def _publisher_chain(publisher: Any) -> list[str]:
    """Publisher names from most specific to broadest (DCAT-US subOrganizationOf)."""
    chain: list[str] = []
    node = publisher
    while node and len(chain) < 10:
        name = _text(node)
        if name and name not in chain:
            chain.append(name)
        node = node.get("subOrganizationOf") if isinstance(node, dict) else None
    return chain


def _lineage(parents: dict[str, str], name: str) -> set[str]:
    """``name`` and every ancestor recorded for it in ``parents``."""
    seen: set[str] = set()
    while name and name not in seen:
        seen.add(name)
        name = parents.get(name, "")
    return seen


def _mailto(value: Any) -> str:
    email = _text(value).removeprefix("mailto:").strip()
    return email if _EMAIL_RE.fullmatch(email) else ""


def _license_key(url: str) -> str:
    parsed = urlparse(url.strip().lower())
    host = parsed.netloc.removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/')}" if host else url.strip().lower()


def _dcat_license_id(url: str) -> str:
    return _LICENSE_IDS.get(_license_key(url), "") if url else ""


def _media_format(media_type: str, declared: str) -> str:
    return declared or _MEDIA_TYPE_FORMATS.get(media_type.lower(), "")


def _dcat_extras(raw: dict[str, Any]) -> list[dict[str, str]]:
    """Every DCAT dataset field as a ``dcat_``-prefixed extra.

    Fields copied into a CKAN core field (title, description, landing page,
    version) are not duplicated; keyword, theme, license and contactPoint are
    kept verbatim because their CKAN counterparts are normalized.
    """
    skipped = {"distribution", "title", "description", "landingPage", "version"}
    return [
        {"key": f"dcat_{key.lstrip('@')}", "value": _extra_value(_stable(key, value))}
        for key, value in raw.items()
        if key not in skipped and not _is_empty(value)
    ]


# Catalog fields that are sets. Socrata serves them in a different order on
# each fetch, so they are sorted to keep identical content identical.
_UNORDERED_FIELDS = frozenset({"keyword", "theme", "domain_tags"})


def _stable(key: str, value: Any) -> Any:
    if key in _UNORDERED_FIELDS and isinstance(value, list):
        return sorted(value, key=lambda item: json.dumps(item, sort_keys=True))
    return value


def _socrata_owner(row: dict[str, Any]) -> tuple[str, str]:
    """The owning agency Socrata records for a dataset, and the field it came from."""
    for item in (row.get("classification") or {}).get("domain_metadata") or []:
        key, value = str(item.get("key") or ""), str(item.get("value") or "").strip()
        if value and _SOCRATA_OWNER_KEY_RE.search(key):
            return value, key
    attribution = str((row.get("resource") or {}).get("attribution") or "").strip()
    return (attribution, "attribution") if attribution else ("", "")


def _socrata_extras(row: dict[str, Any]) -> list[dict[str, str]]:
    """Discovery API metadata the DCAT export omits, as ``socrata_`` extras.

    Volatile analytics (page views, downloads) and account names are left out.
    """
    resource = row.get("resource") or {}
    classification = row.get("classification") or {}
    values: dict[str, Any] = {
        "socrata_id": resource.get("id"),
        "socrata_type": resource.get("type"),
        "socrata_attribution": resource.get("attribution"),
        "socrata_attribution_link": resource.get("attribution_link"),
        "socrata_provenance": resource.get("provenance"),
        "socrata_created_at": resource.get("createdAt"),
        "socrata_updated_at": resource.get("updatedAt"),
        "socrata_data_updated_at": resource.get("data_updated_at"),
        "socrata_metadata_updated_at": resource.get("metadata_updated_at"),
        "socrata_publication_date": resource.get("publication_date"),
        "socrata_category": classification.get("domain_category"),
        "socrata_domain_tags": _stable("domain_tags", classification.get("domain_tags")),
        "socrata_license": (row.get("metadata") or {}).get("license"),
        "socrata_permalink": row.get("permalink"),
    }
    for item in classification.get("domain_metadata") or []:
        if item.get("key"):
            values[f"socrata_{item['key']}"] = item.get("value")
    return [
        {"key": key, "value": _extra_value(value)}
        for key, value in values.items()
        if not _is_empty(value)
    ]


def _socrata_columns(row: dict[str, Any]) -> list[dict[str, str]]:
    """The column list Socrata publishes for a dataset, as a data dictionary."""
    resource = row.get("resource") or {}
    fields = resource.get("columns_field_name") or []

    def at(key: str, index: int) -> str:
        values = resource.get(key) or []
        return str(values[index] or "") if index < len(values) else ""

    # The column arrays agree with one another but not from fetch to fetch,
    # so the dictionary is ordered by field name.
    return sorted(
        (
            {
                "name": str(field),
                "label": at("columns_name", index),
                "type": at("columns_datatype", index),
                "description": _scrub_secrets(at("columns_description", index)),
            }
            for index, field in enumerate(fields)
        ),
        key=lambda column: column["name"],
    )


def _dcat_resources(raw: dict[str, Any], *, dataset_id: str) -> list[dict[str, Any]]:
    """CKAN resources for a dataset's DCAT distributions."""
    identifier = _text(raw.get("identifier") or raw.get("@id"))
    resources: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    distributions = [d for d in raw.get("distribution") or [] if isinstance(d, dict)]
    for index, dist in enumerate(distributions):
        download_url = _text(dist.get("downloadURL"))
        url = download_url or _text(dist.get("accessURL"))
        media_type = _text(dist.get("mediaType"))
        fmt = _media_format(media_type, _text(dist.get("format")))
        key = url or str(index)
        if key in used_keys:
            key = f"{key}#{index}"
        used_keys.add(key)
        resource: dict[str, Any] = {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset_id}/{key}")),
            "_source_id": url or f"{identifier}#distribution-{index}",
            "url": url,
            "name": _text(dist.get("title")) or fmt or media_type or f"Distribution {index + 1}",
            "description": _text(dist.get("description")),
            "format": fmt,
            "mimetype": media_type or None,
        }
        for field_name, value in dist.items():
            if field_name in {"title", "description", "mediaType", "format", "downloadURL"}:
                continue
            if field_name == "accessURL" and not download_url:
                continue
            if not _is_empty(value):
                resource[f"dcat_{field_name.lstrip('@')}"] = value
        resources.append(resource)
    return resources


def _dcat_snapshot(
    catalog: Any,
    *,
    site_title: str,
    source_url: str,
    socrata: dict[str, dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Translate a DCAT catalog into the CKAN-shaped snapshot the mirror writes.

    Organizations come from each dataset's publisher chain, most specific first,
    headed by the Socrata owning agency when the Discovery API names one. A
    publisher that is the portal itself is represented by the portal's root
    organization rather than duplicated beneath it.
    """
    nodes = raw_dataset_nodes(catalog)
    if limit is not None:
        nodes = nodes[:limit]
    host = (urlparse(source_url).hostname or "").lower()
    portal_names = {host, host.removeprefix("www."), site_title.strip().lower()} - {""}

    org_parent: dict[str, str] = {}
    org_field: dict[str, str] = {}
    themes: dict[str, str] = {}
    packages: list[dict[str, Any]] = []
    names: set[str] = set()

    def org_id(title: str) -> str:
        return _dcat_uuid(title, source_url, "publisher")

    for raw in nodes:
        identifier = _text(raw.get("identifier") or raw.get("@id"))
        title = _text(raw.get("title"))
        dataset_id = _dcat_uuid(identifier or title, source_url, "dataset")
        row = (socrata or {}).get(_socrata_id(identifier))

        chain = _publisher_chain(raw.get("publisher"))
        fields = ["dcat:publisher"] * len(chain)
        if row:
            owner, owner_key = _socrata_owner(row)
            if owner and owner not in chain:
                chain.insert(0, owner)
                fields.insert(0, f"socrata:{owner_key}")
        kept = [(n, f) for n, f in zip(chain, fields, strict=True) if n.lower() not in portal_names]
        for (child, _), (parent, _) in zip(kept, kept[1:], strict=False):
            # A parent learned from any dataset fills in one not yet known, but
            # a catalog that contradicts itself must not create a loop, which
            # ckanext-hierarchy rejects mid-run.
            if not org_parent.get(child) and child not in _lineage(org_parent, parent):
                org_parent[child] = parent
        for name_, field_ in kept:
            org_parent.setdefault(name_, "")
            org_field.setdefault(name_, field_)

        theme_slugs: list[str] = []
        for theme in _as_list(raw.get("theme")):
            slug = _slug(theme, _dcat_uuid(theme, source_url, "theme")[:8])
            themes.setdefault(slug, theme)
            if slug not in theme_slugs:
                theme_slugs.append(slug)

        name = _dcat_dataset_name(title or identifier, identifier, dataset_id)
        if name in names:
            name = f"{name[:91]}-{dataset_id[:8]}"
        names.add(name)

        contact = raw.get("contactPoint") if isinstance(raw.get("contactPoint"), dict) else {}
        extras = _dcat_extras(raw) + (_socrata_extras(row) if row else [])
        parsed = parse_dataset(raw)
        tabular = parsed.tabular_distributions
        first_tabular_url = tabular[0].best_url if tabular else ""
        resources = _dcat_resources(raw, dataset_id=dataset_id)
        columns = _socrata_columns(row) if row else []
        if columns:
            # The column list goes on the file it describes, beside any qsv
            # profile. As a dataset extra it would also be indexed as one Solr
            # term, and a wide table's list overruns the term limit.
            dictionary = {
                "source_data_dictionary": json.dumps(columns),
                "source_column_count": str(len(columns)),
            }
            described = next(
                (r for r in resources if first_tabular_url and r["url"] == first_tabular_url),
                resources[0] if resources else None,
            )
            if described is not None:
                described.update(dictionary)
            else:
                extras += [{"key": key, "value": value} for key, value in dictionary.items()]

        packages.append(
            {
                "id": dataset_id,
                "_source_id": identifier or title,
                # How onboard_dcat.py files this dataset's qsv profile.
                "_short_id": parsed.id,
                "_first_tabular_url": first_tabular_url,
                "name": name,
                "title": title or identifier,
                "notes": _text(raw.get("description")),
                "url": _text(raw.get("landingPage")) or None,
                "version": _text(raw.get("version")) or None,
                "maintainer": _text(contact.get("fn")) or None,
                "maintainer_email": _mailto(contact.get("hasEmail")) or None,
                "license_id": _dcat_license_id(_text(raw.get("license"))) or None,
                "owner_org": org_id(kept[0][0]) if kept else "",
                "tags": [
                    {"name": keyword} for keyword in sorted(_as_list(raw.get("keyword")), key=str)
                ],
                "groups": [{"name": slug} for slug in theme_slugs],
                "extras": extras,
                "resources": resources,
            }
        )

    # Names are assigned after every publisher is known so they do not depend on
    # catalog order; two titles that slug alike are told apart by UUID prefix.
    org_names: dict[str, str] = {}
    for title in sorted(org_parent):
        slug = _slug(title, org_id(title)[:8])
        org_names[title] = (
            slug if slug not in org_names.values() else f"{slug[:91]}-{org_id(title)[:8]}"
        )
    organizations = [
        {
            "id": org_id(title),
            "_source_id": title,
            "name": org_names[title],
            "title": title,
            "groups": (
                [{"name": org_names[org_parent[title]], "capacity": "parent"}]
                if org_parent[title]
                else []
            ),
            "extras": [{"key": "publisher_source", "value": org_field[title]}],
        }
        for title in sorted(org_parent)
    ]
    groups = [
        {
            "id": _dcat_uuid(theme, source_url, "theme"),
            "_source_id": theme,
            "name": slug,
            "title": theme,
        }
        for slug, theme in sorted(themes.items())
    ]
    return {
        "organizations": organizations,
        "groups": groups,
        "packages": packages,
        "copies": _empty_copies(),
    }


def _dcat_qsv_index(index: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Re-key a DCAT onboarding index to the mirrored resource UUIDs.

    ``onboard_dcat.py`` profiles each dataset's first tabular distribution and
    files it under the dataset's short ID rather than a resource ID.
    """
    by_short_id: dict[str, str] = {}
    for package in snapshot["packages"]:
        for resource in package.get("resources", []):
            if (
                package.get("_first_tabular_url")
                and resource.get("url") == package["_first_tabular_url"]
            ):
                by_short_id[str(package.get("_short_id"))] = str(resource["id"])
                break
    rekeyed = [
        {**resource, "resource_id": by_short_id[str(resource.get("resource_id"))]}
        for dataset in index.get("datasets", []) or []
        for resource in dataset.get("resources", []) or []
        if str(resource.get("resource_id")) in by_short_id
    ]
    return {"datasets": [{"resources": rekeyed}]} if rekeyed else {}


async def _socrata_page(client: httpx.AsyncClient, domain: str, offset: int) -> dict[str, Any]:
    """One Discovery API page, retried when the failure is transient."""
    params = {"domains": domain, "search_context": domain, "limit": 1000, "offset": offset}
    for attempt in range(_ATTEMPTS):
        try:
            resp = await client.get(_SOCRATA_DISCOVERY_URL, params=params)
            if resp.status_code != 429 and resp.status_code < 500:
                resp.raise_for_status()
                body = resp.json()
                if isinstance(body, dict):
                    return body
                raise ValueError(f"unexpected {type(body).__name__} body")
            error: Exception = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
        except httpx.TransportError as exc:
            error = exc
        if attempt == _ATTEMPTS - 1:
            raise error
        await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


async def _socrata_metadata(catalog: Any, *, required: bool) -> dict[str, dict[str, Any]]:
    """Discovery API records for a Socrata-hosted catalog, keyed by 4x4 ID.

    Returns ``{}`` for a catalog that is not Socrata's. The owning agency comes
    only from this API — data.pa.gov's own catalog names one publisher for every
    dataset — so when ``required`` (a run that writes) a failure raises instead
    of mirroring a snapshot that would move every dataset to the portal root and
    drop its ``socrata_*`` extras. Otherwise it warns and returns what it has.
    """
    domains = {
        (urlparse(_text(node.get("identifier"))).hostname or "")
        for node in raw_dataset_nodes(catalog)
        if _socrata_id(_text(node.get("identifier")))
    } - {""}
    rows: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for domain in sorted(domains):
            offset = 0
            while True:
                try:
                    body = await _socrata_page(client, domain, offset)
                except (httpx.HTTPError, ValueError) as exc:
                    message = f"Socrata enrichment failed for {domain} at offset {offset}: {exc}"
                    if required:
                        raise MirrorError(
                            f"{message}. Retry later; --no-enrich mirrors without it, which "
                            "moves every dataset to the portal root organization."
                        ) from exc
                    print(f"warning: {message}", file=sys.stderr)
                    break
                batch = body.get("results") or []
                for result in batch:
                    view_id = (result.get("resource") or {}).get("id")
                    if view_id:
                        rows[str(view_id)] = result
                offset += len(batch)
                total = int(body.get("resultSetSize") or 0)
                if offset >= total:
                    break
                if not batch:
                    message = f"Socrata enrichment for {domain} stopped at {offset} of {total}"
                    if required:
                        raise MirrorError(message)
                    print(f"warning: {message}", file=sys.stderr)
                    break
    return rows


async def _target_snapshot(client: CKANClient) -> dict[str, list[dict[str, Any]]]:
    label = f"target {client.ckan_url}"
    # Extras carry the provenance that tells a rename of this portal's own
    # record apart from a collision with someone else's.
    return {
        "organizations": await _all_group_rows(
            client, "organization_list", label=label, include_extras=True
        ),
        "groups": await _all_group_rows(client, "group_list", label=label, include_extras=True),
        "packages": await _all_packages(client, label=label, include_private=True),
    }


def _pin_target_names(
    source: dict[str, Any], target: dict[str, list[dict[str, Any]]], *, site_id: str
) -> None:
    """Keep the names already published for a DCAT portal's records.

    DCAT names are derived — from a title slug, and for organizations whose
    titles slug alike, from sort order — so an edited title or a newly added
    publisher would otherwise rename published datasets and break their URLs.
    A record this portal already holds keeps its name; a new record whose name
    is taken gets its UUID prefix appended.
    """
    for kind in ("organizations", "packages"):
        held = {
            str(row.get("id")): str(row.get("name"))
            for row in target[kind]
            if _mirrored_from(row) == site_id and row.get("name")
        }
        rows = source[kind]
        renamed: dict[str, str] = {}
        for row in rows:
            name = held.get(str(row.get("id")))
            if name and name != row.get("name"):
                renamed[str(row.get("name"))] = name
                row["name"] = name
        # A new record may not take any name the Fair Store already uses, nor
        # one given to an earlier record of this run.
        taken = {str(row.get("name")) for row in target[kind] if row.get("name")}
        taken |= {str(row.get("name")) for row in rows if str(row.get("id")) in held}
        for row in rows:
            row_id, name = str(row.get("id")), str(row.get("name"))
            if row_id in held:
                continue
            if name in taken:
                new_name = f"{name[:91]}-{row_id[:8]}"
                renamed[name] = new_name
                row["name"] = name = new_name
            taken.add(name)
        if kind == "organizations" and renamed:
            # Parent references are by name.
            for row in rows:
                for parent in row.get("groups") or []:
                    if parent.get("name") in renamed:
                        parent["name"] = renamed[parent["name"]]


def _assert_no_collisions(
    source: dict[str, Any],
    target: dict[str, list[dict[str, Any]]],
    *,
    store_name: str,
    store_id: str,
    site_id: str = "",
    evictions: list[tuple[str, str, str]] | None = None,
) -> list[str]:
    """Raise on anything a write would clobber; return the renames it will apply.

    The same UUID under a new name is a rename when the Fair Store's record is
    this portal's own (its provenance names ``site_id``): the source renamed
    it, and the patch carries the new name. Under any other provenance it is a
    collision, as is a name held by a different UUID — unless the holder is this
    portal's own record that its source no longer publishes and ``evictions``
    is given: the source reused the name, so the withdrawn record is to be
    renamed out of the way, collected as ``(kind, id, new name)``.
    """
    collisions: list[str] = []
    renames: list[str] = []

    def check(
        kind: str,
        incoming: list[dict[str, Any]],
        existing: list[dict[str, Any]],
        *,
        allow_shared_name: bool = False,
    ) -> None:
        by_name = {str(row.get("name")): row for row in existing if row.get("name")}
        by_id = {str(row.get("id")): row for row in existing if row.get("id")}
        incoming_ids = {str(row.get("id", "")) for row in incoming}
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for row in incoming:
            name, row_id = str(row.get("name", "")), str(row.get("id", ""))
            if row_id in seen_ids:
                collisions.append(f"{kind} UUID {row_id!r} appears twice in the source")
            if name in seen_names and not allow_shared_name:
                collisions.append(f"{kind} name {name!r} appears twice in the source")
            seen_ids.add(row_id)
            seen_names.add(name)
            holder = by_name.get(name)
            if not allow_shared_name and holder is not None and str(holder.get("id")) != row_id:
                holder_id = str(holder.get("id"))
                if (
                    evictions is not None
                    and site_id
                    and _mirrored_from(holder) == site_id
                    and holder_id not in incoming_ids
                ):
                    evicted = f"{name[:80]}-withdrawn-{holder_id[:8]}"
                    evictions.append((kind, holder_id, evicted))
                    renames.append(f"{kind} {name!r} (withdrawn at source) -> {evicted!r}")
                else:
                    collisions.append(f"{kind} name {name!r} already has a different UUID")
            held = by_id.get(row_id)
            if held is not None and str(held.get("name")) != name:
                if site_id and _mirrored_from(held) == site_id:
                    renames.append(f"{kind} {held.get('name')!r} -> {name!r}")
                else:
                    collisions.append(f"{kind} UUID {row_id!r} already has a different name")

    root_existing = next(
        (row for row in target["organizations"] if row.get("name") == store_name), None
    )
    if root_existing and str(root_existing.get("id")) != store_id:
        collisions.append(f"source-store organization {store_name!r} already exists")
    if any(str(row.get("id")) == store_id for row in source["organizations"]):
        collisions.append(f"source organization UUID {store_id!r} matches the source-store UUID")
    # A prior run may have used the bare site ID for the synthetic root. It is
    # renamed before source organizations are written, so exclude it here.
    target_organizations = [
        row for row in target["organizations"] if str(row.get("id")) != store_id
    ]
    check("organization", source["organizations"], target_organizations)
    # CKAN group/category names are site-wide. Reuse an existing category with
    # the same name instead of rewriting it with another portal's UUID.
    check("group", source["groups"], target["groups"], allow_shared_name=True)
    check("dataset", source["packages"], target["packages"])

    target_resources = [
        resource
        for package in target["packages"]
        for resource in package.get("resources", []) or []
    ]
    source_package_by_resource = {
        str(resource.get("id")): str(package.get("id"))
        for package in source["packages"]
        for resource in package.get("resources", []) or []
        if resource.get("id")
    }
    target_by_id = {
        str(resource.get("id")): resource for resource in target_resources if resource.get("id")
    }
    for resource_id in sorted(set(source_package_by_resource) & target_by_id.keys()):
        if (
            str(target_by_id[resource_id].get("package_id"))
            != source_package_by_resource[resource_id]
        ):
            collisions.append(f"resource UUID {resource_id!r} belongs to another target dataset")

    if collisions:
        detail = "\n  - ".join(collisions[:50])
        raise MirrorError(f"Mirror preflight found {len(collisions)} collision(s):\n  - {detail}")
    return renames


_SHOW_ACTIONS = {
    "organization_create": "organization_show",
    "group_create": "group_show",
    "package_create": "package_show",
    "resource_create": "resource_show",
}


async def _write_action(
    client: CKANClient,
    action: str,
    payload: dict[str, Any],
    *,
    label: str,
    upload: tuple[str, bytes, str] | None = None,
) -> dict[str, Any]:
    """Run a write, retrying transient failures; ``upload`` is (name, bytes, type).

    A create whose object turns out to exist is resolved by its preserved UUID:
    a gateway can fail after CKAN commits, and the retry must not be reported
    as a failure (or repeated). Writes without a UUID to check — DataStore
    loads, view creation — are made safe by their callers instead.
    """
    show_action = _SHOW_ACTIONS.get(action)
    for attempt in range(_ATTEMPTS):
        try:
            if upload:
                filename, content, content_type = upload
                result = await client.upload_call(
                    action, payload, filename=filename, content=content, content_type=content_type
                )
            else:
                result = await client.call(action, payload)
            return result if isinstance(result, dict) else {"result": result}
        except CKANActionError as exc:
            error = exc

        if show_action and payload.get("id"):
            existing = await _read(
                client, show_action, {"id": payload["id"]}, label=label, missing_ok=True
            )
            if existing:
                if existing.get("state", "active") == "deleted":
                    raise DeletedInTarget(f"{label} was deleted in the Fair Store")
                # Package, organization and group names are identifiers too;
                # a resource name is only a label.
                if (
                    action != "resource_create"
                    and "name" in payload
                    and existing.get("name") != payload["name"]
                ):
                    raise MirrorError(
                        f"{action} failed for {label}: UUID {payload['id']!r} is held by "
                        f"{existing.get('name')!r}: {error}"
                    )
                return existing

        if not error.transient or attempt == _ATTEMPTS - 1:
            raise MirrorError(f"{action} failed for {label}: {error}") from error
        await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


# -- qsv profiles as dataset metadata and resources ---------------------------
#
# onboard_ckan.py / onboard_dcat.py profile a dataset's first CSV with qsv:
# stats, frequency, and describegpt (AI descriptions, labels, tags). The mirror
# publishes that profile twice over: compact fields on the dataset, and the full
# outputs as resources on it. Tables go into the DataStore, so CKAN shows them
# as sortable tables and serves them as CSV/JSON downloads; describegpt's JSON
# is uploaded as a file. The profiled source data itself is never copied.

_QSV_MODEL_RE = re.compile(r"^Model:\s*(\S+)", re.MULTILINE)
_QSV_VERSION_RE = re.compile(r"Generated by qsv v([\w.\-]+)")
# describegpt ends a description with a provenance block — "Generated by qsv
# vX describegpt", its command line, model, API URL, timestamp and an LLM
# warning — introduced however the model chose: "Attribution: Generated by…",
# an "## Attribution" heading, a bare line, a rule, <sub> or "@attribution".
_QSV_PROVENANCE_RE = re.compile(
    r"^[^\n]*\bGenerated by qsv\b|^[ \t]*Command line:[ \t]*qsv describegpt\b",
    re.IGNORECASE | re.MULTILINE,
)
_QSV_TRAILING_RE = re.compile(
    r"(?:\n[ \t]*(?:(?:#{1,6}|\*\*|<sub>|@)?[ \t]*attribution[ \t]*:?[ \t]*(?:\*\*|</sub>)?"
    r"|-{3,}|\*{3,}|_{3,})?[ \t]*)+$",
    re.IGNORECASE,
)
_DATASTORE_BATCH = 1000
_MAX_SUMMARY_CELL = 10_000


def _qsv_dir(qsv: dict[str, Any]) -> Path | None:
    """The onboarding directory holding a profile's qsv output files."""
    local_path = str(qsv.get("local_path") or "")
    if not local_path:
        return None
    path = Path(local_path)
    directory = (path if path.is_absolute() else _PROJECT_ROOT / path).parent
    return directory if directory.is_dir() else None


def _qsv_resource_id(profiled_resource_id: str, kind: str) -> str:
    """Stable UUID for one qsv output of one profiled resource."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qsv:{profiled_resource_id}:{kind}"))


def _profile_row_count(qsv: dict[str, Any]) -> int | None:
    """The profiled file's row count, when it is one.

    A byte-capped DCAT download counts only the rows it kept, and a profile
    whose download failed reports 0; neither is the file's size.
    """
    count = qsv.get("row_count")
    if isinstance(count, bool) or not isinstance(count, int) or qsv.get("truncated_download"):
        return None
    if count == 0 and str(qsv.get("status") or "") != "ok":
        return None
    return count


def _describegpt_prose(description: str) -> str:
    """A describegpt description without its trailing provenance block."""
    description = description or ""
    match = _QSV_PROVENANCE_RE.search(description)
    prose = description[: match.start()] if match else description
    return _QSV_TRAILING_RE.sub("", "\n" + prose).strip()


def _qsv_dataset_extras(qsv: dict[str, Any]) -> dict[str, str]:
    """The profile's dataset-level summary, shown with the dataset's metadata.

    Column names are included so a search for a column finds its dataset. The
    description keeps only describegpt's prose: the model and qsv version are
    fields of their own, and the full output, provenance included, is the
    describegpt resource.
    """
    qsv = _vet_profile(qsv)
    description = _scrub_secrets(str(qsv.get("qsv_description") or ""))
    # The profiled file's own columns: the dictionary also lists the portal's
    # DataStore fields, which a download can lack.
    columns = qsv.get("_header") or [
        str(c["name"]) for c in qsv.get("columns") or [] if c.get("name")
    ]
    model = _QSV_MODEL_RE.search(description)
    version = _QSV_VERSION_RE.search(description)
    values = {
        "qsv_description": _describegpt_prose(description),
        "qsv_tags": json.dumps(qsv.get("qsv_tags") or []) if qsv.get("qsv_tags") else "",
        "qsv_row_count": str(
            _profile_row_count(qsv) if _profile_row_count(qsv) is not None else ""
        ),
        # Statistics from a byte-capped download describe the file's first rows.
        "qsv_profile_truncated": "true" if qsv.get("truncated_download") else "",
        "qsv_column_count": str(len(columns)) if columns else "",
        # JSON, like qsv_tags: column names can contain commas.
        "qsv_columns": json.dumps(columns, ensure_ascii=False) if columns else "",
        "qsv_profiled_resource_id": str(qsv.get("resource_id") or ""),
        "qsv_profiled_at": str(qsv.get("onboarded_at") or ""),
        "qsv_status": str(qsv.get("status") or ""),
        "qsv_model": model.group(1) if model else "",
        "qsv_version": version.group(1) if version else "",
    }
    return {key: value for key, value in values.items() if value}


def _as_number(value: Any) -> float | int | None:
    """A DataStore numeric cell: qsv writes blanks and text where numbers are absent."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _top_values_text(top_values: list[dict[str, Any]] | None) -> str:
    return "; ".join(
        f"{item.get('value')} ({item.get('count')})" for item in (top_values or [])[:10]
    )


def _dictionary_table(qsv: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One row per column: qsv type, AI label and description, key statistics."""
    fields = [
        {
            "id": "column",
            "type": "text",
            "info": {"label": "Column", "notes": "Column name in the file"},
        },
        {"id": "label", "type": "text", "info": {"label": "Label", "notes": "AI-generated label"}},
        {"id": "type", "type": "text", "info": {"label": "Type", "notes": "Type inferred by qsv"}},
        {
            "id": "description",
            "type": "text",
            "info": {"label": "Description", "notes": "AI-generated description"},
        },
        {"id": "null_count", "type": "numeric", "info": {"label": "Null count"}},
        {"id": "cardinality", "type": "numeric", "info": {"label": "Distinct values"}},
        {"id": "min", "type": "text", "info": {"label": "Minimum"}},
        {"id": "max", "type": "text", "info": {"label": "Maximum"}},
        {"id": "mean", "type": "numeric", "info": {"label": "Mean"}},
        {"id": "median", "type": "numeric", "info": {"label": "Median"}},
        {"id": "stddev", "type": "numeric", "info": {"label": "Standard deviation"}},
        {"id": "top_values", "type": "text", "info": {"label": "Most frequent values (count)"}},
    ]
    records = []
    for column in _column_dictionary(qsv):
        stats = column["stats"]
        records.append(
            {
                "column": column["name"],
                "label": column["label"],
                "type": column["type"],
                "description": column["description"],
                "null_count": _as_number(stats.get("nullcount")),
                "cardinality": _as_number(stats.get("cardinality")),
                "min": str(stats.get("min") or ""),
                "max": str(stats.get("max") or ""),
                "mean": _as_number(stats.get("mean")),
                "median": _as_number(stats.get("q2_median")),
                "stddev": _as_number(stats.get("stddev")),
                "top_values": _top_values_text(column["top_values"]),
            }
        )
    return fields, records


def _summary_cell(value: str) -> str:
    """A text cell for a summary table, capped at :data:`_MAX_SUMMARY_CELL` chars.

    qsv reports a column's min/max/mode verbatim, so a geometry column yields
    whole WKT polygons (one 311 mode is 157 KB) that would swamp the table view.
    """
    value = _scrub_secrets(value)
    if len(value) <= _MAX_SUMMARY_CELL:
        return value
    return f"{value[:_MAX_SUMMARY_CELL]}… [truncated: {len(value):,} characters]"


def _csv_table(
    path: Path, numeric: frozenset[str] = frozenset()
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """A qsv CSV as DataStore fields and records; ``numeric`` columns are typed."""
    if not path.is_file():
        return None
    # Geometry cells are read whole, then capped.
    _raise_csv_field_limit()
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        header = [name for name in reader.fieldnames or [] if name]
        records = [
            {
                name: (
                    _as_number(row.get(name))
                    if name in numeric
                    else _summary_cell(row.get(name) or "")
                )
                for name in header
            }
            for row in reader
        ]
    fields = [{"id": name, "type": "numeric" if name in numeric else "text"} for name in header]
    return fields, records


def _qsv_artifacts(qsv: dict[str, Any]) -> list[dict[str, Any]]:
    """The resources to publish for one profile, in display order."""
    qsv = _vet_profile(qsv)
    directory = _qsv_dir(qsv)
    source = str(qsv.get("resource_name") or "the profiled file")
    caveat = "Generated with qsv; AI-written text may contain inaccuracies."
    artifacts: list[dict[str, Any]] = []

    fields, records = _dictionary_table(qsv)
    if records:
        artifacts.append(
            {
                "kind": "dictionary",
                "name": "Data dictionary (qsv)",
                "description": (
                    f"Column-level data dictionary for “{source}”: qsv-inferred types, "
                    f"AI-generated labels and descriptions (qsv describegpt), and key "
                    f"statistics. {caveat}"
                ),
                "table": (fields, records),
            }
        )
    if directory is None:
        return artifacts

    stats = _csv_table(directory / "qsv_stats.csv") if qsv["_stats_ok"] else None
    if stats:
        artifacts.append(
            {
                "kind": "stats",
                "name": "Summary statistics (qsv stats)",
                "description": (
                    f"qsv stats for “{source}”: one row per column with its type, "
                    "min/max, mean, quartiles, cardinality, null count and more."
                ),
                "table": stats,
            }
        )
    frequency = (
        _csv_table(
            directory / "qsv_frequency.csv", numeric=frozenset({"count", "percentage", "rank"})
        )
        if qsv["_frequency_ok"]
        else None
    )
    if frequency:
        artifacts.append(
            {
                "kind": "frequency",
                "name": "Frequency distribution (qsv frequency)",
                "description": (
                    f"qsv frequency for “{source}”: the most common values in each "
                    "column, with counts, percentages and rank."
                ),
                "table": frequency,
            }
        )
    describegpt = directory / "qsv_dict.json"
    if qsv["_describegpt_ok"] and describegpt.is_file():
        artifacts.append(
            {
                "kind": "describegpt",
                "name": "AI descriptions (qsv describegpt)",
                "description": (
                    f"Raw qsv describegpt output for “{source}”: the AI-generated "
                    f"dataset description, tags, and column dictionary. {caveat}"
                ),
                "file": (
                    "describegpt.json",
                    _scrub_secrets(describegpt.read_text(encoding="utf-8")).encode("utf-8"),
                    "application/json",
                ),
            }
        )
    return artifacts


_QSV_KINDS = ("dictionary", "stats", "frequency", "describegpt")


async def _publish_qsv(
    target: CKANClient,
    *,
    package_id: str,
    qsv: dict[str, Any],
    site_id: str,
    existing_resources: dict[str, dict[str, Any]] | set[str],
    apply: bool = True,
) -> dict[str, int]:
    """Create, refresh or remove one profile's resources; counts by outcome.

    Each resource records a digest of what it was built from (``qsv_digest``),
    and one whose digest matches — with its table still loaded — is left alone,
    so a rerun rewrites only the profiles that changed. A kind the profile no
    longer produces is removed: these resources are the Fair Store's own
    derivation, and one left behind would be stale or, for output that turned
    out to describe another file, wrong. ``apply=False`` only counts.
    """
    held = (
        existing_resources
        if isinstance(existing_resources, dict)
        else {resource_id: {} for resource_id in existing_resources}
    )
    counts = {"created": 0, "refreshed": 0, "unchanged": 0, "removed": 0}
    profiled_id = str(qsv.get("resource_id") or "")
    produced: set[str] = set()
    for artifact in _qsv_artifacts(qsv):
        produced.add(artifact["kind"])
        resource_id = _qsv_resource_id(profiled_id, artifact["kind"])
        resource: dict[str, Any] = {
            "id": resource_id,
            "package_id": package_id,
            "name": artifact["name"],
            "description": artifact["description"],
            "qsv_artifact": artifact["kind"],
            "qsv_profiled_resource_id": profiled_id,
            "qsv_profiled_at": str(qsv.get("onboarded_at") or ""),
            "qsv_profile_site": site_id,
        }
        if "file" in artifact:
            resource["format"] = "JSON"
            content = hashlib.sha256(artifact["file"][1]).hexdigest()
            resource["qsv_digest"] = _digest({"resource": resource, "content": content})
        else:
            fields, records = artifact["table"]
            # What datastore_create does for an inline resource, done as a
            # create of its own so a lost response is resolved by its UUID.
            resource.update(format="CSV", url="_datastore_only_resource", url_type="datastore")
            resource["qsv_digest"] = _digest(
                {"resource": resource, "fields": fields, "records": records}
            )
        existing = held.get(resource_id)
        loaded = existing is not None and (
            existing.get("url_type") == "upload"
            if "file" in artifact
            else bool(existing.get("datastore_active"))
        )
        if loaded and existing.get("qsv_digest") == resource["qsv_digest"]:
            counts["unchanged"] += 1
            if apply and "table" in artifact:
                await _ensure_table_view(target, resource_id, label=resource["name"])
            continue
        counts["created" if existing is None else "refreshed"] += 1
        if not apply:
            continue

        label = f"{artifact['kind']} for {profiled_id}"
        action = "resource_create" if existing is None else "resource_patch"
        if "file" in artifact:
            # The file and its digest land in one call.
            await _write_action(target, action, resource, label=label, upload=artifact["file"])
            continue
        # The digest is recorded only once the rows are in and counted: a run
        # that stops mid-load leaves a table that the next run must reload,
        # not one it would take for unchanged.
        digest = resource["qsv_digest"]
        await _write_action(target, action, {**resource, "qsv_digest": ""}, label=label)
        await _load_table(target, resource_id, fields, records, label=label)
        await _write_action(
            target, "resource_patch", {"id": resource_id, "qsv_digest": digest}, label=label
        )
        await _ensure_table_view(target, resource_id, label=label)

    # Without the onboarding directory nothing shows the output is gone (the
    # mirror may just be running from a checkout that lacks it), so nothing
    # is removed.
    removable = _qsv_dir(qsv) is not None
    for kind in _QSV_KINDS:
        resource_id = _qsv_resource_id(profiled_id, kind)
        if not removable or kind in produced or resource_id not in held:
            continue
        counts["removed"] += 1
        if apply:
            await _delete(
                target, "resource_delete", {"id": resource_id}, label=f"{kind} for {profiled_id}"
            )
    return counts


async def _delete(target: CKANClient, action: str, params: dict[str, Any], *, label: str) -> None:
    """A delete; an object that is already gone is fine."""
    for attempt in range(_ATTEMPTS):
        try:
            await target.call(action, params)
            return
        except CKANActionError as exc:
            if exc.not_found:
                return
            if not exc.transient or attempt == _ATTEMPTS - 1:
                raise MirrorError(f"{action} failed for {label}: {exc}") from exc
        await asyncio.sleep(2**attempt)


async def _drop_table(target: CKANClient, resource_id: str, *, label: str) -> None:
    """Delete a resource's DataStore table; one that does not exist is fine."""
    await _delete(
        target, "datastore_delete", {"resource_id": resource_id, "force": True}, label=label
    )


async def _load_table(
    target: CKANClient,
    resource_id: str,
    fields: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    """Replace a resource's DataStore table with ``records``, verified by count.

    The table is rebuilt rather than appended to, so a rerun cannot duplicate
    rows. An insert whose response is lost may still have committed, and its
    retry then adds the batch twice, so the loaded row count is checked and the
    table rebuilt until it matches.
    """
    total: Any = None
    for _ in range(_ATTEMPTS):
        await _drop_table(target, resource_id, label=label)
        await _write_action(
            target,
            "datastore_create",
            {
                "resource_id": resource_id,
                "force": True,
                "fields": fields,
                "records": records[:_DATASTORE_BATCH],
            },
            label=label,
        )
        for start in range(_DATASTORE_BATCH, len(records), _DATASTORE_BATCH):
            await _write_action(
                target,
                "datastore_upsert",
                {
                    "resource_id": resource_id,
                    "force": True,
                    "method": "insert",
                    "records": records[start : start + _DATASTORE_BATCH],
                },
                label=label,
            )
        loaded = await _read(
            target, "datastore_search", {"resource_id": resource_id, "limit": 0}, label=label
        )
        total = loaded.get("total") if isinstance(loaded, dict) else None
        if total == len(records):
            return
        print(
            f"warning: {label} holds {total} rows, expected {len(records)}; reloading",
            file=sys.stderr,
        )
    raise MirrorError(f"{label}: DataStore holds {total} rows, expected {len(records)}")


async def _ensure_table_view(target: CKANClient, resource_id: str, *, label: str) -> None:
    """Give a DataStore resource its table view.

    CKAN picks a resource's default views when the resource is created, before
    its table exists, so the table view never qualifies and has to be added
    once the rows are in. An existing one is left alone so reruns do not stack
    duplicates — which is why a failed listing or create is re-listed, never
    read as "no view yet".
    """
    for attempt in range(_ATTEMPTS):
        views = await _read(
            target, "resource_view_list", {"id": resource_id}, label=f"views of {label}"
        )
        if any(
            isinstance(view, dict) and view.get("view_type") == "datatables_view"
            for view in views or []
        ):
            return
        try:
            await target.call(
                "resource_view_create",
                {"resource_id": resource_id, "view_type": "datatables_view", "title": "Table"},
            )
            return
        except CKANActionError as exc:
            if not exc.transient or attempt == _ATTEMPTS - 1:
                raise MirrorError(f"table view failed for {label}: {exc}") from exc
        await asyncio.sleep(2**attempt)


# The digest of what the mirror last wrote to a record: an unchanged record is
# skipped on the next run instead of being patched (and reindexed) again.
_DIGEST_KEY = "mirror_digest"


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _extra(row: dict[str, Any], key: str) -> str:
    for item in row.get("extras") or []:
        if isinstance(item, dict) and item.get("key") == key:
            return str(item.get("value") or "")
    return ""


def _stamp_package(payload: dict[str, Any]) -> str:
    """Add the package's digest as an extra; extras are compared as a set."""
    extras = sorted(payload.get("extras") or [], key=lambda item: item["key"])
    digest = _digest({**payload, "extras": extras})
    payload["extras"] = [*payload.get("extras", []), {"key": _DIGEST_KEY, "value": digest}]
    return digest


def _stamp_resource(payload: dict[str, Any]) -> str:
    digest = _digest(payload)
    payload[_DIGEST_KEY] = digest
    return digest


def _package_payload(
    package: dict[str, Any],
    *,
    site_id: str,
    source_url: str,
    store_id: str,
    known_org_ids: set[str],
    qsv: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _project(package, _PACKAGE_FIELDS)
    # A patch leaves out what it does not name, so a value the source cleared
    # would survive in the Fair Store; name these as empty instead.
    for field in _CLEARABLE_PACKAGE_FIELDS:
        payload.setdefault(field, "")
    owner_org = str(package.get("owner_org") or "")
    payload["owner_org"] = owner_org if owner_org in known_org_ids else store_id
    payload["tags"] = _tag_refs(package.get("tags"))
    payload["groups"] = _group_refs(package.get("groups"))
    carried = {
        key: _extra_value(value)
        for key, value in package.items()
        if key not in _PACKAGE_MANAGED and not key.startswith("_") and not _is_empty(value)
    }
    if qsv:
        carried.update(_qsv_dataset_extras(qsv))
    for key in ("metadata_created", "metadata_modified"):
        if package.get(key):
            carried[f"mirror_source_{key}"] = str(package[key])
    if not owner_org:
        carried["mirror_source_owner_org"] = ""
    extras = _provenance_extras(
        package.get("extras"),
        site_id=site_id,
        source_url=source_url,
        source_id=_source_id(package),
        carried=carried,
    )
    # CKAN also indexes each dataset extra as a single untokenized Solr term;
    # one longer than Solr's limit fails the whole dataset with an HTTP 500.
    # Such a value cannot be stored on the dataset, so it is named instead.
    oversized = [
        item["key"] for item in extras if len(item["value"].encode("utf-8")) > _SOLR_MAX_TERM_BYTES
    ]
    if oversized:
        print(
            f"warning: {package.get('name')}: extras over Solr's term limit not stored: "
            f"{', '.join(oversized)}",
            file=sys.stderr,
        )
        extras = [item for item in extras if item["key"] not in oversized]
        extras.append({"key": "mirror_oversized_extras", "value": json.dumps(oversized)})
    payload["extras"] = extras
    if not str(payload.get("notes") or "").strip():
        payload["notes"] = "No description was provided by the source catalog."
    return payload


def _resource_payload(
    resource: dict[str, Any],
    *,
    package_id: str,
    site_id: str,
    source_url: str,
    qsv: dict[str, Any] | None,
) -> dict[str, Any]:
    """Every source resource field, minus the source CKAN's own bookkeeping."""
    payload = {
        key: value
        for key, value in resource.items()
        if value is not None
        and key not in _RESOURCE_MANAGED
        and not key.startswith(("_", "datastore_"))
    }
    payload["package_id"] = package_id
    url_type = str(resource.get("url_type") or "")
    if url_type in _LOCAL_URL_TYPES:
        payload.pop("url_type", None)
        payload["mirror_source_url_type"] = url_type
    if resource.get("datastore_active"):
        # Recorded so a consumer knows the source portal can be queried; the
        # Fair Store itself has no DataStore table for this resource.
        payload["mirror_source_datastore_active"] = True
    payload.update(
        {
            _PROVENANCE_KEYS["portal"]: site_id,
            _PROVENANCE_KEYS["url"]: source_url,
            _PROVENANCE_KEYS["id"]: _source_id(resource),
        }
    )
    if qsv:
        qsv = _vet_profile(qsv)
        columns = _column_dictionary(qsv)
        payload.update(
            {
                "qsv_description": _describegpt_prose(
                    _scrub_secrets(str(qsv.get("qsv_description") or ""))
                ),
                "qsv_tags": json.dumps(qsv.get("qsv_tags") or []),
                "data_dictionary": json.dumps(columns),
                "column_count": len(qsv.get("_header") or columns),
                "row_count": _profile_row_count(qsv),
                "qsv_onboarded_at": qsv.get("onboarded_at", ""),
            }
        )
    return payload


def _withdrawn(
    source: dict[str, Any], target: dict[str, list[dict[str, Any]]], *, site_id: str
) -> dict[str, Any]:
    """This portal's mirrored records that its source no longer publishes.

    Reported, never deleted: whether a record withdrawn upstream should leave
    the Fair Store is the operator's call. qsv resources are the Fair Store's
    own and are not counted.
    """
    source_packages = {str(package.get("id")) for package in source["packages"]}
    source_resources = {
        str(resource.get("id"))
        for package in source["packages"]
        for resource in package.get("resources") or []
    }
    ours = [package for package in target["packages"] if _mirrored_from(package) == site_id]
    datasets = sorted(
        str(package.get("name"))
        for package in ours
        if str(package.get("id")) not in source_packages
    )
    resources = [
        str(resource.get("id"))
        for package in ours
        if str(package.get("id")) in source_packages
        for resource in package.get("resources") or []
        if resource.get(_PROVENANCE_KEYS["portal"]) == site_id
        and str(resource.get("id")) not in source_resources
    ]
    return {"datasets": len(datasets), "dataset_names": datasets[:50], "resources": len(resources)}


async def mirror_catalog(
    *,
    source: CKANClient | None,
    target: CKANClient,
    site_id: str,
    site_title: str,
    source_url: str,
    apply: bool = False,
    organization: str | None = None,
    qsv_index: dict[str, Any] | None = None,
    limit: int | None = None,
    snapshot: dict[str, Any] | None = None,
    pin_names: bool = False,
) -> dict[str, Any]:
    """Plan or apply one complete, API-level metadata mirror.

    ``source`` is read as a CKAN portal unless a prepared ``snapshot`` (from
    :func:`_dcat_snapshot`) is passed instead. ``pin_names`` keeps the names
    already published for the portal's records (see :func:`_pin_target_names`);
    without it a source rename is applied.
    """
    if snapshot is not None:
        source_data = snapshot
    elif source is not None:
        source_data = await _source_snapshot(
            source, site_id=site_id, organization=organization, limit=limit
        )
    else:
        raise MirrorError("mirror_catalog needs a source client or a snapshot")
    target_data = await _target_snapshot(target)
    if pin_names:
        _pin_target_names(source_data, target_data, site_id=site_id)
    store_name = _slug(site_id, "source-store")
    if store_name in {str(row.get("name")) for row in source_data["organizations"]}:
        store_name = _slug(f"{store_name}-source", "source-store")
    store_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_url.rstrip("/")))
    evictions: list[tuple[str, str, str]] = []
    renames = _assert_no_collisions(
        source_data,
        target_data,
        store_name=store_name,
        store_id=store_id,
        site_id=site_id,
        evictions=evictions,
    )

    qsv_resources = _qsv_by_resource(qsv_index or {})
    resources = [
        resource
        for package in source_data["packages"]
        for resource in package.get("resources", []) or []
    ]
    summary: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "site": site_id,
        "source_url": source_url,
        "store_organization": store_name,
        "organizations": len(source_data["organizations"]),
        "groups": len(source_data["groups"]),
        "datasets": len(source_data["packages"]),
        "resources": len(resources),
        "qsv_resources": sum(
            1 for resource in resources if str(resource.get("id")) in qsv_resources
        ),
        "skipped_copies": source_data.get("copies", _empty_copies()),
        "renamed": renames,
        # Planned in a dry run, done when applied.
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "qsv": {"created": 0, "refreshed": 0, "unchanged": 0, "removed": 0},
    }
    if organization is None and limit is None:
        summary["withdrawn_at_source"] = _withdrawn(source_data, target_data, site_id=site_id)

    held_qsv = [
        resource
        for package in target_data["packages"]
        for resource in package.get("resources", []) or []
        if resource.get("qsv_profile_site") == site_id
    ]
    if held_qsv and not qsv_resources:
        # Every profiled dataset would lose its qsv fields to the patch.
        message = (
            f"the Fair Store holds {len(held_qsv)} qsv resources for {site_id}, but no qsv "
            "profiles were loaded: is its onboarding index missing (see --index-path)?"
        )
        if apply:
            raise MirrorError(message)
        print(f"warning: {message}", file=sys.stderr)
    # Profiles no longer in the index; their resources are reported, not removed.
    summary["qsv"]["orphaned_profiles"] = len(
        {str(resource.get("qsv_profiled_resource_id")) for resource in held_qsv}
        - set(qsv_resources)
    )

    target_group_ids = {str(row.get("id")) for row in target_data["groups"]}
    target_group_names = {str(row.get("name")) for row in target_data["groups"]}
    target_packages = {str(row.get("id")): row for row in target_data["packages"]}
    target_resources = {
        str(resource.get("id")): resource
        for package in target_data["packages"]
        for resource in package.get("resources", []) or []
    }

    async def write(action: str, payload: dict[str, Any], *, label: str) -> None:
        if apply:
            await _write_action(target, action, payload, label=label)

    target_orgs = {str(row.get("id")): row for row in target_data["organizations"]}
    deleted_orgs: set[str] = set()

    async def upsert_group(
        kind: str, payload: dict[str, Any], held: dict[str, Any] | None, digest: str
    ) -> bool:
        """Create or patch an organization or group; False when unchanged or skipped."""
        if held is not None and _extra(held, _DIGEST_KEY) == digest:
            summary["unchanged"] += 1
            return False
        payload["extras"] = [*payload["extras"], {"key": _DIGEST_KEY, "value": digest}]
        try:
            await write(
                f"{kind}_{'create' if held is None else 'patch'}", payload, label=payload["name"]
            )
        except DeletedInTarget:
            # An admin removed it from the Fair Store; that decision stands.
            summary.setdefault("skipped_deleted_in_target", []).append(payload["name"])
            if kind == "organization":
                deleted_orgs.add(str(payload["id"]))
            return False
        summary["created" if held is None else "updated"] += 1
        return True

    def group_digest(payload: dict[str, Any], parents: Any = None) -> str:
        extras = sorted(payload.get("extras") or [], key=lambda item: item["key"])
        return _digest({**payload, "extras": extras, "parents": parents})

    for kind, holder_id, evicted in evictions:
        await write(
            "organization_patch" if kind == "organization" else "package_patch",
            {"id": holder_id, "name": evicted},
            label=f"{kind} {holder_id} (withdrawn at source)",
        )

    store_payload = {
        "id": store_id,
        "name": store_name,
        "title": site_title,
        "description": f"Datasets mirrored from {source_url.rstrip('/')}",
        "extras": _provenance_extras(
            [], site_id=site_id, source_url=source_url, source_id=store_id
        ),
    }
    await upsert_group(
        "organization", store_payload, target_orgs.get(store_id), group_digest(store_payload)
    )

    # CKAN's no_loops validator resolves both child and parent rows; combining a
    # caller-supplied child UUID with groups during create trips that validator,
    # so the hierarchy is applied only after every source organization exists.
    # A parent that is not being mirrored cannot be referenced, so an
    # organization whose parents are all outside the snapshot hangs off the root
    # — except in an --organization run, which leaves an existing one's parent.
    source_org_names = {str(row.get("name")) for row in source_data["organizations"]}
    hierarchy: list[tuple[dict[str, Any], list[dict[str, str]]]] = []
    for organization_row in source_data["organizations"]:
        org_id = str(organization_row.get("id", ""))
        payload = _with_image(_project(organization_row, _ORGANIZATION_FIELDS), organization_row)
        payload["extras"] = _provenance_extras(
            organization_row.get("extras"),
            site_id=site_id,
            source_url=source_url,
            source_id=_source_id(organization_row),
        )
        held = target_orgs.get(org_id)
        parents: list[dict[str, str]] | None = [
            parent
            for parent in _group_refs(organization_row.get("groups"))
            if parent["name"] in source_org_names
        ] or [{"name": store_name, "capacity": "parent"}]
        if organization is not None and held is not None:
            parents = None
        if await upsert_group("organization", payload, held, group_digest(payload, parents)):
            if parents is not None:
                hierarchy.append((organization_row, parents))

    deleted_names = {
        str(row.get("name"))
        for row in source_data["organizations"]
        if str(row.get("id")) in deleted_orgs
    }
    for organization_row, parents in hierarchy:
        parents = [parent for parent in parents if parent["name"] not in deleted_names] or [
            {"name": store_name, "capacity": "parent"}
        ]
        await write(
            "organization_patch",
            {"id": str(organization_row.get("id", "")), "groups": parents},
            label=f"{organization_row.get('name')} hierarchy",
        )

    target_groups = {str(row.get("id")): row for row in target_data["groups"]}
    for group in source_data["groups"]:
        payload = _with_image(_project(group, _GROUP_FIELDS), group)
        group_id = str(group.get("id", ""))
        if str(group.get("name")) in target_group_names and group_id not in target_group_ids:
            continue
        payload["extras"] = _provenance_extras(
            group.get("extras"), site_id=site_id, source_url=source_url, source_id=_source_id(group)
        )
        await upsert_group("group", payload, target_groups.get(group_id), group_digest(payload))

    # Datasets of an organization an admin deleted here hang off the root.
    known_org_ids = {str(row.get("id")) for row in source_data["organizations"]} - deleted_orgs
    total = len(source_data["packages"])
    for position, package in enumerate(source_data["packages"], start=1):
        profiles = [
            qsv_resources[str(resource.get("id"))]
            for resource in package.get("resources", []) or []
            if str(resource.get("id")) in qsv_resources
        ]
        payload = _package_payload(
            package,
            site_id=site_id,
            source_url=source_url,
            store_id=store_id,
            known_org_ids=known_org_ids,
            qsv=profiles[0] if profiles else None,
        )
        digest = _stamp_package(payload)
        package_id = str(payload.get("id", ""))
        resource_payloads = [
            _resource_payload(
                resource,
                package_id=package_id,
                site_id=site_id,
                source_url=source_url,
                qsv=qsv_resources.get(str(resource.get("id", ""))),
            )
            for resource in package.get("resources", []) or []
        ]
        for resource_payload in resource_payloads:
            _stamp_resource(resource_payload)
        if apply and (position % 50 == 0 or position == total):
            print(f"[{site_id}] {position}/{total} datasets", file=sys.stderr, flush=True)

        existing = target_packages.get(package_id)
        if existing is None:
            # One call creates the dataset and its resources with their
            # preserved UUIDs; a separate create per resource costs a
            # reindex each and made a portal-sized mirror several times slower.
            try:
                await write(
                    "package_create",
                    {**payload, "resources": resource_payloads},
                    label=str(package.get("name")),
                )
            except DeletedInTarget:
                # An admin removed it from the Fair Store; that decision stands.
                summary.setdefault("skipped_deleted_in_target", []).append(package.get("name"))
                continue
            summary["created"] += 1 + len(resource_payloads)
        else:
            if _extra(existing, _DIGEST_KEY) == digest:
                summary["unchanged"] += 1
            else:
                await write("package_patch", payload, label=str(package.get("name")))
                summary["updated"] += 1
            for resource_payload in resource_payloads:
                held = target_resources.get(resource_payload["id"])
                if held is not None and held.get(_DIGEST_KEY) == resource_payload[_DIGEST_KEY]:
                    summary["unchanged"] += 1
                    continue
                # Replaced rather than patched: a field the source dropped
                # must go here too (a patch would keep it).
                await write(
                    "resource_create" if held is None else "resource_update",
                    resource_payload,
                    label=str(resource_payload["id"]),
                )
                summary["created" if held is None else "updated"] += 1

        for profile in profiles:
            counts = await _publish_qsv(
                target,
                package_id=package_id,
                qsv=profile,
                site_id=site_id,
                existing_resources=target_resources,
                apply=apply,
            )
            for outcome, count in counts.items():
                summary["qsv"][outcome] += count

    qsv_counts = summary["qsv"]
    summary["qsv_artifacts"] = (
        qsv_counts["created"] + qsv_counts["refreshed"] + qsv_counts["unchanged"]
    )
    return summary


async def _mirror_site(
    site_id: str, site: dict[str, Any], args: argparse.Namespace, api_key: str
) -> dict[str, Any]:
    # Neither client may fall back to CKAN_API_KEY: the target is written with
    # the mirror's own token, and a source portal must never receive a key.
    target = CKANClient(ckan_url=args.target_url, api_key=api_key or None, use_default_key=False)
    try:
        common = {
            "target": target,
            "site_id": site_id,
            "site_title": site["name"],
            "source_url": site["url"],
            "apply": args.apply,
        }
        if normalize_portal_type(site.get("portal_type")) == PORTAL_TYPE_DCAT:
            if args.organization:
                raise MirrorError("--organization applies to CKAN source portals only")
            dcat = DCATClient(site["url"], site.get("catalog_url"))
            try:
                catalog = await dcat.fetch_raw_catalog()
            except (httpx.HTTPError, ValueError) as exc:
                raise MirrorError(f"DCAT catalog for {site_id} could not be read: {exc}") from exc
            finally:
                await dcat.close()
            socrata = (
                {} if args.no_enrich else await _socrata_metadata(catalog, required=args.apply)
            )
            snapshot = _dcat_snapshot(
                catalog,
                site_title=site["name"],
                source_url=site["url"],
                socrata=socrata,
                limit=args.limit,
            )
            summary = await mirror_catalog(
                source=None,
                snapshot=snapshot,
                pin_names=True,
                qsv_index=_dcat_qsv_index(_load_index(site_id, args.index_path), snapshot),
                **common,
            )
            summary["socrata_enriched"] = sum(
                1
                for package in snapshot["packages"]
                if any(extra["key"] == "socrata_id" for extra in package["extras"])
            )
            return summary

        source = CKANClient(ckan_url=site["url"], use_default_key=False)
        try:
            return await mirror_catalog(
                source=source,
                organization=args.organization,
                qsv_index=_load_index(site_id, args.index_path),
                limit=args.limit,
                **common,
            )
        finally:
            await source.close()
    finally:
        await target.close()


def _same_portal(url: str, other: str) -> bool:
    def key(value: str) -> tuple[str, str]:
        parsed = urlparse(value.strip())
        host = (parsed.hostname or "").lower().removeprefix("www.")
        return host, parsed.path.rstrip("/")

    return key(url) == key(other)


async def _main(args: argparse.Namespace) -> dict[str, Any] | list[dict[str, Any]]:
    if args.site == "all":
        if args.organization or args.index_path:
            raise MirrorError("--organization and --index-path need a single --site")
        sites = [(str(site["id"]), site) for site in list_sites()]
    else:
        site = get_site(args.site)
        if not site:
            raise MirrorError(f"Unknown site {args.site!r}; add it to ckan_sites.json first")
        sites = [(args.site, site)]

    api_key = os.environ.get(args.api_key_env, "")
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if args.apply and not api_key:
        raise MirrorError(
            f"Set {args.api_key_env} or --api-key-file when using --apply; a sysadmin token is required"
        )

    if args.site != "all" and _same_portal(sites[0][1]["url"], args.target_url):
        raise MirrorError(f"{args.site} is the target portal; it cannot be mirrored into itself")

    summaries: list[dict[str, Any]] = []
    for site_id, site in sites:
        if _same_portal(site["url"], args.target_url):
            # data.dathere.com is both a registered portal and a Fair Store.
            summaries.append({"site": site_id, "skipped": "this portal is the target"})
            continue
        try:
            summaries.append(await _mirror_site(site_id, site, args, api_key))
        except Exception as exc:
            if args.site != "all":
                raise
            # One portal's outage must not stop the others from refreshing;
            # the failure is reported and the exit status is non-zero.
            print(f"error: {site_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            summaries.append({"site": site_id, "error": f"{type(exc).__name__}: {exc}"})
    return summaries if args.site == "all" else summaries[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror CKAN/DCAT metadata and qsv results")
    parser.add_argument(
        "--site",
        default="wprdc",
        help="source site ID from ckan_sites.json, or 'all' for every registered portal",
    )
    # Deliberately not CKAN_URL/CKAN_API_KEY: those are the app's portal
    # settings, and in the app container CKAN_URL is data.dathere.com.
    parser.add_argument(
        "--target-url", default=os.environ.get("FAIRSTORE_URL", "http://localhost:5001")
    )
    parser.add_argument("--organization", help="optionally mirror only one source organization")
    parser.add_argument("--index-path", help="override the qsv index.json path")
    parser.add_argument("--limit", type=int, help="limit datasets for a smoke test")
    parser.add_argument("--apply", action="store_true", help="write after collision preflight")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="skip Socrata Discovery API enrichment (owning agency, column list) for DCAT portals",
    )
    parser.add_argument("--api-key-env", default="FAIRSTORE_API_KEY")
    parser.add_argument("--api-key-file", help="read the target sysadmin token from a file")
    args = parser.parse_args()
    # Development logging is DEBUG on stdout, where this command prints its
    # JSON summary; a line per HTTP request would bury it.
    for noisy in ("asyncio", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        summary = asyncio.run(_main(args))
    except MirrorError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if isinstance(summary, list) and any("error" in item for item in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()
