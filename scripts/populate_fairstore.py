#!/usr/bin/env python
"""Mirror a source CKAN catalog into the Fair Store.

The mirror keeps source organization, group, dataset, and resource names and
UUIDs. A top-level organization represents the source portal; source root
organizations are attached beneath it through ckanext-hierarchy. Available
qsv profiling metadata from ``data/ckan_onboard/<site>/index.json`` is added to
the corresponding resource without replacing the source description.

The command is read-only unless ``--apply`` is supplied. Before any write it
checks the whole source snapshot for name and UUID collisions. Re-running is
idempotent: existing objects are patched by their preserved source UUIDs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from data_concierge.data_layer.connectors.ckan import CKANClient  # noqa: E402
from data_concierge.data_layer.onboard_index import _scrub_secrets  # noqa: E402
from data_concierge.gateway.ckan_sites import get_site  # noqa: E402

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
_RESOURCE_FIELDS = (
    "id",
    "url",
    "description",
    "format",
    "hash",
    "name",
    "resource_type",
    "url_type",
    "mimetype",
    "mimetype_inner",
    "cache_url",
    "size",
    "created",
    "last_modified",
    "cache_last_updated",
)


class MirrorError(RuntimeError):
    """Raised when a collision or failed CKAN action makes a mirror unsafe."""


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
        str(resource["resource_id"]): resource
        for dataset in index.get("datasets", []) or []
        for resource in dataset.get("resources", []) or []
        if resource.get("resource_id")
    }


def _project(source: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {
        field: source[field] for field in fields if field in source and source[field] is not None
    }


def _provenance_extras(
    extras: list[dict[str, Any]] | None,
    *,
    site_id: str,
    source_url: str,
    source_id: str,
) -> list[dict[str, str]]:
    """Preserve source extras and add stable mirror identity fields."""
    values = {
        str(item.get("key")): str(item.get("value", ""))
        for item in (extras or [])
        if item.get("key")
    }
    values.update(
        {
            _PROVENANCE_KEYS["portal"]: site_id,
            _PROVENANCE_KEYS["url"]: source_url,
            _PROVENANCE_KEYS["id"]: source_id,
        }
    )
    return [{"key": key, "value": value} for key, value in values.items()]


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


async def _all_packages(client: CKANClient) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    start = 0
    while True:
        page = await client.action(
            "package_search", {"q": "*:*", "rows": 1000, "start": start, "sort": "name asc"}
        )
        batch = page.get("results", []) if isinstance(page, dict) else []
        packages.extend(batch)
        start += len(batch)
        if not batch or start >= int(page.get("count", 0)):
            return packages


async def _all_group_rows(client: CKANClient, action_name: str) -> list[dict[str, Any]]:
    """Read every organization or group despite portal-side page-size caps."""
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    offset = 0
    while True:
        result = await client.action(
            action_name,
            {"all_fields": True, "limit": 1000, "offset": offset},
        )
        batch = list(result) if isinstance(result, list) else []
        if not batch:
            return rows
        new_rows = [row for row in batch if str(row.get("id")) not in seen_ids]
        if not new_rows:
            return rows
        rows.extend(new_rows)
        seen_ids.update(str(row.get("id")) for row in new_rows)
        offset += len(batch)


async def _source_snapshot(
    client: CKANClient, *, organization: str | None = None, limit: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    organization_rows = await _all_group_rows(client, "organization_list")
    organizations: list[dict[str, Any]] = []
    for row in organization_rows if isinstance(organization_rows, list) else []:
        if organization and row.get("name") != organization:
            continue
        full = await client.action(
            "organization_show",
            {
                "id": row["id"],
                "include_datasets": False,
                "include_users": False,
                "include_groups": True,
                "include_tags": True,
            },
        )
        organizations.append(full or row)

    groups = await _all_group_rows(client, "group_list")

    packages = await _all_packages(client)
    if organization:
        org_ids = {str(item.get("id")) for item in organizations}
        packages = [
            package
            for package in packages
            if package.get("organization", {}).get("name") == organization
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
    return {"organizations": organizations, "groups": groups, "packages": packages}


async def _target_snapshot(client: CKANClient) -> dict[str, list[dict[str, Any]]]:
    organizations = await _all_group_rows(client, "organization_list")
    groups = await _all_group_rows(client, "group_list")
    packages = await _all_packages(client)
    return {
        "organizations": list(organizations) if isinstance(organizations, list) else [],
        "groups": list(groups) if isinstance(groups, list) else [],
        "packages": packages,
    }


def _assert_no_collisions(
    source: dict[str, list[dict[str, Any]]],
    target: dict[str, list[dict[str, Any]]],
    *,
    store_name: str,
    store_id: str,
) -> None:
    collisions: list[str] = []

    def check(
        kind: str,
        incoming: list[dict[str, Any]],
        existing: list[dict[str, Any]],
        *,
        allow_shared_name: bool = False,
    ) -> None:
        by_name = {str(row.get("name")): row for row in existing if row.get("name")}
        by_id = {str(row.get("id")): row for row in existing if row.get("id")}
        for row in incoming:
            name, row_id = str(row.get("name", "")), str(row.get("id", ""))
            if not allow_shared_name and name in by_name and str(by_name[name].get("id")) != row_id:
                collisions.append(f"{kind} name {name!r} already has a different UUID")
            if row_id in by_id and str(by_id[row_id].get("name")) != name:
                collisions.append(f"{kind} UUID {row_id!r} already has a different name")

    root_existing = next(
        (row for row in target["organizations"] if row.get("name") == store_name), None
    )
    if root_existing and str(root_existing.get("id")) != store_id:
        collisions.append(f"source-store organization {store_name!r} already exists")
    check("organization", source["organizations"], target["organizations"])
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


async def _write_action(
    client: CKANClient, action: str, payload: dict[str, Any], *, label: str
) -> dict[str, Any]:
    result = await client.action(action, payload)
    if not result:
        raise MirrorError(f"{action} failed for {label}")
    return result


def _resource_payload(
    resource: dict[str, Any],
    *,
    package_id: str,
    site_id: str,
    source_url: str,
    qsv: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = _project(resource, _RESOURCE_FIELDS)
    payload["package_id"] = package_id
    payload.update(
        {
            _PROVENANCE_KEYS["portal"]: site_id,
            _PROVENANCE_KEYS["url"]: source_url,
            _PROVENANCE_KEYS["id"]: str(resource.get("id", "")),
        }
    )
    # Preserve extension-owned spatial metadata when the destination has the
    # same extension, but never claim that a DataStore table was copied.
    for key, value in resource.items():
        if key.startswith("dataspatial_") and value is not None:
            payload[key] = value
    if qsv:
        columns = _column_dictionary(qsv)
        payload.update(
            {
                "qsv_description": _scrub_secrets(qsv.get("qsv_description", "")),
                "qsv_tags": json.dumps(qsv.get("qsv_tags") or []),
                "data_dictionary": json.dumps(columns),
                "column_count": len(columns),
                "row_count": qsv.get("row_count", 0),
                "qsv_onboarded_at": qsv.get("onboarded_at", ""),
            }
        )
    return payload


async def mirror_catalog(
    *,
    source: CKANClient,
    target: CKANClient,
    site_id: str,
    site_title: str,
    source_url: str,
    apply: bool = False,
    organization: str | None = None,
    qsv_index: dict[str, Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Plan or apply one complete, API-level CKAN metadata mirror."""
    source_data = await _source_snapshot(source, organization=organization, limit=limit)
    target_data = await _target_snapshot(target)
    store_name = _slug(site_id, "source-store")
    store_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_url.rstrip("/")))
    _assert_no_collisions(source_data, target_data, store_name=store_name, store_id=store_id)

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
        "created": 0,
        "updated": 0,
    }
    if not apply:
        return summary

    target_org_ids = {str(row.get("id")) for row in target_data["organizations"]}
    target_group_ids = {str(row.get("id")) for row in target_data["groups"]}
    target_group_names = {str(row.get("name")) for row in target_data["groups"]}
    target_package_ids = {str(row.get("id")) for row in target_data["packages"]}
    target_resource_ids = {
        str(resource.get("id"))
        for package in target_data["packages"]
        for resource in package.get("resources", []) or []
    }

    store_payload = {
        "id": store_id,
        "name": store_name,
        "title": site_title,
        "description": f"Datasets mirrored from {source_url.rstrip('/')}",
        "extras": _provenance_extras(
            [], site_id=site_id, source_url=source_url, source_id=store_id
        ),
    }
    store_exists = store_id in target_org_ids
    await _write_action(
        target,
        "organization_patch" if store_exists else "organization_create",
        store_payload,
        label=store_name,
    )
    summary["updated" if store_exists else "created"] += 1

    source_org_names = {str(row.get("name")) for row in source_data["organizations"]}
    for organization_row in source_data["organizations"]:
        payload = _project(organization_row, _ORGANIZATION_FIELDS)
        source_id = str(organization_row.get("id", ""))
        payload["extras"] = _provenance_extras(
            organization_row.get("extras"),
            site_id=site_id,
            source_url=source_url,
            source_id=source_id,
        )
        exists = source_id in target_org_ids
        await _write_action(
            target,
            "organization_patch" if exists else "organization_create",
            payload,
            label=str(organization_row.get("name")),
        )
        summary["updated" if exists else "created"] += 1

    # Apply hierarchy only after every source organization exists. CKAN's
    # no_loops validator resolves both child and parent rows; combining a
    # caller-supplied child UUID with groups during create trips that validator.
    for organization_row in source_data["organizations"]:
        parents = _group_refs(organization_row.get("groups"))
        if not any(parent["name"] in source_org_names for parent in parents):
            parents.append({"name": store_name, "capacity": "parent"})
        await _write_action(
            target,
            "organization_patch",
            {"id": str(organization_row.get("id", "")), "groups": parents},
            label=f"{organization_row.get('name')} hierarchy",
        )

    for group in source_data["groups"]:
        payload = _project(group, _GROUP_FIELDS)
        source_id = str(group.get("id", ""))
        if str(group.get("name")) in target_group_names and source_id not in target_group_ids:
            continue
        payload["extras"] = _provenance_extras(
            group.get("extras"), site_id=site_id, source_url=source_url, source_id=source_id
        )
        exists = source_id in target_group_ids
        await _write_action(
            target,
            "group_patch" if exists else "group_create",
            payload,
            label=str(group.get("name")),
        )
        summary["updated" if exists else "created"] += 1

    known_org_ids = {str(row.get("id")) for row in source_data["organizations"]}
    for package in source_data["packages"]:
        payload = _project(package, _PACKAGE_FIELDS)
        package_id = str(package.get("id", ""))
        owner_org = str(package.get("owner_org") or "")
        payload["owner_org"] = owner_org if owner_org in known_org_ids else store_id
        payload["tags"] = [
            {
                "name": str(tag["name"]),
                **({"vocabulary_id": tag["vocabulary_id"]} if tag.get("vocabulary_id") else {}),
            }
            for tag in package.get("tags", []) or []
            if tag.get("name")
        ]
        payload["groups"] = _group_refs(package.get("groups"))
        payload["extras"] = _provenance_extras(
            package.get("extras"),
            site_id=site_id,
            source_url=source_url,
            source_id=package_id,
        )
        if not str(payload.get("notes") or "").strip():
            payload["notes"] = "No description was provided by the source catalog."
        if not owner_org:
            payload["extras"].append({"key": "mirror_source_owner_org", "value": ""})
        exists = package_id in target_package_ids
        await _write_action(
            target,
            "package_patch" if exists else "package_create",
            payload,
            label=str(package.get("name")),
        )
        summary["updated" if exists else "created"] += 1

        for resource in package.get("resources", []) or []:
            resource_id = str(resource.get("id", ""))
            resource_payload = _resource_payload(
                resource,
                package_id=package_id,
                site_id=site_id,
                source_url=source_url,
                qsv=qsv_resources.get(resource_id),
            )
            resource_exists = resource_id in target_resource_ids
            await _write_action(
                target,
                "resource_patch" if resource_exists else "resource_create",
                resource_payload,
                label=resource_id,
            )
            summary["updated" if resource_exists else "created"] += 1

    return summary


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    site = get_site(args.site)
    if not site:
        raise MirrorError(f"Unknown site {args.site!r}; add it to ckan_sites.json first")
    if site.get("portal_type", "ckan") != "ckan":
        raise MirrorError("Exact organization mirroring currently requires a CKAN source portal")

    api_key = os.environ.get(args.api_key_env, "")
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if args.apply and not api_key:
        raise MirrorError(
            f"Set {args.api_key_env} or --api-key-file when using --apply; a sysadmin token is required"
        )

    source = CKANClient(ckan_url=site["url"])
    target = CKANClient(ckan_url=args.target_url, api_key=api_key or None)
    try:
        return await mirror_catalog(
            source=source,
            target=target,
            site_id=args.site,
            site_title=site["name"],
            source_url=site["url"],
            apply=args.apply,
            organization=args.organization,
            qsv_index=_load_index(args.site, args.index_path),
            limit=args.limit,
        )
    finally:
        await source.close()
        await target.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror CKAN metadata and qsv results")
    parser.add_argument("--site", default="wprdc", help="source site ID from ckan_sites.json")
    parser.add_argument("--target-url", default=os.environ.get("CKAN_URL", "http://localhost:5001"))
    parser.add_argument("--organization", help="optionally mirror only one source organization")
    parser.add_argument("--index-path", help="override the qsv index.json path")
    parser.add_argument("--limit", type=int, help="limit datasets for a smoke test")
    parser.add_argument("--apply", action="store_true", help="write after collision preflight")
    parser.add_argument("--api-key-env", default="CKAN_API_KEY")
    parser.add_argument("--api-key-file", help="read the target sysadmin token from a file")
    args = parser.parse_args()
    try:
        summary = asyncio.run(_main(args))
    except MirrorError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
