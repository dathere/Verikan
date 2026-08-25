#!/usr/bin/env python
"""Publish onboarded data dictionaries into the Fair Store (CKAN).

Issue #133. ``scripts/onboard_ckan.py`` builds a rich per-column dictionary
(qsv stats, labels, types, top values) and stores it as ``index.json``. This
takes that dictionary and writes it into the containerized CKAN instance as
real dataset and resource records, so the structured metadata is queryable
through CKAN's deterministic API rather than only through the in-process
search index.

What lands in CKAN, per resource:

* a **dataset** (CKAN package) carrying the dataset title, description,
  organization and tags;
* a **resource** under it carrying the qsv description and format;
* the **column-level data dictionary** — every column's label, type, stats
  and top values — as a JSON blob in the resource's ``extras`` under
  ``data_dictionary``, plus flat ``column_count`` / ``row_count`` extras for
  filtering.

The vector store keeps the fuzzy half and points back at these CKAN ids; this
script never touches it. Re-running is idempotent: a dataset is matched by its
slug and updated in place rather than duplicated.

Usage::

    # Boot the stack and create a token first:
    #   docker compose --profile fairstore up -d
    #   docker compose exec fairstore ckan -c /srv/app/ckan.ini sysadmin add admin
    # then generate an API token in the CKAN UI and:
    python -m scripts.populate_fairstore \\
        --ckan-url http://localhost:5001 \\
        --api-key <token> \\
        --site wprdc \\
        --org city-of-pittsburgh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from data_concierge.core.logging import get_logger  # noqa: E402
from data_concierge.data_layer.connectors.ckan import CKANClient  # noqa: E402
from data_concierge.data_layer.onboard_index import _scrub_secrets  # noqa: E402

logger = get_logger("populate_fairstore")

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _slug(text: str, fallback: str) -> str:
    """CKAN dataset names must be lowercase slugs, 2-100 chars."""
    s = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    if len(s) < 2:
        s = fallback
    return s[:100]


def _load_index(site: str) -> dict[str, Any]:
    """Read the onboarded index.json for a site (local path)."""
    path = _PROJECT_ROOT / "data" / "ckan_onboard" / site / "index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No onboarded index at {path}. Run scripts/onboard_ckan.py for {site!r} first."
        )
    return json.loads(path.read_text())


def _column_dictionary(resource: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact, publishable dictionary for one resource's columns."""
    columns = []
    for col in resource.get("columns", []) or []:
        stats = col.get("stats", {}) or {}
        columns.append(
            {
                "name": col.get("name", ""),
                "label": col.get("qsv_label") or col.get("ckan_info", {}).get("label", ""),
                "description": _scrub_secrets(col.get("qsv_description", "")),
                "type": col.get("qsv_type") or col.get("ckan_type", ""),
                "stats": {
                    k: stats.get(k)
                    for k in ("type", "min", "max", "mean", "q2_median", "stddev",
                              "nullcount", "cardinality")
                    if stats.get(k) not in (None, "")
                },
                "top_values": (col.get("top_values") or [])[:10],
            }
        )
    return columns


async def _ensure_org(ckan: CKANClient, org_slug: str, org_title: str) -> None:
    """Create the organization if it does not already exist."""
    existing = await ckan.action("organization_show", {"id": org_slug})
    if existing:
        return
    await ckan.action("organization_create", {"name": org_slug, "title": org_title})
    logger.info("Created organization", org=org_slug)


async def _upsert_dataset(
    ckan: CKANClient, name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Create the dataset, or update it in place if the slug already exists."""
    existing = await ckan.action("package_show", {"id": name})
    if existing:
        payload["id"] = existing["id"]
        return await ckan.action("package_update", payload)
    return await ckan.action("package_create", payload)


async def populate(
    ckan_url: str, api_key: str, site: str, org_slug: str, org_title: str, dry_run: bool
) -> dict[str, int]:
    index = _load_index(site)
    datasets = index.get("datasets", []) or []
    logger.info("Loaded onboarded index", site=site, datasets=len(datasets))

    counts = {"datasets": 0, "resources": 0, "skipped": 0}

    if dry_run:
        for ds in datasets:
            resources = [
                r for r in ds.get("resources", []) if r.get("status") in ("ok", None)
            ]
            print(
                f"  would publish: {_slug(ds.get('dataset_title', ''), ds.get('dataset_id', 'x'))} "
                f"({len(resources)} resources, "
                f"{sum(len(r.get('columns') or []) for r in resources)} columns)"
            )
            counts["datasets"] += 1
            counts["resources"] += len(resources)
        return counts

    ckan = CKANClient(ckan_url=ckan_url, api_key=api_key)
    try:
        await _ensure_org(ckan, org_slug, org_title)

        for ds in datasets:
            resources = [
                r for r in ds.get("resources", []) if r.get("status") in ("ok", None)
            ]
            if not resources:
                counts["skipped"] += 1
                continue

            name = _slug(ds.get("dataset_title", ""), ds.get("dataset_id", "dataset"))
            tags = [{"name": _slug(t, "tag")} for t in (ds.get("tags") or []) if t][:20]

            pkg = await _upsert_dataset(
                ckan,
                name,
                {
                    "name": name,
                    "title": ds.get("dataset_title", name),
                    "notes": _scrub_secrets(ds.get("dataset_description", "")),
                    "owner_org": org_slug,
                    "tags": tags,
                    "extras": [
                        {"key": "source_dataset_id", "value": str(ds.get("dataset_id", ""))},
                        {"key": "onboarded_from", "value": site},
                    ],
                },
            )
            if not pkg:
                logger.warning("Dataset upsert failed", dataset=name)
                counts["skipped"] += 1
                continue
            counts["datasets"] += 1

            for res in resources:
                columns = _column_dictionary(res)
                await ckan.action(
                    "resource_create",
                    {
                        "package_id": pkg["id"],
                        "name": res.get("resource_name", "resource"),
                        "description": _scrub_secrets(res.get("qsv_description", "")),
                        "format": res.get("format", "CSV"),
                        "data_dictionary": json.dumps(columns),
                        "column_count": len(columns),
                        "row_count": res.get("row_count", 0),
                    },
                )
                counts["resources"] += 1

        return counts
    finally:
        await ckan.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish onboarded dictionaries into CKAN")
    ap.add_argument("--ckan-url", default="http://localhost:5001")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--site", default="wprdc")
    ap.add_argument("--org", default="city-of-pittsburgh", help="organization slug")
    ap.add_argument("--org-title", default="City of Pittsburgh")
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would be published, write nothing"
    )
    args = ap.parse_args()

    if not args.dry_run and not args.api_key:
        ap.error("--api-key is required unless --dry-run")

    counts = asyncio.run(
        populate(
            args.ckan_url, args.api_key, args.site, args.org, args.org_title, args.dry_run
        )
    )
    print(
        f"\nFair Store populate ({'dry run' if args.dry_run else 'done'}): "
        f"{counts['datasets']} datasets, {counts['resources']} resources, "
        f"{counts['skipped']} skipped."
    )


if __name__ == "__main__":
    main()
