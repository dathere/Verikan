#!/usr/bin/env python3
"""Onboard a DCAT portal: download CSV distributions, profile with qsv, index them.

The DCAT counterpart to ``onboard_ckan.py``.  Only dataset *discovery* differs:
a DCAT portal publishes a catalog document (``/data.json``) instead of a live
action API, and it carries no field-level data dictionary — so every column is
described from qsv output alone.

Everything downstream is shared with the CKAN script via
``data_layer.qsv_profiling``.

Usage::

    # Profile the whole portal (qsv stats + frequency, no LLM needed)
    python scripts/onboard_dcat.py --site-id data-pa-gov --skip-qsv

    # Add AI column descriptions, then push to Pinecone
    python scripts/onboard_dcat.py --site-id data-pa-gov \\
        --openrouter-api-key sk-or-... --pinecone

    # Try a subset first
    python scripts/onboard_dcat.py --site-id data-pa-gov \\
        --dataset-filter opioid --limit 5 --skip-qsv

    # See what would be sent to Pinecone without writing
    python scripts/onboard_dcat.py --site-id data-pa-gov \\
        --pinecone --pinecone-dry-run --rebuild-index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_concierge.data_layer.connectors.dcat import (  # noqa: E402
    DCATClient,
    DCATDataset,
)
from data_concierge.data_layer.qsv_profiling import (  # noqa: E402
    _extract_qsv_description,
    _extract_qsv_tags,
    _now_iso,
    merge_columns,
    run_qsv_describegpt,
    run_qsv_frequency,
    run_qsv_stats,
    sync_to_storage,
)
from data_concierge.gateway.ckan_sites import get_site, normalize_portal_type  # noqa: E402

# A DCAT distribution is a static file with no row limit, and a state portal
# publishes some very large ones.  Cap what we pull so one dataset cannot fill
# the disk; the cap is reported in meta.json so a truncated profile is never
# mistaken for a complete one.
DEFAULT_MAX_MB = 200


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "unnamed"


async def download_distribution(
    client: DCATClient,
    url: str,
    dest: Path,
    max_bytes: int,
) -> tuple[int, bool]:
    """Stream a distribution to disk, stopping at ``max_bytes``.

    Returns ``(bytes_written, truncated)``.  Writing straight to disk (rather
    than through the connector's row-level loader) keeps whole files available
    for qsv, which needs the real CSV to compute stats.
    """
    safe_url = client._safe_url(url)
    http = await client._http()
    dest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    truncated = False
    with open(dest, "wb") as fh:
        async with http.stream("GET", safe_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if written + len(chunk) > max_bytes:
                    fh.write(chunk[: max_bytes - written])
                    written = max_bytes
                    truncated = True
                    break
                fh.write(chunk)
                written += len(chunk)

    if truncated:
        # A byte-truncated CSV usually ends mid-row; drop the partial line so
        # qsv sees a well-formed file.
        raw = dest.read_bytes()
        cut = raw.rfind(b"\n")
        if cut > 0:
            dest.write_bytes(raw[:cut])
            written = cut

    return written, truncated


async def process_dataset(
    client: DCATClient,
    dataset: DCATDataset,
    base_dir: Path,
    api_key: str | None,
    skip_download: bool,
    skip_qsv: bool,
    max_bytes: int,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Download one dataset's first CSV distribution and profile it with qsv."""
    async with semaphore:
        tabular = dataset.tabular_distributions
        if not tabular:
            return None

        dist = tabular[0]
        ds_slug = _slugify(dataset.id or dataset.title)
        res_slug = _slugify(dist.title or dataset.title or dataset.id)

        resource_dir = base_dir / ds_slug
        csv_path = resource_dir / f"{res_slug}.csv"
        qsv_dict_path = resource_dir / "qsv_dict.json"
        stats_path = resource_dir / "qsv_stats.csv"
        freq_path = resource_dir / "qsv_frequency.csv"
        meta_path = resource_dir / "meta.json"

        print(f"\n  {dataset.id}  {dataset.title[:70]}")

        result: dict = {
            "dataset_id": dataset.id,
            "dataset_title": dataset.title,
            "dataset_description": dataset.description,
            "organization": dataset.publisher,
            "dataset_tags": list(dataset.keywords or []),
            "resource_id": dataset.id,
            "resource_name": dist.title or dataset.title,
            "format": dist.format or "CSV",
            "local_path": str(csv_path),
            "source_url": dist.best_url,
            "landing_page": dataset.landing_page,
            "modified": dataset.modified,
            "license": dataset.license,
            "themes": list(dataset.themes or []),
            "columns": [],
            "qsv_description": None,
            "qsv_tags": None,
            "status": "ok",
            "error": None,
            "truncated_download": False,
            "onboarded_at": _now_iso(),
        }

        # Step 1: download the distribution
        if skip_download and csv_path.exists():
            size = csv_path.stat().st_size
            print(f"    Skipping download (exists): {csv_path.name} ({size:,} bytes)")
        else:
            try:
                print(f"    Downloading {dist.best_url}")
                size, truncated = await download_distribution(
                    client, dist.best_url, csv_path, max_bytes
                )
                result["truncated_download"] = truncated
                note = " (TRUNCATED at cap)" if truncated else ""
                print(f"    Downloaded {csv_path.name} ({size:,} bytes){note}")
            except Exception as e:
                result["status"] = "download_failed"
                result["error"] = str(e)
                print(f"    Download failed: {e}")
                resource_dir.mkdir(parents=True, exist_ok=True)
                meta_path.write_text(json.dumps(result, indent=2))
                return result

        result["file_size_bytes"] = csv_path.stat().st_size if csv_path.exists() else 0

        # Step 2: qsv describegpt (LLM column labels/descriptions/tags)
        qsv_data = None
        if not skip_qsv and api_key and csv_path.exists():
            print("    Running qsv describegpt...")
            qsv_data = await run_qsv_describegpt(csv_path, api_key, qsv_dict_path)
            if qsv_data:
                result["qsv_description"] = _extract_qsv_description(qsv_data)
                result["qsv_tags"] = _extract_qsv_tags(qsv_data)
                print(f"    qsv: description + {len(result['qsv_tags'] or [])} tags")
            elif result["status"] == "ok":
                result["status"] = "qsv_failed"
        elif skip_qsv:
            print("    Skipping qsv describegpt (--skip-qsv)")

        # Step 3+4: qsv stats and frequency (no LLM required)
        stats_data = None
        freq_data = None
        if csv_path.exists():
            print("    Running qsv stats...")
            stats_data = await run_qsv_stats(csv_path, stats_path)
            if stats_data:
                print(f"    Stats: {len(stats_data)} columns profiled")
            print("    Running qsv frequency...")
            freq_data = await run_qsv_frequency(csv_path, freq_path)
            if freq_data:
                print(f"    Frequency: top values for {len(freq_data)} columns")

        # A DCAT catalog publishes no field-level dictionary, so the portal
        # side of the merge is empty by definition.
        result["columns"] = merge_columns(None, qsv_data, stats_data, freq_data)

        # Step 5: exact row count
        if csv_path.exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    "qsv", "count", str(csv_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    result["row_count"] = int(stdout.decode().strip())
            except Exception:
                pass

        resource_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(result, indent=2))
        return result


def rebuild_from_disk(base_dir: Path) -> list[dict]:
    """Re-read meta.json files already on disk (no downloads, no API calls)."""
    results: list[dict] = []
    for meta_path in sorted(base_dir.rglob("meta.json")):
        try:
            results.append(json.loads(meta_path.read_text()))
        except Exception as e:
            print(f"  Skipping {meta_path}: {e}")
    return results


def build_index(site_info: dict, results: list[dict]) -> dict:
    """Build the master index.json from processed dataset results.

    Mirrors the CKAN index shape so ``data_layer.onboard_index`` and
    ``data_layer.pinecone_upload`` read both portal types identically.
    """
    datasets_map: dict[str, dict] = {}
    for res in results:
        ds_id = res["dataset_id"]
        entry = datasets_map.setdefault(
            ds_id,
            {
                "dataset_id": ds_id,
                "dataset_title": res.get("dataset_title", ""),
                "dataset_description": res.get("dataset_description", ""),
                "organization": res.get("organization", ""),
                "tags": res.get("dataset_tags", []),
                "resources": [],
            },
        )
        entry["resources"].append(
            {
                k: v
                for k, v in res.items()
                if k
                not in (
                    "dataset_title",
                    "dataset_description",
                    "organization",
                    "dataset_tags",
                )
            }
        )

    return {
        "site_id": site_info["id"],
        "site_url": site_info["url"],
        "site_name": site_info["name"],
        "portal_type": "dcat",
        "catalog_url": site_info.get("catalog_url") or "",
        "onboarded_at": _now_iso(),
        "total_datasets": len(datasets_map),
        "total_resources": len(results),
        "datasets": list(datasets_map.values()),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Onboard a DCAT portal: download CSVs, profile with qsv, index.",
    )
    parser.add_argument("--site-id", default="data-pa-gov", help="Portal ID from the registry")
    parser.add_argument("--portal-url", default=None, help="Portal URL (bypasses the registry)")
    parser.add_argument("--catalog-url", default=None, help="Explicit DCAT catalog URL")
    parser.add_argument(
        "--openrouter-api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key for qsv describegpt (or set OPENROUTER_API_KEY)",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel datasets")
    parser.add_argument("--limit", type=int, default=0, help="Max datasets to process (0 = all)")
    parser.add_argument("--dataset-filter", default=None, help="Only datasets matching this text")
    parser.add_argument(
        "--max-mb", type=int, default=DEFAULT_MAX_MB, help="Per-file download cap in MB"
    )
    parser.add_argument("--skip-download", action="store_true", help="Reuse CSVs already on disk")
    parser.add_argument("--skip-qsv", action="store_true", help="Skip qsv describegpt (no LLM)")
    parser.add_argument("--output-dir", default="data/dcat_onboard", help="Output directory")
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing to storage backend")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild index.json from meta.json files on disk (no downloads)",
    )
    parser.add_argument(
        "--pinecone", action="store_true", help="Upload the built index to Pinecone"
    )
    parser.add_argument("--pinecone-namespace", default=None, help="Override Pinecone namespace")
    parser.add_argument("--pinecone-index", default=None, help="Override Pinecone index name")
    parser.add_argument(
        "--pinecone-dry-run",
        action="store_true",
        help="Build Pinecone records and show a sample without writing",
    )
    args = parser.parse_args()

    # Resolve the portal, from the registry or from explicit flags.
    if args.portal_url:
        site = {
            "id": args.site_id or _slugify(args.portal_url),
            "url": args.portal_url.rstrip("/"),
            "name": args.portal_url,
            "catalog_url": args.catalog_url,
        }
    else:
        registered = get_site(args.site_id)
        if not registered:
            print(f"Portal '{args.site_id}' is not registered. Add it in the admin panel,")
            print("or pass --portal-url to onboard an unregistered portal.")
            sys.exit(1)
        if normalize_portal_type(registered.get("portal_type")) != "dcat":
            print(
                f"Portal '{args.site_id}' is registered as a CKAN portal. "
                "Use scripts/onboard_ckan.py for it."
            )
            sys.exit(1)
        site = dict(registered)
        if args.catalog_url:
            site["catalog_url"] = args.catalog_url

    base_dir = Path(args.output_dir) / site["id"]
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Onboarding DCAT portal: {site['name']}  ({site['url']})")
    print(f"Output: {base_dir}")

    if args.rebuild_index:
        print("\nRebuilding index from files on disk...")
        results = rebuild_from_disk(base_dir)
        print(f"  Found {len(results)} profiled datasets")
    else:
        client = DCATClient(site["url"], catalog_url=site.get("catalog_url"))
        try:
            catalog = await client.fetch_catalog()
            print(f"\nCatalog: {catalog.catalog_url}")
            print(f"  {len(catalog.datasets)} datasets")

            candidates = [ds for ds in catalog.datasets if ds.tabular_distributions]
            print(f"  {len(candidates)} with a tabular (CSV) distribution")

            if args.dataset_filter:
                needle = args.dataset_filter.lower()
                candidates = [
                    ds
                    for ds in candidates
                    if needle in ds.title.lower()
                    or needle in ds.description.lower()
                    or any(needle in k.lower() for k in ds.keywords)
                ]
                print(f"  {len(candidates)} match --dataset-filter {args.dataset_filter!r}")

            if args.limit:
                candidates = candidates[: args.limit]
                print(f"  Processing {len(candidates)} (--limit {args.limit})")

            if not candidates:
                print("\nNothing to process.")
                return

            semaphore = asyncio.Semaphore(args.concurrency)
            max_bytes = args.max_mb * 1024 * 1024
            tasks = [
                process_dataset(
                    client,
                    ds,
                    base_dir,
                    args.openrouter_api_key or None,
                    args.skip_download,
                    args.skip_qsv,
                    max_bytes,
                    semaphore,
                )
                for ds in candidates
            ]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await client.close()

        results = []
        for item in gathered:
            if isinstance(item, BaseException):
                print(f"  Dataset failed: {item}")
            elif item is not None:
                results.append(item)

    if not results:
        print("\nNo datasets were profiled.")
        return

    index = build_index(site, results)
    index_path = base_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))

    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n{'=' * 60}")
    print(f"Profiled {len(results)} datasets ({ok} clean)")
    print(f"Index: {index_path}")

    if not args.no_sync:
        synced = sync_to_storage(base_dir, site["id"], prefix="dcat_onboard")
        print(f"Synced {synced} files to the storage backend")

    if args.pinecone:
        from data_concierge.data_layer.pinecone_upload import upload_index

        print("\nUploading to Pinecone...")
        summary = upload_index(
            index,
            site_id=site["id"],
            namespace=args.pinecone_namespace,
            index_name=args.pinecone_index,
            dry_run=args.pinecone_dry_run,
        )
        print(f"  Records built: {summary.get('records_built', 0)}")
        if args.pinecone_dry_run:
            sample = summary.get("sample")
            if sample:
                preview = dict(sample)
                preview["text"] = preview["text"][:400] + "..."
                print("  DRY RUN — sample record:")
                print(json.dumps(preview, indent=4)[:2000])
        else:
            print(f"  Upserted: {summary.get('upserted', 0)}")
            print(f"  Failed:   {summary.get('failed', 0)}")
            print(f"  Index/ns: {summary.get('index_name')} / {summary.get('namespace')}")
        for err in summary.get("errors", [])[:5]:
            print(f"  ERROR: {err}")


if __name__ == "__main__":
    asyncio.run(main())
