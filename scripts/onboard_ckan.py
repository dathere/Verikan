#!/usr/bin/env python3
"""Onboard a CKAN portal: download CSVs, build data dictionaries, create searchable index.

Usage:
    python scripts/onboard_ckan.py --site-id wprdc --openrouter-api-key <key>
    python scripts/onboard_ckan.py --site-id wprdc --dataset-filter 311 --skip-qsv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB — qsv stats can produce large fields


from data_concierge.data_layer.connectors.ckan import CKANClient  # noqa: E402
from data_concierge.data_layer.qsv_profiling import (  # noqa: E402
    _extract_qsv_description,
    _extract_qsv_tags,
    _now_iso,
    build_index,
    download_csv,
    merge_columns,
    run_qsv_describegpt,
    run_qsv_frequency,
    run_qsv_stats,
    sync_to_storage,
)
from data_concierge.gateway.ckan_sites import get_site  # noqa: E402


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "unnamed"


async def list_all_datasets(
    client: CKANClient,
    organization: str | None = None,
) -> list[dict]:
    """Paginate package_search to get every dataset from the portal."""
    datasets: list[dict] = []
    page_size = 500
    start = 0

    fq = [f"organization:{organization}"] if organization else None

    while True:
        result = await client.package_search("*:*", rows=page_size, start=start, filter_queries=fq)
        batch = result.get("results", [])
        if not batch:
            break
        datasets.extend(batch)
        total = result.get("count", 0)
        start += page_size
        if start >= total:
            break

    return datasets


def extract_csv_resources(datasets: list[dict]) -> list[tuple[dict, dict]]:
    """Return (dataset, resource) pairs for every CSV resource."""
    pairs = []
    for ds in datasets:
        for res in ds.get("resources", []):
            fmt = (res.get("format") or "").upper()
            if fmt == "CSV":
                pairs.append((ds, res))
    return pairs


async def fetch_ckan_dict(client: CKANClient, resource_id: str) -> list[dict] | None:
    """Fetch field metadata from CKAN datastore (limit=0 to skip records)."""
    result = await client.datastore_search(resource_id, limit=0)
    if not result:
        return None
    fields = result.get("fields", [])
    return [f for f in fields if f.get("id") != "_id"]


def build_resource_download_url(ckan_url: str, resource: dict) -> str:
    """Get the best download URL for a CSV resource."""
    url = resource.get("url", "")
    if url:
        return url
    resource_id = resource.get("id", "")
    return f"{ckan_url}/datastore/dump/{resource_id}"


async def process_resource(
    client: CKANClient,
    dataset: dict,
    resource: dict,
    base_dir: Path,
    api_key: str | None,
    skip_download: bool,
    skip_qsv: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process a single CSV resource: download, fetch dict, run qsv."""
    async with semaphore:
        ds_name = dataset.get("name", _slugify(dataset.get("title", "unknown")))
        res_name = _slugify(resource.get("name", resource.get("id", "unknown")))
        res_id = resource.get("id", "")

        resource_dir = base_dir / ds_name
        csv_path = resource_dir / f"{res_name}.csv"
        ckan_dict_path = resource_dir / "ckan_dict.json"
        qsv_dict_path = resource_dir / "qsv_dict.json"
        stats_path = resource_dir / "qsv_stats.csv"
        freq_path = resource_dir / "qsv_frequency.csv"
        meta_path = resource_dir / "meta.json"

        result: dict = {
            "dataset_id": dataset.get("name", ""),
            "dataset_title": dataset.get("title", ""),
            "dataset_description": dataset.get("notes", ""),
            "organization": (dataset.get("organization") or {}).get("name", ""),
            "dataset_tags": [t.get("name", "") for t in dataset.get("tags", [])],
            "resource_id": res_id,
            "resource_name": resource.get("name", ""),
            "format": "CSV",
            "local_path": str(csv_path),
            "columns": [],
            "qsv_description": None,
            "qsv_tags": None,
            "status": "ok",
            "error": None,
            "onboarded_at": _now_iso(),
        }

        # Step 1: Download CSV
        file_size = 0
        if skip_download and csv_path.exists():
            file_size = csv_path.stat().st_size
            print(f"    Skipping download (exists): {csv_path.name} ({file_size:,} bytes)")
        else:
            try:
                download_url = build_resource_download_url(client.ckan_url, resource)
                print(f"    Downloading {download_url}")
                file_size = await download_csv(download_url, csv_path)
                print(f"    Downloaded {csv_path.name} ({file_size:,} bytes)")
            except Exception as e:
                result["status"] = "download_failed"
                result["error"] = str(e)
                print(f"    Download failed: {e}")
                return result

        result["file_size_bytes"] = file_size

        # Step 2: Fetch CKAN data dictionary
        ckan_fields = None
        try:
            ckan_fields = await fetch_ckan_dict(client, res_id)
            if ckan_fields is not None:
                resource_dir.mkdir(parents=True, exist_ok=True)
                with open(ckan_dict_path, "w") as f:
                    json.dump({"fields": ckan_fields}, f, indent=2)
                print(f"    CKAN dict: {len(ckan_fields)} fields")
            else:
                result["status"] = "no_datastore"
                print("    No datastore (CKAN dict unavailable)")
        except Exception as e:
            print(f"    CKAN dict fetch failed: {e}")

        # Step 3: Run qsv describegpt
        qsv_data = None
        if not skip_qsv and api_key and csv_path.exists():
            print("    Running qsv describegpt...")
            qsv_data = await run_qsv_describegpt(csv_path, api_key, qsv_dict_path)
            if qsv_data:
                result["qsv_description"] = _extract_qsv_description(qsv_data)
                result["qsv_tags"] = _extract_qsv_tags(qsv_data)
                print(f"    qsv: description + {len(result['qsv_tags'] or [])} tags")
            else:
                if result["status"] == "ok":
                    result["status"] = "qsv_failed"
        elif skip_qsv:
            print("    Skipping qsv describegpt (--skip-qsv)")

        # Step 4: Run qsv stats (no LLM needed — always run if CSV exists)
        stats_data = None
        if csv_path.exists():
            print("    Running qsv stats...")
            stats_data = await run_qsv_stats(csv_path, stats_path)
            if stats_data:
                print(f"    Stats: {len(stats_data)} columns profiled")

        # Step 5: Run qsv frequency (no LLM needed — always run if CSV exists)
        freq_data = None
        if csv_path.exists():
            print("    Running qsv frequency...")
            freq_data = await run_qsv_frequency(csv_path, freq_path)
            if freq_data:
                print(f"    Frequency: top values for {len(freq_data)} columns")

        # Step 6: Merge columns
        result["columns"] = merge_columns(ckan_fields, qsv_data, stats_data, freq_data)

        # Step 7: Count rows from CSV
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

        # Save meta.json
        resource_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(result, f, indent=2)

        return result


def rebuild_from_disk(base_dir: Path) -> list[dict]:
    """Re-read existing output files and rebuild meta.json for each resource.

    Walks dataset directories, re-parses ckan_dict.json, qsv_dict.json,
    qsv_stats.csv, and qsv_frequency.csv, then rewrites meta.json with
    correctly merged columns.
    """
    results = []
    dataset_dirs = sorted(
        d for d in base_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()
    )

    for i, ds_dir in enumerate(dataset_dirs, 1):
        meta_path = ds_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        print(f"  [{i}/{len(dataset_dirs)}] {ds_dir.name}")

        # Re-read CKAN dict
        ckan_fields = None
        ckan_dict_path = ds_dir / "ckan_dict.json"
        if ckan_dict_path.exists():
            with open(ckan_dict_path) as f:
                ckan_fields = json.load(f).get("fields", [])

        # Re-read qsv dict
        qsv_data = None
        qsv_dict_path = ds_dir / "qsv_dict.json"
        if qsv_dict_path.exists():
            with open(qsv_dict_path) as f:
                qsv_data = json.load(f)
            meta["qsv_description"] = _extract_qsv_description(qsv_data)
            meta["qsv_tags"] = _extract_qsv_tags(qsv_data)

        # Re-read qsv stats
        stats_data = None
        stats_path = ds_dir / "qsv_stats.csv"
        if stats_path.exists():
            reader = csv.DictReader(StringIO(stats_path.read_text()))
            stats_data = {}
            for row in reader:
                col_name = row.get("field", "")
                if col_name:
                    stats_data[col_name] = {k: v for k, v in row.items() if k != "field"}

        # Re-read qsv frequency
        freq_data = None
        freq_path = ds_dir / "qsv_frequency.csv"
        if freq_path.exists():
            reader = csv.DictReader(StringIO(freq_path.read_text()))
            freq_data: dict[str, list[dict]] = {}  # type: ignore[no-redef]
            for row in reader:
                col_name = row.get("field", "")
                if col_name:
                    freq_data.setdefault(col_name, []).append({
                        "value": row.get("value", ""),
                        "count": int(row.get("count", 0)),
                    })

        # Merge and update
        meta["columns"] = merge_columns(ckan_fields, qsv_data, stats_data, freq_data)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        results.append(meta)

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Onboard a CKAN portal: download CSVs and build data dictionaries.",
    )
    parser.add_argument("--site-id", default="wprdc", help="CKAN site ID from registry")
    parser.add_argument(
        "--openrouter-api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Max parallel resources")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download if CSV exists (resume mode)"
    )
    parser.add_argument("--skip-qsv", action="store_true", help="Skip qsv describegpt")
    parser.add_argument("--dataset-filter", default=None, help="Only process matching datasets")
    parser.add_argument(
        "--output-dir", default="data/ckan_onboard", help="Base output directory"
    )
    parser.add_argument("--no-sync", action="store_true", help="Skip syncing to storage backend")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild meta.json and index.json from existing files on disk (no downloads, no API calls)",
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

    # Resolve site
    site = get_site(args.site_id)
    if not site:
        print(f"Error: Site '{args.site_id}' not found in CKAN sites registry.")
        print("Registered sites: ", end="")
        from data_concierge.gateway.ckan_sites import list_site_ids
        print(", ".join(list_site_ids()))
        sys.exit(1)

    base_dir = Path(args.output_dir) / args.site_id

    if args.rebuild_index:
        print(f"Rebuilding index from existing files in {base_dir}...")
        results = rebuild_from_disk(base_dir)
        if not results:
            print("No dataset directories with meta.json found.")
            return

        index = build_index(site, results)
        index_path = base_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Index rebuilt: {len(results)} resources")

        if not args.no_sync:
            print("Syncing to storage backend...")
            synced = sync_to_storage(base_dir, args.site_id)
            print(f"Synced {synced} files")
        print("Done!")
        return

    # Verify qsv is installed
    try:
        subprocess.run(["qsv", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: qsv is not installed or not in PATH.")
        print("Install from: https://github.com/dathere/qsv")
        sys.exit(1)

    site_url = site["url"]
    organization = site.get("organization")
    print(f"Onboarding CKAN site: {site['name']} ({site_url})")
    if organization:
        print(f"  Organization filter: {organization}")

    api_key = args.openrouter_api_key
    if not api_key and not args.skip_qsv:
        print("Warning: No OpenRouter API key provided. Use --openrouter-api-key or set")
        print("  OPENROUTER_API_KEY env var. Running with --skip-qsv behavior.")
        args.skip_qsv = True

    # List datasets
    client = CKANClient(ckan_url=site_url)
    try:
        print("Listing datasets...")
        datasets = await list_all_datasets(client, organization)
        print(f"Found {len(datasets)} datasets")

        if args.dataset_filter:
            datasets = [
                ds for ds in datasets
                if args.dataset_filter.lower() in (ds.get("name", "") + ds.get("title", "")).lower()
            ]
            print(f"Filtered to {len(datasets)} datasets matching '{args.dataset_filter}'")

        # Extract CSV resources
        pairs = extract_csv_resources(datasets)
        print(f"Found {len(pairs)} CSV resources to process")

        if not pairs:
            print("No CSV resources found. Done.")
            return

        # Process resources
        semaphore = asyncio.Semaphore(args.concurrency)
        results = []

        for i, (ds, res) in enumerate(pairs, 1):
            ds_name = ds.get("name", "?")
            res_name = res.get("name", "?")
            print(f"\n[{i}/{len(pairs)}] {ds_name} / {res_name}")

            result = await process_resource(
                client=client,
                dataset=ds,
                resource=res,
                base_dir=base_dir,
                api_key=api_key,
                skip_download=args.skip_download,
                skip_qsv=args.skip_qsv,
                semaphore=semaphore,
            )
            results.append(result)

        # Build index
        print(f"\nBuilding index ({len(results)} resources)...")
        index = build_index(site, results)
        index_path = base_dir / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Index written to {index_path}")

        # Summary
        ok = sum(1 for r in results if r["status"] == "ok")
        failed = len(results) - ok
        print(f"\nSummary: {ok} OK, {failed} failed/partial")
        for r in results:
            if r["status"] != "ok":
                print(f"  {r['status']}: {r['dataset_id']} / {r['resource_name']}")

        # Sync to storage backend
        if not args.no_sync:
            print("\nSyncing to storage backend...")
            synced = sync_to_storage(base_dir, args.site_id)
            print(f"Synced {synced} files")

        # Upload to Pinecone for semantic_search_resources
        if args.pinecone:
            from data_concierge.data_layer.pinecone_upload import upload_index

            print("\nUploading to Pinecone...")
            summary = upload_index(
                index,
                site_id=args.site_id,
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

    finally:
        await client.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
