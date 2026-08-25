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
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB — qsv stats can produce large fields

import httpx  # noqa: E402

from data_concierge.data_layer.connectors.ckan import CKANClient  # noqa: E402
from data_concierge.data_layer.storage import storage  # noqa: E402
from data_concierge.gateway.ckan_sites import get_site  # noqa: E402


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


async def download_csv(url: str, dest: Path) -> int:
    """Stream-download a CSV file. Returns file size in bytes."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as http:
        async with http.stream("GET", url) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
                    size += len(chunk)
    return size


async def fetch_ckan_dict(client: CKANClient, resource_id: str) -> list[dict] | None:
    """Fetch field metadata from CKAN datastore (limit=0 to skip records)."""
    result = await client.datastore_search(resource_id, limit=0)
    if not result:
        return None
    fields = result.get("fields", [])
    return [f for f in fields if f.get("id") != "_id"]


async def run_qsv_describegpt(
    csv_path: Path,
    api_key: str,
    output_path: Path,
) -> dict | None:
    """Run qsv describegpt and return parsed JSON output."""
    cmd = [
        "qsv", "describegpt", str(csv_path),
        "--all",
        "--format", "json",
        "--base-url", "https://openrouter.ai/api/v1",
        "--model", "google/gemini-2.5-flash-lite",
        "--api-key", api_key,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"    qsv describegpt failed (exit {proc.returncode}): {stderr.decode()[:500]}")
        return None

    try:
        data = json.loads(stdout.decode())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return data
    except json.JSONDecodeError as e:
        print(f"    qsv output not valid JSON: {e}")
        return None


async def run_qsv_stats(csv_path: Path, output_path: Path) -> dict[str, dict] | None:
    """Run qsv stats and return per-column stats keyed by column name."""
    proc = await asyncio.create_subprocess_exec(
        "qsv", "stats", str(csv_path), "--everything",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"    qsv stats failed: {stderr.decode()[:300]}")
        return None

    raw = stdout.decode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw)

    reader = csv.DictReader(StringIO(raw))
    stats_by_col: dict[str, dict] = {}
    for row in reader:
        col_name = row.get("field", "")
        if col_name:
            stats_by_col[col_name] = {k: v for k, v in row.items() if k != "field"}
    return stats_by_col


async def run_qsv_frequency(csv_path: Path, output_path: Path) -> dict[str, list[dict]] | None:
    """Run qsv frequency and return top values per column."""
    proc = await asyncio.create_subprocess_exec(
        "qsv", "frequency", str(csv_path), "--limit", "20",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        print(f"    qsv frequency failed: {stderr.decode()[:300]}")
        return None

    raw = stdout.decode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw)

    reader = csv.DictReader(StringIO(raw))
    freq_by_col: dict[str, list[dict]] = {}
    for row in reader:
        col_name = row.get("field", "")
        if col_name:
            freq_by_col.setdefault(col_name, []).append({
                "value": row.get("value", ""),
                "count": int(row.get("count", 0)),
            })
    return freq_by_col


def _extract_qsv_fields(qsv_data: dict) -> list[dict]:
    """Extract field list from qsv describegpt JSON, handling both key casings."""
    for key in ("Dictionary", "dictionary"):
        val = qsv_data.get(key)
        if isinstance(val, dict):
            resp = val.get("response", val)
            fields = resp.get("fields", [])
            if isinstance(fields, list):
                return fields
        elif isinstance(val, list):
            return val
    return []


def _extract_qsv_description(qsv_data: dict) -> str:
    """Extract description string from qsv describegpt JSON."""
    for key in ("Description", "description"):
        val = qsv_data.get(key)
        if isinstance(val, dict):
            resp = val.get("response", val)
            if isinstance(resp, str):
                return resp
            return resp.get("description", "")
        elif isinstance(val, str):
            return val
    return ""


def _extract_qsv_tags(qsv_data: dict) -> list[str]:
    """Extract tags list from qsv describegpt JSON."""
    for key in ("Tags", "tags"):
        val = qsv_data.get(key)
        if isinstance(val, dict):
            resp = val.get("response", val)
            if isinstance(resp, dict):
                raw = resp.get("tags", [])
            else:
                raw = resp
            if isinstance(raw, str):
                return [t.strip() for t in raw.split(",")]
            if isinstance(raw, list):
                return raw
    return []


def merge_columns(
    ckan_fields: list[dict] | None,
    qsv_data: dict | None,
    stats_data: dict[str, dict] | None = None,
    freq_data: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Merge CKAN field metadata with qsv dictionary into a unified column list."""
    ckan_by_name: dict[str, dict] = {}
    if ckan_fields:
        for f in ckan_fields:
            ckan_by_name[f["id"]] = f

    qsv_by_name: dict[str, dict] = {}
    if qsv_data:
        qsv_fields = _extract_qsv_fields(qsv_data)
        for col in qsv_fields:
            name = col.get("name") or col.get("field") or col.get("column")
            if name:
                qsv_by_name[name] = col

    all_names: list[str] = list(ckan_by_name.keys())
    for name in qsv_by_name:
        if name not in ckan_by_name:
            all_names.append(name)
    if stats_data:
        for name in stats_data:
            if name not in ckan_by_name and name not in qsv_by_name:
                all_names.append(name)

    columns = []
    for name in all_names:
        ckan_f = ckan_by_name.get(name, {})
        qsv_f = qsv_by_name.get(name, {})

        col: dict = {"name": name}

        if ckan_f:
            col["ckan_type"] = ckan_f.get("type", "")
            info = ckan_f.get("info", {})
            if info:
                col["ckan_info"] = info

        if qsv_f:
            col["qsv_label"] = qsv_f.get("label", "")
            col["qsv_description"] = qsv_f.get("description", "")
            col["qsv_type"] = qsv_f.get("type", "")
            qsv_extras = {
                k: v
                for k, v in qsv_f.items()
                if k not in ("name", "field", "column", "label", "description", "type")
            }
            if qsv_extras:
                col["qsv_dict_stats"] = qsv_extras

        if stats_data and name in stats_data:
            col["stats"] = stats_data[name]

        if freq_data and name in freq_data:
            col["top_values"] = freq_data[name]

        columns.append(col)

    return columns


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


def build_index(
    site_info: dict,
    resource_results: list[dict],
) -> dict:
    """Build the master index from all processed resource results."""
    datasets_map: dict[str, dict] = {}

    for res in resource_results:
        ds_id = res["dataset_id"]
        if ds_id not in datasets_map:
            datasets_map[ds_id] = {
                "dataset_id": ds_id,
                "dataset_title": res["dataset_title"],
                "dataset_description": res["dataset_description"],
                "organization": res["organization"],
                "tags": res.get("dataset_tags", []),
                "resources": [],
            }

        resource_entry = {
            k: v
            for k, v in res.items()
            if k not in ("dataset_title", "dataset_description", "organization", "dataset_tags")
        }
        datasets_map[ds_id]["resources"].append(resource_entry)

    return {
        "site_id": site_info["id"],
        "site_url": site_info["url"],
        "site_name": site_info["name"],
        "onboarded_at": _now_iso(),
        "total_datasets": len(datasets_map),
        "total_resources": len(resource_results),
        "datasets": list(datasets_map.values()),
    }


def sync_to_storage(base_dir: Path, site_id: str) -> int:
    """Sync JSON and qsv CSV files from local disk to the unified storage backend."""
    synced = 0
    for json_file in base_dir.rglob("*.json"):
        rel = json_file.relative_to(base_dir.parent)
        key = f"ckan_onboard/{rel}"
        with open(json_file) as f:
            data = json.load(f)
        storage.write_json(key, data)
        synced += 1
    for csv_file in base_dir.rglob("qsv_*.csv"):
        rel = csv_file.relative_to(base_dir.parent)
        key = f"ckan_onboard/{rel}"
        storage.write_bytes(key, csv_file.read_bytes())
        synced += 1
    return synced


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

    finally:
        await client.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
