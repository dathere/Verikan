"""Shared dataset-profiling helpers for portal onboarding.

Both onboarding scripts — ``scripts/onboard_ckan.py`` and
``scripts/onboard_dcat.py`` — download a portal's CSV resources, profile them
with `qsv <https://github.com/dathere/qsv>`_, and write a data dictionary per
resource.  Only the *discovery* half differs between the two (a CKAN action
API versus a DCAT catalog document), so everything downstream of "we have a
CSV on disk" lives here and is shared.

The qsv passes, in the order the scripts run them:

``describegpt``
    LLM-generated column labels, descriptions, a dataset summary, and tags.
    Requires an OpenRouter API key; skipped otherwise.
``stats``
    Per-column type, cardinality, min/max, quartiles. No LLM, always run.
``frequency``
    Top values per column. No LLM, always run.
``count``
    Exact row count.

:func:`merge_columns` folds all of those, plus the portal's own field
metadata when it has any, into the single column list that
``data_layer.onboard_index`` searches and that
``data_layer.pinecone_upload`` turns into embedded records.
"""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import httpx

from data_concierge.data_layer.storage import storage

csv.field_size_limit(10 * 1024 * 1024)  # 10 MB — qsv stats can produce large fields


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
                # describegpt capitalises the key in some responses
                # ({"Attribution": ..., "Tags": [...]}).
                raw = next((v for k, v in resp.items() if str(k).lower() == "tags"), [])
            else:
                raw = resp
            if isinstance(raw, str):
                return [t.strip() for t in raw.split(",")]
            if isinstance(raw, list):
                return raw
    return []


def merge_columns(
    portal_fields: list[dict] | None,
    qsv_data: dict | None,
    stats_data: dict[str, dict] | None = None,
    freq_data: dict[str, list[dict]] | None = None,
) -> list[dict]:
    """Merge portal field metadata with the qsv dictionary into one column list.

    ``portal_fields`` is the portal's own data dictionary when it has one — a
    CKAN DataStore field list, keyed by ``id``.  A DCAT catalog publishes no
    field-level metadata, so DCAT onboarding passes ``None`` and every column
    is described from qsv output alone.
    """
    ckan_by_name: dict[str, dict] = {}
    if portal_fields:
        for f in portal_fields:
            field_id = f.get("id") or f.get("name")
            if field_id:
                ckan_by_name[field_id] = f

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


def sync_to_storage(base_dir: Path, site_id: str, prefix: str = "ckan_onboard") -> int:
    """Sync JSON and qsv CSV files from local disk to the unified storage backend.

    ``prefix`` is the storage key namespace.  It defaults to ``ckan_onboard``
    because ``onboard_index`` reads from there and existing deployments already
    have data under that prefix.
    """
    synced = 0
    for json_file in base_dir.rglob("*.json"):
        rel = json_file.relative_to(base_dir.parent)
        key = f"{prefix}/{rel}"
        with open(json_file) as f:
            data = json.load(f)
        storage.write_json(key, data)
        synced += 1
    for csv_file in base_dir.rglob("qsv_*.csv"):
        rel = csv_file.relative_to(base_dir.parent)
        key = f"{prefix}/{rel}"
        storage.write_bytes(key, csv_file.read_bytes())
        synced += 1
    return synced
