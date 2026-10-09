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

Every pass runs through `qsv-client <https://github.com/dathere/qsv-client>`_, which
kills a run that exceeds its timeout (see ``_DEFAULT_TIMEOUTS``; each one can be
overridden with ``VERIKAN_QSV_<PASS>_TIMEOUT``). A failed, timed-out, or missing
qsv makes the pass return ``None``, and the onboarding scripts carry on without it.

:func:`merge_columns` folds all of those, plus the portal's own field
metadata when it has any, into the single column list that
``data_layer.onboard_index`` searches and that
``data_layer.pinecone_upload`` turns into embedded records.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from qsv_client import (
    AsyncQsv,
    QsvError,
    QsvNotFound,
    QsvTimeout,
    QsvVersionError,
)

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


# Default per-pass timeouts in seconds. ``VERIKAN_QSV_<PASS>_TIMEOUT`` overrides one,
# e.g. ``VERIKAN_QSV_STATS_TIMEOUT=1800`` for a portal with very large CSVs.
_DEFAULT_TIMEOUTS: dict[str, float] = {
    "describegpt": 900.0,  # several LLM round trips
    "stats": 900.0,
    "frequency": 600.0,
    "count": 120.0,
}

# A failed run, a missing or too-old binary, or output that does not parse. QsvNotFound and
# QsvVersionError are not QsvError subclasses, and the client raises QsvNotFound from its
# constructor.
_QSV_FAILURES = (QsvError, QsvNotFound, QsvVersionError, ValueError)

# Binaries whose missing describegpt has already been reported, so a portal run warns once.
_describegpt_checked: set[str] = set()


def _timeout(command: str) -> float:
    var = f"VERIKAN_QSV_{command.upper()}_TIMEOUT"
    raw = os.environ.get(var)
    if raw:
        try:
            return float(raw)
        except ValueError:
            print(f"    ignoring {var}={raw!r}: not a number")
    return _DEFAULT_TIMEOUTS[command]


def _qsv(api_key: str | None = None) -> AsyncQsv:
    """The qsv client every profiling pass runs through.

    The binary is ``$QSV_BIN`` if set, else the first of qsv, qsvmcp, qsvdp, qsvlite on
    ``PATH``. Each run is killed (with its whole process group) once it passes its timeout.
    """
    return AsyncQsv(llm_api_key=api_key)


def _report_failure(command: str, exc: Exception) -> None:
    kind = getattr(exc, "kind", None)
    message = getattr(exc, "message", None) or str(exc)
    label = f"{type(exc).__name__}, {kind}" if kind else type(exc).__name__
    print(f"    qsv {command} failed ({label}): {message[:500]}")


async def _explain_missing_describegpt(qsv: AsyncQsv) -> None:
    """After a describegpt failure, say so if the binary has no describegpt at all.

    qsvlite and qsvdp lack it, and binary discovery falls back to them. Checked lazily so a
    successful run pays nothing for the probe.
    """
    if qsv.binary in _describegpt_checked:
        return
    _describegpt_checked.add(qsv.binary)
    try:
        caps = await qsv.capabilities()
    except _QSV_FAILURES:
        return
    if caps.commands and not caps.has_command("describegpt"):
        print(
            f"    {qsv.binary} ({caps.binary} {caps.version}) has no describegpt command; "
            "install the full qsv binary or point QSV_BIN at it"
        )


async def run_qsv_describegpt(
    csv_path: Path,
    api_key: str,
    output_path: Path,
) -> dict | None:
    """Run qsv describegpt and return parsed JSON output.

    The API key goes to qsv in ``QSV_LLM_APIKEY``, not ``--api-key``: describegpt
    copies its command line into the descriptions it generates, and argv is also
    visible to other local users in ``ps``.

    The output is re-serialised with ``json.dump(indent=2)`` rather than streamed to
    ``output_path`` as qsv wrote it: ``qsv_dict.json`` is published and hashed, so its
    bytes must not change with qsv's own formatting.
    """
    qsv: AsyncQsv | None = None
    try:
        qsv = _qsv(api_key)
        data = await qsv.describegpt(
            csv_path,
            "--all",
            "--format", "json",
            "--base-url", "https://openrouter.ai/api/v1",
            "--model", "google/gemini-2.5-flash-lite",
            timeout=_timeout("describegpt"),
        )
    except _QSV_FAILURES as e:
        _report_failure("describegpt", e)
        if qsv is not None and isinstance(e, QsvError) and not isinstance(e, QsvTimeout):
            await _explain_missing_describegpt(qsv)
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    return data


async def _run_qsv_csv(
    command: str, csv_path: Path, output_path: Path, *args: str
) -> list[dict[str, str]] | None:
    """Stream ``qsv <command>``'s CSV output to ``output_path`` and return its rows.

    qsv writes to a sibling ``.part`` file that replaces ``output_path`` only once the run
    succeeds and parses, so a failed or killed run leaves the previous output in place
    rather than a truncated file.
    """
    part = output_path.with_name(output_path.name + ".part")
    try:
        res = await _qsv().run(
            command, csv_path, *args, stdout_path=part, timeout=_timeout(command)
        )
        rows = res.csv_rows()
        os.replace(part, output_path)
        return rows
    except _QSV_FAILURES as e:
        _report_failure(command, e)
        return None
    finally:
        part.unlink(missing_ok=True)


async def run_qsv_stats(csv_path: Path, output_path: Path) -> dict[str, dict] | None:
    """Run qsv stats and return per-column stats keyed by column name."""
    rows = await _run_qsv_csv("stats", csv_path, output_path, "--everything")
    if rows is None:
        return None

    stats_by_col: dict[str, dict] = {}
    for row in rows:
        col_name = row.get("field", "")
        if col_name:
            stats_by_col[col_name] = {k: v for k, v in row.items() if k != "field"}
    return stats_by_col


async def run_qsv_frequency(csv_path: Path, output_path: Path) -> dict[str, list[dict]] | None:
    """Run qsv frequency and return top values per column."""
    rows = await _run_qsv_csv("frequency", csv_path, output_path, "--limit", "20")
    if rows is None:
        return None

    freq_by_col: dict[str, list[dict]] = {}
    for row in rows:
        col_name = row.get("field", "")
        if col_name:
            freq_by_col.setdefault(col_name, []).append({
                "value": row.get("value", ""),
                "count": int(row.get("count", 0)),
            })
    return freq_by_col


async def run_qsv_count(csv_path: Path) -> int | None:
    """Exact row count of ``csv_path``, or ``None`` if qsv could not count it."""
    try:
        return await _qsv().count(csv_path, timeout=_timeout("count"))
    except _QSV_FAILURES as e:
        _report_failure("count", e)
        return None


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


def csv_header(path: Path) -> list[str]:
    """A CSV's column names, as the Fair Store mirror reads them."""
    csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [name for name in next(csv.reader(handle), []) if name]


def scrubbed_json_bytes(path: Path) -> bytes:
    """A JSON file's bytes with provider keys redacted, as the mirror publishes it.

    ``populate_fairstore`` scrubs the raw text of ``qsv_dict.json`` before
    publishing it and hashes the result, so storing exactly that text keeps a
    mirror run from storage and one from local files identical (a per-value
    scrub keeps the word after the key, which the raw scrub removes, and every
    describegpt resource would flip between the two). Should raw scrubbing
    ever break the JSON — a key right before a closing quote — each string
    value is scrubbed instead.
    """
    from data_concierge.data_layer.onboard_index import _scrub_secrets

    text = path.read_text(encoding="utf-8")
    scrubbed = _scrub_secrets(text)
    try:
        json.loads(scrubbed)
    except ValueError:
        scrubbed = json.dumps(_scrub_values(json.loads(text)), indent=2)
    return scrubbed.encode("utf-8")


def _scrub_values(value: Any) -> Any:
    from data_concierge.data_layer.onboard_index import _scrub_secrets

    if isinstance(value, str):
        return _scrub_secrets(value)
    if isinstance(value, list):
        return [_scrub_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _scrub_values(item) for key, item in value.items()}
    return value


# The qsv outputs populate_fairstore reads from a dataset directory.
MIRRORED_QSV_OUTPUTS = ("qsv_dict.json", "qsv_stats.csv", "qsv_frequency.csv")


def recorded_headers(base_dir: Path, prefix: str) -> dict[str, list[str]]:
    """Each downloaded CSV's header, keyed as its storage key would be."""
    headers: dict[str, list[str]] = {}
    for csv_file in sorted(base_dir.rglob("*.csv")):
        if csv_file.name.startswith("qsv_"):
            continue
        try:
            header = csv_header(csv_file)
        except (OSError, csv.Error):
            continue  # an unreadable download must not stop the sync
        # Even an empty header: locally the mirror reads [] from such a file.
        headers[f"{prefix}/{csv_file.relative_to(base_dir.parent)}"] = header
    return headers


def sync_manifest(base_dir: Path, prefix: str) -> dict[str, Any]:
    """What a mirror run from storage needs to know about the local directory.

    ``headers``: each downloaded CSV's header. ``dirs``: every dataset directory
    and the qsv outputs it holds now — so a directory that is absent stays
    absent, and an output deleted since an earlier sync is not staged.
    """
    dirs = {
        f"{prefix}/{directory.relative_to(base_dir.parent)}": sorted(
            name for name in MIRRORED_QSV_OUTPUTS if (directory / name).is_file()
        )
        for directory in sorted(p for p in base_dir.iterdir() if p.is_dir())
    }
    return {"version": 1, "headers": recorded_headers(base_dir, prefix), "dirs": dirs}


def sync_to_storage(base_dir: Path, site_id: str, prefix: str = "ckan_onboard") -> int:
    """Sync JSON and qsv CSV files from local disk to the unified storage backend.

    ``prefix`` is the storage key namespace.  It defaults to ``ckan_onboard``
    because ``onboard_index`` reads from there and existing deployments already
    have data under that prefix.

    The downloaded CSVs themselves are not synced; ``sync_manifest.json``
    records their headers and which dataset directories and qsv outputs exist
    (:func:`sync_manifest`). ``populate_fairstore`` checks each qsv output
    against the header of the file it describes, and runs from the storage
    backend when the admin panel starts a mirror on Cloud Run.

    JSON is scrubbed of provider keys on the way: qsv describegpt records its
    own command line in ``qsv_dict.json``, ``meta.json`` and ``index.json``,
    and onboarding runs before the key moved to ``QSV_LLM_APIKEY`` passed it
    on ``--api-key``.
    """
    synced = 0
    manifest = sync_manifest(base_dir, prefix)
    for json_file in base_dir.rglob("*.json"):
        rel = json_file.relative_to(base_dir.parent)
        key = f"{prefix}/{rel}"
        storage.write_bytes(key, scrubbed_json_bytes(json_file))
        synced += 1
    for csv_file in base_dir.rglob("qsv_*.csv"):
        rel = csv_file.relative_to(base_dir.parent)
        key = f"{prefix}/{rel}"
        storage.write_bytes(key, csv_file.read_bytes())
        synced += 1
    # Last: a manifest must never list objects an interrupted sync did not write.
    storage.write_json(f"{prefix}/{site_id}/sync_manifest.json", manifest)
    return synced + 1
