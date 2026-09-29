"""Run and monitor portal onboarding jobs from the admin panel.

Onboarding a portal — download every CSV resource, profile it with qsv, build
the search index, optionally push it to Pinecone — is a long, chatty batch job
that previously could only be run by hand on a laptop.  This module runs the
same scripts as a supervised child process so an admin can launch one from the
Data Portals pane and watch its output.

Design notes worth knowing before changing this:

**Arguments are built, never accepted.**  Everything an admin submits is
validated and assembled into an ``argv`` *list* here; the child is spawned
without a shell.  The script to run is chosen from the portal's registered
``portal_type``, not from the request, so a caller cannot point the runner at
an arbitrary program.  Free-text options are length-capped and stripped of
control characters.

**Secrets never reach the command line.**  ``onboard_*.py`` accept
``--openrouter-api-key``, but an argv is visible in ``ps`` and would be echoed
into the job log, so the key is passed through the environment instead and the
log is scrubbed on the way in (qsv's describegpt records its own invocation,
including the key, in the descriptions it generates).

**One job at a time.**  These jobs saturate CPU and network and write to a
shared output directory; two concurrent runs against the same portal would
interleave their downloads.  A second start is refused with the running job's
ID rather than queued, so the caller always knows what is happening.

**Fair Store mirror runs share the runner.**  ``populate_fairstore.py``
(:func:`start_mirror_job`) runs under the same lock, log and records, tagged
``kind="fairstore_mirror"``: a mirror reads the onboarding output a concurrent
onboarding run would be rewriting. The Fair Store's sysadmin token reaches the
child through ``FAIRSTORE_API_KEY``, never argv.

**Any instance can show a run.**  Cloud Run sends each poll to whichever
instance it likes, and only one holds the child process. A running job
re-persists its record (log tail included) every :data:`HEARTBEAT_SECONDS`, so
another instance reports it as running from storage; a "running" record whose
heartbeat has gone stale is reported as interrupted. A cancel that lands on
another instance is written to ``<id>.cancel.json`` and carried out by its
owner's next heartbeat. Log offsets are absolute line numbers
(``log_line_count``), so a poll answered by a lagging instance or after the
tail passes :data:`MAX_LOG_LINES` never repeats or stalls. Two starts landing
on different instances at the same moment can both run; the storage backend
has no create-only write to take a lease with.

**Cloud Run caveat.**  A child process keeps running only while the instance
is alive and scheduled.  Without ``--no-cpu-throttling`` the CPU is throttled
to near zero between requests and a background job crawls; with
``--min-instances 0`` the instance can be reclaimed outright and the job dies.
:func:`runtime_warnings` reports this so the UI can say so up front rather than
letting an admin wonder why a job stalled.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_STORAGE_PREFIX = "onboarding_jobs"
_INDEX_KEY = f"{_STORAGE_PREFIX}/index.json"

# Keep the tail of the log in memory and in the persisted record. Onboarding a
# large portal emits tens of thousands of lines; the tail is what an operator
# reads, and an unbounded list would grow the record without limit.
MAX_LOG_LINES = 4000
MAX_JOB_RECORDS = 50

KIND_ONBOARDING = "onboarding"
KIND_FAIRSTORE_MIRROR = "fairstore_mirror"

# A running job's record is re-persisted this often (see the module docstring),
# and a stored "running" record whose heartbeat is older than STALE_AFTER has no
# live process anywhere.
HEARTBEAT_SECONDS = 10
STALE_AFTER_SECONDS = 60

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED)

# Bounds for the numeric options an admin can set.
_LIMITS: dict[str, tuple[int, int]] = {
    "limit": (0, 5000),
    "concurrency": (1, 10),
    "max_mb": (1, 2000),
    "mirror_limit": (1, 5000),
}
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# structlog's dev renderer colours its output, and a pipe is not a terminal but
# it colours anyway. Rendered in the admin panel's <pre>, those sequences show
# up literally as "\x1b[2m...\x1b[0m" around every log line, so they are
# stripped from what we retain.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# httpcore traces every connection at DEBUG, and the app logs at DEBUG outside
# production (core.logging.get_log_level), so a child inherits hundreds of
# connection lines around each real progress message. The job log is an
# operator's progress view, so that chatter is dropped from it — narrowly, by
# exact known prefixes, so nothing else (errors included) is ever swallowed.
# The child's own output is untouched; only what we retain is filtered.
_NOISE_PREFIXES = (
    "connect_tcp.",
    "start_tls.",
    "send_request_headers.",
    "send_request_body.",
    "receive_response_headers.",
    "receive_response_body.",
    "response_closed.",
    "close.started",
    "close.complete",
    "HTTP Request:",
    "Using selector:",
)


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in _NOISE_PREFIXES)

# job_id -> live process; only ever holds the single running job.
_processes: dict[str, asyncio.subprocess.Process] = {}
# job_id -> in-memory log tail (the persisted record carries a copy)
_logs: dict[str, deque[str]] = {}
_jobs: dict[str, dict[str, Any]] = {}
_start_lock = asyncio.Lock()
# Orders a heartbeat's storage write against the final one: a heartbeat write
# still in flight in a worker thread when the job ends must not land after the
# final record (it would read "running" again, then "interrupted").
_record_lock = threading.Lock()
_finished: set[str] = set()


class JobError(RuntimeError):
    """A job could not be started (bad options, or one already running)."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def scripts_dir() -> Path | None:
    """Locate the onboarding scripts directory.

    ``src/data_concierge/gateway/`` → repo root in a checkout, ``/app`` in the
    container image (which copies ``scripts/`` alongside ``src/``).
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent.parent / "scripts",  # repo root / /app
        Path("/app/scripts"),
        Path.cwd() / "scripts",
    ]
    for candidate in candidates:
        if (candidate / "onboard_dcat.py").exists():
            return candidate
    return None


def runtime_warnings() -> list[str]:
    """Environment problems that would make a job fail or stall.

    Surfaced in the admin UI *before* a launch, because every one of these
    produces a confusing failure deep inside a long run otherwise.
    """
    warnings: list[str] = []

    if scripts_dir() is None:
        warnings.append(
            "The onboarding scripts are not present in this deployment, so jobs "
            "cannot be started. The container image must copy scripts/ to /app/scripts."
        )
    if shutil.which("qsv") is None:
        warnings.append(
            "qsv is not installed on this host. Downloads will succeed but every "
            "profiling pass (stats, frequency, describegpt, count) will be skipped, "
            "so the index will have no column metadata."
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        warnings.append(
            "OPENROUTER_API_KEY is not set, so qsv describegpt cannot run. Column "
            "labels and AI descriptions will be missing unless you run with "
            "'Skip AI descriptions' — stats and frequency still work."
        )
    if os.environ.get("K_SERVICE"):
        # Set by Cloud Run. A child process only runs while the instance is
        # alive and has CPU.
        warnings.append(
            "Running on Cloud Run: a job survives only while this instance is "
            "alive and scheduled. Deploy with --no-cpu-throttling (and a "
            "min-instances of at least 1) or long jobs will stall or be killed "
            "mid-run. For a full portal, prefer running the script locally."
        )
    return warnings


def _scrub(text: str) -> str:
    """Redact provider keys and strip terminal colouring from child output."""
    from data_concierge.data_layer.onboard_index import _scrub_secrets

    return _scrub_secrets(_ANSI_RE.sub("", text))


def _clean_text_option(value: Any, *, max_len: int = 200) -> str | None:
    """Sanitise a free-text option destined for argv.

    The child is spawned without a shell, so quoting is not the concern —
    control characters corrupting the job log are.
    """
    if value is None:
        return None
    text = _CONTROL_RE.sub("", str(value)).strip()
    if not text:
        return None
    return text[:max_len]


def _clean_int_option(value: Any, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise JobError(f"{name} must be a whole number") from exc
    low, high = _LIMITS[name]
    if not (low <= number <= high):
        raise JobError(f"{name} must be between {low} and {high}")
    return number


def _clean_name_option(value: Any, name: str) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not _NAME_RE.match(text):
        raise JobError(
            f"{name} may only contain letters, digits, dot, dash, and underscore"
        )
    return text


def build_command(site: dict[str, Any], options: dict[str, Any]) -> tuple[list[str], str]:
    """Build the validated argv for a portal's onboarding run.

    Returns ``(argv, script_name)``.  Raises :class:`JobError` on any option
    the runner will not pass through.
    """
    from data_concierge.gateway.ckan_sites import PORTAL_TYPE_DCAT, normalize_portal_type

    directory = scripts_dir()
    if directory is None:
        raise JobError("Onboarding scripts are not available in this deployment")

    # The script is selected from the portal's registered type — never from
    # anything the caller supplied.
    portal_type = normalize_portal_type(site.get("portal_type"))
    script_name = (
        "onboard_dcat.py" if portal_type == PORTAL_TYPE_DCAT else "onboard_ckan.py"
    )
    script_path = directory / script_name
    if not script_path.exists():
        raise JobError(f"{script_name} is missing from {directory}")

    argv: list[str] = [sys.executable, str(script_path), "--site-id", str(site["id"])]

    if options.get("skip_qsv"):
        argv.append("--skip-qsv")
    if options.get("skip_download"):
        argv.append("--skip-download")
    if options.get("no_sync"):
        argv.append("--no-sync")
    if options.get("rebuild_index"):
        argv.append("--rebuild-index")

    dataset_filter = _clean_text_option(options.get("dataset_filter"))
    if dataset_filter:
        argv += ["--dataset-filter", dataset_filter]

    concurrency = _clean_int_option(options.get("concurrency"), "concurrency")
    if concurrency is not None:
        argv += ["--concurrency", str(concurrency)]

    # --limit and --max-mb exist only on the DCAT script.
    if portal_type == PORTAL_TYPE_DCAT:
        limit = _clean_int_option(options.get("limit"), "limit")
        if limit:
            argv += ["--limit", str(limit)]
        max_mb = _clean_int_option(options.get("max_mb"), "max_mb")
        if max_mb is not None:
            argv += ["--max-mb", str(max_mb)]

    if options.get("pinecone"):
        argv.append("--pinecone")
        if options.get("pinecone_dry_run"):
            argv.append("--pinecone-dry-run")
        namespace = _clean_name_option(options.get("pinecone_namespace"), "pinecone_namespace")
        if namespace:
            argv += ["--pinecone-namespace", namespace]
        index_name = _clean_name_option(options.get("pinecone_index"), "pinecone_index")
        if index_name:
            argv += ["--pinecone-index", index_name]

    return argv, script_name


def build_mirror_command(
    options: dict[str, Any],
    *,
    target_url: str,
    summary_path: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build the validated argv for a Fair Store mirror run.

    ``options["site"]`` is ``"all"`` or a registered portal ID, or
    ``options["sites"]`` a list of them — never the Fair Store's own
    chat-source entry. Returns ``(argv, sites)``, ``sites`` empty for every
    portal. The token is not an argument: :func:`start_mirror_job` passes it
    through the environment.
    """
    from data_concierge.gateway import ckan_sites
    from data_concierge.gateway.fairstore import PORTAL_ID, normalize_url

    directory = scripts_dir()
    if directory is None:
        raise JobError("The mirror script is not available in this deployment")
    script_path = directory / "populate_fairstore.py"
    if not script_path.exists():
        raise JobError(f"populate_fairstore.py is missing from {directory}")

    try:
        url = normalize_url(target_url)
    except ValueError as exc:
        raise JobError(str(exc)) from exc
    if not url:
        raise JobError("Set the Fair Store URL first")

    requested = options.get("sites") or [options.get("site") or "all"]
    if not isinstance(requested, list) or len(requested) > 50:
        raise JobError("Choose 'all' or up to 50 portals")
    sites: list[dict[str, Any]] = []
    if requested != ["all"]:
        for raw in dict.fromkeys(str(r).strip() for r in requested):
            site = ckan_sites.get_site(raw)
            if (
                site is None
                or str(site.get("id", "")).lower() == PORTAL_ID
                or site.get("managed_by") == "fairstore"
            ):
                raise JobError(f"'{raw}' is not a portal the Fair Store mirrors")
            sites.append(site)

    argv: list[str] = [
        sys.executable,
        str(script_path),
        "--site",
        ",".join(str(site["id"]) for site in sites) or "all",
        "--target-url",
        url,
        "--summary-file",
        summary_path,
    ]
    if options.get("apply"):
        argv.append("--apply")
    if options.get("enrich") is False:
        argv.append("--no-enrich")
    limit = _clean_int_option(options.get("limit"), "mirror_limit")
    if limit is not None:
        argv += ["--limit", str(limit)]
    return argv, sites


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields before a record leaves this module."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _job_key(job_id: str) -> str:
    return f"{_STORAGE_PREFIX}/{job_id}.json"


def _cancel_key(job_id: str) -> str:
    # Separate from the record, which the owner's heartbeat keeps rewriting.
    return f"{_STORAGE_PREFIX}/{job_id}.cancel.json"


def _write_record(record: dict[str, Any]) -> bool:
    """Write a job record and the index; ``False`` if storage failed.

    Pass ``_public(job)``, never the live record: it carries the supervising
    asyncio Task under ``_task``, which is not JSON-serializable, and writing it
    silently lost every completed job.
    """
    job = record
    try:
        storage.write_json(_job_key(job["id"]), record)
        index = storage.read_json(_INDEX_KEY) or {}
        ids = [i for i in index.get("job_ids", []) if i != job["id"]]
        ids.insert(0, job["id"])
        dropped = ids[MAX_JOB_RECORDS:]
        storage.write_json(_INDEX_KEY, {"job_ids": ids[:MAX_JOB_RECORDS]})
        for old in dropped:
            for key in (_job_key(old), _cancel_key(old)):
                try:
                    storage.delete(key)
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
    except Exception as exc:  # noqa: BLE001 - persistence must not kill a job
        logger.warning("Failed to persist onboarding job", job_id=job["id"], error=str(exc))
        return False
    return True


FINAL_WRITE_ATTEMPTS = 5


def _locked_write(record: dict[str, Any]) -> None:
    with _record_lock:
        _write_record(record)


def _final_write(record: dict[str, Any]) -> None:
    """The terminal record: after any in-flight heartbeat, retried on failure.

    Other instances read a job only from storage, so a lost final write would
    leave it "running" until the heartbeat went stale, then "interrupted"
    without its result. Runs in a worker thread (blocking I/O and sleeps).
    """
    with _record_lock:
        _finished.add(record["id"])
        for attempt in range(FINAL_WRITE_ATTEMPTS):
            if _write_record(record):
                break
            if attempt < FINAL_WRITE_ATTEMPTS - 1:
                time.sleep(min(2**attempt, 10))
        else:
            logger.error("Could not persist a finished job", job_id=record["id"])
    try:
        storage.delete(_cancel_key(record["id"]))
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def _summary(job: dict[str, Any]) -> dict[str, Any]:
    """A job record without its log, for list views.

    Records written before mirror runs existed carry no ``kind``; they were
    all onboarding runs.
    """
    record = {k: v for k, v in job.items() if k != "log"}
    record.setdefault("kind", KIND_ONBOARDING)
    return record


def running_job_id() -> str | None:
    for job_id, job in _jobs.items():
        if job.get("status") == STATUS_RUNNING:
            return job_id
    return None


def _seconds_since(timestamp: Any) -> float | None:
    try:
        then = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (datetime.now(UTC) - then).total_seconds()


def _settle_stored(stored: dict[str, Any]) -> dict[str, Any]:
    """Decide what a stored record this instance holds no process for means.

    Running with a fresh heartbeat: another instance is running it. Running
    without one: the instance that ran it is gone, so it was interrupted.
    """
    if stored.get("status") == STATUS_RUNNING:
        age = _seconds_since(stored.get("heartbeat_at"))
        if age is not None and age < STALE_AFTER_SECONDS:
            stored["running_elsewhere"] = True
        else:
            stored["status"] = STATUS_FAILED
            stored["error"] = "interrupted (the server running it restarted)"
    return stored


def _running_elsewhere() -> str | None:
    """A job another instance is running, from its fresh heartbeat."""
    try:
        index = storage.read_json(_INDEX_KEY) or {}
        for job_id in index.get("job_ids", [])[:5]:
            if job_id in _jobs:
                continue
            stored = storage.read_json(_job_key(job_id)) or {}
            if _settle_stored(stored).get("running_elsewhere"):
                return str(job_id)
    except Exception as exc:  # noqa: BLE001 - never block a start on a read error
        logger.warning("Could not check for jobs on other instances", error=str(exc))
    return None


async def _heartbeat(job_id: str) -> None:
    """Re-persist a running job, and honour a cancel recorded by another instance."""
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        job = _jobs.get(job_id)
        if job is None or job.get("status") != STATUS_RUNNING:
            return
        try:
            cancel_requested = await asyncio.to_thread(storage.exists, _cancel_key(job_id))
        except Exception:  # noqa: BLE001
            cancel_requested = False
        if cancel_requested:
            await cancel_job(job_id)
            return
        job["heartbeat_at"] = _now()
        job["log"] = list(_logs.get(job_id, []))
        # Snapshot on the loop thread; only the storage write leaves it.
        await asyncio.to_thread(_heartbeat_write, _public(job))


def _heartbeat_write(record: dict[str, Any]) -> None:
    with _record_lock:
        if record["id"] not in _finished:
            _write_record(record)


async def _pump_output(job_id: str, process: asyncio.subprocess.Process) -> None:
    """Read the child's merged output into the job's log tail."""
    log = _logs[job_id]
    assert process.stdout is not None
    while True:
        raw = await process.stdout.readline()
        if not raw:
            break
        line = _scrub(raw.decode("utf-8", errors="replace").rstrip("\n"))
        if _is_noise(line):
            continue
        log.append(line)
        job = _jobs.get(job_id)
        if job is not None:
            job["log_line_count"] = job.get("log_line_count", 0) + 1


def _collect_result(job: dict[str, Any]) -> None:
    """Fold a run's JSON summary file into the record, then remove the file."""
    path = job.pop("_summary_path", None)
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            job["result"] = json.load(handle)
    except FileNotFoundError:
        pass  # the run failed before writing one; the log says why
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable job summary", job_id=job.get("id"), error=str(exc))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def _supervise(job_id: str, process: asyncio.subprocess.Process) -> None:
    """Wait for the child, then record its outcome."""
    job = _jobs[job_id]
    beat = asyncio.create_task(_heartbeat(job_id))
    try:
        await _pump_output(job_id, process)
        exit_code = await process.wait()
        job["exit_code"] = exit_code
        if job.get("status") == STATUS_CANCELLED:
            pass  # cancel_job already set the terminal status
        elif exit_code == 0:
            job["status"] = STATUS_SUCCEEDED
        else:
            job["status"] = STATUS_FAILED
            job["error"] = f"exited with code {exit_code}"
    except asyncio.CancelledError:
        job["status"] = STATUS_CANCELLED
        job["error"] = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001
        job["status"] = STATUS_FAILED
        job["error"] = str(exc)
        logger.error("Job crashed", job_id=job_id, error=str(exc))
    finally:
        beat.cancel()
        job["finished_at"] = _now()
        job["log"] = list(_logs.get(job_id, []))
        _collect_result(job)
        _processes.pop(job_id, None)
        # Snapshot on the loop thread; the write (and its lock wait) leaves it.
        await asyncio.shield(asyncio.to_thread(_final_write, _public(job)))
        logger.info(
            "Job finished",
            job_id=job_id,
            status=job.get("status"),
            exit_code=job.get("exit_code"),
        )


async def _spawn(
    argv: list[str],
    job: dict[str, Any],
    *,
    env_overrides: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Start ``argv`` as the one running job; the caller holds ``_start_lock``.

    ``env_overrides`` sets (or, with ``None``, removes) variables in the
    child's copy of the server environment — how secrets reach it without
    appearing in argv.
    """
    job_id = job["id"]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(scripts_dir().parent),  # type: ignore[union-attr]
        )
    except OSError as exc:
        raise JobError(f"Could not start {job.get('script') or 'the job'}: {exc}") from exc

    _jobs[job_id] = job
    _logs[job_id] = deque(maxlen=MAX_LOG_LINES)
    _processes[job_id] = process
    # Under the record lock: the previous job's final write may still be
    # updating the index, and interleaving would drop one of the two.
    await asyncio.to_thread(_locked_write, _public(job))

    # Supervised in the background; the caller gets the job record now.
    task = asyncio.create_task(_supervise(job_id, process))
    job["_task"] = task  # not persisted (see _write_record / _public)
    logger.info(
        "Job started",
        job_id=job_id,
        kind=job.get("kind"),
        site_id=job.get("site_id"),
        script=job.get("script"),
        started_by=job.get("started_by"),
    )
    return _public(job)


def _refuse_if_running() -> None:
    active = running_job_id() or _running_elsewhere()
    if active:
        raise JobError(
            f"Job {active} is already running. Wait for it to finish or cancel it — "
            "onboarding and Fair Store mirror runs share one slot, because a mirror "
            "reads what an onboarding run writes."
        )


def _new_job(**fields: Any) -> dict[str, Any]:
    now = _now()
    return {
        "id": uuid.uuid4().hex[:12],
        "status": STATUS_RUNNING,
        "started_at": now,
        "heartbeat_at": now,
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "log_line_count": 0,
        "log": [],
        **fields,
    }


async def start_job(
    site: dict[str, Any],
    options: dict[str, Any],
    *,
    started_by: str = "admin",
) -> dict[str, Any]:
    """Validate options, spawn the onboarding script, and return the job record."""
    async with _start_lock:
        _refuse_if_running()
        argv, script_name = build_command(site, options)
        job = _new_job(
            kind=KIND_ONBOARDING,
            site_id=site.get("id"),
            site_name=site.get("name"),
            portal_type=site.get("portal_type") or "ckan",
            script=script_name,
            # argv is safe to show: secrets go through the environment.
            command=" ".join(argv[1:]),
            options=dict(options),
            started_by=started_by,
        )
        # The child inherits the server's environment so OPENROUTER_API_KEY and
        # PINECONE_API_KEY reach it without ever appearing in argv.
        return await _spawn(argv, job)


async def start_mirror_job(
    options: dict[str, Any],
    *,
    target_url: str,
    token: str,
    started_by: str = "admin",
) -> dict[str, Any]:
    """Start a Fair Store mirror run (``populate_fairstore.py``).

    ``options``: ``site`` (``"all"`` or a portal ID), ``apply`` (write; the
    default is a dry run), ``enrich`` (Socrata enrichment, default on) and
    ``limit`` (smoke test). Writing needs the sysadmin ``token``.
    """
    if options.get("apply") and not token:
        raise JobError("Writing to the Fair Store needs its sysadmin API token")
    async with _start_lock:
        _refuse_if_running()
        fd, summary_path = tempfile.mkstemp(prefix="fairstore-mirror-", suffix=".json")
        os.close(fd)
        os.unlink(summary_path)  # the script creates it; absent means no summary
        argv, sites = build_mirror_command(
            options, target_url=target_url, summary_path=summary_path
        )
        if len(sites) == 1:
            site_name = sites[0].get("name") or sites[0]["id"]
        elif sites:
            site_name = "Portals: " + ", ".join(str(site["id"]) for site in sites)
        else:
            site_name = "All portals"
        job = _new_job(
            kind=KIND_FAIRSTORE_MIRROR,
            site_id=",".join(str(site["id"]) for site in sites) or "all",
            site_name=site_name,
            portal_type=(sites[0].get("portal_type") or "ckan") if len(sites) == 1 else None,
            script="populate_fairstore.py",
            command=" ".join(argv[1:]),
            options={
                "site": ",".join(str(site["id"]) for site in sites) or "all",
                "apply": bool(options.get("apply")),
                "enrich": options.get("enrich") is not False,
                "limit": options.get("limit") or None,
            },
            target_url=target_url,
            started_by=started_by,
            _summary_path=summary_path,
        )
        return await _spawn(
            argv,
            job,
            env_overrides={
                "FAIRSTORE_API_KEY": token or None,
                "FAIRSTORE_URL": target_url,
            },
        )


def list_jobs(limit: int = 25, *, kind: str | None = None) -> list[dict[str, Any]]:
    """Recent jobs, newest first, without their logs (optionally one ``kind``)."""
    records: list[dict[str, Any]] = [_summary(_public(j)) for j in _jobs.values()]
    seen = {r["id"] for r in records}

    # Fold in jobs persisted by an earlier process lifetime.
    try:
        index = storage.read_json(_INDEX_KEY) or {}
        for job_id in index.get("job_ids", []):
            if job_id in seen:
                continue
            stored = storage.read_json(_job_key(job_id))
            if stored:
                records.append(_summary(_settle_stored(stored)))
                seen.add(job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read onboarding job index", error=str(exc))

    if kind:
        records = [r for r in records if r.get("kind") == kind]
    records.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return records[:limit]


def get_job(job_id: str, *, log_offset: int = 0) -> dict[str, Any] | None:
    """One job with a slice of its log.

    ``log_offset`` is an absolute line number (the previous response's
    ``log_next_offset``), so a UI can poll for just what it has not shown yet,
    from any instance, however long the log grows.
    """
    job = _jobs.get(job_id)
    if job is not None:
        record = _summary(_public(job))
        lines = list(_logs.get(job_id, []))
    else:
        stored = storage.read_json(_job_key(job_id))
        if not stored:
            return None
        record = _summary(_settle_stored(stored))
        lines = list(stored.get("log", []))

    # Absolute line numbers: ``lines`` is the tail ending at log_line_count.
    total = max(int(record.get("log_line_count") or 0), len(lines))
    first = total - len(lines)
    offset = max(0, int(log_offset or 0))
    record["log"] = lines[max(0, offset - first) :] if offset < total else []
    record["log_offset"] = offset
    # Never backwards: a poll answered from a lagging snapshot returns nothing.
    record["log_next_offset"] = max(offset, total)
    record["log_truncated"] = offset < first
    return record


async def cancel_job(job_id: str) -> bool:
    """Terminate a running job.  Returns ``False`` if it was not running.

    For a job another instance is running, a ``<id>.cancel.json`` key is
    written; its owner's next heartbeat terminates it.
    """
    process = _processes.get(job_id)
    job = _jobs.get(job_id)
    if process is None or job is None:
        stored = storage.read_json(_job_key(job_id)) or {}
        if job is None and _settle_stored(dict(stored)).get("running_elsewhere"):
            storage.write_json(_cancel_key(job_id), {"requested_at": _now()})
            # The owner may have finished (and cleared cancels) meanwhile.
            again = storage.read_json(_job_key(job_id)) or {}
            if not _settle_stored(dict(again)).get("running_elsewhere"):
                storage.delete(_cancel_key(job_id))
                return False
            logger.info("Cancel requested for a job on another instance", job_id=job_id)
            return True
        return False
    if job.get("status") != STATUS_RUNNING or process.returncode is not None:
        return False  # already exited; _supervise records how

    # Mark first so _supervise does not overwrite the status with "failed"
    # when the child exits non-zero because we killed it.
    previous = (job.get("status"), job.get("error"))
    job["status"] = STATUS_CANCELLED
    job["error"] = "cancelled by admin"
    try:
        process.terminate()
    except ProcessLookupError:
        job["status"], job["error"] = previous
        return False

    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    logger.info("Job cancelled", job_id=job_id)
    return True


__all__ = [
    "KIND_FAIRSTORE_MIRROR",
    "KIND_ONBOARDING",
    "MAX_LOG_LINES",
    "STATUS_CANCELLED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "TERMINAL_STATUSES",
    "JobError",
    "build_command",
    "build_mirror_command",
    "cancel_job",
    "get_job",
    "list_jobs",
    "running_job_id",
    "runtime_warnings",
    "scripts_dir",
    "start_job",
    "start_mirror_job",
]
