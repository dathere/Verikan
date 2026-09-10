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

**Cloud Run caveat.**  A child process keeps running only while the instance
is alive and scheduled.  Without ``--no-cpu-throttling`` the CPU is throttled
to near zero between requests and a background job crawls; with
``--min-instances 0`` the instance can be reclaimed outright and the job dies.
:func:`runtime_warnings` reports this so the UI can say so up front rather than
letting an admin wonder why a job stalled.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
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


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields before a record leaves this module."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _job_key(job_id: str) -> str:
    return f"{_STORAGE_PREFIX}/{job_id}.json"


def _persist(job: dict[str, Any]) -> None:
    """Write a job record, and keep the index of recent jobs trimmed.

    ``_public`` is not optional here: the live record carries the supervising
    asyncio Task under ``_task``, which is not JSON-serializable, and writing
    the raw dict silently loses every completed job.
    """
    try:
        storage.write_json(_job_key(job["id"]), _public(job))
        index = storage.read_json(_INDEX_KEY) or {}
        ids = [i for i in index.get("job_ids", []) if i != job["id"]]
        ids.insert(0, job["id"])
        dropped = ids[MAX_JOB_RECORDS:]
        storage.write_json(_INDEX_KEY, {"job_ids": ids[:MAX_JOB_RECORDS]})
        for old in dropped:
            try:
                storage.delete(_job_key(old))
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
    except Exception as exc:  # noqa: BLE001 - persistence must not kill a job
        logger.warning("Failed to persist onboarding job", job_id=job["id"], error=str(exc))


def _summary(job: dict[str, Any]) -> dict[str, Any]:
    """A job record without its log, for list views."""
    return {k: v for k, v in job.items() if k != "log"}


def running_job_id() -> str | None:
    for job_id, job in _jobs.items():
        if job.get("status") == STATUS_RUNNING:
            return job_id
    return None


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


async def _supervise(job_id: str, process: asyncio.subprocess.Process) -> None:
    """Wait for the child, then record its outcome."""
    job = _jobs[job_id]
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
        logger.error("Onboarding job crashed", job_id=job_id, error=str(exc))
    finally:
        job["finished_at"] = _now()
        job["log"] = list(_logs.get(job_id, []))
        _processes.pop(job_id, None)
        _persist(job)
        logger.info(
            "Onboarding job finished",
            job_id=job_id,
            status=job.get("status"),
            exit_code=job.get("exit_code"),
        )


async def start_job(
    site: dict[str, Any],
    options: dict[str, Any],
    *,
    started_by: str = "admin",
) -> dict[str, Any]:
    """Validate options, spawn the onboarding script, and return the job record."""
    async with _start_lock:
        active = running_job_id()
        if active:
            raise JobError(
                f"Job {active} is already running. Wait for it to finish or cancel it "
                "— concurrent runs would interleave downloads into the same directory."
            )

        argv, script_name = build_command(site, options)
        job_id = uuid.uuid4().hex[:12]

        # The child inherits the server's environment so OPENROUTER_API_KEY and
        # PINECONE_API_KEY reach it without ever appearing in argv.
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        job: dict[str, Any] = {
            "id": job_id,
            "site_id": site.get("id"),
            "site_name": site.get("name"),
            "portal_type": site.get("portal_type") or "ckan",
            "script": script_name,
            # argv is safe to show: secrets go through the environment.
            "command": " ".join(argv[1:]),
            "options": dict(options),
            "status": STATUS_RUNNING,
            "started_by": started_by,
            "started_at": _now(),
            "finished_at": None,
            "exit_code": None,
            "error": None,
            "log_line_count": 0,
            "log": [],
        }

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=str(scripts_dir().parent),  # type: ignore[union-attr]
            )
        except OSError as exc:
            raise JobError(f"Could not start the onboarding script: {exc}") from exc

        _jobs[job_id] = job
        _logs[job_id] = deque(maxlen=MAX_LOG_LINES)
        _processes[job_id] = process
        _persist(job)

        # Supervised in the background; the caller gets the job record now.
        task = asyncio.create_task(_supervise(job_id, process))
        job["_task"] = task  # not persisted (see _persist / _summary consumers)
        logger.info(
            "Onboarding job started",
            job_id=job_id,
            site_id=site.get("id"),
            script=script_name,
            started_by=started_by,
        )
        return _public(job)


def list_jobs(limit: int = 25) -> list[dict[str, Any]]:
    """Recent jobs, newest first, without their logs."""
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
                # A job recorded as running that we have no process for did not
                # survive a restart; reporting it as running would be a lie.
                if stored.get("status") == STATUS_RUNNING:
                    stored["status"] = STATUS_FAILED
                    stored["error"] = "interrupted (server restarted during the run)"
                records.append(_summary(stored))
                seen.add(job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read onboarding job index", error=str(exc))

    records.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return records[:limit]


def get_job(job_id: str, *, log_offset: int = 0) -> dict[str, Any] | None:
    """One job with a slice of its log.

    ``log_offset`` is a line index into the retained tail, so a UI can poll for
    just what it has not shown yet instead of refetching the whole log.
    """
    job = _jobs.get(job_id)
    if job is not None:
        record = _summary(_public(job))
        lines = list(_logs.get(job_id, []))
    else:
        stored = storage.read_json(_job_key(job_id))
        if not stored:
            return None
        if stored.get("status") == STATUS_RUNNING:
            stored["status"] = STATUS_FAILED
            stored["error"] = "interrupted (server restarted during the run)"
        record = _summary(stored)
        lines = list(stored.get("log", []))

    offset = max(0, min(int(log_offset or 0), len(lines)))
    record["log"] = lines[offset:]
    record["log_offset"] = offset
    record["log_next_offset"] = len(lines)
    record["log_truncated"] = record.get("log_line_count", 0) > len(lines)
    return record


async def cancel_job(job_id: str) -> bool:
    """Terminate a running job.  Returns ``False`` if it was not running."""
    process = _processes.get(job_id)
    job = _jobs.get(job_id)
    if process is None or job is None or job.get("status") != STATUS_RUNNING:
        return False

    # Mark first so _supervise does not overwrite the status with "failed"
    # when the child exits non-zero because we killed it.
    job["status"] = STATUS_CANCELLED
    job["error"] = "cancelled by admin"
    try:
        process.terminate()
    except ProcessLookupError:
        return False

    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    logger.info("Onboarding job cancelled", job_id=job_id)
    return True


__all__ = [
    "MAX_LOG_LINES",
    "STATUS_CANCELLED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_SUCCEEDED",
    "TERMINAL_STATUSES",
    "JobError",
    "build_command",
    "cancel_job",
    "get_job",
    "list_jobs",
    "running_job_id",
    "runtime_warnings",
    "scripts_dir",
    "start_job",
]
