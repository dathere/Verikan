"""Schedule, persist and serve notebook verification results.

Issue #131. ``agents/notebook_verifier`` knows how to execute a notebook and
reconcile it against the answer; ``agents/notebook_reviewer`` knows how to
adversarially review its method (the roborev signal); this module decides
*when* both happen and where the combined verdict lives.

Verification runs **after** the answer is returned. Executing a notebook takes
seconds to minutes, and no user should wait that long for a confidence number,
so ``/query`` responds immediately with ``notebook_verification`` pending and
the verdict is folded in later via
:meth:`ConfidenceScore.with_notebook_verification`.

The stored ``notebook_verification/{query_id}.json`` record carries both
signals (``verdict`` for execution/reconciliation, ``review`` for the
adversarial method review) plus the query context, so the admin panel's
Notebook Reviews pane can list them without a second index.

Concurrency is capped. Each verification spawns a subprocess running a Jupyter
kernel, which is heavy; without a limit a burst of queries would exhaust the
container. Work beyond the cap queues rather than piling up. The review is an
API call and rides inside the same slot.

Cloud Run caveat
----------------
Cloud Run throttles CPU outside a request's scope unless the service is
configured with CPU always allocated. A verification scheduled as a background
task can therefore stall, or be lost entirely if the instance is scaled down
before it finishes. Two consequences:

* a verdict may never arrive, so callers must treat "pending" as a state that
  can persist indefinitely rather than a promise;
* before enabling this at any volume, either set CPU always-allocated on the
  service, or move execution to a Cloud Tasks / Pub-Sub worker that runs
  inside a real request.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from data_concierge.agents.notebook_reviewer import ReviewVerdict, review_notebook
from data_concierge.agents.notebook_verifier import NotebookVerdict, verify_notebook
from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import ConfidenceScore
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

_KEY_PREFIX = "notebook_verification"

# Each verification is a Jupyter kernel in a subprocess. Two at a time keeps
# a burst of queries from starving the web workers.
_MAX_CONCURRENT = 2
_semaphore: asyncio.Semaphore | None = None

# Tasks are kept referenced until they finish; asyncio only holds a weak
# reference to a bare create_task() result, so without this the garbage
# collector can cancel work mid-flight.
_in_flight: set[asyncio.Task[None]] = set()

# Within the combined notebook_verification factor, execution+reconciliation
# outweighs the static review: re-running the code and getting the answer's
# numbers is direct evidence, an adversarial read is judgement. When one side
# could not be measured the other carries the factor alone (renormalised),
# mirroring how the composite treats unavailable factors (#132).
_EXECUTION_WEIGHT = 0.7
_REVIEW_WEIGHT = 0.3


def _get_semaphore() -> asyncio.Semaphore:
    """Lazily bind the semaphore to the running loop."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


def _key(query_id: str) -> str:
    return f"{_KEY_PREFIX}/{query_id}.json"


def get_verification(query_id: str) -> dict[str, Any] | None:
    """Stored verdict for a query, or ``None`` if it has not landed."""
    try:
        return storage.read_json(_key(query_id))
    except Exception as e:  # noqa: BLE001 - a read failure must not 500 the caller
        logger.warning("Could not read notebook verification", query_id=query_id, error=str(e))
        return None


# Keys are uuids (no time ordering), so correct newest-first sorting needs
# every record read. Cap the reads so an admin opening the pane on a
# long-lived deployment doesn't trigger thousands of storage round-trips.
_MAX_RECORDS_READ = 1000


def list_verifications(limit: int = 100) -> list[dict[str, Any]]:
    """Stored verification/review records, newest first.

    Backs the admin Notebook Reviews pane. Reads records under the prefix
    (each is a small JSON document), sorts by ``completed_at``, and returns
    at most ``limit``. Reads are capped at ``_MAX_RECORDS_READ``; past that
    the ordering degrades to an arbitrary subset rather than the pane
    blocking on unbounded I/O. Callers on an event loop should invoke this
    via ``asyncio.to_thread`` — every read is blocking storage I/O.
    """
    records: list[dict[str, Any]] = []
    try:
        keys = storage.list_keys(_KEY_PREFIX, suffix=".json")
    except Exception as e:  # noqa: BLE001 - a listing failure must not 500 the caller
        logger.warning("Could not list notebook verifications", error=str(e))
        return []
    if len(keys) > _MAX_RECORDS_READ:
        logger.warning(
            "Notebook verification records exceed the read cap; listing a subset",
            total=len(keys),
            cap=_MAX_RECORDS_READ,
        )
        keys = keys[:_MAX_RECORDS_READ]
    for key in keys:
        try:
            record = storage.read_json(key)
        except Exception:  # noqa: BLE001 - skip unreadable records, keep the rest
            continue
        if isinstance(record, dict):
            records.append(record)
    records.sort(key=lambda r: str(r.get("completed_at") or ""), reverse=True)
    return records[: max(0, limit)]


def combine_scores(
    execution_score: float | None,
    review_score: float | None,
    execution_reason: str | None = None,
    review_reason: str | None = None,
    *,
    review_has_critical: bool = False,
) -> tuple[float | None, str | None]:
    """Merge execution/reconciliation and review into one factor score.

    Weighted 70/30 when both measured; either alone carries the factor when
    the other is unavailable; ``(None, reason)`` when neither ran. Never
    substitutes a stand-in constant for an unmeasured side (#132).

    A **critical** review finding caps the combined factor at the review
    score. The critical class exists for defects execution cannot see — a
    hardcoded answer executes cleanly and reconciles perfectly — so clean
    mechanical signals must not be allowed to outvote it.

    Note the deliberate asymmetry of the one-side-only branches: when only
    the review measured (e.g. the execution environment is broken), its
    score carries the whole factor at face value — same renormalisation the
    composite applies to whole factors. The stored record keeps both
    sub-verdicts, so the admin pane still shows that execution never ran.
    """
    if execution_score is not None and review_score is not None:
        combined = execution_score * _EXECUTION_WEIGHT + review_score * _REVIEW_WEIGHT
        if review_has_critical:
            combined = min(combined, review_score)
        return round(combined, 4), None
    if execution_score is not None:
        return execution_score, None
    if review_score is not None:
        return review_score, None
    reasons = [r for r in (execution_reason, review_reason) if r]
    return None, (
        ". ".join(dict.fromkeys(reasons)) or "Notebook verification could not be completed"
    )


def _persist(
    query_id: str,
    verdict: NotebookVerdict | None,
    confidence: ConfidenceScore | None,
    *,
    status: str,
    error: str | None = None,
    review: ReviewVerdict | None = None,
    combined_score: float | None = None,
    query: str | None = None,
    data_source: str | None = None,
    confidence_before: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "query_id": query_id,
        "status": status,
        "completed_at": datetime.now().isoformat(),
        "error": error,
        "query": (query or "")[:500] or None,
        "data_source": data_source,
        "verdict": verdict.model_dump() if verdict else None,
        "review": review.model_dump() if review else None,
        "combined_score": combined_score,
        "confidence_before": confidence_before,
    }
    if confidence is not None:
        payload["confidence"] = confidence.final_score
        payload["confidence_level"] = confidence.level.value
        payload["confidence_measured_weight"] = confidence.measured_weight
        payload["confidence_explanation"] = confidence.explanation
        payload["confidence_unavailable"] = dict(confidence.unavailable)
        payload["confidence_breakdown"] = {
            "notebook_verification": confidence.notebook_verification,
            "answer_grounding": confidence.answer_grounding,
            "data_retrieval_quality": confidence.data_retrieval_quality,
            "source_metadata_quality": confidence.source_metadata_quality,
            "query_answer_alignment": confidence.query_answer_alignment,
            "computation_complexity": confidence.computation_complexity,
        }
    try:
        storage.write_json(_key(query_id), payload)
    except Exception as e:  # noqa: BLE001 - losing a verdict must not break anything
        logger.warning("Could not persist notebook verification", query_id=query_id, error=str(e))


async def _run(
    query_id: str,
    notebook: dict[str, Any],
    answer: str,
    confidence: ConfidenceScore | None,
    query: str,
    data_source: str | None,
) -> None:
    """Execute + review one notebook and store the updated confidence."""
    confidence_before = confidence.final_score if confidence is not None else None

    async def _safe_execute() -> NotebookVerdict:
        """Execution side, guaranteed not to raise (so gather can't leak the
        review task by propagating an exception past it)."""
        if not settings.notebook_verification_enabled:
            # Review-only mode: the operator disabled execution (e.g. no
            # always-allocated CPU) but the API-only review still runs.
            return NotebookVerdict(
                executed=False,
                score=None,
                reason="Notebook execution is disabled on this deployment",
            )
        try:
            # verify_notebook blocks on subprocess.run, so keep it off the loop.
            return await asyncio.to_thread(
                verify_notebook,
                notebook,
                answer,
                total_timeout=settings.notebook_verification_timeout_seconds,
                cell_timeout=settings.notebook_verification_cell_timeout_seconds,
            )
        except Exception as e:  # noqa: BLE001 - never let a verifier fault escape
            logger.warning("Notebook execution check failed", query_id=query_id, error=str(e))
            return NotebookVerdict(
                executed=False,
                execution_error=str(e)[:500],
                score=None,
                reason="Notebook verification could not run because the verifier itself failed",
            )

    async def _safe_review() -> ReviewVerdict:
        """Review side; review_notebook already never raises, but keep the
        same guarantee explicitly for gather's sake."""
        try:
            return await review_notebook(notebook, query, answer)
        except Exception as e:  # noqa: BLE001 - reviewer fault must not sink the pass
            logger.warning("Notebook review crashed", query_id=query_id, error=str(e))
            return ReviewVerdict(
                reviewed=False,
                score=None,
                reason="Notebook review could not run because the reviewer itself failed",
            )

    async with _get_semaphore():
        # Both sides are exception-proof, so gather cannot raise and neither
        # task can be left running detached outside the semaphore.
        verdict, review = await asyncio.gather(_safe_execute(), _safe_review())

        combined_score, combined_reason = combine_scores(
            verdict.score,
            review.score,
            verdict.reason,
            review.reason,
            review_has_critical=review.severity_counts.get("critical", 0) > 0,
        )
        merged_signals = {**verdict.as_signals(), **review.as_signals()}

        updated = (
            confidence.with_notebook_verification(combined_score, combined_reason, merged_signals)
            if confidence is not None
            else None
        )

        _persist(
            query_id,
            verdict,
            updated,
            status="complete",
            review=review,
            combined_score=combined_score,
            query=query,
            data_source=data_source,
            confidence_before=confidence_before,
        )
        logger.info(
            "Notebook verification complete",
            query_id=query_id,
            executed=verdict.executed,
            reconciliation=verdict.reconciliation_ratio,
            notebook_score=verdict.score,
            review_score=review.score,
            review_findings=len(review.findings),
            combined_score=combined_score,
            confidence_before=confidence_before,
            confidence_after=updated.final_score if updated else None,
        )


def schedule_verification(
    query_id: str,
    notebook: dict[str, Any] | None,
    answer: str,
    confidence: ConfidenceScore | None,
    query: str = "",
    data_source: str | None = None,
) -> bool:
    """Kick off verification + review for a query. Returns whether scheduled.

    Safe to call unconditionally: it is a no-op when both checks are off,
    when there is no notebook, or when there is no running event loop.
    Runs when EITHER check is enabled — an operator who disables execution
    (the heavyweight subprocess) still gets the API-only adversarial review,
    with the execution side reported unavailable rather than skipped
    silently.
    """
    if not (settings.notebook_verification_enabled or settings.notebook_review_enabled):
        return False
    if not notebook or not notebook.get("cells"):
        return False

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("No running loop; skipping notebook verification", query_id=query_id)
        return False

    _persist(
        query_id,
        None,
        confidence,
        status="pending",
        query=query,
        data_source=data_source,
        confidence_before=confidence.final_score if confidence is not None else None,
    )

    task = loop.create_task(_run(query_id, notebook, answer, confidence, query, data_source))
    _in_flight.add(task)
    task.add_done_callback(_in_flight.discard)
    logger.info("Notebook verification scheduled", query_id=query_id, in_flight=len(_in_flight))
    return True


def review_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats over verification records for the admin pane."""
    total = len(records)
    complete = [r for r in records if r.get("status") == "complete"]
    executed_ok = 0
    reviewed = 0
    findings_by_severity: dict[str, int] = {}
    combined_scores: list[float] = []
    for r in complete:
        verdict = r.get("verdict") or {}
        if verdict.get("executed"):
            executed_ok += 1
        review = r.get("review") or {}
        if review.get("reviewed"):
            reviewed += 1
            for f in review.get("findings") or []:
                sev = str(f.get("severity") or "unknown")
                findings_by_severity[sev] = findings_by_severity.get(sev, 0) + 1
        if isinstance(r.get("combined_score"), int | float):
            combined_scores.append(float(r["combined_score"]))
    return {
        "total": total,
        "complete": len(complete),
        "pending": sum(1 for r in records if r.get("status") == "pending"),
        "errors": sum(1 for r in records if r.get("status") == "error"),
        "executed_ok": executed_ok,
        "reviewed": reviewed,
        "findings_by_severity": findings_by_severity,
        "avg_combined_score": (
            round(sum(combined_scores) / len(combined_scores), 4) if combined_scores else None
        ),
    }
