"""Adversarial static review of a generated notebook's method.

Issue #131, third signal. ``agents/notebook_verifier`` answers "does the
notebook run, and do its numbers match the answer" — a mechanical check. This
module answers the question execution cannot: **is the method sound?** A
notebook can execute cleanly, print the answer's exact numbers, and still be
wrong — the number was hardcoded instead of computed, the filter selects the
wrong subset, the aggregation answers a different question, the citation
points at a dataset the code never touches.

This applies to the product's output the same adversarial posture a code
review applies to source: read the artifact assuming it is wrong, and say
concretely where. The review runs in-process via the Anthropic SDK against a
set of guidelines, and returns severity-rated findings.

Design rules, inherited from #131/#132:

* Runs as a **separate signal** merged into the ``notebook_verification``
  confidence factor by ``gateway/notebook_verification``. When the review
  cannot run (no API key, model error, oversized notebook) it reports
  ``score=None`` with a reason — unavailable, never a fabricated zero.
* The **score is derived deterministically from the findings' severities**,
  not asked of the model. Models are much more reliable at naming concrete
  defects than at calibrating a 0-1 number; a fixed deduction schedule keeps
  the factor comparable across queries and testable without the API.
* Never raises: a reviewer fault must degrade to "could not measure" rather
  than take down the verification pass that triggered it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep is absent
    anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False


SEVERITIES = ("critical", "high", "medium", "low")

# Deduction per finding, by severity. A single critical finding (the answer's
# number is hardcoded, the code computes something else entirely) sinks the
# review score on its own — crucially, a hardcoded answer *passes* execution
# and reconciliation, so the review is the only signal that can catch it and
# must be able to speak decisively. Lows are advisory. Deductions accumulate
# and the score floors at 0.0.
_SEVERITY_DEDUCTIONS: dict[str, float] = {
    "critical": 0.90,
    "high": 0.40,
    "medium": 0.15,
    "low": 0.05,
}

# The model reads untrusted content (the notebook embeds text retrieved from
# open data portals), so cap what we send and never let one giant cell blow
# the request. Character budgets, not tokens — cheap and deterministic.
_MAX_CELL_CHARS = 4000
_MAX_NOTEBOOK_CHARS = 60000
_MAX_FINDINGS = 12

_REVIEW_TOOL: dict[str, Any] = {
    "name": "record_review",
    "description": "Record the structured result of the notebook method review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One or two sentences: is the method sound, and why.",
            },
            "findings": {
                "type": "array",
                "description": "Concrete defects found, worst first. Empty if none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": list(SEVERITIES),
                        },
                        "title": {
                            "type": "string",
                            "description": "Short name of the defect",
                        },
                        "detail": {
                            "type": "string",
                            "description": "The failing scenario: what the code does vs "
                            "what the answer claims",
                        },
                        "cell_index": {
                            "type": "integer",
                            "description": "0-based index of the offending cell, if one",
                        },
                    },
                    "required": ["severity", "title", "detail"],
                },
            },
        },
        "required": ["summary", "findings"],
    },
}


class ReviewFinding(BaseModel):
    """One defect the adversarial review found in the notebook's method."""

    severity: str = Field(description="critical | high | medium | low")
    title: str
    detail: str
    cell_index: int | None = None


class ReviewVerdict(BaseModel):
    """Outcome of the adversarial method review of a generated notebook."""

    reviewed: bool = Field(description="The review actually ran and produced findings")
    summary: str | None = None
    findings: list[ReviewFinding] = Field(default_factory=list)
    model: str | None = None
    duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    score: float | None = Field(default=None, description="None when the review could not run")
    reason: str | None = Field(default=None, description="Why the score is None, if it is")

    @property
    def severity_counts(self) -> dict[str, int]:
        """Findings tallied by severity, in fixed severity order."""
        counts = dict.fromkeys(SEVERITIES, 0)
        for f in self.findings:
            if f.severity in counts:
                counts[f.severity] += 1
        return counts

    def as_signals(self) -> dict[str, Any]:
        """Compact debug payload for ``ConfidenceScore.signals``."""
        return {
            "review_ran": self.reviewed,
            "review_findings": len(self.findings),
            "review_severities": {k: v for k, v in self.severity_counts.items() if v},
            "review_model": self.model,
        }


def score_from_findings(findings: list[ReviewFinding]) -> float:
    """Deterministic 0-1 soundness score from severity-rated findings.

    Fixed deductions per severity, floored at zero. Kept as a module-level
    function (not a method) so tests can pin the schedule without an API call.
    """
    total = sum(_SEVERITY_DEDUCTIONS.get(f.severity, 0.0) for f in findings)
    return round(max(0.0, 1.0 - total), 4)


def _cell_excerpt(source: Any) -> str:
    text = "".join(source) if isinstance(source, list) else str(source or "")
    if len(text) > _MAX_CELL_CHARS:
        text = text[:_MAX_CELL_CHARS] + f"\n… [truncated, {len(text)} chars total]"
    return text


def render_notebook_for_review(notebook: dict[str, Any]) -> str:
    """Flatten the notebook into an indexed, size-capped text form.

    Cell indices in the rendering are what ``cell_index`` in findings refers
    to. Outputs are omitted — the review judges the *method*; execution and
    reconciliation are the verifier's job.
    """
    parts: list[str] = []
    total = 0
    for i, cell in enumerate(notebook.get("cells", [])):
        kind = cell.get("cell_type", "code")
        excerpt = _cell_excerpt(cell.get("source"))
        block = f"--- cell {i} ({kind}) ---\n{excerpt}\n"
        total += len(block)
        if total > _MAX_NOTEBOOK_CHARS:
            parts.append("--- remaining cells omitted, notebook exceeds review size cap ---")
            break
        parts.append(block)
    return "\n".join(parts)


def _build_user_message(query: str, answer: str, notebook: dict[str, Any]) -> str:
    return (
        "Review this generated notebook against the question and the answer "
        "shipped with it.\n\n"
        f"## User's question\n{(query or '').strip()[:2000]}\n\n"
        f"## Answer shipped to the user\n{(answer or '').strip()[:4000]}\n\n"
        f"## Notebook cells\n{render_notebook_for_review(notebook)}"
    )


def _effective_model() -> str:
    return settings.notebook_review_model or settings.llm_model


def review_available() -> bool:
    """Whether the adversarial review can currently run."""
    if not settings.notebook_review_enabled:
        return False
    if not ANTHROPIC_AVAILABLE:
        return False
    return bool(settings.anthropic_api_key.get_secret_value())


# Lazily-constructed async Anthropic client (mirrors gateway/match_verifier.py).
_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return _client


def _parse_review_input(tool_input: dict[str, Any]) -> tuple[str, list[ReviewFinding]]:
    """Coerce the model's tool input into a summary + validated findings.

    Tolerant of shape drift: unknown severities are clamped to ``medium``,
    findings beyond the cap are dropped (the worst survive because the tool
    schema asks for worst-first ordering).
    """
    summary = str(tool_input.get("summary") or "").strip()
    findings: list[ReviewFinding] = []
    raw = tool_input.get("findings")
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "").lower().strip()
        if severity not in SEVERITIES:
            severity = "medium"
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if not title and not detail:
            continue
        cell_index = item.get("cell_index")
        findings.append(
            ReviewFinding(
                severity=severity,
                title=title or detail[:80],
                detail=detail or title,
                cell_index=cell_index if isinstance(cell_index, int) else None,
            )
        )
        if len(findings) >= _MAX_FINDINGS:
            break
    return summary, findings


def _unavailable(reason: str) -> ReviewVerdict:
    return ReviewVerdict(reviewed=False, score=None, reason=reason)


async def review_notebook(
    notebook: dict[str, Any],
    query: str,
    answer: str,
) -> ReviewVerdict:
    """Adversarially review ``notebook``'s method against ``answer``.

    Never raises. Returns ``score=None`` with a reason whenever the review
    could not actually run, so the confidence machinery reports the signal
    unavailable instead of folding in a fabricated number.
    """
    if not settings.notebook_review_enabled:
        return _unavailable("Notebook review is disabled")
    if not ANTHROPIC_AVAILABLE or not settings.anthropic_api_key.get_secret_value():
        return _unavailable(
            "Notebook review could not run because the review model is not configured"
        )
    if not notebook or not notebook.get("cells"):
        return _unavailable(
            "Notebook review could not run because the notebook has no cells to review"
        )

    from data_concierge.gateway.system_prompt import (
        DEFAULT_NOTEBOOK_REVIEW_TEMPLATE,
        get_notebook_review_template,
    )

    try:
        system_prompt = get_notebook_review_template().format()
    except (KeyError, IndexError, ValueError) as e:
        logger.warning("Custom review prompt failed to render; using default", error=str(e))
        system_prompt = DEFAULT_NOTEBOOK_REVIEW_TEMPLATE

    model = _effective_model()
    started = asyncio.get_running_loop().time()
    try:
        client = _get_client()
        # No sampling params: the 5-family models (claude-sonnet-5 et al.)
        # reject temperature/top_p/top_k with a 400.
        response = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                tools=[_REVIEW_TOOL],
                tool_choice={"type": "tool", "name": "record_review"},
                messages=[
                    {
                        "role": "user",
                        "content": _build_user_message(query, answer, notebook),
                    }
                ],
            ),
            timeout=settings.notebook_review_timeout_seconds,
        )
    except TimeoutError:
        logger.warning("Notebook review timed out", model=model)
        return _unavailable("Notebook review could not complete within its time limit")
    except Exception as exc:  # noqa: BLE001 - reviewer must never break verification
        logger.warning("Notebook review failed", model=model, error=str(exc))
        return _unavailable(
            "Notebook review could not run because the review model was unavailable"
        )

    duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)

    tool_input: dict[str, Any] | None = None
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == (
            "record_review"
        ):
            candidate = getattr(block, "input", None)
            if isinstance(candidate, dict):
                tool_input = candidate
            break

    if tool_input is None:
        logger.warning(
            "Notebook review returned no structured verdict",
            model=model,
            stop_reason=getattr(response, "stop_reason", None),
        )
        return _unavailable(
            "Notebook review could not run because the review model returned no verdict"
        )

    summary, findings = _parse_review_input(tool_input)
    usage = getattr(response, "usage", None)

    verdict = ReviewVerdict(
        reviewed=True,
        summary=summary or None,
        findings=findings,
        model=str(getattr(response, "model", model)),
        duration_ms=duration_ms,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        score=score_from_findings(findings),
        reason=None,
    )
    logger.info(
        "Notebook review complete",
        model=verdict.model,
        findings=len(findings),
        severities=json.dumps(verdict.severity_counts),
        score=verdict.score,
        duration_ms=duration_ms,
    )
    return verdict
