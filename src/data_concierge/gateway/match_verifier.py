"""LLM-based verification gate for verified-answer matching.

Keyword overlap (Jaccard + query coverage) is fast but semantically blind: a
query for "crime in 2023" scores highly against a verified answer for "crime in
2025" because the topic terms dominate and the differing year is just one token
(issue #64). The same false-positive mode applies to geographic mismatches
("Texas" vs "California"), variable mismatches ("population" vs "poverty"), and
scope mismatches ("violent crime" vs "all crime").

This module adds Stage 2 of a two-stage pipeline: after cheap keyword retrieval
narrows the field to a few candidates, a low-latency model (Claude Haiku by
default) judges whether each candidate verified answer *actually* answers the
new question. It is deliberately conservative — any topic / geography / time /
specificity mismatch is a non-match.

The gate degrades gracefully: if the Anthropic client is unavailable or a
configured number of consecutive calls fail, a process-level circuit breaker
opens and callers fall back to stricter keyword-only matching.
"""

from __future__ import annotations

import json
import re
from typing import Any

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep is absent
    anthropic = None  # type: ignore[assignment]
    ANTHROPIC_AVAILABLE = False


MATCH_VERIFICATION_PROMPT = """\
You are a data matching assistant. Determine whether a previously verified \
answer can correctly answer a user's NEW question.

A match is ONLY valid if ALL of these are true:
1. Same topic/variable (e.g., both about "crime", both about "unemployment").
2. Same geographic scope (e.g., both about "Pittsburgh", both about "Texas").
3. Compatible time period. An EXPLICIT year in the new question (e.g. "2023") \
must be covered by the verified answer. A verified answer phrased with a \
RELATIVE date ("last year", "recently") does NOT satisfy an explicit-year \
request unless its data clearly covers that exact year.
4. Same level of specificity (e.g., "violent crime" is NOT "all crime").

Verified answer's original question: {verified_query}
Verified answer text: {verified_answer}

User's NEW question: {user_query}

Respond with ONLY a JSON object, no other text:
{{"is_match": true or false, "confidence": 0.0 to 1.0, "reason": "one sentence"}}
"""


class _CircuitBreaker:
    """Trips open after N consecutive failures to avoid hammering a dead API."""

    def __init__(self) -> None:
        self._consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        threshold = settings.verified_match_circuit_breaker_threshold
        return self._consecutive_failures >= threshold

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        self._consecutive_failures += 1

    def reset(self) -> None:
        self._consecutive_failures = 0


# Module-level breaker shared across verification calls within a process.
_breaker = _CircuitBreaker()

# Lazily-constructed async Anthropic client (mirrors agents/llm_agent.py).
_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed")
        api_key = settings.anthropic_api_key.get_secret_value()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


def _parse_response(text: str) -> dict[str, Any] | None:
    """Extract the JSON verdict from the model response.

    Tolerant of leading/trailing prose or code fences around the JSON object.
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or "is_match" not in data:
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "is_match": bool(data.get("is_match")),
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(data.get("reason", "")).strip(),
    }


def llm_gate_available() -> bool:
    """Whether the LLM verification gate can currently be used."""
    if not settings.verified_match_llm_enabled:
        return False
    if not ANTHROPIC_AVAILABLE:
        return False
    if not settings.anthropic_api_key.get_secret_value():
        return False
    return not _breaker.is_open


async def verify_match_with_llm(
    user_query: str,
    verified_query: str,
    verified_answer: str,
) -> dict[str, Any] | None:
    """Ask the low-latency model whether a verified answer matches a new query.

    Returns a dict ``{"is_match": bool, "confidence": float, "reason": str}``
    on success, or ``None`` when the gate is unavailable / errored (callers
    should fall back to keyword-only matching in that case).
    """
    if not llm_gate_available():
        return None

    prompt = MATCH_VERIFICATION_PROMPT.format(
        verified_query=verified_query,
        # Truncate the answer body — only enough context to judge topic/scope.
        verified_answer=(verified_answer or "")[:1500],
        user_query=user_query,
    )

    try:
        client = _get_client()
        response = await client.messages.create(
            model=settings.verified_match_model,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on any API error
        _breaker.record_failure()
        logger.warning(
            "Verified-match LLM gate failed; falling back to keyword matching",
            error=str(exc),
            consecutive_failures=_breaker._consecutive_failures,
            circuit_open=_breaker.is_open,
        )
        return None

    _breaker.record_success()

    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")

    verdict = _parse_response(text)
    if verdict is None:
        logger.warning(
            "Verified-match LLM gate returned unparseable response",
            raw=text[:200],
        )
        return None

    logger.info(
        "Verified-match LLM verdict",
        is_match=verdict["is_match"],
        confidence=verdict["confidence"],
        reason=verdict["reason"][:120],
    )
    return verdict
