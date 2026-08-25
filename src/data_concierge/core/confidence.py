"""Confidence scoring system for Data Concierge responses.

Overhauled per issue #104 to replace fabricated scores with real signals
extracted from the LLM tool-calling pipeline.

Six-factor model (reweighted for #131):
  notebook_verification   (30%) — generated notebook executed and its output
                                  re-derives the answer's numbers; supplied
                                  asynchronously, gated on
                                  ``notebook_verification_enabled``
  answer_grounding        (25%) — answer claims backed by tool data
  data_retrieval_quality  (20%) — semantic score, rows, success rate
  source_metadata_quality (10%) — resource freshness, record counts
  query_answer_alignment  (10%) — reserved for a real LLM judge
  computation_complexity   (5%) — penalises long/error-prone chains

The legacy ``calculate()`` method is kept for the deterministic Data
Commons graph; the LLM graph uses ``calculate_from_signals()``.
"""

import re
from datetime import datetime
from typing import Any, NamedTuple

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import (
    ConfidenceLevel,
    ConfidenceScore,
    RetrievedData,
    ToolCallSignals,
)

logger = get_logger(__name__)


class ComponentScore(NamedTuple):
    """One confidence factor: either a measured value, or a reason it is not.

    Issue #132. A factor with no input used to be reported as ``0.0`` (or a
    mid-range stand-in), which reads downstream as a measurement rather than
    the absence of one. ``unavailable()`` keeps the two apart.
    """

    value: float | None
    reason: str | None = None

    @classmethod
    def measured(cls, value: float) -> "ComponentScore":
        """A real measurement, clamped and rounded."""
        return cls(round(min(1.0, max(0.0, value)), 4), None)

    @classmethod
    def unavailable(cls, reason: str) -> "ComponentScore":
        """No input to measure. ``reason`` is shown to the user."""
        return cls(None, reason)


class ConfidenceCalculator:
    """Calculator for multi-factor confidence scores.

    Supports two calculation paths:
    - ``calculate()`` — legacy path for the deterministic Data Commons graph
    - ``calculate_from_signals()`` — new path for the LLM-driven graph
    """

    # Pre-assigned authority scores by source (used by legacy path)
    SOURCE_AUTHORITY: dict[str, float] = {
        "bls": 0.95,
        "census": 0.95,
        "ncses": 0.95,
        "bea": 0.95,
        "cdc": 0.92,
        "data_commons": 0.90,
        "ckan": 0.85,
        "wprdc": 0.88,
        "default": 0.80,
    }

    # Expected update frequencies in days (used by legacy path)
    UPDATE_FREQUENCIES: dict[str, int] = {
        "bls": 30,
        "census": 365,
        "ncses": 365,
        "bea": 90,
        "cdc": 30,
        "data_commons": 7,
        "ckan": 30,
        "wprdc": 30,
        "default": 90,
    }

    def __init__(self) -> None:
        """Initialize the confidence calculator."""
        self.logger = get_logger("confidence_calculator")

    # =================================================================
    # New signal-based path (LLM graph)
    # =================================================================

    def calculate_from_signals(
        self,
        *,
        tool_signals: ToolCallSignals,
        final_answer: str,
        tool_results: list[str],
        data_source: str = "",
        portal_quality_score: float = 0.85,
        portal_quality_measured: bool = False,
    ) -> ConfidenceScore:
        """Calculate confidence from real pipeline signals.

        Args:
            tool_signals: Signals captured during LLM tool execution
            final_answer: The LLM's final answer text
            tool_results: List of raw tool result strings
            data_source: Data source ID (for logging)
            portal_quality_score: Quality score from portal config
            portal_quality_measured: True when the score came from a real
                citation rather than the generic default, so the source
                factor knows whether it has anything to measure

        Returns:
            ConfidenceScore with signal-based factors, and an
            ``unavailable`` map for any factor that had no input
        """
        signals_debug: dict[str, Any] = {"data_source": data_source}

        # ── Factor 1: Answer Grounding (30%) ─────────────────────────
        grounding = self._score_answer_grounding(
            final_answer,
            tool_results,
            tool_signals,
            signals_debug,
        )

        # ── Factor 2: Data Retrieval Quality (25%) ───────────────────
        retrieval = self._score_data_retrieval_quality(
            tool_signals,
            portal_quality_score,
            signals_debug,
        )

        # ── Factor 3: Source Metadata Quality (15%) ──────────────────
        metadata_q = self._score_source_metadata_quality(
            tool_signals,
            portal_quality_score,
            portal_quality_measured,
            signals_debug,
        )

        # ── Factor 4: Query-Answer Alignment (15%) ───────────────────
        alignment = self._score_query_answer_alignment(signals_debug)

        # ── Factor 5: Computation Complexity Penalty (15%) ───────────
        complexity = self._score_computation_complexity(
            tool_signals,
            signals_debug,
        )

        unavailable: dict[str, str] = {}
        for name, scored in (
            ("answer_grounding", grounding),
            ("data_retrieval_quality", retrieval),
            ("source_metadata_quality", metadata_q),
            ("query_answer_alignment", alignment),
            ("computation_complexity", complexity),
        ):
            if scored.value is None and scored.reason:
                unavailable[name] = scored.reason

        # Notebook verification (#131) runs after the answer is returned, so
        # it is always pending here. The caller folds the verdict in later via
        # ConfidenceScore.with_notebook_verification(). When both the
        # execution check and the adversarial review are off, omit the factor
        # entirely — telling every user a check is "still running" when
        # nothing will run it is worse than not mentioning it.
        if settings.notebook_verification_enabled or settings.notebook_review_enabled:
            unavailable["notebook_verification"] = ConfidenceScore.NOTEBOOK_PENDING_REASON

        confidence = ConfidenceScore.compute_from_signals(
            answer_grounding=grounding.value,
            data_retrieval_quality=retrieval.value,
            source_metadata_quality=metadata_q.value,
            query_answer_alignment=alignment.value,
            computation_complexity=complexity.value,
            unavailable=unavailable,
            signals=signals_debug,
        )

        self.logger.debug(
            "Signal-based confidence calculated",
            grounding=grounding.value,
            retrieval=retrieval.value,
            metadata_q=metadata_q.value,
            alignment=alignment.value,
            complexity=complexity.value,
            unavailable=sorted(unavailable),
            measured_weight=confidence.measured_weight,
            final=confidence.final_score,
        )

        return confidence

    # -- individual factor scorers ------------------------------------

    @staticmethod
    def _score_answer_grounding(
        final_answer: str,
        tool_results: list[str],
        tool_signals: ToolCallSignals,
        debug: dict[str, Any],
    ) -> ComponentScore:
        """Score how well the answer is backed by data from tool calls.

        Extracts numeric values from the final answer, then checks how
        many appear in any tool result string.

        Grounding is measured by string-matching numbers, so it is only
        measurable when there is an answer, tool output to match against,
        and at least one numeric claim to match.
        """
        if not final_answer:
            debug["grounding_reason"] = "no_answer"
            return ComponentScore.unavailable(
                "Answer grounding could not be checked because no answer text was produced"
            )

        if not any(tool_results):
            debug["grounding_reason"] = "no_tool_output"
            return ComponentScore.unavailable(
                "Answer grounding could not be checked because no tool output was captured "
                "to compare the answer against"
            )

        # Extract numbers from the answer (integers, decimals, percentages,
        # comma-formatted like 1,234,567).  The regex grabs the core
        # numeric token; a trailing period that's just sentence punctuation
        # (e.g. "in 2025.") is stripped post-match.
        raw_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", final_answer)
        answer_numbers = {n.rstrip(".") for n in raw_numbers}
        # Normalise: strip commas for matching
        normalised_answer = {n.replace(",", "") for n in answer_numbers}
        # Also keep originals for comma-formatted matching in results
        normalised_answer |= answer_numbers

        total_claims = len(normalised_answer)

        if total_claims == 0:
            # Nothing numeric to match, so this check cannot run at all.
            # Common for methodology questions and dataset-discovery answers.
            debug["grounding_reason"] = "no_numeric_claims"
            return ComponentScore.unavailable(
                "Answer grounding could not be checked because the answer makes no numeric "
                "claims to verify against the source data"
            )

        # Check how many answer numbers appear in any tool result.
        # Normalise tool results too (strip commas) so "1,234" in the answer
        # matches "1234" in the result and vice versa.
        combined_results = " ".join(tool_results)
        combined_results_normalised = combined_results.replace(",", "")
        grounded = 0
        for num in normalised_answer:
            if num in combined_results or num in combined_results_normalised:
                grounded += 1

        ratio = grounded / total_claims if total_claims else 0.0
        # Scale: 0 grounded → 0.15 (LLM could be summarising), all grounded → 1.0
        score = 0.15 + 0.85 * ratio

        debug["grounding_total_claims"] = total_claims
        debug["grounding_matched"] = grounded
        debug["grounding_ratio"] = round(ratio, 3)
        return ComponentScore.measured(score)

    @staticmethod
    def _score_data_retrieval_quality(
        signals: ToolCallSignals,
        portal_quality_score: float,
        debug: dict[str, Any],
    ) -> ComponentScore:
        """Score retrieval quality from tool call outcomes.

        Composite of whichever of these the run actually produced:
        (a) semantic search relevance (Pinecone score)
        (b) row coverage: were meaningful rows loaded?
        (c) tool success rate: errors reduce trust
        (d) resource diversity: more resources = more evidence

        (a), (b) and (d) do not apply to every source — an MCP-backed
        answer loads no CKAN rows and uses no CKAN resources — so each is
        dropped and the rest reweighted rather than counted as a zero.
        """
        total_calls = signals.successful_tool_calls + signals.failed_tool_calls

        if total_calls == 0:
            debug["retrieval_reason"] = "no_tool_calls"
            return ComponentScore.unavailable(
                "Retrieval quality could not be measured because no data-retrieval tools ran"
            )

        # (c) Success rate — the only sub-signal that always applies.
        success_rate = signals.successful_tool_calls / total_calls
        parts: list[tuple[float, float]] = [(success_rate, 0.35)]

        # (a) Semantic relevance — only when Pinecone search was used.
        if signals.max_semantic_score > 0:
            parts.append((signals.max_semantic_score, 0.35))
            debug["retrieval_semantic"] = round(signals.max_semantic_score, 3)
        else:
            debug["retrieval_semantic"] = "not_used"

        # (b) Row coverage — only when a tool actually loaded tabular rows.
        if signals.total_rows_loaded > 0:
            parts.append((min(1.0, signals.total_rows_loaded / 50), 0.25))
        else:
            debug["retrieval_rows"] = "none_loaded"
        debug["retrieval_rows_count"] = signals.total_rows_loaded

        # (d) Resource diversity — a CKAN notion; absent for MCP sources.
        if signals.distinct_resources_used > 0:
            parts.append((min(1.0, signals.distinct_resources_used / 3), 0.25))
        else:
            debug["retrieval_diversity"] = "not_applicable"

        weight_total = sum(w for _, w in parts)
        score = sum(v * w for v, w in parts) / weight_total

        # Boost by portal quality (curated portals are more trustworthy)
        score = score * (0.7 + 0.3 * portal_quality_score)

        debug["retrieval_success_rate"] = round(success_rate, 3)
        debug["retrieval_subsignals_used"] = len(parts)
        return ComponentScore.measured(score)

    @staticmethod
    def _score_source_metadata_quality(
        signals: ToolCallSignals,
        portal_quality_score: float,
        portal_quality_measured: bool,
        debug: dict[str, Any],
    ) -> ComponentScore:
        """Score resource quality from CKAN metadata.

        Considers: resource freshness (metadata_modified), record counts,
        and portal-level quality score.

        When none of the three is available this used to return the
        hardcoded 0.85 portal default, presenting a fabricated number as a
        measurement. It now reports itself unavailable instead.
        """
        components: list[float] = []

        # Freshness: how recently was the resource modified?
        if signals.resource_metadata_modified:
            try:
                mod_date = datetime.fromisoformat(
                    signals.resource_metadata_modified.replace("Z", "+00:00")
                )
                days_old = (datetime.now() - mod_date.replace(tzinfo=None)).days
                if days_old <= 30:
                    freshness = 1.0
                elif days_old <= 90:
                    freshness = 0.85
                elif days_old <= 365:
                    freshness = 0.7
                elif days_old <= 730:
                    freshness = 0.5
                else:
                    freshness = 0.3
                components.append(freshness)
                debug["metadata_days_old"] = days_old
            except (ValueError, TypeError):
                # Keep a trace — an unparseable date used to vanish silently.
                debug["metadata_freshness"] = "unparseable_date"

        # Record counts — more records = richer dataset
        if signals.resource_record_counts:
            max_records = max(signals.resource_record_counts)
            if max_records >= 10000:
                size_score = 1.0
            elif max_records >= 1000:
                size_score = 0.85
            elif max_records >= 100:
                size_score = 0.7
            elif max_records > 0:
                size_score = 0.5
            else:
                size_score = 0.2
            components.append(size_score)
            debug["metadata_max_records"] = max_records

        # Portal quality score — only a real signal when the run produced a
        # citation to read it from. Otherwise it is the generic 0.85 default.
        if portal_quality_measured:
            components.append(portal_quality_score)
        else:
            debug["metadata_portal_quality"] = "defaulted"

        debug["metadata_components"] = len(components)
        if not components:
            return ComponentScore.unavailable(
                "Source quality could not be measured because the source published no "
                "freshness date, no record counts and no portal quality rating"
            )
        return ComponentScore.measured(sum(components) / len(components))

    @staticmethod
    def _score_query_answer_alignment(debug: dict[str, Any]) -> ComponentScore:
        """Whether the answer actually addresses the question asked.

        Nothing measures this yet. The previous implementation returned
        either a flat 0.3 or ``0.4 + 0.6 * grounding``, the latter being a
        linear restatement of the grounding factor — so 15% of the
        composite was a second copy of the 30% grounding factor rather
        than an independent signal.

        Reporting it unavailable removes that double-count and renormalises
        the composite over the factors that are genuinely measured. Wiring
        up a real check (an LLM judge or embedding similarity) is the
        follow-up.
        """
        debug["alignment_reason"] = "not_implemented"
        return ComponentScore.unavailable(
            "Query alignment is not scored yet — no independent check that the answer "
            "addresses the question has been implemented"
        )

    @staticmethod
    def _score_computation_complexity(
        signals: ToolCallSignals,
        debug: dict[str, Any],
    ) -> ComponentScore:
        """Penalise long or error-prone tool chains.

        More iterations and errors = more chance of accumulated error.
        """
        total_calls = signals.successful_tool_calls + signals.failed_tool_calls

        if total_calls == 0:
            debug["complexity_reason"] = "no_tool_calls"
            return ComponentScore.unavailable(
                "Computation reliability could not be measured because no tool chain ran"
            )

        # Error penalty: each failed call reduces score
        error_rate = signals.failed_tool_calls / total_calls if total_calls else 0.0
        error_penalty = error_rate * 0.5

        # Iteration penalty: more than 4 iterations suggests difficulty
        iteration_penalty = max(0.0, (signals.iterations_used - 4) * 0.06)

        # SQL retry penalty
        sql_penalty = signals.sql_retry_count * 0.08

        score = 1.0 - error_penalty - iteration_penalty - sql_penalty

        debug["complexity_error_rate"] = round(error_rate, 3)
        debug["complexity_iterations"] = signals.iterations_used
        debug["complexity_sql_retries"] = signals.sql_retry_count
        return ComponentScore.measured(score)

    # =================================================================
    # Legacy path (deterministic Data Commons graph)
    # =================================================================

    def calculate(
        self,
        query_confidence: float | None,
        retrieved_data: RetrievedData | None,
        computation_type: str | None = "direct_lookup",
    ) -> ConfidenceScore:
        """Calculate overall confidence score (legacy path).

        Used by the deterministic Data Commons graph where entity
        extraction and structured API responses provide real parse
        confidence, observation counts, and data vintage dates.

        Args:
            query_confidence: Confidence from query parsing, or ``None``
                if the parser never ran
            retrieved_data: Retrieved data with source info
            computation_type: Type of computation performed, or ``None``
                if no computation step ran

        Returns:
            Complete ConfidenceScore, with an ``unavailable`` map naming
            any factor that had no input
        """
        unavailable: dict[str, str] = {}

        if query_confidence is None:
            query_score: float | None = None
            unavailable["query_answer_alignment"] = (
                "Query interpretation could not be scored because the query parser did not run"
            )
        else:
            query_score = min(1.0, max(0.0, query_confidence))

        source_score = self._calculate_source_authority(retrieved_data)
        if source_score.value is None and source_score.reason:
            unavailable["source_metadata_quality"] = source_score.reason

        retrieval_score = self._calculate_retrieval_match(retrieved_data)
        if retrieval_score.value is None and retrieval_score.reason:
            unavailable["data_retrieval_quality"] = retrieval_score.reason

        recency_score = self._calculate_data_recency(retrieved_data)
        if recency_score.value is None and recency_score.reason:
            unavailable["data_recency"] = recency_score.reason

        computation_score = self._calculate_computation_reliability(computation_type)
        if computation_score.value is None and computation_score.reason:
            unavailable["computation_complexity"] = computation_score.reason

        # Pending until the notebook run lands (#131), same as the signal path.
        if settings.notebook_verification_enabled or settings.notebook_review_enabled:
            unavailable["notebook_verification"] = ConfidenceScore.NOTEBOOK_PENDING_REASON

        confidence = ConfidenceScore.compute(
            query=query_score,
            source=source_score.value,
            retrieval=retrieval_score.value,
            recency=recency_score.value,
            computation=computation_score.value,
            unavailable=unavailable,
        )

        self.logger.debug(
            "Legacy confidence calculated",
            query=query_score,
            source=source_score.value,
            retrieval=retrieval_score.value,
            recency=recency_score.value,
            computation=computation_score.value,
            unavailable=sorted(unavailable),
            measured_weight=confidence.measured_weight,
            final=confidence.final_score,
        )

        return confidence

    def _calculate_source_authority(
        self,
        retrieved_data: RetrievedData | None,
    ) -> ComponentScore:
        # Source authority is a property of the source, not of whether the
        # call succeeded. With no source recorded there is nothing to look
        # up — reporting 0.0 would read as "this source has no authority".
        if not retrieved_data or not retrieved_data.source_info:
            return ComponentScore.unavailable(
                "Source authority could not be scored because the retrieval recorded no source"
            )
        scores = [
            self.SOURCE_AUTHORITY.get(
                source.id,
                source.quality_score or self.SOURCE_AUTHORITY["default"],
            )
            for source in retrieved_data.source_info
        ]
        return ComponentScore.measured(sum(scores) / len(scores))

    def _calculate_retrieval_match(
        self,
        retrieved_data: RetrievedData | None,
    ) -> ComponentScore:
        if not retrieved_data:
            return ComponentScore.unavailable(
                "Retrieval quality could not be scored because no data was retrieved"
            )
        base_score = retrieved_data.retrieval_score
        if retrieved_data.observations:
            obs_factor = min(1.0, len(retrieved_data.observations) / 5)
            return ComponentScore.measured(base_score * (0.7 + 0.3 * obs_factor))
        return ComponentScore.measured(base_score * 0.5)

    def _calculate_data_recency(
        self,
        retrieved_data: RetrievedData | None,
    ) -> ComponentScore:
        if not retrieved_data:
            return ComponentScore.unavailable(
                "Data recency could not be scored because no data was retrieved"
            )
        if retrieved_data.data_vintage:
            try:
                vintage_date = datetime.fromisoformat(
                    retrieved_data.data_vintage.replace("Z", "+00:00")
                )
                days_old = (datetime.now() - vintage_date.replace(tzinfo=None)).days
                source_id = (
                    retrieved_data.source_info[0].id if retrieved_data.source_info else "default"
                )
                expected_days = self.UPDATE_FREQUENCIES.get(
                    source_id,
                    self.UPDATE_FREQUENCIES["default"],
                )
                if days_old <= expected_days:
                    return ComponentScore.measured(1.0)
                elif days_old <= expected_days * 2:
                    return ComponentScore.measured(0.8)
                elif days_old <= expected_days * 4:
                    return ComponentScore.measured(0.6)
                else:
                    return ComponentScore.measured(0.4)
            except (ValueError, TypeError):
                return ComponentScore.unavailable(
                    "Data recency could not be scored because the source reported an "
                    "unreadable publication date"
                )
        return ComponentScore.unavailable(
            "Data recency could not be scored because the source published no vintage date"
        )

    def _calculate_computation_reliability(
        self,
        computation_type: str | None,
    ) -> ComponentScore:
        # ``None`` means no computation step ran. This used to default to
        # "direct_lookup", i.e. a perfect 1.0 substituted for "we did not
        # compute anything".
        if not computation_type:
            return ComponentScore.unavailable(
                "Computation reliability could not be scored because no computation step ran"
            )
        reliability_scores = {
            "direct_lookup": 1.0,
            "simple_aggregate": 0.95,
            "comparison": 0.90,
            "trend_analysis": 0.85,
            "derived_calculation": 0.80,
            "statistical_inference": 0.70,
            "llm_analysis": 0.85,
            "unknown": 0.60,
        }
        return ComponentScore.measured(reliability_scores.get(computation_type, 0.60))

    # =================================================================
    # Shared helpers
    # =================================================================

    def get_response_template(
        self,
        confidence: ConfidenceScore,
    ) -> dict[str, Any]:
        """Get appropriate response template based on confidence level."""
        level = confidence.level

        if level == ConfidenceLevel.HIGH:
            return {
                "template": "direct",
                "prefix": "",
                "suffix": "",
                "include_caveats": False,
                "action": "answer_directly",
            }
        elif level == ConfidenceLevel.MEDIUM:
            return {
                "template": "qualified",
                "prefix": "Based on available data, ",
                "suffix": (
                    " (Note: Please verify with the original source "
                    "for the most current information.)"
                ),
                "include_caveats": True,
                "action": "answer_with_caveats",
            }
        elif level == ConfidenceLevel.LOW:
            return {
                "template": "partial",
                "prefix": "I found related data, but ",
                "suffix": "",
                "include_caveats": True,
                "action": "request_clarification",
            }
        else:
            return {
                "template": "unknown",
                "prefix": "I wasn't able to find a confident answer. ",
                "suffix": "",
                "include_caveats": False,
                "action": "offer_alternatives_or_escalate",
            }

    def should_escalate(
        self,
        confidence: ConfidenceScore,
        retrieval_attempts: int,
    ) -> bool:
        """Determine if query should be escalated to human."""
        if (
            confidence.final_score < settings.escalation_confidence_threshold
            and retrieval_attempts >= settings.max_retrieval_attempts
        ):
            return True
        return False


# Global calculator instance
confidence_calculator = ConfidenceCalculator()
