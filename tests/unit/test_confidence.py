"""Unit tests for the overhauled confidence scoring system (issue #104).

Tests both the new signal-based path (LLM graph) and the legacy path
(deterministic Data Commons graph) to ensure backward compatibility.
"""

import pytest

from data_concierge.core.confidence import ConfidenceCalculator
from data_concierge.core.models import (
    ConfidenceLevel,
    ConfidenceScore,
    DataSource,
    RetrievedData,
    ToolCallSignals,
)

# ────────────────────────────────────────────────────────────────────
# ConfidenceScore model tests
# ────────────────────────────────────────────────────────────────────


class TestConfidenceScoreModel:
    """Tests for the ConfidenceScore Pydantic model."""

    def test_compute_from_signals_weights(self) -> None:
        """Verify the new weighted formula: 30/25/15/15/15."""
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=1.0,
            data_retrieval_quality=1.0,
            source_metadata_quality=1.0,
            query_answer_alignment=1.0,
            computation_complexity=1.0,
        )
        assert score.final_score == pytest.approx(1.0, abs=0.01)

        score_zero = ConfidenceScore.compute_from_signals(
            answer_grounding=0.0,
            data_retrieval_quality=0.0,
            source_metadata_quality=0.0,
            query_answer_alignment=0.0,
            computation_complexity=0.0,
        )
        assert score_zero.final_score == pytest.approx(0.0, abs=0.01)

    def test_compute_from_signals_specific_weights(self) -> None:
        """answer_grounding carries 25% (#131 reweighting, was 30%).

        notebook_verification is absent here, so the composite renormalises
        over the 70% that was measured: 0.25 / 0.70.
        """
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=1.0,
            data_retrieval_quality=0.0,
            source_metadata_quality=0.0,
            query_answer_alignment=0.0,
            computation_complexity=0.0,
        )
        assert score.measured_weight == pytest.approx(0.70)
        assert score.final_score == pytest.approx(0.25 / 0.70, abs=0.01)

    def test_notebook_verification_is_the_heaviest_factor(self) -> None:
        """#131: re-deriving the answer outweighs matching the transcript."""
        nb_only = ConfidenceScore.compute_from_signals(
            answer_grounding=None, data_retrieval_quality=None,
            source_metadata_quality=None, query_answer_alignment=None,
            computation_complexity=None, notebook_verification=1.0,
            unavailable={"answer_grounding": "x"},
        )
        grounding_only = ConfidenceScore.compute_from_signals(
            answer_grounding=1.0, data_retrieval_quality=None,
            source_metadata_quality=None, query_answer_alignment=None,
            computation_complexity=None,
            unavailable={"notebook_verification": "x"},
        )
        assert nb_only.measured_weight > grounding_only.measured_weight
        assert nb_only.measured_weight == pytest.approx(0.30)
        assert grounding_only.measured_weight == pytest.approx(0.25)

    def test_legacy_compute_still_works(self) -> None:
        """The old 5-arg compute() must keep working for backward compat."""
        score = ConfidenceScore.compute(0.8, 0.9, 0.7, 0.6, 0.95)
        expected = 0.25 * 0.8 + 0.25 * 0.9 + 0.20 * 0.7 + 0.15 * 0.6 + 0.15 * 0.95
        assert score.final_score == pytest.approx(expected, abs=0.01)

    def test_legacy_aliases_map_correctly(self) -> None:
        """Legacy property aliases should return the mapped new fields."""
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=0.8,
            data_retrieval_quality=0.7,
            source_metadata_quality=0.9,
            query_answer_alignment=0.6,
            computation_complexity=0.95,
        )
        # query_interpretation → query_answer_alignment
        assert score.query_interpretation == 0.6
        # source_authority → source_metadata_quality
        assert score.source_authority == 0.9
        # retrieval_match → data_retrieval_quality
        assert score.retrieval_match == 0.7
        # data_recency → source_metadata_quality
        assert score.data_recency == 0.9
        # computation_reliability → computation_complexity
        assert score.computation_reliability == 0.95

    def test_level_thresholds(self) -> None:
        """Verify HIGH/MEDIUM/LOW/VERY_LOW thresholds."""
        high = ConfidenceScore.compute_from_signals(
            answer_grounding=0.95,
            data_retrieval_quality=0.90,
            source_metadata_quality=0.85,
            query_answer_alignment=0.85,
            computation_complexity=0.90,
        )
        assert high.level == ConfidenceLevel.HIGH

        low = ConfidenceScore.compute_from_signals(
            answer_grounding=0.55,
            data_retrieval_quality=0.55,
            source_metadata_quality=0.55,
            query_answer_alignment=0.55,
            computation_complexity=0.8,
            notebook_verification=0.6,
        )
        assert low.level == ConfidenceLevel.LOW

    def test_signals_dict_stored(self) -> None:
        """The signals debug dict should be stored on the model."""
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=0.5,
            data_retrieval_quality=0.5,
            source_metadata_quality=0.5,
            query_answer_alignment=0.5,
            computation_complexity=0.5,
            signals={"test_key": "test_value"},
        )
        assert score.signals["test_key"] == "test_value"


# ────────────────────────────────────────────────────────────────────
# ConfidenceCalculator signal-based tests
# ────────────────────────────────────────────────────────────────────


class TestSignalBasedCalculator:
    """Tests for calculate_from_signals() — the new LLM-graph path."""

    def setup_method(self) -> None:
        self.calc = ConfidenceCalculator()

    def test_perfect_signals_high_confidence(self) -> None:
        """A pipeline with many successful tools and grounded answer → HIGH."""
        signals = ToolCallSignals(
            successful_tool_calls=5,
            failed_tool_calls=0,
            sql_retry_count=0,
            total_rows_loaded=200,
            distinct_resources_used=3,
            iterations_used=3,
            max_semantic_score=0.92,
            resource_metadata_modified="2026-05-15",
            resource_record_counts=[5000, 12000],
        )
        # Answer with numbers that ALL appear in tool results
        answer = "There were 1,234 incidents and 5,000 total records."
        tool_results = ["Total records: 5,000\n...1,234 incidents..."]

        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer=answer,
            tool_results=tool_results,
            data_source="wprdc",
            portal_quality_score=0.88,
        )

        assert result.final_score >= 0.70
        assert result.answer_grounding >= 0.7
        assert result.data_retrieval_quality >= 0.5
        assert result.computation_complexity >= 0.7

    def test_no_tool_calls_low_confidence(self) -> None:
        """No tools called at all → very low confidence."""
        signals = ToolCallSignals(
            successful_tool_calls=0,
            failed_tool_calls=0,
            iterations_used=1,
        )
        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer="I could not find any data.",
            tool_results=[],
        )
        assert result.final_score < 0.40

    def test_all_tools_failed(self) -> None:
        """Every tool errored → low confidence."""
        signals = ToolCallSignals(
            successful_tool_calls=0,
            failed_tool_calls=4,
            sql_retry_count=2,
            iterations_used=6,
        )
        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer="I encountered errors.",
            tool_results=["Error: HTTP 500", "SQL error: .", "Error: timeout", "Error: bad"],
        )
        assert result.final_score < 0.35
        assert result.computation_complexity < 0.5

    def test_answer_grounding_with_numbers(self) -> None:
        """Numbers in answer that appear in tool results → high grounding."""
        signals = ToolCallSignals(
            successful_tool_calls=3,
            iterations_used=3,
            total_rows_loaded=100,
            distinct_resources_used=1,
        )
        # Use numbers in a format that matches exactly in the tool output
        answer = "The population is 302971 and there are 45123 households."
        tool_results = [
            "population: 302971\nhouseholds: 45123\ncity: Pittsburgh"
        ]
        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer=answer,
            tool_results=tool_results,
        )
        assert result.answer_grounding >= 0.8

    def test_answer_grounding_no_numbers_is_unmeasurable(self) -> None:
        """No numeric claims → grounding cannot be checked, and says so.

        Grounding works by string-matching numbers from the answer against
        tool output. An answer with no numbers gives it nothing to match,
        so it reports unavailable rather than inventing a stand-in score.
        """
        signals = ToolCallSignals(
            successful_tool_calls=2,
            iterations_used=2,
        )
        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer="The data shows a general upward trend.",
            tool_results=["some data here"],
        )
        assert result.answer_grounding is None
        assert "numeric claims" in result.unavailable["answer_grounding"]
        # The other factors still carry the score.
        assert result.measured_weight > 0.0
        assert result.final_score > 0.0

    def test_answer_grounding_fabricated_numbers(self) -> None:
        """Numbers in answer that DON'T appear in tool results → low grounding."""
        signals = ToolCallSignals(
            successful_tool_calls=2,
            iterations_used=2,
            total_rows_loaded=50,
            distinct_resources_used=1,
        )
        answer = "There were 99,999 incidents."
        tool_results = ["Total records: 1,234\nSome completely different data"]
        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer=answer,
            tool_results=tool_results,
        )
        # 99,999 doesn't appear in results — low grounding
        assert result.answer_grounding < 0.6

    def test_high_iteration_count_penalised(self) -> None:
        """Many iterations → complexity penalty."""
        signals_few = ToolCallSignals(
            successful_tool_calls=3,
            iterations_used=3,
        )
        signals_many = ToolCallSignals(
            successful_tool_calls=3,
            iterations_used=10,
        )
        result_few = self.calc.calculate_from_signals(
            tool_signals=signals_few,
            final_answer="Answer",
            tool_results=["data"],
        )
        result_many = self.calc.calculate_from_signals(
            tool_signals=signals_many,
            final_answer="Answer",
            tool_results=["data"],
        )
        assert result_many.computation_complexity < result_few.computation_complexity

    def test_sql_retries_penalised(self) -> None:
        """SQL retries reduce computation complexity score."""
        signals_clean = ToolCallSignals(
            successful_tool_calls=3,
            iterations_used=3,
            sql_retry_count=0,
        )
        signals_retries = ToolCallSignals(
            successful_tool_calls=3,
            iterations_used=3,
            sql_retry_count=3,
        )
        clean = self.calc.calculate_from_signals(
            tool_signals=signals_clean,
            final_answer="Answer",
            tool_results=["data"],
        )
        retried = self.calc.calculate_from_signals(
            tool_signals=signals_retries,
            final_answer="Answer",
            tool_results=["data"],
        )
        assert retried.computation_complexity < clean.computation_complexity

    def test_semantic_score_boosts_retrieval(self) -> None:
        """High Pinecone score → higher retrieval quality."""
        base_signals = ToolCallSignals(
            successful_tool_calls=3,
            iterations_used=3,
            total_rows_loaded=100,
            distinct_resources_used=1,
        )
        no_semantic = ToolCallSignals(
            **{**base_signals.model_dump(), "max_semantic_score": 0.0}
        )
        high_semantic = ToolCallSignals(
            **{**base_signals.model_dump(), "max_semantic_score": 0.95}
        )
        r_no = self.calc.calculate_from_signals(
            tool_signals=no_semantic,
            final_answer="Answer",
            tool_results=["data"],
        )
        r_high = self.calc.calculate_from_signals(
            tool_signals=high_semantic,
            final_answer="Answer",
            tool_results=["data"],
        )
        assert r_high.data_retrieval_quality > r_no.data_retrieval_quality

    def test_fresh_metadata_boosts_quality(self) -> None:
        """Recently-modified resource → higher source metadata quality."""
        fresh = ToolCallSignals(
            successful_tool_calls=2,
            iterations_used=2,
            resource_metadata_modified="2026-05-30",
            resource_record_counts=[10000],
        )
        stale = ToolCallSignals(
            successful_tool_calls=2,
            iterations_used=2,
            resource_metadata_modified="2020-01-01",
            resource_record_counts=[50],
        )
        r_fresh = self.calc.calculate_from_signals(
            tool_signals=fresh,
            final_answer="Answer",
            tool_results=["data"],
        )
        r_stale = self.calc.calculate_from_signals(
            tool_signals=stale,
            final_answer="Answer",
            tool_results=["data"],
        )
        assert r_fresh.source_metadata_quality > r_stale.source_metadata_quality

    def test_signals_debug_dict_populated(self) -> None:
        """The signals debug dict should contain scoring details."""
        signals = ToolCallSignals(
            successful_tool_calls=2,
            failed_tool_calls=1,
            iterations_used=3,
            total_rows_loaded=50,
            distinct_resources_used=1,
        )
        result = self.calc.calculate_from_signals(
            tool_signals=signals,
            final_answer="There were 100 items.",
            tool_results=["100 items found"],
            data_source="wprdc",
        )
        assert result.signals["data_source"] == "wprdc"
        assert "grounding_total_claims" in result.signals
        assert "retrieval_success_rate" in result.signals
        assert "complexity_error_rate" in result.signals


# ────────────────────────────────────────────────────────────────────
# Legacy calculator tests (deterministic Data Commons graph)
# ────────────────────────────────────────────────────────────────────


class TestLegacyCalculator:
    """Tests for calculate() — the deterministic-graph path."""

    def setup_method(self) -> None:
        self.calc = ConfidenceCalculator()

    def test_legacy_with_observations(self) -> None:
        """Data Commons path with real observations → reasonable score."""
        retrieved = RetrievedData(
            observations=[],  # empty but source_info present
            source_info=[
                DataSource(
                    id="data_commons",
                    name="Data Commons",
                    url="https://datacommons.org",
                    quality_score=0.90,
                )
            ],
            retrieval_method="kg_lookup",
            retrieval_score=0.95,
        )
        result = self.calc.calculate(
            query_confidence=0.85,
            retrieved_data=retrieved,
            computation_type="direct_lookup",
        )
        assert 0.0 < result.final_score <= 1.0
        # Legacy aliases work
        assert result.query_interpretation == result.query_answer_alignment

    def test_legacy_no_data_reports_unmeasurable_factors(self) -> None:
        """No retrieved data → the data-derived factors are unmeasurable.

        They used to return 0.0, which is indistinguishable from "we looked
        and the source scored zero". Only the factors that had inputs
        (query interpretation, computation type) contribute now.
        """
        result = self.calc.calculate(
            query_confidence=0.5,
            retrieved_data=None,
            computation_type="unknown",
        )
        assert result.source_metadata_quality is None
        assert result.data_retrieval_quality is None
        for key in ("source_metadata_quality", "data_retrieval_quality", "data_recency"):
            assert key in result.unavailable
            assert "no data was retrieved" in result.unavailable[key] or "no source" in (
                result.unavailable[key]
            )
        # Query interpretation (0.175) and computation (0.105) were measurable;
        # the legacy weights were rescaled in #131 to leave 30% for the
        # notebook factor, which is pending at this point.
        assert result.measured_weight == pytest.approx(0.28)
        assert result.explanation is not None
        assert "28%" in result.explanation

    def test_response_template_levels(self) -> None:
        """get_response_template returns correct templates per level."""
        high = ConfidenceScore.compute(0.9, 0.95, 0.95, 0.9, 1.0)
        assert self.calc.get_response_template(high)["template"] == "direct"

        medium = ConfidenceScore.compute(0.7, 0.8, 0.7, 0.7, 0.8)
        assert self.calc.get_response_template(medium)["template"] == "qualified"

        # 0.25*0.6 + 0.25*0.6 + 0.20*0.5 + 0.15*0.5 + 0.15*0.6 = 0.54 → LOW
        low = ConfidenceScore.compute(0.6, 0.6, 0.5, 0.5, 0.6)
        assert self.calc.get_response_template(low)["template"] == "partial"

        very_low = ConfidenceScore.compute(0.2, 0.2, 0.2, 0.2, 0.2)
        assert self.calc.get_response_template(very_low)["template"] == "unknown"

    def test_should_escalate(self) -> None:
        """Escalation when below threshold after max attempts."""
        low = ConfidenceScore.compute(0.2, 0.2, 0.2, 0.2, 0.2)
        assert self.calc.should_escalate(low, retrieval_attempts=2)

        high = ConfidenceScore.compute(0.9, 0.9, 0.9, 0.9, 0.9)
        assert not self.calc.should_escalate(high, retrieval_attempts=2)


# ────────────────────────────────────────────────────────────────────
# ToolCallSignals model tests
# ────────────────────────────────────────────────────────────────────


class TestToolCallSignals:
    """Tests for the ToolCallSignals Pydantic model."""

    def test_defaults(self) -> None:
        """All fields have sensible zero defaults."""
        signals = ToolCallSignals()
        assert signals.successful_tool_calls == 0
        assert signals.failed_tool_calls == 0
        assert signals.sql_retry_count == 0
        assert signals.max_semantic_score == 0.0
        assert signals.resource_record_counts == []

    def test_construction_with_values(self) -> None:
        """Can construct with all fields populated."""
        signals = ToolCallSignals(
            successful_tool_calls=5,
            failed_tool_calls=1,
            sql_retry_count=2,
            total_rows_loaded=500,
            distinct_resources_used=3,
            iterations_used=4,
            max_semantic_score=0.88,
            resource_metadata_modified="2026-01-15",
            resource_record_counts=[1000, 5000],
            answer_numeric_claims=3,
            answer_grounded_claims=2,
        )
        assert signals.successful_tool_calls == 5
        assert signals.max_semantic_score == 0.88


class TestUnmeasurableFactors:
    """Issue #132 — a factor with no input is explained, not scored 0.0."""

    def setup_method(self) -> None:
        self.calc = ConfidenceCalculator()

    def test_nothing_measurable_is_unknown_not_zero(self) -> None:
        """Every factor unavailable → level UNKNOWN, not VERY_LOW."""
        score = ConfidenceScore.compute(
            None, None, None, None, None,
            unavailable={"escalated": "Routed to a human before any data was retrieved"},
        )
        assert score.measured_weight == 0.0
        assert score.final_score == 0.0
        assert score.level == ConfidenceLevel.UNKNOWN
        assert score.level != ConfidenceLevel.VERY_LOW
        assert score.explanation is not None
        assert "No confidence score could be computed" in score.explanation

    def test_measured_zero_is_still_very_low(self) -> None:
        """A genuinely-measured zero stays VERY_LOW — the distinction holds."""
        score = ConfidenceScore.compute(0.0, 0.0, 0.0, 0.0, 0.0, notebook_verification=0.0)
        assert score.measured_weight == 1.0
        assert score.level == ConfidenceLevel.VERY_LOW
        assert score.explanation is None
        assert score.is_fully_measured

    def test_composite_renormalises_over_measured_factors(self) -> None:
        """A missing factor is dropped, not averaged in as zero."""
        # Only answer_grounding (0.30) measured, at 0.9.
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=0.9,
            data_retrieval_quality=None,
            source_metadata_quality=None,
            query_answer_alignment=None,
            computation_complexity=None,
            unavailable={"data_retrieval_quality": "no tools ran"},
        )
        # Renormalised: 0.9, not 0.25 * 0.9 = 0.225.
        assert score.final_score == pytest.approx(0.9)
        assert score.measured_weight == pytest.approx(0.25)
        assert not score.is_fully_measured

    def test_explanation_names_the_reason(self) -> None:
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=0.8,
            data_retrieval_quality=0.7,
            source_metadata_quality=0.9,
            query_answer_alignment=None,
            computation_complexity=0.6,
            notebook_verification=0.9,
            unavailable={"query_answer_alignment": "Query alignment is not scored yet"},
        )
        assert score.explanation is not None
        assert "90%" in score.explanation  # 1.0 - 0.10 for alignment
        assert "Query alignment is not scored yet" in score.explanation

    def test_no_tool_calls_leaves_nothing_measurable(self) -> None:
        """An LLM run where no tool ever fired cannot be scored at all."""
        result = self.calc.calculate_from_signals(
            tool_signals=ToolCallSignals(),
            final_answer="I could not find data for that.",
            tool_results=[],
        )
        assert result.level == ConfidenceLevel.UNKNOWN
        assert result.measured_weight == 0.0
        assert set(result.unavailable) >= {
            "answer_grounding",
            "data_retrieval_quality",
            "query_answer_alignment",
            "computation_complexity",
        }

    def test_query_alignment_is_never_silently_derived_from_grounding(self) -> None:
        """Alignment used to be an affine restatement of grounding."""
        result = self.calc.calculate_from_signals(
            tool_signals=ToolCallSignals(successful_tool_calls=2, total_rows_loaded=100),
            final_answer="The rate was 4.2 percent in 2025.",
            tool_results=["rate 4.2 for 2025"],
        )
        assert result.query_answer_alignment is None
        assert "not scored yet" in result.unavailable["query_answer_alignment"]

    def test_source_quality_not_fabricated_from_portal_default(self) -> None:
        """With no citation and no metadata, source quality is unavailable.

        It used to return the hardcoded 0.85 portal default as though it
        had been measured.
        """
        result = self.calc.calculate_from_signals(
            tool_signals=ToolCallSignals(successful_tool_calls=1, total_rows_loaded=10),
            final_answer="The count was 12 units.",
            tool_results=["count 12"],
            portal_quality_measured=False,
        )
        assert result.source_metadata_quality is None
        assert "source_metadata_quality" in result.unavailable

    def test_no_computation_step_is_not_a_perfect_score(self) -> None:
        """Empty computed_results used to default to direct_lookup = 1.0."""
        result = self.calc.calculate(
            query_confidence=0.8,
            retrieved_data=None,
            computation_type=None,
        )
        assert result.computation_complexity is None
        assert "no computation step ran" in result.unavailable["computation_complexity"]

    def test_breakdown_survives_pydantic_roundtrip(self) -> None:
        """The API serialises this shape — None factors must not be rejected."""
        score = ConfidenceScore.compute_from_signals(
            answer_grounding=None,
            data_retrieval_quality=0.5,
            source_metadata_quality=None,
            query_answer_alignment=None,
            computation_complexity=0.6,
            unavailable={"answer_grounding": "no answer text"},
        )
        dumped = score.model_dump()
        assert dumped["answer_grounding"] is None
        assert dumped["unavailable"] == {"answer_grounding": "no answer text"}
        assert ConfidenceScore(**dumped).final_score == score.final_score
