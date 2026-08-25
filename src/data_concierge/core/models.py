"""Core data models for the Data Concierge system."""

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class QueryTier(str, Enum):
    """Query complexity tier for routing."""

    TIER_1 = "tier_1"  # Simple factual queries - AI self-service
    TIER_2 = "tier_2"  # Complex queries - AI-assisted with multi-step reasoning
    TIER_3 = "tier_3"  # Research projects - Human concierge


class QueryIntent(str, Enum):
    """Classified intent of the query."""

    FACTUAL_LOOKUP = "factual_lookup"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    DATASET_DISCOVERY = "dataset_discovery"
    METHODOLOGY_QUESTION = "methodology_question"
    DATA_LINKING = "data_linking"
    OUT_OF_SCOPE = "out_of_scope"


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""

    HIGH = "high"  # >= 0.85
    MEDIUM = "medium"  # 0.70 - 0.84
    LOW = "low"  # 0.50 - 0.69
    VERY_LOW = "very_low"  # < 0.50
    UNKNOWN = "unknown"  # nothing could be measured — not the same as a low score


class DataAccessLevel(str, Enum):
    """Data access level classification."""

    PUBLIC = "public"
    RESTRICTED = "restricted"
    NON_PUBLIC = "non_public"


# =============================================================================
# Entity Models
# =============================================================================


class PlaceEntity(BaseModel):
    """Represents a geographic place entity."""

    dcid: str = Field(..., description="Data Commons ID (e.g., geoId/48 for Texas)")
    name: str = Field(..., description="Human-readable name")
    place_type: str = Field(..., description="Type of place (State, County, City, etc.)")
    geo_json_coords: dict[str, Any] | None = Field(default=None, description="GeoJSON coordinates")


class VariableEntity(BaseModel):
    """Represents a statistical variable entity."""

    dcid: str = Field(..., description="Data Commons variable ID")
    name: str = Field(..., description="Human-readable name")
    unit: str | None = Field(default=None, description="Unit of measurement")
    population_type: str | None = Field(default=None, description="Population type")
    measurement_method: str | None = Field(default=None, description="How the variable is measured")


class TimeEntity(BaseModel):
    """Represents a temporal entity."""

    start_date: str = Field(..., description="Start date (ISO format)")
    end_date: str | None = Field(default=None, description="End date for ranges")
    period_type: str = Field(default="point", description="point, range, or recurring")


class DemographicEntity(BaseModel):
    """Represents a demographic filter entity."""

    attribute: str = Field(..., description="Demographic attribute (age, gender, race, etc.)")
    value: str = Field(..., description="Specific value or range")


# =============================================================================
# Query Models
# =============================================================================


class ParsedEntities(BaseModel):
    """Entities extracted from a query."""

    variables: list[VariableEntity] = Field(default_factory=list)
    places: list[PlaceEntity] = Field(default_factory=list)
    times: list[TimeEntity] = Field(default_factory=list)
    demographics: list[DemographicEntity] = Field(default_factory=list)


# =============================================================================
# Data Models
# =============================================================================


class DataSource(BaseModel):
    """Represents a data source/agency."""

    id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Source name (e.g., Bureau of Labor Statistics)")
    url: str = Field(..., description="Source URL")
    update_frequency: str | None = Field(default=None, description="How often data is updated")
    quality_score: float = Field(default=0.95, ge=0.0, le=1.0, description="Pre-assigned quality score")


class StatisticalObservation(BaseModel):
    """A single statistical observation/data point."""

    variable: VariableEntity
    place: PlaceEntity
    value: float | int | str
    date: str = Field(..., description="Observation date")
    margin_of_error: float | None = Field(default=None, description="Margin of error if available")
    observation_period: str | None = Field(default=None, description="Period covered")
    source: DataSource
    access_level: DataAccessLevel = Field(default=DataAccessLevel.PUBLIC)


class RetrievedData(BaseModel):
    """Data retrieved from knowledge graph or APIs."""

    observations: list[StatisticalObservation] = Field(default_factory=list)
    source_info: list[DataSource] = Field(default_factory=list)
    retrieval_method: str = Field(..., description="kg_lookup, vector_search, or api_call")
    retrieval_score: float = Field(default=1.0, ge=0.0, le=1.0)
    data_vintage: str | None = Field(default=None, description="When the data was last updated")


# =============================================================================
# Citation Models
# =============================================================================


class Citation(BaseModel):
    """A citation for data provenance."""

    source: DataSource
    dataset_title: str
    access_date: str = Field(default_factory=lambda: datetime.now().isoformat()[:10])
    url: str
    methodology_notes: str | None = None
    footnote_text: str | None = None


# =============================================================================
# Response Models
# =============================================================================


class ToolCallSignals(BaseModel):
    """Signals extracted from the LLM tool-calling pipeline.

    Captured during ``LLMAnalysisAgent.process()`` and passed to the
    confidence calculator so it can score from *real* pipeline data
    instead of fabricated placeholders.
    """

    successful_tool_calls: int = Field(default=0, description="Tools that returned data")
    failed_tool_calls: int = Field(default=0, description="Tools that errored")
    sql_retry_count: int = Field(default=0, description="SQL queries retried after error")
    total_rows_loaded: int = Field(default=0, description="Rows loaded via load_resource_data")
    distinct_resources_used: int = Field(
        default=0, description="Unique CKAN resource IDs touched"
    )
    iterations_used: int = Field(default=0, description="LLM conversation turns used")
    max_semantic_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Best Pinecone semantic-search relevance score",
    )
    resource_metadata_modified: str | None = Field(
        default=None, description="ISO date of most-recently-modified resource loaded"
    )
    resource_record_counts: list[int] = Field(
        default_factory=list,
        description="Total-record counts reported by each loaded resource",
    )
    answer_numeric_claims: int = Field(
        default=0, description="Numeric values found in the final answer"
    )
    answer_grounded_claims: int = Field(
        default=0, description="Numeric values that appear in tool results"
    )


class ConfidenceScore(BaseModel):
    """Detailed confidence score breakdown.

    Factors and weights (issue #104, reweighted for #131):
    - notebook_verification (30%): the generated notebook was executed and its
      own output re-derives the numbers the answer claims
    - answer_grounding (25%): answer claims string-match the tool transcript
    - data_retrieval_quality (20%): semantic score, row counts, success rate
    - source_metadata_quality (10%): resource freshness, record counts
    - query_answer_alignment (10%): reserved for a real LLM judge
    - computation_complexity (5%): penalises long/error-prone tool chains

    ``notebook_verification`` carries the largest share because it is the only
    factor that *re-derives* the answer rather than inspecting the context the
    answer was written from. Re-running the published code and getting the
    same number is materially stronger evidence than finding that number in a
    transcript the model already had in front of it, which is all
    ``answer_grounding`` can tell you. Grounding drops 30 -> 25 for that
    reason: the two overlap, and the weaker of the pair should yield.

    Verification is asynchronous — a notebook takes seconds to minutes to run
    — so at answer time this factor is normally unavailable and renormalised
    out (see ``unavailable``), then filled in when the run completes.

    Legacy fields are kept as aliases so the API response, notebook
    generator, and integration tests keep working without a breaking
    change. They are computed from the new factors.
    """

    # ── new factors ──────────────────────────────────────────────────
    # ``None`` means "could not be computed" — see ``unavailable`` for why.
    notebook_verification: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Generated notebook executed and its output re-derives the answer's claims",
    )
    answer_grounding: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of answer claims backed by tool data"
    )
    data_retrieval_quality: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Composite retrieval quality"
    )
    source_metadata_quality: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Resource freshness and richness"
    )
    query_answer_alignment: float | None = Field(
        default=None, ge=0.0, le=1.0, description="How well the answer addresses the query"
    )
    computation_complexity: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Inverse complexity penalty"
    )
    final_score: float = Field(ge=0.0, le=1.0, description="Weighted final score")

    # Raw signals dict for admin review / debugging
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw signal data used for scoring (for admin/debug)",
    )

    # ── explainability (issue #132) ──────────────────────────────────
    # A factor that had no input is recorded here instead of being
    # folded into the composite as 0.0, which would be indistinguishable
    # from a measured-bad score. ``final_score`` is renormalised over
    # whatever *was* measurable, and ``measured_weight`` says how much of
    # the total weight that covers.
    unavailable: dict[str, str] = Field(
        default_factory=dict,
        description="Factor name -> why it could not be computed",
    )
    measured_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the total factor weight that was actually measured",
    )

    # ── legacy aliases (read-only) ───────────────────────────────────
    # Mapped from new factors so existing API consumers, notebooks, and
    # tests that read these fields keep working.

    @property
    def query_interpretation(self) -> float | None:
        """Legacy alias → query_answer_alignment."""
        return self.query_answer_alignment

    @property
    def source_authority(self) -> float | None:
        """Legacy alias → source_metadata_quality."""
        return self.source_metadata_quality

    @property
    def retrieval_match(self) -> float | None:
        """Legacy alias → data_retrieval_quality."""
        return self.data_retrieval_quality

    @property
    def data_recency(self) -> float | None:
        """Legacy alias → source_metadata_quality (recency is folded in)."""
        return self.source_metadata_quality

    @property
    def computation_reliability(self) -> float | None:
        """Legacy alias → computation_complexity."""
        return self.computation_complexity

    @staticmethod
    def _weighted(
        weighted_factors: list[tuple[str, float | None, float]],
    ) -> tuple[float, float]:
        """Reduce ``(name, value, weight)`` triples to ``(score, measured_weight)``.

        Factors whose value is ``None`` are dropped and the remaining
        weights are renormalised, so an unmeasurable factor neither drags
        the composite toward zero nor quietly inflates it.
        """
        measured = [(v, w) for _, v, w in weighted_factors if v is not None]
        measured_weight = sum(w for _, w in measured)
        if measured_weight <= 0.0:
            return 0.0, 0.0
        total = sum(v * w for v, w in measured) / measured_weight
        return round(min(1.0, max(0.0, total)), 4), round(measured_weight, 4)

    @classmethod
    def compute(
        cls,
        query: float | None,
        source: float | None,
        retrieval: float | None,
        recency: float | None,
        computation: float | None,
        unavailable: dict[str, str] | None = None,
        notebook_verification: float | None = None,
    ) -> "ConfidenceScore":
        """Backward-compatible factory used by the deterministic graph.

        Maps the five legacy positional args onto the new factors. Pass
        ``None`` for any factor that could not be computed, together with
        an ``unavailable`` entry saying why; it is then excluded from the
        composite rather than counted as a measured zero.
        """
        # The deterministic graph keeps its own legacy weighting, scaled to
        # leave the same 30% share for notebook verification as the signal
        # path, so the two produce comparable numbers.
        final, measured_weight = cls._weighted(
            [
                ("notebook_verification", notebook_verification, 0.30),
                ("query_answer_alignment", query, 0.175),
                ("source_metadata_quality", source, 0.175),
                ("data_retrieval_quality", retrieval, 0.14),
                ("data_recency", recency, 0.105),
                ("computation_complexity", computation, 0.105),
            ]
        )
        return cls(
            notebook_verification=notebook_verification,
            answer_grounding=retrieval,
            data_retrieval_quality=retrieval,
            source_metadata_quality=source,
            query_answer_alignment=query,
            computation_complexity=computation,
            final_score=final,
            measured_weight=measured_weight,
            unavailable=unavailable or {},
            signals={"legacy_compat": True},
        )

    @classmethod
    def compute_from_signals(
        cls,
        *,
        answer_grounding: float | None,
        data_retrieval_quality: float | None,
        source_metadata_quality: float | None,
        query_answer_alignment: float | None,
        computation_complexity: float | None,
        notebook_verification: float | None = None,
        unavailable: dict[str, str] | None = None,
        signals: dict[str, Any] | None = None,
    ) -> "ConfidenceScore":
        """Compute confidence from the new signal-based factors.

        Weights (#104, reweighted for #131):
          notebook_verification    30%
          answer_grounding         25%
          data_retrieval_quality   20%
          source_metadata_quality  10%
          query_answer_alignment   10%
          computation_complexity    5%

        Pass ``None`` for a factor that could not be computed, plus an
        ``unavailable`` entry saying why (issue #132). The composite is
        renormalised over the factors that were measurable.
        """
        final, measured_weight = cls._weighted(
            [
                ("notebook_verification", notebook_verification, 0.30),
                ("answer_grounding", answer_grounding, 0.25),
                ("data_retrieval_quality", data_retrieval_quality, 0.20),
                ("source_metadata_quality", source_metadata_quality, 0.10),
                ("query_answer_alignment", query_answer_alignment, 0.10),
                ("computation_complexity", computation_complexity, 0.05),
            ]
        )
        return cls(
            notebook_verification=notebook_verification,
            answer_grounding=answer_grounding,
            data_retrieval_quality=data_retrieval_quality,
            source_metadata_quality=source_metadata_quality,
            query_answer_alignment=query_answer_alignment,
            computation_complexity=computation_complexity,
            final_score=final,
            measured_weight=measured_weight,
            unavailable=unavailable or {},
            signals=signals or {},
        )

    @property
    def level(self) -> ConfidenceLevel:
        """Get confidence level classification.

        ``UNKNOWN`` when nothing at all could be measured — distinct from
        ``VERY_LOW``, which means we measured and the news was bad.
        """
        if self.measured_weight <= 0.0:
            return ConfidenceLevel.UNKNOWN
        if self.final_score >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.final_score >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif self.final_score >= 0.50:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    # ClassVar, not a field — Pydantic would otherwise treat it as one.
    NOTEBOOK_PENDING_REASON: ClassVar[str] = (
        "Notebook verification is still running — the generated notebook is being "
        "executed to check that its output reproduces this answer"
    )

    def with_notebook_verification(
        self,
        score: float | None,
        reason: str | None = None,
        signals: dict[str, Any] | None = None,
    ) -> "ConfidenceScore":
        """Return a new score with the notebook factor filled in.

        Verification runs after the answer is returned (#131), so the score
        shown first has this factor pending. When the run lands, the caller
        recomputes through here rather than mutating in place, keeping the
        original score intact for audit.
        """
        unavailable = {
            k: v for k, v in self.unavailable.items() if k != "notebook_verification"
        }
        if score is None:
            unavailable["notebook_verification"] = (
                reason or "Notebook verification could not be completed"
            )
        merged_signals = dict(self.signals)
        merged_signals.update(signals or {})

        if self.signals.get("legacy_compat"):
            return ConfidenceScore.compute(
                query=self.query_answer_alignment,
                source=self.source_metadata_quality,
                retrieval=self.data_retrieval_quality,
                # Recency aliases onto source on the legacy path.
                recency=self.source_metadata_quality,
                computation=self.computation_complexity,
                unavailable=unavailable,
                notebook_verification=score,
            )
        return ConfidenceScore.compute_from_signals(
            answer_grounding=self.answer_grounding,
            data_retrieval_quality=self.data_retrieval_quality,
            source_metadata_quality=self.source_metadata_quality,
            query_answer_alignment=self.query_answer_alignment,
            computation_complexity=self.computation_complexity,
            notebook_verification=score,
            unavailable=unavailable,
            signals=merged_signals,
        )

    @property
    def is_fully_measured(self) -> bool:
        """True when every factor contributed to the score."""
        return not self.unavailable and self.measured_weight >= 0.999

    @property
    def explanation(self) -> str | None:
        """Plain-language note on what the score does and does not cover.

        ``None`` when every factor was measured, so callers can show the
        number on its own without an apologetic caveat.
        """
        if self.is_fully_measured:
            return None
        reasons = list(self.unavailable.values())
        if self.measured_weight <= 0.0:
            head = "No confidence score could be computed for this answer."
        else:
            pct = round(self.measured_weight * 100)
            head = (
                f"This score covers {pct}% of the usual checks; "
                f"the rest could not be measured for this answer."
            )
        if not reasons:
            return head
        return head + " " + " ".join(r.rstrip(".") + "." for r in reasons)


class VisualizationSpec(BaseModel):
    """Vega-Lite visualization specification."""

    chart_type: str = Field(..., description="Type of chart (bar, line, map, etc.)")
    vega_lite_spec: dict[str, Any] = Field(..., description="Vega-Lite specification")
    alt_text: str = Field(..., description="Accessibility description")


class NotebookOutput(BaseModel):
    """Generated Jupyter notebook output."""

    notebook_json: dict[str, Any] = Field(..., description="Complete .ipynb content")
    filename: str = Field(..., description="Suggested filename")
    download_url: str | None = Field(default=None, description="URL to download notebook")


# =============================================================================
# Session Models
# =============================================================================


class Session(BaseModel):
    """User session for context management."""

    session_id: str
    user_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    query_history: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


