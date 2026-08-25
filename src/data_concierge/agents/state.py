"""Agent state definitions for LangGraph workflow."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from data_concierge.core.models import (
    Citation,
    ConfidenceScore,
    NotebookOutput,
    ParsedEntities,
    QueryIntent,
    QueryTier,
    RetrievedData,
    Session,
    ToolCallSignals,
    VisualizationSpec,
)


class GraphState(TypedDict):
    """State passed through the LangGraph workflow.

    This uses TypedDict for compatibility with LangGraph's state management.
    """

    # Input
    query: str
    session: Session | None

    # Data source selection (data_commons, ckan, etc.)
    data_source: str

    # Concierge mode — kept for backward compat; always "analyze"
    concierge_mode: str

    # Classification
    intent: QueryIntent | None
    tier: QueryTier | None

    # Parsed entities
    entities: ParsedEntities | None
    normalized_query: str | None
    query_hash: str | None
    parse_confidence: float

    # Retrieved data
    retrieved_data: RetrievedData | None
    retrieval_attempts: int

    # Computed results
    computed_results: dict[str, Any]

    # Generated content
    visualization: VisualizationSpec | None
    citations: list[Citation]
    notebook: NotebookOutput | None

    # Execution trace for reproducibility
    execution_trace: list[dict[str, Any]]

    # Raw agent log — full LLM conversation + tool I/O for admin review
    agent_log: list[dict[str, Any]]

    # Tool-call signals for confidence scoring (LLM graph only)
    tool_call_signals: ToolCallSignals | None

    # Raw tool result strings for answer-grounding checks
    tool_result_texts: list[str]

    # LLM messages (for agents that use LLM)
    messages: Annotated[list, add_messages]

    # Final output
    answer: str | None
    confidence: ConfidenceScore | None

    # Quick answer mode - skip notebook for simple factual lookups
    quick_answer_mode: bool

    # Quick answer fields (populated when quick_answer_mode is True)
    quick_answer: str | None  # One-line answer text
    source_links: list[dict[str, str]]  # [{name, url, description}]

    # Control flow
    current_agent: str | None
    should_escalate: bool
    needs_clarification: bool
    clarification_question: str | None
    error: str | None


def create_initial_state(
    query: str,
    session: Session | None = None,
    data_source: str = "data_commons",
) -> GraphState:
    """Create initial state for a new query.

    Args:
        query: User's query text
        session: Optional session for context
        data_source: Data source to use (data_commons, ckan, wprdc, etc.)

    Returns:
        Initial GraphState
    """
    return GraphState(
        query=query,
        session=session,
        data_source=data_source,
        concierge_mode="analyze",
        intent=None,
        tier=None,
        entities=None,
        normalized_query=None,
        query_hash=None,
        parse_confidence=0.0,
        retrieved_data=None,
        retrieval_attempts=0,
        computed_results={},
        visualization=None,
        citations=[],
        notebook=None,
        execution_trace=[],
        agent_log=[],
        tool_call_signals=None,
        tool_result_texts=[],
        messages=[],
        answer=None,
        confidence=None,
        quick_answer_mode=False,
        quick_answer=None,
        source_links=[],
        current_agent=None,
        should_escalate=False,
        needs_clarification=False,
        clarification_question=None,
        error=None,
    )
