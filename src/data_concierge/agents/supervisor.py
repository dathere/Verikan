"""Supervisor agent and LangGraph workflow orchestration."""

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from data_concierge.agents.state import GraphState, create_initial_state
from data_concierge.core.confidence import confidence_calculator
from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import ConfidenceScore, QueryIntent, QueryTier

logger = get_logger(__name__)


class SupervisorAgent:
    """Supervisor agent that orchestrates specialist agents.

    Implements the supervisor pattern where a central agent delegates
    to specialists and aggregates results.
    """

    def __init__(self) -> None:
        """Initialize the supervisor agent."""
        self.logger = get_logger("supervisor")

    # Direct source page URLs by source ID and variable keyword
    SOURCE_PAGE_URLS: dict[str, dict[str, str]] = {
        "data_commons": {
            "_base": "https://datacommons.org/tools/timeline",
            "_api": "https://api.datacommons.org/v2/observation",
        },
        "bls": {
            "_base": "https://data.bls.gov",
            "unemployment": "https://www.bls.gov/lau/",
            "employment": "https://www.bls.gov/ces/",
            "cpi": "https://www.bls.gov/cpi/",
            "inflation": "https://www.bls.gov/cpi/",
            "wages": "https://www.bls.gov/oes/",
            "jobs": "https://www.bls.gov/jlt/",
        },
        "census": {
            "_base": "https://data.census.gov",
            "_api": "https://api.census.gov",
            "population": "https://www.census.gov/programs-surveys/popest.html",
            "income": "https://www.census.gov/programs-surveys/acs",
            "poverty": "https://www.census.gov/programs-surveys/acs",
            "housing": "https://www.census.gov/programs-surveys/acs",
        },
        "bea": {
            "_base": "https://apps.bea.gov/iTable/",
            "_api": "https://apps.bea.gov/api",
            "gdp": "https://www.bea.gov/data/gdp",
            "income": "https://www.bea.gov/data/economic-accounts/regional",
        },
        "fred": {
            "_base": "https://fred.stlouisfed.org",
            "_api": "https://api.stlouisfed.org/fred",
            "interest": "https://fred.stlouisfed.org/categories/22",
            "inflation": "https://fred.stlouisfed.org/categories/32992",
        },
    }

    # Human-readable names for source IDs
    _SOURCE_NAMES: dict[str, str] = {
        "bls": "Bureau of Labor Statistics",
        "census": "U.S. Census Bureau",
        "bea": "Bureau of Economic Analysis",
        "fred": "Federal Reserve Economic Data",
        "data_commons": "Google Data Commons",
    }

    def _build_source_links(self, state: GraphState) -> list[dict[str, str]]:
        """Build direct source links for a quick answer.

        Strategy:
        1. Build a Data Commons Explorer link if we have place + variable DCIDs
        2. Match variable keyword against SOURCE_PAGE_URLS to find specific pages
        3. Add citation-derived links as fallback

        Always produces at least one link if entities or citations exist.
        """
        links: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        entities = state.get("entities")
        citations = state.get("citations", [])

        # Extract variable keyword safely
        variable_keyword = ""
        if entities and entities.variables and len(entities.variables) > 0:
            variable_keyword = entities.variables[0].name.lower()

        # 1. Data Commons Explorer deep link (most useful — interactive chart)
        if (
            entities
            and entities.variables and len(entities.variables) > 0
            and entities.places and len(entities.places) > 0
        ):
            place = entities.places[0]
            var = entities.variables[0]
            dc_url = (
                f"https://datacommons.org/tools/timeline#"
                f"place={place.dcid}&statsVar={var.dcid}"
            )
            seen_urls.add(dc_url)
            links.append({
                "name": "Data Commons Explorer",
                "url": dc_url,
                "description": f"Interactive chart: {var.name} in {place.name}",
            })

        # 2. Match variable keyword against known source pages
        if variable_keyword:
            for source_id, source_urls in self.SOURCE_PAGE_URLS.items():
                for keyword, url in source_urls.items():
                    if keyword.startswith("_"):
                        continue
                    if keyword in variable_keyword and url not in seen_urls:
                        seen_urls.add(url)
                        name = self._SOURCE_NAMES.get(source_id, source_id)
                        links.append({
                            "name": name,
                            "url": url,
                            "description": f"Official {keyword} data from {name}",
                        })

        # 3. Add links from citations (if any) that we haven't already covered
        for citation in citations:
            url = citation.url
            if url and url not in seen_urls:
                seen_urls.add(url)
                desc = citation.source.name
                if citation.dataset_title:
                    desc = f"{citation.dataset_title} — {citation.source.name}"
                links.append({
                    "name": citation.source.name,
                    "url": url,
                    "description": desc,
                })

        return links

    async def route_query(self, state: GraphState) -> GraphState:
        """Route the query to appropriate processing based on tier.

        Args:
            state: Current graph state

        Returns:
            Updated state with routing decision
        """
        tier = state.get("tier")
        intent = state.get("intent")
        self.logger.info("Routing query", tier=tier.value if tier else None)

        if tier == QueryTier.TIER_3:
            # Escalate to human
            state["should_escalate"] = True
            state["answer"] = (
                "This query requires assistance from a human data specialist. "
                "Your request has been added to the specialist queue. "
                "A data concierge will review your request and reach out shortly."
            )
            # Nothing has been retrieved or computed at this point — the
            # query was diverted straight to a human. Report every factor
            # as unmeasurable rather than as a measured zero.
            state["confidence"] = ConfidenceScore.compute(
                None,
                None,
                None,
                None,
                None,
                unavailable={
                    "escalated": (
                        "No confidence could be measured because this query was routed to a "
                        "human specialist before any data was retrieved"
                    )
                },
            )

        # Enable quick answer mode for simple factual lookups (TIER_1)
        if tier == QueryTier.TIER_1 and intent == QueryIntent.FACTUAL_LOOKUP:
            state["quick_answer_mode"] = True
            self.logger.info("Quick answer mode enabled for TIER_1 factual lookup")

        return state

    async def calculate_confidence(self, state: GraphState) -> GraphState:
        """Calculate confidence score from all processing results.

        Uses the signal-based calculator when tool_call_signals are
        present (LLM graph), otherwise falls back to the legacy
        calculator (deterministic Data Commons graph).

        Args:
            state: Current graph state

        Returns:
            State with confidence score
        """
        self.logger.debug("Calculating confidence")

        tool_signals = state.get("tool_call_signals")

        if tool_signals is not None:
            # ── Signal-based path (LLM graph) ────────────────────────
            final_answer = state.get("answer", "") or ""
            tool_result_texts = state.get("tool_result_texts", [])
            data_source = state.get("data_source", "")

            # Portal quality is only a measurement when a citation supplied
            # it; otherwise it is a generic default the source factor must
            # not present as if it had been observed.
            portal_quality = 0.85
            portal_quality_measured = False
            citations = state.get("citations", [])
            if citations and hasattr(citations[0], "source"):
                portal_quality = citations[0].source.quality_score
                portal_quality_measured = True

            confidence = confidence_calculator.calculate_from_signals(
                tool_signals=tool_signals,
                final_answer=final_answer,
                tool_results=tool_result_texts,
                data_source=data_source,
                portal_quality_score=portal_quality,
                portal_quality_measured=portal_quality_measured,
            )
        else:
            # ── Legacy path (deterministic graph) ────────────────────
            # The LLM graph's exception path also lands here, having set
            # parse_confidence to 0.0 without ever running a parser. Treat
            # a zero as "never measured" rather than "measured as hopeless".
            parse_confidence = state.get("parse_confidence") or None
            retrieved_data = state.get("retrieved_data")
            computed_results = state.get("computed_results") or {}
            # No computation step ran if computed_results is empty. The old
            # "direct_lookup" default scored that as a perfect 1.0.
            computation_type = computed_results.get("computation_type") or None

            confidence = confidence_calculator.calculate(
                query_confidence=parse_confidence,
                retrieved_data=retrieved_data,
                computation_type=computation_type,
            )

        state["confidence"] = confidence

        self.logger.info(
            "Confidence calculated",
            final_score=confidence.final_score,
            level=confidence.level.value,
            measured_weight=confidence.measured_weight,
            unavailable=sorted(confidence.unavailable),
            path="signals" if tool_signals is not None else "legacy",
        )

        return state

    async def aggregate_results(self, state: GraphState) -> GraphState:
        """Aggregate results from specialist agents into final answer.

        Args:
            state: Current graph state with specialist outputs

        Returns:
            State with aggregated final response
        """
        self.logger.info("Aggregating results")

        # Check if we already have an answer (e.g., from escalation)
        if state.get("answer"):
            return state

        # Check for errors
        if state.get("error"):
            state["answer"] = f"I encountered an error processing your query: {state['error']}"
            if not state.get("confidence"):
                state["confidence"] = ConfidenceScore.compute(
                    None,
                    None,
                    None,
                    None,
                    None,
                    unavailable={
                        "error": (
                            "No confidence could be measured because the query failed before "
                            "scoring could run"
                        )
                    },
                )
            return state

        # Check if we need clarification
        if state.get("needs_clarification"):
            state["answer"] = state.get(
                "clarification_question",
                "Could you please clarify your question?"
            )
            return state

        # Build response from retrieved data and computed results
        retrieved_data = state.get("retrieved_data")
        computed_results = state.get("computed_results", {})
        intent = state.get("intent")
        confidence = state.get("confidence")
        citations = state.get("citations", [])

        # Generate answer based on confidence level and data
        answer = self._format_answer(
            retrieved_data=retrieved_data,
            computed_results=computed_results,
            intent=intent,
            confidence=confidence,
            citations=citations,
        )

        state["answer"] = answer

        # Build quick answer with source links for TIER_1 factual lookups
        if state.get("quick_answer_mode"):
            source_links = self._build_source_links(state)
            state["source_links"] = source_links

            # Build a concise one-line answer
            quick_text = self._format_quick_answer(
                retrieved_data=retrieved_data,
                computed_results=computed_results,
                state=state,
            )
            state["quick_answer"] = quick_text
            self.logger.info(
                "Quick answer built",
                source_link_count=len(source_links),
            )

        return state

    def _format_answer(
        self,
        retrieved_data: Any,
        computed_results: dict[str, Any],
        intent: QueryIntent | None,
        confidence: ConfidenceScore | None,
        citations: list[Any],
    ) -> str:
        """Format the final answer based on data and confidence."""
        has_observations = (
            retrieved_data
            and hasattr(retrieved_data, "observations")
            and len(retrieved_data.observations) > 0
        )
        has_primary = computed_results.get("primary_value") is not None

        if not has_observations and not has_primary:
            return (
                "I couldn't retrieve the specific data from the API at this time. "
                "The data may be temporarily unavailable, or may require an API key. "
                "Please check the source links below to find this data directly."
            )

        # Get confidence level prefix/suffix
        prefix = ""
        suffix = ""
        if confidence:
            template = confidence_calculator.get_response_template(confidence)
            prefix = template.get("prefix", "")
            suffix = template.get("suffix", "")

        comp_type = computed_results.get("computation_type", "direct_lookup")
        answer_parts: list[str] = []

        if comp_type == "direct_lookup":
            if has_primary and has_observations:
                obs = retrieved_data.observations[0]
                unit = obs.variable.unit or ""
                answer_parts.append(
                    f"{prefix}the {obs.variable.name} for {obs.place.name} "
                    f"is {computed_results['primary_value']}{unit} ({obs.date})"
                )
            elif has_primary:
                answer_parts.append(
                    f"{prefix}the value is {computed_results['primary_value']}"
                )
            elif has_observations:
                # No computed primary_value but we have raw observations
                obs = retrieved_data.observations[0]
                unit = obs.variable.unit or ""
                answer_parts.append(
                    f"{prefix}the {obs.variable.name} for {obs.place.name} "
                    f"is {obs.value}{unit} ({obs.date})"
                )

        elif comp_type == "comparison":
            highest = computed_results.get("highest", {})
            lowest = computed_results.get("lowest", {})
            pct_diff = computed_results.get("percentage_difference", 0)
            if highest and lowest:
                answer_parts.append(
                    f"{prefix}{highest.get('place', 'Unknown')} has the highest value "
                    f"at {highest.get('value', 'N/A')}, while {lowest.get('place', 'Unknown')} "
                    f"has the lowest at {lowest.get('value', 'N/A')} "
                    f"(a difference of {pct_diff:.1f}%)"
                )

        elif comp_type == "trend_analysis":
            trend = computed_results.get("trend_direction", "unknown")
            pct_change = computed_results.get("percentage_change", 0)
            start_date = computed_results.get("start_date", "")
            end_date = computed_results.get("end_date", "")
            answer_parts.append(
                f"{prefix}the trend is {trend} with a {abs(pct_change):.1f}% "
                f"{'increase' if pct_change > 0 else 'decrease'} "
                f"from {start_date} to {end_date}"
            )

        # Fallback: use raw observations if nothing else worked
        if not answer_parts and has_observations:
            for obs in retrieved_data.observations[:3]:
                answer_parts.append(
                    f"{obs.variable.name} for {obs.place.name}: "
                    f"{obs.value}{obs.variable.unit or ''} ({obs.date})"
                )

        if not answer_parts:
            return (
                "I couldn't retrieve the specific data from the API at this time. "
                "Please check the source links below to find this data directly."
            )

        answer = ". ".join(answer_parts)
        if not answer.endswith("."):
            answer += "."
        if citations:
            answer += " [1]"
        if suffix:
            answer += suffix
        return answer

    def _format_quick_answer(
        self,
        retrieved_data: Any,
        computed_results: dict[str, Any],
        state: GraphState,
    ) -> str:
        """Format a concise one-line answer for quick answer mode.

        Returns a single sentence. Falls back gracefully when data is missing.
        """
        has_observations = (
            retrieved_data
            and hasattr(retrieved_data, "observations")
            and len(retrieved_data.observations) > 0
        )
        primary_value = computed_results.get("primary_value")

        # Best case: we have observations with values
        if has_observations:
            obs = retrieved_data.observations[0]
            unit = obs.variable.unit or ""
            value = primary_value if primary_value is not None else obs.value
            return (
                f"The {obs.variable.name} for {obs.place.name} "
                f"is {value}{unit} ({obs.date})."
            )

        # No observations — build a useful fallback from entities
        entities = state.get("entities")
        var_name = "the requested data"
        place_name = ""
        if entities:
            if entities.variables and len(entities.variables) > 0:
                var_name = entities.variables[0].name
            if entities.places and len(entities.places) > 0:
                place_name = f" for {entities.places[0].name}"

        return (
            f"Could not retrieve {var_name}{place_name} from the API. "
            f"Please check the source links below for the latest data."
        )

    async def check_confidence(self, state: GraphState) -> GraphState:
        """Check confidence and decide if escalation or retry is needed.

        Args:
            state: Current graph state

        Returns:
            State with escalation decision
        """
        confidence = state.get("confidence")
        retrieval_attempts = state.get("retrieval_attempts", 0)

        if confidence:
            self.logger.debug(
                "Checking confidence",
                score=confidence.final_score,
                measured_weight=confidence.measured_weight,
                threshold=settings.escalation_confidence_threshold,
                attempts=retrieval_attempts,
            )

            # A score of 0.0 because nothing could be measured is not the
            # same as a measured-low score, and must not silently trip the
            # low-confidence branch as if we had looked and found nothing.
            unmeasurable = confidence.measured_weight <= 0.0

            if unmeasurable and retrieval_attempts >= settings.max_retrieval_attempts:
                state["should_escalate"] = True
                state["answer"] = (
                    "I couldn't measure how reliable an answer to your question would be, "
                    "so I'd rather not guess. Would you like me to escalate this to a "
                    "human data specialist?"
                )
                self.logger.info(
                    "Escalating because confidence was unmeasurable",
                    unavailable=sorted(confidence.unavailable),
                    attempts=retrieval_attempts,
                )
            elif (
                not unmeasurable
                and confidence.final_score < settings.escalation_confidence_threshold
                and retrieval_attempts >= settings.max_retrieval_attempts
            ):
                state["should_escalate"] = True
                state["answer"] = (
                    "I wasn't able to find a confident answer to your question "
                    "after multiple attempts. Would you like me to escalate this "
                    "to a human data specialist?"
                )
                self.logger.info(
                    "Escalating due to low confidence",
                    confidence=confidence.final_score,
                    attempts=retrieval_attempts,
                )

        return state


def should_continue_after_route(state: GraphState) -> Literal["end", "continue"]:
    """Determine if we should continue after routing.

    Args:
        state: Current graph state

    Returns:
        "end" if escalated to human, "continue" otherwise
    """
    if state.get("should_escalate"):
        return "end"
    return "continue"


def should_generate_notebook(state: GraphState) -> Literal["notebook", "skip_notebook"]:
    """Determine if notebook generation should be skipped.

    For TIER_1 factual lookups, we skip notebook generation and produce
    a quick one-line answer with source links instead.

    Args:
        state: Current graph state

    Returns:
        "skip_notebook" if quick answer mode, "notebook" otherwise
    """
    if state.get("quick_answer_mode"):
        return "skip_notebook"
    return "notebook"


def should_retry_or_end(state: GraphState) -> Literal["end", "retry"]:
    """Determine if we should retry data finding or end.

    Args:
        state: Current graph state

    Returns:
        "retry" if low confidence and attempts remaining, "end" otherwise
    """
    confidence = state.get("confidence")
    retrieval_attempts = state.get("retrieval_attempts", 0)

    # If we have a meaningful answer and confidence is okay, end
    answer = state.get("answer")
    has_meaningful_answer = bool(
        answer and (not isinstance(answer, str) or answer.strip())
    )
    if has_meaningful_answer:
        if confidence and confidence.final_score >= settings.escalation_confidence_threshold:
            return "end"
        # If already at max attempts, end anyway
        if retrieval_attempts >= settings.max_retrieval_attempts:
            return "end"
        # Low confidence but haven't maxed out attempts. Retrying only helps
        # if we actually measured something and it came out low — an
        # unmeasurable score means retrieval never produced a signal to
        # score, so another identical pass will not change that.
        if (
            confidence
            and confidence.measured_weight > 0.0
            and confidence.final_score < settings.escalation_confidence_threshold
        ):
            return "retry"

    return "end"


def create_llm_graph() -> StateGraph:
    """Create the LLM-driven LangGraph workflow for CKAN/WPRDC queries.

    Replaces the hardcoded pipeline with an LLM agent that uses tool
    calling to search, load, analyze data, and generate answers.

    Returns:
        Compiled StateGraph
    """
    from data_concierge.agents.llm_agent import get_llm_agent
    from data_concierge.agents.notebook_generator import NotebookGeneratorAgent

    supervisor = SupervisorAgent()
    llm_agent = get_llm_agent()
    notebook_generator = NotebookGeneratorAgent()

    workflow = StateGraph(GraphState)

    workflow.add_node("llm_analyze", llm_agent.process)
    workflow.add_node("generate_notebook", notebook_generator.process)
    workflow.add_node("calculate_confidence", supervisor.calculate_confidence)
    workflow.add_node("aggregate", supervisor.aggregate_results)

    workflow.set_entry_point("llm_analyze")
    workflow.add_edge("llm_analyze", "generate_notebook")
    workflow.add_edge("generate_notebook", "calculate_confidence")
    workflow.add_edge("calculate_confidence", "aggregate")
    workflow.add_edge("aggregate", END)

    return workflow.compile()


def create_agent_graph() -> StateGraph:
    """Create the LangGraph workflow for query processing.

    Always runs the full analysis pipeline: parse → route → find → compute
    → visualize → cite → notebook → confidence → aggregate → check.

    Returns:
        Compiled StateGraph
    """
    # Import agents here to avoid circular imports
    from data_concierge.agents.citation_builder import CitationBuilderAgent
    from data_concierge.agents.data_finder import DataFinderAgent
    from data_concierge.agents.notebook_generator import NotebookGeneratorAgent
    from data_concierge.agents.query_parser import QueryParserAgent
    from data_concierge.agents.stats_computer import StatsComputerAgent
    from data_concierge.agents.viz_builder import VizBuilderAgent

    # Initialize agents
    supervisor = SupervisorAgent()
    query_parser = QueryParserAgent()
    data_finder = DataFinderAgent()
    stats_computer = StatsComputerAgent()
    viz_builder = VizBuilderAgent()
    citation_builder = CitationBuilderAgent()
    notebook_generator = NotebookGeneratorAgent()

    # Create the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("parse", query_parser.process)
    workflow.add_node("route", supervisor.route_query)
    workflow.add_node("find_data", data_finder.process)
    workflow.add_node("compute", stats_computer.process)
    workflow.add_node("visualize", viz_builder.process)
    workflow.add_node("cite", citation_builder.process)
    workflow.add_node("generate_notebook", notebook_generator.process)
    workflow.add_node("calculate_confidence", supervisor.calculate_confidence)
    workflow.add_node("aggregate", supervisor.aggregate_results)
    workflow.add_node("check_confidence", supervisor.check_confidence)

    # Define edges — always start with parsing
    workflow.set_entry_point("parse")

    # After parsing, route based on tier
    workflow.add_edge("parse", "route")

    # After routing, check if we should continue or end (escalation)
    workflow.add_conditional_edges(
        "route",
        should_continue_after_route,
        {
            "end": END,
            "continue": "find_data",
        },
    )

    # Data finding leads to computation
    workflow.add_edge("find_data", "compute")

    # Computation leads to visualization
    workflow.add_edge("compute", "visualize")

    # Visualization leads to citation
    workflow.add_edge("visualize", "cite")

    # Citation leads to notebook generation OR directly to confidence
    # (skip notebook for TIER_1 factual lookups — quick answer mode)
    workflow.add_conditional_edges(
        "cite",
        should_generate_notebook,
        {
            "notebook": "generate_notebook",
            "skip_notebook": "calculate_confidence",
        },
    )

    # Notebook generation leads to confidence calculation
    workflow.add_edge("generate_notebook", "calculate_confidence")

    # Confidence calculation leads to aggregation
    workflow.add_edge("calculate_confidence", "aggregate")

    # Aggregation leads to confidence check
    workflow.add_edge("aggregate", "check_confidence")

    # Confidence check determines if we end or retry
    workflow.add_conditional_edges(
        "check_confidence",
        should_retry_or_end,
        {
            "end": END,
            "retry": "find_data",
        },
    )

    return workflow.compile()


# Create the graph instances lazily to avoid import issues
_agent_graph = None
_llm_graph = None


# Data sources that should use the LLM-driven graph.
#
# MCP-backed sources also use the LLM graph since MCP tools are exposed to
# the LLM agent via the MCPDataConnector.  ``_STATIC_LLM_GRAPH_SOURCES`` holds
# the non-CKAN LLM sources; admin-added CKAN sites are merged in dynamically
# by ``is_llm_graph_source()`` via the ``gateway.ckan_sites`` registry so new
# portals take effect immediately without a code change or restart.
_STATIC_LLM_GRAPH_SOURCES: set[str] = {"census-data-api", "fbi-crime-data"}


def is_llm_graph_source(data_source: str) -> bool:
    """Return True if ``data_source`` should be routed through the LLM graph.

    Checks the admin-managed CKAN sites registry first, then falls back to
    the hardcoded static set.  Storage failures are non-fatal — in that case
    we still recognize the legacy ``wprdc`` / ``ckan`` IDs.
    """
    if data_source in _STATIC_LLM_GRAPH_SOURCES:
        return True
    try:
        from data_concierge.gateway import ckan_sites
        if data_source in ckan_sites.list_site_ids():
            return True
    except Exception as exc:  # pragma: no cover
        logger.warning("CKAN sites lookup failed", error=str(exc))
        # Defensive fallback to the legacy hardcoded IDs
        if data_source in {"wprdc", "ckan"}:
            return True
    return False


# Backwards-compatible alias: any code that does
# ``if data_source in LLM_GRAPH_SOURCES`` keeps working because ``__contains__``
# on this proxy defers to ``is_llm_graph_source``.
class _LLMGraphSources:
    def __contains__(self, item: object) -> bool:
        return isinstance(item, str) and is_llm_graph_source(item)

    def __iter__(self):
        try:
            from data_concierge.gateway import ckan_sites
            ckan_ids = set(ckan_sites.list_site_ids())
        except Exception:
            ckan_ids = {"wprdc", "ckan"}
        return iter(_STATIC_LLM_GRAPH_SOURCES | ckan_ids)


LLM_GRAPH_SOURCES = _LLMGraphSources()


def get_llm_graph_instance() -> Any:
    """Get the compiled LLM graph, creating it if necessary."""
    global _llm_graph
    if _llm_graph is None:
        _llm_graph = create_llm_graph()
    return _llm_graph


def get_agent_graph() -> Any:
    """Get the compiled agent graph, creating it if necessary."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph


async def process_query(
    query: str,
    session: Any | None = None,
    data_source: str = "data_commons",
) -> GraphState:
    """Process a query through the agent graph.

    Always runs the full analysis pipeline. For CKAN/WPRDC/MCP sources,
    uses the LLM-driven graph; for Data Commons, uses the deterministic graph.

    Args:
        query: User's query text
        session: Optional session for context
        data_source: Data source to use (data_commons, ckan, wprdc, etc.)

    Returns:
        Final graph state with results
    """
    initial_state = create_initial_state(query, session, data_source)

    # Use LLM-driven graph for CKAN / WPRDC / MCP-backed data sources
    if is_llm_graph_source(data_source):
        logger.info(
            "Using LLM graph for CKAN source",
            data_source=data_source,
            query=query[:100],
        )
        graph = get_llm_graph_instance()
    else:
        # Deterministic graph for Data Commons (always analysis, no concierge)
        graph = get_agent_graph()

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    return final_state


async def process_concierge_query(
    query: str,
    session: Any | None = None,
) -> dict[str, Any]:
    """Process a query through the concierge only (no analysis).

    This is a simplified path that only runs the concierge agent
    to provide recommendations without performing actual data analysis.

    Args:
        query: User's query text
        session: Optional session for context

    Returns:
        Dict with concierge recommendations
    """
    from data_concierge.agents.data_concierge import get_data_concierge

    concierge = get_data_concierge()
    result = await concierge.process_query(query)

    return result
