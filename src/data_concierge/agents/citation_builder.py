"""Citation Builder Agent for generating proper data citations."""

from datetime import datetime
from typing import Any

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.logging import get_logger
from data_concierge.core.models import Citation, DataSource

logger = get_logger(__name__)


class CitationBuilderAgent(BaseAgent):
    """Agent responsible for building proper citations.

    Creates citation objects with:
    - Source attribution
    - Access dates
    - Methodology notes
    - Footnote text for display
    """

    name = "citation_builder"
    description = "Generates proper citations for data sources"

    # Citation templates by source type
    CITATION_TEMPLATES = {
        "federal_agency": (
            "{agency_name}. \"{dataset_title}.\" {url}. "
            "Accessed {access_date}."
        ),
        "data_commons": (
            "Data Commons. \"{variable_name}\" for {place_name}. "
            "{url}. Accessed {access_date}. "
            "Original source: {original_source}."
        ),
        "academic": (
            "{authors}. \"{title}.\" {journal}, vol. {volume}, "
            "{year}, pp. {pages}. {doi}."
        ),
    }

    async def process(self, state: GraphState) -> GraphState:
        """Build citations for all data sources used.

        Args:
            state: Current graph state with retrieved data

        Returns:
            Updated state with citations
        """
        self.logger.info("Building citations")

        retrieved_data = state.get("retrieved_data")
        if not retrieved_data:
            return state

        citations: list[Citation] = []
        sources_cited: set[str] = set()

        # Build citations for each unique source
        for source in retrieved_data.source_info:
            if source.id in sources_cited:
                continue

            citation = self._create_citation(
                source,
                retrieved_data,
                state.get("query", ""),
            )
            citations.append(citation)
            sources_cited.add(source.id)

        # Add citations for observations if they have different sources
        for obs in retrieved_data.observations:
            if obs.source.id not in sources_cited:
                citation = self._create_citation(
                    obs.source,
                    retrieved_data,
                    state.get("query", ""),
                    variable=obs.variable.name,
                    place=obs.place.name,
                )
                citations.append(citation)
                sources_cited.add(obs.source.id)

        state["citations"] = citations

        # Add to execution trace
        state["execution_trace"].append({
            "agent": self.name,
            "action": "build_citations",
            "citation_count": len(citations),
            "sources": list(sources_cited),
        })

        self.logger.info("Citations built", count=len(citations))

        return state

    def _create_citation(
        self,
        source: DataSource,
        retrieved_data: Any,
        query: str,
        variable: str | None = None,
        place: str | None = None,
    ) -> Citation:
        """Create a citation for a data source.

        Args:
            source: Data source to cite
            retrieved_data: Retrieved data context
            query: Original query
            variable: Variable name if applicable
            place: Place name if applicable

        Returns:
            Citation object
        """
        access_date = datetime.now().strftime("%B %d, %Y")

        # If the source name encodes a dataset (e.g., "CKAN Data Portal — Foo"),
        # split it so the citation shows the portal as the publisher and the
        # dataset name as the dataset_title. This makes citations point to the
        # specific queried dataset.
        portal_name = source.name
        if (variable is None or not variable) and " — " in source.name:
            portal_name, _, derived_title = source.name.partition(" — ")
            if derived_title:
                variable = derived_title

        # Determine citation type and build accordingly
        if source.id == "data_commons":
            dataset_title = f"{variable or 'Statistical Data'}"
            if place:
                dataset_title += f" for {place}"

            footnote = self.CITATION_TEMPLATES["data_commons"].format(
                variable_name=variable or "Statistical Variable",
                place_name=place or "Selected Location",
                url=source.url,
                access_date=access_date,
                original_source="U.S. Federal Agencies",
            )
        else:
            # Federal agency citation
            dataset_title = f"{variable or 'Statistical Data'}"

            footnote = self.CITATION_TEMPLATES["federal_agency"].format(
                agency_name=portal_name,
                dataset_title=dataset_title,
                url=source.url,
                access_date=access_date,
            )

        # Build methodology notes
        methodology = self._build_methodology_notes(source, retrieved_data)

        return Citation(
            source=source,
            dataset_title=dataset_title,
            access_date=access_date,
            url=source.url,
            methodology_notes=methodology,
            footnote_text=footnote,
        )

    def _build_methodology_notes(
        self,
        source: DataSource,
        retrieved_data: Any,
    ) -> str:
        """Build methodology notes for the citation.

        Args:
            source: Data source
            retrieved_data: Retrieved data context

        Returns:
            Methodology notes string
        """
        notes = []

        # Add update frequency
        if source.update_frequency:
            notes.append(f"Data updated {source.update_frequency}.")

        # Add data vintage
        if retrieved_data and retrieved_data.data_vintage:
            notes.append(f"Data retrieved on {retrieved_data.data_vintage}.")

        # Add retrieval method
        if retrieved_data and retrieved_data.retrieval_method:
            method_descriptions = {
                "kg_lookup": "Retrieved via knowledge graph query.",
                "vector_search": "Retrieved via semantic search.",
                "api_call": "Retrieved via direct API call.",
            }
            method_note = method_descriptions.get(
                retrieved_data.retrieval_method,
                f"Retrieved via {retrieved_data.retrieval_method}.",
            )
            notes.append(method_note)

        return " ".join(notes) if notes else ""

    def verify_source_link(self, url: str) -> bool:
        """Verify that a source URL is accessible.

        Args:
            url: URL to verify

        Returns:
            True if URL is accessible
        """
        # TODO: Implement actual URL verification
        # For now, just return True
        return True

    def generate_footnotes(
        self,
        citations: list[Citation],
    ) -> list[str]:
        """Generate numbered footnotes for citations.

        Args:
            citations: List of citations

        Returns:
            List of formatted footnote strings
        """
        return [
            f"[{i + 1}] {c.footnote_text}"
            for i, c in enumerate(citations)
        ]

    def format_for_notebook(
        self,
        citations: list[Citation],
    ) -> str:
        """Format citations for notebook markdown cell.

        Args:
            citations: List of citations

        Returns:
            Markdown-formatted citation string
        """
        lines = ["## Data Sources and Citations", ""]

        for i, citation in enumerate(citations, 1):
            # Split publisher from dataset when source.name uses the
            # "Publisher — Dataset" form so the citation header shows
            # the publisher and the dataset title appears below it.
            full_name = citation.source.name
            if " — " in full_name:
                publisher, _, dataset = full_name.partition(" — ")
                dataset_title = dataset or citation.dataset_title
            else:
                publisher = full_name
                dataset_title = citation.dataset_title

            lines.append(f"**[{i}]** {publisher}")
            lines.append(f"- Dataset: {dataset_title}")
            lines.append(f"- URL: [{citation.url}]({citation.url})")
            lines.append(f"- Accessed: {citation.access_date}")
            if citation.methodology_notes:
                lines.append(f"- Notes: {citation.methodology_notes}")
            lines.append("")

        return "\n".join(lines)
