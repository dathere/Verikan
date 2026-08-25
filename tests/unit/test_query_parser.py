"""Tests for the query parser agent."""

import pytest

from data_concierge.agents.query_parser import QueryParserAgent
from data_concierge.agents.state import create_initial_state


@pytest.fixture
def parser() -> QueryParserAgent:
    """Create a parser instance for testing."""
    return QueryParserAgent()


class TestEntityExtraction:
    """Tests for entity extraction."""

    @pytest.mark.asyncio
    async def test_extract_place_entity(self, parser: QueryParserAgent) -> None:
        """Test extraction of place entities."""
        state = create_initial_state("What is unemployment in Texas?")
        result = await parser.process(state)

        entities = result.get("entities")
        assert entities is not None
        assert len(entities.places) > 0
        assert entities.places[0].name == "Texas"
        assert entities.places[0].dcid == "geoId/48"

    @pytest.mark.asyncio
    async def test_extract_variable_entity(self, parser: QueryParserAgent) -> None:
        """Test extraction of statistical variable entities."""
        state = create_initial_state("What is the unemployment rate?")
        result = await parser.process(state)

        entities = result.get("entities")
        assert entities is not None
        assert len(entities.variables) > 0
        assert "Unemployment" in entities.variables[0].name

    @pytest.mark.asyncio
    async def test_extract_time_entity(self, parser: QueryParserAgent) -> None:
        """Test extraction of temporal entities."""
        state = create_initial_state("What was unemployment in 2023?")
        result = await parser.process(state)

        entities = result.get("entities")
        assert entities is not None
        assert len(entities.times) > 0
        assert "2023" in entities.times[0].start_date

    @pytest.mark.asyncio
    async def test_extract_demographic_entity(self, parser: QueryParserAgent) -> None:
        """Test extraction of demographic entities."""
        state = create_initial_state("What is unemployment for young people?")
        result = await parser.process(state)

        entities = result.get("entities")
        assert entities is not None
        assert len(entities.demographics) > 0
        assert entities.demographics[0].attribute == "age"

    @pytest.mark.asyncio
    async def test_extract_place_nickname(self, parser: QueryParserAgent) -> None:
        """Test extraction of place nicknames."""
        state = create_initial_state("What is unemployment in the Lone Star State?")
        result = await parser.process(state)

        entities = result.get("entities")
        assert entities is not None
        assert len(entities.places) > 0
        assert entities.places[0].name == "Texas"


class TestQueryNormalization:
    """Tests for query normalization."""

    @pytest.mark.asyncio
    async def test_query_normalization(self, parser: QueryParserAgent) -> None:
        """Test that queries are normalized correctly."""
        state = create_initial_state("What is UNEMPLOYMENT in Texas?")
        result = await parser.process(state)

        assert result.get("normalized_query") is not None
        assert result.get("query_hash") is not None

    @pytest.mark.asyncio
    async def test_same_query_same_hash(self, parser: QueryParserAgent) -> None:
        """Test that equivalent queries produce the same hash."""
        state1 = create_initial_state("What is unemployment in Texas?")
        state2 = create_initial_state("What is unemployment in Texas?")

        result1 = await parser.process(state1)
        result2 = await parser.process(state2)

        assert result1.get("query_hash") == result2.get("query_hash")


class TestParsingConfidence:
    """Tests for parsing confidence calculation."""

    @pytest.mark.asyncio
    async def test_confidence_with_entities(self, parser: QueryParserAgent) -> None:
        """Test that presence of entities increases confidence."""
        # Query with clear entities
        state1 = create_initial_state("What is unemployment in Texas in 2023?")
        result1 = await parser.process(state1)

        # Query with fewer clear entities
        state2 = create_initial_state("Tell me about jobs")
        result2 = await parser.process(state2)

        assert result1.get("parse_confidence", 0) >= result2.get("parse_confidence", 0)

    @pytest.mark.asyncio
    async def test_execution_trace_added(self, parser: QueryParserAgent) -> None:
        """Test that execution trace is added during processing."""
        state = create_initial_state("What is unemployment in Texas?")
        result = await parser.process(state)

        trace = result.get("execution_trace", [])
        assert len(trace) > 0
        assert trace[0]["agent"] == "query_parser"
        assert trace[0]["action"] == "parse_query"
