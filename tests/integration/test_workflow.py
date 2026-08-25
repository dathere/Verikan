"""Integration tests for the full Data Concierge workflow."""

import pytest
from httpx import ASGITransport, AsyncClient

from data_concierge.agents.supervisor import process_query
from data_concierge.api.main import app
from data_concierge.core.models import QueryIntent, QueryTier
from data_concierge.gateway.router import require_admin, require_auth


@pytest.fixture(autouse=True)
def _authenticated():
    """Satisfy the auth dependencies for endpoint tests.

    /query and the review endpoints began requiring a login after these
    tests were written, which is what the 401s were. The pipeline behaviour
    under test is unrelated to auth, so the dependency is overridden rather
    than every test being taught to log in.
    """
    app.dependency_overrides[require_auth] = lambda: "test-token"
    app.dependency_overrides[require_admin] = lambda: {"user": "test-admin"}
    yield
    app.dependency_overrides.pop(require_auth, None)
    app.dependency_overrides.pop(require_admin, None)


class TestFullWorkflow:
    """Tests for the complete agent pipeline."""

    @pytest.mark.asyncio
    async def test_simple_factual_query(self) -> None:
        """Test processing a simple factual query through the full pipeline."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        # Should have parsed entities
        assert final_state.get("entities") is not None
        entities = final_state["entities"]
        assert len(entities.places) > 0
        assert entities.places[0].name == "Texas"
        assert len(entities.variables) > 0

        # Should be classified as Tier 1
        assert final_state.get("tier") == QueryTier.TIER_1

        # Should have an answer
        assert final_state.get("answer") is not None
        assert len(final_state["answer"]) > 0

        # Should have confidence score
        assert final_state.get("confidence") is not None
        assert final_state["confidence"].final_score >= 0

        # Should have execution trace for notebook
        assert len(final_state.get("execution_trace", [])) > 0

    @pytest.mark.asyncio
    async def test_tier3_escalation(self) -> None:
        """Test that complex linking queries get escalated.

        Note: Data linking was downgraded from TIER_3 to TIER_2 to allow
        automated processing (see the architecture notes in README.md). This test verifies it is at
        least TIER_2 or above.
        """
        query = "Can you combine BLS employment data with Census education data?"

        final_state = await process_query(query)

        # Data linking is now TIER_2 (was TIER_3, downgraded per docs)
        assert final_state.get("tier") in [QueryTier.TIER_2, QueryTier.TIER_3]

        # Should have an answer
        assert final_state.get("answer") is not None

    @pytest.mark.asyncio
    async def test_comparison_query(self) -> None:
        """Test processing a comparison query."""
        query = "Compare unemployment in Texas vs California"

        final_state = await process_query(query)

        # Should be classified as comparison intent
        assert final_state.get("intent") == QueryIntent.COMPARISON

        # Should have multiple places
        entities = final_state.get("entities")
        assert entities is not None
        assert len(entities.places) >= 2

        # Should have an answer
        assert final_state.get("answer") is not None

    @pytest.mark.asyncio
    async def test_trend_query(self) -> None:
        """Test processing a trend analysis query."""
        query = "What is the unemployment trend over the past 5 years in Texas?"

        final_state = await process_query(query)

        # Should be classified as trend intent
        assert final_state.get("intent") == QueryIntent.TREND_ANALYSIS

        # Should have time entities
        entities = final_state.get("entities")
        assert entities is not None

        # Should have an answer
        assert final_state.get("answer") is not None


class TestAPIEndpoints:
    """Tests for the API endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        """Test the health check endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_classify_endpoint(self) -> None:
        """Test the classification endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/classify",
                json={"query": "What is unemployment in Texas?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "factual_lookup"
        assert data["tier"] == "tier_1"
        assert "intent_confidence" in data

    @pytest.mark.asyncio
    async def test_query_endpoint(self) -> None:
        """Test the main query endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "What is unemployment in Texas?"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "query_id" in data
        assert "answer" in data
        assert "confidence" in data
        assert "tier" in data
        assert "processing_time_ms" in data

    @pytest.mark.asyncio
    async def test_sources_endpoint(self) -> None:
        """Test the data sources listing endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/sources")

        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0

        # Check source structure
        source = data["sources"][0]
        assert "id" in source
        assert "name" in source
        assert "status" in source

    @pytest.mark.asyncio
    async def test_session_lifecycle(self) -> None:
        """Test session creation, retrieval, and deletion."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create session
            response = await client.post("/api/v1/session")
            assert response.status_code == 200
            data = response.json()
            assert data["created"] is True
            session_id = data["session_id"]

            # Get session
            response = await client.get(f"/api/v1/session/{session_id}")
            assert response.status_code == 200

            # Delete session
            response = await client.delete(f"/api/v1/session/{session_id}")
            assert response.status_code == 200

            # Verify deleted
            response = await client.get(f"/api/v1/session/{session_id}")
            assert response.status_code == 404


class TestConfidenceScoring:
    """Tests for confidence score calculations."""

    @pytest.mark.asyncio
    async def test_confidence_components(self) -> None:
        """Test that confidence scores have all expected components."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        confidence = final_state.get("confidence")
        assert confidence is not None

        # Check new factor fields are present
        assert hasattr(confidence, "answer_grounding")
        assert hasattr(confidence, "data_retrieval_quality")
        assert hasattr(confidence, "source_metadata_quality")
        assert hasattr(confidence, "query_answer_alignment")
        assert hasattr(confidence, "computation_complexity")
        assert hasattr(confidence, "final_score")

        # Legacy aliases still work
        assert hasattr(confidence, "query_interpretation")
        assert hasattr(confidence, "source_authority")
        assert hasattr(confidence, "retrieval_match")
        assert hasattr(confidence, "data_recency")
        assert hasattr(confidence, "computation_reliability")

        # All should be between 0 and 1
        # Since #132 a factor is None when it could not be computed, with the
        # reason recorded in `unavailable` — so range-check the ones that were
        # measured, and require an explanation for the ones that were not.
        for name in (
            "answer_grounding",
            "data_retrieval_quality",
            "source_metadata_quality",
            "query_answer_alignment",
            "computation_complexity",
        ):
            value = getattr(confidence, name)
            if value is None:
                assert name in confidence.unavailable, (
                    f"{name} is unmeasured but carries no reason"
                )
            else:
                assert 0 <= value <= 1

        assert 0 <= confidence.final_score <= 1
        assert 0 <= confidence.measured_weight <= 1

        # Legacy aliases in range when measured; they read through to the
        # same fields, so they are None whenever their target is.
        for alias in (
            "query_interpretation",
            "source_authority",
            "retrieval_match",
            "data_recency",
            "computation_reliability",
        ):
            value = getattr(confidence, alias)
            assert value is None or 0 <= value <= 1

    @pytest.mark.asyncio
    async def test_confidence_level_classification(self) -> None:
        """Test that confidence levels are correctly classified."""
        from data_concierge.core.models import ConfidenceLevel, ConfidenceScore

        # Test via legacy compute() — backward compat
        high = ConfidenceScore.compute(0.9, 0.95, 0.95, 0.9, 1.0)
        assert high.level == ConfidenceLevel.HIGH

        medium = ConfidenceScore.compute(0.7, 0.8, 0.7, 0.7, 0.8)
        assert medium.level == ConfidenceLevel.MEDIUM

        low = ConfidenceScore.compute(0.5, 0.6, 0.5, 0.5, 0.6)
        assert low.level == ConfidenceLevel.LOW

        very_low = ConfidenceScore.compute(0.3, 0.3, 0.3, 0.3, 0.3)
        assert very_low.level == ConfidenceLevel.VERY_LOW

        # Test via new compute_from_signals()
        high_sig = ConfidenceScore.compute_from_signals(
            answer_grounding=0.95,
            data_retrieval_quality=0.90,
            source_metadata_quality=0.85,
            query_answer_alignment=0.90,
            computation_complexity=0.95,
        )
        assert high_sig.level == ConfidenceLevel.HIGH

        # These must land in the LOW band (0.50-0.69). The previous values
        # summed to 0.43, i.e. VERY_LOW, so this assertion had never passed.
        low_sig = ConfidenceScore.compute_from_signals(
            answer_grounding=0.6,
            data_retrieval_quality=0.6,
            source_metadata_quality=0.6,
            query_answer_alignment=0.6,
            computation_complexity=0.6,
        )
        assert low_sig.level == ConfidenceLevel.LOW


class TestQuickAnswer:
    """Tests for quick answer mode (TIER_1 factual lookups skip notebooks)."""

    @pytest.mark.asyncio
    async def test_quick_answer_mode_enabled(self) -> None:
        """Test that simple factual queries enable quick answer mode."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        # TIER_1 factual lookup should enable quick answer mode
        assert final_state.get("tier") == QueryTier.TIER_1
        assert final_state.get("quick_answer_mode") is True

    @pytest.mark.asyncio
    async def test_quick_answer_skips_notebook(self) -> None:
        """Test that quick answer mode skips notebook generation."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        # Should NOT have a notebook
        assert final_state.get("notebook") is None

        # Should still have an answer
        assert final_state.get("answer") is not None
        assert len(final_state["answer"]) > 0

    @pytest.mark.asyncio
    async def test_quick_answer_has_source_links(self) -> None:
        """Test that quick answers include source links."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        source_links = final_state.get("source_links", [])
        # Should have at least one source link (Data Commons explorer link
        # is always generated from entities even without API data)
        assert len(source_links) > 0
        # Each source link should have name, url, description
        for link in source_links:
            assert "name" in link
            assert "url" in link
            assert "description" in link
            # URL should be a real URL
            assert link["url"].startswith("http")

    @pytest.mark.asyncio
    async def test_quick_answer_text(self) -> None:
        """Test that quick answer text is populated when data is retrieved."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        quick_answer = final_state.get("quick_answer")
        retrieved_data = final_state.get("retrieved_data")
        has_observations = (
            retrieved_data
            and hasattr(retrieved_data, "observations")
            and len(retrieved_data.observations) > 0
        )
        if has_observations:
            assert quick_answer is not None
            assert len(quick_answer) > 0
            assert quick_answer.endswith(".")
        else:
            # Without API keys, quick_answer may be empty
            assert quick_answer is not None  # field should exist

    @pytest.mark.asyncio
    async def test_comparison_query_not_quick_answer(self) -> None:
        """Test that comparison queries do NOT use quick answer mode."""
        query = "Compare unemployment in Texas vs California"

        final_state = await process_query(query)

        # Comparison queries should NOT be quick answer
        assert final_state.get("quick_answer_mode") is not True

    @pytest.mark.asyncio
    async def test_trend_query_not_quick_answer(self) -> None:
        """Test that trend queries do NOT use quick answer mode."""
        query = "What is the unemployment trend over the past 5 years in Texas?"

        final_state = await process_query(query)

        # Trend queries should NOT be quick answer
        assert final_state.get("quick_answer_mode") is not True


class TestQuickAnswerAPI:
    """Tests for quick answer API endpoints."""

    @pytest.mark.asyncio
    async def test_query_endpoint_returns_quick_answer(self) -> None:
        """Test that /query returns quick answer fields for simple queries."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": "What is the unemployment rate in Texas?",
                    "concierge_mode": "analyze",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["is_quick_answer"] is True
        assert data["notebook"] is None
        assert data["notebook_url"] is None

        # Should have source links
        assert isinstance(data["source_links"], list)

    @pytest.mark.asyncio
    async def test_quick_answer_submission_lifecycle(self) -> None:
        """Test submitting, reviewing, and approving a quick answer."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Submit a quick answer
            response = await client.post(
                "/api/v1/answers/submit",
                json={
                    "query": "What is the unemployment rate in Texas?",
                    "answer": "The unemployment rate in Texas is 4.1% (2024).",
                    "source_links": [
                        {
                            "name": "Bureau of Labor Statistics",
                            "url": "https://www.bls.gov/lau/",
                            "description": "Local Area Unemployment Statistics",
                        }
                    ],
                    "data_source": "bls",
                    "confidence": 0.92,
                    "variable": "unemployment rate",
                    "place": "Texas",
                    "date": "2024",
                    "value": "4.1%",
                },
            )
            assert response.status_code == 200
            submit_data = response.json()
            assert submit_data["status"] == "pending"
            submission_id = submit_data["submission_id"]

            # List submissions
            response = await client.get("/api/v1/answers/submissions?status_filter=pending")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] >= 1

            # Get specific submission
            response = await client.get(f"/api/v1/answers/submissions/{submission_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "What is the unemployment rate in Texas?"
            assert len(data["source_links"]) == 1

            # Approve the submission
            response = await client.post(
                f"/api/v1/answers/submissions/{submission_id}/approve",
                json={"reviewed_by": "test_admin", "admin_notes": "Verified correct"},
            )
            assert response.status_code == 200
            approve_data = response.json()
            assert "answer_id" in approve_data
            answer_id = approve_data["answer_id"]

            # List verified answers
            response = await client.get("/api/v1/verified-answers")
            assert response.status_code == 200
            data = response.json()
            assert data["count"] >= 1

            # Get specific verified answer
            response = await client.get(f"/api/v1/verified-answers/{answer_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "The unemployment rate in Texas is 4.1% (2024)."
            assert len(data["source_links"]) == 1

            # Search verified answers
            response = await client.post(
                "/api/v1/verified-answers/search",
                json={"query": "unemployment Texas"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["count"] >= 1

    @pytest.mark.asyncio
    async def test_quick_answer_rejection(self) -> None:
        """Test rejecting a quick answer submission."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Submit a quick answer
            response = await client.post(
                "/api/v1/answers/submit",
                json={
                    "query": "What is GDP?",
                    "answer": "GDP is a measure.",
                    "source_links": [
                        {"name": "BEA", "url": "https://www.bea.gov", "description": "BEA"}
                    ],
                },
            )
            assert response.status_code == 200
            submission_id = response.json()["submission_id"]

            # Reject it
            response = await client.post(
                f"/api/v1/answers/submissions/{submission_id}/reject",
                json={"reviewed_by": "test_admin", "admin_notes": "Too vague"},
            )
            assert response.status_code == 200

            # Verify it's rejected
            response = await client.get(f"/api/v1/answers/submissions/{submission_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_admin_stats_include_answers(self) -> None:
        """Test that admin stats include quick answer counts."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/notebooks/admin/stats")
            assert response.status_code == 200
            data = response.json()
            assert "total_answer_submissions" in data
            assert "answer_pending" in data
            assert "answer_approved" in data
            assert "answer_rejected" in data
            assert "total_verified_answers" in data


class TestNotebookGeneration:
    """Tests for notebook generation."""

    @pytest.mark.asyncio
    async def test_notebook_generated_for_complex_queries(self) -> None:
        """Test that notebooks are generated for non-TIER_1 queries."""
        query = "Compare unemployment in Texas vs California"

        final_state = await process_query(query)

        # Complex queries should still generate notebooks
        if not final_state.get("quick_answer_mode"):
            notebook = final_state.get("notebook")
            assert notebook is not None
            assert notebook.notebook_json is not None
            assert notebook.filename is not None
            assert notebook.filename.endswith(".ipynb")

    @pytest.mark.asyncio
    async def test_notebook_structure(self) -> None:
        """Test notebook has expected structure for complex queries."""
        query = "Compare unemployment in Texas vs California"

        final_state = await process_query(query)

        notebook = final_state.get("notebook")
        if notebook is not None:
            nb_json = notebook.notebook_json
            assert "cells" in nb_json
            assert "metadata" in nb_json
            assert len(nb_json["cells"]) > 0

            cell_types = [cell.get("cell_type") for cell in nb_json["cells"]]
            assert "markdown" in cell_types
            assert "code" in cell_types


class TestCitations:
    """Tests for citation generation."""

    @pytest.mark.asyncio
    async def test_citations_generated(self) -> None:
        """Test that citations are generated for queries with data."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        citations = final_state.get("citations", [])
        retrieved_data = final_state.get("retrieved_data")
        has_observations = (
            retrieved_data
            and hasattr(retrieved_data, "observations")
            and len(retrieved_data.observations) > 0
        )
        # Should have at least one citation if data was actually retrieved
        if has_observations:
            assert len(citations) > 0

    @pytest.mark.asyncio
    async def test_citation_structure(self) -> None:
        """Test citation has expected structure."""
        query = "What is the unemployment rate in Texas?"

        final_state = await process_query(query)

        citations = final_state.get("citations", [])
        if citations:
            citation = citations[0]
            assert hasattr(citation, "source")
            assert hasattr(citation, "dataset_title")
            assert hasattr(citation, "url")
            assert hasattr(citation, "access_date")
