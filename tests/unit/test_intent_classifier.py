"""Tests for the intent classifier."""

import pytest

from data_concierge.core.models import QueryIntent, QueryTier
from data_concierge.gateway.intent_classifier import IntentClassifier


@pytest.fixture
def classifier() -> IntentClassifier:
    """Create a classifier instance for testing."""
    return IntentClassifier()


class TestIntentClassification:
    """Tests for intent classification."""

    def test_factual_lookup_intent(self, classifier: IntentClassifier) -> None:
        """Test classification of factual lookup queries."""
        queries = [
            "What is the unemployment rate in Texas?",
            "What's the population of California?",
            "How many people live in New York?",
        ]

        for query in queries:
            intent, confidence = classifier.classify_intent(query)
            assert intent == QueryIntent.FACTUAL_LOOKUP
            assert confidence >= 0.5

    def test_comparison_intent(self, classifier: IntentClassifier) -> None:
        """Test classification of comparison queries."""
        queries = [
            "Compare unemployment rates between Texas and California",
            "Which state has higher income, Texas vs California?",
            "Is unemployment higher in Texas than in New York?",
        ]

        for query in queries:
            intent, confidence = classifier.classify_intent(query)
            assert intent == QueryIntent.COMPARISON
            assert confidence >= 0.5

    def test_trend_intent(self, classifier: IntentClassifier) -> None:
        """Test classification of trend analysis queries."""
        queries = [
            "What is the unemployment trend over the past 5 years?",
            "How has population changed from 2010 to 2020?",
            "Show me the historical GDP growth",
        ]

        for query in queries:
            intent, confidence = classifier.classify_intent(query)
            assert intent == QueryIntent.TREND_ANALYSIS
            assert confidence >= 0.5


class TestComplexityClassification:
    """Tests for complexity/tier classification."""

    def test_tier1_simple_queries(self, classifier: IntentClassifier) -> None:
        """Test that simple queries are classified as Tier 1."""
        queries = [
            "What is unemployment in Texas?",
            "Population of California in 2020",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result["tier"] == QueryTier.TIER_1

    def test_tier2_complex_queries(self, classifier: IntentClassifier) -> None:
        """Test that complex queries are classified as Tier 2."""
        queries = [
            "Compare unemployment trends across all states from 2015 to 2023",
            "What is the correlation between education and income by state?",
        ]

        for query in queries:
            result = classifier.classify(query)
            # Allow Tier 2 or 3 for complex queries
            assert result["tier"] in [QueryTier.TIER_2, QueryTier.TIER_3]

    def test_tier3_linking_queries(self, classifier: IntentClassifier) -> None:
        """Test that data linking queries are classified correctly.

        Note: Data linking was downgraded from TIER_3 to TIER_2 to allow
        automated processing (see the architecture notes in README.md).
        """
        queries = [
            "Can you link employment data with education outcomes?",
            "Combine census data with BLS employment statistics",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert result["tier"] in [QueryTier.TIER_2, QueryTier.TIER_3]


class TestClassificationConfidence:
    """Tests for confidence scoring in classification."""

    def test_high_confidence_for_clear_queries(
        self,
        classifier: IntentClassifier,
    ) -> None:
        """Test that clear queries get higher confidence."""
        query = "What is the unemployment rate in Texas?"
        result = classifier.classify(query)
        assert result["combined_confidence"] >= 0.6

    def test_classification_returns_all_fields(
        self,
        classifier: IntentClassifier,
    ) -> None:
        """Test that classification returns all expected fields."""
        query = "What is unemployment in Texas?"
        result = classifier.classify(query)

        assert "query" in result
        assert "intent" in result
        assert "intent_confidence" in result
        assert "tier" in result
        assert "tier_confidence" in result
        assert "combined_confidence" in result
