"""Intent classification for routing queries to appropriate tiers."""

import re
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.core.models import QueryIntent, QueryTier

logger = get_logger(__name__)


class IntentClassifier:
    """Classifies query intent and complexity for routing decisions."""

    # Keywords and patterns for intent classification
    FACTUAL_PATTERNS = [
        r"\bwhat is\b",
        r"\bwhat's\b",
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bwhat was\b",
        r"\bcurrent\b",
        r"\blatest\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\bmore than\b",
        r"\bless than\b",
        r"\bhigher\b.*\bthan\b",  # Non-adjacent "higher ... than"
        r"\blower\b.*\bthan\b",   # Non-adjacent "lower ... than"
        r"\bgreater\b.*?\bthan\b",
        r"\bwhich.*(higher|lower|more|less|greater)\b",
        r"\bcorrelation\b",
        r"\bcorrelate\b",
        r"\brelationship between\b",
    ]

    TREND_PATTERNS = [
        r"\btrend\b",
        r"\bover time\b",
        r"\bhistorical\b",
        r"\bchange in\b",
        r"\bgrowth\b",
        r"\bdecline\b",
        r"\bfrom \d{4} to \d{4}\b",
        r"\bpast \d+ years\b",
    ]

    DISCOVERY_PATTERNS = [
        r"\bfind data\b",
        r"\bdatasets?\b",
        r"\bwhere can i find\b",
        r"\bsources for\b",
        r"\bavailable data\b",
        r"\blooking for data\b",
    ]

    METHODOLOGY_PATTERNS = [
        r"\bhow is .* measured\b",
        r"\bmethodology\b",
        r"\bdefinition of\b",
        r"\bwhat does .* mean\b",
        r"\bhow do they calculate\b",
    ]

    LINKING_PATTERNS = [
        r"\bcombine\b",
        r"\blink\b",
        r"\bjoin\b",
        r"\bmerge\b",
        r"\bintegrat\b",
        r"\bcross-reference\b",
        r"\bmatch.*across\b",
    ]

    # Complexity indicators
    SIMPLE_INDICATORS = [
        r"^what is ",
        r"^how many ",
        r"^what's the ",
        r" in \d{4}$",
        r" in [a-zA-Z]+$",  # Ends with a place name
    ]

    COMPLEX_INDICATORS = [
        r"\band\b.*\band\b",  # Multiple "and"s
        r"\bfor each\b",
        r"\bby .+ and .+\b",  # Multiple groupings
        r"\bif .+ then\b",
        r"\bassuming\b",
        r"\badjusted for\b",
        r"\bcorrelation\b",
        r"\ball states\b",
        r"\bacross.*(states|counties|regions)\b",
        r"\bmultiple\b",
        r"\bby state\b",
        r"\bper capita\b",
        r"\bbreakdown\b",
    ]

    def __init__(self) -> None:
        """Initialize the classifier with compiled patterns."""
        self._factual_patterns = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]
        self._comparison_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self._trend_patterns = [re.compile(p, re.IGNORECASE) for p in self.TREND_PATTERNS]
        self._discovery_patterns = [re.compile(p, re.IGNORECASE) for p in self.DISCOVERY_PATTERNS]
        self._methodology_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.METHODOLOGY_PATTERNS
        ]
        self._linking_patterns = [re.compile(p, re.IGNORECASE) for p in self.LINKING_PATTERNS]
        self._simple_patterns = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_INDICATORS]
        self._complex_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_INDICATORS]

    def _count_matches(self, query: str, patterns: list[re.Pattern[str]]) -> int:
        """Count how many patterns match the query."""
        return sum(1 for p in patterns if p.search(query))

    def classify_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Classify the intent of a query.

        Args:
            query: The user's query text

        Returns:
            Tuple of (QueryIntent, confidence score)
        """
        query = query.strip().lower()

        # Score each intent type
        scores: dict[QueryIntent, int] = {
            QueryIntent.FACTUAL_LOOKUP: self._count_matches(query, self._factual_patterns),
            QueryIntent.COMPARISON: self._count_matches(query, self._comparison_patterns),
            QueryIntent.TREND_ANALYSIS: self._count_matches(query, self._trend_patterns),
            QueryIntent.DATASET_DISCOVERY: self._count_matches(query, self._discovery_patterns),
            QueryIntent.METHODOLOGY_QUESTION: self._count_matches(query, self._methodology_patterns),
            QueryIntent.DATA_LINKING: self._count_matches(query, self._linking_patterns),
        }

        # Find the intent with the highest score
        max_score = max(scores.values())

        if max_score == 0:
            # No patterns matched - could be out of scope or needs LLM classification
            return QueryIntent.FACTUAL_LOOKUP, 0.5  # Default with low confidence

        best_intent = max(scores, key=lambda k: scores[k])

        # Calculate confidence based on score distinctiveness
        total_matches = sum(scores.values())
        confidence = min(0.95, 0.5 + (max_score / max(total_matches, 1)) * 0.5)

        logger.debug(
            "Intent classified",
            query=query[:50],
            intent=best_intent.value,
            confidence=confidence,
            scores=scores,
        )

        return best_intent, confidence

    def classify_complexity(self, query: str, intent: QueryIntent) -> tuple[QueryTier, float]:
        """Classify the complexity tier of a query.

        Args:
            query: The user's query text
            intent: Already-classified intent

        Returns:
            Tuple of (QueryTier, confidence score)
        """
        query = query.strip()

        # Data linking is complex but still processable (was TIER_3, now TIER_2)
        if intent == QueryIntent.DATA_LINKING:
            return QueryTier.TIER_2, 0.9

        # Count complexity indicators
        simple_score = self._count_matches(query, self._simple_patterns)
        complex_score = self._count_matches(query, self._complex_patterns)

        # Query length is also a factor
        word_count = len(query.split())
        if word_count > 30:
            complex_score += 1
        elif word_count < 10:
            simple_score += 1

        # Intent-based adjustments
        if intent in [QueryIntent.FACTUAL_LOOKUP, QueryIntent.METHODOLOGY_QUESTION]:
            simple_score += 1
        elif intent in [QueryIntent.TREND_ANALYSIS, QueryIntent.COMPARISON]:
            complex_score += 1
        elif intent == QueryIntent.DATASET_DISCOVERY:
            complex_score += 0.5

        # Determine tier (cap at TIER_2 to avoid human escalation)
        # TIER_3 is reserved for explicit human escalation requests
        if complex_score >= 2 or intent == QueryIntent.DATA_LINKING:
            # Downgrade from TIER_3 to TIER_2 to allow automated processing
            tier = QueryTier.TIER_2
            confidence = min(0.9, 0.6 + complex_score * 0.1)
        elif complex_score > simple_score:
            tier = QueryTier.TIER_2
            confidence = min(0.85, 0.5 + (complex_score - simple_score) * 0.1)
        else:
            tier = QueryTier.TIER_1
            confidence = min(0.9, 0.6 + simple_score * 0.1)

        logger.debug(
            "Complexity classified",
            query=query[:50],
            tier=tier.value,
            confidence=confidence,
            simple_score=simple_score,
            complex_score=complex_score,
        )

        return tier, confidence

    def classify(self, query: str) -> dict[str, Any]:
        """Fully classify a query's intent and complexity.

        Args:
            query: The user's query text

        Returns:
            Classification results including intent, tier, and confidence
        """
        intent, intent_confidence = self.classify_intent(query)
        tier, tier_confidence = self.classify_complexity(query, intent)

        result = {
            "query": query,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "tier": tier,
            "tier_confidence": tier_confidence,
            "combined_confidence": (intent_confidence + tier_confidence) / 2,
        }

        logger.info(
            "Query classified",
            intent=intent.value,
            tier=tier.value,
            confidence=result["combined_confidence"],
        )

        return result


# Global classifier instance
intent_classifier = IntentClassifier()
