"""Query Parser Agent for entity extraction and query formalization."""

import hashlib
import re
from datetime import datetime
from typing import Any

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.logging import get_logger
from data_concierge.core.models import (
    DemographicEntity,
    ParsedEntities,
    PlaceEntity,
    TimeEntity,
    VariableEntity,
)
from data_concierge.gateway.intent_classifier import intent_classifier

logger = get_logger(__name__)


class QueryParserAgent(BaseAgent):
    """Agent responsible for parsing and structuring user queries.

    Extracts entities (places, variables, time periods, demographics),
    resolves ambiguities, and formalizes the query for downstream processing.
    """

    name = "query_parser"
    description = "Extracts entities and formalizes queries"

    # Common place name mappings to Data Commons IDs
    PLACE_MAPPINGS: dict[str, tuple[str, str, str]] = {
        # Country / national
        "united states": ("country/USA", "United States", "Country"),
        "usa": ("country/USA", "United States", "Country"),
        "us": ("country/USA", "United States", "Country"),
        "national": ("country/USA", "United States", "Country"),
        "nation": ("country/USA", "United States", "Country"),
        "america": ("country/USA", "United States", "Country"),
        # All 50 states -> (dcid, name, type)
        "alabama": ("geoId/01", "Alabama", "State"),
        "alaska": ("geoId/02", "Alaska", "State"),
        "arizona": ("geoId/04", "Arizona", "State"),
        "arkansas": ("geoId/05", "Arkansas", "State"),
        "california": ("geoId/06", "California", "State"),
        "golden state": ("geoId/06", "California", "State"),
        "colorado": ("geoId/08", "Colorado", "State"),
        "connecticut": ("geoId/09", "Connecticut", "State"),
        "delaware": ("geoId/10", "Delaware", "State"),
        "florida": ("geoId/12", "Florida", "State"),
        "georgia": ("geoId/13", "Georgia", "State"),
        "hawaii": ("geoId/15", "Hawaii", "State"),
        "idaho": ("geoId/16", "Idaho", "State"),
        "illinois": ("geoId/17", "Illinois", "State"),
        "indiana": ("geoId/18", "Indiana", "State"),
        "iowa": ("geoId/19", "Iowa", "State"),
        "kansas": ("geoId/20", "Kansas", "State"),
        "kentucky": ("geoId/21", "Kentucky", "State"),
        "louisiana": ("geoId/22", "Louisiana", "State"),
        "maine": ("geoId/23", "Maine", "State"),
        "maryland": ("geoId/24", "Maryland", "State"),
        "massachusetts": ("geoId/25", "Massachusetts", "State"),
        "michigan": ("geoId/26", "Michigan", "State"),
        "minnesota": ("geoId/27", "Minnesota", "State"),
        "mississippi": ("geoId/28", "Mississippi", "State"),
        "missouri": ("geoId/29", "Missouri", "State"),
        "montana": ("geoId/30", "Montana", "State"),
        "nebraska": ("geoId/31", "Nebraska", "State"),
        "nevada": ("geoId/32", "Nevada", "State"),
        "new hampshire": ("geoId/33", "New Hampshire", "State"),
        "new jersey": ("geoId/34", "New Jersey", "State"),
        "new mexico": ("geoId/35", "New Mexico", "State"),
        "new york": ("geoId/36", "New York", "State"),
        "north carolina": ("geoId/37", "North Carolina", "State"),
        "north dakota": ("geoId/38", "North Dakota", "State"),
        "ohio": ("geoId/39", "Ohio", "State"),
        "oklahoma": ("geoId/40", "Oklahoma", "State"),
        "oregon": ("geoId/41", "Oregon", "State"),
        "pennsylvania": ("geoId/42", "Pennsylvania", "State"),
        "rhode island": ("geoId/44", "Rhode Island", "State"),
        "south carolina": ("geoId/45", "South Carolina", "State"),
        "south dakota": ("geoId/46", "South Dakota", "State"),
        "tennessee": ("geoId/47", "Tennessee", "State"),
        "texas": ("geoId/48", "Texas", "State"),
        "lone star state": ("geoId/48", "Texas", "State"),
        "utah": ("geoId/49", "Utah", "State"),
        "vermont": ("geoId/50", "Vermont", "State"),
        "virginia": ("geoId/51", "Virginia", "State"),
        "washington": ("geoId/53", "Washington", "State"),
        "west virginia": ("geoId/54", "West Virginia", "State"),
        "wisconsin": ("geoId/55", "Wisconsin", "State"),
        "wyoming": ("geoId/56", "Wyoming", "State"),
        # DC
        "district of columbia": ("geoId/11", "District of Columbia", "State"),
        "dc": ("geoId/11", "District of Columbia", "State"),
        "washington dc": ("geoId/11", "District of Columbia", "State"),
        "washington d.c.": ("geoId/11", "District of Columbia", "State"),
        # Major cities -> (dcid, name, type)
        "chicago": ("geoId/1714000", "Chicago", "City"),
        "los angeles": ("geoId/0644000", "Los Angeles", "City"),
        "new york city": ("geoId/3651000", "New York City", "City"),
        "nyc": ("geoId/3651000", "New York City", "City"),
        "houston": ("geoId/4835000", "Houston", "City"),
        "phoenix": ("geoId/0455000", "Phoenix", "City"),
        "philadelphia": ("geoId/4260000", "Philadelphia", "City"),
        "san antonio": ("geoId/4865000", "San Antonio", "City"),
        "san diego": ("geoId/0666000", "San Diego", "City"),
        "dallas": ("geoId/4819000", "Dallas", "City"),
        "san jose": ("geoId/0668000", "San Jose", "City"),
    }

    # Common variable name mappings (using correct Data Commons DCIDs)
    # See: https://datacommons.org/tools/statvar
    # Maps keyword -> (primary_dcid, display_name, unit, [fallback_dcids])
    VARIABLE_MAPPINGS: dict[str, tuple[str, str, str | None]] = {
        # Unemployment - Multiple DCIDs to try
        # UnemploymentRate_Person is the standard rate variable
        "unemployment": ("Count_Person_Unemployed", "Unemployed Persons", None),
        "unemployment rate": ("UnemploymentRate_Person", "Unemployment Rate", "%"),
        "jobless rate": ("UnemploymentRate_Person", "Unemployment Rate", "%"),
        "unemployed": ("Count_Person_Unemployed", "Unemployed Persons", None),
        # Population - Count_Person is universal
        "population": ("Count_Person", "Population", None),
        "total population": ("Count_Person", "Population", None),
        # Income
        "median income": ("Median_Income_Person", "Median Income", "USD"),
        "median household income": ("Median_Income_Household", "Median Household Income", "USD"),
        "income": ("Median_Income_Household", "Median Household Income", "USD"),
        # Poverty
        "poverty rate": ("Count_Person_BelowPovertyLevelInThePast12Months", "Below Poverty Level", None),
        "poverty": ("Count_Person_BelowPovertyLevelInThePast12Months", "Below Poverty Level", None),
        # Education
        "education": ("Count_Person_EducationalAttainmentBachelorsDegreeOrHigher", "Bachelor's Degree or Higher", None),
        "college degree": ("Count_Person_EducationalAttainmentBachelorsDegreeOrHigher", "Bachelor's Degree or Higher", None),
        # Health
        "life expectancy": ("LifeExpectancy_Person", "Life Expectancy", "years"),
        # GDP
        "gdp": ("Amount_EconomicActivity_GrossDomesticProduction_RealValue", "GDP", "USD"),
        "gross domestic product": ("Amount_EconomicActivity_GrossDomesticProduction_RealValue", "GDP", "USD"),
        # Crime
        "crime rate": ("Count_CriminalActivities_CombinedCrime", "Crime Rate", None),
        "crime": ("Count_CriminalActivities_CombinedCrime", "Crime Rate", None),
    }

    # Fallback variable DCIDs to try if primary returns no data
    # Maps primary DCID -> list of alternative DCIDs to try
    VARIABLE_FALLBACKS: dict[str, list[str]] = {
        "UnemploymentRate_Person": [
            "Count_Person_Unemployed",
            "Count_Person_InLaborForce_Unemployed",
        ],
        "Median_Income_Household": [
            "Median_Income_Person",
            "Mean_Income_Person",
        ],
        "Count_Person_BelowPovertyLevelInThePast12Months": [
            "Percent_Person_BelowPovertyLevelInThePast12Months",
        ],
    }

    # Temporal patterns - capture full 4-digit year
    YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
    RELATIVE_TIME_PATTERNS = {
        r"\blast year\b": -1,
        r"\bthis year\b": 0,
        r"\b(\d+) years? ago\b": None,  # Extract from match
    }

    # Demographic patterns
    DEMOGRAPHIC_PATTERNS = {
        "age": [
            (r"\byoung people\b", "age_16_24"),
            (r"\byouth\b", "age_16_24"),
            (r"\belderly\b", "age_65_plus"),
            (r"\bseniors?\b", "age_65_plus"),
            (r"\badults?\b", "age_18_plus"),
            (r"\bchildren\b", "age_under_18"),
        ],
        "gender": [
            (r"\bmen\b", "male"),
            (r"\bmale\b", "male"),
            (r"\bwomen\b", "female"),
            (r"\bfemale\b", "female"),
        ],
    }

    def __init__(self) -> None:
        """Initialize the query parser."""
        super().__init__()
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._demographic_compiled: dict[str, list[tuple[re.Pattern[str], str]]] = {}
        for attr, patterns in self.DEMOGRAPHIC_PATTERNS.items():
            self._demographic_compiled[attr] = [
                (re.compile(p, re.IGNORECASE), v) for p, v in patterns
            ]

    async def process(self, state: GraphState) -> GraphState:
        """Parse the query and extract structured information.

        Args:
            state: Current graph state

        Returns:
            Updated state with parsed entities and classification
        """
        self.logger.info("Parsing query", query=state["query"][:50])

        query = state["query"]

        # Classify intent and tier
        classification = intent_classifier.classify(query)
        state["intent"] = classification["intent"]
        state["tier"] = classification["tier"]

        # Extract entities
        entities = self._extract_entities(query)
        state["entities"] = entities

        # Normalize query for caching
        normalized = self._normalize_query(query, entities)
        state["normalized_query"] = normalized
        state["query_hash"] = hashlib.sha256(normalized.encode()).hexdigest()

        # Calculate parsing confidence
        parse_confidence = self._calculate_confidence(entities, classification)
        state["parse_confidence"] = parse_confidence

        # Add to execution trace
        state["execution_trace"].append({
            "agent": self.name,
            "action": "parse_query",
            "input": query,
            "entities": {
                "places": [p.model_dump() for p in entities.places],
                "variables": [v.model_dump() for v in entities.variables],
                "times": [t.model_dump() for t in entities.times],
                "demographics": [d.model_dump() for d in entities.demographics],
            },
            "intent": classification["intent"].value,
            "tier": classification["tier"].value,
            "confidence": parse_confidence,
        })

        self.logger.info(
            "Query parsed",
            intent=classification["intent"].value,
            tier=classification["tier"].value,
            entity_count=sum([
                len(entities.places),
                len(entities.variables),
                len(entities.times),
                len(entities.demographics),
            ]),
            confidence=parse_confidence,
        )

        return state

    def _extract_entities(self, query: str) -> ParsedEntities:
        """Extract all entity types from the query.

        Args:
            query: User's query text

        Returns:
            ParsedEntities with extracted entities
        """
        query_lower = query.lower()

        places = self._extract_places(query_lower)
        variables = self._extract_variables(query_lower)
        times = self._extract_times(query, query_lower)
        demographics = self._extract_demographics(query_lower)

        return ParsedEntities(
            places=places,
            variables=variables,
            times=times,
            demographics=demographics,
        )

    def _extract_places(self, query_lower: str) -> list[PlaceEntity]:
        """Extract place entities from the query."""
        places = []

        for name, (dcid, display_name, place_type) in self.PLACE_MAPPINGS.items():
            if name in query_lower:
                places.append(PlaceEntity(
                    dcid=dcid,
                    name=display_name,
                    place_type=place_type,
                ))

        return places

    def _extract_variables(self, query_lower: str) -> list[VariableEntity]:
        """Extract statistical variable entities from the query.

        Matches are resolved most-specific-first. "unemployment rate" and
        "unemployment" both occur in the same query, but they map to
        different variables — a rate versus a headcount — and emitting both
        in dict order put the headcount first, so a question about the rate
        was answered with a count. A term that is a substring of another
        matched term is therefore suppressed by it.
        """
        matched = [name for name in self.VARIABLE_MAPPINGS if name in query_lower]

        # "unemployment" loses to "unemployment rate"; unrelated terms that
        # merely co-occur (e.g. "population" and "poverty rate") both stand.
        specific = [
            name
            for name in matched
            if not any(other != name and name in other for other in matched)
        ]

        variables = []
        for name in sorted(specific, key=len, reverse=True):
            dcid, display_name, unit = self.VARIABLE_MAPPINGS[name]
            variables.append(VariableEntity(dcid=dcid, name=display_name, unit=unit))

        return variables

    def _extract_times(self, query: str, query_lower: str) -> list[TimeEntity]:
        """Extract temporal entities from the query."""
        times = []
        current_year = datetime.now().year

        # Look for explicit years
        year_matches = self.YEAR_PATTERN.findall(query)
        for year in year_matches:
            full_year = f"{year}"
            times.append(TimeEntity(
                start_date=f"{full_year}-01-01",
                period_type="year",
            ))

        # Look for relative time references
        if "last year" in query_lower:
            times.append(TimeEntity(
                start_date=f"{current_year - 1}-01-01",
                period_type="year",
            ))
        elif "this year" in query_lower:
            times.append(TimeEntity(
                start_date=f"{current_year}-01-01",
                period_type="year",
            ))

        # Look for "X years ago"
        years_ago_match = re.search(r"(\d+) years? ago", query_lower)
        if years_ago_match:
            years_back = int(years_ago_match.group(1))
            times.append(TimeEntity(
                start_date=f"{current_year - years_back}-01-01",
                period_type="year",
            ))

        return times

    def _extract_demographics(self, query_lower: str) -> list[DemographicEntity]:
        """Extract demographic filter entities from the query."""
        demographics = []

        for attr, patterns in self._demographic_compiled.items():
            for pattern, value in patterns:
                if pattern.search(query_lower):
                    demographics.append(DemographicEntity(
                        attribute=attr,
                        value=value,
                    ))
                    break  # Only one match per attribute

        return demographics

    def _normalize_query(self, query: str, entities: ParsedEntities) -> str:
        """Normalize query for consistent caching.

        Replaces entity mentions with canonical IDs.

        Args:
            query: Original query
            entities: Extracted entities

        Returns:
            Normalized query string
        """
        normalized = query.lower().strip()

        # Replace place names with dcids
        for place in entities.places:
            for name in self.PLACE_MAPPINGS:
                if name in normalized:
                    normalized = normalized.replace(name, place.dcid)
                    break

        # Replace variable names with dcids
        for var in entities.variables:
            for name in self.VARIABLE_MAPPINGS:
                if name in normalized:
                    normalized = normalized.replace(name, var.dcid)
                    break

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def _calculate_confidence(
        self,
        entities: ParsedEntities,
        classification: dict[str, Any],
    ) -> float:
        """Calculate parsing confidence score.

        Args:
            entities: Extracted entities
            classification: Intent classification result

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from intent classification
        base_confidence = classification["combined_confidence"]

        # Adjust based on entity extraction
        entity_score = 0.0

        # Having at least one variable is important
        if entities.variables:
            entity_score += 0.3

        # Having a place helps
        if entities.places:
            entity_score += 0.2

        # Time context is useful
        if entities.times:
            entity_score += 0.1

        # Combine scores
        final_confidence = 0.6 * base_confidence + 0.4 * min(entity_score, 1.0)

        return min(1.0, max(0.0, final_confidence))
