"""Data Concierge Agent - The first point of contact for users.

This agent acts as a knowledgeable data concierge that:
1. Understands what data the user is looking for
2. Recommends relevant data sources and specific datasets
3. Provides direct links to data portals and APIs
4. Can fetch LIVE data from various government sources
5. Offers to perform deeper analysis if requested

The concierge approach provides a more helpful user experience by guiding
users to the right data sources while making it easy to explore further.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from data_concierge.core.config import settings
from data_concierge.core.data_source_registry import (
    DataCategory,
    get_data_source_registry,
)
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# Try to import Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Try to import connectors
try:
    from data_concierge.data_layer.connectors.bls import BLSClient
    BLS_AVAILABLE = True
except ImportError:
    BLS_AVAILABLE = False
    BLSClient = None

try:
    from data_concierge.data_layer.connectors.fred import FREDClient
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    FREDClient = None

try:
    from data_concierge.data_layer.connectors.census import CensusClient
    CENSUS_AVAILABLE = True
except ImportError:
    CENSUS_AVAILABLE = False
    CensusClient = None

try:
    from data_concierge.data_layer.connectors.bea import BEAClient
    BEA_AVAILABLE = True
except ImportError:
    BEA_AVAILABLE = False
    BEAClient = None


@dataclass
class DataSourceRecommendation:
    """A recommendation for a data source."""
    source_id: str
    source_name: str
    description: str
    relevance_score: float
    portal_url: str | None = None
    api_url: str | None = None
    recommended_datasets: list[dict[str, Any]] = field(default_factory=list)
    access_instructions: str | None = None
    example_queries: list[str] = field(default_factory=list)


@dataclass
class ConciergeResponse:
    """Response from the Data Concierge."""
    query_id: str
    user_query: str
    understanding: str  # How the concierge understood the query
    recommendations: list[DataSourceRecommendation]
    suggested_next_steps: list[str]
    offer_analysis: bool  # Whether to offer deeper analysis
    analysis_prompt: str | None  # Prompt for if user wants analysis
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0


class DataConciergeAgent:
    """The Data Concierge Agent - first point of contact for data queries.

    This agent provides a conversational interface that:
    1. Interprets the user's data needs
    2. Searches the registry for relevant sources
    3. Provides specific recommendations with links
    4. Offers to perform deeper analysis

    The concierge is designed to be helpful and informative, guiding users
    to the right data sources while making it easy to explore further.

    The agent operates in two modes:
    - CONCIERGE mode: For vague first questions, acts as a guide to data sources
    - ANALYSIS mode: For specific questions or follow-ups, fetches and analyzes data
    """

    # System prompt for CONCIERGE mode (vague first questions)
    # This prompt guides users to data sources without immediately analyzing
    CONCIERGE_SYSTEM_PROMPT = """You are a friendly and knowledgeable Data Concierge, an expert guide to government and public data sources. Your role is to help users DISCOVER what data is available and WHERE to find it.

## YOUR ROLE

Since the user is just starting their exploration, your job is to:
1. Understand what topic/information they're interested in
2. Explain what types of data are available in that area
3. Recommend the most relevant data sources with links
4. Ask clarifying questions to better understand their needs
5. Let them know you can analyze data for them if they want specific insights

## AVAILABLE TOOLS

### Discovery Tools (use these primarily)
- `search_data_sources`: Search for data sources by keywords
- `get_source_details`: Get detailed info about a specific source
- `get_category_sources`: Find sources by category
- `list_all_sources`: List all available sources

## RESPONSE STYLE

Be conversational and informative. Your responses should:
- Welcome the user and acknowledge their interest
- Use search tools to find relevant data sources
- Present 2-3 top recommendations with clear descriptions
- Explain what kinds of questions each source can answer
- Provide direct links to data portals
- End with "Would you like me to analyze any specific data for you?" or similar

## EXAMPLE INTERACTION

User: "I'm interested in employment data"

You should:
1. Search for employment-related sources
2. Present BLS, Census, and other relevant sources
3. Explain what each offers (unemployment rates, labor statistics, etc.)
4. Provide portal links
5. Ask what specific aspect interests them

## IMPORTANT

- DO NOT fetch live data unless they ask for specific numbers
- Focus on GUIDING and EXPLAINING what's available
- Be welcoming and encourage them to explore
- Ask clarifying questions to understand their needs better"""

    # System prompt for ANALYSIS mode (specific questions or follow-ups)
    ANALYSIS_SYSTEM_PROMPT = """You are a friendly and knowledgeable Data Concierge, an expert guide to government and public data sources. Your role is to help users find the data they need AND fetch live data when requested.

## YOUR ROLE

When a user asks about data or statistics, you should:
1. Understand what information they're looking for
2. Identify which data sources would be most relevant
3. Recommend specific datasets with links
4. If the user wants actual numbers, USE THE LIVE DATA TOOLS to fetch real data
5. Present the data clearly with context

## AVAILABLE TOOLS

### Discovery Tools (find sources)
- `search_data_sources`: Search for data sources by keywords
- `get_source_details`: Get detailed info about a specific source
- `get_category_sources`: Find sources by category
- `list_all_sources`: List all available sources

### LIVE Data Tools (fetch actual data)
- `fetch_bls_unemployment`: Get unemployment rate data (national or by state)
- `fetch_bls_cpi`: Get Consumer Price Index / inflation data
- `fetch_fred_data`: Get GDP, interest rates, inflation from Federal Reserve
- `fetch_census_data`: Get population, income, poverty, housing data
- `fetch_bea_gdp`: Get GDP data (national or by state)

## WHEN TO USE LIVE DATA TOOLS

Use the live data tools when:
- User asks "What is the unemployment rate in [state]?" → Use fetch_bls_unemployment
- User asks about inflation or CPI → Use fetch_bls_cpi
- User asks about GDP, interest rates → Use fetch_fred_data or fetch_bea_gdp
- User asks about population, income, poverty → Use fetch_census_data
- User explicitly asks for "current", "latest", or "actual" numbers

## RESPONSE STYLE

Be conversational and helpful. Your responses should:
- Acknowledge what the user is looking for
- Use live data tools to get actual numbers when appropriate
- Present data clearly with source attribution
- Explain what the numbers mean
- Offer to explore further

## EXAMPLE INTERACTIONS

User: "What is the unemployment rate in Texas?"

You should:
1. Use `fetch_bls_unemployment` with state="Texas"
2. Present the latest rate and recent trends
3. Provide context and the source link

User: "Where can I find GDP data?"

You should:
1. Use `search_data_sources` with keywords ["gdp"]
2. Recommend BEA and FRED as primary sources
3. Offer to fetch the actual GDP numbers

## IMPORTANT

- When users ask for specific data, FETCH IT using the live tools
- Always cite your source (BLS, FRED, Census, BEA)
- Provide links for users who want to explore more
- If a tool returns an error, explain what happened and suggest alternatives

## CONTEXT

The user is asking a follow-up question or a specific analytical question.
Focus on fetching real data and providing insights, not just pointing to sources."""

    def __init__(self) -> None:
        """Initialize the Data Concierge Agent."""
        self.logger = get_logger("data_concierge")
        self.registry = get_data_source_registry()
        self._anthropic_client: Any = None

        # Model configuration
        self.model = settings.llm_model

        # Initialize data source connectors
        self._bls_client: BLSClient | None = None
        self._fred_client: FREDClient | None = None
        self._census_client: CensusClient | None = None
        self._bea_client: BEAClient | None = None

        # Define tools
        self.tools = self._define_tools()

        self.logger.info(
            "DataConciergeAgent initialized",
            model=self.model,
            sources_available=len(self.registry.get_enabled_sources()),
            bls_available=BLS_AVAILABLE,
            fred_available=FRED_AVAILABLE,
            census_available=CENSUS_AVAILABLE,
            bea_available=BEA_AVAILABLE,
        )

    def is_query_specific(self, query: str) -> bool:
        """Determine if a query is specific enough to warrant immediate analysis.

        Specific queries typically:
        - Ask for specific numbers ("what is", "how much", "how many")
        - Mention specific locations (states, cities)
        - Mention specific time periods (years, months)
        - Ask for comparisons or trends
        - Request specific data types (unemployment rate, GDP, etc.)

        Vague queries typically:
        - Are exploratory ("I'm interested in...", "Tell me about...")
        - Ask general questions ("What data do you have?")
        - Don't specify location, time, or metric

        Args:
            query: The user's query string

        Returns:
            True if the query is specific enough for analysis, False otherwise
        """
        query_lower = query.lower().strip()

        # Patterns that indicate a VAGUE/exploratory query
        vague_patterns = [
            "i'm interested in",
            "i am interested in",
            "tell me about",
            "what do you have",
            "what data do you have",
            "what kind of",
            "what types of",
            "show me",
            "can you help",
            "i want to learn",
            "i want to explore",
            "i need help",
            "help me find",
            "looking for data",
            "any data on",
            "data about",
            "information about",
            "information on",
            "where can i find",
            "how can i access",
            "resources for",
            "sources for",
        ]

        # Check for vague patterns
        for pattern in vague_patterns:
            if pattern in query_lower:
                return False

        # Patterns that indicate a SPECIFIC query
        specific_patterns = [
            # Direct questions about specific metrics
            r"what is the .* rate",
            r"what is the .* in \w+",
            r"what was the",
            r"how much",
            r"how many",
            r"what's the",
            r"what are the",
            # Comparisons
            r"compare",
            r"difference between",
            r"higher|lower than",
            r"more|less than",
            # Trends
            r"trend",
            r"over time",
            r"change in",
            r"has .* changed",
            r"historical",
            # Specific time references
            r"\b20\d{2}\b",  # Year like 2020, 2023
            r"last year",
            r"this year",
            r"past \d+ years?",
            r"since \d{4}",
            # Specific locations (states)
            r"\b(texas|california|new york|florida|illinois|pennsylvania|ohio|georgia|michigan|north carolina|washington|arizona|massachusetts|tennessee|indiana|missouri|maryland|wisconsin|colorado|minnesota|south carolina|alabama|louisiana|kentucky|oregon|oklahoma|connecticut|utah|iowa|nevada|arkansas|mississippi|kansas|new mexico|nebraska|west virginia|idaho|hawaii|new hampshire|maine|montana|rhode island|delaware|south dakota|north dakota|alaska|vermont|wyoming|dc|d\.c\.)\b",
            # Specific metrics
            r"unemployment",
            r"inflation",
            r"gdp",
            r"population",
            r"income",
            r"poverty",
            r"housing",
            r"cpi",
            r"employment",
            r"jobs",
        ]

        import re
        specific_count = 0
        for pattern in specific_patterns:
            if re.search(pattern, query_lower):
                specific_count += 1

        # If query has 2+ specific indicators, consider it specific
        if specific_count >= 2:
            return True

        # If query contains a question word + location or metric, consider it specific
        question_words = ["what", "how", "when", "where"]
        has_question = any(w in query_lower.split()[:3] for w in question_words)

        if has_question and specific_count >= 1:
            return True

        # Default to vague for safety (will show guidance)
        return False

    def get_mode_for_context(
        self,
        query: str,
        is_follow_up: bool = False,
        force_mode: str | None = None,
    ) -> str:
        """Determine the operating mode based on context.

        Args:
            query: The user's query
            is_follow_up: Whether this is a follow-up message in a conversation
            force_mode: Force a specific mode ("concierge" or "analysis")

        Returns:
            "concierge" or "analysis"
        """
        # Allow forcing a mode for testing or explicit requests
        if force_mode in ("concierge", "analysis"):
            return force_mode

        # Follow-up questions always go to analysis mode
        if is_follow_up:
            return "analysis"

        # First question: check if it's specific or vague
        if self.is_query_specific(query):
            return "analysis"

        # Default to concierge mode for vague first questions
        return "concierge"

    def _get_bls_client(self) -> BLSClient | None:
        """Get or create BLS client."""
        if not BLS_AVAILABLE:
            return None
        if self._bls_client is None:
            self._bls_client = BLSClient()
        return self._bls_client

    def _get_fred_client(self) -> FREDClient | None:
        """Get or create FRED client."""
        if not FRED_AVAILABLE:
            return None
        if self._fred_client is None:
            self._fred_client = FREDClient()
        return self._fred_client

    def _get_census_client(self) -> CensusClient | None:
        """Get or create Census client."""
        if not CENSUS_AVAILABLE:
            return None
        if self._census_client is None:
            self._census_client = CensusClient()
        return self._census_client

    def _get_bea_client(self) -> BEAClient | None:
        """Get or create BEA client."""
        if not BEA_AVAILABLE:
            return None
        if self._bea_client is None:
            self._bea_client = BEAClient()
        return self._bea_client

    def _get_anthropic_client(self) -> Any:
        """Get or create async Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic library not installed. Run: pip install anthropic")

        if self._anthropic_client is None:
            api_key = settings.anthropic_api_key.get_secret_value()
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not configured")
            self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)

        return self._anthropic_client

    def _define_tools(self) -> list[dict[str, Any]]:
        """Define tools available to the concierge."""
        tools = [
            {
                "name": "search_data_sources",
                "description": "Search for data sources that match given keywords. Returns a list of relevant data sources with descriptions and links.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keywords to search for (e.g., ['unemployment', 'jobs', 'labor'])"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of sources to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["keywords"]
                }
            },
            {
                "name": "get_source_details",
                "description": "Get detailed information about a specific data source, including its datasets, access instructions, and example queries.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source_id": {
                            "type": "string",
                            "description": "The ID of the source (e.g., 'bls', 'census', 'bea')"
                        }
                    },
                    "required": ["source_id"]
                }
            },
            {
                "name": "get_category_sources",
                "description": "Find data sources by category (e.g., employment, health, demographics).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [cat.value for cat in DataCategory],
                            "description": "The data category to search for"
                        }
                    },
                    "required": ["category"]
                }
            },
            {
                "name": "list_all_sources",
                "description": "List all available data sources with brief descriptions.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            # Live data fetching tools
            {
                "name": "fetch_bls_unemployment",
                "description": "Fetch LIVE unemployment rate data from the Bureau of Labor Statistics. Can get national or state-level data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "description": "State name (e.g., 'Texas', 'California'). Leave empty for national data."
                        },
                        "start_year": {
                            "type": "integer",
                            "description": "Start year for data (default: 10 years ago)"
                        },
                        "end_year": {
                            "type": "integer",
                            "description": "End year for data (default: current year)"
                        }
                    }
                }
            },
            {
                "name": "fetch_bls_cpi",
                "description": "Fetch LIVE Consumer Price Index (CPI/inflation) data from BLS.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["all", "food", "energy", "core"],
                            "description": "CPI category (default: 'all')"
                        },
                        "start_year": {
                            "type": "integer",
                            "description": "Start year for data"
                        },
                        "end_year": {
                            "type": "integer",
                            "description": "End year for data"
                        }
                    }
                }
            },
            {
                "name": "fetch_fred_data",
                "description": "Fetch LIVE economic data from FRED (Federal Reserve Economic Data). Can get GDP, interest rates, inflation, and more.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data_type": {
                            "type": "string",
                            "enum": ["gdp", "unemployment", "inflation", "interest_rate"],
                            "description": "Type of data to fetch"
                        },
                        "rate_type": {
                            "type": "string",
                            "enum": ["fed_funds", "prime", "treasury_10yr", "treasury_2yr", "mortgage_30yr"],
                            "description": "For interest rates, specify which rate (default: fed_funds)"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format"
                        }
                    },
                    "required": ["data_type"]
                }
            },
            {
                "name": "fetch_census_data",
                "description": "Fetch LIVE demographic and economic data from the U.S. Census Bureau.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data_type": {
                            "type": "string",
                            "enum": ["population", "income", "poverty", "housing", "education", "employment"],
                            "description": "Type of Census data to fetch"
                        },
                        "state": {
                            "type": "string",
                            "description": "State name (optional, for state-level data)"
                        },
                        "year": {
                            "type": "integer",
                            "description": "Year for data (default: most recent)"
                        }
                    },
                    "required": ["data_type"]
                }
            },
            {
                "name": "fetch_bea_gdp",
                "description": "Fetch LIVE GDP data from the Bureau of Economic Analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["national", "state"],
                            "description": "National or state-level GDP"
                        },
                        "state": {
                            "type": "string",
                            "description": "State name (required if level is 'state')"
                        },
                        "real": {
                            "type": "boolean",
                            "description": "If true, get real (inflation-adjusted) GDP (default: true)"
                        }
                    }
                }
            },
        ]
        return tools

    async def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result.

        All tool execution is async — does not block the event loop.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input arguments

        Returns:
            Result string
        """
        try:
            if tool_name == "search_data_sources":
                return self._tool_search_sources(
                    tool_input.get("keywords", []),
                    tool_input.get("limit", 5),
                )

            elif tool_name == "get_source_details":
                return self._tool_get_source_details(
                    tool_input.get("source_id", ""),
                )

            elif tool_name == "get_category_sources":
                return self._tool_get_category_sources(
                    tool_input.get("category", ""),
                )

            elif tool_name == "list_all_sources":
                return self._tool_list_all_sources()

            # Live data fetching tools (async)
            elif tool_name == "fetch_bls_unemployment":
                return await self._tool_fetch_bls_unemployment(
                    tool_input.get("state"),
                    tool_input.get("start_year"),
                    tool_input.get("end_year"),
                )

            elif tool_name == "fetch_bls_cpi":
                return await self._tool_fetch_bls_cpi(
                    tool_input.get("category", "all"),
                    tool_input.get("start_year"),
                    tool_input.get("end_year"),
                )

            elif tool_name == "fetch_fred_data":
                return await self._tool_fetch_fred_data(
                    tool_input.get("data_type"),
                    tool_input.get("rate_type", "fed_funds"),
                    tool_input.get("start_date"),
                    tool_input.get("end_date"),
                )

            elif tool_name == "fetch_census_data":
                return await self._tool_fetch_census_data(
                    tool_input.get("data_type"),
                    tool_input.get("state"),
                    tool_input.get("year"),
                )

            elif tool_name == "fetch_bea_gdp":
                return await self._tool_fetch_bea_gdp(
                    tool_input.get("level", "national"),
                    tool_input.get("state"),
                    tool_input.get("real", True),
                )

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            self.logger.error(f"Tool execution error: {tool_name}", error=str(e))
            return f"Error: {str(e)}"

    async def _tool_fetch_bls_unemployment(
        self,
        state: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> str:
        """Fetch unemployment data from BLS."""
        client = self._get_bls_client()
        if not client:
            return "BLS client not available. Please check that the BLS connector is properly installed."

        try:
            result = await client.get_unemployment_rate(
                state=state,
                start_year=start_year,
                end_year=end_year,
            )

            if "error" in result:
                return f"Error fetching BLS data: {result['error']}"

            # Format the output
            location = state or "National"
            output = [f"**Unemployment Rate - {location}**\n"]
            output.append("Source: Bureau of Labor Statistics")
            output.append(f"Series ID: {result.get('series_id', 'N/A')}\n")

            if result.get("latest"):
                latest = result["latest"]
                output.append(f"**Latest Data ({latest.get('date', 'N/A')}):**")
                output.append(f"- Unemployment Rate: {latest.get('value', 'N/A')}%")

                if latest.get("pct_change_12_month"):
                    output.append(f"- 12-Month Change: {latest['pct_change_12_month']}%")

            output.append(f"\n**Historical Data ({result.get('count', 0)} observations):**")

            observations = result.get("observations", [])[:12]  # Last 12 periods
            for obs in observations:
                date = obs.get("date", "N/A")
                value = obs.get("value", "N/A")
                output.append(f"- {date}: {value}%")

            return "\n".join(output)

        except Exception as e:
            return f"Error fetching BLS unemployment data: {str(e)}"

    async def _tool_fetch_bls_cpi(
        self,
        category: str = "all",
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> str:
        """Fetch CPI data from BLS."""
        client = self._get_bls_client()
        if not client:
            return "BLS client not available."

        try:
            result = await client.get_cpi(
                category=category,
                start_year=start_year,
                end_year=end_year,
            )

            if "error" in result:
                return f"Error fetching CPI data: {result['error']}"

            output = [f"**Consumer Price Index ({category.upper()})**\n"]
            output.append("Source: Bureau of Labor Statistics")
            output.append(f"Series ID: {result.get('series_id', 'N/A')}\n")

            if result.get("latest"):
                latest = result["latest"]
                output.append(f"**Latest Data ({latest.get('date', 'N/A')}):**")
                output.append(f"- CPI Index: {latest.get('value', 'N/A')}")

                if latest.get("pct_change_12_month"):
                    output.append(f"- 12-Month Inflation: {latest['pct_change_12_month']}%")

            output.append(f"\n**Historical Data ({result.get('count', 0)} observations):**")

            observations = result.get("observations", [])[:12]
            for obs in observations:
                date = obs.get("date", "N/A")
                value = obs.get("value", "N/A")
                change = obs.get("pct_change_12_month", "")
                line = f"- {date}: {value}"
                if change:
                    line += f" ({change}% YoY)"
                output.append(line)

            return "\n".join(output)

        except Exception as e:
            return f"Error fetching CPI data: {str(e)}"

    async def _tool_fetch_fred_data(
        self,
        data_type: str,
        rate_type: str = "fed_funds",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Fetch data from FRED."""
        client = self._get_fred_client()
        if not client:
            return "FRED client not available. Please configure FRED_API_KEY."

        try:
            if data_type == "gdp":
                result = await client.get_gdp(start_date=start_date, end_date=end_date)
            elif data_type == "unemployment":
                result = await client.get_unemployment_rate(start_date=start_date, end_date=end_date)
            elif data_type == "inflation":
                result = await client.get_inflation(start_date=start_date, end_date=end_date)
            elif data_type == "interest_rate":
                result = await client.get_interest_rate(rate_type=rate_type, start_date=start_date, end_date=end_date)
            else:
                return f"Unknown data type: {data_type}"

            if "error" in result:
                return f"Error fetching FRED data: {result['error']}"

            output = [f"**{result.get('data_type', data_type)}**\n"]
            output.append("Source: Federal Reserve Economic Data (FRED)")
            output.append(f"Series ID: {result.get('series_id', 'N/A')}")
            output.append(f"Units: {result.get('units', 'N/A')}\n")

            if result.get("latest"):
                latest = result["latest"]
                output.append(f"**Latest Data ({latest.get('date', 'N/A')}):**")
                output.append(f"- Value: {latest.get('value', 'N/A')}")

            output.append(f"\n**Historical Data ({result.get('count', 0)} observations):**")

            observations = result.get("observations", [])[-12:]  # Last 12 periods
            for obs in observations:
                date = obs.get("date", "N/A")
                value = obs.get("value", "N/A")
                output.append(f"- {date}: {value}")

            return "\n".join(output)

        except Exception as e:
            return f"Error fetching FRED data: {str(e)}"

    async def _tool_fetch_census_data(
        self,
        data_type: str,
        state: str | None = None,
        year: int | None = None,
    ) -> str:
        """Fetch data from Census Bureau."""
        client = self._get_census_client()
        if not client:
            return "Census client not available. Please configure CENSUS_API_KEY."

        try:
            year = year or 2022  # Default to recent year

            if data_type == "population":
                result = await client.get_population(year=year, geography="state", state=state)
            elif data_type == "income":
                result = await client.get_median_income(year=year, geography="state", state=state)
            elif data_type == "poverty":
                result = await client.get_poverty_rate(year=year, geography="state", state=state)
            elif data_type == "housing":
                result = await client.get_housing_data(year=year, geography="state", state=state)
            elif data_type == "education":
                result = await client.get_education_data(year=year, geography="state", state=state)
            elif data_type == "employment":
                result = await client.get_employment_data(year=year, geography="state", state=state)
            else:
                return f"Unknown data type: {data_type}"

            if "error" in result:
                return f"Error fetching Census data: {result['error']}"

            output = [f"**{result.get('data_type', data_type)}**\n"]
            output.append(f"Source: U.S. Census Bureau - {result.get('survey', 'ACS')}")
            output.append(f"Year: {result.get('year', year)}\n")

            data = result.get("data", [])
            if data:
                output.append(f"**Data ({len(data)} records):**\n")
                for record in data[:20]:  # Limit to 20 records
                    name = record.get("NAME", record.get("state_name", "Unknown"))
                    # Show key metrics based on data type
                    metrics = []
                    for key, value in record.items():
                        if key not in ["NAME", "state", "county", "state_name"] and value is not None:
                            metrics.append(f"{key}: {value:,}" if isinstance(value, int | float) else f"{key}: {value}")
                    if metrics:
                        output.append(f"- **{name}**: {', '.join(metrics[:3])}")

            return "\n".join(output)

        except Exception as e:
            return f"Error fetching Census data: {str(e)}"

    async def _tool_fetch_bea_gdp(
        self,
        level: str = "national",
        state: str | None = None,
        real: bool = True,
    ) -> str:
        """Fetch GDP data from BEA."""
        client = self._get_bea_client()
        if not client:
            return "BEA client not available. Please configure BEA_API_KEY."

        try:
            if level == "national":
                result = await client.get_gdp(real=real)
            else:
                result = await client.get_regional_gdp(state=state)

            if "error" in result:
                return f"Error fetching BEA data: {result['error']}"

            output = [f"**{result.get('data_type', 'GDP')}**\n"]
            output.append("Source: Bureau of Economic Analysis")
            if result.get("units"):
                output.append(f"Units: {result['units']}")
            output.append("")

            # For national GDP, show GDP observations
            gdp_obs = result.get("gdp_observations", result.get("observations", []))
            if gdp_obs:
                output.append(f"**GDP Data ({len(gdp_obs)} periods):**\n")
                for obs in gdp_obs[-12:]:  # Last 12 periods
                    period = obs.get("time_period", obs.get("TimePeriod", "N/A"))
                    value = obs.get("value")
                    if value is not None:
                        output.append(f"- {period}: ${value:,.0f}B")
                    else:
                        output.append(f"- {period}: {obs.get('value_raw', 'N/A')}")

            return "\n".join(output)

        except Exception as e:
            return f"Error fetching BEA data: {str(e)}"

    def _tool_search_sources(self, keywords: list[str], limit: int = 5) -> str:
        """Search for data sources by keywords."""
        if not keywords:
            return "No keywords provided for search."

        results = self.registry.search_by_keywords(keywords)[:limit]

        if not results:
            return f"No data sources found matching keywords: {', '.join(keywords)}"

        output_parts = [f"Found {len(results)} relevant data sources:\n"]

        for source, score in results:
            output_parts.append(f"\n**{source.name}** (ID: {source.id})")
            output_parts.append(f"Relevance: {score:.1%}")
            output_parts.append(f"Description: {source.description[:200]}...")
            output_parts.append(f"Website: {source.base_url}")
            if source.portal_url:
                output_parts.append(f"Data Portal: {source.portal_url}")

            # Show matching datasets
            matching_datasets = []
            for ds in source.datasets:
                ds_keywords = [k.lower() for k in ds.keywords]
                if any(kw.lower() in ds_keywords or kw.lower() in ds.name.lower()
                       for kw in keywords):
                    matching_datasets.append(ds)

            if matching_datasets:
                output_parts.append("Relevant datasets:")
                for ds in matching_datasets[:3]:
                    output_parts.append(f"  - {ds.name}")
                    if ds.url:
                        output_parts.append(f"    Link: {ds.url}")

            output_parts.append("")

        return "\n".join(output_parts)

    def _tool_get_source_details(self, source_id: str) -> str:
        """Get detailed information about a data source."""
        source = self.registry.get_source(source_id)
        if not source:
            return f"Data source '{source_id}' not found. Use list_all_sources to see available sources."

        lines = [
            f"**{source.name}**",
            "",
            f"**Description:** {source.description}",
            "",
            "**URLs:**",
            f"- Website: {source.base_url}",
        ]

        if source.portal_url:
            lines.append(f"- Data Portal: {source.portal_url}")
        if source.api_url:
            lines.append(f"- API: {source.api_url}")

        lines.append("")
        lines.append(f"**Data Quality Score:** {source.quality_score:.0%}")
        lines.append(f"**Update Frequency:** {source.update_frequency}")

        if source.requires_api_key:
            lines.append("**API Key Required:** Yes")

        if source.access_instructions:
            lines.append("")
            lines.append("**How to Access:**")
            lines.append(source.access_instructions)

        if source.datasets:
            lines.append("")
            lines.append(f"**Available Datasets ({len(source.datasets)}):**")
            for ds in source.datasets:
                lines.append("")
                lines.append(f"*{ds.name}*")
                lines.append(f"  {ds.description}")
                if ds.url:
                    lines.append(f"  Link: {ds.url}")
                if ds.update_frequency:
                    lines.append(f"  Updated: {ds.update_frequency}")

        if source.example_queries:
            lines.append("")
            lines.append("**Example Questions This Source Can Answer:**")
            for eq in source.example_queries:
                lines.append(f"- {eq}")

        return "\n".join(lines)

    def _tool_get_category_sources(self, category: str) -> str:
        """Get sources for a specific data category."""
        try:
            cat_enum = DataCategory(category)
        except ValueError:
            categories = ", ".join([c.value for c in DataCategory])
            return f"Invalid category '{category}'. Valid categories are: {categories}"

        sources = self.registry.search_by_category(cat_enum)

        if not sources:
            return f"No data sources found for category: {category}"

        output_parts = [f"Data sources for '{category}':\n"]

        for source in sources:
            output_parts.append(f"**{source.name}** (ID: {source.id})")
            output_parts.append(f"  {source.description[:150]}...")
            output_parts.append(f"  Portal: {source.portal_url or source.base_url}")

            # Show relevant datasets for this category
            relevant_datasets = [
                ds for ds in source.datasets
                if cat_enum in ds.categories
            ]
            if relevant_datasets:
                output_parts.append("  Key datasets:")
                for ds in relevant_datasets[:2]:
                    output_parts.append(f"    - {ds.name}")
            output_parts.append("")

        return "\n".join(output_parts)

    def _tool_list_all_sources(self) -> str:
        """List all available data sources."""
        sources = self.registry.get_enabled_sources()

        output_parts = [f"Available Data Sources ({len(sources)}):\n"]

        for source in sources:
            output_parts.append(f"**{source.name}** (ID: {source.id})")
            output_parts.append(f"  {source.description[:100]}...")
            output_parts.append(f"  Categories: {', '.join(c.value for c in source.categories[:4])}")
            output_parts.append("")

        return "\n".join(output_parts)

    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        is_follow_up: bool = False,
        force_mode: str | None = None,
        conversation_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Process a user query through the concierge.

        This is the main entry point for the concierge. It:
        1. Determines the appropriate mode (concierge vs analysis)
        2. Uses the LLM with the appropriate system prompt
        3. Either guides to sources (concierge) or fetches data (analysis)

        Args:
            query: User's natural language query
            session_id: Optional session ID for context
            is_follow_up: Whether this is a follow-up in a conversation
            force_mode: Force a specific mode ("concierge" or "analysis")
            conversation_history: Previous messages for context

        Returns:
            Dict containing the concierge response
        """
        self.logger.info(f"Processing query: {query[:100]}...")
        start_time = time.time()

        # Determine mode based on context
        mode = self.get_mode_for_context(query, is_follow_up, force_mode)
        self.logger.info(f"Operating in {mode} mode (is_follow_up={is_follow_up})")

        try:
            client = self._get_anthropic_client()

            # Select system prompt based on mode
            if mode == "concierge":
                system_prompt = self.CONCIERGE_SYSTEM_PROMPT
            else:
                system_prompt = self.ANALYSIS_SYSTEM_PROMPT

            # Build messages with conversation history if provided
            messages = []
            if conversation_history:
                # Add relevant conversation history for context
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # Add current query
            messages.append({"role": "user", "content": query})

            # Run the agent loop
            max_iterations = 10
            for iteration in range(max_iterations):
                self.logger.debug(f"Concierge iteration {iteration + 1} (mode={mode})")

                # Call Claude (async — does not block the event loop)
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=self.tools,
                    messages=messages,
                )

                # Process response
                assistant_content = []
                tool_calls_to_process = []

                for content_block in response.content:
                    if content_block.type == "text":
                        assistant_content.append({
                            "type": "text",
                            "text": content_block.text,
                        })

                    elif content_block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input,
                        })
                        tool_calls_to_process.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input,
                        })

                # Add assistant message
                messages.append({"role": "assistant", "content": assistant_content})

                # Process tool calls
                if tool_calls_to_process:
                    tool_results = []
                    for tool_call in tool_calls_to_process:
                        result = await self._execute_tool(
                            tool_call["name"],
                            tool_call["input"],
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call["id"],
                            "content": result,
                        })
                    messages.append({"role": "user", "content": tool_results})

                # Check if we're done
                if response.stop_reason == "end_turn" and not tool_calls_to_process:
                    break

            # Extract final answer
            final_answer = ""
            for content_block in response.content:
                if content_block.type == "text":
                    final_answer = content_block.text
                    break

            elapsed = time.time() - start_time

            self.logger.info(
                "Query processed",
                elapsed=elapsed,
                iterations=iteration + 1,
            )

            return {
                "answer": final_answer,
                "query": query,
                "processing_time_ms": elapsed * 1000,
                "success": True,
                "mode": mode,
                "offer_analysis": mode == "concierge" or self._should_offer_analysis(query, final_answer),
            }

        except Exception as e:
            self.logger.error(f"Concierge error: {e}", exc_info=True)
            elapsed = time.time() - start_time

            return {
                "answer": f"I apologize, but I encountered an error while searching for data sources: {str(e)}",
                "query": query,
                "processing_time_ms": elapsed * 1000,
                "success": False,
                "mode": mode,
                "error": str(e),
            }

    def _should_offer_analysis(self, query: str, answer: str) -> bool:
        """Determine if we should offer deeper analysis.

        Returns True if the query seems like something that could benefit
        from actual data analysis (vs just needing links).
        """
        # Keywords that suggest user wants actual data, not just links
        analysis_keywords = [
            "what is", "how much", "how many", "compare", "trend",
            "over time", "change", "increase", "decrease", "rate",
            "statistics", "numbers", "data for", "latest"
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in analysis_keywords)

    def get_quick_recommendations(self, query: str) -> list[DataSourceRecommendation]:
        """Get quick recommendations without using LLM.

        This provides a fast fallback when LLM is not needed or available.

        Args:
            query: User's query

        Returns:
            List of source recommendations
        """
        # Extract keywords
        query_lower = query.lower()
        stop_words = {"what", "is", "the", "in", "of", "for", "how", "has", "have", "does", "do", "a", "an", "and", "or", "to", "from", "i", "want", "need", "find", "get", "show", "me"}
        words = query_lower.replace("?", "").replace(",", "").replace(".", "").split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        if not keywords:
            return []

        # Search registry
        results = self.registry.search_by_keywords(keywords)

        recommendations = []
        for source, score in results[:5]:
            # Find relevant datasets
            relevant_datasets = []
            for ds in source.datasets:
                ds_keywords = [k.lower() for k in ds.keywords]
                if any(kw in ds_keywords or kw in ds.name.lower() for kw in keywords):
                    relevant_datasets.append({
                        "name": ds.name,
                        "description": ds.description,
                        "url": ds.url,
                    })

            recommendations.append(DataSourceRecommendation(
                source_id=source.id,
                source_name=source.name,
                description=source.description,
                relevance_score=score,
                portal_url=source.portal_url,
                api_url=source.api_url,
                recommended_datasets=relevant_datasets[:3],
                access_instructions=source.access_instructions,
                example_queries=source.example_queries,
            ))

        return recommendations


# Singleton instance
_concierge_agent: DataConciergeAgent | None = None


def get_data_concierge() -> DataConciergeAgent:
    """Get or create the Data Concierge agent singleton."""
    global _concierge_agent
    if _concierge_agent is None:
        _concierge_agent = DataConciergeAgent()
    return _concierge_agent
