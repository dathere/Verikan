"""Data Source Registry for managing multiple data sources.

This module provides a centralized registry for all available data sources,
their metadata, and capabilities. The Data Concierge uses this registry to
recommend appropriate data sources based on user queries.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)


class DataCategory(str, Enum):
    """Categories of data that sources can provide."""
    EMPLOYMENT = "employment"
    UNEMPLOYMENT = "unemployment"
    WAGES = "wages"
    PRICES = "prices"
    DEMOGRAPHICS = "demographics"
    POPULATION = "population"
    GDP = "gdp"
    INCOME = "income"
    TRADE = "trade"
    HEALTH = "health"
    EDUCATION = "education"
    SCIENCE_ENGINEERING = "science_engineering"
    ENVIRONMENT = "environment"
    HOUSING = "housing"
    CRIME = "crime"
    TRANSPORTATION = "transportation"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"
    GENERAL = "general"


@dataclass
class DatasetInfo:
    """Information about a specific dataset within a source."""
    name: str
    description: str
    url: str | None = None
    categories: list[DataCategory] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    update_frequency: str | None = None
    geographic_coverage: str = "United States"
    temporal_coverage: str | None = None


@dataclass
class DataSourceInfo:
    """Complete information about a data source."""
    id: str
    name: str
    description: str
    base_url: str
    api_url: str | None = None
    portal_url: str | None = None
    categories: list[DataCategory] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    datasets: list[DatasetInfo] = field(default_factory=list)
    quality_score: float = 0.90
    update_frequency: str = "varies"
    requires_api_key: bool = False
    is_enabled: bool = True
    source_type: str = "api"  # api, knowledge_graph, portal

    # How to access this data
    access_instructions: str | None = None
    example_queries: list[str] = field(default_factory=list)


class DataSourceRegistry:
    """Registry for managing all available data sources.

    This class maintains a comprehensive catalog of data sources and provides
    methods for searching and matching sources to user queries.
    """

    def __init__(self) -> None:
        """Initialize the registry with default sources."""
        self.sources: dict[str, DataSourceInfo] = {}
        self._load_default_sources()
        logger.info(f"DataSourceRegistry initialized with {len(self.sources)} sources")

    def _load_default_sources(self) -> None:
        """Load default data sources configuration."""

        # Bureau of Labor Statistics (BLS)
        self.register_source(DataSourceInfo(
            id="bls",
            name="Bureau of Labor Statistics (BLS)",
            description="Primary source for labor market data including employment, unemployment, wages, productivity, and consumer prices.",
            base_url="https://www.bls.gov",
            api_url="https://api.bls.gov/publicAPI/v2",
            portal_url="https://data.bls.gov",
            categories=[
                DataCategory.EMPLOYMENT,
                DataCategory.UNEMPLOYMENT,
                DataCategory.WAGES,
                DataCategory.PRICES,
            ],
            keywords=[
                "employment", "unemployment", "jobs", "labor", "wages", "salary",
                "cpi", "consumer price index", "inflation", "ppi", "producer price",
                "payroll", "workforce", "occupation", "industry", "hours worked",
                "productivity", "compensation", "benefits", "layoffs", "hires"
            ],
            datasets=[
                DatasetInfo(
                    name="Current Employment Statistics (CES)",
                    description="Monthly data on employment, hours, and earnings by industry.",
                    url="https://www.bls.gov/ces/",
                    categories=[DataCategory.EMPLOYMENT, DataCategory.WAGES],
                    keywords=["nonfarm payroll", "employment", "industry", "hours", "earnings"],
                    update_frequency="monthly"
                ),
                DatasetInfo(
                    name="Local Area Unemployment Statistics (LAUS)",
                    description="Monthly unemployment data for states, counties, and metropolitan areas.",
                    url="https://www.bls.gov/lau/",
                    categories=[DataCategory.UNEMPLOYMENT],
                    keywords=["unemployment rate", "labor force", "state", "county", "metro"],
                    update_frequency="monthly"
                ),
                DatasetInfo(
                    name="Consumer Price Index (CPI)",
                    description="Monthly measure of price changes for consumer goods and services.",
                    url="https://www.bls.gov/cpi/",
                    categories=[DataCategory.PRICES],
                    keywords=["inflation", "prices", "cost of living", "consumer"],
                    update_frequency="monthly"
                ),
                DatasetInfo(
                    name="Occupational Employment and Wage Statistics (OEWS)",
                    description="Employment and wage estimates by occupation and industry.",
                    url="https://www.bls.gov/oes/",
                    categories=[DataCategory.EMPLOYMENT, DataCategory.WAGES],
                    keywords=["occupation", "wages", "salary", "employment by job"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="Job Openings and Labor Turnover Survey (JOLTS)",
                    description="Data on job openings, hires, and separations.",
                    url="https://www.bls.gov/jlt/",
                    categories=[DataCategory.EMPLOYMENT],
                    keywords=["job openings", "vacancies", "hires", "quits", "layoffs"],
                    update_frequency="monthly"
                ),
            ],
            quality_score=0.95,
            update_frequency="monthly",
            requires_api_key=True,
            source_type="api",
            access_instructions="Register for a free API key at https://www.bls.gov/developers/",
            example_queries=[
                "What is the unemployment rate in California?",
                "How have wages changed over the past 5 years?",
                "What is the current CPI inflation rate?",
            ]
        ))

        # U.S. Census Bureau
        self.register_source(DataSourceInfo(
            id="census",
            name="U.S. Census Bureau",
            description="Comprehensive demographic, economic, and geographic data about the United States population and economy.",
            base_url="https://www.census.gov",
            api_url="https://api.census.gov",
            portal_url="https://data.census.gov",
            categories=[
                DataCategory.DEMOGRAPHICS,
                DataCategory.POPULATION,
                DataCategory.INCOME,
                DataCategory.HOUSING,
                DataCategory.EDUCATION,
            ],
            keywords=[
                "population", "demographics", "census", "income", "poverty",
                "housing", "education", "race", "ethnicity", "age", "gender",
                "household", "family", "migration", "birth", "death", "commute"
            ],
            datasets=[
                DatasetInfo(
                    name="American Community Survey (ACS)",
                    description="Annual survey providing detailed demographic, social, economic, and housing data.",
                    url="https://www.census.gov/programs-surveys/acs",
                    categories=[DataCategory.DEMOGRAPHICS, DataCategory.INCOME, DataCategory.HOUSING],
                    keywords=["demographics", "income", "poverty", "education", "housing"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="Decennial Census",
                    description="Complete count of the U.S. population every 10 years.",
                    url="https://www.census.gov/programs-surveys/decennial-census.html",
                    categories=[DataCategory.POPULATION, DataCategory.DEMOGRAPHICS],
                    keywords=["population", "count", "redistricting"],
                    update_frequency="every 10 years"
                ),
                DatasetInfo(
                    name="Population Estimates Program",
                    description="Annual estimates of population between census years.",
                    url="https://www.census.gov/programs-surveys/popest.html",
                    categories=[DataCategory.POPULATION],
                    keywords=["population estimate", "growth", "change"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="County Business Patterns",
                    description="Annual data on businesses by industry and geography.",
                    url="https://www.census.gov/programs-surveys/cbp.html",
                    categories=[DataCategory.EMPLOYMENT],
                    keywords=["businesses", "establishments", "employment by county"],
                    update_frequency="annual"
                ),
            ],
            quality_score=0.95,
            update_frequency="varies",
            requires_api_key=True,
            source_type="api",
            access_instructions="Get a free API key at https://api.census.gov/data/key_signup.html",
            example_queries=[
                "What is the population of Texas?",
                "What is the median household income by state?",
                "How has the population changed in urban vs rural areas?",
            ]
        ))

        # Bureau of Economic Analysis (BEA)
        self.register_source(DataSourceInfo(
            id="bea",
            name="Bureau of Economic Analysis (BEA)",
            description="Economic statistics including GDP, personal income, corporate profits, and international trade data.",
            base_url="https://www.bea.gov",
            api_url="https://apps.bea.gov/api",
            portal_url="https://apps.bea.gov/iTable/",
            categories=[
                DataCategory.GDP,
                DataCategory.INCOME,
                DataCategory.TRADE,
            ],
            keywords=[
                "gdp", "gross domestic product", "economic growth", "recession",
                "personal income", "disposable income", "savings", "corporate profits",
                "trade deficit", "exports", "imports", "investment", "consumption"
            ],
            datasets=[
                DatasetInfo(
                    name="National Income and Product Accounts (NIPA)",
                    description="Comprehensive measures of U.S. production, including GDP.",
                    url="https://www.bea.gov/data/gdp",
                    categories=[DataCategory.GDP, DataCategory.INCOME],
                    keywords=["gdp", "national income", "economic output"],
                    update_frequency="quarterly"
                ),
                DatasetInfo(
                    name="Regional Economic Accounts",
                    description="State and local area economic data including GDP by state.",
                    url="https://www.bea.gov/data/economic-accounts/regional",
                    categories=[DataCategory.GDP, DataCategory.INCOME],
                    keywords=["state gdp", "regional income", "per capita income"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="International Trade in Goods and Services",
                    description="Monthly data on U.S. exports and imports.",
                    url="https://www.bea.gov/data/intl-trade-investment/international-trade-goods-and-services",
                    categories=[DataCategory.TRADE],
                    keywords=["trade", "exports", "imports", "balance"],
                    update_frequency="monthly"
                ),
            ],
            quality_score=0.95,
            update_frequency="quarterly",
            requires_api_key=True,
            source_type="api",
            access_instructions="Register at https://apps.bea.gov/API/signup/",
            example_queries=[
                "What is the current GDP growth rate?",
                "How does GDP vary by state?",
                "What is the U.S. trade deficit?",
            ]
        ))

        # NCSES ()
        self.register_source(DataSourceInfo(
            id="ncses",
            name="National Center for Science and Engineering Statistics (NCSES)",
            description="Statistics on the science and engineering enterprise, R&D spending, and STEM workforce.",
            base_url="https://ncses..gov",
            api_url="https://ncses..gov/api",
            portal_url="https://ncses..gov/explore-data",
            categories=[
                DataCategory.SCIENCE_ENGINEERING,
                DataCategory.EDUCATION,
            ],
            keywords=[
                "research", "development", "r&d", "science", "engineering", "stem",
                "phd", "doctorate", "graduate", "postdoc", "university", "federal funding",
                "innovation", "patents", "technology"
            ],
            datasets=[
                DatasetInfo(
                    name="Survey of Federal Funds for R&D",
                    description="Federal government obligations and outlays for R&D.",
                    url="https://ncses..gov/surveys/federal-funds-research-development/",
                    categories=[DataCategory.SCIENCE_ENGINEERING],
                    keywords=["federal r&d", "research funding", "agency spending"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="Survey of Graduate Students and Postdoctorates",
                    description="Data on graduate enrollment and postdoctoral appointees in science and engineering.",
                    url="https://ncses..gov/surveys/graduate-students-postdoctorates-science-engineering/",
                    categories=[DataCategory.EDUCATION, DataCategory.SCIENCE_ENGINEERING],
                    keywords=["graduate students", "phd", "postdoc", "stem education"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="Business R&D Survey",
                    description="R&D performed and funded by U.S. businesses.",
                    url="https://ncses..gov/surveys/business-research-development/",
                    categories=[DataCategory.SCIENCE_ENGINEERING],
                    keywords=["business r&d", "corporate research", "private sector"],
                    update_frequency="annual"
                ),
            ],
            quality_score=0.95,
            update_frequency="annual",
            source_type="api",
            example_queries=[
                "How much does the federal government spend on R&D?",
                "How many STEM PhDs are awarded each year?",
                "What industries spend the most on research?",
            ]
        ))

        # CDC
        self.register_source(DataSourceInfo(
            id="cdc",
            name="Centers for Disease Control and Prevention (CDC)",
            description="Public health data including mortality, disease surveillance, and health indicators.",
            base_url="https://www.cdc.gov",
            api_url="https://data.cdc.gov/api",
            portal_url="https://data.cdc.gov",
            categories=[
                DataCategory.HEALTH,
            ],
            keywords=[
                "health", "disease", "mortality", "death", "birth", "vaccination",
                "epidemic", "pandemic", "covid", "flu", "chronic disease", "obesity",
                "smoking", "alcohol", "drug", "mental health", "life expectancy"
            ],
            datasets=[
                DatasetInfo(
                    name="National Vital Statistics System",
                    description="Birth and death statistics for the U.S.",
                    url="https://www.cdc.gov/nchs/nvss/index.htm",
                    categories=[DataCategory.HEALTH],
                    keywords=["births", "deaths", "mortality", "life expectancy"],
                    update_frequency="annual"
                ),
                DatasetInfo(
                    name="COVID-19 Data Tracker",
                    description="COVID-19 cases, deaths, and vaccinations.",
                    url="https://covid.cdc.gov/covid-data-tracker/",
                    categories=[DataCategory.HEALTH],
                    keywords=["covid", "coronavirus", "vaccination", "cases"],
                    update_frequency="weekly"
                ),
            ],
            quality_score=0.92,
            update_frequency="varies",
            source_type="portal",
            example_queries=[
                "What is the life expectancy in the United States?",
                "How do mortality rates vary by state?",
                "What are the leading causes of death?",
            ]
        ))

        # Data Commons
        self.register_source(DataSourceInfo(
            id="data_commons",
            name="Google Data Commons",
            description="Unified knowledge graph aggregating data from multiple federal agencies, international organizations, and academic institutions.",
            base_url="https://datacommons.org",
            api_url="https://api.datacommons.org",
            portal_url="https://datacommons.org/tools/timeline",
            categories=[
                DataCategory.DEMOGRAPHICS,
                DataCategory.EMPLOYMENT,
                DataCategory.HEALTH,
                DataCategory.EDUCATION,
                DataCategory.ENVIRONMENT,
                DataCategory.GENERAL,
            ],
            keywords=[
                "unified data", "knowledge graph", "multiple sources", "aggregated",
                "population", "demographics", "economics", "health", "environment"
            ],
            datasets=[],  # Data Commons aggregates from multiple sources
            quality_score=0.90,
            update_frequency="continuous",
            source_type="knowledge_graph",
            access_instructions="No API key required for basic access.",
            example_queries=[
                "Compare unemployment rates across states",
                "Show population trends over time",
                "What is the median income in different counties?",
            ]
        ))

        # CKAN (general open data portal)
        self.register_source(DataSourceInfo(
            id="ckan",
            name="CKAN Open Data Portal",
            description="Open data portal with semantic search capabilities. Contains diverse datasets from various sources.",
            base_url="https://data.dathere.com",
            api_url="https://data.dathere.com/api/3",
            portal_url="https://data.dathere.com",
            categories=[
                DataCategory.GENERAL,
            ],
            keywords=[
                "open data", "datasets", "csv", "json", "government data"
            ],
            datasets=[],  # Dynamic - discovered via semantic search
            quality_score=0.85,
            update_frequency="varies",
            source_type="portal",
            access_instructions="Data is publicly accessible. Use semantic search to find relevant datasets.",
            example_queries=[
                "Find datasets about climate change",
                "Search for transportation data",
                "What datasets are available about housing?",
            ]
        ))

        # WPRDC (Western PA Regional Data Center) - City of Pittsburgh
        self.register_source(DataSourceInfo(
            id="wprdc",
            name="Western PA Regional Data Center (WPRDC)",
            description="Open data portal for Pittsburgh and Western Pennsylvania. "
            "Contains 126+ datasets from the City of Pittsburgh covering public safety, "
            "property, permits, city services, community, transportation, and finance.",
            base_url="https://data.wprdc.org",
            api_url="https://data.wprdc.org/api/3",
            portal_url="https://data.wprdc.org",
            categories=[
                DataCategory.CRIME,
                DataCategory.HOUSING,
                DataCategory.TRANSPORTATION,
                DataCategory.GENERAL,
            ],
            keywords=[
                "pittsburgh", "allegheny county", "western pennsylvania", "wprdc",
                "311", "service requests", "police", "crime", "incidents",
                "property", "assessment", "building permits", "violations",
                "fire", "community center", "traffic", "potholes", "budget",
                "condemned", "tax delinquency", "city facilities",
                "open data", "city of pittsburgh",
            ],
            datasets=[
                DatasetInfo(
                    name="311 Data",
                    description="City of Pittsburgh 311 service requests including "
                    "complaints, requests, and inquiries with type, neighborhood, "
                    "and status information.",
                    url="https://data.wprdc.org/dataset/311-data",
                    categories=[DataCategory.GENERAL],
                    keywords=["311", "service requests", "complaints", "city services"],
                    update_frequency="daily",
                    geographic_coverage="City of Pittsburgh",
                ),
                DatasetInfo(
                    name="Police Incident Blotter (Monthly NIBRS)",
                    description="Monthly criminal activity data coded using "
                    "NIBRS standards with geocoded locations.",
                    url="https://data.wprdc.org/dataset/uniform-crime-reporting-data",
                    categories=[DataCategory.CRIME],
                    keywords=["crime", "police", "incidents", "arrests", "NIBRS"],
                    update_frequency="monthly",
                    geographic_coverage="City of Pittsburgh",
                ),
                DatasetInfo(
                    name="PLI Permits",
                    description="Building and construction permits issued by "
                    "the City of Pittsburgh since 2019.",
                    url="https://data.wprdc.org/dataset/pli-permits",
                    categories=[DataCategory.HOUSING],
                    keywords=["permits", "building", "construction", "development"],
                    update_frequency="daily",
                    geographic_coverage="City of Pittsburgh",
                ),
                DatasetInfo(
                    name="Property Assessments",
                    description="Allegheny County property assessment data "
                    "including property values and characteristics.",
                    url="https://data.wprdc.org/dataset/property-assessments",
                    categories=[DataCategory.HOUSING],
                    keywords=["property", "assessment", "real estate", "value"],
                    update_frequency="periodic",
                    geographic_coverage="Allegheny County",
                ),
                DatasetInfo(
                    name="Community Center Attendance",
                    description="Daily attendance figures at Pittsburgh "
                    "community centers since March 2011.",
                    url="https://data.wprdc.org/dataset/community-center-attendance",
                    categories=[DataCategory.GENERAL],
                    keywords=["community", "recreation", "attendance", "parks"],
                    update_frequency="daily",
                    geographic_coverage="City of Pittsburgh",
                ),
            ],
            quality_score=0.88,
            update_frequency="varies",
            requires_api_key=False,
            source_type="portal",
            access_instructions=(
                "Data is publicly accessible via the CKAN API at "
                "https://data.wprdc.org/api/3. No API key required."
            ),
            example_queries=[
                "What are the most common types of 311 requests in Pittsburgh?",
                "Which Pittsburgh neighborhoods have the most police incidents?",
                "How many building permits were issued in Pittsburgh last year?",
                "Show me community center attendance trends in Pittsburgh",
                "What is the distribution of property values across Pittsburgh neighborhoods?",
            ],
        ))

        # FRED (Federal Reserve Economic Data)
        self.register_source(DataSourceInfo(
            id="fred",
            name="Federal Reserve Economic Data (FRED)",
            description="Economic data from the Federal Reserve Bank of St. Louis, including interest rates, monetary aggregates, and economic indicators.",
            base_url="https://fred.stlouisfed.org",
            api_url="https://api.stlouisfed.org/fred",
            portal_url="https://fred.stlouisfed.org",
            categories=[
                DataCategory.GDP,
                DataCategory.INCOME,
                DataCategory.PRICES,
                DataCategory.EMPLOYMENT,
            ],
            keywords=[
                "federal reserve", "interest rates", "monetary policy", "money supply",
                "inflation", "gdp", "unemployment", "economic indicators", "recession"
            ],
            datasets=[
                DatasetInfo(
                    name="Interest Rates",
                    description="Federal funds rate, Treasury yields, and other interest rates.",
                    url="https://fred.stlouisfed.org/categories/22",
                    categories=[DataCategory.GDP],
                    keywords=["interest rate", "federal funds", "treasury", "yield"],
                    update_frequency="daily"
                ),
                DatasetInfo(
                    name="Economic Indicators",
                    description="Key economic indicators including GDP, unemployment, and inflation.",
                    url="https://fred.stlouisfed.org/categories/32992",
                    categories=[DataCategory.GDP, DataCategory.UNEMPLOYMENT],
                    keywords=["economic indicators", "leading indicators"],
                    update_frequency="varies"
                ),
            ],
            quality_score=0.95,
            update_frequency="varies",
            requires_api_key=True,
            source_type="api",
            access_instructions="Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html",
            example_queries=[
                "What is the current federal funds rate?",
                "How have interest rates changed over time?",
                "What are the leading economic indicators?",
            ]
        ))

        # U.S. Census Bureau Data API MCP Server (self-hosted, SQLite-backed)
        self.register_source(DataSourceInfo(
            id="census-data-api",
            name="U.S. Census Bureau Data API (via MCP)",
            description=(
                "Official U.S. Census Bureau Data API accessed through an MCP "
                "server. Provides dataset discovery, geography/FIPS resolution, "
                "data-table search, and aggregate data retrieval for population, "
                "demographics, income, housing, and more, backed by a local "
                "SQLite metadata index."
            ),
            base_url="https://api.census.gov",
            api_url="https://api.census.gov/data",
            portal_url="https://data.census.gov",
            categories=[
                DataCategory.DEMOGRAPHICS,
                DataCategory.POPULATION,
                DataCategory.INCOME,
                DataCategory.HOUSING,
                DataCategory.EDUCATION,
            ],
            keywords=[
                "census", "population", "demographics", "acs",
                "american community survey", "fips", "geography", "state",
                "county", "city", "tract", "income", "poverty", "housing",
                "education", "race", "ethnicity", "age", "census api",
            ],
            quality_score=0.95,
            update_frequency="varies",
            requires_api_key=False,
            source_type="mcp",
            access_instructions=(
                "This data source uses an MCP (Model Context Protocol) server. "
                "It is connected by default from the MCP Servers settings page."
            ),
            example_queries=[
                "What is the population of California?",
                "Show median household income for Texas counties",
                "What datasets are available for housing?",
            ],
        ))

        # FBI Crime Data Explorer MCP Server (self-hosted)
        self.register_source(DataSourceInfo(
            id="fbi-crime-data",
            name="FBI Crime Data Explorer (via MCP)",
            description=(
                "Access FBI Uniform Crime Reporting (UCR) and NIBRS data through "
                "a Model Context Protocol (MCP) server. Provides crime trends, "
                "incident-based data, arrests, hate crimes, homicide and property "
                "crime details, police employment, LEOKA, LESDC, and use of force."
            ),
            base_url="https://cde.ucr.cjis.gov",
            api_url="https://api.usa.gov/crime/fbi/cde",
            portal_url="https://cde.ucr.cjis.gov",
            categories=[DataCategory.CRIME],
            keywords=[
                "fbi", "crime", "ucr", "nibrs", "arrests", "hate crime",
                "homicide", "property crime", "violent crime", "police",
                "law enforcement", "leoka", "use of force", "offense",
                "robbery", "burglary", "assault", "larceny", "motor vehicle theft",
            ],
            quality_score=0.9,
            update_frequency="annual",
            requires_api_key=False,
            source_type="mcp",
            access_instructions=(
                "This data source uses an MCP (Model Context Protocol) server. "
                "It is connected by default from the MCP Servers settings page."
            ),
            example_queries=[
                "What was the violent crime rate in Texas from 2018 to 2022?",
                "How many hate crime incidents were reported in California in 2021?",
                "How many law enforcement officers were employed in New York in 2019?",
            ],
        ))

    def register_source(self, source: DataSourceInfo) -> None:
        """Register a new data source."""
        self.sources[source.id] = source
        logger.debug(f"Registered data source: {source.id}")

    def sync_admin_ckan_sites(self) -> None:
        """Pull admin-managed CKAN portals into the in-memory registry.

        Admins register CKAN sites via the ``/admin/ckan-sites`` endpoint; this
        method refreshes the registry so new portals immediately participate in
        concierge recommendations, keyword search, and source listings without
        a restart. Called on every keyword/category search.
        """
        try:
            # Local import keeps data_concierge.core free of gateway imports.
            from data_concierge.gateway import ckan_sites
        except Exception:  # pragma: no cover - shouldn't happen in normal runtime
            return

        try:
            entries = ckan_sites.list_sites()
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load CKAN sites registry", error=str(exc))
            return

        for entry in entries:
            site_id = str(entry.get("id") or "").strip()
            url = str(entry.get("url") or "").strip()
            if not site_id or not url:
                continue
            info = DataSourceInfo(
                id=site_id,
                name=entry.get("name") or site_id,
                description=entry.get("description")
                    or "CKAN open data portal (admin-registered).",
                base_url=url,
                api_url=f"{url.rstrip('/')}/api/3",
                portal_url=url,
                categories=[DataCategory.GENERAL],
                keywords=list(entry.get("keywords") or []),
                datasets=[],
                quality_score=float(entry.get("quality_score", 0.85)),
                update_frequency="varies",
                requires_api_key=False,
                source_type="portal",
                access_instructions=(
                    f"Publicly accessible CKAN API at {url.rstrip('/')}/api/3. "
                    "Agent will search this portal automatically when relevant."
                ),
                example_queries=[],
            )
            # Overwrite any existing entry with the same ID so the admin-edited
            # values win.  This is safe because the default WPRDC / ckan
            # hardcoded entries will simply be refreshed from the registry.
            self.sources[site_id] = info

    def get_source(self, source_id: str) -> DataSourceInfo | None:
        """Get a data source by ID."""
        self.sync_admin_ckan_sites()
        return self.sources.get(source_id)

    def get_all_sources(self) -> list[DataSourceInfo]:
        """Get all registered data sources."""
        self.sync_admin_ckan_sites()
        return list(self.sources.values())

    def get_enabled_sources(self) -> list[DataSourceInfo]:
        """Get all enabled data sources."""
        self.sync_admin_ckan_sites()
        return [s for s in self.sources.values() if s.is_enabled]

    def search_by_category(self, category: DataCategory) -> list[DataSourceInfo]:
        """Find sources that cover a specific data category."""
        self.sync_admin_ckan_sites()
        return [
            source for source in self.sources.values()
            if source.is_enabled and category in source.categories
        ]

    def search_by_keywords(self, keywords: list[str]) -> list[tuple[DataSourceInfo, float]]:
        """Search sources by keywords and return with relevance scores.

        Args:
            keywords: List of keywords to search for

        Returns:
            List of (source, score) tuples, sorted by score descending
        """
        self.sync_admin_ckan_sites()
        results = []
        keywords_lower = [k.lower() for k in keywords]

        for source in self.sources.values():
            if not source.is_enabled:
                continue

            score = 0.0

            # Check source keywords
            source_keywords_lower = [k.lower() for k in source.keywords]
            for kw in keywords_lower:
                if kw in source_keywords_lower:
                    score += 1.0
                elif any(kw in sk for sk in source_keywords_lower):
                    score += 0.5

            # Check source name and description
            name_lower = source.name.lower()
            desc_lower = source.description.lower()
            for kw in keywords_lower:
                if kw in name_lower:
                    score += 0.5
                if kw in desc_lower:
                    score += 0.3

            # Check datasets
            for dataset in source.datasets:
                dataset_keywords_lower = [k.lower() for k in dataset.keywords]
                for kw in keywords_lower:
                    if kw in dataset_keywords_lower:
                        score += 0.8
                    elif any(kw in dk for dk in dataset_keywords_lower):
                        score += 0.3
                    if kw in dataset.name.lower():
                        score += 0.4
                    if kw in dataset.description.lower():
                        score += 0.2

            if score > 0:
                # Normalize by number of keywords
                normalized_score = score / len(keywords)
                results.append((source, normalized_score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_datasets_for_query(
        self,
        query: str,
        categories: list[DataCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """Find relevant datasets across all sources for a query.

        Args:
            query: User's natural language query
            categories: Optional list of categories to filter by

        Returns:
            List of dataset recommendations with source info
        """
        # Extract keywords from query
        query_lower = query.lower()

        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {"what", "is", "the", "in", "of", "for", "how", "has", "have", "does", "do", "a", "an", "and", "or", "to", "from"}
        words = query_lower.replace("?", "").replace(",", "").split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Search sources
        source_results = self.search_by_keywords(keywords)

        recommendations = []

        for source, score in source_results:
            # Filter by category if specified
            if categories:
                if not any(cat in source.categories for cat in categories):
                    continue

            # Find matching datasets within source
            matching_datasets = []
            for dataset in source.datasets:
                dataset_match_score = 0.0
                dataset_keywords_lower = [k.lower() for k in dataset.keywords]

                for kw in keywords:
                    if kw in dataset_keywords_lower:
                        dataset_match_score += 1.0
                    elif any(kw in dk for dk in dataset_keywords_lower):
                        dataset_match_score += 0.5
                    if kw in dataset.name.lower():
                        dataset_match_score += 0.5
                    if kw in dataset.description.lower():
                        dataset_match_score += 0.2

                if dataset_match_score > 0:
                    matching_datasets.append({
                        "dataset": dataset,
                        "score": dataset_match_score,
                    })

            # Sort datasets by match score
            matching_datasets.sort(key=lambda x: x["score"], reverse=True)

            recommendations.append({
                "source": source,
                "source_score": score,
                "matching_datasets": matching_datasets[:3],  # Top 3 datasets
            })

        return recommendations[:5]  # Top 5 sources

    def get_source_summary(self, source_id: str) -> str:
        """Get a formatted summary of a data source."""
        source = self.get_source(source_id)
        if not source:
            return f"Source '{source_id}' not found."

        lines = [
            f"**{source.name}**",
            f"{source.description}",
            "",
            f"**Website:** {source.base_url}",
        ]

        if source.portal_url:
            lines.append(f"**Data Portal:** {source.portal_url}")

        if source.datasets:
            lines.append("")
            lines.append("**Key Datasets:**")
            for ds in source.datasets[:5]:
                lines.append(f"- {ds.name}: {ds.description[:100]}...")
                if ds.url:
                    lines.append(f"  Link: {ds.url}")

        if source.example_queries:
            lines.append("")
            lines.append("**Example Questions:**")
            for eq in source.example_queries:
                lines.append(f"- {eq}")

        return "\n".join(lines)


# Singleton instance
_registry: DataSourceRegistry | None = None


def get_data_source_registry() -> DataSourceRegistry:
    """Get or create the data source registry singleton."""
    global _registry
    if _registry is None:
        _registry = DataSourceRegistry()
    return _registry
