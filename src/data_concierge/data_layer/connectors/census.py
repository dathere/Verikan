"""U.S. Census Bureau API client.

This client interfaces with the Census Bureau API to fetch demographic,
economic, and geographic data from various surveys and programs.

API Documentation: https://www.census.gov/data/developers/data-sets.html
"""

from typing import Any

import httpx

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.connectors import register_closeable

logger = get_logger(__name__)


# Common Census variables
CENSUS_VARIABLES = {
    # Population
    "total_population": "B01003_001E",  # Total Population
    "median_age": "B01002_001E",  # Median Age

    # Income
    "median_household_income": "B19013_001E",  # Median Household Income
    "per_capita_income": "B19301_001E",  # Per Capita Income
    "mean_household_income": "B19025_001E",  # Mean Household Income

    # Poverty
    "poverty_count": "B17001_002E",  # Population below poverty level
    "poverty_universe": "B17001_001E",  # Population for poverty determination

    # Housing
    "total_housing_units": "B25001_001E",  # Total Housing Units
    "median_home_value": "B25077_001E",  # Median Home Value
    "median_rent": "B25064_001E",  # Median Gross Rent
    "owner_occupied": "B25003_002E",  # Owner-Occupied Housing Units
    "renter_occupied": "B25003_003E",  # Renter-Occupied Housing Units

    # Education
    "bachelors_degree_plus": "B15003_022E",  # Bachelor's degree holders
    "high_school_graduate": "B15003_017E",  # High school graduate

    # Employment
    "employed": "B23025_004E",  # Employed Population 16+
    "unemployed": "B23025_005E",  # Unemployed Population 16+
    "labor_force": "B23025_002E",  # In Labor Force

    # Race/Ethnicity
    "white_alone": "B02001_002E",  # White alone
    "black_alone": "B02001_003E",  # Black or African American alone
    "asian_alone": "B02001_005E",  # Asian alone
    "hispanic_latino": "B03003_003E",  # Hispanic or Latino

    # Age
    "under_18": "B09001_001E",  # Population under 18
    "65_and_over": "B01001_020E",  # Population 65 and over (male) - needs summing
}

# State FIPS codes
STATE_FIPS = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
    "california": "06", "colorado": "08", "connecticut": "09", "delaware": "10",
    "florida": "12", "georgia": "13", "hawaii": "15", "idaho": "16",
    "illinois": "17", "indiana": "18", "iowa": "19", "kansas": "20",
    "kentucky": "21", "louisiana": "22", "maine": "23", "maryland": "24",
    "massachusetts": "25", "michigan": "26", "minnesota": "27", "mississippi": "28",
    "missouri": "29", "montana": "30", "nebraska": "31", "nevada": "32",
    "new hampshire": "33", "new jersey": "34", "new mexico": "35", "new york": "36",
    "north carolina": "37", "north dakota": "38", "ohio": "39", "oklahoma": "40",
    "oregon": "41", "pennsylvania": "42", "rhode island": "44", "south carolina": "45",
    "south dakota": "46", "tennessee": "47", "texas": "48", "utah": "49",
    "vermont": "50", "virginia": "51", "washington": "53", "west virginia": "54",
    "wisconsin": "55", "wyoming": "56", "district of columbia": "11", "dc": "11",
    "puerto rico": "72",
}

# Reverse lookup
FIPS_TO_STATE = {v: k.title() for k, v in STATE_FIPS.items()}


class CensusClient:
    """Client for interacting with the U.S. Census Bureau API.

    Provides access to:
    - American Community Survey (ACS) 1-year and 5-year estimates
    - Decennial Census data
    - Population Estimates
    - Economic data

    API Documentation: https://www.census.gov/data/developers.html
    """

    API_URL = "https://api.census.gov/data"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Census client.

        Args:
            api_key: Census API key (free at https://api.census.gov/data/key_signup.html)
        """
        self._api_key = api_key or (
            settings.census_api_key.get_secret_value()
            if hasattr(settings, 'census_api_key') and settings.census_api_key
            else None
        )
        self._client: httpx.AsyncClient | None = None
        self.logger = get_logger("census_client")
        register_closeable(self)

        if not self._api_key:
            self.logger.warning("No Census API key configured - some requests may fail")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_URL,
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> list[list[Any]] | dict[str, Any]:
        """Make a request to the Census API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            API response data (list of lists for data queries)
        """
        client = await self._get_client()

        request_params = params.copy() if params else {}
        if self._api_key:
            request_params["key"] = self._api_key

        try:
            self.logger.debug("Census API request", endpoint=endpoint, params=request_params)
            response = await client.get(endpoint, params=request_params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            self.logger.error("Census HTTP error", status=e.response.status_code, error=str(e))
            return {"error": f"HTTP {e.response.status_code}: {str(e)}"}
        except Exception as e:
            self.logger.error("Census API error", error=str(e))
            return {"error": str(e)}

    async def get_acs_data(
        self,
        variables: list[str],
        year: int = 2022,
        acs_type: str = "acs5",
        geography: str = "state",
        state: str | None = None,
        county: str | None = None,
    ) -> dict[str, Any]:
        """Get American Community Survey data.

        Args:
            variables: List of ACS variable codes
            year: Survey year
            acs_type: ACS type - 'acs1' (1-year) or 'acs5' (5-year)
            geography: Geographic level - 'us', 'state', 'county', 'tract', 'place'
            state: State FIPS code or name (required for county/tract)
            county: County FIPS code (for tract-level data)

        Returns:
            Dict with data and metadata
        """
        # Build endpoint
        endpoint = f"/{year}/{acs_type}"

        # Build get parameter
        get_vars = ",".join(["NAME"] + variables)

        # Build for parameter based on geography
        state_fips = None
        if state:
            state_lower = state.lower()
            state_fips = STATE_FIPS.get(state_lower, state)

        if geography == "us":
            for_param = "us:*"
        elif geography == "state":
            if state_fips:
                for_param = f"state:{state_fips}"
            else:
                for_param = "state:*"
        elif geography == "county":
            if state_fips:
                for_param = f"county:*&in=state:{state_fips}"
            else:
                for_param = "county:*"
        elif geography == "place":
            if state_fips:
                for_param = f"place:*&in=state:{state_fips}"
            else:
                for_param = "place:*"
        else:
            for_param = f"{geography}:*"

        params = {
            "get": get_vars,
            "for": for_param,
        }

        result = await self._request(endpoint, params)

        if isinstance(result, dict) and "error" in result:
            return result

        if not result or len(result) < 2:
            return {"error": "No data returned", "data": []}

        # Parse response (first row is headers)
        headers = result[0]
        data_rows = result[1:]

        records = []
        for row in data_rows:
            record = {}
            for i, header in enumerate(headers):
                value = row[i] if i < len(row) else None

                # Try to convert numeric values
                if value is not None and header not in ["NAME", "state", "county", "place", "tract"]:
                    try:
                        # Census uses negative numbers for missing data
                        value = float(value)
                        if value < 0:
                            value = None
                    except (ValueError, TypeError):
                        pass

                record[header] = value

            # Add state name if we have state FIPS
            if "state" in record:
                record["state_name"] = FIPS_TO_STATE.get(record["state"], record["state"])

            records.append(record)

        return {
            "year": year,
            "survey": acs_type.upper(),
            "geography": geography,
            "variables": variables,
            "data": records,
            "count": len(records),
        }

    async def get_population(
        self,
        year: int = 2022,
        geography: str = "state",
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get population data.

        Args:
            year: Survey year
            geography: Geographic level
            state: State name or FIPS code

        Returns:
            Population data
        """
        result = await self.get_acs_data(
            variables=[CENSUS_VARIABLES["total_population"]],
            year=year,
            geography=geography,
            state=state,
        )

        if "error" not in result and result.get("data"):
            # Rename variable to friendly name
            for record in result["data"]:
                if CENSUS_VARIABLES["total_population"] in record:
                    record["population"] = record.pop(CENSUS_VARIABLES["total_population"])

            result["data_type"] = "Total Population"

        return result

    async def get_median_income(
        self,
        year: int = 2022,
        geography: str = "state",
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get median household income data.

        Args:
            year: Survey year
            geography: Geographic level
            state: State name or FIPS code

        Returns:
            Income data
        """
        result = await self.get_acs_data(
            variables=[CENSUS_VARIABLES["median_household_income"]],
            year=year,
            geography=geography,
            state=state,
        )

        if "error" not in result and result.get("data"):
            for record in result["data"]:
                if CENSUS_VARIABLES["median_household_income"] in record:
                    record["median_household_income"] = record.pop(CENSUS_VARIABLES["median_household_income"])

            result["data_type"] = "Median Household Income"
            result["units"] = "Dollars"

        return result

    async def get_poverty_rate(
        self,
        year: int = 2022,
        geography: str = "state",
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get poverty rate data.

        Args:
            year: Survey year
            geography: Geographic level
            state: State name or FIPS code

        Returns:
            Poverty data
        """
        result = await self.get_acs_data(
            variables=[
                CENSUS_VARIABLES["poverty_count"],
                CENSUS_VARIABLES["poverty_universe"],
            ],
            year=year,
            geography=geography,
            state=state,
        )

        if "error" not in result and result.get("data"):
            for record in result["data"]:
                poverty_count = record.get(CENSUS_VARIABLES["poverty_count"])
                poverty_universe = record.get(CENSUS_VARIABLES["poverty_universe"])

                if poverty_count and poverty_universe and poverty_universe > 0:
                    record["poverty_rate"] = round((poverty_count / poverty_universe) * 100, 1)
                else:
                    record["poverty_rate"] = None

                record["poverty_count"] = record.pop(CENSUS_VARIABLES["poverty_count"], None)
                record["poverty_universe"] = record.pop(CENSUS_VARIABLES["poverty_universe"], None)

            result["data_type"] = "Poverty Rate"
            result["units"] = "Percent"

        return result

    async def get_housing_data(
        self,
        year: int = 2022,
        geography: str = "state",
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get housing data including median home value and rent.

        Args:
            year: Survey year
            geography: Geographic level
            state: State name or FIPS code

        Returns:
            Housing data
        """
        result = await self.get_acs_data(
            variables=[
                CENSUS_VARIABLES["total_housing_units"],
                CENSUS_VARIABLES["median_home_value"],
                CENSUS_VARIABLES["median_rent"],
                CENSUS_VARIABLES["owner_occupied"],
                CENSUS_VARIABLES["renter_occupied"],
            ],
            year=year,
            geography=geography,
            state=state,
        )

        if "error" not in result and result.get("data"):
            for record in result["data"]:
                record["total_housing_units"] = record.pop(CENSUS_VARIABLES["total_housing_units"], None)
                record["median_home_value"] = record.pop(CENSUS_VARIABLES["median_home_value"], None)
                record["median_rent"] = record.pop(CENSUS_VARIABLES["median_rent"], None)
                record["owner_occupied"] = record.pop(CENSUS_VARIABLES["owner_occupied"], None)
                record["renter_occupied"] = record.pop(CENSUS_VARIABLES["renter_occupied"], None)

            result["data_type"] = "Housing Statistics"

        return result

    async def get_education_data(
        self,
        year: int = 2022,
        geography: str = "state",
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get education attainment data.

        Args:
            year: Survey year
            geography: Geographic level
            state: State name or FIPS code

        Returns:
            Education data
        """
        result = await self.get_acs_data(
            variables=[
                CENSUS_VARIABLES["total_population"],
                CENSUS_VARIABLES["bachelors_degree_plus"],
                CENSUS_VARIABLES["high_school_graduate"],
            ],
            year=year,
            geography=geography,
            state=state,
        )

        if "error" not in result and result.get("data"):
            for record in result["data"]:
                record["total_population"] = record.pop(CENSUS_VARIABLES["total_population"], None)
                record["bachelors_degree_plus"] = record.pop(CENSUS_VARIABLES["bachelors_degree_plus"], None)
                record["high_school_graduate"] = record.pop(CENSUS_VARIABLES["high_school_graduate"], None)

            result["data_type"] = "Educational Attainment"

        return result

    async def get_employment_data(
        self,
        year: int = 2022,
        geography: str = "state",
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get employment status data.

        Args:
            year: Survey year
            geography: Geographic level
            state: State name or FIPS code

        Returns:
            Employment data
        """
        result = await self.get_acs_data(
            variables=[
                CENSUS_VARIABLES["labor_force"],
                CENSUS_VARIABLES["employed"],
                CENSUS_VARIABLES["unemployed"],
            ],
            year=year,
            geography=geography,
            state=state,
        )

        if "error" not in result and result.get("data"):
            for record in result["data"]:
                labor_force = record.get(CENSUS_VARIABLES["labor_force"])
                unemployed = record.get(CENSUS_VARIABLES["unemployed"])

                if labor_force and unemployed and labor_force > 0:
                    record["unemployment_rate"] = round((unemployed / labor_force) * 100, 1)
                else:
                    record["unemployment_rate"] = None

                record["labor_force"] = record.pop(CENSUS_VARIABLES["labor_force"], None)
                record["employed"] = record.pop(CENSUS_VARIABLES["employed"], None)
                record["unemployed"] = record.pop(CENSUS_VARIABLES["unemployed"], None)

            result["data_type"] = "Employment Status"

        return result

    def get_state_fips(self, state_name: str) -> str | None:
        """Get FIPS code for a state name.

        Args:
            state_name: State name

        Returns:
            FIPS code or None
        """
        return STATE_FIPS.get(state_name.lower())

    def get_common_variables(self) -> dict[str, str]:
        """Get a dictionary of common Census variable codes.

        Returns:
            Dict mapping descriptive names to variable codes
        """
        return CENSUS_VARIABLES.copy()

    def generate_reproducible_code(
        self,
        variables: list[str],
        year: int,
        geography: str,
        state: str | None = None,
    ) -> str:
        """Generate reproducible Python code for Census data retrieval.

        Args:
            variables: Variable codes
            year: Survey year
            geography: Geographic level
            state: Optional state

        Returns:
            Python code string
        """
        variables_str = repr(variables)
        state_param = f'"{state}"' if state else "None"

        code = f'''"""
Census API Data Retrieval
Generated for reproducibility in Jupyter notebooks

This code retrieves data from the U.S. Census Bureau API.
Get a free API key at: https://api.census.gov/data/key_signup.html
"""

import requests
import pandas as pd

# Census API Configuration
CENSUS_API_URL = "https://api.census.gov/data"
# Replace with your actual API key
API_KEY = "your_api_key_here"

def fetch_census_data(
    variables: list,
    year: int = {year},
    acs_type: str = "acs5",
    geography: str = "{geography}",
    state: str = {state_param},
    api_key: str = API_KEY,
) -> pd.DataFrame:
    """Fetch data from Census API.

    Args:
        variables: List of Census variable codes
        year: Survey year
        acs_type: 'acs1' or 'acs5'
        geography: Geographic level ('us', 'state', 'county', etc.)
        state: State FIPS code or None
        api_key: Census API key

    Returns:
        DataFrame with Census data
    """
    endpoint = f"{{CENSUS_API_URL}}/{{year}}/{{acs_type}}"

    get_vars = ",".join(["NAME"] + variables)

    if geography == "us":
        for_param = "us:*"
    elif geography == "state":
        for_param = f"state:{{state}}" if state else "state:*"
    elif geography == "county":
        for_param = f"county:*&in=state:{{state}}" if state else "county:*"
    else:
        for_param = f"{{geography}}:*"

    params = {{
        "get": get_vars,
        "for": for_param,
        "key": api_key,
    }}

    response = requests.get(endpoint, params=params)
    response.raise_for_status()

    data = response.json()

    # First row is headers
    headers = data[0]
    records = data[1:]

    df = pd.DataFrame(records, columns=headers)

    # Convert numeric columns
    for var in variables:
        if var in df.columns:
            df[var] = pd.to_numeric(df[var], errors="coerce")
            # Census uses negative numbers for missing data
            df.loc[df[var] < 0, var] = None

    return df

# Fetch the data
print("Fetching Census data...")
print(f"Variables: {variables_str}")
print(f"Year: {year}")
print(f"Geography: {geography}")

df = fetch_census_data({variables_str})

print(f"\\nLoaded {{len(df):,}} records")

# Display the data
print("\\n=== Data ===")
print(df.to_string())

# Summary statistics
print("\\n=== Summary Statistics ===")
print(df.describe())
'''

        return code
