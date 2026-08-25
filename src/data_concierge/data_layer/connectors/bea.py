"""Bureau of Economic Analysis (BEA) API client.

This client interfaces with the BEA API to fetch GDP, personal income,
trade, and other economic data.

API Documentation: https://apps.bea.gov/api/
"""

from typing import Any

import httpx

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.connectors import register_closeable

logger = get_logger(__name__)


# State FIPS codes for regional data
STATE_FIPS = {
    "alabama": "01000", "alaska": "02000", "arizona": "04000", "arkansas": "05000",
    "california": "06000", "colorado": "08000", "connecticut": "09000", "delaware": "10000",
    "florida": "12000", "georgia": "13000", "hawaii": "15000", "idaho": "16000",
    "illinois": "17000", "indiana": "18000", "iowa": "19000", "kansas": "20000",
    "kentucky": "21000", "louisiana": "22000", "maine": "23000", "maryland": "24000",
    "massachusetts": "25000", "michigan": "26000", "minnesota": "27000", "mississippi": "28000",
    "missouri": "29000", "montana": "30000", "nebraska": "31000", "nevada": "32000",
    "new hampshire": "33000", "new jersey": "34000", "new mexico": "35000", "new york": "36000",
    "north carolina": "37000", "north dakota": "38000", "ohio": "39000", "oklahoma": "40000",
    "oregon": "41000", "pennsylvania": "42000", "rhode island": "44000", "south carolina": "45000",
    "south dakota": "46000", "tennessee": "47000", "texas": "48000", "utah": "49000",
    "vermont": "50000", "virginia": "51000", "washington": "53000", "west virginia": "54000",
    "wisconsin": "55000", "wyoming": "56000", "district of columbia": "11000", "dc": "11000",
}


class BEAClient:
    """Client for interacting with the BEA API.

    Provides access to:
    - National Income and Product Accounts (NIPA) - GDP, consumption, investment
    - Regional Economic Accounts - State GDP, personal income
    - International Trade data
    - Industry data

    API Documentation: https://apps.bea.gov/api/
    """

    API_URL = "https://apps.bea.gov/api/data"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the BEA client.

        Args:
            api_key: BEA API key (free at https://apps.bea.gov/API/signup/)
        """
        self._api_key = api_key or (
            settings.bea_api_key.get_secret_value()
            if hasattr(settings, 'bea_api_key') and settings.bea_api_key
            else None
        )
        self._client: httpx.AsyncClient | None = None
        self.logger = get_logger("bea_client")
        register_closeable(self)

        if not self._api_key:
            self.logger.warning("No BEA API key configured - API calls will fail")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make a request to the BEA API.

        Args:
            params: Query parameters

        Returns:
            API response data
        """
        if not self._api_key:
            return {"error": "BEA API key not configured"}

        client = await self._get_client()

        request_params = {
            "UserID": self._api_key,
            "ResultFormat": "JSON",
            **params,
        }

        try:
            self.logger.debug("BEA API request", params=request_params)
            response = await client.get(self.API_URL, params=request_params)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if "BEAAPI" in data:
                results = data["BEAAPI"].get("Results", {})
                if "Error" in results:
                    error = results["Error"]
                    error_msg = error.get("ErrorDetail", {}).get("Description", str(error))
                    return {"error": error_msg}
                return results

            return data

        except httpx.HTTPStatusError as e:
            self.logger.error("BEA HTTP error", status=e.response.status_code, error=str(e))
            return {"error": f"HTTP {e.response.status_code}: {str(e)}"}
        except Exception as e:
            self.logger.error("BEA API error", error=str(e))
            return {"error": str(e)}

    async def get_dataset_list(self) -> list[dict[str, Any]]:
        """Get list of available BEA datasets.

        Returns:
            List of dataset info
        """
        result = await self._request({"method": "GETDATASETLIST"})

        if "error" in result:
            return []

        datasets = result.get("Dataset", [])
        return [
            {
                "name": ds.get("DatasetName"),
                "description": ds.get("DatasetDescription"),
            }
            for ds in datasets
        ]

    async def get_parameter_list(self, dataset: str) -> list[dict[str, Any]]:
        """Get parameters for a BEA dataset.

        Args:
            dataset: Dataset name (e.g., 'NIPA', 'Regional')

        Returns:
            List of parameter info
        """
        result = await self._request({
            "method": "GETPARAMETERLIST",
            "DatasetName": dataset,
        })

        if "error" in result:
            return []

        params = result.get("Parameter", [])
        return [
            {
                "name": p.get("ParameterName"),
                "description": p.get("ParameterDescription"),
                "required": p.get("ParameterIsRequiredFlag") == "1",
                "default": p.get("ParameterDefaultValue"),
            }
            for p in params
        ]

    async def get_nipa_data(
        self,
        table_name: str,
        frequency: str = "Q",
        year: str | list[str] = "X",
    ) -> dict[str, Any]:
        """Get National Income and Product Accounts (NIPA) data.

        Args:
            table_name: NIPA table name (e.g., 'T10101' for GDP)
            frequency: 'A' (annual), 'Q' (quarterly), or 'M' (monthly)
            year: Year(s) to retrieve or 'X' for all years

        Returns:
            NIPA data
        """
        if isinstance(year, list):
            year = ",".join(str(y) for y in year)

        result = await self._request({
            "method": "GETDATA",
            "DatasetName": "NIPA",
            "TableName": table_name,
            "Frequency": frequency,
            "Year": year,
        })

        if "error" in result:
            return result

        # Parse the data
        data = result.get("Data", [])
        notes = result.get("Notes", [])

        observations = []
        for row in data:
            obs = {
                "table_name": row.get("TableName"),
                "series_code": row.get("SeriesCode"),
                "line_number": row.get("LineNumber"),
                "line_description": row.get("LineDescription"),
                "time_period": row.get("TimePeriod"),
                "value": None,
                "value_raw": row.get("DataValue"),
            }

            # Parse numeric value
            value = row.get("DataValue", "")
            if value and value not in ["(NA)", "(D)", "--"]:
                try:
                    # Remove commas from numbers
                    value_clean = value.replace(",", "")
                    obs["value"] = float(value_clean)
                except ValueError:
                    pass

            observations.append(obs)

        return {
            "dataset": "NIPA",
            "table_name": table_name,
            "frequency": frequency,
            "observations": observations,
            "notes": [n.get("NoteText") for n in notes if n.get("NoteText")],
            "count": len(observations),
        }

    async def get_gdp(
        self,
        frequency: str = "Q",
        year: str | list[str] = "X",
        real: bool = True,
    ) -> dict[str, Any]:
        """Get GDP data.

        Args:
            frequency: 'A' (annual) or 'Q' (quarterly)
            year: Year(s) or 'X' for all
            real: If True, get real GDP; if False, nominal GDP

        Returns:
            GDP data
        """
        # T10101 = Real GDP, T10105 = Nominal GDP
        table_name = "T10101" if real else "T10105"

        result = await self.get_nipa_data(table_name, frequency, year)

        if "error" not in result:
            result["data_type"] = "Real GDP" if real else "Nominal GDP"
            result["units"] = "Billions of chained 2017 dollars" if real else "Billions of dollars"

            # Filter to just the top-level GDP line
            gdp_obs = [
                obs for obs in result["observations"]
                if obs.get("line_number") == "1"
            ]
            result["gdp_observations"] = gdp_obs

        return result

    async def get_regional_gdp(
        self,
        state: str | None = None,
        year: str | list[str] = "LAST5",
        component: str = "RGDP_SAN",
    ) -> dict[str, Any]:
        """Get regional (state-level) GDP data.

        Args:
            state: State name or FIPS code, or None for all states
            year: Year(s) or 'LAST5', 'LAST10', etc.
            component: Component code:
                      'RGDP_SAN' = Real GDP (millions, chained 2017 dollars)
                      'RGDP_MAN' = Real GDP percent change
                      'SAGDP1' = GDP by state

        Returns:
            Regional GDP data
        """
        if isinstance(year, list):
            year = ",".join(str(y) for y in year)

        geo_fips = "STATE"
        if state:
            state_lower = state.lower()
            geo_fips = STATE_FIPS.get(state_lower, state)

        result = await self._request({
            "method": "GETDATA",
            "DatasetName": "Regional",
            "TableName": component,
            "GeoFIPS": geo_fips,
            "Year": year,
        })

        if "error" in result:
            return result

        data = result.get("Data", [])

        observations = []
        for row in data:
            obs = {
                "geo_fips": row.get("GeoFips"),
                "geo_name": row.get("GeoName"),
                "time_period": row.get("TimePeriod"),
                "unit": row.get("Unit"),
                "value": None,
                "value_raw": row.get("DataValue"),
            }

            value = row.get("DataValue", "")
            if value and value not in ["(NA)", "(D)", "--"]:
                try:
                    value_clean = value.replace(",", "")
                    obs["value"] = float(value_clean)
                except ValueError:
                    pass

            observations.append(obs)

        return {
            "dataset": "Regional",
            "component": component,
            "data_type": "Regional GDP",
            "observations": observations,
            "count": len(observations),
        }

    async def get_personal_income(
        self,
        state: str | None = None,
        year: str | list[str] = "LAST5",
        table_name: str = "SAINC1",
    ) -> dict[str, Any]:
        """Get personal income data by state.

        Args:
            state: State name or FIPS code, or None for all states
            year: Year(s) or 'LAST5', 'LAST10'
            table_name: Table name:
                       'SAINC1' = Personal Income Summary
                       'SAINC4' = Personal Income and Employment by Major Component

        Returns:
            Personal income data
        """
        if isinstance(year, list):
            year = ",".join(str(y) for y in year)

        geo_fips = "STATE"
        if state:
            state_lower = state.lower()
            geo_fips = STATE_FIPS.get(state_lower, state)

        result = await self._request({
            "method": "GETDATA",
            "DatasetName": "Regional",
            "TableName": table_name,
            "GeoFIPS": geo_fips,
            "Year": year,
            "LineCode": "1,2,3",  # Total personal income, per capita, population
        })

        if "error" in result:
            return result

        data = result.get("Data", [])

        observations = []
        for row in data:
            obs = {
                "geo_fips": row.get("GeoFips"),
                "geo_name": row.get("GeoName"),
                "line_code": row.get("LineCode"),
                "description": row.get("Description"),
                "time_period": row.get("TimePeriod"),
                "unit": row.get("Unit"),
                "value": None,
                "value_raw": row.get("DataValue"),
            }

            value = row.get("DataValue", "")
            if value and value not in ["(NA)", "(D)", "--"]:
                try:
                    value_clean = value.replace(",", "")
                    obs["value"] = float(value_clean)
                except ValueError:
                    pass

            observations.append(obs)

        return {
            "dataset": "Regional",
            "table_name": table_name,
            "data_type": "Personal Income",
            "observations": observations,
            "count": len(observations),
        }

    async def get_trade_data(
        self,
        indicator: str = "Balance",
        area: str = "AllCountries",
        frequency: str = "A",
        year: str = "LAST5",
    ) -> dict[str, Any]:
        """Get international trade data.

        Args:
            indicator: Trade indicator:
                      'Balance' = Trade balance
                      'Exports' = Exports of goods and services
                      'Imports' = Imports of goods and services
            area: Geographic area or 'AllCountries'
            frequency: 'A' (annual), 'Q' (quarterly), or 'M' (monthly)
            year: Year(s) or 'LAST5', 'ALL'

        Returns:
            Trade data
        """
        result = await self._request({
            "method": "GETDATA",
            "DatasetName": "ITA",
            "Indicator": indicator,
            "AreaOrCountry": area,
            "Frequency": frequency,
            "Year": year,
        })

        if "error" in result:
            return result

        data = result.get("Data", [])

        observations = []
        for row in data:
            obs = {
                "indicator": row.get("Indicator"),
                "area": row.get("AreaOrCountry"),
                "time_period": row.get("TimePeriod"),
                "value": None,
                "value_raw": row.get("DataValue"),
            }

            value = row.get("DataValue", "")
            if value and value not in ["(NA)", "(D)", "--"]:
                try:
                    value_clean = value.replace(",", "")
                    obs["value"] = float(value_clean)
                except ValueError:
                    pass

            observations.append(obs)

        return {
            "dataset": "ITA",
            "indicator": indicator,
            "data_type": f"International Trade - {indicator}",
            "observations": observations,
            "count": len(observations),
        }

    def get_state_fips(self, state_name: str) -> str | None:
        """Get BEA FIPS code for a state name.

        Args:
            state_name: State name

        Returns:
            FIPS code or None
        """
        return STATE_FIPS.get(state_name.lower())

    def generate_reproducible_code(
        self,
        dataset: str,
        table_name: str,
        params: dict[str, Any],
    ) -> str:
        """Generate reproducible Python code for BEA data retrieval.

        Args:
            dataset: BEA dataset name
            table_name: Table name
            params: Additional parameters

        Returns:
            Python code string
        """
        params_str = repr(params)

        code = f'''"""
BEA API Data Retrieval
Generated for reproducibility in Jupyter notebooks

This code retrieves data from the Bureau of Economic Analysis API.
Get a free API key at: https://apps.bea.gov/API/signup/
"""

import requests
import pandas as pd

# BEA API Configuration
BEA_API_URL = "https://apps.bea.gov/api/data"
# Replace with your actual API key
API_KEY = "your_api_key_here"

def fetch_bea_data(
    dataset: str,
    table_name: str,
    api_key: str = API_KEY,
    **kwargs,
) -> pd.DataFrame:
    """Fetch data from BEA API.

    Args:
        dataset: BEA dataset name (e.g., 'NIPA', 'Regional')
        table_name: Table name
        api_key: BEA API key
        **kwargs: Additional API parameters

    Returns:
        DataFrame with the data
    """
    params = {{
        "UserID": api_key,
        "method": "GETDATA",
        "DatasetName": dataset,
        "TableName": table_name,
        "ResultFormat": "JSON",
        **kwargs,
    }}

    response = requests.get(BEA_API_URL, params=params)
    response.raise_for_status()

    data = response.json()

    # Extract results
    results = data.get("BEAAPI", {{}}).get("Results", {{}})
    if "Error" in results:
        raise ValueError(f"BEA API error: {{results['Error']}}")

    records = results.get("Data", [])

    df = pd.DataFrame(records)

    # Convert DataValue to numeric
    if "DataValue" in df.columns:
        df["Value"] = pd.to_numeric(
            df["DataValue"].str.replace(",", ""),
            errors="coerce"
        )

    return df

# Fetch the data
print("Fetching BEA data...")
print(f"Dataset: {dataset}")
print(f"Table: {table_name}")
print(f"Parameters: {params_str}")

df = fetch_bea_data(
    "{dataset}",
    "{table_name}",
    **{params_str},
)

print(f"\\nLoaded {{len(df):,}} records")

# Display the data
print("\\n=== Data ===")
print(df.to_string())

# Summary
if "Value" in df.columns:
    print("\\n=== Summary ===")
    print(df["Value"].describe())
'''

        return code
