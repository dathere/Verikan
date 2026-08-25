"""Federal Reserve Economic Data (FRED) API client.

This client interfaces with the FRED API from the Federal Reserve Bank of St. Louis
to fetch economic data including interest rates, GDP, inflation, and more.

API Documentation: https://fred.stlouisfed.org/docs/api/fred/
"""

from typing import Any

import httpx

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.connectors import register_closeable

logger = get_logger(__name__)


# Common FRED series IDs for quick lookups
FRED_SERIES = {
    # GDP
    "gdp": "GDP",  # Gross Domestic Product
    "gdp_real": "GDPC1",  # Real GDP
    "gdp_growth": "A191RL1Q225SBEA",  # Real GDP growth rate

    # Unemployment
    "unemployment_rate": "UNRATE",  # Civilian Unemployment Rate
    "unemployment_claims": "ICSA",  # Initial Jobless Claims
    "labor_force_participation": "CIVPART",  # Labor Force Participation Rate

    # Inflation
    "cpi": "CPIAUCSL",  # Consumer Price Index
    "pce": "PCEPI",  # Personal Consumption Expenditures Price Index
    "inflation_expectations": "MICH",  # University of Michigan Inflation Expectations

    # Interest Rates
    "fed_funds_rate": "FEDFUNDS",  # Federal Funds Rate
    "fed_funds_target": "DFEDTARU",  # Federal Funds Target Rate Upper
    "prime_rate": "DPRIME",  # Bank Prime Loan Rate
    "treasury_10yr": "DGS10",  # 10-Year Treasury Constant Maturity Rate
    "treasury_2yr": "DGS2",  # 2-Year Treasury Constant Maturity Rate
    "treasury_30yr": "DGS30",  # 30-Year Treasury Constant Maturity Rate
    "treasury_3mo": "DTB3",  # 3-Month Treasury Bill Rate

    # Housing
    "housing_starts": "HOUST",  # Housing Starts
    "home_prices": "CSUSHPINSA",  # S&P/Case-Shiller Home Price Index
    "mortgage_rate_30yr": "MORTGAGE30US",  # 30-Year Fixed Rate Mortgage Average

    # Money Supply
    "m1": "M1SL",  # M1 Money Stock
    "m2": "M2SL",  # M2 Money Stock

    # Trade
    "trade_balance": "BOPGSTB",  # Trade Balance: Goods and Services

    # Stock Market
    "sp500": "SP500",  # S&P 500 Index

    # Personal Income
    "personal_income": "PI",  # Personal Income
    "disposable_income": "DSPIC96",  # Real Disposable Personal Income
    "savings_rate": "PSAVERT",  # Personal Savings Rate
}


class FREDClient:
    """Client for interacting with the FRED API.

    FRED provides access to hundreds of thousands of economic time series
    from various sources including:
    - Federal Reserve
    - Bureau of Labor Statistics
    - Bureau of Economic Analysis
    - Census Bureau
    - And many more

    API Documentation: https://fred.stlouisfed.org/docs/api/fred/
    """

    API_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the FRED client.

        Args:
            api_key: FRED API key (required for API access)
                    Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self._api_key = api_key or (
            settings.fred_api_key.get_secret_value()
            if hasattr(settings, 'fred_api_key') and settings.fred_api_key
            else None
        )
        self._client: httpx.AsyncClient | None = None
        self.logger = get_logger("fred_client")
        register_closeable(self)

        if not self._api_key:
            self.logger.warning("No FRED API key configured - API calls will fail")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.API_URL,
                timeout=30.0,
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
    ) -> dict[str, Any]:
        """Make a request to the FRED API.

        Args:
            endpoint: API endpoint (e.g., '/series/observations')
            params: Query parameters

        Returns:
            API response data
        """
        if not self._api_key:
            return {"error": "FRED API key not configured"}

        client = await self._get_client()

        request_params = {
            "api_key": self._api_key,
            "file_type": "json",
            **(params or {}),
        }

        try:
            response = await client.get(endpoint, params=request_params)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            self.logger.error("FRED HTTP error", status=e.response.status_code, error=str(e))
            return {"error": f"HTTP {e.response.status_code}: {str(e)}"}
        except Exception as e:
            self.logger.error("FRED API error", error=str(e))
            return {"error": str(e)}

    async def get_series_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        units: str = "lin",
        frequency: str | None = None,
        aggregation_method: str = "avg",
        limit: int = 100000,
    ) -> dict[str, Any]:
        """Get observations for a FRED series.

        Args:
            series_id: FRED series ID
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            units: Data units transformation:
                   'lin' (levels), 'chg' (change), 'ch1' (change from year ago),
                   'pch' (percent change), 'pc1' (percent change from year ago),
                   'pca' (compounded annual rate of change),
                   'cch' (continuously compounded rate of change),
                   'cca' (continuously compounded annual rate of change),
                   'log' (natural log)
            frequency: Data frequency: 'd', 'w', 'bw', 'm', 'q', 'sa', 'a'
            aggregation_method: How to aggregate: 'avg', 'sum', 'eop'
            limit: Maximum observations to return

        Returns:
            Dict with series observations
        """
        params: dict[str, Any] = {
            "series_id": series_id,
            "units": units,
            "aggregation_method": aggregation_method,
            "limit": limit,
        }

        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        if frequency:
            params["frequency"] = frequency

        result = await self._request("/series/observations", params)

        if "error" in result:
            return result

        observations = []
        for obs in result.get("observations", []):
            value = obs.get("value")
            # FRED uses "." for missing values
            if value and value != ".":
                try:
                    value_float = float(value)
                except ValueError:
                    value_float = None
            else:
                value_float = None

            observations.append({
                "date": obs.get("date"),
                "value": value_float,
                "value_raw": obs.get("value"),
            })

        return {
            "series_id": series_id,
            "observations": observations,
            "count": len(observations),
            "latest": observations[-1] if observations else None,
        }

    async def get_series_info(self, series_id: str) -> dict[str, Any]:
        """Get metadata about a FRED series.

        Args:
            series_id: FRED series ID

        Returns:
            Series metadata
        """
        result = await self._request("/series", {"series_id": series_id})

        if "error" in result:
            return result

        seriess = result.get("seriess", [])
        if not seriess:
            return {"error": f"Series {series_id} not found"}

        series = seriess[0]
        return {
            "series_id": series.get("id"),
            "title": series.get("title"),
            "frequency": series.get("frequency"),
            "frequency_short": series.get("frequency_short"),
            "units": series.get("units"),
            "units_short": series.get("units_short"),
            "seasonal_adjustment": series.get("seasonal_adjustment"),
            "seasonal_adjustment_short": series.get("seasonal_adjustment_short"),
            "last_updated": series.get("last_updated"),
            "observation_start": series.get("observation_start"),
            "observation_end": series.get("observation_end"),
            "popularity": series.get("popularity"),
            "notes": series.get("notes"),
        }

    async def search_series(
        self,
        search_text: str,
        limit: int = 20,
        order_by: str = "popularity",
    ) -> list[dict[str, Any]]:
        """Search for FRED series.

        Args:
            search_text: Search query
            limit: Maximum results
            order_by: Sort order: 'search_rank', 'series_id', 'title',
                     'units', 'frequency', 'seasonal_adjustment',
                     'realtime_start', 'realtime_end', 'last_updated',
                     'observation_start', 'observation_end', 'popularity'

        Returns:
            List of matching series
        """
        result = await self._request("/series/search", {
            "search_text": search_text,
            "limit": limit,
            "order_by": order_by,
        })

        if "error" in result:
            return []

        series_list = []
        for series in result.get("seriess", []):
            series_list.append({
                "series_id": series.get("id"),
                "title": series.get("title"),
                "frequency": series.get("frequency_short"),
                "units": series.get("units_short"),
                "seasonal_adjustment": series.get("seasonal_adjustment_short"),
                "popularity": series.get("popularity"),
                "observation_start": series.get("observation_start"),
                "observation_end": series.get("observation_end"),
            })

        return series_list

    async def get_gdp(
        self,
        real: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get GDP data.

        Args:
            real: If True, get real (inflation-adjusted) GDP
            start_date: Start date
            end_date: End date

        Returns:
            GDP data
        """
        series_id = FRED_SERIES["gdp_real"] if real else FRED_SERIES["gdp"]
        result = await self.get_series_observations(
            series_id,
            start_date=start_date,
            end_date=end_date,
        )

        if "error" not in result:
            result["data_type"] = "Real GDP" if real else "Nominal GDP"
            result["units"] = "Billions of Chained 2017 Dollars" if real else "Billions of Dollars"

        return result

    async def get_unemployment_rate(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get unemployment rate data.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Unemployment rate data
        """
        result = await self.get_series_observations(
            FRED_SERIES["unemployment_rate"],
            start_date=start_date,
            end_date=end_date,
        )

        if "error" not in result:
            result["data_type"] = "Unemployment Rate"
            result["units"] = "Percent"

        return result

    async def get_inflation(
        self,
        measure: str = "cpi",
        percent_change: bool = True,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get inflation data.

        Args:
            measure: Inflation measure - 'cpi' or 'pce'
            percent_change: If True, return year-over-year percent change
            start_date: Start date
            end_date: End date

        Returns:
            Inflation data
        """
        series_id = FRED_SERIES["cpi"] if measure == "cpi" else FRED_SERIES["pce"]
        units = "pc1" if percent_change else "lin"

        result = await self.get_series_observations(
            series_id,
            start_date=start_date,
            end_date=end_date,
            units=units,
        )

        if "error" not in result:
            measure_name = "CPI" if measure == "cpi" else "PCE"
            if percent_change:
                result["data_type"] = f"{measure_name} Year-over-Year Change"
                result["units"] = "Percent"
            else:
                result["data_type"] = f"{measure_name} Index"
                result["units"] = "Index"

        return result

    async def get_interest_rate(
        self,
        rate_type: str = "fed_funds",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Get interest rate data.

        Args:
            rate_type: Rate type - 'fed_funds', 'prime', 'treasury_10yr',
                      'treasury_2yr', 'treasury_30yr', 'treasury_3mo', 'mortgage_30yr'
            start_date: Start date
            end_date: End date

        Returns:
            Interest rate data
        """
        series_map = {
            "fed_funds": FRED_SERIES["fed_funds_rate"],
            "prime": FRED_SERIES["prime_rate"],
            "treasury_10yr": FRED_SERIES["treasury_10yr"],
            "treasury_2yr": FRED_SERIES["treasury_2yr"],
            "treasury_30yr": FRED_SERIES["treasury_30yr"],
            "treasury_3mo": FRED_SERIES["treasury_3mo"],
            "mortgage_30yr": FRED_SERIES["mortgage_rate_30yr"],
        }

        series_id = series_map.get(rate_type, FRED_SERIES["fed_funds_rate"])

        result = await self.get_series_observations(
            series_id,
            start_date=start_date,
            end_date=end_date,
        )

        rate_names = {
            "fed_funds": "Federal Funds Rate",
            "prime": "Prime Rate",
            "treasury_10yr": "10-Year Treasury Rate",
            "treasury_2yr": "2-Year Treasury Rate",
            "treasury_30yr": "30-Year Treasury Rate",
            "treasury_3mo": "3-Month Treasury Bill Rate",
            "mortgage_30yr": "30-Year Mortgage Rate",
        }

        if "error" not in result:
            result["data_type"] = rate_names.get(rate_type, rate_type)
            result["units"] = "Percent"

        return result

    def get_common_series(self) -> dict[str, str]:
        """Get a dictionary of common FRED series IDs.

        Returns:
            Dict mapping descriptive names to series IDs
        """
        return FRED_SERIES.copy()

    def generate_reproducible_code(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Generate reproducible Python code for FRED data retrieval.

        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date

        Returns:
            Python code string
        """
        start_param = f'"{start_date}"' if start_date else "None"
        end_param = f'"{end_date}"' if end_date else "None"

        code = f'''"""
FRED Data Retrieval: {series_id}
Generated for reproducibility in Jupyter notebooks

This code retrieves data from the FRED API.
Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import requests
import pandas as pd

# FRED API Configuration
FRED_API_URL = "https://api.stlouisfed.org/fred"
# Replace with your actual API key
API_KEY = "your_api_key_here"

def fetch_fred_data(
    series_id: str,
    start_date: str = {start_param},
    end_date: str = {end_param},
    api_key: str = API_KEY,
) -> pd.DataFrame:
    """Fetch data from FRED API.

    Args:
        series_id: FRED series ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FRED API key

    Returns:
        DataFrame with the time series data
    """
    params = {{
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }}

    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    response = requests.get(f"{{FRED_API_URL}}/series/observations", params=params)
    response.raise_for_status()

    data = response.json()
    observations = data.get("observations", [])

    records = []
    for obs in observations:
        value = obs.get("value")
        if value and value != ".":
            try:
                value = float(value)
            except ValueError:
                value = None
        else:
            value = None

        records.append({{
            "date": obs.get("date"),
            "value": value,
        }})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"])

    return df

# Fetch the data
print(f"Fetching FRED series: {series_id}")

df = fetch_fred_data("{series_id}")

print(f"\\nLoaded {{len(df):,}} observations")
print(f"Date range: {{df['date'].min()}} to {{df['date'].max()}}")

# Display recent data
print("\\n=== Recent Data ===")
print(df.tail(12).to_string(index=False))

# Plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["value"])
plt.xlabel("Date")
plt.ylabel("Value")
plt.title(f"FRED Series: {series_id}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

        return code
