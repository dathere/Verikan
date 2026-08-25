"""Bureau of Labor Statistics (BLS) API client.

This client interfaces with the BLS Public Data API v2 to fetch
employment, unemployment, wages, and price data.

API Documentation: https://www.bls.gov/developers/api_signature_v2.htm
"""

from datetime import datetime
from typing import Any

import httpx

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger

logger = get_logger(__name__)


# Common BLS series IDs for quick lookups
BLS_SERIES = {
    # Unemployment Rate
    "unemployment_rate_national": "LNS14000000",  # National unemployment rate
    "unemployment_rate_men": "LNS14000001",  # Men 20+
    "unemployment_rate_women": "LNS14000002",  # Women 20+

    # Employment
    "total_nonfarm_employment": "CES0000000001",  # Total nonfarm employment
    "total_private_employment": "CES0500000001",  # Total private employment

    # Consumer Price Index
    "cpi_all_urban": "CUUR0000SA0",  # CPI-U All items
    "cpi_food": "CUUR0000SAF1",  # CPI Food
    "cpi_energy": "CUUR0000SA0E",  # CPI Energy
    "cpi_core": "CUUR0000SA0L1E",  # CPI less food and energy

    # Producer Price Index
    "ppi_final_demand": "WPSFD4",  # PPI Final Demand

    # Average Hourly Earnings
    "avg_hourly_earnings_private": "CES0500000003",  # All private employees

    # Labor Force
    "labor_force_participation": "LNS11300000",  # Labor force participation rate
    "civilian_labor_force": "LNS11000000",  # Civilian labor force level
}

# State FIPS codes for LAUS series
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


class BLSClient:
    """Client for interacting with the BLS Public Data API.

    The BLS API provides access to labor statistics including:
    - Employment and unemployment data
    - Consumer Price Index (CPI)
    - Producer Price Index (PPI)
    - Wages and earnings
    - Productivity measures

    API Documentation: https://www.bls.gov/developers/
    """

    API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the BLS client.

        Args:
            api_key: BLS API key. Optional but recommended for higher rate limits.
                    Without key: 25 requests/day, 10 years of data
                    With key: 500 requests/day, 20 years of data
        """
        self._api_key = api_key or (
            settings.bls_api_key.get_secret_value()
            if hasattr(settings, 'bls_api_key') and settings.bls_api_key
            else None
        )
        self.logger = get_logger("bls_client")

    async def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a request to the BLS API with a fresh connection.

        This creates a new HTTP client for each request to avoid
        connection reuse issues in async contexts.

        Args:
            payload: Request payload

        Returns:
            API response data
        """
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        ) as client:
            response = await client.post(self.API_URL, json=payload)
            response.raise_for_status()
            return response.json()

    async def get_series_data(
        self,
        series_ids: list[str],
        start_year: int | None = None,
        end_year: int | None = None,
        calculations: bool = True,
        annual_average: bool = True,
    ) -> dict[str, Any]:
        """Fetch time series data from BLS.

        Args:
            series_ids: List of BLS series IDs (max 50 for registered users, 25 for unregistered)
            start_year: Start year for data (default: 10 years ago)
            end_year: End year for data (default: current year)
            calculations: Include percent changes and other calculations
            annual_average: Include annual averages

        Returns:
            Dict with series data and metadata
        """
        # Default to last 10 years if not specified
        current_year = datetime.now().year
        if end_year is None:
            end_year = current_year
        if start_year is None:
            start_year = end_year - 10

        # Build request payload
        payload: dict[str, Any] = {
            "seriesid": series_ids,
            "startyear": str(start_year),
            "endyear": str(end_year),
        }

        # Add optional parameters if API key is available
        if self._api_key:
            payload["registrationkey"] = self._api_key
            payload["calculations"] = calculations
            payload["annualaverage"] = annual_average

        try:
            self.logger.debug(
                "Fetching BLS data",
                series_ids=series_ids,
                start_year=start_year,
                end_year=end_year,
            )

            data = await self._make_request(payload)

            if data.get("status") != "REQUEST_SUCCEEDED":
                error_msg = data.get("message", ["Unknown error"])
                self.logger.error("BLS API returned error", errors=error_msg)
                return {"error": error_msg, "series": []}

            series_count = len(data.get("Results", {}).get("series", []))
            self.logger.info("BLS data retrieved successfully", series_count=series_count)

            return data.get("Results", {})

        except httpx.TimeoutException as e:
            self.logger.error("BLS API request timed out", error=str(e))
            return {"error": f"Request timed out: {e}", "series": []}

        except httpx.HTTPStatusError as e:
            self.logger.error("BLS API HTTP error", status_code=e.response.status_code, error=str(e))
            return {"error": f"HTTP error {e.response.status_code}: {e}", "series": []}

        except Exception as e:
            self.logger.error("BLS API request failed", error_type=type(e).__name__, error=str(e))
            return {"error": f"Request failed: {e}", "series": []}

    async def get_unemployment_rate(
        self,
        state: str | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        """Get unemployment rate data.

        Args:
            state: State name (optional, national if not specified)
            start_year: Start year
            end_year: End year

        Returns:
            Dict with unemployment rate data
        """
        if state:
            # State unemployment rate from LAUS
            state_lower = state.lower()
            fips = STATE_FIPS.get(state_lower)
            if not fips:
                return {"error": f"Unknown state: {state}"}
            # LAUS series: LAUST{fips}0000000000003 = unemployment rate
            series_id = f"LAUST{fips}0000000000003"
        else:
            # National unemployment rate
            series_id = BLS_SERIES["unemployment_rate_national"]

        result = await self.get_series_data(
            [series_id],
            start_year=start_year,
            end_year=end_year,
        )

        # Parse and format the result
        return self._format_series_result(result, "Unemployment Rate", state or "National")

    async def get_cpi(
        self,
        category: str = "all",
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        """Get Consumer Price Index data.

        Args:
            category: CPI category - 'all', 'food', 'energy', 'core'
            start_year: Start year
            end_year: End year

        Returns:
            Dict with CPI data
        """
        series_map = {
            "all": BLS_SERIES["cpi_all_urban"],
            "food": BLS_SERIES["cpi_food"],
            "energy": BLS_SERIES["cpi_energy"],
            "core": BLS_SERIES["cpi_core"],
        }

        series_id = series_map.get(category.lower(), series_map["all"])

        result = await self.get_series_data(
            [series_id],
            start_year=start_year,
            end_year=end_year,
        )

        return self._format_series_result(result, f"CPI ({category})", "National")

    async def get_employment(
        self,
        sector: str = "total_nonfarm",
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        """Get employment data.

        Args:
            sector: Employment sector - 'total_nonfarm', 'private'
            start_year: Start year
            end_year: End year

        Returns:
            Dict with employment data
        """
        series_map = {
            "total_nonfarm": BLS_SERIES["total_nonfarm_employment"],
            "private": BLS_SERIES["total_private_employment"],
        }

        series_id = series_map.get(sector.lower(), series_map["total_nonfarm"])

        result = await self.get_series_data(
            [series_id],
            start_year=start_year,
            end_year=end_year,
        )

        return self._format_series_result(result, f"Employment ({sector})", "National")

    async def get_wages(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, Any]:
        """Get average hourly earnings data.

        Args:
            start_year: Start year
            end_year: End year

        Returns:
            Dict with wage data
        """
        result = await self.get_series_data(
            [BLS_SERIES["avg_hourly_earnings_private"]],
            start_year=start_year,
            end_year=end_year,
        )

        return self._format_series_result(result, "Average Hourly Earnings", "National")

    async def search_series(self, query: str) -> list[dict[str, Any]]:
        """Search for BLS series matching a query.

        This is a local search through common series - BLS doesn't have a search API.

        Args:
            query: Search query

        Returns:
            List of matching series info
        """
        query_lower = query.lower()
        results = []

        series_descriptions = {
            "unemployment_rate_national": "National Unemployment Rate (Seasonally Adjusted)",
            "total_nonfarm_employment": "Total Nonfarm Employment (Thousands)",
            "total_private_employment": "Total Private Employment (Thousands)",
            "cpi_all_urban": "Consumer Price Index - All Urban Consumers, All Items",
            "cpi_food": "Consumer Price Index - Food",
            "cpi_energy": "Consumer Price Index - Energy",
            "cpi_core": "Consumer Price Index - All Items Less Food and Energy",
            "ppi_final_demand": "Producer Price Index - Final Demand",
            "avg_hourly_earnings_private": "Average Hourly Earnings - All Private Employees",
            "labor_force_participation": "Labor Force Participation Rate",
            "civilian_labor_force": "Civilian Labor Force Level",
        }

        for key, series_id in BLS_SERIES.items():
            description = series_descriptions.get(key, key.replace("_", " ").title())
            if query_lower in key.lower() or query_lower in description.lower():
                results.append({
                    "series_id": series_id,
                    "name": key,
                    "description": description,
                })

        return results

    def _format_series_result(
        self,
        result: dict[str, Any],
        data_type: str,
        geography: str,
    ) -> dict[str, Any]:
        """Format BLS API result into a standardized structure.

        Args:
            result: Raw API result
            data_type: Type of data (e.g., "Unemployment Rate")
            geography: Geographic area

        Returns:
            Formatted result dict
        """
        if "error" in result:
            return result

        series_list = result.get("series", [])
        if not series_list:
            return {"error": "No data returned", "observations": []}

        series = series_list[0]
        series_id = series.get("seriesID", "")
        data_points = series.get("data", [])

        observations = []
        for point in data_points:
            year = point.get("year")
            period = point.get("period", "")
            value = point.get("value")

            # Skip annual averages (M13) unless that's all we have
            if period == "M13":
                period_name = "Annual"
                date = f"{year}"
            else:
                # Convert M01-M12 to month names
                month_num = int(period[1:]) if period.startswith("M") else 0
                if month_num > 0:
                    date = f"{year}-{month_num:02d}"
                    period_name = datetime(2000, month_num, 1).strftime("%B")
                else:
                    date = year
                    period_name = period

            try:
                value_float = float(value)
            except (TypeError, ValueError):
                value_float = None

            obs = {
                "date": date,
                "year": year,
                "period": period,
                "period_name": period_name,
                "value": value_float,
                "value_raw": value,
            }

            # Add calculations if available
            if "calculations" in point:
                calcs = point["calculations"]
                if "pct_changes" in calcs:
                    pct = calcs["pct_changes"]
                    obs["pct_change_1_month"] = pct.get("1")
                    obs["pct_change_3_month"] = pct.get("3")
                    obs["pct_change_6_month"] = pct.get("6")
                    obs["pct_change_12_month"] = pct.get("12")

            observations.append(obs)

        # Sort by date descending (most recent first)
        observations.sort(key=lambda x: x["date"], reverse=True)

        return {
            "series_id": series_id,
            "data_type": data_type,
            "geography": geography,
            "observations": observations,
            "latest": observations[0] if observations else None,
            "count": len(observations),
        }

    def get_state_unemployment_series_id(self, state: str) -> str | None:
        """Get the LAUS series ID for a state's unemployment rate.

        Args:
            state: State name

        Returns:
            Series ID or None if state not found
        """
        state_lower = state.lower()
        fips = STATE_FIPS.get(state_lower)
        if fips:
            return f"LAUST{fips}0000000000003"
        return None

    def generate_reproducible_code(
        self,
        series_ids: list[str],
        start_year: int,
        end_year: int,
    ) -> str:
        """Generate reproducible Python code for BLS data retrieval.

        Args:
            series_ids: Series IDs to fetch
            start_year: Start year
            end_year: End year

        Returns:
            Python code string
        """
        series_str = repr(series_ids)

        code = f'''"""
BLS Data Retrieval
Generated for reproducibility in Jupyter notebooks

This code retrieves data from the BLS Public Data API v2.
"""

import requests
import pandas as pd

# BLS API Configuration
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

def fetch_bls_data(
    series_ids: list,
    start_year: int = {start_year},
    end_year: int = {end_year},
    api_key: str = None,
) -> pd.DataFrame:
    """Fetch data from BLS API.

    Args:
        series_ids: List of BLS series IDs
        start_year: Start year for data
        end_year: End year for data
        api_key: Optional BLS API key for higher limits

    Returns:
        DataFrame with the time series data
    """
    payload = {{
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }}

    if api_key:
        payload["registrationkey"] = api_key
        payload["calculations"] = True
        payload["annualaverage"] = True

    response = requests.post(BLS_API_URL, json=payload)
    response.raise_for_status()

    data = response.json()
    if data.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError(f"BLS API error: {{data.get('message')}}")

    # Parse results into DataFrame
    records = []
    for series in data.get("Results", {{}}).get("series", []):
        series_id = series.get("seriesID")
        for point in series.get("data", []):
            records.append({{
                "series_id": series_id,
                "year": point.get("year"),
                "period": point.get("period"),
                "value": float(point.get("value")) if point.get("value") else None,
            }})

    df = pd.DataFrame(records)

    # Create proper date column
    def parse_period(row):
        if row["period"] == "M13":
            return f"{{row['year']}}-12-31"  # Annual average
        month = int(row["period"][1:]) if row["period"].startswith("M") else 1
        return f"{{row['year']}}-{{month:02d}}-01"

    df["date"] = pd.to_datetime(df.apply(parse_period, axis=1))
    df = df.sort_values("date")

    return df

# Fetch the data
print("Fetching BLS data...")
print(f"Series: {series_str}")
print(f"Period: {start_year} to {end_year}")

df = fetch_bls_data({series_str})

print(f"\\nLoaded {{len(df):,}} observations")
print(f"\\nDate range: {{df['date'].min()}} to {{df['date'].max()}}")

# Display recent data
print("\\n=== Recent Data ===")
print(df.tail(12).to_string(index=False))

# Plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for series_id in df["series_id"].unique():
    series_data = df[df["series_id"] == series_id]
    plt.plot(series_data["date"], series_data["value"], label=series_id)

plt.xlabel("Date")
plt.ylabel("Value")
plt.title("BLS Time Series Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
'''

        return code
