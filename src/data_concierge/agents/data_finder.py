"""Data Finder Agent for retrieving data from Data Commons.

This agent handles the deterministic graph path. CKAN/WPRDC queries are
routed to LLMAnalysisAgent via the LLM graph in supervisor.py.
"""

from datetime import datetime
from typing import Any

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import (
    DataAccessLevel,
    DataSource,
    PlaceEntity,
    RetrievedData,
    StatisticalObservation,
    VariableEntity,
)
from data_concierge.data_layer.connectors.data_commons import DataCommonsClient

logger = get_logger(__name__)


class DataFinderAgent(BaseAgent):
    """Agent responsible for finding and retrieving data from Data Commons."""

    name = "data_finder"
    description = "Finds and retrieves statistical data from Data Commons"

    DATA_SOURCES: dict[str, DataSource] = {
        "data_commons": DataSource(
            id="data_commons",
            name="Google Data Commons",
            url="https://datacommons.org",
            update_frequency="continuous",
            quality_score=0.90,
        ),
        "bls": DataSource(
            id="bls",
            name="Bureau of Labor Statistics",
            url="https://www.bls.gov",
            update_frequency="monthly",
            quality_score=0.95,
        ),
        "census": DataSource(
            id="census",
            name="U.S. Census Bureau",
            url="https://www.census.gov",
            update_frequency="varies",
            quality_score=0.95,
        ),
    }

    VARIABLE_FALLBACKS: dict[str, list[str]] = {
        "UnemploymentRate_Person": [
            "Count_Person_Unemployed",
            "Count_Person_InLaborForce_Unemployed",
        ],
        "Median_Income_Household": [
            "Median_Income_Person",
        ],
        "Count_Person_BelowPovertyLevelInThePast12Months": [
            "Percent_Person_BelowPovertyLevelInThePast12Months",
        ],
    }

    def __init__(self) -> None:
        super().__init__()
        self._dc_client: DataCommonsClient | None = None

    def _get_dc_client(self) -> DataCommonsClient:
        if self._dc_client is None:
            self._dc_client = DataCommonsClient()
        return self._dc_client

    async def process(self, state: GraphState) -> GraphState:
        """Find and retrieve data from Data Commons based on parsed entities."""
        self.logger.info("Finding data", query=state["query"][:50])

        entities = state.get("entities")
        if not entities:
            state["error"] = "No entities parsed from query"
            return state

        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1

        observations: list[StatisticalObservation] = []
        sources_used: list[DataSource] = []
        retrieval_method = "none"
        retrieval_score = 0.0

        if entities.variables and entities.places:
            dc_result = await self._query_data_commons(
                entities.variables,
                entities.places,
                entities.times if hasattr(entities, "times") else [],
            )
            if dc_result:
                observations.extend(dc_result["observations"])
                sources_used.append(self.DATA_SOURCES["data_commons"])
                retrieval_method = "kg_lookup"
                retrieval_score = 0.95

                state["execution_trace"].append({
                    "agent": self.name,
                    "action": "query_data_commons",
                    "api_call": {
                        "endpoint": f"{settings.data_commons_api_url}/v2/observation",
                        "method": "GET",
                        "params": dc_result.get("params", {}),
                    },
                    "code": dc_result.get("code_snippet", ""),
                    "result_count": len(dc_result["observations"]),
                })

        if not observations:
            self.logger.warning("No data retrieved from Data Commons")
            retrieval_method = "none"
            retrieval_score = 0.0

        state["retrieved_data"] = RetrievedData(
            observations=observations,
            source_info=sources_used,
            retrieval_method=retrieval_method,
            retrieval_score=retrieval_score,
            data_vintage=datetime.now().isoformat()[:10],
        )

        self.logger.info(
            "Data retrieval complete",
            observation_count=len(observations),
            method=retrieval_method,
            score=retrieval_score,
        )

        return state

    async def _query_data_commons(
        self,
        variables: list[VariableEntity],
        places: list[PlaceEntity],
        times: list[Any],
    ) -> dict[str, Any] | None:
        """Query Data Commons API for statistical observations.

        Includes fallback mechanism to try alternative variable DCIDs if primary
        returns no data.
        """
        dc_client = self._get_dc_client()

        observations = []

        date_param = "latest"
        if times:
            time = times[0]
            if hasattr(time, "start_date"):
                date_str = time.start_date
                if len(date_str) >= 4:
                    date_param = date_str[:4]

        entity_dcids = [p.dcid for p in places]

        variable_dcids_to_try = []
        var_lookup = {v.dcid: v for v in variables}

        for var in variables:
            variable_dcids_to_try.append(var.dcid)
            if var.dcid in self.VARIABLE_FALLBACKS:
                for fallback in self.VARIABLE_FALLBACKS[var.dcid]:
                    if fallback not in variable_dcids_to_try:
                        variable_dcids_to_try.append(fallback)
                        var_lookup[fallback] = VariableEntity(
                            dcid=fallback,
                            name=fallback.replace("_", " "),
                            unit=var.unit,
                        )

        params_used = {
            "variable_dcids": variable_dcids_to_try,
            "entity_dcids": entity_dcids,
            "date": date_param,
            "select": ["entity", "variable", "value", "date"],
        }

        code_snippet = self._generate_dc_notebook_code(
            variables=variables,
            places=places,
            date_param=date_param,
        )

        place_lookup = {p.dcid: p for p in places}

        try:
            self.logger.info(
                "Querying Data Commons",
                variables=variable_dcids_to_try,
                entities=entity_dcids,
                date=date_param,
            )

            records = await dc_client.get_observations_as_records(
                entity_dcids=entity_dcids,
                variable_dcids=variable_dcids_to_try,
                date=date_param,
            )

            if records:
                for record in records:
                    var_dcid = record.get("variable", "")
                    entity_dcid = record.get("entity", "")

                    var = var_lookup.get(var_dcid)
                    place = place_lookup.get(entity_dcid)

                    if var is None:
                        var = VariableEntity(
                            dcid=var_dcid,
                            name=var_dcid.replace("_", " "),
                            unit=None,
                        )

                    if place is None:
                        place = PlaceEntity(
                            dcid=entity_dcid,
                            name=entity_dcid,
                            place_type="Place",
                        )

                    observations.append(StatisticalObservation(
                        variable=var,
                        place=place,
                        value=record.get("value", 0),
                        date=record.get("date", ""),
                        source=self.DATA_SOURCES["data_commons"],
                        access_level=DataAccessLevel.PUBLIC,
                        observation_period=record.get("observationPeriod"),
                    ))

                self.logger.info(
                    "Data Commons query successful",
                    observation_count=len(observations),
                )
            else:
                self.logger.warning(
                    "Data Commons returned no records",
                    variables=variable_dcids_to_try,
                    entities=entity_dcids,
                )

        except Exception as e:
            self.logger.error(
                "Data Commons API request failed",
                error=str(e),
                variables=variable_dcids_to_try,
                entities=entity_dcids,
            )

        if observations:
            return {
                "observations": observations,
                "params": params_used,
                "code_snippet": code_snippet,
            }

        return None

    def _generate_dc_notebook_code(
        self,
        variables: list[VariableEntity],
        places: list[PlaceEntity],
        date_param: str,
    ) -> str:
        variable_dcids = [v.dcid for v in variables]
        entity_dcids = [p.dcid for p in places]
        var_names = ", ".join([v.name for v in variables])
        place_names = ", ".join([p.name for p in places])

        return f'''"""
Data Commons Query: {var_names} for {place_names}
Using the Data Commons Python API (mirrors MCP server tools)

API Documentation: https://docs.datacommons.org/api/python/
"""

import pandas as pd

# Method 1: Using datacommons_pandas (recommended for DataFrames)
try:
    import datacommons_pandas as dcpd

    df = dcpd.build_time_series_dataframe(
        places={entity_dcids!r},
        stat_vars={variable_dcids!r}
    )

    print("=== Data Commons Query Results ===")
    print(f"Variables: {var_names}")
    print(f"Places: {place_names}")
    print(f"Date filter: {date_param}")
    print()
    print(df.to_string())

except ImportError:
    print("datacommons_pandas not installed. Using REST API instead.")
    df = None

# Method 2: Using the REST API directly (fallback)
if df is None or df.empty:
    import requests

    url = "https://api.datacommons.org/v2/observation"
    params = {{
        "entity.dcids": {entity_dcids!r},
        "variable.dcids": {variable_dcids!r},
        "select": ["entity", "variable", "value", "date"],
        "date": "{date_param}",
    }}

    response = requests.get(url, params=params)
    data = response.json()

    records = []
    facets = data.get("facets", {{}})

    for var_dcid, var_data in data.get("byVariable", {{}}).items():
        for entity_dcid, entity_data in var_data.get("byEntity", {{}}).items():
            for facet in entity_data.get("orderedFacets", []):
                facet_id = facet.get("facetId", "")
                facet_info = facets.get(facet_id, {{}})

                for obs in facet.get("observations", []):
                    records.append({{
                        "entity": entity_dcid,
                        "variable": var_dcid,
                        "date": obs.get("date", ""),
                        "value": obs.get("value"),
                        "source": facet_info.get("provenanceUrl", ""),
                    }})

    df = pd.DataFrame(records)

    if not df.empty:
        print("=== Data Commons Query Results ===")
        print(df.to_string())
    else:
        print("No data found for the specified query.")

if df is not None and not df.empty:
    print()
    print("=== Summary Statistics ===")
    if "date" in df.columns:
        df_sorted = df.sort_values("date", ascending=False)
        latest = df_sorted.iloc[0]
        print(f"Most recent observation:")
        print(f"  Date: {{latest.get('date', 'N/A')}}")
        print(f"  Value: {{latest.get('value', 'N/A'):,}}")
'''
