"""Data Commons API client for knowledge graph queries."""

import hashlib
from typing import Any

import httpx

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.connectors import register_closeable

logger = get_logger(__name__)


class DataCommonsClient:
    """Client for interacting with the Data Commons API.

    Data Commons provides access to a unified knowledge graph of
    statistical data from various sources including federal agencies.

    API Documentation: https://docs.datacommons.org/api/
    """

    def __init__(self, api_url: str | None = None) -> None:
        """Initialize the Data Commons client.

        Args:
            api_url: Base URL for Data Commons API. Defaults to settings.
        """
        self.api_url = api_url or settings.data_commons_api_url
        self._client: httpx.AsyncClient | None = None
        self.logger = get_logger("data_commons_client")
        register_closeable(self)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}

            # Add API key if configured
            api_key = settings.data_commons_api_key.get_secret_value()
            if api_key:
                headers["X-API-Key"] = api_key

            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                timeout=30.0,
                headers=headers,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    def _add_api_key(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add API key to request parameters if configured.

        Args:
            params: Request parameters dictionary

        Returns:
            Parameters dict with API key added if available
        """
        api_key = settings.data_commons_api_key.get_secret_value()
        if api_key and api_key != "your-data-commons-api-key-here":
            params["key"] = api_key
            # Log a short non-reversible fingerprint, never the key material itself (issue #99).
            key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:8]
            self.logger.debug("Added API key to request params", key_fingerprint=key_fingerprint)
        else:
            self.logger.warning("No valid Data Commons API key configured")
        return params

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_property_values(
        self,
        dcids: list[str],
        property_name: str,
    ) -> dict[str, list[Any]]:
        """Get property values for entities.

        Args:
            dcids: List of Data Commons IDs
            property_name: Property to retrieve

        Returns:
            Dict mapping dcids to their property values
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/v2/node",
                json={
                    "nodes": dcids,
                    "property": f"->{property_name}",
                },
            )
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            self.logger.error("Failed to get property values", error=str(e))
            return {}

    async def get_observations(
        self,
        entity_dcids: list[str],
        variable_dcids: list[str],
        date: str = "",
        select: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get statistical observations for entities and variables.

        Args:
            entity_dcids: List of place/entity DCIDs
            variable_dcids: List of statistical variable DCIDs
            date: Date filter - empty string for all dates, "latest", or specific date like "2020"
            select: Fields to return - defaults to ["entity", "variable", "value", "date"]

        Returns:
            Observation data from Data Commons (raw response)
        """
        client = await self._get_client()

        if select is None:
            select = ["entity", "variable", "value", "date"]

        params: dict[str, Any] = {
            "entity.dcids": entity_dcids,
            "variable.dcids": variable_dcids,
            "select": select,
        }

        # Only add date parameter if specified
        if date:
            params["date"] = date

        # Add API key to params
        params = self._add_api_key(params)

        try:
            response = await client.get("/v2/observation", params=params)
            response.raise_for_status()
            data = response.json()

            # Debug logging to see the actual response
            self.logger.debug(
                "Data Commons API response",
                response_keys=list(data.keys()) if data else [],
                has_byVariable="byVariable" in data,
                byVariable_keys=list(data.get("byVariable", {}).keys()) if data else [],
                response_preview=str(data)[:500] if data else "empty",
            )

            return data
        except Exception as e:
            self.logger.error(
                "Failed to get observations",
                error=str(e),
                entities=entity_dcids,
                variables=variable_dcids,
            )
            return {}

    async def get_observations_as_records(
        self,
        entity_dcids: list[str],
        variable_dcids: list[str],
        date: str = "latest",
    ) -> list[dict[str, Any]]:
        """Get observations as flat records (similar to to_observations_as_records).

        This is the preferred method for getting data ready for DataFrames.

        Args:
            entity_dcids: List of place/entity DCIDs
            variable_dcids: List of statistical variable DCIDs
            date: Date filter - "latest" to get most recent for each entity/variable combo,
                  "" or None for all dates, or specific date like "2020"

        Returns:
            List of flat observation records
        """
        # Fetch all data first (don't use date=latest with the API)
        raw_response = await self.get_observations(
            entity_dcids=entity_dcids,
            variable_dcids=variable_dcids,
            date="" if date == "latest" else date,
        )
        records = self._parse_observation_response(raw_response)

        # If "latest" was requested, filter to most recent observation per entity/variable
        if date == "latest" and records:
            # Group by entity and variable, keep only the most recent date
            latest_records = {}
            for record in records:
                key = (record.get("entity"), record.get("variable"))
                existing = latest_records.get(key)

                # Keep the record with the most recent date
                if existing is None or record.get("date", "") > existing.get("date", ""):
                    latest_records[key] = record

            return list(latest_records.values())

        return records

    async def fetch_available_variables(
        self,
        entity_dcids: list[str],
    ) -> list[str]:
        """Fetch statistical variables available for given entities.

        Args:
            entity_dcids: List of entity DCIDs

        Returns:
            List of available variable DCIDs
        """
        client = await self._get_client()

        params: dict[str, Any] = {
            "entity.dcids": entity_dcids,
            "select": ["entity", "variable"],
        }
        params = self._add_api_key(params)

        try:
            response = await client.get("/v2/observation", params=params)
            response.raise_for_status()
            data = response.json()

            # Extract all variable DCIDs from the response
            variables = set()
            by_variable = data.get("byVariable", {})
            for var_dcid in by_variable.keys():
                variables.add(var_dcid)

            return list(variables)

        except Exception as e:
            self.logger.error(
                "Failed to fetch available variables",
                error=str(e),
                entities=entity_dcids,
            )
            return []

    async def fetch_observations_by_entity_type(
        self,
        parent_entity: str,
        entity_type: str,
        variable_dcids: list[str],
        date: str = "latest",
    ) -> list[dict[str, Any]]:
        """Fetch observations for child entities of a parent by type.

        Example: Get population for all counties in California
            parent_entity="geoId/06", entity_type="County", variable_dcids=["Count_Person"]

        Args:
            parent_entity: Parent entity DCID (e.g., "geoId/06" for California)
            entity_type: Type of child entities (e.g., "County", "City")
            variable_dcids: Variables to fetch
            date: Date filter

        Returns:
            List of observation records
        """
        client = await self._get_client()

        # Use entity expression for child places
        entity_expression = f"{parent_entity}<-containedInPlace+{{typeOf:{entity_type}}}"

        params: dict[str, Any] = {
            "variable.dcids": variable_dcids,
            "entity.expression": entity_expression,
            "select": ["entity", "variable", "value", "date"],
            "date": date,
        }
        params = self._add_api_key(params)

        try:
            response = await client.get("/v2/observation", params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_observation_response(data)

        except Exception as e:
            self.logger.error(
                "Failed to fetch observations by entity type",
                error=str(e),
                parent=parent_entity,
                entity_type=entity_type,
            )
            return []

    async def get_observation_series(
        self,
        entity_dcid: str,
        variable_dcid: str,
        date: str = "all",
    ) -> list[dict[str, Any]]:
        """Get time series of observations.

        Args:
            entity_dcid: Place/entity DCID
            variable_dcid: Statistical variable DCID
            date: Date filter ("all", "latest", or specific date like "2020")

        Returns:
            List of observations over time
        """
        client = await self._get_client()

        params = {
            "entity.dcids": [entity_dcid],
            "variable.dcids": [variable_dcid],
            "select": ["entity", "variable", "value", "date"],
            "date": date,
        }
        params = self._add_api_key(params)

        try:
            response = await client.get("/v2/observation", params=params)
            response.raise_for_status()

            data = response.json()

            # Parse the V2 API response structure:
            # byVariable -> byEntity -> orderedFacets -> observations
            observations = self._parse_observation_response(data)

            # Sort by date
            return sorted(observations, key=lambda x: x.get("date", ""))

        except Exception as e:
            self.logger.error(
                "Failed to get observation series",
                error=str(e),
                entity=entity_dcid,
                variable=variable_dcid,
            )
            return []

    def _parse_observation_response(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse the Data Commons V2 API observation response.

        The V2 API response structure is:
        {
            "byVariable": {
                "VARIABLE_DCID": {
                    "byEntity": {
                        "ENTITY_DCID": {
                            "orderedFacets": [
                                {
                                    "facetId": "...",
                                    "observations": [
                                        {"date": "2020", "value": 123},
                                        ...
                                    ]
                                }
                            ]
                        }
                    }
                }
            },
            "facets": {...}
        }

        Args:
            data: Raw API response

        Returns:
            Flattened list of observation records
        """
        observations = []
        facets = data.get("facets", {})

        by_variable = data.get("byVariable", {})
        for var_dcid, var_data in by_variable.items():
            by_entity = var_data.get("byEntity", {})
            for entity_dcid, entity_data in by_entity.items():
                ordered_facets = entity_data.get("orderedFacets", [])
                for facet in ordered_facets:
                    facet_id = facet.get("facetId", "")
                    facet_info = facets.get(facet_id, {})

                    facet_observations = facet.get("observations", [])
                    for obs in facet_observations:
                        observations.append({
                            "entity": entity_dcid,
                            "variable": var_dcid,
                            "date": obs.get("date", ""),
                            "value": obs.get("value"),
                            "facetId": facet_id,
                            "importName": facet_info.get("importName"),
                            "provenanceUrl": facet_info.get("provenanceUrl"),
                            "measurementMethod": facet_info.get("measurementMethod"),
                            "observationPeriod": facet_info.get("observationPeriod"),
                        })

        return observations

    async def search_entities(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for entities by name or keyword.

        Args:
            query: Search query
            entity_types: Optional list of entity types to filter
            limit: Maximum results to return

        Returns:
            List of matching entities
        """
        client = await self._get_client()

        try:
            params: dict[str, Any] = {
                "query": query,
                "maxResults": limit,
            }
            if entity_types:
                params["types"] = entity_types

            # Add API key to params
            params = self._add_api_key(params)

            response = await client.get("/v2/search", params=params)
            response.raise_for_status()

            return response.json().get("entities", [])

        except Exception as e:
            self.logger.error(
                "Failed to search entities",
                error=str(e),
                query=query,
            )
            return []

    async def get_variable_info(
        self,
        variable_dcid: str,
    ) -> dict[str, Any] | None:
        """Get detailed information about a statistical variable.

        Args:
            variable_dcid: Variable DCID

        Returns:
            Variable metadata or None
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/v2/node",
                json={
                    "nodes": [variable_dcid],
                    "property": "->*",
                },
            )
            response.raise_for_status()

            data = response.json().get("data", {})
            return data.get(variable_dcid)

        except Exception as e:
            self.logger.error(
                "Failed to get variable info",
                error=str(e),
                variable=variable_dcid,
            )
            return None

    async def get_place_info(
        self,
        place_dcid: str,
    ) -> dict[str, Any] | None:
        """Get detailed information about a place.

        Args:
            place_dcid: Place DCID

        Returns:
            Place metadata or None
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/v2/node",
                json={
                    "nodes": [place_dcid],
                    "property": "->*",
                },
            )
            response.raise_for_status()

            data = response.json().get("data", {})
            return data.get(place_dcid)

        except Exception as e:
            self.logger.error(
                "Failed to get place info",
                error=str(e),
                place=place_dcid,
            )
            return None

    def generate_sparql_query(
        self,
        variable_dcid: str,
        place_dcid: str,
    ) -> str:
        """Generate a SPARQL query for Data Commons.

        Args:
            variable_dcid: Variable DCID
            place_dcid: Place DCID

        Returns:
            SPARQL query string
        """
        return f"""
SELECT ?date ?value
WHERE {{
  ?obs typeOf StatisticalVariable .
  ?obs dcid "{variable_dcid}" .
  ?obs observationAbout ?place .
  ?place dcid "{place_dcid}" .
  ?obs observationDate ?date .
  ?obs value ?value .
}}
ORDER BY DESC(?date)
LIMIT 10
"""
