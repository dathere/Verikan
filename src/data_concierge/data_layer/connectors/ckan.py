"""CKAN API client for data portal queries.

This client interfaces with CKAN data portals and uses Pinecone for semantic
search over pre-indexed CKAN resources.
"""

from typing import Any

import httpx
import pandas as pd

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.data_layer.connectors import register_closeable

logger = get_logger(__name__)


class CKANClient:
    """Client for interacting with CKAN data portals.

    Provides methods to:
    - Search for datasets and resources
    - Query metadata without loading full data
    - Load and retrieve resource data via DataStore API
    - Filter datasets by metadata criteria

    The client also integrates with Pinecone for semantic search over
    pre-indexed CKAN resources with AI-generated metadata.
    """

    def __init__(
        self,
        ckan_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the CKAN client.

        Args:
            ckan_url: Base URL for CKAN instance. Defaults to settings.
            api_key: Optional CKAN API key for authenticated access.
        """
        self.ckan_url = (ckan_url or settings.ckan_url).rstrip("/")
        self._api_key = api_key or (
            settings.ckan_api_key.get_secret_value()
            if hasattr(settings, 'ckan_api_key') and settings.ckan_api_key
            else None
        )
        self._client: httpx.AsyncClient | None = None
        self.logger = get_logger("ckan_client")
        register_closeable(self)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = self._api_key

            self._client = httpx.AsyncClient(
                base_url=self.ckan_url,
                timeout=60.0,
                headers=headers,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def action(self, action_name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a CKAN API action.

        Args:
            action_name: CKAN action name (e.g., 'package_search', 'datastore_search')
            params: Action parameters

        Returns:
            API response result
        """
        client = await self._get_client()

        try:
            response = await client.post(
                f"/api/3/action/{action_name}",
                json=params or {},
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", {}).get("message", "Unknown error")
                self.logger.error("CKAN action failed", action=action_name, error=error_msg)
                return {}

            return data.get("result", {})

        except httpx.HTTPStatusError as e:
            # Log 404s at debug level since they're expected for invalid resource IDs
            if e.response.status_code == 404:
                self.logger.debug(
                    "CKAN resource not found",
                    action=action_name,
                    status_code=404,
                    resource_id=params.get("resource_id") if params else None,
                )
            else:
                self.logger.error(
                    "CKAN HTTP error",
                    action=action_name,
                    status_code=e.response.status_code,
                    error=str(e),
                )
            return {}
        except Exception as e:
            self.logger.error("CKAN API error", action=action_name, error=str(e))
            return {}

    async def package_search(
        self,
        query: str,
        rows: int = 10,
        start: int = 0,
        sort: str = "score desc",
        filter_queries: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search for packages (datasets) in CKAN.

        Args:
            query: Search query (Solr query syntax)
            rows: Number of results to return
            start: Starting offset for pagination
            sort: Sort order
            filter_queries: Additional filter queries (fq)

        Returns:
            Search results with count and dataset list
        """
        params: dict[str, Any] = {
            "q": query,
            "rows": rows,
            "start": start,
            "sort": sort,
        }

        if filter_queries:
            params["fq_list"] = filter_queries

        return await self.action("package_search", params)

    async def package_show(self, package_id: str) -> dict[str, Any]:
        """Get detailed information about a specific package.

        Args:
            package_id: Package name or ID

        Returns:
            Package metadata including resources
        """
        return await self.action("package_show", {"id": package_id})

    async def resource_show(self, resource_id: str) -> dict[str, Any]:
        """Get detailed information about a specific resource.

        Args:
            resource_id: Resource ID

        Returns:
            Resource metadata
        """
        return await self.action("resource_show", {"id": resource_id})

    async def datastore_search(
        self,
        resource_id: str,
        q: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        fields: list[str] | None = None,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Search records within a DataStore resource.

        Args:
            resource_id: Resource ID to search
            q: Full-text search query
            filters: Field-specific filters
            limit: Maximum records to return
            offset: Starting offset
            fields: Specific fields to return
            sort: Sort specification

        Returns:
            DataStore search results with records and total count
        """
        params: dict[str, Any] = {
            "resource_id": resource_id,
            "limit": limit,
            "offset": offset,
        }

        if q:
            params["q"] = q
        if filters:
            params["filters"] = filters
        if fields:
            params["fields"] = fields
        if sort:
            params["sort"] = sort

        return await self.action("datastore_search", params)

    async def datastore_search_sql(self, sql: str) -> dict[str, Any]:
        """Execute SQL query against DataStore.

        Args:
            sql: SQL query string

        Returns:
            Query results with records
        """
        return await self.action("datastore_search_sql", {"sql": sql})

    async def get_resource_data(
        self,
        resource_id: str,
        limit: int = 100000,
        filters: dict[str, Any] | None = None,
    ) -> pd.DataFrame | None:
        """Load resource data as a pandas DataFrame.

        Args:
            resource_id: Resource ID to load
            limit: Maximum records to load
            filters: Optional filters to apply

        Returns:
            DataFrame with resource data or None if failed
        """
        try:
            all_records = []
            offset = 0
            batch_size = min(32000, limit)  # CKAN typically limits to 32000 per request

            while offset < limit:
                result = await self.datastore_search(
                    resource_id=resource_id,
                    limit=min(batch_size, limit - offset),
                    offset=offset,
                    filters=filters,
                )

                if not result:
                    break

                records = result.get("records", [])
                if not records:
                    break

                all_records.extend(records)
                offset += len(records)

                total = result.get("total", 0)
                if offset >= total:
                    break

            if not all_records:
                return None

            df = pd.DataFrame(all_records)

            # Remove CKAN internal fields
            internal_cols = ["_id", "_full_text"]
            df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors="ignore")

            self.logger.info(
                "Loaded CKAN resource data",
                resource_id=resource_id,
                rows=len(df),
                columns=len(df.columns),
            )

            return df

        except Exception as e:
            self.logger.error("Failed to load resource data", resource_id=resource_id, error=str(e))
            return None

    async def query_dataset_metadata(
        self,
        query_type: str,
        search_terms: str | None = None,
        dataset_name: str | None = None,
        organization: str | None = None,
        include_resources: bool = False,
        limit: int = 10,
    ) -> str:
        """Query CKAN metadata without loading full datasets.

        This is a fast operation for answering metadata questions.

        Args:
            query_type: Type of query ('search_count', 'dataset_info', 'list_all',
                       'search_datasets', 'organization_datasets')
            search_terms: Keywords to search for
            dataset_name: Specific dataset name/ID for info queries
            organization: Organization name for filtering
            include_resources: Whether to include resource details
            limit: Maximum results to return

        Returns:
            Formatted string with query results
        """
        self.logger.info(f"Metadata query: {query_type}, terms: {search_terms}, dataset: {dataset_name}")

        try:
            if query_type == "search_count":
                if not search_terms:
                    return "Error: search_terms required for search_count query"

                result = await self.package_search(search_terms, rows=0)
                count = result.get("count", 0)

                output = f"Found **{count:,} datasets** matching '{search_terms}'\n\n"

                if 0 < count <= 20:
                    examples_result = await self.package_search(search_terms, rows=min(5, count))
                    output += "Examples:\n"
                    for pkg in examples_result.get("results", []):
                        output += f"• {pkg.get('title', pkg.get('name'))}\n"

                return output

            elif query_type == "dataset_info":
                if not dataset_name:
                    return "Error: dataset_name required for dataset_info query"

                dataset = await self.package_show(dataset_name)
                if not dataset:
                    search_result = await self.package_search(dataset_name, rows=1)
                    if search_result.get("count", 0) == 0:
                        return f"No dataset found with name/ID '{dataset_name}'"
                    dataset = search_result["results"][0]

                output_parts = []
                output_parts.append(f"**Dataset: {dataset.get('title', dataset.get('name'))}**\n")

                if dataset.get("notes"):
                    desc = dataset["notes"][:200]
                    output_parts.append(f"**Description:** {desc}...")

                output_parts.append(f"**License:** {dataset.get('license_title', dataset.get('license_id', 'Not specified'))}")

                if dataset.get("author"):
                    output_parts.append(f"**Author:** {dataset['author']}")

                if dataset.get("maintainer"):
                    output_parts.append(f"**Maintainer:** {dataset['maintainer']}")

                if dataset.get("organization"):
                    org = dataset["organization"]
                    output_parts.append(f"**Organization:** {org.get('title', org.get('name'))}")

                if dataset.get("metadata_created"):
                    output_parts.append(f"**Created:** {dataset['metadata_created'][:10]}")

                if dataset.get("metadata_modified"):
                    output_parts.append(f"**Last Modified:** {dataset['metadata_modified'][:10]}")

                tags = dataset.get("tags", [])
                if tags:
                    tag_names = [tag["name"] if isinstance(tag, dict) else str(tag) for tag in tags[:10]]
                    output_parts.append(f"**Tags:** {', '.join(tag_names)}")

                resources = dataset.get("resources", [])
                output_parts.append(f"\n**Resources:** {len(resources)}")

                if include_resources and resources:
                    output_parts.append("\n**Resource Details:**")
                    for i, res in enumerate(resources[:5], 1):
                        output_parts.append(f"\n{i}. **{res.get('name', 'Unnamed')}**")
                        output_parts.append(f"   ID: `{res.get('id')}`")
                        output_parts.append(f"   Format: {res.get('format', 'unknown')}")
                        if res.get("size"):
                            size_mb = int(res["size"]) / (1024 * 1024)
                            output_parts.append(f"   Size: {size_mb:.2f} MB")

                return "\n".join(output_parts)

            elif query_type == "list_all":
                result = await self.package_search("*:*", rows=limit, sort="metadata_modified desc")

                datasets = result.get("results", [])
                total = result.get("count", 0)

                output = f"**Total Datasets:** {total:,}\n\n"
                output += f"**Showing {len(datasets)} most recently updated:**\n\n"

                for i, pkg in enumerate(datasets, 1):
                    output += f"{i}. **{pkg.get('title', pkg.get('name'))}**\n"
                    output += f"   ID: `{pkg.get('name')}`\n"
                    output += f"   Resources: {len(pkg.get('resources', []))}\n\n"

                return output

            elif query_type == "search_datasets":
                if not search_terms:
                    return "Error: search_terms required for search_datasets query"

                result = await self.package_search(search_terms, rows=limit)

                datasets = result.get("results", [])
                total = result.get("count", 0)

                output = f"Found **{total:,} datasets** matching '{search_terms}'\n\n"

                if datasets:
                    output += f"**Top {len(datasets)} Results:**\n\n"
                    for i, pkg in enumerate(datasets, 1):
                        output += f"{i}. **{pkg.get('title', pkg.get('name'))}**\n"
                        output += f"   ID: `{pkg.get('name')}`\n"
                        output += f"   Resources: {len(pkg.get('resources', []))}\n"

                        if pkg.get("notes"):
                            desc = pkg["notes"][:150]
                            output += f"   Description: {desc}...\n"

                        output += "\n"

                return output

            elif query_type == "organization_datasets":
                if not organization:
                    return "Error: organization required for organization_datasets query"

                org = await self.action("organization_show", {
                    "id": organization,
                    "include_datasets": True,
                })

                if not org:
                    return f"Error retrieving organization '{organization}'"

                datasets = org.get("packages", [])

                output = f"**Organization:** {org.get('title', org.get('name'))}\n\n"
                output += f"**Total Datasets:** {len(datasets)}\n\n"

                if datasets:
                    output += "**Datasets:**\n\n"
                    for i, pkg in enumerate(datasets[:limit], 1):
                        if isinstance(pkg, str):
                            output += f"{i}. {pkg}\n"
                            continue

                        output += f"{i}. **{pkg.get('title', pkg.get('name'))}**\n"
                        output += f"   ID: `{pkg.get('name')}`\n"
                        output += f"   Resources: {len(pkg.get('resources', []))}\n\n"

                return output

            else:
                return f"Unknown query_type: {query_type}"

        except Exception as e:
            self.logger.error(f"Metadata query error: {e}")
            return f"Error executing metadata query: {str(e)}"

    async def filter_datasets_by_metadata(
        self,
        search_query: str = "*:*",
        filter_queries: list[str] | None = None,
        date_range: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        organization: str | None = None,
        resource_format: str | None = None,
        sort: str = "metadata_modified desc",
        rows: int = 20,
    ) -> str:
        """Advanced filtering of datasets using CKAN's package_search API.

        Args:
            search_query: Main search query (Solr syntax)
            filter_queries: Additional filter queries
            date_range: Date range filter with 'field', 'start', 'end'
            tags: Filter by specific tags
            organization: Filter by organization
            resource_format: Filter by resource format (CSV, JSON, etc.)
            sort: Sort order
            rows: Number of results

        Returns:
            Formatted string with filtered results
        """
        self.logger.info(f"Advanced filter - query: {search_query}, filters: {filter_queries}")

        try:
            fq_list = list(filter_queries) if filter_queries else []

            if date_range and date_range.get("field"):
                field = date_range["field"]
                start = date_range.get("start", "*")
                end = date_range.get("end", "NOW")
                fq_list.append(f"{field}:[{start} TO {end}]")

            if tags:
                for tag in tags:
                    fq_list.append(f"tags:{tag}")

            if organization:
                fq_list.append(f"organization:{organization}")

            if resource_format:
                fq_list.append(f"res_format:{resource_format.upper()}")

            result = await self.package_search(
                search_query,
                rows=min(rows, 1000),
                sort=sort,
                filter_queries=fq_list if fq_list else None,
            )

            datasets = result.get("results", [])
            total = result.get("count", 0)

            output_parts = []
            output_parts.append(f"**Found {total:,} datasets matching criteria**\n")

            if fq_list or search_query != "*:*":
                output_parts.append("**Active Filters:**")
                if search_query != "*:*":
                    output_parts.append(f"  • Search: {search_query}")
                for fq in fq_list:
                    output_parts.append(f"  • {fq}")
                output_parts.append("")

            if datasets:
                output_parts.append(f"**Top {len(datasets)} Results:**\n")

                for i, pkg in enumerate(datasets, 1):
                    output_parts.append(f"**{i}. {pkg.get('title', pkg.get('name'))}**")
                    output_parts.append(f"   • ID: `{pkg.get('name')}`")
                    output_parts.append(f"   • Resources: {len(pkg.get('resources', []))}")

                    if pkg.get("organization"):
                        org_name = pkg["organization"].get("title") or pkg["organization"].get("name")
                        output_parts.append(f"   • Organization: {org_name}")

                    if pkg.get("tags"):
                        tag_names = [
                            t.get("name", str(t)) if isinstance(t, dict) else str(t)
                            for t in pkg["tags"][:5]
                        ]
                        if tag_names:
                            output_parts.append(f"   • Tags: {', '.join(tag_names)}")

                    if pkg.get("metadata_modified"):
                        modified = pkg["metadata_modified"][:10]
                        output_parts.append(f"   • Last Modified: {modified}")

                    if pkg.get("resources"):
                        formats = {res.get("format", "").upper() for res in pkg["resources"] if res.get("format")}
                        if formats:
                            output_parts.append(f"   • Formats: {', '.join(sorted(formats))}")

                    output_parts.append("")
            else:
                output_parts.append("*No datasets found matching the criteria.*")

            return "\n".join(output_parts)

        except Exception as e:
            self.logger.error(f"Filter datasets error: {e}", exc_info=True)
            return f"Error filtering datasets: {str(e)}"

    def generate_reproducible_code(
        self,
        resource_id: str,
        resource_name: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> str:
        """Generate reproducible Python code for CKAN data retrieval.

        This code can be used in Jupyter notebooks for reproducibility.

        Args:
            resource_id: Resource ID to query
            resource_name: Optional resource name for documentation
            filters: Optional filters applied
            limit: Row limit

        Returns:
            Python code string for notebook reproduction
        """
        resource_display = resource_name or resource_id
        filters_repr = repr(filters) if filters else "None"

        code = f'''"""
CKAN Data Retrieval: {resource_display}
Generated for reproducibility in Jupyter notebooks

This code retrieves data from CKAN DataStore API.
"""

import requests
import pandas as pd

# CKAN Configuration
CKAN_URL = "{self.ckan_url}"
RESOURCE_ID = "{resource_id}"

def fetch_ckan_data(resource_id: str, limit: int = {limit}, filters: dict = {filters_repr}) -> pd.DataFrame:
    """Fetch data from CKAN DataStore API.

    Args:
        resource_id: The CKAN resource ID
        limit: Maximum number of records to fetch
        filters: Optional field filters

    Returns:
        DataFrame with the resource data
    """
    all_records = []
    offset = 0
    batch_size = min(32000, limit)

    while offset < limit:
        params = {{
            "resource_id": resource_id,
            "limit": min(batch_size, limit - offset),
            "offset": offset,
        }}

        if filters:
            params["filters"] = filters

        response = requests.post(
            f"{{CKAN_URL}}/api/3/action/datastore_search",
            json=params,
            headers={{"Content-Type": "application/json"}},
        )

        if response.status_code != 200:
            print(f"Error: {{response.status_code}} - {{response.text}}")
            break

        result = response.json()
        if not result.get("success"):
            print(f"CKAN error: {{result.get('error')}}")
            break

        records = result.get("result", {{}}).get("records", [])
        if not records:
            break

        all_records.extend(records)
        offset += len(records)

        total = result.get("result", {{}}).get("total", 0)
        if offset >= total:
            break

    df = pd.DataFrame(all_records)

    # Remove CKAN internal fields
    internal_cols = ["_id", "_full_text"]
    df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors="ignore")

    return df

# Fetch the data
print(f"Fetching data from CKAN resource: {resource_display}")
print(f"Resource ID: {{RESOURCE_ID}}")
print(f"Limit: {limit} records")
{"print(f'Filters: {filters_repr}')" if filters else ""}

df = fetch_ckan_data(RESOURCE_ID)

print(f"\\nLoaded {{len(df):,}} records with {{len(df.columns)}} columns")
print(f"\\nColumns: {{', '.join(df.columns.tolist())}}")

# Display first few rows
print("\\n=== Sample Data ===")
print(df.head(10).to_string())

# Basic statistics for numeric columns
print("\\n=== Summary Statistics ===")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if numeric_cols:
    print(df[numeric_cols].describe().to_string())
else:
    print("No numeric columns found")
'''

        return code
