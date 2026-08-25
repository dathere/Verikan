"""MCP data connector - bridges MCP servers into the Data Concierge agent pipeline.

Provides high-level methods for using MCP tools in the context of data retrieval
and analysis, including code generation for reproducible notebooks.
"""

import json
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.mcp.models import MCPToolCallResult
from data_concierge.mcp.registry import MCPServerRegistry, get_mcp_registry

logger = get_logger(__name__)


class MCPDataConnector:
    """Connector that uses MCP server tools for data retrieval.

    Wraps the MCP registry to provide data-oriented methods used by
    the agent pipeline (DataFinder, LLM agent, etc.).
    """

    def __init__(self, registry: MCPServerRegistry | None = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> MCPServerRegistry:
        if self._registry is None:
            self._registry = get_mcp_registry()
        return self._registry

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get all available MCP tools across connected servers.

        Returns:
            List of tool descriptors with server info
        """
        return self.registry.get_all_tools()

    def get_connected_server_ids(self) -> list[str]:
        """Get IDs of all connected MCP servers."""
        return [s.config.id for s in self.registry.get_connected_servers()]

    async def call_tool(
        self, server_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        """Call a tool on a connected MCP server.

        Args:
            server_id: MCP server ID
            tool_name: Tool to call
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        return await self.registry.call_tool(server_id, tool_name, arguments)

    async def census_list_datasets(self) -> MCPToolCallResult:
        """List available Census Bureau datasets via MCP.

        Returns:
            Result containing dataset metadata
        """
        return await self.registry.call_tool("census-mcp", "list-datasets", {})

    async def census_fetch_geography(
        self, dataset: str, year: int | None = None
    ) -> MCPToolCallResult:
        """Fetch available geography levels for a Census dataset.

        Args:
            dataset: Dataset identifier (e.g., 'acs/acs1')
            year: Optional vintage year

        Returns:
            Available geography levels
        """
        args: dict[str, Any] = {"dataset": dataset}
        if year:
            args["year"] = year
        return await self.registry.call_tool("census-mcp", "fetch-dataset-geography", args)

    async def census_fetch_data(
        self,
        dataset: str,
        year: int,
        variables: list[str] | None = None,
        group: str | None = None,
        geo_for: str | None = None,
        geo_in: str | None = None,
        descriptive: bool = False,
    ) -> MCPToolCallResult:
        """Fetch aggregate data from the Census Bureau via MCP.

        Args:
            dataset: Dataset identifier (e.g., 'acs/acs1')
            year: Vintage year
            variables: List of variable codes
            group: Variable group
            geo_for: Geography level filter
            geo_in: Geography sub-filter
            descriptive: Add variable labels

        Returns:
            Census data
        """
        get_params: dict[str, Any] = {}
        if variables:
            get_params["variables"] = variables
        if group:
            get_params["group"] = group

        args: dict[str, Any] = {
            "dataset": dataset,
            "year": year,
            "get": get_params,
        }
        if geo_for:
            args["for"] = geo_for
        if geo_in:
            args["in"] = geo_in
        if descriptive:
            args["descriptive"] = True

        return await self.registry.call_tool("census-mcp", "fetch-aggregate-data", args)

    async def census_resolve_fips(
        self, geography_name: str, summary_level: str | None = None
    ) -> MCPToolCallResult:
        """Resolve a geography name to FIPS codes via MCP.

        Args:
            geography_name: Name to search (e.g., 'Philadelphia')
            summary_level: Optional level filter (e.g., 'Place', '160')

        Returns:
            Matching geographies with FIPS codes
        """
        args: dict[str, Any] = {"geography_name": geography_name}
        if summary_level:
            args["summary_level"] = summary_level
        return await self.registry.call_tool("census-mcp", "resolve-geography-fips", args)

    def get_tool_definitions_for_llm(self) -> list[dict[str, Any]]:
        """Get MCP tools formatted for LLM function calling.

        Returns tool definitions in Anthropic tool-use format so the
        LLM agent can decide which MCP tools to call.

        Returns:
            List of tool definitions
        """
        tools = []
        for tool_info in self.get_available_tools():
            tool = tool_info["tool"]
            server_id = tool_info["server_id"]
            server_name = tool_info["server_name"]

            # Build the input schema
            input_schema = (
                tool.input_schema.copy()
                if tool.input_schema
                else {
                    "type": "object",
                    "properties": {},
                }
            )

            tools.append(
                {
                    "name": f"mcp__{server_id}__{tool.name}",
                    "description": (f"[MCP: {server_name}] {tool.description}"),
                    "input_schema": input_schema,
                }
            )

        return tools

    async def handle_llm_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Handle an LLM tool call that targets an MCP tool.

        Parses the prefixed tool name (mcp__{server_id}__{tool_name})
        and dispatches to the correct MCP server.

        Args:
            tool_name: Prefixed tool name from LLM
            arguments: Tool arguments from LLM

        Returns:
            Text result from the tool
        """
        parts = tool_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError(f"Invalid MCP tool name format: {tool_name}")

        server_id = parts[1]
        actual_tool_name = parts[2]

        result = await self.registry.call_tool(server_id, actual_tool_name, arguments)

        if result.is_error:
            return f"Error calling {actual_tool_name}: {result.text}"

        return result.text

    def generate_reproducible_code(
        self,
        server_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_text: str,
    ) -> str:
        """Generate reproducible Python code for an MCP tool call.

        This is used by the notebook generator to create cells that
        show how to reproduce the data retrieval.

        Args:
            server_id: MCP server ID
            tool_name: Tool that was called
            arguments: Arguments used
            result_text: Result text from the call

        Returns:
            Python code string
        """
        # For Census MCP tools, generate Census API-specific code
        if server_id == "census-mcp":
            return self._generate_census_code(tool_name, arguments, result_text)

        # Generic MCP tool code. Everything is embedded via repr() so the
        # cell is ALWAYS valid Python regardless of what the arguments or
        # result contain (multi-line JSON, quotes, backslashes) — the old
        # f-string template commented only the first line of the arguments
        # JSON and pasted the raw result between triple quotes, which made
        # every MCP notebook fail execution with a SyntaxError (#131).
        # The verbatim result is embedded as data and parsed, so later cells
        # can compute from `result` instead of re-reading prose.
        truncated = len(result_text) > 8000
        embedded = result_text[:8000]
        truncation_note = (
            "# NOTE: the embedded result was truncated for notebook size; the full\n"
            "# response is available by re-running the tool against the MCP server.\n"
            if truncated
            else ""
        )
        return f"""# MCP tool call: {tool_name} (via {server_id})
# The verbatim result retrieved at analysis time is embedded below, so this
# notebook is self-contained and later cells can compute from `result`.
# Re-run the tool against the MCP server to refresh the data.
import json

mcp_tool = {tool_name!r}
arguments = {arguments!r}
{truncation_note}result_text = {embedded!r}

try:
    result = json.loads(result_text)
    print(json.dumps(result, indent=2)[:2000])
except ValueError:
    # Not JSON — keep the raw text available for inspection.
    result = result_text
    print(result_text[:2000])
"""

    def _generate_census_code(
        self, tool_name: str, arguments: dict[str, Any], result_text: str
    ) -> str:
        """Generate Census Bureau API code for notebook reproduction."""
        if tool_name == "fetch-aggregate-data":
            dataset = arguments.get("dataset", "acs/acs5")
            year = arguments.get("year", 2022)
            get_params = arguments.get("get", {})
            variables = get_params.get("variables", [])
            geo_for = arguments.get("for", "")
            geo_in = arguments.get("in", "")

            var_str = ",".join(variables) if variables else "NAME"
            code = f'''import requests
import pandas as pd

# Fetch data from the U.S. Census Bureau API
# Dataset: {dataset}, Year: {year}
BASE_URL = "https://api.census.gov/data/{year}/{dataset}"

params = {{
    "get": "{var_str}",
}}
'''
            if geo_for:
                code += f'params["for"] = "{geo_for}"\n'
            if geo_in:
                code += f'params["in"] = "{geo_in}"\n'

            code += """
# Add your Census API key for higher rate limits:
# params["key"] = "YOUR_CENSUS_API_KEY"

response = requests.get(BASE_URL, params=params)
response.raise_for_status()

data = response.json()
df = pd.DataFrame(data[1:], columns=data[0])
print(f"Retrieved {len(df)} rows")
df.head(10)
"""
            return code

        elif tool_name == "resolve-geography-fips":
            name = arguments.get("geography_name", "")
            return f'''# Resolve geography to FIPS code
# Searched for: {name}
# The Census Bureau uses FIPS codes to identify geographic areas.
# See: https://www.census.gov/library/reference/code-lists/ansi.html

print("""{result_text[:1000]}""")
'''

        elif tool_name == "list-datasets":
            return """# List available Census Bureau API datasets
# Full catalog: https://api.census.gov/data.html

import requests

response = requests.get("https://api.census.gov/data.json")
datasets = response.json().get("dataset", [])
print(f"Found {len(datasets)} datasets")

# Show a sample
for ds in datasets[:10]:
    title = ds.get("title", "Unknown")
    desc = ds.get("description", "")[:100]
    print(f"  - {title}: {desc}")
"""

        # Default
        return f'''# Census MCP Tool: {tool_name}
# Arguments: {json.dumps(arguments, indent=2)}
print("""{result_text[:2000]}""")
'''


# Singleton
_connector: MCPDataConnector | None = None


def get_mcp_connector() -> MCPDataConnector:
    """Get or create the MCP data connector singleton."""
    global _connector
    if _connector is None:
        _connector = MCPDataConnector()
    return _connector
