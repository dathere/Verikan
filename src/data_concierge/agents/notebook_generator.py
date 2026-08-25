"""Notebook Generator Agent for creating reproducible Google Colab notebooks.

This module generates Google Colab-compatible Jupyter notebooks that capture
all the steps of the data analysis, making the results fully reproducible.

The generated notebooks include:
- Google Colab compatibility (badges, install cells)
- Step-by-step documentation
- Reproducible code for all data operations
- Clear explanations and citations
"""

import json
from datetime import datetime
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import NotebookOutput

logger = get_logger(__name__)


def _compile_safe_code_cell(code: str, label: str = "generated code") -> nbformat.NotebookNode:
    """A trace-derived code cell guaranteed not to break notebook execution.

    Tool-call codegen embeds retrieved content and LLM-shaped arguments, so a
    template bug or hostile content can produce code that does not parse — and
    one SyntaxError cell used to fail verification (#131) for the whole
    notebook. Code that does not compile ships commented out, preserved for
    provenance, instead of executable.
    """
    try:
        compile(code, "<notebook-cell>", "exec")
        return new_code_cell(code)
    except SyntaxError as e:
        logger.warning(
            "Generated notebook cell does not compile; shipping it commented out",
            label=label,
            error=str(e)[:200],
        )
        commented = "\n".join(f"# {line}" for line in code.splitlines())
        return new_code_cell(
            f"# {label}: the generated code below did not parse as valid Python\n"
            "# and is preserved for provenance only.\n" + commented
        )


class NotebookGeneratorAgent(BaseAgent):
    """Agent responsible for generating reproducible Google Colab notebooks.

    Creates complete .ipynb files that are:
    - Fully compatible with Google Colab (one-click execution)
    - Structured as step-by-step documentation
    - Self-contained with all dependencies installed
    - Reproducible with clear explanations at each step

    For CKAN queries, the notebook captures:
    - Semantic search queries to find datasets
    - Data loading from CKAN DataStore API
    - All analysis operations (filter, aggregate, etc.)
    - The final answer and reasoning

    For Data Commons queries, the notebook captures:
    - Entity extraction and mapping
    - Data Commons API calls
    - Statistical computations
    - Visualization generation
    """

    name = "notebook_generator"
    description = "Generates reproducible Google Colab notebooks"

    # Colab installation cell - ensures all dependencies are available
    COLAB_SETUP_CODE = """# ============================================================
# STEP 1: Environment Setup
# ============================================================
# This cell installs all required packages for Google Colab.
# If running locally, you can skip this cell if packages are installed.

# Check if running in Google Colab
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Running in Google Colab - Installing dependencies...")
    !pip install -q pandas numpy matplotlib seaborn requests
    print("✅ Dependencies installed successfully!")
else:
    print("💻 Running locally - assuming dependencies are installed")
    print("   If not, run: pip install pandas numpy matplotlib seaborn requests")
"""

    # Standard imports for notebooks
    STANDARD_IMPORTS = """# ============================================================
# STEP 2: Import Libraries
# ============================================================
# These are the core libraries used throughout this notebook.
# Each import serves a specific purpose in the data pipeline.

import requests      # For making HTTP requests to data APIs
import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical computations
from datetime import datetime  # For timestamp handling
import json          # For JSON parsing

# Visualization libraries (with graceful fallback)
try:
    import matplotlib.pyplot as plt  # For creating plots
    import seaborn as sns            # For statistical visualizations
    VISUALIZATION_AVAILABLE = True
    plt.style.use('seaborn-v0_8-whitegrid')
    print("📊 Visualization libraries loaded successfully")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization libraries not available")
    print("   Install with: pip install matplotlib seaborn")

# ============================================================
# STEP 3: Configuration
# ============================================================
# Data source configuration - modify these if needed

CKAN_URL = "{ckan_url}"

# Display configuration info
print(f"\\n📅 Notebook generated: {{datetime.now().isoformat()}}")
print(f"🔗 CKAN URL: {{CKAN_URL}}")
print(f"📌 Timestamp: {timestamp}")
"""

    # CKAN data fetching helper
    CKAN_HELPER_CODE = '''# ============================================================
# STEP 4: Helper Functions
# ============================================================
# These utility functions handle data fetching from the CKAN API.
# You can reuse these functions for your own data exploration.

def fetch_ckan_data(resource_id: str, limit: int = 10000, filters: dict = None) -> pd.DataFrame:
    """
    Fetch data from CKAN DataStore API.

    This function handles pagination automatically and returns all records
    up to the specified limit.

    Parameters:
    -----------
    resource_id : str
        The unique identifier for the CKAN resource (dataset)
    limit : int, default=10000
        Maximum number of records to fetch
    filters : dict, optional
        Field filters to apply (e.g., {"state": "CA"})

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the fetched records

    Example:
    --------
    >>> df = fetch_ckan_data("abc123", limit=1000, filters={"year": 2023})
    >>> print(f"Loaded {len(df)} records")
    """
    print(f"📥 Fetching data from resource: {resource_id}")

    all_records = []
    offset = 0
    batch_size = min(32000, limit)

    while offset < limit:
        params = {
            "resource_id": resource_id,
            "limit": min(batch_size, limit - offset),
            "offset": offset,
        }

        if filters:
            params["filters"] = filters

        response = requests.post(
            f"{CKAN_URL}/api/3/action/datastore_search",
            json=params,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            break

        result = response.json()
        if not result.get("success"):
            print(f"❌ CKAN error: {result.get('error')}")
            break

        records = result.get("result", {}).get("records", [])
        if not records:
            break

        all_records.extend(records)
        offset += len(records)

        total = result.get("result", {}).get("total", 0)
        print(f"   Progress: {len(all_records):,} / {total:,} records")

        if offset >= total:
            break

    df = pd.DataFrame(all_records)

    # Remove CKAN internal fields that aren't useful for analysis
    internal_cols = ["_id", "_full_text"]
    df = df.drop(columns=[c for c in internal_cols if c in df.columns], errors="ignore")

    print(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
    return df


def search_ckan_packages(query: str, rows: int = 10) -> list:
    """
    Search CKAN for datasets (packages) matching a query.

    Parameters:
    -----------
    query : str
        The search query (e.g., "employment statistics")
    rows : int, default=10
        Maximum number of results to return

    Returns:
    --------
    list
        A list of matching package dictionaries

    Example:
    --------
    >>> packages = search_ckan_packages("housing prices", rows=5)
    >>> for pkg in packages:
    ...     print(pkg['title'])
    """
    print(f"🔍 Searching for: '{query}'")

    response = requests.post(
        f"{CKAN_URL}/api/3/action/package_search",
        json={"q": query, "rows": rows},
        headers={"Content-Type": "application/json"},
    )

    if response.status_code != 200:
        print(f"❌ Search failed: {response.status_code}")
        return []

    result = response.json()
    if not result.get("success"):
        print(f"❌ Search error: {result.get('error')}")
        return []

    packages = result.get("result", {}).get("results", [])
    print(f"✅ Found {len(packages)} matching datasets")
    return packages


print("✅ Helper functions loaded successfully!")
print("   - fetch_ckan_data(resource_id, limit, filters)")
print("   - search_ckan_packages(query, rows)")
'''

    async def process(self, state: GraphState) -> GraphState:
        """Generate a reproducible Google Colab notebook.

        Creates a notebook that:
        - Is fully compatible with Google Colab (one-click run)
        - Includes step-by-step documentation
        - Has all dependencies auto-installed
        - Provides clear explanations at each step

        Args:
            state: Current graph state with execution trace

        Returns:
            Updated state with notebook output including Colab URL
        """
        self.logger.info("Generating Colab-compatible notebook")

        # Check data source
        data_source = state.get("data_source", "data_commons")

        # Create notebook with Colab-compatible metadata
        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
            # Colab-specific metadata
            "colab": {
                "name": f"Data Concierge - {state.get('query', 'Query')[:50]}",
                "provenance": [],
                "toc_visible": True,
            },
            # Custom metadata for tracking
            "data_concierge": {
                "version": "0.1.0",
                "generated": datetime.now().isoformat(),
                "query": state.get("query", ""),
                "data_source": data_source,
                "colab_compatible": True,
            },
        }

        cells = []

        # Cell 1: Title with Colab badge
        cells.append(self._create_title_cell(state))

        # Cell 2: Colab setup (install dependencies)
        cells.append(self._create_colab_setup_cell())

        # Cell 3: Imports
        cells.append(self._create_imports_cell(data_source))

        # Cell 4: Helper functions (for CKAN-based portals)
        if data_source in ("ckan", "wprdc"):
            cells.append(self._create_helper_functions_cell())

        # Generate cells based on data source and execution trace
        execution_trace = state.get("execution_trace", [])

        # Determine whether this query used the LLM agent (CKAN or MCP sources)
        from data_concierge.agents.supervisor import is_llm_graph_source

        if is_llm_graph_source(data_source):
            cells.extend(self._create_llm_agent_cells(state, execution_trace))
        else:
            # Original behavior for Data Commons
            has_code_cells = False
            for i, trace_entry in enumerate(execution_trace):
                trace_cells = self._create_trace_cells(trace_entry, i + 1)
                cells.extend(trace_cells)
                if trace_entry.get("code"):
                    has_code_cells = True

            if not has_code_cells:
                cells.extend(self._create_default_query_cells(state))

        # Final cells: Results, confidence methodology, and citations
        cells.append(self._create_results_cell(state))
        cells.append(self._create_confidence_methodology_cell(state))
        cells.append(self._create_citations_cell(state))

        nb.cells = cells

        # Convert to JSON
        notebook_json = json.loads(nbformat.writes(nb))

        # Generate filename
        query_slug = self._slugify(state.get("query", "query")[:30])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_concierge_{query_slug}_{timestamp}.ipynb"

        state["notebook"] = NotebookOutput(
            notebook_json=notebook_json,
            filename=filename,
            download_url=None,
        )

        # Add to execution trace
        state["execution_trace"].append(
            {
                "agent": self.name,
                "action": "generate_notebook",
                "filename": filename,
                "cell_count": len(cells),
            }
        )

        self.logger.info("Notebook generated", filename=filename, cells=len(cells))

        return state

    def _create_llm_agent_cells(
        self,
        state: GraphState,
        execution_trace: list[dict[str, Any]],
    ) -> list[nbformat.NotebookNode]:
        """Create notebook cells for any LLM-agent-driven source (CKAN or MCP).

        Dispatches to the MCP or CKAN cell builder based on whether the trace
        contains MCP tool calls (prefixed with ``mcp__``).
        """
        llm_entries = [
            e for e in execution_trace if e.get("agent") in ("llm_analyst", "data_finder")
        ]
        has_mcp_entries = any(str(e.get("action", "")).startswith("mcp__") for e in llm_entries)

        if has_mcp_entries:
            return self._create_mcp_cells(state, llm_entries)

        # Fallback: existing CKAN cell logic
        return self._create_ckan_cells(state, execution_trace)

    def _create_mcp_cells(
        self,
        state: GraphState,
        entries: list[dict[str, Any]],
    ) -> list[nbformat.NotebookNode]:
        """Create notebook cells for MCP tool calls.

        Renders each MCP tool call as a markdown description + code cell,
        then shows the actual data that was returned.
        """
        cells: list[nbformat.NotebookNode] = []

        # Section header
        cells.append(
            new_markdown_cell(
                "# Data Retrieval via MCP\n\n"
                "The following steps show how data was retrieved through the "
                "**U.S. Census Bureau MCP server**. Each step includes reproducible "
                "code you can run directly.\n\n---"
            )
        )

        # Filter to only MCP entries with code
        mcp_entries = [e for e in entries if str(e.get("action", "")).startswith("mcp__")]

        for i, entry in enumerate(mcp_entries, 1):
            action = entry.get("action", "")
            # Strip the mcp__{server_id}__ prefix to get the tool name
            parts = action.split("__", 2)
            tool_name = parts[2] if len(parts) == 3 else action
            server_name = parts[1].replace("-", " ").title() if len(parts) >= 2 else "MCP"

            args = entry.get("arguments", {})
            preview = entry.get("result_preview", "")
            code = entry.get("code", "")

            label = self._format_mcp_tool_name(tool_name)

            # Markdown description
            md = f"## Step {i}: {label}\n\n"
            md += f"**Source:** {server_name}\n\n"

            if tool_name == "list-datasets":
                md += "Lists all available Census Bureau datasets.\n"
            elif tool_name == "resolve-geography-fips":
                md += f"**Geography:** `{args.get('geography_name', '')}`\n"
                if args.get("summary_level"):
                    md += f"**Level:** `{args['summary_level']}`\n"
            elif tool_name == "fetch-dataset-geography":
                md += f"**Dataset:** `{args.get('dataset', '')}`\n"
                if args.get("year"):
                    md += f"**Year:** {args['year']}\n"
            elif tool_name == "fetch-aggregate-data":
                md += f"**Dataset:** `{args.get('dataset', '')}`\n"
                md += f"**Year:** {args.get('year', '')}\n"
                get_p = args.get("get", {})
                if get_p.get("group"):
                    md += f"**Table group:** `{get_p['group']}`\n"
                elif get_p.get("variables"):
                    md += f"**Variables:** `{', '.join(get_p['variables'])}`\n"
                if args.get("for"):
                    md += f"**Geography (for):** `{args['for']}`\n"
                if args.get("in"):
                    md += f"**Geography (in):** `{args['in']}`\n"
            elif tool_name == "search-data-tables":
                md += f"**Query:** `{args.get('label_query', args.get('data_table_id', ''))}`\n"
                if args.get("api_endpoint"):
                    md += f"**Endpoint:** `{args['api_endpoint']}`\n"

            if preview:
                # Show a trimmed result so the notebook contains the actual data
                trimmed = preview[:1500]
                md += f"\n**Data retrieved:**\n```\n{trimmed}\n```\n"

            # A failed exploratory call stays for provenance but must not be
            # an executable step (same rule as the CKAN trace cells).
            failed = preview.startswith(("Error", "HTTP ", "SQL error"))
            if failed:
                md = md.replace(
                    f"## Step {i}: {label}\n\n",
                    f"## Step {i}: {label} (exploratory attempt — failed)\n\n",
                    1,
                )

            cells.append(new_markdown_cell(md))

            if code:
                if failed:
                    commented = "\n".join(f"# {line}" for line in code.splitlines())
                    cells.append(
                        new_code_cell(
                            f"# Step {i}: {label} — failed exploratory attempt, "
                            "kept for provenance\n" + commented
                        )
                    )
                else:
                    cells.append(
                        _compile_safe_code_cell(f"# Step {i}: {label}\n\n{code}", f"Step {i}")
                    )

        if not mcp_entries:
            cells.append(
                new_markdown_cell(
                    "> No MCP tool calls were recorded. "
                    "The server may not have been connected when the query ran.\n"
                )
            )

        return cells

    def _create_ckan_cells(
        self,
        state: GraphState,
        execution_trace: list[dict[str, Any]],
    ) -> list[nbformat.NotebookNode]:
        """Create cells for CKAN agent execution.

        This generates notebook cells that capture all the tool calls
        made by the CKAN agent or the LLM analysis agent.
        """
        cells = []

        ckan_entries = [
            e for e in execution_trace if e.get("agent") in ["data_finder", "llm_analyst"]
        ]

        if ckan_entries:
            # Entries with code (from LLM agent or old format)
            code_entries = [e for e in ckan_entries if e.get("code")]
            if code_entries:
                cells.extend(self._create_cells_from_llm_trace(code_entries))
            else:
                cells.extend(self._create_cells_from_old_trace(ckan_entries))
        else:
            # Fallback: Create basic CKAN query cells
            cells.extend(self._create_default_ckan_cells(state))

        return cells

    def _create_cells_from_llm_trace(
        self,
        entries: list[dict[str, Any]],
    ) -> list[nbformat.NotebookNode]:
        """Create cells from LLM analysis agent trace entries.

        Each entry has 'action', 'arguments', 'result_preview', and 'code'.
        """
        cells = []

        cells.append(
            new_markdown_cell(
                "# Data Analysis Pipeline\n\n"
                "The following steps show how the data was searched, loaded, and "
                "analyzed. Each step includes executable code you can modify and re-run.\n\n---"
            )
        )

        for i, entry in enumerate(entries, 1):
            action = entry.get("action", "unknown")
            args = entry.get("arguments", {})
            preview = entry.get("result_preview", "")

            # An exploratory attempt that failed at analysis time (wrong
            # column, bad SQL) is kept for provenance but must not be an
            # executable step: reproduced verbatim it fails again, so the
            # notebook never runs top to bottom and verification (#131)
            # scores every answer as broken.
            failed = preview.startswith(("Error", "HTTP ", "SQL error"))

            # Markdown description
            label = self._format_action(action)
            md = f"## Step {i}: {label}\n\n"
            if failed:
                md = (
                    f"## Step {i}: {label} (exploratory attempt — failed)\n\n"
                    "This attempt errored during the analysis and was retried "
                    "differently in a later step. It is preserved for "
                    "provenance; its code is commented out so the notebook "
                    "still runs end to end.\n\n"
                )

            if action == "search_datasets":
                md += f"**Search query:** `{args.get('query', '')}`\n"
                if args.get("organization"):
                    md += f"**Organization:** `{args['organization']}`\n"
            elif action == "get_dataset_info":
                md += f"**Dataset:** `{args.get('dataset_id', '')}`\n"
            elif action == "load_resource_data":
                md += f"**Resource ID:** `{args.get('resource_id', '')}`\n"
                if args.get("limit"):
                    md += f"**Limit:** {args['limit']}\n"
                if args.get("filters"):
                    md += f"**Filters:** `{json.dumps(args['filters'])}`\n"
            elif action == "run_sql_query":
                md += f"**SQL:**\n```sql\n{args.get('sql', '')}\n```\n"

            if preview:
                md += f"\n**Result preview:**\n```\n{preview[:600]}\n```\n"

            cells.append(new_markdown_cell(md))

            # Code cell
            code = entry.get("code", "")
            if code:
                if failed:
                    commented = "\n".join(f"# {line}" for line in code.splitlines())
                    cells.append(
                        new_code_cell(
                            f"# Step {i}: {label} — failed exploratory attempt, "
                            "kept for provenance\n" + commented
                        )
                    )
                else:
                    cells.append(
                        _compile_safe_code_cell(f"# Step {i}: {label}\n\n{code}", f"Step {i}")
                    )

        return cells

    def _create_cells_from_old_trace(
        self,
        entries: list[dict[str, Any]],
    ) -> list[nbformat.NotebookNode]:
        """Create cells from the old execution trace format."""
        cells = []

        for i, entry in enumerate(entries, 1):
            action = entry.get("action", "unknown")

            # Create markdown explanation
            md_content = f"## Step {i}: {self._format_action(action)}\n\n"

            if action == "query_ckan":
                resources = entry.get("resources_found", [])
                if resources:
                    md_content += f"**Found {len(resources)} resources:**\n\n"
                    for j, res in enumerate(resources[:5], 1):
                        md_content += f"{j}. {res.get('resource_name', 'Unknown')} (score: {res.get('score', 0):.2f})\n"
                md_content += f"\n**Results:** {entry.get('result_count', 0)} observations\n"

            elif action == "ckan_vector_search":
                results = entry.get("search_results", [])
                if results:
                    md_content += f"**Vector Search Results:** {len(results)}\n\n"
                    for j, res in enumerate(results, 1):
                        md_content += f"{j}. {res.get('resource_name', 'Unknown')} (score: {res.get('score', 0):.2f})\n"

                loaded = entry.get("loaded_resource")
                if loaded:
                    md_content += f"\n**Loaded Resource:** `{loaded}`\n"

            cells.append(new_markdown_cell(md_content))

            # Add code cell if available
            code = entry.get("code", "")
            if code:
                cells.append(_compile_safe_code_cell(code))

        return cells

    def _create_default_ckan_cells(
        self,
        state: GraphState,
    ) -> list[nbformat.NotebookNode]:
        """Create default CKAN query cells when no detailed trace is available."""
        cells = []

        query = state.get("query", "")

        cells.append(
            new_markdown_cell("""## Data Query

This section contains code to query CKAN for data relevant to your question.
""")
        )

        code = f"""# Search for relevant datasets
query = {query!r}

print("Searching for relevant datasets...")
packages = search_ckan_packages(query, rows=5)

print(f"\\nFound {{len(packages)}} packages:")
for i, pkg in enumerate(packages, 1):
    print(f"{{i}}. {{pkg.get('title', pkg.get('name'))}}")
    resources = pkg.get('resources', [])
    print(f"   Resources: {{len(resources)}}")
    for res in resources[:2]:
        print(f"   - {{res.get('name', 'Unnamed')}} ({{res.get('format', 'unknown')}})")
"""
        cells.append(new_code_cell(code))

        # Check if we have retrieved data
        retrieved_data = state.get("retrieved_data")
        if retrieved_data and retrieved_data.observations:
            # Get resource info from first observation
            obs = retrieved_data.observations[0]
            if obs.variable.dcid.startswith("ckan:"):
                resource_id = obs.variable.dcid.replace("ckan:", "")

                load_code = f"""# Load the selected dataset
resource_id = "{resource_id}"

print(f"Loading data from resource: {{resource_id}}")
df = fetch_ckan_data(resource_id, limit=10000)

print(f"\\nLoaded {{len(df):,}} rows with {{len(df.columns)}} columns")
print(f"\\nColumns: {{', '.join(df.columns.tolist())}}")
print("\\n=== Sample Data ===")
print(df.head(10).to_string())

# Basic statistics
print("\\n=== Summary Statistics ===")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if numeric_cols:
    print(df[numeric_cols].describe().to_string())
"""
                cells.append(new_code_cell(load_code))

        return cells

    def _create_default_query_cells(self, state: GraphState) -> list[nbformat.NotebookNode]:
        """Create default Data Commons query cells when no execution trace code is available."""
        cells = []
        entities = state.get("entities")

        markdown_content = """## Data Query

This section contains code to query Data Commons for the data relevant to your question.
"""
        cells.append(new_markdown_cell(markdown_content))

        if entities and (entities.variables or entities.places):
            variable_dcids = (
                [v.dcid for v in entities.variables] if entities.variables else ["Count_Person"]
            )
            entity_dcids = [p.dcid for p in entities.places] if entities.places else ["country/USA"]

            var_names = (
                ", ".join([v.name for v in entities.variables])
                if entities.variables
                else "Population"
            )
            place_names = (
                ", ".join([p.name for p in entities.places]) if entities.places else "United States"
            )

            code = f"""# Query Data Commons for: {var_names} in {place_names}

# Variables to query
variable_dcids = {variable_dcids!r}

# Entities (places) to query
entity_dcids = {entity_dcids!r}

# REST API query
url = "https://api.datacommons.org/v2/observation"
params = {{
    "entity.dcids": entity_dcids,
    "variable.dcids": variable_dcids,
    "select": ["entity", "variable", "value", "date"],
    "date": "latest",
}}

response = requests.get(url, params=params)
data = response.json()

# Parse the nested V2 API response
records = []
facets = data.get("facets", {{}})

for var_dcid, var_data in data.get("byVariable", {{}}).items():
    for ent_dcid, ent_data in var_data.get("byEntity", {{}}).items():
        for facet in ent_data.get("orderedFacets", []):
            facet_id = facet.get("facetId", "")
            facet_info = facets.get(facet_id, {{}})
            for obs in facet.get("observations", []):
                records.append({{
                    "entity": ent_dcid,
                    "variable": var_dcid,
                    "date": obs.get("date", ""),
                    "value": obs.get("value"),
                    "source": facet_info.get("provenanceUrl", ""),
                }})

df = pd.DataFrame(records)

# Display results
if not df.empty:
    print("=== Query Results ===")
    print(f"Variables: {var_names}")
    print(f"Places: {place_names}")
    print()
    print(df.to_string())
else:
    print("No data found. Try different variables or entities.")
"""
        else:
            code = """# Generic Data Commons Query Example
variable_dcids = ["Count_Person"]
entity_dcids = ["country/USA"]

url = "https://api.datacommons.org/v2/observation"
params = {
    "entity.dcids": entity_dcids,
    "variable.dcids": variable_dcids,
    "select": ["entity", "variable", "value", "date"],
    "date": "latest",
}

response = requests.get(url, params=params)
data = response.json()
print(json.dumps(data, indent=2))
"""

        cells.append(new_code_cell(code))
        return cells

    def _create_title_cell(self, state: GraphState) -> nbformat.NotebookNode:
        """Create the title markdown cell with Colab badge.

        The markdown is an admin-editable template (``gateway/system_prompt``);
        a custom template that fails to render falls back to the shipped
        default so a bad override can never break notebook generation.
        """
        from data_concierge.gateway.system_prompt import (
            DEFAULT_NOTEBOOK_HEADER_TEMPLATE,
            get_notebook_header_template,
        )

        query = state.get("query", "Unknown query")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_source = state.get("data_source", "data_commons")

        # Build data sources list
        sources = []
        retrieved_data = state.get("retrieved_data")
        if retrieved_data:
            sources = [s.name for s in retrieved_data.source_info]

        fields = {
            "query": query,
            "timestamp": timestamp,
            "data_source": data_source.upper(),
            "sources": ", ".join(sources) if sources else "N/A",
        }
        try:
            content = get_notebook_header_template().format(**fields)
        except (KeyError, IndexError, ValueError) as e:
            self.logger.warning(
                "Custom notebook header failed to render; using default", error=str(e)
            )
            content = DEFAULT_NOTEBOOK_HEADER_TEMPLATE.format(**fields)
        return new_markdown_cell(content)

    def _create_colab_setup_cell(self) -> nbformat.NotebookNode:
        """Create the Colab setup/installation cell."""
        return new_code_cell(self.COLAB_SETUP_CODE)

    def _create_imports_cell(self, data_source: str = "data_commons") -> nbformat.NotebookNode:
        """Create the imports code cell."""
        # Use the correct portal URL based on data source
        if data_source == "wprdc":
            ckan_url = settings.wprdc_ckan_url
        else:
            ckan_url = settings.ckan_url

        code = self.STANDARD_IMPORTS.format(
            timestamp=datetime.now().isoformat(),
            ckan_url=ckan_url,
        )
        return new_code_cell(code)

    def _create_helper_functions_cell(self) -> nbformat.NotebookNode:
        """Create helper functions cell for CKAN queries."""
        return new_code_cell(self.CKAN_HELPER_CODE)

    def _create_trace_cells(
        self,
        trace_entry: dict[str, Any],
        step_num: int,
    ) -> list[nbformat.NotebookNode]:
        """Create cells from an execution trace entry."""
        cells = []
        agent = trace_entry.get("agent", "unknown")
        action = trace_entry.get("action", "unknown")

        md_content = f"""## Step {step_num}: {self._format_action(action)}

**Agent**: {agent}
"""

        if action == "parse_query":
            entities = trace_entry.get("entities", {})
            md_content += f"""
**Extracted Entities**:
- Places: {[p.get("name", p.get("dcid", "Unknown")) for p in entities.get("places", [])]}
- Variables: {[v.get("name", v.get("dcid", "Unknown")) for v in entities.get("variables", [])]}
- Time Periods: {[t.get("start_date", "Unknown") for t in entities.get("times", [])]}
- Demographics: {[d.get("value", "Unknown") for d in entities.get("demographics", [])]}

**Classification**:
- Intent: {trace_entry.get("intent", "Unknown")}
- Tier: {trace_entry.get("tier", "Unknown")}
- Confidence: {trace_entry.get("confidence", 0):.2f}
"""

        elif action == "query_data_commons":
            api_call = trace_entry.get("api_call", {})
            md_content += f"""
**API Call**:
- Endpoint: `{api_call.get("endpoint", "N/A")}`
- Method: {api_call.get("method", "GET")}
- Results: {trace_entry.get("result_count", 0)} observations
"""

        elif action == "query_ckan":
            resources = trace_entry.get("resources_found", [])
            if resources:
                md_content += f"\n**Resources Found**: {len(resources)}\n\n"
                for i, res in enumerate(resources[:3], 1):
                    md_content += f"{i}. {res.get('resource_name', 'Unknown')} (score: {res.get('score', 0):.2f})\n"
            md_content += f"\n**Results**: {trace_entry.get('result_count', 0)} observations\n"

        elif action == "ckan_vector_search":
            results = trace_entry.get("search_results", [])
            if results:
                md_content += f"\n**Vector Search Results**: {len(results)}\n\n"
                for i, res in enumerate(results, 1):
                    md_content += f"{i}. {res.get('resource_name', 'Unknown')} (score: {res.get('score', 0):.2f})\n"
                    md_content += f"   - Resource ID: `{res.get('resource_id', 'N/A')}`\n"
            loaded = trace_entry.get("loaded_resource")
            if loaded:
                md_content += f"\n**Loaded Resource**: `{loaded}`\n"

        elif action == "compute_statistics":
            md_content += f"""
**Computation**:
- Type: {trace_entry.get("computation_type", "Unknown")}
- Input Count: {trace_entry.get("input_count", 0)}
"""

        elif action == "generate_visualization":
            md_content += f"""
**Visualization**:
- Chart Type: {trace_entry.get("chart_type", "Unknown")}
"""

        cells.append(new_markdown_cell(md_content))

        # Add code cell if available
        code = trace_entry.get("code", "")
        if code:
            cells.append(_compile_safe_code_cell(code))

        return cells

    def _create_results_cell(self, state: GraphState) -> nbformat.NotebookNode:
        """Create the results markdown cell with step-by-step explanation.

        The surrounding markdown is an admin-editable template; the confidence
        table is computed here and injected via ``{confidence_block}``.
        """
        from data_concierge.gateway.system_prompt import (
            DEFAULT_NOTEBOOK_RESULTS_TEMPLATE,
            get_notebook_results_template,
        )

        answer = state.get("answer", "No answer generated")
        confidence = state.get("confidence")

        if confidence:
            # Calculate confidence level
            final = confidence.final_score
            if confidence.measured_weight <= 0.0:
                level = "⚪ NOT MEASURED"
            else:
                level = "🟢 HIGH" if final >= 0.75 else "🟡 MEDIUM" if final >= 0.5 else "🔴 LOW"

            # A factor that could not be computed is reported as such, with
            # the reason in place of a percentage (issue #132).
            def _cell(value: float | None, key: str) -> str:
                if value is None:
                    why = confidence.unavailable.get(key, "not measured")
                    return f"— | {why}"
                return f"{value:.0%} | "

            rows = [
                (
                    "Answer Grounding",
                    confidence.answer_grounding,
                    "answer_grounding",
                    "How well the answer is backed by retrieved data",
                ),
                (
                    "Data Retrieval Quality",
                    confidence.data_retrieval_quality,
                    "data_retrieval_quality",
                    "Quality and breadth of data retrieval",
                ),
                (
                    "Source Metadata Quality",
                    confidence.source_metadata_quality,
                    "source_metadata_quality",
                    "Freshness and richness of data sources",
                ),
                (
                    "Query-Answer Alignment",
                    confidence.query_answer_alignment,
                    "query_answer_alignment",
                    "How well the answer addresses the query",
                ),
                (
                    "Computation Complexity",
                    confidence.computation_complexity,
                    "computation_complexity",
                    "Simplicity and reliability of analysis steps",
                ),
            ]
            table_rows = "\n".join(
                f"| {label} | {'—' if value is None else format(value, '.0%')} | "
                f"{confidence.unavailable.get(key, desc) if value is None else desc} |"
                for label, value, key, desc in rows
            )

            caveat = ""
            if confidence.explanation:
                caveat = f"\n> ⚠️ {confidence.explanation}\n"

            confidence_block = f"""{caveat}
| Factor | Score | Description |
|--------|-------|-------------|
{table_rows}
| **Final Score** | **{confidence.final_score:.0%}** | **{level}** |

### Interpretation Guide

- 🟢 **HIGH (75%+)**: Results are reliable and well-supported by data
- 🟡 **MEDIUM (50-74%)**: Results are reasonable but may need verification
- 🔴 **LOW (<50%)**: Results should be treated with caution
- ⚪ **NOT MEASURED**: None of the confidence factors could be computed — the
  score is absent, not zero
"""
        else:
            confidence_block = """
> ⚠️ Confidence score not available for this query.
"""

        fields = {"answer": answer, "confidence_block": confidence_block}
        try:
            content = get_notebook_results_template().format(**fields)
        except (KeyError, IndexError, ValueError) as e:
            self.logger.warning(
                "Custom notebook results template failed to render; using default", error=str(e)
            )
            content = DEFAULT_NOTEBOOK_RESULTS_TEMPLATE.format(**fields)

        return new_markdown_cell(content)

    def _create_confidence_methodology_cell(self, state: GraphState) -> nbformat.NotebookNode:
        """Create a markdown cell explaining the confidence scoring methodology."""
        content = """# ============================================================
# 📐 CONFIDENCE SCORING METHODOLOGY
# ============================================================

## How We Calculate Confidence

The Data Concierge uses a **weighted composite score** to assess the reliability
of each answer. The final confidence score is a weighted average of five independent
factors, each measuring a different aspect of answer quality.

### Scoring Formula

```
Final Score = (0.25 × Query Interpretation)
            + (0.25 × Source Authority)
            + (0.20 × Retrieval Match)
            + (0.15 × Data Recency)
            + (0.15 × Computation Reliability)
```

### Factor Descriptions

| Factor | Weight | What It Measures | How It's Calculated |
|--------|--------|------------------|---------------------|
| **Query Interpretation** | 25% | How well the system understood the query | Entity extraction confidence × intent classification confidence |
| **Source Authority** | 25% | Trustworthiness of the data source | Pre-assigned per source (BLS/Census: 0.95, Data Commons: 0.90, CKAN: 0.85) |
| **Retrieval Match** | 20% | How well the retrieved data matches the query | Retrieval score, boosted by observation count (up to 5 observations) |
| **Data Recency** | 15% | How fresh the data is | 1.0 if within expected update cycle, decays to 0.4 floor for older data |
| **Computation Reliability** | 15% | Accuracy of the computation method | By type: direct lookup 1.0, trend analysis 0.85, statistical inference 0.70 |

### Confidence Levels

| Level | Score Range | Interpretation |
|-------|-------------|----------------|
| 🟢 **HIGH** | ≥ 85% | Results are reliable and well-supported by authoritative data |
| 🟡 **MEDIUM** | 50% – 84% | Results are reasonable but may benefit from verification |
| 🔴 **LOW** | 25% – 49% | Results should be treated with caution; data may be incomplete |
| ⚫ **VERY LOW** | < 25% | Insufficient data; consider alternative sources or queries |

### Source Authority Ratings

| Data Source | Authority Score | Rationale |
|-------------|----------------|-----------|
| Bureau of Labor Statistics (BLS) | 0.95 | Official federal statistics, rigorous methodology |
| U.S. Census Bureau | 0.95 | Comprehensive national data collection |
| Bureau of Economic Analysis (BEA) | 0.95 | Official GDP and economic accounts |
| FRED (Federal Reserve) | 0.95 | Curated economic data from the Fed |
| Google Data Commons | 0.90 | Aggregated from authoritative sources |
| WPRDC (Pittsburgh) | 0.88 | Curated regional open data portal |
| Generic CKAN Portals | 0.85 | Quality varies by portal and dataset |

### Data Recency Decay

The recency score decays based on how old the data is relative to its expected
update frequency:

- **Within 1× update cycle**: 1.0 (fully current)
- **Within 2× update cycle**: 0.8
- **Within 4× update cycle**: 0.6
- **Older than 4× update cycle**: 0.4 (floor)

### Escalation Policy

When the final confidence score falls **below 50%** after **2 retrieval attempts**,
the system flags the query for human review rather than providing a potentially
unreliable answer.

---
"""
        return new_markdown_cell(content)

    def _create_citations_cell(self, state: GraphState) -> nbformat.NotebookNode:
        """Create the citations markdown cell with detailed reproducibility info."""
        citations = state.get("citations", [])

        content = """# ============================================================
# 📚 CITATIONS & REFERENCES
# ============================================================

## 📖 Data Sources

"""
        if not citations:
            content += "> No specific citations available for this query.\n\n"
        else:
            from data_concierge.agents.citation_builder import CitationBuilderAgent

            builder = CitationBuilderAgent()
            content += builder.format_for_notebook(citations)

        content += f"""
---

## 🔄 Reproducibility Guide

This notebook was automatically generated by **Verikan**.
Follow these steps to reproduce or extend the analysis:

### Prerequisites

```bash
pip install pandas numpy requests matplotlib seaborn
```

### Running the Notebook

| Step | Action | Notes |
|------|--------|-------|
| 1 | **Open in Colab** | Click the "Open in Colab" badge at the top |
| 2 | **Run All Cells** | `Runtime` → `Run all` or `Ctrl+F9` |
| 3 | **Wait for completion** | Dependencies install automatically in Colab |
| 4 | **Review results** | Scroll down to see the analysis results |

### ⚠️ Important Notes

- **Data freshness**: Results may differ if data sources have been updated since generation
- **API limits**: Some data sources have rate limits; wait if you encounter errors
- **Modifications**: Feel free to modify parameters and re-run cells to explore further

### 📅 Generation Info

- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Query**: {state.get("query", "N/A")}
- **Data Source**: {state.get("data_source", "N/A").upper()}

---

*Generated by Verikan*
"""
        return new_markdown_cell(content)

    def _format_mcp_tool_name(self, tool_name: str) -> str:
        """Human-readable label for an MCP tool name."""
        labels = {
            "list-datasets": "List Available Census Datasets",
            "fetch-dataset-geography": "Fetch Dataset Geography Levels",
            "fetch-aggregate-data": "Fetch Aggregate Census Data",
            "resolve-geography-fips": "Resolve Geography to FIPS Code",
            "search-data-tables": "Search Census Data Tables",
        }
        return labels.get(tool_name, tool_name.replace("-", " ").title())

    def _format_action(self, action: str) -> str:
        """Format action name for display."""
        action_labels = {
            "parse_query": "Query Parsing & Entity Extraction",
            "query_data_commons": "Data Retrieval from Data Commons",
            "query_ckan": "CKAN Data Query",
            "ckan_vector_search": "CKAN Semantic Search",
            "search_data": "Dataset Search",
            "search_datasets": "Search for Datasets",
            "get_dataset_info": "Dataset Details",
            "load_resource_data": "Load Data from Resource",
            "run_sql_query": "SQL Analysis Query",
            "load_and_explore": "Data Loading",
            "analyze_data": "Data Analysis",
            "compute_statistics": "Statistical Computation",
            "generate_visualization": "Visualization Generation",
            "build_citations": "Citation Building",
        }
        return action_labels.get(action, action.replace("_", " ").title())

    def _format_tool_name(self, tool_name: str) -> str:
        """Format tool name for display."""
        tool_labels = {
            "search_data": "Dataset Search (Onboarded Index)",
            "load_and_explore": "Load & Explore Data",
            "analyze_data": "Data Analysis",
            "query_ckan_metadata": "CKAN Metadata Query",
            "create_simple_chart": "Create Chart",
            "create_custom_visualization": "Custom Visualization",
        }
        return tool_labels.get(tool_name, tool_name.replace("_", " ").title())

    def _slugify(self, text: str) -> str:
        """Convert text to a filename-safe slug."""
        import re

        slug = text.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "_", slug)
        slug = slug.strip("_")
        return slug or "query"


def generate_notebook_from_trace(
    trace: Any,
    query: str,
    answer: str,
) -> NotebookOutput:
    """Generate a notebook directly from a CKANAgentSystem execution trace.

    This is a standalone function that can be used outside the LangGraph workflow.

    Args:
        trace: AgentExecutionTrace from CKANAgentSystem
        query: Original user query
        answer: Final answer

    Returns:
        NotebookOutput with the generated notebook
    """

    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "data_concierge": {
            "version": "0.1.0",
            "generated": datetime.now().isoformat(),
            "query": query,
            "data_source": "ckan",
        },
    }

    cells = []

    # Title cell
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title_content = f"""# Verikan — Query Results

## Query
> {query}

## Metadata
- **Generated**: {timestamp}
- **Data Source**: CKAN
- **Notebook Version**: 1.0

---
"""
    cells.append(new_markdown_cell(title_content))

    # Imports cell
    imports_code = NotebookGeneratorAgent.STANDARD_IMPORTS.format(
        timestamp=datetime.now().isoformat(),
        ckan_url=settings.ckan_url,
    )
    cells.append(new_code_cell(imports_code))

    # Helper functions
    cells.append(new_code_cell(NotebookGeneratorAgent.CKAN_HELPER_CODE))

    # Tool calls
    if hasattr(trace, "tool_calls") and trace.tool_calls:
        cells.append(
            new_markdown_cell("## Data Operations\n\nThe following operations were performed:")
        )

        for i, tool_call in enumerate(trace.tool_calls, 1):
            tool_name = tool_call.tool_name if hasattr(tool_call, "tool_name") else "unknown"
            code_snippet = tool_call.code_snippet if hasattr(tool_call, "code_snippet") else None

            cells.append(new_markdown_cell(f"### Step {i}: {tool_name.replace('_', ' ').title()}"))

            if code_snippet:
                cells.append(_compile_safe_code_cell(code_snippet))

    # Answer cell
    answer_content = f"""## Results

### Answer
{answer}

---

## Reproducibility Notes

Run all cells in order to reproduce these results.
Data may differ if sources have been updated since generation.
"""
    cells.append(new_markdown_cell(answer_content))

    nb.cells = cells

    # Generate filename
    import re

    slug = re.sub(r"[^\w\s-]", "", query.lower()[:30])
    slug = re.sub(r"[\s-]+", "_", slug).strip("_") or "query"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_concierge_{slug}_{timestamp_str}.ipynb"

    notebook_json = json.loads(nbformat.writes(nb))

    return NotebookOutput(
        notebook_json=notebook_json,
        filename=filename,
        download_url=None,
    )
