"""Viz Builder Agent for generating visualizations."""

from typing import Any

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.logging import get_logger
from data_concierge.core.models import QueryIntent, VisualizationSpec

logger = get_logger(__name__)


class VizBuilderAgent(BaseAgent):
    """Agent responsible for building visualizations.

    Generates Vega-Lite specifications for charts and maps
    based on the data and query type.
    """

    name = "viz_builder"
    description = "Generates Vega-Lite visualizations"

    # Chart type recommendations based on intent
    INTENT_CHART_MAP = {
        QueryIntent.FACTUAL_LOOKUP: "text",  # Just show the number
        QueryIntent.COMPARISON: "bar",
        QueryIntent.TREND_ANALYSIS: "line",
        QueryIntent.DATASET_DISCOVERY: None,  # No viz needed
        QueryIntent.METHODOLOGY_QUESTION: None,
        QueryIntent.DATA_LINKING: "scatter",
        QueryIntent.OUT_OF_SCOPE: None,
    }

    async def process(self, state: GraphState) -> GraphState:
        """Generate visualization for the data.

        Args:
            state: Current graph state with computed results

        Returns:
            Updated state with visualization spec
        """
        self.logger.info("Building visualization")

        intent = state.get("intent")
        computed_results = state.get("computed_results", {})
        retrieved_data = state.get("retrieved_data")

        # Determine if visualization is appropriate
        chart_type = self._select_chart_type(intent, computed_results)

        if chart_type is None or chart_type == "text":
            self.logger.debug("No visualization needed for this query type")
            return state

        # Generate the visualization
        viz_spec = self._generate_visualization(
            chart_type,
            computed_results,
            retrieved_data,
            state.get("query", ""),
        )

        if viz_spec:
            state["visualization"] = viz_spec

            # Add to execution trace
            state["execution_trace"].append({
                "agent": self.name,
                "action": "generate_visualization",
                "chart_type": chart_type,
                "code": self._generate_viz_code(viz_spec),
            })

            self.logger.info("Visualization generated", chart_type=chart_type)

        return state

    def _select_chart_type(
        self,
        intent: QueryIntent | None,
        computed_results: dict[str, Any],
    ) -> str | None:
        """Select appropriate chart type based on intent and data.

        Args:
            intent: Query intent
            computed_results: Computed statistical results

        Returns:
            Chart type string or None
        """
        if intent is None:
            return None

        # Get base recommendation from intent
        base_type = self.INTENT_CHART_MAP.get(intent)

        # Adjust based on data characteristics
        comp_type = computed_results.get("computation_type")

        if comp_type == "trend_analysis":
            return "line"
        elif comp_type == "comparison":
            ranking = computed_results.get("ranking", [])
            if len(ranking) > 10:
                return "bar"  # Horizontal bar for many items
            return "bar"
        elif comp_type == "direct_lookup":
            values = computed_results.get("values", [])
            if len(values) <= 1:
                return "text"  # Just the number
            return "bar"

        return base_type

    def _generate_visualization(
        self,
        chart_type: str,
        computed_results: dict[str, Any],
        retrieved_data: Any,
        query: str,
    ) -> VisualizationSpec | None:
        """Generate Vega-Lite specification.

        Args:
            chart_type: Type of chart to generate
            computed_results: Computed results with values
            retrieved_data: Original retrieved data
            query: Original query for title

        Returns:
            VisualizationSpec or None
        """
        if chart_type == "bar":
            return self._generate_bar_chart(computed_results, query)
        elif chart_type == "line":
            return self._generate_line_chart(computed_results, query)
        elif chart_type == "scatter":
            return self._generate_scatter_plot(computed_results, query)

        return None

    def _generate_bar_chart(
        self,
        computed_results: dict[str, Any],
        query: str,
    ) -> VisualizationSpec:
        """Generate a bar chart specification.

        Args:
            computed_results: Data for the chart
            query: Query for title

        Returns:
            VisualizationSpec for bar chart
        """
        # Build data for the chart
        ranking = computed_results.get("ranking", [])
        summary = computed_results.get("observations_summary", [])

        data_values = []
        if ranking:
            data_values = [
                {"category": r["place"], "value": r["value"]}
                for r in ranking
            ]
        elif summary:
            data_values = [
                {"category": s["place"], "value": s["value"]}
                for s in summary
            ]

        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": query[:50] + "..." if len(query) > 50 else query,
            "width": 400,
            "height": 300,
            "data": {"values": data_values},
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "category",
                    "type": "nominal",
                    "axis": {"title": None},
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "axis": {"title": "Value"},
                },
                "color": {
                    "field": "category",
                    "type": "nominal",
                    "legend": None,
                },
            },
        }

        alt_text = f"Bar chart showing {len(data_values)} values. "
        if ranking:
            alt_text += f"Highest: {ranking[0]['place']} ({ranking[0]['value']}). "
            alt_text += f"Lowest: {ranking[-1]['place']} ({ranking[-1]['value']})."

        return VisualizationSpec(
            chart_type="bar",
            vega_lite_spec=vega_spec,
            alt_text=alt_text,
        )

    def _generate_line_chart(
        self,
        computed_results: dict[str, Any],
        query: str,
    ) -> VisualizationSpec:
        """Generate a line chart specification.

        Args:
            computed_results: Data for the chart
            query: Query for title

        Returns:
            VisualizationSpec for line chart
        """
        time_series = computed_results.get("time_series", [])

        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": query[:50] + "..." if len(query) > 50 else query,
            "width": 500,
            "height": 300,
            "data": {"values": time_series},
            "mark": {"type": "line", "point": True},
            "encoding": {
                "x": {
                    "field": "date",
                    "type": "temporal",
                    "axis": {"title": "Date"},
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "axis": {"title": "Value"},
                },
            },
        }

        trend = computed_results.get("trend_direction", "unknown")
        pct_change = computed_results.get("percentage_change", 0)
        alt_text = (
            f"Line chart showing trend over time. "
            f"Trend is {trend} with {pct_change:.1f}% change."
        )

        return VisualizationSpec(
            chart_type="line",
            vega_lite_spec=vega_spec,
            alt_text=alt_text,
        )

    def _generate_scatter_plot(
        self,
        computed_results: dict[str, Any],
        query: str,
    ) -> VisualizationSpec:
        """Generate a scatter plot specification.

        Args:
            computed_results: Data for the chart
            query: Query for title

        Returns:
            VisualizationSpec for scatter plot
        """
        # Placeholder for scatter plot
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": query[:50] + "..." if len(query) > 50 else query,
            "width": 400,
            "height": 400,
            "data": {"values": []},
            "mark": "point",
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
            },
        }

        return VisualizationSpec(
            chart_type="scatter",
            vega_lite_spec=vega_spec,
            alt_text="Scatter plot (no data available)",
        )

    def _generate_viz_code(self, viz_spec: VisualizationSpec) -> str:
        """Generate Python code to create the visualization.

        Args:
            viz_spec: Visualization specification

        Returns:
            Python code string
        """
        return f'''import altair as alt
import pandas as pd

# Create visualization
spec = {viz_spec.vega_lite_spec}

# Convert to Altair chart
chart = alt.Chart.from_dict(spec)

# Display the chart
chart.display()

# Or save as HTML
# chart.save('chart.html')
'''
