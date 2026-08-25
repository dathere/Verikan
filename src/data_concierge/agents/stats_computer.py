"""Stats Computer Agent for statistical calculations and analysis."""

from typing import Any

from data_concierge.agents.base import BaseAgent
from data_concierge.agents.state import GraphState
from data_concierge.core.logging import get_logger
from data_concierge.core.models import QueryIntent

logger = get_logger(__name__)


class StatsComputerAgent(BaseAgent):
    """Agent responsible for statistical computations.

    Performs calculations on retrieved data:
    - Aggregations (sum, average, median)
    - Comparisons between values
    - Trend analysis over time
    - Rankings
    """

    name = "stats_computer"
    description = "Performs statistical calculations and analysis"

    async def process(self, state: GraphState) -> GraphState:
        """Perform statistical computations on retrieved data.

        Args:
            state: Current graph state with retrieved data

        Returns:
            Updated state with computed results
        """
        self.logger.info("Computing statistics")

        retrieved_data = state.get("retrieved_data")
        intent = state.get("intent")

        if not retrieved_data or not retrieved_data.observations:
            self.logger.debug("No data to compute")
            state["computed_results"] = {}
            return state

        observations = retrieved_data.observations

        # Determine what computation to perform based on intent
        if intent == QueryIntent.FACTUAL_LOOKUP:
            results = self._compute_direct_lookup(observations)
        elif intent == QueryIntent.COMPARISON:
            results = self._compute_comparison(observations)
        elif intent == QueryIntent.TREND_ANALYSIS:
            results = self._compute_trend(observations)
        else:
            results = self._compute_direct_lookup(observations)

        state["computed_results"] = results

        # Add to execution trace
        state["execution_trace"].append({
            "agent": self.name,
            "action": "compute_statistics",
            "intent": intent.value if intent else "unknown",
            "input_count": len(observations),
            "computation_type": results.get("computation_type", "direct_lookup"),
            "code": self._generate_code_snippet(results),
        })

        self.logger.info(
            "Statistics computed",
            computation_type=results.get("computation_type"),
            result_count=len(results.get("values", [])),
        )

        return state

    def _compute_direct_lookup(
        self,
        observations: list[Any],
    ) -> dict[str, Any]:
        """Handle direct lookup - just return the value(s).

        Args:
            observations: List of statistical observations

        Returns:
            Computation results
        """
        if len(observations) == 1:
            obs = observations[0]
            return {
                "computation_type": "direct_lookup",
                "primary_value": obs.value,
                "variable": obs.variable.name,
                "place": obs.place.name,
                "date": obs.date,
                "unit": obs.variable.unit,
                "values": [obs.value],
            }

        # Multiple observations - list them all
        values = [obs.value for obs in observations]
        return {
            "computation_type": "direct_lookup",
            "primary_value": values[0] if values else None,
            "values": values,
            "observations_summary": [
                {
                    "variable": obs.variable.name,
                    "place": obs.place.name,
                    "value": obs.value,
                    "date": obs.date,
                }
                for obs in observations
            ],
        }

    def _compute_comparison(
        self,
        observations: list[Any],
    ) -> dict[str, Any]:
        """Compare values across observations.

        Args:
            observations: List of statistical observations

        Returns:
            Comparison results
        """
        if len(observations) < 2:
            return self._compute_direct_lookup(observations)

        # Sort by value
        sorted_obs = sorted(observations, key=lambda x: float(x.value), reverse=True)
        values = [float(obs.value) for obs in sorted_obs]

        # Calculate differences
        max_val = max(values)
        min_val = min(values)
        diff = max_val - min_val
        pct_diff = (diff / min_val * 100) if min_val != 0 else 0

        return {
            "computation_type": "comparison",
            "highest": {
                "value": sorted_obs[0].value,
                "place": sorted_obs[0].place.name,
            },
            "lowest": {
                "value": sorted_obs[-1].value,
                "place": sorted_obs[-1].place.name,
            },
            "difference": diff,
            "percentage_difference": round(pct_diff, 2),
            "ranking": [
                {"place": obs.place.name, "value": obs.value, "rank": i + 1}
                for i, obs in enumerate(sorted_obs)
            ],
            "values": values,
        }

    def _compute_trend(
        self,
        observations: list[Any],
    ) -> dict[str, Any]:
        """Analyze trend over time.

        Args:
            observations: List of statistical observations

        Returns:
            Trend analysis results
        """
        if len(observations) < 2:
            return self._compute_direct_lookup(observations)

        # Sort by date
        sorted_obs = sorted(observations, key=lambda x: x.date)
        values = [float(obs.value) for obs in sorted_obs]
        dates = [obs.date for obs in sorted_obs]

        # Calculate basic trend metrics
        first_value = values[0]
        last_value = values[-1]
        change = last_value - first_value
        pct_change = (change / first_value * 100) if first_value != 0 else 0

        # Determine trend direction
        if pct_change > 5:
            trend_direction = "increasing"
        elif pct_change < -5:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # Calculate average
        avg_value = sum(values) / len(values)

        return {
            "computation_type": "trend_analysis",
            "start_value": first_value,
            "end_value": last_value,
            "start_date": dates[0],
            "end_date": dates[-1],
            "absolute_change": round(change, 2),
            "percentage_change": round(pct_change, 2),
            "trend_direction": trend_direction,
            "average": round(avg_value, 2),
            "min": min(values),
            "max": max(values),
            "time_series": [
                {"date": d, "value": v} for d, v in zip(dates, values, strict=False)
            ],
            "values": values,
        }

    def _generate_code_snippet(self, results: dict[str, Any]) -> str:
        """Generate Python code snippet for the computation.

        Args:
            results: Computation results

        Returns:
            Python code as string
        """
        comp_type = results.get("computation_type", "direct_lookup")

        if comp_type == "direct_lookup":
            return '''# Get the most recent value
if not df.empty:
    df_sorted = df.sort_values("date", ascending=False)
    result = df_sorted.iloc[0]

    print(f"Most Recent Data:")
    print(f"  Date: {result['date']}")
    print(f"  Value: {result['value']:,.0f}")
    print(f"\\nHistorical range:")
    print(f"  Earliest: {df_sorted.iloc[-1]['date']} - {df_sorted.iloc[-1]['value']:,.0f}")
    print(f"  Latest: {result['date']} - {result['value']:,.0f}")
    print(f"  Total observations: {len(df)}")
else:
    print("No data available")
    result = None
'''

        elif comp_type == "comparison":
            return '''# Compare values across observations
df_sorted = df.sort_values('value', ascending=False)
highest = df_sorted.iloc[0]
lowest = df_sorted.iloc[-1]
difference = highest['value'] - lowest['value']
pct_diff = (difference / lowest['value']) * 100 if lowest['value'] != 0 else 0

print(f"Highest: {highest['place']} = {highest['value']}")
print(f"Lowest: {lowest['place']} = {lowest['value']}")
print(f"Difference: {difference:.2f} ({pct_diff:.1f}%)")
'''

        elif comp_type == "trend_analysis":
            return '''# Analyze trend over time
df_sorted = df.sort_values('date')
first_value = df_sorted.iloc[0]['value']
last_value = df_sorted.iloc[-1]['value']
change = last_value - first_value
pct_change = (change / first_value) * 100 if first_value != 0 else 0

trend = "increasing" if pct_change > 5 else "decreasing" if pct_change < -5 else "stable"
print(f"Trend: {trend}")
print(f"Change: {change:.2f} ({pct_change:.1f}%)")
print(f"Period: {df_sorted.iloc[0]['date']} to {df_sorted.iloc[-1]['date']}")
'''

        return "# Results computed"

    def calculate_aggregate(
        self,
        values: list[float],
        method: str = "mean",
    ) -> float:
        """Calculate aggregate statistic.

        Args:
            values: List of numeric values
            method: Aggregation method (mean, sum, median, min, max)

        Returns:
            Aggregated value
        """
        if not values:
            return 0.0

        if method == "mean":
            return sum(values) / len(values)
        elif method == "sum":
            return sum(values)
        elif method == "median":
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mid = n // 2
            if n % 2 == 0:
                return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
            return sorted_vals[mid]
        elif method == "min":
            return min(values)
        elif method == "max":
            return max(values)

        return sum(values) / len(values)

    def rank_values(
        self,
        observations: list[Any],
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """Rank observations by value.

        Args:
            observations: List of observations
            ascending: If True, rank from lowest to highest

        Returns:
            Ranked list with rank numbers
        """
        sorted_obs = sorted(
            observations,
            key=lambda x: float(x.value),
            reverse=not ascending,
        )

        return [
            {
                "rank": i + 1,
                "place": obs.place.name,
                "value": obs.value,
                "date": obs.date,
            }
            for i, obs in enumerate(sorted_obs)
        ]
