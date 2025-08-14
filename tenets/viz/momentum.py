"""Momentum visualization module.

This module provides visualization capabilities for development momentum
and velocity metrics, including burndown charts, velocity trends, and
sprint analytics.
"""

from typing import Any, Dict, List, Optional

from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay


class MomentumVisualizer(BaseVisualizer):
    """Visualizer for momentum and velocity metrics.

    Creates visualizations for development velocity, sprint progress,
    and team momentum analytics.
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize momentum visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        super().__init__(chart_config, display_config)
        self.terminal_display = TerminalDisplay(display_config)

    def create_velocity_chart(
        self, velocity_data: List[Dict[str, Any]], show_trend: bool = True
    ) -> Dict[str, Any]:
        """Create velocity trend chart.

        Args:
            velocity_data: List of velocity data points
            show_trend: Whether to show trend line

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        labels = []
        velocity = []

        for point in velocity_data:
            labels.append(point.get("period", ""))
            velocity.append(point.get("velocity", 0))

        datasets = [
            {
                "label": "Velocity",
                "data": velocity,
                "borderColor": ColorPalette.DEFAULT[0],
                "backgroundColor": ColorPalette.DEFAULT[0] + "20",
                "fill": True,
            }
        ]

        # Add trend line if requested
        if show_trend and len(velocity) > 1:
            trend_values = self._calculate_trend_line(velocity)
            datasets.append(
                {
                    "label": "Trend",
                    "data": trend_values,
                    "borderColor": ColorPalette.DEFAULT[1],
                    "borderDash": [5, 5],
                    "fill": False,
                    "pointRadius": 0,
                }
            )

        config = ChartConfig(type=ChartType.LINE, title="Development Velocity")

        return self.create_chart(ChartType.LINE, {"labels": labels, "datasets": datasets}, config)

    def create_burndown_chart(self, burndown_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create sprint burndown chart.

        Args:
            burndown_data: Burndown data with ideal and actual lines

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        labels = burndown_data.get("dates", [])
        ideal = burndown_data.get("ideal_line", [])
        actual = burndown_data.get("actual_line", [])

        datasets = [
            {
                "label": "Ideal",
                "data": ideal,
                "borderColor": ColorPalette.DEFAULT[2],
                "borderDash": [10, 5],
                "fill": False,
            },
            {
                "label": "Actual",
                "data": actual,
                "borderColor": ColorPalette.DEFAULT[0],
                "backgroundColor": ColorPalette.DEFAULT[0] + "10",
                "fill": True,
            },
        ]

        # Add scope changes if present
        if "scope_changes" in burndown_data:
            datasets.append(
                {
                    "label": "Scope Changes",
                    "data": burndown_data["scope_changes"],
                    "type": "bar",
                    "backgroundColor": ColorPalette.SEVERITY["medium"] + "50",
                }
            )

        config = ChartConfig(type=ChartType.LINE, title="Sprint Burndown")

        return self.create_chart(ChartType.LINE, {"labels": labels, "datasets": datasets}, config)

    def create_sprint_comparison(self, sprint_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create sprint comparison chart.

        Args:
            sprint_data: List of sprint metrics

        Returns:
            Dict[str, Any]: Grouped bar chart configuration
        """
        labels = []
        planned = []
        completed = []
        carryover = []

        for sprint in sprint_data:
            labels.append(sprint.get("name", ""))
            planned.append(sprint.get("planned", 0))
            completed.append(sprint.get("completed", 0))
            carryover.append(sprint.get("carryover", 0))

        datasets = [
            {"label": "Planned", "data": planned, "backgroundColor": ColorPalette.DEFAULT[0]},
            {
                "label": "Completed",
                "data": completed,
                "backgroundColor": ColorPalette.HEALTH["good"],
            },
            {
                "label": "Carried Over",
                "data": carryover,
                "backgroundColor": ColorPalette.SEVERITY["medium"],
            },
        ]

        config = ChartConfig(type=ChartType.BAR, title="Sprint Comparison")

        return {
            "type": "bar",
            "data": {"labels": labels, "datasets": datasets},
            "options": self._get_chart_options(config),
        }

    def create_team_velocity_radar(self, team_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create team velocity radar chart.

        Args:
            team_metrics: Dictionary of metric name to value

        Returns:
            Dict[str, Any]: Radar chart configuration
        """
        # Normalize metrics to 0-100 scale
        labels = []
        values = []

        metric_max = {
            "velocity": 100,
            "predictability": 100,
            "quality": 100,
            "collaboration": 100,
            "innovation": 100,
            "delivery": 100,
        }

        for metric, value in team_metrics.items():
            labels.append(metric.replace("_", " ").title())
            max_val = metric_max.get(metric, 100)
            normalized = min(100, (value / max_val) * 100)
            values.append(normalized)

        datasets = [
            {
                "label": "Current Sprint",
                "data": values,
                "borderColor": ColorPalette.DEFAULT[0],
                "backgroundColor": ColorPalette.DEFAULT[0] + "40",
            }
        ]

        # Add previous sprint if available
        if "previous" in team_metrics:
            prev_values = []
            for metric in team_metrics["previous"]:
                max_val = metric_max.get(metric, 100)
                normalized = min(100, (team_metrics["previous"][metric] / max_val) * 100)
                prev_values.append(normalized)

            datasets.append(
                {
                    "label": "Previous Sprint",
                    "data": prev_values,
                    "borderColor": ColorPalette.DEFAULT[1],
                    "backgroundColor": ColorPalette.DEFAULT[1] + "20",
                }
            )

        config = ChartConfig(type=ChartType.RADAR, title="Team Performance Metrics")

        return self.create_chart(ChartType.RADAR, {"labels": labels, "datasets": datasets}, config)

    def create_cumulative_flow(self, flow_data: Dict[str, List[int]]) -> Dict[str, Any]:
        """Create cumulative flow diagram.

        Args:
            flow_data: Dictionary of status to daily counts

        Returns:
            Dict[str, Any]: Stacked area chart configuration
        """
        # Assume first list defines the time axis
        days = len(next(iter(flow_data.values())))
        labels = [f"Day {i + 1}" for i in range(days)]

        datasets = []
        colors = {
            "todo": ColorPalette.DEFAULT[2],
            "in_progress": ColorPalette.DEFAULT[0],
            "review": ColorPalette.DEFAULT[1],
            "done": ColorPalette.HEALTH["good"],
            "blocked": ColorPalette.SEVERITY["high"],
        }

        for status, values in flow_data.items():
            datasets.append(
                {
                    "label": status.replace("_", " ").title(),
                    "data": values,
                    "backgroundColor": colors.get(
                        status, ColorPalette.DEFAULT[len(datasets) % len(ColorPalette.DEFAULT)]
                    ),
                    "fill": True,
                }
            )

        config = ChartConfig(type=ChartType.LINE, title="Cumulative Flow Diagram")

        chart_config = self.create_chart(
            ChartType.LINE, {"labels": labels, "datasets": datasets}, config
        )

        # Make it stacked
        chart_config["options"]["scales"] = {"y": {"stacked": True}}

        return chart_config

    def create_productivity_gauge(self, productivity_score: float) -> Dict[str, Any]:
        """Create productivity gauge chart.

        Args:
            productivity_score: Productivity score (0-100)

        Returns:
            Dict[str, Any]: Gauge chart configuration
        """
        # Determine color based on score
        if productivity_score >= 80:
            color = ColorPalette.HEALTH["excellent"]
        elif productivity_score >= 60:
            color = ColorPalette.HEALTH["good"]
        elif productivity_score >= 40:
            color = ColorPalette.HEALTH["fair"]
        else:
            color = ColorPalette.SEVERITY["high"]

        config = ChartConfig(
            type=ChartType.GAUGE,
            title=f"Team Productivity: {productivity_score:.0f}%",
            colors=[color],
        )

        return self.create_chart(ChartType.GAUGE, {"value": productivity_score, "max": 100}, config)

    def display_terminal(self, momentum_data: Dict[str, Any], show_details: bool = True) -> None:
        """Display momentum analysis in terminal.

        Args:
            momentum_data: Momentum analysis data
            show_details: Whether to show detailed breakdown
        """
        # Display header
        self.terminal_display.display_header("Momentum Analysis", style="double")

        # Display current sprint summary
        if "current_sprint" in momentum_data:
            sprint = momentum_data["current_sprint"]
            summary_data = {
                "Sprint": sprint.get("name", "Current"),
                "Velocity": sprint.get("velocity", 0),
                "Completed": f"{sprint.get('completed', 0)}/{sprint.get('planned', 0)}",
                "Days Remaining": sprint.get("days_remaining", 0),
            }

            self.terminal_display.display_metrics(summary_data, title="Current Sprint")

        # Display velocity trend
        if "velocity_trend" in momentum_data:
            trend = momentum_data["velocity_trend"]
            trend_symbol = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            trend_color = "green" if trend > 0 else "red" if trend < 0 else "yellow"

            print(
                f"\nVelocity Trend: {self.terminal_display.colorize(trend_symbol, trend_color)} {abs(trend):.1f}%"
            )

        # Display team metrics
        if show_details and "team_metrics" in momentum_data:
            headers = ["Metric", "Value", "Target", "Status"]
            rows = []

            for metric in momentum_data["team_metrics"]:
                value = metric.get("value", 0)
                target = metric.get("target", 0)
                status = "✓" if value >= target else "✗"
                status_color = "green" if value >= target else "red"

                rows.append(
                    [
                        metric.get("name", ""),
                        self.format_number(value, precision=1),
                        self.format_number(target, precision=1),
                        self.terminal_display.colorize(status, status_color),
                    ]
                )

            self.terminal_display.display_table(headers, rows, title="Team Performance Metrics")

        # Display burndown status
        if "burndown" in momentum_data:
            burndown = momentum_data["burndown"]
            on_track = burndown.get("on_track", False)
            completion = burndown.get("completion_percentage", 0)

            status_text = "On Track" if on_track else "Behind Schedule"
            status_color = "green" if on_track else "red"

            print(f"\nBurndown Status: {self.terminal_display.colorize(status_text, status_color)}")
            print(f"Completion: {self.terminal_display.create_progress_bar(completion, 100)}")

        # Display recommendations
        if "recommendations" in momentum_data:
            self.terminal_display.display_list(
                momentum_data["recommendations"], title="Recommendations", style="numbered"
            )

    def create_contributor_velocity(
        self, contributor_data: List[Dict[str, Any]], limit: int = 10
    ) -> Dict[str, Any]:
        """Create contributor velocity chart.

        Args:
            contributor_data: List of contributor velocity data
            limit: Maximum contributors to show

        Returns:
            Dict[str, Any]: Bar chart configuration
        """
        # Sort by velocity
        sorted_data = sorted(contributor_data, key=lambda x: x.get("velocity", 0), reverse=True)[
            :limit
        ]

        labels = []
        velocity = []
        colors = []

        for contributor in sorted_data:
            labels.append(contributor.get("name", "Unknown"))
            velocity.append(contributor.get("velocity", 0))

            # Color based on trend
            trend = contributor.get("trend", "stable")
            if trend == "increasing":
                colors.append(ColorPalette.HEALTH["good"])
            elif trend == "decreasing":
                colors.append(ColorPalette.SEVERITY["medium"])
            else:
                colors.append(ColorPalette.DEFAULT[0])

        config = ChartConfig(
            type=ChartType.HORIZONTAL_BAR, title="Individual Velocity", colors=colors
        )

        return self.create_chart(
            ChartType.HORIZONTAL_BAR, {"labels": labels, "values": velocity}, config
        )

    def _calculate_trend_line(self, values: List[float]) -> List[float]:
        """Calculate trend line for values.

        Args:
            values: List of values

        Returns:
            List[float]: Trend line values
        """
        if len(values) < 2:
            return values

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return [y_mean] * n

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        return [intercept + slope * i for i in range(n)]
