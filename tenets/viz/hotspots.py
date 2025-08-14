"""Hotspot visualization module.

This module provides visualization capabilities for code hotspots,
including change frequency, complexity hotspots, and risk areas.
"""

from typing import Any, Dict, List, Optional

from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay


class HotspotVisualizer(BaseVisualizer):
    """Visualizer for code hotspots.

    Creates visualizations for hotspot analysis including heatmaps,
    bubble charts, and risk matrices.
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize hotspot visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        super().__init__(chart_config, display_config)
        self.terminal_display = TerminalDisplay(display_config)

    def create_hotspot_heatmap(
        self,
        hotspot_data: List[Dict[str, Any]],
        metric_x: str = "change_frequency",
        metric_y: str = "complexity",
    ) -> Dict[str, Any]:
        """Create hotspot heatmap.

        Args:
            hotspot_data: List of files with hotspot metrics
            metric_x: X-axis metric
            metric_y: Y-axis metric

        Returns:
            Dict[str, Any]: Heatmap configuration
        """
        # Create grid for heatmap
        # Bin the data into grid cells
        x_values = [d.get(metric_x, 0) for d in hotspot_data]
        y_values = [d.get(metric_y, 0) for d in hotspot_data]

        if not x_values or not y_values:
            return {}

        # Create 10x10 grid
        grid_size = 10
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        x_step = (x_max - x_min) / grid_size if x_max > x_min else 1
        y_step = (y_max - y_min) / grid_size if y_max > y_min else 1

        # Initialize grid
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

        # Populate grid
        for d in hotspot_data:
            x_val = d.get(metric_x, 0)
            y_val = d.get(metric_y, 0)

            x_idx = min(int((x_val - x_min) / x_step), grid_size - 1) if x_step > 0 else 0
            y_idx = min(int((y_val - y_min) / y_step), grid_size - 1) if y_step > 0 else 0

            grid[y_idx][x_idx] += 1

        # Create labels
        x_labels = [f"{x_min + i * x_step:.1f}" for i in range(grid_size)]
        y_labels = [f"{y_min + i * y_step:.1f}" for i in range(grid_size)]

        config = ChartConfig(
            type=ChartType.HEATMAP,
            title=f"Hotspot Map: {metric_x.replace('_', ' ').title()} vs {metric_y.replace('_', ' ').title()}",
        )

        return self.create_chart(
            ChartType.HEATMAP, {"matrix": grid, "x_labels": x_labels, "y_labels": y_labels}, config
        )

    def create_hotspot_bubble(
        self, hotspot_data: List[Dict[str, Any]], limit: int = 50
    ) -> Dict[str, Any]:
        """Create hotspot bubble chart.

        Args:
            hotspot_data: List of files with hotspot metrics
            limit: Maximum bubbles to show

        Returns:
            Dict[str, Any]: Bubble chart configuration
        """
        # Sort by risk score
        sorted_data = sorted(hotspot_data, key=lambda x: x.get("risk_score", 0), reverse=True)[
            :limit
        ]

        points = []
        labels = []

        for item in sorted_data:
            complexity = item.get("complexity", 0)
            changes = item.get("change_frequency", 0)
            size = item.get("lines", 100)

            # Scale size for visualization
            bubble_size = min(50, 5 + (size / 100))

            points.append((complexity, changes, bubble_size))
            labels.append(item.get("file", "Unknown"))

        config = ChartConfig(
            type=ChartType.BUBBLE, title="Code Hotspots (Complexity vs Change Frequency)"
        )

        chart_config = self.create_chart(ChartType.BUBBLE, {"points": points}, config)

        # Customize axes
        chart_config["options"]["scales"] = {
            "x": {"title": {"display": True, "text": "Complexity"}},
            "y": {"title": {"display": True, "text": "Change Frequency"}},
        }

        return chart_config

    def create_risk_matrix(self, hotspot_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create risk matrix visualization.

        Args:
            hotspot_data: List of files with risk metrics

        Returns:
            Dict[str, Any]: Scatter plot as risk matrix
        """
        # Categorize by risk level
        risk_categories = {
            "low": {"points": [], "color": ColorPalette.HEALTH["excellent"]},
            "medium": {"points": [], "color": ColorPalette.HEALTH["fair"]},
            "high": {"points": [], "color": ColorPalette.SEVERITY["high"]},
            "critical": {"points": [], "color": ColorPalette.SEVERITY["critical"]},
        }

        for item in hotspot_data:
            risk = item.get("risk_level", "low")
            impact = item.get("impact", 0)
            likelihood = item.get("likelihood", 0)

            if risk in risk_categories:
                risk_categories[risk]["points"].append((likelihood, impact))

        # Create datasets for each risk level
        datasets = []
        for risk_level, data in risk_categories.items():
            if data["points"]:
                datasets.append(
                    {
                        "label": risk_level.title(),
                        "data": [{"x": x, "y": y} for x, y in data["points"]],
                        "backgroundColor": data["color"],
                        "pointRadius": 5,
                    }
                )

        config = ChartConfig(type=ChartType.SCATTER, title="Risk Matrix")

        chart_config = {
            "type": "scatter",
            "data": {"datasets": datasets},
            "options": {
                **self._get_chart_options(config),
                "scales": {
                    "x": {"title": {"display": True, "text": "Likelihood"}, "min": 0, "max": 100},
                    "y": {"title": {"display": True, "text": "Impact"}, "min": 0, "max": 100},
                },
            },
        }

        return chart_config

    def create_hotspot_trend(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hotspot trend chart over time.

        Args:
            trend_data: List of data points with date and metrics

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        labels = []
        total_hotspots = []
        critical_hotspots = []
        avg_risk = []

        for point in trend_data:
            labels.append(point.get("date", ""))
            total_hotspots.append(point.get("total_hotspots", 0))
            critical_hotspots.append(point.get("critical_hotspots", 0))
            avg_risk.append(point.get("avg_risk_score", 0))

        datasets = [
            {
                "label": "Total Hotspots",
                "data": total_hotspots,
                "borderColor": ColorPalette.DEFAULT[0],
                "fill": False,
                "yAxisID": "y",
            },
            {
                "label": "Critical Hotspots",
                "data": critical_hotspots,
                "borderColor": ColorPalette.SEVERITY["critical"],
                "fill": False,
                "yAxisID": "y",
            },
            {
                "label": "Avg Risk Score",
                "data": avg_risk,
                "borderColor": ColorPalette.DEFAULT[2],
                "fill": False,
                "yAxisID": "y1",
            },
        ]

        config = ChartConfig(type=ChartType.LINE, title="Hotspot Trends")

        chart_config = self.create_chart(
            ChartType.LINE, {"labels": labels, "datasets": datasets}, config
        )

        # Dual y-axis
        chart_config["options"]["scales"] = {
            "y": {
                "type": "linear",
                "display": True,
                "position": "left",
                "title": {"display": True, "text": "Count"},
            },
            "y1": {
                "type": "linear",
                "display": True,
                "position": "right",
                "title": {"display": True, "text": "Risk Score"},
                "grid": {"drawOnChartArea": False},
            },
        }

        return chart_config

    def create_file_activity_chart(
        self, activity_data: List[Dict[str, Any]], limit: int = 20
    ) -> Dict[str, Any]:
        """Create file activity chart.

        Args:
            activity_data: File activity data
            limit: Maximum files to show

        Returns:
            Dict[str, Any]: Stacked bar chart configuration
        """
        # Sort by total activity
        sorted_data = sorted(activity_data, key=lambda x: x.get("total_changes", 0), reverse=True)[
            :limit
        ]

        labels = []
        additions = []
        deletions = []
        modifications = []

        for item in sorted_data:
            labels.append(self._truncate_filename(item.get("file", "")))
            additions.append(item.get("additions", 0))
            deletions.append(item.get("deletions", 0))
            modifications.append(item.get("modifications", 0))

        datasets = [
            {
                "label": "Additions",
                "data": additions,
                "backgroundColor": ColorPalette.HEALTH["good"],
            },
            {
                "label": "Modifications",
                "data": modifications,
                "backgroundColor": ColorPalette.DEFAULT[0],
            },
            {
                "label": "Deletions",
                "data": deletions,
                "backgroundColor": ColorPalette.SEVERITY["high"],
            },
        ]

        config = ChartConfig(type=ChartType.STACKED_BAR, title="File Change Activity")

        return {
            "type": "bar",
            "data": {"labels": labels, "datasets": datasets},
            "options": {
                **self._get_chart_options(config),
                "scales": {"x": {"stacked": True}, "y": {"stacked": True}},
            },
        }

    def display_terminal(self, hotspot_data: Dict[str, Any], show_details: bool = True) -> None:
        """Display hotspot analysis in terminal.

        Args:
            hotspot_data: Hotspot analysis data
            show_details: Whether to show detailed breakdown
        """
        # Display header
        self.terminal_display.display_header("Hotspot Analysis", style="double")

        # Display summary metrics
        summary_data = {
            "Total Hotspots": hotspot_data.get("total_hotspots", 0),
            "Critical": hotspot_data.get("critical_count", 0),
            "High Risk": hotspot_data.get("high_count", 0),
            "Files Analyzed": hotspot_data.get("files_analyzed", 0),
        }

        self.terminal_display.display_metrics(summary_data, title="Summary")

        # Display risk distribution
        if "risk_distribution" in hotspot_data:
            self.terminal_display.display_distribution(
                hotspot_data["risk_distribution"],
                title="Risk Distribution",
                labels=["Low", "Medium", "High", "Critical"],
            )

        # Display top hotspots
        if show_details and "hotspots" in hotspot_data:
            headers = ["File", "Risk", "Changes", "Complexity", "Score"]
            rows = []

            for hotspot in hotspot_data["hotspots"][:10]:
                risk = hotspot.get("risk_level", "low")
                risk_colored = self.terminal_display.colorize(
                    risk.upper(), self._get_risk_color(risk)
                )

                rows.append(
                    [
                        self._truncate_filename(hotspot.get("file", "")),
                        risk_colored,
                        str(hotspot.get("change_frequency", 0)),
                        str(hotspot.get("complexity", 0)),
                        self.format_number(hotspot.get("risk_score", 0), precision=1),
                    ]
                )

            self.terminal_display.display_table(headers, rows, title="Top Hotspots")

        # Display recommendations
        if "recommendations" in hotspot_data:
            self.terminal_display.display_list(
                hotspot_data["recommendations"], title="Recommendations", style="numbered"
            )

    def _truncate_filename(self, filename: str, max_length: int = 35) -> str:
        """Truncate filename for display.

        Args:
            filename: File path
            max_length: Maximum length

        Returns:
            str: Truncated filename
        """
        if len(filename) <= max_length:
            return filename

        # Try to keep the actual filename
        parts = filename.split("/")
        if parts:
            name = parts[-1]
            if len(name) <= max_length:
                # Show partial path + full filename
                remaining = max_length - len(name) - 4
                if remaining > 0:
                    prefix = filename[:remaining]
                    return f"{prefix}.../{name}"

        return filename[: max_length - 3] + "..."

    def _get_risk_color(self, risk: str) -> str:
        """Get color for risk level.

        Args:
            risk: Risk level

        Returns:
            str: Color name
        """
        colors = {"critical": "red", "high": "yellow", "medium": "cyan", "low": "green"}
        return colors.get(risk.lower(), "white")
