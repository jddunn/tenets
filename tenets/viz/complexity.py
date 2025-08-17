"""Complexity visualization module.

This module provides visualization capabilities for complexity metrics,
including cyclomatic complexity, cognitive complexity, and other
complexity-related visualizations.
"""

from typing import Any, Dict, List, Optional

from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay


class ComplexityVisualizer(BaseVisualizer):
    """Visualizer for complexity metrics.

    Creates visualizations for complexity analysis results including
    distribution charts, heatmaps, and trend analysis.
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize complexity visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        super().__init__(chart_config, display_config)
        self.terminal_display = TerminalDisplay(display_config)

    def create_distribution_chart(
        self, complexity_data: Dict[str, Any], chart_type: ChartType = ChartType.BAR
    ) -> Dict[str, Any]:
        """Create complexity distribution chart.

        Args:
            complexity_data: Complexity analysis data
            chart_type: Type of chart to create

        Returns:
            Dict[str, Any]: Chart configuration
        """
        distribution = complexity_data.get("distribution", {})

        # Default distribution if not provided
        if not distribution:
            distribution = {
                "low": complexity_data.get("low_complexity_count", 0),
                "medium": complexity_data.get("medium_complexity_count", 0),
                "high": complexity_data.get("high_complexity_count", 0),
                "very_high": complexity_data.get("very_high_complexity_count", 0),
            }

        labels = ["Low (1-5)", "Medium (6-10)", "High (11-20)", "Very High (>20)"]
        values = [
            distribution.get("low", 0),
            distribution.get("medium", 0),
            distribution.get("high", 0),
            distribution.get("very_high", 0),
        ]

        # Use severity colors for complexity levels
        colors = [
            ColorPalette.HEALTH["excellent"],
            ColorPalette.HEALTH["good"],
            ColorPalette.HEALTH["fair"],
            ColorPalette.HEALTH["critical"],
        ]

        config = ChartConfig(type=chart_type, title="Complexity Distribution", colors=colors)

        return self.create_chart(chart_type, {"labels": labels, "values": values}, config)

    def create_top_complex_chart(
        self, complex_items: List[Dict[str, Any]], limit: int = 10
    ) -> Dict[str, Any]:
        """Create chart of top complex items.

        Args:
            complex_items: List of complex items with name and complexity
            limit: Maximum items to show

        Returns:
            Dict[str, Any]: Chart configuration
        """
        # Sort and limit items
        sorted_items = sorted(complex_items, key=lambda x: x.get("complexity", 0), reverse=True)[
            :limit
        ]

        labels = []
        values = []
        colors = []

        for item in sorted_items:
            name = item.get("name", "Unknown")
            # Truncate long names
            if len(name) > 30:
                name = name[:27] + "..."
            labels.append(name)

            complexity = item.get("complexity", 0)
            values.append(complexity)

            # Color based on complexity level
            if complexity > 20:
                colors.append(ColorPalette.SEVERITY["critical"])
            elif complexity > 10:
                colors.append(ColorPalette.SEVERITY["high"])
            elif complexity > 5:
                colors.append(ColorPalette.SEVERITY["medium"])
            else:
                colors.append(ColorPalette.SEVERITY["low"])

        config = ChartConfig(
            type=ChartType.HORIZONTAL_BAR,
            title=f"Top {len(sorted_items)} Most Complex Functions",
            colors=colors,
        )

        return self.create_chart(
            ChartType.HORIZONTAL_BAR, {"labels": labels, "values": values}, config
        )

    def create_complexity_heatmap(
        self, file_complexities: Dict[str, List[int]], max_functions: int = 50
    ) -> Dict[str, Any]:
        """Create complexity heatmap for files.

        Args:
            file_complexities: Dictionary of file paths to complexity values
            max_functions: Maximum functions per file to show

        Returns:
            Dict[str, Any]: Heatmap configuration
        """
        # Prepare matrix data
        file_names = []
        matrix = []

        for file_path, complexities in file_complexities.items():
            # Extract filename from path
            file_name = file_path.split("/")[-1]
            if len(file_name) > 20:
                file_name = file_name[:17] + "..."
            file_names.append(file_name)

            # Pad or truncate complexity list
            row = complexities[:max_functions]
            if len(row) < max_functions:
                row.extend([0] * (max_functions - len(row)))
            matrix.append(row)

        # Create function labels
        function_labels = [f"F{i + 1}" for i in range(max_functions)]

        config = ChartConfig(type=ChartType.HEATMAP, title="Complexity Heatmap (Files × Functions)")

        return self.create_chart(
            ChartType.HEATMAP,
            {"matrix": matrix, "x_labels": function_labels, "y_labels": file_names},
            config,
        )

    def create_trend_chart(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create complexity trend chart over time.

        Args:
            trend_data: List of data points with date and metrics

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        if not trend_data:
            return {}

        labels = []
        avg_complexity = []
        max_complexity = []
        total_complex = []

        for point in trend_data:
            labels.append(point.get("date", ""))
            avg_complexity.append(point.get("avg_complexity", 0))
            max_complexity.append(point.get("max_complexity", 0))
            total_complex.append(point.get("complex_functions", 0))

        datasets = [
            {
                "label": "Average Complexity",
                "data": avg_complexity,
                "borderColor": ColorPalette.DEFAULT[0],
                "fill": False,
            },
            {
                "label": "Max Complexity",
                "data": max_complexity,
                "borderColor": ColorPalette.SEVERITY["high"],
                "fill": False,
            },
        ]

        config = ChartConfig(type=ChartType.LINE, title="Complexity Trend Over Time")

        return self.create_chart(ChartType.LINE, {"labels": labels, "datasets": datasets}, config)

    def create_comparison_chart(
        self, current_data: Dict[str, Any], baseline_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comparison chart between current and baseline.

        Args:
            current_data: Current complexity metrics
            baseline_data: Baseline complexity metrics

        Returns:
            Dict[str, Any]: Grouped bar chart configuration
        """
        metrics = ["avg_complexity", "max_complexity", "complex_functions"]
        labels = ["Average", "Maximum", "Complex Count"]

        current_values = [current_data.get(m, 0) for m in metrics]
        baseline_values = [baseline_data.get(m, 0) for m in metrics]

        datasets = [
            {
                "label": "Current",
                "data": current_values,
                "backgroundColor": ColorPalette.DEFAULT[0],
            },
            {
                "label": "Baseline",
                "data": baseline_values,
                "backgroundColor": ColorPalette.DEFAULT[1],
            },
        ]

        config = ChartConfig(type=ChartType.BAR, title="Complexity Comparison")

        return {
            "type": "bar",
            "data": {"labels": labels, "datasets": datasets},
            "options": self._get_chart_options(config),
        }

    def display_terminal(self, complexity_data: Dict[str, Any], show_details: bool = True) -> None:
        """Display complexity analysis in terminal.

        Args:
            complexity_data: Complexity analysis data
            show_details: Whether to show detailed breakdown
        """
        # Display header
        self.terminal_display.display_header("Complexity Analysis", style="double")

        # Display summary metrics
        summary_data = {
            "Average Complexity": self.format_number(
                complexity_data.get("avg_complexity", 0), precision=2
            ),
            "Maximum Complexity": complexity_data.get("max_complexity", 0),
            "Complex Functions": complexity_data.get("complex_functions", 0),
            "Total Functions": complexity_data.get("total_functions", 0),
        }

        self.terminal_display.display_metrics(summary_data, title="Summary")

        # Display distribution
        if "distribution" in complexity_data:
            self.terminal_display.display_distribution(
                complexity_data["distribution"],
                title="Complexity Distribution",
                labels=["Low", "Medium", "High", "Very High"],
            )

        # Display top complex functions
        if show_details and "complex_items" in complexity_data:
            headers = ["Function", "File", "Complexity", "Risk"]
            rows = []

            for item in complexity_data["complex_items"][:10]:
                risk = self._get_risk_level(item.get("complexity", 0))
                rows.append(
                    [
                        item.get("name", "Unknown"),
                        self._truncate_path(item.get("file", "")),
                        str(item.get("complexity", 0)),
                        self.terminal_display.colorize(risk, self._get_risk_color(risk)),
                    ]
                )

            self.terminal_display.display_table(headers, rows, title="Top Complex Functions")

        # Display recommendations
        if "recommendations" in complexity_data:
            self.terminal_display.display_list(
                complexity_data["recommendations"], title="Recommendations", style="numbered"
            )

    def create_radar_chart(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create radar chart for complexity metrics.

        Args:
            metrics: Dictionary of metric names to values

        Returns:
            Dict[str, Any]: Radar chart configuration
        """
        # Normalize metrics to 0-100 scale
        normalized = {}
        max_values = {
            "cyclomatic": 50,
            "cognitive": 100,
            "halstead": 1000,
            "maintainability": 100,
            "lines": 500,
        }

        labels = []
        values = []

        for metric, value in metrics.items():
            labels.append(metric.replace("_", " ").title())
            max_val = max_values.get(metric, 100)
            normalized_value = min(100, (value / max_val) * 100)
            values.append(normalized_value)

        config = ChartConfig(type=ChartType.RADAR, title="Complexity Metrics Radar")

        return self.create_chart(
            ChartType.RADAR,
            {"labels": labels, "datasets": [{"label": "Current", "data": values}]},
            config,
        )

    def _get_risk_level(self, complexity: int) -> str:
        """Get risk level for complexity value.

        Args:
            complexity: Complexity value

        Returns:
            str: Risk level
        """
        if complexity > 20:
            return "Critical"
        elif complexity > 10:
            return "High"
        elif complexity > 5:
            return "Medium"
        else:
            return "Low"

    def _get_risk_color(self, risk: str) -> str:
        """Get color for risk level.

        Args:
            risk: Risk level

        Returns:
            str: Color name
        """
        colors = {"Critical": "red", "High": "yellow", "Medium": "cyan", "Low": "green"}
        return colors.get(risk, "white")

    def _truncate_path(self, path: str, max_length: int = 40) -> str:
        """Truncate file path for display.

        Args:
            path: File path
            max_length: Maximum length

        Returns:
            str: Truncated path
        """
        if len(path) <= max_length:
            return path

        # Try to keep filename while respecting max_length strictly
        parts = path.split("/")
        filename = parts[-1] if parts else path
        # Reserve space for ellipsis and '/' if adding prefix
        reserve = 4  # '.../'
        if len(filename) + reserve < max_length:
            prefix_len = max_length - len(filename) - reserve
            prefix = path[:prefix_len]
            result = f"{prefix}.../{filename}"
            return result[:max_length]

        # Fallback: plain truncation with ellipsis, capped
        return (path[: max_length - 3] + "...")[:max_length]
