"""Base visualization module providing common functionality.

This module provides the base classes and utilities for all visualization
components. It includes chart configuration, color management, and common
visualization patterns used throughout the viz package.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tenets.utils.logger import get_logger


class ChartType(Enum):
    """Supported chart types."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    RADAR = "radar"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    NETWORK = "network"
    BUBBLE = "bubble"
    STACKED_BAR = "stacked_bar"
    HORIZONTAL_BAR = "horizontal_bar"
    DONUT = "donut"
    AREA = "area"
    TIMELINE = "timeline"


class DisplayFormat(Enum):
    """Supported display formats."""

    TERMINAL = "terminal"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    SVG = "svg"
    PNG = "png"


@dataclass
class ChartConfig:
    """Configuration for chart generation.

    Attributes:
        type: Type of chart to generate
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        colors: Custom color palette
        theme: Visual theme (light, dark, etc.)
        interactive: Whether chart should be interactive
        show_legend: Whether to show legend
        show_grid: Whether to show grid lines
        animation: Whether to animate chart
        responsive: Whether chart should be responsive
        export_options: Export format options
    """

    type: ChartType
    title: str = ""
    width: int = 800
    height: int = 400
    colors: Optional[List[str]] = None
    theme: str = "light"
    interactive: bool = True
    show_legend: bool = True
    show_grid: bool = True
    animation: bool = True
    responsive: bool = True
    export_options: List[str] = field(default_factory=lambda: ["png", "svg"])


@dataclass
class DisplayConfig:
    """Configuration for terminal display.

    Attributes:
        use_colors: Whether to use colors in terminal
        use_unicode: Whether to use unicode characters
        max_width: Maximum display width
        max_rows: Maximum rows to display
        truncate: Whether to truncate long text
        show_progress: Whether to show progress indicators
        style: Display style (compact, detailed, etc.)
    """

    use_colors: bool = True
    use_unicode: bool = True
    max_width: int = 120
    max_rows: int = 50
    truncate: bool = True
    show_progress: bool = True
    style: str = "detailed"


class ColorPalette:
    """Color palette management for visualizations.

    Provides consistent color schemes across all visualizations with
    support for different themes and accessibility considerations.
    """

    # Default color palettes
    DEFAULT = [
        "#2563eb",  # Blue
        "#8b5cf6",  # Purple
        "#10b981",  # Green
        "#f59e0b",  # Amber
        "#ef4444",  # Red
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#84cc16",  # Lime
        "#f97316",  # Orange
        "#6366f1",  # Indigo
    ]

    SEVERITY = {
        "critical": "#dc2626",  # Red-600
        "high": "#ea580c",  # Orange-600
        "medium": "#ca8a04",  # Yellow-600
        "low": "#16a34a",  # Green-600
        "info": "#0891b2",  # Cyan-600
    }

    HEALTH = {
        "excellent": "#10b981",  # Green
        "good": "#84cc16",  # Lime
        "fair": "#f59e0b",  # Amber
        "poor": "#f97316",  # Orange
        "critical": "#ef4444",  # Red
    }

    MONOCHROME = [
        "#1e293b",  # Slate-800
        "#334155",  # Slate-700
        "#475569",  # Slate-600
        "#64748b",  # Slate-500
        "#94a3b8",  # Slate-400
        "#cbd5e1",  # Slate-300
        "#e2e8f0",  # Slate-200
        "#f1f5f9",  # Slate-100
    ]

    @classmethod
    def get_palette(cls, name: str = "default") -> List[str]:
        """Get a color palette by name.

        Args:
            name: Palette name (default, monochrome, etc.)

        Returns:
            List[str]: List of color hex codes
        """
        palettes = {
            "default": cls.DEFAULT,
            "monochrome": cls.MONOCHROME,
            "severity": list(cls.SEVERITY.values()),
            "health": list(cls.HEALTH.values()),
        }
        return palettes.get(name.lower(), cls.DEFAULT)

    @classmethod
    def get_color(cls, value: Any, category: str = "default") -> str:
        """Get a color for a specific value.

        Args:
            value: Value to get color for
            category: Category (severity, health, etc.)

        Returns:
            str: Color hex code
        """
        if category == "severity":
            return cls.SEVERITY.get(str(value).lower(), cls.DEFAULT[0])
        elif category == "health":
            return cls.HEALTH.get(str(value).lower(), cls.DEFAULT[0])
        else:
            # Use default palette with modulo for cycling
            if isinstance(value, int):
                return cls.DEFAULT[value % len(cls.DEFAULT)]
            return cls.DEFAULT[0]

    @classmethod
    def interpolate_color(
        cls,
        value: float,
        min_val: float = 0,
        max_val: float = 100,
        start_color: str = "#10b981",
        end_color: str = "#ef4444",
    ) -> str:
        """Interpolate color based on value.

        Args:
            value: Value to interpolate
            min_val: Minimum value
            max_val: Maximum value
            start_color: Color for minimum value
            end_color: Color for maximum value

        Returns:
            str: Interpolated color hex code
        """
        # Normalize value
        if max_val == min_val:
            ratio = 0.5
        else:
            ratio = (value - min_val) / (max_val - min_val)
            ratio = max(0, min(1, ratio))

        # Parse colors
        start_rgb = cls._hex_to_rgb(start_color)
        end_rgb = cls._hex_to_rgb(end_color)

        # Interpolate
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)

        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB.

        Args:
            hex_color: Hex color code

        Returns:
            Tuple[int, int, int]: RGB values
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class BaseVisualizer:
    """Base class for all visualizers.

    Provides common functionality for creating visualizations including
    chart generation, color management, and data formatting.

    Attributes:
        logger: Logger instance
        chart_config: Default chart configuration
        display_config: Default display configuration
        color_palette: Color palette to use
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize base visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        self.logger = get_logger(self.__class__.__name__)
        self.chart_config = chart_config or ChartConfig(type=ChartType.BAR)
        self.display_config = display_config or DisplayConfig()
        self.color_palette = ColorPalette.get_palette("default")

    def create_chart(
        self, chart_type: ChartType, data: Dict[str, Any], config: Optional[ChartConfig] = None
    ) -> Dict[str, Any]:
        """Create a chart configuration.

        Args:
            chart_type: Type of chart
            data: Chart data
            config: Optional chart configuration

        Returns:
            Dict[str, Any]: Chart configuration for rendering
        """
        config = config or self.chart_config
        config.type = chart_type

        # Route to specific chart creator
        creators = {
            ChartType.BAR: self._create_bar_chart,
            ChartType.HORIZONTAL_BAR: self._create_horizontal_bar_chart,
            ChartType.LINE: self._create_line_chart,
            ChartType.PIE: self._create_pie_chart,
            ChartType.SCATTER: self._create_scatter_chart,
            ChartType.RADAR: self._create_radar_chart,
            ChartType.GAUGE: self._create_gauge_chart,
            ChartType.HEATMAP: self._create_heatmap,
            ChartType.TREEMAP: self._create_treemap,
            ChartType.NETWORK: self._create_network_graph,
            ChartType.BUBBLE: self._create_bubble_chart,
        }

        creator = creators.get(chart_type, self._create_bar_chart)
        return creator(data, config)

    def _create_horizontal_bar_chart(
        self, data: Dict[str, Any], config: ChartConfig
    ) -> Dict[str, Any]:
        """Create horizontal bar chart configuration.

        Args:
            data: Chart data with 'labels' and 'values'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Horizontal bar chart configuration
        """
        colors = config.colors or self.color_palette
        return {
            "type": "horizontal_bar",
            "data": {
                "labels": data.get("labels", []),
                "datasets": [
                    {
                        "label": config.title,
                        "data": data.get("values", []),
                        "backgroundColor": colors,
                        "borderColor": colors,
                        "borderWidth": 1,
                    }
                ],
            },
            "options": self._get_chart_options(config),
        }

    def _create_bar_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create bar chart configuration.

        Args:
            data: Chart data with 'labels' and 'values'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Bar chart configuration
        """
        colors = config.colors or self.color_palette

        return {
            "type": "bar",
            "data": {
                "labels": data.get("labels", []),
                "datasets": [
                    {
                        "label": config.title,
                        "data": data.get("values", []),
                        "backgroundColor": colors,
                        "borderColor": colors,
                        "borderWidth": 1,
                    }
                ],
            },
            "options": self._get_chart_options(config),
        }

    def _create_line_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create line chart configuration.

        Args:
            data: Chart data with 'labels' and 'datasets'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        datasets = []
        for i, dataset in enumerate(data.get("datasets", [])):
            color = self.color_palette[i % len(self.color_palette)]
            datasets.append(
                {
                    "label": dataset.get("label", f"Series {i + 1}"),
                    "data": dataset.get("data", []),
                    "borderColor": color,
                    "backgroundColor": color + "20",
                    "borderWidth": 2,
                    "fill": dataset.get("fill", False),
                    "tension": 0.1,
                }
            )

        return {
            "type": "line",
            "data": {"labels": data.get("labels", []), "datasets": datasets},
            "options": self._get_chart_options(config),
        }

    def _create_pie_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create pie chart configuration.

        Args:
            data: Chart data with 'labels' and 'values'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Pie chart configuration
        """
        colors = config.colors or self.color_palette

        return {
            "type": "pie",
            "data": {
                "labels": data.get("labels", []),
                "datasets": [
                    {
                        "data": data.get("values", []),
                        "backgroundColor": colors,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    }
                ],
            },
            "options": self._get_chart_options(config),
        }

    def _create_scatter_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create scatter chart configuration.

        Args:
            data: Chart data with 'points' as [(x, y), ...]
            config: Chart configuration

        Returns:
            Dict[str, Any]: Scatter chart configuration
        """
        points = data.get("points", [])
        chart_data = [{"x": x, "y": y} for x, y in points]

        return {
            "type": "scatter",
            "data": {
                "datasets": [
                    {
                        "label": config.title,
                        "data": chart_data,
                        "backgroundColor": self.color_palette[0],
                        "pointRadius": 5,
                    }
                ]
            },
            "options": self._get_chart_options(config),
        }

    def _create_radar_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create radar chart configuration.

        Args:
            data: Chart data with 'labels' and 'datasets'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Radar chart configuration
        """
        datasets = []
        for i, dataset in enumerate(data.get("datasets", [])):
            color = self.color_palette[i % len(self.color_palette)]
            datasets.append(
                {
                    "label": dataset.get("label", f"Series {i + 1}"),
                    "data": dataset.get("data", []),
                    "borderColor": color,
                    "backgroundColor": color + "40",
                    "borderWidth": 2,
                    "pointRadius": 4,
                }
            )

        return {
            "type": "radar",
            "data": {"labels": data.get("labels", []), "datasets": datasets},
            "options": self._get_chart_options(config),
        }

    def _create_gauge_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create gauge chart configuration.

        Args:
            data: Chart data with 'value' and 'max'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Gauge chart configuration
        """
        value = data.get("value", 0)
        max_value = data.get("max", 100)

        # Determine color based on value
        color = ColorPalette.interpolate_color(
            value, 0, max_value, start_color="#10b981", end_color="#ef4444"
        )

        return {
            "type": "doughnut",
            "data": {
                "datasets": [
                    {
                        "data": [value, max_value - value],
                        "backgroundColor": [color, "#e5e7eb"],
                        "borderWidth": 0,
                    }
                ]
            },
            "options": {
                **self._get_chart_options(config),
                "circumference": 180,
                "rotation": 270,
                "cutout": "75%",
            },
        }

    def _create_heatmap(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create heatmap configuration.

        Args:
            data: Chart data with 'matrix', 'x_labels', 'y_labels'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Heatmap configuration
        """
        matrix = data.get("matrix", [])
        x_labels = data.get("x_labels", [])
        y_labels = data.get("y_labels", [])

        # Convert matrix to data points
        data_points = []
        for y_idx, row in enumerate(matrix):
            for x_idx, value in enumerate(row):
                data_points.append({"x": x_idx, "y": y_idx, "value": value})

        return {
            "type": "heatmap",
            "data": {
                "labels": {"x": x_labels, "y": y_labels},
                "datasets": [{"label": config.title, "data": data_points}],
            },
            "options": self._get_chart_options(config),
        }

    def _create_treemap(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create treemap configuration.

        Args:
            data: Hierarchical data structure
            config: Chart configuration

        Returns:
            Dict[str, Any]: Treemap configuration
        """
        return {
            "type": "treemap",
            "data": {
                "datasets": [
                    {
                        "label": config.title,
                        "tree": data.get("tree", []),
                        "backgroundColor": self.color_palette,
                    }
                ]
            },
            "options": self._get_chart_options(config),
        }

    def _create_network_graph(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create network graph configuration.

        Args:
            data: Graph data with 'nodes' and 'edges'
            config: Chart configuration

        Returns:
            Dict[str, Any]: Network graph configuration
        """
        return {
            "type": "network",
            "data": {"nodes": data.get("nodes", []), "edges": data.get("edges", [])},
            "options": {**self._get_chart_options(config), "layout": data.get("layout", "force")},
        }

    def _create_bubble_chart(self, data: Dict[str, Any], config: ChartConfig) -> Dict[str, Any]:
        """Create bubble chart configuration.

        Args:
            data: Chart data with 'points' as [(x, y, size), ...]
            config: Chart configuration

        Returns:
            Dict[str, Any]: Bubble chart configuration
        """
        points = data.get("points", [])
        chart_data = [{"x": x, "y": y, "r": r} for x, y, r in points]

        return {
            "type": "bubble",
            "data": {
                "datasets": [
                    {
                        "label": config.title,
                        "data": chart_data,
                        "backgroundColor": self.color_palette[0] + "80",
                    }
                ]
            },
            "options": self._get_chart_options(config),
        }

    def _get_chart_options(self, config: ChartConfig) -> Dict[str, Any]:
        """Get common chart options.

        Args:
            config: Chart configuration

        Returns:
            Dict[str, Any]: Chart options
        """
        return {
            "responsive": config.responsive,
            "maintainAspectRatio": False,
            "animation": {"duration": 1000 if config.animation else 0},
            "plugins": {
                "title": {"display": bool(config.title), "text": config.title},
                "legend": {"display": config.show_legend},
            },
            "scales": {"x": {"display": config.show_grid}, "y": {"display": config.show_grid}},
        }

    def format_number(
        self, value: Union[int, float], precision: int = 2, use_thousands: bool = True
    ) -> str:
        """Format a number for display.

        Args:
            value: Number to format
            precision: Decimal precision
            use_thousands: Use thousands separator

        Returns:
            str: Formatted number
        """
        if isinstance(value, float):
            formatted = f"{value:.{precision}f}"
        else:
            formatted = str(value)

        if use_thousands and abs(value) >= 1000:
            parts = formatted.split(".")
            parts[0] = f"{int(parts[0]):,}"
            formatted = ".".join(parts) if len(parts) > 1 else parts[0]

        return formatted

    def format_percentage(
        self, value: float, precision: int = 1, include_sign: bool = False
    ) -> str:
        """Format a value as percentage.

        Args:
            value: Value (0-1 or 0-100 depending on context)
            precision: Decimal precision
            include_sign: Include + sign for positive values

        Returns:
            str: Formatted percentage
        """
        # Assume 0-1 range if value <= 1
        if -1 <= value <= 1:
            percentage = value * 100
        else:
            percentage = value

        formatted = f"{percentage:.{precision}f}%"

        if include_sign and percentage > 0:
            formatted = f"+{formatted}"

        return formatted

    def export_chart(
        self, chart_config: Dict[str, Any], output_path: Path, format: str = "json"
    ) -> Path:
        """Export chart configuration to file.

        Args:
            chart_config: Chart configuration
            output_path: Output file path
            format: Export format (json, html, etc.)

        Returns:
            Path: Path to exported file
        """
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(chart_config, f, indent=2)
        elif format == "html":
            # Generate standalone HTML with chart
            html = self._generate_standalone_html(chart_config)
            with open(output_path, "w") as f:
                f.write(html)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.debug(f"Exported chart to {output_path}")
        return output_path

    def _generate_standalone_html(self, chart_config: Dict[str, Any]) -> str:
        """Generate standalone HTML for chart.

        Args:
            chart_config: Chart configuration

        Returns:
            str: HTML content
        """
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Chart</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <!-- Powered by Chart.js -->
    <canvas id="chart"></canvas>
    <script>
        new Chart(document.getElementById('chart'), {json.dumps(chart_config)});
    </script>
    <!-- Chart.js -->
</body>
</html>"""
