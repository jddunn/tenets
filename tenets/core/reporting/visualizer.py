"""Visualization module for report generation.

This module provides chart and graph generation functionality for creating
visual representations of analysis data. It supports various chart types
and can generate both static and interactive visualizations.

The visualizer creates data visualizations that help understand code metrics,
trends, and patterns at a glance.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


class ChartGenerator:
    """Generator for various chart types.

    Creates chart configurations and data structures for visualization
    libraries like Chart.js, D3.js, or server-side rendering.

    Attributes:
        config: Configuration object
        logger: Logger instance
        color_palette: Default color palette
    """

    def __init__(self, config: TenetsConfig):
        """Initialize chart generator.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Default color palette
        self.color_palette = [
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

    def create_bar_chart(
        self,
        labels: List[str],
        values: List[Union[int, float]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        colors: Optional[List[str]] = None,
        horizontal: bool = False,
    ) -> Dict[str, Any]:
        """Create a bar chart configuration.

        Args:
            labels: Bar labels
            values: Bar values
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            colors: Custom colors
            horizontal: Use horizontal bars

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> generator = ChartGenerator(config)
            >>> chart = generator.create_bar_chart(
            ...     ["Low", "Medium", "High"],
            ...     [10, 25, 5],
            ...     title="Issue Distribution"
            ... )
        """
        if not colors:
            colors = self._get_colors(len(values))

        config = {
            "type": "horizontalBar" if horizontal else "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": title,
                        "data": values,
                        "backgroundColor": colors,
                        "borderColor": colors,
                        "borderWidth": 1,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": False},
                },
                "scales": {
                    "x": {"title": {"display": bool(x_label), "text": x_label}},
                    "y": {"title": {"display": bool(y_label), "text": y_label}},
                },
            },
        }

        return config

    def create_line_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        smooth: bool = True,
    ) -> Dict[str, Any]:
        """Create a line chart configuration.

        Args:
            labels: X-axis labels
            datasets: List of dataset configurations
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            smooth: Use smooth lines

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_line_chart(
            ...     ["Jan", "Feb", "Mar"],
            ...     [
            ...         {"label": "Bugs", "data": [10, 8, 12]},
            ...         {"label": "Features", "data": [5, 7, 9]}
            ...     ],
            ...     title="Monthly Trends"
            ... )
        """
        # Process datasets
        processed_datasets = []
        for i, dataset in enumerate(datasets):
            color = self.color_palette[i % len(self.color_palette)]
            processed_datasets.append(
                {
                    "label": dataset.get("label", f"Series {i + 1}"),
                    "data": dataset.get("data", []),
                    "borderColor": dataset.get("color", color),
                    "backgroundColor": dataset.get("color", color) + "20",
                    "borderWidth": 2,
                    "fill": dataset.get("fill", False),
                    "tension": 0.1 if smooth else 0,
                }
            )

        config = {
            "type": "line",
            "data": {"labels": labels, "datasets": processed_datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": len(processed_datasets) > 1},
                },
                "scales": {
                    "x": {"title": {"display": bool(x_label), "text": x_label}},
                    "y": {"title": {"display": bool(y_label), "text": y_label}},
                },
            },
        }

        return config

    def create_pie_chart(
        self,
        labels: List[str],
        values: List[Union[int, float]],
        title: str = "",
        colors: Optional[List[str]] = None,
        as_donut: bool = False,
    ) -> Dict[str, Any]:
        """Create a pie chart configuration.

        Args:
            labels: Slice labels
            values: Slice values
            title: Chart title
            colors: Custom colors
            as_donut: Create as donut chart

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_pie_chart(
            ...     ["Python", "JavaScript", "Java"],
            ...     [450, 320, 180],
            ...     title="Language Distribution"
            ... )
        """
        if not colors:
            colors = self._get_colors(len(values))

        config = {
            "type": "doughnut" if as_donut else "pie",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": colors,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"position": "right"},
                    "tooltip": {
                        "callbacks": {
                            "label": "function(context) { "
                            'var label = context.label || ""; '
                            "var value = context.parsed; "
                            "var total = context.dataset.data.reduce((a, b) => a + b, 0); "
                            "var percentage = ((value / total) * 100).toFixed(1); "
                            'return label + ": " + value + " (" + percentage + "%)"; }'
                        }
                    },
                },
            },
        }

        if as_donut:
            config["options"]["cutout"] = "50%"

        return config

    def create_scatter_plot(
        self,
        data_points: List[Tuple[float, float]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        point_labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a scatter plot configuration.

        Args:
            data_points: List of (x, y) tuples
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            point_labels: Labels for points
            colors: Point colors

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_scatter_plot(
            ...     [(10, 5), (20, 8), (15, 12)],
            ...     title="Complexity vs Size",
            ...     x_label="Lines of Code",
            ...     y_label="Complexity"
            ... )
        """
        # Convert data points to Chart.js format
        chart_data = [{"x": x, "y": y} for x, y in data_points]

        config = {
            "type": "scatter",
            "data": {
                "datasets": [
                    {
                        "label": title,
                        "data": chart_data,
                        "backgroundColor": colors[0] if colors else self.color_palette[0],
                        "pointRadius": 5,
                        "pointHoverRadius": 7,
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": False},
                },
                "scales": {
                    "x": {
                        "type": "linear",
                        "position": "bottom",
                        "title": {"display": bool(x_label), "text": x_label},
                    },
                    "y": {"title": {"display": bool(y_label), "text": y_label}},
                },
            },
        }

        # Add point labels if provided
        if point_labels and len(point_labels) == len(data_points):
            config["options"]["plugins"]["tooltip"] = {
                "callbacks": {
                    "label": f"function(context) {{ "
                    f"var labels = {json.dumps(point_labels)}; "
                    f'return labels[context.dataIndex] + ": (" + '
                    f'context.parsed.x + ", " + context.parsed.y + ")"; }}'
                }
            }

        return config

    def create_radar_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str = "",
        max_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a radar chart configuration.

        Args:
            labels: Axis labels
            datasets: List of dataset configurations
            title: Chart title
            max_value: Maximum value for axes

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_radar_chart(
            ...     ["Quality", "Performance", "Security", "Maintainability"],
            ...     [{"label": "Current", "data": [7, 8, 6, 9]}],
            ...     title="Code Metrics"
            ... )
        """
        # Process datasets
        processed_datasets = []
        for i, dataset in enumerate(datasets):
            color = self.color_palette[i % len(self.color_palette)]
            processed_datasets.append(
                {
                    "label": dataset.get("label", f"Series {i + 1}"),
                    "data": dataset.get("data", []),
                    "borderColor": dataset.get("color", color),
                    "backgroundColor": dataset.get("color", color) + "40",
                    "borderWidth": 2,
                    "pointRadius": 4,
                    "pointHoverRadius": 6,
                }
            )

        config = {
            "type": "radar",
            "data": {"labels": labels, "datasets": processed_datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": len(processed_datasets) > 1},
                },
                "scales": {
                    "r": {
                        "beginAtZero": True,
                        "max": max_value,
                        "ticks": {"stepSize": max_value / 5 if max_value else None},
                    }
                },
            },
        }

        return config

    def create_gauge_chart(
        self,
        value: float,
        max_value: float = 100,
        title: str = "",
        thresholds: Optional[List[Tuple[float, str]]] = None,
    ) -> Dict[str, Any]:
        """Create a gauge chart configuration.

        Args:
            value: Current value
            max_value: Maximum value
            title: Chart title
            thresholds: List of (value, color) thresholds

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_gauge_chart(
            ...     75,
            ...     100,
            ...     title="Health Score",
            ...     thresholds=[(60, "yellow"), (80, "green")]
            ... )
        """
        # Default thresholds if not provided
        if not thresholds:
            thresholds = [
                (40, "#ef4444"),  # Red
                (60, "#f59e0b"),  # Yellow
                (80, "#10b981"),  # Green
            ]

        # Determine color based on value
        color = "#ef4444"  # Default red
        for threshold_value, threshold_color in thresholds:
            if value >= threshold_value:
                color = threshold_color

        # Create as a doughnut chart with rotation
        config = {
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
                "responsive": True,
                "maintainAspectRatio": False,
                "circumference": 180,
                "rotation": 270,
                "cutout": "75%",
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": False},
                    "tooltip": {"enabled": False},
                },
            },
        }

        return config

    def create_stacked_bar_chart(
        self,
        labels: List[str],
        datasets: List[Dict[str, Any]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        horizontal: bool = False,
    ) -> Dict[str, Any]:
        """Create a stacked bar chart configuration.

        Args:
            labels: Bar labels
            datasets: List of dataset configurations
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            horizontal: Use horizontal bars

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_stacked_bar_chart(
            ...     ["Sprint 1", "Sprint 2", "Sprint 3"],
            ...     [
            ...         {"label": "Completed", "data": [8, 10, 12]},
            ...         {"label": "In Progress", "data": [3, 2, 4]},
            ...         {"label": "Blocked", "data": [1, 0, 2]}
            ...     ],
            ...     title="Sprint Progress"
            ... )
        """
        # Process datasets
        processed_datasets = []
        for i, dataset in enumerate(datasets):
            color = self.color_palette[i % len(self.color_palette)]
            processed_datasets.append(
                {
                    "label": dataset.get("label", f"Series {i + 1}"),
                    "data": dataset.get("data", []),
                    "backgroundColor": dataset.get("color", color),
                    "borderColor": dataset.get("color", color),
                    "borderWidth": 1,
                }
            )

        config = {
            "type": "bar",
            "data": {"labels": labels, "datasets": processed_datasets},
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "indexAxis": "y" if horizontal else "x",
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": True},
                },
                "scales": {
                    "x": {"stacked": True, "title": {"display": bool(x_label), "text": x_label}},
                    "y": {"stacked": True, "title": {"display": bool(y_label), "text": y_label}},
                },
            },
        }

        return config

    def create_bubble_chart(
        self,
        data_points: List[Tuple[float, float, float]],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        bubble_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a bubble chart configuration.

        Args:
            data_points: List of (x, y, size) tuples
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            bubble_labels: Labels for bubbles

        Returns:
            Dict[str, Any]: Chart configuration

        Example:
            >>> chart = generator.create_bubble_chart(
            ...     [(10, 5, 20), (20, 8, 35), (15, 12, 15)],
            ...     title="File Analysis",
            ...     x_label="Complexity",
            ...     y_label="Changes"
            ... )
        """
        # Convert data points to Chart.js format
        chart_data = [{"x": x, "y": y, "r": r} for x, y, r in data_points]

        config = {
            "type": "bubble",
            "data": {
                "datasets": [
                    {
                        "label": title,
                        "data": chart_data,
                        "backgroundColor": self.color_palette[0] + "80",
                        "borderColor": self.color_palette[0],
                        "borderWidth": 1,
                    }
                ]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "title": {"display": bool(title), "text": title},
                    "legend": {"display": False},
                },
                "scales": {
                    "x": {"title": {"display": bool(x_label), "text": x_label}},
                    "y": {"title": {"display": bool(y_label), "text": y_label}},
                },
            },
        }

        return config

    def _get_colors(self, count: int) -> List[str]:
        """Get colors from palette.

        Args:
            count: Number of colors needed

        Returns:
            List[str]: List of colors
        """
        colors = []
        for i in range(count):
            colors.append(self.color_palette[i % len(self.color_palette)])
        return colors


def create_chart(
    chart_type: str, data: Dict[str, Any], title: str = "", config: Optional[TenetsConfig] = None
) -> Dict[str, Any]:
    """Convenience function to create a chart.

    Args:
        chart_type: Type of chart (bar, line, pie, etc.)
        data: Chart data
        title: Chart title
        config: Optional configuration

    Returns:
        Dict[str, Any]: Chart configuration

    Example:
        >>> from tenets.core.reporting.visualizer import create_chart
        >>> chart = create_chart(
        ...     "bar",
        ...     {"labels": ["A", "B", "C"], "values": [1, 2, 3]},
        ...     title="Sample Chart"
        ... )
    """
    if config is None:
        config = TenetsConfig()

    generator = ChartGenerator(config)

    if chart_type == "bar":
        return generator.create_bar_chart(
            data.get("labels", []), data.get("values", []), title=title
        )
    elif chart_type == "line":
        return generator.create_line_chart(
            data.get("labels", []), data.get("datasets", []), title=title
        )
    elif chart_type == "pie":
        return generator.create_pie_chart(
            data.get("labels", []), data.get("values", []), title=title
        )
    elif chart_type == "scatter":
        return generator.create_scatter_plot(data.get("points", []), title=title)
    elif chart_type == "radar":
        return generator.create_radar_chart(
            data.get("labels", []), data.get("datasets", []), title=title
        )
    elif chart_type == "gauge":
        return generator.create_gauge_chart(data.get("value", 0), data.get("max", 100), title=title)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")


def create_heatmap(
    matrix_data: List[List[float]],
    x_labels: List[str],
    y_labels: List[str],
    title: str = "",
    color_scale: str = "viridis",
) -> Dict[str, Any]:
    """Create a heatmap visualization.

    Args:
        matrix_data: 2D matrix of values
        x_labels: X-axis labels
        y_labels: Y-axis labels
        title: Chart title
        color_scale: Color scale name

    Returns:
        Dict[str, Any]: Heatmap configuration

    Example:
        >>> from tenets.core.reporting.visualizer import create_heatmap
        >>> heatmap = create_heatmap(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...     ["A", "B", "C"],
        ...     ["X", "Y", "Z"],
        ...     title="Correlation Matrix"
        ... )
    """
    # Find min and max values for color scaling
    flat_values = [val for row in matrix_data for val in row]
    min_val = min(flat_values) if flat_values else 0
    max_val = max(flat_values) if flat_values else 1

    # Convert matrix to chart data points
    data_points = []
    for y_idx, row in enumerate(matrix_data):
        for x_idx, value in enumerate(row):
            data_points.append(
                {
                    "x": x_idx,
                    "y": y_idx,
                    "value": value,
                    "color": _value_to_color(value, min_val, max_val, color_scale),
                }
            )

    config = {
        "type": "heatmap",
        "data": {
            "labels": {"x": x_labels, "y": y_labels},
            "datasets": [
                {
                    "label": title,
                    "data": data_points,
                    "backgroundColor": "context.dataset.data[context.dataIndex].color",
                    "borderWidth": 1,
                }
            ],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {"display": bool(title), "text": title},
                "legend": {"display": False},
            },
            "scales": {
                "x": {"type": "category", "labels": x_labels},
                "y": {"type": "category", "labels": y_labels},
            },
        },
    }

    return config


def create_timeline(
    events: List[Dict[str, Any]],
    title: str = "",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Create a timeline visualization.

    Args:
        events: List of event dictionaries with 'date' and 'label' keys
        title: Timeline title
        start_date: Timeline start date
        end_date: Timeline end date

    Returns:
        Dict[str, Any]: Timeline configuration

    Example:
        >>> from tenets.core.reporting.visualizer import create_timeline
        >>> timeline = create_timeline(
        ...     [
        ...         {"date": "2024-01-01", "label": "Project Start"},
        ...         {"date": "2024-02-15", "label": "First Release"}
        ...     ],
        ...     title="Project Timeline"
        ... )
    """
    # Sort events by date
    sorted_events = sorted(events, key=lambda e: e.get("date", ""))

    # Determine date range
    if not start_date and sorted_events:
        start_date = datetime.fromisoformat(sorted_events[0]["date"])
    if not end_date and sorted_events:
        end_date = datetime.fromisoformat(sorted_events[-1]["date"])

    if not start_date:
        start_date = datetime.now() - timedelta(days=30)
    if not end_date:
        end_date = datetime.now()

    # Create timeline data
    timeline_data = []
    for event in sorted_events:
        event_date = datetime.fromisoformat(event["date"])
        position = (event_date - start_date).days / max(1, (end_date - start_date).days)

        timeline_data.append(
            {
                "date": event["date"],
                "label": event.get("label", ""),
                "description": event.get("description", ""),
                "position": position * 100,  # Convert to percentage
                "type": event.get("type", "default"),
            }
        )

    config = {
        "type": "timeline",
        "data": timeline_data,
        "options": {
            "title": title,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "responsive": True,
        },
    }

    return config


def create_network_graph(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], title: str = "", layout: str = "force"
) -> Dict[str, Any]:
    """Create a network graph visualization.

    Args:
        nodes: List of node dictionaries with 'id' and 'label' keys
        edges: List of edge dictionaries with 'source' and 'target' keys
        title: Graph title
        layout: Layout algorithm (force, circular, hierarchical)

    Returns:
        Dict[str, Any]: Network graph configuration

    Example:
        >>> from tenets.core.reporting.visualizer import create_network_graph
        >>> graph = create_network_graph(
        ...     nodes=[
        ...         {"id": "A", "label": "Node A"},
        ...         {"id": "B", "label": "Node B"}
        ...     ],
        ...     edges=[
        ...         {"source": "A", "target": "B", "weight": 1}
        ...     ],
        ...     title="Dependency Graph"
        ... )
    """
    # Process nodes
    processed_nodes = []
    for node in nodes:
        processed_nodes.append(
            {
                "id": node.get("id"),
                "label": node.get("label", node.get("id")),
                "size": node.get("size", 10),
                "color": node.get("color", "#2563eb"),
                "x": node.get("x"),
                "y": node.get("y"),
            }
        )

    # Process edges
    processed_edges = []
    for edge in edges:
        processed_edges.append(
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "weight": edge.get("weight", 1),
                "color": edge.get("color", "#94a3b8"),
                "style": edge.get("style", "solid"),
            }
        )

    config = {
        "type": "network",
        "data": {"nodes": processed_nodes, "edges": processed_edges},
        "options": {
            "title": title,
            "layout": {"type": layout, "options": _get_layout_options(layout)},
            "interaction": {"dragNodes": True, "dragView": True, "zoomView": True},
            "physics": {"enabled": layout == "force"},
        },
    }

    return config


def create_treemap(
    hierarchical_data: Dict[str, Any],
    title: str = "",
    value_key: str = "value",
    label_key: str = "name",
) -> Dict[str, Any]:
    """Create a treemap visualization.

    Args:
        hierarchical_data: Hierarchical data structure
        title: Chart title
        value_key: Key for value in data
        label_key: Key for label in data

    Returns:
        Dict[str, Any]: Treemap configuration

    Example:
        >>> from tenets.core.reporting.visualizer import create_treemap
        >>> treemap = create_treemap(
        ...     {
        ...         "name": "root",
        ...         "children": [
        ...             {"name": "A", "value": 10},
        ...             {"name": "B", "value": 20}
        ...         ]
        ...     },
        ...     title="Code Distribution"
        ... )
    """
    # Flatten hierarchical data for visualization
    flat_data = _flatten_hierarchy(hierarchical_data, value_key, label_key)

    config = {
        "type": "treemap",
        "data": {
            "datasets": [
                {
                    "label": title,
                    "tree": flat_data,
                    "key": value_key,
                    "groups": ["parent", label_key],
                    "backgroundColor": _generate_treemap_colors(flat_data),
                }
            ]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "title": {"display": bool(title), "text": title},
                "legend": {"display": False},
            },
        },
    }

    return config


def _value_to_color(
    value: float, min_val: float, max_val: float, color_scale: str = "viridis"
) -> str:
    """Convert a value to a color based on scale.

    Args:
        value: Value to convert
        min_val: Minimum value in range
        max_val: Maximum value in range
        color_scale: Color scale name

    Returns:
        str: Hex color code
    """
    # Normalize value to 0-1 range
    if max_val == min_val:
        normalized = 0.5
    else:
        normalized = (value - min_val) / (max_val - min_val)

    # Simple color scales
    if color_scale == "viridis":
        # Simplified viridis approximation
        r = int(68 + (253 - 68) * normalized)
        g = int(1 + (231 - 1) * normalized)
        b = int(84 + (36 - 84) * normalized)
    elif color_scale == "coolwarm":
        # Blue to red
        if normalized < 0.5:
            r = int(59 + (247 - 59) * (normalized * 2))
            g = int(76 + (247 - 76) * (normalized * 2))
            b = 192
        else:
            r = 192
            g = int(247 - (247 - 76) * ((normalized - 0.5) * 2))
            b = int(247 - (247 - 59) * ((normalized - 0.5) * 2))
    else:
        # Default grayscale
        gray = int(255 * (1 - normalized))
        r = g = b = gray

    return f"#{r:02x}{g:02x}{b:02x}"


def _get_layout_options(layout: str) -> Dict[str, Any]:
    """Get layout options for network graph.

    Args:
        layout: Layout type

    Returns:
        Dict[str, Any]: Layout options
    """
    if layout == "force":
        return {
            "repulsion": {
                "centralGravity": 0.2,
                "springLength": 100,
                "springConstant": 0.05,
                "nodeDistance": 150,
                "damping": 0.09,
            }
        }
    elif layout == "circular":
        return {"radius": 200}
    elif layout == "hierarchical":
        return {"direction": "UD", "sortMethod": "hubsize", "shakeTowards": "leaves"}  # Up-Down
    else:
        return {}


def _flatten_hierarchy(
    data: Dict[str, Any], value_key: str, label_key: str, parent: str = None
) -> List[Dict[str, Any]]:
    """Flatten hierarchical data for treemap.

    Args:
        data: Hierarchical data
        value_key: Key for values
        label_key: Key for labels
        parent: Parent label

    Returns:
        List[Dict[str, Any]]: Flattened data
    """
    result = []

    current = {
        "parent": parent,
        label_key: data.get(label_key, "Unknown"),
        value_key: data.get(value_key, 0),
    }
    result.append(current)

    # Process children
    if "children" in data:
        for child in data["children"]:
            result.extend(_flatten_hierarchy(child, value_key, label_key, current[label_key]))

    return result


def _generate_treemap_colors(data: List[Dict[str, Any]]) -> List[str]:
    """Generate colors for treemap cells.

    Args:
        data: Treemap data

    Returns:
        List[str]: List of colors
    """
    # Simple color generation based on depth
    colors = []
    base_colors = [
        "#2563eb",
        "#8b5cf6",
        "#10b981",
        "#f59e0b",
        "#ef4444",
        "#06b6d4",
        "#ec4899",
        "#84cc16",
    ]

    for i, item in enumerate(data):
        colors.append(base_colors[i % len(base_colors)])

    return colors
