"""Visualization package for code analysis.

This package provides comprehensive visualization capabilities for code metrics,
analysis results, and development patterns. It includes support for various
chart types, terminal displays, and export formats.

The viz package is designed to be used by CLI commands and reporting modules
to create rich visualizations without duplicating visualization logic.

Example:
    Basic usage for creating visualizations::

        from tenets.viz import ComplexityVisualizer

        # Create visualizer
        viz = ComplexityVisualizer()

        # Generate chart from analysis data
        chart = viz.create_distribution_chart(complexity_data)

        # Display in terminal
        viz.display_terminal(complexity_data, show_details=True)

Available Visualizers:
    - ComplexityVisualizer: Visualize complexity metrics and distributions
    - ContributorVisualizer: Display contributor patterns and collaboration
    - CouplingVisualizer: Show module coupling and dependencies
    - DependencyVisualizer: Visualize dependency graphs and structures
    - HotspotVisualizer: Display code hotspots and risk areas
    - MomentumVisualizer: Track development velocity and trends

Display Utilities:
    - TerminalDisplay: Rich terminal output with tables and charts
    - ProgressDisplay: Progress indicators for long operations

Chart Types:
    - Bar, Line, Pie, Scatter, Radar, Gauge
    - Heatmap, Treemap, Network, Bubble
    - Stacked bars, Area charts, Timelines

Export Formats:
    - Terminal (ASCII/Unicode)
    - HTML (interactive charts)
    - JSON (chart configurations)
    - SVG/PNG (static images)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

# Base visualization classes
from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig, DisplayFormat

# Specialized visualizers
from .complexity import ComplexityVisualizer
from .contributors import ContributorVisualizer
from .coupling import CouplingVisualizer
from .dependencies import DependencyVisualizer

# Display utilities
from .displays import ProgressDisplay, TerminalDisplay
from .hotspots import HotspotVisualizer
from .momentum import MomentumVisualizer

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    # Base classes
    "BaseVisualizer",
    "ChartConfig",
    "ChartType",
    "ColorPalette",
    "DisplayConfig",
    "DisplayFormat",
    # Visualizers
    "ComplexityVisualizer",
    "ContributorVisualizer",
    "CouplingVisualizer",
    "DependencyVisualizer",
    "HotspotVisualizer",
    "MomentumVisualizer",
    # Display utilities
    "TerminalDisplay",
    "ProgressDisplay",
    # Factory functions
    "create_visualizer",
    "create_chart",
    "create_terminal_display",
    # Utility functions
    "detect_visualization_type",
    "export_visualization",
    "combine_visualizations",
]


def create_visualizer(
    viz_type: str,
    chart_config: Optional[ChartConfig] = None,
    display_config: Optional[DisplayConfig] = None,
) -> BaseVisualizer:
    """Factory function to create a visualizer instance.

    Creates the appropriate visualizer based on the specified type,
    with optional configuration for charts and display settings.

    Args:
        viz_type: Type of visualizer to create
            Options: 'complexity', 'contributors', 'coupling',
                    'dependencies', 'hotspots', 'momentum'
        chart_config: Optional chart configuration
        display_config: Optional display configuration

    Returns:
        BaseVisualizer: Configured visualizer instance

    Raises:
        ValueError: If viz_type is not recognized

    Example:
        >>> from tenets.viz import create_visualizer
        >>> viz = create_visualizer('complexity')
        >>> chart = viz.create_distribution_chart(data)

    Note:
        This factory function provides a convenient way to create
        visualizers without importing specific classes.
    """
    visualizers = {
        "complexity": ComplexityVisualizer,
        "contributors": ContributorVisualizer,
        "coupling": CouplingVisualizer,
        "dependencies": DependencyVisualizer,
        "hotspots": HotspotVisualizer,
        "momentum": MomentumVisualizer,
    }

    viz_class = visualizers.get(viz_type.lower())
    if not viz_class:
        raise ValueError(
            f"Unknown visualizer type: {viz_type}. Available types: {', '.join(visualizers.keys())}"
        )

    return viz_class(chart_config=chart_config, display_config=display_config)


def create_chart(
    chart_type: Union[str, ChartType],
    data: Dict[str, Any],
    title: Optional[str] = None,
    config: Optional[ChartConfig] = None,
) -> Dict[str, Any]:
    """Create a chart configuration from data.

    This is a convenience function that creates a chart without
    needing to instantiate a specific visualizer.

    Args:
        chart_type: Type of chart to create
            Options: 'bar', 'line', 'pie', 'scatter', 'radar',
                    'gauge', 'heatmap', 'treemap', 'network'
        data: Chart data with appropriate structure for chart type
            - Bar/Pie: {'labels': [...], 'values': [...]}
            - Line: {'labels': [...], 'datasets': [...]}
            - Scatter: {'points': [(x, y), ...]}
            - Heatmap: {'matrix': [[...]], 'x_labels': [...], 'y_labels': [...]}
        title: Optional chart title
        config: Optional chart configuration

    Returns:
        Dict[str, Any]: Chart configuration for rendering

    Example:
        >>> from tenets.viz import create_chart
        >>> chart = create_chart(
        ...     'bar',
        ...     {'labels': ['A', 'B', 'C'], 'values': [1, 2, 3]},
        ...     title='Sample Chart'
        ... )

    Note:
        The returned configuration is compatible with Chart.js
        and can be rendered in HTML or exported as JSON.
    """
    # Convert string to ChartType enum
    if isinstance(chart_type, str):
        try:
            chart_type = ChartType[chart_type.upper()]
        except KeyError:
            raise ValueError(f"Unknown chart type: {chart_type}")

    # Create base visualizer for chart generation
    viz = BaseVisualizer(chart_config=config)

    # Override title if provided
    if config is None:
        config = ChartConfig(type=chart_type, title=title or "")
    elif title:
        config.title = title

    return viz.create_chart(chart_type, data, config)


def create_terminal_display(config: Optional[DisplayConfig] = None) -> TerminalDisplay:
    """Create a terminal display instance.

    Factory function for creating configured terminal displays
    with rich formatting capabilities.

    Args:
        config: Optional display configuration
            Controls colors, unicode, width, truncation, etc.

    Returns:
        TerminalDisplay: Configured terminal display instance

    Example:
        >>> from tenets.viz import create_terminal_display
        >>> display = create_terminal_display()
        >>> display.display_table(
        ...     headers=['Name', 'Value'],
        ...     rows=[['A', 1], ['B', 2]]
        ... )

    Note:
        Terminal display automatically detects terminal capabilities
        and adjusts output accordingly.
    """
    return TerminalDisplay(config=config)


def detect_visualization_type(data: Any) -> str:
    """Auto-detect appropriate visualization type from data structure.

    Analyzes the structure and content of data to determine
    the most appropriate visualization type.

    Args:
        data: Data to analyze
            Can be dict, list, or other data structure

    Returns:
        str: Detected visualization type
            Options: 'complexity', 'contributors', 'coupling',
                    'dependencies', 'hotspots', 'momentum', 'custom'

    Example:
        >>> from tenets.viz import detect_visualization_type
        >>> data = {'complexity': {...}, 'avg_complexity': 15.2}
        >>> viz_type = detect_visualization_type(data)
        >>> print(viz_type)  # 'complexity'

    Note:
        This function uses heuristics based on common field names
        and data structures to detect the visualization type.
    """
    if isinstance(data, dict):
        # Check for known data structures
        indicators = {
            "complexity": [
                "complexity",
                "avg_complexity",
                "max_complexity",
                "complex_functions",
                "complexity_distribution",
            ],
            "contributors": [
                "contributors",
                "total_contributors",
                "active_contributors",
                "bus_factor",
                "contributor",
            ],
            "hotspots": [
                "hotspots",
                "risk_score",
                "change_frequency",
                "risk_level",
                "critical_count",
            ],
            "momentum": [
                "velocity",
                "momentum",
                "burndown",
                "sprint",
                "velocity_trend",
                "productivity",
            ],
            "dependencies": [
                "dependencies",
                "imports",
                "requires",
                "dependency_graph",
                "circular_dependencies",
            ],
            "coupling": [
                "coupling",
                "afferent_coupling",
                "efferent_coupling",
                "instability",
                "coupling_matrix",
            ],
        }

        for viz_type, fields in indicators.items():
            if any(field in data for field in fields):
                return viz_type

    elif isinstance(data, list) and data:
        # Check first item in list
        first = data[0] if isinstance(data[0], dict) else {}

        if any(field in first for field in ["complexity", "cyclomatic"]):
            return "complexity"
        elif any(field in first for field in ["author", "contributor", "commits"]):
            return "contributors"
        elif any(field in first for field in ["risk", "hotspot", "risk_score"]):
            return "hotspots"
        elif any(field in first for field in ["velocity", "throughput"]):
            return "momentum"

    return "custom"


def export_visualization(
    visualization: Dict[str, Any],
    output_path: str,
    format: str = "html",
    config: Optional[ChartConfig] = None,
) -> None:
    """Export visualization to file.

    Exports a visualization configuration to various formats
    including HTML, JSON, SVG, or PNG.

    Args:
        visualization: Visualization configuration
            Chart.js compatible configuration dict
        output_path: Path to output file
        format: Export format
            Options: 'html', 'json', 'svg', 'png'
        config: Optional chart configuration
            Used for HTML generation settings

    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written

    Example:
        >>> from tenets.viz import create_chart, export_visualization
        >>> chart = create_chart('bar', data)
        >>> export_visualization(chart, 'chart.html', format='html')

    Note:
        - HTML format creates standalone files with embedded Chart.js
        - JSON format exports raw configuration
        - SVG/PNG formats require additional dependencies
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(visualization, f, indent=2)

    elif format == "html":
        html_content = _generate_html_visualization(visualization, config)
        with open(output_path, "w") as f:
            f.write(html_content)

    elif format in ["svg", "png"]:
        raise NotImplementedError(
            f"{format.upper()} export requires additional dependencies. "
            "Use HTML or JSON format instead."
        )
    else:
        raise ValueError(f"Unsupported export format: {format}")


def combine_visualizations(
    visualizations: List[Dict[str, Any]], layout: str = "grid", title: Optional[str] = None
) -> Dict[str, Any]:
    """Combine multiple visualizations into a dashboard.

    Creates a composite visualization containing multiple charts
    arranged in a specified layout.

    Args:
        visualizations: List of visualization configurations
        layout: Layout style
            Options: 'grid', 'vertical', 'horizontal', 'tabs'
        title: Optional dashboard title

    Returns:
        Dict[str, Any]: Combined visualization configuration

    Example:
        >>> from tenets.viz import create_chart, combine_visualizations
        >>> chart1 = create_chart('bar', data1)
        >>> chart2 = create_chart('line', data2)
        >>> dashboard = combine_visualizations(
        ...     [chart1, chart2],
        ...     layout='grid',
        ...     title='Analysis Dashboard'
        ... )

    Note:
        Combined visualizations are best rendered in HTML format
        for proper layout and interactivity.
    """
    return {
        "type": "dashboard",
        "title": title or "Dashboard",
        "layout": layout,
        "visualizations": visualizations,
        "options": {"responsive": True, "maintainAspectRatio": False},
    }


def _generate_html_visualization(
    visualization: Dict[str, Any], config: Optional[ChartConfig] = None
) -> str:
    """Generate HTML content for visualization.

    Internal function to create standalone HTML with embedded
    Chart.js and visualization configuration.

    Args:
        visualization: Visualization configuration
        config: Optional chart configuration

    Returns:
        str: Complete HTML content
    """
    import json

    if config is None:
        config = ChartConfig(type=ChartType.BAR, title="Visualization")

    # Handle dashboard layout
    if visualization.get("type") == "dashboard":
        return _generate_dashboard_html(visualization, config)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            padding: 30px;
            width: 100%;
            max-width: {config.width + 60}px;
        }}
        h1 {{
            color: #2d3748;
            margin-bottom: 30px;
            font-size: 24px;
            font-weight: 600;
            text-align: center;
        }}
        .chart-container {{
            position: relative;
            height: {config.height}px;
            margin: 20px 0;
        }}
        .metadata {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{config.title}</h1>
        <div class="chart-container">
            <canvas id="chart"></canvas>
        </div>
        <div class="metadata">
            Generated by Tenets Visualization System
        </div>
    </div>
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const chartConfig = {json.dumps(visualization)};

        // Ensure responsive settings
        if (!chartConfig.options) chartConfig.options = {{}};
        chartConfig.options.responsive = true;
        chartConfig.options.maintainAspectRatio = false;

        const chart = new Chart(ctx, chartConfig);
    </script>
</body>
</html>"""


def _generate_dashboard_html(dashboard: Dict[str, Any], config: ChartConfig) -> str:
    """Generate HTML for dashboard with multiple visualizations.

    Internal function to create dashboard layout with multiple charts.

    Args:
        dashboard: Dashboard configuration
        config: Chart configuration

    Returns:
        str: Dashboard HTML content
    """
    import json

    visualizations = dashboard.get("visualizations", [])
    layout = dashboard.get("layout", "grid")

    # Generate chart containers
    chart_html = []
    chart_scripts = []

    for i, viz in enumerate(visualizations):
        chart_id = f"chart{i}"

        # Determine grid layout
        if layout == "grid":
            width_class = "half" if len(visualizations) > 1 else "full"
        elif layout == "vertical":
            width_class = "full"
        else:
            width_class = "half"

        chart_html.append(
            f"""
        <div class="chart-item {width_class}">
            <div class="chart-container">
                <canvas id="{chart_id}"></canvas>
            </div>
        </div>
        """
        )

        chart_scripts.append(
            f"""
        const ctx{i} = document.getElementById('{chart_id}').getContext('2d');
        const chart{i} = new Chart(ctx{i}, {json.dumps(viz)});
        """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard.get("title", "Dashboard")}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f7fafc;
            padding: 20px;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2d3748;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: 600;
        }}
        .charts-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .chart-item {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            padding: 20px;
        }}
        .chart-item.full {{
            width: 100%;
        }}
        .chart-item.half {{
            width: calc(50% - 10px);
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        @media (max-width: 768px) {{
            .chart-item.half {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>{dashboard.get("title", "Dashboard")}</h1>
        <div class="charts-grid">
            {"".join(chart_html)}
        </div>
    </div>
    <script>
        {"".join(chart_scripts)}
    </script>
</body>
</html>"""
