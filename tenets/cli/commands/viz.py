"""Viz command implementation.

This command provides standalone visualization capabilities for pre-analyzed
data, allowing users to create charts and visualizations from JSON, CSV, or
other data files without re-running analysis.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import click

from tenets.utils.logger import get_logger
from tenets.viz import (
    BaseVisualizer,
    ChartConfig,
    ChartType,
    ComplexityVisualizer,
    ContributorVisualizer,
    CouplingVisualizer,
    DependencyVisualizer,
    HotspotVisualizer,
    MomentumVisualizer,
)


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--type",
    "-t",
    "viz_type",
    type=click.Choice(
        [
            "auto",
            "complexity",
            "contributors",
            "coupling",
            "dependencies",
            "hotspots",
            "momentum",
            "custom",
        ]
    ),
    default="auto",
    help="Visualization type",
)
@click.option(
    "--chart",
    "-c",
    type=click.Choice(
        ["bar", "line", "pie", "scatter", "radar", "gauge", "heatmap", "network", "treemap"]
    ),
    help="Chart type for custom visualization",
)
@click.option("--output", "-o", type=click.Path(), help="Output file for chart")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["terminal", "html", "svg", "png", "json"]),
    default="terminal",
    help="Output format",
)
@click.option("--title", help="Chart title")
@click.option("--width", type=int, default=800, help="Chart width in pixels")
@click.option("--height", type=int, default=400, help="Chart height in pixels")
@click.option("--x-field", help="Field name for X axis (custom viz)")
@click.option("--y-field", help="Field name for Y axis (custom viz)")
@click.option("--value-field", help="Field name for values (custom viz)")
@click.option("--label-field", help="Field name for labels (custom viz)")
@click.option("--limit", type=int, help="Limit number of data points")
@click.option("--interactive", is_flag=True, help="Launch interactive visualization")
@click.pass_context
def viz(
    ctx,
    input_file: str,
    viz_type: str,
    chart: Optional[str],
    output: Optional[str],
    format: str,
    title: Optional[str],
    width: int,
    height: int,
    x_field: Optional[str],
    y_field: Optional[str],
    value_field: Optional[str],
    label_field: Optional[str],
    limit: Optional[int],
    interactive: bool,
):
    """Create visualizations from data files.

    This command generates visualizations from pre-analyzed data files
    without needing to re-run analysis. It supports JSON and CSV input
    files and can auto-detect the appropriate visualization type.

    Examples:
        tenets viz analysis_results.json
        tenets viz metrics.csv --type=custom --chart=bar --label-field=name --value-field=score
        tenets viz hotspots.json --format=html --output=hotspots.html
        tenets viz data.json --interactive
    """
    logger = get_logger(__name__)
    config = ctx.obj["config"]

    input_path = Path(input_file)
    logger.info(f"Loading data from: {input_path}")

    try:
        # Load data
        data = _load_data(input_path)

        # Auto-detect visualization type if needed
        if viz_type == "auto":
            viz_type = _detect_viz_type(data)
            logger.info(f"Auto-detected visualization type: {viz_type}")

        # Create visualization
        if viz_type == "custom":
            visualization = _create_custom_viz(
                data, chart, x_field, y_field, value_field, label_field, title, limit
            )
        else:
            visualization = _create_typed_viz(viz_type, data, chart, title, limit)

        # Configure chart
        if visualization:
            chart_config = ChartConfig(
                type=ChartType[chart.upper()] if chart else ChartType.BAR,
                title=title or _generate_title(viz_type),
                width=width,
                height=height,
                interactive=interactive,
            )

            # Apply configuration
            if isinstance(visualization, dict):
                visualization["options"] = {
                    **visualization.get("options", {}),
                    "width": width,
                    "height": height,
                }

        # Output visualization
        if format == "terminal":
            _display_terminal(viz_type, data, visualization)
        elif format == "json":
            _output_json(visualization, output)
        elif format in ["html", "svg", "png"]:
            _generate_chart_file(visualization, format, output, chart_config)

        # Launch interactive mode if requested
        if interactive:
            _launch_interactive(visualization, data, viz_type)

        # Success message
        if output:
            click.echo(f"✅ Visualization saved to: {output}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise click.ClickException(str(e))


def _load_data(input_path: Path) -> Any:
    """Load data from file.

    Args:
        input_path: Path to input file

    Returns:
        Loaded data
    """
    if input_path.suffix.lower() == ".json":
        with open(input_path) as f:
            return json.load(f)
    elif input_path.suffix.lower() == ".csv":
        with open(input_path) as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        # Try JSON first
        try:
            with open(input_path) as f:
                return json.load(f)
        except:
            # Try CSV
            with open(input_path) as f:
                reader = csv.DictReader(f)
                return list(reader)


def _detect_viz_type(data: Any) -> str:
    """Auto-detect visualization type from data structure.

    Args:
        data: Input data

    Returns:
        Detected visualization type
    """
    if isinstance(data, dict):
        # Check for known data structures
        if "complexity" in data or "avg_complexity" in data:
            return "complexity"
        elif "contributors" in data or "total_contributors" in data:
            return "contributors"
        elif "hotspots" in data or "risk_score" in data:
            return "hotspots"
        elif "velocity" in data or "momentum" in data:
            return "momentum"
        elif "dependencies" in data or "coupling" in data:
            return "dependencies"
        elif "afferent_coupling" in data or "efferent_coupling" in data:
            return "coupling"
    elif isinstance(data, list) and data:
        # Check first item
        first = data[0]
        if "complexity" in first:
            return "complexity"
        elif "author" in first or "contributor" in first:
            return "contributors"
        elif "risk" in first or "hotspot" in first:
            return "hotspots"

    return "custom"


def _create_typed_viz(
    viz_type: str, data: Any, chart: Optional[str], title: Optional[str], limit: Optional[int]
) -> Dict[str, Any]:
    """Create visualization based on type.

    Args:
        viz_type: Visualization type
        data: Input data
        chart: Chart type override
        title: Chart title
        limit: Data limit

    Returns:
        Visualization configuration
    """
    if viz_type == "complexity":
        viz = ComplexityVisualizer()
        if "distribution" in data:
            return viz.create_distribution_chart(data)
        elif "complex_items" in data:
            return viz.create_top_complex_chart(data["complex_items"], limit or 10)
        elif isinstance(data, list):
            # List of complexity data
            return viz.create_top_complex_chart(data, limit or 10)

    elif viz_type == "contributors":
        viz = ContributorVisualizer()
        if "contributors" in data:
            return viz.create_contribution_chart(data["contributors"], limit=limit or 10)
        elif isinstance(data, list):
            return viz.create_contribution_chart(data, limit=limit or 10)

    elif viz_type == "hotspots":
        viz = HotspotVisualizer()
        if "hotspots" in data:
            return viz.create_hotspot_bubble(data["hotspots"], limit or 50)
        elif isinstance(data, list):
            return viz.create_hotspot_bubble(data, limit or 50)

    elif viz_type == "momentum":
        viz = MomentumVisualizer()
        if "velocity_data" in data:
            return viz.create_velocity_chart(data["velocity_data"])
        elif "burndown" in data:
            return viz.create_burndown_chart(data["burndown"])

    elif viz_type == "dependencies":
        viz = DependencyVisualizer()
        if "dependencies" in data:
            return viz.create_dependency_graph(data["dependencies"])
        elif "circular_dependencies" in data:
            return viz.create_circular_dependencies_chart(data["circular_dependencies"])

    elif viz_type == "coupling":
        viz = CouplingVisualizer()
        if "coupling_data" in data:
            return viz.create_coupling_network(data["coupling_data"])
        elif "modules" in data:
            return viz.create_afferent_efferent_chart(data["modules"])

    return {}


def _create_custom_viz(
    data: Any,
    chart: Optional[str],
    x_field: Optional[str],
    y_field: Optional[str],
    value_field: Optional[str],
    label_field: Optional[str],
    title: Optional[str],
    limit: Optional[int],
) -> Dict[str, Any]:
    """Create custom visualization from data.

    Args:
        data: Input data
        chart: Chart type
        x_field: X axis field
        y_field: Y axis field
        value_field: Value field
        label_field: Label field
        title: Chart title
        limit: Data limit

    Returns:
        Visualization configuration
    """
    viz = BaseVisualizer()

    # Determine chart type
    if not chart:
        if x_field and y_field:
            chart = "scatter"
        elif value_field and label_field:
            chart = "bar"
        else:
            chart = "bar"

    chart_type = ChartType[chart.upper()]

    # Extract data based on fields
    if isinstance(data, list):
        # List of records
        if limit:
            data = data[:limit]

        if chart_type in [ChartType.BAR, ChartType.PIE]:
            labels = [str(d.get(label_field, "")) for d in data] if label_field else []
            values = [float(d.get(value_field, 0)) for d in data] if value_field else []

            return viz.create_chart(
                chart_type,
                {"labels": labels, "values": values},
                ChartConfig(type=chart_type, title=title or "Custom Chart"),
            )

        elif chart_type == ChartType.SCATTER:
            points = []
            for d in data:
                x = float(d.get(x_field, 0)) if x_field else 0
                y = float(d.get(y_field, 0)) if y_field else 0
                points.append((x, y))

            return viz.create_chart(
                chart_type,
                {"points": points},
                ChartConfig(type=chart_type, title=title or "Custom Scatter"),
            )

        elif chart_type == ChartType.LINE:
            # Assume data is time series
            labels = [str(d.get(label_field or x_field, "")) for d in data]
            values = [float(d.get(value_field or y_field, 0)) for d in data]

            return viz.create_chart(
                chart_type,
                {"labels": labels, "datasets": [{"data": values}]},
                ChartConfig(type=chart_type, title=title or "Custom Line"),
            )

    elif isinstance(data, dict):
        # Dictionary data
        if chart_type in [ChartType.BAR, ChartType.PIE]:
            # Use keys as labels, values as values
            items = list(data.items())
            if limit:
                items = items[:limit]

            labels = [str(k) for k, _ in items]
            values = [float(v) if isinstance(v, (int, float)) else 0 for _, v in items]

            return viz.create_chart(
                chart_type,
                {"labels": labels, "values": values},
                ChartConfig(type=chart_type, title=title or "Custom Chart"),
            )

    return {}


def _display_terminal(viz_type: str, data: Any, visualization: Dict[str, Any]) -> None:
    """Display visualization in terminal.

    Args:
        viz_type: Visualization type
        data: Original data
        visualization: Visualization configuration
    """
    if viz_type == "complexity":
        viz = ComplexityVisualizer()
        viz.display_terminal(data, show_details=True)
    elif viz_type == "contributors":
        viz = ContributorVisualizer()
        viz.display_terminal(data, show_details=True)
    elif viz_type == "hotspots":
        viz = HotspotVisualizer()
        viz.display_terminal(data, show_details=True)
    elif viz_type == "momentum":
        viz = MomentumVisualizer()
        viz.display_terminal(data, show_details=True)
    elif viz_type == "dependencies":
        viz = DependencyVisualizer()
        viz.display_terminal(data, show_details=True)
    elif viz_type == "coupling":
        viz = CouplingVisualizer()
        viz.display_terminal(data, show_details=True)
    else:
        # Custom visualization - show summary
        click.echo("Custom Visualization Generated")
        click.echo(f"Type: {visualization.get('type', 'unknown')}")
        if "data" in visualization:
            dataset_count = len(visualization["data"].get("datasets", []))
            click.echo(f"Datasets: {dataset_count}")


def _output_json(visualization: Dict[str, Any], output: Optional[str]) -> None:
    """Output visualization as JSON.

    Args:
        visualization: Visualization configuration
        output: Output path
    """
    if output:
        with open(output, "w") as f:
            json.dump(visualization, f, indent=2)
    else:
        click.echo(json.dumps(visualization, indent=2))


def _generate_chart_file(
    visualization: Dict[str, Any], format: str, output: Optional[str], config: ChartConfig
) -> None:
    """Generate chart file.

    Args:
        visualization: Visualization configuration
        format: Output format
        output: Output path
        config: Chart configuration
    """
    output_path = Path(output) if output else Path(f"chart.{format}")

    if format == "html":
        html_content = _generate_html_chart(visualization, config)
        with open(output_path, "w") as f:
            f.write(html_content)

    elif format == "svg":
        # For SVG, we'd need a different approach
        # This is a placeholder
        click.echo("SVG export not yet implemented")
        return

    elif format == "png":
        # For PNG, we'd need a headless browser or chart rendering library
        click.echo("PNG export not yet implemented")
        return


def _generate_html_chart(visualization: Dict[str, Any], config: ChartConfig) -> str:
    """Generate standalone HTML with chart.

    Args:
        visualization: Chart configuration
        config: Chart configuration

    Returns:
        HTML content
    """
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{config.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: {config.width + 40}px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 20px 0;
            color: #333;
        }}
        canvas {{
            max-width: 100%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{config.title}</h1>
        <canvas id="chart" width="{config.width}" height="{config.height}"></canvas>
    </div>
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const chart = new Chart(ctx, {json.dumps(visualization)});
    </script>
</body>
</html>"""


def _launch_interactive(visualization: Dict[str, Any], data: Any, viz_type: str) -> None:
    """Launch interactive visualization mode.

    Args:
        visualization: Visualization configuration
        data: Original data
        viz_type: Visualization type
    """
    click.echo("\n🚀 Launching interactive mode...")

    # This would typically launch a local web server
    # For now, just save and open HTML
    import tempfile
    import webbrowser

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        config = ChartConfig(
            title=f"Interactive {viz_type.title()} Visualization",
            width=1200,
            height=600,
            interactive=True,
        )
        html = _generate_html_chart(visualization, config)
        f.write(html.encode())

        # Open in browser
        webbrowser.open(f"file://{f.name}")
        click.echo(f"Opened in browser: {f.name}")


def _generate_title(viz_type: str) -> str:
    """Generate default title for visualization type.

    Args:
        viz_type: Visualization type

    Returns:
        Default title
    """
    titles = {
        "complexity": "Complexity Analysis",
        "contributors": "Contributor Analysis",
        "hotspots": "Code Hotspots",
        "momentum": "Development Momentum",
        "dependencies": "Dependency Analysis",
        "coupling": "Coupling Analysis",
        "custom": "Data Visualization",
    }
    return titles.get(viz_type, "Visualization")
