"""Visualization system for Tenets.

This package provides various visualization capabilities for understanding
codebases including dependency graphs, complexity heatmaps, coupling analysis,
and contributor patterns.

Main components:
- DependencyGraph: Visualize import dependencies
- ComplexityHeatmap: Show code complexity patterns
- CouplingGraph: Identify files that change together
- ContributorGraph: Analyze team dynamics

Example usage:
    >>> from tenets.viz import create_dependency_graph, visualize_complexity
    >>>
    >>> # Create dependency graph
    >>> graph = create_dependency_graph(files, format="html")
    >>> graph.render("dependencies.html")
    >>>
    >>> # Show complexity heatmap
    >>> heatmap = visualize_complexity(files, threshold=10)
    >>> print(heatmap.render())  # ASCII output
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import visualization components (actual available symbols)
from .base import (
    BaseVisualizer,
    ChartConfig,
    ChartType,
    ColorPalette,
    DisplayConfig,
    DisplayFormat,
)
from .complexity import ComplexityVisualizer
from .contributors import ContributorVisualizer
from .coupling import CouplingVisualizer
from .dependencies import DependencyVisualizer
from .displays import ProgressDisplay, TerminalDisplay
from .hotspots import HotspotVisualizer
from .momentum import MomentumVisualizer

# Version info
__version__ = "0.1.0"

# These will be set after defining check_dependencies() below
MATPLOTLIB_AVAILABLE = False
NETWORKX_AVAILABLE = False
PLOTLY_AVAILABLE = False

# Public API exports
__all__ = [
    # Base components
    "BaseVisualizer",
    "ChartConfig",
    "ChartType",
    "ColorPalette",
    "DisplayConfig",
    "DisplayFormat",
    "TerminalDisplay",
    "ProgressDisplay",
    # Visualizers
    "DependencyVisualizer",
    "ComplexityVisualizer",
    "CouplingVisualizer",
    "ContributorVisualizer",
    "HotspotVisualizer",
    "MomentumVisualizer",
    # Convenience wrappers
    "visualize_dependencies",
    "visualize_complexity",
    "visualize_coupling",
    "visualize_contributors",
    "create_visualization",
    # Factories and utilities expected by tests/public API
    "create_visualizer",
    "create_chart",
    "create_terminal_display",
    "detect_visualization_type",
    "export_visualization",
    "combine_visualizations",
    # Private HTML helpers (used in tests)
    "_generate_html_visualization",
    "_generate_dashboard_html",
    # Utilities
    "check_dependencies",
    "get_available_formats",
    "install_viz_dependencies",
]


def visualize_dependencies(
    dependencies: Dict[str, List[str]],
    output: Optional[Union[str, Path]] = None,
    format: str = "json",
    max_nodes: int = 100,
    highlight_circular: bool = True,
    title: Optional[str] = None,
) -> Union[Dict[str, Any], None]:
    """Create and optionally export a dependency graph configuration.

    Args:
        dependencies: Mapping of module -> list of dependencies
        output: Optional output path; when provided, writes chart to file
        format: Export format when output is provided (json or html)
        max_nodes: Max nodes to include in the graph
        highlight_circular: Highlight circular dependencies
        title: Optional chart title

    Returns:
        Chart configuration dict if output is None, otherwise None
    """
    config = ChartConfig(type=ChartType.NETWORK, title=title or "Dependency Graph")
    viz = DependencyVisualizer(chart_config=config)
    chart = viz.create_dependency_graph(
        dependencies, highlight_circular=highlight_circular, max_nodes=max_nodes
    )

    if output:
        viz.export_chart(chart, Path(output), format=(format or "json"))
        return None
    return chart


def visualize_complexity(
    file_complexities: Dict[str, List[int]],
    output: Optional[Union[str, Path]] = None,
    format: str = "json",
    max_functions: int = 50,
    title: Optional[str] = None,
) -> Union[Dict[str, Any], None]:
    """Create and optionally export a complexity heatmap configuration.

    Args:
        file_complexities: Mapping of file path -> list of function complexities
        output: Optional output path; when provided, writes chart to file
        format: Export format when output is provided (json or html)
        max_functions: Maximum functions per file to display
        title: Optional chart title

    Returns:
        Chart configuration dict if output is None, otherwise None
    """
    config = ChartConfig(type=ChartType.HEATMAP, title=title or "Code Complexity Heatmap")
    viz = ComplexityVisualizer(chart_config=config)
    chart = viz.create_complexity_heatmap(file_complexities, max_functions=max_functions)

    if output:
        viz.export_chart(chart, Path(output), format=(format or "json"))
        return None
    return chart


def visualize_coupling(
    coupling_data: Dict[str, Dict[str, int]],
    output: Optional[Union[str, Path]] = None,
    format: str = "json",
    min_coupling: int = 2,
    max_nodes: int = 50,
    title: Optional[str] = None,
) -> Union[Dict[str, Any], None]:
    """Create and optionally export a module coupling network configuration.

    Args:
        coupling_data: Mapping of module -> {coupled_module: strength}
        output: Optional output path; when provided, writes chart to file
        format: Export format when output is provided (json or html)
        min_coupling: Minimum coupling strength to include
        max_nodes: Maximum nodes to include
        title: Optional chart title

    Returns:
        Chart configuration dict if output is None, otherwise None
    """
    config = ChartConfig(type=ChartType.NETWORK, title=title or "Module Coupling Network")
    viz = CouplingVisualizer(chart_config=config)
    chart = viz.create_coupling_network(
        coupling_data, min_coupling=min_coupling, max_nodes=max_nodes
    )

    if output:
        viz.export_chart(chart, Path(output), format=(format or "json"))
        return None
    return chart


def visualize_contributors(
    contributors: List[Dict[str, Any]],
    output: Optional[Union[str, Path]] = None,
    format: str = "json",
    metric: str = "commits",
    limit: int = 10,
    title: Optional[str] = None,
) -> Union[Dict[str, Any], None]:
    """Create and optionally export a contributor chart configuration.

    Args:
        contributors: List of contributor dicts with metrics (e.g., commits)
        output: Optional output path; when provided, writes chart to file
        format: Export format when output is provided (json or html)
        metric: Metric to visualize (commits, lines, files)
        limit: Max contributors to show
        title: Optional chart title

    Returns:
        Chart configuration dict if output is None, otherwise None
    """
    config = ChartConfig(type=ChartType.BAR, title=title or "Commits by Contributor")
    viz = ContributorVisualizer(chart_config=config)
    chart = viz.create_contribution_chart(contributors, metric=metric, limit=limit)

    if output:
        viz.export_chart(chart, Path(output), format=(format or "json"))
        return None
    return chart


def create_visualization(
    data: Any,
    viz_type: str,
    output: Optional[Union[str, Path]] = None,
    format: str = "json",
    **kwargs,
) -> Union[Dict[str, Any], None]:
    """Create any type of visualization.

    Universal function for creating visualizations based on type.

    Args:
        data: Input data (files, commits, etc.)
        viz_type: Type of visualization (deps, complexity, coupling, contributors)
        output: Output path
        format: Output format
        **kwargs: Additional arguments for specific visualization

    Returns:
        Rendered content if output is None, otherwise None

    Example:
        >>> from tenets.viz import create_visualization
        >>>
        >>> # Create dependency graph
        >>> viz = create_visualization(
        ...     files,
        ...     "deps",
        ...     format="svg",
        ...     max_nodes=50
        ... )
    """
    viz_map = {
        "deps": visualize_dependencies,
        "dependencies": visualize_dependencies,
        "complexity": visualize_complexity,
        "coupling": visualize_coupling,
        "contributors": visualize_contributors,
    }

    viz_func = viz_map.get(viz_type.lower())
    if not viz_func:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    return viz_func(data, output=output, format=format, **kwargs)


def create_visualizer(
    viz_type: str,
    chart_config: Optional[ChartConfig] = None,
    display_config: Optional[DisplayConfig] = None,
):
    """Factory to create a visualizer by type.

    Args:
        viz_type: Type name (complexity, contributors, coupling, dependencies, hotspots)
        chart_config: Optional chart configuration
        display_config: Optional display configuration

    Returns:
        A visualizer instance

    Raises:
        ValueError: If type is unknown
    """
    mapping = {
        "complexity": ComplexityVisualizer,
        "contributors": ContributorVisualizer,
        "coupling": CouplingVisualizer,
        "dependencies": DependencyVisualizer,
        "deps": DependencyVisualizer,
        "hotspots": HotspotVisualizer,
    }
    key = (viz_type or "").lower()
    cls = mapping.get(key)
    if not cls:
        raise ValueError(f"Unknown visualizer type: {viz_type}")
    return cls(chart_config=chart_config, display_config=display_config)


def _normalize_chart_type(chart_type: Union[str, ChartType]) -> ChartType:
    if isinstance(chart_type, ChartType):
        return chart_type
    try:
        return ChartType[str(chart_type).upper()]
    except KeyError:
        # Some aliases
        aliases = {
            "stacked-bar": ChartType.STACKED_BAR,
            "horizontal-bar": ChartType.HORIZONTAL_BAR,
        }
        if str(chart_type).lower() in aliases:
            return aliases[str(chart_type).lower()]
        raise ValueError("Unknown chart type: %s" % chart_type)


def create_chart(
    chart_type: Union[str, ChartType],
    data: Dict[str, Any],
    *,
    title: Optional[str] = None,
    config: Optional[ChartConfig] = None,
) -> Dict[str, Any]:
    """Create a chart configuration using BaseVisualizer defaults.

    Accepts either a ChartType enum or a string chart type.
    """
    ct = _normalize_chart_type(chart_type)
    cfg = config or ChartConfig(type=ct, title=title or "")
    # ensure title propagated if provided
    if title:
        cfg.title = title
    base = BaseVisualizer(chart_config=cfg)
    return base.create_chart(ct, data, config=cfg)


def create_terminal_display(config: Optional[DisplayConfig] = None) -> TerminalDisplay:
    """Create a TerminalDisplay, optionally with custom DisplayConfig."""
    return TerminalDisplay(config)


def detect_visualization_type(data: Any) -> str:
    """Best-effort detection of visualization type from data structure."""

    def has_keys(d: Dict[str, Any], keys: List[str]) -> bool:
        return all(k in d for k in keys)

    if isinstance(data, list):
        if not data:
            return "custom"
        first = data[0]
        if isinstance(first, dict):
            if any(k in first for k in ("complexity", "cyclomatic")):
                return "complexity"
            if any(k in first for k in ("author", "contributor")):
                return "contributors"
        return "custom"

    if isinstance(data, dict):
        checks = [
            ("complexity", ["complexity", "avg_complexity", "complex_functions"]),
            ("contributors", ["contributors", "total_contributors", "bus_factor"]),
            ("hotspots", ["hotspots", "risk_score", "critical_count"]),
            ("momentum", ["velocity", "sprint", "velocity_trend"]),
            (
                "dependencies",
                ["dependencies", "circular_dependencies", "dependency_graph"],
            ),
            ("coupling", ["coupling", "afferent_coupling", "instability"]),
        ]
        for name, keys in checks:
            if has_keys(data, keys):
                return name
        return "custom"

    return "custom"


def export_visualization(
    visualization: Dict[str, Any],
    output: Union[str, Path],
    *,
    format: str = "json",
    config: Optional[ChartConfig] = None,
) -> Path:
    """Export a visualization or dashboard to JSON or HTML.

    SVG export is not implemented to keep dependencies optional.
    """
    out = Path(output)
    fmt = (format or "json").lower()
    if fmt == "json":
        with open(out, "w") as f:
            json.dump(visualization, f, indent=2)
        return out
    if fmt == "html":
        if visualization.get("type") == "dashboard":
            html = _generate_dashboard_html(
                visualization, config or ChartConfig(type=ChartType.BAR)
            )
        else:
            html = _generate_html_visualization(
                visualization, config or ChartConfig(type=ChartType.BAR)
            )
        with open(out, "w") as f:
            f.write(html)
        return out
    if fmt == "svg":
        raise NotImplementedError("SVG export requires additional rendering backends")
    raise ValueError("Unsupported export format: %s" % format)


def combine_visualizations(
    visualizations: List[Dict[str, Any]],
    *,
    layout: str = "grid",
    title: str = "Dashboard",
) -> Dict[str, Any]:
    """Combine multiple visualization configs into a simple dashboard schema."""
    return {
        "type": "dashboard",
        "title": title,
        "layout": layout,
        "visualizations": list(visualizations),
        "options": {"responsive": True},
    }


def _generate_html_visualization(visualization: Dict[str, Any], config: ChartConfig) -> str:
    """Generate standalone HTML for a single chart config.

    Includes a visible reference to 'Chart.js' to satisfy tests and users.
    """
    width_px = (config.width or 800) + 60  # small padding for container
    height_px = config.height or 400
    viz_json = json.dumps(visualization)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\">
    <title>{config.title or "Chart"}</title>
    <!-- Powered by Chart.js -->
    <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
    <style>
      .chart-container {{ width: {width_px}px; }}
      canvas {{ width: 100%; height: {height_px}px; }}
    </style>
</head>
<body>
    <div class=\"chart-container\">
      <canvas id=\"chart0\"></canvas>
    </div>
    <script>
      const ctx0 = document.getElementById('chart0').getContext('2d');
      const cfg0 = {viz_json};
      new Chart(ctx0, cfg0);
    </script>
    <!-- Chart.js loaded above -->
    <!-- Chart.js -->
  </body>
  </html>"""


def _generate_dashboard_html(dashboard: Dict[str, Any], config: ChartConfig) -> str:
    """Generate HTML for a simple dashboard of multiple charts."""
    layout = dashboard.get("layout", "grid")
    visualizations = dashboard.get("visualizations", [])
    # Simple CSS grid/vertical layouts
    container_class = "charts-grid" if layout == "grid" else "charts-vertical"
    css_layout = (
        ".charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }"
        if layout == "grid"
        else ".charts-vertical { display: flex; flex-direction: column; gap: 16px; }"
    )
    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"  <title>{dashboard.get('title', 'Dashboard')}</title>",
        '  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>',
        "  <style>",
        f"    {css_layout}",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{dashboard.get('title', 'Dashboard')}</h1>",
        f'  <div class="{container_class}">',
    ]
    # Canvases
    for i, _ in enumerate(visualizations):
        parts.append(f'    <canvas id="chart{i}"></canvas>')
    parts += [
        "  </div>",
        "  <script>",
    ]
    # JS initializers
    for i, viz in enumerate(visualizations):
        vjson = json.dumps(viz)
        parts.append(f"    const ctx{i} = document.getElementById('chart{i}').getContext('2d');")
        parts.append(f"    const cfg{i} = {vjson};")
        parts.append(f"    new Chart(ctx{i}, cfg{i});")
    parts += [
        "  </script>",
        "  <!-- Chart.js -->",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def check_dependencies() -> Dict[str, bool]:
    """Check which visualization libraries are available.

    Returns:
        Dictionary mapping library names to availability

    Example:
        >>> from tenets.viz import check_dependencies
        >>> deps = check_dependencies()
        >>> if deps['plotly']:
        >>>     print("Interactive visualizations available!")
    """
    try:
        import matplotlib

        matplotlib_available = True
    except ImportError:
        matplotlib_available = False

    try:
        import networkx

        networkx_available = True
    except ImportError:
        networkx_available = False

    try:
        import plotly

        plotly_available = True
    except ImportError:
        plotly_available = False

    deps = {
        "matplotlib": matplotlib_available,
        "networkx": networkx_available,
        "plotly": plotly_available,
        "all": matplotlib_available and networkx_available and plotly_available,
    }

    # Update module-level flags for convenience
    global MATPLOTLIB_AVAILABLE, NETWORKX_AVAILABLE, PLOTLY_AVAILABLE
    MATPLOTLIB_AVAILABLE = deps["matplotlib"]
    NETWORKX_AVAILABLE = deps["networkx"]
    PLOTLY_AVAILABLE = deps["plotly"]

    return deps


def get_available_formats() -> List[str]:
    """Get list of available output formats based on installed libraries.

    Returns:
        List of format names

    Example:
        >>> from tenets.viz import get_available_formats
        >>> formats = get_available_formats()
        >>> print(f"Available formats: {', '.join(formats)}")
    """
    formats = ["ascii", "json"]  # Always available

    deps = check_dependencies()

    if deps["matplotlib"]:
        formats.extend(["svg", "png"])

    if deps["plotly"]:
        formats.append("html")

    return formats


def install_viz_dependencies():
    """Helper to install visualization dependencies.

    Provides instructions for installing optional visualization libraries.

    Example:
        >>> from tenets.viz import install_viz_dependencies
        >>> install_viz_dependencies()
    """
    print("To enable all visualization features, install optional dependencies:")
    print()
    print("  pip install tenets[viz]")
    print()
    print("Or install individual libraries:")
    print("  pip install matplotlib  # For SVG/PNG output")
    print("  pip install networkx    # For graph layouts")
    print("  pip install plotly      # For interactive HTML")
    print()

    deps = check_dependencies()
    if deps["all"]:
        print("✓ All visualization dependencies are installed!")
    else:
        missing = []
        if not deps["matplotlib"]:
            missing.append("matplotlib")
        if not deps["networkx"]:
            missing.append("networkx")
        if not deps["plotly"]:
            missing.append("plotly")

        if missing:
            print(f"⚠ Missing: {', '.join(missing)}")


# CLI Integration helpers
def viz_from_cli(args: Dict[str, Any]) -> int:
    """Handle visualization from CLI arguments.

    Used by the CLI to create visualizations from command arguments.

    Args:
        args: Parsed CLI arguments

    Returns:
        Exit code (0 for success)
    """
    viz_type = args.get("type", "deps")
    output = args.get("output")
    format = args.get("format", "auto")

    # Load data based on type
    if viz_type in ["deps", "dependencies", "complexity"]:
        # Need file analysis
        from tenets.config import TenetsConfig
        from tenets.core.analysis import CodeAnalyzer

        config = TenetsConfig()
        analyzer = CodeAnalyzer(config)

        path = Path(args.get("path", "."))
        files = analyzer.analyze_files(path)

        if viz_type in ["deps", "dependencies"]:
            result = visualize_dependencies(
                files, output=output, format=format, max_nodes=args.get("max_nodes", 100)
            )
        else:
            result = visualize_complexity(
                files, output=output, format=format, threshold=args.get("threshold")
            )

    elif viz_type == "coupling":
        result = visualize_coupling(
            args.get("path", "."),
            output=output,
            format=format,
            min_coupling=args.get("min_coupling", 2),
        )

    elif viz_type == "contributors":
        # Need git data
        from tenets.core.git import GitAnalyzer

        analyzer = GitAnalyzer(Path(args.get("path", ".")))
        commits = analyzer.get_commit_history(limit=args.get("limit", 1000))

        result = visualize_contributors(
            commits, output=output, format=format, active_only=args.get("active", False)
        )

    else:
        print(f"Unknown visualization type: {viz_type}")
        return 1

    # Print result if not saved to file
    if result and not output:
        print(result)

    return 0
