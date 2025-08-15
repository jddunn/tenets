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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import visualization components (actual available symbols)
from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay
from .dependencies import DependencyVisualizer
from .complexity import ComplexityVisualizer
from .contributors import ContributorVisualizer
from .coupling import CouplingVisualizer

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
    "TerminalDisplay",
    # Visualizers
    "DependencyVisualizer",
    "ComplexityVisualizer",
    "CouplingVisualizer",
    "ContributorVisualizer",
    # Convenience wrappers
    "visualize_dependencies",
    "visualize_complexity",
    "visualize_coupling",
    "visualize_contributors",
    "create_visualization",
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
