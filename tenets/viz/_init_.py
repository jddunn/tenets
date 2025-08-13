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

# Import visualization components
from .base import (
    ColorScheme,
    VisualizationBase,
    VisualizationFormat,
    check_dependencies,
    create_graph_layout,
    format_size,
    truncate_text,
)
from .complexity import (
    ComplexityHeatmap,
    ComplexityMetrics,
    FunctionComplexity,
    analyze_complexity_trends,
    create_complexity_heatmap,
)
from .contributors import (
    ContributorGraph,
    ContributorStats,
    TeamDynamics,
    analyze_contributors,
)
from .coupling import (
    CouplingGraph,
    FileChangeHistory,
    FileCoupling,
    analyze_coupling_from_git,
)
from .dependencies import (
    DependencyGraph,
    DependencyNode,
    create_dependency_graph,
)

# Version info
__version__ = "0.1.0"

# Check which visualization libraries are available
DEPENDENCIES = check_dependencies()
MATPLOTLIB_AVAILABLE = DEPENDENCIES.get("matplotlib", False)
NETWORKX_AVAILABLE = DEPENDENCIES.get("networkx", False)
PLOTLY_AVAILABLE = DEPENDENCIES.get("plotly", False)

# Public API exports
__all__ = [
    # Base components
    "ColorScheme",
    "VisualizationBase",
    "VisualizationFormat",
    # Dependency visualization
    "DependencyGraph",
    "DependencyNode",
    "create_dependency_graph",
    "visualize_dependencies",
    # Complexity visualization
    "ComplexityHeatmap",
    "ComplexityMetrics",
    "FunctionComplexity",
    "create_complexity_heatmap",
    "visualize_complexity",
    "analyze_complexity_trends",
    # Coupling visualization
    "CouplingGraph",
    "FileCoupling",
    "FileChangeHistory",
    "analyze_coupling_from_git",
    "visualize_coupling",
    # Contributor visualization
    "ContributorGraph",
    "ContributorStats",
    "TeamDynamics",
    "analyze_contributors",
    "visualize_contributors",
    # Utilities
    "create_visualization",
    "check_dependencies",
    "format_size",
    "truncate_text",
]


def visualize_dependencies(
    files: List[Any],
    output: Optional[Union[str, Path]] = None,
    format: str = "auto",
    max_nodes: int = 100,
    layout: str = "spring",
    title: Optional[str] = None,
) -> Union[str, bytes, None]:
    """Create and render a dependency graph.

    Convenience function that creates a dependency graph and optionally
    renders it to a file or returns the rendered content.

    Args:
        files: List of FileAnalysis objects
        output: Output path (if None, returns content)
        format: Output format (svg, png, html, ascii, json, auto)
        max_nodes: Maximum nodes to display
        layout: Graph layout algorithm
        title: Custom title

    Returns:
        Rendered content if output is None, otherwise None

    Example:
        >>> from tenets.viz import visualize_dependencies
        >>>
        >>> # Render to file
        >>> visualize_dependencies(files, output="deps.svg")
        >>>
        >>> # Get ASCII representation
        >>> ascii_graph = visualize_dependencies(files, format="ascii")
        >>> print(ascii_graph)
    """
    title = title or f"Dependencies ({len(files)} files)"

    # Create graph
    graph = create_dependency_graph(files, title=title, max_nodes=max_nodes, format=format)

    # Render
    if output:
        graph.render(Path(output))
        return None
    else:
        return graph.render()


def visualize_complexity(
    files: List[Any],
    output: Optional[Union[str, Path]] = None,
    format: str = "auto",
    threshold: Optional[int] = None,
    title: Optional[str] = None,
) -> Union[str, bytes, None]:
    """Create and render a complexity heatmap.

    Visualizes code complexity patterns to identify maintenance hotspots
    and refactoring candidates.

    Args:
        files: List of FileAnalysis objects with complexity metrics
        output: Output path (if None, returns content)
        format: Output format
        threshold: Minimum complexity to include
        title: Custom title

    Returns:
        Rendered content if output is None, otherwise None

    Example:
        >>> from tenets.viz import visualize_complexity
        >>>
        >>> # Show high complexity files
        >>> heatmap = visualize_complexity(files, threshold=10)
        >>> print(heatmap)  # ASCII output
    """
    title = title or "Code Complexity Heatmap"

    # Create heatmap
    heatmap = create_complexity_heatmap(files, title=title, threshold=threshold, format=format)

    # Render
    if output:
        heatmap.render(Path(output))
        return None
    else:
        return heatmap.render()


def visualize_coupling(
    repo_path: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    format: str = "auto",
    min_coupling: int = 2,
    since_days: int = 90,
    title: Optional[str] = None,
) -> Union[str, bytes, None]:
    """Analyze and visualize file coupling from git history.

    Shows which files frequently change together, helping identify
    hidden dependencies and refactoring opportunities.

    Args:
        repo_path: Path to git repository
        output: Output path
        format: Output format
        min_coupling: Minimum changes together to show
        since_days: Analyze commits from last N days
        title: Custom title

    Returns:
        Rendered content if output is None, otherwise None

    Example:
        >>> from tenets.viz import visualize_coupling
        >>>
        >>> # Analyze coupling in current repo
        >>> visualize_coupling(".", output="coupling.html", since_days=30)
    """
    from datetime import datetime, timedelta

    title = title or f"File Coupling (last {since_days} days)"
    since = datetime.now() - timedelta(days=since_days)

    # Analyze coupling
    graph = analyze_coupling_from_git(Path(repo_path), since=since, min_coupling=min_coupling)
    graph.title = title

    # Render
    if output:
        graph.render(Path(output))
        return None
    else:
        return graph.render()


def visualize_contributors(
    commits: List[Dict[str, Any]],
    output: Optional[Union[str, Path]] = None,
    format: str = "auto",
    active_only: bool = False,
    title: Optional[str] = None,
) -> Union[str, bytes, None]:
    """Visualize contributor activity and patterns.

    Shows who works on what, collaboration patterns, and potential
    bus factor risks.

    Args:
        commits: List of commit data dictionaries
        output: Output path
        format: Output format
        active_only: Only show active contributors
        title: Custom title

    Returns:
        Rendered content if output is None, otherwise None

    Example:
        >>> from tenets.viz import visualize_contributors
        >>>
        >>> # Show contributor patterns
        >>> viz = visualize_contributors(commit_data, format="html")
    """
    title = title or "Contributor Activity"

    # Analyze contributors
    graph = analyze_contributors(commits, title=title, active_only=active_only)

    # Render
    if output:
        graph.render(Path(output))
        return None
    else:
        return graph.render()


def create_visualization(
    data: Any,
    viz_type: str,
    output: Optional[Union[str, Path]] = None,
    format: str = "auto",
    **kwargs,
) -> Union[str, bytes, None]:
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

    return {
        "matplotlib": matplotlib_available,
        "networkx": networkx_available,
        "plotly": plotly_available,
        "all": matplotlib_available and networkx_available and plotly_available,
    }


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
