"""Viz command implementation.

This command provides visualization capabilities for codebase analysis,
including dependency graphs, complexity visualizations, and more.
"""

import glob
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import typer

from tenets.config import TenetsConfig
from tenets.core.analysis.analyzer import CodeAnalyzer
from tenets.core.project_detector import ProjectDetector
from tenets.utils.logger import get_logger

# Expose FileScanner and CodeAnalyzer under expected names for tests/patching
from tenets.utils.scanner import FileScanner

# Re-export visualizer classes used in tests for patching
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
    detect_visualization_type,
)
from tenets.viz.graph_generator import GraphGenerator


def _generate_complexity_html(data: list, project_path: str, hotspots_only: bool = False) -> str:
    """Generate HTML visualization for complexity data."""

    # Data is already a list of complexity items
    items = data if isinstance(data, list) else []
    if hotspots_only:
        items = [i for i in items if i.get("complexity", 0) >= 10]

    # Sort by complexity
    items = sorted(items, key=lambda x: x.get("complexity", 0), reverse=True)

    # Calculate average complexity
    avg_complexity = sum(i.get("complexity", 0) for i in items) / len(items) if items else 0

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Complexity Analysis - {project_path}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .chart-container {{
            margin: 30px 0;
            height: 400px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .complexity-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            text-align: center;
            min-width: 40px;
        }}
        .complexity-low {{ background: #28a745; color: white; }}
        .complexity-medium {{ background: #ffc107; color: #333; }}
        .complexity-high {{ background: #dc3545; color: white; }}
        .complexity-extreme {{ background: #6f42c1; color: white; }}
        .file-path {{
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
            color: #555;
        }}
        .timestamp {{
            text-align: right;
            color: #999;
            font-size: 0.85em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Code Complexity Analysis</h1>
        <p style="color: #666;">Project: <strong>{project_path}</strong></p>

        <div class="summary">
            <div class="metric-card">
                <div class="metric-label">Total Files</div>
                <div class="metric-value">{len(items)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Complexity</div>
                <div class="metric-value">{avg_complexity:.1f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">High Complexity Files</div>
                <div class="metric-value">{len([i for i in items if i.get("complexity", 0) >= 10])}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Complexity</div>
                <div class="metric-value">{max((i.get("complexity", 0) for i in items), default=0)}</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="complexityChart"></canvas>
        </div>

        <h2>{"🔥 Complexity Hotspots" if hotspots_only else "📊 File Complexity Details"}</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Complexity</th>
                    <th>Lines</th>
                    <th>Functions</th>
                    <th>Classes</th>
                </tr>
            </thead>
            <tbody>
"""

    for item in items[:50]:  # Limit to top 50 for performance
        complexity = item.get("complexity", 0)
        if complexity < 5:
            badge_class = "complexity-low"
        elif complexity < 10:
            badge_class = "complexity-medium"
        elif complexity < 20:
            badge_class = "complexity-high"
        else:
            badge_class = "complexity-extreme"

        html += f"""
                <tr>
                    <td><span class="file-path">{item.get("file", "Unknown")}</span></td>
                    <td><span class="complexity-badge {badge_class}">{complexity}</span></td>
                    <td>{item.get("lines", 0)}</td>
                    <td>{item.get("functions", 0)}</td>
                    <td>{item.get("classes", 0)}</td>
                </tr>
"""

    html += f"""
            </tbody>
        </table>

        <div class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    <script>
        // Create bar chart
        const ctx = document.getElementById('complexityChart').getContext('2d');
        const data = {json.dumps(items[:20])};  // Top 20 for chart

        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: data.map(d => {{
                    const parts = d.file.split('/');
                    return parts[parts.length - 1];
                }}),
                datasets: [{{
                    label: 'Complexity',
                    data: data.map(d => d.complexity || 0),
                    backgroundColor: data.map(d => {{
                        const c = d.complexity || 0;
                        if (c < 5) return 'rgba(40, 167, 69, 0.8)';
                        if (c < 10) return 'rgba(255, 193, 7, 0.8)';
                        if (c < 20) return 'rgba(220, 53, 69, 0.8)';
                        return 'rgba(111, 66, 193, 0.8)';
                    }}),
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{
                        display: true,
                        text: 'Top 20 Files by Complexity'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Cyclomatic Complexity'
                        }}
                    }},
                    x: {{
                        ticks: {{
                            autoSkip: false,
                            maxRotation: 45,
                            minRotation: 45
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


viz_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Visualize codebase insights"
)


def setup_verbose_logging(verbose: bool, command_name: str = "") -> bool:
    """Setup verbose logging, checking both command flag and global context.

    Returns:
        True if verbose mode is enabled
    """
    # Check for verbose from either command flag or global context
    ctx = click.get_current_context(silent=True)
    global_verbose = ctx.obj.get("verbose", False) if ctx and ctx.obj else False
    verbose = verbose or global_verbose

    # Set logging level based on verbose flag
    if verbose:
        import logging

        logging.getLogger("tenets").setLevel(logging.DEBUG)
        logger = get_logger(__name__)
        if command_name:
            logger.debug(f"Verbose mode enabled for {command_name}")
        else:
            logger.debug("Verbose mode enabled")

    return verbose


@viz_app.command("deps")
def deps(
    path: str = typer.Argument(".", help="Path to analyze (use quotes for globs, e.g., **/*.py)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (e.g., architecture.svg)"
    ),
    format: str = typer.Option(
        "ascii", "--format", "-f", help="Output format (ascii, svg, png, html, json, dot)"
    ),
    level: str = typer.Option(
        "file", "--level", "-l", help="Dependency level (file, module, package)"
    ),
    cluster_by: Optional[str] = typer.Option(
        None, "--cluster-by", help="Cluster nodes by (directory, module, package)"
    ),
    max_nodes: Optional[int] = typer.Option(
        None, "--max-nodes", help="Maximum number of nodes to display"
    ),
    include: Optional[str] = typer.Option(None, "--include", "-i", help="Include file patterns"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Exclude file patterns"),
    layout: str = typer.Option(
        "hierarchical", "--layout", help="Graph layout (hierarchical, circular, shell, kamada)"
    ),
    include_minified: bool = typer.Option(
        False, "--include-minified", help="Include minified files"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug output"),
):
    """Visualize dependencies between files and modules.

    Automatically detects project type (Python, Node.js, Java, Go, etc.) and
    generates dependency graphs in multiple formats.

    Examples:
        tenets viz deps                              # Auto-detect and show ASCII tree
        tenets viz deps . --output arch.svg          # Generate SVG dependency graph
        tenets viz deps --format html -o deps.html   # Interactive HTML visualization
        tenets viz deps --level module                # Module-level dependencies
        tenets viz deps --level package --cluster-by package  # Package architecture
        tenets viz deps --layout circular --max-nodes 50      # Circular layout
        tenets viz deps src/ --include "*.py" --exclude "*test*"  # Filter files

    Install visualization libraries:
        pip install tenets[viz]  # For SVG, PNG, HTML support
    """
    logger = get_logger(__name__)

    # Setup verbose logging
    verbose = setup_verbose_logging(verbose, "viz deps")
    if verbose:
        logger.debug(f"Analyzing path(s): {path}")
        logger.debug(f"Output format: {format}")
        logger.debug(f"Dependency level: {level}")

    try:
        # Get config from context if available
        ctx = click.get_current_context(silent=True)
        config = None
        if ctx and ctx.obj:
            config = (
                ctx.obj.get("config")
                if isinstance(ctx.obj, dict)
                else getattr(ctx.obj, "config", None)
            )
        if not config:
            config = TenetsConfig()

        # Override minified exclusion if flag is set
        if include_minified:
            config.exclude_minified = False

        # Create analyzer and scanner
        analyzer = CodeAnalyzer(config)
        scanner = FileScanner(config)

        # Normalize include/exclude patterns from CLI
        include_patterns = include.split(",") if include else None
        exclude_patterns = exclude.split(",") if exclude else None

        # Detect project type
        detector = ProjectDetector()
        if verbose:
            logger.debug(f"Starting project detection for: {path}")
        project_info = detector.detect_project(Path(path))

        # Echo key detection info so it's visible in CLI output (also logged)
        click.echo(f"Detected project type: {project_info['type']}")
        logger.info(f"Detected project type: {project_info['type']}")
        logger.info(
            ", ".join(f"{lang} ({pct}%)" for lang, pct in project_info.get("languages", {}).items())
        )
        if project_info.get("frameworks"):
            logger.info(f"Frameworks: {', '.join(project_info['frameworks'])}")
        if project_info.get("entry_points"):
            logger.info(f"Entry points: {', '.join(project_info['entry_points'][:5])}")

        if verbose:
            logger.debug(f"Full project info: {project_info}")
            logger.debug(f"Project structure: {project_info.get('structure', {})}")

        # Resolve path globs ourselves (Windows shells often don't expand globs)
        scan_paths: List[Path] = []
        contains_glob = any(ch in path for ch in ["*", "?", "["])
        if contains_glob:
            matched = [Path(p) for p in glob.glob(path, recursive=True)]
            if matched:
                scan_paths = matched
                if verbose:
                    logger.debug(f"Expanded glob to {len(matched)} paths")
        if not scan_paths:
            scan_paths = [Path(path)]

        # Scan files (pass patterns correctly)
        logger.info(f"Scanning {path} for dependencies...")
        files = scanner.scan(
            scan_paths,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        if not files:
            click.echo("No files found to analyze")
            raise typer.Exit(1)

        # Analyze files for dependencies
        dependency_graph: Dict[str, List[str]] = {}

        logger.info(f"Analyzing {len(files)} files for dependencies...")
        for i, file in enumerate(files, 1):
            if verbose:
                logger.debug(f"Analyzing file {i}/{len(files)}: {file}")
            analysis = analyzer.analyze_file(file, use_cache=False, deep=True)
            if analysis:
                # Prefer imports on structure; fall back to analysis.imports
                imports = []
                if getattr(analysis, "structure", None) and getattr(
                    analysis.structure, "imports", None
                ):
                    imports = analysis.structure.imports
                elif getattr(analysis, "imports", None):
                    imports = analysis.imports

                if imports:
                    deps = []
                    for imp in imports:
                        # Extract module name - handle different import types
                        module_name = None
                        if hasattr(imp, "module") and getattr(imp, "module", None):
                            module_name = imp.module
                        elif hasattr(imp, "from_module") and getattr(imp, "from_module", None):
                            module_name = imp.from_module

                        if module_name:
                            deps.append(module_name)

                    if deps:
                        dependency_graph[str(file)] = deps
                        if verbose:
                            logger.debug(f"Found {len(deps)} dependencies in {file}")
                elif verbose:
                    logger.debug(f"No imports found in {file}")
            elif verbose:
                logger.debug(f"No analysis for {file}")

        logger.info(f"Found dependencies in {len(dependency_graph)} files")

        # Aggregate dependencies based on level
        if level != "file":
            dependency_graph = aggregate_dependencies(dependency_graph, level, project_info)
            logger.info(f"Aggregated to {len(dependency_graph)} {level}s")

        if not dependency_graph:
            click.echo("No dependencies found in analyzed files.")
            click.echo("This could mean:")
            click.echo("  - Files don't have imports/dependencies")
            click.echo("  - File types are not supported yet")
            click.echo("  - Analysis couldn't extract import information")
            if output:
                click.echo("\nNo output file created as there's no data to save.")
            raise typer.Exit(0)

        # Generate visualization using GraphGenerator
        if format == "ascii":
            # Simple ASCII tree output for terminal
            click.echo("\nDependency Graph:")
            click.echo("=" * 50)

            # Apply max_nodes limit for ASCII output
            items = list(dependency_graph.items())
            if max_nodes:
                items = items[:max_nodes]

            for file_path, deps in sorted(items):
                click.echo(f"\n{Path(file_path).name}")
                for dep in deps[:10]:  # Limit deps per file for readability
                    click.echo(f"  └─> {dep}")

            if max_nodes and len(dependency_graph) > max_nodes:
                click.echo(f"\n... and {len(dependency_graph) - max_nodes} more files")
        else:
            # Use GraphGenerator for all other formats
            generator = GraphGenerator()

            # Auto-generate output filename if format requires a file but none specified
            if not output and format in ["html", "svg", "png", "pdf", "dot"]:
                # Generate a descriptive filename
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Get project name, handle "." and empty cases
                if path == "." or not path:
                    project_name = Path.cwd().name
                else:
                    project_name = Path(path).name

                # Clean up project name for filename
                project_name = project_name.replace(" ", "_").replace("-", "_")
                if not project_name or project_name == ".":
                    project_name = "project"

                # Include level and other options in filename for clarity
                filename_parts = ["dependency_graph", project_name]
                if level != "file":
                    filename_parts.append(level)
                if cluster_by:
                    filename_parts.append(f"by_{cluster_by}")
                if max_nodes:
                    filename_parts.append(f"top{max_nodes}")
                filename_parts.append(timestamp)

                output = "_".join(filename_parts) + f".{format}"
                click.echo(f"Auto-generating output file: {output}")

            try:
                result = generator.generate_graph(
                    dependency_graph=dependency_graph,
                    output_path=Path(output) if output else None,
                    format=format,
                    layout=layout,
                    cluster_by=cluster_by,
                    max_nodes=max_nodes,
                    project_info=project_info,
                )

                if output:
                    click.echo(f"\n✓ Dependency graph saved to: {result}")
                    click.echo(f"  Format: {format}")
                    click.echo(f"  Nodes: {len(dependency_graph)}")
                    click.echo(f"  Project type: {project_info['type']}")

                    # Provide helpful messages based on format
                    if format == "html":
                        click.echo(
                            "\nOpen the HTML file in a browser for an interactive visualization."
                        )
                        # Optionally offer to open it
                        if click.confirm(
                            "Would you like to open it in your browser now?", default=False
                        ):
                            import webbrowser

                            # Ensure absolute path for file URI
                            file_path = Path(result).resolve()
                            webbrowser.open(file_path.as_uri())
                    elif format == "dot":
                        click.echo("\nYou can render this DOT file with Graphviz tools.")
                    elif format in ["svg", "png", "pdf"]:
                        click.echo(f"\nGenerated {format.upper()} image with dependency graph.")
                # Only output to terminal for formats that make sense (json, ascii)
                elif format in ["json", "ascii"]:
                    click.echo(result)
                else:
                    click.echo(
                        f"Error: Format '{format}' requires an output file. Use --output or let auto-naming handle it."
                    )

            except Exception as e:
                logger.error(f"Failed to generate {format} visualization: {e}")
                click.echo(f"Error generating visualization: {e}")
                click.echo("\nFalling back to JSON output...")

                # Fallback to JSON
                output_data = {
                    "dependency_graph": dependency_graph,
                    "project_info": project_info,
                    "cluster_by": cluster_by,
                }

                if output:
                    output_path = Path(output).with_suffix(".json")
                    with open(output_path, "w") as f:
                        json.dump(output_data, f, indent=2)
                    click.echo(f"Dependency data saved to {output_path}")
                else:
                    click.echo(json.dumps(output_data, indent=2))

    except Exception as e:
        logger.error(f"Failed to generate dependency visualization: {e}")
        # Provide a helpful hint for Windows users about quoting globs
        if any(ch in path for ch in ["*", "?", "["]):
            click.echo(
                'Hint: Quote your glob patterns to avoid shell parsing issues, e.g., "**/*.py".'
            )
        raise typer.Exit(1)


@viz_app.command("complexity")
def complexity(
    path: str = typer.Argument(".", help="Path to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option(
        "ascii", "--format", "-f", help="Output format (ascii, svg, png, html)"
    ),
    threshold: Optional[int] = typer.Option(
        None, "--threshold", help="Minimum complexity threshold"
    ),
    hotspots: bool = typer.Option(False, "--hotspots", help="Show only hotspot files"),
    include: Optional[str] = typer.Option(None, "--include", "-i", help="Include file patterns"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Exclude file patterns"),
    include_minified: bool = typer.Option(
        False,
        "--include-minified",
        help="Include minified/built files (*.min.js, dist/, etc.) normally excluded",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug output"),
):
    """Visualize code complexity metrics.

    Examples:
        tenets viz complexity              # ASCII bar chart
        tenets viz complexity --threshold 10 --hotspots  # High complexity only
        tenets viz complexity --output complexity.png    # Save as image
    """
    logger = get_logger(__name__)

    # Setup verbose logging
    verbose = setup_verbose_logging(verbose, "viz complexity")

    # Get config from context if available
    ctx = click.get_current_context(silent=True)
    config = None
    if ctx and ctx.obj:
        config = (
            ctx.obj.get("config") if isinstance(ctx.obj, dict) else getattr(ctx.obj, "config", None)
        )
    if not config:
        config = TenetsConfig()

    # Create scanner
    scanner = FileScanner(config)

    # Scan files
    logger.info(f"Scanning {path} for complexity analysis...")
    files = scanner.scan(
        [Path(path)],
        include_patterns=include.split(",") if include else None,
        exclude_patterns=exclude.split(",") if exclude else None,
    )

    if not files:
        click.echo("No files found to analyze")
        raise typer.Exit(1)

    # Analyze files for complexity
    analyzer = CodeAnalyzer(config)
    complexity_data: List[Dict[str, Any]] = []

    for file in files:
        analysis = analyzer.analyze_file(file, use_cache=False, deep=True)
        if analysis and getattr(analysis, "complexity", None):
            complexity_score = analysis.complexity.cyclomatic
            if threshold and complexity_score < threshold:
                continue
            if hotspots and complexity_score < 10:  # Hotspot threshold
                continue
            complexity_data.append(
                {
                    "file": str(file),
                    "complexity": complexity_score,
                    "cognitive": getattr(analysis.complexity, "cognitive", 0),
                    "lines": (
                        len(file.read_text(encoding="utf-8", errors="ignore").splitlines())
                        if file.exists()
                        else 0
                    ),
                }
            )

    # Sort by complexity
    complexity_data.sort(key=lambda x: x["complexity"], reverse=True)

    # Auto-generate output filename if format requires a file but none specified
    if not output and format in ["html", "svg", "png", "json"] and format != "ascii":
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = Path(path).name if Path(path).name != "." else "project"

        filename_parts = ["complexity", project_name]
        if threshold:
            filename_parts.append(f"threshold{threshold}")
        if hotspots:
            filename_parts.append("hotspots")
        filename_parts.append(timestamp)

        output = "_".join(filename_parts) + f".{format if format != 'html' else 'json'}"
        click.echo(f"Auto-generating output file: {output}")

    # Generate visualization based on format
    if format == "ascii":
        # ASCII bar chart
        click.echo("\nComplexity Analysis:")
        click.echo("=" * 60)

        if not complexity_data:
            click.echo("No files meet the criteria")
        else:
            max_complexity = max(c["complexity"] for c in complexity_data)
            for item in complexity_data[:20]:  # Show top 20
                file_name = Path(item["file"]).name
                complexity = item["complexity"]
                bar_length = int((complexity / max_complexity) * 40) if max_complexity > 0 else 0
                bar = "█" * bar_length
                click.echo(f"{file_name:30} {bar} {complexity}")

    elif output or format != "ascii":
        # Save to file
        if output:
            output_path = Path(output)
        # Auto-generate filename based on format
        elif format == "html":
            output_path = Path(
                f"complexity__hotspots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
        else:
            output_path = Path(
                f"complexity__hotspots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        if format == "html":
            # Generate HTML visualization
            html_content = _generate_complexity_html(complexity_data, path, hotspots)
            output_path = output_path.with_suffix(".html")
            output_path.write_text(html_content, encoding="utf-8")
            click.echo(f"Complexity HTML visualization saved to {output_path}")

            # Offer to open in browser
            if click.confirm("\nWould you like to open it in your browser now?", default=False):
                import webbrowser

                file_path = output_path.resolve()
                webbrowser.open(file_path.as_uri())
                click.echo("✓ Opened in browser")
        elif format in ["json", "svg", "png"]:
            # Save as JSON for now (SVG/PNG can be added later)
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(complexity_data, f, indent=2)
            click.echo(f"Complexity data saved to {output_path}")
        else:
            # Default to JSON
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(complexity_data, f, indent=2)
            click.echo(f"Complexity data saved to {output_path}")
    # Output JSON to stdout only if explicitly no output and format is compatible
    elif format == "json":
        click.echo(json.dumps(complexity_data, indent=2))
    else:
        click.echo("Use --output to specify output file or --format ascii for terminal display")


@viz_app.command("data")
def data(
    input_file: str = typer.Argument(help="Data file to visualize (JSON/CSV)"),
    chart: Optional[str] = typer.Option(None, "--chart", "-c", help="Chart type"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("terminal", "--format", "-f", help="Output format"),
    title: Optional[str] = typer.Option(None, "--title", help="Chart title"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug output"),
):
    """Create visualizations from data files.

    This command generates visualizations from pre-analyzed data files
    without needing to re-run analysis.
    """
    logger = get_logger(__name__)

    input_path = Path(input_file)
    if not input_path.exists():
        click.echo(f"Error: File not found: {input_file}")
        raise typer.Exit(1)

    # Load data
    if input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
        click.echo(f"Loaded JSON data from {input_file}")
        click.echo(f"Data type: {data.get('type', 'unknown')}")
        # TODO: Generate actual visualization
    else:
        click.echo(f"Unsupported file format: {input_path.suffix}")
        raise typer.Exit(1)


def aggregate_dependencies(
    dependency_graph: Dict[str, List[str]], level: str, project_info: Dict
) -> Dict[str, List[str]]:
    """Aggregate file-level dependencies to module or package level.

    Args:
        dependency_graph: File-level dependency graph
        level: Aggregation level (module or package)
        project_info: Project detection information

    Returns:
        Aggregated dependency graph
    """
    aggregated = defaultdict(set)

    # First, ensure all source modules are in the result
    for source_file in dependency_graph:
        source_key = get_aggregate_key(source_file, level, project_info)
        if source_key not in aggregated:
            aggregated[source_key] = set()

    # Then add dependencies
    for source_file, dependencies in dependency_graph.items():
        # Get aggregate key for source
        source_key = get_aggregate_key(source_file, level, project_info)

        for dep in dependencies:
            # Get aggregate key for dependency
            dep_key = get_aggregate_key(dep, level, project_info)

            # Don't add self-dependencies
            if source_key != dep_key:
                aggregated[source_key].add(dep_key)

    # Convert sets to lists
    return {k: sorted(list(v)) for k, v in aggregated.items()}


def get_aggregate_key(path_str: str, level: str, project_info: Dict) -> str:
    """Get the aggregate key for a path based on the specified level.

    Args:
        path_str: File path or module name
        level: Aggregation level (module or package)
        project_info: Project information for context

    Returns:
        Aggregate key string
    """
    # Handle different path formats
    path_str = path_str.replace("\\", "/")

    # Check if it's a module name (not a file) - module names use dots as separators
    # but don't have file extensions like .py, .js, etc.
    is_module_name = (
        "." in path_str
        and "/" not in path_str
        and not any(
            path_str.endswith(ext)
            for ext in [
                ".py",
                ".js",
                ".java",
                ".go",
                ".rs",
                ".rb",
                ".ts",
                ".jsx",
                ".tsx",
                ".cpp",
                ".c",
                ".h",
            ]
        )
    )

    if is_module_name:
        # It's already a module name like "src.utils.helpers"
        parts = path_str.split(".")
    else:
        # Convert file path to parts
        parts = path_str.split("/")

        # Remove file extension from last part if it's a file
        if parts and "." in parts[-1]:
            filename = parts[-1]
            name_without_ext = filename.rsplit(".", 1)[0]
            parts[-1] = name_without_ext

    if level == "module":
        # Module level - group by immediate parent directory
        if len(parts) > 1:
            # For Python projects, use dot notation
            if project_info.get("type", "").startswith("python"):
                return ".".join(parts[:-1])
            else:
                # For other projects, use directory path
                return "/".join(parts[:-1])
        else:
            # Single file at root level always returns "root" for module level
            return "root"

    elif level == "package":
        # Package level - group by top-level package
        if len(parts) > 1:
            # For Python, find the top-level package
            if project_info.get("type", "").startswith("python"):
                # Look for __init__.py to determine package boundaries
                # For now, use the first directory as package
                return parts[0] if parts[0] not in [".", "root"] else "root"
            else:
                # For other languages, use top directory
                return parts[0] if parts[0] not in [".", "root"] else "root"
        else:
            # Single file at root level
            return "root"

    return path_str  # Default to original path


# Standalone data visualization command used by tests via runner.invoke(viz, ...)
@click.command()
@click.argument("input_path")
@click.option("--type", default="auto", help="Visualization type or 'auto'")
@click.option("--chart", "-c", default=None, help="Chart type for custom viz")
@click.option(
    "--format", "-f", default="terminal", help="Output format (terminal,json,html,svg,png)"
)
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--title", default=None, help="Chart title")
@click.option("--width", type=int, default=None, help="Chart width")
@click.option("--height", type=int, default=None, help="Chart height")
@click.option("--label-field", default=None, help="Label field for custom charts")
@click.option("--value-field", default=None, help="Value field for custom charts")
@click.option("--x-field", default=None, help="X field for custom charts")
@click.option("--y-field", default=None, help="Y field for custom charts")
@click.option("--limit", type=int, default=None, help="Limit number of data points")
@click.option("--interactive", is_flag=True, help="Open interactive HTML in browser")
def viz(
    input_path: str,
    type: str,
    chart: Optional[str],
    format: str,
    output: Optional[str],
    title: Optional[str],
    width: Optional[int],
    height: Optional[int],
    label_field: Optional[str],
    value_field: Optional[str],
    x_field: Optional[str],
    y_field: Optional[str],
    limit: Optional[int],
    interactive: bool,
):
    logger = get_logger(__name__)
    p = Path(input_path)
    if not p.exists():
        click.echo(f"Error: File does not exist: {input_path}")
        raise click.ClickException(f"File does not exist: {input_path}")

    # Load data (attempt JSON first for unknown extensions)
    data: Any
    try:
        if p.suffix.lower() == ".json" or p.suffix.lower() not in {".csv"}:
            data = json.loads(p.read_text())
        else:
            # CSV
            import csv

            with open(p, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
    except Exception:
        click.echo("Visualization failed: Could not parse data file")
        raise click.ClickException("Could not parse data file")

    # Auto-detect type if requested
    viz_type = type
    if viz_type == "auto":
        try:
            viz_type = detect_visualization_type(data)
        except Exception:
            viz_type = "custom"

    # Build chart via the appropriate visualizer
    chart_cfg = ChartConfig(type=ChartType.BAR, title=title or "")
    if width:
        chart_cfg.width = width
    if height:
        chart_cfg.height = height

    try:
        if viz_type == "complexity":
            viz_ = ComplexityVisualizer(chart_config=chart_cfg)
            chart_data = viz_.create_distribution_chart(data.get("complexity") or data)
        elif viz_type == "contributors":
            viz_ = ContributorVisualizer(chart_config=chart_cfg)
            chart_data = viz_.create_contribution_chart(
                data if isinstance(data, list) else data.get("contributors", [])
            )
        elif viz_type == "hotspots":
            viz_ = HotspotVisualizer(chart_config=chart_cfg)
            chart_data = viz_.create_hotspot_bubble(data.get("hotspots") or data)
        elif viz_type == "momentum":
            viz_ = MomentumVisualizer(chart_config=chart_cfg)
            # choose a generic momentum chart
            chart_data = viz_.create_velocity_chart(data)
        elif viz_type in ("dependencies", "deps"):
            viz_ = DependencyVisualizer(chart_config=chart_cfg)  # type: ignore[name-defined]
            chart_data = viz_.create_dependency_graph(
                data.get("dependencies") or data.get("dependency_graph") or data
            )
        elif viz_type == "coupling":
            viz_ = CouplingVisualizer(chart_config=chart_cfg)
            chart_data = viz_.create_coupling_network(data.get("coupling_data") or data)
        else:
            # custom
            viz_ = BaseVisualizer(chart_config=chart_cfg)
            # Build dataset from fields if provided
            inferred_type = chart or "bar"
            payload: Dict[str, Any]
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # Respect limit
                rows = data[: limit or len(data)]
                if label_field and (value_field or (x_field and y_field)):
                    if value_field:
                        payload = {
                            "labels": [str(r.get(label_field, "")) for r in rows],
                            "values": [float(r.get(value_field, 0) or 0) for r in rows],
                        }
                    else:
                        payload = {
                            "labels": [str(r.get(x_field, "")) for r in rows],
                            "datasets": [
                                {
                                    "label": y_field or "value",
                                    "data": [float(r.get(y_field, 0) or 0) for r in rows],
                                }
                            ],
                        }
                else:
                    # Fallback minimal structure
                    payload = {"labels": list(range(len(rows))), "values": [1] * len(rows)}
            else:
                payload = {"labels": ["A", "B"], "values": [1, 2]}
            chart_data = viz_.create_chart(inferred_type, payload, config=chart_cfg)

        # Output handling
        fmt = (format or "terminal").lower()

        # If interactive requested, prefer HTML regardless of requested format
        if interactive:
            fmt = "html"

        # Terminal output only if not overridden by interactive or other formats
        if fmt == "terminal" and hasattr(viz_, "display_terminal"):
            # Show summary style output expected by tests
            click.echo(
                "Custom Visualization Generated"
                if viz_type == "custom"
                else "Visualization Generated"
            )
            click.echo(f"Type: {chart_data.get('type', 'unknown')}")
            if isinstance(chart_data.get("data"), dict) and chart_data["data"].get("datasets"):
                click.echo(f"Datasets: {len(chart_data['data']['datasets'])}")
            # Also call display if available
            try:
                viz_.display_terminal(chart_data)  # type: ignore[attr-defined]
            except Exception:
                pass
            return  # Exit successfully

        # File outputs
        if output:
            out_path = Path(output)
            if fmt == "json":
                out_path.write_text(json.dumps(chart_data))
                click.echo(f"Visualization saved to: {out_path}")
            elif fmt == "html":
                # Compose minimal HTML using tenets.viz HTML helper through export
                from tenets.viz import export_visualization

                export_visualization(chart_data, out_path, format="html", config=chart_cfg)
                click.echo(f"Visualization saved to: {out_path}")
            elif fmt in {"svg", "png"}:
                click.echo(f"{fmt.upper()} export not yet implemented")
            else:
                # default to JSON
                out_path.write_text(json.dumps(chart_data))
                click.echo(f"Visualization saved to: {out_path}")
            return  # Exit successfully

        # No output specified: print JSON for json format, else treat as success
        if fmt == "json":
            click.echo(json.dumps(chart_data))
        elif fmt == "html":
            # Create temp HTML and open if interactive
            import tempfile
            import webbrowser

            from tenets.viz import export_visualization

            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tf:
                export_visualization(chart_data, Path(tf.name), format="html", config=chart_cfg)
                if interactive:
                    click.echo("Launching interactive mode...")
                    webbrowser.open(Path(tf.name).as_uri())
                    click.echo("Opened in browser")
        elif fmt in {"svg", "png"}:
            # Explicitly acknowledge not implemented even without output
            click.echo(f"{fmt.upper()} export not yet implemented")
        else:
            # Terminal default already echoed above for many types
            pass

    except KeyError as e:
        click.echo(f"Visualization failed: missing field {e}")
        raise click.ClickException(f"Missing field: {e}")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Visualization failed: {e}")
        click.echo("Visualization failed")
        raise click.ClickException(str(e))
