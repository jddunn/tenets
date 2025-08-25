"""Viz command implementation.

This command provides visualization capabilities for codebase analysis,
including dependency graphs, complexity visualizations, and more.
"""

import json
from collections import defaultdict
import glob
from pathlib import Path
from typing import Dict, List, Optional

import click
import typer

from tenets.config import TenetsConfig
from tenets.utils.scanner import FileScanner as Scanner
from tenets.core.analysis.analyzer import CodeAnalyzer as Analyzer
from tenets.core.project_detector import ProjectDetector
from tenets.viz.graph_generator import GraphGenerator
from tenets.utils.logger import get_logger

viz_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Visualize codebase insights"
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
    path: str = typer.Argument(".", help="Path to analyze (use quotes for globs, e.g., ""**/*.py"")"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (e.g., architecture.svg)"),
    format: str = typer.Option("ascii", "--format", "-f", help="Output format (ascii, svg, png, html, json, dot)"),
    level: str = typer.Option("file", "--level", "-l", help="Dependency level (file, module, package)"),
    cluster_by: Optional[str] = typer.Option(None, "--cluster-by", help="Cluster nodes by (directory, module, package)"),
    max_nodes: Optional[int] = typer.Option(None, "--max-nodes", help="Maximum number of nodes to display"),
    include: Optional[str] = typer.Option(None, "--include", "-i", help="Include file patterns"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Exclude file patterns"),
    layout: str = typer.Option("hierarchical", "--layout", help="Graph layout (hierarchical, circular, shell, kamada)"),
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
            config = ctx.obj.get("config") if isinstance(ctx.obj, dict) else getattr(ctx.obj, "config", None)
        if not config:
            config = TenetsConfig()

        # Create analyzer and scanner
        analyzer = Analyzer(config)
        scanner = Scanner(config)

        # Normalize include/exclude patterns from CLI
        include_patterns = include.split(",") if include else None
        exclude_patterns = exclude.split(",") if exclude else None

        # Detect project type
        detector = ProjectDetector()
        if verbose:
            logger.debug(f"Starting project detection for: {path}")
        project_info = detector.detect_project(Path(path))

        logger.info(f"Detected project type: {project_info['type']}")
        logger.info(f"Languages: {', '.join(f'{lang} ({pct}%)' for lang, pct in project_info['languages'].items())}")
        if project_info['frameworks']:
            logger.info(f"Frameworks: {', '.join(project_info['frameworks'])}")
        if project_info['entry_points']:
            logger.info(f"Entry points: {', '.join(project_info['entry_points'][:5])}")

        if verbose:
            logger.debug(f"Full project info: {project_info}")
            logger.debug(f"Project structure: {project_info.get('structure', {})}")

        # Resolve path globs ourselves (Windows shells often don't expand globs)
        scan_paths = []
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
        dependency_graph = {}

        logger.info(f"Analyzing {len(files)} files for dependencies...")
        for i, file in enumerate(files, 1):
            if verbose:
                logger.debug(f"Analyzing file {i}/{len(files)}: {file}")
            analysis = analyzer.analyze_file(file, use_cache=False, deep=True)
            if analysis:
                # Prefer imports on structure; fall back to analysis.imports
                imports = []
                if analysis.structure and getattr(analysis.structure, "imports", None):
                    imports = analysis.structure.imports
                elif getattr(analysis, "imports", None):
                    imports = analysis.imports

                if imports:
                    deps = []
                    for imp in imports:
                        # Extract module name - handle different import types
                        module_name = None
                        if hasattr(imp, "module") and imp.module:
                            module_name = imp.module
                        elif hasattr(imp, "from_module") and imp.from_module:
                            module_name = imp.from_module

                        if module_name:
                            deps.append(module_name)

                    if deps:
                        dependency_graph[str(file)] = deps
                        if verbose:
                            logger.debug(f"Found {len(deps)} dependencies in {file}")
                else:
                    if verbose:
                        logger.debug(f"No imports found in {file}")
            else:
                if verbose:
                    logger.debug(f"No analysis for {file}")
        
        logger.info(f"Found dependencies in {len(dependency_graph)} files")
        
        # Aggregate dependencies based on level
        if level != "file":
            dependency_graph = aggregate_dependencies(
                dependency_graph, level, project_info
            )
            logger.info(f"Aggregated to {len(dependency_graph)} {level}s")
        
        if not dependency_graph:
            click.echo("No dependencies found in analyzed files.")
            click.echo("This could mean:")
            click.echo("  - Files don't have imports/dependencies")
            click.echo("  - File types are not supported yet")
            click.echo("  - Analysis couldn't extract import information")
            if output:
                click.echo(f"\nNo output file created as there's no data to save.")
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
            
            # Layout is now passed as parameter, no need to override
            
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
                        click.echo("\nOpen the HTML file in a browser for an interactive visualization.")
                    elif format == "dot":
                        click.echo("\nYou can render this DOT file with Graphviz tools.")
                    elif format in ["svg", "png", "pdf"]:
                        click.echo(f"\nGenerated {format.upper()} image with dependency graph.")
                else:
                    # Output to terminal if no file specified
                    click.echo(result)
                    
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
            click.echo("Hint: Quote your glob patterns to avoid shell parsing issues, e.g., \"**/*.py\".")
        raise typer.Exit(1)


@viz_app.command("complexity")
def complexity(
    path: str = typer.Argument(".", help="Path to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("ascii", "--format", "-f", help="Output format (ascii, svg, png, html)"),
    threshold: Optional[int] = typer.Option(None, "--threshold", help="Minimum complexity threshold"),
    hotspots: bool = typer.Option(False, "--hotspots", help="Show only hotspot files"),
    include: Optional[str] = typer.Option(None, "--include", "-i", help="Include file patterns"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Exclude file patterns"),
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
    
    try:
        # Get config from context if available
        ctx = click.get_current_context(silent=True)
        config = None
        if ctx and ctx.obj:
            config = ctx.obj.get("config") if isinstance(ctx.obj, dict) else getattr(ctx.obj, "config", None)
        if not config:
            config = TenetsConfig()
        
        # Create scanner
        scanner = Scanner(config)
        
        # Configure scanner
        if include:
            scanner.include_patterns = include.split(",")
        if exclude:
            scanner.exclude_patterns = exclude.split(",")
        
        # Scan files
        logger.info(f"Scanning {path} for complexity analysis...")
        files = scanner.scan([Path(path)])
        
        if not files:
            click.echo("No files found to analyze")
            raise typer.Exit(1)
        
        # Analyze files for complexity
        analyzer = Analyzer(config)
        complexity_data = []
        
        for file in files:
            analysis = analyzer.analyze_file(file, use_cache=False, deep=True)
            if analysis and analysis.complexity:
                complexity_score = analysis.complexity.cyclomatic
                if threshold and complexity_score < threshold:
                    continue
                if hotspots and complexity_score < 10:  # Hotspot threshold
                    continue
                complexity_data.append({
                    "file": str(file),
                    "complexity": complexity_score,
                    "cognitive": getattr(analysis.complexity, 'cognitive', 0),
                    "lines": len(file.read_text().splitlines()) if file.exists() else 0
                })
        
        # Sort by complexity
        complexity_data.sort(key=lambda x: x["complexity"], reverse=True)
        
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
        
        elif output:
            # Save to file
            output_path = Path(output)
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(complexity_data, f, indent=2)
                click.echo(f"Complexity data saved to {output_path}")
            else:
                # TODO: Implement other formats
                click.echo(f"Format {format} not yet implemented. Saving as JSON.")
                output_path = output_path.with_suffix(".json")
                with open(output_path, "w") as f:
                    json.dump(complexity_data, f, indent=2)
                click.echo(f"Complexity data saved to {output_path}")
        else:
            # Output JSON to stdout
            click.echo(json.dumps(complexity_data, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to generate complexity visualization: {e}")
        raise typer.Exit(1)


@viz_app.command("contributors")
def contributors(
    path: str = typer.Argument(".", help="Path to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("ascii", "--format", "-f", help="Output format (ascii, json, html)"),
    active: bool = typer.Option(False, "--active", help="Show only active contributors"),
    since: Optional[str] = typer.Option(None, "--since", help="Analyze since date (e.g., '2 weeks ago')"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug output"),
):
    """Visualize contributor activity.
    
    Examples:
        tenets viz contributors --active
        tenets viz contributors --since "1 month ago"
    """
    click.echo("Contributor visualization coming soon!")
    click.echo(f"Would analyze: {path}")
    if active:
        click.echo("Filtering for active contributors")
    if since:
        click.echo(f"Since: {since}")


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


def aggregate_dependencies(dependency_graph: Dict[str, List[str]], level: str, project_info: Dict) -> Dict[str, List[str]]:
    """Aggregate file-level dependencies to module or package level.
    
    Args:
        dependency_graph: File-level dependency graph
        level: Aggregation level (module or package)
        project_info: Project detection information
        
    Returns:
        Aggregated dependency graph
    """
    aggregated = defaultdict(set)
    
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
    
    # If it's already a module name (contains dots but no slashes)
    if "." in path_str and "/" not in path_str:
        parts = path_str.split(".")
    else:
        # Convert file path to parts
        parts = path_str.split("/")
        
        # Remove file extension from last part
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
        return "root"
    
    elif level == "package":
        # Package level - group by top-level package
        if len(parts) > 0:
            # For Python, find the top-level package
            if project_info.get("type", "").startswith("python"):
                # Look for __init__.py to determine package boundaries
                # For now, use the first directory as package
                return parts[0] if parts[0] not in [".", "root"] else "root"
            else:
                # For other languages, use top directory
                return parts[0] if parts[0] not in [".", "root"] else "root"
        return "root"
    
    return path_str  # Default to original path