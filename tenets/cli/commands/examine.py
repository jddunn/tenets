"""Examine command - analyze codebase structure and health."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from tenets import Tenets

console = Console()


def examine(
    path: Path = typer.Argument(Path("."), help="Path to examine (directory or file)"),
    # Analysis options
    deep: bool = typer.Option(False, "--deep", "-d", help="Perform deep analysis with AST parsing"),
    # Output options
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save analysis results to file"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table (default), json, yaml"
    ),
    # What to show
    metrics: bool = typer.Option(False, "--metrics", help="Show detailed code metrics"),
    complexity: bool = typer.Option(False, "--complexity", help="Show complexity analysis"),
    ownership: bool = typer.Option(False, "--ownership", help="Show code ownership (requires git)"),
    hotspots: bool = typer.Option(
        False, "--hotspots", help="Show code hotspots (frequently changed files)"
    ),
    structure: bool = typer.Option(False, "--structure", help="Show directory structure"),
    # Git options
    no_git: bool = typer.Option(False, "--no-git", help="Disable git analysis"),
    # Context
    ctx: typer.Context = typer.Context,
):
    """
    Examine codebase structure, complexity, and health.

    Provides detailed analysis of your code including metrics, dependencies,
    complexity scores, and potential issues.

    Examples:

        # Basic examination
        tenets examine

        # Deep analysis with metrics
        tenets examine ./src --deep --metrics

        # Show complexity and hotspots
        tenets examine --complexity --hotspots

        # Export full analysis
        tenets examine --output analysis.json --format json

        # Show code ownership
        tenets examine --ownership --no-git
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        tenets = Tenets()

        # Run analysis
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(f"Examining {path}...", total=None)

                result = tenets.examine(
                    path=path, deep=deep, include_git=not no_git, output_metadata=True
                )
        else:
            result = tenets.examine(
                path=path, deep=deep, include_git=not no_git, output_metadata=True
            )

        # Format output based on requested format
        if format == "table":
            _display_analysis_table(result, quiet)

            if metrics:
                _display_metrics_table(result)

            if complexity:
                _display_complexity_table(result)

            if ownership and result.git_analysis:
                _display_ownership_table(result.git_analysis)

            if hotspots and result.git_analysis:
                _display_hotspots_table(result.git_analysis)

            if structure:
                _display_structure_tree(result)

        elif format == "json":
            output_data = result.to_dict()
            if output:
                output.write_text(json.dumps(output_data, indent=2))
                if not quiet:
                    console.print(f"[green]✓[/green] Analysis saved to {output}")
            else:
                console.print_json(data=output_data)

        elif format == "yaml":
            import yaml

            output_data = result.to_dict()
            yaml_str = yaml.dump(output_data, default_flow_style=False, sort_keys=False)
            if output:
                output.write_text(yaml_str)
                if not quiet:
                    console.print(f"[green]✓[/green] Analysis saved to {output}")
            else:
                console.print(yaml_str)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _display_analysis_table(analysis, quiet: bool = False):
    """Display main analysis results in a table."""
    if not quiet:
        console.print(
            Panel(
                f"[bold]Codebase Examination Results[/bold]\n" f"Root: {analysis.root_path}",
                title="📊 Analysis Summary",
                border_style="blue",
            )
        )

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add summary metrics
    summary = analysis.summary
    table.add_row("Total Files", str(analysis.total_files))
    table.add_row("Total Lines", f"{analysis.total_lines:,}")

    # Language breakdown
    lang_str = ", ".join(
        f"{lang} ({sum(1 for f in analysis.files if f.language == lang)})"
        for lang in analysis.languages[:5]
    )
    if len(analysis.languages) > 5:
        lang_str += f", +{len(analysis.languages) - 5} more"
    table.add_row("Languages", lang_str)

    # Average complexity if available
    if any(f.complexity for f in analysis.files):
        complexities = [f.complexity.cyclomatic for f in analysis.files if f.complexity]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        table.add_row("Avg Complexity", f"{avg_complexity:.2f}")

    # Git stats if available
    if hasattr(analysis, "git_analysis") and analysis.git_analysis:
        git = analysis.git_analysis
        if hasattr(git, "current_branch"):
            table.add_row("Git Branch", git.current_branch or "detached")
        if hasattr(git, "total_commits"):
            table.add_row("Total Commits", str(git.total_commits))
        if hasattr(git, "total_authors"):
            table.add_row("Contributors", str(git.total_authors))

    console.print(table)


def _display_metrics_table(analysis):
    """Display detailed metrics by language."""
    table = Table(title="Code Metrics by Language")

    table.add_column("Language", style="cyan")
    table.add_column("Files", style="green", justify="right")
    table.add_column("Lines", style="yellow", justify="right")
    table.add_column("Avg Lines/File", style="blue", justify="right")
    table.add_column("Avg Complexity", style="red", justify="right")

    # Group by language
    from collections import defaultdict

    lang_stats = defaultdict(lambda: {"files": 0, "lines": 0, "complexity": []})

    for file in analysis.files:
        stats = lang_stats[file.language]
        stats["files"] += 1
        stats["lines"] += file.lines
        if file.complexity:
            stats["complexity"].append(file.complexity.cyclomatic)

    # Sort by lines descending
    sorted_langs = sorted(lang_stats.items(), key=lambda x: x[1]["lines"], reverse=True)

    # Display
    for lang, stats in sorted_langs[:15]:  # Top 15 languages
        avg_lines = stats["lines"] // stats["files"] if stats["files"] > 0 else 0
        avg_complexity = (
            sum(stats["complexity"]) / len(stats["complexity"]) if stats["complexity"] else 0
        )

        table.add_row(
            lang,
            str(stats["files"]),
            f"{stats['lines']:,}",
            str(avg_lines),
            f"{avg_complexity:.1f}" if avg_complexity > 0 else "-",
        )

    console.print(table)


def _display_complexity_table(analysis):
    """Display complexity analysis."""
    # Get complex files
    complex_files = [f for f in analysis.files if f.complexity and f.complexity.cyclomatic > 10]

    if not complex_files:
        console.print("[yellow]No files with high complexity found (threshold: 10)[/yellow]")
        return

    # Sort by complexity
    complex_files.sort(key=lambda f: f.complexity.cyclomatic, reverse=True)

    table = Table(title="High Complexity Files")
    table.add_column("File", style="cyan")
    table.add_column("Complexity", style="red", justify="right")
    table.add_column("Lines", style="yellow", justify="right")
    table.add_column("Functions", style="green", justify="right")
    table.add_column("Classes", style="blue", justify="right")

    for file in complex_files[:20]:  # Top 20
        # Shorten path for display
        display_path = file.path
        if len(display_path) > 60:
            parts = Path(display_path).parts
            if len(parts) > 3:
                display_path = f".../{'/'.join(parts[-3:])}"

        table.add_row(
            display_path,
            str(file.complexity.cyclomatic),
            str(file.lines),
            str(len(file.functions)),
            str(len(file.classes)),
        )

    console.print(table)


def _display_ownership_table(git_analysis):
    """Display code ownership information."""
    if not hasattr(git_analysis, "author_stats") or not git_analysis.author_stats:
        console.print("[yellow]No author statistics available[/yellow]")
        return

    table = Table(title="Code Ownership")
    table.add_column("Author", style="cyan")
    table.add_column("Commits", style="green", justify="right")
    table.add_column("Lines Added", style="yellow", justify="right")
    table.add_column("Lines Removed", style="red", justify="right")
    table.add_column("Files Touched", style="blue", justify="right")

    # Sort by commits
    sorted_authors = sorted(
        git_analysis.author_stats.items(), key=lambda x: x[1].get("commit_count", 0), reverse=True
    )

    for author, stats in sorted_authors[:20]:  # Top 20
        # Truncate long author names
        display_author = author[:30] + "..." if len(author) > 30 else author

        table.add_row(
            display_author,
            str(stats.get("commit_count", 0)),
            f"{stats.get('lines_added', 0):,}",
            f"{stats.get('lines_removed', 0):,}",
            str(stats.get("files_touched", 0)),
        )

    console.print(table)


def _display_hotspots_table(git_analysis):
    """Display code hotspots (frequently changed files)."""
    if not hasattr(git_analysis, "hotspots") or not git_analysis.hotspots:
        console.print("[yellow]No hotspots analysis available[/yellow]")
        return

    table = Table(title="Code Hotspots (Frequently Changed Files)")
    table.add_column("File", style="cyan")
    table.add_column("Changes", style="red", justify="right")
    table.add_column("Authors", style="yellow", justify="right")
    table.add_column("Last Modified", style="blue")
    table.add_column("Complexity", style="green", justify="right")

    for hotspot in git_analysis.hotspots[:20]:  # Top 20
        # Shorten path
        display_path = hotspot.get("file", "")
        if len(display_path) > 50:
            parts = Path(display_path).parts
            if len(parts) > 3:
                display_path = f".../{'/'.join(parts[-3:])}"

        table.add_row(
            display_path,
            str(hotspot.get("change_count", 0)),
            str(hotspot.get("author_count", 0)),
            hotspot.get("last_modified", "")[:10],  # Date only
            str(hotspot.get("complexity", "-")),
        )

    console.print(table)


def _display_structure_tree(analysis):
    """Display directory structure as a tree."""
    tree = Tree(f"[bold]{analysis.root_path}[/bold]")

    # Build directory structure
    from collections import defaultdict

    dir_structure = defaultdict(list)

    for file in analysis.files[:100]:  # Limit to first 100 files
        parts = Path(file.path).parts
        for i in range(len(parts) - 1):
            parent = "/".join(parts[:i]) if i > 0 else "."
            child = parts[i]
            if child not in dir_structure[parent]:
                dir_structure[parent].append(child)

        # Add file to its directory
        if len(parts) > 1:
            parent = "/".join(parts[:-1])
            dir_structure[parent].append(parts[-1])

    # Build tree recursively
    def add_nodes(node, path):
        children = sorted(dir_structure.get(path, []))
        for child in children[:20]:  # Limit children
            child_path = f"{path}/{child}" if path != "." else child
            if child_path in dir_structure:
                # Directory
                child_node = node.add(f"[blue]{child}/[/blue]")
                add_nodes(child_node, child_path)
            else:
                # File
                node.add(child)

    add_nodes(tree, ".")

    console.print(Panel(tree, title="Directory Structure", border_style="green"))
