"""Rank command - show ranked files without content."""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from tenets import Tenets
from tenets.utils.timing import CommandTimer

console = Console()


def _get_language_from_extension(file_path: Path) -> str:
    """Get language from file extension."""
    ext = file_path.suffix.lower()
    # Common language mappings
    lang_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".m": "objc",
        ".h": "c",
        ".hpp": "cpp",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".md": "markdown",
        ".rst": "rst",
        ".sql": "sql",
    }
    return lang_map.get(ext, "text")


# Expose pyperclip for optional clipboard support
try:
    import pyperclip as _pyperclip  # type: ignore

    pyperclip = _pyperclip
except Exception:
    pyperclip = None  # type: ignore


def rank(
    prompt: str = typer.Argument(..., help="Your query or task to rank files against"),
    path: Path = typer.Argument(Path(), help="Path to analyze (directory or files)"),
    # Output options
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown, json, xml, html, tree"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save output to file instead of stdout"
    ),
    # Ranking options
    mode: str = typer.Option(
        "balanced",  # Use same default as distill command for consistency
        "--mode",
        "-m",
        help="Ranking mode: fast (keyword only), balanced (TF-IDF + structure), thorough (deep analysis)",
    ),
    top: Optional[int] = typer.Option(None, "--top", "-t", help="Show only top N files"),
    min_score: Optional[float] = typer.Option(
        None, "--min-score", help="Minimum relevance score (0.0-1.0)"
    ),
    max_files: Optional[int] = typer.Option(
        None, "--max-files", help="Maximum number of files to show"
    ),
    # Display options
    tree_view: bool = typer.Option(False, "--tree", help="Show results as directory tree"),
    show_scores: bool = typer.Option(True, "--scores/--no-scores", help="Show relevance scores"),
    show_factors: bool = typer.Option(False, "--factors", help="Show ranking factor breakdown"),
    show_path: str = typer.Option(
        "relative", "--path-style", help="Path display: relative, absolute, name"
    ),
    # Filtering
    include: Optional[str] = typer.Option(
        None, "--include", "-i", help="Include file patterns (e.g., '*.py,*.js')"
    ),
    exclude: Optional[str] = typer.Option(
        None, "--exclude", "-e", help="Exclude file patterns (e.g., 'test_*,*.backup')"
    ),
    include_tests: bool = typer.Option(False, "--include-tests", help="Include test files"),
    exclude_tests: bool = typer.Option(
        False, "--exclude-tests", help="Explicitly exclude test files"
    ),
    # Features
    no_git: bool = typer.Option(False, "--no-git", help="Disable git signals in ranking"),
    session: Optional[str] = typer.Option(
        None, "--session", "-s", help="Use session for stateful ranking"
    ),
    # Info options
    show_stats: bool = typer.Option(False, "--stats", help="Show ranking statistics"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed debug information"),
    copy: bool = typer.Option(False, "--copy", help="Copy file list to clipboard"),
):
    """
    Rank files by relevance without showing their content.

    This command runs the same intelligent ranking as 'distill' but only shows
    the list of relevant files, their scores, and optionally the ranking factors.
    Useful for understanding what files would be included in context or for
    feeding file lists to other tools.

    Examples:

        # Show top 10 most relevant files
        tenets rank "implement OAuth2" --top 10

        # Show files above a score threshold
        tenets rank "fix bug" --min-score 0.3

        # Tree view with ranking factors
        tenets rank "add caching" --tree --factors

        # Export as JSON for automation
        tenets rank "review API" --format json -o ranked_files.json

        # Quick file list to clipboard
        tenets rank "database queries" --top 20 --copy --no-scores
    """
    # Initialize timer
    is_json_quiet = format.lower() == "json" and not output
    timer = CommandTimer(console, is_json_quiet)

    try:
        timer.start("Initializing ranking...")

        # Initialize tenets with same distiller pipeline
        tenets_instance = Tenets()

        # Use the same distiller pipeline that the distill command uses
        # This ensures consistent ranking behavior

        # Show progress only for non-JSON formats
        if format.lower() != "json" or output:
            console.print(f"[yellow]Ranking files for: {prompt[:50]}...[/yellow]")

        # Use distiller's ranking pipeline by calling rank_files directly
        # This ensures we get the same sophisticated ranking as distill
        result = tenets_instance.rank_files(
            prompt=prompt,
            paths=[path] if path else None,
            mode=mode,
            include_patterns=include.split(",") if include else None,
            exclude_patterns=exclude.split(",") if exclude else None,
            include_tests=include_tests if include_tests else None,
            exclude_tests=exclude_tests if exclude_tests else False,
            explain=show_factors,
        )

        ranked_files = result.files

        # Apply threshold filtering if min_score is set
        if min_score:
            ranked_files = [
                f for f in ranked_files if getattr(f, "relevance_score", 0) >= min_score
            ]

        # Apply limits
        if top:
            ranked_files = ranked_files[:top]
        if max_files:
            ranked_files = ranked_files[:max_files]

        # Format output
        output_content = _format_ranked_files(
            ranked_files,
            format=format,
            tree_view=tree_view,
            show_scores=show_scores,
            show_factors=show_factors,
            show_path=show_path,
            prompt=prompt,
            stats=None,  # Stats not available from rank_files yet
        )

        # Output results
        if output:
            output.write_text(output_content)
            console.print(f"[green]✓[/green] Saved ranking to {output}")
        elif format == "markdown" or format == "tree":
            console.print(output_content)
        else:
            print(output_content)

        # Check if we should copy to clipboard
        do_copy = copy
        try:
            # Check config flag for auto-copy (similar to distill command)
            cfg = getattr(tenets_instance, "config", None)
            if cfg and getattr(getattr(cfg, "output", None), "copy_on_rank", False):
                do_copy = True
        except Exception:
            pass

        # Copy to clipboard if requested or config enabled
        if do_copy and pyperclip:
            # Create simple file list for clipboard
            if show_scores:
                clip_content = "\n".join(
                    f"{f.path} ({f.relevance_score:.3f})" for f in ranked_files
                )
            else:
                clip_content = "\n".join(str(f.path) for f in ranked_files)
            pyperclip.copy(clip_content)
            console.print("[green]✓[/green] Copied file list to clipboard")

        # Show stats if requested
        if show_stats:
            # Stats not available from rank_files yet
            console.print("[yellow]Stats not available yet[/yellow]")

        timer.stop()

    except Exception as e:
        console.print(f"[red]Error:[/red] {e!s}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _format_ranked_files(
    files: List,
    format: str,
    tree_view: bool,
    show_scores: bool,
    show_factors: bool,
    show_path: str,
    prompt: str,
    stats: Optional[Dict] = None,
) -> str:
    """Format ranked files for output."""

    if format == "json":
        return _format_json(files, show_scores, show_factors, stats)
    elif format == "xml":
        return _format_xml(files, show_scores, show_factors, prompt)
    elif format == "html":
        return _format_html(files, show_scores, show_factors, prompt, tree_view)
    elif tree_view or format == "tree":
        return _format_tree(files, show_scores, show_factors, show_path)
    else:  # markdown
        return _format_markdown(files, show_scores, show_factors, show_path)


def _format_markdown(files: List, show_scores: bool, show_factors: bool, show_path: str) -> str:
    """Format as markdown list."""
    lines = ["# Ranked Files\n"]

    for i, file in enumerate(files, 1):
        path = _get_display_path(file.path, show_path)

        if show_scores:
            score = getattr(file, "relevance_score", 0.0)
            lines.append(f"{i}. **{path}** - Score: {score:.3f}")
        else:
            lines.append(f"{i}. **{path}**")

        if show_factors and hasattr(file, "relevance_factors"):
            factors = file.relevance_factors
            lines.append("   - Factors:")
            for factor, value in factors.items():
                lines.append(f"     - {factor}: {value:.2%}")

        lines.append("")

    return "\n".join(lines)


def _format_tree(files: List, show_scores: bool, show_factors: bool, show_path: str) -> str:
    """Format as tree structure sorted by relevance."""
    import platform

    # Use simple characters on Windows to avoid encoding issues
    if platform.system() == "Windows":
        tree = Tree("[Ranked Files (sorted by relevance)]")
    else:
        tree = Tree("📁 Ranked Files (sorted by relevance)")

    # Group by directory while preserving order
    from collections import defaultdict

    dirs = defaultdict(list)

    for file in files:
        dir_path = Path(file.path).parent
        dirs[dir_path].append(file)

    # Sort directories by the highest scoring file in each
    def get_max_score(dir_path):
        return max((getattr(f, "relevance_score", 0.0) for f in dirs[dir_path]), default=0.0)

    sorted_dirs = sorted(dirs.keys(), key=get_max_score, reverse=True)

    # Build tree with sorted directories and files
    import platform

    # Use simple characters on Windows to avoid encoding issues
    dir_prefix = "[D]" if platform.system() == "Windows" else "📂"
    file_prefix = "[F]" if platform.system() == "Windows" else "📄"

    for dir_path in sorted_dirs:
        dir_branch = tree.add(f"{dir_prefix} {dir_path}")
        # Sort files within directory by score
        sorted_files = sorted(
            dirs[dir_path], key=lambda f: getattr(f, "relevance_score", 0.0), reverse=True
        )
        for file in sorted_files:
            name = Path(file.path).name
            if show_scores:
                score = getattr(file, "relevance_score", 0.0)
                file_text = f"{file_prefix} {name} [{score:.3f}]"
            else:
                file_text = f"{file_prefix} {name}"

            file_branch = dir_branch.add(file_text)

            if show_factors and hasattr(file, "relevance_factors"):
                factors = file.relevance_factors
                for factor, value in factors.items():
                    file_branch.add(f"{factor}: {value:.2%}")

    # Convert to string
    from io import StringIO

    from rich.console import Console

    string_io = StringIO()
    temp_console = Console(file=string_io, force_terminal=True)
    temp_console.print(tree)
    return string_io.getvalue()


def _format_json(files: List, show_scores: bool, show_factors: bool, stats: Optional[Dict]) -> str:
    """Format as JSON."""
    data = {"total_files": len(files), "files": []}

    for file in files:
        file_data = {
            "path": str(file.path),
            "rank": getattr(file, "relevance_rank", 0),
        }

        if show_scores:
            file_data["score"] = getattr(file, "relevance_score", 0.0)

        if show_factors and hasattr(file, "relevance_factors"):
            file_data["factors"] = file.relevance_factors

        data["files"].append(file_data)

    if stats:
        data["stats"] = stats.to_dict() if hasattr(stats, "to_dict") else stats

    return json.dumps(data, indent=2)


def _format_xml(files: List, show_scores: bool, show_factors: bool, prompt: str) -> str:
    """Format as XML."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append("<ranking>")
    lines.append(f"  <prompt>{prompt}</prompt>")
    lines.append(f"  <total_files>{len(files)}</total_files>")
    lines.append("  <files>")

    for file in files:
        lines.append("    <file>")
        lines.append(f"      <path>{file.path}</path>")
        lines.append(f"      <rank>{getattr(file, 'relevance_rank', 0)}</rank>")

        if show_scores:
            lines.append(f"      <score>{getattr(file, 'relevance_score', 0.0):.3f}</score>")

        if show_factors and hasattr(file, "relevance_factors"):
            lines.append("      <factors>")
            for factor, value in file.relevance_factors.items():
                lines.append(f"        <{factor}>{value:.3f}</{factor}>")
            lines.append("      </factors>")

        lines.append("    </file>")

    lines.append("  </files>")
    lines.append("</ranking>")

    return "\n".join(lines)


def _format_html(
    files: List, show_scores: bool, show_factors: bool, prompt: str, tree_view: bool
) -> str:
    """Format as HTML with interactive features."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ranked Files - Tenets</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
        .prompt {{ background: #f0f0f0; padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
        .file-list {{ list-style: none; padding: 0; }}
        .file-item {{ background: #f8f8f8; margin: 10px 0; padding: 15px; border-radius: 4px; border-left: 3px solid #4CAF50; }}
        .file-path {{ font-family: 'Monaco', 'Consolas', monospace; font-size: 14px; color: #2196F3; }}
        .file-score {{ float: right; background: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
        .factors {{ margin-top: 10px; font-size: 12px; color: #666; }}
        .factor-item {{ display: inline-block; margin-right: 15px; }}
        .tree-view {{ font-family: 'Monaco', 'Consolas', monospace; white-space: pre; background: #2d2d2d; color: #f8f8f2; padding: 20px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Ranked Files</h1>
        <div class="prompt">
            <strong>Query:</strong> {prompt}
        </div>
        <div class="stats">
            <strong>Total Files:</strong> {len(files)}
        </div>
"""

    if tree_view:
        # Tree view
        html += '<div class="tree-view">'
        html += _build_html_tree(files, show_scores, show_factors)
        html += "</div>"
    else:
        # List view
        html += '<ul class="file-list">'
        for i, file in enumerate(files, 1):
            score_html = ""
            if show_scores:
                score = getattr(file, "relevance_score", 0.0)
                score_html = f'<span class="file-score">{score:.3f}</span>'

            factors_html = ""
            if show_factors and hasattr(file, "relevance_factors"):
                factors_html = '<div class="factors">'
                for factor, value in file.relevance_factors.items():
                    factors_html += f'<span class="factor-item">{factor}: {value:.2%}</span>'
                factors_html += "</div>"

            html += f"""
            <li class="file-item">
                {score_html}
                <span class="file-path">{file.path}</span>
                {factors_html}
            </li>
            """

        html += "</ul>"

    html += """
    </div>
</body>
</html>
"""
    return html


def _build_html_tree(files: List, show_scores: bool, show_factors: bool) -> str:
    """Build HTML tree representation sorted by relevance."""
    # Group by directory
    from collections import defaultdict

    dirs = defaultdict(list)

    for file in files:
        dir_path = Path(file.path).parent
        dirs[dir_path].append(file)

    # Sort directories by the highest scoring file in each
    def get_max_score(dir_path):
        return max((getattr(f, "relevance_score", 0.0) for f in dirs[dir_path]), default=0.0)

    sorted_dirs = sorted(dirs.keys(), key=get_max_score, reverse=True)

    lines = []
    for dir_path in sorted_dirs:
        lines.append(f"📂 {dir_path}/")
        # Sort files within directory by score
        sorted_files = sorted(
            dirs[dir_path], key=lambda f: getattr(f, "relevance_score", 0.0), reverse=True
        )
        for file in sorted_files:
            name = Path(file.path).name
            score_str = ""
            if show_scores:
                score = getattr(file, "relevance_score", 0.0)
                score_str = f" [{score:.3f}]"
            lines.append(f"  📄 {name}{score_str}")

            if show_factors and hasattr(file, "relevance_factors"):
                for factor, value in file.relevance_factors.items():
                    lines.append(f"      {factor}: {value:.2%}")

    return "\n".join(lines)


def _get_display_path(path, style: str) -> str:
    """Get display path based on style."""
    # Ensure path is a Path object
    if not isinstance(path, Path):
        path = Path(path)

    if style == "absolute":
        return str(path.absolute())
    elif style == "name":
        return path.name
    else:  # relative
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)


def _show_stats(stats) -> None:
    """Show ranking statistics."""
    table = Table(title="Ranking Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if hasattr(stats, "to_dict"):
        stats_dict = stats.to_dict()
    else:
        stats_dict = stats

    for key, value in stats_dict.items():
        if isinstance(value, float):
            table.add_row(key.replace("_", " ").title(), f"{value:.3f}")
        else:
            table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)
