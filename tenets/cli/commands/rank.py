"""Rank command - show ranked files without content."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
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
            output.write_text(output_content, encoding="utf-8")
            console.print(f"[green]OK[/green] Saved ranking to {output}")
            # Offer to open HTML in browser
            if format == "html" and sys.stdin.isatty():
                if click.confirm("\nWould you like to open it in your browser now?", default=False):
                    import webbrowser

                    file_path = output.resolve()
                    webbrowser.open(file_path.as_uri())
                    console.print("[green]OK[/green] Opened in browser")
        elif format in ["html", "xml", "json"]:
            # For HTML/XML/JSON, auto-save to a default file like distill does
            if sys.stdin.isatty():  # Interactive mode
                import re
                from datetime import datetime

                # Create filename from prompt
                safe_prompt = re.sub(r"[^\w\s-]", "", prompt[:30]).strip()
                safe_prompt = re.sub(r"[-\s]+", "-", safe_prompt)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Determine file extension
                ext = format.lower()
                if ext == "html":
                    ext = "html"
                elif ext == "xml":
                    ext = "xml"
                else:  # json
                    ext = "json"

                default_file = Path(f"tenets_rank_{safe_prompt}_{timestamp}.{ext}")
                default_file.write_text(output_content, encoding="utf-8")

                console.print(
                    f"[green]OK[/green] {format.upper()} output saved to [cyan]{default_file}[/cyan]"
                )
                console.print(f"[dim]File size:[/dim] {len(output_content):,} bytes")

                # Offer to open in browser for HTML, or folder for XML/JSON
                if format == "html":
                    if click.confirm(
                        "\nWould you like to open it in your browser now?", default=False
                    ):
                        import webbrowser

                        file_path = default_file.resolve()
                        webbrowser.open(file_path.as_uri())
                        console.print("[green]OK[/green] Opened in browser")
                    else:
                        console.print(
                            "[cyan]Tip:[/cyan] Open the file in a browser or use --output to specify a different path"
                        )
                # For XML/JSON, offer to open the folder
                elif click.confirm(
                    f"\nWould you like to open the folder containing the {format.upper()} file?",
                    default=False,
                ):
                    import platform
                    import webbrowser

                    folder = default_file.parent.resolve()
                    if platform.system() == "Windows":
                        import os

                        os.startfile(folder)
                    elif platform.system() == "Darwin":  # macOS
                        import subprocess

                        subprocess.run(["open", folder], check=False)
                    else:  # Linux
                        import subprocess

                        subprocess.run(["xdg-open", folder], check=False)
                    console.print(f"[green]OK[/green] Opened folder: {folder}")
            else:
                # Non-interactive mode: print raw output
                print(output_content)
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
            console.print("[green]OK[/green] Copied file list to clipboard")

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
    """Format as HTML with interactive features using shared template system."""
    import json

    # Prepare data for JavaScript
    files_data = []
    for file in files:
        file_data = {
            "path": str(file.path),
            "score": getattr(file, "relevance_score", 0.0),
            "rank": getattr(file, "relevance_rank", 0),
        }
        if hasattr(file, "relevance_factors"):
            file_data["factors"] = file.relevance_factors
        files_data.append(file_data)

    files_json = json.dumps(files_data)

    # Calculate statistics
    total_files = len(files)
    high_relevance_count = len([f for f in files if getattr(f, "relevance_score", 0) >= 0.5])
    max_score = max((getattr(f, "relevance_score", 0) for f in files), default=0)
    avg_score = sum(getattr(f, "relevance_score", 0) for f in files) / len(files) if files else 0

    # Enhanced custom styles for rank report
    custom_styles = """
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --info-color: #06b6d4;
            --background: #ffffff;
            --surface: #f8fafc;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border: #e2e8f0;
            --shadow: rgba(0, 0, 0, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--background);
            margin: 0;
            padding: 0;
        }

        /* Enhanced Rank Report Styles */
        .rank-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
        }

        .controls {
            background: #f8fafc;
            padding: 1rem 2rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .search-box {
            flex: 1;
            min-width: 200px;
            padding: 0.5rem 1rem;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            font-size: 1rem;
        }

        .export-button {
            background: #667eea;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.3s;
        }

        .export-button:hover {
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }

        .stat-card {
            background: white;
            border-radius: 0.5rem;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }

        .stat-label {
            font-size: 0.875rem;
            color: #64748b;
            margin-top: 0.25rem;
        }

        .file-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .file-item {
            background: white;
            margin: 1rem 0;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #667eea;
            transition: all 0.3s;
            position: relative;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .file-item:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .file-path {
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.9rem;
            color: #1e293b;
            margin-bottom: 0.5rem;
            word-break: break-all;
        }

        .file-score {
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
        }

        .copy-button {
            position: absolute;
            top: 3rem;
            right: 1rem;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 0.25rem;
            padding: 0.25rem 0.5rem;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.3s;
        }

        .copy-button:hover {
            background: #f1f5f9;
        }

        .copy-button.copied {
            background: #10b981;
            color: white;
            border-color: #10b981;
        }

        .factors {
            margin-top: 0.75rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .factor-item {
            background: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            color: #64748b;
            border: 1px solid #e2e8f0;
        }

        .tree-view {
            font-family: 'Monaco', 'Consolas', monospace;
            white-space: pre;
            background: #1e293b;
            color: #e2e8f0;
            padding: 1.5rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            margin: 1rem 0;
        }

        .highlight {
            background: yellow;
            font-weight: bold;
            padding: 0 2px;
        }

        /* Tab Interface */
        .tab-container {
            margin-bottom: 2rem;
        }

        .tab-nav {
            display: flex;
            align-items: center;
            background: #f8fafc;
            border-radius: 8px;
            padding: 8px;
            gap: 8px;
            flex-wrap: wrap;
        }

        .tab-button {
            background: none;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
            font-weight: 500;
            color: #64748b;
            transition: all 0.3s;
        }

        .tab-button:hover {
            background: #e2e8f0;
            color: #2d3748;
        }

        .tab-button.active {
            background: #667eea;
            color: white;
        }

        .tab-content {
            display: none;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-top: 1rem;
        }

        .tab-content.active {
            display: block;
        }

        /* Charts */
        .chart-container {
            position: relative;
            height: 400px;
            margin: 2rem 0;
        }
    </style>
    """

    # JavaScript for interactivity
    # Use chr(10) for newline to avoid f-string backslash issues in Python 3.11
    newline = chr(10)
    scripts = f"""
    <script>
        // Store files data globally
        window.filesData = {files_json};

        // Copy individual file path
        function copyFilePath(index) {{
            const file = window.filesData[index];
            if (file) {{
                navigator.clipboard.writeText(file.path).then(function() {{
                    const button = document.getElementById('copy-' + index);
                    const originalText = button.innerText;
                    button.innerText = 'Copied!';
                    button.classList.add('copied');
                    setTimeout(() => {{
                        button.innerText = originalText;
                        button.classList.remove('copied');
                    }}, 2000);
                }});
            }}
        }}

        // Copy all file paths
        function copyAllPaths() {{
            const paths = window.filesData.map(f => f.path).join('{newline}');
            navigator.clipboard.writeText(paths).then(function() {{
                const button = document.getElementById('copy-all-btn');
                const originalText = button.innerText;
                button.innerText = '✓ Copied!';
                button.classList.add('copied');
                setTimeout(() => {{
                    button.innerText = originalText;
                    button.classList.remove('copied');
                }}, 2000);
            }});
        }}

        // Export as JSON
        function exportAsJSON() {{
            const data = {{
                prompt: "{prompt.replace('"', '\\"')}",
                total_files: {total_files},
                files: window.filesData,
                statistics: {{
                    high_relevance: {high_relevance_count},
                    max_score: {max_score:.3f},
                    avg_score: {avg_score:.3f}
                }},
                generated_at: "{datetime.now().isoformat()}"
            }};
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ranked_files.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}

        // Export as CSV
        function exportAsCSV() {{
            let csv = 'Rank,Path,Score{newline}';
            window.filesData.forEach((file, index) => {{
                csv += `${{index + 1}},"${{file.path}}",${{file.score.toFixed(3)}}{newline}`;
            }});
            const blob = new Blob([csv], {{type: 'text/csv'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ranked_files.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}

        // Search/filter files
        function searchFiles() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const fileItems = document.querySelectorAll('.file-item');

            fileItems.forEach((item, index) => {{
                const file = window.filesData[index];
                const matchesSearch = searchTerm === '' ||
                    file.path.toLowerCase().includes(searchTerm);
                item.style.display = matchesSearch ? 'block' : 'none';
            }});
        }}

        // Tab switching
        function openTab(evt, tabName) {{
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {{
                tabContents[i].classList.remove('active');
            }}

            const tabButtons = document.getElementsByClassName('tab-button');
            for (let i = 0; i < tabButtons.length; i++) {{
                tabButtons[i].classList.remove('active');
            }}

            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');

            // Initialize chart if switching to chart tab
            if (tabName === 'chart-tab' && !window.chartInitialized) {{
                initializeChart();
                window.chartInitialized = true;
            }}
        }}

        // Initialize distribution chart
        function initializeChart() {{
            const ctx = document.getElementById('distChart');
            if (!ctx) return;

            // Group files by score ranges
            const scoreRanges = {{
                '0.0-0.2': 0,
                '0.2-0.4': 0,
                '0.4-0.6': 0,
                '0.6-0.8': 0,
                '0.8-1.0': 0
            }};

            window.filesData.forEach(file => {{
                const score = file.score;
                if (score <= 0.2) scoreRanges['0.0-0.2']++;
                else if (score <= 0.4) scoreRanges['0.2-0.4']++;
                else if (score <= 0.6) scoreRanges['0.4-0.6']++;
                else if (score <= 0.8) scoreRanges['0.6-0.8']++;
                else scoreRanges['0.8-1.0']++;
            }});

            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: Object.keys(scoreRanges),
                    datasets: [{{
                        label: 'Number of Files',
                        data: Object.values(scoreRanges),
                        backgroundColor: [
                            'rgba(239, 68, 68, 0.8)',
                            'rgba(245, 158, 11, 0.8)',
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(139, 92, 246, 0.8)'
                        ],
                        borderColor: [
                            'rgba(239, 68, 68, 1)',
                            'rgba(245, 158, 11, 1)',
                            'rgba(59, 130, 246, 1)',
                            'rgba(16, 185, 129, 1)',
                            'rgba(139, 92, 246, 1)'
                        ],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'File Score Distribution'
                        }},
                        legend: {{
                            display: false
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{
                                stepSize: 1
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    """

    # Build the HTML content sections
    header_html = f"""
    <div class="rank-header">
        <h1>🎯 Ranked Files</h1>
        <p>Query: <strong>{prompt}</strong></p>
        <p>Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    """

    # Controls section
    controls_html = """
    <div class="controls">
        <input type="text" id="searchBox" class="search-box" placeholder="🔍 Search files..." onkeyup="searchFiles()">
        <button class="export-button" onclick="exportAsJSON()">📥 Export JSON</button>
        <button class="export-button" onclick="exportAsCSV()">📊 Export CSV</button>
        <button id="copy-all-btn" class="export-button" onclick="copyAllPaths()">📋 Copy All Paths</button>
    </div>
    """

    # Statistics grid
    stats_html = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{total_files}</div>
            <div class="stat-label">Total Files</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{high_relevance_count}</div>
            <div class="stat-label">High Relevance</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{max_score:.2f}</div>
            <div class="stat-label">Top Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_score:.2f}</div>
            <div class="stat-label">Avg Score</div>
        </div>
    </div>
    """

    # Tab navigation
    tab_nav_html = """
    <div class="tab-container">
        <div class="tab-nav">
            <button class="tab-button active" onclick="openTab(event, 'list-tab')">📄 File List</button>
            <button class="tab-button" onclick="openTab(event, 'tree-tab')">🌳 Tree View</button>
            <button class="tab-button" onclick="openTab(event, 'chart-tab')">📊 Charts</button>
        </div>
    </div>
    """

    # File list content
    file_list_html = '<div id="list-tab" class="tab-content active"><ul class="file-list">'
    for i, file in enumerate(files):
        path = str(file.path)
        score = getattr(file, "relevance_score", 0.0)

        file_list_html += f"""
        <li class="file-item">
            <div class="file-path">{path}</div>
            {f'<div class="file-score">{score:.3f}</div>' if show_scores else ""}
            <button id="copy-{i}" class="copy-button" onclick="copyFilePath({i})">📋 Copy</button>
        """

        if show_factors and hasattr(file, "relevance_factors"):
            file_list_html += '<div class="factors">'
            for factor, value in file.relevance_factors.items():
                file_list_html += f'<span class="factor-item">{factor}: {value:.2%}</span>'
            file_list_html += "</div>"

        file_list_html += "</li>"

    file_list_html += "</ul></div>"

    # Tree view content
    tree_html = '<div id="tree-tab" class="tab-content">'
    if tree_view:
        # Generate tree structure
        from collections import defaultdict

        dirs = defaultdict(list)
        for file in files:
            dir_path = Path(file.path).parent
            dirs[dir_path].append(file)

        tree_html += '<div class="tree-view">'
        for dir_path in sorted(dirs.keys()):
            tree_html += f"📂 {dir_path}\n"
            for file in sorted(
                dirs[dir_path], key=lambda f: getattr(f, "relevance_score", 0.0), reverse=True
            ):
                name = Path(file.path).name
                score = getattr(file, "relevance_score", 0.0)
                tree_html += f"  📄 {name} [{score:.3f}]\n"
        tree_html += "</div>"
    else:
        tree_html += "<p>Enable --tree flag to see tree view</p>"
    tree_html += "</div>"

    # Chart content
    chart_html = """
    <div id="chart-tab" class="tab-content">
        <div class="chart-container">
            <canvas id="distChart"></canvas>
        </div>
    </div>
    """

    # Combine all sections
    content_html = f"""
    <div class="container">
        {header_html}
        {controls_html}
        {stats_html}
        {tab_nav_html}
        {file_list_html}
        {tree_html}
        {chart_html}
    </div>
    """

    # Build final HTML using template
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ranked Files - Tenets</title>
    {custom_styles}
    {scripts}
</head>
<body>
    {content_html}
</body>
</html>"""

    return html


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
