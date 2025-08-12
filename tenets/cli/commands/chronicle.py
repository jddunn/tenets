"""Chronicle command - summarize recent git history."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tenets.core.git import GitAnalyzer

console = Console()


def chronicle(
    path: Path = typer.Option(
        Path("."),
        "--path",
        "-p",
        help="Repository path",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        "-s",
        help="Time period (e.g., '2 weeks', '30 days') [not strictly enforced]",
    ),
    author: Optional[str] = typer.Option(
        None,
        "--author",
        "-a",
        help="Filter by author name/email",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum commits to display",
    ),
    ctx: typer.Context = typer.Context,
):
    """Show a concise chronicle of recent commits.

    Lists recent commits (optionally filtered by author) to provide
    quick project history context.
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        ga = GitAnalyzer(path)
        if not ga.is_repo():
            console.print(f"[red]No git repository found at {path}[/red]")
            raise typer.Exit(1)

        # Fetch recent commits
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Reading git history...", total=None)
            commits = ga.recent_commits(limit=limit)

        # Filter by author if requested
        if author:
            needle = author.lower()
            commits = [
                c
                for c in commits
                if needle in (c.author or "").lower() or needle in (c.email or "").lower()
            ]

        if not commits:
            console.print("[yellow]No commits found.[/yellow]")
            return

        if not quiet:
            console.print(
                Panel(
                    f"[bold]Recent Commits[/bold]\nRepo: {Path(path).resolve()}",
                    title="🕘 Chronicle",
                    border_style="blue",
                )
            )

        table = Table()
        table.add_column("SHA", style="cyan")
        table.add_column("Author", style="green")
        table.add_column("Message", style="white")
        table.add_column("Date", style="yellow", justify="right")

        from datetime import datetime

        for c in commits:
            date_str = (
                datetime.fromtimestamp(c.committed_date).strftime("%Y-%m-%d")
                if c.committed_date
                else ""
            )
            table.add_row(c.hexsha[:7], f"{c.author}", c.message.splitlines()[0][:80], date_str)

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
