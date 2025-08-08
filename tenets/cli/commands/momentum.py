"""Momentum command - track development velocity and productivity."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from tenets import Tenets

console = Console()


def momentum(
    path: Path = typer.Option(
        Path("."),
        "--path", "-p",
        help="Repository path"
    ),
    since: str = typer.Option(
        "last-month",
        "--since", "-s",
        help="Time period to analyze"
    ),
    team: bool = typer.Option(
        False,
        "--team",
        help="Show team-wide statistics"
    ),
    author: Optional[str] = typer.Option(
        None,
        "--author", "-a",
        help="Show stats for specific author"
    ),
    weekly: bool = typer.Option(
        False,
        "--weekly",
        help="Show weekly breakdown"
    ),
    ctx: typer.Context = typer.Context,
):
    """
    Track development momentum and velocity metrics.
    
    Measures coding velocity, team productivity, and development trends
    to understand your project's momentum.
    
    Examples:
    
        # Personal momentum for last month
        tenets momentum
        
        # Team momentum for the quarter
        tenets momentum --team --since "3 months"
        
        # Individual contributor stats
        tenets momentum --author "alice@example.com"
        
        # Weekly breakdown
        tenets momentum --weekly --since "2 months"
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    try:
        tenets = Tenets()
        
        # Get velocity data
        velocity_data = tenets.momentum(
            path=path,
            since=since,
            team=team,
            author=author
        )
        
        if "error" in velocity_data:
            console.print(f"[red]Error:[/red] {velocity_data['error']}")
            raise typer.Exit(1)
        
        # Display header
        if not quiet:
            title = "Development Momentum"
            if team:
                title = "Team " + title
            elif author:
                title = f"{author}'s " + title
                
            console.print(Panel(
                f"[bold]{title}[/bold]\n"
                f"Period: {since}\n"
                f"Repository: {path}",
                title="🚀 Momentum Analysis",
                border_style="blue"
            ))
        
        # Show velocity chart
        if weekly and "weekly" in velocity_data:
            _display_velocity_chart(velocity_data["weekly"], "Weekly")
        
        # Show overall stats
        if "overall" in velocity_data:
            _display_overall_stats(velocity_data["overall"], team, author)
        
        # Show team stats if requested
        if team and "team" in velocity_data:
            _display_team_stats(velocity_data["team"])
        
        # Show author stats if requested
        if author and "author" in velocity_data:
            _display_author_stats(author, velocity_data["author"])
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _display_velocity_chart(weekly_data, title):
    """Display velocity chart."""
    if not weekly_data:
        return
    
    console.print(f"\n[bold]{title} Momentum[/bold]")
    
    # Find max for scaling
    max_commits = max(w.get("commits", 0) for w in weekly_data) if weekly_data else 1
    
    for week in weekly_data[-12:]:  # Last 12 weeks
        commits = week.get("commits", 0)
        bar_length = int((commits / max_commits) * 40) if max_commits > 0 else 0
        bar = "█" * bar_length
        
        week_label = week.get("week", "?")
        if isinstance(week_label, str):
            week_display = week_label
        else:
            week_display = f"W{week_label}"
        
        # Color based on velocity
        if commits == 0:
            color = "dim"
        elif commits < max_commits * 0.3:
            color = "red"
        elif commits < max_commits * 0.7:
            color = "yellow"
        else:
            color = "green"
        
        console.print(f"{week_display:>6} [{color}]{bar:<40}[/{color}] {commits:>3} commits")


def _display_overall_stats(stats, team: bool, author: Optional[str]):
    """Display overall statistics."""
    table = Table(title="Overall Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Basic stats
    table.add_row("Total Commits", str(stats.get("commit_count", 0)))
    table.add_row("Active Days", str(stats.get("active_days", 0)))
    table.add_row("Lines Added", f"{stats.get('lines_added', 0):,}")
    table.add_row("Lines Removed", f"{stats.get('lines_removed', 0):,}")
    table.add_row("Net Lines", f"{stats.get('net_lines', 0):+,}")
    table.add_row("Files Changed", str(stats.get("files_changed", 0)))
    
    # Calculated metrics
    if stats.get("active_days", 0) > 0:
        avg_commits = stats.get("commit_count", 0) / stats["active_days"]
        table.add_row("Avg Commits/Day", f"{avg_commits:.1f}")
    
    if stats.get("commit_count", 0) > 0:
        avg_change = stats.get("net_lines", 0) / stats["commit_count"]
        table.add_row("Avg Lines/Commit", f"{avg_change:+.1f}")
    
    console.print(table)


def _display_team_stats(team_data):
    """Display team statistics."""
    console.print(f"\n[bold]Team Statistics[/bold]")
    console.print(f"Total contributors: {team_data.get('total_authors', 0)}")
    
    # Top contributors table
    if team_data.get("author_stats"):
        table = Table(title="Top Contributors")
        table.add_column("Author", style="cyan")
        table.add_column("Commits", style="green", justify="right")
        table.add_column("Lines Added", style="yellow", justify="right")
        table.add_column("Active Days", style="blue", justify="right")
        table.add_column("Velocity", style="magenta", justify="right")
        
        # Sort by commits
        sorted_authors = sorted(
            team_data["author_stats"].items(),
            key=lambda x: x[1].get("commit_count", 0),
            reverse=True
        )
        
        for author, stats in sorted_authors[:10]:  # Top 10
            # Calculate velocity (commits per active day)
            active_days = stats.get("active_days", 1)
            velocity = stats.get("commit_count", 0) / active_days if active_days > 0 else 0
            
            table.add_row(
                author[:30] + "..." if len(author) > 30 else author,
                str(stats.get("commit_count", 0)),
                f"{stats.get('lines_added', 0):,}",
                str(active_days),
                f"{velocity:.1f}/day"
            )
        
        console.print(table)


def _display_author_stats(author, stats):
    """Display individual author statistics."""
    console.print(f"\n[bold]Statistics for {author}[/bold]")
    
    # Create a nice summary
    summary_items = [
        f"Commits: {stats.get('commit_count', 0)}",
        f"Active days: {stats.get('active_days', 0)}",
        f"Lines added: {stats.get('lines_added', 0):,}",
        f"Lines removed: {stats.get('lines_removed', 0):,}",
        f"Files touched: {stats.get('files_touched', 0)}",
    ]
    
    if stats.get("first_commit"):
        summary_items.append(f"First commit: {stats['first_commit'][:10]}")
    if stats.get("last_commit"):
        summary_items.append(f"Last commit: {stats['last_commit'][:10]}")
    
    for item in summary_items:
        console.print(f"  • {item}")
    
    # Show frequently changed files
    if stats.get("frequent_files"):
        console.print(f"\n[bold]Frequently Changed Files:[/bold]")
        for file_info in stats["frequent_files"][:10]:  # Top 10
            console.print(f"  • {file_info['file']}: {file_info['changes']} changes")