"""Momentum command implementation.

This command tracks and visualizes development velocity and team momentum
metrics over time.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from tenets.core.git import GitAnalyzer
from tenets.core.momentum import MomentumTracker
from tenets.core.reporting import ReportGenerator
from tenets.utils.logger import get_logger
from tenets.viz import MomentumVisualizer, TerminalDisplay


@click.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option(
    "--period",
    "-p",
    type=click.Choice(["day", "week", "sprint", "month"]),
    default="week",
    help="Time period for velocity calculation",
)
@click.option("--duration", "-d", type=int, default=12, help="Number of periods to analyze")
@click.option("--sprint-length", type=int, default=14, help="Sprint length in days")
@click.option("--output", "-o", type=click.Path(), help="Output file for report")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["terminal", "html", "json", "markdown"]),
    default="terminal",
    help="Output format",
)
@click.option(
    "--metrics",
    "-m",
    multiple=True,
    help="Specific metrics to track (velocity, throughput, cycle-time)",
)
@click.option("--team", is_flag=True, help="Show team metrics")
@click.option("--burndown", is_flag=True, help="Show burndown chart")
@click.option("--forecast", is_flag=True, help="Include velocity forecast")
@click.pass_context
def momentum(
    ctx,
    path: str,
    period: str,
    duration: int,
    sprint_length: int,
    output: Optional[str],
    format: str,
    metrics: List[str],
    team: bool,
    burndown: bool,
    forecast: bool,
):
    """Track development momentum and velocity.

    Analyzes repository activity to measure development velocity,
    team productivity, and momentum trends over time.

    Examples:
        tenets momentum
        tenets momentum --period=sprint --duration=6
        tenets momentum --burndown --team
        tenets momentum --forecast --format=html --output=velocity.html
    """
    logger = get_logger(__name__)
    config = ctx.obj["config"]

    # Initialize path
    target_path = Path(path).resolve()
    logger.info(f"Tracking momentum at: {target_path}")

    # Initialize momentum tracker
    tracker = MomentumTracker(config)
    git_analyzer = GitAnalyzer(target_path)

    # Calculate date range based on period and duration
    date_range = _calculate_date_range(period, duration, sprint_length)

    # Determine which metrics to calculate
    if metrics:
        selected_metrics = list(metrics)
    else:
        selected_metrics = ["velocity", "throughput", "cycle_time"]

    try:
        # Track momentum
        logger.info(f"Calculating {period}ly momentum...")
        momentum_data = tracker.track_momentum(
            target_path,
            period=period,
            since=date_range["since"],
            until=date_range["until"],
            metrics=selected_metrics,
            sprint_length=sprint_length,
        )

        # Add team metrics if requested
        if team:
            logger.info("Calculating team metrics...")
            momentum_data["team_metrics"] = _calculate_team_metrics(
                git_analyzer, date_range, sprint_length
            )

        # Add burndown if requested
        if burndown and period == "sprint":
            logger.info("Generating burndown data...")
            momentum_data["burndown"] = _generate_burndown_data(git_analyzer, sprint_length)

        # Add forecast if requested
        if forecast:
            logger.info("Generating velocity forecast...")
            momentum_data["forecast"] = _generate_forecast(momentum_data.get("velocity_data", []))

        # Display or save results
        if format == "terminal":
            _display_terminal_momentum(momentum_data, team, burndown, forecast)
        elif format == "json":
            _output_json_momentum(momentum_data, output)
        else:
            _generate_momentum_report(momentum_data, format, output, config)

        # Summary
        _print_momentum_summary(momentum_data)

    except Exception as e:
        logger.error(f"Momentum tracking failed: {e}")
        raise click.ClickException(str(e))


def _calculate_date_range(period: str, duration: int, sprint_length: int) -> Dict[str, datetime]:
    """Calculate date range for momentum tracking.

    Args:
        period: Time period type
        duration: Number of periods
        sprint_length: Sprint length in days

    Returns:
        Date range dictionary
    """
    now = datetime.now()

    if period == "day":
        days_back = duration
    elif period == "week":
        days_back = duration * 7
    elif period == "sprint":
        days_back = duration * sprint_length
    elif period == "month":
        days_back = duration * 30
    else:
        days_back = 90

    return {"since": now - timedelta(days=days_back), "until": now}


def _calculate_team_metrics(
    git_analyzer: GitAnalyzer, date_range: Dict[str, datetime], sprint_length: int
) -> Dict[str, Any]:
    """Calculate team-level metrics.

    Args:
        git_analyzer: Git analyzer instance
        date_range: Date range for analysis
        sprint_length: Sprint length in days

    Returns:
        Team metrics data
    """
    # Get commits in range
    commits = git_analyzer.get_commits(since=date_range["since"], until=date_range["until"])

    # Calculate metrics
    contributors = set()
    daily_commits = {}

    for commit in commits:
        contributors.add(commit.get("author", "Unknown"))
        date = commit.get("date", datetime.now()).date()
        daily_commits[date] = daily_commits.get(date, 0) + 1

    # Calculate velocity metrics
    total_days = (date_range["until"] - date_range["since"]).days
    active_days = len(daily_commits)

    return {
        "team_size": len(contributors),
        "active_contributors": len(contributors),
        "total_commits": len(commits),
        "avg_commits_per_day": len(commits) / max(1, total_days),
        "active_days": active_days,
        "productivity": (active_days / max(1, total_days)) * 100,
        "collaboration_index": _calculate_collaboration_index(commits),
    }


def _calculate_collaboration_index(commits: List[Dict[str, Any]]) -> float:
    """Calculate collaboration index from commits.

    Args:
        commits: List of commits

    Returns:
        Collaboration index (0-100)
    """
    # Simple heuristic: files touched by multiple authors
    file_authors = {}

    for commit in commits:
        author = commit.get("author", "Unknown")
        for file in commit.get("files", []):
            if file not in file_authors:
                file_authors[file] = set()
            file_authors[file].add(author)

    if not file_authors:
        return 0.0

    # Calculate percentage of files with multiple authors
    multi_author_files = sum(1 for authors in file_authors.values() if len(authors) > 1)
    return (multi_author_files / len(file_authors)) * 100


def _generate_burndown_data(git_analyzer: GitAnalyzer, sprint_length: int) -> Dict[str, Any]:
    """Generate burndown chart data.

    Args:
        git_analyzer: Git analyzer instance
        sprint_length: Sprint length in days

    Returns:
        Burndown data
    """
    # Get current sprint data
    now = datetime.now()
    sprint_start = now - timedelta(days=sprint_length)

    commits = git_analyzer.get_commits(since=sprint_start, until=now)

    # Calculate daily progress
    daily_work = {}
    for commit in commits:
        date = commit.get("date", now).date()
        # Simple metric: use files changed as work unit
        work = len(commit.get("files", []))
        daily_work[date] = daily_work.get(date, 0) + work

    # Generate burndown lines
    total_work = sum(daily_work.values())
    dates = []
    ideal_line = []
    actual_line = []

    remaining_work = total_work
    for day in range(sprint_length):
        date = (sprint_start + timedelta(days=day)).date()
        dates.append(str(date))

        # Ideal line
        ideal_remaining = total_work * (1 - (day / sprint_length))
        ideal_line.append(ideal_remaining)

        # Actual line
        if date in daily_work:
            remaining_work -= daily_work[date]
        actual_line.append(remaining_work)

    return {
        "dates": dates,
        "ideal_line": ideal_line,
        "actual_line": actual_line,
        "total_work": total_work,
        "remaining_work": remaining_work,
        "on_track": remaining_work <= ideal_line[-1] if ideal_line else True,
        "completion_percentage": ((total_work - remaining_work) / max(1, total_work)) * 100,
    }


def _generate_forecast(velocity_data: List[float]) -> Dict[str, Any]:
    """Generate velocity forecast.

    Args:
        velocity_data: Historical velocity data

    Returns:
        Forecast data
    """
    if len(velocity_data) < 3:
        return {"available": False, "reason": "Insufficient data"}

    # Simple moving average forecast
    recent_velocity = velocity_data[-3:]
    avg_velocity = sum(recent_velocity) / len(recent_velocity)

    # Calculate trend
    if len(velocity_data) >= 6:
        older_avg = sum(velocity_data[-6:-3]) / 3
        trend = ((avg_velocity - older_avg) / max(1, older_avg)) * 100
    else:
        trend = 0

    # Generate forecast
    forecast_periods = 3
    forecast = []
    for i in range(forecast_periods):
        # Apply trend
        forecast_value = avg_velocity * (1 + (trend / 100) * (i + 1))
        forecast.append(forecast_value)

    return {
        "available": True,
        "current_velocity": avg_velocity,
        "trend_percentage": trend,
        "forecast_values": forecast,
        "confidence": "medium" if len(velocity_data) >= 10 else "low",
    }


def _display_terminal_momentum(
    momentum_data: Dict[str, Any], show_team: bool, show_burndown: bool, show_forecast: bool
) -> None:
    """Display momentum data in terminal.

    Args:
        momentum_data: Momentum tracking data
        show_team: Whether to show team metrics
        show_burndown: Whether to show burndown
        show_forecast: Whether to show forecast
    """
    viz = MomentumVisualizer()
    viz.display_terminal(momentum_data, show_details=True)

    display = TerminalDisplay()

    # Show additional visualizations if requested
    if show_team and "team_metrics" in momentum_data:
        display.display_header("Team Metrics", style="single")
        display.display_metrics(momentum_data["team_metrics"], columns=2)

    if show_burndown and "burndown" in momentum_data:
        burndown = momentum_data["burndown"]
        display.display_header("Sprint Burndown", style="single")

        # Show progress bar
        completion = burndown.get("completion_percentage", 0)
        progress_bar = display.create_progress_bar(completion, 100)
        print(f"Progress: {progress_bar}")

        status = "On Track" if burndown.get("on_track", False) else "Behind Schedule"
        color = "green" if burndown.get("on_track", False) else "red"
        print(f"Status: {display.colorize(status, color)}")

    if show_forecast and "forecast" in momentum_data:
        forecast = momentum_data["forecast"]
        if forecast.get("available", False):
            display.display_header("Velocity Forecast", style="single")

            trend = forecast.get("trend_percentage", 0)
            trend_symbol = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            trend_color = "green" if trend > 5 else "red" if trend < -5 else "yellow"

            print(f"Current Velocity: {forecast.get('current_velocity', 0):.1f}")
            print(f"Trend: {display.colorize(trend_symbol, trend_color)} {abs(trend):.1f}%")
            print(f"Confidence: {forecast.get('confidence', 'low').upper()}")

            print("\nForecast (next 3 periods):")
            for i, value in enumerate(forecast.get("forecast_values", []), 1):
                print(f"  Period +{i}: {value:.1f}")


def _generate_momentum_report(
    momentum_data: Dict[str, Any], format: str, output: Optional[str], config: Any
) -> None:
    """Generate momentum report.

    Args:
        momentum_data: Momentum data
        format: Report format
        output: Output path
        config: Configuration
    """
    from tenets.core.reporting import ReportConfig

    generator = ReportGenerator(config)

    report_config = ReportConfig(
        title="Development Momentum Report", format=format, include_charts=True
    )

    output_path = Path(output) if output else Path(f"momentum_report.{format}")

    generator.generate(data=momentum_data, output_path=output_path, config=report_config)

    click.echo(f"Momentum report generated: {output_path}")


def _output_json_momentum(momentum_data: Dict[str, Any], output: Optional[str]) -> None:
    """Output momentum data as JSON.

    Args:
        momentum_data: Momentum data
        output: Output path
    """
    import json

    if output:
        with open(output, "w") as f:
            json.dump(momentum_data, f, indent=2, default=str)
        click.echo(f"Momentum data saved to: {output}")
    else:
        click.echo(json.dumps(momentum_data, indent=2, default=str))


def _print_momentum_summary(momentum_data: Dict[str, Any]) -> None:
    """Print momentum summary.

    Args:
        momentum_data: Momentum data
    """
    click.echo("\n" + "=" * 50)
    click.echo("MOMENTUM SUMMARY")
    click.echo("=" * 50)

    # Current velocity
    if "current_velocity" in momentum_data:
        click.echo(f"Current Velocity: {momentum_data['current_velocity']:.1f}")

    # Trend
    if "velocity_trend" in momentum_data:
        trend = momentum_data["velocity_trend"]
        if trend > 0:
            click.secho(f"Trend: ↑ +{trend:.1f}%", fg="green")
        elif trend < 0:
            click.secho(f"Trend: ↓ {trend:.1f}%", fg="red")
        else:
            click.echo("Trend: → Stable")

    # Team metrics
    if "team_metrics" in momentum_data:
        team = momentum_data["team_metrics"]
        click.echo(f"\nTeam Size: {team.get('team_size', 0)}")
        click.echo(f"Productivity: {team.get('productivity', 0):.1f}%")

    # Forecast
    if "forecast" in momentum_data:
        forecast = momentum_data["forecast"]
        if forecast.get("available", False):
            click.echo(f"\nForecast Confidence: {forecast.get('confidence', 'low').upper()}")
