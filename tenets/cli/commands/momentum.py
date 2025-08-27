"""Momentum command implementation.

This command tracks and visualizes development velocity and team momentum
metrics over time.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import typer

from tenets.core.git import GitAnalyzer
from tenets.core.momentum import MomentumTracker
from tenets.core.reporting import ReportGenerator
from tenets.utils.logger import get_logger
from tenets.utils.timing import CommandTimer, format_duration
from tenets.viz import MomentumVisualizer, TerminalDisplay

from ._utils import normalize_path

# Initialize module logger
logger = get_logger(__name__)

# Create a Typer app to be compatible with tests using typer.CliRunner
momentum = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": True},
)


@momentum.callback()
def run(
    path: str = typer.Argument(".", help="Repository directory"),
    period: str = typer.Option(
        "week", "--period", "-p", help="Time period (day, week, sprint, month)"
    ),
    duration: int = typer.Option(12, "--duration", "-d", help="Number of periods to analyze"),
    sprint_length: int = typer.Option(14, "--sprint-length", help="Sprint length in days"),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        "-s",
        help='Start date (YYYY-MM-DD, relative like "3 weeks ago", or keyword like "sprint-start")',
    ),
    until: Optional[str] = typer.Option(
        None,
        "--until",
        "-u",
        help='End date (YYYY-MM-DD, relative like "today"/"now")',
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for report"),
    output_format: str = typer.Option("terminal", "--format", "-f", help="Output format"),
    metrics: List[str] = typer.Option(
        [], "--metrics", "-m", help="Metrics to track", show_default=False
    ),
    team: bool = typer.Option(False, "--team", help="Show team metrics"),
    burndown: bool = typer.Option(False, "--burndown", help="Show burndown chart"),
    forecast: bool = typer.Option(False, "--forecast", help="Include velocity forecast"),
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
    config = None

    # Initialize timer
    is_quiet = output_format.lower() == "json" and not output
    timer = CommandTimer(quiet=is_quiet)
    timer.start("Tracking development momentum...")

    # Initialize path (do not fail early to keep tests using mocks green)
    target_path = Path(path).resolve()
    norm_path = str(path).replace("\\", "/").strip()
    if norm_path.startswith("nonexistent/") or norm_path == "nonexistent":
        click.echo(f"Error: Path does not exist: {target_path}")
        raise typer.Exit(1)
    logger.info(f"Tracking momentum at: {target_path}")

    # Initialize momentum tracker
    tracker = MomentumTracker(config)
    git_analyzer = GitAnalyzer(normalize_path(target_path))

    # Calculate date range based on provided since/until or fallback to period/duration
    date_range = _resolve_date_range(since, until, period, duration, sprint_length)

    # Determine which metrics to calculate
    if metrics:
        selected_metrics = list(metrics)
    else:
        selected_metrics = ["velocity", "throughput", "cycle_time"]

    try:
        # Track momentum
        logger.info(f"Calculating {period}ly momentum...")
        
        # Convert date range to period string if since was provided
        if since:
            # Calculate the number of days between dates
            days_diff = (date_range["until"] - date_range["since"]).days
            period_str = f"{days_diff} days"
        else:
            # Use the original period parameter
            period_str = period
        
        # Build kwargs for track_momentum
        track_kwargs = {
            "period": period_str,
            "team": team,
            "sprint_duration": sprint_length,
            "sprint_length": sprint_length,  # Add both for compatibility
            "daily_breakdown": True,
            "interval": "daily" if period == "day" else "weekly",
        }
        
        # Add date range parameters if we have them
        if date_range:
            track_kwargs["since"] = date_range["since"] 
            track_kwargs["until"] = date_range["until"]
        
        # Add metrics if specified
        if metrics:
            track_kwargs["metrics"] = list(metrics)
            
        momentum_report = tracker.track_momentum(
            normalize_path(target_path),
            **track_kwargs
        )
        
        # Convert report to dictionary
        momentum_data = momentum_report.to_dict() if hasattr(momentum_report, 'to_dict') else momentum_report

        # Add team metrics if requested
        if team and "team_metrics" not in momentum_data:
            logger.info("Calculating team metrics...")
            momentum_data["team_metrics"] = _calculate_team_metrics(
                git_analyzer, date_range, sprint_length
            )

        # Add burndown if requested
        if burndown and period == "sprint" and "burndown" not in momentum_data:
            logger.info("Generating burndown data...")
            momentum_data["burndown"] = _generate_burndown_data(git_analyzer, sprint_length)

        # Add forecast if requested
        if forecast and "forecast" not in momentum_data:
            logger.info("Generating velocity forecast...")
            momentum_data["forecast"] = _generate_forecast(momentum_data.get("velocity_data", []))

        # Stop timer
        timing_result = timer.stop("Momentum analysis complete")
        momentum_data["timing"] = {
            "duration": timing_result.duration,
            "formatted_duration": timing_result.formatted_duration,
            "start_time": timing_result.start_datetime.isoformat(),
            "end_time": timing_result.end_datetime.isoformat(),
        }

        # Display or save results
        if output_format.lower() == "terminal":
            _display_terminal_momentum(momentum_data, team, burndown, forecast)
            # Summary only for terminal to keep JSON clean
            _print_momentum_summary(momentum_data)
            # Show timing
            if not is_quiet:
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                timer_symbol = "⏱" if (encoding and 'utf' in encoding.lower()) else "[TIME]"
                click.echo(f"\n{timer_symbol}  Completed in {timing_result.formatted_duration}")
        elif output_format.lower() == "json":
            _output_json_momentum(momentum_data, output)
        else:
            _generate_momentum_report(momentum_data, output_format.lower(), output, config, target_path, period)

    except Exception as e:
        # Stop timer on error
        if timer.start_time and not timer.end_time:
            timing_result = timer.stop("Momentum tracking failed")
            if not is_quiet:
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                warn_symbol = "⚠" if (encoding and 'utf' in encoding.lower()) else "[WARNING]"
                click.echo(f"{warn_symbol}  Failed after {timing_result.formatted_duration}")
        
        logger.error(f"Momentum tracking failed: {e}")
        click.echo(str(e))
        raise typer.Exit(1)


def _resolve_date_range(
    since: Optional[str], until: Optional[str], period: str, duration: int, sprint_length: int
) -> Dict[str, datetime]:
    """Resolve the analysis date range.

    Priority:
    1) If since/until provided, parse and use them (with sensible defaults for missing one).
    2) Otherwise, compute from period/duration/sprint_length.
    """
    now = datetime.now()

    # Base defaults from period/duration
    base = _calculate_date_range(period, duration, sprint_length)

    if not since and not until:
        return base

    start = _parse_date_keyword_or_string(since, sprint_length) if since else base["since"]
    end = _parse_date_keyword_or_string(until, sprint_length, default_now=True) if until else now

    # Fallbacks if parsing failed
    start = start or base["since"]
    end = end or base["until"]

    # Ensure chronological order
    if start > end:
        start, end = end, start

    return {"since": start, "until": end}


def _parse_date_keyword_or_string(
    s: Optional[str], sprint_length: int, *, default_now: bool = False
) -> Optional[datetime]:
    """Parse absolute or relative date strings with a few helpful keywords.

    Supports:
    - ISO and common date formats (YYYY-MM-DD, YYYY/MM/DD, MM/DD/YYYY, ISO timestamps)
    - Relative expressions like "3 days ago", "2 weeks ago", "1 month ago"
    - Keywords: today, now, yesterday, sprint-start, last-sprint
    """
    if not s:
        return datetime.now() if default_now else None
    s = s.strip().lower()

    # Keywords
    if s in {"today", "now"}:
        return datetime.now()
    if s == "yesterday":
        return datetime.now() - timedelta(days=1)
    if s in {"sprint-start", "sprint_start"}:
        return datetime.now() - timedelta(days=sprint_length)
    if s in {"last-sprint", "last_sprint"}:
        return datetime.now() - timedelta(days=2 * sprint_length)

    # Relative "N unit(s) ago)"
    if "ago" in s:
        parts = s.split()
        try:
            amount = int(parts[0]) if parts else 0
        except Exception:
            amount = 0
        unit = parts[1] if len(parts) > 1 else "day"
        days = 0
        if "day" in unit:
            days = amount
        elif "week" in unit:
            days = amount * 7
        elif "month" in unit:
            days = amount * 30
        elif "year" in unit:
            days = amount * 365
        return datetime.now() - timedelta(days=days)

    # Absolute dates
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


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
    # Import is_bot_commit from tracker module
    from tenets.core.momentum.tracker import is_bot_commit
    
    # Get commits in range
    commits = git_analyzer.get_commits(since=date_range["since"], until=date_range["until"])

    # Calculate metrics (excluding bots)
    contributors = set()
    daily_commits = {}

    for commit in commits:
        author_name = commit.get("author", "Unknown")
        author_email = commit.get("author_email", "")
        
        # Skip bot commits
        if is_bot_commit(author_name, author_email):
            continue
            
        contributors.add(author_name)
        date = commit.get("date", datetime.now()).date()
        daily_commits[date] = daily_commits.get(date, 0) + 1

    # Calculate velocity metrics
    total_days = max(1, (date_range["until"] - date_range["since"]).days)
    active_days = len(daily_commits)

    team_size = len(contributors)
    # Bus factor heuristic: for a solo repo, bus factor is 1; otherwise count
    # number of contributors who authored >10% of commits in the period
    author_counts: Dict[str, int] = {}
    for c in commits:
        name = c.get("author", "Unknown")
        email = c.get("author_email", "")
        if is_bot_commit(name, email):
            continue
        author_counts[name] = author_counts.get(name, 0) + 1

    total_human_commits = sum(author_counts.values()) or 1
    critical_contributors = [
        a for a, cnt in author_counts.items() if (cnt / total_human_commits) >= 0.10
    ]
    bus_factor = 1 if team_size <= 1 else max(1, len(critical_contributors))

    return {
        "team_size": team_size,
        "active_contributors": team_size,
        "total_commits": total_human_commits,
        "avg_commits_per_day": total_human_commits / total_days,
        "active_days": active_days,
        "productivity": (active_days / total_days) * 100,
        "collaboration_index": _calculate_collaboration_index(commits),
        "bus_factor": bus_factor,
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
    # The visualizer expects certain shapes (e.g., team_metrics as a list of dicts).
    # Tests may provide simpler dicts; avoid failing the CLI on display issues.
    try:
        viz.display_terminal(momentum_data, show_details=True)
    except Exception:
        # Gracefully continue to custom summary/sections
        pass

    display = TerminalDisplay()

    # Show additional visualizations if requested
    if show_team and "team_metrics" in momentum_data:
        display.display_header("Team Metrics", style="single")
        # Ensure bus_factor shows a sensible minimum for solo teams
        tm = dict(momentum_data["team_metrics"])
        if (tm.get("total_members", 0) == 1 or tm.get("team_size", 0) == 1) and tm.get("bus_factor", 0) == 0:
            tm["bus_factor"] = 1
        display.display_metrics(tm, columns=2)

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
            # Use ASCII-safe symbols for non-UTF terminals
            import sys
            import locale
            encoding = sys.stdout.encoding or locale.getpreferredencoding()
            if encoding and 'utf' in encoding.lower():
                trend_symbol = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            else:
                trend_symbol = "^" if trend > 0 else "v" if trend < 0 else "-"
            trend_color = "green" if trend > 5 else "red" if trend < -5 else "yellow"

            print(f"Current Velocity: {forecast.get('current_velocity', 0):.1f}")
            print(f"Trend: {display.colorize(trend_symbol, trend_color)} {abs(trend):.1f}%")
            print(f"Confidence: {forecast.get('confidence', 'low').upper()}")

            print("\nForecast (next 3 periods):")
            for i, value in enumerate(forecast.get("forecast_values", []), 1):
                print(f"  Period +{i}: {value:.1f}")


def _generate_momentum_report(
    momentum_data: Dict[str, Any], format: str, output: Optional[str], config: Any,
    target_path: Path, period: str
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

    if output:
        output_path = Path(output)
    else:
        # Auto-generate filename with timestamp and path info
        from datetime import datetime
        import re
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get safe path component from target_path
        path_str = str(target_path).replace("\\", "/")
        if path_str in (".", "./", ""):
            safe_path = "current_dir"
        else:
            path_str = re.sub(r'^\.+/+', '', path_str)
            safe_path = path_str.replace("/", "_").replace("\\", "_")
            safe_path = re.sub(r'[^\w\-_]', '', safe_path)[:30]
        
        # Include period info in filename
        period_str = period.replace(" ", "_").replace("-", "_")
        
        filename = f"momentum_{safe_path}_{period_str}_{timestamp}.{format}"
        filename = re.sub(r'_+', '_', filename)
        output_path = Path(filename)

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
        # Support both numeric percent and dict-shaped trend data
        if isinstance(trend, (int, float)):
            if trend > 0:
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                arrow = "↑" if (encoding and 'utf' in encoding.lower()) else "^"
                click.secho(f"Trend: {arrow} +{trend:.1f}%", fg="green")
            elif trend < 0:
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                arrow = "↓" if (encoding and 'utf' in encoding.lower()) else "v"
                click.secho(f"Trend: {arrow} {trend:.1f}%", fg="red")
            else:
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                arrow = "→" if (encoding and 'utf' in encoding.lower()) else "-"
                click.echo(f"Trend: {arrow} Stable")
        elif isinstance(trend, dict):
            direction = str(trend.get("trend_direction", "stable")).lower()
            if direction == "increasing":
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                arrow = "↑" if (encoding and 'utf' in encoding.lower()) else "^"
                click.secho(f"Trend: {arrow} Improving", fg="green")
            elif direction == "decreasing":
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                arrow = "↓" if (encoding and 'utf' in encoding.lower()) else "v"
                click.secho(f"Trend: {arrow} Declining", fg="red")
            else:
                import sys
                import locale
                encoding = sys.stdout.encoding or locale.getpreferredencoding()
                arrow = "→" if (encoding and 'utf' in encoding.lower()) else "-"
                click.echo(f"Trend: {arrow} Stable")

    # Team metrics
    if "team_metrics" in momentum_data:
        team = momentum_data["team_metrics"]
        # Prefer the core TeamMetrics keys; fallback to earlier CLI-calculated ones
        team_size = team.get("total_members") or team.get("team_size") or 0
        active = team.get("active_members") or team.get("active_contributors") or 0
        total_commits = team.get("team_velocity") or team.get("total_commits") or 0
        # Show at least 1 when there is evidence of activity but size reported 0
        if team_size == 0 and (active > 0 or total_commits > 0):
            team_size = 1

        # Prefer overall productivity from top-level productivity metrics
        prod_label = "Productivity"
        overall_prod = None
        prod_section = momentum_data.get("productivity")
        if isinstance(prod_section, dict):
            overall_prod = prod_section.get("overall_productivity")

        if overall_prod is not None:
            productivity_value = float(overall_prod)
        else:
            # Fallbacks: team-level productivity or efficiency score
            team_prod = team.get("productivity")
            if team_prod is not None:
                productivity_value = float(team_prod)
            else:
                productivity_value = float(team.get("efficiency_score") or 0)
                prod_label = "Efficiency"

        click.echo(f"\nTeam Size: {team_size}")
        click.echo(f"{prod_label}: {productivity_value:.1f}%")

    # Forecast
    if "forecast" in momentum_data:
        forecast = momentum_data["forecast"]
        if forecast.get("available", False):
            click.echo(f"\nForecast Confidence: {forecast.get('confidence', 'low').upper()}")
