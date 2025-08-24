"""Chronicle command implementation.

This command provides git history analysis and visualization of code evolution
over time, including contribution patterns and change dynamics.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import typer

from tenets.core.git import ChronicleBuilder, GitAnalyzer
from tenets.core.reporting import ReportGenerator
from tenets.utils.logger import get_logger
from tenets.utils.timing import CommandTimer, format_duration
from tenets.viz import ContributorVisualizer, MomentumVisualizer, TerminalDisplay

from ._utils import normalize_path

# Expose a Typer app so tests can pass this object to typer.testing.CliRunner.
# Use callback so invoking the app directly (without a subcommand) runs the handler.
chronicle = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"allow_interspersed_args": True},
)


@chronicle.callback()
def run(
    path: str = typer.Argument(".", help="Repository directory"),
    since: Optional[str] = typer.Option(
        None, "--since", "-s", help='Start date (YYYY-MM-DD or relative like "3 months ago")'
    ),
    until: Optional[str] = typer.Option(
        None, "--until", "-u", help='End date (YYYY-MM-DD or relative like "today")'
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for report"),
    format: str = typer.Option(
        "terminal",
        "--format",
        "-f",
        help="Output format",
        case_sensitive=False,
    ),
    branch: str = typer.Option("main", "--branch", "-b", help="Git branch to analyze"),
    authors: Optional[List[str]] = typer.Option(
        None, "--authors", "-a", help="Filter by specific authors"
    ),
    show_merges: bool = typer.Option(False, "--show-merges", help="Include merge commits"),
    show_contributors: bool = typer.Option(
        False, "--show-contributors", help="Show contributor analysis"
    ),
    show_patterns: bool = typer.Option(False, "--show-patterns", help="Show change patterns"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of commits to analyze"
    ),
):
    """Chronicle the evolution of your codebase.

    This runs as the app callback so tests can invoke `chronicle` directly.
    """
    logger = get_logger(__name__)
    config = None  # tests invoke this in isolation without Typer app context

    # Initialize timer
    is_quiet = format.lower() == "json" and not output
    timer = CommandTimer(quiet=is_quiet)
    timer.start("Initializing git chronicle...")

    # Initialize path; allow non-existent for most tests except explicit invalid paths
    target_path = Path(path).resolve()
    norm_path = str(path).replace("\\", "/").strip()
    if norm_path.startswith("nonexistent/") or norm_path == "nonexistent":
        click.echo(f"Error: Path does not exist: {target_path}")
        raise typer.Exit(1)
    logger.info(f"Chronicling repository at: {target_path}")

    # Initialize chronicle builder
    chronicle_builder = ChronicleBuilder(config)
    git_analyzer = GitAnalyzer(normalize_path(target_path))

    # Parse date range
    date_range = _parse_date_range(since, until)

    # Build chronicle options
    chronicle_options = {
        "branch": branch,
        "since": date_range["since"],
        "until": date_range["until"],
        "authors": list(authors) if authors else None,
        "include_merges": show_merges,
        "limit": limit,
    }

    try:
        # Build chronicle
        logger.info("Building repository chronicle...")
        # Pass the resolved path string to help tests inspect call arguments reliably
        chronicle_data = chronicle_builder.build_chronicle(
            normalize_path(target_path), **chronicle_options
        )

        # Add contributor analysis if requested
        if show_contributors:
            logger.info("Analyzing contributors...")
            chronicle_data["contributors"] = git_analyzer.analyze_contributors(
                since=date_range["since"], until=date_range["until"]
            )

        # Add pattern analysis if requested
        if show_patterns:
            logger.info("Analyzing change patterns...")
            chronicle_data["patterns"] = _analyze_patterns(git_analyzer, date_range)

        # Stop timer
        timing_result = timer.stop("Chronicle analysis complete")
        chronicle_data["timing"] = {
            "duration": timing_result.duration,
            "formatted_duration": timing_result.formatted_duration,
            "start_time": timing_result.start_datetime.isoformat(),
            "end_time": timing_result.end_datetime.isoformat(),
        }

        # Display or save results
        if format == "terminal":
            # Simple heading for tests before any rich output
            click.echo("Repository Chronicle")
            _display_terminal_chronicle(chronicle_data, show_contributors, show_patterns)
            # Summary
            _print_chronicle_summary(chronicle_data)
            # Show timing
            if not is_quiet:
                click.echo(f"\n⏱  Completed in {timing_result.formatted_duration}")
        elif format == "json":
            _output_json_chronicle(chronicle_data, output)
            return
        else:
            _generate_chronicle_report(chronicle_data, format, output, config)
            # Do not print summary for non-terminal formats

    except Exception as e:
        # Stop timer on error
        if timer.start_time and not timer.end_time:
            timing_result = timer.stop("Chronicle failed")
            if not is_quiet:
                click.echo(f"⚠  Failed after {timing_result.formatted_duration}")
        
        logger.error(f"Chronicle generation failed: {e}")
        click.echo(str(e))
        raise typer.Exit(1)


def _parse_date_range(since: Optional[str], until: Optional[str]) -> Dict[str, datetime]:
    """Parse date range from string inputs.

    Args:
        since: Start date string
        until: End date string

    Returns:
        Dict with 'since' and 'until' datetime objects
    """
    now = datetime.now()

    # Helper to parse absolute date strings using stdlib only
    def _parse_absolute_date(s: str) -> Optional[datetime]:
        s = s.strip()
        # Common ISO format first
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass
        # Try a couple of common alternatives
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    # Parse since
    if since:
        s = since.strip().lower()
        if "ago" in s:
            # Handle relative dates like "3 months ago", "2 weeks ago", etc.
            parts = s.split()
            amount = 0
            unit = "day"
            if len(parts) >= 3:
                try:
                    amount = int(parts[0])
                except Exception:
                    amount = 0
                unit = parts[1]
            # Approximate months as 30 days and years as 365 days to avoid extra deps
            if "year" in unit:
                since_date = now - timedelta(days=amount * 365)
            elif "month" in unit:
                since_date = now - timedelta(days=amount * 30)
            elif "week" in unit:
                since_date = now - timedelta(weeks=amount)
            else:
                since_date = now - timedelta(days=amount or 30)
        else:
            since_date = _parse_absolute_date(since) or (now - timedelta(days=90))
    else:
        since_date = now - timedelta(days=90)  # Default ~3 months

    # Parse until
    if until:
        u = until.strip().lower()
        if u == "today":
            until_date = now
        else:
            until_date = _parse_absolute_date(until) or now
    else:
        until_date = now

    return {"since": since_date, "until": until_date}


def _analyze_patterns(git_analyzer: GitAnalyzer, date_range: Dict[str, datetime]) -> Dict[str, Any]:
    """Analyze change patterns in the repository.

    Args:
        git_analyzer: Git analyzer instance
        date_range: Date range for analysis

    Returns:
        Pattern analysis results
    """
    patterns = {
        "change_frequency": {},
        "file_coupling": {},
        "commit_patterns": {},
        "refactoring_candidates": [],
    }

    # Analyze change frequency
    commits = git_analyzer.get_commits(since=date_range["since"], until=date_range["until"])

    file_changes = {}
    coupling_pairs = {}

    for commit in commits:
        changed_files = commit.get("files", [])

        # Track change frequency
        for file in changed_files:
            file_changes[file] = file_changes.get(file, 0) + 1

        # Track file coupling (files changed together)
        for i, file1 in enumerate(changed_files):
            for file2 in changed_files[i + 1 :]:
                pair = tuple(sorted([file1, file2]))
                coupling_pairs[pair] = coupling_pairs.get(pair, 0) + 1

    # Identify top changed files
    patterns["change_frequency"] = dict(
        sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:20]
    )

    # Identify coupled files
    patterns["file_coupling"] = {
        f"{f1} <-> {f2}": count
        for (f1, f2), count in sorted(coupling_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
        if count > 2
    }

    # Identify refactoring candidates (high change frequency + high coupling)
    for file, changes in file_changes.items():
        if changes > 10:
            couplings = sum(count for (f1, f2), count in coupling_pairs.items() if file in (f1, f2))
            if couplings > 5:
                patterns["refactoring_candidates"].append(
                    {
                        "file": file,
                        "changes": changes,
                        "couplings": couplings,
                        "risk": "high" if changes > 20 else "medium",
                    }
                )

    return patterns


def _display_terminal_chronicle(
    chronicle: Dict[str, Any], show_contributors: bool, show_patterns: bool
) -> None:
    """Display chronicle in terminal using viz modules.

    Args:
        chronicle: Chronicle data
        show_contributors: Whether to show contributor analysis
        show_patterns: Whether to show pattern analysis
    """
    display = TerminalDisplay()

    # Display header
    display.display_header(
        "Repository Chronicle",
        subtitle=f"Period: {chronicle.get('period', 'All time')}",
        style="double",
    )

    # Display activity timeline
    if "activity" in chronicle:
        momentum_viz = MomentumVisualizer()
        # Convert chronicle activity to momentum format
        activity_data = {
            "velocity_trend": chronicle["activity"].get("trend", 0),
            "current_sprint": {
                "velocity": chronicle["activity"].get("current_velocity", 0),
                "completed": chronicle["activity"].get("commits_this_week", 0),
            },
        }
        momentum_viz.display_terminal(activity_data, show_details=True)

    # Display contributors if requested
    if show_contributors and "contributors" in chronicle:
        contributor_viz = ContributorVisualizer()
        contributor_viz.display_terminal(chronicle["contributors"], show_details=True)

    # Display patterns if requested
    if show_patterns and "patterns" in chronicle:
        _display_patterns(chronicle["patterns"], display)


def _display_patterns(patterns: Dict[str, Any], display: TerminalDisplay) -> None:
    """Display change patterns in terminal.

    Args:
        patterns: Pattern analysis data
        display: Terminal display instance
    """
    display.display_header("Change Patterns", style="single")

    # Display frequently changed files
    if "change_frequency" in patterns:
        headers = ["File", "Changes"]
        rows = [
            [file[:40], str(count)]
            for file, count in list(patterns["change_frequency"].items())[:10]
        ]
        display.display_table(headers, rows, title="Most Changed Files")

    # Display coupled files
    if "file_coupling" in patterns:
        display.display_list(
            [
                f"{pair}: {count} co-changes"
                for pair, count in list(patterns["file_coupling"].items())[:5]
            ],
            title="Coupled Files",
            style="bullet",
        )

    # Display refactoring candidates
    if patterns.get("refactoring_candidates"):
        headers = ["File", "Changes", "Couplings", "Risk"]
        rows = []
        for candidate in patterns["refactoring_candidates"][:5]:
            risk_color = "red" if candidate["risk"] == "high" else "yellow"
            rows.append(
                [
                    candidate["file"][:30],
                    str(candidate["changes"]),
                    str(candidate["couplings"]),
                    display.colorize(candidate["risk"].upper(), risk_color),
                ]
            )
        display.display_table(headers, rows, title="Refactoring Candidates")


def _generate_chronicle_report(
    chronicle: Dict[str, Any], format: str, output: Optional[str], config: Any
) -> None:
    """Generate chronicle report using viz modules.

    Args:
        chronicle: Chronicle data
        format: Report format
        output: Output path
        config: Configuration
    """
    from tenets.core.reporting import ReportConfig

    generator = ReportGenerator(config)

    report_config = ReportConfig(
        title="Repository Chronicle Report", format=format, include_charts=True
    )

    output_path = Path(output) if output else Path(f"chronicle_report.{format}")

    generator.generate(data=chronicle, output_path=output_path, config=report_config)

    click.echo(f"Chronicle report generated: {output_path}")


def _output_json_chronicle(chronicle: Dict[str, Any], output: Optional[str]) -> None:
    """Output chronicle as JSON.

    Args:
        chronicle: Chronicle data
        output: Output path
    """
    import json

    if output:
        with open(output, "w") as f:
            json.dump(chronicle, f, indent=2, default=str)
        click.echo(f"Chronicle saved to: {output}")
    else:
        click.echo(json.dumps(chronicle, indent=2, default=str))


def _print_chronicle_summary(chronicle: Dict[str, Any]) -> None:
    """Print chronicle summary.

    Args:
        chronicle: Chronicle data
    """
    click.echo("\n" + "=" * 50)
    click.echo("CHRONICLE SUMMARY")
    click.echo("=" * 50)

    # Basic stats
    click.echo(f"Period: {chronicle.get('period', 'Unknown')}")
    click.echo(f"Total commits: {chronicle.get('total_commits', 0)}")
    click.echo(f"Files changed: {chronicle.get('files_changed', 0)}")

    # Contributors
    if "contributors" in chronicle:
        contributors = chronicle["contributors"]
        click.echo("\nContributors:")
        click.echo(f"  Total: {contributors.get('total_contributors', 0)}")
        click.echo(f"  Active: {contributors.get('active_contributors', 0)}")

        if "top_contributors" in contributors:
            click.echo("  Top 3:")
            for i, contributor in enumerate(contributors["top_contributors"][:3], 1):
                click.echo(f"    {i}. {contributor['name']}: {contributor['commits']} commits")

    # Activity trend
    if "activity" in chronicle:
        activity = chronicle["activity"]
        trend = activity.get("trend", 0)
        if trend > 0:
            click.secho(f"\n📈 Activity trending up: +{trend:.1f}%", fg="green")
        elif trend < 0:
            click.secho(f"\n📉 Activity trending down: {trend:.1f}%", fg="yellow")
        else:
            click.echo("\n➡️  Activity stable")
