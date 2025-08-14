"""Examine command implementation.

This command provides comprehensive code examination including complexity
analysis, metrics calculation, and quality assessment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from tenets.core.examiner import (
    CodeExaminer,
    HotspotDetector,
    OwnershipAnalyzer,
)
from tenets.core.reporting import ReportGenerator
from tenets.utils.logger import get_logger
from tenets.viz import ComplexityVisualizer, HotspotVisualizer, TerminalDisplay


@click.command()
@click.argument("path", type=click.Path(exists=True), default=".")
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
    help="Specific metrics to calculate (complexity, duplication, etc.)",
)
@click.option(
    "--threshold", "-t", type=int, default=10, help="Complexity threshold for flagging issues"
)
@click.option("--include", "-i", multiple=True, help="File patterns to include")
@click.option("--exclude", "-e", multiple=True, help="File patterns to exclude")
@click.option("--max-depth", type=int, default=5, help="Maximum directory depth")
@click.option("--show-details", is_flag=True, help="Show detailed breakdown")
@click.option("--hotspots", is_flag=True, help="Include hotspot analysis")
@click.option("--ownership", is_flag=True, help="Include ownership analysis")
@click.pass_context
def examine(
    ctx,
    path: str,
    output: Optional[str],
    format: str,
    metrics: List[str],
    threshold: int,
    include: List[str],
    exclude: List[str],
    max_depth: int,
    show_details: bool,
    hotspots: bool,
    ownership: bool,
):
    """Examine code quality and complexity.

    Performs comprehensive code analysis including complexity metrics,
    code quality assessment, and optional hotspot detection.

    Examples:
        tenets examine
        tenets examine src/ --format=html --output=report.html
        tenets examine --metrics=complexity --threshold=15
        tenets examine --hotspots --show-details
    """
    logger = get_logger(__name__)
    config = ctx.obj["config"]

    # Initialize path
    target_path = Path(path).resolve()
    logger.info(f"Examining code at: {target_path}")

    # Initialize examiner
    examiner = CodeExaminer(config)

    # Configure examination options
    exam_options = {
        "threshold": threshold,
        "max_depth": max_depth,
        "include_patterns": list(include) if include else None,
        "exclude_patterns": list(exclude) if exclude else None,
        "calculate_metrics": list(metrics) if metrics else ["all"],
        "include_hotspots": hotspots,
        "include_ownership": ownership,
    }

    try:
        # Perform examination
        logger.info("Starting code examination...")
        examination_results = examiner.examine(target_path, **exam_options)

        # Add specialized analysis if requested
        if hotspots:
            logger.info("Performing hotspot analysis...")
            hotspot_detector = HotspotDetector(config)
            examination_results["hotspots"] = hotspot_detector.detect_hotspots(
                target_path, threshold=threshold
            )

        if ownership:
            logger.info("Analyzing code ownership...")
            ownership_analyzer = OwnershipAnalyzer(config)
            examination_results["ownership"] = ownership_analyzer.analyze_ownership(target_path)

        # Display or save results based on format
        if format == "terminal":
            _display_terminal_results(examination_results, show_details)
        elif format == "json":
            _output_json_results(examination_results, output)
        else:
            # Generate report using viz modules
            _generate_report(examination_results, format, output, config)

        # Summary
        _print_summary(examination_results)

    except Exception as e:
        logger.error(f"Examination failed: {e}")
        raise click.ClickException(str(e))


def _display_terminal_results(results: Dict[str, Any], show_details: bool) -> None:
    """Display results in terminal using viz modules.

    Args:
        results: Examination results
        show_details: Whether to show detailed breakdown
    """
    display = TerminalDisplay()

    # Display header
    display.display_header(
        "Code Examination Results",
        subtitle=f"Files analyzed: {results.get('total_files', 0)}",
        style="double",
    )

    # Display complexity analysis if available
    if "complexity" in results:
        complexity_viz = ComplexityVisualizer()
        complexity_viz.display_terminal(results["complexity"], show_details)

    # Display hotspots if available
    if "hotspots" in results:
        hotspot_viz = HotspotVisualizer()
        hotspot_viz.display_terminal(results["hotspots"], show_details)

    # Display ownership if available
    if "ownership" in results:
        _display_ownership_results(results["ownership"], display, show_details)

    # Display overall metrics
    if "metrics" in results:
        display.display_metrics(results["metrics"], title="Overall Metrics", columns=2)


def _display_ownership_results(
    ownership: Dict[str, Any], display: TerminalDisplay, show_details: bool
) -> None:
    """Display ownership results in terminal.

    Args:
        ownership: Ownership data
        display: Terminal display instance
        show_details: Whether to show details
    """
    display.display_header("Code Ownership", style="single")

    if "by_contributor" in ownership and show_details:
        headers = ["Contributor", "Files", "Lines", "Percentage"]
        rows = []

        total_lines = ownership.get("total_lines", 1)
        for contributor in ownership["by_contributor"][:10]:
            percentage = (contributor["lines"] / total_lines) * 100
            rows.append(
                [
                    contributor["name"][:30],
                    str(contributor["files"]),
                    str(contributor["lines"]),
                    f"{percentage:.1f}%",
                ]
            )

        display.display_table(headers, rows, title="Top Contributors")

    # Display bus factor warning if low
    bus_factor = ownership.get("bus_factor", 0)
    if bus_factor <= 2:
        display.display_warning(f"Low bus factor ({bus_factor}) - knowledge concentration risk!")


def _generate_report(
    results: Dict[str, Any], format: str, output: Optional[str], config: Any
) -> None:
    """Generate report using viz modules and reporting.

    Args:
        results: Examination results
        format: Report format
        output: Output path
        config: Configuration
    """
    from tenets.core.reporting import ReportConfig

    # Initialize report generator
    generator = ReportGenerator(config)

    # Create report configuration
    report_config = ReportConfig(
        title="Code Examination Report",
        format=format,
        include_charts=True,
        include_code_snippets=True,
    )

    # Generate report using viz modules for charts
    output_path = Path(output) if output else Path(f"examination_report.{format}")

    # The generator will internally use viz modules
    generator.generate(data=results, output_path=output_path, config=report_config)

    click.echo(f"Report generated: {output_path}")


def _output_json_results(results: Dict[str, Any], output: Optional[str]) -> None:
    """Output results as JSON.

    Args:
        results: Examination results
        output: Output path
    """
    import json

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"Results saved to: {output}")
    else:
        click.echo(json.dumps(results, indent=2, default=str))


def _print_summary(results: Dict[str, Any]) -> None:
    """Print examination summary.

    Args:
        results: Examination results
    """
    click.echo("\n" + "=" * 50)
    click.echo("EXAMINATION SUMMARY")
    click.echo("=" * 50)

    # Files analyzed
    click.echo(f"Files analyzed: {results.get('total_files', 0)}")
    click.echo(f"Total lines: {results.get('total_lines', 0):,}")

    # Complexity summary
    if "complexity" in results:
        complexity = results["complexity"]
        click.echo("\nComplexity:")
        click.echo(f"  Average: {complexity.get('avg_complexity', 0):.2f}")
        click.echo(f"  Maximum: {complexity.get('max_complexity', 0)}")
        click.echo(f"  Complex functions: {complexity.get('complex_functions', 0)}")

    # Hotspot summary
    if "hotspots" in results:
        hotspots = results["hotspots"]
        click.echo("\nHotspots:")
        click.echo(f"  Total: {hotspots.get('total_hotspots', 0)}")
        click.echo(f"  Critical: {hotspots.get('critical_count', 0)}")

    # Health score
    if "health_score" in results:
        score = results["health_score"]
        if score >= 80:
            color = "green"
            status = "Excellent"
        elif score >= 60:
            color = "yellow"
            status = "Good"
        elif score >= 40:
            color = "yellow"
            status = "Fair"
        else:
            color = "red"
            status = "Needs Improvement"

        click.echo("\nHealth Score: ", nl=False)
        click.secho(f"{score:.1f}/100 ({status})", fg=color)
