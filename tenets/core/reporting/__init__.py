"""Reporting package for generating analysis reports.

This package provides comprehensive reporting functionality for all analysis
results. It supports multiple output formats including HTML, Markdown, JSON,
and PDF, with rich visualizations and interactive dashboards.

The reporting system creates professional, actionable reports that help teams
understand code quality, track progress, and make data-driven decisions.

Main components:
- ReportGenerator: Main report generation orchestrator
- HTMLReporter: HTML report generation with interactive charts
- MarkdownReporter: Markdown report generation
- JSONReporter: JSON data export
- PDFReporter: PDF report generation
- Dashboard: Interactive dashboard generation
- Visualizer: Chart and graph generation

Example usage:
    >>> from tenets.core.reporting import ReportGenerator
    >>> from tenets.config import TenetsConfig
    >>>
    >>> config = TenetsConfig()
    >>> generator = ReportGenerator(config)
    >>>
    >>> # Generate comprehensive report
    >>> generator.generate(
    ...     analysis_results,
    ...     output_path="report.html",
    ...     format="html",
    ...     include_charts=True
    ... )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .generator import (
    ReportConfig,
    ReportGenerator,
    ReportSection,
    generate_report,
    generate_summary,
)
from .html_reporter import (
    HTMLReporter,
    HTMLTemplate,
    create_dashboard,
    create_html_report,
)
from .markdown_reporter import (
    MarkdownReporter,
    create_markdown_report,
    format_markdown_table,
)
from .visualizer import (
    ChartGenerator,
    create_chart,
    create_heatmap,
    create_network_graph,
    create_timeline,
)

# Version
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Main generator
    "ReportGenerator",
    "ReportConfig",
    "ReportSection",
    "generate_report",
    "generate_summary",
    # Reporters
    "HTMLReporter",
    "HTMLTemplate",
    "create_html_report",
    "create_dashboard",
    "MarkdownReporter",
    "create_markdown_report",
    "format_markdown_table",
    # Visualization
    "ChartGenerator",
    "create_chart",
    "create_heatmap",
    "create_timeline",
    "create_network_graph",
    # Convenience functions
    "quick_report",
    "export_data",
    "create_executive_summary",
]


def quick_report(
    analysis_results: Dict[str, Any],
    output_path: Optional[Path] = None,
    format: str = "html",
    title: str = "Code Analysis Report",
    config: Optional[Any] = None,
) -> Path:
    """Generate a quick report from analysis results.

    Creates a comprehensive report with sensible defaults for quick
    reporting needs.

    Args:
        analysis_results: Analysis results to report
        output_path: Output file path (auto-generated if None)
        format: Report format (html, markdown, json)
        title: Report title
        config: Optional TenetsConfig instance

    Returns:
        Path: Path to generated report

    Example:
        >>> from tenets.core.reporting import quick_report
        >>>
        >>> report_path = quick_report(
        ...     analysis_results,
        ...     format="html",
        ...     title="Sprint 23 Analysis"
        ... )
        >>> print(f"Report generated: {report_path}")
    """
    from datetime import datetime

    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    # Auto-generate output path if needed
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = "html" if format == "html" else format
        output_path = Path(f"tenets_report_{timestamp}.{extension}")

    # Create report config
    report_config = ReportConfig(
        title=title,
        format=format,
        include_charts=True,
        include_code_snippets=True,
        include_recommendations=True,
        theme="default",
    )

    # Generate report
    generator = ReportGenerator(config)
    return generator.generate(analysis_results, output_path, report_config)


def export_data(
    analysis_results: Dict[str, Any],
    output_path: Path,
    format: str = "json",
    include_metadata: bool = True,
    config: Optional[Any] = None,
) -> Path:
    """Export analysis data in specified format.

    Exports raw analysis data for further processing or integration
    with other tools.

    Args:
        analysis_results: Analysis results to export
        output_path: Output file path
        format: Export format (json, csv, xlsx)
        include_metadata: Include analysis metadata
        config: Optional TenetsConfig instance

    Returns:
        Path: Path to exported data

    Example:
        >>> from tenets.core.reporting import export_data
        >>>
        >>> export_path = export_data(
        ...     analysis_results,
        ...     Path("data.json"),
        ...     format="json"
        ... )
    """
    import json
    from datetime import datetime

    if format == "json":
        # Prepare data with metadata
        export_data = {
            "analysis_results": analysis_results,
            "metadata": (
                {
                    "exported_at": datetime.now().isoformat(),
                    "version": __version__,
                    "format": format,
                }
                if include_metadata
                else {}
            ),
        }

        # Write JSON
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    elif format == "csv":
        # Export as CSV (simplified)
        import csv

        # Flatten results for CSV
        rows = []
        for category, items in analysis_results.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        row = {"category": category}
                        row.update(item)
                        rows.append(row)

        # Write CSV
        if rows:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    return output_path


def create_executive_summary(
    analysis_results: Dict[str, Any],
    max_length: int = 500,
    include_metrics: bool = True,
    include_risks: bool = True,
    include_recommendations: bool = True,
) -> str:
    """Create an executive summary of analysis results.

    Generates a concise, high-level summary suitable for executives
    and stakeholders.

    Args:
        analysis_results: Analysis results to summarize
        max_length: Maximum summary length in words
        include_metrics: Include key metrics
        include_risks: Include top risks
        include_recommendations: Include top recommendations

    Returns:
        str: Executive summary text

    Example:
        >>> from tenets.core.reporting import create_executive_summary
        >>>
        >>> summary = create_executive_summary(
        ...     analysis_results,
        ...     max_length=300
        ... )
        >>> print(summary)
    """
    summary_parts = []

    # Opening statement
    if "overview" in analysis_results:
        overview = analysis_results["overview"]
        summary_parts.append(
            f"Code analysis reveals a codebase with "
            f"{overview.get('total_files', 0)} files, "
            f"{overview.get('total_lines', 0)} lines of code, "
            f"and a health score of {overview.get('health_score', 0):.1f}/100."
        )

    # Key metrics
    if include_metrics and "metrics" in analysis_results:
        metrics = analysis_results["metrics"]
        summary_parts.append(
            f"Key metrics: "
            f"Complexity: {metrics.get('avg_complexity', 0):.1f}, "
            f"Duplication: {metrics.get('duplication_ratio', 0):.1%}, "
            f"Test Coverage: {metrics.get('test_coverage', 0):.1%}."
        )

    # Top risks
    if include_risks and "risks" in analysis_results:
        risks = analysis_results["risks"]
        if risks:
            top_risks = risks[:3]  # Top 3 risks
            risk_text = "Critical risks: " + "; ".join(r.get("description", "") for r in top_risks)
            summary_parts.append(risk_text)

    # Top recommendations
    if include_recommendations and "recommendations" in analysis_results:
        recommendations = analysis_results["recommendations"]
        if recommendations:
            top_recs = recommendations[:2]  # Top 2 recommendations
            rec_text = "Priority actions: " + "; ".join(
                r.get("action", "") if isinstance(r, dict) else str(r) for r in top_recs
            )
            summary_parts.append(rec_text)

    # Conclusion
    health_score = analysis_results.get("overview", {}).get("health_score", 50)
    if health_score >= 80:
        summary_parts.append(
            "Overall, the codebase is in excellent condition with minor improvements needed."
        )
    elif health_score >= 60:
        summary_parts.append(
            "The codebase is in good condition but requires attention to identified issues."
        )
    elif health_score >= 40:
        summary_parts.append("The codebase needs significant improvement in multiple areas.")
    else:
        summary_parts.append("The codebase requires urgent attention to address critical issues.")

    # Combine and limit length
    summary = " ".join(summary_parts)

    # Truncate if needed (simple word count limit)
    words = summary.split()
    if len(words) > max_length:
        summary = " ".join(words[:max_length]) + "..."

    return summary


def create_comparison_report(
    baseline_results: Dict[str, Any],
    current_results: Dict[str, Any],
    output_path: Path,
    format: str = "html",
    title: str = "Comparison Report",
    config: Optional[Any] = None,
) -> Path:
    """Create a comparison report between two analysis results.

    Generates a report highlighting differences and trends between
    baseline and current analysis results.

    Args:
        baseline_results: Baseline analysis results
        current_results: Current analysis results
        output_path: Output file path
        format: Report format
        title: Report title
        config: Optional configuration

    Returns:
        Path: Path to comparison report

    Example:
        >>> from tenets.core.reporting import create_comparison_report
        >>>
        >>> report_path = create_comparison_report(
        ...     baseline_results,
        ...     current_results,
        ...     Path("comparison.html"),
        ...     title="Sprint 22 vs Sprint 23"
        ... )
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    # Calculate differences
    comparison_data = {
        "baseline": baseline_results,
        "current": current_results,
        "changes": _calculate_changes(baseline_results, current_results),
        "improvements": _identify_improvements(baseline_results, current_results),
        "regressions": _identify_regressions(baseline_results, current_results),
    }

    # Generate comparison report
    generator = ReportGenerator(config)
    report_config = ReportConfig(
        title=title, format=format, template="comparison", include_charts=True
    )

    return generator.generate(comparison_data, output_path, report_config)


def create_trend_report(
    historical_results: List[Dict[str, Any]],
    output_path: Path,
    format: str = "html",
    title: str = "Trend Analysis Report",
    config: Optional[Any] = None,
) -> Path:
    """Create a trend analysis report from historical data.

    Generates a report showing trends and patterns over time based
    on multiple analysis snapshots.

    Args:
        historical_results: List of historical analysis results
        output_path: Output file path
        format: Report format
        title: Report title
        config: Optional configuration

    Returns:
        Path: Path to trend report

    Example:
        >>> from tenets.core.reporting import create_trend_report
        >>>
        >>> report_path = create_trend_report(
        ...     [sprint20_results, sprint21_results, sprint22_results],
        ...     Path("trends.html"),
        ...     title="Quarterly Trend Analysis"
        ... )
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    # Calculate trends
    trend_data = {
        "snapshots": historical_results,
        "trends": _calculate_trends(historical_results),
        "predictions": _predict_trends(historical_results),
        "patterns": _identify_patterns(historical_results),
    }

    # Generate trend report
    generator = ReportGenerator(config)
    report_config = ReportConfig(
        title=title, format=format, template="trends", include_charts=True, chart_type="line"
    )

    return generator.generate(trend_data, output_path, report_config)


def _calculate_changes(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate changes between baseline and current results.

    Args:
        baseline: Baseline results
        current: Current results

    Returns:
        Dict[str, Any]: Calculated changes
    """
    changes = {}

    # Compare numeric metrics
    for key in ["health_score", "complexity", "duplication", "coverage"]:
        if key in baseline and key in current:
            baseline_val = baseline[key]
            current_val = current[key]
            if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                changes[key] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "change": current_val - baseline_val,
                    "change_percent": (
                        ((current_val - baseline_val) / baseline_val * 100)
                        if baseline_val != 0
                        else 0
                    ),
                }

    return changes


def _identify_improvements(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Identify improvements between baseline and current.

    Args:
        baseline: Baseline results
        current: Current results

    Returns:
        List[str]: List of improvements
    """
    improvements = []

    # Check for improved metrics
    changes = _calculate_changes(baseline, current)

    for metric, change_data in changes.items():
        if change_data["change"] > 0:
            if metric in ["health_score", "coverage"]:
                improvements.append(
                    f"{metric.replace('_', ' ').title()} improved by {change_data['change']:.1f}"
                )
        elif change_data["change"] < 0:
            if metric in ["complexity", "duplication"]:
                improvements.append(
                    f"{metric.replace('_', ' ').title()} reduced by {abs(change_data['change']):.1f}"
                )

    return improvements


def _identify_regressions(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Identify regressions between baseline and current.

    Args:
        baseline: Baseline results
        current: Current results

    Returns:
        List[str]: List of regressions
    """
    regressions = []

    # Check for degraded metrics
    changes = _calculate_changes(baseline, current)

    for metric, change_data in changes.items():
        if change_data["change"] < 0:
            if metric in ["health_score", "coverage"]:
                regressions.append(
                    f"{metric.replace('_', ' ').title()} decreased by {abs(change_data['change']):.1f}"
                )
        elif change_data["change"] > 0:
            if metric in ["complexity", "duplication"]:
                regressions.append(
                    f"{metric.replace('_', ' ').title()} increased by {change_data['change']:.1f}"
                )

    return regressions


def _calculate_trends(historical_results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Calculate trends from historical data.

    Args:
        historical_results: List of historical results

    Returns:
        Dict[str, List[float]]: Trend data by metric
    """
    trends = {}

    # Extract time series for each metric
    metrics = ["health_score", "complexity", "duplication", "coverage"]

    for metric in metrics:
        values = []
        for result in historical_results:
            if metric in result:
                values.append(result[metric])

        if values:
            trends[metric] = values

    return trends


def _predict_trends(historical_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Predict future trends based on historical data.

    Args:
        historical_results: List of historical results

    Returns:
        Dict[str, float]: Predicted values
    """
    predictions = {}

    trends = _calculate_trends(historical_results)

    for metric, values in trends.items():
        if len(values) >= 3:
            # Simple linear extrapolation
            recent_change = values[-1] - values[-2] if len(values) > 1 else 0
            predictions[metric] = values[-1] + recent_change

    return predictions


def _identify_patterns(historical_results: List[Dict[str, Any]]) -> List[str]:
    """Identify patterns in historical data.

    Args:
        historical_results: List of historical results

    Returns:
        List[str]: Identified patterns
    """
    patterns = []

    trends = _calculate_trends(historical_results)

    for metric, values in trends.items():
        if len(values) >= 3:
            # Check for consistent improvement
            if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
                if metric in ["health_score", "coverage"]:
                    patterns.append(f"Consistent improvement in {metric}")

            # Check for consistent degradation
            elif all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
                if metric in ["health_score", "coverage"]:
                    patterns.append(f"Consistent degradation in {metric}")

            # Check for volatility
            elif len(values) >= 4:
                changes = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
                avg_change = sum(changes) / len(changes)
                if avg_change > abs(values[-1] * 0.1):  # >10% average change
                    patterns.append(f"High volatility in {metric}")

    return patterns
