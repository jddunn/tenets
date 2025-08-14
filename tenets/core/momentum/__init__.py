"""Development momentum and velocity tracking package.

This package provides comprehensive velocity tracking and momentum analysis
for software development teams. It analyzes git history to understand
development patterns, team productivity, and project velocity trends.

The momentum tracker helps teams understand their development pace, identify
bottlenecks, and make data-driven decisions about resource allocation and
sprint planning.

Main components:
- VelocityTracker: Main tracker for development velocity
- MomentumMetrics: Metrics calculation for momentum
- SprintAnalyzer: Sprint-based velocity analysis
- TeamVelocity: Team-level velocity tracking
- ProductivityAnalyzer: Individual and team productivity analysis

Example usage:
    >>> from tenets.core.momentum import VelocityTracker
    >>> from tenets.config import TenetsConfig
    >>>
    >>> config = TenetsConfig()
    >>> tracker = VelocityTracker(config)
    >>>
    >>> # Track momentum for the last month
    >>> report = tracker.track_momentum(
    ...     repo_path=Path("."),
    ...     period="last-month",
    ...     team=True
    ... )
    >>>
    >>> print(f"Team velocity: {report.team_velocity}")
    >>> print(f"Sprint completion: {report.sprint_completion}%")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import (
    MomentumMetrics,
    ProductivityMetrics,
    SprintMetrics,
    TeamMetrics,
    VelocityTrend,
    calculate_momentum_metrics,
)
from .tracker import (
    MomentumReport,
    VelocityTracker,
    track_individual_velocity,
    track_momentum,
    track_team_velocity,
)

# Version
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Main tracker
    "VelocityTracker",
    "MomentumReport",
    "track_momentum",
    "track_team_velocity",
    "track_individual_velocity",
    # Metrics
    "MomentumMetrics",
    "ProductivityMetrics",
    "SprintMetrics",
    "TeamMetrics",
    "VelocityTrend",
    "calculate_momentum_metrics",
    # Convenience functions
    "analyze_sprint_velocity",
    "analyze_team_productivity",
    "predict_completion",
]


def analyze_sprint_velocity(
    repo_path: Path,
    sprint_duration: int = 14,
    lookback_sprints: int = 6,
    config: Optional[Any] = None,
) -> SprintMetrics:
    """Analyze velocity across recent sprints.

    Calculates sprint-based velocity metrics to understand team
    performance and predictability over time.

    Args:
        repo_path: Path to git repository
        sprint_duration: Sprint length in days
        lookback_sprints: Number of sprints to analyze
        config: Optional TenetsConfig instance

    Returns:
        SprintMetrics: Sprint velocity analysis

    Example:
        >>> from tenets.core.momentum import analyze_sprint_velocity
        >>>
        >>> metrics = analyze_sprint_velocity(
        ...     Path("."),
        ...     sprint_duration=14,
        ...     lookback_sprints=6
        ... )
        >>> print(f"Average velocity: {metrics.avg_velocity}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)
    report = tracker.track_momentum(
        repo_path=repo_path, period=f"{sprint_duration * lookback_sprints} days"
    )

    return report.sprint_metrics


def analyze_team_productivity(
    repo_path: Path,
    period: str = "last-month",
    team_mapping: Optional[Dict[str, List[str]]] = None,
    config: Optional[Any] = None,
) -> TeamMetrics:
    """Analyze team productivity metrics.

    Provides detailed analysis of team productivity including
    individual contributions, collaboration patterns, and efficiency.

    Args:
        repo_path: Path to git repository
        period: Time period to analyze
        team_mapping: Optional mapping of team names to members
        config: Optional TenetsConfig instance

    Returns:
        TeamMetrics: Team productivity analysis

    Example:
        >>> from tenets.core.momentum import analyze_team_productivity
        >>>
        >>> team_metrics = analyze_team_productivity(
        ...     Path("."),
        ...     period="last-quarter",
        ...     team_mapping={
        ...         "backend": ["alice@example.com", "bob@example.com"],
        ...         "frontend": ["charlie@example.com", "diana@example.com"]
        ...     }
        ... )
        >>> print(f"Team efficiency: {team_metrics.efficiency_score}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)
    report = tracker.track_momentum(
        repo_path=repo_path, period=period, team=True, team_mapping=team_mapping
    )

    return report.team_metrics


def predict_completion(
    repo_path: Path,
    remaining_work: int,
    team_size: Optional[int] = None,
    confidence_level: float = 0.8,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Predict project completion based on velocity.

    Uses historical velocity data to predict when a certain amount
    of work will be completed.

    Args:
        repo_path: Path to git repository
        remaining_work: Estimated remaining work (in points/tasks)
        team_size: Current team size (uses historical if not provided)
        confidence_level: Confidence level for prediction (0-1)
        config: Optional TenetsConfig instance

    Returns:
        Dict[str, Any]: Completion prediction including date and confidence

    Example:
        >>> from tenets.core.momentum import predict_completion
        >>>
        >>> prediction = predict_completion(
        ...     Path("."),
        ...     remaining_work=100,
        ...     team_size=5,
        ...     confidence_level=0.8
        ... )
        >>> print(f"Expected completion: {prediction['expected_date']}")
        >>> print(f"Confidence: {prediction['confidence']}%")
    """
    from datetime import datetime, timedelta

    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)

    # Get historical velocity
    report = tracker.track_momentum(repo_path=repo_path, period="last-quarter")

    if not report.velocity_trend or not report.velocity_trend.avg_velocity:
        return {
            "expected_date": None,
            "confidence": 0,
            "message": "Insufficient historical data for prediction",
        }

    # Calculate completion time
    avg_velocity = report.velocity_trend.avg_velocity

    # Adjust for team size if provided
    if team_size and report.team_metrics and report.team_metrics.active_members > 0:
        velocity_per_person = avg_velocity / report.team_metrics.active_members
        adjusted_velocity = velocity_per_person * team_size
    else:
        adjusted_velocity = avg_velocity

    # Calculate days to completion
    if adjusted_velocity > 0:
        days_to_complete = remaining_work / adjusted_velocity

        # Apply confidence adjustment (add buffer for lower confidence)
        buffer_factor = 1 + (1 - confidence_level)
        adjusted_days = days_to_complete * buffer_factor

        expected_date = datetime.now() + timedelta(days=adjusted_days)

        # Calculate actual confidence based on velocity stability
        if report.velocity_trend.stability_score:
            actual_confidence = min(confidence_level, report.velocity_trend.stability_score / 100)
        else:
            actual_confidence = confidence_level * 0.7  # Lower confidence without stability data

        return {
            "expected_date": expected_date.strftime("%Y-%m-%d"),
            "days_remaining": int(adjusted_days),
            "confidence": round(actual_confidence * 100, 1),
            "velocity_used": adjusted_velocity,
            "buffer_days": int(adjusted_days - days_to_complete),
            "message": "Prediction based on historical velocity",
        }
    else:
        return {"expected_date": None, "confidence": 0, "message": "No velocity detected"}


def calculate_burndown(
    repo_path: Path,
    total_work: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Calculate burndown chart data.

    Generates data for burndown visualization showing work
    completion over time.

    Args:
        repo_path: Path to git repository
        total_work: Total work to complete
        start_date: Sprint start date (ISO format)
        end_date: Sprint end date (ISO format)
        config: Optional TenetsConfig instance

    Returns:
        Dict[str, Any]: Burndown data including ideal and actual lines

    Example:
        >>> from tenets.core.momentum import calculate_burndown
        >>>
        >>> burndown = calculate_burndown(
        ...     Path("."),
        ...     total_work=100,
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-14"
        ... )
        >>> print(f"Completion: {burndown['completion_percentage']}%")
    """
    from datetime import datetime, timedelta

    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)

    # Parse dates
    if start_date:
        start = datetime.fromisoformat(start_date)
    else:
        start = datetime.now() - timedelta(days=14)  # Default 2-week sprint

    if end_date:
        end = datetime.fromisoformat(end_date)
    else:
        end = datetime.now()

    # Get work completed per day
    report = tracker.track_momentum(
        repo_path=repo_path, period=f"{(end - start).days} days", daily_breakdown=True
    )

    # Build burndown data
    days = (end - start).days
    ideal_line = []
    actual_line = []
    remaining_work = total_work

    for day in range(days + 1):
        # Ideal line (linear burndown)
        ideal_remaining = total_work * (1 - day / days) if days > 0 else total_work
        ideal_line.append(
            {
                "day": day,
                "date": (start + timedelta(days=day)).strftime("%Y-%m-%d"),
                "remaining": ideal_remaining,
            }
        )

        # Actual line (based on velocity data)
        if report.daily_velocity and day < len(report.daily_velocity):
            work_done = report.daily_velocity[day]
            remaining_work -= work_done
            actual_line.append(
                {
                    "day": day,
                    "date": (start + timedelta(days=day)).strftime("%Y-%m-%d"),
                    "remaining": max(0, remaining_work),
                }
            )

    completion_percentage = (
        (total_work - remaining_work) / total_work * 100 if total_work > 0 else 0
    )

    return {
        "ideal_line": ideal_line,
        "actual_line": actual_line,
        "total_work": total_work,
        "remaining_work": max(0, remaining_work),
        "completion_percentage": round(completion_percentage, 1),
        "days_elapsed": days,
        "on_track": remaining_work <= ideal_line[-1]["remaining"] if ideal_line else False,
    }


def get_velocity_chart_data(
    repo_path: Path,
    period: str = "last-quarter",
    interval: str = "weekly",
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Get data for velocity chart visualization.

    Prepares velocity data in a format suitable for charting,
    with configurable time intervals.

    Args:
        repo_path: Path to git repository
        period: Time period to analyze
        interval: Data interval (daily, weekly, monthly)
        config: Optional TenetsConfig instance

    Returns:
        Dict[str, Any]: Chart-ready velocity data

    Example:
        >>> from tenets.core.momentum import get_velocity_chart_data
        >>>
        >>> chart_data = get_velocity_chart_data(
        ...     Path("."),
        ...     period="last-quarter",
        ...     interval="weekly"
        ... )
        >>> # Use chart_data for visualization
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)

    report = tracker.track_momentum(repo_path=repo_path, period=period, interval=interval)

    # Format for charting
    chart_data = {"labels": [], "velocity": [], "commits": [], "contributors": [], "trend_line": []}

    if report.velocity_trend and report.velocity_trend.data_points:
        for point in report.velocity_trend.data_points:
            chart_data["labels"].append(point["date"])
            chart_data["velocity"].append(point["velocity"])
            chart_data["commits"].append(point.get("commits", 0))
            chart_data["contributors"].append(point.get("contributors", 0))

        # Add trend line (simple linear regression)
        if len(chart_data["velocity"]) > 1:
            # Simple trend calculation
            n = len(chart_data["velocity"])
            x_mean = n / 2
            y_mean = sum(chart_data["velocity"]) / n

            numerator = sum(
                (i - x_mean) * (v - y_mean) for i, v in enumerate(chart_data["velocity"])
            )
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator != 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean

                chart_data["trend_line"] = [intercept + slope * i for i in range(n)]

    chart_data["summary"] = {
        "avg_velocity": report.velocity_trend.avg_velocity if report.velocity_trend else 0,
        "max_velocity": report.velocity_trend.max_velocity if report.velocity_trend else 0,
        "min_velocity": report.velocity_trend.min_velocity if report.velocity_trend else 0,
        "trend": report.velocity_trend.trend_direction if report.velocity_trend else "stable",
    }

    return chart_data
