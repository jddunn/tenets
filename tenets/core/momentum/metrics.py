"""Metrics calculation module for momentum tracking.

This module provides various metrics classes and calculation functions
for development momentum analysis. It includes sprint metrics, team metrics,
productivity metrics, and velocity trend analysis.

The metrics in this module help quantify development pace, team efficiency,
and project health through data-driven measurements.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tenets.utils.logger import get_logger


@dataclass
class VelocityTrend:
    """Velocity trend analysis over time.

    Tracks how development velocity changes over time, identifying patterns,
    trends, and stability in the development process.

    Attributes:
        trend_direction: Direction of trend (increasing, decreasing, stable)
        avg_velocity: Average velocity over period
        max_velocity: Maximum velocity observed
        min_velocity: Minimum velocity observed
        std_deviation: Standard deviation of velocity
        stability_score: Stability score (0-100, higher is more stable)
        acceleration: Rate of change in velocity
        data_points: List of velocity data points for visualization
        forecast: Predicted future velocity
        confidence_level: Confidence in forecast (0-1)
        seasonal_pattern: Detected seasonal patterns
        anomalies: Detected anomalies in velocity
    """

    trend_direction: str = "stable"
    avg_velocity: float = 0.0
    max_velocity: float = 0.0
    min_velocity: float = 0.0
    std_deviation: float = 0.0
    stability_score: float = 0.0
    acceleration: float = 0.0
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    forecast: Optional[float] = None
    confidence_level: float = 0.0
    seasonal_pattern: Optional[str] = None
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "trend_direction": self.trend_direction,
            "avg_velocity": round(self.avg_velocity, 2),
            "max_velocity": round(self.max_velocity, 2),
            "min_velocity": round(self.min_velocity, 2),
            "std_deviation": round(self.std_deviation, 2),
            "stability_score": round(self.stability_score, 1),
            "acceleration": round(self.acceleration, 3),
            "data_points": self.data_points[:50],  # Limit for serialization
            "forecast": round(self.forecast, 2) if self.forecast else None,
            "confidence_level": round(self.confidence_level, 2),
            "seasonal_pattern": self.seasonal_pattern,
            "anomaly_count": len(self.anomalies),
        }

    @property
    def is_stable(self) -> bool:
        """Check if velocity is stable.

        Returns:
            bool: True if velocity is stable
        """
        return self.stability_score >= 70

    @property
    def is_improving(self) -> bool:
        """Check if velocity is improving.

        Returns:
            bool: True if velocity is increasing
        """
        return self.trend_direction == "increasing" and self.acceleration > 0

    @property
    def volatility(self) -> float:
        """Calculate velocity volatility.

        Coefficient of variation as a measure of volatility.

        Returns:
            float: Volatility score (0-1)
        """
        if self.avg_velocity == 0:
            return 0.0
        return min(1.0, self.std_deviation / self.avg_velocity)


@dataclass
class SprintMetrics:
    """Sprint-based velocity and performance metrics.

    Provides sprint-level analysis for teams using agile methodologies,
    tracking velocity, completion rates, and sprint health.

    Attributes:
        total_sprints: Total number of sprints analyzed
        avg_velocity: Average sprint velocity
        max_velocity: Maximum sprint velocity
        min_velocity: Minimum sprint velocity
        velocity_trend: Trend in sprint velocity
        sprint_data: Detailed data for each sprint
        completion_rate: Average sprint completion rate
        predictability: Sprint predictability score
        burndown_efficiency: Burndown chart efficiency
        scope_change_rate: Rate of scope changes mid-sprint
        carry_over_rate: Rate of work carried to next sprint
        sprint_health: Overall sprint health assessment
    """

    total_sprints: int = 0
    avg_velocity: float = 0.0
    max_velocity: float = 0.0
    min_velocity: float = 0.0
    velocity_trend: str = "stable"
    sprint_data: List[Dict[str, Any]] = field(default_factory=list)
    completion_rate: float = 0.0
    predictability: float = 0.0
    burndown_efficiency: float = 0.0
    scope_change_rate: float = 0.0
    carry_over_rate: float = 0.0
    sprint_health: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "total_sprints": self.total_sprints,
            "avg_velocity": round(self.avg_velocity, 2),
            "max_velocity": round(self.max_velocity, 2),
            "min_velocity": round(self.min_velocity, 2),
            "velocity_trend": self.velocity_trend,
            "completion_rate": round(self.completion_rate, 1),
            "predictability": round(self.predictability, 1),
            "burndown_efficiency": round(self.burndown_efficiency, 1),
            "scope_change_rate": round(self.scope_change_rate, 1),
            "carry_over_rate": round(self.carry_over_rate, 1),
            "sprint_health": self.sprint_health,
            "recent_sprints": self.sprint_data[-5:] if self.sprint_data else [],
        }

    @property
    def velocity_consistency(self) -> float:
        """Calculate velocity consistency across sprints.

        Returns:
            float: Consistency score (0-100)
        """
        if not self.sprint_data or len(self.sprint_data) < 2:
            return 0.0

        velocities = [s.get("velocity", 0) for s in self.sprint_data]
        if not velocities or max(velocities) == 0:
            return 0.0

        # Calculate coefficient of variation
        mean = sum(velocities) / len(velocities)
        if mean == 0:
            return 0.0

        variance = sum((v - mean) ** 2 for v in velocities) / len(velocities)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean

        # Convert to consistency score (lower CV = higher consistency)
        return max(0, min(100, 100 * (1 - cv)))

    @property
    def is_healthy(self) -> bool:
        """Check if sprint metrics indicate healthy process.

        Returns:
            bool: True if sprints are healthy
        """
        return (
            self.completion_rate >= 80
            and self.predictability >= 70
            and self.scope_change_rate <= 20
        )


@dataclass
class TeamMetrics:
    """Team-level productivity and collaboration metrics.

    Measures team dynamics, collaboration patterns, and overall team
    effectiveness in delivering value.

    Attributes:
        total_members: Total team members
        active_members: Currently active members
        team_velocity: Overall team velocity
        collaboration_score: Team collaboration score
        efficiency_score: Team efficiency score
        bus_factor: Team bus factor (knowledge distribution)
        skill_diversity: Skill diversity index
        communication_score: Team communication effectiveness
        team_health: Overall team health assessment
        teams: Sub-team metrics if applicable
        knowledge_silos: Identified knowledge silos
        collaboration_matrix: Who collaborates with whom
    """

    total_members: int = 0
    active_members: int = 0
    team_velocity: float = 0.0
    collaboration_score: float = 0.0
    efficiency_score: float = 0.0
    bus_factor: int = 0
    skill_diversity: float = 0.0
    communication_score: float = 0.0
    team_health: str = "unknown"
    teams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    knowledge_silos: List[str] = field(default_factory=list)
    collaboration_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "total_members": self.total_members,
            "active_members": self.active_members,
            "team_velocity": round(self.team_velocity, 2),
            "collaboration_score": round(self.collaboration_score, 1),
            "efficiency_score": round(self.efficiency_score, 1),
            "bus_factor": self.bus_factor,
            "skill_diversity": round(self.skill_diversity, 2),
            "communication_score": round(self.communication_score, 1),
            "team_health": self.team_health,
            "teams": self.teams,
            "knowledge_silo_count": len(self.knowledge_silos),
            "collaboration_pairs": len(self.collaboration_matrix),
        }

    @property
    def participation_rate(self) -> float:
        """Calculate team participation rate.

        Returns:
            float: Participation rate (0-100)
        """
        if self.total_members == 0:
            return 0.0
        return (self.active_members / self.total_members) * 100

    @property
    def velocity_per_member(self) -> float:
        """Calculate average velocity per team member.

        Returns:
            float: Velocity per member
        """
        if self.active_members == 0:
            return 0.0
        return self.team_velocity / self.active_members

    @property
    def needs_attention(self) -> bool:
        """Check if team metrics indicate issues.

        Returns:
            bool: True if team needs attention
        """
        return (
            self.participation_rate < 50
            or self.collaboration_score < 30
            or self.bus_factor <= 1
            or self.team_health in ["poor", "needs improvement"]
        )


@dataclass
class ProductivityMetrics:
    """Individual and team productivity measurements.

    Tracks various productivity indicators to understand work efficiency,
    output quality, and areas for improvement.

    Attributes:
        overall_productivity: Overall productivity score
        avg_daily_commits: Average commits per day
        avg_daily_lines: Average lines changed per day
        code_churn: Code churn rate
        rework_rate: Rate of rework/refactoring
        review_turnaround: Average review turnaround time
        peak_productivity_date: Date of peak productivity
        peak_productivity_score: Peak productivity score
        productivity_trend: Productivity trend direction
        top_performers: List of top performing contributors
        bottlenecks: Identified productivity bottlenecks
        focus_areas: Main areas of focus
        time_distribution: How time is distributed across activities
    """

    overall_productivity: float = 0.0
    avg_daily_commits: float = 0.0
    avg_daily_lines: float = 0.0
    code_churn: float = 0.0
    rework_rate: float = 0.0
    review_turnaround: float = 0.0
    peak_productivity_date: Optional[datetime] = None
    peak_productivity_score: float = 0.0
    productivity_trend: str = "stable"
    top_performers: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    focus_areas: List[Tuple[str, int]] = field(default_factory=list)
    time_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "overall_productivity": round(self.overall_productivity, 1),
            "avg_daily_commits": round(self.avg_daily_commits, 2),
            "avg_daily_lines": round(self.avg_daily_lines, 1),
            "code_churn": round(self.code_churn, 2),
            "rework_rate": round(self.rework_rate, 2),
            "review_turnaround": round(self.review_turnaround, 1),
            "peak_productivity_date": (
                self.peak_productivity_date.isoformat() if self.peak_productivity_date else None
            ),
            "peak_productivity_score": round(self.peak_productivity_score, 1),
            "productivity_trend": self.productivity_trend,
            "top_performers": self.top_performers[:5],
            "bottleneck_count": len(self.bottlenecks),
            "focus_areas": self.focus_areas[:10],
            "time_distribution": {k: round(v, 1) for k, v in self.time_distribution.items()},
        }

    @property
    def efficiency_rating(self) -> str:
        """Get efficiency rating based on productivity.

        Returns:
            str: Efficiency rating (excellent, good, fair, poor)
        """
        if self.overall_productivity >= 80:
            return "excellent"
        elif self.overall_productivity >= 60:
            return "good"
        elif self.overall_productivity >= 40:
            return "fair"
        else:
            return "poor"

    @property
    def has_bottlenecks(self) -> bool:
        """Check if bottlenecks are identified.

        Returns:
            bool: True if bottlenecks exist
        """
        return len(self.bottlenecks) > 0


@dataclass
class MomentumMetrics:
    """Overall momentum metrics for development.

    Aggregates various metrics to provide a comprehensive view of
    development momentum and project health.

    Attributes:
        momentum_score: Overall momentum score (0-100)
        velocity_score: Velocity component score
        quality_score: Quality component score
        collaboration_score: Collaboration component score
        productivity_score: Productivity component score
        momentum_trend: Momentum trend direction
        acceleration: Rate of momentum change
        sustainability: Momentum sustainability score
        risk_factors: Identified risk factors
        opportunities: Identified opportunities
        health_indicators: Key health indicators
    """

    momentum_score: float = 0.0
    velocity_score: float = 0.0
    quality_score: float = 0.0
    collaboration_score: float = 0.0
    productivity_score: float = 0.0
    momentum_trend: str = "stable"
    acceleration: float = 0.0
    sustainability: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    health_indicators: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "momentum_score": round(self.momentum_score, 1),
            "velocity_score": round(self.velocity_score, 1),
            "quality_score": round(self.quality_score, 1),
            "collaboration_score": round(self.collaboration_score, 1),
            "productivity_score": round(self.productivity_score, 1),
            "momentum_trend": self.momentum_trend,
            "acceleration": round(self.acceleration, 3),
            "sustainability": round(self.sustainability, 1),
            "risk_count": len(self.risk_factors),
            "opportunity_count": len(self.opportunities),
            "health_indicators": self.health_indicators,
        }

    @property
    def is_healthy(self) -> bool:
        """Check if momentum is healthy.

        Returns:
            bool: True if momentum is healthy
        """
        return (
            self.momentum_score >= 60 and self.sustainability >= 70 and len(self.risk_factors) <= 2
        )

    @property
    def momentum_category(self) -> str:
        """Categorize momentum level.

        Returns:
            str: Momentum category (excellent, good, fair, poor)
        """
        if self.momentum_score >= 80:
            return "excellent"
        elif self.momentum_score >= 60:
            return "good"
        elif self.momentum_score >= 40:
            return "fair"
        else:
            return "poor"


def calculate_momentum_metrics(
    daily_velocities: List[Any], individual_velocities: List[Any]
) -> MomentumMetrics:
    """Calculate overall momentum metrics from velocity data.

    Aggregates various velocity and productivity data to compute
    comprehensive momentum metrics.

    Args:
        daily_velocities: List of daily velocity data
        individual_velocities: List of individual contributor velocities

    Returns:
        MomentumMetrics: Calculated momentum metrics

    Example:
        >>> metrics = calculate_momentum_metrics(
        ...     daily_data,
        ...     contributor_data
        ... )
        >>> print(f"Momentum score: {metrics.momentum_score}")
    """
    logger = get_logger(__name__)
    metrics = MomentumMetrics()

    # Calculate velocity score
    if daily_velocities:
        active_days = [d for d in daily_velocities if d.is_active]
        if active_days:
            # Average velocity
            avg_velocity = sum(d.velocity_points for d in active_days) / len(active_days)

            # Velocity consistency
            if len(active_days) > 1:
                velocities = [d.velocity_points for d in active_days]
                mean = sum(velocities) / len(velocities)
                variance = sum((v - mean) ** 2 for v in velocities) / len(velocities)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean if mean > 0 else 0
                consistency = max(0, 100 * (1 - cv))

                metrics.velocity_score = (avg_velocity * 2 + consistency) / 3
            else:
                metrics.velocity_score = avg_velocity * 2

            # Cap velocity score at 100
            metrics.velocity_score = min(100, metrics.velocity_score)

    # Calculate productivity score
    if individual_velocities:
        individual_scores = [v.productivity_score for v in individual_velocities]
        if individual_scores:
            metrics.productivity_score = sum(individual_scores) / len(individual_scores)

        # Calculate collaboration score
        all_files = set()
        contributor_files = {}

        for velocity in individual_velocities:
            all_files.update(velocity.files_touched)
            contributor_files[velocity.email] = velocity.files_touched

        # Files touched by multiple people indicate collaboration
        if all_files:
            shared_files = 0
            for file in all_files:
                contributors_on_file = sum(
                    1 for files in contributor_files.values() if file in files
                )
                if contributors_on_file > 1:
                    shared_files += 1

            metrics.collaboration_score = (shared_files / len(all_files)) * 100

    # Estimate quality score (simplified heuristic)
    # In a real system, this would incorporate test coverage, bug rates, etc.
    metrics.quality_score = 70.0  # Default moderate quality

    # Adjust quality based on productivity patterns
    if metrics.productivity_score > 80:
        metrics.quality_score += 10
    elif metrics.productivity_score < 40:
        metrics.quality_score -= 10

    # Calculate overall momentum score
    weights = {"velocity": 0.3, "productivity": 0.3, "quality": 0.2, "collaboration": 0.2}

    metrics.momentum_score = (
        metrics.velocity_score * weights["velocity"]
        + metrics.productivity_score * weights["productivity"]
        + metrics.quality_score * weights["quality"]
        + metrics.collaboration_score * weights["collaboration"]
    )

    # Determine momentum trend
    if daily_velocities and len(daily_velocities) > 7:
        # Compare recent vs older velocity
        mid_point = len(daily_velocities) // 2
        recent = daily_velocities[mid_point:]
        older = daily_velocities[:mid_point]

        recent_avg = sum(d.velocity_points for d in recent if d.is_active) / max(1, len(recent))
        older_avg = sum(d.velocity_points for d in older if d.is_active) / max(1, len(older))

        if recent_avg > older_avg * 1.1:
            metrics.momentum_trend = "increasing"
            metrics.acceleration = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        elif recent_avg < older_avg * 0.9:
            metrics.momentum_trend = "decreasing"
            metrics.acceleration = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            metrics.momentum_trend = "stable"
            metrics.acceleration = 0

    # Calculate sustainability
    # Based on consistency and participation
    consistency_factor = metrics.velocity_score / 100
    participation_factor = min(1.0, len(individual_velocities) / 5)  # Assume 5 is good team size

    metrics.sustainability = (consistency_factor * 0.6 + participation_factor * 0.4) * 100

    # Identify risk factors
    if metrics.velocity_score < 40:
        metrics.risk_factors.append("Low velocity")
    if metrics.collaboration_score < 30:
        metrics.risk_factors.append("Poor collaboration")
    if metrics.productivity_score < 40:
        metrics.risk_factors.append("Low productivity")
    if metrics.sustainability < 50:
        metrics.risk_factors.append("Unsustainable pace")
    if len(individual_velocities) < 3:
        metrics.risk_factors.append("Small team size")

    # Identify opportunities
    if metrics.momentum_trend == "increasing":
        metrics.opportunities.append("Momentum is building - capitalize on it")
    if metrics.collaboration_score > 70:
        metrics.opportunities.append("Strong collaboration - leverage for knowledge sharing")
    if metrics.productivity_score > 70:
        metrics.opportunities.append("High productivity - consider tackling technical debt")

    # Set health indicators
    metrics.health_indicators = {
        "velocity_healthy": metrics.velocity_score >= 60,
        "productivity_healthy": metrics.productivity_score >= 60,
        "collaboration_healthy": metrics.collaboration_score >= 50,
        "sustainable": metrics.sustainability >= 70,
        "trending_positive": metrics.momentum_trend in ["increasing", "stable"],
    }

    logger.debug(f"Calculated momentum metrics: score={metrics.momentum_score:.1f}")

    return metrics


def calculate_sprint_velocity(commits: List[Any], sprint_duration: int = 14) -> float:
    """Calculate velocity for a sprint period.

    Calculates story points or velocity equivalent based on
    commit activity and code changes.

    Args:
        commits: List of commits in sprint
        sprint_duration: Sprint length in days

    Returns:
        float: Calculated sprint velocity

    Example:
        >>> velocity = calculate_sprint_velocity(
        ...     sprint_commits,
        ...     sprint_duration=14
        ... )
        >>> print(f"Sprint velocity: {velocity}")
    """
    if not commits:
        return 0.0

    velocity = 0.0

    # Base velocity on commit count
    velocity += len(commits) * 1.0

    # Add points for code changes
    total_changes = 0
    for commit in commits:
        if hasattr(commit, "stats") and hasattr(commit.stats, "total"):
            total_changes += commit.stats.total.get("lines", 0)

    # Logarithmic scale for changes
    if total_changes > 0:
        velocity += math.log(1 + total_changes) * 0.5

    # Normalize by sprint duration
    if sprint_duration > 0:
        velocity = velocity * (14 / sprint_duration)  # Normalize to 2-week sprint

    return velocity


def calculate_team_efficiency(team_metrics: TeamMetrics) -> float:
    """Calculate team efficiency score.

    Combines various team metrics to compute an overall
    efficiency score.

    Args:
        team_metrics: Team metrics data

    Returns:
        float: Team efficiency score (0-100)

    Example:
        >>> efficiency = calculate_team_efficiency(team_metrics)
        >>> print(f"Team efficiency: {efficiency}%")
    """
    if team_metrics.total_members == 0:
        return 0.0

    score = 0.0

    # Participation rate (30%)
    participation = team_metrics.participation_rate
    score += participation * 0.3

    # Collaboration score (25%)
    score += team_metrics.collaboration_score * 0.25

    # Velocity per member (25%)
    # Normalize velocity per member (assume 10 velocity/member is good)
    normalized_velocity = min(100, team_metrics.velocity_per_member * 10)
    score += normalized_velocity * 0.25

    # Bus factor (20%)
    # Higher bus factor is better (assume 3+ is good)
    bus_factor_score = min(100, team_metrics.bus_factor * 33.33)
    score += bus_factor_score * 0.2

    return min(100, score)


def predict_velocity(
    historical_velocities: List[float], periods_ahead: int = 1, confidence_level: float = 0.8
) -> Tuple[float, float]:
    """Predict future velocity based on historical data.

    Uses simple linear regression to predict future velocity
    with confidence intervals.

    Args:
        historical_velocities: List of historical velocity values
        periods_ahead: Number of periods to predict ahead
        confidence_level: Confidence level for prediction

    Returns:
        Tuple[float, float]: (predicted_velocity, confidence)

    Example:
        >>> prediction, confidence = predict_velocity(
        ...     [10, 12, 11, 13, 14],
        ...     periods_ahead=2
        ... )
        >>> print(f"Predicted: {prediction} (confidence: {confidence})")
    """
    if not historical_velocities or len(historical_velocities) < 2:
        return 0.0, 0.0

    n = len(historical_velocities)

    # Simple linear regression
    x_values = list(range(n))
    x_mean = sum(x_values) / n
    y_mean = sum(historical_velocities) / n

    # Calculate slope and intercept
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, historical_velocities))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        # No variance in x, return average
        return y_mean, confidence_level

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    # Predict future value
    future_x = n - 1 + periods_ahead
    predicted = intercept + slope * future_x

    # Calculate confidence based on model fit
    # Calculate R-squared
    ss_tot = sum((y - y_mean) ** 2 for y in historical_velocities)
    if ss_tot == 0:
        r_squared = 1.0
    else:
        predicted_values = [intercept + slope * x for x in x_values]
        ss_res = sum((y - pred) ** 2 for y, pred in zip(historical_velocities, predicted_values))
        r_squared = 1 - (ss_res / ss_tot)

    # Adjust confidence based on R-squared and prediction distance
    actual_confidence = confidence_level * r_squared * (0.9 ** (periods_ahead - 1))

    return max(0, predicted), max(0, min(1, actual_confidence))


def calculate_burndown_rate(
    completed_work: List[float], total_work: float, time_elapsed: int, total_time: int
) -> Dict[str, Any]:
    """Calculate burndown rate and projections.

    Analyzes work completion rate for burndown charts and
    sprint completion predictions.

    Args:
        completed_work: List of completed work per time unit
        total_work: Total work to complete
        time_elapsed: Time units elapsed
        total_time: Total time units available

    Returns:
        Dict[str, Any]: Burndown metrics and projections

    Example:
        >>> burndown = calculate_burndown_rate(
        ...     [10, 8, 12, 9],
        ...     100,
        ...     4,
        ...     14
        ... )
        >>> print(f"On track: {burndown['on_track']}")
    """
    if not completed_work or total_work <= 0 or total_time <= 0:
        return {
            "actual_rate": 0.0,
            "required_rate": 0.0,
            "on_track": False,
            "projected_completion": None,
            "completion_percentage": 0.0,
        }

    work_done = sum(completed_work)
    remaining_work = total_work - work_done
    remaining_time = total_time - time_elapsed

    # Calculate actual burn rate
    actual_rate = work_done / time_elapsed if time_elapsed > 0 else 0

    # Calculate required burn rate
    required_rate = remaining_work / remaining_time if remaining_time > 0 else 0

    # Project completion
    if actual_rate > 0:
        time_to_complete = remaining_work / actual_rate
        projected_completion = time_elapsed + time_to_complete
    else:
        projected_completion = None

    # Check if on track
    on_track = actual_rate >= required_rate if required_rate > 0 else work_done >= total_work

    return {
        "actual_rate": round(actual_rate, 2),
        "required_rate": round(required_rate, 2),
        "on_track": on_track,
        "projected_completion": round(projected_completion, 1) if projected_completion else None,
        "completion_percentage": round(work_done / total_work * 100, 1),
        "work_remaining": round(remaining_work, 2),
        "time_remaining": remaining_time,
        "ahead_behind": round(work_done - (total_work * time_elapsed / total_time), 2),
    }
