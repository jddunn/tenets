"""Velocity tracker module for development momentum analysis.

This module provides the main tracking functionality for development velocity
and momentum. It analyzes git history to understand development patterns,
team productivity, and project velocity trends over time.

The VelocityTracker class orchestrates the analysis of commits, code changes,
and contributor activity to provide actionable insights into team momentum.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.config import TenetsConfig
from tenets.core.git import GitAnalyzer
from tenets.utils.logger import get_logger

from .metrics import (
    MomentumMetrics,
    ProductivityMetrics,
    SprintMetrics,
    TeamMetrics,
    VelocityTrend,
    calculate_momentum_metrics,
)

# Common bot patterns for Git services
BOT_PATTERNS = {
    # GitHub bots
    "github-actions[bot]",
    "dependabot[bot]",
    "renovate[bot]",
    "greenkeeper[bot]",
    "semantic-release-bot",
    "release-bot",
    "github-classroom[bot]",
    "codecov[bot]",
    "coveralls[bot]",
    "mergify[bot]",
    "allcontributors[bot]",
    "imgbot[bot]",
    "restyled-io[bot]",
    "whitesource-bolt-for-github[bot]",
    "snyk-bot",
    "pull[bot]",
    "stale[bot]",
    # GitLab bots
    "gitlab-bot",
    "gitlab-ci",
    # Generic bot patterns
    "bot@",
    "noreply@",
    "automated",
    "auto-commit",
    "auto-merge",
}


def is_bot_commit(author_name: str, author_email: str) -> bool:
    """Check if a commit is from a bot or automated system.

    Args:
        author_name: Commit author name
        author_email: Commit author email

    Returns:
        bool: True if commit appears to be from a bot
    """
    # Convert to lowercase for case-insensitive comparison
    name_lower = author_name.lower() if author_name else ""
    email_lower = author_email.lower() if author_email else ""

    # Check exact matches
    if name_lower in BOT_PATTERNS or email_lower in BOT_PATTERNS:
        return True

    # Check patterns in name
    bot_indicators = ["[bot]", "bot", "automation", "ci", "release", "deploy"]
    for indicator in bot_indicators:
        if indicator in name_lower:
            return True

    # Check email patterns
    if "noreply" in email_lower or "bot@" in email_lower or "automated" in email_lower:
        return True

    # Check for GitHub/GitLab automated emails
    if email_lower.endswith("@users.noreply.github.com"):
        # Could be a real user, check if it has bot indicators
        if any(bot in name_lower for bot in ["bot", "action", "automated"]):
            return True

    return False


@dataclass
class DailyVelocity:
    """Velocity metrics for a single day.

    Tracks development activity and productivity for a specific day,
    used for building velocity trends and burndown charts.

    Attributes:
        date: Date of activity
        commits: Number of commits
        lines_added: Lines of code added
        lines_removed: Lines of code removed
        files_changed: Number of files modified
        contributors: Set of active contributors
        pull_requests: Number of PRs merged
        issues_closed: Number of issues closed
        velocity_points: Calculated velocity points
        productivity_score: Daily productivity score
    """

    date: datetime
    commits: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_changed: int = 0
    contributors: Set[str] = field(default_factory=set)
    pull_requests: int = 0
    issues_closed: int = 0
    velocity_points: float = 0.0
    productivity_score: float = 0.0

    @property
    def net_lines(self) -> int:
        """Calculate net lines changed.

        Returns:
            int: Lines added minus lines removed
        """
        return self.lines_added - self.lines_removed

    @property
    def contributor_count(self) -> int:
        """Get number of unique contributors.

        Returns:
            int: Unique contributor count
        """
        return len(self.contributors)

    @property
    def is_active(self) -> bool:
        """Check if this was an active day.

        Returns:
            bool: True if any activity occurred
        """
        return self.commits > 0 or self.files_changed > 0


@dataclass
class WeeklyVelocity:
    """Velocity metrics aggregated by week.

    Provides week-level velocity metrics for sprint tracking and
    longer-term trend analysis.

    Attributes:
        week_start: Start date of the week
        week_end: End date of the week
        week_number: Week number in year
        daily_velocities: List of daily velocities
        total_commits: Total commits in week
        total_lines_changed: Total lines changed
        unique_contributors: Unique contributors in week
        avg_daily_velocity: Average daily velocity
        velocity_variance: Variance in daily velocity
        sprint_completion: Sprint completion percentage if applicable
    """

    week_start: datetime
    week_end: datetime
    week_number: int
    daily_velocities: List[DailyVelocity] = field(default_factory=list)
    total_commits: int = 0
    total_lines_changed: int = 0
    unique_contributors: Set[str] = field(default_factory=set)
    avg_daily_velocity: float = 0.0
    velocity_variance: float = 0.0
    sprint_completion: Optional[float] = None

    @property
    def active_days(self) -> int:
        """Count active days in the week.

        Returns:
            int: Number of days with activity
        """
        return sum(1 for d in self.daily_velocities if d.is_active)

    @property
    def productivity_score(self) -> float:
        """Calculate weekly productivity score.

        Returns:
            float: Productivity score (0-100)
        """
        if not self.daily_velocities:
            return 0.0

        daily_scores = [d.productivity_score for d in self.daily_velocities]
        return sum(daily_scores) / len(daily_scores)


@dataclass
class ContributorVelocity:
    """Velocity metrics for an individual contributor.

    Tracks individual developer productivity and contribution patterns
    to understand team dynamics and individual performance.

    Attributes:
        name: Contributor name
        email: Contributor email
        commits: Total commits
        lines_added: Total lines added
        lines_removed: Total lines removed
        files_touched: Set of files modified
        active_days: Days with commits
        first_commit: First commit date in period
        last_commit: Last commit date in period
        velocity_trend: Individual velocity trend
        productivity_score: Individual productivity score
        consistency_score: Consistency of contributions
        impact_score: Impact/influence score
        collaboration_score: Collaboration with others
        specialization_areas: Areas of expertise
    """

    name: str
    email: str
    commits: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_touched: Set[str] = field(default_factory=set)
    active_days: Set[str] = field(default_factory=set)
    first_commit: Optional[datetime] = None
    last_commit: Optional[datetime] = None
    velocity_trend: str = "stable"
    productivity_score: float = 0.0
    consistency_score: float = 0.0
    impact_score: float = 0.0
    collaboration_score: float = 0.0
    specialization_areas: List[str] = field(default_factory=list)

    @property
    def net_lines(self) -> int:
        """Calculate net lines contributed.

        Returns:
            int: Lines added minus lines removed
        """
        return self.lines_added - self.lines_removed

    @property
    def avg_commit_size(self) -> float:
        """Calculate average commit size.

        Returns:
            float: Average lines changed per commit
        """
        if self.commits == 0:
            return 0.0
        return (self.lines_added + self.lines_removed) / self.commits

    @property
    def daily_commit_rate(self) -> float:
        """Calculate average commits per active day.

        Returns:
            float: Commits per active day
        """
        if not self.active_days:
            return 0.0
        return self.commits / len(self.active_days)


@dataclass
class MomentumReport:
    """Comprehensive momentum and velocity analysis report.

    Aggregates all velocity metrics and trends to provide a complete
    picture of development momentum and team productivity.

    Attributes:
        period_start: Start date of analysis period
        period_end: End date of analysis period
        total_commits: Total commits in period
        total_contributors: Total unique contributors
        active_contributors: Currently active contributors
        momentum_metrics: Overall momentum metrics
        velocity_trend: Velocity trend analysis
        sprint_metrics: Sprint-based metrics
        team_metrics: Team-level metrics
        individual_velocities: Individual contributor velocities
        daily_breakdown: Daily velocity breakdown
        weekly_breakdown: Weekly velocity breakdown
        productivity_metrics: Productivity analysis
        recommendations: Actionable recommendations
        health_score: Overall momentum health score
    """

    period_start: datetime
    period_end: datetime
    total_commits: int = 0
    total_contributors: int = 0
    active_contributors: int = 0
    momentum_metrics: Optional[MomentumMetrics] = None
    velocity_trend: Optional[VelocityTrend] = None
    sprint_metrics: Optional[SprintMetrics] = None
    team_metrics: Optional[TeamMetrics] = None
    individual_velocities: List[ContributorVelocity] = field(default_factory=list)
    daily_breakdown: List[DailyVelocity] = field(default_factory=list)
    weekly_breakdown: List[WeeklyVelocity] = field(default_factory=list)
    productivity_metrics: Optional[ProductivityMetrics] = None
    recommendations: List[str] = field(default_factory=list)
    health_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "days": (self.period_end - self.period_start).days,
            },
            "summary": {
                "total_commits": self.total_commits,
                "total_contributors": self.total_contributors,
                "active_contributors": self.active_contributors,
                "health_score": round(self.health_score, 1),
            },
            "momentum": self.momentum_metrics.to_dict() if self.momentum_metrics else {},
            "velocity_trend": self.velocity_trend.to_dict() if self.velocity_trend else {},
            "sprint_metrics": self.sprint_metrics.to_dict() if self.sprint_metrics else {},
            "team_metrics": self.team_metrics.to_dict() if self.team_metrics else {},
            "productivity": (
                self.productivity_metrics.to_dict() if self.productivity_metrics else {}
            ),
            "top_contributors": [
                {
                    "name": c.name,
                    "commits": c.commits,
                    "productivity": round(c.productivity_score, 1),
                }
                for c in sorted(self.individual_velocities, key=lambda x: x.commits, reverse=True)[
                    :10
                ]
            ],
            "recommendations": self.recommendations,
        }

    @property
    def avg_daily_velocity(self) -> float:
        """Calculate average daily velocity.

        Returns:
            float: Average velocity per day
        """
        if not self.daily_breakdown:
            return 0.0

        active_days = [d for d in self.daily_breakdown if d.is_active]
        if not active_days:
            return 0.0

        return sum(d.velocity_points for d in active_days) / len(active_days)

    @property
    def velocity_stability(self) -> float:
        """Calculate velocity stability score.

        Lower variance indicates more stable/predictable velocity.

        Returns:
            float: Stability score (0-100)
        """
        if not self.daily_breakdown:
            return 0.0

        velocities = [d.velocity_points for d in self.daily_breakdown if d.is_active]
        if len(velocities) < 2:
            return 100.0

        # Calculate coefficient of variation
        mean = sum(velocities) / len(velocities)
        if mean == 0:
            return 0.0

        variance = sum((v - mean) ** 2 for v in velocities) / len(velocities)
        std_dev = variance**0.5
        cv = std_dev / mean

        # Convert to stability score (lower CV = higher stability)
        stability = max(0, 100 * (1 - cv))
        return stability


class VelocityTracker:
    """Main tracker for development velocity and momentum.

    Orchestrates the analysis of git history to track development velocity,
    team productivity, and momentum trends over time.

    Attributes:
        config: Configuration object
        logger: Logger instance
        git_analyzer: Git analyzer instance
    """

    def __init__(self, config: TenetsConfig):
        """Initialize velocity tracker.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.git_analyzer: Optional[GitAnalyzer] = None

    def track_momentum(
        self,
        repo_path: Path,
        period: str = "last-month",
        team: bool = False,
        author: Optional[str] = None,
        team_mapping: Optional[Dict[str, List[str]]] = None,
        sprint_duration: int = 14,
        daily_breakdown: bool = False,
        interval: str = "weekly",
        exclude_bots: bool = True,
        **kwargs,  # Accept additional parameters for compatibility
    ) -> MomentumReport:
        """Track development momentum for a repository.

        Analyzes git history to calculate velocity metrics, identify trends,
        and provide insights into development momentum.

        Args:
            repo_path: Path to git repository
            period: Time period to analyze (e.g., "last-month", "30 days")
            team: Whether to include team-wide metrics
            author: Specific author to analyze
            team_mapping: Optional mapping of team names to members
            sprint_duration: Sprint length in days for sprint metrics
            daily_breakdown: Whether to include daily velocity data
            interval: Aggregation interval (daily, weekly, monthly)
            exclude_bots: Whether to exclude bot commits from analysis

        Returns:
            MomentumReport: Comprehensive momentum analysis

        Example:
            >>> tracker = VelocityTracker(config)
            >>> report = tracker.track_momentum(
            ...     Path("."),
            ...     period="last-quarter",
            ...     team=True
            ... )
            >>> print(f"Team velocity: {report.avg_daily_velocity}")
        """
        self.logger.debug(f"Tracking momentum for {repo_path} over {period}")
        import time

        start_time = time.time()

        # Initialize git analyzer
        self.git_analyzer = GitAnalyzer(repo_path)

        if not self.git_analyzer.is_repo():
            self.logger.warning(f"Not a git repository: {repo_path}")
            return MomentumReport(period_start=datetime.now(), period_end=datetime.now())

        # Parse period - check if since/until are provided in kwargs
        if "since" in kwargs and "until" in kwargs:
            period_start = kwargs["since"]
            period_end = kwargs["until"]
        elif "since" in kwargs:
            period_start = kwargs["since"]
            period_end = datetime.now()
        else:
            period_start, period_end = self._parse_period(period)

        # Initialize report
        report = MomentumReport(period_start=period_start, period_end=period_end)

        # Get commit data
        self.logger.info(f"Fetching commits from {period_start} to {period_end}")
        fetch_start = time.time()
        commits = self._get_commits_in_period(period_start, period_end, author, exclude_bots)
        self.logger.info(f"Fetched {len(commits)} commits in {time.time() - fetch_start:.2f}s")

        if not commits:
            self.logger.info("No commits found in period")
            return report

        # Analyze daily velocity
        analyze_start = time.time()
        daily_data = self._analyze_daily_velocity(commits, period_start, period_end)
        # Human-friendly timing (avoid confusing 0.00s output)
        _elapsed = time.time() - analyze_start
        _elapsed_str = "<0.01s" if _elapsed < 0.01 else f"{_elapsed:.2f}s"
        self.logger.info(f"Analyzed daily velocity in {_elapsed_str}")
        if daily_breakdown:
            report.daily_breakdown = daily_data

        # Analyze weekly velocity
        if interval in ["weekly", "sprint"]:
            report.weekly_breakdown = self._analyze_weekly_velocity(daily_data)

        # Analyze individual velocities
        report.individual_velocities = self._analyze_individual_velocities(
            commits, period_start, period_end
        )

        # Calculate overall metrics
        report.total_commits = len(commits)
        report.total_contributors = len(
            set(
                c.author.email
                for c in commits
                if hasattr(c, "author")
                and hasattr(c.author, "email")
                and not is_bot_commit(getattr(c.author, "name", ""), getattr(c.author, "email", ""))
            )
        )
        report.active_contributors = sum(
            1
            for v in report.individual_velocities
            if v.last_commit and (datetime.now() - v.last_commit).days <= 7
        )

        # Calculate momentum metrics
        report.momentum_metrics = calculate_momentum_metrics(
            daily_data, report.individual_velocities
        )

        # Analyze velocity trend
        report.velocity_trend = self._analyze_velocity_trend(daily_data, report.weekly_breakdown)

        # Calculate sprint metrics if requested
        if sprint_duration > 0:
            report.sprint_metrics = self._calculate_sprint_metrics(daily_data, sprint_duration)

        # Calculate team metrics if requested
        if team:
            report.team_metrics = self._calculate_team_metrics(
                report.individual_velocities, team_mapping
            )

        # Calculate productivity metrics
        report.productivity_metrics = self._calculate_productivity_metrics(report)

        # Calculate health score
        report.health_score = self._calculate_health_score(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        self.logger.debug(
            f"Momentum tracking complete: {report.total_commits} commits, "
            f"{report.total_contributors} contributors"
        )

        return report

    def _parse_period(self, period: str) -> Tuple[datetime, datetime]:
        """Parse period string into date range.

        Args:
            period: Period string (e.g., "last-month", "30 days")

        Returns:
            Tuple[datetime, datetime]: Start and end dates
        """
        end_date = datetime.now()

        # Parse common period formats
        period_lower = period.lower().replace("-", " ").replace("_", " ")

        if "last month" in period_lower or period_lower == "month":
            start_date = end_date - timedelta(days=30)
        elif "last quarter" in period_lower or period_lower == "quarter":
            start_date = end_date - timedelta(days=90)
        elif "last year" in period_lower or period_lower == "year":
            start_date = end_date - timedelta(days=365)
        elif "last week" in period_lower or period_lower == "week":
            start_date = end_date - timedelta(days=7)
        elif "yesterday" in period_lower:
            start_date = end_date - timedelta(days=1)
        elif "today" in period_lower:
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Try to parse as "N days/weeks/months"
            import re

            match = re.match(r"(\d+)\s*(day|week|month)", period_lower)
            if match:
                num = int(match.group(1))
                unit = match.group(2)
                if unit == "day":
                    start_date = end_date - timedelta(days=num)
                elif unit == "week":
                    start_date = end_date - timedelta(weeks=num)
                elif unit == "month":
                    start_date = end_date - timedelta(days=num * 30)
                else:
                    start_date = end_date - timedelta(days=30)  # Default
            else:
                # Default to last month
                start_date = end_date - timedelta(days=30)

        return start_date, end_date

    def _get_commits_in_period(
        self,
        start_date: datetime,
        end_date: datetime,
        author: Optional[str] = None,
        exclude_bots: bool = True,
        max_commits: int = 500,
    ) -> List[Any]:
        """Get commits within a date range.

        Args:
            start_date: Start date
            end_date: End date
            author: Optional author filter
            exclude_bots: Whether to exclude bot commits
            max_commits: Maximum number of commits to fetch

        Returns:
            List[Any]: List of commit objects
        """
        commits = self.git_analyzer.get_commits_since(start_date, max_count=max_commits)

        # Filter by end date and optionally exclude bots
        filtered_commits = []
        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            if commit_date <= end_date:
                # Check if it's a bot commit
                if exclude_bots and hasattr(commit, "author"):
                    author_name = getattr(commit.author, "name", "")
                    author_email = getattr(commit.author, "email", "")
                    if is_bot_commit(author_name, author_email):
                        self.logger.debug(
                            f"Excluding bot commit from {author_name} <{author_email}>"
                        )
                        continue

                # Filter by author if specified
                if author:
                    if hasattr(commit, "author"):
                        if (
                            author.lower() in commit.author.name.lower()
                            or author.lower() in commit.author.email.lower()
                        ):
                            filtered_commits.append(commit)
                else:
                    filtered_commits.append(commit)

        return filtered_commits

    def _analyze_daily_velocity(
        self, commits: List[Any], start_date: datetime, end_date: datetime
    ) -> List[DailyVelocity]:
        """Analyze velocity on a daily basis.

        Args:
            commits: List of commits
            start_date: Period start
            end_date: Period end

        Returns:
            List[DailyVelocity]: Daily velocity data
        """
        # Group commits by day
        daily_map: Dict[str, DailyVelocity] = {}

        # Initialize all days in period
        current_date = start_date
        while current_date <= end_date:
            day_key = current_date.strftime("%Y-%m-%d")
            daily_map[day_key] = DailyVelocity(date=current_date)
            current_date += timedelta(days=1)

        # Process commits
        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            day_key = commit_date.strftime("%Y-%m-%d")

            if day_key not in daily_map:
                continue

            daily = daily_map[day_key]
            daily.commits += 1

            # Add contributor (excluding bots)
            if hasattr(commit, "author") and hasattr(commit.author, "email"):
                author_name = getattr(commit.author, "name", "")
                author_email = getattr(commit.author, "email", "")
                if not is_bot_commit(author_name, author_email):
                    daily.contributors.add(commit.author.email)

            # Process file changes (skip by default for performance)
            # Note: Accessing commit.stats is VERY expensive in GitPython
            # Only enable if explicitly needed
            include_file_stats = False  # Disabled for performance
            if include_file_stats and hasattr(commit, "stats") and hasattr(commit.stats, "files"):
                for file_path, stats in commit.stats.files.items():
                    daily.files_changed += 1
                    daily.lines_added += stats.get("insertions", 0)
                    daily.lines_removed += stats.get("deletions", 0)

            # Check for PR/issue references in commit message
            if hasattr(commit, "message"):
                message = commit.message.lower()
                # Simple heuristic for PR merges
                if any(
                    pattern in message for pattern in ["merge pull request", "merge pr", "merged #"]
                ):
                    daily.pull_requests += 1
                # Simple heuristic for issue closes
                if any(pattern in message for pattern in ["closes #", "fixes #", "resolves #"]):
                    daily.issues_closed += 1

        # Calculate velocity points for each day
        for daily in daily_map.values():
            daily.velocity_points = self._calculate_velocity_points(daily)
            daily.productivity_score = self._calculate_daily_productivity(daily)

        # Return sorted list
        return sorted(daily_map.values(), key=lambda d: d.date)

    def _analyze_weekly_velocity(
        self, daily_velocities: List[DailyVelocity]
    ) -> List[WeeklyVelocity]:
        """Aggregate daily velocities into weekly data.

        Args:
            daily_velocities: List of daily velocities

        Returns:
            List[WeeklyVelocity]: Weekly velocity data
        """
        if not daily_velocities:
            return []

        weekly_map: Dict[int, WeeklyVelocity] = {}

        for daily in daily_velocities:
            # Get week number
            week_num = daily.date.isocalendar()[1]
            year = daily.date.year
            week_key = f"{year}-{week_num}"

            if week_key not in weekly_map:
                # Calculate week boundaries
                week_start = daily.date - timedelta(days=daily.date.weekday())
                week_end = week_start + timedelta(days=6)

                weekly_map[week_key] = WeeklyVelocity(
                    week_start=week_start, week_end=week_end, week_number=week_num
                )

            weekly = weekly_map[week_key]
            weekly.daily_velocities.append(daily)
            weekly.total_commits += daily.commits
            weekly.total_lines_changed += daily.lines_added + daily.lines_removed
            weekly.unique_contributors.update(daily.contributors)

        # Calculate weekly metrics
        for weekly in weekly_map.values():
            if weekly.daily_velocities:
                velocities = [d.velocity_points for d in weekly.daily_velocities]
                weekly.avg_daily_velocity = sum(velocities) / len(velocities)

                # Calculate variance
                if len(velocities) > 1:
                    mean = weekly.avg_daily_velocity
                    weekly.velocity_variance = sum((v - mean) ** 2 for v in velocities) / len(
                        velocities
                    )

        return sorted(weekly_map.values(), key=lambda w: w.week_start)

    def _analyze_individual_velocities(
        self, commits: List[Any], start_date: datetime, end_date: datetime
    ) -> List[ContributorVelocity]:
        """Analyze velocity by individual contributor.

        Args:
            commits: List of commits
            start_date: Period start
            end_date: Period end

        Returns:
            List[ContributorVelocity]: Individual velocity data
        """
        contributor_map: Dict[str, ContributorVelocity] = {}

        for commit in commits:
            if not hasattr(commit, "author"):
                continue

            email = getattr(commit.author, "email", "unknown")
            name = getattr(commit.author, "name", "Unknown")

            # Skip bot contributors
            if is_bot_commit(name, email):
                self.logger.debug(f"Skipping bot contributor: {name} <{email}>")
                continue

            if email not in contributor_map:
                contributor_map[email] = ContributorVelocity(name=name, email=email)

            contributor = contributor_map[email]
            contributor.commits += 1

            # Track dates
            commit_date = datetime.fromtimestamp(commit.committed_date)
            if not contributor.first_commit or commit_date < contributor.first_commit:
                contributor.first_commit = commit_date
            if not contributor.last_commit or commit_date > contributor.last_commit:
                contributor.last_commit = commit_date

            # Track active days
            day_key = commit_date.strftime("%Y-%m-%d")
            contributor.active_days.add(day_key)

            # Process file changes (skip by default for performance)
            # Note: Accessing commit.stats is VERY expensive in GitPython
            include_file_stats = False  # Disabled for performance
            if include_file_stats and hasattr(commit, "stats") and hasattr(commit.stats, "files"):
                for file_path, stats in commit.stats.files.items():
                    contributor.files_touched.add(file_path)
                    contributor.lines_added += stats.get("insertions", 0)
                    contributor.lines_removed += stats.get("deletions", 0)

        # Calculate derived metrics
        for contributor in contributor_map.values():
            contributor.productivity_score = self._calculate_contributor_productivity(
                contributor, (end_date - start_date).days
            )
            contributor.consistency_score = self._calculate_consistency(contributor)
            contributor.impact_score = self._calculate_impact(contributor)
            contributor.velocity_trend = self._determine_velocity_trend(contributor)
            contributor.specialization_areas = self._identify_specializations(contributor)

        return sorted(contributor_map.values(), key=lambda c: c.commits, reverse=True)

    def _calculate_velocity_points(self, daily: DailyVelocity) -> float:
        """Calculate velocity points for a day.

        Args:
            daily: Daily velocity data

        Returns:
            float: Calculated velocity points
        """
        points = 0.0

        # Base points for commits
        points += daily.commits * 1.0

        # Points for code changes (logarithmic scale)
        if daily.lines_added + daily.lines_removed > 0:
            import math

            points += math.log(1 + daily.lines_added + daily.lines_removed) * 0.5

        # Points for files touched
        points += daily.files_changed * 0.2

        # Bonus for PRs and issues
        points += daily.pull_requests * 2.0
        points += daily.issues_closed * 1.5

        # Bonus for collaboration
        if daily.contributor_count > 1:
            points += (daily.contributor_count - 1) * 0.5

        return points

    def _calculate_daily_productivity(self, daily: DailyVelocity) -> float:
        """Calculate productivity score for a day.

        Args:
            daily: Daily velocity data

        Returns:
            float: Productivity score (0-100)
        """
        if daily.commits == 0:
            return 0.0

        score = 0.0

        # Base on velocity points
        score += min(50, daily.velocity_points * 5)

        # Factor in code quality (smaller commits are often better)
        avg_change_size = (daily.lines_added + daily.lines_removed) / max(1, daily.commits)
        if avg_change_size < 100:
            score += 20
        elif avg_change_size < 500:
            score += 10

        # Factor in collaboration
        if daily.contributor_count > 1:
            score += min(20, daily.contributor_count * 5)

        # Factor in PR/issue completion
        if daily.pull_requests > 0 or daily.issues_closed > 0:
            score += 10

        return min(100, score)

    def _calculate_contributor_productivity(
        self, contributor: ContributorVelocity, period_days: int
    ) -> float:
        """Calculate productivity score for a contributor.

        Args:
            contributor: Contributor velocity data
            period_days: Total days in period

        Returns:
            float: Productivity score (0-100)
        """
        score = 0.0

        # Base on commit frequency
        if period_days > 0:
            commit_rate = contributor.commits / period_days
            score += min(30, commit_rate * 30)

        # Factor in code contribution
        if contributor.net_lines > 0:
            import math

            score += min(30, math.log(1 + contributor.net_lines) * 2)

        # Factor in consistency
        if len(contributor.active_days) > 0 and period_days > 0:
            consistency = len(contributor.active_days) / period_days
            score += consistency * 20

        # Factor in breadth (files touched)
        if len(contributor.files_touched) > 0:
            score += min(20, len(contributor.files_touched) / 5)

        return min(100, score)

    def _calculate_consistency(self, contributor: ContributorVelocity) -> float:
        """Calculate consistency score for a contributor.

        Args:
            contributor: Contributor velocity data

        Returns:
            float: Consistency score (0-100)
        """
        if not contributor.first_commit or not contributor.last_commit:
            return 0.0

        total_days = (contributor.last_commit - contributor.first_commit).days + 1
        if total_days == 0:
            return 100.0

        active_ratio = len(contributor.active_days) / total_days
        return min(100, active_ratio * 100)

    def _calculate_impact(self, contributor: ContributorVelocity) -> float:
        """Calculate impact score for a contributor.

        Args:
            contributor: Contributor velocity data

        Returns:
            float: Impact score (0-100)
        """
        score = 0.0

        # Base on total contribution
        if contributor.commits > 0:
            score += min(40, contributor.commits)

        # Factor in file coverage
        if len(contributor.files_touched) > 0:
            score += min(30, len(contributor.files_touched) / 2)

        # Factor in net lines (productive contribution)
        if contributor.net_lines > 0:
            import math

            score += min(30, math.log(1 + contributor.net_lines))

        return min(100, score)

    def _determine_velocity_trend(self, contributor: ContributorVelocity) -> str:
        """Determine velocity trend for a contributor.

        Args:
            contributor: Contributor velocity data

        Returns:
            str: Trend direction (increasing, decreasing, stable)
        """
        # Simple heuristic based on recent activity
        if not contributor.last_commit:
            return "inactive"

        days_since_last = (datetime.now() - contributor.last_commit).days

        if days_since_last > 30:
            return "inactive"
        elif days_since_last > 14:
            return "decreasing"
        elif contributor.daily_commit_rate > 1:
            return "increasing"
        else:
            return "stable"

    def _identify_specializations(self, contributor: ContributorVelocity) -> List[str]:
        """Identify areas of specialization for a contributor.

        Args:
            contributor: Contributor velocity data

        Returns:
            List[str]: Specialization areas
        """
        areas = []

        # Analyze file paths for patterns
        path_patterns = defaultdict(int)

        for file_path in contributor.files_touched:
            parts = Path(file_path).parts
            for part in parts[:-1]:  # Exclude filename
                if part not in {".", "..", ""}:
                    path_patterns[part] += 1

        # Top directories are specialization areas
        sorted_patterns = sorted(path_patterns.items(), key=lambda x: x[1], reverse=True)

        areas.extend([pattern for pattern, _ in sorted_patterns[:3]])

        return areas

    def _analyze_velocity_trend(
        self, daily_data: List[DailyVelocity], weekly_data: List[WeeklyVelocity]
    ) -> VelocityTrend:
        """Analyze velocity trend over time.

        Args:
            daily_data: Daily velocity data
            weekly_data: Weekly velocity data

        Returns:
            VelocityTrend: Velocity trend analysis
        """
        trend = VelocityTrend()

        if not daily_data:
            return trend

        # Calculate basic statistics
        active_velocities = [d.velocity_points for d in daily_data if d.is_active]

        if active_velocities:
            trend.avg_velocity = sum(active_velocities) / len(active_velocities)
            trend.max_velocity = max(active_velocities)
            trend.min_velocity = min(active_velocities)

            # Calculate trend direction (simple linear regression)
            if len(active_velocities) > 1:
                n = len(active_velocities)
                x_mean = n / 2
                y_mean = trend.avg_velocity

                numerator = sum(
                    (i - x_mean) * (v - y_mean) for i, v in enumerate(active_velocities)
                )
                denominator = sum((i - x_mean) ** 2 for i in range(n))

                if denominator != 0:
                    slope = numerator / denominator
                    if slope > 0.1:
                        trend.trend_direction = "increasing"
                    elif slope < -0.1:
                        trend.trend_direction = "decreasing"
                    else:
                        trend.trend_direction = "stable"

        # Add data points for visualization
        if weekly_data:
            trend.data_points = [
                {
                    "date": w.week_start.isoformat(),
                    "velocity": w.avg_daily_velocity,
                    "commits": w.total_commits,
                    "contributors": len(w.unique_contributors),
                }
                for w in weekly_data
            ]

        # Calculate stability score
        if len(active_velocities) > 1:
            mean = trend.avg_velocity
            variance = sum((v - mean) ** 2 for v in active_velocities) / len(active_velocities)
            std_dev = variance**0.5
            cv = std_dev / mean if mean > 0 else 0
            trend.stability_score = max(0, 100 * (1 - cv))

        return trend

    def _calculate_sprint_metrics(
        self, daily_data: List[DailyVelocity], sprint_duration: int
    ) -> SprintMetrics:
        """Calculate sprint-based metrics.

        Args:
            daily_data: Daily velocity data
            sprint_duration: Sprint length in days

        Returns:
            SprintMetrics: Sprint metrics analysis
        """
        metrics = SprintMetrics()

        if not daily_data or sprint_duration <= 0:
            return metrics

        # Group days into sprints
        sprints = []
        current_sprint = []

        for i, daily in enumerate(daily_data):
            current_sprint.append(daily)

            if len(current_sprint) >= sprint_duration:
                sprints.append(current_sprint)
                current_sprint = []

        if current_sprint:
            sprints.append(current_sprint)

        metrics.total_sprints = len(sprints)

        # Calculate metrics per sprint
        sprint_velocities = []

        for sprint in sprints:
            sprint_velocity = sum(d.velocity_points for d in sprint)
            sprint_velocities.append(sprint_velocity)

            sprint_commits = sum(d.commits for d in sprint)
            sprint_contributors = set()
            for d in sprint:
                sprint_contributors.update(d.contributors)

            metrics.sprint_data.append(
                {
                    "start_date": sprint[0].date,
                    "end_date": sprint[-1].date,
                    "velocity": sprint_velocity,
                    "commits": sprint_commits,
                    "contributors": len(sprint_contributors),
                    "active_days": sum(1 for d in sprint if d.is_active),
                }
            )

        # Calculate statistics
        if sprint_velocities:
            metrics.avg_velocity = sum(sprint_velocities) / len(sprint_velocities)
            metrics.max_velocity = max(sprint_velocities)
            metrics.min_velocity = min(sprint_velocities)

            # Velocity trend
            if len(sprint_velocities) > 1:
                recent_avg = sum(sprint_velocities[-3:]) / len(sprint_velocities[-3:])
                older_avg = sum(sprint_velocities[:-3]) / max(1, len(sprint_velocities[:-3]))

                if recent_avg > older_avg * 1.1:
                    metrics.velocity_trend = "increasing"
                elif recent_avg < older_avg * 0.9:
                    metrics.velocity_trend = "decreasing"
                else:
                    metrics.velocity_trend = "stable"

        return metrics

    def _calculate_team_metrics(
        self,
        individual_velocities: List[ContributorVelocity],
        team_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> TeamMetrics:
        """Calculate team-level metrics.

        Args:
            individual_velocities: Individual contributor velocities
            team_mapping: Optional team structure mapping

        Returns:
            TeamMetrics: Team metrics analysis
        """
        metrics = TeamMetrics()

        metrics.total_members = len(individual_velocities)
        # For solo repos, ensure at least one member when there is any activity
        if metrics.total_members == 0 and individual_velocities:
            metrics.total_members = 1
        metrics.active_members = sum(
            1 for v in individual_velocities if v.velocity_trend != "inactive"
        )

        # Calculate team velocity
        metrics.team_velocity = sum(v.commits for v in individual_velocities)

        # Calculate collaboration score
        all_files = set()
        for v in individual_velocities:
            all_files.update(v.files_touched)

        if all_files:
            # Files touched by multiple people indicate collaboration
            file_contributor_count = defaultdict(int)
            for v in individual_velocities:
                for file in v.files_touched:
                    file_contributor_count[file] += 1

            shared_files = sum(1 for count in file_contributor_count.values() if count > 1)
            metrics.collaboration_score = (shared_files / len(all_files)) * 100

        # Efficiency score
        if metrics.total_members > 0:
            metrics.efficiency_score = (metrics.active_members / metrics.total_members) * 100

        # Team health
        avg_productivity = sum(v.productivity_score for v in individual_velocities) / max(
            1, len(individual_velocities)
        )

        if avg_productivity > 70 and metrics.efficiency_score > 70:
            metrics.team_health = "excellent"
        elif avg_productivity > 50 and metrics.efficiency_score > 50:
            metrics.team_health = "good"
        elif avg_productivity > 30 and metrics.efficiency_score > 30:
            metrics.team_health = "fair"
        else:
            metrics.team_health = "needs improvement"

        # Process team mapping if provided
        if team_mapping:
            metrics.teams = {}
            for team_name, members in team_mapping.items():
                team_contributors = [v for v in individual_velocities if v.email in members]

                if team_contributors:
                    metrics.teams[team_name] = {
                        "members": len(team_contributors),
                        "commits": sum(v.commits for v in team_contributors),
                        "avg_productivity": sum(v.productivity_score for v in team_contributors)
                        / len(team_contributors),
                    }

        # Derive a pragmatic bus factor:
        # - If only one active member, bus factor should be 1
        # - Otherwise count contributors with at least 10% of commits in the period
        total_commits = sum(v.commits for v in individual_velocities)
        if metrics.active_members <= 1:
            metrics.bus_factor = 1 if (metrics.active_members + metrics.total_members) > 0 else 0
        elif total_commits > 0:
            critical = 0
            for v in individual_velocities:
                if v.commits / total_commits >= 0.10:
                    critical += 1
            metrics.bus_factor = max(1, critical)
        else:
            metrics.bus_factor = 1 if metrics.total_members == 1 else 0

        return metrics

    def _calculate_productivity_metrics(self, report: MomentumReport) -> ProductivityMetrics:
        """Calculate overall productivity metrics.

        Args:
            report: Momentum report

        Returns:
            ProductivityMetrics: Productivity analysis
        """
        metrics = ProductivityMetrics()

        # Calculate from daily data
        if report.daily_breakdown:
            active_days = [d for d in report.daily_breakdown if d.is_active]

            if active_days:
                metrics.avg_daily_commits = sum(d.commits for d in active_days) / len(active_days)
                metrics.avg_daily_lines = sum(d.net_lines for d in active_days) / len(active_days)

                # Peak productivity
                peak_day = max(active_days, key=lambda d: d.productivity_score)
                metrics.peak_productivity_date = peak_day.date
                metrics.peak_productivity_score = peak_day.productivity_score

        # Calculate from individual data
        if report.individual_velocities:
            metrics.top_performers = [
                {"name": v.name, "productivity": v.productivity_score, "commits": v.commits}
                for v in sorted(
                    report.individual_velocities, key=lambda x: x.productivity_score, reverse=True
                )[:5]
            ]

            # Focus areas (most changed areas)
            area_counts = defaultdict(int)
            for v in report.individual_velocities:
                for area in v.specialization_areas:
                    area_counts[area] += 1

            metrics.focus_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Calculate productivity score
        if report.momentum_metrics:
            metrics.overall_productivity = report.momentum_metrics.productivity_score

        return metrics

    def _calculate_health_score(self, report: MomentumReport) -> float:
        """Calculate overall momentum health score.

        Args:
            report: Momentum report

        Returns:
            float: Health score (0-100)
        """
        score = 50.0  # Start at neutral

        # Factor in velocity stability
        stability = report.velocity_stability
        if stability > 80:
            score += 15
        elif stability > 60:
            score += 10
        elif stability < 30:
            score -= 10

        # Factor in team participation
        if report.total_contributors > 0:
            active_ratio = report.active_contributors / report.total_contributors
            if active_ratio > 0.8:
                score += 15
            elif active_ratio > 0.5:
                score += 10
            elif active_ratio < 0.3:
                score -= 15

        # Factor in velocity trend
        if report.velocity_trend:
            if report.velocity_trend.trend_direction == "increasing":
                score += 10
            elif report.velocity_trend.trend_direction == "decreasing":
                score -= 10

        # Factor in productivity
        if report.productivity_metrics and report.productivity_metrics.overall_productivity:
            prod_score = report.productivity_metrics.overall_productivity
            if prod_score > 70:
                score += 10
            elif prod_score < 30:
                score -= 10

        return max(0, min(100, score))

    def _generate_recommendations(self, report: MomentumReport) -> List[str]:
        """Generate recommendations based on momentum analysis.

        Args:
            report: Momentum report

        Returns:
            List[str]: Actionable recommendations
        """
        recommendations = []

        # Velocity stability recommendations
        if report.velocity_stability < 50:
            recommendations.append(
                "Velocity is unstable. Consider: "
                "1) More consistent sprint planning, "
                "2) Reducing scope changes mid-sprint, "
                "3) Addressing blockers more quickly."
            )

        # Participation recommendations
        if report.total_contributors > 0:
            active_ratio = report.active_contributors / report.total_contributors
            if active_ratio < 0.5:
                recommendations.append(
                    f"Only {report.active_contributors}/{report.total_contributors} "
                    "contributors are active. Consider re-engaging inactive members "
                    "or adjusting team size."
                )

        # Velocity trend recommendations
        if report.velocity_trend:
            if report.velocity_trend.trend_direction == "decreasing":
                recommendations.append(
                    "Velocity is decreasing. Investigate: "
                    "1) Team morale and burnout, "
                    "2) Technical debt accumulation, "
                    "3) Process inefficiencies."
                )
            elif report.velocity_trend.trend_direction == "increasing":
                recommendations.append(
                    "Velocity is increasing! Maintain momentum by: "
                    "1) Documenting successful practices, "
                    "2) Avoiding overcommitment, "
                    "3) Investing in automation."
                )

        # Sprint recommendations
        if report.sprint_metrics:
            if report.sprint_metrics.velocity_trend == "decreasing":
                recommendations.append(
                    "Sprint velocity is declining. Review sprint retrospectives "
                    "and address recurring impediments."
                )

        # Team health recommendations
        if report.team_metrics:
            if report.team_metrics.team_health == "needs improvement":
                recommendations.append(
                    "Team health needs attention. Focus on: "
                    "1) Improving collaboration, "
                    "2) Clarifying roles and responsibilities, "
                    "3) Addressing skill gaps."
                )

            if report.team_metrics.collaboration_score < 30:
                recommendations.append(
                    "Low collaboration detected. Encourage: "
                    "1) Pair programming, "
                    "2) Code reviews, "
                    "3) Knowledge sharing sessions."
                )

        # Individual performance recommendations
        if report.individual_velocities:
            # Check for imbalanced contributions
            top_contributor = report.individual_velocities[0]
            if len(report.individual_velocities) > 1:
                total_commits = sum(v.commits for v in report.individual_velocities)
                if total_commits > 0 and top_contributor.commits / total_commits > 0.5:
                    recommendations.append(
                        f"{top_contributor.name} is contributing >50% of commits. "
                        "Consider distributing work more evenly to avoid burnout "
                        "and knowledge silos."
                    )

        # Overall health recommendations
        if report.health_score < 40:
            recommendations.append(
                "Overall momentum health is poor. Immediate action needed: "
                "1) Conduct team health check, "
                "2) Review and adjust processes, "
                "3) Address technical and organizational debt."
            )
        elif report.health_score > 80:
            recommendations.append(
                "Excellent momentum! Maintain by: "
                "1) Celebrating successes, "
                "2) Continuing current practices, "
                "3) Sharing learnings with other teams."
            )

        return recommendations


def track_momentum(
    repo_path: Path, period: str = "last-month", config: Optional[TenetsConfig] = None, **kwargs
) -> MomentumReport:
    """Convenience function to track momentum.

    Args:
        repo_path: Path to repository
        period: Time period to analyze
        config: Optional configuration
        **kwargs: Additional arguments for tracker

    Returns:
        MomentumReport: Momentum analysis
    """
    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)
    return tracker.track_momentum(repo_path, period, **kwargs)


def track_team_velocity(
    repo_path: Path,
    period: str = "last-month",
    team_mapping: Optional[Dict[str, List[str]]] = None,
    config: Optional[TenetsConfig] = None,
) -> TeamMetrics:
    """Track team velocity metrics.

    Args:
        repo_path: Path to repository
        period: Time period to analyze
        team_mapping: Team structure mapping
        config: Optional configuration

    Returns:
        TeamMetrics: Team velocity metrics
    """
    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)
    report = tracker.track_momentum(repo_path, period, team=True, team_mapping=team_mapping)

    return report.team_metrics


def track_individual_velocity(
    repo_path: Path, author: str, period: str = "last-month", config: Optional[TenetsConfig] = None
) -> Optional[ContributorVelocity]:
    """Track individual contributor velocity.

    Args:
        repo_path: Path to repository
        author: Author name or email
        period: Time period to analyze
        config: Optional configuration

    Returns:
        Optional[ContributorVelocity]: Individual velocity metrics
    """
    if config is None:
        config = TenetsConfig()

    tracker = VelocityTracker(config)
    report = tracker.track_momentum(repo_path, period, author=author)

    # Find the specific contributor
    for velocity in report.individual_velocities:
        if author.lower() in velocity.name.lower() or author.lower() in velocity.email.lower():
            return velocity

    return None
