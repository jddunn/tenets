"""Chronicle module for git history analysis.

This module provides functionality for analyzing and summarizing git repository
history, including commit patterns, contributor activity, and development trends.
It extracts historical insights to help understand project evolution and team
dynamics over time.

The chronicle functionality provides a narrative view of repository changes,
making it easy to understand what happened, when, and by whom.
"""

import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.config import TenetsConfig
from tenets.core.git.analyzer import GitAnalyzer  # Fix circular import
from tenets.utils.logger import get_logger


@dataclass
class CommitSummary:
    """Summary information for a single commit.

    Provides a concise representation of a commit with key information
    for historical analysis and reporting.

    Attributes:
        sha: Commit SHA (short form)
        author: Commit author name
        email: Author email
        date: Commit date
        message: Commit message (first line)
        files_changed: Number of files changed
        lines_added: Lines added
        lines_removed: Lines removed
        is_merge: Whether this is a merge commit
        is_revert: Whether this is a revert commit
        tags: Associated tags
        branch: Branch name if available
        issue_refs: Referenced issue numbers
        pr_refs: Referenced PR numbers
    """

    sha: str
    author: str
    email: str
    date: datetime
    message: str
    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    is_merge: bool = False
    is_revert: bool = False
    tags: List[str] = field(default_factory=list)
    branch: Optional[str] = None
    issue_refs: List[str] = field(default_factory=list)
    pr_refs: List[str] = field(default_factory=list)

    @property
    def net_lines(self) -> int:
        """Calculate net lines changed.

        Returns:
            int: Lines added minus lines removed
        """
        return self.lines_added - self.lines_removed

    @property
    def commit_type(self) -> str:
        """Determine commit type from message.

        Returns:
            str: Commit type (feat, fix, docs, etc.)
        """
        message_lower = self.message.lower()

        # Special types first
        if self.is_merge or message_lower.startswith("merge "):
            return "merge"
        if self.is_revert or message_lower.startswith("revert "):
            return "revert"

        # Check for conventional commit format
        match = re.match(r"^(\w+)(?:\(.+?\))?:", self.message)
        if match:
            return match.group(1)

        # Heuristic-based detection (order matters)
        if any(word in message_lower for word in ["fix", "bug", "patch"]):
            return "fix"
        elif any(word in message_lower for word in ["test", "spec"]):
            return "test"
        elif any(word in message_lower for word in ["feat", "add", "new"]):
            return "feat"
        elif any(word in message_lower for word in ["doc", "readme"]):
            return "docs"
        elif any(word in message_lower for word in ["refactor", "clean"]):
            return "refactor"
        elif self.is_merge:
            return "merge"
        elif self.is_revert:
            return "revert"
        else:
            return "other"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "sha": self.sha,
            "author": self.author,
            "email": self.email,
            "date": self.date.isoformat(),
            "message": self.message,
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "type": self.commit_type,
            "is_merge": self.is_merge,
            "is_revert": self.is_revert,
            "tags": self.tags,
            "branch": self.branch,
            "issue_refs": self.issue_refs,
            "pr_refs": self.pr_refs,
        }


@dataclass
class DayActivity:
    """Activity summary for a single day.

    Aggregates all repository activity for a specific day to provide
    daily development rhythm insights.

    Attributes:
        date: Date of activity
        commits: List of commits on this day
        total_commits: Total commit count
        unique_authors: Set of unique authors
        lines_added: Total lines added
        lines_removed: Total lines removed
        files_touched: Set of files modified
        commit_types: Distribution of commit types
        peak_hour: Hour with most commits
        first_commit_time: Time of first commit
        last_commit_time: Time of last commit
    """

    date: datetime
    commits: List[CommitSummary] = field(default_factory=list)
    total_commits: int = 0
    unique_authors: Set[str] = field(default_factory=set)
    lines_added: int = 0
    lines_removed: int = 0
    files_touched: Set[str] = field(default_factory=set)
    commit_types: Dict[str, int] = field(default_factory=dict)
    peak_hour: Optional[int] = None
    first_commit_time: Optional[datetime] = None
    last_commit_time: Optional[datetime] = None

    @property
    def net_lines(self) -> int:
        """Calculate net lines changed.

        Returns:
            int: Net lines changed for the day
        """
        return self.lines_added - self.lines_removed

    @property
    def productivity_score(self) -> float:
        """Calculate daily productivity score.

        Returns:
            float: Productivity score (0-100)
        """
        score = 0.0

        # Base on commit count
        score += min(30, self.total_commits * 3)

        # Factor in unique authors (collaboration)
        score += min(20, len(self.unique_authors) * 5)

        # Factor in code changes
        if self.lines_added + self.lines_removed > 0:
            import math

            score += min(30, math.log(1 + self.lines_added + self.lines_removed) * 3)

        # Factor in commit quality (prefer fixes and features)
        quality_commits = self.commit_types.get("feat", 0) + self.commit_types.get("fix", 0)
        if self.total_commits > 0:
            quality_ratio = quality_commits / self.total_commits
            score += quality_ratio * 20

        return min(100, score)


@dataclass
class ChronicleReport:
    """Comprehensive chronicle report of repository history.

    Provides a complete narrative view of repository evolution including
    commits, contributors, trends, and significant events.

    Attributes:
        period_start: Start of chronicle period
        period_end: End of chronicle period
        total_commits: Total commits in period
        total_contributors: Total unique contributors
        commits: List of commit summaries
        daily_activity: Daily activity breakdown
        contributor_stats: Statistics by contributor
        commit_type_distribution: Distribution of commit types
        file_change_frequency: Most frequently changed files
        hot_periods: Periods of high activity
        quiet_periods: Periods of low activity
        significant_events: Notable events (releases, major changes)
        trends: Identified trends in development
        summary: Executive summary of the period
    """

    period_start: datetime
    period_end: datetime
    total_commits: int = 0
    total_contributors: int = 0
    commits: List[CommitSummary] = field(default_factory=list)
    daily_activity: List[DayActivity] = field(default_factory=list)
    contributor_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    commit_type_distribution: Dict[str, int] = field(default_factory=dict)
    file_change_frequency: List[Tuple[str, int]] = field(default_factory=list)
    hot_periods: List[Dict[str, Any]] = field(default_factory=list)
    quiet_periods: List[Dict[str, Any]] = field(default_factory=list)
    significant_events: List[Dict[str, Any]] = field(default_factory=list)
    trends: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

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
                "avg_commits_per_day": (
                    self.total_commits / max(1, (self.period_end - self.period_start).days)
                ),
                "narrative": self.summary,
            },
            "commit_types": self.commit_type_distribution,
            "top_contributors": list(self.contributor_stats.items())[:10],
            "top_files": self.file_change_frequency[:20],
            "hot_periods": self.hot_periods[:5],
            "quiet_periods": self.quiet_periods[:5],
            "significant_events": self.significant_events,
            "trends": self.trends,
        }

    @property
    def most_active_day(self) -> Optional[DayActivity]:
        """Get the most active day.

        Returns:
            Optional[DayActivity]: Most active day or None
        """
        if not self.daily_activity:
            return None
        return max(self.daily_activity, key=lambda d: d.total_commits)

    @property
    def activity_level(self) -> str:
        """Determine overall activity level.

        Returns:
            str: Activity level (high, moderate, low)
        """
        if not self.daily_activity:
            return "none"

        avg_daily_commits = self.total_commits / max(1, len(self.daily_activity))

        if avg_daily_commits > 10:
            return "high"
        elif avg_daily_commits > 3:
            return "moderate"
        else:
            return "low"


class Chronicle:
    """Main chronicle analyzer for git repositories.

    Analyzes git history to create a narrative view of repository evolution,
    identifying patterns, trends, and significant events.

    Attributes:
        config: Configuration object
        logger: Logger instance
        git_analyzer: Git analyzer instance
    """

    def __init__(self, config: TenetsConfig):
        """Initialize chronicle analyzer.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.git_analyzer: Optional[GitAnalyzer] = None

    def analyze(
        self,
        repo_path: Path,
        since: Optional[str] = None,
        until: Optional[str] = None,
        author: Optional[str] = None,
        branch: Optional[str] = None,
        include_merges: bool = True,
        include_stats: bool = True,
        max_commits: int = 1000,
    ) -> ChronicleReport:
        """Analyze repository history and create chronicle report.

        Creates a comprehensive narrative of repository evolution including
        commits, contributors, trends, and significant events.

        Args:
            repo_path: Path to git repository
            since: Start date or relative time (e.g., "2 weeks ago")
            until: End date or relative time
            author: Filter by specific author
            branch: Specific branch to analyze
            include_merges: Whether to include merge commits
            include_stats: Whether to include detailed statistics
            max_commits: Maximum commits to analyze

        Returns:
            ChronicleReport: Comprehensive chronicle analysis

        Example:
            >>> chronicle = Chronicle(config)
            >>> report = chronicle.analyze(
            ...     Path("."),
            ...     since="1 month ago",
            ...     include_stats=True
            ... )
            >>> print(report.summary)
        """
        self.logger.debug(f"Analyzing chronicle for {repo_path}")

        # Initialize git analyzer
        self.git_analyzer = GitAnalyzer(repo_path)

        if not self.git_analyzer.is_repo():
            self.logger.warning(f"Not a git repository: {repo_path}")
            return ChronicleReport(
                period_start=datetime.now(),
                period_end=datetime.now(),
                summary="No git repository found",
            )

        # Parse time period
        period_start, period_end = self._parse_time_period(since, until)

        # Initialize report
        report = ChronicleReport(period_start=period_start, period_end=period_end)

        # Get commits
        commits = self._get_commits(
            period_start, period_end, author, branch, include_merges, max_commits
        )

        if not commits:
            report.summary = "No commits found in the specified period"
            return report

        # Process commits sequentially
        # Note: Parallelization was attempted but GitPython commit objects
        # are not thread-safe and accessing commit.stats is very expensive
        for commit in commits:
            commit_summary = self._process_commit(commit, include_stats)
            report.commits.append(commit_summary)

        # Sort commits by date
        report.commits.sort(key=lambda c: c.date)

        # Update basic stats
        report.total_commits = len(report.commits)

        # Analyze daily activity
        report.daily_activity = self._analyze_daily_activity(report.commits)

        # Analyze contributors
        report.contributor_stats = self._analyze_contributors(report.commits)
        report.total_contributors = len(report.contributor_stats)

        # Analyze commit types
        report.commit_type_distribution = self._analyze_commit_types(report.commits)

        # Analyze file changes
        if include_stats:
            report.file_change_frequency = self._analyze_file_changes(commits)

        # Identify hot and quiet periods
        report.hot_periods = self._identify_hot_periods(report.daily_activity)
        report.quiet_periods = self._identify_quiet_periods(report.daily_activity)

        # Identify significant events
        report.significant_events = self._identify_significant_events(report.commits)

        # Identify trends
        report.trends = self._identify_trends(report)

        # Generate summary
        report.summary = self._generate_summary(report)

        self.logger.debug(
            f"Chronicle analysis complete: {report.total_commits} commits, "
            f"{report.total_contributors} contributors"
        )

        return report

    def _parse_time_period(
        self, since: Optional[str], until: Optional[str]
    ) -> Tuple[datetime, datetime]:
        """Parse time period strings.

        Args:
            since: Start time string
            until: End time string

        Returns:
            Tuple[datetime, datetime]: Start and end dates
        """
        end_date = datetime.now()

        # Parse until
        if until:
            try:
                # Try ISO format first
                end_date = datetime.fromisoformat(until)
            except:
                # Try relative format
                end_date = self._parse_relative_time(until, base=datetime.now())

        # Parse since
        if since:
            try:
                # Try ISO format first
                start_date = datetime.fromisoformat(since)
            except:
                # Try relative format
                start_date = self._parse_relative_time(since, base=end_date)
        else:
            # Default to 30 days ago
            start_date = end_date - timedelta(days=30)

        return start_date, end_date

    def _parse_relative_time(self, time_str: str, base: datetime) -> datetime:
        """Parse relative time string.

        Args:
            time_str: Relative time string (e.g., "2 weeks ago")
            base: Base datetime for calculation

        Returns:
            datetime: Calculated datetime
        """
        time_str = time_str.lower().strip()

        # Common patterns
        patterns = [
            (r"(\d+)\s*days?\s*ago", lambda m: base - timedelta(days=int(m.group(1)))),
            (r"(\d+)\s*weeks?\s*ago", lambda m: base - timedelta(weeks=int(m.group(1)))),
            (r"(\d+)\s*months?\s*ago", lambda m: base - timedelta(days=int(m.group(1)) * 30)),
            (r"yesterday", lambda m: base - timedelta(days=1)),
            (r"today", lambda m: base),
            (r"last\s*week", lambda m: base - timedelta(weeks=1)),
            (r"last\s*month", lambda m: base - timedelta(days=30)),
        ]

        for pattern, handler in patterns:
            match = re.search(pattern, time_str)
            if match:
                return handler(match)

        # Default to 30 days ago
        return base - timedelta(days=30)

    def _get_commits(
        self,
        start_date: datetime,
        end_date: datetime,
        author: Optional[str],
        branch: Optional[str],
        include_merges: bool,
        max_commits: int,
    ) -> List[Any]:
        """Get commits within specified criteria.

        Args:
            start_date: Start date
            end_date: End date
            author: Author filter
            branch: Branch filter
            include_merges: Include merge commits
            max_commits: Maximum commits to retrieve

        Returns:
            List[Any]: List of commit objects
        """
        # Get commits from git
        commits = self.git_analyzer.get_commits_since(start_date, max_count=max_commits)

        # Filter commits
        filtered = []
        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)

            # Date filter
            if commit_date > end_date:
                continue

            # Author filter
            if author:
                if not (
                    hasattr(commit, "author")
                    and (
                        author.lower() in commit.author.name.lower()
                        or author.lower() in commit.author.email.lower()
                    )
                ):
                    continue

            # Merge filter
            if not include_merges and len(commit.parents) > 1:
                continue

            filtered.append(commit)

        return filtered

    def _process_commit(self, commit: Any, include_stats: bool) -> CommitSummary:
        """Process a commit into a summary.

        Args:
            commit: Git commit object
            include_stats: Whether to include detailed stats

        Returns:
            CommitSummary: Processed commit summary
        """
        summary = CommitSummary(
            sha=commit.hexsha[:7],
            author=commit.author.name if hasattr(commit, "author") else "Unknown",
            email=commit.author.email if hasattr(commit, "author") else "",
            date=datetime.fromtimestamp(commit.committed_date),
            message=commit.message.split("\n")[0][:100],  # First line, truncated
            is_merge=len(commit.parents) > 1,
        )

        # Check if revert
        if "revert" in summary.message.lower():
            summary.is_revert = True

        # Extract issue/PR references
        summary.issue_refs = re.findall(r"#(\d+)", commit.message)
        summary.pr_refs = re.findall(r"PR[- ]?#?(\d+)", commit.message, re.IGNORECASE)

        # Get stats if requested
        if include_stats and hasattr(commit, "stats"):
            if hasattr(commit.stats, "files"):
                summary.files_changed = len(commit.stats.files)
                for file_stats in commit.stats.files.values():
                    summary.lines_added += file_stats.get("insertions", 0)
                    summary.lines_removed += file_stats.get("deletions", 0)

        return summary

    def _analyze_daily_activity(self, commits: List[CommitSummary]) -> List[DayActivity]:
        """Analyze commits by day.

        Args:
            commits: List of commit summaries

        Returns:
            List[DayActivity]: Daily activity breakdown
        """
        daily_map: Dict[str, DayActivity] = {}

        for commit in commits:
            day_key = commit.date.strftime("%Y-%m-%d")

            if day_key not in daily_map:
                daily_map[day_key] = DayActivity(
                    date=commit.date.replace(hour=0, minute=0, second=0, microsecond=0)
                )

            activity = daily_map[day_key]
            activity.commits.append(commit)
            activity.total_commits += 1
            activity.unique_authors.add(commit.author)
            activity.lines_added += commit.lines_added
            activity.lines_removed += commit.lines_removed

            # Track commit types
            commit_type = commit.commit_type
            activity.commit_types[commit_type] = activity.commit_types.get(commit_type, 0) + 1

            # Track times
            if not activity.first_commit_time or commit.date < activity.first_commit_time:
                activity.first_commit_time = commit.date
            if not activity.last_commit_time or commit.date > activity.last_commit_time:
                activity.last_commit_time = commit.date

        # Calculate peak hours
        for activity in daily_map.values():
            if activity.commits:
                hour_counts = defaultdict(int)
                for commit in activity.commits:
                    hour_counts[commit.date.hour] += 1
                activity.peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]

        return sorted(daily_map.values(), key=lambda d: d.date)

    def _analyze_contributors(self, commits: List[CommitSummary]) -> Dict[str, Dict[str, Any]]:
        """Analyze contributor statistics.

        Args:
            commits: List of commit summaries

        Returns:
            Dict[str, Dict[str, Any]]: Contributor statistics
        """
        contributors = defaultdict(
            lambda: {
                "commits": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "first_commit": None,
                "last_commit": None,
                "commit_types": defaultdict(int),
                "active_days": set(),
            }
        )

        for commit in commits:
            stats = contributors[commit.author]
            stats["commits"] += 1
            stats["lines_added"] += commit.lines_added
            stats["lines_removed"] += commit.lines_removed
            stats["commit_types"][commit.commit_type] += 1
            stats["active_days"].add(commit.date.strftime("%Y-%m-%d"))

            if not stats["first_commit"] or commit.date < stats["first_commit"]:
                stats["first_commit"] = commit.date
            if not stats["last_commit"] or commit.date > stats["last_commit"]:
                stats["last_commit"] = commit.date

        # Convert sets to counts for serialization
        for author, stats in contributors.items():
            stats["active_days"] = len(stats["active_days"])
            stats["commit_types"] = dict(stats["commit_types"])
            stats["net_lines"] = stats["lines_added"] - stats["lines_removed"]

        return dict(contributors)

    def _analyze_commit_types(self, commits: List[CommitSummary]) -> Dict[str, int]:
        """Analyze distribution of commit types.

        Args:
            commits: List of commit summaries

        Returns:
            Dict[str, int]: Commit type distribution
        """
        type_counts = defaultdict(int)

        for commit in commits:
            type_counts[commit.commit_type] += 1

        return dict(type_counts)

    def _analyze_file_changes(self, commits: List[Any]) -> List[Tuple[str, int]]:
        """Analyze most frequently changed files.

        Args:
            commits: List of commit objects

        Returns:
            List[Tuple[str, int]]: Files and change counts
        """
        file_counts = defaultdict(int)

        for commit in commits:
            if hasattr(commit, "stats") and hasattr(commit.stats, "files"):
                for file_path in commit.stats.files.keys():
                    file_counts[file_path] += 1

        return sorted(file_counts.items(), key=lambda x: x[1], reverse=True)

    def _identify_hot_periods(self, daily_activity: List[DayActivity]) -> List[Dict[str, Any]]:
        """Identify periods of high activity.

        Args:
            daily_activity: Daily activity data

        Returns:
            List[Dict[str, Any]]: Hot period descriptions
        """
        if not daily_activity:
            return []

        hot_periods = []
        avg_commits = sum(d.total_commits for d in daily_activity) / len(daily_activity)
        threshold = avg_commits * 2  # 2x average is "hot"

        current_period = None

        for activity in daily_activity:
            if activity.total_commits >= threshold:
                if current_period:
                    current_period["end"] = activity.date
                    current_period["days"] += 1
                    current_period["commits"] += activity.total_commits
                else:
                    current_period = {
                        "start": activity.date,
                        "end": activity.date,
                        "days": 1,
                        "commits": activity.total_commits,
                    }
            elif current_period:
                hot_periods.append(current_period)
                current_period = None

        if current_period:
            hot_periods.append(current_period)

        # Sort by commit count
        hot_periods.sort(key=lambda p: p["commits"], reverse=True)

        return hot_periods

    def _identify_quiet_periods(self, daily_activity: List[DayActivity]) -> List[Dict[str, Any]]:
        """Identify periods of low activity.

        Args:
            daily_activity: Daily activity data

        Returns:
            List[Dict[str, Any]]: Quiet period descriptions
        """
        if not daily_activity:
            return []

        quiet_periods = []

        # Look for gaps in activity
        for i in range(len(daily_activity) - 1):
            current = daily_activity[i]
            next_day = daily_activity[i + 1]

            gap_days = (next_day.date - current.date).days - 1

            if gap_days >= 3:  # 3+ days of no activity
                quiet_periods.append(
                    {
                        "start": current.date + timedelta(days=1),
                        "end": next_day.date - timedelta(days=1),
                        "days": gap_days,
                    }
                )

        # Sort by duration
        quiet_periods.sort(key=lambda p: p["days"], reverse=True)

        return quiet_periods

    def _identify_significant_events(self, commits: List[CommitSummary]) -> List[Dict[str, Any]]:
        """Identify significant events in history.

        Args:
            commits: List of commit summaries

        Returns:
            List[Dict[str, Any]]: Significant events
        """
        events = []

        for commit in commits:
            # Look for version tags
            if commit.tags:
                for tag in commit.tags:
                    if re.match(r"v?\d+\.\d+", tag):
                        events.append(
                            {
                                "type": "release",
                                "date": commit.date,
                                "description": f"Release {tag}",
                                "commit": commit.sha,
                            }
                        )

            # Look for major changes (large commits)
            if commit.files_changed > 50 or commit.lines_added > 1000:
                events.append(
                    {
                        "type": "major_change",
                        "date": commit.date,
                        "description": f"Major change: {commit.files_changed} files, {commit.lines_added} lines",
                        "commit": commit.sha,
                        "message": commit.message,
                    }
                )

            # Look for initial commit
            if "initial commit" in commit.message.lower():
                events.append(
                    {
                        "type": "initial_commit",
                        "date": commit.date,
                        "description": "Repository initialized",
                        "commit": commit.sha,
                    }
                )

        # Sort by date
        events.sort(key=lambda e: e["date"])

        return events

    def _identify_trends(self, report: ChronicleReport) -> List[str]:
        """Identify trends in development.

        Args:
            report: Chronicle report

        Returns:
            List[str]: Identified trends
        """
        trends = []

        # Activity trend
        if report.daily_activity:
            recent = (
                report.daily_activity[-7:]
                if len(report.daily_activity) > 7
                else report.daily_activity
            )
            older = report.daily_activity[:-7] if len(report.daily_activity) > 7 else []

            if recent and older:
                recent_avg = sum(d.total_commits for d in recent) / len(recent)
                older_avg = sum(d.total_commits for d in older) / len(older)

                if recent_avg > older_avg * 1.5:
                    trends.append("Development activity is increasing")
                elif recent_avg < older_avg * 0.5:
                    trends.append("Development activity is decreasing")

        # Contributor trend
        if report.contributor_stats:
            active_recent = sum(
                1
                for stats in report.contributor_stats.values()
                if stats["last_commit"] and (report.period_end - stats["last_commit"]).days < 7
            )

            if active_recent < len(report.contributor_stats) * 0.5:
                trends.append("Many contributors have become inactive")
            elif active_recent == len(report.contributor_stats):
                trends.append("All contributors remain active")

        # Commit type trends
        if report.commit_type_distribution:
            total = sum(report.commit_type_distribution.values())
            if total > 0:
                fix_ratio = report.commit_type_distribution.get("fix", 0) / total
                feat_ratio = report.commit_type_distribution.get("feat", 0) / total

                if fix_ratio > 0.4:
                    trends.append("High proportion of bug fixes indicates stability issues")
                if feat_ratio > 0.5:
                    trends.append("Heavy feature development in progress")

        # Hot period trends
        if len(report.hot_periods) > 3:
            trends.append("Development shows burst patterns with intense activity periods")

        # Quiet period trends
        if report.quiet_periods and max(p["days"] for p in report.quiet_periods) > 7:
            trends.append("Extended quiet periods suggest irregular development rhythm")

        return trends

    def _generate_summary(self, report: ChronicleReport) -> str:
        """Generate executive summary of the chronicle.

        Args:
            report: Chronicle report

        Returns:
            str: Summary narrative
        """
        period_days = (report.period_end - report.period_start).days

        summary_parts = []

        # Opening
        summary_parts.append(
            f"Over the past {period_days} days, the repository saw "
            f"{report.total_commits} commits from {report.total_contributors} contributors."
        )

        # Activity level
        activity_level = report.activity_level
        if activity_level == "high":
            summary_parts.append("Development activity has been very high.")
        elif activity_level == "moderate":
            summary_parts.append("Development activity has been steady.")
        else:
            summary_parts.append("Development activity has been relatively low.")

        # Top contributors
        if report.contributor_stats:
            top_contributor = max(report.contributor_stats.items(), key=lambda x: x[1]["commits"])
            summary_parts.append(
                f"The most active contributor was {top_contributor[0]} with "
                f"{top_contributor[1]['commits']} commits."
            )

        # Commit types
        if report.commit_type_distribution:
            top_type = max(report.commit_type_distribution.items(), key=lambda x: x[1])
            if top_type[0] == "feat":
                summary_parts.append("Development focused primarily on new features.")
            elif top_type[0] == "fix":
                summary_parts.append("Significant effort was spent on bug fixes.")
            elif top_type[0] == "refactor":
                summary_parts.append("The team focused on code refactoring.")

        # Significant events
        if report.significant_events:
            releases = [e for e in report.significant_events if e["type"] == "release"]
            if releases:
                summary_parts.append(f"The period included {len(releases)} releases.")

        # Trends
        if report.trends:
            summary_parts.append(f"Key trends: {report.trends[0]}")

        return " ".join(summary_parts)


def create_chronicle(
    repo_path: Path, since: Optional[str] = None, config: Optional[TenetsConfig] = None, **kwargs
) -> ChronicleReport:
    """Convenience function to create a repository chronicle.

    Args:
        repo_path: Path to repository
        since: Start time for chronicle
        config: Optional configuration
        **kwargs: Additional arguments for chronicle

    Returns:
        ChronicleReport: Chronicle analysis

    Example:
        >>> from tenets.core.git.chronicle import create_chronicle
        >>> report = create_chronicle(Path("."), since="1 month ago")
        >>> print(report.summary)
    """
    if config is None:
        config = TenetsConfig()

    chronicle = Chronicle(config)
    return chronicle.analyze(repo_path, since=since, **kwargs)


class ChronicleBuilder:
    """High-level builder that assembles a simple chronicle dict for CLI.

    This composes the existing Chronicle and GitAnalyzer without duplicating
    analysis logic. It converts inputs to what Chronicle expects and returns a
    compact, CLI-friendly dictionary.

    The CLI tests patch this class, but we provide a functional default for
    real usage.
    """

    def __init__(self, config: Optional[TenetsConfig] = None) -> None:
        self.config = config or TenetsConfig()
        self.logger = get_logger(__name__)

    def build_chronicle(
        self,
        repo_path: Path,
        *,
        since: Optional[object] = None,
        until: Optional[object] = None,
        branch: Optional[str] = None,
        authors: Optional[List[str]] = None,
        include_merges: bool = True,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a chronicle summary for the given repository.

        Args:
            repo_path: Path to a git repository
            since: Start time (datetime or relative/ISO string)
            until: End time (datetime or relative/ISO string)
            branch: Branch name to analyze
            authors: Optional author filters (currently advisory)
            include_merges: Include merge commits
            limit: Max commits to analyze (advisory to Chronicle)

        Returns:
            A dictionary with keys expected by the CLI views.
        """

        # Normalize time parameters to strings for Chronicle
        def _to_str(t: Optional[object]) -> Optional[str]:
            if t is None:
                return None
            if isinstance(t, str):
                return t
            try:
                from datetime import datetime as _dt

                if isinstance(t, _dt):
                    return t.isoformat()
            except Exception:
                pass
            # Fallback to string repr
            return str(t)

        since_s = _to_str(since)
        until_s = _to_str(until)

        # Run detailed analysis via Chronicle
        chron = Chronicle(self.config)
        report = chron.analyze(
            repo_path,
            since=since_s,
            until=until_s,
            author=(authors[0] if authors else None),  # basic filter support
            branch=branch,
            include_merges=include_merges,
            include_stats=False,  # Disabled for performance - commit.stats is very expensive
            max_commits=limit or 1000,
        )

        # Summarize fields commonly displayed by CLI
        period = (
            f"{report.period_start.date().isoformat()} to {report.period_end.date().isoformat()}"
        )
        files_changed = (
            len({p[0] for p in report.file_change_frequency}) if report.file_change_frequency else 0
        )

        # Lightweight activity signal (placeholder using totals)
        activity = {
            "trend": 0.0,  # real trend computation is beyond this builder
            "current_velocity": report.total_commits,
            "commits_this_week": (
                sum(d.total_commits for d in report.daily_activity[-7:])
                if report.daily_activity
                else 0
            ),
        }

        return {
            "period": period,
            "total_commits": report.total_commits,
            "files_changed": files_changed,
            "activity": activity,
            # Include a small slice of richer data for reports
            "commit_types": report.commit_type_distribution,
            "top_files": report.file_change_frequency[:10],
            "top_contributors": sorted(
                ((a, s.get("commits", 0)) for a, s in report.contributor_stats.items()),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            # Preserve the original report for advanced formatting if needed
            "_report": report,
        }
