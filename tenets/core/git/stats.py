"""Git statistics module.

This module provides comprehensive statistical analysis of git repositories,
including commit patterns, contributor metrics, file statistics, and
repository growth analysis. It helps understand repository health,
development patterns, and team dynamics through data-driven insights.

The statistics module aggregates various git metrics to provide actionable
insights for project management and technical decision-making.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.config import TenetsConfig
from tenets.core.git.analyzer import GitAnalyzer  # Fix circular import
from tenets.utils.logger import get_logger


@dataclass
class CommitStats:
    """Statistics for commits.

    Provides detailed statistical analysis of commit patterns including
    frequency, size, timing, and distribution metrics.

    Attributes:
        total_commits: Total number of commits
        commits_per_day: Average commits per day
        commits_per_week: Average commits per week
        commits_per_month: Average commits per month
        commit_size_avg: Average commit size (lines changed)
        commit_size_median: Median commit size
        commit_size_std: Standard deviation of commit size
        largest_commit: Largest single commit
        smallest_commit: Smallest single commit
        merge_commits: Number of merge commits
        revert_commits: Number of revert commits
        fix_commits: Number of fix commits
        feature_commits: Number of feature commits
        hourly_distribution: Commits by hour of day
        daily_distribution: Commits by day of week
        monthly_distribution: Commits by month
    """

    total_commits: int = 0
    commits_per_day: float = 0.0
    commits_per_week: float = 0.0
    commits_per_month: float = 0.0
    commit_size_avg: float = 0.0
    commit_size_median: float = 0.0
    commit_size_std: float = 0.0
    largest_commit: Dict[str, Any] = field(default_factory=dict)
    smallest_commit: Dict[str, Any] = field(default_factory=dict)
    merge_commits: int = 0
    revert_commits: int = 0
    fix_commits: int = 0
    feature_commits: int = 0
    hourly_distribution: List[int] = field(default_factory=lambda: [0] * 24)
    daily_distribution: List[int] = field(default_factory=lambda: [0] * 7)
    monthly_distribution: List[int] = field(default_factory=lambda: [0] * 12)

    @property
    def merge_ratio(self) -> float:
        """Calculate merge commit ratio.

        Returns:
            float: Ratio of merge commits to total
        """
        if self.total_commits == 0:
            return 0.0
        return self.merge_commits / self.total_commits

    @property
    def fix_ratio(self) -> float:
        """Calculate fix commit ratio.

        Returns:
            float: Ratio of fix commits to total
        """
        if self.total_commits == 0:
            return 0.0
        return self.fix_commits / self.total_commits

    @property
    def peak_hour(self) -> int:
        """Find peak commit hour.

        Returns:
            int: Hour with most commits (0-23)
        """
        if not any(self.hourly_distribution):
            return 0
        return self.hourly_distribution.index(max(self.hourly_distribution))

    @property
    def peak_day(self) -> str:
        """Find peak commit day.

        Returns:
            str: Day with most commits
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if not any(self.daily_distribution):
            return "Unknown"
        peak_index = self.daily_distribution.index(max(self.daily_distribution))
        return days[peak_index]


@dataclass
class ContributorStats:
    """Statistics for contributors.

    Provides analysis of contributor patterns, productivity metrics,
    and team dynamics based on git history.

    Attributes:
        total_contributors: Total unique contributors
        active_contributors: Contributors active in last 30 days
        new_contributors: New contributors in period
        contributor_commits: Commits per contributor
        contributor_lines: Lines changed per contributor
        contributor_files: Files touched per contributor
        top_contributors: Most active contributors
        contribution_inequality: Gini coefficient of contributions
        collaboration_graph: Who works with whom
        timezone_distribution: Contributors by timezone
        retention_rate: Contributor retention rate
        churn_rate: Contributor churn rate
    """

    total_contributors: int = 0
    active_contributors: int = 0
    new_contributors: int = 0
    contributor_commits: Dict[str, int] = field(default_factory=dict)
    contributor_lines: Dict[str, int] = field(default_factory=dict)
    contributor_files: Dict[str, Set[str]] = field(default_factory=dict)
    top_contributors: List[Tuple[str, int]] = field(default_factory=list)
    contribution_inequality: float = 0.0
    collaboration_graph: Dict[Tuple[str, str], int] = field(default_factory=dict)
    timezone_distribution: Dict[str, int] = field(default_factory=dict)
    retention_rate: float = 0.0
    churn_rate: float = 0.0

    @property
    def avg_commits_per_contributor(self) -> float:
        """Calculate average commits per contributor.

        Returns:
            float: Average commits
        """
        if self.total_contributors == 0:
            return 0.0
        total_commits = sum(self.contributor_commits.values())
        return total_commits / self.total_contributors

    @property
    def bus_factor(self) -> int:
        """Calculate bus factor.

        Returns:
            int: Number of key contributors
        """
        if not self.contributor_commits:
            return 0

        total_commits = sum(self.contributor_commits.values())
        if total_commits == 0:
            return 0

        # Sort contributors by commits
        sorted_contributors = sorted(
            self.contributor_commits.items(), key=lambda x: x[1], reverse=True
        )

        # Find minimum contributors for 50% of commits
        cumulative = 0
        bus_factor = 0

        for _, commits in sorted_contributors:
            cumulative += commits
            bus_factor += 1
            if cumulative >= total_commits * 0.5:
                break

        return bus_factor

    @property
    def collaboration_score(self) -> float:
        """Calculate collaboration score.

        Returns:
            float: Collaboration score (0-100)
        """
        if self.total_contributors <= 1:
            return 0.0

        # Maximum possible collaborations
        max_collaborations = (self.total_contributors * (self.total_contributors - 1)) / 2

        # Actual collaborations
        actual_collaborations = len(self.collaboration_graph)

        if max_collaborations == 0:
            return 0.0

        return min(100, (actual_collaborations / max_collaborations) * 100)


@dataclass
class FileStats:
    """Statistics for files.

    Provides analysis of file-level metrics including change frequency,
    size distribution, and file lifecycle patterns.

    Attributes:
        total_files: Total files in repository
        active_files: Files changed in period
        new_files: Files added in period
        deleted_files: Files deleted in period
        file_changes: Number of changes per file
        file_sizes: Size distribution of files
        largest_files: Largest files by line count
        most_changed: Most frequently changed files
        file_age: Age distribution of files
        file_churn: Churn rate per file
        hot_files: Files with high activity
        stable_files: Files with low activity
        file_types: Distribution by file type
    """

    total_files: int = 0
    active_files: int = 0
    new_files: int = 0
    deleted_files: int = 0
    file_changes: Dict[str, int] = field(default_factory=dict)
    file_sizes: Dict[str, int] = field(default_factory=dict)
    largest_files: List[Tuple[str, int]] = field(default_factory=list)
    most_changed: List[Tuple[str, int]] = field(default_factory=list)
    file_age: Dict[str, int] = field(default_factory=dict)
    file_churn: Dict[str, float] = field(default_factory=dict)
    hot_files: List[str] = field(default_factory=list)
    stable_files: List[str] = field(default_factory=list)
    file_types: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_file_size(self) -> float:
        """Calculate average file size.

        Returns:
            float: Average size in lines
        """
        if not self.file_sizes:
            return 0.0
        return sum(self.file_sizes.values()) / len(self.file_sizes)

    @property
    def file_stability(self) -> float:
        """Calculate overall file stability.

        Returns:
            float: Stability score (0-100)
        """
        if self.total_files == 0:
            return 0.0

        stable_ratio = len(self.stable_files) / self.total_files
        return min(100, stable_ratio * 100)

    @property
    def churn_rate(self) -> float:
        """Calculate overall churn rate.

        Returns:
            float: Average churn rate
        """
        if not self.file_churn:
            return 0.0
        return sum(self.file_churn.values()) / len(self.file_churn)


@dataclass
class RepositoryStats:
    """Overall repository statistics.

    Aggregates various statistical analyses to provide comprehensive
    insights into repository health and development patterns.

    Attributes:
        repo_age_days: Age of repository in days
        total_commits: Total commits
        total_contributors: Total contributors
        total_files: Total files
        total_lines: Total lines of code
        languages: Programming languages used
        commit_stats: Commit statistics
        contributor_stats: Contributor statistics
        file_stats: File statistics
        growth_rate: Repository growth rate
        activity_trend: Recent activity trend
        health_score: Overall health score
        risk_factors: Identified risk factors
        strengths: Identified strengths
    """

    repo_age_days: int = 0
    total_commits: int = 0
    total_contributors: int = 0
    total_files: int = 0
    total_lines: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    commit_stats: CommitStats = field(default_factory=CommitStats)
    contributor_stats: ContributorStats = field(default_factory=ContributorStats)
    file_stats: FileStats = field(default_factory=FileStats)
    growth_rate: float = 0.0
    activity_trend: str = "stable"
    health_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "overview": {
                "repo_age_days": self.repo_age_days,
                "total_commits": self.total_commits,
                "total_contributors": self.total_contributors,
                "total_files": self.total_files,
                "total_lines": self.total_lines,
                "health_score": round(self.health_score, 1),
            },
            "languages": dict(
                sorted(self.languages.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "commit_metrics": {
                "total": self.commit_stats.total_commits,
                "per_day": round(self.commit_stats.commits_per_day, 2),
                "merge_ratio": round(self.commit_stats.merge_ratio * 100, 1),
                "fix_ratio": round(self.commit_stats.fix_ratio * 100, 1),
                "peak_hour": self.commit_stats.peak_hour,
                "peak_day": self.commit_stats.peak_day,
            },
            "contributor_metrics": {
                "total": self.contributor_stats.total_contributors,
                "active": self.contributor_stats.active_contributors,
                "bus_factor": self.contributor_stats.bus_factor,
                "collaboration_score": round(self.contributor_stats.collaboration_score, 1),
                "top_contributors": self.contributor_stats.top_contributors[:5],
            },
            "file_metrics": {
                "total": self.file_stats.total_files,
                "active": self.file_stats.active_files,
                "stability": round(self.file_stats.file_stability, 1),
                "churn_rate": round(self.file_stats.churn_rate, 2),
                "hot_files": len(self.file_stats.hot_files),
            },
            "trends": {
                "growth_rate": round(self.growth_rate, 2),
                "activity_trend": self.activity_trend,
            },
            "risk_factors": self.risk_factors,
            "strengths": self.strengths,
        }


class GitStatsAnalyzer:
    """Analyzer for git repository statistics.

    Provides comprehensive statistical analysis of git repositories
    to understand development patterns, team dynamics, and code health.

    Attributes:
        config: Configuration object
        logger: Logger instance
        git_analyzer: Git analyzer instance
    """

    def __init__(self, config: TenetsConfig):
        """Initialize statistics analyzer.

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
        branch: Optional[str] = None,
        include_files: bool = True,
        include_languages: bool = True,
        max_commits: int = 10000,
    ) -> RepositoryStats:
        """Analyze repository statistics.

        Performs comprehensive statistical analysis of a git repository
        to provide insights into development patterns and health.

        Args:
            repo_path: Path to git repository
            since: Start date or relative time
            until: End date or relative time
            branch: Specific branch to analyze
            include_files: Whether to include file statistics
            include_languages: Whether to analyze languages
            max_commits: Maximum commits to analyze

        Returns:
            RepositoryStats: Comprehensive statistics

        Example:
            >>> analyzer = GitStatsAnalyzer(config)
            >>> stats = analyzer.analyze(Path("."))
            >>> print(f"Health score: {stats.health_score}")
        """
        self.logger.debug(f"Analyzing statistics for {repo_path}")

        # Initialize git analyzer
        self.git_analyzer = GitAnalyzer(repo_path)

        if not self.git_analyzer.is_repo():
            self.logger.warning(f"Not a git repository: {repo_path}")
            return RepositoryStats()

        # Initialize stats
        stats = RepositoryStats()

        # Get time period
        start_date, end_date = self._parse_time_period(since, until)

        # Get commits
        commits = self._get_commits(start_date, end_date, branch, max_commits)

        if not commits:
            self.logger.info("No commits found in specified period")
            return stats

        # Calculate basic metrics
        stats.total_commits = len(commits)
        stats.repo_age_days = (end_date - start_date).days

        # Analyze commits
        stats.commit_stats = self._analyze_commits(commits, start_date, end_date)

        # Analyze contributors
        stats.contributor_stats = self._analyze_contributors(commits, end_date)
        stats.total_contributors = stats.contributor_stats.total_contributors

        # Analyze files if requested
        if include_files:
            stats.file_stats = self._analyze_files(commits, repo_path)
            stats.total_files = stats.file_stats.total_files

            # Get total lines
            stats.total_lines = sum(stats.file_stats.file_sizes.values())

        # Analyze languages if requested
        if include_languages:
            stats.languages = self._analyze_languages(repo_path)

        # Calculate trends
        stats.growth_rate = self._calculate_growth_rate(commits)
        stats.activity_trend = self._determine_activity_trend(commits)

        # Calculate health score
        stats.health_score = self._calculate_health_score(stats)

        # Identify risks and strengths
        stats.risk_factors = self._identify_risks(stats)
        stats.strengths = self._identify_strengths(stats)

        self.logger.debug(
            f"Statistics analysis complete: {stats.total_commits} commits, "
            f"{stats.total_contributors} contributors"
        )

        return stats

    def _parse_time_period(
        self, since: Optional[str], until: Optional[str]
    ) -> Tuple[datetime, datetime]:
        """Parse time period for analysis.

        Args:
            since: Start time
            until: End time

        Returns:
            Tuple[datetime, datetime]: Start and end dates
        """
        import subprocess

        end_date = datetime.now()

        # Parse until
        if until:
            try:
                end_date = datetime.fromisoformat(until)
            except:
                # Try git date parsing
                pass

        # Parse since
        if since:
            try:
                start_date = datetime.fromisoformat(since)
            except:
                # Default to 1 year ago
                start_date = end_date - timedelta(days=365)
        else:
            # Get repository creation date
            try:
                result = subprocess.run(
                    ["git", "log", "--reverse", "--format=%at", "-1"],
                    cwd=self.git_analyzer.repo.working_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                timestamp = int(result.stdout.strip())
                start_date = datetime.fromtimestamp(timestamp)
            except:
                start_date = end_date - timedelta(days=365)

        return start_date, end_date

    def _get_commits(
        self, start_date: datetime, end_date: datetime, branch: Optional[str], max_commits: int
    ) -> List[Any]:
        """Get commits for analysis.

        Args:
            start_date: Start date
            end_date: End date
            branch: Branch to analyze
            max_commits: Maximum commits

        Returns:
            List[Any]: List of commits
        """
        commits = self.git_analyzer.get_commits_since(start_date, max_count=max_commits)

        # Filter by end date
        filtered = []
        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            if commit_date <= end_date:
                filtered.append(commit)

        return filtered

    def _analyze_commits(
        self, commits: List[Any], start_date: datetime, end_date: datetime
    ) -> CommitStats:
        """Analyze commit patterns.

        Args:
            commits: List of commits
            start_date: Period start
            end_date: Period end

        Returns:
            CommitStats: Commit statistics
        """
        stats = CommitStats()
        stats.total_commits = len(commits)

        if not commits:
            return stats

        # Calculate commit frequency
        period_days = max(1, (end_date - start_date).days)
        stats.commits_per_day = stats.total_commits / period_days
        stats.commits_per_week = stats.commits_per_day * 7
        stats.commits_per_month = stats.commits_per_day * 30

        # Analyze commit sizes and types
        commit_sizes = []

        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            message = commit.message.lower()

            # Time distribution
            stats.hourly_distribution[commit_date.hour] += 1
            stats.daily_distribution[commit_date.weekday()] += 1
            stats.monthly_distribution[commit_date.month - 1] += 1

            # Commit types
            if len(commit.parents) > 1:
                stats.merge_commits += 1
            if "revert" in message:
                stats.revert_commits += 1
            if any(word in message for word in ["fix", "bug", "patch", "repair"]):
                stats.fix_commits += 1
            if any(word in message for word in ["feat", "feature", "add", "new"]):
                stats.feature_commits += 1

            # Commit size
            if hasattr(commit, "stats"):
                total_changes = 0
                if hasattr(commit.stats, "total"):
                    total_changes = commit.stats.total.get("lines", 0)
                elif hasattr(commit.stats, "files"):
                    for file_stats in commit.stats.files.values():
                        total_changes += file_stats.get("insertions", 0)
                        total_changes += file_stats.get("deletions", 0)

                commit_sizes.append(total_changes)

                # Track largest/smallest
                commit_info = {
                    "sha": commit.hexsha[:7],
                    "message": commit.message.split("\n")[0][:80],
                    "size": total_changes,
                    "date": commit_date,
                }

                if not stats.largest_commit or total_changes > stats.largest_commit.get("size", 0):
                    stats.largest_commit = commit_info

                if not stats.smallest_commit or total_changes < stats.smallest_commit.get(
                    "size", float("inf")
                ):
                    if total_changes > 0:  # Ignore empty commits
                        stats.smallest_commit = commit_info

        # Calculate size statistics
        if commit_sizes:
            stats.commit_size_avg = statistics.mean(commit_sizes)
            stats.commit_size_median = statistics.median(commit_sizes)
            if len(commit_sizes) > 1:
                stats.commit_size_std = statistics.stdev(commit_sizes)

        return stats

    def _analyze_contributors(self, commits: List[Any], end_date: datetime) -> ContributorStats:
        """Analyze contributor patterns.

        Args:
            commits: List of commits
            end_date: Analysis end date

        Returns:
            ContributorStats: Contributor statistics
        """
        stats = ContributorStats()

        # Track contributor metrics
        contributor_first_commit = {}
        contributor_last_commit = {}
        files_by_contributor = defaultdict(set)

        for commit in commits:
            if not hasattr(commit, "author"):
                continue

            author = commit.author.email
            commit_date = datetime.fromtimestamp(commit.committed_date)

            # Track commits
            stats.contributor_commits[author] = stats.contributor_commits.get(author, 0) + 1

            # Track first/last commits
            if author not in contributor_first_commit:
                contributor_first_commit[author] = commit_date
            contributor_last_commit[author] = commit_date

            # Track changes
            if hasattr(commit, "stats"):
                lines_changed = 0
                if hasattr(commit.stats, "total"):
                    lines_changed = commit.stats.total.get("lines", 0)
                elif hasattr(commit.stats, "files"):
                    for file_path, file_stats in commit.stats.files.items():
                        files_by_contributor[author].add(file_path)
                        lines_changed += file_stats.get("insertions", 0)
                        lines_changed += file_stats.get("deletions", 0)

                stats.contributor_lines[author] = (
                    stats.contributor_lines.get(author, 0) + lines_changed
                )

        # Set contributor files
        stats.contributor_files = dict(files_by_contributor)

        # Calculate derived metrics
        stats.total_contributors = len(stats.contributor_commits)

        # Active contributors (active in last 30 days)
        for author, last_commit in contributor_last_commit.items():
            if (end_date - last_commit).days <= 30:
                stats.active_contributors += 1

        # New contributors (first commit in last 90 days)
        for author, first_commit in contributor_first_commit.items():
            if (end_date - first_commit).days <= 90:
                stats.new_contributors += 1

        # Top contributors
        stats.top_contributors = sorted(
            stats.contributor_commits.items(), key=lambda x: x[1], reverse=True
        )[:20]

        # Calculate inequality (Gini coefficient)
        if stats.contributor_commits:
            values = sorted(stats.contributor_commits.values())
            n = len(values)
            cumsum = 0
            for i, value in enumerate(values):
                cumsum += (n - i) * value
            stats.contribution_inequality = (n + 1 - 2 * cumsum / sum(values)) / n

        # Build collaboration graph
        # Simplified: contributors who changed same files collaborate
        for file_path in set().union(*files_by_contributor.values()):
            contributors_on_file = [
                author for author, files in files_by_contributor.items() if file_path in files
            ]

            for i, author1 in enumerate(contributors_on_file):
                for author2 in contributors_on_file[i + 1 :]:
                    key = tuple(sorted([author1, author2]))
                    stats.collaboration_graph[key] = stats.collaboration_graph.get(key, 0) + 1

        # Calculate retention rate
        if stats.total_contributors > 0:
            stats.retention_rate = (stats.active_contributors / stats.total_contributors) * 100

        # Calculate churn rate
        inactive = stats.total_contributors - stats.active_contributors
        if stats.total_contributors > 0:
            stats.churn_rate = (inactive / stats.total_contributors) * 100

        return stats

    def _analyze_files(self, commits: List[Any], repo_path: Path) -> FileStats:
        """Analyze file statistics.

        Args:
            commits: List of commits
            repo_path: Repository path

        Returns:
            FileStats: File statistics
        """
        import subprocess

        stats = FileStats()

        # Track file changes
        file_first_seen = {}
        file_last_seen = {}

        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)

            if hasattr(commit, "stats") and hasattr(commit.stats, "files"):
                for file_path in commit.stats.files.keys():
                    stats.file_changes[file_path] = stats.file_changes.get(file_path, 0) + 1

                    if file_path not in file_first_seen:
                        file_first_seen[file_path] = commit_date
                    file_last_seen[file_path] = commit_date

        # Get current files and sizes
        try:
            result = subprocess.run(
                ["git", "ls-files"], cwd=repo_path, capture_output=True, text=True, check=True
            )

            current_files = result.stdout.strip().split("\n")
            stats.total_files = len(current_files)

            # Get file sizes (line counts)
            for file_path in current_files[:1000]:  # Limit for performance
                try:
                    with open(repo_path / file_path, encoding="utf-8", errors="ignore") as f:
                        line_count = sum(1 for _ in f)
                        stats.file_sizes[file_path] = line_count
                except:
                    continue

                # Track file type
                ext = Path(file_path).suffix.lower()
                if ext:
                    stats.file_types[ext] = stats.file_types.get(ext, 0) + 1

        except subprocess.CalledProcessError:
            pass

        # Identify active files
        stats.active_files = len(stats.file_changes)

        # Most changed files
        stats.most_changed = sorted(stats.file_changes.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]

        # Largest files
        stats.largest_files = sorted(stats.file_sizes.items(), key=lambda x: x[1], reverse=True)[
            :20
        ]

        # Calculate file age
        now = datetime.now()
        for file_path, first_seen in file_first_seen.items():
            age_days = (now - first_seen).days
            stats.file_age[file_path] = age_days

        # Calculate churn
        for file_path, changes in stats.file_changes.items():
            if file_path in stats.file_age:
                age = max(1, stats.file_age[file_path])
                stats.file_churn[file_path] = changes / age

        # Identify hot and stable files
        if stats.file_churn:
            churn_values = sorted(stats.file_churn.values())
            if len(churn_values) > 0:
                high_threshold = churn_values[int(len(churn_values) * 0.8)]
                low_threshold = churn_values[int(len(churn_values) * 0.2)]

                for file_path, churn in stats.file_churn.items():
                    if churn >= high_threshold:
                        stats.hot_files.append(file_path)
                    elif churn <= low_threshold:
                        stats.stable_files.append(file_path)

        return stats

    def _analyze_languages(self, repo_path: Path) -> Dict[str, int]:
        """Analyze programming languages.

        Args:
            repo_path: Repository path

        Returns:
            Dict[str, int]: Language distribution
        """
        import subprocess

        languages = defaultdict(int)

        # Map extensions to languages
        ext_to_lang = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".h": "C/C++",
            ".cs": "C#",
            ".rb": "Ruby",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".r": "R",
            ".m": "Objective-C",
            ".sh": "Shell",
            ".yml": "YAML",
            ".yaml": "YAML",
            ".json": "JSON",
            ".xml": "XML",
            ".html": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".sql": "SQL",
            ".md": "Markdown",
        }

        try:
            # Get all files
            result = subprocess.run(
                ["git", "ls-files"], cwd=repo_path, capture_output=True, text=True, check=True
            )

            for file_path in result.stdout.strip().split("\n"):
                ext = Path(file_path).suffix.lower()
                if ext in ext_to_lang:
                    # Try to get file size
                    try:
                        file_full_path = repo_path / file_path
                        if file_full_path.exists():
                            with open(file_full_path, encoding="utf-8", errors="ignore") as f:
                                line_count = sum(1 for _ in f)
                                languages[ext_to_lang[ext]] += line_count
                    except:
                        languages[ext_to_lang[ext]] += 1

        except subprocess.CalledProcessError:
            pass

        return dict(languages)

    def _calculate_growth_rate(self, commits: List[Any]) -> float:
        """Calculate repository growth rate.

        Args:
            commits: List of commits

        Returns:
            float: Growth rate (commits per day)
        """
        if len(commits) < 2:
            return 0.0

        # Get date range
        dates = [datetime.fromtimestamp(c.committed_date) for c in commits]
        min_date = min(dates)
        max_date = max(dates)

        days = max(1, (max_date - min_date).days)

        return len(commits) / days

    def _determine_activity_trend(self, commits: List[Any]) -> str:
        """Determine activity trend.

        Args:
            commits: List of commits

        Returns:
            str: Trend direction
        """
        if len(commits) < 10:
            return "insufficient_data"

        # Split commits into halves
        mid_point = len(commits) // 2
        first_half = commits[:mid_point]
        second_half = commits[mid_point:]

        # Get date ranges
        first_dates = [datetime.fromtimestamp(c.committed_date) for c in first_half]
        second_dates = [datetime.fromtimestamp(c.committed_date) for c in second_half]

        first_days = max(1, (max(first_dates) - min(first_dates)).days)
        second_days = max(1, (max(second_dates) - min(second_dates)).days)

        first_rate = len(first_half) / first_days
        second_rate = len(second_half) / second_days

        if second_rate > first_rate * 1.2:
            return "increasing"
        elif second_rate < first_rate * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _calculate_health_score(self, stats: RepositoryStats) -> float:
        """Calculate repository health score.

        Args:
            stats: Repository statistics

        Returns:
            float: Health score (0-100)
        """
        score = 50.0  # Start at neutral

        # Positive factors

        # Active contributors
        if stats.contributor_stats.active_contributors > 5:
            score += 10
        elif stats.contributor_stats.active_contributors > 2:
            score += 5

        # Good bus factor
        if stats.contributor_stats.bus_factor > 3:
            score += 10
        elif stats.contributor_stats.bus_factor > 1:
            score += 5

        # Regular commits
        if stats.commit_stats.commits_per_week > 10:
            score += 10
        elif stats.commit_stats.commits_per_week > 3:
            score += 5

        # Good collaboration
        if stats.contributor_stats.collaboration_score > 70:
            score += 10
        elif stats.contributor_stats.collaboration_score > 40:
            score += 5

        # File stability
        if stats.file_stats.file_stability > 70:
            score += 10
        elif stats.file_stats.file_stability > 40:
            score += 5

        # Negative factors

        # High fix ratio (many bugs)
        if stats.commit_stats.fix_ratio > 0.4:
            score -= 10
        elif stats.commit_stats.fix_ratio > 0.25:
            score -= 5

        # High contributor churn
        if stats.contributor_stats.churn_rate > 50:
            score -= 10
        elif stats.contributor_stats.churn_rate > 30:
            score -= 5

        # High file churn
        if stats.file_stats.churn_rate > 0.5:
            score -= 10
        elif stats.file_stats.churn_rate > 0.3:
            score -= 5

        # Declining activity
        if stats.activity_trend == "decreasing":
            score -= 10

        # Low bus factor
        if stats.contributor_stats.bus_factor <= 1:
            score -= 15

        return max(0, min(100, score))

    def _identify_risks(self, stats: RepositoryStats) -> List[str]:
        """Identify risk factors.

        Args:
            stats: Repository statistics

        Returns:
            List[str]: Risk factors
        """
        risks = []

        if stats.contributor_stats.bus_factor <= 1:
            risks.append("Critical: Bus factor is 1 - single point of failure")

        if stats.contributor_stats.churn_rate > 50:
            risks.append("High contributor churn rate")

        if stats.commit_stats.fix_ratio > 0.4:
            risks.append("High bug fix ratio indicates quality issues")

        if stats.file_stats.churn_rate > 0.5:
            risks.append("High file churn rate indicates instability")

        if stats.contributor_stats.active_contributors < 2:
            risks.append("Very few active contributors")

        if stats.activity_trend == "decreasing":
            risks.append("Declining development activity")

        if len(stats.file_stats.hot_files) > stats.file_stats.total_files * 0.1:
            risks.append("Many unstable files requiring attention")

        return risks

    def _identify_strengths(self, stats: RepositoryStats) -> List[str]:
        """Identify strengths.

        Args:
            stats: Repository statistics

        Returns:
            List[str]: Strengths
        """
        strengths = []

        if stats.contributor_stats.bus_factor > 3:
            strengths.append("Good bus factor - knowledge well distributed")

        if stats.contributor_stats.collaboration_score > 70:
            strengths.append("High collaboration between contributors")

        if stats.commit_stats.commits_per_week > 10:
            strengths.append("Very active development")

        if stats.file_stats.file_stability > 70:
            strengths.append("Most files are stable")

        if stats.contributor_stats.retention_rate > 80:
            strengths.append("High contributor retention")

        if stats.activity_trend == "increasing":
            strengths.append("Growing development activity")

        if stats.contributor_stats.new_contributors > 0:
            strengths.append("Attracting new contributors")

        return strengths


def analyze_git_stats(
    repo_path: Path, config: Optional[TenetsConfig] = None, **kwargs
) -> RepositoryStats:
    """Convenience function to analyze git statistics.

    Args:
        repo_path: Path to repository
        config: Optional configuration
        **kwargs: Additional arguments

    Returns:
        RepositoryStats: Repository statistics

    Example:
        >>> from tenets.core.git.stats import analyze_git_stats
        >>> stats = analyze_git_stats(Path("."))
        >>> print(f"Health score: {stats.health_score}")
    """
    if config is None:
        config = TenetsConfig()

    analyzer = GitStatsAnalyzer(config)
    return analyzer.analyze(repo_path, **kwargs)
