"""Git integration package.

This package provides comprehensive git repository analysis capabilities including
repository metrics, blame analysis, history chronicling, and statistical insights.
It extracts valuable context from version control history to understand code
evolution, team dynamics, and development patterns.

The git package enables tenets to leverage version control information for
better context building, all without requiring any external API calls.

Main components:
- GitAnalyzer: Core git repository analyzer
- BlameAnalyzer: Line-by-line authorship tracking
- Chronicle: Repository history narrative generator
- GitStatsAnalyzer: Comprehensive repository statistics

Example usage:
    >>> from tenets.core.git import GitAnalyzer
    >>> from tenets.config import TenetsConfig
    >>>
    >>> config = TenetsConfig()
    >>> analyzer = GitAnalyzer(config)
    >>>
    >>> # Get recent commits
    >>> commits = analyzer.get_recent_commits(limit=10)
    >>> for commit in commits:
    >>>     print(f"{commit['sha']}: {commit['message']}")
    >>>
    >>> # Analyze repository statistics
    >>> from tenets.core.git import analyze_git_stats
    >>> stats = analyze_git_stats(Path("."))
    >>> print(f"Health score: {stats.health_score}")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .analyzer import CommitInfo, GitAnalyzer
from .blame import (
    BlameAnalyzer,
    BlameLine,
    BlameReport,
    FileBlame,
)
from .chronicle import (
    Chronicle,
    ChronicleBuilder,
    ChronicleReport,
    CommitSummary,
    DayActivity,
)
from .stats import (
    CommitStats,
    ContributorStats,
    FileStats,
    GitStatsAnalyzer,
    RepositoryStats,
)

# Version
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Core analyzer
    "GitAnalyzer",
    "CommitInfo",
    "analyze_repository",
    "get_git_context",
    # Blame analysis
    "BlameAnalyzer",
    "BlameLine",
    "FileBlame",
    "BlameReport",
    "analyze_blame",
    "get_file_ownership",
    # Chronicle generation
    "Chronicle",
    "ChronicleBuilder",
    "ChronicleReport",
    "CommitSummary",
    "DayActivity",
    "create_chronicle",
    "get_recent_history",
    # Statistics
    "GitStatsAnalyzer",
    "CommitStats",
    "ContributorStats",
    "FileStats",
    "RepositoryStats",
    "analyze_git_stats",
    "get_repository_health",
]


def analyze_repository(path: Optional[Path] = None, config: Optional[Any] = None) -> Dict[str, Any]:
    """Analyze a git repository comprehensively.

    This is a convenience function that creates a GitAnalyzer instance
    and performs basic repository analysis.

    Args:
        path: Path to repository (defaults to current directory)
        config: Optional TenetsConfig instance

    Returns:
        Dictionary with repository information including branch,
        recent commits, and contributors

    Example:
        >>> from tenets.core.git import analyze_repository
        >>>
        >>> repo_info = analyze_repository(Path("./my_project"))
        >>> print(f"Current branch: {repo_info['branch']}")
        >>> print(f"Recent commits: {len(repo_info['recent_commits'])}")
        >>> print(f"Contributors: {len(repo_info['contributors'])}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    if path is None:
        path = Path.cwd()

    analyzer = GitAnalyzer(config)

    if not analyzer.is_git_repo(path):
        return {"is_repo": False, "error": "Not a git repository"}

    return {
        "is_repo": True,
        "branch": analyzer.get_current_branch(path),
        "recent_commits": analyzer.get_recent_commits(path, limit=10),
        "contributors": analyzer.get_contributors(path, limit=10),
        "changes_last_week": analyzer.get_changes_since(path, since="1 week ago"),
    }


def get_git_context(
    path: Optional[Path] = None,
    files: Optional[List[str]] = None,
    since: str = "1 week ago",
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Get git context for specific files or time period.

    Retrieves relevant git information to provide context about
    recent changes, contributors, and activity patterns.

    Args:
        path: Repository path (defaults to current directory)
        files: Specific files to get context for
        since: Time period to analyze (e.g., "2 weeks ago")
        config: Optional TenetsConfig instance

    Returns:
        Dictionary with git context including commits, contributors,
        and activity summary

    Example:
        >>> from tenets.core.git import get_git_context
        >>>
        >>> context = get_git_context(
        ...     files=["src/main.py", "src/utils.py"],
        ...     since="1 month ago"
        ... )
        >>> print(f"Changes: {len(context['commits'])}")
        >>> print(f"Active contributors: {len(context['contributors'])}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    if path is None:
        path = Path.cwd()

    analyzer = GitAnalyzer(config)

    if not analyzer.is_git_repo(path):
        return {}

    return {
        "commits": analyzer.get_recent_commits(path, files=files, limit=50),
        "contributors": analyzer.get_contributors(path, files=files),
        "changes": analyzer.get_changes_since(path, since=since, files=files),
        "branch": analyzer.get_current_branch(path),
    }


def analyze_blame(
    repo_path: Path,
    target: str = ".",
    ignore_whitespace: bool = True,
    follow_renames: bool = True,
    max_files: int = 100,
    config: Optional[Any] = None,
) -> BlameReport:
    """Analyze code ownership using git blame.

    Performs line-by-line authorship analysis to understand
    code ownership patterns and identify knowledge holders.

    Args:
        repo_path: Path to git repository
        target: File or directory to analyze
        ignore_whitespace: Ignore whitespace changes in blame
        follow_renames: Track file renames
        max_files: Maximum files to analyze (for directories)
        config: Optional TenetsConfig instance

    Returns:
        BlameReport with comprehensive ownership analysis

    Example:
        >>> from tenets.core.git import analyze_blame
        >>>
        >>> blame = analyze_blame(Path("."), target="src/")
        >>> print(f"Bus factor: {blame.bus_factor}")
        >>> print(f"Primary authors: {blame.author_summary}")
        >>>
        >>> # Analyze single file
        >>> file_blame = analyze_blame(Path("."), target="main.py")
        >>> print(f"Primary author: {file_blame.primary_author}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    analyzer = BlameAnalyzer(config)

    target_path = Path(repo_path) / target

    if target_path.is_file():
        # Single file analysis
        file_blame = analyzer.analyze_file(
            repo_path, target, ignore_whitespace=ignore_whitespace, follow_renames=follow_renames
        )
        report = BlameReport(files_analyzed=1)
        report.file_blames[target] = file_blame
        analyzer._calculate_report_stats(report)
        report.recommendations = analyzer._generate_recommendations(report)
        return report
    else:
        # Directory analysis
        return analyzer.analyze_directory(
            repo_path, directory=target, recursive=True, max_files=max_files
        )


def get_file_ownership(
    repo_path: Path, file_path: str, config: Optional[Any] = None
) -> Dict[str, Any]:
    """Get ownership information for a specific file.

    Quick function to get the primary author and ownership
    distribution for a single file.

    Args:
        repo_path: Repository path
        file_path: Path to file relative to repo
        config: Optional TenetsConfig instance

    Returns:
        Dictionary with ownership information

    Example:
        >>> from tenets.core.git import get_file_ownership
        >>>
        >>> ownership = get_file_ownership(Path("."), "src/main.py")
        >>> print(f"Primary author: {ownership['primary_author']}")
        >>> print(f"Contributors: {ownership['contributors']}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    analyzer = BlameAnalyzer(config)
    file_blame = analyzer.analyze_file(repo_path, file_path)

    return {
        "primary_author": file_blame.primary_author,
        "contributors": list(file_blame.authors),
        "author_stats": file_blame.author_stats,
        "total_lines": file_blame.total_lines,
        "age_distribution": file_blame.age_distribution,
        "freshness_score": file_blame.freshness_score,
        "author_diversity": file_blame.author_diversity,
    }


def create_chronicle(
    repo_path: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    author: Optional[str] = None,
    include_stats: bool = True,
    max_commits: int = 1000,
    config: Optional[Any] = None,
) -> ChronicleReport:
    """Create a narrative chronicle of repository history.

    Generates a comprehensive narrative view of repository evolution
    including commits, contributors, trends, and significant events.

    Args:
        repo_path: Path to repository
        since: Start date or relative time (e.g., "1 month ago")
        until: End date or relative time
        author: Filter by specific author
        include_stats: Include detailed statistics
        max_commits: Maximum commits to analyze
        config: Optional TenetsConfig instance

    Returns:
        ChronicleReport with repository narrative

    Example:
        >>> from tenets.core.git import create_chronicle
        >>>
        >>> chronicle = create_chronicle(
        ...     Path("."),
        ...     since="3 months ago",
        ...     include_stats=True
        ... )
        >>> print(chronicle.summary)
        >>> print(f"Activity level: {chronicle.activity_level}")
        >>> for event in chronicle.significant_events:
        >>>     print(f"{event['date']}: {event['description']}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    chronicler = Chronicle(config)

    return chronicler.analyze(
        repo_path,
        since=since,
        until=until,
        author=author,
        include_stats=include_stats,
        max_commits=max_commits,
    )


def get_recent_history(
    repo_path: Path, days: int = 7, config: Optional[Any] = None
) -> Dict[str, Any]:
    """Get recent repository history summary.

    Quick function to get a summary of recent repository activity.

    Args:
        repo_path: Repository path
        days: Number of days to look back
        config: Optional TenetsConfig instance

    Returns:
        Dictionary with recent history summary

    Example:
        >>> from tenets.core.git import get_recent_history
        >>>
        >>> history = get_recent_history(Path("."), days=14)
        >>> print(f"Commits: {history['total_commits']}")
        >>> print(f"Active contributors: {history['contributors']}")
        >>> print(f"Most active day: {history['most_active_day']}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    chronicle = create_chronicle(repo_path, since=f"{days} days ago", config=config)

    return {
        "total_commits": chronicle.total_commits,
        "contributors": chronicle.total_contributors,
        "most_active_day": (
            chronicle.most_active_day.date.isoformat() if chronicle.most_active_day else None
        ),
        "activity_level": chronicle.activity_level,
        "commit_types": chronicle.commit_type_distribution,
        "trends": chronicle.trends,
        "summary": chronicle.summary,
    }


def analyze_git_stats(
    repo_path: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    include_files: bool = True,
    include_languages: bool = True,
    max_commits: int = 10000,
    config: Optional[Any] = None,
) -> RepositoryStats:
    """Analyze comprehensive repository statistics.

    Performs statistical analysis of repository to understand
    development patterns, team dynamics, and code health.

    Args:
        repo_path: Path to repository
        since: Start date or relative time
        until: End date or relative time
        include_files: Whether to include file statistics
        include_languages: Whether to analyze languages
        max_commits: Maximum commits to analyze
        config: Optional TenetsConfig instance

    Returns:
        RepositoryStats with comprehensive metrics

    Example:
        >>> from tenets.core.git import analyze_git_stats
        >>>
        >>> stats = analyze_git_stats(
        ...     Path("."),
        ...     since="6 months ago",
        ...     include_languages=True
        ... )
        >>> print(f"Health score: {stats.health_score}")
        >>> print(f"Bus factor: {stats.contributor_stats.bus_factor}")
        >>> print(f"Top languages: {stats.languages}")
        >>>
        >>> # View risk factors
        >>> for risk in stats.risk_factors:
        >>>     print(f"Risk: {risk}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    analyzer = GitStatsAnalyzer(config)

    return analyzer.analyze(
        repo_path,
        since=since,
        until=until,
        include_files=include_files,
        include_languages=include_languages,
        max_commits=max_commits,
    )


def get_repository_health(repo_path: Path, config: Optional[Any] = None) -> Dict[str, Any]:
    """Get a quick repository health assessment.

    Provides a simplified health check with key metrics and
    actionable recommendations.

    Args:
        repo_path: Repository path
        config: Optional TenetsConfig instance

    Returns:
        Dictionary with health assessment

    Example:
        >>> from tenets.core.git import get_repository_health
        >>>
        >>> health = get_repository_health(Path("."))
        >>> print(f"Score: {health['score']}/100")
        >>> print(f"Status: {health['status']}")
        >>> for issue in health['issues']:
        >>>     print(f"Issue: {issue}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    stats = analyze_git_stats(repo_path, since="3 months ago", config=config)

    # Determine status based on score
    if stats.health_score >= 80:
        status = "Excellent"
    elif stats.health_score >= 60:
        status = "Good"
    elif stats.health_score >= 40:
        status = "Fair"
    else:
        status = "Needs Attention"

    return {
        "score": stats.health_score,
        "status": status,
        "bus_factor": stats.contributor_stats.bus_factor,
        "active_contributors": stats.contributor_stats.active_contributors,
        "activity_trend": stats.activity_trend,
        "issues": stats.risk_factors,
        "strengths": stats.strengths,
        "recommendations": (
            stats.risk_factors[:3]
            if stats.risk_factors
            else ["Repository is healthy - maintain current practices"]
        ),
    }
