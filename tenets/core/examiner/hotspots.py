"""Hotspot detection module for code examination.

This module identifies code hotspots - areas of the codebase that change
frequently, have high complexity, or exhibit other problematic patterns.
Hotspots often indicate areas that need refactoring, have bugs, or are
difficult to maintain.

The hotspot detector combines git history, complexity metrics, and other
indicators to identify problematic areas that deserve attention.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.config import TenetsConfig
from tenets.core.git import GitAnalyzer
from tenets.utils.logger import get_logger


@dataclass
class HotspotMetrics:
    """Metrics for identifying and scoring hotspots.

    Combines various indicators to determine if a code area is a hotspot
    that requires attention or refactoring.

    Attributes:
        change_frequency: How often the file changes
        commit_count: Total number of commits
        author_count: Number of unique authors
        lines_changed: Total lines added/removed
        bug_fix_commits: Number of bug fix commits
        refactor_commits: Number of refactoring commits
        complexity: Code complexity if available
        coupling: How many other files change with this one
        age_days: Days since file creation
        recency_days: Days since last change
        churn_rate: Rate of change over time
        defect_density: Estimated defect density
        stability_score: File stability score (0-100)
    """

    change_frequency: float = 0.0
    commit_count: int = 0
    author_count: int = 0
    lines_changed: int = 0
    bug_fix_commits: int = 0
    refactor_commits: int = 0
    complexity: float = 0.0
    coupling: int = 0
    age_days: int = 0
    recency_days: int = 0
    churn_rate: float = 0.0
    defect_density: float = 0.0
    stability_score: float = 100.0

    @property
    def hotspot_score(self) -> float:
        """Calculate overall hotspot score.

        Higher scores indicate more problematic areas.

        Returns:
            float: Hotspot score (0-100)
        """
        score = 0.0

        # Factor in change frequency (high frequency = hot)
        if self.change_frequency > 0:
            score += min(25, self.change_frequency * 10)

        # Factor in complexity (complex + changing = very hot)
        if self.complexity > 10:
            score += min(25, (self.complexity - 10) * 2)

        # Factor in bug fixes (many fixes = problematic)
        if self.commit_count > 0:
            bug_ratio = self.bug_fix_commits / self.commit_count
            score += min(20, bug_ratio * 40)

        # Factor in author count (many authors = coordination issues)
        if self.author_count > 5:
            score += min(15, (self.author_count - 5) * 3)

        # Factor in coupling (high coupling = risky changes)
        if self.coupling > 5:
            score += min(15, self.coupling)

        return min(100, score)

    @property
    def risk_level(self) -> str:
        """Determine risk level based on hotspot score.

        Returns:
            str: Risk level (critical, high, medium, low)
        """
        score = self.hotspot_score
        if score >= 70:
            return "critical"
        elif score >= 50:
            return "high"
        elif score >= 30:
            return "medium"
        else:
            return "low"

    @property
    def needs_attention(self) -> bool:
        """Check if this hotspot needs immediate attention.

        Returns:
            bool: True if attention needed
        """
        return self.risk_level in ["critical", "high"]


@dataclass
class FileHotspot:
    """Hotspot information for a single file.

    Tracks detailed information about why a file is considered a hotspot
    and what actions might be needed.

    Attributes:
        path: File path
        name: File name
        metrics: Hotspot metrics
        recent_commits: Recent commit history
        coupled_files: Files that frequently change together
        problem_indicators: Specific problems detected
        recommended_actions: Suggested actions to address issues
        last_modified: Last modification date
        created: Creation date
        size: File size in lines
        language: Programming language
    """

    path: str
    name: str
    metrics: HotspotMetrics = field(default_factory=HotspotMetrics)
    recent_commits: List[Dict[str, Any]] = field(default_factory=list)
    coupled_files: List[str] = field(default_factory=list)
    problem_indicators: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    last_modified: Optional[datetime] = None
    created: Optional[datetime] = None
    size: int = 0
    language: str = "unknown"

    @property
    def summary(self) -> str:
        """Generate summary of hotspot issues.

        Returns:
            str: Human-readable summary
        """
        issues = []

        if self.metrics.change_frequency > 0.5:
            issues.append(f"changes {self.metrics.change_frequency:.1f} times/day")

        if self.metrics.complexity > 20:
            issues.append(f"complexity {self.metrics.complexity:.0f}")

        if self.metrics.bug_fix_commits > 5:
            issues.append(f"{self.metrics.bug_fix_commits} bug fixes")

        if self.metrics.author_count > 10:
            issues.append(f"{self.metrics.author_count} authors")

        if issues:
            return f"{self.name}: " + ", ".join(issues)
        else:
            return f"{self.name}: stable"


@dataclass
class ModuleHotspot:
    """Hotspot information for a module/directory.

    Aggregates hotspot information at the module level to identify
    problematic areas of the codebase.

    Attributes:
        path: Module path
        name: Module name
        file_count: Number of files in module
        hotspot_files: List of hotspot files in module
        total_commits: Total commits to module
        total_authors: Total unique authors
        avg_complexity: Average complexity across files
        total_bugs: Total bug fixes in module
        stability_score: Module stability score
        cohesion: Module cohesion score
        coupling: Module coupling score
    """

    path: str
    name: str
    file_count: int = 0
    hotspot_files: List[FileHotspot] = field(default_factory=list)
    total_commits: int = 0
    total_authors: int = 0
    avg_complexity: float = 0.0
    total_bugs: int = 0
    stability_score: float = 100.0
    cohesion: float = 1.0
    coupling: float = 0.0

    @property
    def hotspot_density(self) -> float:
        """Calculate hotspot density in module.

        Returns:
            float: Ratio of hotspot files to total files
        """
        if self.file_count == 0:
            return 0.0
        return len(self.hotspot_files) / self.file_count

    @property
    def module_health(self) -> str:
        """Assess overall module health.

        Returns:
            str: Health status (healthy, warning, unhealthy)
        """
        if self.hotspot_density > 0.5 or self.stability_score < 40:
            return "unhealthy"
        elif self.hotspot_density > 0.3 or self.stability_score < 60:
            return "warning"
        else:
            return "healthy"


@dataclass
class HotspotReport:
    """Comprehensive hotspot analysis report.

    Provides detailed insights into code hotspots, including problematic
    files, modules, trends, and recommendations for improvement.

    Attributes:
        total_files_analyzed: Total files analyzed
        total_hotspots: Total hotspots detected
        critical_count: Number of critical hotspots
        high_count: Number of high-risk hotspots
        file_hotspots: List of file-level hotspots
        module_hotspots: List of module-level hotspots
        coupling_clusters: Groups of tightly coupled files
        temporal_patterns: Time-based patterns detected
        hotspot_trends: Trends in hotspot evolution
        top_problems: Most common problem types
        estimated_effort: Estimated effort to address hotspots
        recommendations: Actionable recommendations
        risk_matrix: Risk assessment matrix
    """

    total_files_analyzed: int = 0
    total_hotspots: int = 0
    critical_count: int = 0
    high_count: int = 0
    file_hotspots: List[FileHotspot] = field(default_factory=list)
    module_hotspots: List[ModuleHotspot] = field(default_factory=list)
    coupling_clusters: List[List[str]] = field(default_factory=list)
    temporal_patterns: Dict[str, Any] = field(default_factory=dict)
    hotspot_trends: Dict[str, Any] = field(default_factory=dict)
    top_problems: List[Tuple[str, int]] = field(default_factory=list)
    estimated_effort: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    risk_matrix: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "total_files_analyzed": self.total_files_analyzed,
            "total_hotspots": self.total_hotspots,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "hotspot_summary": [
                {
                    "path": h.path,
                    "score": h.metrics.hotspot_score,
                    "risk": h.metrics.risk_level,
                    "issues": h.problem_indicators,
                }
                for h in self.file_hotspots[:20]
            ],
            "module_summary": [
                {"path": m.path, "health": m.module_health, "hotspot_density": m.hotspot_density}
                for m in self.module_hotspots[:10]
            ],
            "top_problems": self.top_problems[:10],
            "estimated_effort_days": round(self.estimated_effort / 8, 1),
            "recommendations": self.recommendations,
        }

    @property
    def total_count(self) -> int:
        """Get total hotspot count.

        Returns:
            int: Total number of hotspots
        """
        return len(self.file_hotspots)

    @property
    def health_score(self) -> float:
        """Calculate overall codebase health score.

        Lower scores indicate more hotspots and problems.

        Returns:
            float: Health score (0-100)
        """
        if self.total_files_analyzed == 0:
            return 100.0

        # Start with perfect score
        score = 100.0

        # Deduct for hotspot ratio
        hotspot_ratio = self.total_hotspots / self.total_files_analyzed
        score -= min(40, hotspot_ratio * 100)

        # Deduct for critical hotspots
        score -= min(30, self.critical_count * 5)

        # Deduct for high-risk hotspots
        score -= min(20, self.high_count * 2)

        # Deduct for coupling clusters
        score -= min(10, len(self.coupling_clusters))

        return max(0, score)


class HotspotDetector:
    """Detector for code hotspots.

    Analyzes code repository to identify hotspots - areas that change
    frequently, have high complexity, or show other problematic patterns.

    Attributes:
        config: Configuration object
        logger: Logger instance
        git_analyzer: Git analyzer instance
    """

    def __init__(self, config: TenetsConfig):
        """Initialize hotspot detector.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.git_analyzer: Optional[GitAnalyzer] = None

    def detect(
        self,
        repo_path: Path,
        files: Optional[List[Any]] = None,
        since_days: int = 90,
        threshold: int = 10,
        include_stable: bool = False,
    ) -> HotspotReport:
        """Detect hotspots in a repository.

        Analyzes git history and code metrics to identify problematic
        areas that need attention or refactoring.

        Args:
            repo_path: Path to git repository
            files: Optional list of analyzed file objects
            since_days: Days of history to analyze
            threshold: Minimum score to consider as hotspot
            include_stable: Whether to include stable files in report

        Returns:
            HotspotReport: Comprehensive hotspot analysis

        Example:
            >>> detector = HotspotDetector(config)
            >>> report = detector.detect(Path("."), since_days=30)
            >>> print(f"Found {report.total_hotspots} hotspots")
        """
        self.logger.debug(f"Detecting hotspots in {repo_path}")

        # Initialize git analyzer
        self.git_analyzer = GitAnalyzer(repo_path)

        if not self.git_analyzer.is_repo():
            self.logger.warning(f"Not a git repository: {repo_path}")
            return HotspotReport()

        report = HotspotReport()

        # Get file change data
        since_date = datetime.now() - timedelta(days=since_days)
        file_changes = self._analyze_file_changes(since_date)

        # Analyze each file for hotspot indicators
        for file_path, change_data in file_changes.items():
            # Skip if not enough activity
            if not include_stable and change_data["commit_count"] < 2:
                continue

            report.total_files_analyzed += 1

            # Create hotspot analysis
            hotspot = self._analyze_file_hotspot(file_path, change_data, files, since_days)

            # Check if meets threshold
            if hotspot.metrics.hotspot_score >= threshold:
                report.file_hotspots.append(hotspot)
                report.total_hotspots += 1

                # Count by risk level
                if hotspot.metrics.risk_level == "critical":
                    report.critical_count += 1
                elif hotspot.metrics.risk_level == "high":
                    report.high_count += 1

        # Sort hotspots by score
        report.file_hotspots.sort(key=lambda h: h.metrics.hotspot_score, reverse=True)

        # Analyze module-level hotspots
        report.module_hotspots = self._analyze_module_hotspots(report.file_hotspots)

        # Detect coupling clusters
        report.coupling_clusters = self._detect_coupling_clusters(file_changes)

        # Analyze temporal patterns
        report.temporal_patterns = self._analyze_temporal_patterns(file_changes)

        # Identify top problems
        report.top_problems = self._identify_top_problems(report.file_hotspots)

        # Estimate effort
        report.estimated_effort = self._estimate_remediation_effort(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Build risk matrix
        report.risk_matrix = self._build_risk_matrix(report.file_hotspots)

        self.logger.debug(f"Hotspot detection complete: {report.total_hotspots} hotspots found")

        return report


# Backward-compatible functional API expected by tests
def detect_hotspots(
    repo_path: Path,
    files: Optional[List[Any]] = None,
    since_days: int = 90,
    threshold: int = 10,
    include_stable: bool = False,
    config: Optional[TenetsConfig] = None,
) -> HotspotReport:
    """Detect hotspots in a repository path.

    Thin wrapper that constructs a HotspotDetector and delegates to detect().

    Args:
        repo_path: Path to the repository
        files: Optional analyzed files list
        since_days: History window
        threshold: Minimum hotspot score
        include_stable: Include stable files
        config: Optional TenetsConfig

    Returns:
        HotspotReport
    """
    cfg = config or TenetsConfig()
    detector = HotspotDetector(cfg)
    return detector.detect(
        repo_path=repo_path,
        files=files,
        since_days=since_days,
        threshold=threshold,
        include_stable=include_stable,
    )

    def _analyze_file_changes(self, since_date: datetime) -> Dict[str, Dict[str, Any]]:
        """Analyze file change patterns from git history.

        Args:
            since_date: Start date for analysis

        Returns:
            Dict[str, Dict[str, Any]]: Change data by file path
        """
        file_changes = defaultdict(
            lambda: {
                "commit_count": 0,
                "authors": set(),
                "commits": [],
                "lines_added": 0,
                "lines_removed": 0,
                "bug_fixes": 0,
                "refactors": 0,
                "coupled_files": defaultdict(int),
                "first_commit": None,
                "last_commit": None,
            }
        )

        # Get commits since date
        commits = self.git_analyzer.get_commits_since(since_date)

        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)
            commit_message = commit.message.lower()

            # Detect bug fixes and refactors
            is_bug_fix = any(
                keyword in commit_message
                for keyword in ["fix", "bug", "issue", "error", "crash", "patch"]
            )
            is_refactor = any(
                keyword in commit_message
                for keyword in ["refactor", "cleanup", "reorganize", "restructure"]
            )

            # Get changed files
            changed_files = []
            if hasattr(commit, "stats") and hasattr(commit.stats, "files"):
                for file_path, stats in commit.stats.files.items():
                    changed_files.append(file_path)

                    # Update file change data
                    data = file_changes[file_path]
                    data["commit_count"] += 1
                    data["authors"].add(
                        commit.author.email if hasattr(commit, "author") else "unknown"
                    )
                    data["commits"].append(
                        {
                            "sha": commit.hexsha[:7],
                            "date": commit_date,
                            "message": commit.message[:100],
                            "author": getattr(commit.author, "name", "unknown"),
                        }
                    )
                    data["lines_added"] += stats.get("insertions", 0)
                    data["lines_removed"] += stats.get("deletions", 0)

                    if is_bug_fix:
                        data["bug_fixes"] += 1
                    if is_refactor:
                        data["refactors"] += 1

                    # Track dates
                    if not data["first_commit"] or commit_date < data["first_commit"]:
                        data["first_commit"] = commit_date
                    if not data["last_commit"] or commit_date > data["last_commit"]:
                        data["last_commit"] = commit_date

            # Track coupling (files that change together)
            for i, file1 in enumerate(changed_files):
                for file2 in changed_files[i + 1 :]:
                    file_changes[file1]["coupled_files"][file2] += 1
                    file_changes[file2]["coupled_files"][file1] += 1

        return dict(file_changes)

    def _analyze_file_hotspot(
        self,
        file_path: str,
        change_data: Dict[str, Any],
        analyzed_files: Optional[List[Any]],
        since_days: int,
    ) -> FileHotspot:
        """Analyze a single file for hotspot indicators.

        Args:
            file_path: Path to file
            change_data: Change history data
            analyzed_files: Optional list of analyzed file objects
            since_days: Days of history analyzed

        Returns:
            FileHotspot: Hotspot analysis for the file
        """
        hotspot = FileHotspot(
            path=file_path,
            name=Path(file_path).name,
            last_modified=change_data.get("last_commit"),
            created=change_data.get("first_commit"),
        )

        # Calculate basic metrics
        hotspot.metrics.commit_count = change_data["commit_count"]
        hotspot.metrics.author_count = len(change_data["authors"])
        hotspot.metrics.lines_changed = change_data["lines_added"] + change_data["lines_removed"]
        hotspot.metrics.bug_fix_commits = change_data["bug_fixes"]
        hotspot.metrics.refactor_commits = change_data["refactors"]

        # Calculate change frequency
        if since_days > 0:
            hotspot.metrics.change_frequency = change_data["commit_count"] / since_days

        # Calculate age and recency
        if change_data["first_commit"]:
            hotspot.metrics.age_days = (datetime.now() - change_data["first_commit"]).days
        if change_data["last_commit"]:
            hotspot.metrics.recency_days = (datetime.now() - change_data["last_commit"]).days

        # Calculate churn rate
        if hotspot.metrics.age_days > 0:
            hotspot.metrics.churn_rate = hotspot.metrics.lines_changed / hotspot.metrics.age_days

        # Get complexity from analyzed files if available
        if analyzed_files:
            for file_obj in analyzed_files:
                if hasattr(file_obj, "path") and file_obj.path == file_path:
                    if hasattr(file_obj, "complexity") and file_obj.complexity:
                        hotspot.metrics.complexity = getattr(file_obj.complexity, "cyclomatic", 0)
                    if hasattr(file_obj, "lines"):
                        hotspot.size = file_obj.lines
                    if hasattr(file_obj, "language"):
                        hotspot.language = file_obj.language
                    break

        # Count coupled files
        hotspot.metrics.coupling = len(change_data["coupled_files"])
        hotspot.coupled_files = [
            f
            for f, count in change_data["coupled_files"].items()
            if count >= 3  # Minimum coupling threshold
        ][:10]  # Top 10 coupled files

        # Add recent commits
        hotspot.recent_commits = sorted(
            change_data["commits"], key=lambda c: c["date"], reverse=True
        )[:10]

        # Identify problem indicators
        hotspot.problem_indicators = self._identify_problems(hotspot)

        # Generate recommendations
        hotspot.recommended_actions = self._recommend_actions(hotspot)

        # Calculate stability score
        hotspot.metrics.stability_score = self._calculate_stability(hotspot)

        # Estimate defect density
        if hotspot.size > 0:
            hotspot.metrics.defect_density = hotspot.metrics.bug_fix_commits / hotspot.size * 1000

        return hotspot

    def _identify_problems(self, hotspot: FileHotspot) -> List[str]:
        """Identify specific problems in a hotspot.

        Args:
            hotspot: File hotspot

        Returns:
            List[str]: Problem descriptions
        """
        problems = []

        # High change frequency
        if hotspot.metrics.change_frequency > 0.5:
            problems.append(
                f"Very high change frequency ({hotspot.metrics.change_frequency:.1f}/day)"
            )
        elif hotspot.metrics.change_frequency > 0.2:
            problems.append(f"High change frequency ({hotspot.metrics.change_frequency:.1f}/day)")

        # High complexity
        if hotspot.metrics.complexity > 20:
            problems.append(f"Very high complexity ({hotspot.metrics.complexity:.0f})")
        elif hotspot.metrics.complexity > 10:
            problems.append(f"High complexity ({hotspot.metrics.complexity:.0f})")

        # Many bug fixes
        if hotspot.metrics.bug_fix_commits > 10:
            problems.append(f"Frequent bug fixes ({hotspot.metrics.bug_fix_commits})")
        elif hotspot.metrics.bug_fix_commits > 5:
            problems.append(f"Several bug fixes ({hotspot.metrics.bug_fix_commits})")

        # Many authors (coordination issues)
        if hotspot.metrics.author_count > 10:
            problems.append(f"Many contributors ({hotspot.metrics.author_count})")

        # High coupling
        if hotspot.metrics.coupling > 10:
            problems.append(f"Highly coupled ({hotspot.metrics.coupling} files)")
        elif hotspot.metrics.coupling > 5:
            problems.append(f"Moderately coupled ({hotspot.metrics.coupling} files)")

        # High churn
        if hotspot.metrics.churn_rate > 10:
            problems.append("Very high code churn")
        elif hotspot.metrics.churn_rate > 5:
            problems.append("High code churn")

        # Recent instability
        if hotspot.metrics.recency_days < 7 and hotspot.metrics.commit_count > 5:
            problems.append("Recent instability")

        # Large file
        if hotspot.size > 1000:
            problems.append(f"Very large file ({hotspot.size} lines)")
        elif hotspot.size > 500:
            problems.append(f"Large file ({hotspot.size} lines)")

        return problems

    def _recommend_actions(self, hotspot: FileHotspot) -> List[str]:
        """Generate recommended actions for a hotspot.

        Args:
            hotspot: File hotspot

        Returns:
            List[str]: Recommended actions
        """
        actions = []

        # Complexity-based recommendations
        if hotspot.metrics.complexity > 20:
            actions.append("Refactor to reduce complexity (extract methods/classes)")
        elif hotspot.metrics.complexity > 10:
            actions.append("Consider simplifying complex logic")

        # Size-based recommendations
        if hotspot.size > 1000:
            actions.append("Split into smaller, more focused modules")
        elif hotspot.size > 500:
            actions.append("Consider breaking into smaller files")

        # Bug-based recommendations
        if hotspot.metrics.bug_fix_commits > 5:
            actions.append("Add comprehensive test coverage")
            actions.append("Perform thorough code review")

        # Coupling-based recommendations
        if hotspot.metrics.coupling > 10:
            actions.append("Reduce coupling through better abstraction")
        elif hotspot.metrics.coupling > 5:
            actions.append("Review dependencies and interfaces")

        # Author-based recommendations
        if hotspot.metrics.author_count > 10:
            actions.append("Establish clear ownership")
            actions.append("Improve documentation")

        # Churn-based recommendations
        if hotspot.metrics.churn_rate > 10:
            actions.append("Stabilize requirements before implementing")
        elif hotspot.metrics.churn_rate > 5:
            actions.append("Review design for stability")

        # Recent changes recommendations
        if hotspot.metrics.recency_days < 7:
            actions.append("Monitor closely for new issues")

        # General recommendations
        if not actions:
            if hotspot.metrics.hotspot_score > 50:
                actions.append("Schedule for refactoring")
            else:
                actions.append("Keep monitoring")

        return actions

    def _calculate_stability(self, hotspot: FileHotspot) -> float:
        """Calculate stability score for a file.

        Args:
            hotspot: File hotspot

        Returns:
            float: Stability score (0-100, higher is more stable)
        """
        score = 100.0

        # Penalize for frequent changes
        score -= min(30, hotspot.metrics.change_frequency * 30)

        # Penalize for bug fixes
        if hotspot.metrics.commit_count > 0:
            bug_ratio = hotspot.metrics.bug_fix_commits / hotspot.metrics.commit_count
            score -= min(25, bug_ratio * 50)

        # Penalize for many authors
        score -= min(20, max(0, hotspot.metrics.author_count - 3) * 4)

        # Penalize for high churn
        score -= min(15, hotspot.metrics.churn_rate * 1.5)

        # Bonus for recent stability
        if hotspot.metrics.recency_days > 30:
            score += 10

        return max(0, score)

    def _analyze_module_hotspots(self, file_hotspots: List[FileHotspot]) -> List[ModuleHotspot]:
        """Analyze hotspots at module/directory level.

        Args:
            file_hotspots: List of file hotspots

        Returns:
            List[ModuleHotspot]: Module-level hotspot analysis
        """
        module_map: Dict[str, ModuleHotspot] = {}

        for hotspot in file_hotspots:
            # Get module path (parent directory)
            module_path = str(Path(hotspot.path).parent)

            if module_path not in module_map:
                module_map[module_path] = ModuleHotspot(
                    path=module_path, name=Path(module_path).name or "root"
                )

            module = module_map[module_path]
            module.hotspot_files.append(hotspot)
            module.total_commits += hotspot.metrics.commit_count
            module.total_bugs += hotspot.metrics.bug_fix_commits

        # Calculate module metrics
        for module in module_map.values():
            if module.hotspot_files:
                # Average complexity
                complexities = [
                    h.metrics.complexity for h in module.hotspot_files if h.metrics.complexity > 0
                ]
                if complexities:
                    module.avg_complexity = sum(complexities) / len(complexities)

                # Unique authors
                authors = set()
                for hotspot in module.hotspot_files:
                    # This would need actual author data
                    pass

                # Stability score
                stabilities = [h.metrics.stability_score for h in module.hotspot_files]
                if stabilities:
                    module.stability_score = sum(stabilities) / len(stabilities)

        # Sort by hotspot density
        modules = list(module_map.values())
        modules.sort(key=lambda m: len(m.hotspot_files), reverse=True)

        return modules

    def _detect_coupling_clusters(self, file_changes: Dict[str, Dict[str, Any]]) -> List[List[str]]:
        """Detect clusters of tightly coupled files.

        Args:
            file_changes: File change data

        Returns:
            List[List[str]]: Coupling clusters
        """
        # Build coupling graph
        coupling_graph: Dict[str, Set[str]] = defaultdict(set)

        for file_path, change_data in file_changes.items():
            for coupled_file, count in change_data["coupled_files"].items():
                if count >= 5:  # Minimum coupling threshold
                    coupling_graph[file_path].add(coupled_file)
                    coupling_graph[coupled_file].add(file_path)

        # Find connected components (clusters)
        visited = set()
        clusters = []

        for file_path in coupling_graph:
            if file_path not in visited:
                cluster = self._find_cluster(file_path, coupling_graph, visited)
                if len(cluster) > 2:  # Minimum cluster size
                    clusters.append(sorted(cluster))

        # Sort by size
        clusters.sort(key=len, reverse=True)

        return clusters[:10]  # Top 10 clusters

    def _find_cluster(self, start: str, graph: Dict[str, Set[str]], visited: Set[str]) -> List[str]:
        """Find connected component in coupling graph.

        Args:
            start: Starting node
            graph: Coupling graph
            visited: Set of visited nodes

        Returns:
            List[str]: Cluster of connected files
        """
        cluster = []
        stack = [start]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                cluster.append(node)
                stack.extend(graph[node] - visited)

        return cluster

    def _analyze_temporal_patterns(self, file_changes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in changes.

        Args:
            file_changes: File change data

        Returns:
            Dict[str, Any]: Temporal pattern analysis
        """
        patterns = {
            "burst_changes": [],  # Files with burst activity
            "periodic_changes": [],  # Files with periodic patterns
            "declining_activity": [],  # Files with declining changes
            "increasing_activity": [],  # Files with increasing changes
        }

        for file_path, change_data in file_changes.items():
            if len(change_data["commits"]) < 5:
                continue

            # Analyze commit timeline
            commits = sorted(change_data["commits"], key=lambda c: c["date"])

            # Check for burst activity (many commits in short time)
            for i in range(len(commits) - 3):
                window = commits[i : i + 4]
                time_span = (window[-1]["date"] - window[0]["date"]).days
                if time_span <= 2:  # 4 commits in 2 days
                    patterns["burst_changes"].append(
                        {"file": file_path, "date": window[0]["date"], "commits": 4}
                    )
                    break

            # Check for trends (simplified)
            if len(commits) >= 10:
                first_half = commits[: len(commits) // 2]
                second_half = commits[len(commits) // 2 :]

                first_rate = len(first_half) / max(
                    1, (first_half[-1]["date"] - first_half[0]["date"]).days
                )
                second_rate = len(second_half) / max(
                    1, (second_half[-1]["date"] - second_half[0]["date"]).days
                )

                if second_rate > first_rate * 1.5:
                    patterns["increasing_activity"].append(file_path)
                elif second_rate < first_rate * 0.5:
                    patterns["declining_activity"].append(file_path)

        return patterns

    def _identify_top_problems(self, file_hotspots: List[FileHotspot]) -> List[Tuple[str, int]]:
        """Identify most common problems across hotspots.

        Args:
            file_hotspots: List of file hotspots

        Returns:
            List[Tuple[str, int]]: Problem types and counts
        """
        problem_counts: Dict[str, int] = defaultdict(int)

        for hotspot in file_hotspots:
            for problem in hotspot.problem_indicators:
                # Extract problem type
                if "complexity" in problem.lower():
                    problem_counts["High Complexity"] += 1
                elif "change frequency" in problem.lower():
                    problem_counts["Frequent Changes"] += 1
                elif "bug" in problem.lower():
                    problem_counts["Bug Prone"] += 1
                elif "coupled" in problem.lower():
                    problem_counts["High Coupling"] += 1
                elif "contributor" in problem.lower():
                    problem_counts["Many Contributors"] += 1
                elif "churn" in problem.lower():
                    problem_counts["High Churn"] += 1
                elif "large file" in problem.lower():
                    problem_counts["Large Files"] += 1
                else:
                    problem_counts["Other"] += 1

        return sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)

    def _estimate_remediation_effort(self, report: HotspotReport) -> float:
        """Estimate effort to address hotspots.

        Args:
            report: Hotspot report

        Returns:
            float: Estimated hours
        """
        total_hours = 0.0

        for hotspot in report.file_hotspots:
            # Base estimate on risk level and size
            if hotspot.metrics.risk_level == "critical":
                base_hours = 16
            elif hotspot.metrics.risk_level == "high":
                base_hours = 8
            elif hotspot.metrics.risk_level == "medium":
                base_hours = 4
            else:
                base_hours = 2

            # Adjust for file size
            if hotspot.size > 1000:
                base_hours *= 2
            elif hotspot.size > 500:
                base_hours *= 1.5

            # Adjust for complexity
            if hotspot.metrics.complexity > 20:
                base_hours *= 1.5

            # Adjust for coupling
            if hotspot.metrics.coupling > 10:
                base_hours *= 1.3

            total_hours += base_hours

        # Add time for testing and review
        total_hours *= 1.5

        return total_hours

    def _generate_recommendations(self, report: HotspotReport) -> List[str]:
        """Generate recommendations based on hotspot analysis.

        Args:
            report: Hotspot report

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        # Critical hotspots
        if report.critical_count > 0:
            recommendations.append(
                f"URGENT: Address {report.critical_count} critical hotspots immediately. "
                "These files have severe issues that impact stability."
            )

        # High-risk hotspots
        if report.high_count > 5:
            recommendations.append(
                f"Schedule refactoring for {report.high_count} high-risk files. "
                "Consider dedicating a sprint to technical debt reduction."
            )

        # Coupling clusters
        if len(report.coupling_clusters) > 3:
            recommendations.append(
                f"Found {len(report.coupling_clusters)} coupling clusters. "
                "Review architecture to reduce interdependencies."
            )

        # Module health
        unhealthy_modules = [m for m in report.module_hotspots if m.module_health == "unhealthy"]
        if unhealthy_modules:
            recommendations.append(
                f"{len(unhealthy_modules)} modules are unhealthy. "
                "Consider module-level refactoring or splitting."
            )

        # Common problems
        if report.top_problems:
            top_problem = report.top_problems[0]
            if top_problem[0] == "High Complexity":
                recommendations.append(
                    "Complexity is the main issue. Focus on simplifying logic "
                    "and extracting methods/classes."
                )
            elif top_problem[0] == "Frequent Changes":
                recommendations.append(
                    "Files change too frequently. Stabilize requirements and improve abstractions."
                )
            elif top_problem[0] == "Bug Prone":
                recommendations.append(
                    "Many bugs detected. Increase test coverage and implement stricter code review."
                )

        # Temporal patterns
        if report.temporal_patterns.get("burst_changes"):
            recommendations.append(
                "Detected burst change patterns. Avoid rushed implementations "
                "and allow time for proper design."
            )

        # Effort estimate
        if report.estimated_effort > 160:  # More than 4 weeks
            recommendations.append(
                f"Estimated {report.estimated_effort / 40:.1f} person-weeks to address all hotspots. "
                "Consider a dedicated tech debt reduction initiative."
            )

        # General health
        if report.health_score < 40:
            recommendations.append(
                "Overall codebase health is poor. Implement: "
                "1) Complexity limits in CI/CD, "
                "2) Mandatory code review, "
                "3) Regular refactoring sessions."
            )

        return recommendations

    def _build_risk_matrix(self, file_hotspots: List[FileHotspot]) -> Dict[str, List[str]]:
        """Build risk assessment matrix.

        Args:
            file_hotspots: List of file hotspots

        Returns:
            Dict[str, List[str]]: Risk matrix by category
        """
        matrix = {"critical": [], "high": [], "medium": [], "low": []}

        for hotspot in file_hotspots:
            risk_level = hotspot.metrics.risk_level
            matrix[risk_level].append(hotspot.path)

        # Limit each category
        for level in matrix:
            matrix[level] = matrix[level][:20]

        return matrix


# Backward-compatible functional API expected by tests
def detect_hotspots(
    repo_path: Path,
    files: Optional[List[Any]] = None,
    threshold: int = 5,
    config: Optional[TenetsConfig] = None,
) -> HotspotReport:
    """Detect hotspots via a simple functional wrapper.

    Args:
        repo_path: Path to repository
        files: Optional analyzed files
        threshold: Minimum hotspot score to include
        config: Optional TenetsConfig

    Returns:
        HotspotReport
    """
    detector = HotspotDetector(config or TenetsConfig())
    return detector.detect(repo_path, files=files, threshold=threshold)
