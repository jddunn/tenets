"""Git blame analysis module.

This module provides functionality for analyzing line-by-line authorship
of files using git blame. It helps understand who wrote what code, when
changes were made, and how code ownership is distributed within files.

The blame analyzer provides detailed insights into code authorship patterns,
helping identify knowledge owners and understanding code evolution.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


@dataclass
class BlameLine:
    """Information for a single line from git blame.

    Represents authorship information for a specific line of code,
    including who wrote it, when, and in which commit.

    Attributes:
        line_number: Line number in file
        content: Content of the line
        author: Author name
        author_email: Author email
        commit_sha: Commit SHA that introduced this line
        commit_date: Date when line was introduced
        commit_message: Commit message (first line)
        is_original: Whether this is from the original commit
        age_days: Age of the line in days
        previous_authors: List of previous authors if line was modified
    """

    line_number: int
    content: str
    author: str
    author_email: str
    commit_sha: str
    commit_date: datetime
    commit_message: str
    is_original: bool = False
    age_days: int = 0
    previous_authors: List[str] = field(default_factory=list)

    @property
    def is_recent(self) -> bool:
        """Check if line was recently modified.

        Returns:
            bool: True if modified within last 30 days
        """
        return self.age_days <= 30

    @property
    def is_old(self) -> bool:
        """Check if line is old.

        Returns:
            bool: True if older than 180 days
        """
        return self.age_days > 180

    @property
    def is_documentation(self) -> bool:
        """Check if line appears to be documentation.

        Returns:
            bool: True if line looks like documentation
        """
        stripped = self.content.strip()
        return (
            stripped.startswith("#")
            or stripped.startswith("//")
            or stripped.startswith("/*")
            or stripped.startswith("*")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
            or "TODO" in stripped
            or "FIXME" in stripped
            or "NOTE" in stripped
        )

    @property
    def is_empty(self) -> bool:
        """Check if line is empty or whitespace only.

        Returns:
            bool: True if empty or whitespace
        """
        return len(self.content.strip()) == 0


@dataclass
class FileBlame:
    """Blame information for an entire file.

    Aggregates line-by-line blame information to provide file-level
    authorship insights and ownership patterns.

    Attributes:
        file_path: Path to the file
        total_lines: Total number of lines
        blame_lines: List of blame information per line
        authors: Set of unique authors
        author_stats: Statistics per author
        commit_shas: Set of unique commits
        oldest_line: Oldest line in file
        newest_line: Newest line in file
        age_distribution: Distribution of line ages
        ownership_map: Line ranges owned by each author
        hot_spots: Lines that changed frequently
    """

    file_path: str
    total_lines: int = 0
    blame_lines: List[BlameLine] = field(default_factory=list)
    authors: Set[str] = field(default_factory=set)
    author_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    commit_shas: Set[str] = field(default_factory=set)
    oldest_line: Optional[BlameLine] = None
    newest_line: Optional[BlameLine] = None
    age_distribution: Dict[str, int] = field(default_factory=dict)
    ownership_map: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    hot_spots: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def primary_author(self) -> Optional[str]:
        """Get the primary author of the file.

        Returns:
            Optional[str]: Author with most lines or None
        """
        if not self.author_stats:
            return None

        return max(self.author_stats.items(), key=lambda x: x[1].get("lines", 0))[0]

    @property
    def author_diversity(self) -> float:
        """Calculate author diversity score.

        Higher scores indicate more distributed authorship.

        Returns:
            float: Diversity score (0-1)
        """
        if not self.author_stats or self.total_lines == 0:
            return 0.0

        # Calculate Shannon entropy
        import math

        entropy = 0.0

        for stats in self.author_stats.values():
            proportion = stats.get("lines", 0) / self.total_lines
            if proportion > 0:
                entropy -= proportion * math.log(proportion)

        # Normalize by maximum possible entropy
        max_entropy = math.log(len(self.author_stats)) if len(self.author_stats) > 1 else 1

        return entropy / max_entropy if max_entropy > 0 else 0.0

    @property
    def average_age_days(self) -> float:
        """Calculate average age of lines in days.

        Returns:
            float: Average age in days
        """
        if not self.blame_lines:
            return 0.0

        total_age = sum(line.age_days for line in self.blame_lines)
        return total_age / len(self.blame_lines)

    @property
    def freshness_score(self) -> float:
        """Calculate code freshness score.

        Higher scores indicate more recently modified code.

        Returns:
            float: Freshness score (0-100)
        """
        if not self.blame_lines:
            return 0.0

        recent_lines = sum(1 for line in self.blame_lines if line.is_recent)
        return (recent_lines / len(self.blame_lines)) * 100


@dataclass
class BlameReport:
    """Comprehensive blame analysis report.

    Provides detailed authorship analysis across multiple files,
    identifying ownership patterns, knowledge distribution, and
    collaboration insights.

    Attributes:
        files_analyzed: Number of files analyzed
        total_lines: Total lines analyzed
        total_authors: Total unique authors
        file_blames: Blame data for each file
        author_summary: Summary statistics per author
        ownership_distribution: How ownership is distributed
        collaboration_matrix: Who modified whose code
        knowledge_map: Knowledge areas per author
        recommendations: Actionable recommendations
        hot_files: Files with most contributors
        single_author_files: Files with only one author
        abandoned_code: Code from inactive authors
    """

    files_analyzed: int = 0
    total_lines: int = 0
    total_authors: int = 0
    file_blames: Dict[str, FileBlame] = field(default_factory=dict)
    author_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ownership_distribution: Dict[str, float] = field(default_factory=dict)
    collaboration_matrix: Dict[Tuple[str, str], int] = field(default_factory=dict)
    knowledge_map: Dict[str, Set[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    hot_files: List[Dict[str, Any]] = field(default_factory=list)
    single_author_files: List[str] = field(default_factory=list)
    abandoned_code: Dict[str, int] = field(default_factory=dict)
    # Allow tests to override computed bus_factor via setter
    _bus_factor_override: Optional[int] = field(default=None, repr=False)
    # Allow tests to override computed collaboration score via setter
    _collab_score_override: Optional[float] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "summary": {
                "files_analyzed": self.files_analyzed,
                "total_lines": self.total_lines,
                "total_authors": self.total_authors,
            },
            "top_authors": sorted(
                self.author_summary.items(),
                key=lambda x: x[1].get("total_lines", 0),
                reverse=True,
            )[:10],
            "ownership_distribution": self.ownership_distribution,
            "hot_files": self.hot_files[:10],
            "single_author_files": len(self.single_author_files),
            "abandoned_lines": sum(self.abandoned_code.values()),
            "recommendations": self.recommendations,
        }

    @property
    def bus_factor(self) -> int:
        """Calculate bus factor based on blame data.

        Returns:
            int: Bus factor (number of critical authors)
        """
        if self._bus_factor_override is not None:
            return int(self._bus_factor_override)
        if not self.author_summary or self.total_lines == 0:
            return 0

        # Authors who own >10% of code are critical
        critical_authors = sum(
            1
            for stats in self.author_summary.values()
            if stats.get("total_lines", 0) / self.total_lines > 0.1
        )

        return max(1, critical_authors)

    @bus_factor.setter
    def bus_factor(self, value: int) -> None:
        """Allow overriding the computed bus factor (used in tests)."""
        try:
            self._bus_factor_override = int(value)
        except Exception:
            self._bus_factor_override = None

    @property
    def collaboration_score(self) -> float:
        """Calculate collaboration score.

        Higher scores indicate more collaborative development.

        Returns:
            float: Collaboration score (0-100)
        """
        # Allow test overrides
        if self._collab_score_override is not None:
            try:
                return float(self._collab_score_override)
            except Exception:
                pass

        if self.files_analyzed == 0:
            return 0.0

        # Files with multiple authors indicate collaboration
        multi_author_files = sum(1 for blame in self.file_blames.values() if len(blame.authors) > 1)

        return (multi_author_files / self.files_analyzed) * 100

    @collaboration_score.setter
    def collaboration_score(self, value: float) -> None:
        """Allow overriding the computed collaboration score (used in tests)."""
        try:
            self._collab_score_override = float(value)
        except Exception:
            self._collab_score_override = None


class BlameAnalyzer:
    """Analyzer for git blame operations.

    Provides line-by-line authorship analysis using git blame,
    helping understand code ownership and evolution patterns.

    Attributes:
        config: Configuration object
        logger: Logger instance
        _blame_cache: Cache for blame results
    """

    def __init__(self, config: TenetsConfig):
        """Initialize blame analyzer.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._blame_cache: Dict[str, FileBlame] = {}

    def analyze_file(
        self,
        repo_path: Path,
        file_path: str,
        ignore_whitespace: bool = True,
        follow_renames: bool = True,
    ) -> FileBlame:
        """Analyze blame for a single file.

        Performs git blame analysis on a file to understand
        line-by-line authorship.

        Args:
            repo_path: Path to git repository
            file_path: Path to file relative to repo root
            ignore_whitespace: Ignore whitespace changes
            follow_renames: Follow file renames

        Returns:
            FileBlame: Blame analysis for the file

        Example:
            >>> analyzer = BlameAnalyzer(config)
            >>> blame = analyzer.analyze_file(Path("."), "src/main.py")
            >>> print(f"Primary author: {blame.primary_author}")
        """
        import subprocess

        self.logger.debug(f"Analyzing blame for {file_path}")

        # Check cache
        cache_key = f"{repo_path}/{file_path}"
        if cache_key in self._blame_cache:
            return self._blame_cache[cache_key]

        file_blame = FileBlame(file_path=file_path)

        # Build git blame command
        cmd = ["git", "blame", "--line-porcelain"]

        if ignore_whitespace:
            cmd.append("-w")

        if follow_renames:
            cmd.append("-C")

        cmd.append(file_path)

        try:
            # Run git blame
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, check=True)

            # Parse blame output
            self._parse_blame_output(result.stdout, file_blame)

            # Calculate statistics
            self._calculate_file_stats(file_blame)

            # Cache result
            self._blame_cache[cache_key] = file_blame

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git blame failed for {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error analyzing blame for {file_path}: {e}")

        return file_blame

    def analyze_directory(
        self,
        repo_path: Path,
        directory: str = ".",
        file_pattern: str = "*",
        recursive: bool = True,
        max_files: int = 100,
    ) -> BlameReport:
        """Analyze blame for all files in a directory.

        Performs comprehensive blame analysis across multiple files
        to understand ownership patterns.

        Args:
            repo_path: Path to git repository
            directory: Directory to analyze
            file_pattern: File pattern to match
            recursive: Whether to recurse into subdirectories
            max_files: Maximum files to analyze

        Returns:
            BlameReport: Comprehensive blame analysis

        Example:
            >>> analyzer = BlameAnalyzer(config)
            >>> report = analyzer.analyze_directory(
            ...     Path("."),
            ...     directory="src",
            ...     file_pattern="*.py"
            ... )
            >>> print(f"Bus factor: {report.bus_factor}")
        """
        from pathlib import Path as PathLib

        self.logger.debug(f"Analyzing blame for directory {directory}")

        report = BlameReport()

        # Get list of files
        if recursive:
            pattern = f"**/{file_pattern}"
        else:
            pattern = file_pattern

        target_dir = PathLib(repo_path) / directory
        files = list(target_dir.glob(pattern))

        # Filter to only files (not directories)
        files = [f for f in files if f.is_file()]

        # Limit number of files
        if len(files) > max_files:
            self.logger.info(f"Limiting analysis to {max_files} files")
            files = files[:max_files]

        # Analyze each file
        for file_path in files:
            # Get relative path
            try:
                rel_path = file_path.relative_to(repo_path)
            except ValueError:
                continue

            # Skip binary files and common non-source files
            if self._should_skip_file(str(rel_path)):
                continue

            # Analyze file
            file_blame = self.analyze_file(repo_path, str(rel_path))

            if file_blame.total_lines > 0:
                report.file_blames[str(rel_path)] = file_blame
                report.files_analyzed += 1

        # Calculate report statistics
        self._calculate_report_stats(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        self.logger.debug(
            f"Blame analysis complete: {report.files_analyzed} files, "
            f"{report.total_authors} authors"
        )

        return report

    def get_line_history(
        self, repo_path: Path, file_path: str, line_number: int, max_depth: int = 10
    ) -> List[Dict[str, Any]]:
        """Get history of changes for a specific line.

        Traces the evolution of a specific line through git history.

        Args:
            repo_path: Path to git repository
            file_path: Path to file
            line_number: Line number to trace
            max_depth: Maximum history depth to retrieve

        Returns:
            List[Dict[str, Any]]: History of line changes

        Example:
            >>> analyzer = BlameAnalyzer(config)
            >>> history = analyzer.get_line_history(
            ...     Path("."),
            ...     "src/main.py",
            ...     42
            ... )
            >>> for change in history:
            ...     print(f"{change['date']}: {change['author']}")
        """
        import subprocess

        history = []
        current_line = line_number
        current_file = file_path

        for depth in range(max_depth):
            try:
                # Get blame for current line
                cmd = [
                    "git",
                    "blame",
                    "-L",
                    f"{current_line},{current_line}",
                    "--line-porcelain",
                    current_file,
                ]

                result = subprocess.run(
                    cmd, cwd=repo_path, capture_output=True, text=True, check=True
                )

                # Parse blame output
                blame_data = self._parse_single_blame(result.stdout)

                if not blame_data:
                    break

                history.append(blame_data)

                # Get previous version
                if blame_data["commit"] == "0000000000000000000000000000000000000000":
                    break  # Uncommitted changes

                # Find line in parent commit
                parent_cmd = ["git", "show", f"{blame_data['commit']}^:{current_file}"]

                try:
                    subprocess.run(
                        parent_cmd, cwd=repo_path, capture_output=True, text=True, check=True
                    )
                    # Continue with parent
                    # This is simplified - real implementation would track line movement
                except subprocess.CalledProcessError:
                    break  # File didn't exist in parent

            except subprocess.CalledProcessError:
                break
            except Exception as e:
                self.logger.error(f"Error getting line history: {e}")
                break

        return history

    def _parse_blame_output(self, output: str, file_blame: FileBlame) -> None:
        """Parse git blame porcelain output.

        Args:
            output: Git blame output
            file_blame: FileBlame object to populate
        """
        lines = output.strip().split("\n")
        current_commit = {}
        line_number = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Parse commit header
            if line and not line.startswith("\t"):
                parts = line.split(" ")
                if len(parts) >= 3 and len(parts[0]) == 40:  # SHA line
                    current_commit = {
                        "sha": parts[0],
                        "original_line": int(parts[1]) if len(parts) > 1 else 0,
                        "final_line": int(parts[2]) if len(parts) > 2 else 0,
                    }
                    line_number = current_commit["final_line"]
                elif line.startswith("author "):
                    current_commit["author"] = line[7:]
                elif line.startswith("author-mail "):
                    current_commit["author_email"] = line[12:].strip("<>")
                elif line.startswith("author-time "):
                    timestamp = int(line[12:])
                    current_commit["date"] = datetime.fromtimestamp(timestamp)
                elif line.startswith("summary "):
                    current_commit["message"] = line[8:]
            elif line.startswith("\t"):
                # This is the actual line content
                content = line[1:]  # Remove tab

                if current_commit:
                    blame_line = BlameLine(
                        line_number=line_number,
                        content=content,
                        author=current_commit.get("author", "Unknown"),
                        author_email=current_commit.get("author_email", ""),
                        commit_sha=current_commit.get("sha", "")[:7],
                        commit_date=current_commit.get("date", datetime.now()),
                        commit_message=current_commit.get("message", ""),
                        age_days=(datetime.now() - current_commit.get("date", datetime.now())).days,
                    )

                    file_blame.blame_lines.append(blame_line)
                    file_blame.authors.add(blame_line.author)
                    file_blame.commit_shas.add(blame_line.commit_sha)

                    # Reset for next line
                    current_commit = {}

            i += 1

        file_blame.total_lines = len(file_blame.blame_lines)

    def _parse_single_blame(self, output: str) -> Dict[str, Any]:
        """Parse single line blame output.

        Args:
            output: Git blame output for single line

        Returns:
            Dict[str, Any]: Parsed blame data
        """
        lines = output.strip().split("\n")
        blame_data = {}

        for line in lines:
            if line and not line.startswith("\t"):
                if len(line) >= 40 and line[:40].replace("0", "").replace("a-f", ""):
                    parts = line.split(" ")
                    blame_data["commit"] = parts[0]
                elif line.startswith("author "):
                    blame_data["author"] = line[7:]
                elif line.startswith("author-time "):
                    timestamp = int(line[12:])
                    blame_data["date"] = datetime.fromtimestamp(timestamp)
                elif line.startswith("summary "):
                    blame_data["message"] = line[8:]
            elif line.startswith("\t"):
                blame_data["content"] = line[1:]

        return blame_data

    def _calculate_file_stats(self, file_blame: FileBlame) -> None:
        """Calculate statistics for file blame.

        Args:
            file_blame: FileBlame object to calculate stats for
        """
        if not file_blame.blame_lines:
            return

        # Find oldest and newest lines
        file_blame.oldest_line = min(file_blame.blame_lines, key=lambda l: l.commit_date)
        file_blame.newest_line = max(file_blame.blame_lines, key=lambda l: l.commit_date)

        # Calculate author statistics
        for line in file_blame.blame_lines:
            if line.author not in file_blame.author_stats:
                file_blame.author_stats[line.author] = {
                    "lines": 0,
                    "commits": set(),
                    "first_contribution": line.commit_date,
                    "last_contribution": line.commit_date,
                    "code_lines": 0,
                    "doc_lines": 0,
                }

            stats = file_blame.author_stats[line.author]
            stats["lines"] += 1
            stats["commits"].add(line.commit_sha)

            stats["first_contribution"] = min(stats["first_contribution"], line.commit_date)
            stats["last_contribution"] = max(stats["last_contribution"], line.commit_date)

            if line.is_documentation:
                stats["doc_lines"] += 1
            elif not line.is_empty:
                stats["code_lines"] += 1

        # Convert commit sets to counts
        for stats in file_blame.author_stats.values():
            stats["commit_count"] = len(stats["commits"])
            del stats["commits"]  # Remove set for serialization

        # Calculate age distribution
        age_buckets = {
            "recent": 0,  # < 30 days
            "moderate": 0,  # 30-90 days
            "old": 0,  # 90-180 days
            "ancient": 0,  # > 180 days
        }

        for line in file_blame.blame_lines:
            if line.age_days < 30:
                age_buckets["recent"] += 1
            elif line.age_days < 90:
                age_buckets["moderate"] += 1
            elif line.age_days < 180:
                age_buckets["old"] += 1
            else:
                age_buckets["ancient"] += 1

        file_blame.age_distribution = age_buckets

        # Build ownership map (continuous line ranges per author)
        current_author = None
        start_line = 0

        for i, line in enumerate(file_blame.blame_lines):
            if line.author != current_author:
                if current_author:
                    # Save previous range
                    if current_author not in file_blame.ownership_map:
                        file_blame.ownership_map[current_author] = []
                    file_blame.ownership_map[current_author].append((start_line, i - 1))

                current_author = line.author
                start_line = i

        # Save last range
        if current_author:
            if current_author not in file_blame.ownership_map:
                file_blame.ownership_map[current_author] = []
            file_blame.ownership_map[current_author].append(
                (start_line, len(file_blame.blame_lines) - 1)
            )

    def _calculate_report_stats(self, report: BlameReport) -> None:
        """Calculate statistics for blame report.

        Args:
            report: BlameReport to calculate stats for
        """
        # Aggregate author statistics
        author_totals = defaultdict(
            lambda: {
                "total_lines": 0,
                "total_files": 0,
                "code_lines": 0,
                "doc_lines": 0,
                "files": set(),
            }
        )

        for file_path, file_blame in report.file_blames.items():
            report.total_lines += file_blame.total_lines

            for author, stats in file_blame.author_stats.items():
                author_totals[author]["total_lines"] += stats["lines"]
                author_totals[author]["code_lines"] += stats.get("code_lines", 0)
                author_totals[author]["doc_lines"] += stats.get("doc_lines", 0)
                author_totals[author]["files"].add(file_path)

            # Identify single-author files
            if len(file_blame.authors) == 1:
                report.single_author_files.append(file_path)

            # Identify hot files (many contributors)
            if len(file_blame.authors) > 5:
                report.hot_files.append(
                    {
                        "file": file_path,
                        "authors": len(file_blame.authors),
                        "primary_author": file_blame.primary_author,
                        "diversity": file_blame.author_diversity,
                    }
                )

        # Convert sets to counts
        for author, totals in author_totals.items():
            totals["total_files"] = len(totals["files"])
            del totals["files"]

        report.author_summary = dict(author_totals)
        report.total_authors = len(report.author_summary)

        # Sort hot files by author count
        report.hot_files.sort(key=lambda x: x["authors"], reverse=True)

        # Calculate ownership distribution
        if report.total_lines > 0:
            for author, stats in report.author_summary.items():
                report.ownership_distribution[author] = (
                    stats["total_lines"] / report.total_lines * 100
                )

        # Identify abandoned code (from inactive authors)
        # This is a simplified check - in reality would check against active contributors
        for author, stats in report.author_summary.items():
            # Simple heuristic: authors with < 10 lines might be inactive
            if stats["total_lines"] < 10:
                report.abandoned_code[author] = stats["total_lines"]

    def _generate_recommendations(self, report: BlameReport) -> List[str]:
        """Generate recommendations based on blame analysis.

        Args:
            report: BlameReport to analyze

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        # Bus factor recommendations
        if report.bus_factor <= 1:
            recommendations.append(
                "Critical: Bus factor is 1. Single person owns majority of code. "
                "Urgent knowledge transfer needed."
            )
        elif report.bus_factor <= 2:
            recommendations.append(
                "Low bus factor detected. Increase code review participation "
                "and pair programming to spread knowledge."
            )

        # Single-author file recommendations
        if len(report.single_author_files) > report.files_analyzed * 0.3:
            recommendations.append(
                f"{len(report.single_author_files)} files have single authors. "
                "Implement mandatory code review to increase knowledge sharing."
            )

        # Collaboration recommendations
        if report.collaboration_score < 50:
            recommendations.append(
                "Low collaboration detected. Most files have single authors. "
                "Consider mob programming sessions and team rotations."
            )

        # Hot file recommendations
        if report.hot_files:
            top_hot = report.hot_files[0]
            recommendations.append(
                f"File '{top_hot['file']}' has {top_hot['authors']} authors. "
                "Consider refactoring to reduce coupling and clarify ownership."
            )

        # Abandoned code recommendations
        if report.abandoned_code:
            total_abandoned = sum(report.abandoned_code.values())
            recommendations.append(
                f"{total_abandoned} lines from potentially inactive authors. "
                "Review and reassign ownership or remove unused code."
            )

        # Ownership concentration recommendations
        if report.ownership_distribution:
            top_owner = max(report.ownership_distribution.items(), key=lambda x: x[1])
            if top_owner[1] > 50:
                recommendations.append(
                    f"{top_owner[0]} owns {top_owner[1]:.1f}% of code. "
                    "Distribute ownership more evenly across team."
                )

        return recommendations

    def _should_skip_file(self, file_path: str) -> bool:
        """Check if file should be skipped.

        Args:
            file_path: File path to check

        Returns:
            bool: True if file should be skipped
        """
        skip_patterns = [
            ".git/",
            "__pycache__",
            ".pyc",
            ".pyo",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".bin",
            ".jar",
            ".class",
            "node_modules/",
            "vendor/",
            ".min.js",
            ".min.css",
            ".map",
            ".lock",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            ".sum",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
        ]

        file_path_lower = file_path.lower()

        for pattern in skip_patterns:
            if pattern in file_path_lower:
                return True

        return False


def analyze_blame(
    repo_path: Path, target: str = ".", config: Optional[TenetsConfig] = None, **kwargs
) -> BlameReport:
    """Convenience function to analyze blame.

    Args:
        repo_path: Path to repository
        target: File or directory to analyze
        config: Optional configuration
        **kwargs: Additional arguments

    Returns:
        BlameReport: Blame analysis report

    Example:
        >>> from tenets.core.git.blame import analyze_blame
        >>> report = analyze_blame(Path("."), target="src/")
        >>> print(f"Bus factor: {report.bus_factor}")
    """
    if config is None:
        config = TenetsConfig()

    analyzer = BlameAnalyzer(config)

    target_path = Path(repo_path) / target

    if target_path.is_file():
        # Single file analysis
        file_blame = analyzer.analyze_file(repo_path, target)
        report = BlameReport(files_analyzed=1)
        report.file_blames[target] = file_blame
        analyzer._calculate_report_stats(report)
        report.recommendations = analyzer._generate_recommendations(report)
        return report
    else:
        # Directory analysis
        return analyzer.analyze_directory(repo_path, target, **kwargs)
