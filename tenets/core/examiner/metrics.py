"""Metrics calculation module for code analysis.

This module provides comprehensive metrics calculation for codebases,
including size metrics, complexity aggregations, code quality indicators,
and statistical analysis across files and languages.

The MetricsCalculator class processes analyzed files to extract quantitative
measurements that help assess code health, maintainability, and quality.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


@dataclass
class MetricsReport:
    """Comprehensive metrics report for analyzed code.

    Aggregates various code metrics to provide quantitative insights into
    codebase characteristics, including size, complexity, documentation,
    and quality indicators.

    Attributes:
        total_files: Total number of files analyzed
        total_lines: Total lines of code across all files
        total_blank_lines: Total blank lines
        total_comment_lines: Total comment lines
        total_code_lines: Total actual code lines (excluding blanks/comments)
        total_functions: Total number of functions/methods
        total_classes: Total number of classes
        total_imports: Total number of import statements
        avg_file_size: Average file size in lines
        avg_complexity: Average cyclomatic complexity
        max_complexity: Maximum cyclomatic complexity found
        min_complexity: Minimum cyclomatic complexity found
        complexity_std_dev: Standard deviation of complexity
        documentation_ratio: Ratio of comment lines to code lines
        test_coverage: Estimated test coverage (if test files found)
        languages: Dictionary of language-specific metrics
        file_types: Distribution of file types
        size_distribution: File size distribution buckets
        complexity_distribution: Complexity distribution buckets
        largest_files: List of largest files by line count
        most_complex_files: List of files with highest complexity
        most_imported_modules: Most frequently imported modules
        code_duplication_ratio: Estimated code duplication ratio
        technical_debt_score: Calculated technical debt score
        maintainability_index: Overall maintainability index
    """

    # Basic counts
    total_files: int = 0
    total_lines: int = 0
    total_blank_lines: int = 0
    total_comment_lines: int = 0
    total_code_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0

    # Averages and statistics
    avg_file_size: float = 0.0
    avg_complexity: float = 0.0
    max_complexity: float = 0.0
    min_complexity: float = float("inf")
    complexity_std_dev: float = 0.0

    # Ratios and scores
    documentation_ratio: float = 0.0
    test_coverage: float = 0.0
    code_duplication_ratio: float = 0.0
    technical_debt_score: float = 0.0
    maintainability_index: float = 0.0

    # Distributions
    languages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    file_types: Dict[str, int] = field(default_factory=dict)
    size_distribution: Dict[str, int] = field(default_factory=dict)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)

    # Top lists
    largest_files: List[Dict[str, Any]] = field(default_factory=list)
    most_complex_files: List[Dict[str, Any]] = field(default_factory=list)
    most_imported_modules: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics report to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of metrics
        """
        return {
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "total_blank_lines": self.total_blank_lines,
            "total_comment_lines": self.total_comment_lines,
            "total_code_lines": self.total_code_lines,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "total_imports": self.total_imports,
            "avg_file_size": round(self.avg_file_size, 2),
            "avg_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "min_complexity": self.min_complexity if self.min_complexity != float("inf") else 0,
            "complexity_std_dev": round(self.complexity_std_dev, 2),
            "documentation_ratio": round(self.documentation_ratio, 3),
            "test_coverage": round(self.test_coverage, 2),
            "code_duplication_ratio": round(self.code_duplication_ratio, 3),
            "technical_debt_score": round(self.technical_debt_score, 2),
            "maintainability_index": round(self.maintainability_index, 2),
            "languages": self.languages,
            "file_types": self.file_types,
            "size_distribution": self.size_distribution,
            "complexity_distribution": self.complexity_distribution,
            "largest_files": self.largest_files[:10],
            "most_complex_files": self.most_complex_files[:10],
            "most_imported_modules": self.most_imported_modules[:10],
        }

    @property
    def code_to_comment_ratio(self) -> float:
        """Calculate code to comment ratio.

        Returns:
            float: Ratio of code lines to comment lines
        """
        if self.total_comment_lines == 0:
            return float("inf")
        return self.total_code_lines / self.total_comment_lines

    @property
    def avg_file_complexity(self) -> float:
        """Calculate average complexity per file.

        Returns:
            float: Average complexity across all files
        """
        if self.total_files == 0:
            return 0.0
        return self.avg_complexity

    @property
    def quality_score(self) -> float:
        """Calculate overall code quality score (0-100).

        Combines various metrics to produce a single quality indicator.

        Returns:
            float: Quality score between 0 and 100
        """
        score = 100.0

        # Penalize high complexity
        if self.avg_complexity > 10:
            score -= min(20, (self.avg_complexity - 10) * 2)

        # Penalize poor documentation
        if self.documentation_ratio < 0.1:
            score -= 15

        # Penalize very large files
        if self.avg_file_size > 500:
            score -= 10

        # Penalize high duplication
        if self.code_duplication_ratio > 0.1:
            score -= min(15, self.code_duplication_ratio * 150)

        # Bonus for good test coverage
        if self.test_coverage > 0.8:
            score += 10

        return max(0, min(100, score))


class MetricsCalculator:
    """Calculator for code metrics extraction and aggregation.

    Processes analyzed files to compute comprehensive metrics including
    size measurements, complexity statistics, quality indicators, and
    distributional analysis.

    Attributes:
        config: Configuration object
        logger: Logger instance
    """

    def __init__(self, config: TenetsConfig):
        """Initialize metrics calculator with configuration.

        Args:
            config: TenetsConfig instance with metrics settings
        """
        self.config = config
        self.logger = get_logger(__name__)

    def calculate(self, files: List[Any]) -> MetricsReport:
        """Calculate comprehensive metrics for analyzed files.

        Processes a list of analyzed file objects to extract and aggregate
        various code metrics, producing a complete metrics report.

        Args:
            files: List of analyzed file objects

        Returns:
            MetricsReport: Comprehensive metrics analysis

        Example:
            >>> calculator = MetricsCalculator(config)
            >>> report = calculator.calculate(analyzed_files)
            >>> print(f"Average complexity: {report.avg_complexity}")
        """
        self.logger.debug(f"Calculating metrics for {len(files)} files")

        report = MetricsReport()

        if not files:
            return report

        # Collect raw metrics
        self._collect_basic_metrics(files, report)

        # Calculate distributions
        self._calculate_distributions(files, report)

        # Identify top items
        self._identify_top_items(files, report)

        # Calculate derived metrics
        self._calculate_derived_metrics(files, report)

        # Calculate language-specific metrics
        self._calculate_language_metrics(files, report)

        # Estimate quality indicators
        self._estimate_quality_indicators(files, report)

        self.logger.debug(f"Metrics calculation complete: {report.total_files} files")

        return report

    def calculate_file_metrics(self, file_analysis: Any) -> Dict[str, Any]:
        """Calculate metrics for a single file.

        Extracts detailed metrics from a single file analysis object,
        providing file-specific measurements and statistics.

        Args:
            file_analysis: Analyzed file object

        Returns:
            Dict[str, Any]: File-specific metrics

        Example:
            >>> metrics = calculator.calculate_file_metrics(file_analysis)
            >>> print(f"File complexity: {metrics['complexity']}")
        """
        metrics = {
            "lines": getattr(file_analysis, "lines", 0),
            "blank_lines": getattr(file_analysis, "blank_lines", 0),
            "comment_lines": getattr(file_analysis, "comment_lines", 0),
            "code_lines": 0,
            "functions": len(getattr(file_analysis, "functions", [])),
            "classes": len(getattr(file_analysis, "classes", [])),
            "imports": len(getattr(file_analysis, "imports", [])),
            "complexity": 0,
            "documentation_ratio": 0.0,
        }

        # Calculate code lines
        metrics["code_lines"] = metrics["lines"] - metrics["blank_lines"] - metrics["comment_lines"]

        # Extract complexity
        if hasattr(file_analysis, "complexity") and file_analysis.complexity:
            metrics["complexity"] = getattr(file_analysis.complexity, "cyclomatic", 0)

        # Calculate documentation ratio
        if metrics["code_lines"] > 0:
            metrics["documentation_ratio"] = metrics["comment_lines"] / metrics["code_lines"]

        # Add language and path info
        metrics["language"] = getattr(file_analysis, "language", "unknown")
        metrics["path"] = getattr(file_analysis, "path", "")
        metrics["name"] = Path(metrics["path"]).name if metrics["path"] else "unknown"

        return metrics

    def _collect_basic_metrics(self, files: List[Any], report: MetricsReport) -> None:
        """Collect basic counting metrics from files.

        Aggregates fundamental metrics like line counts, function counts,
        and class counts across all files.

        Args:
            files: List of analyzed files
            report: Report to populate with metrics
        """
        complexities = []

        for file in files:
            # Count files
            report.total_files += 1

            # Count lines
            if hasattr(file, "lines"):
                report.total_lines += file.lines
            if hasattr(file, "blank_lines"):
                report.total_blank_lines += file.blank_lines
            if hasattr(file, "comment_lines"):
                report.total_comment_lines += file.comment_lines

            # Count structures
            if hasattr(file, "functions"):
                report.total_functions += len(file.functions)
            if hasattr(file, "classes"):
                report.total_classes += len(file.classes)
            if hasattr(file, "imports"):
                report.total_imports += len(file.imports)

            # Collect complexity
            if hasattr(file, "complexity") and file.complexity:
                complexity = getattr(file.complexity, "cyclomatic", 0)
                complexities.append(complexity)
                report.max_complexity = max(report.max_complexity, complexity)
                report.min_complexity = min(report.min_complexity, complexity)

            # Track file types
            if hasattr(file, "language"):
                report.file_types[file.language] = report.file_types.get(file.language, 0) + 1

        # Calculate code lines
        report.total_code_lines = (
            report.total_lines - report.total_blank_lines - report.total_comment_lines
        )

        # Calculate averages
        if report.total_files > 0:
            report.avg_file_size = report.total_lines / report.total_files

        if complexities:
            report.avg_complexity = sum(complexities) / len(complexities)

            # Calculate standard deviation
            mean = report.avg_complexity
            variance = sum((x - mean) ** 2 for x in complexities) / len(complexities)
            report.complexity_std_dev = variance**0.5

    def _calculate_distributions(self, files: List[Any], report: MetricsReport) -> None:
        """Calculate statistical distributions of metrics.

        Creates histogram-like distributions for file sizes and complexity
        values to understand metric spread across the codebase.

        Args:
            files: List of analyzed files
            report: Report to populate with distributions
        """
        # Size distribution buckets
        size_buckets = {
            "tiny (1-50)": 0,
            "small (51-200)": 0,
            "medium (201-500)": 0,
            "large (501-1000)": 0,
            "huge (1000+)": 0,
        }

        # Complexity distribution buckets
        complexity_buckets = {
            "simple (1-5)": 0,
            "moderate (6-10)": 0,
            "complex (11-20)": 0,
            "very complex (21+)": 0,
        }

        for file in files:
            # Categorize by size
            if hasattr(file, "lines"):
                lines = file.lines
                if lines <= 50:
                    size_buckets["tiny (1-50)"] += 1
                elif lines <= 200:
                    size_buckets["small (51-200)"] += 1
                elif lines <= 500:
                    size_buckets["medium (201-500)"] += 1
                elif lines <= 1000:
                    size_buckets["large (501-1000)"] += 1
                else:
                    size_buckets["huge (1000+)"] += 1

            # Categorize by complexity
            if hasattr(file, "complexity") and file.complexity:
                complexity = getattr(file.complexity, "cyclomatic", 0)
                if complexity <= 5:
                    complexity_buckets["simple (1-5)"] += 1
                elif complexity <= 10:
                    complexity_buckets["moderate (6-10)"] += 1
                elif complexity <= 20:
                    complexity_buckets["complex (11-20)"] += 1
                else:
                    complexity_buckets["very complex (21+)"] += 1

        report.size_distribution = size_buckets
        report.complexity_distribution = complexity_buckets

    def _identify_top_items(self, files: List[Any], report: MetricsReport) -> None:
        """Identify top files by various metrics.

        Finds and records the files with extreme values for size,
        complexity, and other metrics for focused attention.

        Args:
            files: List of analyzed files
            report: Report to populate with top items
        """
        # Sort by size
        files_with_size = [f for f in files if hasattr(f, "lines") and hasattr(f, "path")]
        files_by_size = sorted(files_with_size, key=lambda f: f.lines, reverse=True)

        report.largest_files = [
            {
                "path": f.path,
                "name": Path(f.path).name,
                "lines": f.lines,
                "language": getattr(f, "language", "unknown"),
            }
            for f in files_by_size[:10]
        ]

        # Sort by complexity
        files_with_complexity = [
            f for f in files if hasattr(f, "complexity") and f.complexity and hasattr(f, "path")
        ]
        files_by_complexity = sorted(
            files_with_complexity,
            key=lambda f: getattr(f.complexity, "cyclomatic", 0),
            reverse=True,
        )

        report.most_complex_files = [
            {
                "path": f.path,
                "name": Path(f.path).name,
                "complexity": getattr(f.complexity, "cyclomatic", 0),
                "lines": getattr(f, "lines", 0),
                "language": getattr(f, "language", "unknown"),
            }
            for f in files_by_complexity[:10]
        ]

        # Track most imported modules
        import_counts = defaultdict(int)
        for file in files:
            if hasattr(file, "imports"):
                for imp in file.imports:
                    # Extract module name from import
                    if hasattr(imp, "module"):
                        module = imp.module
                    else:
                        module = str(imp)
                    if module:
                        import_counts[module] += 1

        report.most_imported_modules = sorted(
            import_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

    def _calculate_derived_metrics(self, files: List[Any], report: MetricsReport) -> None:
        """Calculate derived metrics from basic metrics.

        Computes higher-level metrics that are derived from the basic
        measurements, such as ratios and composite scores.

        Args:
            files: List of analyzed files
            report: Report to populate with derived metrics
        """
        # Documentation ratio
        if report.total_code_lines > 0:
            report.documentation_ratio = report.total_comment_lines / report.total_code_lines

        # Estimate test coverage (based on test file presence)
        test_files = [
            f
            for f in files
            if hasattr(f, "path") and ("test" in f.path.lower() or "spec" in f.path.lower())
        ]

        if report.total_files > 0:
            test_file_ratio = len(test_files) / report.total_files
            # Rough estimate: assume test files indicate ~60-80% coverage
            report.test_coverage = min(0.95, test_file_ratio * 3.5)

        # Estimate code duplication (simplified heuristic)
        # High similarity in file sizes and complexity might indicate duplication
        if files:
            size_variance = self._calculate_variance(
                [getattr(f, "lines", 0) for f in files if hasattr(f, "lines")]
            )

            # Lower variance might indicate more duplication
            if size_variance > 0:
                report.code_duplication_ratio = min(0.5, 1000 / (size_variance + 1000))

        # Calculate technical debt score (0-100, higher is worse)
        debt_score = 0.0

        # Factor in complexity
        if report.avg_complexity > 10:
            debt_score += min(30, (report.avg_complexity - 10) * 3)

        # Factor in large files
        if report.avg_file_size > 300:
            debt_score += min(20, (report.avg_file_size - 300) / 20)

        # Factor in poor documentation
        if report.documentation_ratio < 0.15:
            debt_score += 20 * (0.15 - report.documentation_ratio) / 0.15

        # Factor in lack of tests
        if report.test_coverage < 0.5:
            debt_score += 20 * (0.5 - report.test_coverage) / 0.5

        report.technical_debt_score = min(100, debt_score)

        # Calculate maintainability index (0-100, higher is better)
        # Based on Halstead volume, cyclomatic complexity, and lines of code
        # Simplified version of the MI formula
        maintainability = 171  # Base score

        # Reduce based on complexity
        maintainability -= 5.2 * min(10, report.avg_complexity)

        # Reduce based on size
        import math

        if report.avg_file_size > 0:
            maintainability -= 0.23 * math.log(report.avg_file_size + 1)

        # Normalize to 0-100
        report.maintainability_index = max(0, min(100, maintainability * 100 / 171))

    def _calculate_language_metrics(self, files: List[Any], report: MetricsReport) -> None:
        """Calculate per-language metrics breakdown.

        Computes metrics grouped by programming language to understand
        language-specific patterns and characteristics.

        Args:
            files: List of analyzed files
            report: Report to populate with language metrics
        """
        language_stats = defaultdict(
            lambda: {
                "files": 0,
                "lines": 0,
                "functions": 0,
                "classes": 0,
                "complexity_sum": 0,
                "complexity_count": 0,
            }
        )

        for file in files:
            if not hasattr(file, "language"):
                continue

            lang = file.language
            stats = language_stats[lang]

            # Count files
            stats["files"] += 1

            # Sum metrics
            if hasattr(file, "lines"):
                stats["lines"] += file.lines
            if hasattr(file, "functions"):
                stats["functions"] += len(file.functions)
            if hasattr(file, "classes"):
                stats["classes"] += len(file.classes)

            # Track complexity
            if hasattr(file, "complexity") and file.complexity:
                complexity = getattr(file.complexity, "cyclomatic", 0)
                if complexity > 0:
                    stats["complexity_sum"] += complexity
                    stats["complexity_count"] += 1

        # Calculate averages and store
        for lang, stats in language_stats.items():
            lang_metrics = {
                "files": stats["files"],
                "lines": stats["lines"],
                "functions": stats["functions"],
                "classes": stats["classes"],
                "avg_file_size": stats["lines"] / stats["files"] if stats["files"] > 0 else 0,
                "avg_complexity": (
                    stats["complexity_sum"] / stats["complexity_count"]
                    if stats["complexity_count"] > 0
                    else 0
                ),
            }
            report.languages[lang] = lang_metrics

    def _estimate_quality_indicators(self, files: List[Any], report: MetricsReport) -> None:
        """Estimate code quality indicators.

        Uses heuristics to estimate various quality indicators that
        would normally require more sophisticated analysis tools.

        Args:
            files: List of analyzed files
            report: Report to populate with quality estimates
        """
        # Additional quality indicators can be added here
        # For now, most are calculated in _calculate_derived_metrics

        # Example: Detect potential security issues
        security_patterns = ["eval", "exec", "password", "secret", "token"]
        security_mentions = 0

        for file in files:
            if hasattr(file, "content"):
                content_lower = file.content.lower()
                for pattern in security_patterns:
                    security_mentions += content_lower.count(pattern)

        # This could be used to adjust technical debt or add a security score
        if security_mentions > 50:  # Arbitrary threshold
            report.technical_debt_score = min(100, report.technical_debt_score + 10)

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values.

        Args:
            values: List of numeric values

        Returns:
            float: Variance of the values
        """
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


def calculate_metrics(files: List[Any], config: Optional[TenetsConfig] = None) -> MetricsReport:
    """Convenience function to calculate metrics for files.

    Creates a MetricsCalculator instance and calculates comprehensive
    metrics for the provided files.

    Args:
        files: List of analyzed file objects
        config: Optional configuration (uses defaults if None)

    Returns:
        MetricsReport: Comprehensive metrics analysis

    Example:
        >>> report = calculate_metrics(analyzed_files)
        >>> print(f"Quality score: {report.quality_score}")
    """
    if config is None:
        config = TenetsConfig()

    calculator = MetricsCalculator(config)
    return calculator.calculate(files)
