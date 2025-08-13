"""Code examination and inspection package.

This package provides comprehensive code analysis capabilities including
metrics calculation, complexity analysis, ownership tracking, and hotspot
detection. It extracts the core examination logic from CLI commands to
provide a reusable, testable API.

The examiner package works in conjunction with the analyzer package,
adding higher-level insights and aggregations on top of basic file analysis.

Main components:
- Examiner: Main orchestrator for code examination
- MetricsCalculator: Calculate code metrics and statistics
- ComplexityAnalyzer: Analyze code complexity patterns
- OwnershipTracker: Track code ownership and contribution patterns
- HotspotDetector: Identify frequently changed or problematic areas

Example usage:
    >>> from tenets.core.examiner import Examiner
    >>> from tenets.config import TenetsConfig
    >>>
    >>> config = TenetsConfig()
    >>> examiner = Examiner(config)
    >>>
    >>> # Comprehensive examination
    >>> results = examiner.examine_project(
    ...     path=Path("./src"),
    ...     deep=True,
    ...     include_git=True
    ... )
    >>>
    >>> print(f"Total files: {results.total_files}")
    >>> print(f"Average complexity: {results.metrics.avg_complexity}")
    >>> print(f"Top contributors: {results.ownership.top_contributors}")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .complexity import ComplexityAnalyzer, ComplexityReport
from .examiner import ExaminationResult, Examiner
from .hotspots import HotspotDetector, HotspotReport
from .metrics import MetricsCalculator, MetricsReport
from .ownership import OwnershipReport, OwnershipTracker

# Version
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Main examiner
    "Examiner",
    "ExaminationResult",
    "examine_project",
    "examine_file",
    # Metrics
    "MetricsCalculator",
    "MetricsReport",
    "calculate_metrics",
    # Complexity
    "ComplexityAnalyzer",
    "ComplexityReport",
    "analyze_complexity",
    # Ownership
    "OwnershipTracker",
    "OwnershipReport",
    "track_ownership",
    # Hotspots
    "HotspotDetector",
    "HotspotReport",
    "detect_hotspots",
]


def examine_project(
    path: Path,
    deep: bool = False,
    include_git: bool = True,
    include_metrics: bool = True,
    include_complexity: bool = True,
    include_ownership: bool = True,
    include_hotspots: bool = True,
    config: Optional[Any] = None,
) -> ExaminationResult:
    """Examine a project comprehensively.

    This is a convenience function that creates an Examiner instance
    and performs a full examination of the specified project.

    Args:
        path: Path to project directory
        deep: Whether to perform deep analysis with AST parsing
        include_git: Whether to include git analysis
        include_metrics: Whether to calculate detailed metrics
        include_complexity: Whether to analyze complexity
        include_ownership: Whether to track code ownership
        include_hotspots: Whether to detect hotspots
        config: Optional TenetsConfig instance

    Returns:
        ExaminationResult with comprehensive analysis

    Example:
        >>> from tenets.core.examiner import examine_project
        >>>
        >>> results = examine_project(
        ...     Path("./my_project"),
        ...     deep=True,
        ...     include_git=True
        ... )
        >>>
        >>> # Access various reports
        >>> print(f"Files analyzed: {results.total_files}")
        >>> print(f"Languages: {results.languages}")
        >>> print(f"Top complex files: {results.complexity.top_files}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    examiner = Examiner(config)

    return examiner.examine_project(
        path=path,
        deep=deep,
        include_git=include_git,
        include_metrics=include_metrics,
        include_complexity=include_complexity,
        include_ownership=include_ownership,
        include_hotspots=include_hotspots,
    )


def examine_file(
    file_path: Path, deep: bool = False, config: Optional[Any] = None
) -> Dict[str, Any]:
    """Examine a single file.

    Performs detailed analysis on a single file including
    complexity, metrics, and structure analysis.

    Args:
        file_path: Path to file
        deep: Whether to perform deep analysis
        config: Optional TenetsConfig instance

    Returns:
        Dictionary with file examination results

    Example:
        >>> from tenets.core.examiner import examine_file
        >>>
        >>> results = examine_file(Path("main.py"), deep=True)
        >>> print(f"Complexity: {results['complexity']}")
        >>> print(f"Lines: {results['lines']}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    examiner = Examiner(config)

    return examiner.examine_file(file_path, deep=deep)


def calculate_metrics(files: List[Any], config: Optional[Any] = None) -> MetricsReport:
    """Calculate metrics for a list of files.

    Args:
        files: List of FileAnalysis objects
        config: Optional TenetsConfig instance

    Returns:
        MetricsReport with calculated metrics

    Example:
        >>> from tenets.core.examiner import calculate_metrics
        >>>
        >>> metrics = calculate_metrics(analyzed_files)
        >>> print(f"Total lines: {metrics.total_lines}")
        >>> print(f"Average complexity: {metrics.avg_complexity}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    calculator = MetricsCalculator(config)
    return calculator.calculate(files)


def analyze_complexity(
    files: List[Any], threshold: int = 10, config: Optional[Any] = None
) -> ComplexityReport:
    """Analyze complexity patterns in files.

    Args:
        files: List of FileAnalysis objects
        threshold: Minimum complexity threshold
        config: Optional TenetsConfig instance

    Returns:
        ComplexityReport with analysis results

    Example:
        >>> from tenets.core.examiner import analyze_complexity
        >>>
        >>> complexity = analyze_complexity(files, threshold=10)
        >>> for file in complexity.high_complexity_files:
        >>>     print(f"{file.path}: {file.complexity}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    analyzer = ComplexityAnalyzer(config)
    return analyzer.analyze(files, threshold=threshold)


def track_ownership(
    repo_path: Path, since_days: int = 90, config: Optional[Any] = None
) -> OwnershipReport:
    """Track code ownership patterns.

    Args:
        repo_path: Path to git repository
        since_days: Days to look back
        config: Optional TenetsConfig instance

    Returns:
        OwnershipReport with ownership data

    Example:
        >>> from tenets.core.examiner import track_ownership
        >>>
        >>> ownership = track_ownership(Path("."), since_days=30)
        >>> for author in ownership.top_contributors:
        >>>     print(f"{author.name}: {author.commits}")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    tracker = OwnershipTracker(config)
    return tracker.track(repo_path, since_days=since_days)


def detect_hotspots(
    repo_path: Path,
    files: Optional[List[Any]] = None,
    threshold: int = 5,
    config: Optional[Any] = None,
) -> HotspotReport:
    """Detect code hotspots (frequently changed areas).

    Args:
        repo_path: Path to git repository
        files: Optional list of FileAnalysis objects
        threshold: Minimum change count for hotspot
        config: Optional TenetsConfig instance

    Returns:
        HotspotReport with detected hotspots

    Example:
        >>> from tenets.core.examiner import detect_hotspots
        >>>
        >>> hotspots = detect_hotspots(Path("."), threshold=10)
        >>> for hotspot in hotspots.files:
        >>>     print(f"{hotspot.path}: {hotspot.change_count} changes")
    """
    from tenets.config import TenetsConfig

    if config is None:
        config = TenetsConfig()

    detector = HotspotDetector(config)
    return detector.detect(repo_path, files=files, threshold=threshold)
