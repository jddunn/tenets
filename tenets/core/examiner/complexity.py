"""Complexity analysis module for code examination.

This module provides deep complexity analysis for codebases, calculating
various complexity metrics including cyclomatic complexity, cognitive
complexity, and Halstead metrics. It identifies complex areas that may
need refactoring and tracks complexity trends.

The complexity analyzer works with the examination system to provide
detailed insights into code maintainability and potential problem areas.
"""

import math
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


@dataclass
class ComplexityMetrics:
    """Detailed complexity metrics for a code element.

    Captures various complexity measurements for functions, classes,
    or files, providing a comprehensive view of code complexity.

    Attributes:
        cyclomatic: McCabe's cyclomatic complexity
        cognitive: Cognitive complexity (how hard to understand)
        halstead_volume: Halstead volume metric
        halstead_difficulty: Halstead difficulty metric
        halstead_effort: Halstead effort metric
        maintainability_index: Maintainability index (0-100)
        nesting_depth: Maximum nesting depth
        parameter_count: Number of parameters (for functions)
        line_count: Number of lines
        token_count: Number of tokens
        operator_count: Number of unique operators
        operand_count: Number of unique operands
    """

    cyclomatic: int = 1
    cognitive: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    maintainability_index: float = 100.0
    nesting_depth: int = 0
    parameter_count: int = 0
    line_count: int = 0
    token_count: int = 0
    operator_count: int = 0
    operand_count: int = 0

    @property
    def complexity_per_line(self) -> float:
        """Calculate complexity per line of code.

        Returns:
            float: Cyclomatic complexity divided by lines
        """
        if self.line_count == 0:
            return 0.0
        return self.cyclomatic / self.line_count

    @property
    def risk_level(self) -> str:
        """Determine risk level based on cyclomatic complexity.

        Uses industry-standard thresholds to categorize risk.

        Returns:
            str: Risk level (low, medium, high, very high)
        """
        if self.cyclomatic <= 5:
            return "low"
        elif self.cyclomatic <= 10:
            return "medium"
        elif self.cyclomatic <= 20:
            return "high"
        else:
            return "very high"

    @property
    def cognitive_risk_level(self) -> str:
        """Determine risk level based on cognitive complexity.

        Returns:
            str: Cognitive risk level
        """
        if self.cognitive <= 7:
            return "low"
        elif self.cognitive <= 15:
            return "medium"
        elif self.cognitive <= 25:
            return "high"
        else:
            return "very high"


@dataclass
class FunctionComplexity:
    """Complexity analysis for a single function or method.

    Tracks detailed complexity metrics for individual functions,
    including their location, parameters, and various complexity scores.

    Attributes:
        name: Function name
        full_name: Fully qualified name (with class if method)
        file_path: Path to containing file
        line_start: Starting line number
        line_end: Ending line number
        metrics: Detailed complexity metrics
        calls: Functions called by this function
        called_by: Functions that call this function
        is_recursive: Whether function is recursive
        is_generator: Whether function is a generator
        is_async: Whether function is async
        has_decorator: Whether function has decorators
        docstring: Function docstring if present
    """

    name: str
    full_name: str
    file_path: str
    line_start: int
    line_end: int
    metrics: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    calls: Set[str] = field(default_factory=set)
    called_by: Set[str] = field(default_factory=set)
    is_recursive: bool = False
    is_generator: bool = False
    is_async: bool = False
    has_decorator: bool = False
    docstring: Optional[str] = None

    @property
    def lines(self) -> int:
        """Get number of lines in function.

        Returns:
            int: Line count
        """
        return self.line_end - self.line_start + 1

    @property
    def has_documentation(self) -> bool:
        """Check if function has documentation.

        Returns:
            bool: True if docstring exists
        """
        return bool(self.docstring)


@dataclass
class ClassComplexity:
    """Complexity analysis for a class.

    Aggregates complexity metrics for an entire class including
    all its methods and nested classes.

    Attributes:
        name: Class name
        file_path: Path to containing file
        line_start: Starting line number
        line_end: Ending line number
        metrics: Aggregated complexity metrics
        methods: List of method complexity analyses
        nested_classes: List of nested class complexities
        inheritance_depth: Depth in inheritance hierarchy
        parent_classes: List of parent class names
        abstract_methods: Count of abstract methods
        static_methods: Count of static methods
        properties: Count of properties
        instance_attributes: Count of instance attributes
    """

    name: str
    file_path: str
    line_start: int
    line_end: int
    metrics: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    methods: List[FunctionComplexity] = field(default_factory=list)
    nested_classes: List["ClassComplexity"] = field(default_factory=list)
    inheritance_depth: int = 0
    parent_classes: List[str] = field(default_factory=list)
    abstract_methods: int = 0
    static_methods: int = 0
    properties: int = 0
    instance_attributes: int = 0

    @property
    def total_methods(self) -> int:
        """Get total number of methods.

        Returns:
            int: Method count
        """
        return len(self.methods)

    @property
    def avg_method_complexity(self) -> float:
        """Calculate average method complexity.

        Returns:
            float: Average cyclomatic complexity of methods
        """
        if not self.methods:
            return 0.0
        return sum(m.metrics.cyclomatic for m in self.methods) / len(self.methods)

    @property
    def weighted_methods_per_class(self) -> int:
        """Calculate WMC (Weighted Methods per Class) metric.

        Sum of complexities of all methods in the class.

        Returns:
            int: WMC metric value
        """
        return sum(m.metrics.cyclomatic for m in self.methods)


@dataclass
class FileComplexity:
    """Complexity analysis for an entire file.

    Aggregates all complexity metrics for a source file including
    functions, classes, and overall file metrics.

    Attributes:
        path: File path
        name: File name
        language: Programming language
        metrics: File-level complexity metrics
        functions: List of function complexities
        classes: List of class complexities
        total_complexity: Sum of all complexity in file
        max_complexity: Maximum complexity found in file
        complexity_hotspots: Areas of high complexity
        import_complexity: Complexity from imports/dependencies
        coupling: Coupling metric
        cohesion: Cohesion metric
    """

    path: str
    name: str
    language: str
    metrics: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    functions: List[FunctionComplexity] = field(default_factory=list)
    classes: List[ClassComplexity] = field(default_factory=list)
    total_complexity: int = 0
    max_complexity: int = 0
    complexity_hotspots: List[Dict[str, Any]] = field(default_factory=list)
    import_complexity: int = 0
    coupling: float = 0.0
    cohesion: float = 0.0

    @property
    def avg_complexity(self) -> float:
        """Calculate average complexity across all functions.

        Returns:
            float: Average complexity
        """
        all_functions = self.functions.copy()
        for cls in self.classes:
            all_functions.extend(cls.methods)

        if not all_functions:
            return 0.0

        return sum(f.metrics.cyclomatic for f in all_functions) / len(all_functions)

    @property
    def needs_refactoring(self) -> bool:
        """Determine if file needs refactoring based on complexity.

        Returns:
            bool: True if refactoring is recommended
        """
        return self.max_complexity > 20 or self.avg_complexity > 10 or self.total_complexity > 100


@dataclass
class ComplexityReport:
    """Comprehensive complexity analysis report.

    Aggregates complexity analysis across an entire codebase,
    providing statistics, trends, and actionable insights.

    Attributes:
        total_files: Total files analyzed
        total_functions: Total functions analyzed
        total_classes: Total classes analyzed
        avg_complexity: Average cyclomatic complexity
        max_complexity: Maximum cyclomatic complexity found
        median_complexity: Median cyclomatic complexity
        std_dev_complexity: Standard deviation of complexity
        high_complexity_count: Count of high complexity items
        very_high_complexity_count: Count of very high complexity items
        files: List of file complexity analyses
        top_complex_functions: Most complex functions
        top_complex_classes: Most complex classes
        top_complex_files: Most complex files
        complexity_distribution: Distribution of complexity values
        refactoring_candidates: Items recommended for refactoring
        technical_debt_hours: Estimated hours to address complexity
        trend_direction: Whether complexity is increasing/decreasing
        recommendations: List of actionable recommendations
    """

    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    avg_complexity: float = 0.0
    max_complexity: int = 0
    median_complexity: float = 0.0
    std_dev_complexity: float = 0.0
    high_complexity_count: int = 0
    very_high_complexity_count: int = 0
    files: List[FileComplexity] = field(default_factory=list)
    top_complex_functions: List[FunctionComplexity] = field(default_factory=list)
    top_complex_classes: List[ClassComplexity] = field(default_factory=list)
    top_complex_files: List[FileComplexity] = field(default_factory=list)
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    refactoring_candidates: List[Dict[str, Any]] = field(default_factory=list)
    technical_debt_hours: float = 0.0
    trend_direction: str = "stable"
    recommendations: List[str] = field(default_factory=list)
    # Accept an optional constructor-only override for complexity score (tests pass this)
    # Note: Using InitVar keeps it out of instance __dict__, we defensively validate type in __post_init__
    complexity_score: InitVar[Optional[float]] = None
    # Internal override storage (not part of dataclass init)
    _override_complexity_score: Optional[float] = field(default=None, repr=False, compare=False)

    def __post_init__(self, complexity_score: Optional[float] = None) -> None:
        """Store validated override if provided.

        Some tests instantiate with complexity_score=NN. Also be resilient to
        Mock/property objects that may slip in; only keep numeric overrides.
        """
        try:
            if complexity_score is None:
                self._override_complexity_score = None
            else:
                # Accept ints/floats or strings that can parse to float
                val = float(complexity_score)  # may raise
                self._override_complexity_score = val
        except Exception:
            # Ignore invalid overrides
            self._override_complexity_score = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "total_files": self.total_files,
            "total_functions": self.total_functions,
            "total_classes": self.total_classes,
            "avg_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "median_complexity": round(self.median_complexity, 2),
            "std_dev_complexity": round(self.std_dev_complexity, 2),
            "high_complexity_count": self.high_complexity_count,
            "very_high_complexity_count": self.very_high_complexity_count,
            "complexity_distribution": self.complexity_distribution,
            "refactoring_candidates": self.refactoring_candidates[:10],
            "technical_debt_hours": round(self.technical_debt_hours, 1),
            "trend_direction": self.trend_direction,
            "recommendations": self.recommendations,
        }

    @property
    def complexity_score(self) -> float:
        """Calculate overall complexity score (0-100).

        Lower scores indicate better (less complex) code.

        Returns:
            float: Complexity score
        """
        # Honor explicit override when provided (used in some tests)
        override = self._override_complexity_score
        if isinstance(override, (int, float)):
            return float(max(0.0, min(100.0, float(override))))

        score = 0.0

        # Base score on average complexity
        score += min(40, self.avg_complexity * 4)

        # Add penalty for high complexity items
        score += min(30, self.high_complexity_count * 2)

        # Add penalty for very high complexity items
        score += min(30, self.very_high_complexity_count * 5)

        return min(100, score)


class ComplexityAnalyzer:
    """Analyzer for code complexity metrics.

    Provides comprehensive complexity analysis including cyclomatic
    complexity, cognitive complexity, and various other metrics to
    assess code maintainability and identify refactoring opportunities.

    Attributes:
        config: Configuration object
        logger: Logger instance
        complexity_cache: Cache of computed complexities
    """

    def __init__(self, config: TenetsConfig):
        """Initialize complexity analyzer.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.complexity_cache: Dict[str, Any] = {}

    # -------- Safe coercion helpers --------
    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Best-effort conversion to int.

        Handles None, strings, and Mock-like objects gracefully.
        """
        if value is None:
            return default
        try:
            # Avoid converting booleans as they are ints in Python
            if isinstance(value, bool):
                return int(value)
            return int(value)
        except Exception:
            try:
                # Try float then int
                return int(float(str(value)))
            except Exception:
                return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Best-effort conversion to float.

        Handles None, strings, and Mock-like objects gracefully.
        """
        if value is None:
            return default
        try:
            if isinstance(value, bool):
                return float(value)
            return float(value)
        except Exception:
            try:
                return float(str(value))
            except Exception:
                return default

    def analyze(
        self, files: List[Any], threshold: float = 10.0, deep: bool = False
    ) -> ComplexityReport:
        """Analyze complexity for a list of files.

        Performs comprehensive complexity analysis across all provided
        files, calculating various metrics and identifying problem areas.

        Args:
            files: List of analyzed file objects
            threshold: Complexity threshold for flagging
            deep: Whether to perform deep analysis

        Returns:
            ComplexityReport: Comprehensive complexity analysis

        Example:
            >>> analyzer = ComplexityAnalyzer(config)
            >>> report = analyzer.analyze(files, threshold=10)
            >>> print(f"Average complexity: {report.avg_complexity}")
        """
        self.logger.debug(f"Analyzing complexity for {len(files)} files")

        report = ComplexityReport()
        all_complexities = []

        for file in files:
            if not self._should_analyze_file(file):
                continue

            # Analyze file complexity
            file_complexity = self._analyze_file_complexity(file, deep)
            if file_complexity:
                report.files.append(file_complexity)
                report.total_files += 1

                # Collect all function complexities
                for func in file_complexity.functions:
                    all_complexities.append(func.metrics.cyclomatic)
                    report.total_functions += 1

                    # Track high complexity functions
                    if func.metrics.cyclomatic > threshold:
                        report.high_complexity_count += 1
                        if func.metrics.cyclomatic > threshold * 2:
                            report.very_high_complexity_count += 1

                    # Update max complexity
                    report.max_complexity = max(report.max_complexity, func.metrics.cyclomatic)

                # Process classes
                for cls in file_complexity.classes:
                    report.total_classes += 1
                    for method in cls.methods:
                        all_complexities.append(method.metrics.cyclomatic)
                        report.total_functions += 1

                        if method.metrics.cyclomatic > threshold:
                            report.high_complexity_count += 1
                            if method.metrics.cyclomatic > threshold * 2:
                                report.very_high_complexity_count += 1

        # Calculate statistics
        if all_complexities:
            report.avg_complexity = sum(all_complexities) / len(all_complexities)
            report.median_complexity = self._calculate_median(all_complexities)
            report.std_dev_complexity = self._calculate_std_dev(all_complexities)

        # Calculate distribution
        report.complexity_distribution = self._calculate_distribution(all_complexities)

        # Identify top complex items
        self._identify_top_complex_items(report)

        # Identify refactoring candidates
        self._identify_refactoring_candidates(report, threshold)

        # Estimate technical debt
        report.technical_debt_hours = self._estimate_technical_debt(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        self.logger.debug(f"Complexity analysis complete: avg={report.avg_complexity:.2f}")

        return report

    def analyze_file(self, file_analysis: Any) -> Dict[str, Any]:
        """Analyze complexity for a single file.

        Args:
            file_analysis: Analyzed file object

        Returns:
            Dict[str, Any]: File complexity details
        """
        file_complexity = self._analyze_file_complexity(file_analysis, deep=True)

        if not file_complexity:
            return {}

        return {
            "cyclomatic": file_complexity.metrics.cyclomatic,
            "cognitive": file_complexity.metrics.cognitive,
            "avg_complexity": file_complexity.avg_complexity,
            "max_complexity": file_complexity.max_complexity,
            "total_complexity": file_complexity.total_complexity,
            "functions": len(file_complexity.functions),
            "classes": len(file_complexity.classes),
            "needs_refactoring": file_complexity.needs_refactoring,
            "risk_level": file_complexity.metrics.risk_level,
            "maintainability_index": file_complexity.metrics.maintainability_index,
        }

    def _should_analyze_file(self, file: Any) -> bool:
        """Determine if file should be analyzed for complexity.

        Args:
            file: File object to check

        Returns:
            bool: True if file should be analyzed
        """
        # Skip files without necessary attributes (be robust to Mock objects)
        path_val = getattr(file, "path", None)
        lang_val = getattr(file, "language", None)
        if not path_val or not isinstance(path_val, (str, Path)):
            return False
        if not lang_val or not isinstance(lang_val, str):
            return False

        # Skip non-code files
        non_code_extensions = {".txt", ".md", ".json", ".yml", ".yaml", ".xml", ".html"}
        try:
            suffix = Path(str(path_val)).suffix.lower()
        except Exception:
            return False
        if suffix in non_code_extensions:
            return False

        # Skip very large files (likely generated)
        lines = getattr(file, "lines", None)
        if isinstance(lines, (int, float)) and lines > 10000:
            return False

        return True

    def _analyze_file_complexity(self, file: Any, deep: bool = False) -> Optional[FileComplexity]:
        """Analyze complexity for a single file.

        Args:
            file: Analyzed file object
            deep: Whether to perform deep analysis

        Returns:
            Optional[FileComplexity]: File complexity analysis
        """
        try:
            file_complexity = FileComplexity(
                path=file.path,
                name=Path(file.path).name,
                language=getattr(file, "language", "unknown"),
            )

            # Extract basic metrics
            if hasattr(file, "complexity") and file.complexity:
                # file.complexity is a ComplexityMetrics object from models/analysis.py
                file_complexity.metrics.cyclomatic = self._safe_int(
                    file.complexity.cyclomatic if hasattr(file.complexity, "cyclomatic") else 1, 1
                )
                file_complexity.metrics.cognitive = self._safe_int(
                    file.complexity.cognitive if hasattr(file.complexity, "cognitive") else 0, 0
                )
                # Copy other metrics if available
                if hasattr(file.complexity, "halstead_volume"):
                    file_complexity.metrics.halstead_volume = file.complexity.halstead_volume
                if hasattr(file.complexity, "maintainability_index"):
                    file_complexity.metrics.maintainability_index = (
                        file.complexity.maintainability_index
                    )
                if hasattr(file.complexity, "max_depth"):
                    file_complexity.metrics.nesting_depth = file.complexity.max_depth

            # Process functions
            if hasattr(file, "functions"):
                for func in file.functions:
                    func_complexity = self._analyze_function_complexity(func, file.path)
                    if func_complexity:
                        file_complexity.functions.append(func_complexity)
                        file_complexity.total_complexity += func_complexity.metrics.cyclomatic
                        file_complexity.max_complexity = max(
                            file_complexity.max_complexity, func_complexity.metrics.cyclomatic
                        )

            # Process classes
            if hasattr(file, "classes"):
                for cls in file.classes:
                    class_complexity = self._analyze_class_complexity(cls, file.path)
                    if class_complexity:
                        file_complexity.classes.append(class_complexity)

                        # Add class methods to total
                        for method in class_complexity.methods:
                            file_complexity.total_complexity += method.metrics.cyclomatic
                            file_complexity.max_complexity = max(
                                file_complexity.max_complexity, method.metrics.cyclomatic
                            )

            # Calculate file-level metrics
            if hasattr(file, "lines"):
                file_complexity.metrics.line_count = self._safe_int(getattr(file, "lines", 0), 0)

            # Identify hotspots
            file_complexity.complexity_hotspots = self._identify_hotspots(file_complexity)

            # Calculate maintainability index
            file_complexity.metrics.maintainability_index = self._calculate_maintainability(
                file_complexity
            )

            return file_complexity

        except Exception as e:
            self.logger.warning(f"Failed to analyze complexity for {file.path}: {e}")
            return None

    def _analyze_function_complexity(
        self, func: Any, file_path: str
    ) -> Optional[FunctionComplexity]:
        """Analyze complexity for a function.

        Args:
            func: Function object
            file_path: Path to containing file

        Returns:
            Optional[FunctionComplexity]: Function complexity analysis
        """
        try:
            func_complexity = FunctionComplexity(
                name=str(getattr(func, "name", "unknown")),
                full_name=str(getattr(func, "full_name", getattr(func, "name", "unknown"))),
                file_path=file_path,
                line_start=self._safe_int(getattr(func, "line_start", 0), 0),
                line_end=self._safe_int(getattr(func, "line_end", 0), 0),
            )

            # Extract complexity metrics
            if hasattr(func, "complexity"):
                func_complexity.metrics.cyclomatic = self._safe_int(
                    getattr(func.complexity, "cyclomatic", 1), 1
                )
                func_complexity.metrics.cognitive = self._safe_int(
                    getattr(func.complexity, "cognitive", 0), 0
                )
                func_complexity.metrics.nesting_depth = self._safe_int(
                    getattr(func.complexity, "nesting_depth", 0), 0
                )

            # Extract other metrics
            if hasattr(func, "parameters"):
                try:
                    func_complexity.metrics.parameter_count = len(func.parameters)
                except Exception:
                    func_complexity.metrics.parameter_count = 0

            func_complexity.metrics.line_count = max(
                0, self._safe_int(func_complexity.line_end - func_complexity.line_start + 1, 0)
            )

            # Set flags
            func_complexity.is_async = getattr(func, "is_async", False)
            func_complexity.is_generator = getattr(func, "is_generator", False)
            func_complexity.has_decorator = getattr(func, "has_decorator", False)
            func_complexity.docstring = getattr(func, "docstring", None)

            return func_complexity

        except Exception as e:
            self.logger.debug(f"Failed to analyze function complexity: {e}")
            return None

    def _analyze_class_complexity(self, cls: Any, file_path: str) -> Optional[ClassComplexity]:
        """Analyze complexity for a class.

        Args:
            cls: Class object
            file_path: Path to containing file

        Returns:
            Optional[ClassComplexity]: Class complexity analysis
        """
        try:
            class_complexity = ClassComplexity(
                name=str(getattr(cls, "name", "unknown")),
                file_path=file_path,
                line_start=self._safe_int(getattr(cls, "line_start", 0), 0),
                line_end=self._safe_int(getattr(cls, "line_end", 0), 0),
            )

            # Process methods
            if hasattr(cls, "methods"):
                for method in cls.methods:
                    method_complexity = self._analyze_function_complexity(method, file_path)
                    if method_complexity:
                        class_complexity.methods.append(method_complexity)

            # Extract class metrics
            class_complexity.inheritance_depth = self._safe_int(
                getattr(cls, "inheritance_depth", 0), 0
            )
            try:
                class_complexity.parent_classes = list(getattr(cls, "parent_classes", []))
            except Exception:
                class_complexity.parent_classes = []

            # Calculate aggregate complexity
            if class_complexity.methods:
                total_cyclomatic = sum(
                    self._safe_int(m.metrics.cyclomatic, 0) for m in class_complexity.methods
                )
                class_complexity.metrics.cyclomatic = total_cyclomatic

            return class_complexity

        except Exception as e:
            self.logger.debug(f"Failed to analyze class complexity: {e}")
            return None

    def _identify_hotspots(self, file_complexity: FileComplexity) -> List[Dict[str, Any]]:
        """Identify complexity hotspots in a file.

        Args:
            file_complexity: File complexity analysis

        Returns:
            List[Dict[str, Any]]: List of hotspot locations
        """
        hotspots = []

        # Check functions
        for func in file_complexity.functions:
            if func.metrics.cyclomatic > 10:
                hotspots.append(
                    {
                        "type": "function",
                        "name": func.name,
                        "lines": f"{func.line_start}-{func.line_end}",
                        "complexity": func.metrics.cyclomatic,
                        "risk": func.metrics.risk_level,
                    }
                )

        # Check methods
        for cls in file_complexity.classes:
            for method in cls.methods:
                if method.metrics.cyclomatic > 10:
                    hotspots.append(
                        {
                            "type": "method",
                            "name": f"{cls.name}.{method.name}",
                            "lines": f"{method.line_start}-{method.line_end}",
                            "complexity": method.metrics.cyclomatic,
                            "risk": method.metrics.risk_level,
                        }
                    )

        # Sort by complexity
        hotspots.sort(key=lambda x: x["complexity"], reverse=True)

        return hotspots

    def _calculate_maintainability(self, file_complexity: FileComplexity) -> float:
        """Calculate maintainability index for a file.

        Uses a simplified version of the Maintainability Index formula.

        Args:
            file_complexity: File complexity analysis

        Returns:
            float: Maintainability index (0-100)
        """
        # MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)
        # Where V = Halstead Volume, CC = Cyclomatic Complexity, LOC = Lines of Code

        mi = 171.0

        # Cyclomatic complexity component
        mi -= 0.23 * file_complexity.total_complexity

        # Lines of code component
        line_count = self._safe_int(file_complexity.metrics.line_count, 0)
        if line_count > 0:
            mi -= 16.2 * math.log(line_count)

        # Halstead volume component (estimated)
        # Estimate volume as lines * 10 for simplicity
        estimated_volume = line_count * 10
        if estimated_volume > 0:
            mi -= 5.2 * math.log(estimated_volume)

        # Normalize to 0-100
        mi = max(0, min(100, mi * 100 / 171))

        return mi

    def _identify_top_complex_items(self, report: ComplexityReport) -> None:
        """Identify the most complex functions, classes, and files.

        Args:
            report: Report to populate with top items
        """
        all_functions = []
        all_classes = []

        # Collect all functions and classes
        for file in report.files:
            all_functions.extend(file.functions)
            all_classes.extend(file.classes)

            for cls in file.classes:
                all_functions.extend(cls.methods)

        # Sort and select top functions
        all_functions.sort(key=lambda f: f.metrics.cyclomatic, reverse=True)
        report.top_complex_functions = all_functions[:10]

        # Sort and select top classes
        all_classes.sort(key=lambda c: c.metrics.cyclomatic, reverse=True)
        report.top_complex_classes = all_classes[:10]

        # Sort and select top files
        report.files.sort(key=lambda f: f.total_complexity, reverse=True)
        report.top_complex_files = report.files[:10]

    def _identify_refactoring_candidates(self, report: ComplexityReport, threshold: float) -> None:
        """Identify items that should be refactored.

        Args:
            report: Report to populate with candidates
            threshold: Complexity threshold
        """
        candidates = []

        for file in report.files:
            # Check functions
            for func in file.functions:
                if func.metrics.cyclomatic > threshold:
                    candidates.append(
                        {
                            "type": "function",
                            "name": func.name,
                            "file": file.name,
                            "complexity": func.metrics.cyclomatic,
                            "lines": func.lines,
                            "reason": f"Complexity {func.metrics.cyclomatic} exceeds threshold {threshold}",
                            "priority": (
                                "high" if func.metrics.cyclomatic > threshold * 2 else "medium"
                            ),
                        }
                    )

            # Check classes
            for cls in file.classes:
                if cls.avg_method_complexity > threshold:
                    candidates.append(
                        {
                            "type": "class",
                            "name": cls.name,
                            "file": file.name,
                            "complexity": cls.avg_method_complexity,
                            "methods": cls.total_methods,
                            "reason": f"Average method complexity {cls.avg_method_complexity:.1f} exceeds threshold",
                            "priority": "medium",
                        }
                    )

                # Check individual methods
                for method in cls.methods:
                    if method.metrics.cyclomatic > threshold:
                        candidates.append(
                            {
                                "type": "method",
                                "name": f"{cls.name}.{method.name}",
                                "file": file.name,
                                "complexity": method.metrics.cyclomatic,
                                "lines": method.lines,
                                "reason": f"Method complexity {method.metrics.cyclomatic} exceeds threshold",
                                "priority": (
                                    "high"
                                    if method.metrics.cyclomatic > threshold * 2
                                    else "medium"
                                ),
                            }
                        )

        # Sort by complexity and priority
        candidates.sort(key=lambda x: (x["priority"] == "high", x["complexity"]), reverse=True)

        report.refactoring_candidates = candidates

    def _estimate_technical_debt(self, report: ComplexityReport) -> float:
        """Estimate technical debt in hours.

        Uses complexity metrics to estimate refactoring time.

        Args:
            report: Complexity report

        Returns:
            float: Estimated hours of technical debt
        """
        hours = 0.0

        # Estimate based on high complexity items
        # Rough estimates:
        # - High complexity function: 2-4 hours to refactor
        # - Very high complexity function: 4-8 hours to refactor

        for candidate in report.refactoring_candidates:
            if candidate["priority"] == "high":
                hours += 6.0  # Average for very high complexity
            else:
                hours += 3.0  # Average for high complexity

        # Add time for testing and review (50% of refactoring time)
        hours *= 1.5

        return hours

    def _generate_recommendations(self, report: ComplexityReport) -> List[str]:
        """Generate actionable recommendations based on analysis.

        Args:
            report: Complexity report

        Returns:
            List[str]: List of recommendations
        """
        recommendations = []

        # Check overall complexity
        if report.avg_complexity > 10:
            recommendations.append(
                f"Average complexity ({report.avg_complexity:.1f}) is high. "
                "Consider breaking down complex functions."
            )

        # Check for very complex items
        if report.very_high_complexity_count > 0:
            recommendations.append(
                f"Found {report.very_high_complexity_count} functions with very high complexity. "
                "These should be refactored immediately."
            )

        # Check for complexity concentration
        if report.top_complex_files and len(report.top_complex_files) > 0:
            top_file = report.top_complex_files[0]
            if top_file.total_complexity > 100:
                recommendations.append(
                    f"File '{top_file.name}' has total complexity of {top_file.total_complexity}. "
                    "Consider splitting this file into smaller modules."
                )

        # Check for large classes
        if report.top_complex_classes:
            for cls in report.top_complex_classes[:3]:
                if cls.total_methods > 20:
                    recommendations.append(
                        f"Class '{cls.name}' has {cls.total_methods} methods. "
                        "Consider applying the Single Responsibility Principle."
                    )

        # Technical debt recommendation
        if report.technical_debt_hours > 40:
            recommendations.append(
                f"Estimated {report.technical_debt_hours:.0f} hours of technical debt. "
                "Consider dedicating sprint time to refactoring."
            )

        # General recommendations based on score
        if report.complexity_score > 70:
            recommendations.append(
                "Overall complexity is very high. Consider establishing complexity "
                "limits in your CI/CD pipeline."
            )
        elif report.complexity_score > 50:
            recommendations.append(
                "Moderate complexity detected. Regular refactoring sessions "
                "would help maintain code quality."
            )

        return recommendations

    def _calculate_median(self, values: List[float]) -> float:
        """Calculate median of values.

        Args:
            values: List of numeric values

        Returns:
            float: Median value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            return sorted_values[n // 2]

    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of values.

        Args:
            values: List of numeric values

        Returns:
            float: Standard deviation
        """
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _calculate_distribution(self, complexities: List[int]) -> Dict[str, int]:
        """Calculate complexity distribution.

        Args:
            complexities: List of complexity values

        Returns:
            Dict[str, int]: Distribution buckets
        """
        distribution = {
            "simple (1-5)": 0,
            "moderate (6-10)": 0,
            "complex (11-20)": 0,
            "very complex (21+)": 0,
        }

        for complexity in complexities:
            if complexity <= 5:
                distribution["simple (1-5)"] += 1
            elif complexity <= 10:
                distribution["moderate (6-10)"] += 1
            elif complexity <= 20:
                distribution["complex (11-20)"] += 1
            else:
                distribution["very complex (21+)"] += 1

        return distribution


# Backward-compatible functional API expected by tests
def analyze_complexity(
    files: List[Any], threshold: int = 10, config: Optional[TenetsConfig] = None
) -> ComplexityReport:
    """Analyze complexity for a list of files.

    Thin wrapper that constructs a ComplexityAnalyzer and returns its report.

    Args:
        files: List of analyzed file-like objects
        threshold: Threshold for high/very high classification
        config: Optional TenetsConfig instance

    Returns:
        ComplexityReport
    """
    cfg = config or TenetsConfig()
    analyzer = ComplexityAnalyzer(cfg)
    return analyzer.analyze(files, threshold=float(threshold), deep=False)
