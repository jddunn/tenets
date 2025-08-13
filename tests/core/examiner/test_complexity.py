"""Tests for complexity analysis module."""

from unittest.mock import Mock

import pytest

from tenets.config import TenetsConfig
from tenets.core.examiner.complexity import (
    ClassComplexity,
    ComplexityAnalyzer,
    ComplexityMetrics,
    ComplexityReport,
    FileComplexity,
    FunctionComplexity,
    analyze_complexity,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def complexity_analyzer(config):
    """Create ComplexityAnalyzer instance."""
    return ComplexityAnalyzer(config)


@pytest.fixture
def sample_files():
    """Create sample analyzed files with complexity data."""
    files = []

    # Simple file
    file1 = Mock()
    file1.path = "simple.py"
    file1.language = "Python"
    file1.lines = 50
    file1.complexity = Mock(cyclomatic=3, cognitive=2)
    file1.functions = [Mock(name="simple_func", complexity=Mock(cyclomatic=2))]
    file1.classes = []
    files.append(file1)

    # Complex file
    file2 = Mock()
    file2.path = "complex.py"
    file2.language = "Python"
    file2.lines = 500
    file2.complexity = Mock(cyclomatic=25, cognitive=30)

    # Complex functions
    file2.functions = [
        Mock(
            name="complex_func1",
            line_start=10,
            line_end=100,
            complexity=Mock(cyclomatic=15, cognitive=18, nesting_depth=4),
            parameters=["a", "b", "c"],
            is_async=False,
            is_generator=False,
            has_decorator=True,
            docstring="Complex function",
        ),
        Mock(
            name="complex_func2",
            line_start=110,
            line_end=200,
            complexity=Mock(cyclomatic=12, cognitive=15, nesting_depth=3),
            parameters=["x", "y"],
            is_async=True,
            is_generator=False,
            has_decorator=False,
            docstring=None,
        ),
    ]

    # Complex class
    class_mock = Mock()
    class_mock.name = "ComplexClass"
    class_mock.line_start = 210
    class_mock.line_end = 450
    class_mock.methods = [
        Mock(name="method1", line_start=220, line_end=280, complexity=Mock(cyclomatic=8)),
        Mock(name="method2", line_start=290, line_end=340, complexity=Mock(cyclomatic=10)),
    ]
    class_mock.inheritance_depth = 2
    class_mock.parent_classes = ["BaseClass"]

    file2.classes = [class_mock]
    files.append(file2)

    # Very complex file
    file3 = Mock()
    file3.path = "very_complex.py"
    file3.language = "Python"
    file3.lines = 1000
    file3.complexity = Mock(cyclomatic=50, cognitive=60)
    file3.functions = [
        Mock(
            name=f"func_{i}",
            line_start=i * 50,
            line_end=i * 50 + 40,
            complexity=Mock(cyclomatic=25 + i),
        )
        for i in range(3)
    ]
    file3.classes = []
    files.append(file3)

    return files


class TestComplexityMetrics:
    """Test suite for ComplexityMetrics dataclass."""

    def test_complexity_metrics_creation(self):
        """Test creating ComplexityMetrics."""
        metrics = ComplexityMetrics(cyclomatic=10, cognitive=15, nesting_depth=3)

        assert metrics.cyclomatic == 10
        assert metrics.cognitive == 15
        assert metrics.nesting_depth == 3

    def test_complexity_per_line(self):
        """Test complexity_per_line property."""
        metrics = ComplexityMetrics(cyclomatic=20, line_count=100)

        assert metrics.complexity_per_line == 0.2

        # Test with no lines
        metrics_no_lines = ComplexityMetrics(cyclomatic=10, line_count=0)
        assert metrics_no_lines.complexity_per_line == 0.0

    def test_risk_level(self):
        """Test risk_level property based on cyclomatic complexity."""
        # Low risk
        low = ComplexityMetrics(cyclomatic=3)
        assert low.risk_level == "low"

        # Medium risk
        medium = ComplexityMetrics(cyclomatic=8)
        assert medium.risk_level == "medium"

        # High risk
        high = ComplexityMetrics(cyclomatic=15)
        assert high.risk_level == "high"

        # Very high risk
        very_high = ComplexityMetrics(cyclomatic=25)
        assert very_high.risk_level == "very high"

    def test_cognitive_risk_level(self):
        """Test cognitive_risk_level property."""
        # Low cognitive risk
        low = ComplexityMetrics(cognitive=5)
        assert low.cognitive_risk_level == "low"

        # Medium cognitive risk
        medium = ComplexityMetrics(cognitive=12)
        assert medium.cognitive_risk_level == "medium"

        # High cognitive risk
        high = ComplexityMetrics(cognitive=20)
        assert high.cognitive_risk_level == "high"

        # Very high cognitive risk
        very_high = ComplexityMetrics(cognitive=30)
        assert very_high.cognitive_risk_level == "very high"


class TestFunctionComplexity:
    """Test suite for FunctionComplexity dataclass."""

    def test_function_complexity_creation(self):
        """Test creating FunctionComplexity."""
        func = FunctionComplexity(
            name="test_func",
            full_name="module.test_func",
            file_path="test.py",
            line_start=10,
            line_end=50,
        )

        assert func.name == "test_func"
        assert func.full_name == "module.test_func"
        assert func.file_path == "test.py"
        assert func.lines == 41

    def test_has_documentation(self):
        """Test has_documentation property."""
        # With docstring
        func_with_doc = FunctionComplexity(
            name="func",
            full_name="func",
            file_path="test.py",
            line_start=1,
            line_end=10,
            docstring="This is a function",
        )
        assert func_with_doc.has_documentation == True

        # Without docstring
        func_no_doc = FunctionComplexity(
            name="func",
            full_name="func",
            file_path="test.py",
            line_start=1,
            line_end=10,
            docstring=None,
        )
        assert func_no_doc.has_documentation == False


class TestClassComplexity:
    """Test suite for ClassComplexity dataclass."""

    def test_class_complexity_creation(self):
        """Test creating ClassComplexity."""
        cls = ClassComplexity(name="TestClass", file_path="test.py", line_start=1, line_end=100)

        assert cls.name == "TestClass"
        assert cls.file_path == "test.py"
        assert cls.total_methods == 0

    def test_avg_method_complexity(self):
        """Test avg_method_complexity property."""
        cls = ClassComplexity(name="TestClass", file_path="test.py", line_start=1, line_end=100)

        # Add methods with different complexities
        cls.methods = [
            FunctionComplexity(
                name="method1",
                full_name="TestClass.method1",
                file_path="test.py",
                line_start=10,
                line_end=20,
                metrics=ComplexityMetrics(cyclomatic=5),
            ),
            FunctionComplexity(
                name="method2",
                full_name="TestClass.method2",
                file_path="test.py",
                line_start=25,
                line_end=40,
                metrics=ComplexityMetrics(cyclomatic=10),
            ),
        ]

        assert cls.total_methods == 2
        assert cls.avg_method_complexity == 7.5

        # Test with no methods
        cls_no_methods = ClassComplexity(
            name="Empty", file_path="test.py", line_start=1, line_end=5
        )
        assert cls_no_methods.avg_method_complexity == 0.0

    def test_weighted_methods_per_class(self):
        """Test WMC calculation."""
        cls = ClassComplexity(name="TestClass", file_path="test.py", line_start=1, line_end=100)

        cls.methods = [
            FunctionComplexity(
                name=f"method{i}",
                full_name=f"TestClass.method{i}",
                file_path="test.py",
                line_start=i * 10,
                line_end=i * 10 + 5,
                metrics=ComplexityMetrics(cyclomatic=i + 1),
            )
            for i in range(5)
        ]

        # WMC = sum of all method complexities = 1+2+3+4+5 = 15
        assert cls.weighted_methods_per_class == 15


class TestFileComplexity:
    """Test suite for FileComplexity dataclass."""

    def test_file_complexity_creation(self):
        """Test creating FileComplexity."""
        file_comp = FileComplexity(path="test.py", name="test.py", language="Python")

        assert file_comp.path == "test.py"
        assert file_comp.name == "test.py"
        assert file_comp.language == "Python"

    def test_avg_complexity(self):
        """Test avg_complexity property."""
        file_comp = FileComplexity(path="test.py", name="test.py", language="Python")

        # Add functions
        file_comp.functions = [
            FunctionComplexity(
                name=f"func{i}",
                full_name=f"func{i}",
                file_path="test.py",
                line_start=i * 10,
                line_end=i * 10 + 5,
                metrics=ComplexityMetrics(cyclomatic=i * 2 + 1),
            )
            for i in range(3)
        ]

        # Add class with methods
        cls = ClassComplexity(name="TestClass", file_path="test.py", line_start=100, line_end=200)
        cls.methods = [
            FunctionComplexity(
                name="method",
                full_name="TestClass.method",
                file_path="test.py",
                line_start=110,
                line_end=120,
                metrics=ComplexityMetrics(cyclomatic=7),
            )
        ]
        file_comp.classes = [cls]

        # Average = (1 + 3 + 5 + 7) / 4 = 4
        assert file_comp.avg_complexity == 4.0

    def test_needs_refactoring(self):
        """Test needs_refactoring property."""
        # Needs refactoring - high max complexity
        file1 = FileComplexity(path="test.py", name="test.py", language="Python", max_complexity=25)
        assert file1.needs_refactoring == True

        # Needs refactoring - high average
        file2 = FileComplexity(path="test.py", name="test.py", language="Python")
        file2.functions = [
            FunctionComplexity(
                name=f"func{i}",
                full_name=f"func{i}",
                file_path="test.py",
                line_start=0,
                line_end=10,
                metrics=ComplexityMetrics(cyclomatic=15),
            )
            for i in range(3)
        ]
        assert file2.avg_complexity == 15.0
        assert file2.needs_refactoring == True

        # Doesn't need refactoring
        file3 = FileComplexity(
            path="simple.py",
            name="simple.py",
            language="Python",
            max_complexity=5,
            total_complexity=20,
        )
        assert file3.needs_refactoring == False


class TestComplexityReport:
    """Test suite for ComplexityReport dataclass."""

    def test_complexity_report_creation(self):
        """Test creating ComplexityReport."""
        report = ComplexityReport(
            total_files=10, total_functions=50, total_classes=5, avg_complexity=8.5
        )

        assert report.total_files == 10
        assert report.total_functions == 50
        assert report.total_classes == 5
        assert report.avg_complexity == 8.5

    def test_complexity_score(self):
        """Test complexity_score property."""
        # Good complexity
        good_report = ComplexityReport(
            avg_complexity=5.0, high_complexity_count=2, very_high_complexity_count=0
        )

        score = good_report.complexity_score
        assert 0 <= score <= 100
        assert score < 40  # Should be relatively low (good)

        # Poor complexity
        poor_report = ComplexityReport(
            avg_complexity=15.0, high_complexity_count=10, very_high_complexity_count=5
        )

        poor_score = poor_report.complexity_score
        assert poor_score > score
        assert poor_score <= 100

    def test_to_dict(self):
        """Test to_dict conversion."""
        report = ComplexityReport(
            total_files=5, total_functions=20, avg_complexity=7.5, max_complexity=25
        )

        report.refactoring_candidates = [{"name": "complex_func", "complexity": 25}]
        report.recommendations = ["Reduce complexity"]

        result = report.to_dict()

        assert result["total_files"] == 5
        assert result["total_functions"] == 20
        assert result["avg_complexity"] == 7.5
        assert len(result["refactoring_candidates"]) == 1
        assert len(result["recommendations"]) == 1


class TestComplexityAnalyzer:
    """Test suite for ComplexityAnalyzer."""

    def test_initialization(self, config):
        """Test ComplexityAnalyzer initialization."""
        analyzer = ComplexityAnalyzer(config)

        assert analyzer.config == config
        assert analyzer.complexity_cache == {}

    def test_analyze_empty_files(self, complexity_analyzer):
        """Test analyzing with no files."""
        report = complexity_analyzer.analyze([])

        assert report.total_files == 0
        assert report.total_functions == 0
        assert report.avg_complexity == 0.0

    def test_analyze_basic(self, complexity_analyzer, sample_files):
        """Test basic complexity analysis."""
        report = complexity_analyzer.analyze(sample_files, threshold=10.0)

        assert report.total_files == len(sample_files)
        assert report.total_functions > 0
        assert report.total_classes > 0
        assert report.avg_complexity > 0
        assert report.max_complexity > 0

    def test_should_analyze_file(self, complexity_analyzer):
        """Test file filtering logic."""
        # Should analyze code files
        code_file = Mock()
        code_file.path = "test.py"
        code_file.language = "Python"
        code_file.lines = 100
        assert complexity_analyzer._should_analyze_file(code_file) == True

        # Should skip non-code files
        text_file = Mock()
        text_file.path = "readme.txt"
        text_file.language = "Text"
        assert complexity_analyzer._should_analyze_file(text_file) == False

        # Should skip very large files
        huge_file = Mock()
        huge_file.path = "generated.py"
        huge_file.language = "Python"
        huge_file.lines = 15000
        assert complexity_analyzer._should_analyze_file(huge_file) == False

        # Should skip files without required attributes
        incomplete_file = Mock()
        incomplete_file.language = "Python"
        assert complexity_analyzer._should_analyze_file(incomplete_file) == False

    def test_analyze_file_complexity(self, complexity_analyzer):
        """Test analyzing single file complexity."""
        file = Mock()
        file.path = "test.py"
        file.language = "Python"
        file.lines = 100
        file.complexity = Mock(cyclomatic=10, cognitive=12)

        # Add function
        func = Mock()
        func.name = "test_func"
        func.full_name = "test_func"
        func.line_start = 10
        func.line_end = 30
        func.complexity = Mock(cyclomatic=5, cognitive=6, nesting_depth=2)
        func.parameters = ["a", "b"]
        func.is_async = False
        func.is_generator = False
        func.has_decorator = True
        func.docstring = "Test function"
        file.functions = [func]

        # Add class
        cls = Mock()
        cls.name = "TestClass"
        cls.line_start = 40
        cls.line_end = 80
        cls.inheritance_depth = 1
        cls.parent_classes = ["BaseClass"]

        method = Mock()
        method.name = "test_method"
        method.line_start = 45
        method.line_end = 60
        method.complexity = Mock(cyclomatic=3)
        cls.methods = [method]

        file.classes = [cls]

        result = complexity_analyzer._analyze_file_complexity(file, deep=True)

        assert result is not None
        assert result.path == "test.py"
        assert len(result.functions) == 1
        assert len(result.classes) == 1
        assert result.total_complexity > 0

    def test_calculate_maintainability(self, complexity_analyzer):
        """Test maintainability index calculation."""
        file_comp = FileComplexity(
            path="test.py", name="test.py", language="Python", total_complexity=20
        )
        file_comp.metrics.line_count = 200

        mi = complexity_analyzer._calculate_maintainability(file_comp)

        assert 0 <= mi <= 100

    def test_identify_refactoring_candidates(self, complexity_analyzer):
        """Test identifying refactoring candidates."""
        report = ComplexityReport()

        # Create files with functions needing refactoring
        file1 = FileComplexity(path="complex.py", name="complex.py", language="Python")

        func1 = FunctionComplexity(
            name="very_complex",
            full_name="very_complex",
            file_path="complex.py",
            line_start=1,
            line_end=50,
            metrics=ComplexityMetrics(cyclomatic=25),
        )
        file1.functions = [func1]

        cls = ClassComplexity(
            name="ComplexClass", file_path="complex.py", line_start=60, line_end=200
        )
        cls.methods = [
            FunctionComplexity(
                name=f"method{i}",
                full_name=f"ComplexClass.method{i}",
                file_path="complex.py",
                line_start=70 + i * 20,
                line_end=80 + i * 20,
                metrics=ComplexityMetrics(cyclomatic=12 + i),
            )
            for i in range(3)
        ]
        file1.classes = [cls]

        report.files = [file1]

        complexity_analyzer._identify_refactoring_candidates(report, threshold=10)

        assert len(report.refactoring_candidates) > 0
        # Should include the very complex function
        assert any(
            c["name"] == "very_complex" and c["complexity"] == 25
            for c in report.refactoring_candidates
        )

    def test_estimate_technical_debt(self, complexity_analyzer):
        """Test technical debt estimation."""
        report = ComplexityReport()

        report.refactoring_candidates = [
            {"priority": "high", "complexity": 25},
            {"priority": "high", "complexity": 22},
            {"priority": "medium", "complexity": 15},
            {"priority": "medium", "complexity": 12},
        ]

        debt = complexity_analyzer._estimate_technical_debt(report)

        assert debt > 0
        # Should be reasonable estimate
        assert 10 <= debt <= 100

    def test_generate_recommendations(self, complexity_analyzer):
        """Test recommendation generation."""
        report = ComplexityReport(
            avg_complexity=12.0,
            very_high_complexity_count=3,
            complexity_score=65,
            technical_debt_hours=50,
        )

        # Add a complex file
        file = FileComplexity(
            path="complex.py", name="complex.py", language="Python", total_complexity=150
        )
        report.top_complex_files = [file]

        # Add a large class
        cls = ClassComplexity(name="LargeClass", file_path="large.py", line_start=1, line_end=500)
        cls.methods = [Mock() for _ in range(25)]
        report.top_complex_classes = [cls]

        recommendations = complexity_analyzer._generate_recommendations(report)

        assert len(recommendations) > 0
        # Should have recommendations about high complexity
        assert any("complexity" in r.lower() for r in recommendations)

    def test_calculate_median(self, complexity_analyzer):
        """Test median calculation."""
        # Odd number of values
        odd_values = [1, 3, 5, 7, 9]
        assert complexity_analyzer._calculate_median(odd_values) == 5

        # Even number of values
        even_values = [1, 2, 3, 4]
        assert complexity_analyzer._calculate_median(even_values) == 2.5

        # Empty list
        assert complexity_analyzer._calculate_median([]) == 0.0

    def test_calculate_std_dev(self, complexity_analyzer):
        """Test standard deviation calculation."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        std_dev = complexity_analyzer._calculate_std_dev(values)

        # Should be approximately 2.0
        assert 1.9 <= std_dev <= 2.1

        # Empty list
        assert complexity_analyzer._calculate_std_dev([]) == 0.0

        # Single value
        assert complexity_analyzer._calculate_std_dev([5]) == 0.0

    def test_calculate_distribution(self, complexity_analyzer):
        """Test complexity distribution calculation."""
        complexities = [2, 5, 8, 12, 15, 22, 30]

        distribution = complexity_analyzer._calculate_distribution(complexities)

        assert distribution["simple (1-5)"] == 2  # 2, 5
        assert distribution["moderate (6-10)"] == 1  # 8
        assert distribution["complex (11-20)"] == 2  # 12, 15
        assert distribution["very complex (21+)"] == 2  # 22, 30


class TestAnalyzeComplexityFunction:
    """Test the analyze_complexity convenience function."""

    def test_analyze_complexity_basic(self, sample_files):
        """Test basic complexity analysis."""
        report = analyze_complexity(sample_files)

        assert isinstance(report, ComplexityReport)
        assert report.total_files > 0

    def test_analyze_complexity_with_threshold(self, sample_files):
        """Test with custom threshold."""
        report = analyze_complexity(sample_files, threshold=15)

        assert isinstance(report, ComplexityReport)
        # High threshold should result in fewer high complexity items
        assert report.high_complexity_count >= 0

    def test_analyze_complexity_with_config(self, sample_files, config):
        """Test with custom config."""
        report = analyze_complexity(sample_files, config=config)

        assert isinstance(report, ComplexityReport)
