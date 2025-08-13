"""Tests for metrics calculation module."""

from unittest.mock import Mock

import pytest

from tenets.config import TenetsConfig
from tenets.core.examiner.metrics import (
    MetricsCalculator,
    MetricsReport,
    calculate_metrics,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def metrics_calculator(config):
    """Create MetricsCalculator instance."""
    return MetricsCalculator(config)


@pytest.fixture
def sample_files():
    """Create sample analyzed file objects."""
    files = []

    # Python file with various metrics
    file1 = Mock()
    file1.path = "main.py"
    file1.language = "Python"
    file1.lines = 150
    file1.blank_lines = 20
    file1.comment_lines = 30
    file1.functions = [Mock(name="func1"), Mock(name="func2")]
    file1.classes = [Mock(name="Class1")]
    file1.imports = ["os", "sys", "pathlib"]
    file1.complexity = Mock(cyclomatic=15)
    files.append(file1)

    # JavaScript file
    file2 = Mock()
    file2.path = "app.js"
    file2.language = "JavaScript"
    file2.lines = 200
    file2.blank_lines = 30
    file2.comment_lines = 40
    file2.functions = [Mock(name="init"), Mock(name="render"), Mock(name="update")]
    file2.classes = []
    file2.imports = ["react", "redux"]
    file2.complexity = Mock(cyclomatic=8)
    files.append(file2)

    # Large complex file
    file3 = Mock()
    file3.path = "complex.py"
    file3.language = "Python"
    file3.lines = 1500
    file3.blank_lines = 100
    file3.comment_lines = 200
    file3.functions = [Mock(name=f"func{i}") for i in range(20)]
    file3.classes = [Mock(name=f"Class{i}") for i in range(5)]
    file3.imports = ["numpy", "pandas", "sklearn"]
    file3.complexity = Mock(cyclomatic=35)
    files.append(file3)

    # Test file
    file4 = Mock()
    file4.path = "test_main.py"
    file4.language = "Python"
    file4.lines = 100
    file4.blank_lines = 15
    file4.comment_lines = 10
    file4.functions = [Mock(name="test_1"), Mock(name="test_2")]
    file4.classes = []
    file4.imports = ["pytest", "unittest"]
    file4.complexity = Mock(cyclomatic=3)
    files.append(file4)

    return files


class TestMetricsReport:
    """Test suite for MetricsReport dataclass."""

    def test_metrics_report_creation(self):
        """Test creating MetricsReport."""
        report = MetricsReport()

        assert report.total_files == 0
        assert report.total_lines == 0
        assert report.avg_complexity == 0.0
        assert report.maintainability_index == 0.0

    def test_code_to_comment_ratio(self):
        """Test code_to_comment_ratio property."""
        report = MetricsReport(total_code_lines=1000, total_comment_lines=200)

        assert report.code_to_comment_ratio == 5.0

        # Test with no comments
        report_no_comments = MetricsReport(total_code_lines=1000, total_comment_lines=0)
        assert report_no_comments.code_to_comment_ratio == float("inf")

    def test_quality_score(self):
        """Test quality_score calculation."""
        # Good quality
        good_report = MetricsReport(
            avg_complexity=5.0,
            documentation_ratio=0.2,
            avg_file_size=200,
            code_duplication_ratio=0.05,
            test_coverage=0.85,
        )

        score = good_report.quality_score
        assert 70 <= score <= 100

        # Poor quality
        poor_report = MetricsReport(
            avg_complexity=15.0,
            documentation_ratio=0.05,
            avg_file_size=700,
            code_duplication_ratio=0.2,
            test_coverage=0.3,
        )

        poor_score = poor_report.quality_score
        assert poor_score < score
        assert 0 <= poor_score <= 100

    def test_to_dict(self):
        """Test to_dict conversion."""
        report = MetricsReport(
            total_files=10,
            total_lines=1500,
            avg_complexity=8.5,
            max_complexity=25.0,
            test_coverage=0.75,
        )

        report.languages = {"Python": {"files": 8}, "JavaScript": {"files": 2}}
        report.largest_files = [{"path": "big.py", "lines": 500}]

        result = report.to_dict()

        assert result["total_files"] == 10
        assert result["total_lines"] == 1500
        assert result["avg_complexity"] == 8.5
        assert "Python" in result["languages"]
        assert len(result["largest_files"]) == 1


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""

    def test_initialization(self, config):
        """Test MetricsCalculator initialization."""
        calculator = MetricsCalculator(config)

        assert calculator.config == config

    def test_calculate_empty_files(self, metrics_calculator):
        """Test calculating metrics with no files."""
        report = metrics_calculator.calculate([])

        assert report.total_files == 0
        assert report.total_lines == 0
        assert report.avg_complexity == 0.0

    def test_calculate_basic_metrics(self, metrics_calculator, sample_files):
        """Test calculating basic metrics."""
        report = metrics_calculator.calculate(sample_files)

        assert report.total_files == len(sample_files)
        assert report.total_lines == sum(f.lines for f in sample_files)
        assert report.total_functions == sum(len(f.functions) for f in sample_files)
        assert report.total_classes == sum(len(f.classes) for f in sample_files)
        assert report.total_imports == sum(len(f.imports) for f in sample_files)

        # Check complexity metrics
        assert report.avg_complexity > 0
        assert report.max_complexity == 35  # From complex.py
        assert report.min_complexity == 3  # From test_main.py

    def test_calculate_distributions(self, metrics_calculator, sample_files):
        """Test distribution calculations."""
        report = metrics_calculator.calculate(sample_files)

        # Size distribution
        assert "tiny (1-50)" in report.size_distribution
        assert "small (51-200)" in report.size_distribution
        assert "huge (1000+)" in report.size_distribution

        # At least one file in each relevant bucket
        assert report.size_distribution["small (51-200)"] >= 2  # main.py, test_main.py
        assert report.size_distribution["huge (1000+)"] >= 1  # complex.py

        # Complexity distribution
        assert "simple (1-5)" in report.complexity_distribution
        assert "very complex (21+)" in report.complexity_distribution
        assert report.complexity_distribution["very complex (21+)"] >= 1  # complex.py

    def test_identify_top_items(self, metrics_calculator, sample_files):
        """Test identifying top files."""
        report = metrics_calculator.calculate(sample_files)

        # Largest files
        assert len(report.largest_files) > 0
        assert report.largest_files[0]["path"] == "complex.py"
        assert report.largest_files[0]["lines"] == 1500

        # Most complex files
        assert len(report.most_complex_files) > 0
        assert report.most_complex_files[0]["path"] == "complex.py"
        assert report.most_complex_files[0]["complexity"] == 35

        # Most imported modules
        assert len(report.most_imported_modules) > 0

    def test_calculate_file_metrics(self, metrics_calculator):
        """Test calculating metrics for a single file."""
        file = Mock()
        file.path = "test.py"
        file.language = "Python"
        file.lines = 100
        file.blank_lines = 10
        file.comment_lines = 20
        file.functions = [Mock(), Mock()]
        file.classes = [Mock()]
        file.imports = ["os", "sys"]
        file.complexity = Mock(cyclomatic=10)

        metrics = metrics_calculator.calculate_file_metrics(file)

        assert metrics["lines"] == 100
        assert metrics["blank_lines"] == 10
        assert metrics["comment_lines"] == 20
        assert metrics["code_lines"] == 70
        assert metrics["functions"] == 2
        assert metrics["classes"] == 1
        assert metrics["imports"] == 2
        assert metrics["complexity"] == 10
        assert metrics["documentation_ratio"] == 20 / 70

    def test_language_metrics(self, metrics_calculator, sample_files):
        """Test language-specific metrics."""
        report = metrics_calculator.calculate(sample_files)

        assert "Python" in report.languages
        assert "JavaScript" in report.languages

        python_metrics = report.languages["Python"]
        assert python_metrics["files"] == 3  # main.py, complex.py, test_main.py
        assert python_metrics["lines"] == 1750  # 150 + 1500 + 100

        js_metrics = report.languages["JavaScript"]
        assert js_metrics["files"] == 1
        assert js_metrics["lines"] == 200

    def test_derived_metrics(self, metrics_calculator, sample_files):
        """Test calculation of derived metrics."""
        report = metrics_calculator.calculate(sample_files)

        # Documentation ratio
        assert report.documentation_ratio > 0

        # Test coverage estimate
        assert 0 <= report.test_coverage <= 1

        # Code duplication estimate
        assert 0 <= report.code_duplication_ratio <= 0.5

        # Technical debt score
        assert 0 <= report.technical_debt_score <= 100

        # Maintainability index
        assert 0 <= report.maintainability_index <= 100

    def test_variance_calculation(self, metrics_calculator):
        """Test variance calculation method."""
        values = [1, 2, 3, 4, 5]
        variance = metrics_calculator._calculate_variance(values)

        assert variance == 2.0  # Variance of [1,2,3,4,5] is 2

        # Test empty list
        empty_variance = metrics_calculator._calculate_variance([])
        assert empty_variance == 0.0

        # Test single value
        single_variance = metrics_calculator._calculate_variance([5])
        assert single_variance == 0.0

    def test_quality_indicators(self, metrics_calculator):
        """Test quality indicator estimation."""
        # Create file with security patterns
        file = Mock()
        file.path = "auth.py"
        file.content = "password = 'secret123'\ntoken = get_token()\neval(user_input)"
        file.language = "Python"
        file.lines = 100
        file.blank_lines = 10
        file.comment_lines = 5
        file.functions = []
        file.classes = []
        file.imports = []
        file.complexity = Mock(cyclomatic=5)

        report = metrics_calculator.calculate([file])

        # Should detect security mentions and adjust technical debt
        assert report.technical_debt_score > 0

    def test_files_without_complexity(self, metrics_calculator):
        """Test handling files without complexity data."""
        file = Mock()
        file.path = "simple.txt"
        file.language = "Text"
        file.lines = 50
        file.blank_lines = 5
        file.comment_lines = 0
        file.functions = []
        file.classes = []
        file.imports = []
        file.complexity = None

        report = metrics_calculator.calculate([file])

        assert report.total_files == 1
        assert report.avg_complexity == 0.0

    def test_edge_cases(self, metrics_calculator):
        """Test edge cases in metrics calculation."""
        # File with all comments
        file1 = Mock()
        file1.path = "comments.py"
        file1.language = "Python"
        file1.lines = 100
        file1.blank_lines = 0
        file1.comment_lines = 100
        file1.functions = []
        file1.classes = []
        file1.imports = []
        file1.complexity = Mock(cyclomatic=1)

        # File with no content
        file2 = Mock()
        file2.path = "empty.py"
        file2.language = "Python"
        file2.lines = 0
        file2.blank_lines = 0
        file2.comment_lines = 0
        file2.functions = []
        file2.classes = []
        file2.imports = []
        file2.complexity = Mock(cyclomatic=1)

        report = metrics_calculator.calculate([file1, file2])

        assert report.total_files == 2
        assert report.total_code_lines == 0


class TestCalculateMetricsFunction:
    """Test the calculate_metrics convenience function."""

    def test_calculate_metrics_basic(self, sample_files):
        """Test basic metrics calculation."""
        report = calculate_metrics(sample_files)

        assert isinstance(report, MetricsReport)
        assert report.total_files == len(sample_files)

    def test_calculate_metrics_with_config(self, sample_files, config):
        """Test with custom config."""
        report = calculate_metrics(sample_files, config=config)

        assert isinstance(report, MetricsReport)

    def test_calculate_metrics_empty(self):
        """Test with empty file list."""
        report = calculate_metrics([])

        assert report.total_files == 0
        assert report.total_lines == 0
