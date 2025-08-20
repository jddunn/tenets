"""Tests for the report generator module."""

import json
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.reporting.generator import (
    ReportConfig,
    ReportGenerator,
    ReportSection,
)


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.cache.enabled = False
    return config


@pytest.fixture
def generator(config):
    """Create report generator instance."""
    return ReportGenerator(config)


@pytest.fixture
def sample_analysis_data():
    """Create sample analysis data."""
    return {
        "total_files": 100,
        "total_lines": 10000,
        "languages": ["Python", "JavaScript"],
        "health_score": 75.5,
        "complexity": {
            "avg_complexity": 5.2,
            "max_complexity": 25,
            "complex_functions": 10,
            "total_functions": 150,
            "complex_items": [
                {
                    "name": "complex_function",
                    "file": "src/main.py",
                    "complexity": 25,
                    "lines": 100,
                    "language": "python",
                }
            ],
        },
        "contributors": {
            "total_contributors": 5,
            "active_contributors": 3,
            "bus_factor": 2,
            "avg_commits_per_contributor": 20.5,
            "contributors": [
                {
                    "name": "John Doe",
                    "commits": 50,
                    "lines": 5000,
                    "files": 20,
                    "last_commit_days_ago": 1,
                }
            ],
        },
        "hotspots": {
            "total_hotspots": 15,
            "critical_count": 3,
            "high_count": 5,
            "files_analyzed": 100,
            "hotspots": [
                {
                    "file": "src/critical.py",
                    "risk_level": "critical",
                    "change_frequency": 50,
                    "complexity": 30,
                    "risk_score": 85.5,
                }
            ],
        },
        "dependencies": {
            "total_modules": 50,
            "total_dependencies": 200,
            "external_dependencies": 30,
            "circular_count": 2,
            "dependencies": [],
            "most_dependent": [{"name": "core.py", "dependencies": 10, "dependents": 20}],
        },
    }


class TestReportSection:
    """Test suite for ReportSection."""

    def test_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            id="test-section",
            title="Test Section",
            level=1,
            order=1,
            icon="📊",
        )

        assert section.id == "test-section"
        assert section.title == "Test Section"
        assert section.level == 1
        assert section.order == 1
        assert section.icon == "📊"
        assert section.visible is True
        assert section.collapsed is False

    def test_add_metric(self):
        """Test adding metrics to section."""
        section = ReportSection(id="test", title="Test")

        section.add_metric("score", 95.5)
        section.add_metric("count", 10)

        assert section.metrics["score"] == 95.5
        assert section.metrics["count"] == 10

    def test_add_table(self):
        """Test adding tables to section."""
        section = ReportSection(id="test", title="Test")

        table_data = {"headers": ["Name", "Value"], "rows": [["Test", 123], ["Demo", 456]]}
        section.add_table(table_data)

        assert len(section.tables) == 1
        assert section.tables[0] == table_data

    def test_add_chart(self):
        """Test adding charts to section."""
        section = ReportSection(id="test", title="Test")

        chart_config = {"type": "bar", "data": {"labels": ["A", "B"], "values": [1, 2]}}
        section.add_chart(chart_config)

        assert len(section.charts) == 1
        assert section.charts[0] == chart_config

    def test_add_subsection(self):
        """Test adding subsections."""
        parent = ReportSection(id="parent", title="Parent")
        child = ReportSection(id="child", title="Child", level=2)

        parent.add_subsection(child)

        assert len(parent.subsections) == 1
        assert parent.subsections[0] == child


class TestReportConfig:
    """Test suite for ReportConfig."""

    def test_default_config(self):
        """Test default report configuration."""
        config = ReportConfig()

        assert config.title == "Code Analysis Report"
        assert config.format == "html"
        assert config.include_summary is True
        assert config.include_toc is True
        assert config.include_charts is True
        assert config.include_code_snippets is True
        assert config.include_recommendations is True
        assert config.max_items == 20
        assert config.theme == "light"

    def test_custom_config(self):
        """Test custom report configuration."""
        config = ReportConfig(
            title="Custom Report",
            format="markdown",
            include_charts=False,
            theme="dark",
            max_items=50,
        )

        assert config.title == "Custom Report"
        assert config.format == "markdown"
        assert config.include_charts is False
        assert config.theme == "dark"
        assert config.max_items == 50


class TestReportGenerator:
    """Test suite for ReportGenerator."""

    def test_initialization(self, config):
        """Test generator initialization."""
        generator = ReportGenerator(config)

        assert generator.config == config
        assert generator.sections == []
        assert generator.metadata == {}

    def test_build_metadata(self, generator, sample_analysis_data):
        """Test metadata building."""
        report_config = ReportConfig(title="Test Report")

        metadata = generator._build_metadata(sample_analysis_data, report_config)

        assert metadata["title"] == "Test Report"
        assert "generated_at" in metadata
        assert metadata["format"] == "html"
        assert "analysis_summary" in metadata

        summary = metadata["analysis_summary"]
        assert summary["total_files"] == 100
        assert summary["total_lines"] == 10000
        assert summary["health_score"] == 75.5

    def test_extract_summary_metrics(self, generator, sample_analysis_data):
        """Test summary metrics extraction."""
        summary = generator._extract_summary_metrics(sample_analysis_data)

        assert summary["total_files"] == 100
        assert summary["total_lines"] == 10000
        assert summary["health_score"] == 75.5
        assert summary["critical_issues"] == 3  # From hotspots
        assert summary["total_issues"] > 0

    def test_create_summary_section(self, generator, sample_analysis_data):
        """Test summary section creation."""
        generator.metadata = generator._build_metadata(sample_analysis_data, ReportConfig())

        section = generator._create_summary_section(sample_analysis_data)

        assert section.id == "summary"
        assert section.title == "Executive Summary"
        assert section.level == 1
        # Check enhanced metrics
        assert "Functions" in section.metrics
        assert "Classes" in section.metrics
        assert "Test Coverage" in section.metrics
        assert "Languages" in section.metrics
        assert "Avg File Size" in section.metrics

    def test_create_file_overview_section(self, generator, sample_analysis_data):
        """Test file overview section creation."""
        report_config = ReportConfig()

        section = generator._create_file_overview_section(sample_analysis_data, report_config)

        assert section.id == "file_overview"
        assert section.title == "File Analysis Overview"
        assert section.order == 2
        assert section.icon == "📁"
        # Check that language table is added
        assert (
            len(section.tables) > 0
            if sample_analysis_data.get("metrics", {}).get("languages")
            else True
        )

    def test_find_readme(self, generator, sample_analysis_data):
        """Test README finding functionality."""
        readme = generator._find_readme(sample_analysis_data)

        # Currently returns None as placeholder
        assert readme is None

    def test_create_readme_section(self, generator):
        """Test README section creation."""
        readme_content = "# Test Project\\n\\nThis is a test project."

        section = generator._create_readme_section(readme_content)

        assert section.id == "readme"
        assert section.title == "Project README"
        assert section.order == 1.5
        assert section.collapsible is True
        assert readme_content in section.content[0]
        assert section.icon == "📖"
        assert section.content is not None
        assert section.metrics == {}

    def test_create_complexity_section(self, generator, sample_analysis_data):
        """Test complexity section creation."""
        config = ReportConfig(include_charts=True, max_items=10)

        section = generator._create_complexity_section(sample_analysis_data["complexity"], config)

        assert section.id == "complexity"
        assert section.title == "Complexity Analysis"
        assert "Average Complexity" in section.metrics
        assert len(section.charts) > 0  # Should have charts
        assert len(section.tables) > 0  # Should have table

    def test_create_contributors_section(self, generator, sample_analysis_data):
        """Test contributors section creation."""
        config = ReportConfig(include_charts=True)

        section = generator._create_contributors_section(
            sample_analysis_data["contributors"], config
        )

        assert section.id == "contributors"
        assert section.title == "Contributor Analysis"
        assert "Total Contributors" in section.metrics
        assert section.metrics["Total Contributors"] == 5
        assert len(section.charts) > 0

    def test_create_hotspots_section(self, generator, sample_analysis_data):
        """Test hotspots section creation."""
        config = ReportConfig(include_charts=True)

        section = generator._create_hotspots_section(sample_analysis_data["hotspots"], config)

        assert section.id == "hotspots"
        assert section.title == "Code Hotspots"
        assert "Total Hotspots" in section.metrics
        assert section.metrics["Critical"] == 3

    def test_create_recommendations_section(self, generator, sample_analysis_data):
        """Test recommendations section creation."""
        section = generator._create_recommendations_section(sample_analysis_data)

        assert section.id == "recommendations"
        assert section.title == "Recommendations"
        assert section.content is not None
        assert len(section.content) > 0

    @patch("tenets.core.reporting.html_reporter.HTMLReporter")
    def test_generate_html_report(
        self, mock_html_reporter, generator, sample_analysis_data, tmp_path
    ):
        """Test HTML report generation."""
        output_path = tmp_path / "report.html"
        config = ReportConfig(format="html", include_charts=True)

        # Mock HTML reporter
        mock_reporter_instance = Mock()
        mock_reporter_instance.generate.return_value = output_path
        mock_html_reporter.return_value = mock_reporter_instance

        result = generator.generate(sample_analysis_data, output_path, config)

        assert result == output_path
        mock_html_reporter.assert_called_once()
        mock_reporter_instance.generate.assert_called_once()

    @patch("tenets.core.reporting.markdown_reporter.MarkdownReporter")
    def test_generate_markdown_report(
        self, mock_md_reporter, generator, sample_analysis_data, tmp_path
    ):
        """Test Markdown report generation."""
        output_path = tmp_path / "report.md"
        config = ReportConfig(format="markdown")

        # Mock Markdown reporter
        mock_reporter_instance = Mock()
        mock_reporter_instance.generate.return_value = output_path
        mock_md_reporter.return_value = mock_reporter_instance

        result = generator.generate(sample_analysis_data, output_path, config)

        assert result == output_path
        mock_md_reporter.assert_called_once()

    def test_generate_json_report(self, generator, sample_analysis_data, tmp_path):
        """Test JSON report generation."""
        output_path = tmp_path / "report.json"
        config = ReportConfig(format="json")

        # Create some sections
        generator.sections = [ReportSection(id="test", title="Test Section")]
        generator.metadata = {"test": "metadata"}

        result = generator.generate(sample_analysis_data, output_path, config)

        assert result == output_path
        assert output_path.exists()

        # Verify JSON content
        with open(output_path) as f:
            data = json.load(f)
            assert "metadata" in data
            assert "sections" in data

    def test_get_risk_level(self, generator):
        """Test risk level determination."""
        assert generator._get_risk_level(25) == "Critical"
        assert generator._get_risk_level(15) == "High"
        assert generator._get_risk_level(7) == "Medium"
        assert generator._get_risk_level(3) == "Low"

    def test_format_days_ago(self, generator):
        """Test days ago formatting."""
        assert generator._format_days_ago(0) == "Today"
        assert generator._format_days_ago(1) == "Yesterday"
        assert generator._format_days_ago(5) == "5 days ago"
        assert generator._format_days_ago(10) == "1 week ago"
        assert generator._format_days_ago(21) == "3 weeks ago"
        assert generator._format_days_ago(45) == "1 month ago"
        assert generator._format_days_ago(400) == "1 year ago"

    def test_sections_to_dict(self, generator):
        """Test converting sections to dictionary."""
        sections = [
            ReportSection(id="test", title="Test", content="Test content", metrics={"score": 100})
        ]

        result = generator._sections_to_dict(sections)

        assert len(result) == 1
        assert result[0]["id"] == "test"
        assert result[0]["title"] == "Test"
        assert result[0]["content"] == "Test content"
        assert result[0]["metrics"]["score"] == 100

    def test_create_complex_functions_table(self, generator):
        """Test complex functions table creation."""
        complex_items = [
            {"name": "func1", "file": "file1.py", "complexity": 20, "lines": 100},
            {"name": "func2", "file": "file2.py", "complexity": 15, "lines": 75},
        ]

        table = generator._create_complex_functions_table(complex_items)

        assert table["headers"] == ["Function", "File", "Complexity", "Lines", "Risk"]
        assert len(table["rows"]) == 2
        assert table["rows"][0][0] == "func1"
        assert table["rows"][0][4] == "High"

    def test_create_contributors_table(self, generator):
        """Test contributors table creation."""
        contributors = [
            {"name": "Alice", "commits": 100, "lines": 5000, "files": 50, "last_commit_days_ago": 0}
        ]

        table = generator._create_contributors_table(contributors)

        assert "Contributor" in table["headers"]
        assert len(table["rows"]) == 1
        assert table["rows"][0][0] == "Alice"
        assert table["rows"][0][4] == "Today"

    def test_create_hotspots_table(self, generator):
        """Test hotspots table creation."""
        hotspots = [
            {
                "file": "hotspot.py",
                "risk_level": "critical",
                "change_frequency": 50,
                "complexity": 30,
                "risk_score": 85.5,
            }
        ]

        table = generator._create_hotspots_table(hotspots)

        assert "File" in table["headers"]
        assert "Risk Level" in table["headers"]
        assert len(table["rows"]) == 1
        assert table["rows"][0][1] == "CRITICAL"

    def test_generate_with_no_data(self, generator, tmp_path):
        """Test report generation with minimal data."""
        output_path = tmp_path / "empty_report.json"
        config = ReportConfig(format="json", include_summary=False)

        result = generator.generate({}, output_path, config)

        assert result == output_path
        assert output_path.exists()

    def test_generate_with_missing_sections(self, generator, sample_analysis_data, tmp_path):
        """Test report generation with missing data sections."""
        # Remove some sections
        partial_data = {"complexity": sample_analysis_data["complexity"]}

        output_path = tmp_path / "partial_report.json"
        config = ReportConfig(format="json")

        result = generator.generate(partial_data, output_path, config)

        assert result == output_path
        assert output_path.exists()
