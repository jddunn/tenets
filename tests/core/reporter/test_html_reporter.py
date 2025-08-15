"""Tests for the HTML report generator module."""

import base64
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest

from tenets.config import TenetsConfig
from tenets.core.reporting.html_reporter import (
    HTMLReporter,
    HTMLTemplate,
    create_html_report,
    create_dashboard,
)
from tenets.core.reporting.generator import ReportConfig, ReportSection


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.cache.enabled = False
    return config


@pytest.fixture
def reporter(config):
    """Create HTML reporter instance."""
    return HTMLReporter(config)


@pytest.fixture
def template():
    """Create HTML template instance."""
    return HTMLTemplate()


@pytest.fixture
def sample_sections():
    """Create sample report sections."""
    sections = [
        ReportSection(
            id="summary",
            title="Executive Summary",
            level=1,
            order=1,
            icon="📊",
            content=["The codebase is in good condition.", "10 issues found."],
            metrics={"Health Score": 85.5, "Files": 100},
        ),
        ReportSection(
            id="complexity",
            title="Complexity Analysis",
            level=1,
            order=2,
            icon="🔍",
            tables=[
                {"headers": ["Function", "Complexity"], "rows": [["func1", 10], ["func2", 20]]}
            ],
            charts=[
                {
                    "type": "bar",
                    "data": {"labels": ["Low", "Medium", "High"], "values": [50, 30, 20]},
                }
            ],
        ),
    ]
    return sections


@pytest.fixture
def sample_metadata():
    """Create sample metadata."""
    return {
        "title": "Test Report",
        "generated_at": datetime.now().isoformat(),
        "analysis_summary": {
            "health_score": 85.5,
            "total_files": 100,
            "total_lines": 10000,
            "critical_issues": 2,
            "total_issues": 10,
        },
    }


class TestHTMLTemplate:
    """Test suite for HTMLTemplate."""

    def test_initialization_default(self):
        """Test template initialization with defaults."""
        template = HTMLTemplate()

        assert template.theme == "default"
        assert template.custom_css is None
        assert template.include_charts is True

    def test_initialization_custom(self):
        """Test template initialization with custom settings."""
        template = HTMLTemplate(
            theme="dark", custom_css="body { color: red; }", include_charts=False
        )

        assert template.theme == "dark"
        assert template.custom_css == "body { color: red; }"
        assert template.include_charts is False

    def test_get_base_template(self, template):
        """Test base template generation."""
        html = template.get_base_template()

        assert "<!DOCTYPE html>" in html
        assert "{title}" in html
        assert "{content}" in html
        assert "{header}" in html
        assert "{footer}" in html

    def test_get_styles(self, template):
        """Test CSS styles generation."""
        styles = template.get_styles()

        assert "<style>" in styles
        assert ":root {" in styles
        assert "--primary-color:" in styles
        assert ".container {" in styles
        assert ".section {" in styles

    def test_get_styles_with_custom_css(self):
        """Test CSS with custom styles."""
        template = HTMLTemplate(custom_css=".custom { color: blue; }")
        styles = template.get_styles()

        assert ".custom { color: blue; }" in styles

    def test_get_styles_dark_theme(self):
        """Test dark theme styles."""
        template = HTMLTemplate(theme="dark")
        styles = template.get_styles()

        assert "--background: #0f172a" in styles

    def test_get_styles_corporate_theme(self):
        """Test corporate theme styles."""
        template = HTMLTemplate(theme="corporate")
        styles = template.get_styles()

        assert "--primary-color: #1e40af" in styles

    def test_get_scripts(self, template):
        """Test JavaScript inclusion."""
        scripts = template.get_scripts()

        assert "chart.js" in scripts.lower()
        assert "prism" in scripts.lower()

    def test_get_scripts_no_charts(self):
        """Test scripts without charts."""
        template = HTMLTemplate(include_charts=False)
        scripts = template.get_scripts()

        assert scripts == ""

    def test_get_navigation(self, template, sample_sections):
        """Test navigation generation."""
        nav = template.get_navigation(sample_sections)

        assert '<nav class="nav">' in nav
        assert 'href="#summary"' in nav
        assert 'href="#complexity"' in nav
        assert "📊" in nav  # Icon should be included


class TestHTMLReporter:
    """Test suite for HTMLReporter."""

    def test_initialization(self, config):
        """Test reporter initialization."""
        reporter = HTMLReporter(config)

        assert reporter.config == config
        assert isinstance(reporter.template, HTMLTemplate)

    def test_generate(self, reporter, sample_sections, sample_metadata, tmp_path):
        """Test HTML report generation."""
        output_path = tmp_path / "report.html"
        report_config = ReportConfig(title="Test Report", format="html", include_charts=True)

        result = reporter.generate(sample_sections, sample_metadata, output_path, report_config)

        assert result == output_path
        assert output_path.exists()

        # Check content
        content = output_path.read_text()
        assert "Test Report" in content
        assert "Executive Summary" in content
        assert "Complexity Analysis" in content

    def test_generate_header(self, reporter, sample_metadata):
        """Test header generation."""
        report_config = ReportConfig(title="Test Title")

        header = reporter._generate_header(sample_metadata, report_config)

        assert '<header class="header">' in header
        assert "Test Title" in header
        assert "Health Score: 85.5/100" in header
        assert "Files: 100" in header

    def test_generate_header_with_logo(self, reporter, sample_metadata, tmp_path):
        """Test header with custom logo."""
        # Create a fake logo file
        logo_path = tmp_path / "logo.png"
        logo_path.write_bytes(b"fake_image_data")

        report_config = ReportConfig(title="Test", custom_logo=logo_path)

        with patch.object(reporter, "_encode_image", return_value="base64data"):
            header = reporter._generate_header(sample_metadata, report_config)

            assert '<img src="data:image/png;base64,base64data"' in header

    def test_generate_sections(self, reporter, sample_sections):
        """Test sections generation."""
        report_config = ReportConfig()

        html = reporter._generate_sections(sample_sections, report_config)

        assert 'id="summary"' in html
        assert 'id="complexity"' in html
        assert "Executive Summary" in html
        assert "Complexity Analysis" in html

    def test_generate_section(self, reporter):
        """Test single section generation."""
        section = ReportSection(
            id="test",
            title="Test Section",
            level=2,
            content="Test content",
            metrics={"Metric1": 100},
        )
        report_config = ReportConfig()

        html = reporter._generate_section(section, report_config)

        assert 'id="test"' in html
        assert "<h2" in html
        assert "Test Section" in html
        assert "Test content" in html

    def test_generate_section_collapsible(self, reporter):
        """Test collapsible section generation."""
        section = ReportSection(id="test", title="Collapsible", collapsible=True)
        report_config = ReportConfig()

        html = reporter._generate_section(section, report_config)

        assert 'class="collapsible"' in html
        assert 'class="collapsible-content"' in html

    def test_render_content_list(self, reporter):
        """Test rendering list content."""
        content = ["Item 1", "Item 2", "Item 3"]

        html = reporter._render_content(content)

        assert "<ul>" in html
        assert "<li>Item 1</li>" in html
        assert "<li>Item 2</li>" in html
        assert "<li>Item 3</li>" in html

    def test_render_content_dict(self, reporter):
        """Test rendering dictionary content as metrics."""
        content = {"Metric1": 100, "Metric2": 200}

        html = reporter._render_content(content)

        assert 'class="metrics-grid"' in html
        assert "Metric1" in html
        assert "100" in html

    def test_render_table(self, reporter):
        """Test table rendering."""
        table_data = {
            "headers": ["Name", "Value", "Status"],
            "rows": [["Test1", 100, "Pass"], ["Test2", 50, "Fail"]],
        }

        html = reporter._render_table(table_data)

        assert "<table>" in html
        assert "<th>Name</th>" in html
        assert "<td>Test1</td>" in html
        assert "<td>Pass</td>" in html

    def test_render_chart(self, reporter):
        """Test chart rendering."""
        chart_data = {
            "type": "bar",
            "data": {"title": "Test Chart", "labels": ["A", "B"], "values": [1, 2]},
        }

        html = reporter._render_chart(chart_data)

        assert 'class="chart-container"' in html
        assert '<canvas id="chart-' in html
        assert "Test Chart" in html

    def test_render_code_snippet(self, reporter):
        """Test code snippet rendering."""
        snippet = {
            "language": "python",
            "code": "def test():\n    return True",
            "highlight_lines": [2],
        }

        html = reporter._render_code_snippet(snippet)

        assert 'class="code-snippet"' in html
        assert 'data-language="python"' in html
        assert 'class="highlight"' in html
        assert "def test():" in html

    def test_render_metrics(self, reporter):
        """Test metrics rendering."""
        metrics = {"Score": 95.5, "Count": 100, "Status": "Good"}

        html = reporter._render_metrics(metrics)

        assert 'class="metrics-grid"' in html
        assert 'class="metric-card"' in html
        assert "Score" in html
        assert "95.5" in html

    def test_generate_footer(self, reporter):
        """Test footer generation."""
        report_config = ReportConfig(footer_text="Custom Footer")

        footer = reporter._generate_footer(report_config)

        assert '<footer class="footer">' in footer
        assert "Custom Footer" in footer

    def test_generate_chart_scripts(self, reporter, sample_sections):
        """Test Chart.js initialization scripts."""
        scripts = reporter._generate_chart_scripts(sample_sections)

        assert "<script>" in scripts
        assert "new Chart(" in scripts
        assert "getElementById(" in scripts
        assert "collapsible" in scripts.lower()

    def test_generate_chart_config(self, reporter):
        """Test Chart.js configuration generation."""
        config = reporter._generate_chart_config(
            "bar", {"labels": ["A", "B", "C"], "values": [1, 2, 3], "title": "Test Chart"}
        )

        assert config["type"] == "bar"
        assert config["data"]["labels"] == ["A", "B", "C"]
        assert config["data"]["datasets"][0]["data"] == [1, 2, 3]

    def test_generate_chart_config_line(self, reporter):
        """Test line chart configuration."""
        config = reporter._generate_chart_config("line", {"labels": ["A", "B"], "values": [1, 2]})

        assert config["type"] == "line"
        assert config["data"]["datasets"][0]["fill"] is False
        assert config["data"]["datasets"][0]["tension"] == 0.1

    def test_generate_chart_config_gauge(self, reporter):
        """Test gauge chart configuration."""
        config = reporter._generate_chart_config("gauge", {"value": 75, "max": 100})

        assert config["type"] == "doughnut"
        assert config["data"]["datasets"][0]["data"] == [75, 25]
        assert config["options"]["circumference"] == 180

    def test_format_cell(self, reporter):
        """Test table cell formatting."""
        assert reporter._format_cell(True) == "✓"
        assert reporter._format_cell(False) == "✗"
        assert reporter._format_cell(None) == "-"
        assert reporter._format_cell(3.14159) == "3.14"
        assert reporter._format_cell(100) == "100"

    def test_format_cell_badges(self, reporter):
        """Test badge formatting in cells."""
        critical = reporter._format_cell("critical")
        assert 'class="badge badge-critical"' in critical

        high = reporter._format_cell("High")
        assert 'class="badge badge-high"' in high

    def test_escape_html(self, reporter):
        """Test HTML escaping."""
        text = '<script>alert("XSS")</script>'
        escaped = reporter._escape_html(text)

        assert "&lt;script&gt;" in escaped
        assert "&quot;" in escaped
        assert "alert" in escaped

    def test_load_custom_css(self, reporter, tmp_path):
        """Test loading custom CSS from file."""
        css_file = tmp_path / "custom.css"
        css_file.write_text("body { color: red; }")

        css = reporter._load_custom_css(css_file)

        assert css == "body { color: red; }"

    def test_load_custom_css_not_found(self, reporter):
        """Test loading non-existent CSS file."""
        css = reporter._load_custom_css(Path("/nonexistent.css"))

        assert css is None

    def test_encode_image(self, reporter, tmp_path):
        """Test image encoding to base64."""
        image_path = tmp_path / "test.png"
        image_data = b"fake_image_data"
        image_path.write_bytes(image_data)

        encoded = reporter._encode_image(image_path)

        assert encoded == base64.b64encode(image_data).decode("utf-8")

    def test_encode_image_error(self, reporter):
        """Test image encoding error handling."""
        encoded = reporter._encode_image(Path("/nonexistent.png"))

        assert encoded == ""


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_create_html_report(self, tmp_path):
        """Test convenience function for HTML report creation."""
        sections = [ReportSection(id="test", title="Test", content="Test content")]
        output_path = tmp_path / "report.html"

        result = create_html_report(sections, output_path, title="Quick Report")

        assert result == output_path
        assert output_path.exists()

        content = output_path.read_text()
        assert "Quick Report" in content
        assert "Test content" in content

    @patch("tenets.core.reporting.html_reporter.ReportGenerator")
    def test_create_dashboard(self, mock_generator, tmp_path):
        """Test dashboard creation."""
        output_path = tmp_path / "dashboard.html"
        analysis_results = {"test": "data"}

        mock_instance = Mock()
        mock_instance.generate.return_value = output_path
        mock_generator.return_value = mock_instance

        result = create_dashboard(analysis_results, output_path)

        assert result == output_path
        mock_generator.assert_called_once()
        mock_instance.generate.assert_called_once()
