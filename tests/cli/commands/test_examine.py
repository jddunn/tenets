"""Unit tests for the examine CLI command.

Tests cover all examination functionality including:
- Code quality and complexity analysis
- Metrics calculation
- Hotspot detection
- Ownership analysis
- Output formats (terminal, HTML, JSON, markdown)
- Threshold configuration
- Include/exclude patterns
- Error handling
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.examine import examine


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_examiner():
    """Create a mock CodeExaminer."""
    examiner = MagicMock()

    # Mock examination results
    examiner.examine.return_value = {
        "total_files": 50,
        "total_lines": 5000,
        "overview": {"total_files": 50, "total_lines": 5000, "health_score": 75.0},
        "metrics": {"avg_complexity": 3.5, "duplication_ratio": 0.15, "test_coverage": 0.80},
        "complexity": {
            "avg_complexity": 3.5,
            "max_complexity": 15,
            "complex_functions": 5,
            "distribution": {"low": 30, "medium": 15, "high": 5},
        },
        "hotspots": {
            "total_hotspots": 3,
            "critical_count": 1,
            "files": [{"path": "src/core.py", "change_count": 25, "risk": "high"}],
        },
        "ownership": {
            "by_contributor": [
                {"name": "Alice", "files": 20, "lines": 2000, "commits": 50},
                {"name": "Bob", "files": 15, "lines": 1500, "commits": 30},
            ],
            "total_lines": 5000,
            "bus_factor": 2,
        },
        "health_score": 75.0,
        "risks": [
            {"description": "High complexity in core modules"},
            {"description": "Low test coverage in API layer"},
        ],
        "recommendations": [
            "Refactor complex functions in src/core.py",
            "Increase test coverage for API endpoints",
        ],
    }

    examiner.examine_file.return_value = {
        "complexity": 5,
        "lines": 150,
        "functions": 10,
        "classes": 2,
    }

    return examiner


@pytest.fixture
def mock_hotspot_detector():
    """Create a mock HotspotDetector."""
    detector = MagicMock()
    detector.detect_hotspots.return_value = {
        "total_hotspots": 3,
        "critical_count": 1,
        "files": [{"path": "src/core.py", "change_count": 25, "risk": "high"}],
    }
    return detector


@pytest.fixture
def mock_ownership_analyzer():
    """Create a mock OwnershipAnalyzer."""
    analyzer = MagicMock()
    analyzer.analyze_ownership.return_value = {
        "by_contributor": [
            {"name": "Alice", "files": 20, "lines": 2000},
            {"name": "Bob", "files": 15, "lines": 1500},
        ],
        "top_contributors": [
            {"name": "Alice", "commits": 50, "files": 20, "expertise": "Core"},
            {"name": "Bob", "commits": 30, "files": 15, "expertise": "API"},
        ],
        "total_lines": 5000,
        "bus_factor": 2,
    }
    return analyzer


@pytest.fixture
def mock_report_generator():
    """Create a mock ReportGenerator."""
    generator = MagicMock()
    generator.generate.return_value = Path("report.html")
    return generator


class TestExamineBasicFunctionality:
    """Test basic examine command functionality."""

    def test_examine_default(self, runner, mock_examiner):
        """Test basic examination with defaults."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["."])

                assert result.exit_code == 0
                assert "EXAMINATION SUMMARY" in result.stdout
                assert "Files analyzed: 50" in result.stdout
                assert "Total lines: 5,000" in result.stdout
                assert "Health Score:" in result.stdout
                mock_examiner.examine.assert_called_once()

    def test_examine_specific_path(self, runner, mock_examiner):
        """Test examining specific path."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["src/"])

                assert result.exit_code == 0
                call_args = mock_examiner.examine.call_args
                first_arg = call_args.args[0] if call_args and call_args.args else None
                assert str(Path("src/").resolve()) == str(first_arg)

    def test_examine_with_threshold(self, runner, mock_examiner):
        """Test examination with complexity threshold."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, [".", "--threshold", "15"])

                assert result.exit_code == 0
                call_args = mock_examiner.examine.call_args[1]
                assert call_args["threshold"] == 15

    def test_examine_with_include_patterns(self, runner, mock_examiner):
        """Test examination with include patterns."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, [".", "--include", "*.py", "--include", "*.js"])

                assert result.exit_code == 0
                call_args = mock_examiner.examine.call_args[1]
                assert call_args["include_patterns"] == ["*.py", "*.js"]

    def test_examine_with_exclude_patterns(self, runner, mock_examiner):
        """Test examination with exclude patterns."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(
                    examine, [".", "--exclude", "test_*", "--exclude", "*.backup"]
                )

                assert result.exit_code == 0
                call_args = mock_examiner.examine.call_args[1]
                assert call_args["exclude_patterns"] == ["test_*", "*.backup"]

    def test_examine_with_max_depth(self, runner, mock_examiner):
        """Test examination with max depth limit."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, [".", "--max-depth", "3"])

                assert result.exit_code == 0
                call_args = mock_examiner.examine.call_args[1]
                assert call_args["max_depth"] == 3


class TestExamineSpecializedAnalysis:
    """Test specialized analysis options."""

    def test_examine_with_hotspots(self, runner, mock_examiner, mock_hotspot_detector):
        """Test examination with hotspot analysis."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.HotspotDetector", return_value=mock_hotspot_detector
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    result = runner.invoke(examine, [".", "--hotspots"])

                    assert result.exit_code == 0
                    assert "Hotspots:" in result.stdout
                    assert "Total: 3" in result.stdout
                    assert "Critical: 1" in result.stdout
                    mock_hotspot_detector.detect_hotspots.assert_called_once()

    def test_examine_with_ownership(self, runner, mock_examiner, mock_ownership_analyzer):
        """Test examination with ownership analysis."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.OwnershipAnalyzer",
                return_value=mock_ownership_analyzer,
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    result = runner.invoke(examine, [".", "--ownership"])

                    assert result.exit_code == 0
                    mock_ownership_analyzer.analyze_ownership.assert_called_once()

    def test_examine_with_specific_metrics(self, runner, mock_examiner):
        """Test examination with specific metrics."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(
                    examine, [".", "--metrics", "complexity", "--metrics", "duplication"]
                )

                assert result.exit_code == 0
                call_args = mock_examiner.examine.call_args[1]
                assert call_args["calculate_metrics"] == ["complexity", "duplication"]

    def test_examine_show_details(self, runner, mock_examiner):
        """Test examination with detailed breakdown."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                with patch("tenets.cli.commands.examine.ComplexityVisualizer") as mock_viz:
                    result = runner.invoke(examine, [".", "--show-details"])

                    assert result.exit_code == 0
                    mock_viz.return_value.display_terminal.assert_called_with(
                        mock_examiner.examine.return_value["complexity"],
                        True,  # show_details=True
                    )


class TestExamineOutputFormats:
    """Test different output formats."""

    def test_examine_terminal_output(self, runner, mock_examiner):
        """Test default terminal output."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                with patch("tenets.cli.commands.examine.TerminalDisplay") as mock_display:
                    result = runner.invoke(examine, [".", "--format", "terminal"])

                    assert result.exit_code == 0
                    mock_display.return_value.display_header.assert_called()
                    assert "EXAMINATION SUMMARY" in result.stdout

    def test_examine_json_output(self, runner, mock_examiner):
        """Test JSON output format."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                with patch("tenets.cli.commands.examine.TenetsConfig"):
                    result = runner.invoke(examine, [".", "--format", "json"])

                    assert result.exit_code == 0
                    # Should output valid JSON
                    output_data = json.loads(result.stdout)
                    assert output_data["total_files"] == 50
                    assert output_data["health_score"] == 75.0

    def test_examine_json_output_to_file(self, runner, mock_examiner, tmp_path):
        """Test JSON output to file."""
        output_file = tmp_path / "report.json"

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                with patch("tenets.cli.commands.examine.TenetsConfig"):
                    result = runner.invoke(
                        examine, [".", "--format", "json", "--output", str(output_file)]
                    )

                    assert result.exit_code == 0
                    assert f"Results saved to: {output_file}" in result.stdout
                    assert output_file.exists()

                # Verify file content
                with open(output_file) as f:
                    data = json.load(f)
                    assert data["total_files"] == 50

    def test_examine_html_output(self, runner, mock_examiner, mock_report_generator):
        """Test HTML report generation."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.ReportGenerator", return_value=mock_report_generator
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    result = runner.invoke(examine, [".", "--format", "html"])

                    assert result.exit_code == 0
                    # Check that a report was generated with auto-generated filename
                    assert "Report generated: tenets_report_" in result.stdout
                    assert ".html" in result.stdout
                    mock_report_generator.generate.assert_called_once()

    def test_examine_markdown_output(self, runner, mock_examiner, mock_report_generator):
        """Test Markdown report generation."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.ReportGenerator", return_value=mock_report_generator
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    result = runner.invoke(
                        examine, [".", "--format", "markdown", "--output", "report.md"]
                    )

                    assert result.exit_code == 0
                    assert "Report generated: report.md" in result.stdout


class TestExamineHealthScore:
    """Test health score calculation and display."""

    def test_health_score_excellent(self, runner, mock_examiner):
        """Test excellent health score display."""
        mock_examiner.examine.return_value["health_score"] = 85
        mock_examiner.examine.return_value["overview"]["health_score"] = 85

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["."])

                assert result.exit_code == 0
                assert "Health Score:" in result.stdout
                assert "85" in result.stdout
                assert "Excellent" in result.stdout

    def test_health_score_good(self, runner, mock_examiner):
        """Test good health score display."""
        mock_examiner.examine.return_value["health_score"] = 65
        mock_examiner.examine.return_value["overview"]["health_score"] = 65

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["."])

                assert result.exit_code == 0
                assert "65" in result.stdout
                assert "Good" in result.stdout

    def test_health_score_fair(self, runner, mock_examiner):
        """Test fair health score display."""
        mock_examiner.examine.return_value["health_score"] = 45
        mock_examiner.examine.return_value["overview"]["health_score"] = 45

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["."])

                assert result.exit_code == 0
                assert "45" in result.stdout
                assert "Fair" in result.stdout

    def test_health_score_needs_improvement(self, runner, mock_examiner):
        """Test low health score display."""
        mock_examiner.examine.return_value["health_score"] = 30
        mock_examiner.examine.return_value["overview"]["health_score"] = 30

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["."])

                assert result.exit_code == 0
                assert "30" in result.stdout
                assert "Needs Improvement" in result.stdout


class TestExamineComplexityDisplay:
    """Test complexity analysis display."""

    def test_complexity_terminal_display(self, runner, mock_examiner):
        """Test complexity display in terminal."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                with patch("tenets.cli.commands.examine.ComplexityVisualizer") as mock_viz:
                    result = runner.invoke(examine, ["."])

                    assert result.exit_code == 0
                    assert "Complexity:" in result.stdout
                    assert "Average: 3.50" in result.stdout
                    assert "Maximum: 15" in result.stdout
                    assert "Complex functions: 5" in result.stdout


class TestExamineOwnershipDisplay:
    """Test ownership analysis display."""

    def test_ownership_display(self, runner, mock_examiner, mock_ownership_analyzer):
        """Test ownership display with details."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.OwnershipAnalyzer",
                return_value=mock_ownership_analyzer,
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    with patch("tenets.cli.commands.examine.TerminalDisplay") as mock_display:
                        result = runner.invoke(examine, [".", "--ownership", "--show-details"])

                        assert result.exit_code == 0
                        # Verify display methods were called
                        mock_display.return_value.display_header.assert_any_call(
                            "Code Ownership", style="single"
                        )
                        mock_display.return_value.display_table.assert_called()

    def test_low_bus_factor_warning(self, runner, mock_examiner, mock_ownership_analyzer):
        """Test warning for low bus factor."""
        mock_ownership_analyzer.analyze_ownership.return_value["bus_factor"] = 1

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.OwnershipAnalyzer",
                return_value=mock_ownership_analyzer,
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    with patch("tenets.cli.commands.examine.TerminalDisplay") as mock_display:
                        result = runner.invoke(examine, [".", "--ownership"])

                        assert result.exit_code == 0
                        mock_display.return_value.display_warning.assert_called_with(
                            "Low bus factor (1) - knowledge concentration risk!"
                        )


class TestExamineErrorHandling:
    """Test error handling scenarios."""

    def test_examine_path_not_exists(self, runner):
        """Test error when path doesn't exist."""
        result = runner.invoke(examine, ["nonexistent/path"])

        # Click validates path existence
        assert result.exit_code != 0
        assert "does not exist" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_examine_analysis_error(self, runner, mock_examiner):
        """Test error during analysis."""
        mock_examiner.examine.side_effect = Exception("Analysis failed")

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                result = runner.invoke(examine, ["."])

                assert result.exit_code != 0
                assert "Analysis failed" in result.stdout

    def test_examine_report_generation_error(self, runner, mock_examiner, mock_report_generator):
        """Test error during report generation."""
        mock_report_generator.generate.side_effect = Exception("Report generation failed")

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.ReportGenerator", return_value=mock_report_generator
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    result = runner.invoke(examine, [".", "--format", "html"])

                    assert result.exit_code != 0
                    assert "Report generation failed" in result.stdout


class TestExamineSummaryOutput:
    """Test summary output formatting."""

    def test_summary_with_all_sections(self, runner, mock_examiner):
        """Test complete summary output."""
        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch("tenets.cli.commands.examine.get_logger"):
                with patch("tenets.cli.commands.examine.HotspotDetector") as mock_hotspot:
                    with patch("tenets.cli.commands.examine.OwnershipAnalyzer") as mock_ownership:
                        # Configure the mocked detectors
                        mock_hotspot_instance = MagicMock()
                        mock_hotspot_instance.detect_hotspots.return_value = {
                            "total_hotspots": 5,
                            "critical_count": 2,
                        }
                        mock_hotspot.return_value = mock_hotspot_instance

                        mock_ownership_instance = MagicMock()
                        mock_ownership_instance.analyze_ownership.return_value = {
                            "by_contributor": [],
                            "bus_factor": 3,
                        }
                        mock_ownership.return_value = mock_ownership_instance

                        result = runner.invoke(examine, [".", "--hotspots", "--ownership"])

                        assert result.exit_code == 0
                        assert "EXAMINATION SUMMARY" in result.stdout
                        assert "Files analyzed:" in result.stdout
                        assert "Total lines:" in result.stdout
                        assert "Complexity:" in result.stdout
                        assert "Hotspots:" in result.stdout
                        assert "Health Score:" in result.stdout


class TestAutoFilenameGeneration:
    """Test automatic report filename generation."""

    def test_generate_auto_filename_basic(self):
        """Test basic filename generation."""
        from datetime import datetime

        from tenets.cli.commands.examine import generate_auto_filename

        # Use a fixed timestamp for consistent testing
        test_time = datetime(2024, 1, 15, 14, 30, 45)

        # Test with simple path
        filename = generate_auto_filename("myproject", "html", test_time)
        assert filename == "tenets_report_myproject_20240115_143045.html"

        # Test with different format
        filename = generate_auto_filename("myproject", "json", test_time)
        assert filename == "tenets_report_myproject_20240115_143045.json"

    def test_generate_auto_filename_with_path(self):
        """Test filename generation with full paths."""
        from datetime import datetime
        from pathlib import Path

        from tenets.cli.commands.examine import generate_auto_filename

        test_time = datetime(2024, 1, 15, 14, 30, 45)

        # Test with absolute path
        filename = generate_auto_filename("/home/user/projects/myapp", "html", test_time)
        assert filename == "tenets_report_myapp_20240115_143045.html"

        # Test with Windows path
        filename = generate_auto_filename("C:\\Users\\dev\\myapp", "html", test_time)
        assert filename == "tenets_report_myapp_20240115_143045.html"

        # Test with Path object
        filename = generate_auto_filename(Path("/projects/webapp"), "markdown", test_time)
        assert filename == "tenets_report_webapp_20240115_143045.markdown"

    def test_generate_auto_filename_special_chars(self):
        """Test filename generation with special characters."""
        from datetime import datetime

        from tenets.cli.commands.examine import generate_auto_filename

        test_time = datetime(2024, 1, 15, 14, 30, 45)

        # Test with spaces and special chars
        filename = generate_auto_filename("my-project_v2", "html", test_time)
        assert filename == "tenets_report_my-project_v2_20240115_143045.html"

        # Test with invalid filename chars
        filename = generate_auto_filename("my@project#1", "html", test_time)
        assert filename == "tenets_report_my_project_1_20240115_143045.html"

        # Test with dots
        filename = generate_auto_filename("my.project.name", "html", test_time)
        assert filename == "tenets_report_my_project_name_20240115_143045.html"

    def test_generate_auto_filename_edge_cases(self):
        """Test filename generation edge cases."""
        from datetime import datetime

        from tenets.cli.commands.examine import generate_auto_filename

        test_time = datetime(2024, 1, 15, 14, 30, 45)

        # Test with current directory "."
        filename = generate_auto_filename(".", "html", test_time)
        assert filename == "tenets_report_project_20240115_143045.html"

        # Test with empty string
        filename = generate_auto_filename("", "html", test_time)
        assert filename == "tenets_report_project_20240115_143045.html"

        # Test with only special chars - should become "project" since it's all underscores
        filename = generate_auto_filename("@#$%", "html", test_time)
        # After converting special chars to underscores, it should detect it's all underscores and use "project"
        assert filename == "tenets_report_project_20240115_143045.html"

    def test_generate_auto_filename_no_timestamp(self):
        """Test that current time is used when no timestamp provided."""

        from tenets.cli.commands.examine import generate_auto_filename

        # Generate two filenames quickly
        filename1 = generate_auto_filename("test", "html")
        filename2 = generate_auto_filename("test", "html")

        # Both should start with the expected prefix
        assert filename1.startswith("tenets_report_test_")
        assert filename1.endswith(".html")

        # Check timestamp format (YYYYMMDD_HHMMSS)
        parts = filename1.replace("tenets_report_test_", "").replace(".html", "")
        assert len(parts) == 15  # 8 for date + 1 underscore + 6 for time
        assert "_" in parts

    def test_examine_auto_filename_integration(
        self, runner, mock_examiner, mock_report_generator, tmp_path
    ):
        """Test that examine command uses auto filename when no output specified."""
        import re

        with patch("tenets.cli.commands.examine.CodeExaminer", return_value=mock_examiner):
            with patch(
                "tenets.cli.commands.examine.ReportGenerator", return_value=mock_report_generator
            ):
                with patch("tenets.cli.commands.examine.get_logger"):
                    # Run examine without specifying output
                    result = runner.invoke(examine, [str(tmp_path), "--format", "html"])

                    assert result.exit_code == 0
                    # Check that the auto-generated filename is in the output
                    assert "Report generated: tenets_report_" in result.stdout

                    # Verify the filename pattern
                    match = re.search(r"tenets_report_(\w+)_(\d{8}_\d{6})\.html", result.stdout)
                    assert match is not None

                    # Verify the path component
                    path_component = match.group(1)
                    assert len(path_component) > 0

                    # Verify timestamp format
                    timestamp_component = match.group(2)
                    assert len(timestamp_component) == 15  # YYYYMMDD_HHMMSS
