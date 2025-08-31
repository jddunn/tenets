"""Unit tests for the chronicle CLI command.

Tests cover all chronicle functionality including:
- Git history analysis
- Date range parsing
- Contributor analysis
- Pattern detection
- Output formats
- Branch selection
- Author filtering
- Merge commit handling
- Error handling
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.chronicle import chronicle


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_chronicle_builder():
    """Create a mock ChronicleBuilder."""
    builder = MagicMock()

    builder.build_chronicle.return_value = {
        "period": "2024-01-01 to 2024-03-01",
        "total_commits": 150,
        "files_changed": 75,
        "activity": {"trend": 15.5, "current_velocity": 25, "commits_this_week": 12},
        "contributors": {
            "total_contributors": 5,
            "active_contributors": 3,
            "top_contributors": [
                {"name": "Alice", "commits": 60, "lines_added": 2000},
                {"name": "Bob", "commits": 45, "lines_added": 1500},
                {"name": "Charlie", "commits": 30, "lines_added": 1000},
            ],
        },
        "patterns": {
            "change_frequency": {"src/core.py": 15, "src/api.py": 12, "tests/test_core.py": 10},
            "file_coupling": {
                "src/core.py <-> tests/test_core.py": 8,
                "src/api.py <-> src/models.py": 5,
            },
            "refactoring_candidates": [
                {"file": "src/core.py", "changes": 15, "couplings": 8, "risk": "high"}
            ],
        },
    }

    return builder


@pytest.fixture
def mock_git_analyzer():
    """Create a mock GitAnalyzer."""
    analyzer = MagicMock()

    analyzer.get_commits.return_value = [
        {
            "sha": "abc123",
            "message": "Fix bug in authentication",
            "author": "Alice",
            "date": datetime.now() - timedelta(days=1),
            "files": ["src/auth.py", "tests/test_auth.py"],
        },
        {
            "sha": "def456",
            "message": "Add new feature",
            "author": "Bob",
            "date": datetime.now() - timedelta(days=2),
            "files": ["src/feature.py"],
        },
    ]

    analyzer.analyze_contributors.return_value = {
        "total_contributors": 5,
        "active_contributors": 3,
        "top_contributors": [{"name": "Alice", "commits": 60}, {"name": "Bob", "commits": 45}],
    }

    return analyzer


class TestChronicleBasicFunctionality:
    """Test basic chronicle command functionality."""

    def test_chronicle_default(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test basic chronicle with defaults."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, ["."])

                    assert result.exit_code == 0
                    assert "CHRONICLE SUMMARY" in result.stdout
                    assert "Total commits: 150" in result.stdout
                    assert "Files changed: 75" in result.stdout
                    mock_chronicle_builder.build_chronicle.assert_called_once()

    def test_chronicle_specific_path(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle for specific path."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, ["src/"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args
                    # Compare the actual first positional argument to the resolved path string
                    first_arg = call_args[0][0] if call_args and call_args[0] else None
                    assert str(first_arg) == str(Path("src/").resolve())


class TestChronicleDateRangeParsing:
    """Test date range parsing functionality."""

    def test_chronicle_with_since_date(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with specific start date."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--since", "2024-01-01"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert isinstance(call_args["since"], datetime)
                    assert call_args["since"].year == 2024
                    assert call_args["since"].month == 1

    def test_chronicle_with_until_date(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with specific end date."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--until", "2024-03-01"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert isinstance(call_args["until"], datetime)

    def test_chronicle_with_relative_date(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with relative date (e.g., '3 months ago')."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--since", "3 months ago"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert isinstance(call_args["since"], datetime)
                    # Should be approximately 3 months ago
                    days_diff = (datetime.now() - call_args["since"]).days
                    assert 85 <= days_diff <= 95  # Around 90 days

    def test_chronicle_with_week_ago(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with 'X weeks ago' format."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--since", "2 weeks ago"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    days_diff = (datetime.now() - call_args["since"]).days
                    assert 13 <= days_diff <= 15  # Around 14 days

    def test_chronicle_with_today(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with 'today' as until date."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--until", "today"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert call_args["until"].date() == datetime.now().date()


class TestChronicleFiltering:
    """Test filtering options."""

    def test_chronicle_with_branch(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle for specific branch."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--branch", "develop"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert call_args["branch"] == "develop"

    def test_chronicle_with_authors(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle filtered by authors."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(
                        chronicle, [".", "--authors", "alice", "--authors", "bob"]
                    )

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert call_args["authors"] == ["alice", "bob"]

    def test_chronicle_with_show_merges(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle including merge commits."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--show-merges"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert call_args["include_merges"] is True

    def test_chronicle_with_limit(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with commit limit."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--limit", "100"])

                    assert result.exit_code == 0
                    call_args = mock_chronicle_builder.build_chronicle.call_args[1]
                    assert call_args["limit"] == 100


class TestChronicleAnalysisOptions:
    """Test additional analysis options."""

    def test_chronicle_with_contributors(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with contributor analysis."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    with patch("tenets.cli.commands.chronicle.ContributorVisualizer"):
                        result = runner.invoke(chronicle, [".", "--show-contributors"])

                        assert result.exit_code == 0
                        mock_git_analyzer.analyze_contributors.assert_called_once()
                        assert "Contributors:" in result.stdout
                        assert "Total: 5" in result.stdout
                        assert "Active: 3" in result.stdout

    def test_chronicle_with_patterns(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test chronicle with pattern analysis."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    with patch("tenets.cli.commands.chronicle.TerminalDisplay") as mock_display:
                        result = runner.invoke(chronicle, [".", "--show-patterns"])

                        assert result.exit_code == 0
                        # Verify pattern analysis was performed
                        assert "patterns" in mock_chronicle_builder.build_chronicle.return_value
                        mock_display.return_value.display_table.assert_called()


class TestChronicleOutputFormats:
    """Test different output formats."""

    def test_chronicle_terminal_output(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test terminal output format."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    with patch("tenets.cli.commands.chronicle.TerminalDisplay"):
                        result = runner.invoke(chronicle, [".", "--format", "terminal"])

                        assert result.exit_code == 0
                        assert "Repository Chronicle" in result.stdout
                        assert "CHRONICLE SUMMARY" in result.stdout

    def test_chronicle_json_output(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test JSON output format."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, [".", "--format", "json"])

                    assert result.exit_code == 0
                    # Should output valid JSON
                    output_data = json.loads(result.stdout)
                    assert output_data["total_commits"] == 150
                    assert output_data["files_changed"] == 75

    def test_chronicle_json_to_file(
        self, runner, mock_chronicle_builder, mock_git_analyzer, tmp_path
    ):
        """Test JSON output to file."""
        output_file = tmp_path / "chronicle.json"

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(
                        chronicle, [".", "--format", "json", "--output", str(output_file)]
                    )

                    assert result.exit_code == 0
                    assert f"Chronicle saved to: {output_file}" in result.stdout
                    assert output_file.exists()

    @pytest.mark.skipif(
        sys.version_info[:2] >= (3, 13),
        reason="Threading tests hang with coverage on Python 3.13+"
    )
    def test_chronicle_html_output(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test HTML report generation."""
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = Path("chronicle_report.html")

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch(
                    "tenets.cli.commands.chronicle.ReportGenerator", return_value=mock_report_gen
                ):
                    with patch("tenets.cli.commands.chronicle.get_logger"):
                        with patch("tenets.cli.commands.chronicle.click.confirm", return_value=False):
                            result = runner.invoke(chronicle, [".", "--format", "html"])

                            assert result.exit_code == 0
                            assert "Chronicle report generated: chronicle" in result.stdout
                            assert "Would you like to open it in your browser now?" in result.stdout
                            mock_report_gen.generate.assert_called_once()

    @pytest.mark.skipif(
        sys.version_info[:2] >= (3, 13),
        reason="Threading tests hang with coverage on Python 3.13+"
    )
    def test_chronicle_html_opens_browser(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test HTML report opens browser when confirmed."""
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = Path("chronicle_report.html")

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch(
                    "tenets.cli.commands.chronicle.ReportGenerator", return_value=mock_report_gen
                ):
                    with patch("tenets.cli.commands.chronicle.get_logger"):
                        with patch("tenets.cli.commands.chronicle.click.confirm", return_value=True):
                            with patch("tenets.cli.commands.chronicle.webbrowser.open") as mock_browser:
                                result = runner.invoke(chronicle, [".", "--format", "html"])

                                assert result.exit_code == 0
                                assert "Chronicle report generated: chronicle" in result.stdout
                                assert "✓ Opened in browser" in result.stdout
                                mock_browser.assert_called_once()

    def test_chronicle_markdown_output(
        self, runner, mock_chronicle_builder, mock_git_analyzer, tmp_path
    ):
        """Test Markdown report generation."""
        output_file = tmp_path / "chronicle.md"
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = output_file

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch(
                    "tenets.cli.commands.chronicle.ReportGenerator", return_value=mock_report_gen
                ):
                    with patch("tenets.cli.commands.chronicle.get_logger"):
                        result = runner.invoke(
                            chronicle, [".", "--format", "markdown", "--output", str(output_file)]
                        )

                        assert result.exit_code == 0
                        assert f"Chronicle report generated: {output_file}" in result.stdout


class TestChronicleActivityTrend:
    """Test activity trend display."""

    def test_activity_trending_up(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test display of upward activity trend."""
        mock_chronicle_builder.build_chronicle.return_value["activity"]["trend"] = 25.5

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, ["."])

                    assert result.exit_code == 0
                    assert "Activity trending up: +25.5%" in result.stdout

    def test_activity_trending_down(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test display of downward activity trend."""
        mock_chronicle_builder.build_chronicle.return_value["activity"]["trend"] = -15.0

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, ["."])

                    assert result.exit_code == 0
                    assert "Activity trending down: -15.0%" in result.stdout

    def test_activity_stable(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test display of stable activity."""
        mock_chronicle_builder.build_chronicle.return_value["activity"]["trend"] = 0

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, ["."])

                    assert result.exit_code == 0
                    assert "Activity stable" in result.stdout


class TestChroniclePatternAnalysis:
    """Test pattern analysis and display."""

    def test_pattern_display(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test pattern analysis display."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    with patch("tenets.cli.commands.chronicle.TerminalDisplay") as mock_display:
                        result = runner.invoke(chronicle, [".", "--show-patterns"])

                        assert result.exit_code == 0
                        # Verify display methods were called for patterns
                        mock_display.return_value.display_header.assert_any_call(
                            "Change Patterns", style="single"
                        )
                        mock_display.return_value.display_table.assert_called()
                        mock_display.return_value.display_list.assert_called()


class TestChronicleErrorHandling:
    """Test error handling scenarios."""

    def test_chronicle_path_not_exists(self, runner):
        """Test error when path doesn't exist."""
        result = runner.invoke(chronicle, ["nonexistent/path"])

        assert result.exit_code != 0
        assert "does not exist" in result.stdout.lower() or "invalid" in result.stdout.lower()

    def test_chronicle_generation_error(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test error during chronicle generation."""
        mock_chronicle_builder.build_chronicle.side_effect = Exception(
            "Chronicle generation failed"
        )

        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(chronicle, ["."])

                    assert result.exit_code != 0
                    assert "Chronicle generation failed" in result.stdout


class TestChronicleSummaryOutput:
    """Test summary output formatting."""

    def test_complete_summary(self, runner, mock_chronicle_builder, mock_git_analyzer):
        """Test complete summary output."""
        with patch(
            "tenets.cli.commands.chronicle.ChronicleBuilder", return_value=mock_chronicle_builder
        ):
            with patch("tenets.cli.commands.chronicle.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.chronicle.get_logger"):
                    result = runner.invoke(
                        chronicle, [".", "--show-contributors", "--show-patterns"]
                    )

                    assert result.exit_code == 0
                    assert "CHRONICLE SUMMARY" in result.stdout
                    assert "Period:" in result.stdout
                    assert "Total commits: 150" in result.stdout
                    assert "Files changed: 75" in result.stdout
                    assert "Contributors:" in result.stdout
                    assert "Top 3:" in result.stdout
                    assert "Alice: 60 commits" in result.stdout
