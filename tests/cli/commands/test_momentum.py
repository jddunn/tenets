"""Unit tests for the momentum CLI command.

Tests cover all momentum tracking functionality including:
- Velocity tracking
- Sprint analysis
- Team metrics
- Burndown charts
- Forecasting
- Period selection
- Output formats
- Error handling
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.momentum import momentum


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_momentum_tracker():
    """Create a mock MomentumTracker."""
    tracker = MagicMock()

    tracker.track_momentum.return_value = {
        "velocity_data": [20, 25, 22, 28, 30, 35],
        "current_velocity": 30,
        "velocity_trend": 15.5,
        "team_metrics": {
            "team_size": 5,
            "active_contributors": 5,
            "total_commits": 150,
            "avg_commits_per_day": 5,
            "active_days": 25,
            "productivity": 85.0,
            "collaboration_index": 65.0,
        },
        "burndown": {
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ideal_line": [100, 75, 50],
            "actual_line": [100, 80, 55],
            "total_work": 100,
            "remaining_work": 55,
            "on_track": False,
            "completion_percentage": 45.0,
        },
        "forecast": {
            "available": True,
            "current_velocity": 30,
            "trend_percentage": 15.0,
            "forecast_values": [32, 34, 36],
            "confidence": "medium",
        },
    }

    return tracker


@pytest.fixture
def mock_git_analyzer():
    """Create a mock GitAnalyzer."""
    analyzer = MagicMock()

    analyzer.get_commits.return_value = [
        {
            "author": "Alice",
            "date": datetime.now() - timedelta(days=1),
            "files": ["src/file1.py", "src/file2.py"],
        },
        {"author": "Bob", "date": datetime.now() - timedelta(days=2), "files": ["src/file3.py"]},
    ]

    return analyzer


class TestMomentumBasicFunctionality:
    """Test basic momentum command functionality."""

    def test_momentum_default(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test basic momentum tracking with defaults."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, ["."])

                    assert result.exit_code == 0
                    assert "MOMENTUM SUMMARY" in result.stdout
                    assert "Current Velocity: 30.0" in result.stdout
                    mock_momentum_tracker.track_momentum.assert_called_once()

    def test_momentum_specific_path(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test momentum for specific path."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, ["src/"])

                    assert result.exit_code == 0
                    # Path should be resolved
                    call_args = mock_momentum_tracker.track_momentum.call_args
                    # Check the first argument directly for equality
                    expected_path = str(Path("src/").resolve())
                    actual_path = call_args[0][0] if call_args and call_args[0] else None
                    assert actual_path == expected_path


class TestMomentumPeriodOptions:
    """Test different period options."""

    def test_momentum_daily_period(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test daily velocity tracking."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--period", "day"])

                    assert result.exit_code == 0
                    call_args = mock_momentum_tracker.track_momentum.call_args[1]
                    assert call_args["period"] == "day"

    def test_momentum_weekly_period(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test weekly velocity tracking."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--period", "week"])

                    assert result.exit_code == 0
                    call_args = mock_momentum_tracker.track_momentum.call_args[1]
                    assert call_args["period"] == "week"

    def test_momentum_sprint_period(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test sprint-based velocity tracking."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(
                        momentum, [".", "--period", "sprint", "--sprint-length", "14"]
                    )

                    assert result.exit_code == 0
                    call_args = mock_momentum_tracker.track_momentum.call_args[1]
                    assert call_args["period"] == "sprint"
                    assert call_args["sprint_length"] == 14

    def test_momentum_monthly_period(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test monthly velocity tracking."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--period", "month"])

                    assert result.exit_code == 0
                    call_args = mock_momentum_tracker.track_momentum.call_args[1]
                    assert call_args["period"] == "month"

    def test_momentum_duration(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test custom duration."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--duration", "8"])

                    assert result.exit_code == 0
                    # Duration affects the date range calculation
                    call_args = mock_momentum_tracker.track_momentum.call_args[1]
                    assert "since" in call_args
                    assert isinstance(call_args["since"], datetime)


class TestMomentumTeamMetrics:
    """Test team metrics functionality."""

    def test_momentum_with_team_metrics(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test momentum with team metrics."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    with patch("tenets.cli.commands.momentum.TerminalDisplay") as mock_display:
                        result = runner.invoke(momentum, [".", "--team"])

                        assert result.exit_code == 0
                        assert "Team Size: 5" in result.stdout
                        assert "Productivity: 85.0%" in result.stdout
                        mock_display.return_value.display_header.assert_any_call(
                            "Team Metrics", style="single"
                        )
                        mock_display.return_value.display_metrics.assert_called()


class TestMomentumBurndown:
    """Test burndown chart functionality."""

    def test_momentum_with_burndown(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test momentum with burndown chart."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    with patch("tenets.cli.commands.momentum.TerminalDisplay") as mock_display:
                        result = runner.invoke(momentum, [".", "--period", "sprint", "--burndown"])

                        assert result.exit_code == 0
                        mock_display.return_value.display_header.assert_any_call(
                            "Sprint Burndown", style="single"
                        )
                        # Check progress bar creation
                        mock_display.return_value.create_progress_bar.assert_called_with(45.0, 100)

    def test_burndown_on_track(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test burndown when on track."""
        mock_momentum_tracker.track_momentum.return_value["burndown"]["on_track"] = True

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--period", "sprint", "--burndown"])

                    assert result.exit_code == 0
                    assert "On Track" in result.stdout

    def test_burndown_behind_schedule(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test burndown when behind schedule."""
        mock_momentum_tracker.track_momentum.return_value["burndown"]["on_track"] = False

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--period", "sprint", "--burndown"])

                    assert result.exit_code == 0
                    assert "Behind Schedule" in result.stdout


class TestMomentumForecast:
    """Test forecasting functionality."""

    def test_momentum_with_forecast(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test momentum with velocity forecast."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    with patch("tenets.cli.commands.momentum.TerminalDisplay") as mock_display:
                        result = runner.invoke(momentum, [".", "--forecast"])

                        assert result.exit_code == 0
                        mock_display.return_value.display_header.assert_any_call(
                            "Velocity Forecast", style="single"
                        )
                        assert "Current Velocity: 30.0" in result.stdout
                        assert "Trend:" in result.stdout
                        assert "15.0%" in result.stdout
                        assert "Confidence: MEDIUM" in result.stdout
                        assert "Period +1: 32.0" in result.stdout

    def test_forecast_not_available(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test when forecast is not available."""
        mock_momentum_tracker.track_momentum.return_value["forecast"]["available"] = False

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--forecast"])

                    assert result.exit_code == 0
                    # Should handle unavailable forecast gracefully


class TestMomentumMetrics:
    """Test specific metrics selection."""

    def test_momentum_with_specific_metrics(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test momentum with specific metrics."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(
                        momentum,
                        [
                            ".",
                            "--metrics",
                            "velocity",
                            "--metrics",
                            "throughput",
                            "--metrics",
                            "cycle-time",
                        ],
                    )

                    assert result.exit_code == 0
                    call_args = mock_momentum_tracker.track_momentum.call_args[1]
                    assert call_args["metrics"] == ["velocity", "throughput", "cycle-time"]


class TestMomentumOutputFormats:
    """Test different output formats."""

    def test_momentum_terminal_output(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test terminal output format."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    with patch("tenets.cli.commands.momentum.MomentumVisualizer"):
                        result = runner.invoke(momentum, [".", "--format", "terminal"])

                        assert result.exit_code == 0
                        assert "MOMENTUM SUMMARY" in result.stdout

    def test_momentum_json_output(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test JSON output format."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--format", "json"])

                    assert result.exit_code == 0
                    # Should output valid JSON
                    output_data = json.loads(result.stdout)
                    assert output_data["current_velocity"] == 30
                    assert output_data["velocity_trend"] == 15.5

    def test_momentum_json_to_file(
        self, runner, mock_momentum_tracker, mock_git_analyzer, tmp_path
    ):
        """Test JSON output to file."""
        output_file = tmp_path / "momentum.json"

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(
                        momentum, [".", "--format", "json", "--output", str(output_file)]
                    )

                    assert result.exit_code == 0
                    assert f"Momentum data saved to: {output_file}" in result.stdout
                    assert output_file.exists()

    @pytest.mark.skipif(
        sys.version_info[:2] >= (3, 13),
        reason="Threading tests hang with coverage on Python 3.13+"
    )
    def test_momentum_html_output(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test HTML report generation."""
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = Path("momentum_report.html")

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch(
                    "tenets.cli.commands.momentum.ReportGenerator", return_value=mock_report_gen
                ):
                    with patch("tenets.cli.commands.momentum.get_logger"):
                        with patch("tenets.cli.commands.momentum.click.confirm", return_value=False):
                            result = runner.invoke(momentum, [".", "--format", "html"])

                            assert result.exit_code == 0
                            assert "Momentum report generated: momentum" in result.stdout
                            assert "Would you like to open it in your browser now?" in result.stdout
                            mock_report_gen.generate.assert_called_once()

    @pytest.mark.skipif(
        sys.version_info[:2] >= (3, 13),
        reason="Threading tests hang with coverage on Python 3.13+"
    )
    def test_momentum_html_opens_browser(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test HTML report opens browser when confirmed."""
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = Path("momentum_report.html")

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch(
                    "tenets.cli.commands.momentum.ReportGenerator", return_value=mock_report_gen
                ):
                    with patch("tenets.cli.commands.momentum.get_logger"):
                        with patch("tenets.cli.commands.momentum.click.confirm", return_value=True):
                            with patch("tenets.cli.commands.momentum.webbrowser.open") as mock_browser:
                                result = runner.invoke(momentum, [".", "--format", "html"])

                                assert result.exit_code == 0
                                assert "Momentum report generated: momentum" in result.stdout
                                assert "✓ Opened in browser" in result.stdout
                                mock_browser.assert_called_once()

    def test_momentum_markdown_output(
        self, runner, mock_momentum_tracker, mock_git_analyzer, tmp_path
    ):
        """Test Markdown report generation."""
        output_file = tmp_path / "momentum.md"
        mock_report_gen = MagicMock()
        mock_report_gen.generate.return_value = output_file

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch(
                    "tenets.cli.commands.momentum.ReportGenerator", return_value=mock_report_gen
                ):
                    with patch("tenets.cli.commands.momentum.get_logger"):
                        result = runner.invoke(
                            momentum, [".", "--format", "markdown", "--output", str(output_file)]
                        )

                        assert result.exit_code == 0
                        assert f"Momentum report generated: {output_file}" in result.stdout


class TestMomentumTrendDisplay:
    """Test velocity trend display."""

    def test_positive_trend_display(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test positive trend display."""
        mock_momentum_tracker.track_momentum.return_value["velocity_trend"] = 20.5

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, ["."])

                    assert result.exit_code == 0
                    assert "Trend: ↑ +20.5%" in result.stdout

    def test_negative_trend_display(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test negative trend display."""
        mock_momentum_tracker.track_momentum.return_value["velocity_trend"] = -10.0

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, ["."])

                    assert result.exit_code == 0
                    assert "Trend: ↓ -10.0%" in result.stdout

    def test_stable_trend_display(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test stable trend display."""
        mock_momentum_tracker.track_momentum.return_value["velocity_trend"] = 0

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, ["."])

                    assert result.exit_code == 0
                    assert "Trend: → Stable" in result.stdout


class TestMomentumErrorHandling:
    """Test error handling scenarios."""

    def test_momentum_path_not_exists(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test error when path doesn't exist."""
        result = runner.invoke(momentum, ["nonexistent/path"])

        assert result.exit_code != 0
        assert "does not exist" in result.stdout.lower() or "invalid" in result.stdout.lower()

        """Test error during momentum tracking."""
        mock_momentum_tracker.track_momentum.side_effect = Exception("Tracking failed")

        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, ["."])

                    assert result.exit_code != 0
                    assert "Tracking failed" in result.stdout


class TestMomentumSummary:
    """Test summary output."""

    def test_complete_summary(self, runner, mock_momentum_tracker, mock_git_analyzer):
        """Test complete summary output."""
        with patch(
            "tenets.cli.commands.momentum.MomentumTracker", return_value=mock_momentum_tracker
        ):
            with patch("tenets.cli.commands.momentum.GitAnalyzer", return_value=mock_git_analyzer):
                with patch("tenets.cli.commands.momentum.get_logger"):
                    result = runner.invoke(momentum, [".", "--team", "--forecast"])

                    assert result.exit_code == 0
                    assert "MOMENTUM SUMMARY" in result.stdout
                    assert "Current Velocity:" in result.stdout
                    assert "Trend:" in result.stdout
                    assert "Team Size:" in result.stdout
                    assert "Productivity:" in result.stdout
                    assert "Forecast Confidence:" in result.stdout
