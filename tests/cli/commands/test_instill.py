"""Unit tests for the instill CLI command.

Tests cover all instill functionality including:
- Tenet injection with different frequencies
- Session management and tracking
- File pinning and unpinning
- Injection statistics and analysis
- Configuration management
- History tracking and export
- Dry run mode
- Error handling
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from tenets.cli.commands.instill import instill


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_tenets():
    """Create a mock Tenets instance with instiller."""
    mock = MagicMock()

    # Mock instiller
    mock.instiller = MagicMock()
    mock.instiller.session_histories = {}
    mock.instiller.metrics_tracker = MagicMock()
    mock.instiller.metrics_tracker.instillations = []
    mock.instiller.metrics_tracker.get_metrics = MagicMock(
        return_value={
            "total_instillations": 10,
            "total_tenets_instilled": 25,
            "total_token_increase": 5000,
            "avg_tenets_per_context": 2.5,
            "avg_token_increase": 500,
            "avg_complexity": 3.2,
            "strategy_distribution": {"top": 5, "inline": 3, "bottom": 2},
            "skip_distribution": {"frequency": 3, "manual": 2},
        }
    )

    # Mock instill result
    mock_result = MagicMock()
    mock_result.skip_reason = None
    mock_result.tenets_instilled = [MagicMock(content="Always use type hints", id="abc123")]
    mock_result.session = "default"
    mock_result.strategy_used = "top"
    mock_result.token_increase = 250
    mock_result.complexity_score = 3.5

    mock.instiller.instill.return_value = (
        "Modified content",
        {"system_instruction_injected": True, "tenets_injected": True},
    )
    mock.instiller._cache = {"last_key": mock_result}

    # Mock tenet manager
    mock.tenet_manager = MagicMock()
    mock.get_pending_tenets = MagicMock(
        return_value=[
            MagicMock(
                id="abc123",
                content="Always use type hints",
                priority=MagicMock(value="high"),
                category=MagicMock(value="style"),
                created_at=datetime.now(),
                instilled_at=None,
                metrics=MagicMock(
                    injection_count=5, contexts_appeared_in=3, reinforcement_needed=False
                ),
            )
        ]
    )

    # Mock file management
    mock.add_file_to_session = MagicMock(return_value=True)
    mock.add_folder_to_session = MagicMock(return_value=3)
    mock.config = MagicMock()
    mock.config.custom = {"pinned_files": {}}

    return mock


@pytest.fixture
def mock_config():
    """Create a mock TenetsConfig."""
    config = MagicMock()
    config.tenet = MagicMock()
    config.tenet.injection_frequency = "periodic"
    config.tenet.injection_interval = 5
    config.tenet.session_complexity_threshold = 3.0
    config.tenet.min_session_length = 3
    config.tenet.max_per_context = 5
    config.tenet.injection_strategy = "top"
    config.tenet.decay_rate = 0.9
    config.tenet.reinforcement_interval = 10
    config.tenet.track_injection_history = True
    config.tenet.session_aware = True
    config.config_file = Path(".tenets.yml")
    config.save = MagicMock()
    return config


class TestInstillBasicFunctionality:
    """Test basic instill command functionality."""

    def test_basic_instill(self, runner, mock_tenets, mock_config):
        """Test basic tenet instillation."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, [])

                assert result.exit_code == 0
                assert "Tenets Instilled" in result.stdout
                mock_tenets.instiller.instill.assert_called()

    def test_instill_with_force(self, runner, mock_tenets, mock_config):
        """Test forced instillation."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--force"])

                assert result.exit_code == 0
                call_args = mock_tenets.instiller.instill.call_args[1]
                assert call_args["force"] is True

    def test_instill_with_session(self, runner, mock_tenets, mock_config):
        """Test instillation with specific session."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--session", "feature-x"])

                assert result.exit_code == 0
                call_args = mock_tenets.instiller.instill.call_args[1]
                assert call_args["session"] == "feature-x"

    def test_instill_with_frequency_override(self, runner, mock_tenets, mock_config):
        """Test overriding injection frequency."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--frequency", "always", "--interval", "3"])

                assert result.exit_code == 0
                # Frequency should be used in logic


class TestInstillConfigurationCommands:
    """Test configuration-related instill commands."""

    def test_show_config(self, runner, mock_tenets, mock_config):
        """Test showing injection configuration."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--show-config"])

                assert result.exit_code == 0
                assert "Tenet Injection Configuration" in result.stdout
                assert "Frequency Mode" in result.stdout
                assert "periodic" in result.stdout

    def test_set_frequency(self, runner, mock_tenets, mock_config):
        """Test setting injection frequency."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--set-frequency", "adaptive", "--set-interval", "7"])

                assert result.exit_code == 0
                assert "Configuration Updated" in result.stdout
                assert mock_config.tenet.injection_frequency == "adaptive"
                assert mock_config.tenet.injection_interval == 7
                mock_config.save.assert_called_once()

    def test_set_invalid_frequency(self, runner, mock_tenets, mock_config):
        """Test setting invalid frequency."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--set-frequency", "invalid"])

                assert result.exit_code == 1
                assert "Invalid frequency" in result.stdout


class TestInstillSessionManagement:
    """Test session management commands."""

    def test_list_sessions(self, runner, mock_tenets, mock_config):
        """Test listing tracked sessions."""
        mock_tenets.instiller.session_histories = {
            "session1": MagicMock(
                last_injection=datetime.now() - timedelta(hours=2),
                get_stats=lambda: {
                    "total_distills": 10,
                    "total_injections": 5,
                    "injection_rate": 0.5,
                    "average_complexity": 3.2,
                },
            )
        }

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--list-sessions"])

                assert result.exit_code == 0
                assert "Tracked Sessions" in result.stdout
                assert "session1" in result.stdout

    def test_reset_session(self, runner, mock_tenets, mock_config):
        """Test resetting session history."""
        mock_tenets.instiller.reset_session_history = MagicMock(return_value=True)

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--reset-session", "--session", "test-session"])

                assert result.exit_code == 0
                assert "Reset injection history" in result.stdout
                mock_tenets.instiller.reset_session_history.assert_called_with("test-session")

    def test_clear_all_sessions(self, runner, mock_tenets, mock_config):
        """Test clearing all session histories."""
        mock_tenets.instiller.session_histories = {"s1": {}, "s2": {}}
        mock_tenets.instiller._save_session_histories = MagicMock()

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                with patch("typer.confirm", return_value=True):
                    app = typer.Typer()
                    app.command()(instill)

                    result = runner.invoke(app, ["--clear-all-sessions"])

                    assert result.exit_code == 0
                    assert "Cleared 2 session histories" in result.stdout
                    assert len(mock_tenets.instiller.session_histories) == 0


class TestInstillFileManagement:
    """Test file pinning functionality."""

    def test_add_file(self, runner, mock_tenets, mock_config):
        """Test pinning a file to session."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--add-file", "src/core.py", "--session", "main"])

                assert result.exit_code == 0
                assert "Pinned: src/core.py" in result.stdout
                mock_tenets.add_file_to_session.assert_called_with("src/core.py", session="main")

    def test_add_folder(self, runner, mock_tenets, mock_config):
        """Test pinning all files in a folder."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--add-folder", "src/auth", "--session", "auth-work"])

                assert result.exit_code == 0
                assert "Pinned 3 files from: src/auth" in result.stdout
                mock_tenets.add_folder_to_session.assert_called_with(
                    "src/auth", session="auth-work"
                )

    def test_remove_file(self, runner, mock_tenets, mock_config):
        """Test unpinning a file."""
        mock_tenets.config.custom = {"pinned_files": {"test": [str(Path("src/file.py").resolve())]}}

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--remove-file", "src/file.py", "--session", "test"])

                assert result.exit_code == 0
                assert "Unpinned: src/file.py" in result.stdout

    def test_list_pinned(self, runner, mock_tenets, mock_config):
        """Test listing pinned files."""
        mock_tenets.config.custom = {"pinned_files": {"default": ["file1.py", "file2.py"]}}

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--list-pinned"])

                assert result.exit_code == 0
                assert "Pinned Files" in result.stdout
                assert "file1.py" in result.stdout
                assert "file2.py" in result.stdout


class TestInstillStatisticsAndAnalysis:
    """Test statistics and analysis commands."""

    def test_show_stats(self, runner, mock_tenets, mock_config):
        """Test showing injection statistics."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--stats"])

                assert result.exit_code == 0
                assert "Injection Statistics" in result.stdout
                assert "Total Instillations: 10" in result.stdout
                assert "Total Tenets: 25" in result.stdout

    def test_show_history(self, runner, mock_tenets, mock_config):
        """Test showing injection history."""
        mock_tenets.instiller.metrics_tracker.instillations = [
            {
                "timestamp": datetime.now().isoformat(),
                "session": "test",
                "tenet_count": 3,
                "token_increase": 500,
                "strategy": "top",
                "complexity": 3.5,
                "skip_reason": None,
            }
        ]

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--list-history"])

                assert result.exit_code == 0
                assert "Injection History" in result.stdout
                assert "test" in result.stdout

    def test_analyze_effectiveness(self, runner, mock_tenets, mock_config):
        """Test analyzing injection effectiveness."""
        mock_tenets.instiller.analyze_effectiveness = MagicMock(
            return_value={
                "configuration": {"frequency": "periodic", "interval": 5},
                "instillation_metrics": {"total_instillations": 10},
                "tenet_effectiveness": {"total_tenets": 5, "by_priority": {"high": 3}},
                "recommendations": ["Increase injection frequency"],
            }
        )

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--analyze"])

                assert result.exit_code == 0
                assert "Tenet Injection Analysis" in result.stdout
                assert "Recommendations" in result.stdout


class TestInstillListingCommands:
    """Test listing and display commands."""

    def test_list_pending_tenets(self, runner, mock_tenets, mock_config):
        """Test listing pending tenets."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--list-pending"])

                assert result.exit_code == 0
                assert "Pending Tenets" in result.stdout
                assert "Always use type hints" in result.stdout

    def test_list_pending_empty(self, runner, mock_tenets, mock_config):
        """Test listing when no pending tenets."""
        mock_tenets.get_pending_tenets.return_value = []

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--list-pending"])

                assert result.exit_code == 0
                assert "No pending tenets found" in result.stdout
                assert "tenets tenet add" in result.stdout


class TestInstillExportAndDryRun:
    """Test export and dry run functionality."""

    def test_dry_run(self, runner, mock_tenets, mock_config):
        """Test dry run mode."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--dry-run"])

                assert result.exit_code == 0
                assert "Would instill the following tenets" in result.stdout
                assert "Always use type hints" in result.stdout

    def test_export_history_json(self, runner, mock_tenets, mock_config, tmp_path):
        """Test exporting history to JSON."""
        output_file = tmp_path / "history.json"
        mock_tenets.instiller.export_instillation_history = MagicMock()

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(
                    app, ["--export-history", str(output_file), "--export-format", "json"]
                )

                assert result.exit_code == 0
                assert "Exported history to" in result.stdout
                mock_tenets.instiller.export_instillation_history.assert_called_with(
                    output_file, format="json", session=None
                )

    def test_export_history_csv(self, runner, mock_tenets, mock_config, tmp_path):
        """Test exporting history to CSV."""
        output_file = tmp_path / "history.csv"
        mock_tenets.instiller.export_instillation_history = MagicMock()

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(
                    app,
                    [
                        "--export-history",
                        str(output_file),
                        "--export-format",
                        "csv",
                        "--session",
                        "test-session",
                    ],
                )

                assert result.exit_code == 0
                mock_tenets.instiller.export_instillation_history.assert_called_with(
                    output_file, format="csv", session="test-session"
                )


class TestInstillErrorHandling:
    """Test error handling scenarios."""

    def test_no_tenet_system(self, runner, mock_config):
        """Test error when tenet system is not available."""
        mock_tenets = MagicMock()
        mock_tenets.instiller = None

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, [])

                assert result.exit_code == 1
                assert "Tenet system is not available" in result.stdout

    def test_instill_skip_reason(self, runner, mock_tenets, mock_config):
        """Test when injection is skipped."""
        mock_result = MagicMock()
        mock_result.skip_reason = "Frequency limit not reached"
        mock_tenets.instiller._cache = {"last_key": mock_result}

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, [])

                assert result.exit_code == 0
                assert "Injection skipped" in result.stdout
                assert "Frequency limit not reached" in result.stdout

    def test_reset_session_no_session_specified(self, runner, mock_tenets, mock_config):
        """Test reset without session specified."""
        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--reset-session"])

                assert result.exit_code == 1
                assert "--reset-session requires --session" in result.stdout

    def test_export_history_error(self, runner, mock_tenets, mock_config, tmp_path):
        """Test error during history export."""
        output_file = tmp_path / "history.json"
        mock_tenets.instiller.export_instillation_history.side_effect = Exception("Export failed")

        with patch("tenets.cli.commands.instill.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.instill.Tenets", return_value=mock_tenets):
                app = typer.Typer()
                app.command()(instill)

                result = runner.invoke(app, ["--export-history", str(output_file)])

                assert result.exit_code == 1
                assert "Error exporting history" in result.stdout
                assert "Export failed" in result.stdout
