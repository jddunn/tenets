"""Unit tests for the config CLI command.

Tests cover all configuration management functionality including:
- Initializing configuration files
- Showing configuration
- Setting configuration values
- Validating configuration
- Cache management
- Export and diff operations
- Model and summarizer information
- Error handling
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.config import config_app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Create a mock TenetsConfig."""
    config = MagicMock()
    config.max_tokens = 100000
    config.ranking = MagicMock()
    config.ranking.algorithm = "balanced"
    config.ranking.threshold = 0.10
    config.summarizer = MagicMock()
    config.summarizer.default_mode = "auto"
    config.summarizer.target_ratio = 0.3
    config.summarizer.llm_model = "gpt-3.5-turbo"
    config.cache = MagicMock()
    config.cache.enabled = True
    config.cache.ttl_days = 7
    config.cache.max_size_mb = 500
    config.cache.directory = Path.home() / ".tenets" / "cache"
    config.git = MagicMock()
    config.git.enabled = True
    config.tenet = MagicMock()
    config.tenet.auto_instill = True
    config.config_file = Path(".tenets.yml")
    config.to_dict = MagicMock(
        return_value={
            "max_tokens": 100000,
            "ranking": {"algorithm": "balanced", "threshold": 0.10},
            "summarizer": {"default_mode": "auto", "target_ratio": 0.3},
            "cache": {"enabled": True, "ttl_days": 7, "max_size_mb": 500},
        }
    )
    config.save = MagicMock()
    return config


class TestConfigInit:
    """Test configuration initialization."""

    def test_init_new_config(self, runner, tmp_path):
        """Test creating a new config file."""
        config_file = tmp_path / ".tenets.yml"

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(config_app, ["init"])

            assert result.exit_code == 0
            assert "Created .tenets.yml" in result.stdout
            assert config_file.exists()

            # Check config content
            content = config_file.read_text()
            assert "max_tokens: 100000" in content
            assert "algorithm: balanced" in content
            assert "threshold: 0.10" in content

    def test_init_existing_config(self, runner, tmp_path):
        """Test init when config already exists."""
        config_file = tmp_path / ".tenets.yml"
        config_file.write_text("existing: config")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(config_app, ["init"])

            assert result.exit_code == 1
            assert "Config file .tenets.yml already exists" in result.stdout
            assert "Use --force to overwrite" in result.stdout

    def test_init_force_overwrite(self, runner, tmp_path):
        """Test forcing overwrite of existing config."""
        config_file = tmp_path / ".tenets.yml"
        config_file.write_text("existing: config")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(config_app, ["init", "--force"])

            assert result.exit_code == 0
            assert "Created .tenets.yml" in result.stdout

            # Check old content was replaced
            content = config_file.read_text()
            assert "existing: config" not in content
            assert "max_tokens: 100000" in content

    def test_init_instructions(self, runner, tmp_path):
        """Test that init shows next steps."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = runner.invoke(config_app, ["init"])

            assert result.exit_code == 0
            assert "Next steps:" in result.stdout
            assert "Edit .tenets.yml" in result.stdout
            assert "tenets config show" in result.stdout


class TestConfigShow:
    """Test showing configuration."""

    def test_show_full_config(self, runner, mock_config):
        """Test showing full configuration."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show"])

            assert result.exit_code == 0
            assert "max_tokens" in result.stdout
            assert "100000" in result.stdout
            assert "algorithm" in result.stdout
            assert "balanced" in result.stdout

    def test_show_specific_key(self, runner, mock_config):
        """Test showing specific config key."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show", "--key", "max_tokens"])

            assert result.exit_code == 0
            assert "max_tokens: 100000" in result.stdout

    def test_show_nested_key(self, runner, mock_config):
        """Test showing nested config key."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show", "--key", "ranking.algorithm"])

            assert result.exit_code == 0
            assert "ranking.algorithm: balanced" in result.stdout

    def test_show_nonexistent_key(self, runner, mock_config):
        """Test showing non-existent key."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show", "--key", "nonexistent"])

            assert result.exit_code == 1
            assert "Key not found: nonexistent" in result.stdout

    def test_show_json_format(self, runner, mock_config):
        """Test showing config in JSON format."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show", "--format", "json"])

            assert result.exit_code == 0
            # Verify it's valid JSON
            config_data = json.loads(result.stdout)
            assert config_data["max_tokens"] == 100000

    def test_show_models_info(self, runner, mock_config):
        """Test showing model information."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show", "--key", "models"])

            assert result.exit_code == 0
            assert "Supported LLM Models" in result.stdout
            assert "gpt" in result.stdout.lower() or "claude" in result.stdout.lower()

    def test_show_summarizers_info(self, runner, mock_config):
        """Test showing summarizer information."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["show", "--key", "summarizers"])

            assert result.exit_code == 0
            assert "Summarization Strategies" in result.stdout
            assert "extractive" in result.stdout
            assert "transformer" in result.stdout


class TestConfigSet:
    """Test setting configuration values."""

    def test_set_simple_value(self, runner, mock_config):
        """Test setting a simple config value."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "max_tokens", "150000"])

            assert result.exit_code == 0
            assert "Set max_tokens = 150000" in result.stdout
            assert mock_config.max_tokens == 150000

    def test_set_nested_value(self, runner, mock_config):
        """Test setting a nested config value."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "ranking.algorithm", "thorough"])

            assert result.exit_code == 0
            assert "Set ranking.algorithm = thorough" in result.stdout
            assert mock_config.ranking.algorithm == "thorough"

    def test_set_with_save(self, runner, mock_config):
        """Test setting and saving config."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "max_tokens", "200000", "--save"])

            assert result.exit_code == 0
            assert "Set max_tokens = 200000" in result.stdout
            assert "Saved to .tenets.yml" in result.stdout
            mock_config.save.assert_called_once()

    def test_set_boolean_value(self, runner, mock_config):
        """Test setting boolean config value."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "cache.enabled", "false"])

            assert result.exit_code == 0
            assert mock_config.cache.enabled is False

    def test_set_float_value(self, runner, mock_config):
        """Test setting float config value."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "ranking.threshold", "0.25"])

            assert result.exit_code == 0
            assert mock_config.ranking.threshold == 0.25

    def test_set_list_value(self, runner, mock_config):
        """Test setting list config value."""
        mock_config.scanner = MagicMock()
        mock_config.scanner.additional_ignore_patterns = []

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(
                config_app, ["set", "scanner.additional_ignore_patterns", "*.tmp,*.bak"]
            )

            assert result.exit_code == 0
            assert mock_config.scanner.additional_ignore_patterns == ["*.tmp", "*.bak"]

    def test_set_invalid_key(self, runner, mock_config):
        """Test setting invalid config key."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "invalid.key", "value"])

            assert result.exit_code == 1
            assert "Invalid configuration key: invalid.key" in result.stdout


class TestConfigValidate:
    """Test configuration validation."""

    def test_validate_valid_config(self, runner, mock_config):
        """Test validating a valid configuration."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["validate"])

            assert result.exit_code == 0
            assert "Configuration file .tenets.yml is valid" in result.stdout
            assert "Key Configuration Settings" in result.stdout
            assert "Max Tokens" in result.stdout
            assert "100000" in result.stdout

    def test_validate_specific_file(self, runner, mock_config, tmp_path):
        """Test validating a specific config file."""
        config_file = tmp_path / "custom.yml"
        config_file.write_text("max_tokens: 50000")

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["validate", "--file", str(config_file)])

            assert result.exit_code == 0
            assert f"Configuration file {config_file} is valid" in result.stdout

    def test_validate_no_config_file(self, runner, mock_config):
        """Test validation with default config."""
        mock_config.config_file = None

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["validate"])

            assert result.exit_code == 0
            assert "Using default configuration (no config file)" in result.stdout

    def test_validate_invalid_config(self, runner):
        """Test validation of invalid config."""
        with patch(
            "tenets.cli.commands.config.TenetsConfig", side_effect=Exception("Invalid YAML")
        ):
            result = runner.invoke(config_app, ["validate"])

            assert result.exit_code == 1
            assert "Configuration validation failed" in result.stdout
            assert "Invalid YAML" in result.stdout


class TestConfigCacheCommands:
    """Test cache management commands."""

    def test_clear_cache_with_confirmation(self, runner, mock_config):
        """Test clearing cache with confirmation."""
        mock_cache_manager = MagicMock()
        mock_cache_manager.clear_all = MagicMock()

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.config.CacheManager", return_value=mock_cache_manager):
                with patch("typer.confirm", return_value=True):
                    result = runner.invoke(config_app, ["clear-cache"])

                    assert result.exit_code == 0
                    assert "Cache cleared" in result.stdout
                    mock_cache_manager.clear_all.assert_called_once()

    def test_clear_cache_cancelled(self, runner, mock_config):
        """Test cancelling cache clear."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            with patch("typer.confirm", return_value=False):
                result = runner.invoke(config_app, ["clear-cache"])

                # typer.confirm with abort=True will exit
                assert result.exit_code == 1

    def test_clear_cache_forced(self, runner, mock_config):
        """Test forced cache clear."""
        mock_cache_manager = MagicMock()

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.config.CacheManager", return_value=mock_cache_manager):
                result = runner.invoke(config_app, ["clear-cache", "--yes"])

                assert result.exit_code == 0
                mock_cache_manager.clear_all.assert_called_once()

    def test_cleanup_cache(self, runner, mock_config):
        """Test cache cleanup."""
        mock_cache_manager = MagicMock()
        mock_cache_manager.analysis.disk.cleanup.return_value = "10 files deleted"
        mock_cache_manager.general.cleanup.return_value = "5 files deleted"

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.config.CacheManager", return_value=mock_cache_manager):
                result = runner.invoke(config_app, ["cleanup-cache"])

                assert result.exit_code == 0
                assert "Cache Cleanup" in result.stdout
                assert "Analysis deletions: 10 files deleted" in result.stdout
                assert "General deletions: 5 files deleted" in result.stdout

    def test_cache_stats(self, runner, mock_config):
        """Test showing cache statistics."""
        cache_dir = Path.home() / ".tenets" / "cache"

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.rglob") as mock_rglob:
                    # Mock some cache files
                    mock_files = [
                        MagicMock(is_file=lambda: True, stat=lambda: MagicMock(st_size=1024)),
                        MagicMock(is_file=lambda: True, stat=lambda: MagicMock(st_size=2048)),
                    ]
                    mock_rglob.return_value = mock_files

                    result = runner.invoke(config_app, ["cache-stats"])

                    assert result.exit_code == 0
                    assert "Cache Statistics" in result.stdout
                    assert "Total Files" in result.stdout
                    assert "Total Size" in result.stdout

    def test_cache_stats_no_cache(self, runner, mock_config):
        """Test cache stats when no cache exists."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            with patch("pathlib.Path.exists", return_value=False):
                result = runner.invoke(config_app, ["cache-stats"])

                assert result.exit_code == 0
                assert "Cache directory does not exist" in result.stdout


class TestConfigExportDiff:
    """Test export and diff operations."""

    def test_export_yaml(self, runner, mock_config, tmp_path):
        """Test exporting config to YAML."""
        output_file = tmp_path / "config.yml"

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["export", str(output_file)])

            assert result.exit_code == 0
            assert f"Configuration exported to {output_file}" in result.stdout
            mock_config.save.assert_called_once_with(output_file)

    def test_export_json(self, runner, mock_config, tmp_path):
        """Test exporting config to JSON."""
        output_file = tmp_path / "config.json"

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["export", str(output_file), "--format", "json"])

            assert result.exit_code == 0
            assert output_file.suffix == ".json"
            mock_config.save.assert_called_once()

    def test_export_auto_extension(self, runner, mock_config, tmp_path):
        """Test export auto-corrects file extension."""
        output_file = tmp_path / "config.txt"

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["export", str(output_file), "--format", "json"])

            assert result.exit_code == 0
            # Should change to .json
            call_args = mock_config.save.call_args[0][0]
            assert str(call_args).endswith(".json")

    def test_diff_current_vs_defaults(self, runner, mock_config):
        """Test diff between current and default config."""
        mock_config2 = MagicMock()
        mock_config2.to_dict.return_value = {
            "max_tokens": 50000,  # Different from mock_config
            "ranking": {"algorithm": "balanced", "threshold": 0.10},
        }

        with patch("tenets.cli.commands.config.TenetsConfig") as mock_cls:
            mock_cls.side_effect = [mock_config, mock_config2]

            result = runner.invoke(config_app, ["diff"])

            assert result.exit_code == 0
            assert "Configuration Differences" in result.stdout
            assert "max_tokens" in result.stdout
            assert "100000" in result.stdout
            assert "50000" in result.stdout

    def test_diff_two_files(self, runner, mock_config, tmp_path):
        """Test diff between two config files."""
        file1 = tmp_path / "config1.yml"
        file2 = tmp_path / "config2.yml"
        file1.write_text("max_tokens: 100000")
        file2.write_text("max_tokens: 200000")

        mock_config1 = MagicMock()
        mock_config1.to_dict.return_value = {"max_tokens": 100000}
        mock_config2 = MagicMock()
        mock_config2.to_dict.return_value = {"max_tokens": 200000}

        with patch("tenets.cli.commands.config.TenetsConfig") as mock_cls:
            mock_cls.side_effect = [mock_config1, mock_config2]

            result = runner.invoke(
                config_app, ["diff", "--file1", str(file1), "--file2", str(file2)]
            )

            assert result.exit_code == 0
            assert "Configuration Differences" in result.stdout
            assert "100000" in result.stdout
            assert "200000" in result.stdout

    def test_diff_no_differences(self, runner, mock_config):
        """Test diff when configs are identical."""
        with patch("tenets.cli.commands.config.TenetsConfig") as mock_cls:
            mock_cls.side_effect = [mock_config, mock_config]

            result = runner.invoke(config_app, ["diff"])

            assert result.exit_code == 0
            assert "No differences" in result.stdout


class TestConfigErrorHandling:
    """Test error handling scenarios."""

    def test_set_error(self, runner, mock_config):
        """Test error during set operation."""
        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["set", "max_tokens", "invalid"])

            assert result.exit_code == 1
            assert "Error setting configuration" in result.stdout

    def test_export_error(self, runner, mock_config):
        """Test error during export."""
        mock_config.save.side_effect = Exception("Export failed")

        with patch("tenets.cli.commands.config.TenetsConfig", return_value=mock_config):
            result = runner.invoke(config_app, ["export", "config.yml"])

            assert result.exit_code == 1
            assert "Error exporting configuration" in result.stdout
            assert "Export failed" in result.stdout

    def test_diff_error(self, runner):
        """Test error during diff."""
        with patch("tenets.cli.commands.config.TenetsConfig", side_effect=Exception("Load failed")):
            result = runner.invoke(config_app, ["diff"])

            assert result.exit_code == 1
            assert "Error comparing configurations" in result.stdout
            assert "Load failed" in result.stdout
