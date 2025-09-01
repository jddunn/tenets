"""
Unit tests for the TenetsConfig configuration system.

This module tests the configuration loading, validation, and management
functionality. It ensures configs can be loaded from files, environment
variables, and dictionaries, with proper precedence and validation.

Test Coverage:
    - Configuration initialization with defaults
    - Loading from YAML/JSON files
    - Environment variable overrides
    - Configuration validation
    - Path resolution
    - Subsystem configurations (scanner, ranking, tenet, etc.)
    - Configuration saving and exporting
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

from tenets.config import (
    CacheConfig,
    GitConfig,
    OutputConfig,
    RankingConfig,
    ScannerConfig,
    TenetConfig,
    TenetsConfig,
)


class TestTenetsConfigInitialization:
    """Test suite for TenetsConfig initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = TenetsConfig()

        # Check default values
        assert config.max_tokens == 100_000
        assert config.debug is False
        assert config.quiet is False

        # Check subsystem configs are initialized
        assert isinstance(config.scanner, ScannerConfig)
        assert isinstance(config.ranking, RankingConfig)
        assert isinstance(config.tenet, TenetConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.git, GitConfig)

    def test_init_with_project_root(self, temp_dir):
        """Test initialization with custom project root."""
        config = TenetsConfig(project_root=temp_dir)

        assert config.project_root == temp_dir
        assert config.project_root.is_absolute()

    def test_auto_find_config_file(self, temp_dir, monkeypatch):
        """Test automatic config file discovery."""
        # Create a config file in temp dir
        config_file = temp_dir / ".tenets.yml"
        config_data = {"max_tokens": 50000}
        config_file.write_text(yaml.dump(config_data))

        # Change to temp dir
        monkeypatch.chdir(temp_dir)

        config = TenetsConfig()

        assert config.config_file == config_file
        assert config.max_tokens == 50000

    def test_init_with_explicit_config_file(self, temp_dir):
        """Test initialization with explicit config file."""
        config_file = temp_dir / "custom.yml"
        config_data = {"max_tokens": 75000, "debug": True, "scanner": {"max_files": 500}}
        config_file.write_text(yaml.dump(config_data))

        config = TenetsConfig(config_file=config_file)

        assert config.config_file == config_file
        assert config.max_tokens == 75000
        assert config.debug is True
        assert config.scanner.max_files == 500


class TestConfigFileLoading:
    """Test suite for configuration file loading."""

    def test_load_yaml_config(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_file = temp_dir / "config.yml"
        config_data = {
            "max_tokens": 60000,
            "scanner": {"respect_gitignore": False, "max_file_size": 1000000},
            "ranking": {"algorithm": "thorough", "threshold": 0.3},
        }
        config_file.write_text(yaml.dump(config_data))

        config = TenetsConfig(config_file=config_file)

        assert config.max_tokens == 60000
        assert config.scanner.respect_gitignore is False
        assert config.scanner.max_file_size == 1000000
        assert config.ranking.algorithm == "thorough"
        assert config.ranking.threshold == 0.3

    def test_load_json_config(self, temp_dir):
        """Test loading configuration from JSON file."""
        config_file = temp_dir / "config.json"
        config_data = {
            "max_tokens": 80000,
            "output": {"default_format": "json", "syntax_highlighting": False},
        }
        config_file.write_text(json.dumps(config_data))

        config = TenetsConfig(config_file=config_file)

        assert config.max_tokens == 80000
        assert config.output.default_format == "json"
        assert config.output.syntax_highlighting is False

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML raises error."""
        config_file = temp_dir / "invalid.yml"
        config_file.write_text("invalid: yaml: content:")

        with pytest.raises(yaml.YAMLError):
            TenetsConfig(config_file=config_file)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file is handled gracefully."""
        # When explicit file is given but doesn't exist
        with pytest.raises(FileNotFoundError):
            TenetsConfig(config_file=Path("/nonexistent/config.yml"))

    def test_config_file_search_order(self, temp_dir, monkeypatch):
        """Test config file search follows correct precedence."""
        monkeypatch.chdir(temp_dir)

        # Create multiple config files
        (temp_dir / ".tenets.yml").write_text("max_tokens: 1000")
        (temp_dir / ".tenets.yaml").write_text("max_tokens: 2000")
        (temp_dir / "tenets.yml").write_text("max_tokens: 3000")

        config = TenetsConfig()

        # Should use .tenets.yml (first in search order)
        assert config.max_tokens == 1000


class TestEnvironmentVariables:
    """Test suite for environment variable configuration."""

    def test_env_var_override_top_level(self, monkeypatch):
        """Test environment variables override top-level config."""
        monkeypatch.setenv("TENETS_MAX_TOKENS", "200000")
        monkeypatch.setenv("TENETS_DEBUG", "true")
        monkeypatch.setenv("TENETS_QUIET", "yes")

        config = TenetsConfig()

        assert config.max_tokens == 200000
        assert config.debug is True
        assert config.quiet is True

    def test_env_var_override_subsystem(self, monkeypatch):
        """Test environment variables override subsystem configs."""
        monkeypatch.setenv("TENETS_SCANNER_MAX_FILES", "5000")
        monkeypatch.setenv("TENETS_SCANNER_RESPECT_GITIGNORE", "false")
        monkeypatch.setenv("TENETS_RANKING_ALGORITHM", "ml")
        monkeypatch.setenv("TENETS_RANKING_THRESHOLD", "0.5")

        config = TenetsConfig()

        assert config.scanner.max_files == 5000
        assert config.scanner.respect_gitignore is False
        assert config.ranking.algorithm == "ml"
        assert config.ranking.threshold == 0.5

    @pytest.mark.skipif(
        sys.version_info[:2] >= (3, 13), reason="Hangs on Python 3.13+ due to pytest issue"
    )
    def test_env_var_list_parsing(self, monkeypatch):
        """Test environment variables with list values."""
        monkeypatch.setenv("TENETS_SCANNER_ADDITIONAL_IGNORE_PATTERNS", "*.log,*.tmp,*.bak")

        config = TenetsConfig()

        assert "*.log" in config.scanner.additional_ignore_patterns
        assert "*.tmp" in config.scanner.additional_ignore_patterns
        assert "*.bak" in config.scanner.additional_ignore_patterns

    def test_env_var_precedence_over_file(self, temp_dir, monkeypatch):
        """Test environment variables take precedence over file config."""
        # Create config file
        config_file = temp_dir / ".tenets.yml"
        config_file.write_text("max_tokens: 50000")

        # Set environment variable
        monkeypatch.setenv("TENETS_MAX_TOKENS", "150000")
        monkeypatch.chdir(temp_dir)

        config = TenetsConfig()

        # Env var should override file
        assert config.max_tokens == 150000


class TestSubsystemConfigs:
    """Test suite for subsystem configurations."""

    def test_scanner_config_defaults(self):
        """Test ScannerConfig default values."""
        scanner = ScannerConfig()

        assert scanner.respect_gitignore is True
        assert scanner.follow_symlinks is False
        assert scanner.max_file_size == 5_000_000
        assert scanner.max_files == 10_000

        # Test new test exclusion fields
        assert scanner.exclude_tests_by_default is True
        assert isinstance(scanner.test_patterns, list)
        assert len(scanner.test_patterns) > 0
        assert isinstance(scanner.test_directories, list)
        assert len(scanner.test_directories) > 0

        # Check that common test patterns are included
        test_patterns_str = ",".join(scanner.test_patterns)
        assert "test_*.py" in test_patterns_str
        assert "*_test.py" in test_patterns_str
        assert "*.test.js" in test_patterns_str
        assert "*.spec.js" in test_patterns_str
        assert "*Test.java" in test_patterns_str
        assert "*_test.go" in test_patterns_str

        # Check that common test directories are included
        test_dirs_str = ",".join(scanner.test_directories)
        assert "test" in test_dirs_str
        assert "tests" in test_dirs_str
        assert "__tests__" in test_dirs_str
        assert "spec" in test_dirs_str
        assert scanner.binary_check is True
        assert scanner.encoding == "utf-8"
        assert scanner.workers == 4
        assert "*.pyc" in scanner.additional_ignore_patterns

    def test_ranking_config_defaults(self):
        """Test RankingConfig default values."""
        ranking = RankingConfig()

        assert ranking.algorithm == "balanced"
        assert ranking.threshold == 0.1
        assert ranking.use_tfidf is True
        assert ranking.use_embeddings is False
        assert ranking.embedding_model == "all-MiniLM-L6-v2"
        assert ranking.workers == 2
        assert "keyword_match" in ranking.custom_weights

    def test_tenet_config_defaults(self):
        """Test TenetConfig default values."""
        tenet = TenetConfig()

        assert tenet.auto_instill is True
        assert tenet.max_per_context == 5
        assert tenet.reinforcement is True
        assert tenet.injection_strategy == "strategic"
        assert tenet.min_distance_between == 1000
        assert tenet.prefer_natural_breaks is True

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        cache = CacheConfig()

        assert cache.enabled is True
        assert cache.ttl_days == 7
        assert cache.max_size_mb == 500
        assert cache.compression is False
        assert cache.memory_cache_size == 1000
        assert cache.sqlite_pragmas["journal_mode"] == "WAL"

    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        output = OutputConfig()

        assert output.default_format == "markdown"
        assert output.syntax_highlighting is True
        assert output.line_numbers is False
        assert output.max_line_length == 120
        assert output.include_metadata is True
        assert output.compression_threshold == 10_000
        assert output.summary_ratio == 0.25

    def test_git_config_defaults(self):
        """Test GitConfig default values."""
        git = GitConfig()

        assert git.enabled is True
        assert git.include_history is True
        assert git.history_limit == 100
        assert git.include_blame is False
        assert git.include_stats is True
        assert "dependabot[bot]" in git.ignore_authors
        assert "main" in git.main_branches


class TestPathResolution:
    """Test suite for path resolution in configuration."""

    def test_cache_directory_creation(self, temp_dir):
        """Test cache directory is created if it doesn't exist."""
        cache_dir = temp_dir / "custom_cache"
        config = TenetsConfig()
        config.cache.directory = cache_dir
        config._resolve_paths()

        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_tenet_storage_path_creation(self, temp_dir):
        """Test tenet storage path is created if it doesn't exist."""
        storage_path = temp_dir / "custom_tenets"
        config = TenetsConfig()
        config.tenet.storage_path = storage_path
        config._resolve_paths()

        assert storage_path.exists()
        assert storage_path.is_dir()

    def test_default_paths_in_home(self):
        """Test default paths are created in home directory."""
        config = TenetsConfig()

        expected_cache = Path.home() / ".tenets" / "cache"
        expected_tenets = Path.home() / ".tenets" / "tenets"

        assert config.cache.directory == expected_cache
        assert config.tenet.storage_path == expected_tenets


class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_validate_max_tokens_minimum(self):
        """Test max_tokens minimum validation."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1000"):
            config = TenetsConfig()
            config.max_tokens = 500
            config._validate()

    def test_validate_max_tokens_maximum(self):
        """Test max_tokens maximum validation."""
        with pytest.raises(ValueError, match="max_tokens cannot exceed 2,000,000"):
            config = TenetsConfig()
            config.max_tokens = 3_000_000
            config._validate()

    def test_validate_ranking_algorithm(self):
        """Test ranking algorithm validation."""
        with pytest.raises(ValueError, match="Invalid ranking algorithm"):
            config = TenetsConfig()
            config.ranking.algorithm = "invalid_algo"
            config._validate()

    def test_validate_ranking_threshold(self):
        """Test ranking threshold validation."""
        with pytest.raises(ValueError, match="Ranking threshold must be between 0 and 1"):
            config = TenetsConfig()
            config.ranking.threshold = 1.5
            config._validate()

    def test_validate_cache_ttl(self):
        """Test cache TTL validation."""
        with pytest.raises(ValueError, match="Cache TTL cannot be negative"):
            config = TenetsConfig()
            config.cache.ttl_days = -1
            config._validate()

    def test_validate_output_format(self):
        """Test output format validation."""
        with pytest.raises(ValueError, match="Invalid output format"):
            config = TenetsConfig()
            config.output.default_format = "invalid_format"
            config._validate()


class TestConfigSerialization:
    """Test suite for configuration serialization."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TenetsConfig()
        config.max_tokens = 75000
        config.debug = True

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["max_tokens"] == 75000
        assert config_dict["debug"] is True
        assert "scanner" in config_dict
        assert "ranking" in config_dict
        assert isinstance(config_dict["scanner"], dict)

    def test_save_to_yaml(self, temp_dir):
        """Test saving configuration to YAML file."""
        config = TenetsConfig()
        config.max_tokens = 90000
        config.scanner.max_files = 2000

        save_path = temp_dir / "saved_config.yml"
        config.save(save_path)

        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["max_tokens"] == 90000
        assert loaded["scanner"]["max_files"] == 2000

    def test_save_to_json(self, temp_dir):
        """Test saving configuration to JSON file."""
        config = TenetsConfig()
        config.max_tokens = 95000

        save_path = temp_dir / "saved_config.json"
        config.save(save_path)

        assert save_path.exists()

        # Load and verify
        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded["max_tokens"] == 95000

    def test_save_without_path(self, temp_dir):
        """Test saving without explicit path uses config_file."""
        config_file = temp_dir / "config.yml"
        config = TenetsConfig(config_file=config_file)
        config.max_tokens = 85000

        config.save()  # Should use config_file

        with open(config_file) as f:
            loaded = yaml.safe_load(f)

        assert loaded["max_tokens"] == 85000

    def test_save_without_any_path(self):
        """Test saving without any path raises error."""
        config = TenetsConfig()

        with pytest.raises(ValueError, match="No path specified"):
            config.save()


class TestConfigProperties:
    """Test suite for configuration properties."""

    def test_cache_dir_property(self):
        """Test cache_dir property returns correct path."""
        config = TenetsConfig()
        assert config.cache_dir == config.cache.directory

    def test_scanner_workers_property(self):
        """Test scanner_workers property."""
        config = TenetsConfig()
        config.scanner.workers = 8
        assert config.scanner_workers == 8

    def test_ranking_workers_property(self):
        """Test ranking_workers property."""
        config = TenetsConfig()
        config.ranking.workers = 4
        assert config.ranking_workers == 4

    def test_ranking_algorithm_property(self):
        """Test ranking_algorithm property."""
        config = TenetsConfig()
        config.ranking.algorithm = "thorough"
        assert config.ranking_algorithm == "thorough"

    def test_respect_gitignore_property(self):
        """Test respect_gitignore property."""
        config = TenetsConfig()
        assert config.respect_gitignore is True
        config.scanner.respect_gitignore = False
        assert config.respect_gitignore is False

    def test_auto_instill_tenets_property(self):
        """Test auto_instill_tenets property."""
        config = TenetsConfig()
        assert config.auto_instill_tenets is True
        config.tenet.auto_instill = False
        assert config.auto_instill_tenets is False

    def test_tenet_injection_config_property(self):
        """Test tenet_injection_config property returns dict."""
        config = TenetsConfig()
        injection_config = config.tenet_injection_config

        assert isinstance(injection_config, dict)
        assert injection_config["strategy"] == "strategic"
        assert injection_config["min_distance_between"] == 1000
        assert injection_config["prefer_natural_breaks"] is True
        assert injection_config["reinforce_at_end"] is True


class TestConfigEdgeCases:
    """Test suite for configuration edge cases."""

    def test_nested_custom_config(self):
        """Test deeply nested custom configuration."""
        config = TenetsConfig()
        config.custom = {"level1": {"level2": {"level3": {"value": "deep"}}}}

        config_dict = config.to_dict()
        assert config_dict["custom"]["level1"]["level2"]["level3"]["value"] == "deep"

    def test_empty_config_file(self, temp_dir):
        """Test loading empty config file uses defaults."""
        config_file = temp_dir / "empty.yml"
        config_file.write_text("")

        config = TenetsConfig(config_file=config_file)

        # Should use defaults
        assert config.max_tokens == 100_000

    def test_partial_subsystem_config(self, temp_dir):
        """Test partial subsystem config merges with defaults."""
        config_file = temp_dir / "partial.yml"
        config_data = {
            "scanner": {
                "max_files": 3000
                # Other scanner fields not specified
            }
        }
        config_file.write_text(yaml.dump(config_data))

        config = TenetsConfig(config_file=config_file)

        # Specified value
        assert config.scanner.max_files == 3000
        # Default values
        assert config.scanner.respect_gitignore is True
        assert config.scanner.max_file_size == 5_000_000

    def test_config_with_extra_fields(self, temp_dir):
        """Test config with unknown fields stores them in custom."""
        config_file = temp_dir / "extra.yml"
        config_data = {
            "max_tokens": 70000,
            "unknown_field": "value",
            "another_unknown": {"nested": "data"},
        }
        config_file.write_text(yaml.dump(config_data))

        config = TenetsConfig(config_file=config_file)

        assert config.max_tokens == 70000
        assert config.custom["unknown_field"] == "value"
        assert config.custom["another_unknown"]["nested"] == "data"
