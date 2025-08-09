"""Configuration management for Tenets.

This module handles all configuration for the Tenets system, including loading
from files, environment variables, and providing defaults. Configuration can
be specified at multiple levels with proper precedence.

Configuration precedence (highest to lowest):
1. Runtime parameters (passed to methods)
2. Environment variables (TENETS_*)
3. Project config file (.tenets.yml in project)
4. User config file (~/.config/tenets/config.yml)
5. Default values

The configuration system is designed to work with zero configuration (sensible
defaults) while allowing full customization when needed.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from datetime import timedelta

from tenets.utils.logger import get_logger


@dataclass
class ScannerConfig:
    """Configuration for file scanning subsystem.

    Controls how tenets discovers and filters files in a codebase.

    Attributes:
        respect_gitignore: Whether to respect .gitignore files
        follow_symlinks: Whether to follow symbolic links
        max_file_size: Maximum file size in bytes to analyze
        max_files: Maximum number of files to scan
        binary_check: Whether to check for and skip binary files
        encoding: Default file encoding
        additional_ignore_patterns: Extra patterns to ignore
        additional_include_patterns: Extra patterns to include
        workers: Number of parallel workers for scanning
    """

    respect_gitignore: bool = True
    follow_symlinks: bool = False
    max_file_size: int = 5_000_000  # 5MB
    max_files: int = 10_000
    binary_check: bool = True
    encoding: str = "utf-8"
    additional_ignore_patterns: List[str] = field(
        default_factory=lambda: [
            "*.pyc",
            "*.pyo",
            "__pycache__",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.egg-info",
            "*.dist-info",
            ".tox",
            ".nox",
            ".coverage",
            ".hypothesis",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        ]
    )
    additional_include_patterns: List[str] = field(default_factory=list)
    workers: int = 4


@dataclass
class RankingConfig:
    """Configuration for relevance ranking system.

    Controls how files are scored and ranked for relevance to prompts.

    Attributes:
        algorithm: Default ranking algorithm (fast, balanced, thorough, ml)
        threshold: Minimum relevance score to include file
        use_tfidf: Whether to use TF-IDF for keyword matching
        use_embeddings: Whether to use semantic embeddings (requires ML)
        embedding_model: Which embedding model to use
        custom_weights: Custom weights for ranking factors
        workers: Number of parallel workers for ranking
    """

    algorithm: str = "balanced"
    threshold: float = 0.1
    use_tfidf: bool = True
    use_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    custom_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "keyword_match": 0.25,
            "path_relevance": 0.20,
            "import_graph": 0.20,
            "git_activity": 0.15,
            "file_type": 0.10,
            "complexity": 0.10,
        }
    )
    workers: int = 2


@dataclass
class TenetConfig:
    """Configuration for the tenet (guiding principles) system.

    Controls how tenets are managed and injected into context.

    Attributes:
        auto_instill: Whether to automatically apply tenets to context
        max_per_context: Maximum tenets to inject per context
        reinforcement: Whether to reinforce critical tenets
        injection_strategy: Default injection strategy
        min_distance_between: Minimum character distance between injections
        prefer_natural_breaks: Whether to inject at natural break points
        storage_path: Where to store tenet database
        collections_enabled: Whether to enable tenet collections
    """

    auto_instill: bool = True
    max_per_context: int = 5
    reinforcement: bool = True
    injection_strategy: str = "strategic"
    min_distance_between: int = 1000
    prefer_natural_breaks: bool = True
    storage_path: Optional[Path] = None
    collections_enabled: bool = True


@dataclass
class CacheConfig:
    """Configuration for caching system.

    Controls cache behavior for analysis results and other expensive operations.

    Attributes:
        enabled: Whether caching is enabled
        directory: Cache directory path
        ttl_days: Time-to-live for cache entries in days
        max_size_mb: Maximum cache size in megabytes
        compression: Whether to compress cached data
        memory_cache_size: Number of items in memory cache
        sqlite_pragmas: SQLite performance settings
    """

    enabled: bool = True
    directory: Optional[Path] = None
    ttl_days: int = 7
    max_size_mb: int = 500
    compression: bool = False
    memory_cache_size: int = 1000
    sqlite_pragmas: Dict[str, str] = field(
        default_factory=lambda: {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": "-64000",  # 64MB
            "temp_store": "MEMORY",
        }
    )


@dataclass
class OutputConfig:
    """Configuration for output formatting.

    Controls how context and analysis results are formatted.

    Attributes:
        default_format: Default output format (markdown, xml, json)
        syntax_highlighting: Whether to enable syntax highlighting
        line_numbers: Whether to include line numbers
        max_line_length: Maximum line length before wrapping
        include_metadata: Whether to include metadata in output
        compression_threshold: File size threshold for summarization
        summary_ratio: Target compression ratio for summaries
    """

    default_format: str = "markdown"
    syntax_highlighting: bool = True
    line_numbers: bool = False
    max_line_length: int = 120
    include_metadata: bool = True
    compression_threshold: int = 10_000  # Characters
    summary_ratio: float = 0.25  # Target 25% of original size


@dataclass
class GitConfig:
    """Configuration for git integration.

    Controls how git information is gathered and used.

    Attributes:
        enabled: Whether git integration is enabled
        include_history: Whether to include commit history
        history_limit: Maximum number of commits to include
        include_blame: Whether to include git blame info
        include_stats: Whether to include statistics
        ignore_authors: Authors to ignore in analysis
        main_branches: Branch names considered "main"
    """

    enabled: bool = True
    include_history: bool = True
    history_limit: int = 100
    include_blame: bool = False
    include_stats: bool = True
    ignore_authors: List[str] = field(
        default_factory=lambda: ["dependabot[bot]", "github-actions[bot]", "renovate[bot]"]
    )
    main_branches: List[str] = field(default_factory=lambda: ["main", "master", "develop", "trunk"])


@dataclass
class TenetsConfig:
    """Main configuration for the Tenets system.

    This is the root configuration object that contains all subsystem configs
    and global settings. It handles loading from files, environment variables,
    and provides sensible defaults.

    Attributes:
        config_file: Path to configuration file (if any)
        project_root: Root directory of the project
        max_tokens: Default maximum tokens for context
        version: Tenets version (for compatibility checking)
        debug: Enable debug mode
        quiet: Suppress non-essential output
        scanner: Scanner subsystem configuration
        ranking: Ranking subsystem configuration
        tenet: Tenet subsystem configuration
        cache: Cache subsystem configuration
        output: Output formatting configuration
        git: Git integration configuration
        custom: Custom user configuration
    """

    # Global settings
    config_file: Optional[Path] = None
    project_root: Optional[Path] = None
    max_tokens: int = 100_000
    version: str = "0.1.0"
    debug: bool = False
    quiet: bool = False

    # Subsystem configurations
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    tenet: TenetConfig = field(default_factory=TenetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    git: GitConfig = field(default_factory=GitConfig)

    # Custom configuration
    custom: Dict[str, Any] = field(default_factory=dict)

    # Derived attributes (not in config files)
    _logger: Any = field(default=None, init=False, repr=False)
    _resolved_paths: Dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize configuration after creation.

        This method:
        1. Sets up logging
        2. Resolves paths
        3. Loads configuration from files
        4. Applies environment variables
        5. Validates configuration
        """
        self._logger = get_logger(__name__)

        # Resolve project root
        if not self.project_root:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(self.project_root).resolve()

        # Find and load config file if not specified
        if not self.config_file:
            self.config_file = self._find_config_file()

        if self.config_file and self.config_file.exists():
            self._load_from_file(self.config_file)

        # Apply environment variables (override file config)
        self._apply_environment_variables()

        # Resolve derived paths
        self._resolve_paths()

        # Validate configuration
        self._validate()

        self._logger.debug(f"Configuration loaded from {self.config_file or 'defaults'}")

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations.

        Searches in order:
        1. .tenets.yml in current directory
        2. .tenets.yaml in current directory
        3. tenets.yml in current directory
        4. .config/tenets.yml in current directory
        5. ~/.config/tenets/config.yml
        6. ~/.tenets.yml

        Returns:
            Path to config file if found, None otherwise
        """
        search_locations = [
            self.project_root / ".tenets.yml",
            self.project_root / ".tenets.yaml",
            self.project_root / "tenets.yml",
            self.project_root / ".config" / "tenets.yml",
            Path.home() / ".config" / "tenets" / "config.yml",
            Path.home() / ".tenets.yml",
        ]

        for location in search_locations:
            if location.exists():
                self._logger.debug(f"Found config file: {location}")
                return location

        return None

    def _load_from_file(self, path: Path):
        """Load configuration from YAML or JSON file.

        Args:
            path: Path to configuration file

        Raises:
            ValueError: If file format is unsupported
            yaml.YAMLError: If YAML parsing fails
        """
        self._logger.info(f"Loading configuration from {path}")

        try:
            with open(path, "r") as f:
                if path.suffix in [".yml", ".yaml"]:
                    data = yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")

            # Apply loaded configuration
            self._apply_dict_config(data)

        except Exception as e:
            self._logger.error(f"Failed to load config from {path}: {e}")
            raise

    def _apply_dict_config(self, data: Dict[str, Any], prefix: str = ""):
        """Recursively apply dictionary configuration.

        Args:
            data: Configuration dictionary
            prefix: Prefix for nested configuration
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Handle subsystem configs
            if key == "scanner" and isinstance(value, dict):
                self.scanner = ScannerConfig(**value)
            elif key == "ranking" and isinstance(value, dict):
                self.ranking = RankingConfig(**value)
            elif key == "tenet" and isinstance(value, dict):
                self.tenet = TenetConfig(**value)
            elif key == "cache" and isinstance(value, dict):
                self.cache = CacheConfig(**value)
            elif key == "output" and isinstance(value, dict):
                self.output = OutputConfig(**value)
            elif key == "git" and isinstance(value, dict):
                self.git = GitConfig(**value)
            elif key == "custom" and isinstance(value, dict):
                self.custom = value
            elif hasattr(self, key):
                # Direct attribute assignment
                setattr(self, key, value)
            else:
                # Store unknown config in custom
                self.custom[full_key] = value

    def _apply_environment_variables(self):
        """Apply environment variable overrides.

        Environment variables follow the pattern:
        TENETS_<SECTION>_<KEY> = value

        Examples:
            TENETS_MAX_TOKENS=150000
            TENETS_SCANNER_MAX_FILE_SIZE=10000000
            TENETS_RANKING_ALGORITHM=thorough
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith("TENETS_"):
                continue

            # Parse environment variable
            parts = env_key[7:].lower().split("_")  # Remove TENETS_ prefix

            if len(parts) == 1:
                # Top-level config
                attr = parts[0]
                if hasattr(self, attr):
                    setattr(self, attr, self._parse_env_value(env_value))
            elif len(parts) >= 2:
                # Subsystem config
                subsystem = parts[0]
                attr = "_".join(parts[1:])

                if hasattr(self, subsystem):
                    subsystem_config = getattr(self, subsystem)
                    if hasattr(subsystem_config, attr):
                        setattr(subsystem_config, attr, self._parse_env_value(env_value))

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value with appropriate type
        """
        # Boolean
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # List (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # String
        return value

    def _resolve_paths(self):
        """Resolve all path configurations.

        Ensures all paths are absolute and creates directories as needed.
        """
        # Cache directory
        if not self.cache.directory:
            self.cache.directory = Path.home() / ".tenets" / "cache"
        else:
            self.cache.directory = Path(self.cache.directory).resolve()

        # Ensure cache directory exists
        self.cache.directory.mkdir(parents=True, exist_ok=True)

        # Tenet storage path
        if not self.tenet.storage_path:
            self.tenet.storage_path = Path.home() / ".tenets" / "tenets"
        else:
            self.tenet.storage_path = Path(self.tenet.storage_path).resolve()

        # Ensure tenet storage exists
        self.tenet.storage_path.mkdir(parents=True, exist_ok=True)

        self._resolved_paths = {
            "cache": self.cache.directory,
            "tenets": self.tenet.storage_path,
            "project": self.project_root,
        }

    def _validate(self):
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate max_tokens
        if self.max_tokens < 1000:
            raise ValueError(f"max_tokens must be at least 1000, got {self.max_tokens}")
        if self.max_tokens > 2_000_000:
            raise ValueError(f"max_tokens cannot exceed 2,000,000, got {self.max_tokens}")

        # Validate ranking algorithm
        valid_algorithms = ["fast", "balanced", "thorough", "ml", "custom"]
        if self.ranking.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid ranking algorithm: {self.ranking.algorithm}")

        # Validate ranking threshold
        if not 0 <= self.ranking.threshold <= 1:
            raise ValueError(
                f"Ranking threshold must be between 0 and 1, got {self.ranking.threshold}"
            )

        # Validate cache TTL
        if self.cache.ttl_days < 0:
            raise ValueError(f"Cache TTL cannot be negative, got {self.cache.ttl_days}")

        # Validate output format
        valid_formats = ["markdown", "xml", "json"]
        if self.output.default_format not in valid_formats:
            raise ValueError(f"Invalid output format: {self.output.default_format}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "max_tokens": self.max_tokens,
            "version": self.version,
            "debug": self.debug,
            "quiet": self.quiet,
            "scanner": asdict(self.scanner),
            "ranking": asdict(self.ranking),
            "tenet": asdict(self.tenet),
            "cache": asdict(self.cache),
            "output": asdict(self.output),
            "git": asdict(self.git),
            "custom": self.custom,
        }

    def save(self, path: Optional[Path] = None):
        """Save configuration to file.

        Args:
            path: Path to save to (uses config_file if not specified)

        Raises:
            ValueError: If no path specified and config_file not set
        """
        save_path = path or self.config_file
        if not save_path:
            raise ValueError("No path specified for saving configuration")

        save_path = Path(save_path)
        config_dict = self.to_dict()

        # Remove version from saved config (managed by package)
        config_dict.pop("version", None)

        with open(save_path, "w") as f:
            if save_path.suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self._logger.info(f"Configuration saved to {save_path}")

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path.

        Returns:
            Resolved cache directory path
        """
        return self._resolved_paths.get("cache", self.cache.directory)

    @cache_dir.setter
    def cache_dir(self, value: Union[str, Path]) -> None:
        # Update underlying cache config and resolved path
        path = Path(value).resolve()
        self.cache.directory = path
        path.mkdir(parents=True, exist_ok=True)
        self._resolved_paths["cache"] = path

    @property
    def scanner_workers(self) -> int:
        """Get number of scanner workers.

        Returns:
            Number of parallel workers for scanning
        """
        return self.scanner.workers

    @property
    def ranking_workers(self) -> int:
        """Get number of ranking workers.

        Returns:
            Number of parallel workers for ranking
        """
        return self.ranking.workers

    @property
    def ranking_algorithm(self) -> str:
        """Get the ranking algorithm.

        Returns:
            Name of ranking algorithm to use
        """
        return self.ranking.algorithm

    @property
    def respect_gitignore(self) -> bool:
        """Whether to respect .gitignore files.

        Returns:
            True if .gitignore should be respected
        """
        return self.scanner.respect_gitignore

    @respect_gitignore.setter
    def respect_gitignore(self, value: bool) -> None:
        self.scanner.respect_gitignore = bool(value)

    @property
    def follow_symlinks(self) -> bool:
        """Whether to follow symbolic links.

        Returns:
            True if symlinks should be followed
        """
        return self.scanner.follow_symlinks

    @follow_symlinks.setter
    def follow_symlinks(self, value: bool) -> None:
        self.scanner.follow_symlinks = bool(value)

    @property
    def additional_ignore_patterns(self) -> List[str]:
        """Get additional ignore patterns.

        Returns:
            List of patterns to ignore
        """
        return self.scanner.additional_ignore_patterns

    @property
    def auto_instill_tenets(self) -> bool:
        """Whether to automatically instill tenets.

        Returns:
            True if tenets should be auto-instilled
        """
        return self.tenet.auto_instill

    @property
    def max_tenets_per_context(self) -> int:
        """Maximum tenets to inject per context.

        Returns:
            Maximum number of tenets
        """
        return self.tenet.max_per_context

    @property
    def tenet_injection_config(self) -> Dict[str, Any]:
        """Get tenet injection configuration.

        Returns:
            Dictionary of injection settings
        """
        return {
            "strategy": self.tenet.injection_strategy,
            "min_distance_between": self.tenet.min_distance_between,
            "prefer_natural_breaks": self.tenet.prefer_natural_breaks,
            "reinforce_at_end": self.tenet.reinforcement,
        }

    @property
    def cache_ttl_days(self) -> int:
        """Cache time-to-live in days.

        Returns:
            Number of days before cache expires
        """
        return self.cache.ttl_days

    @property
    def max_cache_size_mb(self) -> int:
        """Maximum cache size in megabytes.

        Returns:
            Maximum cache size
        """
        return self.cache.max_size_mb
