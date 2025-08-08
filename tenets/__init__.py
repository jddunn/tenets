"""Tenets - Context that feeds your prompts.

Tenets is a code intelligence platform that analyzes codebases locally to surface
relevant files, track development velocity, and build optimal context for both
human understanding and AI pair programming - all without making any LLM API calls.

This package provides:
- Intelligent context extraction (distill)
- Guiding principles management (tenets/instill)
- Code analysis and metrics (examine)
- Development tracking (chronicle/momentum)
- Visualization capabilities (viz)

Example:
    Basic usage for context extraction:

    >>> from tenets import Tenets
    >>> ten = Tenets()
    >>> result = ten.distill("implement OAuth2 authentication")
    >>> print(result.context)

    With tenet system:

    >>> ten.add_tenet("Always use type hints in Python", priority="high")
    >>> ten.instill_tenets()
    >>> result = ten.distill("add user model")  # Context now includes tenets
"""

__version__ = "0.5.0"
__author__ = "manic.agency"
__license__ = "MIT"

import os
import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

# Check Python version
if sys.version_info < (3, 9):
    raise RuntimeError("Tenets requires Python 3.9 or higher")

# Import core components
from tenets.config import TenetsConfig
from tenets.core.distiller import Distiller
from tenets.core.instiller import Instiller
from tenets.core.instiller.manager import TenetManager
from tenets.models.context import ContextResult
from tenets.models.tenet import Tenet, Priority, TenetCategory
from tenets.utils.logger import get_logger


class Tenets:
    """Main API interface for the Tenets system.

    This is the primary class that users interact with to access all Tenets
    functionality. It coordinates between the various subsystems (distiller,
    instiller, analyzer, etc.) to provide a unified interface.

    The Tenets class can be used both programmatically through Python and via
    the CLI. It maintains configuration, manages sessions, and orchestrates
    the various analysis and context generation operations.

    Attributes:
        config: TenetsConfig instance containing all configuration
        distiller: Distiller instance for context extraction
        instiller: Instiller instance for tenet management
        tenet_manager: Direct access to TenetManager for advanced operations
        logger: Logger instance for this class
        _session: Current session name if any
        _cache: Internal cache for results

    Example:
        >>> from tenets import Tenets
        >>>
        >>> # Initialize with default config
        >>> ten = Tenets()
        >>>
        >>> # Or with custom config
        >>> from tenets.config import TenetsConfig
        >>> config = TenetsConfig(max_tokens=150000, ranking_algorithm="thorough")
        >>> ten = Tenets(config=config)
        >>>
        >>> # Extract context
        >>> result = ten.distill("implement user authentication")
        >>> print(f"Generated {result.token_count} tokens of context")
        >>>
        >>> # Add and apply tenets
        >>> ten.add_tenet("Use dependency injection", priority="high")
        >>> ten.instill_tenets()
    """

    def __init__(self, config: Optional[Union[TenetsConfig, Dict[str, Any], Path]] = None):
        """Initialize Tenets with configuration.

        Args:
            config: Can be:
                - TenetsConfig instance
                - Dictionary of configuration values
                - Path to configuration file
                - None (uses default configuration)

        Raises:
            ValueError: If config format is invalid
            FileNotFoundError: If config file path doesn't exist
        """
        # Handle different config input types
        if config is None:
            self.config = TenetsConfig()
        elif isinstance(config, TenetsConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = TenetsConfig(**config)
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            self.config = TenetsConfig(config_file=config_path)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing Tenets v{__version__}")

        # Initialize core components
        self.distiller = Distiller(self.config)
        self.instiller = Instiller(self.config)
        self.tenet_manager = self.instiller.manager

        # Session management
        self._session = None
        self._session_data = {}

        # Internal cache
        self._cache = {}

        self.logger.info("Tenets initialization complete")

    # ============= Core Distillation Methods =============

    def distill(
        self,
        prompt: str,
        files: Optional[Union[str, Path, List[Path]]] = None,
        *,  # Force keyword-only arguments
        format: str = "markdown",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: str = "balanced",
        include_git: bool = True,
        session_name: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        apply_tenets: Optional[bool] = None,
    ) -> ContextResult:
        """Distill relevant context from codebase based on prompt.

        This is the main method for extracting context. It analyzes your codebase,
        finds relevant files, ranks them by importance, and aggregates them into
        an optimized context that fits within token limits.

        Args:
            prompt: Your query or task description. Can be plain text or a URL
                   to a GitHub issue, JIRA ticket, etc.
            files: Paths to analyze. Can be a single path, list of paths, or None
                  to use current directory
            format: Output format - 'markdown', 'xml' (Claude), or 'json'
            model: Target LLM model for token counting (e.g., 'gpt-4o', 'claude-3-opus')
            max_tokens: Maximum tokens for context (overrides model default)
            mode: Analysis mode - 'fast', 'balanced', or 'thorough'
            include_git: Whether to include git context (commits, contributors, etc.)
            session_name: Session name for stateful context building
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: File patterns to exclude (e.g., ['test_*', '*.backup'])
            apply_tenets: Whether to apply tenets (None = use config default)

        Returns:
            ContextResult containing the generated context, metadata, and statistics

        Raises:
            ValueError: If prompt is empty or invalid
            FileNotFoundError: If specified files don't exist

        Example:
            >>> # Basic usage
            >>> result = tenets.distill("implement OAuth2 authentication")
            >>>
            >>> # With specific files and options
            >>> result = tenets.distill(
            ...     "add caching layer",
            ...     files="./src",
            ...     mode="thorough",
            ...     max_tokens=50000,
            ...     include_patterns=["*.py"],
            ...     exclude_patterns=["test_*.py"]
            ... )
            >>>
            >>> # From GitHub issue
            >>> result = tenets.distill("https://github.com/org/repo/issues/123")
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        self.logger.info(f"Distilling context for: {prompt[:100]}...")

        # Use session if specified or default session
        session = session_name or self._session

        # Run distillation
        result = self.distiller.distill(
            prompt=prompt,
            paths=files,
            format=format,
            model=model,
            max_tokens=max_tokens,
            mode=mode,
            include_git=include_git,
            session_name=session,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        # Apply tenets if configured
        should_apply_tenets = (
            apply_tenets if apply_tenets is not None else self.config.auto_instill_tenets
        )

        if should_apply_tenets and self.tenet_manager.get_pending_tenets(session):
            self.logger.info("Applying tenets to context")
            result = self.instiller.instill(
                context=result, session=session, max_tenets=self.config.max_tenets_per_context
            )

        # Cache result
        cache_key = f"{prompt[:50]}_{session or 'global'}"
        self._cache[cache_key] = result

        return result

    # ============= Tenet Management Methods =============

    def add_tenet(
        self,
        content: str,
        priority: Union[str, Priority] = "medium",
        category: Optional[Union[str, TenetCategory]] = None,
        session: Optional[str] = None,
        author: Optional[str] = None,
    ) -> Tenet:
        """Add a new guiding principle (tenet).

        Tenets are persistent instructions that get strategically injected into
        generated context to maintain consistency across AI interactions. They
        help combat context drift and ensure important principles are followed.

        Args:
            content: The guiding principle text
            priority: Priority level - 'low', 'medium', 'high', or 'critical'
            category: Optional category - 'architecture', 'security', 'style',
                     'performance', 'testing', 'documentation', etc.
            session: Optional session to bind this tenet to
            author: Optional author identifier

        Returns:
            The created Tenet object

        Example:
            >>> # Add a high-priority security tenet
            >>> tenet = ten.add_tenet(
            ...     "Always validate and sanitize user input",
            ...     priority="high",
            ...     category="security"
            ... )
            >>>
            >>> # Add a session-specific tenet
            >>> ten.add_tenet(
            ...     "Use async/await for all I/O operations",
            ...     session="async-refactor"
            ... )
        """
        return self.tenet_manager.add_tenet(
            content=content,
            priority=priority,
            category=category,
            session=session or self._session,
            author=author,
        )

    def instill_tenets(self, session: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Instill pending tenets.

        This marks tenets as active and ready to be injected into future contexts.
        By default, only pending tenets are instilled, but you can force
        re-instillation of all tenets.

        Args:
            session: Optional session to instill tenets for
            force: If True, re-instill even already instilled tenets

        Returns:
            Dictionary with instillation results including count and tenets

        Example:
            >>> # Instill all pending tenets
            >>> result = ten.instill_tenets()
            >>> print(f"Instilled {result['count']} tenets")
            >>>
            >>> # Force re-instillation
            >>> ten.instill_tenets(force=True)
        """
        return self.tenet_manager.instill_tenets(session=session or self._session, force=force)

    def list_tenets(
        self,
        pending_only: bool = False,
        instilled_only: bool = False,
        session: Optional[str] = None,
        category: Optional[Union[str, TenetCategory]] = None,
    ) -> List[Dict[str, Any]]:
        """List tenets with optional filtering.

        Args:
            pending_only: Only show pending (not yet instilled) tenets
            instilled_only: Only show instilled tenets
            session: Filter by session binding
            category: Filter by category

        Returns:
            List of tenet dictionaries

        Example:
            >>> # List all tenets
            >>> all_tenets = ten.list_tenets()
            >>>
            >>> # List only pending security tenets
            >>> pending_security = ten.list_tenets(
            ...     pending_only=True,
            ...     category="security"
            ... )
        """
        return self.tenet_manager.list_tenets(
            pending_only=pending_only,
            instilled_only=instilled_only,
            session=session or self._session,
            category=category,
        )

    def get_tenet(self, tenet_id: str) -> Optional[Tenet]:
        """Get a specific tenet by ID.

        Args:
            tenet_id: Tenet ID (can be partial)

        Returns:
            The Tenet object or None if not found
        """
        return self.tenet_manager.get_tenet(tenet_id)

    def remove_tenet(self, tenet_id: str) -> bool:
        """Remove (archive) a tenet.

        Args:
            tenet_id: Tenet ID (can be partial)

        Returns:
            True if removed, False if not found
        """
        return self.tenet_manager.remove_tenet(tenet_id)

    def get_pending_tenets(self, session: Optional[str] = None) -> List[Tenet]:
        """Get all pending tenets.

        Args:
            session: Optional session filter

        Returns:
            List of pending Tenet objects
        """
        return self.tenet_manager.get_pending_tenets(session or self._session)

    def export_tenets(self, format: str = "yaml", session: Optional[str] = None) -> str:
        """Export tenets to YAML or JSON.

        Args:
            format: Export format - 'yaml' or 'json'
            session: Optional session filter

        Returns:
            Serialized tenets string
        """
        return self.tenet_manager.export_tenets(format=format, session=session or self._session)

    def import_tenets(self, file_path: Union[str, Path], session: Optional[str] = None) -> int:
        """Import tenets from file.

        Args:
            file_path: Path to import file (YAML or JSON)
            session: Optional session to bind imported tenets to

        Returns:
            Number of tenets imported
        """
        return self.tenet_manager.import_tenets(
            file_path=file_path, session=session or self._session
        )

    # ============= Analysis Methods =============

    def examine(
        self,
        path: Optional[Union[str, Path]] = None,
        deep: bool = False,
        include_git: bool = True,
        output_metadata: bool = False,
    ) -> Any:  # Returns AnalysisResult
        """Examine codebase structure and metrics.

        Provides detailed analysis of your code including file counts, language
        distribution, complexity metrics, and potential issues.

        Args:
            path: Path to examine (default: current directory)
            deep: Perform deep analysis with AST parsing
            include_git: Include git statistics
            output_metadata: Include detailed metadata in result

        Returns:
            AnalysisResult object with comprehensive codebase analysis

        Example:
            >>> # Basic examination
            >>> analysis = ten.examine()
            >>> print(f"Found {analysis.total_files} files")
            >>> print(f"Languages: {', '.join(analysis.languages)}")
            >>>
            >>> # Deep analysis with git
            >>> analysis = ten.examine(deep=True, include_git=True)
        """
        # This would call the analyzer module (not shown in detail here)
        # Placeholder for now
        from tenets.core.analysis import CodeAnalyzer

        analyzer = CodeAnalyzer(self.config)

        # Would return proper AnalysisResult
        return {
            "total_files": 0,
            "languages": [],
            "message": "Examine functionality to be implemented",
        }

    def track_changes(
        self,
        path: Optional[Union[str, Path]] = None,
        since: str = "1 week",
        author: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track code changes over time.

        Args:
            path: Repository path (default: current directory)
            since: Time period (e.g., '1 week', '3 days', 'yesterday')
            author: Filter by author
            file_pattern: Filter by file pattern

        Returns:
            Dictionary with change information
        """
        # Placeholder - would integrate with git module
        return {
            "commits": [],
            "files": [],
            "message": "Track changes functionality to be implemented",
        }

    def momentum(
        self,
        path: Optional[Union[str, Path]] = None,
        since: str = "last-month",
        team: bool = False,
        author: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track development momentum and velocity.

        Args:
            path: Repository path
            since: Time period to analyze
            team: Show team-wide statistics
            author: Show stats for specific author

        Returns:
            Dictionary with momentum metrics
        """
        # Placeholder - would integrate with git analyzer
        return {"overall": {}, "weekly": [], "message": "Momentum functionality to be implemented"}

    def estimate_cost(self, result: ContextResult, model: str) -> Dict[str, Any]:
        """Estimate the cost of using generated context with an LLM.

        Args:
            result: ContextResult from distill()
            model: Target model name

        Returns:
            Dictionary with token counts and cost estimates
        """
        from tenets.models.llm import estimate_cost as _estimate_cost, get_model_limits

        input_tokens = result.token_count
        # Use a conservative default for expected output if not specified elsewhere
        default_output = get_model_limits(model).max_output
        return _estimate_cost(input_tokens=input_tokens, output_tokens=default_output, model=model)


# Convenience exports
__all__ = [
    "Tenets",
    "TenetsConfig",
    "ContextResult",
    "Tenet",
    "Priority",
    "TenetCategory",
    "__version__",
]
