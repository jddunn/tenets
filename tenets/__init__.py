"""Tenets - Context that feeds your prompts.

Tenets is a code intelligence platform that analyzes codebases locally to surface
relevant files, track development velocity, and build optimal context for both
human understanding and AI pair programming - all without making any LLM API calls.

This package provides:

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

from __future__ import annotations

__version__ = "0.7.3"
__author__ = "Johnny Dunn"
__license__ = "MIT"

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# Check Python version
if sys.version_info < (3, 9):
    raise RuntimeError("Tenets requires Python 3.9 or higher")

# Keep runtime imports lightweight. Only import heavy modules lazily.
from tenets.config import TenetsConfig
from tenets.models.context import ContextResult  # re-export for public API/tests
from tenets.models.tenet import Priority, Tenet, TenetCategory  # re-export for public API/tests
from tenets.utils.logger import get_logger

# Lazy imports using standard Python 3.7+ __getattr__ (PEP 562)
# This allows tests to patch at package level and improves import performance
_LAZY_IMPORTS = {
    "Distiller": "tenets.core.distiller.Distiller",
    "Instiller": "tenets.core.instiller.Instiller",
    "CodeAnalyzer": "tenets.core.analysis.analyzer.CodeAnalyzer",
    "TenetManager": "tenets.core.instiller.manager.TenetManager",
    "ContextResult": "tenets.models.context.ContextResult",
    "Priority": "tenets.models.tenet.Priority",
    "Tenet": "tenets.models.tenet.Tenet",
    "TenetCategory": "tenets.models.tenet.TenetCategory",
}


def __getattr__(name):
    """Lazy import heavy components on first access.

    This is the standard Python 3.7+ way to implement lazy imports (PEP 562).
    It preserves type hints, works with IDEs, and maintains proper class identity.
    """
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr_name = _LAZY_IMPORTS[name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        # Cache for future access
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Type-checking only imports (no runtime side-effects)
if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from tenets.core.analysis.analyzer import CodeAnalyzer
    from tenets.core.distiller import Distiller
    from tenets.core.instiller import Instiller
    from tenets.core.instiller.manager import TenetManager
    from tenets.models.context import ContextResult
    from tenets.models.tenet import Priority, Tenet, TenetCategory


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
        >>> from pathlib import Path
        >>>
        >>> # Initialize with default config
        >>> ten = Tenets()
        >>>
        >>> # Or with custom config
        >>> from tenets.config import TenetsConfig
        >>> config = TenetsConfig(max_tokens=150000, ranking_algorithm="thorough")
        >>> ten = Tenets(config=config)
        >>>
        >>> # Extract context (uses default session automatically)
        >>> result = ten.distill("implement user authentication")
        >>> print(f"Generated {result.token_count} tokens of context")
        >>>
        >>> # Generate HTML report
        >>> result = ten.distill("review API endpoints", format="html")
        >>> Path("api-review.html").write_text(result.context)
        >>>
        >>> # Add and apply tenets
        >>> ten.add_tenet("Use dependency injection", priority="high")
        >>> ten.add_tenet("Follow RESTful conventions", category="architecture")
        >>> ten.instill_tenets()
        >>>
        >>> # Pin critical files for priority inclusion
        >>> ten.pin_file("src/core/auth.py")
        >>> ten.pin_folder("src/api/endpoints")
        >>>
        >>> # Work with named sessions
        >>> result = ten.distill(
        ...     "implement OAuth2",
        ...     session_name="oauth-feature",
        ...     mode="thorough"
        ... )
    """

    def __init__(self, config: Optional[Union[TenetsConfig, dict[str, Any], Path]] = None):
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
            # Map common top-level aliases into nested config structure
            cfg = TenetsConfig()
            # Known top-level shortcuts used in docs/tests
            if "max_tokens" in config:
                cfg.max_tokens = int(config["max_tokens"])  # type: ignore[arg-type]
            if "debug" in config:
                cfg.debug = bool(config["debug"])  # type: ignore[arg-type]
            if "ranking_algorithm" in config:
                cfg.ranking.algorithm = str(config["ranking_algorithm"])  # type: ignore[arg-type]
            # Apply any nested sections if provided
            if "scanner" in config and isinstance(config["scanner"], dict):
                cfg.scanner = type(cfg.scanner)(**config["scanner"])  # type: ignore[call-arg]
            if "ranking" in config and isinstance(config["ranking"], dict):
                cfg.ranking = type(cfg.ranking)(**config["ranking"])  # type: ignore[call-arg]
            if "tenet" in config and isinstance(config["tenet"], dict):
                cfg.tenet = type(cfg.tenet)(**config["tenet"])  # type: ignore[call-arg]
            if "cache" in config and isinstance(config["cache"], dict):
                cfg.cache = type(cfg.cache)(**config["cache"])  # type: ignore[call-arg]
            if "output" in config and isinstance(config["output"], dict):
                cfg.output = type(cfg.output)(**config["output"])  # type: ignore[call-arg]
            if "git" in config and isinstance(config["git"], dict):
                cfg.git = type(cfg.git)(**config["git"])  # type: ignore[call-arg]
            # Any other keys go to custom
            for k, v in config.items():
                if k not in {
                    "max_tokens",
                    "debug",
                    "ranking_algorithm",
                    "scanner",
                    "ranking",
                    "tenet",
                    "cache",
                    "output",
                    "git",
                }:
                    cfg.custom[k] = v
            self.config = cfg
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            self.config = TenetsConfig(config_file=config_path)
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # Initialize logger (import locally to avoid circular import)
        from tenets.utils.logger import get_logger

        self.logger = get_logger(__name__)
        self.logger.info(f"Initializing Tenets v{__version__}")

        # Lazy-load core components to improve import performance
        self._distiller = None
        self._instiller = None
        self._tenet_manager = None

        # Session management
        self._session = None
        self._session_data = {}

        # Internal cache
        self._cache = {}

        self.logger.info("Tenets initialization complete")

    @property
    def distiller(self):
        """Lazy load distiller when needed."""
        if self._distiller is None:
            # Import locally to trigger lazy loading
            from tenets.core.distiller import Distiller

            self._distiller = Distiller(self.config)
        return self._distiller

    @property
    def instiller(self):
        """Lazy load instiller when needed."""
        if self._instiller is None:
            # Import locally to trigger lazy loading
            from tenets.core.instiller import Instiller

            self._instiller = Instiller(self.config)
        return self._instiller

    @property
    def tenet_manager(self):
        """Lazy load tenet manager when needed."""
        if self._tenet_manager is None:
            if self._instiller is None:
                # Import locally to trigger lazy loading
                from tenets.core.instiller import Instiller

                self._instiller = Instiller(self.config)
            self._tenet_manager = self._instiller.manager
        return self._tenet_manager

    # ============= Core Distillation Methods =============

    def distill(
        self,
        prompt: str,
        files: Optional[Union[str, Path, list[Path]]] = None,
        *,  # Force keyword-only arguments
        format: str = "markdown",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: str = "balanced",
        include_git: bool = True,
        session_name: Optional[str] = None,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        apply_tenets: Optional[bool] = None,
        full: bool = False,
        condense: bool = False,
        remove_comments: bool = False,
        include_tests: Optional[bool] = None,
        docstring_weight: Optional[float] = None,
        summarize_imports: bool = True,
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
            format: Output format - 'markdown', 'xml' (Claude), 'json', or 'html' (interactive report)
            model: Target LLM model for token counting (e.g., 'gpt-4o', 'claude-3-opus')
            max_tokens: Maximum tokens for context (overrides model default)
            mode: Analysis mode - 'fast', 'balanced', or 'thorough'
            include_git: Whether to include git context (commits, contributors, etc.)
            session_name: Session name for stateful context building
            include_patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: File patterns to exclude (e.g., ['test_*', '*.backup'])
            apply_tenets: Whether to apply tenets (None = use config default)

        Returns:
            ContextResult containing the generated context, metadata, and statistics.
            The metadata field includes timing information when available:
                metadata['timing'] = {
                    'duration': 2.34,  # seconds
                    'formatted_duration': '2.34s',  # Human-readable duration string
                    'start_datetime': '2024-01-15T10:30:45',
                    'end_datetime': '2024-01-15T10:30:47'
                }

        Raises:
            ValueError: If prompt is empty or invalid
            FileNotFoundError: If specified files don't exist

        Example:
            >>> # Basic usage (uses default session automatically)
            >>> result = tenets.distill("implement OAuth2 authentication")
            >>> print(result.context[:100])  # First 100 chars of context
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
            >>> # Generate HTML report
            >>> result = tenets.distill(
            ...     "analyze authentication flow",
            ...     format="html"
            ... )
            >>> Path("report.html").write_text(result.context)
            >>>
            >>> # With session management
            >>> result = tenets.distill(
            ...     "implement validation",
            ...     session_name="validation-feature"
            ... )
            >>>
            >>> # From GitHub issue
            >>> result = tenets.distill("https://github.com/org/repo/issues/123")
            >>>
            >>> # Access timing information
            >>> result = tenets.distill("analyze performance")
            >>> if 'timing' in result.metadata:
            ...     print(f"Analysis took {result.metadata['timing']['formatted_duration']}")
            ...     # Output: "Analysis took 2.34s"
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        self.logger.info(f"Distilling context for: {prompt[:100]}...")

        # Use session if specified or default session
        session = session_name or self._session

        # Run distillation
        pinned_files = []
        try:
            pf_map = self.config.custom.get("pinned_files", {})
            if session and pf_map and session in pf_map:
                pinned_files = [Path(p) for p in pf_map[session] if Path(p).exists()]
            # Supplement from session DB metadata
            if session and not pinned_files:
                try:
                    from tenets.storage.session_db import SessionDB

                    sdb = SessionDB(self.config)
                    rec = sdb.get_session(session)
                    if rec and rec.metadata.get("pinned_files"):
                        pinned_files = [
                            Path(p)
                            for p in rec.metadata.get("pinned_files", [])
                            if Path(p).exists()
                        ]
                except Exception:  # pragma: no cover
                    pass
        except Exception:  # pragma: no cover
            pinned_files = []
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
            full=full,
            condense=condense,
            remove_comments=remove_comments,
            pinned_files=pinned_files or None,
            include_tests=include_tests,
            docstring_weight=docstring_weight,
            summarize_imports=summarize_imports,
        )

        # Inject system instruction if configured (skip for HTML reports meant for humans)
        if format.lower() != "html":
            try:
                modified, meta = self.instiller.inject_system_instruction(
                    result.context, format=result.format, session=session
                )
                if meta.get("system_instruction_injected"):
                    result = ContextResult(
                        files=result.files,
                        context=modified,
                        format=result.format,
                        metadata={**result.metadata, "system_instruction": meta},
                    )
            except Exception:
                # Best-effort; don't fail distill if injection fails
                pass

        # Apply tenets if configured
        should_apply_tenets = (
            apply_tenets if apply_tenets is not None else self.config.auto_instill_tenets
        )

        pending = None
        if should_apply_tenets:
            try:
                pending = self.tenet_manager.get_pending_tenets(session)
            except Exception:
                pending = []

        def _has_real_pending(items) -> bool:
            try:
                return isinstance(items, list) and len(items) > 0
            except Exception:
                return False

        if should_apply_tenets and _has_real_pending(pending):
            self.logger.info("Applying tenets to context")
            result = self.instiller.instill(
                context=result,
                session=session,
                max_tenets=self.config.max_tenets_per_context,
                inject_system_instruction=False,  # Already injected above
            )

        # Cache result
        cache_key = f"{prompt[:50]}_{session or 'global'}"
        self._cache[cache_key] = result

        return result

    def rank_files(
        self,
        prompt: str,
        paths: Optional[Union[str, Path, List[Path]]] = None,
        *,  # Force keyword-only arguments
        mode: str = "balanced",
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_tests: Optional[bool] = None,
        exclude_tests: bool = False,
        explain: bool = False,
    ) -> RankResult:
        """Rank files by relevance without generating full context.

        This method uses the same sophisticated ranking pipeline as distill()
        but returns only the ranked files without aggregating content.
        Perfect for understanding which files are relevant or for automation.

        Args:
            prompt: Your query or task description
            paths: Paths to analyze (default: current directory)
            mode: Analysis mode - 'fast', 'balanced', or 'thorough'
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            include_tests: Whether to include test files
            exclude_tests: Whether to exclude test files
            explain: Whether to include ranking factor explanations

        Returns:
            RankResult containing the ranked files and metadata

        Example:
            >>> result = ten.rank_files("fix summarizing truncation bug")
            >>> for file in result.files:
            ...     print(f"{file.path}: {file.relevance_score:.3f}")
        """
        # Use the same pipeline as distill but stop at ranking

        # 1. Parse and understand the prompt
        prompt_context = self.distiller._parse_prompt(prompt)

        # Override test inclusion if explicitly specified
        if include_tests is not None:
            prompt_context.include_tests = include_tests
        elif exclude_tests:
            prompt_context.include_tests = False

        # 2. Determine paths to analyze
        paths = self.distiller._normalize_paths(paths)

        # 3. Discover relevant files
        files = self.distiller._discover_files(
            paths=paths,
            prompt_context=prompt_context,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        # 4. Analyze files for structure and content
        analyzed_files = self.distiller._analyze_files(
            files=files, mode=mode, prompt_context=prompt_context
        )

        # 5. Rank files by relevance (this is what we want!)
        ranked_files = self.distiller._rank_files(
            files=analyzed_files, prompt_context=prompt_context, mode=mode
        )

        # Create result object
        from collections import namedtuple

        RankResult = namedtuple("RankResult", ["files", "prompt_context", "mode", "total_scanned"])

        return RankResult(
            files=ranked_files, prompt_context=prompt_context, mode=mode, total_scanned=len(files)
        )

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

    # ============= Session Pinning Utilities =============

    def _ensure_session(self, session: Optional[str]) -> str:
        """Ensure a session exists and return its name."""
        name = session or self._session or "default"
        if not self._session:
            self._session = name
        # Create in session manager (if available)
        try:
            from tenets.core.session.session import SessionManager  # type: ignore

            # Lazy create a manager if not present (some tests may not use it directly)
        except Exception:  # pragma: no cover - defensive
            return name
        return name

    def add_file_to_session(
        self, file_path: Union[str, Path], session: Optional[str] = None
    ) -> bool:
        """Pin a single file into a session so it is prioritized in future distill calls.

        Args:
            file_path: Path to file
            session: Optional session name
        Returns:
            True if file pinned, False otherwise
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return False
        sess_name = session or self._session or "default"
        # Attach to in-memory session context held by session manager if available
        try:
            from tenets.core.session.session import SessionManager  # local import

            # There may or may not be a global session manager; instantiate lightweight if needed
        except Exception:  # pragma: no cover
            pass
        # For now store pinned files in config.custom for persistence stub
        if "pinned_files" not in self.config.custom:
            self.config.custom["pinned_files"] = {}
        self.config.custom["pinned_files"].setdefault(sess_name, set())
        resolved = str(path.resolve())
        self.config.custom["pinned_files"][sess_name].add(resolved)
        # Persist in session DB metadata if available
        try:
            from tenets.storage.session_db import SessionDB  # local import

            sdb = SessionDB(self.config)
            # Read current metadata and merge
            rec = sdb.get_session(sess_name)
            existing = rec.metadata.get("pinned_files") if rec else []
            if isinstance(existing, list):
                if resolved not in existing:
                    existing.append(resolved)
            else:
                existing = [resolved]
            sdb.update_session_metadata(sess_name, {"pinned_files": existing})
        except Exception:  # pragma: no cover - best effort
            pass
        return True

    def add_folder_to_session(
        self,
        folder_path: Union[str, Path],
        session: Optional[str] = None,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        respect_gitignore: bool = True,
        recursive: bool = True,
    ) -> int:
        """Pin all files in a folder (optionally filtered) into a session.

        Args:
            folder_path: Directory to scan
            session: Session name
            include_patterns: Include filter
            exclude_patterns: Exclude filter
            respect_gitignore: Respect .gitignore
            recursive: Recurse into subdirectories
        Returns:
            Count of files pinned.
        """
        root = Path(folder_path)
        if not root.exists() or not root.is_dir():
            return 0
        from tenets.utils.scanner import FileScanner

        scanner = FileScanner(self.config)
        paths = [root]
        files = scanner.scan(
            paths,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            follow_symlinks=False,
            respect_gitignore=respect_gitignore,
        )
        count = 0
        for f in files:
            if self.add_file_to_session(f, session=session):
                count += 1
        return count

    def list_tenets(
        self,
        pending_only: bool = False,
        instilled_only: bool = False,
        session: Optional[str] = None,
        category: Optional[Union[str, TenetCategory]] = None,
    ) -> list[dict[str, Any]]:
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
        from tenets.models.llm import estimate_cost as _estimate_cost
        from tenets.models.llm import get_model_limits

        input_tokens = result.token_count
        # Use a conservative default for expected output if not specified elsewhere
        default_output = get_model_limits(model).max_output
        return _estimate_cost(input_tokens=input_tokens, output_tokens=default_output, model=model)

    # ============= System Instruction Management =============

    def set_system_instruction(
        self,
        instruction: str,
        enable: bool = True,
        position: str = "top",
        format: str = "markdown",
        save: bool = False,
    ) -> None:
        """Set the system instruction for AI interactions.

        Args:
            instruction: The system instruction text
            enable: Whether to auto-inject
            position: Where to inject ('top', 'after_header', 'before_content')
            format: Format type ('markdown', 'xml', 'comment', 'plain')
            save: Whether to save to config file
        """
        self.config.tenet.system_instruction = instruction
        self.config.tenet.system_instruction_enabled = enable
        self.config.tenet.system_instruction_position = position
        self.config.tenet.system_instruction_format = format

        if save and getattr(self.config, "config_file", None):
            self.config.save()

        self.logger.info(f"System instruction set ({len(instruction)} chars)")

    def get_system_instruction(self) -> Optional[str]:
        """Get the current system instruction.

        Returns:
            The system instruction text or None
        """
        return self.config.tenet.system_instruction

    def clear_system_instruction(self, save: bool = False) -> None:
        """Clear the system instruction.

        Args:
            save: Whether to save to config file
        """
        self.config.tenet.system_instruction = None
        self.config.tenet.system_instruction_enabled = False

        if save and getattr(self.config, "config_file", None):
            self.config.save()

        self.logger.info("System instruction cleared")


# Convenience exports
__all__ = [
    "CodeAnalyzer",
    "ContextResult",
    "Distiller",
    "Instiller",
    "Priority",
    "Tenet",
    "TenetCategory",
    "TenetManager",
    "Tenets",
    "TenetsConfig",
    "__version__",
    "get_logger",
]
