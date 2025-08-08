"""Main distiller orchestration.

The Distiller coordinates the entire context extraction process, from
understanding the prompt to delivering optimized context.
"""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from tenets.config import TenetsConfig
from tenets.models.context import ContextResult, PromptContext
from tenets.models.analysis import FileAnalysis
from tenets.core.distiller.aggregator import ContextAggregator
from tenets.core.distiller.optimizer import TokenOptimizer
from tenets.core.distiller.formatter import ContextFormatter
from tenets.core.analysis import CodeAnalyzer
from tenets.core.ranking import RelevanceRanker
from tenets.core.prompt import PromptParser
from tenets.core.git import GitAnalyzer
from tenets.utils.scanner import FileScanner
from tenets.utils.logger import get_logger


class Distiller:
    """Orchestrates context extraction from codebases.

    The Distiller is the main engine that powers the 'distill' command.
    It coordinates all the components to extract the most relevant context
    based on a user's prompt.
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the distiller with configuration.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize components
        self.scanner = FileScanner(config)
        self.analyzer = CodeAnalyzer(config)
        self.ranker = RelevanceRanker(config)
        self.parser = PromptParser(config)
        self.git = GitAnalyzer(config)
        self.aggregator = ContextAggregator(config)
        self.optimizer = TokenOptimizer(config)
        self.formatter = ContextFormatter(config)

    def distill(
        self,
        prompt: str,
        paths: Optional[Union[str, Path, List[Path]]] = None,
        *,  # Force keyword-only arguments for clarity
        format: str = "markdown",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        mode: str = "balanced",
        include_git: bool = True,
        session_name: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> ContextResult:
        """Distill relevant context from codebase based on prompt.

        This is the main method that extracts, ranks, and aggregates
        the most relevant files and information for a given prompt.

        Args:
            prompt: The user's query or task description
            paths: Paths to analyze (default: current directory)
            format: Output format (markdown, xml, json)
            model: Target LLM model for token counting
            max_tokens: Maximum tokens for context
            mode: Analysis mode (fast, balanced, thorough)
            include_git: Whether to include git context
            session_name: Session name for stateful context
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude

        Returns:
            ContextResult with the distilled context

        Example:
            >>> distiller = Distiller(config)
            >>> result = distiller.distill(
            ...     "implement OAuth2 authentication",
            ...     paths="./src",
            ...     mode="thorough",
            ...     max_tokens=50000
            ... )
            >>> print(result.context)
        """
        self.logger.info(f"Distilling context for: {prompt[:100]}...")

        # 1. Parse and understand the prompt
        prompt_context = self._parse_prompt(prompt)

        # 2. Determine paths to analyze
        paths = self._normalize_paths(paths)

        # 3. Discover relevant files
        files = self._discover_files(
            paths=paths,
            prompt_context=prompt_context,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        # 4. Analyze files for structure and content
        analyzed_files = self._analyze_files(files=files, mode=mode, prompt_context=prompt_context)

        # 5. Rank files by relevance
        ranked_files = self._rank_files(
            files=analyzed_files, prompt_context=prompt_context, mode=mode
        )

        # 6. Add git context if requested
        git_context = None
        if include_git:
            git_context = self._get_git_context(
                paths=paths, prompt_context=prompt_context, files=ranked_files
            )

        # 7. Aggregate files within token budget
        aggregated = self._aggregate_files(
            files=ranked_files,
            prompt_context=prompt_context,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model,
            git_context=git_context,
        )

        # 8. Format the output
        formatted = self._format_output(
            aggregated=aggregated,
            format=format,
            prompt_context=prompt_context,
            session_name=session_name,
        )

        # 9. Build final result
        return self._build_result(
            formatted=formatted,
            metadata={
                "mode": mode,
                "files_analyzed": len(files),
                "files_included": len(aggregated["included_files"]),
                "model": model,
                "session": session_name,
                "prompt": prompt,
            },
        )

    def _parse_prompt(self, prompt: str) -> PromptContext:
        """Parse the prompt to understand intent and extract information."""
        return self.parser.parse(prompt)

    def _normalize_paths(self, paths: Optional[Union[str, Path, List[Path]]]) -> List[Path]:
        """Normalize paths to a list of Path objects."""
        if paths is None:
            return [Path.cwd()]

        if isinstance(paths, (str, Path)):
            paths = [paths]

        return [Path(p) for p in paths]

    def _discover_files(
        self,
        paths: List[Path],
        prompt_context: PromptContext,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Discover files to analyze."""
        self.logger.debug(f"Discovering files in {len(paths)} paths")

        # Use prompt context to guide discovery
        if prompt_context.file_patterns:
            # Merge with include patterns
            include_patterns = (include_patterns or []) + prompt_context.file_patterns

        # Scan for files
        files = self.scanner.scan(
            paths=paths,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            follow_symlinks=self.config.follow_symlinks,
            respect_gitignore=self.config.respect_gitignore,
        )

        self.logger.info(f"Discovered {len(files)} files")
        return files

    def _analyze_files(
        self, files: List[Path], mode: str, prompt_context: PromptContext
    ) -> List[FileAnalysis]:
        """Analyze files for content and structure."""
        # Determine analysis depth based on mode
        deep_analysis = mode in ["balanced", "thorough"]

        analyzed = []
        for file in files:
            try:
                analysis = self.analyzer.analyze_file(
                    file_path=file, deep=deep_analysis, extract_keywords=True
                )
                analyzed.append(analysis)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file}: {e}")

        return analyzed

    def _rank_files(
        self, files: List[FileAnalysis], prompt_context: PromptContext, mode: str
    ) -> List[FileAnalysis]:
        """Rank files by relevance to the prompt."""
        return self.ranker.rank_files(files=files, prompt_context=prompt_context, algorithm=mode)

    def _get_git_context(
        self, paths: List[Path], prompt_context: PromptContext, files: List[FileAnalysis]
    ) -> Optional[Dict[str, Any]]:
        """Get relevant git context."""
        # Find git root
        git_root = None
        for path in paths:
            if self.git.is_git_repo(path):
                git_root = path
                break

        if not git_root:
            return None

        # Get git information
        context = {
            "recent_commits": self.git.get_recent_commits(
                path=git_root, limit=10, files=[f.path for f in files[:20]]  # Top 20 files
            ),
            "contributors": self.git.get_contributors(
                path=git_root, files=[f.path for f in files[:20]]
            ),
            "branch": self.git.get_current_branch(git_root),
        }

        # Add temporal context if prompt suggests it
        if any(
            word in prompt_context.text.lower() for word in ["recent", "latest", "new", "changed"]
        ):
            context["recent_changes"] = self.git.get_changes_since(
                path=git_root, since="1 week ago", files=[f.path for f in files[:20]]
            )

        return context

    def _aggregate_files(
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        max_tokens: int,
        model: Optional[str],
        git_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate files within token budget."""
        return self.aggregator.aggregate(
            files=files,
            prompt_context=prompt_context,
            max_tokens=max_tokens,
            model=model,
            git_context=git_context,
        )

    def _format_output(
        self,
        aggregated: Dict[str, Any],
        format: str,
        prompt_context: PromptContext,
        session_name: Optional[str],
    ) -> str:
        """Format the aggregated context for output."""
        return self.formatter.format(
            aggregated=aggregated,
            format=format,
            prompt_context=prompt_context,
            session_name=session_name,
        )

    def _build_result(self, formatted: str, metadata: Dict[str, Any]) -> ContextResult:
        """Build the final context result."""
        return ContextResult(
            context=formatted,
            format=metadata.get("format", "markdown"),
            metadata=metadata,
            files_included=[f["path"] for f in metadata.get("included_files", [])],
        )
