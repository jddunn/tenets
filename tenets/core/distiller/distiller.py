"""Main distiller orchestration.

The Distiller coordinates the entire context extraction process, from
understanding the prompt to delivering optimized context.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tenets.config import TenetsConfig
from tenets.core.analysis import CodeAnalyzer
from tenets.core.distiller.aggregator import ContextAggregator
from tenets.core.distiller.formatter import ContextFormatter
from tenets.core.distiller.optimizer import TokenOptimizer
from tenets.core.git import GitAnalyzer
from tenets.core.prompt import PromptParser
from tenets.core.ranking import RelevanceRanker
from tenets.models.analysis import FileAnalysis
from tenets.models.context import ContextResult, PromptContext
from tenets.utils.logger import get_logger
from tenets.utils.scanner import FileScanner


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

        # Log multiprocessing configuration
        import os

        from tenets.utils.multiprocessing import get_ranking_workers, get_scanner_workers

        cpu_count = os.cpu_count() or 1
        scanner_workers = get_scanner_workers(config)
        ranking_workers = get_ranking_workers(config)
        
        # Only mention ML status if not in fast mode
        ml_info = ""
        if config.ranking.algorithm != "fast":
            ml_info = f", ML enabled: {config.ranking.use_ml}"
        
        self.logger.info(
            f"Distiller initialized (CPU cores: {cpu_count}, "
            f"scanner workers: {scanner_workers}, "
            f"ranking workers: {ranking_workers}{ml_info})"
        )

        # Initialize components (lazy load heavy ones)
        self.scanner = FileScanner(config)
        self.parser = PromptParser(config)
        self.formatter = ContextFormatter(config)
        
        # Lazy-load heavy components to improve startup time
        self._analyzer = None
        self._ranker = None
        self._git = None
        self._aggregator = None
        self._optimizer = None
        
    @property
    def analyzer(self):
        """Lazy load analyzer when needed."""
        if self._analyzer is None:
            self._analyzer = CodeAnalyzer(self.config)
        return self._analyzer
    
    @property
    def ranker(self):
        """Lazy load ranker when needed."""
        if self._ranker is None:
            self._ranker = RelevanceRanker(self.config)
        return self._ranker
    
    @property
    def git(self):
        """Lazy load git analyzer when needed."""
        if self._git is None:
            self._git = GitAnalyzer(self.config)
        return self._git
    
    @property
    def aggregator(self):
        """Lazy load aggregator when needed."""
        if self._aggregator is None:
            self._aggregator = ContextAggregator(self.config)
        return self._aggregator
    
    @property
    def optimizer(self):
        """Lazy load optimizer when needed."""
        if self._optimizer is None:
            self._optimizer = TokenOptimizer(self.config)
        return self._optimizer

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
        full: bool = False,
        condense: bool = False,
        remove_comments: bool = False,
        pinned_files: Optional[List[Path]] = None,
        include_tests: Optional[bool] = None,
        docstring_weight: Optional[float] = None,
        summarize_imports: bool = True,
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
        import time

        start_time = time.time()
        self.logger.info(f"Distilling context for: {prompt[:100]}...")

        # 1. Parse and understand the prompt
        parse_start = time.time()
        prompt_context = self._parse_prompt(prompt)
        self.logger.debug(f"Prompt parsing took {time.time() - parse_start:.2f}s")

        # Override test inclusion if explicitly specified
        if include_tests is not None:
            prompt_context.include_tests = include_tests
            self.logger.debug(f"Override: test inclusion set to {include_tests}")

        # 2. Determine paths to analyze
        paths = self._normalize_paths(paths)

        # 3. Discover relevant files
        discover_start = time.time()
        files = self._discover_files(
            paths=paths,
            prompt_context=prompt_context,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        self.logger.debug(f"File discovery took {time.time() - discover_start:.2f}s")

        # 4. Analyze files for structure and content
        # Prepend pinned files (avoid duplicates) while preserving original discovery order
        if pinned_files:
            # Preserve the explicit order given by the caller (tests rely on this)
            # Do NOT filter by existence – tests pass synthetic Paths.
            pinned_strs = [str(p) for p in pinned_files]
            pinned_set = set(pinned_strs)
            ordered: List[Path] = []
            # First, add pinned files (re-using the discovered Path object if present
            # so downstream identity / patch assertions still work).
            discovered_map = {str(f): f for f in files}
            for p_str, p_obj in zip(pinned_strs, pinned_files):
                if p_str in discovered_map:
                    f = discovered_map[p_str]
                else:
                    f = p_obj  # fallback to provided Path
                if f not in ordered:
                    ordered.append(f)
            # Then append remaining discovered files preserving original discovery order.
            for f in files:
                if str(f) not in pinned_set and f not in ordered:
                    ordered.append(f)
            files = ordered

        # Two-phase analysis for optimal performance
        if mode == "fast" and self.config.ranking.use_lightweight_analysis:
            # Phase 1: Lightweight analysis for all files
            analyzed_files = self._analyze_files(files=files, mode=mode, prompt_context=prompt_context)
            
            # Phase 2: Rank with lightweight data
            rank_start = time.time()
            ranked_files = self._rank_files(
                files=analyzed_files, prompt_context=prompt_context, mode=mode
            )
            self.logger.debug(f"File ranking took {time.time() - rank_start:.2f}s")
            
            # Phase 3: Optional deep analysis for top files
            # This can be enabled for better accuracy with minimal performance impact
            if self.config.ranking.deep_analysis_top_n > 0:
                ranked_files = self._deep_analyze_selected_files(
                    ranked_files, 
                    top_n=self.config.ranking.deep_analysis_top_n,
                    prompt_context=prompt_context
                )
            
        else:
            # Traditional approach for balanced/thorough modes
            # Full analysis before ranking for better accuracy
            analyzed_files = self._analyze_files(files=files, mode=mode, prompt_context=prompt_context)

            # Rank files by relevance
            rank_start = time.time()
            ranked_files = self._rank_files(
                files=analyzed_files, prompt_context=prompt_context, mode=mode
            )
            self.logger.debug(f"File ranking took {time.time() - rank_start:.2f}s")

        # 6. Add git context if requested
        git_context = None
        if include_git:
            git_context = self._get_git_context(
                paths=paths, prompt_context=prompt_context, files=ranked_files
            )

        # 7. Aggregate files within token budget
        aggregate_start = time.time()
        
        # Calculate elapsed time before aggregation so it's available for formatter
        elapsed_time = time.time() - start_time
        
        aggregated = self._aggregate_files(
            files=ranked_files,
            prompt_context=prompt_context,
            max_tokens=max_tokens or self.config.max_tokens,
            model=model,
            git_context=git_context,
            full=full,
            condense=condense,
            remove_comments=remove_comments,
            docstring_weight=docstring_weight,
            summarize_imports=summarize_imports,
            mode=mode,
        )
        self.logger.debug(f"File aggregation took {time.time() - aggregate_start:.2f}s")
        
        # Add timing to aggregated metadata for formatter
        if "metadata" not in aggregated:
            aggregated["metadata"] = {}
        aggregated["metadata"]["time_elapsed"] = f"{elapsed_time:.2f}s"

        # 8. Format the output
        formatted = self._format_output(
            aggregated=aggregated,
            format=format,
            prompt_context=prompt_context,
            session_name=session_name,
        )
        
        # Calculate final elapsed time after formatting
        final_elapsed = time.time() - start_time

        # 9. Build final result with debug information
        metadata = {
            "mode": mode,
            "files_analyzed": len(files),
            "files_included": len(aggregated["included_files"]),
            "model": model,
            "session": session_name,
            "prompt": prompt,
            "full_mode": full,
            "condense": condense,
            "remove_comments": remove_comments,
            # Include the aggregated data for _build_result to use
            "included_files": aggregated["included_files"],
            "total_tokens": aggregated.get("total_tokens", 0),
            # Add final timing information
            "time_elapsed": f"{final_elapsed:.2f}s",
        }

        # Add debug information for verbose mode
        # Add prompt parsing details
        metadata["prompt_context"] = {
            "task_type": prompt_context.task_type,
            "intent": prompt_context.intent,
            "keywords": prompt_context.keywords,
            "synonyms": getattr(prompt_context, "synonyms", []),
            "entities": prompt_context.entities,
        }

        # Expose NLP normalization metrics if available from parser
        try:
            if (
                isinstance(prompt_context.metadata, dict)
                and "nlp_normalization" in prompt_context.metadata
            ):
                metadata["nlp_normalization"] = prompt_context.metadata["nlp_normalization"]
        except Exception:
            pass

        # Add ranking details
        metadata["ranking_details"] = {
            "algorithm": mode,
            "threshold": self.config.ranking.threshold,
            "files_ranked": len(analyzed_files),
            "files_above_threshold": len(ranked_files),
            "top_files": [
                {
                    "path": str(f.path),
                    "score": f.relevance_score,
                    "match_details": {
                        "keywords_matched": getattr(f, "keywords_matched", []),
                        "semantic_score": getattr(f, "semantic_score", 0),
                    },
                }
                for f in ranked_files[:10]  # Top 10 files
            ],
        }

        # Add aggregation details
        metadata["aggregation_details"] = {
            "strategy": aggregated.get("strategy", "unknown"),
            "min_relevance": aggregated.get("min_relevance", 0),
            "files_considered": len(ranked_files),
            "files_rejected": len(ranked_files) - len(aggregated["included_files"]),
            "rejection_reasons": aggregated.get("rejection_reasons", {}),
        }

        return self._build_result(
            formatted=formatted,
            metadata=metadata,
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

        # Handle test file exclusion/inclusion based on configuration and prompt context
        exclude_patterns = exclude_patterns or []

        # If tests should be excluded and not explicitly included in prompt
        if self.config.scanner.exclude_tests_by_default and not prompt_context.include_tests:
            # Add test patterns to exclusion list
            test_exclusions = []
            for pattern in self.config.scanner.test_patterns:
                test_exclusions.append(pattern)

            # Add test directories to exclusion list
            for test_dir in self.config.scanner.test_directories:
                test_exclusions.append(f"**/{test_dir}/**")
                test_exclusions.append(f"{test_dir}/**")

            exclude_patterns.extend(test_exclusions)
            self.logger.debug(f"Excluding test files: added {len(test_exclusions)} test patterns")

        # Scan for files
        files = self.scanner.scan(
            paths=paths,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            follow_symlinks=self.config.follow_symlinks,
            respect_gitignore=self.config.respect_gitignore,
        )

        # Log test inclusion/exclusion status
        if prompt_context.include_tests:
            self.logger.info(f"Discovered {len(files)} files (including tests)")
        elif self.config.scanner.exclude_tests_by_default:
            self.logger.info(f"Discovered {len(files)} files (excluding tests)")
        else:
            self.logger.info(f"Discovered {len(files)} files")

        return files

    def _analyze_files(
        self, files: List[Path], mode: str, prompt_context: PromptContext
    ) -> List[FileAnalysis]:
        """Analyze files for content and structure.
        
        This method implements a two-phase analysis strategy:
        - Fast mode: Lightweight analysis only
        - Other modes: Full deep analysis
        
        This dramatically improves performance for fast mode by avoiding
        expensive AST parsing and language-specific analysis.
        
        Args:
            files: List of file paths to analyze
            mode: Analysis mode ('fast', 'balanced', or 'thorough')
            prompt_context: Context from prompt parsing
            
        Returns:
            List of FileAnalysis objects
        """
        # Fast mode: Use lightweight analysis for speed
        if mode == "fast":
            return self._lightweight_analyze_files(files)
        
        # Other modes: Full deep analysis
        deep_analysis = mode in ["balanced", "thorough"]

        analyzed = []
        for file in files:
            try:
                # Use positional path argument for compatibility with tests that
                # patch analyze_file expecting a first positional parameter named 'path'.
                analysis = self.analyzer.analyze_file(
                    file, deep=deep_analysis, extract_keywords=True
                )
                analyzed.append(analysis)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file}: {e}")

        return analyzed
    
    def _lightweight_analyze_files(self, files: List[Path]) -> List[FileAnalysis]:
        """Perform lightweight analysis for fast mode.
        
        Uses the LightweightAnalyzer to quickly extract minimal information
        needed for ranking. This is 10-100x faster than full analysis.
        
        Args:
            files: List of file paths to analyze
            
        Returns:
            List of FileAnalysis objects (with minimal fields populated)
        """
        from tenets.core.analysis.lightweight import LightweightAnalyzer
        
        # Create lightweight analyzer if not exists
        if not hasattr(self, '_lightweight_analyzer'):
            self._lightweight_analyzer = LightweightAnalyzer()
        
        self.logger.info(f"Using lightweight analysis for {len(files)} files")
        
        # Analyze files with lightweight analyzer
        lightweight_results = self._lightweight_analyzer.analyze_files(files)
        
        # Convert to FileAnalysis objects for compatibility
        analyzed = []
        for result in lightweight_results:
            analyzed.append(result.to_file_analysis())
        
        self.logger.info(f"Lightweight analysis complete: {len(analyzed)} files processed")
        return analyzed
    
    def _deep_analyze_selected_files(
        self, 
        ranked_files: List[FileAnalysis], 
        top_n: int = 20,
        prompt_context: Optional[PromptContext] = None
    ) -> List[FileAnalysis]:
        """Perform deep analysis only on top-ranked files.
        
        This method enables a hybrid approach where we can:
        1. Quickly rank all files with lightweight analysis
        2. Then deeply analyze only the most relevant files
        
        This gives us the best of both worlds: fast initial processing
        with accurate analysis of important files.
        
        Args:
            ranked_files: Files already ranked (with lightweight analysis)
            top_n: Number of top files to deeply analyze
            prompt_context: Optional context for guided analysis
            
        Returns:
            List with top N files replaced with deep analysis versions
            
        Note:
            Files beyond top_n retain their lightweight analysis.
            This is usually fine since they won't be included in output.
        """
        if not ranked_files:
            return ranked_files
        
        # Separate top files for deep analysis
        files_to_analyze = ranked_files[:top_n]
        remaining_files = ranked_files[top_n:]
        
        self.logger.info(f"Performing deep analysis on top {len(files_to_analyze)} files")
        
        deeply_analyzed = []
        for file_analysis in files_to_analyze:
            try:
                # Get the original path
                file_path = Path(file_analysis.path)
                
                # Perform deep analysis
                deep_analysis = self.analyzer.analyze_file(
                    file_path, 
                    deep=True, 
                    extract_keywords=True
                )
                
                # Preserve the ranking score from lightweight analysis
                if hasattr(file_analysis, 'relevance_score'):
                    deep_analysis.relevance_score = file_analysis.relevance_score
                
                deeply_analyzed.append(deep_analysis)
                
            except Exception as e:
                self.logger.warning(
                    f"Deep analysis failed for {file_analysis.path}, "
                    f"keeping lightweight version: {e}"
                )
                deeply_analyzed.append(file_analysis)
        
        # Combine deep-analyzed top files with remaining lightweight files
        result = deeply_analyzed + remaining_files
        
        self.logger.info(
            f"Deep analysis complete: {len(deeply_analyzed)} files enhanced"
        )
        
        return result

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
                path=git_root,
                limit=10,
                files=[f.path for f in files[:20]],  # Top 20 files
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
        full: bool = False,
        condense: bool = False,
        remove_comments: bool = False,
        docstring_weight: Optional[float] = None,
        summarize_imports: bool = True,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate files within token budget."""
        return self.aggregator.aggregate(
            files=files,
            prompt_context=prompt_context,
            max_tokens=max_tokens,
            model=model,
            git_context=git_context,
            full=full,
            condense=condense,
            remove_comments=remove_comments,
            docstring_weight=docstring_weight,
            summarize_imports=summarize_imports,
            mode=mode,
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
        # Extract file paths from the aggregated included_files structure
        included_files = []
        for file_info in metadata.get("included_files", []):
            if isinstance(file_info, dict) and "file" in file_info:
                # file_info["file"] is a FileAnalysis object with a path attribute
                included_files.append(str(file_info["file"].path))
            elif hasattr(file_info, "path"):
                # Direct FileAnalysis object
                included_files.append(str(file_info.path))

        return ContextResult(
            context=formatted,
            format=metadata.get("format", "markdown"),
            metadata=metadata,
            files_included=included_files,
            token_count=metadata.get("total_tokens", 0),
        )
