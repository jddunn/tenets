"""Context aggregation - intelligently combine files within token limits.

The aggregator is responsible for selecting and combining files in a way that
maximizes relevance while staying within token constraints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext
from tenets.models.summary import FileSummary
from tenets.utils.logger import get_logger
from tenets.utils.tokens import count_tokens

# Lazy import Summarizer to avoid loading ML dependencies at import time


@dataclass
class AggregationStrategy:
    """Strategy for how to aggregate files."""

    name: str
    max_full_files: int = 10
    summarize_threshold: float = 0.7  # Files below this score get summarized
    min_relevance: float = 0.3  # Don't include files below this
    preserve_structure: bool = True  # Try to keep related files together


class ContextAggregator:
    """Aggregates files intelligently within token constraints."""

    def __init__(self, config: TenetsConfig):
        """Initialize the aggregator.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._summarizer = None  # Lazy loaded when needed

        # Define aggregation strategies
        # Note: min_relevance should be <= ranking threshold (default 0.1) to avoid filtering out ranked files
        self.strategies = {
            "greedy": AggregationStrategy(
                name="greedy", max_full_files=20, summarize_threshold=0.6, min_relevance=0.05
            ),
            "balanced": AggregationStrategy(
                name="balanced", max_full_files=10, summarize_threshold=0.7, min_relevance=0.08
            ),
            "conservative": AggregationStrategy(
                name="conservative", max_full_files=5, summarize_threshold=0.8, min_relevance=0.15
            ),
        }

    @property
    def summarizer(self):
        """Lazy load summarizer when needed."""
        if self._summarizer is None:
            from tenets.core.summarizer import Summarizer

            self._summarizer = Summarizer(self.config)
        return self._summarizer

    def aggregate(
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        max_tokens: int,
        model: Optional[str] = None,
        git_context: Optional[Dict[str, Any]] = None,
        strategy: str = "balanced",
        full: bool = False,
        condense: bool = False,
        remove_comments: bool = False,
        docstring_weight: Optional[float] = None,
        summarize_imports: bool = True,
    ) -> Dict[str, Any]:
        """Aggregate files within token budget.

        Args:
            files: Ranked files to aggregate
            prompt_context: Context about the prompt
            max_tokens: Maximum token budget
            model: Target model for token counting
            git_context: Optional git context to include
            strategy: Aggregation strategy to use

        Returns:
            Dictionary with aggregated content and metadata
        """
        self.logger.info(f"Aggregating {len(files)} files with {strategy} strategy")

        strat = self.strategies.get(strategy, self.strategies["balanced"])

        # Reserve tokens for structure and git context
        structure_tokens = 500  # Headers, formatting, etc.
        git_tokens = self._estimate_git_tokens(git_context) if git_context else 0
        available_tokens = max_tokens - structure_tokens - git_tokens

        # Select files to include
        included_files = []
        summarized_files = []
        total_tokens = 0

        # Track rejection reasons for verbose mode
        rejection_reasons = {
            "below_min_relevance": 0,
            "token_budget_exceeded": 0,
            "insufficient_tokens_for_summary": 0,
        }

        # Full mode: attempt to include full content for all files (still respecting token budget)
        for i, file in enumerate(files):
            # Skip files below minimum relevance
            if file.relevance_score < strat.min_relevance:
                self.logger.debug(
                    f"Skipping {file.path} (relevance {file.relevance_score:.2f} < {strat.min_relevance})"
                )
                rejection_reasons["below_min_relevance"] += 1
                continue

            # Estimate tokens for this file
            original_content = file.content
            transformed_stats = {}
            if remove_comments or condense:
                try:
                    from .transform import (  # local import
                        apply_transformations,
                        detect_language_from_extension,
                    )

                    lang = detect_language_from_extension(str(file.path))
                    transformed, transformed_stats = apply_transformations(
                        original_content,
                        lang,
                        remove_comments=remove_comments,
                        condense=condense,
                    )
                    if transformed_stats.get("changed"):
                        file.content = transformed
                except Exception as e:  # pragma: no cover - defensive
                    self.logger.debug(f"Transformation failed for {file.path}: {e}")
            file_tokens = count_tokens(file.content, model)

            # Decide whether to include full or summarized
            if full:
                if total_tokens + file_tokens <= available_tokens:
                    included_files.append(
                        {
                            "file": file,
                            "content": file.content,
                            "tokens": file_tokens,
                            "summarized": False,
                            "transformations": transformed_stats,
                        }
                    )
                    total_tokens += file_tokens
                else:
                    self.logger.debug(
                        f"Skipping {file.path} (token budget exceeded in full mode: {total_tokens + file_tokens} > {available_tokens})"
                    )
                    rejection_reasons["token_budget_exceeded"] += 1
                continue

            if (
                i < strat.max_full_files
                and file.relevance_score >= strat.summarize_threshold
                and total_tokens + file_tokens <= available_tokens
            ):
                # Include full file
                included_files.append(
                    {
                        "file": file,
                        "content": file.content,
                        "tokens": file_tokens,
                        "summarized": False,
                        "transformations": transformed_stats,
                    }
                )
                total_tokens += file_tokens

            elif total_tokens < available_tokens * 0.9:  # Leave some buffer
                # Try to summarize
                remaining_tokens = available_tokens - total_tokens
                summary_tokens = min(
                    file_tokens // 4,  # Aim for 25% of original
                    remaining_tokens // 2,  # Don't use more than half remaining
                )

                if summary_tokens > 100:  # Worth summarizing
                    # Calculate target ratio based on desired token reduction
                    target_ratio = min(0.5, summary_tokens / file_tokens)

                    # Apply config overrides if provided
                    if docstring_weight is not None or not summarize_imports:
                        # Temporarily override the config
                        original_weight = getattr(self.config.summarizer, "docstring_weight", 0.5)
                        original_summarize = getattr(
                            self.config.summarizer, "summarize_imports", True
                        )

                        if docstring_weight is not None:
                            self.config.summarizer.docstring_weight = docstring_weight
                        if not summarize_imports:
                            self.config.summarizer.summarize_imports = False

                        summary = self.summarizer.summarize_file(
                            file=file,
                            target_ratio=target_ratio,
                            preserve_structure=True,
                            prompt_keywords=prompt_context.keywords if prompt_context else None,
                        )

                        # Restore original values
                        self.config.summarizer.docstring_weight = original_weight
                        self.config.summarizer.summarize_imports = original_summarize
                    else:
                        summary = self.summarizer.summarize_file(
                            file=file,
                            target_ratio=target_ratio,
                            preserve_structure=True,
                            prompt_keywords=prompt_context.keywords if prompt_context else None,
                        )

                    # Get actual token count of summary
                    summary_content = (
                        summary.summary if hasattr(summary, "summary") else str(summary)
                    )
                    actual_summary_tokens = count_tokens(summary_content, model)

                    # Extract metadata from summary if available
                    metadata = {}
                    if hasattr(summary, "metadata") and summary.metadata:
                        metadata = summary.metadata

                    summarized_files.append(
                        {
                            "file": file,
                            "content": summary_content,
                            "tokens": actual_summary_tokens,
                            "summarized": True,
                            "summary": self._convert_summarization_result_to_file_summary(summary, str(file.path)),
                            "transformations": transformed_stats,
                            "metadata": metadata,
                        }
                    )
                    total_tokens += actual_summary_tokens
                else:
                    self.logger.debug(
                        f"Skipping {file.path} summary (insufficient remaining tokens: {remaining_tokens})"
                    )
                    rejection_reasons["insufficient_tokens_for_summary"] += 1
            else:
                self.logger.debug(
                    f"Skipping {file.path} (token budget exceeded: {total_tokens + file_tokens} > {available_tokens})"
                )
                rejection_reasons["token_budget_exceeded"] += 1

        # Combine full and summarized files
        all_files = included_files + summarized_files

        # Sort by relevance to maintain importance order
        all_files.sort(key=lambda x: x["file"].relevance_score, reverse=True)

        # Build result
        result = {
            "included_files": all_files,
            "total_tokens": total_tokens,
            "available_tokens": available_tokens,
            "git_context": git_context,  # include for tests/consumers
            "strategy": strategy,
            "min_relevance": strat.min_relevance,
            "rejection_reasons": rejection_reasons,
            "statistics": {
                "files_analyzed": len(files),
                "files_included": len(included_files),
                "files_summarized": len(summarized_files),
                "files_skipped": len(files) - len(all_files),
                "token_utilization": total_tokens / available_tokens if available_tokens > 0 else 0,
            },
        }

        self.logger.info(
            f"Aggregated {len(all_files)} files "
            f"({len(included_files)} full, {len(summarized_files)} summarized) "
            f"using {total_tokens:,} tokens"
        )

        return result

    def _convert_summarization_result_to_file_summary(self, result, file_path: str) -> FileSummary:
        """Convert a SummarizationResult to a FileSummary."""
        from tenets.core.summarizer.summarizer import SummarizationResult
        
        if isinstance(result, SummarizationResult):
            return FileSummary(
                content=result.summary,
                was_summarized=True,
                original_tokens=len(result.original_text.split()) * 1.3,  # Rough token estimate
                summary_tokens=len(result.summary.split()) * 1.3,  # Rough token estimate
                original_lines=result.original_text.count('\n') + 1,
                summary_lines=result.summary.count('\n') + 1,
                strategy=result.strategy_used,
                compression_ratio=result.compression_ratio,
                file_path=file_path,
                instructions=[f"Summarized from {result.original_length} to {result.summary_length} characters using {result.strategy_used} strategy"],
                metadata={"time_elapsed": result.time_elapsed}
            )
        else:
            # If it's already a FileSummary or other format, return as-is
            return result

    def _estimate_git_tokens(self, git_context: Dict[str, Any]) -> int:
        """Estimate tokens needed for git context."""
        if not git_context:
            return 0

        # Rough estimates
        tokens = 0

        if "recent_commits" in git_context:
            # ~50 tokens per commit
            tokens += len(git_context["recent_commits"]) * 50

        if "contributors" in git_context:
            # ~20 tokens per contributor
            tokens += len(git_context["contributors"]) * 20

        if "recent_changes" in git_context:
            # ~100 tokens for change summary
            tokens += 100

        return tokens

    def optimize_packing(
        self, files: List[FileAnalysis], max_tokens: int, model: Optional[str] = None
    ) -> List[Tuple[FileAnalysis, bool]]:
        """Optimize file packing using dynamic programming.

        This is a more sophisticated packing algorithm that tries to
        maximize total relevance score within token constraints.

        Args:
            files: Files to pack
            max_tokens: Token budget
            model: Model for token counting

        Returns:
            List of (file, should_summarize) tuples
        """
        n = len(files)
        if n == 0:
            return []

        # Calculate tokens for each file (full and summarized)
        file_tokens = []
        for file in files:
            full_tokens = count_tokens(file.content, model)
            summary_tokens = full_tokens // 4  # Rough estimate
            file_tokens.append((full_tokens, summary_tokens))

        # Dynamic programming: dp[i][j] = max score using first i files with j tokens
        dp = [[0.0 for _ in range(max_tokens + 1)] for _ in range(n + 1)]
        choice = [[None for _ in range(max_tokens + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            file = files[i - 1]
            full_tokens, summary_tokens = file_tokens[i - 1]

            for j in range(max_tokens + 1):
                # Option 1: Skip this file
                dp[i][j] = dp[i - 1][j]
                choice[i][j] = "skip"

                # Option 2: Include full file
                if j >= full_tokens:
                    score = dp[i - 1][j - full_tokens] + file.relevance_score
                    if score > dp[i][j]:
                        dp[i][j] = score
                        choice[i][j] = "full"

                # Option 3: Include summarized file
                if j >= summary_tokens:
                    score = dp[i - 1][j - summary_tokens] + file.relevance_score * 0.6
                    if score > dp[i][j]:
                        dp[i][j] = score
                        choice[i][j] = "summary"

        # Backtrack to find optimal selection
        result = []
        i, j = n, max_tokens

        while i > 0 and j > 0:
            if choice[i][j] == "full":
                result.append((files[i - 1], False))
                j -= file_tokens[i - 1][0]
            elif choice[i][j] == "summary":
                result.append((files[i - 1], True))
                j -= file_tokens[i - 1][1]
            i -= 1

        result.reverse()
        return result
