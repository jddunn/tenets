"""Main summarizer orchestrator for content compression.

This module provides the main Summarizer class that coordinates different
summarization strategies to compress code, documentation, and other text
content while preserving important information.

The summarizer supports multiple strategies:
- Extractive: Selects important sentences
- Compressive: Removes redundant content
- TextRank: Graph-based ranking
- Transformer: Neural summarization (requires ML)
- LLM: Large language model summarization (costs $)
"""

import ast
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis
from tenets.utils.logger import get_logger
from tenets.utils.tokens import count_tokens

from .llm import create_llm_summarizer
from .strategies import (
    CompressiveStrategy,
    ExtractiveStrategy,
    SummarizationStrategy,
    TextRankStrategy,
    TransformerStrategy,
)


class SummarizationMode(Enum):
    """Available summarization modes."""

    EXTRACTIVE = "extractive"
    COMPRESSIVE = "compressive"
    TEXTRANK = "textrank"
    TRANSFORMER = "transformer"
    LLM = "llm"
    AUTO = "auto"  # Automatically select best strategy


@dataclass
class SummarizationResult:
    """Result from summarization operation.

    Attributes:
        original_text: Original text
        summary: Summarized text
        original_length: Original text length
        summary_length: Summary length
        compression_ratio: Actual compression ratio achieved
        strategy_used: Which strategy was used
        time_elapsed: Time taken to summarize
        metadata: Additional metadata
    """

    original_text: str
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    strategy_used: str
    time_elapsed: float
    metadata: Dict[str, Any] = None

    @property
    def reduction_percent(self) -> float:
        """Get reduction percentage."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "compression_ratio": self.compression_ratio,
            "reduction_percent": self.reduction_percent,
            "strategy_used": self.strategy_used,
            "time_elapsed": self.time_elapsed,
            "metadata": self.metadata or {},
        }


@dataclass
class BatchSummarizationResult:
    """Result from batch summarization."""

    results: List[SummarizationResult]
    total_original_length: int
    total_summary_length: int
    overall_compression_ratio: float
    total_time_elapsed: float
    files_processed: int
    files_failed: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_original_length": self.total_original_length,
            "total_summary_length": self.total_summary_length,
            "overall_compression_ratio": self.overall_compression_ratio,
            "total_time_elapsed": self.total_time_elapsed,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "reduction_percent": (1 - self.overall_compression_ratio) * 100,
        }


class Summarizer:
    """Main summarization orchestrator.

    Coordinates different summarization strategies and provides a unified
    interface for content compression. Supports single and batch processing,
    strategy selection, and caching.

    Attributes:
        config: TenetsConfig instance
        logger: Logger instance
        strategies: Available summarization strategies
        cache: Summary cache for repeated content
        stats: Summarization statistics
    """

    def __init__(
        self,
        config: Optional[TenetsConfig] = None,
        default_mode: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """Initialize summarizer.

        Args:
            config: Tenets configuration
            default_mode: Default summarization mode
            enable_cache: Whether to enable caching
        """
        self.config = config or TenetsConfig()
        self.logger = get_logger(__name__)

        # Determine default mode
        if default_mode:
            self.default_mode = SummarizationMode(default_mode)
        else:
            self.default_mode = SummarizationMode.AUTO

        # Initialize strategies
        self.strategies: Dict[SummarizationMode, SummarizationStrategy] = {
            SummarizationMode.EXTRACTIVE: ExtractiveStrategy(),
            SummarizationMode.COMPRESSIVE: CompressiveStrategy(),
            SummarizationMode.TEXTRANK: TextRankStrategy(),
        }

        # Try to initialize ML strategies
        self._init_ml_strategies()

        # Cache for summaries
        self.enable_cache = enable_cache
        self.cache: Dict[str, SummarizationResult] = {}

        # Statistics
        self.stats = {
            "total_summarized": 0,
            "total_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "strategies_used": {},
        }

        self.logger.info(
            f"Summarizer initialized with mode={self.default_mode.value}, "
            f"strategies={list(self.strategies.keys())}"
        )

    def _init_ml_strategies(self):
        """Initialize ML-based strategies if available."""
        # Honor configuration to enable/disable ML strategies (avoids heavy downloads in tests/CI)
        try:
            enable_ml = True
            if hasattr(self, "config") and hasattr(self.config, "summarizer"):
                enable_ml = bool(getattr(self.config.summarizer, "enable_ml_strategies", True))
        except Exception:
            enable_ml = True

        if not enable_ml:
            self.logger.info(
                "ML strategies disabled by config; skipping transformer/LLM initialization"
            )
            return

        # Try transformer strategy
        try:
            self.strategies[SummarizationMode.TRANSFORMER] = TransformerStrategy()
            self.logger.info("Transformer strategy available")
        except Exception as e:
            self.logger.debug(f"Transformer strategy not available: {e}")

        # LLM strategy is initialized on demand due to API keys

    def summarize(
        self,
        text: str,
        mode: Optional[Union[str, SummarizationMode]] = None,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        force_strategy: Optional[SummarizationStrategy] = None,
    ) -> SummarizationResult:
        """Summarize text content.

        Args:
            text: Text to summarize
            mode: Summarization mode (uses default if None)
            target_ratio: Target compression ratio (0.3 = 30% of original)
            max_length: Maximum summary length in characters
            min_length: Minimum summary length in characters
            force_strategy: Force specific strategy instance

        Returns:
            SummarizationResult with summary and metadata

        Example:
            >>> summarizer = Summarizer()
            >>> result = summarizer.summarize(
            ...     long_text,
            ...     mode="extractive",
            ...     target_ratio=0.25
            ... )
            >>> print(f"Reduced by {result.reduction_percent:.1f}%")
        """
        if not text:
            return SummarizationResult(
                original_text="",
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=1.0,
                strategy_used="none",
                time_elapsed=0.0,
            )

        start_time = time.time()

        # Check cache
        if self.enable_cache:
            cache_key = self._get_cache_key(text, target_ratio, max_length, min_length)
            if cache_key in self.cache:
                self.stats["cache_hits"] += 1
                self.logger.debug("Cache hit for summary")
                return self.cache[cache_key]
            else:
                self.stats["cache_misses"] += 1

        # Select strategy
        if force_strategy:
            strategy = force_strategy
            strategy_name = getattr(strategy, "name", "custom")
        else:
            strategy, strategy_name = self._select_strategy(text, mode, target_ratio)

        if not strategy:
            # Fallback to extractive
            strategy = self.strategies[SummarizationMode.EXTRACTIVE]
            strategy_name = "extractive"

        self.logger.debug(f"Using {strategy_name} strategy for summarization")

        # Perform summarization
        try:
            summary = strategy.summarize(
                text, target_ratio=target_ratio, max_length=max_length, min_length=min_length
            )
        except Exception as e:
            self.logger.error(f"Summarization failed with {strategy_name}: {e}")
            # Fallback to simple truncation
            summary = self._simple_truncate(text, target_ratio, max_length)
            strategy_name = "truncate"

        # Create result
        result = SummarizationResult(
            original_text=text,
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 1.0,
            strategy_used=strategy_name,
            time_elapsed=time.time() - start_time,
            metadata={
                "target_ratio": target_ratio,
                "max_length": max_length,
                "min_length": min_length,
            },
        )

        # Update statistics
        self.stats["total_summarized"] += 1
        self.stats["total_time"] += result.time_elapsed
        self.stats["strategies_used"][strategy_name] = (
            self.stats["strategies_used"].get(strategy_name, 0) + 1
        )

        # Cache result
        if self.enable_cache:
            self.cache[cache_key] = result

        self.logger.info(
            f"Summarized {result.original_length} chars to {result.summary_length} chars "
            f"({result.reduction_percent:.1f}% reduction) using {strategy_name}"
        )

        return result

    def summarize_file(
        self,
        file: FileAnalysis,
        mode: Optional[Union[str, SummarizationMode]] = None,
        target_ratio: float = 0.3,
        preserve_structure: bool = True,
    ) -> SummarizationResult:
        """Summarize a code file intelligently.

        Handles code files specially by preserving important elements
        like class/function signatures while summarizing implementations.

        Args:
            file: FileAnalysis object
            mode: Summarization mode
            target_ratio: Target compression ratio
            preserve_structure: Whether to preserve code structure

        Returns:
            SummarizationResult
        """
        if not file.content:
            return SummarizationResult(
                original_text="",
                summary="",
                original_length=0,
                summary_length=0,
                compression_ratio=1.0,
                strategy_used="none",
                time_elapsed=0.0,
            )

        if preserve_structure and file.language:
            # Intelligent code summarization
            summary = self._summarize_code(file, target_ratio)

            return SummarizationResult(
                original_text=file.content,
                summary=summary,
                original_length=len(file.content),
                summary_length=len(summary),
                compression_ratio=len(summary) / len(file.content),
                strategy_used="code-aware",
                time_elapsed=0.0,
                metadata={"file": file.path, "language": file.language},
            )
        else:
            # Regular text summarization
            return self.summarize(file.content, mode, target_ratio)

    def _summarize_code(self, file: FileAnalysis, target_ratio: float) -> str:
        """Intelligently summarize code files.

        Preserves structure (imports, signatures) while compressing
        implementations and removing verbose comments.

        Args:
            file: FileAnalysis with code
            target_ratio: Target compression ratio

        Returns:
            Summarized code
        """
        lines = file.content.split("\n")
        summary_lines = []

        # Always preserve imports
        for line in lines[:50]:  # Check first 50 lines for imports
            if self._is_import_line(line, file.language):
                summary_lines.append(line)

        # Add separator if we have imports
        if summary_lines:
            summary_lines.append("")

        # Preserve class and function signatures
        if file.structure:
            # Add class signatures
            for cls in file.structure.classes[:5]:  # Limit to top 5 classes
                if hasattr(cls, "definition"):
                    summary_lines.append(cls.definition)
                    summary_lines.append("    # ... implementation ...")
                    summary_lines.append("")

            # Add function signatures
            for func in file.structure.functions[:10]:  # Limit to top 10 functions
                if hasattr(func, "signature"):
                    summary_lines.append(func.signature)
                    if hasattr(func, "docstring") and func.docstring:
                        # Include first line of docstring
                        doc_first = func.docstring.split("\n")[0]
                        summary_lines.append(f'    """{doc_first}"""')
                    summary_lines.append("    # ... implementation ...")
                    summary_lines.append("")

        # If still too long, apply text summarization to comments/docstrings
        current_summary = "\n".join(summary_lines)

        if len(current_summary) > len(file.content) * target_ratio:
            # Need more compression
            # Extract and summarize comments/docstrings
            comments = self._extract_comments(file.content, file.language)
            if comments:
                comment_summary = self.summarize(
                    comments, mode=SummarizationMode.EXTRACTIVE, target_ratio=0.3
                ).summary

                summary_lines.append("# Summary of comments/documentation:")
                summary_lines.append(f"# {comment_summary}")

        return "\n".join(summary_lines)

    def _is_import_line(self, line: str, language: str) -> bool:
        """Check if line is an import statement.

        Args:
            line: Code line
            language: Programming language

        Returns:
            True if import line
        """
        line = line.strip()

        if language == "python":
            return line.startswith(("import ", "from "))
        elif language in ["javascript", "typescript"]:
            return line.startswith(("import ", "const ", "require("))
        elif language == "java":
            return line.startswith("import ")
        elif language in ["c", "cpp"]:
            return line.startswith("#include")
        elif language == "go":
            return line.startswith("import")
        elif language == "rust":
            return line.startswith("use ")
        else:
            # Generic patterns
            return any(line.startswith(p) for p in ["import", "include", "require", "use"])

    def _extract_comments(self, code: str, language: str) -> str:
        """Extract comments and docstrings from code.

        Args:
            code: Source code
            language: Programming language

        Returns:
            Extracted comments as text
        """
        comments = []
        lines = code.split("\n")

        in_block_comment = False
        block_delimiter = None

        for line in lines:
            stripped = line.strip()

            # Handle block comments
            if not in_block_comment:
                if language == "python" and stripped.startswith(('"""', "'''")):
                    in_block_comment = True
                    block_delimiter = stripped[:3]
                    comments.append(stripped[3:])
                elif language in ["c", "cpp", "java", "javascript"] and stripped.startswith("/*"):
                    in_block_comment = True
                    block_delimiter = "*/"
                    comments.append(stripped[2:])
                elif language == "html" and stripped.startswith("<!--"):
                    in_block_comment = True
                    block_delimiter = "-->"
                    comments.append(stripped[4:])
            elif block_delimiter in stripped:
                in_block_comment = False
                comments.append(stripped.replace(block_delimiter, ""))
            else:
                comments.append(stripped)

            # Handle single-line comments
            if not in_block_comment:
                if language == "python" and stripped.startswith("#"):
                    comments.append(stripped[1:].strip())
                elif language in ["c", "cpp", "java", "javascript"] and stripped.startswith("//"):
                    comments.append(stripped[2:].strip())

        return " ".join(comments)

    def batch_summarize(
        self,
        texts: List[Union[str, FileAnalysis]],
        mode: Optional[Union[str, SummarizationMode]] = None,
        target_ratio: float = 0.3,
        parallel: bool = True,
    ) -> BatchSummarizationResult:
        """Summarize multiple texts in batch.

        Args:
            texts: List of texts or FileAnalysis objects
            mode: Summarization mode
            target_ratio: Target compression ratio
            parallel: Whether to process in parallel

        Returns:
            BatchSummarizationResult
        """
        start_time = time.time()
        results = []

        total_original = 0
        total_summary = 0
        files_failed = 0

        for item in texts:
            try:
                if isinstance(item, FileAnalysis):
                    result = self.summarize_file(item, mode, target_ratio)
                else:
                    result = self.summarize(item, mode, target_ratio)

                results.append(result)
                total_original += result.original_length
                total_summary += result.summary_length

            except Exception as e:
                self.logger.error(f"Failed to summarize item: {e}")
                files_failed += 1

        overall_ratio = total_summary / total_original if total_original > 0 else 1.0

        return BatchSummarizationResult(
            results=results,
            total_original_length=total_original,
            total_summary_length=total_summary,
            overall_compression_ratio=overall_ratio,
            total_time_elapsed=time.time() - start_time,
            files_processed=len(results),
            files_failed=files_failed,
        )

    def _select_strategy(
        self, text: str, mode: Optional[Union[str, SummarizationMode]], target_ratio: float
    ) -> Tuple[Optional[SummarizationStrategy], str]:
        """Select best summarization strategy.

        Args:
            text: Text to summarize
            mode: Requested mode or None for auto
            target_ratio: Target compression ratio

        Returns:
            Tuple of (strategy, strategy_name)
        """
        # Convert string to enum
        if isinstance(mode, str):
            try:
                mode = SummarizationMode(mode)
            except ValueError:
                mode = self.default_mode
        elif mode is None:
            mode = self.default_mode

        # Handle explicit mode
        if mode != SummarizationMode.AUTO:
            # Special handling for LLM mode
            if mode == SummarizationMode.LLM:
                if mode not in self.strategies:
                    # Initialize LLM strategy on demand
                    try:
                        self.strategies[mode] = create_llm_summarizer()
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize LLM strategy: {e}")
                        mode = SummarizationMode.EXTRACTIVE

            strategy = self.strategies.get(mode)
            return strategy, mode.value if strategy else None

        # Auto mode - select based on content characteristics
        text_length = len(text)

        # For very short text, use extractive
        if text_length < 500:
            return self.strategies[SummarizationMode.EXTRACTIVE], "extractive"

        # For code-like content, use extractive
        if self._looks_like_code(text):
            return self.strategies[SummarizationMode.EXTRACTIVE], "extractive"

        # For medium text, use TextRank if available
        if text_length < 5000:
            if SummarizationMode.TEXTRANK in self.strategies:
                return self.strategies[SummarizationMode.TEXTRANK], "textrank"
            else:
                return self.strategies[SummarizationMode.COMPRESSIVE], "compressive"

        # For long text, prefer transformer if available and ratio is aggressive
        if target_ratio < 0.2 and SummarizationMode.TRANSFORMER in self.strategies:
            return self.strategies[SummarizationMode.TRANSFORMER], "transformer"

        # Default to extractive
        return self.strategies[SummarizationMode.EXTRACTIVE], "extractive"

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code.

        Args:
            text: Text to check

        Returns:
            True if text appears to be code
        """
        code_indicators = [
            "def ",
            "class ",
            "function ",
            "const ",
            "var ",
            "let ",
            "import ",
            "from ",
            "#include",
            "return ",
            "if (",
            "for (",
            "```",
            "{",
            "}",
            ";",
            "->",
            "=>",
            "::",
        ]

        indicator_count = sum(1 for ind in code_indicators if ind in text)
        return indicator_count >= 3

    def _simple_truncate(self, text: str, target_ratio: float, max_length: Optional[int]) -> str:
        """Simple truncation fallback.

        Args:
            text: Text to truncate
            target_ratio: Target ratio
            max_length: Maximum length

        Returns:
            Truncated text
        """
        target_len = int(len(text) * target_ratio)
        if max_length:
            target_len = min(target_len, max_length)

        if len(text) <= target_len:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:target_len]
        last_period = truncated.rfind(".")
        if last_period > target_len * 0.8:
            return truncated[: last_period + 1]

        # Truncate at word boundary
        return truncated.rsplit(" ", 1)[0] + "..."

    def _get_cache_key(
        self, text: str, target_ratio: float, max_length: Optional[int], min_length: Optional[int]
    ) -> str:
        """Generate cache key for summary.

        Args:
            text: Input text
            target_ratio: Target ratio
            max_length: Max length
            min_length: Min length

        Returns:
            Cache key string
        """
        # Use hash of text + parameters
        key_parts = [
            hashlib.md5(text.encode()).hexdigest(),
            str(target_ratio),
            str(max_length),
            str(min_length),
        ]
        return "_".join(key_parts)

    def clear_cache(self):
        """Clear the summary cache."""
        self.cache.clear()
        self.logger.info("Summary cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get summarization statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()

        # Add cache stats
        stats["cache_size"] = len(self.cache)
        if self.stats["cache_hits"] + self.stats["cache_misses"] > 0:
            stats["cache_hit_rate"] = self.stats["cache_hits"] / (
                self.stats["cache_hits"] + self.stats["cache_misses"]
            )
        else:
            stats["cache_hit_rate"] = 0.0

        # Add average time
        if self.stats["total_summarized"] > 0:
            stats["avg_time"] = self.stats["total_time"] / self.stats["total_summarized"]
        else:
            stats["avg_time"] = 0.0

        return stats


class FileSummarizer:
    """Backward-compatible file summarizer used by tests.

    This lightweight class focuses on extracting a concise summary from a single
    file using deterministic heuristics (docstrings, leading comments, or head
    lines). It integrates with Tenets token utilities and returns the
    `FileSummary` model expected by the tests.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.logger = get_logger(__name__)

    # --- Public API used by tests ---
    def summarize_file(self, path: Union[str, Path], max_lines: int = 50):
        """Summarize a file from disk into a FileSummary.

        Args:
            path: Path to the file
            max_lines: Maximum number of lines in the summary

        Returns:
            FileSummary: summary object with metadata
        """
        from tenets.models.summary import FileSummary  # local import to avoid cycles

        p = Path(path)
        text = self._read_text(p)
        summary_text = self._extract_summary(text, max_lines=max_lines, file_path=p)

        tokens = count_tokens(summary_text, model=self.model)
        metadata = {"strategy": "heuristic", "max_lines": max_lines}

        return FileSummary(
            path=str(p),
            summary=summary_text,
            token_count=tokens,
            metadata=metadata,
        )

    # --- Helpers expected by tests ---
    def _read_text(self, path: Union[str, Path]) -> str:
        """Read text from file, tolerating different encodings and binary data.

        Returns empty string if the file does not exist.
        """
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""

        # Try common encodings, then fall back to permissive decode
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return p.read_text(encoding=enc)
            except Exception:
                continue

        # Final fallback: binary-safe read and decode with errors ignored
        try:
            data = p.read_bytes()
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _extract_summary(
        self, text: str, max_lines: int = 50, file_path: Optional[Path] = None
    ) -> str:
        """Extract a human-friendly summary from the file content.

        Preference order:
        1) Python module docstring (if parseable)
        2) Leading comment block (Python // JS /**/ styles)
        3) First N lines of the file
        """
        if not text:
            return ""

        # 1) Try Python module docstring via AST when it looks like Python
        looks_py = False
        if file_path is not None and file_path.suffix.lower() in {".py", ".pyw"}:
            looks_py = True
        else:
            # Heuristic look for Python indicators
            indicators = ("def ", "class ", "import ", "from ", '"""', "'''")
            looks_py = sum(1 for ind in indicators if ind in text) >= 2

        if looks_py:
            try:
                module = ast.parse(text)
                doc = ast.get_docstring(module, clean=True)
                if doc:
                    return self._limit_lines(doc, max_lines)
            except Exception:
                # Fall through to other strategies on parse errors
                pass

        # 2) Leading comment block extraction (supports Python/JS/TS)
        comment_block = self._extract_leading_comments(text)
        if comment_block:
            return self._limit_lines(comment_block, max_lines)

        # 3) Fallback to head extraction
        return self._limit_lines(text, max_lines)

    # --- Internal helpers ---
    def _limit_lines(self, text: str, max_lines: int) -> str:
        if max_lines <= 0:
            return ""
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return "\n".join(lines)
        return "\n".join(lines[:max_lines])

    def _extract_leading_comments(self, text: str) -> str:
        lines = text.splitlines()
        buf: List[str] = []
        i = 0

        # Skip shebang or coding lines
        while i < len(lines) and (lines[i].startswith("#!/") or "coding:" in lines[i]):
            i += 1

        # Triple-quoted doc/comment block at top
        if i < len(lines) and (
            lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''")
        ):
            quote = '"""' if '"""' in lines[i] else "'''"
            first = lines[i].split(quote, 1)[-1]
            if first:
                buf.append(first)
            i += 1
            while i < len(lines):
                line = lines[i]
                if quote in line:
                    before, _sep, _after = line.partition(quote)
                    buf.append(before)
                    break
                buf.append(line)
                i += 1
            if buf:
                return "\n".join(l.strip("\n") for l in buf).strip()

        # Line-comment blocks (Python # or JS //) at file head
        j = i
        while j < len(lines) and (
            lines[j].lstrip().startswith("#") or lines[j].lstrip().startswith("//")
        ):
            # Strip leading comment markers while keeping content
            line = lines[j].lstrip()
            if line.startswith("#"):
                content = line[1:].strip()
            else:
                content = line[2:].strip()
            buf.append(content)
            j += 1
        if buf:
            return "\n".join(buf).strip()

        # JS/TS block comments /* ... */ at head
        if i < len(lines) and lines[i].lstrip().startswith("/*"):
            inner = lines[i].lstrip()[2:]
            if inner:
                buf.append(inner.rstrip("*/ "))
            i += 1
            while i < len(lines):
                line = lines[i]
                if "*/" in line:
                    before, _sep, _after = line.partition("*/")
                    buf.append(before)
                    break
                buf.append(line)
                i += 1
            if buf:
                return "\n".join(buf).strip()

        return ""
