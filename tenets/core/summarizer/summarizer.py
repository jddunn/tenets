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
from .summarizer_utils import (
    ASTParser,
    CodeDetector,
    ImportParser,
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
        }

        # Try to initialize TextRank (requires scikit-learn)
        try:
            self.strategies[SummarizationMode.TEXTRANK] = TextRankStrategy()
        except ImportError:
            self.logger.debug("TextRank unavailable (scikit-learn not installed)")

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

        # Enforce min_length: if requested min_length exceeds original, do not make it shorter
        if min_length and min_length > len(text) and len(text) > 0 and len(summary) < len(text):
            summary = text

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
        prompt_keywords: Optional[List[str]] = None,
    ) -> SummarizationResult:
        """Summarize a code file intelligently.

        Handles code files specially by preserving important elements
        like class/function signatures while summarizing implementations.
        Enhanced with context-aware documentation summarization that preserves
        relevant sections based on prompt keywords.

        Args:
            file: FileAnalysis object
            mode: Summarization mode
            target_ratio: Target compression ratio
            preserve_structure: Whether to preserve code structure
            prompt_keywords: Keywords from user prompt for context-aware summarization

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

        # Determine if this is a documentation file
        file_path = Path(file.path)
        is_documentation = self._is_documentation_file(file_path)

        # Check if context-aware documentation summarization is enabled
        docs_context_aware = getattr(self.config.summarizer, "docs_context_aware", True)
        docs_show_in_place_context = getattr(
            self.config.summarizer, "docs_show_in_place_context", True
        )

        # Apply documentation-specific summarization if enabled and applicable
        if (
            is_documentation
            and docs_context_aware
            and docs_show_in_place_context
            and prompt_keywords
        ):
            summary = self._summarize_documentation_with_context(
                file, target_ratio, prompt_keywords
            )

            return SummarizationResult(
                original_text=file.content,
                summary=summary,
                original_length=len(file.content),
                summary_length=len(summary),
                compression_ratio=len(summary) / len(file.content),
                strategy_used="docs-context-aware",
                time_elapsed=0.0,
                metadata={
                    "file": file.path,
                    "is_documentation": True,
                    "prompt_keywords": prompt_keywords,
                    "context_aware": True,
                },
            )
        elif preserve_structure and file.language and not is_documentation:
            # Intelligent code summarization (skip for documentation files)
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

        Preserves structure (imports, class definitions, method signatures,
        function signatures with arguments and return types) while compressing
        implementations and removing verbose comments.

        Args:
            file: FileAnalysis with code
            target_ratio: Target compression ratio

        Returns:
            Summarized code with enhanced structure information
        """
        lines = file.content.split("\n")
        summary_lines = []

        # Handle imports based on configuration
        import_lines = []
        for line in lines[:50]:  # Check first 50 lines for imports
            if self._is_import_line(line, file.language):
                import_lines.append(line)

        # Check if we should summarize imports
        summarize_imports = getattr(self.config.summarizer, "summarize_imports", True)
        if summarize_imports and len(import_lines) > 5:  # Only summarize if more than 5 imports
            import_summary = self._summarize_imports(import_lines, file.language)
            summary_lines.append(import_summary)
        else:
            # Preserve imports as-is
            summary_lines.extend(import_lines)

        # Add separator if we have imports
        if summary_lines:
            summary_lines.append("")

        # Get docstring weight configuration
        docstring_weight = getattr(self.config.summarizer, "docstring_weight", 0.5)
        include_all_signatures = getattr(self.config.summarizer, "include_all_signatures", True)

        # Determine limits based on configuration
        max_classes = None if include_all_signatures else 10  # Show more classes by default
        max_functions = None if include_all_signatures else 20  # Show more functions by default

        # Preserve class and function signatures with enhanced details
        if file.structure:
            # Add class signatures with methods
            # Handle both list and Mock objects for classes
            if hasattr(file.structure.classes, "__iter__") and not isinstance(
                file.structure.classes, str
            ):
                all_classes = list(file.structure.classes)
                classes_to_include = all_classes[:max_classes] if max_classes else all_classes
            else:
                classes_to_include = []
            for cls in classes_to_include:
                # Include class definition
                if hasattr(cls, "definition"):
                    summary_lines.append(cls.definition)
                elif hasattr(cls, "name"):
                    # Fallback: construct class definition
                    base_classes = getattr(cls, "base_classes", [])
                    if base_classes:
                        summary_lines.append(f"class {cls.name}({', '.join(base_classes)}):")
                    else:
                        summary_lines.append(f"class {cls.name}:")

                # Include class docstring based on weight
                doc = getattr(cls, "docstring", None)
                if isinstance(doc, str) and doc and docstring_weight > 0:
                    # Include docstring based on weight (1.0 = always, 0.5 = sometimes, 0 = never)
                    if docstring_weight >= 1.0 or (docstring_weight > 0 and len(doc) < 200):
                        doc_lines = doc.split("\n")
                        if len(doc_lines) <= 3 or docstring_weight >= 0.8:
                            # Include full docstring for short ones or high weight
                            summary_lines.append(f'    """{doc}"""')
                        else:
                            # Include first line only for longer docstrings with lower weight
                            summary_lines.append(f'    """{doc_lines[0]}..."""')

                # Include class methods with signatures
                methods = getattr(cls, "methods", [])
                if methods:
                    # Handle both list and Mock objects
                    if hasattr(methods, "__iter__") and not isinstance(methods, str):
                        methods_to_process = list(methods)[:10]  # Limit methods per class
                    else:
                        methods_to_process = []

                    for method in methods_to_process:
                        method_name = getattr(method, "name", "")
                        method_args = getattr(method, "arguments", [])
                        method_return = getattr(method, "return_type", None)

                        # Build method signature
                        if method_name:
                            args_str = ", ".join(method_args) if method_args else ""
                            sig = f"    def {method_name}({args_str})"
                            if method_return:
                                sig += f" -> {method_return}"
                            sig += ": ..."
                            summary_lines.append(sig)
                        elif hasattr(method, "signature"):
                            summary_lines.append(f"    {method.signature}: ...")

                summary_lines.append("    # ... additional methods ...")
                summary_lines.append("")

            # Add function signatures with enhanced details
            # Handle both list and Mock objects for functions
            if hasattr(file.structure.functions, "__iter__") and not isinstance(
                file.structure.functions, str
            ):
                all_functions = list(file.structure.functions)
                functions_to_include = (
                    all_functions[:max_functions] if max_functions else all_functions
                )
            else:
                functions_to_include = []
            for func in functions_to_include:
                # Build enhanced function signature
                func_name = getattr(func, "name", "")
                func_args = getattr(func, "arguments", [])
                func_return = getattr(func, "return_type", None)

                if func_name:
                    # Build detailed signature
                    # Handle different types of func_args
                    if func_args:
                        if isinstance(func_args, (list, tuple)):
                            args_str = ", ".join(func_args)
                        elif isinstance(func_args, str):
                            args_str = func_args
                        else:
                            args_str = ""
                    else:
                        args_str = ""
                    sig = f"def {func_name}({args_str})"
                    if func_return:
                        sig += f" -> {func_return}"
                    sig += ":"
                    summary_lines.append(sig)
                elif hasattr(func, "signature"):
                    summary_lines.append(func.signature)
                else:
                    continue

                # Include docstring based on weight
                doc = getattr(func, "docstring", None)
                if isinstance(doc, str) and doc and docstring_weight > 0:
                    if docstring_weight >= 1.0:
                        # Always include full docstring
                        summary_lines.append(f'    """{doc}"""')
                    elif docstring_weight >= 0.7:
                        # Include first paragraph or up to 3 lines
                        doc_lines = doc.split("\n")
                        if len(doc_lines) <= 3:
                            summary_lines.append(f'    """{doc}"""')
                        else:
                            first_para = doc.split("\n\n")[0]
                            summary_lines.append(f'    """{first_para}..."""')
                    elif docstring_weight > 0:
                        # Include only first line
                        doc_first = doc.split("\n")[0]
                        if doc_first:
                            summary_lines.append(f'    """{doc_first}"""')

                summary_lines.append("    # ... implementation ...")
                summary_lines.append("")

        # If structure is missing, fall back to AST/regex extraction with enhanced parsing
        if (
            not file.structure or len(summary_lines) < 10
        ):  # Also use if we got too little from structure
            try:
                # Python-specific AST parse for enhanced function/class signatures
                if file.language == "python":
                    module = ast.parse(file.content)

                    # Get docstring weight for fallback extraction
                    docstring_weight = getattr(self.config.summarizer, "docstring_weight", 0.5)

                    for node in module.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Extract detailed function signature
                            args = []
                            for arg in node.args.args:
                                arg_str = arg.arg
                                # Add type annotation if available
                                if arg.annotation:
                                    try:
                                        arg_str += f": {ast.unparse(arg.annotation)}"
                                    except:
                                        pass
                                args.append(arg_str)

                            # Add return type if available
                            sig = f"def {node.name}({', '.join(args)})"
                            if node.returns:
                                try:
                                    sig += f" -> {ast.unparse(node.returns)}"
                                except:
                                    pass
                            sig += ": ..."
                            summary_lines.append(sig)

                            # Include docstring based on weight
                            docstring = ast.get_docstring(node)
                            if docstring and docstring_weight > 0.3:
                                doc_first = docstring.split("\n")[0]
                                if doc_first:
                                    summary_lines.append(f'    """{doc_first}"""')

                        elif isinstance(node, ast.ClassDef):
                            # Extract class with base classes
                            bases = []
                            for base in node.bases:
                                try:
                                    bases.append(ast.unparse(base))
                                except:
                                    if hasattr(base, "id"):
                                        bases.append(base.id)

                            if bases:
                                summary_lines.append(f"class {node.name}({', '.join(bases)}):")
                            else:
                                summary_lines.append(f"class {node.name}:")

                            # Include class docstring if weight is high
                            docstring = ast.get_docstring(node)
                            if docstring and docstring_weight >= 0.5:
                                doc_first = docstring.split("\n")[0]
                                if doc_first:
                                    summary_lines.append(f'    """{doc_first}"""')

                            # Extract method signatures from class
                            method_count = 0
                            for item in node.body:
                                if (
                                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and method_count < 5
                                ):
                                    args = [a.arg for a in item.args.args]
                                    method_sig = f"    def {item.name}({', '.join(args)})"
                                    if item.returns:
                                        try:
                                            method_sig += f" -> {ast.unparse(item.returns)}"
                                        except:
                                            pass
                                    method_sig += ": ..."
                                    summary_lines.append(method_sig)
                                    method_count += 1

                            if method_count > 0:
                                summary_lines.append("    # ... additional methods ...")
                            summary_lines.append("")

                    if summary_lines:
                        summary_lines.append("")
                else:
                    # Enhanced regex patterns for other languages
                    import re

                    lines = file.content.splitlines()

                    # Language-specific patterns
                    if file.language in ["javascript", "typescript"]:
                        # Match class, function, arrow functions, methods
                        patterns = [
                            re.compile(r"^\s*(export\s+)?(class|interface)\s+(\w+).*$"),
                            re.compile(
                                r"^\s*(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\).*$"
                            ),
                            re.compile(
                                r"^\s*(export\s+)?const\s+(\w+)\s*=\s*(async\s+)?\([^)]*\)\s*=>.*$"
                            ),
                            re.compile(r"^\s*(\w+)\s*\([^)]*\)\s*\{.*$"),  # Method in class
                        ]
                    elif file.language == "java":
                        patterns = [
                            re.compile(
                                r"^\s*(public|private|protected)\s+(class|interface)\s+(\w+).*$"
                            ),
                            re.compile(
                                r"^\s*(public|private|protected)\s+.*\s+(\w+)\s*\([^)]*\).*$"
                            ),
                        ]
                    elif file.language in ["c", "cpp"]:
                        patterns = [
                            re.compile(r"^\s*(class|struct)\s+(\w+).*$"),
                            re.compile(r"^\s*\w+\s+\w+\s*\([^)]*\).*$"),
                        ]
                    else:
                        # Generic patterns
                        patterns = [
                            re.compile(r"\b(class|interface|struct)\s+(\w+)"),
                            re.compile(r"\b(def|function|func)\s+(\w+)\s*\([^)]*\)"),
                        ]

                    for i, line in enumerate(lines[:300]):  # Check more lines
                        for pattern in patterns:
                            if pattern.search(line):
                                # Include the signature line
                                summary_lines.append(line.strip())
                                # Look for return type on next line (for multi-line signatures)
                                if i + 1 < len(lines) and "->" in lines[i + 1]:
                                    summary_lines.append(lines[i + 1].strip())
                                break

                    if summary_lines:
                        summary_lines.append("")
            except Exception:
                # Ignore parse errors and continue
                pass

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
        return ImportParser.is_import_line(line, language)

    def _summarize_imports(self, import_lines: List[str], language: str) -> str:
        """Summarize import statements into a concise description.

        Args:
            import_lines: List of import statement lines
            language: Programming language

        Returns:
            Human-readable summary of imports
        """
        import re

        if not import_lines:
            return ""

        # Parse imports to extract library names
        libraries = set()
        local_imports = 0

        for line in import_lines:
            line = line.strip()

            if language == "python":
                if line.startswith("from "):
                    # Extract module name from "from X import Y"
                    parts = line.split()
                    if len(parts) >= 2:
                        module = parts[1].split(".")[0]
                        if module.startswith("."):
                            local_imports += 1
                        else:
                            libraries.add(module)
                elif line.startswith("import "):
                    # Extract module name from "import X"
                    parts = line.replace("import ", "").split(",")
                    for part in parts:
                        module = part.strip().split(".")[0].split(" as ")[0]
                        libraries.add(module)

            elif language in ["javascript", "typescript"]:
                if "from " in line:
                    # Extract module from import statements
                    match = re.search(r'from\s+["\']([^"\']+)["\']', line)
                    if match:
                        module = match.group(1)
                        if module.startswith("."):
                            local_imports += 1
                        else:
                            # Extract package name (first part before /)
                            package = module.split("/")[0].replace("@", "")
                            libraries.add(package)
                elif "require(" in line:
                    match = re.search(r'require\(["\']([^"\']+)["\']\)', line)
                    if match:
                        module = match.group(1)
                        if module.startswith("."):
                            local_imports += 1
                        else:
                            libraries.add(module.split("/")[0])

            elif language == "java":
                # Extract package from import statement
                parts = line.replace("import ", "").replace(";", "").split(".")
                if len(parts) >= 2:
                    # First two parts usually form the main package
                    libraries.add(".".join(parts[:2]))

            elif language in ["c", "cpp"]:
                # Extract header file
                match = re.search(r'#include\s*[<"]([^>"]+)[>"]', line)
                if match:
                    header = match.group(1)
                    if "/" in header:
                        libraries.add(header.split("/")[0])
                    else:
                        libraries.add(header.replace(".h", "").replace(".hpp", ""))

            elif language == "rust":
                # Extract crate name
                parts = line.replace("use ", "").replace(";", "").split("::")
                if parts[0] not in ["self", "super", "crate"]:
                    libraries.add(parts[0])
                elif parts[0] == "crate" and len(parts) > 1:
                    local_imports += 1

            elif language == "go":
                # Extract package name
                if '"' in line:
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        pkg = match.group(1)
                        if not pkg.startswith("."):
                            # Extract main package name
                            libraries.add(pkg.split("/")[0])

        # Build summary
        total_imports = len(import_lines)
        unique_libs = sorted(libraries)

        summary_parts = [f"# Imports: {total_imports} total"]

        if unique_libs:
            if len(unique_libs) <= 8:
                summary_parts.append(f"# Dependencies: {', '.join(unique_libs)}")
            else:
                # Show first 6 libraries and count of others
                shown_libs = unique_libs[:6]
                remaining = len(unique_libs) - 6
                summary_parts.append(
                    f"# Dependencies: {', '.join(shown_libs)}, and {remaining} others"
                )

        if local_imports > 0:
            summary_parts.append(f"# Local imports: {local_imports}")

        return "\n".join(summary_parts)

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
        prompt_keywords: Optional[List[str]] = None,
    ) -> BatchSummarizationResult:
        """Summarize multiple texts in batch.

        Args:
            texts: List of texts or FileAnalysis objects
            mode: Summarization mode
            target_ratio: Target compression ratio
            parallel: Whether to process in parallel
            prompt_keywords: Keywords from user prompt for context-aware documentation summarization

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
                    result = self.summarize_file(
                        item, mode, target_ratio, prompt_keywords=prompt_keywords
                    )
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
        # Use shared utility with threshold of 3 for consistency
        return CodeDetector.looks_like_code(text, threshold=3)

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

    def _is_documentation_file(self, file_path: Path) -> bool:
        """Determine if a file is a documentation file.

        Args:
            file_path: Path to the file

        Returns:
            True if file is considered documentation
        """
        # Common documentation file extensions
        doc_extensions = {
            ".md",
            ".markdown",
            ".rst",
            ".txt",
            ".text",
            ".adoc",
            ".asciidoc",
            ".tex",
            ".org",
        }

        # Configuration files that often contain documentation
        config_extensions = {
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
            ".properties",
            ".env",
        }

        # Common documentation file names
        doc_names = {
            "readme",
            "changelog",
            "changes",
            "history",
            "news",
            "authors",
            "contributors",
            "license",
            "copying",
            "install",
            "installation",
            "usage",
            "tutorial",
            "guide",
            "manual",
            "help",
            "faq",
            "todo",
            "notes",
            "docs",
            "documentation",
        }

        extension = file_path.suffix.lower()
        name_lower = file_path.stem.lower()

        # Check extension
        if extension in doc_extensions:
            return True

        # Check configuration files (often contain documentation)
        if extension in config_extensions:
            return True

        # Check common documentation file names
        if name_lower in doc_names:
            return True

        # Check if file is in docs-related directories
        parts = [part.lower() for part in file_path.parts]
        doc_dirs = {"docs", "doc", "documentation", "guide", "guides", "manual", "help"}
        if any(part in doc_dirs for part in parts):
            return True

        return False

    def _summarize_documentation_with_context(
        self, file: FileAnalysis, target_ratio: float, prompt_keywords: List[str]
    ) -> str:
        """Summarize documentation file with context-aware approach.

        This method preserves sections that are relevant to the prompt keywords
        and shows them in-place within the summary, maintaining the document
        structure while highlighting relevant context.

        Args:
            file: FileAnalysis object for the documentation file
            target_ratio: Target compression ratio
            prompt_keywords: Keywords from user prompt for relevance matching

        Returns:
            Context-aware summary with relevant sections preserved
        """
        from tenets.core.analysis.implementations.generic_analyzer import GenericAnalyzer

        # Get configuration parameters
        search_depth = getattr(self.config.summarizer, "docs_context_search_depth", 2)
        min_confidence = getattr(self.config.summarizer, "docs_context_min_confidence", 0.6)
        max_sections = getattr(self.config.summarizer, "docs_context_max_sections", 10)
        preserve_examples = getattr(self.config.summarizer, "docs_context_preserve_examples", True)

        # Use generic analyzer to extract context-relevant sections
        analyzer = GenericAnalyzer()
        file_path = Path(file.path)

        context_result = analyzer.extract_context_relevant_sections(
            content=file.content,
            file_path=file_path,
            prompt_keywords=prompt_keywords,
            search_depth=search_depth,
            min_confidence=min_confidence,
            max_sections=max_sections,
        )

        relevant_sections = context_result["relevant_sections"]
        metadata = context_result["metadata"]

        summary_parts = []

        # Add file header with context information
        summary_parts.append(f"# {file_path.name}")
        summary_parts.append(
            f"*Documentation file with {len(relevant_sections)} relevant sections (search depth: {search_depth})*"
        )
        summary_parts.append("")

        # If no relevant sections found, fall back to generic summarization
        if not relevant_sections:
            self.logger.debug(
                f"No relevant sections found in {file.path}, using generic summarization"
            )
            generic_summary = self.summarize(
                file.content, mode=SummarizationMode.EXTRACTIVE, target_ratio=target_ratio
            ).summary
            summary_parts.append("## Summary")
            summary_parts.append(generic_summary)
            return "\n".join(summary_parts)

        # Add table of contents for relevant sections
        if len(relevant_sections) > 1:
            summary_parts.append("## Relevant Sections")
            for i, section in enumerate(relevant_sections, 1):
                context_type = section.get("context_type", "general")
                score = section.get("relevance_score", 0.0)
                summary_parts.append(
                    f"{i}. **{section['title']}** ({context_type}, relevance: {score:.2f})"
                )
            summary_parts.append("")

        # Process and include relevant sections with full context
        total_length = 0
        target_length = int(len(file.content) * target_ratio)

        for section in relevant_sections:
            if total_length >= target_length:
                break

            section_summary = self._format_relevant_section(section, preserve_examples)
            section_length = len(section_summary)

            # Include section if it fits within our target
            if total_length + section_length <= target_length * 1.2:  # Allow 20% overage
                summary_parts.append(section_summary)
                summary_parts.append("")
                total_length += section_length
            else:
                # For configuration sections, truncate rather than compress
                section_type = section.get("section_type", "")
                if section_type in ["yaml_key", "json_key", "ini_section", "toml_section"]:
                    # Truncate content to fit
                    truncated_section = section.copy()
                    content_lines = section["content"].split("\n")
                    # Keep first N lines that fit
                    max_lines = min(10, len(content_lines))
                    truncated_section["content"] = "\n".join(content_lines[:max_lines])
                    if len(content_lines) > max_lines:
                        truncated_section["content"] += "\n# ... (truncated)"
                    section_summary = self._format_relevant_section(
                        truncated_section, preserve_examples
                    )
                else:
                    # Compress the section content to fit while preserving code examples
                    compressed_section = section.copy()

                    # Extract code examples before compression
                    code_examples = section.get("code_examples", [])

                    if code_examples and preserve_examples:
                        # If we have code examples to preserve, compress the text around them
                        content_lines = section["content"].split("\n")
                        preserved_lines = []
                        in_code_block = False
                        code_block_lines = []

                        for line in content_lines:
                            # Check if we're entering or exiting a code block
                            if line.strip().startswith("```"):
                                if not in_code_block:
                                    in_code_block = True
                                    code_block_lines = [line]
                                else:
                                    code_block_lines.append(line)
                                    preserved_lines.extend(code_block_lines)
                                    in_code_block = False
                                    code_block_lines = []
                            elif in_code_block:
                                code_block_lines.append(line)
                            # For non-code lines, only keep the most important ones
                            elif any(kw.lower() in line.lower() for kw in prompt_keywords) or (
                                line.strip() and len(preserved_lines) < 5
                            ):
                                preserved_lines.append(line)

                        # If still in a code block at the end, preserve it
                        if in_code_block and code_block_lines:
                            preserved_lines.extend(code_block_lines)
                            preserved_lines.append("```")  # Close the code block

                        compressed_section["content"] = "\n".join(preserved_lines)
                    else:
                        # No code examples to preserve, use regular compression
                        compressed_content = self.summarize(
                            section["content"], mode=SummarizationMode.EXTRACTIVE, target_ratio=0.5
                        ).summary
                        compressed_section["content"] = compressed_content

                    section_summary = self._format_relevant_section(
                        compressed_section, preserve_examples
                    )
                summary_parts.append(section_summary)
                summary_parts.append("")
                total_length += len(section_summary)

        # Add metadata footer
        summary_parts.append("---")
        summary_parts.append(
            f"*Context analysis: {metadata['matched_sections']}/{metadata['total_sections']} sections matched, "
            f"avg relevance: {metadata['avg_relevance_score']:.2f}*"
        )

        return "\n".join(summary_parts)

    def _format_relevant_section(
        self, section: Dict[str, Any], preserve_examples: bool = True
    ) -> str:
        """Format a relevant section for inclusion in the summary.

        Args:
            section: Section dictionary with content and metadata
            preserve_examples: Whether to preserve code examples

        Returns:
            Formatted section content
        """
        parts = []

        # Section header with relevance info
        title = section.get("title", "Section")
        context_type = section.get("context_type", "general")
        relevance_score = section.get("relevance_score", 0.0)

        # Determine header level (ensure it's at least ##)
        level = max(section.get("level", 2), 2)
        header_prefix = "#" * level

        parts.append(f"{header_prefix} {title}")
        parts.append(f"*{context_type.title()} content (relevance: {relevance_score:.2f})*")
        parts.append("")

        # Add keyword matches summary
        keyword_matches = section.get("keyword_matches", [])
        if keyword_matches:
            match_summary = []
            for match in keyword_matches[:3]:  # Show top 3 matches
                keyword = match.get("keyword", "")
                match_type = match.get("match_type", "")
                count = match.get("count", 0)
                if count > 0:
                    match_summary.append(f"{keyword} ({match_type}: {count})")

            if match_summary:
                parts.append(f"**Keywords found:** {', '.join(match_summary)}")
                parts.append("")

        # Process section content
        content = section.get("content", "")

        # For configuration sections, preserve the actual configuration syntax
        section_type = section.get("section_type", "")
        if section_type in ["yaml_key", "json_key", "ini_section", "toml_section"]:
            # Show a snippet of the actual configuration
            content_lines = content.split("\n")
            if len(content_lines) > 10:
                # Show first 8 lines and indicate truncation
                snippet = "\n".join(content_lines[:8])
                parts.append(f"```yaml\n{snippet}\n# ... (truncated)\n```")
            else:
                parts.append(f"```yaml\n{content.strip()}\n```")
        else:
            # Extract and highlight code examples if preserve_examples is True
            if preserve_examples:
                code_examples = section.get("code_examples", [])
                if code_examples:
                    # Replace code blocks with highlighted versions
                    for example in code_examples:
                        if example["type"] == "code_block":
                            lang = example.get("language", "")
                            code = example.get("code", "")
                            raw_match = example.get("raw_match", "")
                            # Only replace if we have a valid raw_match
                            if raw_match:
                                # Ensure code blocks are properly highlighted
                                highlighted = f"```{lang}\n{code}\n```"
                                # Find and replace in content (this is a simplified approach)
                                content = content.replace(raw_match, highlighted)

            # Add the content
            parts.append(content.strip())

        # Add extracted references if significant
        api_refs = section.get("api_references", [])
        config_refs = section.get("config_references", [])

        if api_refs or config_refs:
            parts.append("")
            if api_refs:
                api_names = [ref.get("name", "") for ref in api_refs[:5]]
                parts.append(f"**API References:** {', '.join(filter(None, api_names))}")

            if config_refs:
                config_keys = [ref.get("key", "") for ref in config_refs[:5]]
                parts.append(f"**Configuration:** {', '.join(filter(None, config_keys))}")

        return "\n".join(parts)


class FileSummarizer:
    """Backward-compatible file summarizer used by tests.

    This class now delegates to the main Summarizer to avoid code duplication,
    while maintaining the same API expected by tests.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model
        self.logger = get_logger(__name__)
        # Create a main summarizer instance for delegation
        config = TenetsConfig()
        self.summarizer = Summarizer(config)

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

        # Delegate to shared utilities for summary extraction
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
        language = CodeDetector.detect_language(text, str(file_path) if file_path else None)

        if language == "python":
            structure = ASTParser.extract_python_structure(text)
            if structure.get("docstrings"):
                doc = structure["docstrings"][0]  # Module docstring
                return self._limit_lines(doc, max_lines)

        # 2) Leading comment block extraction (supports Python/JS/TS)
        comment_block = ASTParser.extract_leading_comments(text)
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

    # Removed _extract_leading_comments - now using ASTParser.extract_leading_comments from shared utils

    def _extract_leading_comments_legacy(self, text: str) -> str:
        # This method is kept temporarily for backward compatibility but delegates to shared util
        return ASTParser.extract_leading_comments(text)

    def _extract_leading_comments_old_implementation(self, text: str) -> str:
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
