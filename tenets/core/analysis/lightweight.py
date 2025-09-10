"""Lightweight file analysis for fast pre-ranking.

This module provides minimal file analysis capabilities optimized for speed.
It extracts just enough information for ranking without deep parsing or
language-specific analysis, dramatically improving performance for fast mode.

The lightweight analyzer trades accuracy for speed by:
- Reading only file headers and samples
- Using simple heuristics instead of AST parsing
- Deferring language detection until needed
- Caching aggressively
"""

import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from tenets.models.analysis import CodeStructure, FileAnalysis
from tenets.utils.logger import get_logger


@dataclass
class LightweightAnalysis:
    """Minimal file analysis data for ranking.

    This class contains just enough information to rank files effectively
    without the overhead of full language-specific analysis.

    Attributes:
        path: Path to the file
        size: File size in bytes
        extension: File extension (e.g., '.py', '.js')
        mime_type: MIME type if detectable
        content_sample: First N bytes of content for keyword matching
        line_count: Approximate number of lines
        has_tests: Quick heuristic check if file contains tests
        last_modified: Timestamp of last modification
        keywords: Set of prominent keywords found in sample
    """

    path: Path
    size: int
    extension: str
    mime_type: Optional[str]
    content_sample: str
    line_count: int
    has_tests: bool
    last_modified: float
    keywords: Set[str]

    def to_file_analysis(self) -> FileAnalysis:
        """Convert to full FileAnalysis object for compatibility.

        Creates a FileAnalysis object with minimal fields populated.
        This allows the lightweight analysis to be used with existing
        ranking infrastructure.

        Returns:
            FileAnalysis object with basic fields set
        """
        # Create a minimal CodeStructure if this is a test file
        structure = None
        if self.has_tests:
            structure = CodeStructure(
                classes=[],
                functions=[],
                imports=[],
                # exports field doesn't exist in CodeStructure
                variables=[],
                constants=[],
                # decorators, docstrings, comments fields don't exist
                is_test_file=True,  # Mark as test file
            )

        return FileAnalysis(
            path=str(self.path),
            content=self.content_sample,  # Use sample as content for ranking
            size=self.size,
            file_extension=self.extension,  # Use correct field name
            language=self._guess_language(),
            lines=self.line_count,  # Use 'lines' not 'line_count'
            file_name=self.path.name,  # Add file_name field
            last_modified=datetime.fromtimestamp(
                self.last_modified
            ),  # Convert timestamp to datetime
            # Leave complex fields empty for speed
            imports=[],
            functions=[],
            classes=[],
            structure=structure,  # Include structure if test file
            complexity=None,  # complexity is a ComplexityMetrics object
            keywords=list(self.keywords),
        )

    def _guess_language(self) -> str:
        """Quick language detection based on extension.

        Uses a simple mapping of extensions to language names.
        This avoids the overhead of initializing language analyzers.

        Returns:
            Language name or 'unknown'
        """
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "matlab",
            ".jl": "julia",
            ".dart": "dart",
            ".lua": "lua",
            ".pl": "perl",
            ".sh": "bash",
            ".ps1": "powershell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".md": "markdown",
            ".rst": "restructuredtext",
            ".tex": "latex",
            ".sql": "sql",
            ".proto": "protobuf",
            ".graphql": "graphql",
            ".dockerfile": "dockerfile",
            ".makefile": "makefile",
            ".cmake": "cmake",
        }
        return ext_to_lang.get(self.extension.lower(), "unknown")


class LightweightAnalyzer:
    """Fast file analyzer for pre-ranking phase.

    This analyzer performs minimal analysis to extract just enough
    information for effective ranking. It's designed to be 10-100x
    faster than full analysis by avoiding:
    - AST parsing
    - Complex pattern matching
    - Language-specific analysis
    - Deep content inspection

    The analyzer is perfect for fast mode where speed is critical
    and approximate ranking is acceptable.
    """

    # Configuration constants
    DEFAULT_SAMPLE_SIZE = 8192  # 8KB sample for keyword extraction
    MAX_SAMPLE_SIZE = 32768  # 32KB maximum to prevent memory issues
    KEYWORD_MIN_LENGTH = 3  # Minimum keyword length
    KEYWORD_MAX_COUNT = 50  # Maximum keywords to extract

    # Common test indicators for quick detection
    TEST_INDICATORS = {
        "test_",
        "_test.",
        "spec.",
        ".spec.",
        "tests/",
        "test/",
        "__tests__/",
        "specs/",
        "unittest",
        "pytest",
        "jest",
        "mocha",
        "describe(",
        "it(",
        "test(",
        "expect(",
    }

    def __init__(self, sample_size: int = DEFAULT_SAMPLE_SIZE):
        """Initialize the lightweight analyzer.

        Args:
            sample_size: Number of bytes to read for content sampling.
                        Larger samples improve accuracy but reduce speed.
        """
        self.logger = get_logger(__name__)
        self.sample_size = min(sample_size, self.MAX_SAMPLE_SIZE)
        self._cache: Dict[str, LightweightAnalysis] = {}

        self.logger.debug(f"LightweightAnalyzer initialized with {self.sample_size} byte samples")

    def analyze_file(self, file_path: Path) -> Optional[LightweightAnalysis]:
        """Perform lightweight analysis on a single file.

        This method extracts minimal information needed for ranking:
        - File metadata (size, extension, modification time)
        - Content sample for keyword matching
        - Quick heuristics (test detection, line count)

        The analysis is cached based on file path and modification time
        to avoid redundant work.

        Args:
            file_path: Path to the file to analyze

        Returns:
            LightweightAnalysis object or None if file cannot be read

        Note:
            Binary files are detected and skipped automatically.
            Large files have their samples truncated for speed.
        """
        # Check cache first
        cache_key = str(file_path)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            try:
                # Validate cache based on modification time
                current_mtime = os.path.getmtime(file_path)
                if cached.last_modified == current_mtime:
                    return cached
            except OSError:
                pass  # File might have been deleted

        try:
            # Get file metadata
            stat = os.stat(file_path)
            size = stat.st_size
            mtime = stat.st_mtime

            # Skip empty files
            if size == 0:
                return None

            # Extract extension and guess MIME type
            extension = file_path.suffix.lower()
            mime_type, _ = mimetypes.guess_type(str(file_path))

            # Skip binary files based on MIME type
            if mime_type and not mime_type.startswith("text/"):
                if not mime_type.startswith("application/"):
                    return None
                # Allow some application types that are text-based
                allowed_app_types = {
                    "application/json",
                    "application/xml",
                    "application/javascript",
                    "application/x-yaml",
                }
                if mime_type not in allowed_app_types:
                    return None

            # Read content sample
            sample_size = min(self.sample_size, size)
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content_sample = f.read(sample_size)

            # Quick binary detection on content
            if "\x00" in content_sample:
                return None  # Binary file

            # Extract basic metrics
            line_count = content_sample.count("\n")
            if size > sample_size:
                # Estimate total lines based on sample
                line_count = int(line_count * (size / sample_size))

            # Quick test detection
            # Check filename and immediate parent directory only
            filename_lower = file_path.name.lower()
            parent_dir_lower = file_path.parent.name.lower() if file_path.parent else ""
            content_lower = content_sample[:1000].lower()  # Check first 1KB

            # Check if any test indicator is in the filename, parent dir, or content
            has_tests = False
            for indicator in self.TEST_INDICATORS:
                # Skip path-based indicators for checking (they're for directory structure)
                if "/" in indicator:
                    # Check if this directory pattern is in the relative path
                    if indicator.rstrip("/") == parent_dir_lower:
                        has_tests = True
                        break
                # Check filename and content
                elif indicator in filename_lower or indicator in content_lower:
                    has_tests = True
                    break

            # Extract keywords (simple word extraction)
            keywords = self._extract_keywords(content_sample)

            # Create analysis object
            analysis = LightweightAnalysis(
                path=file_path,
                size=size,
                extension=extension,
                mime_type=mime_type,
                content_sample=content_sample,
                line_count=line_count,
                has_tests=has_tests,
                last_modified=mtime,
                keywords=keywords,
            )

            # Cache the result
            self._cache[cache_key] = analysis

            return analysis

        except Exception as e:
            self.logger.debug(f"Failed to analyze {file_path}: {e}")
            return None

    def analyze_files(self, file_paths: List[Path]) -> List[LightweightAnalysis]:
        """Analyze multiple files in batch.

        Processes files sequentially for simplicity. The lightweight
        nature of the analysis makes parallelization less critical.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            List of successful analyses (skips failures)
        """
        results = []
        for path in file_paths:
            analysis = self.analyze_file(path)
            if analysis:
                results.append(analysis)

        self.logger.info(
            f"Lightweight analysis complete: {len(results)}/{len(file_paths)} files analyzed"
        )
        return results

    def _extract_keywords(self, content: str) -> Set[str]:
        """Extract prominent keywords from content sample.

        Uses simple heuristics to identify important terms:
        - Alphanumeric words of sufficient length
        - Common programming terms
        - Identifiers that appear multiple times

        This is much faster than NLP-based extraction but less accurate.

        Args:
            content: Content sample to extract keywords from

        Returns:
            Set of keyword strings
        """
        import re
        from collections import Counter

        # Simple word extraction (alphanumeric + underscore)
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", content)

        # Filter and count
        word_counts = Counter(
            word.lower() for word in words if len(word) >= self.KEYWORD_MIN_LENGTH
        )

        # Skip very common programming keywords
        common_keywords = {
            "the",
            "and",
            "for",
            "if",
            "else",
            "return",
            "def",
            "class",
            "function",
            "var",
            "let",
            "const",
            "import",
            "from",
            "export",
            "public",
            "private",
            "static",
            "void",
            "int",
            "string",
            "bool",
            "true",
            "false",
            "null",
            "none",
            "self",
            "this",
            "new",
            "try",
            "catch",
            "throw",
            "async",
            "await",
            "with",
            "while",
            "break",
        }

        # Take top N keywords that aren't too common
        keywords = set()
        for word, count in word_counts.most_common(self.KEYWORD_MAX_COUNT * 2):
            if word not in common_keywords:
                keywords.add(word)
                if len(keywords) >= self.KEYWORD_MAX_COUNT:
                    break

        return keywords

    def clear_cache(self):
        """Clear the analysis cache.

        Useful when files have been modified externally or to free memory.
        """
        self._cache.clear()
        self.logger.debug("Lightweight analysis cache cleared")
