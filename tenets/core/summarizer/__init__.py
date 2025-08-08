"""Summarization package.

Provides both a lightweight file summarizer (FileSummarizer) and a
high-level Summarizer adapter used by other components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Union

from tenets.models.summary import FileSummary
from tenets.models.analysis import FileAnalysis
from tenets.utils.tokens import count_tokens
from tenets.utils.logger import get_logger

# Re-export the lightweight file summarizer
from .summarizer import FileSummarizer  # noqa: F401

__all__ = ["Summarizer", "FileSummarizer"]


class Summarizer:
    """High-level summarizer used by the distiller/aggregator.

    This adapter avoids optional ML dependencies and implements a
    simple heuristic summarization that fits content within a token
    budget, returning a FileSummary compatible with the aggregator.
    """

    def __init__(self, config: object):  # Accept any config shape
        self.config = config
        self.logger = get_logger(__name__)

    def summarize_file(
        self,
        *,
        file: Union[FileAnalysis, str, Path],
        max_tokens: int,
        preserve_sections: Optional[List[str]] = None,
    ) -> FileSummary:
        """Summarize a file or content to fit within max_tokens.

        Args:
            file: FileAnalysis object or path to a file
            max_tokens: Target maximum token budget for the summary
            preserve_sections: Names of sections to try to preserve (metadata)
        Returns:
            FileSummary instance with condensed content
        """
        # Resolve input to raw text/content
        path_str: Optional[str] = None
        language: Optional[str] = None

        if isinstance(file, FileAnalysis):
            text = file.content or ""
            path_str = getattr(file, "path", None)
            language = getattr(file, "language", None)
            original_lines = getattr(file, "lines", 0) or (text.count("\n") + 1 if text else 0)
        else:
            p = Path(file)
            path_str = str(p)
            text = self._read_text(p)
            original_lines = text.count("\n") + 1 if text else 0

        original_tokens = count_tokens(text, None)
        summary_text = self._truncate_to_tokens(text, max_tokens)
        summary_tokens = count_tokens(summary_text, None)

        # Build FileSummary used by aggregator
        summary = FileSummary(
            content=summary_text,
            was_summarized=True,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            original_lines=original_lines,
            summary_lines=summary_text.count("\n") + 1 if summary_text else 0,
            preserved_sections=preserve_sections or [],
            strategy="extract",
            file_path=path_str,
            metadata={"language": language} if language else {},
        )
        return summary

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            try:
                return path.read_text(encoding="latin-1")
            except Exception:
                self.logger.debug("Failed to read %s", path)
                return ""

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        # First approximate by characters, then refine by tokens
        approx_chars = max_tokens * 4  # rough chars/token
        snippet = text[: approx_chars + 500]  # small buffer before precise cut

        lines = snippet.splitlines()
        out_lines: List[str] = []
        for line in lines:
            candidate = ("\n".join(out_lines + [line])).rstrip()
            if count_tokens(candidate, None) > max_tokens:
                break
            out_lines.append(line)

        result = "\n".join(out_lines)
        if len(result) < len(text):
            result += "\n... [truncated]"
        return result
