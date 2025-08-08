"""Lightweight file summarizer (non-ML).

Heuristic summarization suitable when ML deps are not installed. Extracts
module docstrings, top comments, or the first N lines.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from tenets.models.summary import FileSummary
from tenets.utils.tokens import count_tokens
from tenets.utils.logger import get_logger

logger = get_logger(__name__)


class FileSummarizer:
    def __init__(self, model: Optional[str] = None):
        self.model = model

    def summarize_file(self, path: Path, max_lines: int = 150) -> FileSummary:
        text = self._read_text(path)
        summary = self._extract_summary(text, max_lines=max_lines)
        return FileSummary(
            path=str(path),
            summary=summary,
            token_count=count_tokens(summary, self.model),
            metadata={"strategy": "heuristic", "max_lines": max_lines},
        )

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            try:
                return path.read_text(encoding="latin-1")
            except Exception:
                logger.debug("Failed to read %s", path)
                return ""

    def _extract_summary(self, text: str, max_lines: int) -> str:
        if not text:
            return ""
        # Try Python docstring when applicable
        try:
            tree = ast.parse(text)
            doc = ast.get_docstring(tree)
            if doc:
                return doc.strip()
        except Exception:
            pass
        # Fallback: leading comments or head of file
        lines = text.splitlines()
        if not lines:
            return ""
        # Capture leading comment block
        i = 0
        while i < len(lines) and (lines[i].strip().startswith("#") or not lines[i].strip()):
            i += 1
        if i > 0:
            comment_block = "\n".join(line.lstrip("# ") for line in lines[:i]).strip()
            if comment_block:
                return comment_block
        # Head snippet
        return "\n".join(lines[:max_lines]).strip()
