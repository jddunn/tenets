"""Content transformation utilities for distillation.

Provides reusable helpers for optional modes:
- full mode (handled outside here)
- remove-comments
- condense whitespace

The functions here are intentionally conservative: they aim to reduce
noise and token usage without breaking code structure. Comment stripping
is heuristic and language-aware at a shallow level; if an operation would
remove an excessive proportion of non-empty lines (>60%), the original
content is returned to avoid accidental destruction of meaning.

"""

from __future__ import annotations

import re
from typing import Tuple

COMMENT_SYNTAX = {
    # language: (line_comment_markers, block_comment_patterns)
    "python": (["#"], [('"""', '"""'), ("'''", "'''")]),
    "javascript": (["//"], [("/*", "*/")]),
    "typescript": (["//"], [("/*", "*/")]),
    "java": (["//"], [("/*", "*/")]),
    "c": (["//"], [("/*", "*/")]),
    "cpp": (["//"], [("/*", "*/")]),
    "csharp": (["//"], [("/*", "*/")]),
    "go": (["//"], [("/*", "*/")]),
    "rust": (["//"], [("/*", "*/")]),
    "php": (["//", "#"], [("/*", "*/")]),
    "ruby": (["#"], []),
    "shell": (["#"], []),
    "bash": (["#"], []),
    "sql": (["--"], [("/*", "*/")]),
    "kotlin": (["//"], [("/*", "*/")]),
    "scala": (["//"], [("/*", "*/")]),
    "swift": (["//"], [("/*", "*/")]),
    "haskell": (["--"], [("{-", "-}")]),
    "lua": (["--"], [("--[[", "]]--")]),
}

WHITESPACE_RE = re.compile(r"\n{3,}")
TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def detect_language_from_extension(path: str) -> str:
    """Best-effort language detection from file extension.

    Args:
        path: File path.
    Returns:
        Lowercase language key used in COMMENT_SYNTAX or empty string.
    """
    ext = path.lower().rsplit(".", 1)
    if len(ext) == 2:
        ext = ext[1]
    else:
        return ""
    mapping = {
        "py": "python",
        "pyw": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
        "java": "java",
        "c": "c",
        "cc": "cpp",
        "cpp": "cpp",
        "cs": "csharp",
        "go": "go",
        "rs": "rust",
        "php": "php",
        "rb": "ruby",
        "sh": "shell",
        "bash": "bash",
        "sql": "sql",
        "kt": "kotlin",
        "kts": "kotlin",
        "scala": "scala",
        "swift": "swift",
        "hs": "haskell",
        "lua": "lua",
    }
    return mapping.get(ext, "")


def strip_comments(content: str, language: str) -> str:
    """Strip comments from source content.

    Heuristic removal; skips removal if more than 60% of non-empty lines
    would disappear.

    Args:
        content: Original file content.
        language: Detected language key.
    Returns:
        Content with comments removed (or original on safeguard trigger).
    """
    if not content or not language:
        return content
    syntax = COMMENT_SYNTAX.get(language)
    if not syntax:
        return content
    line_markers, block_pairs = syntax

    lines = content.splitlines()
    non_empty_before = sum(1 for l in lines if l.strip())

    # Helper: remove inline comments while preserving strings
    def _strip_inline(line: str) -> str:
        in_single = False
        in_double = False
        escaped = False
        i = 0
        while i < len(line):
            ch = line[i]
            # Toggle string states
            if not escaped and ch == '"' and not in_single:
                in_double = not in_double
            elif not escaped and ch == "'" and not in_double:
                in_single = not in_single
            # Handle escapes within strings
            escaped = (ch == "\\") and (in_single or in_double) and not escaped

            if not in_single and not in_double:
                for marker in line_markers:
                    if line.startswith(marker, i):
                        # Check if only whitespace before marker (full-line comment)
                        if line[:i].strip() == "":
                            return line[:i]
                        else:
                            # Inline comment: keep code before marker
                            return line[:i].rstrip()
            i += 1
        return line

    stripped_lines = [_strip_inline(l) for l in lines]

    text = "\n".join(stripped_lines)

    # Remove block comments with simple loop
    for start, end in block_pairs:
        # Non-greedy to avoid spanning across code; iterative removal
        pattern = re.compile(re.escape(start) + r"[\s\S]*?" + re.escape(end))
        text = pattern.sub("", text)

    # Safeguard
    non_empty_after = sum(1 for l in text.splitlines() if l.strip())
    if non_empty_before and non_empty_after / non_empty_before < 0.4:
        return content  # Too destructive
    return text


def condense_whitespace(content: str) -> str:
    """Condense extraneous whitespace while preserving code structure.

    Operations:
      * Collapse runs of >=3 blank lines to a single blank line.
      * Trim trailing spaces.
      * Ensure single final newline.

    Args:
        content: File content.
    Returns:
        Condensed content.
    """
    if not content:
        return content
    text = TRAILING_SPACE_RE.sub("", content)
    text = WHITESPACE_RE.sub("\n\n", text)
    if not text.endswith("\n"):
        text += "\n"
    return text


def apply_transformations(
    content: str, language: str, *, remove_comments: bool, condense: bool
) -> Tuple[str, dict]:
    """Apply selected transformations.

    Args:
        content: Original content.
        language: Language key.
        remove_comments: Whether to strip comments.
        condense: Whether to condense whitespace.
    Returns:
        Tuple of (transformed_content, stats_dict).
    """
    stats = {"removed_comment_lines": 0, "condensed_blank_runs": 0}
    original = content
    if remove_comments:
        before_lines = [l for l in content.splitlines() if l.strip()]
        content = strip_comments(content, language)
        after_lines = [l for l in content.splitlines() if l.strip()]
        stats["removed_comment_lines"] = max(0, len(before_lines) - len(after_lines))
    if condense:
        blank_runs_before = content.count("\n\n\n")
        content = condense_whitespace(content)
        blank_runs_after = content.count("\n\n\n")
        stats["condensed_blank_runs"] = max(0, blank_runs_before - blank_runs_after)
    stats["changed"] = content != original
    return content, stats
