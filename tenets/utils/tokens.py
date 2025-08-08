"""Token utilities.

Lightweight helpers for token counting and text chunking used across the
project. When available, this module uses the optional `tiktoken` package
for accurate tokenization. If `tiktoken` is not installed, a conservative
heuristic (~4 characters per token) is used instead.

Notes:
- This module is dependency-light by design. `tiktoken` is optional.
- The fallback heuristic intentionally overestimates in some cases to
  keep chunk sizes well under model limits.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)

try:  # Optional dependency
    import tiktoken  # type: ignore

    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAS_TIKTOKEN = False

_MODEL_TO_ENCODING = {
    # OpenAI families (best-effort mappings)
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


def _get_encoding_for_model(model: Optional[str]):
    """Return a tiktoken encoding for the given model, if possible.

    Args:
        model: Optional model identifier (e.g., "gpt-4o", "gpt-4.1"). Used to
            select the most appropriate `tiktoken` encoding.

    Returns:
        A `tiktoken.Encoding` instance when `tiktoken` is available and an
        encoding can be resolved. Returns None if `tiktoken` is not installed
        or an encoding cannot be determined.

    Notes:
        - Falls back to `cl100k_base` if the model is unknown.
        - Returns None on any exception to allow graceful degradation to the
          heuristic path.
    """
    if not _HAS_TIKTOKEN:
        return None
    try:
        if model:
            enc_name = _MODEL_TO_ENCODING.get(model)
            if enc_name:
                return tiktoken.get_encoding(enc_name)
        # Fallback if model not mapped
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Approximate the number of tokens in a string.

    Uses `tiktoken` for accurate counts when available; otherwise falls back
    to a simple heuristic (~4 characters per token).

    Args:
        text: Input text to tokenize.
        model: Optional model name used to select an appropriate tokenizer
            (only relevant when `tiktoken` is available).

    Returns:
        Approximate number of tokens in ``text``.

    Examples:
        >>> count_tokens("hello world") > 0
        True
    """
    if not text:
        return 0

    enc = _get_encoding_for_model(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            # Fall through to heuristic on any failure
            pass

    # Fallback heuristic: ~4 chars per token
    return max(1, int(len(text) / 4))


def get_model_max_tokens(model: Optional[str]) -> int:
    """Return a conservative maximum context size (in tokens) for a model.

    This is a best-effort mapping that may lag behind provider updates. Values
    are deliberately conservative to avoid overruns when accounting for prompts,
    system messages, and tool outputs.

    Args:
        model: Optional model name. If None or unknown, a safe default is used.

    Returns:
        Maximum supported tokens for the given model, or a default of 100,000
        when the model is unspecified/unknown.
    """
    default = 100_000
    if not model:
        return default
    table = {
        "gpt-4": 8_192,
        "gpt-4.1": 128_000,
        "gpt-4o": 128_000,
        "gpt-4o-mini": 128_000,
        "gpt-3.5-turbo": 16_385,
        "claude-3-opus": 200_000,
        "claude-3-5-sonnet": 200_000,
        "claude-3-haiku": 200_000,
    }
    return table.get(model, default)


def chunk_text(text: str, max_tokens: int, model: Optional[str] = None) -> List[str]:
    """Split text into chunks whose token counts do not exceed ``max_tokens``.

    Chunking is line-aware: the input is split on line boundaries and lines are
    accumulated until the next line would exceed ``max_tokens``. This preserves
    readability and structure for code or prose.

    Args:
        text: The input text to split.
        max_tokens: Maximum tokens per chunk. If <= 0, returns the original
            ``text`` as a single-element list.
        model: Optional model name used for token counting (relevant only when
            `tiktoken` is available).

    Returns:
        A list of text chunks, each approximately within the ``max_tokens``
        budget.

    Examples:
        >>> chunks = chunk_text("line1\nline2\nline3", max_tokens=10)
        >>> isinstance(chunks, list)
        True
    """
    if max_tokens <= 0:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for line in lines:
        t = count_tokens(line, model)
        if current and current_tokens + t > max_tokens:
            chunks.append("".join(current))
            current = [line]
            current_tokens = t
        else:
            current.append(line)
            current_tokens += t

    if current:
        chunks.append("".join(current))

    if not chunks:
        return [text]
    return chunks
