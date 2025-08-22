"""Tokenization utilities for code and text.

This module provides tokenizers that understand programming language
constructs and can handle camelCase, snake_case, and other patterns.
"""

import re
from typing import List, Optional, Set
from pathlib import Path

from tenets.utils.logger import get_logger


class CodeTokenizer:
    """Tokenizer optimized for source code.

    Handles:
    - camelCase and PascalCase splitting
    - snake_case splitting
    - Preserves original tokens for exact matching
    - Language-specific keywords
    - Optional stopword filtering
    """

    def __init__(self, use_stopwords: bool = False):
        """Initialize code tokenizer.

        Args:
            use_stopwords: Whether to filter stopwords
        """
        self.logger = get_logger(__name__)
        self.use_stopwords = use_stopwords

        if use_stopwords:
            from .stopwords import StopwordManager

            self.stopwords = StopwordManager().get_set("code")
        else:
            self.stopwords = None

        # Patterns for tokenization
        self.token_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
        self.camel_case_pattern = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)")
        self.snake_case_pattern = re.compile(r"[a-z]+|[A-Z]+")

    def tokenize(
        self, text: str, language: Optional[str] = None, preserve_original: bool = True
    ) -> List[str]:
        """Tokenize code text.

        Args:
            text: Code to tokenize
            language: Programming language (for language-specific handling)
            preserve_original: Keep original tokens alongside splits

        Returns:
            List of tokens
        """
        if not text:
            return []

        tokens = []
        raw_tokens = self.token_pattern.findall(text)

        for token in raw_tokens:
            # Skip single chars except important ones
            if len(token) == 1 and token.lower() not in {"i", "a", "x", "y", "z"}:
                continue

            token_parts = []

            # Handle camelCase/PascalCase
            if any(c.isupper() for c in token) and not token.isupper():
                parts = self.camel_case_pattern.findall(token)
                token_parts.extend(p.lower() for p in parts if len(p) > 1)
                if preserve_original:
                    token_parts.append(token.lower())

            # Handle snake_case
            elif "_" in token:
                parts = token.split("_")
                token_parts.extend(p.lower() for p in parts if p and len(p) > 1)
                if preserve_original:
                    token_parts.append(token.lower())

            else:
                # Regular token
                token_parts.append(token.lower())

            tokens.extend(token_parts)

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        # Filter stopwords if enabled
        if self.use_stopwords and self.stopwords:
            unique_tokens = [t for t in unique_tokens if t not in self.stopwords.words]

        return unique_tokens

    def tokenize_identifier(self, identifier: str) -> List[str]:
        """Tokenize a single identifier (function/class/variable name).

        Args:
            identifier: Identifier to tokenize

        Returns:
            List of component tokens
        """
        tokens = []

        # camelCase/PascalCase
        if any(c.isupper() for c in identifier) and not identifier.isupper():
            tokens = [p.lower() for p in self.camel_case_pattern.findall(identifier)]

        # snake_case
        elif "_" in identifier:
            tokens = [p.lower() for p in identifier.split("_") if p]

        # SCREAMING_SNAKE_CASE
        elif identifier.isupper() and "_" in identifier:
            tokens = [p.lower() for p in identifier.split("_") if p]

        else:
            tokens = [identifier.lower()]

        return [t for t in tokens if len(t) > 1]


class TextTokenizer:
    """Tokenizer for natural language text (prompts, comments, docs).

    More aggressive than CodeTokenizer, designed for understanding
    user intent rather than exact matching.
    """

    def __init__(self, use_stopwords: bool = True):
        """Initialize text tokenizer.

        Args:
            use_stopwords: Whether to filter stopwords (default True)
        """
        self.logger = get_logger(__name__)
        self.use_stopwords = use_stopwords

        if use_stopwords:
            from .stopwords import StopwordManager

            self.stopwords = StopwordManager().get_set("prompt")
        else:
            self.stopwords = None

        # More permissive pattern for natural language
        self.token_pattern = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*\b")

    def tokenize(self, text: str, min_length: int = 2) -> List[str]:
        """Tokenize natural language text.

        Args:
            text: Text to tokenize
            min_length: Minimum token length

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Extract tokens
        tokens = self.token_pattern.findall(text.lower())

        # Filter by length
        tokens = [t for t in tokens if len(t) >= min_length]

        # Filter stopwords
        if self.use_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords.words]

        return tokens

    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text.

        Args:
            text: Input text
            n: Size of n-grams

        Returns:
            List of n-grams
        """
        tokens = self.tokenize(text)

        if len(tokens) < n:
            return []

        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams.append(ngram)

        return ngrams
