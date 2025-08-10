"""Tests for token utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from tenets.utils.tokens import (
    count_tokens,
    get_model_max_tokens,
    chunk_text,
    _get_encoding_for_model,
    _HAS_TIKTOKEN,
)


class TestTokenCounting:
    """Test suite for token counting functionality."""

    def test_count_tokens_empty(self):
        """Test counting tokens in empty string."""
        assert count_tokens("") == 0
        assert count_tokens("", model="gpt-4") == 0

    def test_count_tokens_heuristic(self):
        """Test token counting with heuristic (no tiktoken)."""
        # Mock tiktoken not being available
        with patch("tenets.utils.tokens._HAS_TIKTOKEN", False):
            # Heuristic is ~4 chars per token
            text = "This is a test string with some words."
            token_count = count_tokens(text)

            # Should be approximately len(text) / 4
            expected = max(1, len(text) // 4)
            assert token_count == expected

    def test_count_tokens_short_text(self):
        """Test token counting with short text."""
        text = "Hello"
        count = count_tokens(text)

        assert count > 0
        # Short text should be at least 1 token
        assert count >= 1

    @pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_count_tokens_with_tiktoken(self):
        """Test token counting with tiktoken."""
        text = "The quick brown fox jumps over the lazy dog."

        # Test with different models
        count_gpt4 = count_tokens(text, model="gpt-4")
        count_gpt4o = count_tokens(text, model="gpt-4o")

        assert count_gpt4 > 0
        assert count_gpt4o > 0

        # Both should give reasonable counts
        assert 5 <= count_gpt4 <= 20
        assert 5 <= count_gpt4o <= 20

    @pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_count_tokens_model_specific(self):
        """Test model-specific token counting."""
        text = "Test text for tokenization"

        # Different models might have different tokenizations
        models = ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

        counts = {}
        for model in models:
            counts[model] = count_tokens(text, model=model)

        # All should produce counts
        for model, count in counts.items():
            assert count > 0

    def test_count_tokens_unicode(self):
        """Test token counting with unicode text."""
        texts = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "🚀 Emoji text 🎉",  # Emojis
            "Ñoño",  # Spanish
        ]

        for text in texts:
            count = count_tokens(text)
            assert count > 0

    def test_count_tokens_code(self):
        """Test token counting with code."""
        code = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """

        count = count_tokens(code)
        assert count > 10  # Code should have reasonable token count

    @pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
    def test_get_encoding_for_model(self):
        """Test getting encoding for different models."""
        # Known models
        enc = _get_encoding_for_model("gpt-4")
        assert enc is not None

        enc = _get_encoding_for_model("gpt-4o")
        assert enc is not None

        # Unknown model should fall back to cl100k_base
        enc = _get_encoding_for_model("unknown-model")
        assert enc is not None

        # No model should still return an encoding
        enc = _get_encoding_for_model(None)
        assert enc is not None

    def test_get_encoding_error_handling(self):
        """Test encoding error handling."""
        with patch("tenets.utils.tokens._HAS_TIKTOKEN", True):
            with patch("tiktoken.get_encoding", side_effect=Exception("Encoding error")):
                enc = _get_encoding_for_model("gpt-4")
                assert enc is None

                # Should fall back to heuristic
                count = count_tokens("test text")
                assert count > 0


class TestModelMaxTokens:
    """Test suite for model max tokens functionality."""

    def test_get_model_max_tokens_known_models(self):
        """Test getting max tokens for known models."""
        test_cases = [
            ("gpt-4", 8_192),
            ("gpt-4.1", 128_000),
            ("gpt-4o", 128_000),
            ("gpt-4o-mini", 128_000),
            ("gpt-3.5-turbo", 16_385),
            ("claude-3-opus", 200_000),
            ("claude-3-5-sonnet", 200_000),
            ("claude-3-haiku", 200_000),
        ]

        for model, expected in test_cases:
            assert get_model_max_tokens(model) == expected

    def test_get_model_max_tokens_unknown_model(self):
        """Test getting max tokens for unknown model."""
        assert get_model_max_tokens("unknown-model") == 100_000
        assert get_model_max_tokens("gpt-5-future") == 100_000

    def test_get_model_max_tokens_none(self):
        """Test getting max tokens with None model."""
        assert get_model_max_tokens(None) == 100_000


class TestTextChunking:
    """Test suite for text chunking functionality."""

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", max_tokens=100)
        assert chunks == [""]

    def test_chunk_text_small(self):
        """Test chunking text smaller than limit."""
        text = "This is a small text."
        chunks = chunk_text(text, max_tokens=100)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_multiline(self):
        """Test chunking multiline text."""
        text = """Line 1
Line 2
Line 3
Line 4
Line 5"""

        # Force small chunks
        chunks = chunk_text(text, max_tokens=5)

        assert len(chunks) > 1
        # Chunks should be line-aware
        for chunk in chunks:
            # Each chunk should contain complete lines
            assert chunk.strip() != ""

    def test_chunk_text_preserves_lines(self):
        """Test that chunking preserves line boundaries."""
        lines = [f"This is line number {i} with some content" for i in range(20)]
        text = "\n".join(lines)

        chunks = chunk_text(text, max_tokens=50)

        # Reconstruct text from chunks
        reconstructed = "".join(chunks)

        # Should preserve all content
        assert reconstructed == text

        # Each chunk should end at line boundaries (except possibly the last)
        for chunk in chunks[:-1]:
            if chunk and not chunk.endswith("\n"):
                # If no newline at end, it should be the last chunk
                assert chunk == chunks[-1]

    def test_chunk_text_with_model(self):
        """Test chunking with specific model."""
        text = "Test text " * 100  # Repeat to make longer

        chunks = chunk_text(text, max_tokens=50, model="gpt-4")

        assert len(chunks) > 1

        # Each chunk should be within token limit
        for chunk in chunks:
            token_count = count_tokens(chunk, model="gpt-4")
            # Allow some flexibility for edge cases
            assert token_count <= 60  # Small buffer for boundary issues

    def test_chunk_text_zero_max_tokens(self):
        """Test chunking with zero or negative max tokens."""
        text = "Some text content"

        chunks = chunk_text(text, max_tokens=0)
        assert chunks == [text]

        chunks = chunk_text(text, max_tokens=-10)
        assert chunks == [text]

    def test_chunk_text_single_long_line(self):
        """Test chunking a single very long line."""
        # Single line that's too long for one chunk
        text = "word " * 1000  # No newlines

        chunks = chunk_text(text, max_tokens=100)

        # Should still chunk even without line breaks
        assert len(chunks) > 1

        # Reconstruct should preserve content
        assert "".join(chunks) == text

    def test_chunk_text_code(self):
        """Test chunking code with proper line preservation."""
        code = """
def example_function(param1, param2):
    '''This is a docstring.'''
    result = param1 + param2
    
    if result > 100:
        print("Large result")
        return result * 2
    else:
        print("Small result")
        return result
        
    # This should not be reached
    raise ValueError("Unexpected")

class ExampleClass:
    def __init__(self):
        self.value = 42
        
    def method(self):
        return self.value * 2
"""

        chunks = chunk_text(code, max_tokens=50)

        # Should preserve code structure
        reconstructed = "".join(chunks)
        assert reconstructed == code

        # Each chunk should be valid code lines
        for chunk in chunks:
            lines = chunk.split("\n")
            # Check indentation is preserved
            for line in lines:
                if line.strip():  # Non-empty lines
                    # Indentation should be spaces or nothing
                    assert line[0] in " #'\"def class" or line[0].isalpha()

    def test_chunk_text_mixed_content(self):
        """Test chunking mixed content (prose and code)."""
        content = (
            """
# Documentation

This is some documentation about the following code:

```python
def hello():
    print("Hello, world!")
```

And here's more text after the code block.
"""
            * 10
        )  # Repeat to make longer

        chunks = chunk_text(content, max_tokens=100)

        assert len(chunks) > 1

        # Content should be preserved
        assert "".join(chunks) == content

    def test_chunk_text_boundary_cases(self):
        """Test chunk boundaries with exact token limits."""
        # Create text with known token count
        # Assuming ~4 chars per token for heuristic
        text_10_tokens = "x" * 40  # ~10 tokens
        text_20_tokens = "x" * 80  # ~20 tokens

        # Chunk at exact boundary
        chunks = chunk_text(text_10_tokens, max_tokens=10)
        # Should fit in one chunk
        assert len(chunks) == 1

        # Just over boundary
        chunks = chunk_text(text_20_tokens, max_tokens=10)
        # Should need two chunks
        assert len(chunks) >= 2

    def test_chunk_text_unicode_handling(self):
        """Test chunking with unicode characters."""
        text = (
            """
        English text
        中文文本
        Текст на русском
        日本語のテキスト
        한국어 텍스트
        عربي نص
        🚀 Emoji line 🎉
        """
            * 5
        )

        chunks = chunk_text(text, max_tokens=50)

        # Should handle unicode properly
        reconstructed = "".join(chunks)
        assert reconstructed == text

        # No corruption of unicode
        assert "中文文本" in reconstructed
        assert "🚀" in reconstructed
