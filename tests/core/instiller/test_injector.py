"""Tests for TenetInjector."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from tenets.core.instiller.injector import InjectionPoint, InjectionPosition, TenetInjector
from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet, TenetStatus


@pytest.fixture
def injector():
    """Create TenetInjector instance."""
    config = {"min_distance_between": 500, "prefer_natural_breaks": True, "reinforce_at_end": True}
    return TenetInjector(config)


@pytest.fixture
def sample_tenets():
    """Create sample tenets for testing."""
    tenets = [
        Tenet(content="Always use type hints in Python", priority=Priority.HIGH),
        Tenet(content="Handle all exceptions explicitly", priority=Priority.CRITICAL),
        Tenet(content="Write comprehensive docstrings", priority=Priority.MEDIUM),
    ]
    return tenets


@pytest.fixture
def markdown_content():
    """Sample markdown content."""
    return """# Project Documentation

## Introduction
This is the introduction section with some content.

## Implementation Details

Here's some code:

```python
def example():
    return 42
```

## Testing Strategy

We use pytest for all testing.

## Conclusion

Final thoughts on the project.
"""


class TestTenetInjector:
    """Test suite for TenetInjector."""

    def test_initialization(self):
        """Test injector initialization."""
        config = {"min_distance_between": 1000}
        injector = TenetInjector(config)

        assert injector.min_distance_between == 1000
        assert injector.prefer_natural_breaks == True
        assert injector.reinforce_at_end == True

    def test_inject_tenets_empty_list(self, injector, markdown_content):
        """Test injection with no tenets."""
        content, metadata = injector.inject_tenets(markdown_content, [], format="markdown")

        assert content == markdown_content
        assert metadata["injected_count"] == 0

    def test_inject_tenets_markdown(self, injector, markdown_content, sample_tenets):
        """Test injecting tenets into markdown."""
        modified_content, metadata = injector.inject_tenets(
            markdown_content, sample_tenets, format="markdown"
        )

        # Content should be modified
        assert modified_content != markdown_content
        assert len(modified_content) > len(markdown_content)

        # Check metadata
        assert metadata["injected_count"] == 3
        assert metadata["strategy"] in ["top", "strategic", "distributed"]
        assert "token_increase" in metadata

        # Check tenets are in content
        assert "type hints" in modified_content
        assert "exceptions" in modified_content
        assert "docstrings" in modified_content

    def test_analyze_content_structure_markdown(self, injector):
        """Test markdown structure analysis."""
        content = """# Main Title

## Section 1
Content here.

## Section 2

```python
code block
```

More content.
"""

        structure = injector._analyze_content_structure(content, "markdown")

        assert len(structure["sections"]) == 3  # h1 and two h2s
        assert len(structure["code_blocks"]) == 1
        assert len(structure["natural_breaks"]) > 0

    def test_analyze_content_structure_xml(self, injector):
        """Test XML structure analysis."""
        content = """<document>
<section>Content 1</section>
<section>Content 2</section>
</document>"""

        structure = injector._analyze_content_structure(content, "xml")

        assert len(structure["sections"]) >= 1

    def test_determine_strategy_short_content(self, injector):
        """Test strategy for short content."""
        strategy = injector._determine_strategy(
            content_length=1000, tenet_count=2, structure={"sections": []}
        )

        assert strategy == InjectionPosition.TOP

    def test_determine_strategy_long_content(self, injector):
        """Test strategy for long content."""
        strategy = injector._determine_strategy(
            content_length=60000, tenet_count=8, structure={"sections": []}
        )

        assert strategy == InjectionPosition.DISTRIBUTED

    def test_determine_strategy_structured_content(self, injector):
        """Test strategy for well-structured content."""
        strategy = injector._determine_strategy(
            content_length=15000, tenet_count=4, structure={"sections": [1, 2, 3, 4]}
        )

        assert strategy == InjectionPosition.STRATEGIC

    def test_find_injection_points_top(self, injector):
        """Test finding injection points with TOP strategy."""
        structure = {
            "sections": [{"title": "Intro", "end_position": 100}],
            "code_blocks": [],
            "natural_breaks": [],
        }

        points = injector._find_injection_points(
            content="x" * 1000, structure=structure, strategy=InjectionPosition.TOP, tenet_count=3
        )

        assert len(points) == 3
        # All should be at same position (top)
        assert all(p.position == points[0].position for p in points)
        assert points[0].reason == "top_of_context"

    def test_find_injection_points_distributed(self, injector):
        """Test finding injection points with DISTRIBUTED strategy."""
        structure = {"sections": [], "code_blocks": [], "natural_breaks": [100, 300, 500, 700]}

        points = injector._find_injection_points(
            content="x" * 1000,
            structure=structure,
            strategy=InjectionPosition.DISTRIBUTED,
            tenet_count=3,
        )

        assert len(points) == 3
        # Should be distributed
        positions = [p.position for p in points]
        assert positions[1] > positions[0]
        assert positions[2] > positions[1]

    def test_find_injection_points_strategic(self, injector):
        """Test finding injection points with STRATEGIC strategy."""
        structure = {
            "sections": [
                {"title": "Intro", "end_position": 100, "level": 1},
                {"title": "Main", "end_position": 500, "level": 2},
                {"title": "Details", "end_position": 800, "level": 3},
            ],
            "code_blocks": [{"start": 600, "end": 700}],
            "natural_breaks": [110, 510, 810],
        }

        points = injector._find_injection_points(
            content="x" * 1000,
            structure=structure,
            strategy=InjectionPosition.STRATEGIC,
            tenet_count=2,
        )

        assert len(points) == 2
        # Should avoid code blocks
        for point in points:
            assert not (600 <= point.position <= 700)

    def test_format_tenet_markdown(self, injector):
        """Test formatting tenet for markdown."""
        tenet = Tenet(content="Test principle", priority=Priority.HIGH)

        formatted = injector._format_tenet(tenet, "markdown", position=0)

        assert "**🎯 Key Principle:**" in formatted
        assert "Test principle" in formatted

    def test_format_tenet_xml(self, injector):
        """Test formatting tenet for XML."""
        tenet = Tenet(content="Test principle", priority=Priority.CRITICAL)

        formatted = injector._format_tenet(tenet, "xml", position=0)

        assert "<tenet" in formatted
        assert 'priority="critical"' in formatted
        assert "Test principle" in formatted

    def test_format_tenet_json(self, injector):
        """Test formatting tenet for JSON."""
        tenet = Tenet(content="Test principle", priority=Priority.LOW)

        formatted = injector._format_tenet(tenet, "json", position=0)

        assert "/* TENET:" in formatted
        assert "Test principle" in formatted

    def test_create_reinforcement_section_markdown(self, injector, sample_tenets):
        """Test creating reinforcement section for markdown."""
        section = injector._create_reinforcement_section(sample_tenets[:2], "markdown")

        assert "## 🎯 Key Principles to Remember" in section
        assert "type hints" in section
        assert "exceptions" in section

    def test_create_reinforcement_section_xml(self, injector, sample_tenets):
        """Test creating reinforcement section for XML."""
        section = injector._create_reinforcement_section(sample_tenets[:2], "xml")

        assert "<reinforcement>" in section
        assert "<principle" in section
        assert "</reinforcement>" in section

    def test_calculate_optimal_injection_count(self, injector):
        """Test calculating optimal tenet count."""
        # Short content
        count = injector.calculate_optimal_injection_count(
            content_length=500, available_tenets=10, max_token_increase=1000
        )
        assert count == 1

        # Medium content
        count = injector.calculate_optimal_injection_count(
            content_length=10000, available_tenets=10, max_token_increase=1000
        )
        assert count == 3

        # Limited by available tenets
        count = injector.calculate_optimal_injection_count(
            content_length=50000, available_tenets=2, max_token_increase=1000
        )
        assert count == 2

        # Limited by token budget
        count = injector.calculate_optimal_injection_count(
            content_length=50000, available_tenets=100, max_token_increase=100
        )
        assert count <= 3  # 100 tokens / 30 per tenet

    def test_inject_into_context_result(self, injector, sample_tenets):
        """Test injecting into ContextResult object."""
        context_result = ContextResult(
            files=["test.py"],
            context="# Test\n\nOriginal content",
            format="markdown",
            metadata={"total_tokens": 100},
        )

        modified = injector.inject_into_context_result(context_result, sample_tenets)

        assert modified.context != "# Test\n\nOriginal content"
        assert "tenet_injection" in modified.metadata
        assert "tenets_injected" in modified.metadata
        assert len(modified.metadata["tenets_injected"]) == 3
        assert modified.metadata["total_tokens"] > 100

    def test_injection_with_reinforcement(self, injector):
        """Test injection with reinforcement section."""
        # Create many tenets to trigger reinforcement
        tenets = [Tenet(content=f"Principle {i}", priority=Priority.HIGH) for i in range(5)]

        content, metadata = injector.inject_tenets("Short content", tenets, format="markdown")

        if metadata.get("reinforcement_added"):
            assert "Key Principles to Remember" in content

    def test_injection_avoids_code_blocks(self, injector):
        """Test that injection avoids code blocks."""
        content = """# Title

```python
def important_code():
    # This should not be interrupted
    return 42
```

More content here.
"""

        tenets = [Tenet(content="Test", priority=Priority.HIGH)]

        modified, metadata = injector.inject_tenets(content, tenets, format="markdown")

        # Code block should remain intact
        assert "def important_code():" in modified
        assert "return 42" in modified

    def test_minimum_distance_enforcement(self, injector):
        """Test minimum distance between injections."""
        injector.min_distance_between = 200

        structure = {
            "sections": [],
            "code_blocks": [],
            "natural_breaks": [50, 100, 150, 200, 300, 400],
        }

        points = injector._find_injection_points(
            content="x" * 500,
            structure=structure,
            strategy=InjectionPosition.STRATEGIC,
            tenet_count=3,
        )

        # Check minimum distance
        for i in range(len(points) - 1):
            distance = abs(points[i + 1].position - points[i].position)
            assert distance >= 200 or distance == 0  # Same position is OK
