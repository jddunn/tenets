"""Tests for TenetInjector with system instruction support."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from tenets.core.instiller.injector import InjectionPoint, InjectionPosition, TenetInjector
from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet, TenetStatus


@pytest.fixture
def injector():
    """Create TenetInjector instance."""
    config = {
        "min_distance_between": 500,
        "prefer_natural_breaks": True,
        "reinforce_at_end": True,
    }
    return TenetInjector(config)


@pytest.fixture
def injector_with_system():
    """Create TenetInjector with system instruction config."""
    config = {
        "min_distance_between": 500,
        "prefer_natural_breaks": True,
        "reinforce_at_end": True,
        "system_instruction": "You are a helpful coding assistant.",
        "system_instruction_position": "top",
        "system_instruction_format": "markdown",
        "system_instruction_separator": "\n---\n\n",
        "system_instruction_label": "🎯 System Context",
    }
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
    
    def test_inject_with_system_instruction(self, injector_with_system, markdown_content):
        """Test that system instruction doesn't interfere with tenet injection."""
        tenets = [Tenet(content="Test tenet", priority=Priority.HIGH)]
        
        # Inject tenets (system instruction handled separately in Instiller)
        modified, metadata = injector_with_system.inject_tenets(
            markdown_content, tenets, format="markdown"
        )
        
        assert "Test tenet" in modified
        assert metadata["injected_count"] == 1
    
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


class TestInstillerSystemInstruction:
    """Test suite for system instruction feature in Instiller."""
    
    @pytest.fixture
    def instiller(self):
        """Create Instiller instance with system instruction config."""
        from tenets.core.instiller.instiller import Instiller
        from tenets.config import TenetsConfig
        
        config = TenetsConfig()
        return Instiller(config)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample ContextResult."""
        return ContextResult(
            files=["test.py"],
            context="# Test\n\nOriginal content",
            format="markdown",
            metadata={"total_tokens": 100},
        )
    
    def test_inject_system_instruction_top(self, instiller):
        """Test injecting system instruction at top."""
        instiller.config.tenet.system_instruction = "System context here"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.system_instruction_position = "top"
        
        content = "# Main Content\n\nActual content here."
        modified, metadata = instiller.inject_system_instruction(content, format="markdown")
        
        assert modified.startswith("## 🎯 System Context")
        assert "System context here" in modified
        assert "Main Content" in modified
        assert metadata["system_instruction_injected"] is True
        assert metadata["system_instruction_position"] == "top"
    
    def test_inject_system_instruction_after_header(self, instiller):
        """Test injecting system instruction after header."""
        instiller.config.tenet.system_instruction = "System context"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.system_instruction_position = "after_header"
        
        content = "# Main Title\n\nContent here."
        modified, metadata = instiller.inject_system_instruction(content, format="markdown")
        
        lines = modified.split('\n')
        assert lines[0] == "# Main Title"
        assert "System context" in modified
        assert modified.index("System context") > modified.index("Main Title")
    
    def test_inject_system_instruction_disabled(self, instiller):
        """Test system instruction when disabled."""
        instiller.config.tenet.system_instruction = "System context"
        instiller.config.tenet.system_instruction_enabled = False
        
        content = "Original content"
        modified, metadata = instiller.inject_system_instruction(content)
        
        assert modified == content
        assert metadata["system_instruction_injected"] is False
    
    def test_inject_system_instruction_once_per_session(self, instiller):
        """Test system instruction only injected once per session."""
        instiller.config.tenet.system_instruction = "System context"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.system_instruction_once_per_session = True
        
        # First injection
        content1 = "Content 1"
        modified1, metadata1 = instiller.inject_system_instruction(
            content1, session="test-session"
        )
        assert metadata1["system_instruction_injected"] is True
        
        # Second injection - should skip
        content2 = "Content 2"
        modified2, metadata2 = instiller.inject_system_instruction(
            content2, session="test-session"
        )
        assert metadata2["system_instruction_injected"] is False
        assert metadata2["reason"] == "already_injected_in_session"
        assert modified2 == content2
    
    def test_format_system_instruction_markdown(self, instiller):
        """Test formatting system instruction as markdown."""
        formatted = instiller._format_system_instruction(
            "Test instruction", "markdown"
        )
        
        assert formatted.startswith("## 🎯 System Context")
        assert "Test instruction" in formatted
    
    def test_format_system_instruction_xml(self, instiller):
        """Test formatting system instruction as XML."""
        formatted = instiller._format_system_instruction(
            "Test instruction", "xml"
        )
        
        assert formatted.startswith("<system_instruction>")
        assert formatted.endswith("</system_instruction>")
        assert "Test instruction" in formatted
    
    def test_format_system_instruction_comment(self, instiller):
        """Test formatting system instruction as comment."""
        formatted = instiller._format_system_instruction(
            "Test instruction", "comment"
        )
        
        assert formatted.startswith("//")
        assert "Test instruction" in formatted
    
    def test_instill_with_system_instruction(self, instiller, sample_tenets):
        """Test full instill with system instruction."""
        instiller.config.tenet.system_instruction = "You are a helpful assistant."
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.injection_frequency = "always"
        
        with patch.object(instiller.manager, 'get_pending_tenets', return_value=sample_tenets):
            with patch.object(instiller.injector, 'inject_tenets') as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})
                
                result = instiller.instill(
                    "Test content",
                    session="sys-test",
                    inject_system_instruction=True
                )
                
                # System instruction should be injected
                assert "sys-test" in instiller.system_instruction_injected
                assert instiller.system_instruction_injected["sys-test"] is True
    
    def test_instill_context_result_with_system(self, instiller, sample_context, sample_tenets):
        """Test instilling into ContextResult with system instruction."""
        instiller.config.tenet.system_instruction = "System prompt"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.injection_frequency = "always"
        
        with patch.object(instiller.manager, 'get_pending_tenets', return_value=sample_tenets):
            with patch.object(instiller.injector, 'inject_tenets') as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})
                
                result = instiller.instill(sample_context, session="test")
                
                assert isinstance(result, ContextResult)
                assert "system_instruction" in result.metadata
    
    def test_system_instruction_no_content(self, instiller):
        """Test system instruction when no instruction configured."""
        instiller.config.tenet.system_instruction = None
        
        content = "Original"
        modified, metadata = instiller.inject_system_instruction(content)
        
        assert modified == content
        assert metadata["system_instruction_injected"] is False
    
    def test_system_instruction_token_counting(self, instiller):
        """Test that token increase is calculated for system instruction."""
        instiller.config.tenet.system_instruction = "A" * 1000  # Long instruction
        instiller.config.tenet.system_instruction_enabled = True
        
        with patch('tenets.core.instiller.instiller.estimate_tokens') as mock_estimate:
            mock_estimate.side_effect = [100, 350]  # Original, then modified
            
            content = "Short content"
            modified, metadata = instiller.inject_system_instruction(content)
            
            assert metadata["token_increase"] == 250  # 350 - 100
    
    def test_system_instruction_with_separator(self, instiller):
        """Test system instruction with custom separator."""
        instiller.config.tenet.system_instruction = "Instruction"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.system_instruction_separator = "\n===\n"
        
        content = "Content"
        modified, metadata = instiller.inject_system_instruction(content)
        
        assert "\n===\n" in modified
        assert modified.index("===") > modified.index("Instruction")
        assert modified.index("===") < modified.index("Content")
    
    def test_system_instruction_with_label(self, instiller):
        """Test system instruction with custom label."""
        instiller.config.tenet.system_instruction = "Instruction"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.system_instruction_label = "🚀 Custom Label"
        
        content = "Content"
        modified, metadata = instiller.inject_system_instruction(content, format="markdown")
        
        assert "🚀 Custom Label" in modified
    
    def test_system_instruction_persist_in_context(self, instiller):
        """Test system instruction persistence in context metadata."""
        instiller.config.tenet.system_instruction = "Persistent instruction"
        instiller.config.tenet.system_instruction_enabled = True
        instiller.config.tenet.system_instruction_persist_in_context = True
        
        content = "Content"
        modified, metadata = instiller.inject_system_instruction(content)
        
        assert metadata.get("system_instruction_persisted") is True
        assert metadata.get("system_instruction_content") == "Persistent instruction"
    
    def test_system_instruction_format_plain(self, instiller):
        """Test system instruction with plain format."""
        formatted = instiller._format_system_instruction(
            "Test instruction", "plain"
        )
        
        assert formatted == "🎯 System Context\n\nTest instruction"
        assert "<" not in formatted
        assert "#" not in formatted
        assert "//" not in formatted