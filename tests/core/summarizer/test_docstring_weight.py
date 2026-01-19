"""Tests for docstring weight functionality in summarizer."""

from pathlib import Path
from unittest.mock import MagicMock

from tenets.config import TenetsConfig
from tenets.core.summarizer import Summarizer
from tenets.models.analysis import CodeStructure, FileAnalysis


class TestDocstringWeight:
    """Test docstring weight configuration and behavior."""

    def test_default_docstring_weight(self):
        """Test that default docstring weight is 0.5."""
        config = TenetsConfig()
        assert config.summarizer.docstring_weight == 0.5

    def test_custom_docstring_weight_config(self):
        """Test setting custom docstring weight in config."""
        config = TenetsConfig()
        config.summarizer.docstring_weight = 0.8
        assert config.summarizer.docstring_weight == 0.8

    def test_include_all_signatures_default(self):
        """Test that include_all_signatures defaults to True."""
        config = TenetsConfig()
        assert config.summarizer.include_all_signatures is True

    def test_summarize_code_with_high_docstring_weight(self):
        """Test that high docstring weight includes more docstrings."""
        config = TenetsConfig()
        config.summarizer.docstring_weight = 1.0  # Always include docstrings
        config.summarizer.include_all_signatures = True

        summarizer = Summarizer(config)

        # Create a mock file with docstrings
        code = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        The sum of a and b
    """
    return a + b

class Calculator:
    """A simple calculator class for basic operations."""

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers together."""
        return x * y
'''

        file = FileAnalysis(
            path=Path("test.py"),
            language="python",
            content=code,
            lines=len(code.splitlines()),
            size=len(code),
            structure=CodeStructure(),
        )

        # Summarize with high docstring weight
        result = summarizer._summarize_code(file, target_ratio=0.3)

        # Should include docstrings
        assert "Calculate the sum" in result or "simple calculator" in result
        assert "def calculate_sum" in result or "class Calculator" in result

    def test_summarize_code_with_low_docstring_weight(self):
        """Test that low docstring weight excludes most docstrings."""
        config = TenetsConfig()
        config.summarizer.docstring_weight = 0.0  # Never include docstrings
        config.summarizer.include_all_signatures = True

        summarizer = Summarizer(config)

        # Create a mock file with docstrings
        code = '''
def process_data(data: list) -> dict:
    """Process input data and return results.

    This is a detailed docstring that should be excluded
    when docstring_weight is 0.
    """
    return {"processed": data}

class DataProcessor:
    """Main data processing class with various methods."""

    def transform(self, input_data):
        """Transform the input data."""
        return input_data
'''

        file = FileAnalysis(
            path=Path("processor.py"),
            language="python",
            content=code,
            lines=len(code.splitlines()),
            size=len(code),
            structure=CodeStructure(),
        )

        # Summarize with zero docstring weight
        result = summarizer._summarize_code(file, target_ratio=0.5)

        # Should include signatures but not docstring content
        assert "def process_data" in result or "class DataProcessor" in result
        # Should not include docstring content
        assert "detailed docstring" not in result
        assert "various methods" not in result

    def test_summarize_code_with_medium_docstring_weight(self):
        """Test that medium docstring weight includes some docstrings."""
        config = TenetsConfig()
        config.summarizer.docstring_weight = 0.5  # Balanced inclusion
        config.summarizer.include_all_signatures = True

        summarizer = Summarizer(config)

        # Create a mock file with various docstrings
        code = '''
def simple_func():
    """Short docstring."""
    pass

def complex_func(param1, param2, param3):
    """This is a very long and detailed docstring that explains
    everything about this function in great detail. It goes on
    and on with multiple paragraphs and examples.

    Args:
        param1: First parameter
        param2: Second parameter
        param3: Third parameter

    Returns:
        Some complex result
    """
    return None
'''

        file = FileAnalysis(
            path=Path("mixed.py"),
            language="python",
            content=code,
            lines=len(code.splitlines()),
            size=len(code),
            structure=CodeStructure(),
        )

        # Summarize with medium docstring weight
        result = summarizer._summarize_code(file, target_ratio=0.4)

        # Should include function signatures
        assert "def simple_func" in result or "def complex_func" in result
        # May include short docstrings or first lines of long ones
        # but not full long docstrings

    def test_ast_fallback_with_docstring_weight(self):
        """Test AST fallback extraction respects docstring weight."""
        config = TenetsConfig()
        config.summarizer.docstring_weight = 0.7  # High weight

        summarizer = Summarizer(config)

        # Python code that will use AST fallback
        code = '''
def api_endpoint(request: dict) -> dict:
    """Handle API request and return response."""
    return {"status": "ok"}

class APIHandler:
    """Handles all API requests."""

    def __init__(self):
        """Initialize the handler."""
        self.routes = {}
'''

        file = FileAnalysis(
            path=Path("api.py"),
            language="python",
            content=code,
            lines=len(code.splitlines()),
            size=len(code),
            structure=None,  # No structure, will use AST fallback
        )

        # Summarize should use AST fallback
        result = summarizer._summarize_code(file, target_ratio=0.5)

        # Should extract signatures via AST
        assert "def api_endpoint" in result or "class APIHandler" in result
        # With high weight, should include some docstrings
        assert "Handle API" in result or "Handles all API" in result

    def test_all_signatures_config(self):
        """Test include_all_signatures configuration."""
        config = TenetsConfig()
        config.summarizer.include_all_signatures = False  # Limit signatures

        summarizer = Summarizer(config)

        # Create file with many functions
        functions = []
        for i in range(30):
            functions.append(f'''
def function_{i}():
    """Function {i} docstring."""
    pass
''')

        code = "\n".join(functions)

        file = FileAnalysis(
            path=Path("many_funcs.py"),
            language="python",
            content=code,
            lines=len(code.splitlines()),
            size=len(code),
            structure=CodeStructure(functions=[MagicMock(name=f"function_{i}") for i in range(30)]),
        )

        # With include_all_signatures=False, should limit number of functions
        result = summarizer._summarize_code(file, target_ratio=0.2)

        # Should not include all 30 functions when limited
        function_count = result.count("def function_")
        assert function_count < 30  # Should be limited

    def test_class_method_extraction_with_docstrings(self):
        """Test that class methods are extracted with appropriate docstring handling."""
        config = TenetsConfig()
        config.summarizer.docstring_weight = 0.6
        config.summarizer.include_all_signatures = True

        summarizer = Summarizer(config)

        code = '''
class DataService:
    """Service for data operations."""

    def fetch_data(self, id: int) -> dict:
        """Fetch data by ID."""
        return {}

    def save_data(self, data: dict) -> bool:
        """Save data to database."""
        return True

    def validate_data(self, data: dict) -> bool:
        """Validate data against schema."""
        return True
'''

        file = FileAnalysis(
            path=Path("service.py"),
            language="python",
            content=code,
            lines=len(code.splitlines()),
            size=len(code),
            structure=None,  # Will use AST extraction
        )

        result = summarizer._summarize_code(file, target_ratio=0.4)

        # Should extract class and method signatures
        assert "class DataService" in result
        assert "def fetch_data" in result or "def save_data" in result
        # With weight 0.6, should include some docstrings
        assert "Service for data" in result or "Fetch data" in result
