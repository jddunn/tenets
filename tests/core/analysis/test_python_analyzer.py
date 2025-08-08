"""
Unit tests for the Python code analyzer.

This module tests the Python-specific code analysis functionality including
AST parsing, import extraction, structure analysis, and complexity calculation.
The Python analyzer is one of the most sophisticated analyzers as it uses
Python's built-in AST module for accurate parsing.

Test Coverage:
    - Import extraction (standard, from, relative, aliased)
    - Export detection (__all__, public symbols)
    - Structure extraction (classes, functions, variables)
    - Complexity metrics (cyclomatic, cognitive, Halstead)
    - Error handling for invalid Python code
    - Edge cases and corner cases
"""

import ast
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.python_analyzer import PythonAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestPythonAnalyzerInitialization:
    """Test suite for PythonAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = PythonAnalyzer()

        assert analyzer.language_name == "python"
        assert ".py" in analyzer.file_extensions
        assert ".pyw" in analyzer.file_extensions
        assert ".pyi" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Python import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PythonAnalyzer instance."""
        return PythonAnalyzer()

    def test_extract_standard_imports(self, analyzer):
        """Test extraction of standard import statements."""
        code = """
import os
import sys
import json
"""
        imports = analyzer.extract_imports(code, Path("test.py"))

        assert len(imports) == 3
        assert any(imp.module == "os" for imp in imports)
        assert any(imp.module == "sys" for imp in imports)
        assert any(imp.module == "json" for imp in imports)

        # Check import details
        os_import = next(imp for imp in imports if imp.module == "os")
        assert os_import.type == "import"
        assert os_import.is_relative is False
        assert os_import.line == 2

    def test_extract_from_imports(self, analyzer):
        """Test extraction of from-import statements."""
        code = """
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
"""
        imports = analyzer.extract_imports(code, Path("test.py"))

        # Should have one ImportInfo per imported item
        assert len(imports) >= 6

        # Check datetime import
        datetime_import = next(imp for imp in imports if "datetime" in imp.module)
        assert datetime_import.type == "from"
        assert datetime_import.is_relative is False

    def test_extract_aliased_imports(self, analyzer):
        """Test extraction of aliased imports."""
        code = """
import numpy as np
import pandas as pd
from datetime import datetime as dt
"""
        imports = analyzer.extract_imports(code, Path("test.py"))

        # Check aliased imports
        numpy_import = next(imp for imp in imports if imp.module == "numpy")
        assert numpy_import.alias == "np"

        pandas_import = next(imp for imp in imports if imp.module == "pandas")
        assert pandas_import.alias == "pd"

    def test_extract_relative_imports(self, analyzer):
        """Test extraction of relative imports."""
        code = """
from . import module1
from .. import module2
from ...package import module3
from .sibling import function
"""
        imports = analyzer.extract_imports(code, Path("test.py"))

        # All should be marked as relative
        for imp in imports:
            assert imp.is_relative is True

        # Check relative levels
        level1 = next(imp for imp in imports if imp.level == 1)
        level2 = next(imp for imp in imports if imp.level == 2)
        level3 = next(imp for imp in imports if imp.level == 3)

        assert level1 is not None
        assert level2 is not None
        assert level3 is not None

    def test_extract_star_imports(self, analyzer):
        """Test extraction of star imports."""
        code = """
from module import *
from package.submodule import *
"""
        imports = analyzer.extract_imports(code, Path("test.py"))

        assert len(imports) == 2
        assert all(imp.type == "from" for imp in imports)

    def test_extract_imports_with_syntax_error(self, analyzer):
        """Test import extraction falls back to regex when AST parsing fails."""
        code = """
import os
from pathlib import Path
this is invalid python code {[}]
import sys
"""
        imports = analyzer.extract_imports(code, Path("test.py"))

        # Should still extract valid imports using regex
        assert len(imports) >= 2
        assert any(imp.module == "os" for imp in imports)
        assert any(imp.module == "sys" for imp in imports)


class TestExportExtraction:
    """Test suite for Python export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PythonAnalyzer instance."""
        return PythonAnalyzer()

    def test_extract_exports_with_all(self, analyzer):
        """Test export extraction when __all__ is defined."""
        code = """
__all__ = ['public_function', 'PublicClass', 'CONSTANT']

def public_function():
    pass

def _private_function():
    pass

class PublicClass:
    pass

CONSTANT = 42
"""
        exports = analyzer.extract_exports(code, Path("test.py"))

        # Should only include items in __all__
        assert len(exports) == 3

        export_names = [exp["name"] for exp in exports]
        assert "public_function" in export_names
        assert "PublicClass" in export_names
        assert "CONSTANT" in export_names

        # All should be marked as explicit exports
        assert all(exp.get("defined_in_all") for exp in exports)

    def test_extract_exports_without_all(self, analyzer):
        """Test export extraction without __all__ (public symbols)."""
        code = """
def public_function():
    pass

def _private_function():
    pass

class PublicClass:
    pass

class _PrivateClass:
    pass

PUBLIC_CONSTANT = 42
_PRIVATE_CONSTANT = 0
"""
        exports = analyzer.extract_exports(code, Path("test.py"))

        # Should only include public symbols (not starting with _)
        export_names = [exp["name"] for exp in exports]
        assert "public_function" in export_names
        assert "PublicClass" in export_names
        assert "PUBLIC_CONSTANT" in export_names

        # Private symbols should not be exported
        assert "_private_function" not in export_names
        assert "_PrivateClass" not in export_names
        assert "_PRIVATE_CONSTANT" not in export_names

    def test_extract_exports_with_decorators(self, analyzer):
        """Test export extraction includes decorator information."""
        code = """
@decorator1
@decorator2
def decorated_function():
    pass

@dataclass
class DecoratedClass:
    pass
"""
        exports = analyzer.extract_exports(code, Path("test.py"))

        func_export = next(exp for exp in exports if exp["name"] == "decorated_function")
        assert "decorators" in func_export

        class_export = next(exp for exp in exports if exp["name"] == "DecoratedClass")
        assert "decorators" in class_export


class TestStructureExtraction:
    """Test suite for code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PythonAnalyzer instance."""
        return PythonAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class definitions."""
        code = '''
class SimpleClass:
    """A simple class."""
    pass

class InheritedClass(BaseClass):
    """Class with inheritance."""
    pass

class MultipleInheritance(Base1, Base2, Base3):
    """Class with multiple inheritance."""
    pass

@dataclass
class DecoratedClass:
    """Class with decorators."""
    field1: str
    field2: int
'''
        structure = analyzer.extract_structure(code, Path("test.py"))

        assert len(structure.classes) == 4

        # Check simple class
        simple = next(c for c in structure.classes if c.name == "SimpleClass")
        assert simple.docstring == "A simple class."
        assert len(simple.bases) == 0

        # Check inherited class
        inherited = next(c for c in structure.classes if c.name == "InheritedClass")
        assert "BaseClass" in inherited.bases

        # Check multiple inheritance
        multiple = next(c for c in structure.classes if c.name == "MultipleInheritance")
        assert len(multiple.bases) == 3

        # Check decorated class
        decorated = next(c for c in structure.classes if c.name == "DecoratedClass")
        assert len(decorated.decorators) > 0

    def test_extract_class_methods(self, analyzer):
        """Test extraction of class methods and attributes."""
        code = """
class MyClass:
    class_var = 42
    
    def __init__(self, name):
        self.name = name
    
    def regular_method(self):
        pass
    
    @staticmethod
    def static_method():
        pass
    
    @classmethod
    def class_method(cls):
        pass
    
    @property
    def name_property(self):
        return self.name
    
    def _private_method(self):
        pass
"""
        structure = analyzer.extract_structure(code, Path("test.py"))

        my_class = structure.classes[0]
        assert len(my_class.methods) == 6

        # Check method types
        init_method = next(m for m in my_class.methods if m["name"] == "__init__")
        assert init_method["is_constructor"] is True

        static = next(m for m in my_class.methods if m["name"] == "static_method")
        assert static["is_static"] is True

        classmethod = next(m for m in my_class.methods if m["name"] == "class_method")
        assert classmethod["is_class"] is True

        prop = next(m for m in my_class.methods if m["name"] == "name_property")
        assert prop["is_property"] is True

        private = next(m for m in my_class.methods if m["name"] == "_private_method")
        assert private["is_private"] is True

    def test_extract_functions(self, analyzer):
        """Test extraction of function definitions."""
        code = '''
def simple_function():
    """Simple function."""
    pass

def function_with_args(arg1, arg2, *args, **kwargs):
    """Function with various argument types."""
    pass

def function_with_defaults(arg1, arg2="default", arg3=None):
    """Function with default arguments."""
    pass

def function_with_annotations(name: str, age: int) -> str:
    """Function with type annotations."""
    return f"{name} is {age}"

async def async_function():
    """Async function."""
    pass

@decorator
def decorated_function():
    """Decorated function."""
    pass
'''
        structure = analyzer.extract_structure(code, Path("test.py"))

        assert len(structure.functions) == 6

        # Check simple function
        simple = next(f for f in structure.functions if f.name == "simple_function")
        assert simple.docstring == "Simple function."

        # Check function with args
        with_args = next(f for f in structure.functions if f.name == "function_with_args")
        assert len(with_args.args) == 4
        assert "*args" in str(with_args.args)
        assert "**kwargs" in str(with_args.args)

        # Check function with annotations
        annotated = next(f for f in structure.functions if f.name == "function_with_annotations")
        assert annotated.return_type == "str"

        # Check async function
        async_func = next(f for f in structure.functions if f.name == "async_function")
        assert async_func.is_async is True

        # Check decorated function
        decorated = next(f for f in structure.functions if f.name == "decorated_function")
        assert len(decorated.decorators) > 0

    def test_extract_variables_and_constants(self, analyzer):
        """Test extraction of variables and constants."""
        code = """
# Constants (uppercase)
MAX_SIZE = 1000
DEFAULT_TIMEOUT = 30
API_KEY = "secret"

# Variables
counter = 0
name = "test"
data_list = [1, 2, 3]

# Type annotated variables
typed_var: int = 42
optional_var: Optional[str] = None
"""
        structure = analyzer.extract_structure(code, Path("test.py"))

        # Check constants
        assert "MAX_SIZE" in structure.constants
        assert "DEFAULT_TIMEOUT" in structure.constants
        assert "API_KEY" in structure.constants

        # Check variables
        var_names = [v["name"] for v in structure.variables]
        assert "counter" in var_names
        assert "name" in var_names
        assert "typed_var" in var_names


class TestComplexityCalculation:
    """Test suite for complexity metrics calculation."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PythonAnalyzer instance."""
        return PythonAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
def complex_function(x):
    if x > 0:
        if x > 10:
            return "big"
        else:
            return "small"
    elif x < 0:
        return "negative"
    else:
        return "zero"
    
    for i in range(10):
        if i % 2 == 0:
            print("even")
    
    while x > 0:
        x -= 1
    
    try:
        risky_operation()
    except ValueError:
        handle_error()
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        # Base complexity = 1
        # +1 for each if, elif, for, while, except
        assert metrics.cyclomatic >= 7

    def test_calculate_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """
def nested_function(x):
    if x > 0:  # +1
        for i in range(x):  # +2 (1 + nesting)
            if i % 2 == 0:  # +3 (1 + 2*nesting)
                if i > 5:  # +4 (1 + 3*nesting)
                    print(i)
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        # Cognitive complexity considers nesting
        assert metrics.cognitive >= 10

    def test_calculate_halstead_metrics(self, analyzer):
        """Test Halstead complexity metrics calculation."""
        code = """
def calculate(a, b):
    result = a + b
    result = result * 2
    return result
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        assert metrics.halstead is not None
        assert "vocabulary" in metrics.halstead
        assert "length" in metrics.halstead
        assert "volume" in metrics.halstead
        assert "difficulty" in metrics.halstead
        assert "effort" in metrics.halstead

    def test_calculate_max_depth(self, analyzer):
        """Test maximum nesting depth calculation."""
        code = """
def deeply_nested():
    if True:
        while True:
            for i in range(10):
                try:
                    with open("file"):
                        if condition:
                            pass  # Depth = 6
                except:
                    pass
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        assert metrics.max_depth >= 6

    def test_calculate_line_metrics(self, analyzer):
        """Test line count metrics."""
        code = '''
# This is a comment
def function():
    """Docstring."""
    # Another comment
    x = 1  # Inline comment
    y = 2
    
    return x + y

# More comments
'''
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        assert metrics.line_count == 11  # Total lines
        assert metrics.code_lines > 0  # Non-comment lines
        assert metrics.comment_lines > 0  # Comment lines
        assert metrics.comment_ratio > 0  # Should have some comments

    def test_calculate_maintainability_index(self, analyzer):
        """Test maintainability index calculation."""
        code = '''
def well_structured_function(param1, param2):
    """
    Well-documented function with clear logic.
    
    Args:
        param1: First parameter
        param2: Second parameter
    
    Returns:
        Processed result
    """
    # Clear, simple logic
    if param1 > param2:
        result = param1 - param2
    else:
        result = param2 - param1
    
    return result * 2
'''
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        # Well-structured code should have good maintainability
        assert metrics.maintainability_index > 50
        assert metrics.maintainability_index <= 100


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PythonAnalyzer instance."""
        return PythonAnalyzer()

    def test_handle_syntax_error_in_structure(self, analyzer):
        """Test structure extraction with syntax errors."""
        code = """
def valid_function():
    pass

this is invalid syntax {[}]

class ValidClass:
    pass
"""
        structure = analyzer.extract_structure(code, Path("test.py"))

        # Should return empty structure due to syntax error
        # (Can't parse partial AST)
        assert isinstance(structure, CodeStructure)

    def test_handle_syntax_error_in_complexity(self, analyzer):
        """Test complexity calculation with syntax errors."""
        code = """
def function():
    this is not valid python
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        # Should still return basic metrics
        assert metrics.line_count > 0
        # But complex metrics might be missing/zero
        assert metrics.cyclomatic == 0 or metrics.cyclomatic == 1

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.py"))
        exports = analyzer.extract_exports(code, Path("test.py"))
        structure = analyzer.extract_structure(code, Path("test.py"))
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        assert imports == []
        assert exports == []
        assert len(structure.classes) == 0
        assert len(structure.functions) == 0
        assert metrics.line_count == 1  # Empty string counts as 1 line

    def test_handle_comments_only(self, analyzer):
        """Test handling of files with only comments."""
        code = """
# This file contains only comments
# No actual code
# Just documentation
"""

        structure = analyzer.extract_structure(code, Path("test.py"))
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        assert len(structure.classes) == 0
        assert len(structure.functions) == 0
        assert metrics.code_lines == 0
        assert metrics.comment_lines > 0


class TestEdgeCases:
    """Test suite for edge cases and corner cases."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PythonAnalyzer instance."""
        return PythonAnalyzer()

    def test_nested_classes(self, analyzer):
        """Test extraction of nested classes."""
        code = """
class OuterClass:
    class InnerClass:
        class DeepNested:
            pass
"""
        structure = analyzer.extract_structure(code, Path("test.py"))

        # Should find all classes including nested
        assert len(structure.classes) >= 1
        outer = structure.classes[0]
        assert outer.name == "OuterClass"

    def test_lambda_functions(self, analyzer):
        """Test handling of lambda functions."""
        code = """
simple_lambda = lambda x: x * 2
complex_lambda = lambda x, y: x + y if x > y else y - x
lambda_list = [
    lambda n: n + 1,
    lambda n: n * 2,
    lambda n: n ** 2
]
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        # Lambda functions should contribute to complexity
        assert metrics.cognitive > 0

    def test_comprehensions(self, analyzer):
        """Test handling of list/dict/set comprehensions."""
        code = """
# List comprehension with condition
squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dict comprehension
squared_dict = {x: x**2 for x in range(10)}

# Set comprehension with multiple conditions
special_set = {x for x in range(100) if x % 2 == 0 if x % 3 == 0}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        # Comprehensions add to complexity
        assert metrics.cyclomatic > 1

    def test_abstract_methods(self, analyzer):
        """Test detection of abstract classes and methods."""
        code = """
from abc import ABC, abstractmethod

class AbstractBase(ABC):
    @abstractmethod
    def must_implement(self):
        pass
    
    @abstractmethod
    def another_abstract(self):
        ...

class ConcreteClass(AbstractBase):
    def must_implement(self):
        return "implemented"
    
    def another_abstract(self):
        return "also implemented"
"""
        structure = analyzer.extract_structure(code, Path("test.py"))

        # Check abstract class detection
        abstract_class = next(c for c in structure.classes if c.name == "AbstractBase")
        assert abstract_class.is_abstract is True

        # Check abstract method detection
        assert any(m["is_abstract"] for m in abstract_class.methods)

        # Concrete class should not be abstract
        concrete_class = next(c for c in structure.classes if c.name == "ConcreteClass")
        assert concrete_class.is_abstract is False

    def test_metaclass_detection(self, analyzer):
        """Test detection of metaclasses."""
        code = """
class MetaClass(type):
    pass

class MyClass(metaclass=MetaClass):
    pass
"""
        structure = analyzer.extract_structure(code, Path("test.py"))

        my_class = next(c for c in structure.classes if c.name == "MyClass")
        assert my_class.metaclass == "MetaClass"

    def test_very_long_file(self, analyzer):
        """Test handling of very long files."""
        # Generate a long file
        code = "# Long file\n"
        for i in range(1000):
            code += f"variable_{i} = {i}\n"
            if i % 100 == 0:
                code += f"def function_{i}(): pass\n"

        # Should handle without issues
        metrics = analyzer.calculate_complexity(code, Path("test.py"))

        assert metrics.line_count > 1000
        assert metrics.function_count >= 10
