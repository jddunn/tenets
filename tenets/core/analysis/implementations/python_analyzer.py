"""Python-specific code analyzer using AST.

This module provides comprehensive analysis of Python source code using
the Abstract Syntax Tree (AST) module for accurate parsing. It extracts
imports, exports, code structure, and calculates various complexity metrics.
"""

import ast
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ImportInfo,
)
from tenets.utils.logger import get_logger

from ..base import LanguageAnalyzer


class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analyzer using AST.

    Provides deep analysis of Python code including:
    - Import analysis with tracking of relative imports
    - Function and class extraction with signatures
    - Decorator detection
    - Complexity metrics (cyclomatic, cognitive, Halstead)
    - Type hint analysis
    - Docstring extraction
    - Async function detection

    This analyzer uses Python's built-in AST module for accurate parsing,
    falling back to regex-based extraction when AST parsing fails.
    """

    language_name = "python"
    file_extensions = [".py", ".pyw", ".pyi"]

    def __init__(self):
        """Initialize the Python analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract imports from Python code using AST.

        Identifies all import statements including:
        - Standard imports: import os, import sys
        - From imports: from datetime import datetime
        - Relative imports: from . import module
        - Aliased imports: import numpy as np

        Args:
            content: Python source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with details about each import
        """
        imports = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # Handle: import module1, module2 as m2
                    for alias in node.names:
                        imports.append(
                            ImportInfo(
                                module=alias.name,
                                alias=alias.asname,
                                line=node.lineno,
                                type="import",
                                is_relative=False,
                                level=0,
                            )
                        )

                elif isinstance(node, ast.ImportFrom):
                    # Handle: from module import name1, name2
                    module = node.module or ""
                    for alias in node.names:
                        imported_name = alias.name

                        # Determine full module path
                        if imported_name == "*":
                            full_module = module
                        else:
                            full_module = f"{module}.{imported_name}" if module else imported_name

                        imports.append(
                            ImportInfo(
                                module=full_module,
                                alias=alias.asname,
                                line=node.lineno,
                                type="from",
                                is_relative=node.level > 0,
                                level=node.level,
                                from_module=module,
                            )
                        )

        except SyntaxError as e:
            self.logger.debug(f"Syntax error parsing {file_path}: {e}")
            # Fallback to regex-based extraction
            imports = self._extract_imports_regex(content)

        return imports

    def _extract_imports_regex(self, content: str) -> List[ImportInfo]:
        """Fallback regex-based import extraction when AST parsing fails.

        Args:
            content: Python source code

        Returns:
            List of ImportInfo objects extracted using regex
        """
        imports = []
        lines = content.split("\n")

        import_pattern = re.compile(r"^\s*import\s+([\w\.,\s]+)(?:\s+as\s+(\w+))?")
        from_pattern = re.compile(
            r"^\s*from\s+([\w\.]+)\s+import\s+([\w\.,\s\*]+)(?:\s+as\s+(\w+))?"
        )

        for i, line in enumerate(lines, 1):
            # Skip comments and strings
            if line.strip().startswith("#"):
                continue

            # Standard import
            match = import_pattern.match(line)
            if match:
                modules = match.group(1).split(",")
                alias = match.group(2)
                for module in modules:
                    imports.append(
                        ImportInfo(
                            module=module.strip(),
                            alias=alias,
                            line=i,
                            type="import",
                            is_relative=False,
                            level=0,
                        )
                    )

            # From import
            match = from_pattern.match(line)
            if match:
                from_module = match.group(1)
                imported = match.group(2).split(",")
                alias = match.group(3)

                for item in imported:
                    item = item.strip()
                    imports.append(
                        ImportInfo(
                            module=item if item != "*" else from_module,
                            alias=alias if len(imported) == 1 else None,
                            line=i,
                            type="from",
                            is_relative=from_module.startswith("."),
                            level=len(from_module) - len(from_module.lstrip(".")),
                            from_module=from_module.lstrip("."),
                        )
                    )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from Python code.

        Python exports are determined by:
        1. Explicit __all__ definition
        2. Public symbols (not starting with underscore)

        Args:
            content: Python source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols with their metadata
        """
        exports = []

        try:
            tree = ast.parse(content)

            # Look for __all__ definition
            has_all = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            has_all = True
                            if isinstance(node.value, ast.List):
                                for item in node.value.elts:
                                    if isinstance(item, (ast.Constant, ast.Str)):
                                        value = (
                                            item.value if isinstance(item, ast.Constant) else item.s
                                        )
                                        exports.append(
                                            {
                                                "name": value,
                                                "type": "explicit",
                                                "line": node.lineno,
                                                "defined_in_all": True,
                                            }
                                        )

            # If no __all__, consider all public symbols
            if not has_all:
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                        exports.append(
                            {
                                "name": node.name,
                                "type": "function",
                                "line": node.lineno,
                                "is_async": isinstance(node, ast.AsyncFunctionDef),
                                "decorators": [self._get_name(d) for d in node.decorator_list],
                            }
                        )
                    elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                        exports.append(
                            {
                                "name": node.name,
                                "type": "class",
                                "line": node.lineno,
                                "bases": [self._get_name(base) for base in node.bases],
                                "decorators": [self._get_name(d) for d in node.decorator_list],
                            }
                        )
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                                exports.append(
                                    {
                                        "name": target.id,
                                        "type": "variable",
                                        "line": node.lineno,
                                        "is_constant": target.id.isupper(),
                                    }
                                )

        except SyntaxError:
            self.logger.debug(f"Syntax error parsing exports from {file_path}")

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract comprehensive code structure from Python file.

        Parses the AST to extract:
        - Classes with inheritance, methods, and docstrings
        - Functions with signatures, decorators, and complexity
        - Global variables and constants
        - Nested functions and classes

        Args:
            content: Python source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with complete structural information
        """
        structure = CodeStructure()

        try:
            tree = ast.parse(content)

            # Extract classes with full information
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = ClassInfo(
                        name=node.name,
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        base_classes=[self._get_name(base) for base in node.bases],
                        decorators=[self._get_name(d) for d in node.decorator_list],
                        methods=[],
                        docstring=ast.get_docstring(node),
                        is_abstract=self._is_abstract_class(node),
                        metaclass=self._get_metaclass(node),
                    )

                    # Extract methods and attributes
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = FunctionInfo(
                                name=item.name,
                                line=item.lineno,
                                end_line=getattr(item, "end_lineno", item.lineno),
                                decorators=[self._get_name(d) for d in item.decorator_list],
                                is_async=isinstance(item, ast.AsyncFunctionDef),
                                docstring=ast.get_docstring(item),
                                complexity=self._calculate_function_complexity(item),
                                return_type=self._get_name(item.returns) if item.returns else None,
                                is_constructor=item.name == "__init__",
                                is_abstract=any(
                                    self._get_name(d) == "abstractmethod"
                                    for d in item.decorator_list
                                ),
                                is_static=self._is_static_method(item),
                                is_class=self._is_class_method(item),
                                is_property=self._is_property(item),
                                is_private=item.name.startswith("_")
                                and not item.name.startswith("__"),
                            )
                            class_info.methods.append(method_info)
                        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            # Class attributes with type hints
                            class_info.attributes.append(
                                {
                                    "name": item.target.id,
                                    "line": item.lineno,
                                    "type_hint": (
                                        self._get_name(item.annotation) if item.annotation else None
                                    ),
                                }
                            )

                    structure.classes.append(class_info)

            # Extract top-level functions
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = FunctionInfo(
                        name=node.name,
                        line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        args=self._extract_function_args(node),
                        decorators=[self._get_name(d) for d in node.decorator_list],
                        is_async=isinstance(node, ast.AsyncFunctionDef),
                        docstring=ast.get_docstring(node),
                        complexity=self._calculate_function_complexity(node),
                        return_type=self._get_name(node.returns) if node.returns else None,
                        is_constructor=False,  # Top-level functions are never constructors
                        is_abstract=any(
                            self._get_name(d) == "abstractmethod" for d in node.decorator_list
                        ),
                        is_static=False,  # Top-level functions are not static methods
                        is_class=False,  # Top-level functions are not class methods
                        is_property=self._is_property(
                            node
                        ),  # Top-level properties possible with decorators
                        is_private=node.name.startswith("_") and not node.name.startswith("__"),
                    )
                    structure.functions.append(func_info)

            # Extract global variables and constants
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_info = {
                                "name": target.id,
                                "line": node.lineno,
                                "type": "constant" if target.id.isupper() else "variable",
                            }
                            structure.variables.append(var_info)

                            if target.id.isupper():
                                structure.constants.append(target.id)

                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    # Variables with type hints
                    var_info = {
                        "name": node.target.id,
                        "line": node.lineno,
                        "type": "typed_variable",
                        "type_hint": self._get_name(node.annotation) if node.annotation else None,
                    }
                    structure.variables.append(var_info)

            # Extract type aliases (Python 3.10+)
            if hasattr(ast, "TypeAlias"):
                for node in ast.walk(tree):
                    if isinstance(node, ast.TypeAlias):
                        structure.type_aliases.append(
                            {
                                "name": node.name.id,
                                "line": node.lineno,
                                "value": self._get_name(node.value),
                            }
                        )

        except SyntaxError as e:
            self.logger.debug(f"Syntax error parsing structure from {file_path}: {e}")

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate comprehensive complexity metrics for Python code.

        Calculates:
        - Cyclomatic complexity (McCabe)
        - Cognitive complexity
        - Halstead metrics
        - Maintainability index
        - Maximum nesting depth
        - Lines of code metrics

        Args:
            content: Python source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with all calculated metrics
        """
        metrics = ComplexityMetrics()

        try:
            tree = ast.parse(content)

            # Calculate cyclomatic complexity (McCabe)
            cyclomatic = self._calculate_cyclomatic_complexity(tree)
            metrics.cyclomatic = cyclomatic

            # Calculate cognitive complexity
            cognitive = self._calculate_cognitive_complexity(tree)
            metrics.cognitive = cognitive

            # Calculate Halstead metrics
            halstead = self._calculate_halstead_metrics(tree)
            metrics.halstead = halstead

            # Calculate nesting metrics
            metrics.max_depth = self._calculate_max_depth(tree)

            # Count code elements
            metrics.line_count = content.count("\n") + 1
            metrics.function_count = len(
                [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            )
            metrics.class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            metrics.method_count = self._count_methods(tree)

            # Calculate comment ratio
            metrics.comment_lines = self._count_comment_lines(content)
            metrics.comment_ratio = (
                metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
            )

            # Calculate code lines (non-empty, non-comment)
            metrics.code_lines = self._count_code_lines(content)

            # Calculate maintainability index
            # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))
            if halstead and halstead.get("volume", 0) > 0 and metrics.code_lines > 0:
                halstead_volume = halstead["volume"]
                mi = (
                    171
                    - 5.2 * math.log(halstead_volume)
                    - 0.23 * cyclomatic
                    - 16.2 * math.log(metrics.code_lines)
                    + 50 * math.sin(math.sqrt(2.4 * metrics.comment_ratio))
                )
                metrics.maintainability_index = max(0, min(100, mi))

        except SyntaxError as e:
            self.logger.debug(f"Syntax error calculating complexity for {file_path}: {e}")
            # Return basic metrics from text analysis
            metrics.line_count = content.count("\n") + 1
            metrics.code_lines = self._count_code_lines(content)
            metrics.comment_lines = self._count_comment_lines(content)

        return metrics

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity.

        Args:
            tree: AST tree of the Python code

        Returns:
            Cyclomatic complexity score
        """
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            # Decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # Each 'and'/'or' adds a decision point
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                # List/dict/set comprehensions
                complexity += sum(1 for _ in node.ifs) + 1
            elif isinstance(node, ast.IfExp):
                # Ternary operator
                complexity += 1

        return complexity

    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity score.

        Args:
            tree: AST tree of the Python code

        Returns:
            Cognitive complexity score
        """

        class CognitiveVisitor(ast.NodeVisitor):
            """Visitor to calculate cognitive complexity."""

            def __init__(self):
                self.complexity = 0
                self.nesting_level = 0

            def visit_If(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_While(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_For(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_ExceptHandler(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_With(self, node):
                self.complexity += 1 + self.nesting_level
                self.nesting_level += 1
                self.generic_visit(node)
                self.nesting_level -= 1

            def visit_BoolOp(self, node):
                # Boolean operators add complexity
                self.complexity += len(node.values) - 1
                self.generic_visit(node)

            def visit_Lambda(self, node):
                # Lambda functions add complexity
                self.complexity += 1
                self.generic_visit(node)

            def visit_IfExp(self, node):
                # Ternary expressions add complexity
                self.complexity += 1
                self.generic_visit(node)

        visitor = CognitiveVisitor()
        visitor.visit(tree)
        return visitor.complexity

    def _calculate_halstead_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics.

        Args:
            tree: AST tree of the Python code

        Returns:
            Dictionary of Halstead metrics
        """
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        for node in ast.walk(tree):
            # Count operators
            if isinstance(
                node,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Mod,
                    ast.Pow,
                    ast.LShift,
                    ast.RShift,
                    ast.BitOr,
                    ast.BitXor,
                    ast.BitAnd,
                    ast.FloorDiv,
                    ast.And,
                    ast.Or,
                    ast.Not,
                    ast.Eq,
                    ast.NotEq,
                    ast.Lt,
                    ast.LtE,
                    ast.Gt,
                    ast.GtE,
                    ast.Is,
                    ast.IsNot,
                    ast.In,
                    ast.NotIn,
                    ast.UAdd,
                    ast.USub,
                    ast.Invert,
                ),
            ):
                op_name = node.__class__.__name__
                operators.add(op_name)
                operator_count += 1

            # Count operands (variables, constants)
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                operand_count += 1
            elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
                value = (
                    node.value
                    if isinstance(node, ast.Constant)
                    else (node.n if isinstance(node, ast.Num) else node.s)
                )
                operands.add(str(value))
                operand_count += 1

        n1 = len(operators)  # Unique operators
        n2 = len(operands)  # Unique operands
        N1 = operator_count  # Total operators
        N2 = operand_count  # Total operands

        # Calculate metrics
        metrics = {}
        if n1 > 0 and n2 > 0:
            metrics["unique_operators"] = n1
            metrics["unique_operands"] = n2
            metrics["total_operators"] = N1
            metrics["total_operands"] = N2
            metrics["vocabulary"] = n1 + n2
            metrics["length"] = N1 + N2
            metrics["volume"] = metrics["length"] * (
                math.log2(metrics["vocabulary"]) if metrics["vocabulary"] > 0 else 0
            )
            metrics["difficulty"] = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            metrics["effort"] = metrics["volume"] * metrics["difficulty"]
            metrics["time"] = metrics["effort"] / 18  # Seconds to write
            metrics["bugs"] = metrics["volume"] / 3000  # Estimated bugs

        return metrics

    def _calculate_max_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth in the code.

        Args:
            tree: AST tree of the Python code

        Returns:
            Maximum nesting depth
        """

        class DepthVisitor(ast.NodeVisitor):
            """Visitor to calculate maximum nesting depth."""

            def __init__(self):
                self.max_depth = 0
                self.current_depth = 0

            def visit_If(self, node):
                self._visit_block(node)

            def visit_While(self, node):
                self._visit_block(node)

            def visit_For(self, node):
                self._visit_block(node)

            def visit_With(self, node):
                self._visit_block(node)

            def visit_Try(self, node):
                self._visit_block(node)

            def visit_FunctionDef(self, node):
                self._visit_block(node)

            def visit_AsyncFunctionDef(self, node):
                self._visit_block(node)

            def visit_ClassDef(self, node):
                self._visit_block(node)

            def _visit_block(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1

        visitor = DepthVisitor()
        visitor.visit(tree)
        return visitor.max_depth

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity of a single function.

        Args:
            node: Function definition AST node

        Returns:
            Cyclomatic complexity of the function
        """
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.IfExp):
                complexity += 1
        return complexity

    def _get_name(self, node: ast.AST) -> str:
        """Get string representation of an AST node.

        Args:
            node: AST node

        Returns:
            String representation of the node
        """
        if node is None:
            return ""
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
            value = (
                node.value
                if isinstance(node, ast.Constant)
                else (node.n if isinstance(node, ast.Num) else node.s)
            )
            return str(value)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        else:
            # Use unparse if available (Python 3.9+)
            if hasattr(ast, "unparse"):
                return ast.unparse(node)
            else:
                return str(node.__class__.__name__)

    def _extract_function_args(self, node: ast.FunctionDef) -> List[str]:
        """Extract function arguments with type hints.

        Args:
            node: Function definition AST node

        Returns:
            List of argument names with type hints if available
        """
        args = []

        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {self._get_name(node.args.vararg.annotation)}"
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {self._get_name(node.args.kwarg.annotation)}"
            args.append(arg_str)

        return args

    def _is_static_method(self, node: ast.FunctionDef) -> bool:
        """Check if a method is a static method.

        Args:
            node: Function definition AST node

        Returns:
            True if the method has @staticmethod decorator
        """
        for decorator in node.decorator_list:
            if self._get_name(decorator) == "staticmethod":
                return True
        return False

    def _is_class_method(self, node: ast.FunctionDef) -> bool:
        """Check if a method is a class method.

        Args:
            node: Function definition AST node

        Returns:
            True if the method has @classmethod decorator
        """
        for decorator in node.decorator_list:
            if self._get_name(decorator) == "classmethod":
                return True
        return False

    def _is_property(self, node: ast.FunctionDef) -> bool:
        """Check if a method is a property.

        Args:
            node: Function definition AST node

        Returns:
            True if the method has @property decorator
        """
        for decorator in node.decorator_list:
            decorator_name = self._get_name(decorator)
            if (
                decorator_name == "property"
                or ".setter" in decorator_name
                or ".getter" in decorator_name
            ):
                return True
        return False

    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if a method is abstract.

        Args:
            node: Function definition AST node

        Returns:
            True if the method has @abstractmethod decorator
        """
        for decorator in node.decorator_list:
            if "abstractmethod" in self._get_name(decorator):
                return True
        return False

    def _is_abstract_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is abstract.

        Args:
            node: Class definition AST node

        Returns:
            True if the class inherits from ABC or has abstract methods
        """
        # Check if inherits from ABC
        for base in node.bases:
            base_name = self._get_name(base)
            if "ABC" in base_name or "ABCMeta" in base_name:
                return True

        # Check if has abstract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and self._is_abstract_method(item):
                return True

        return False

    def _get_metaclass(self, node: ast.ClassDef) -> Optional[str]:
        """Get the metaclass of a class if specified.

        Args:
            node: Class definition AST node

        Returns:
            Metaclass name if specified, None otherwise
        """
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                return self._get_name(keyword.value)
        return None

    def _count_methods(self, tree: ast.AST) -> int:
        """Count total number of methods in all classes.

        Args:
            tree: AST tree of the Python code

        Returns:
            Total number of methods
        """
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        count += 1
        return count

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: Python source code

        Returns:
            Number of code lines
        """
        count = 0
        in_multiline_string = False
        multiline_delimiter = None

        for line in content.split("\n"):
            stripped = line.strip()

            # Handle multiline strings
            if '"""' in line or "'''" in line:
                if not in_multiline_string:
                    delimiter = '"""' if '"""' in line else "'''"
                    if line.count(delimiter) >= 2:
                        # Single line docstring
                        count += 1
                    else:
                        in_multiline_string = True
                        multiline_delimiter = delimiter
                elif multiline_delimiter in line:
                    in_multiline_string = False
                    multiline_delimiter = None
                continue

            if in_multiline_string:
                continue

            # Skip empty lines and comments
            if stripped and not stripped.startswith("#"):
                count += 1

        return count

    def _count_comment_lines(self, content: str) -> int:
        """Count lines that are comments or docstrings.

        Args:
            content: Python source code

        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_string = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Handle multiline strings (docstrings)
            if '"""' in line or "'''" in line:
                count += 1
                if not in_multiline_string:
                    delimiter = '"""' if '"""' in line else "'''"
                    if line.count(delimiter) < 2:
                        in_multiline_string = True
                else:
                    in_multiline_string = False
                continue

            if in_multiline_string:
                count += 1
                continue

            # Count comment lines
            if stripped.startswith("#"):
                count += 1

        return count
