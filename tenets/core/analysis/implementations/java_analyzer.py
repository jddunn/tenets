"""Java code analyzer.

This module provides comprehensive analysis for Java source files,
including support for modern Java features, annotations, and frameworks.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ImportInfo,
)
from tenets.utils.logger import get_logger

from ..base import LanguageAnalyzer


class JavaAnalyzer(LanguageAnalyzer):
    """Java code analyzer.

    Provides comprehensive analysis for Java files including:
    - Import analysis with static and wildcard imports
    - Package structure analysis
    - Class, interface, enum, and record extraction
    - Annotation processing
    - Generic type analysis
    - Method and field extraction with modifiers
    - Inner and anonymous class detection
    - Lambda expression support
    - Framework detection (Spring, JUnit, etc.)

    Supports modern Java features including records, sealed classes,
    pattern matching, and text blocks.
    """

    language_name = "java"
    file_extensions = [".java"]
    entry_points = [
        "Main.java", "Application.java", "App.java",
        "pom.xml", "build.gradle", "build.gradle.kts"
    ]
    project_indicators = {
        "spring": ["application.properties", "application.yml", "@SpringBootApplication"],
        "maven": ["pom.xml", "src/main/java/"],
        "gradle": ["build.gradle", "build.gradle.kts", "settings.gradle"],
        "android": ["AndroidManifest.xml", "build.gradle", "res/"],
    }

    def __init__(self):
        """Initialize the Java analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract imports from Java code.

        Handles:
        - Standard imports: import java.util.List;
        - Static imports: import static java.lang.Math.PI;
        - Wildcard imports: import java.util.*;
        - Static wildcard: import static org.junit.Assert.*;

        Args:
            content: Java source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        # Import patterns
        import_pattern = re.compile(r"^\s*import\s+(?:(static)\s+)?([a-zA-Z0-9_.]+(?:\.\*)?)\s*;")

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Stop at class/interface/enum declaration
            if re.match(r"^\s*(?:public\s+)?(?:class|interface|enum|record)\s+", line):
                break

            match = import_pattern.match(line)
            if match:
                is_static = match.group(1) == "static"
                module = match.group(2)
                is_wildcard = module.endswith(".*")

                # Determine import category
                category = self._categorize_java_import(module)

                imports.append(
                    ImportInfo(
                        module=module,
                        line=i,
                        type="static" if is_static else "import",
                        is_wildcard=is_wildcard,
                        is_relative=False,
                        category=category,
                        package=(
                            module.rsplit(".", 1)[0]
                            if "." in module and not is_wildcard
                            else module.rstrip(".*")
                        ),
                    )
                )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract public members from Java code.

        In Java, public members are exported from a class/package.
        This includes public classes, interfaces, enums, methods, and fields.

        Args:
            content: Java source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported (public) symbols with metadata
        """
        exports = []

        # Extract package name
        package_match = re.search(r"^\s*package\s+([\w.]+)\s*;", content, re.MULTILINE)
        package_name = package_match.group(1) if package_match else ""

        # Public classes
        class_pattern = r"(?:^|\n)\s*public\s+(?:(abstract|final)\s+)?class\s+(\w+)(?:<[^>]+>)?"
        for match in re.finditer(class_pattern, content):
            modifiers = [match.group(1)] if match.group(1) else []
            exports.append(
                {
                    "name": match.group(2),
                    "type": "class",
                    "line": content[: match.start()].count("\n") + 1,
                    "package": package_name,
                    "modifiers": modifiers,
                    "is_abstract": "abstract" in modifiers,
                    "is_final": "final" in modifiers,
                }
            )

        # Public interfaces
        interface_pattern = r"(?:^|\n)\s*public\s+interface\s+(\w+)(?:<[^>]+>)?"
        for match in re.finditer(interface_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "interface",
                    "line": content[: match.start()].count("\n") + 1,
                    "package": package_name,
                }
            )

        # Public enums
        enum_pattern = r"(?:^|\n)\s*public\s+enum\s+(\w+)"
        for match in re.finditer(enum_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "enum",
                    "line": content[: match.start()].count("\n") + 1,
                    "package": package_name,
                }
            )

        # Public records (Java 14+)
        record_pattern = r"(?:^|\n)\s*public\s+record\s+(\w+)\s*\([^)]*\)"
        for match in re.finditer(record_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "record",
                    "line": content[: match.start()].count("\n") + 1,
                    "package": package_name,
                }
            )

        # Public methods
        method_pattern = r"(?:^|\n)\s*public\s+(?:(?:static|final|abstract|synchronized|native)\s+)*(?:<[^>]+>\s+)?(?:[\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)"
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            # Filter out keywords that might match the pattern
            if method_name not in [
                "if",
                "for",
                "while",
                "switch",
                "catch",
                "new",
                "return",
                "throw",
            ]:
                line_content = content[match.start() : match.end()]
                exports.append(
                    {
                        "name": method_name,
                        "type": "method",
                        "line": content[: match.start()].count("\n") + 1,
                        "is_static": "static" in line_content,
                        "is_final": "final" in line_content,
                        "is_abstract": "abstract" in line_content,
                        "is_synchronized": "synchronized" in line_content,
                    }
                )

        # Public fields
        field_pattern = r"(?:^|\n)\s*public\s+(?:(?:static|final|volatile|transient)\s+)*(?:[\w<>\[\]]+)\s+(\w+)\s*[;=]"
        for match in re.finditer(field_pattern, content):
            field_name = match.group(1)
            line_content = content[match.start() : match.end()]
            exports.append(
                {
                    "name": field_name,
                    "type": "field",
                    "line": content[: match.start()].count("\n") + 1,
                    "is_static": "static" in line_content,
                    "is_final": "final" in line_content,
                    "is_constant": "static" in line_content and "final" in line_content,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Java file.

        Extracts:
        - Package declaration
        - Classes with inheritance and interfaces
        - Interfaces with extension
        - Enums with values
        - Records (Java 14+)
        - Methods with full signatures
        - Fields with types
        - Annotations
        - Inner classes
        - Lambda expressions

        Args:
            content: Java source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Extract package declaration
        package_match = re.search(r"^\s*package\s+([\w.]+)\s*;", content, re.MULTILINE)
        if package_match:
            structure.package = package_match.group(1)

        # Extract classes
        class_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?(?:(abstract|final)\s+)?(?:(sealed)\s+)?class\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+([\w<>]+))?(?:\s+implements\s+([\w,\s<>]+))?"

        for match in re.finditer(class_pattern, content):
            visibility = match.group(1) or "package-private"
            modifiers = []
            if match.group(2):
                modifiers.append(match.group(2))
            if match.group(3):
                modifiers.append(match.group(3))

            class_name = match.group(4)
            generics = match.group(5)
            extends = match.group(6)
            implements = match.group(7)

            class_info = ClassInfo(
                name=class_name,
                line=content[: match.start()].count("\n") + 1,
                visibility=visibility,
                modifiers=modifiers,
                generics=generics,
                bases=[extends] if extends else [],
                interfaces=self._parse_implements_list(implements) if implements else [],
                methods=[],
                fields=[],
                inner_classes=[],
            )

            # Find class body and extract members
            class_body = self._extract_class_body(content, match.end())
            if class_body:
                class_info.methods = self._extract_methods(class_body)
                class_info.fields = self._extract_fields(class_body)
                class_info.inner_classes = self._extract_inner_classes(class_body)

            structure.classes.append(class_info)

        # Extract interfaces
        interface_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?(?:(sealed)\s+)?interface\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+([\w,\s<>]+))?"

        for match in re.finditer(interface_pattern, content):
            visibility = match.group(1) or "package-private"
            is_sealed = match.group(2) == "sealed"
            interface_name = match.group(3)
            generics = match.group(4)
            extends = match.group(5)

            # Extract interface methods
            interface_body = self._extract_class_body(content, match.end())
            methods = self._extract_interface_methods(interface_body) if interface_body else []

            structure.interfaces.append(
                {
                    "name": interface_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "visibility": visibility,
                    "is_sealed": is_sealed,
                    "generics": generics,
                    "extends": self._parse_implements_list(extends) if extends else [],
                    "methods": methods,
                    "is_functional": len(methods) == 1,  # Functional interface
                }
            )

        # Extract enums
        enum_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?enum\s+(\w+)(?:\s+implements\s+([\w,\s<>]+))?"

        for match in re.finditer(enum_pattern, content):
            visibility = match.group(1) or "package-private"
            enum_name = match.group(2)
            implements = match.group(3)

            # Extract enum values
            enum_body = self._extract_class_body(content, match.end())
            values = self._extract_enum_values(enum_body) if enum_body else []

            structure.enums.append(
                {
                    "name": enum_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "visibility": visibility,
                    "implements": self._parse_implements_list(implements) if implements else [],
                    "values": values,
                }
            )

        # Extract records (Java 14+)
        record_pattern = (
            r"(?:^|\n)\s*(?:(public|private|protected)\s+)?record\s+(\w+)\s*\(([^)]*)\)"
        )

        for match in re.finditer(record_pattern, content):
            visibility = match.group(1) or "package-private"
            record_name = match.group(2)
            components = match.group(3)

            structure.records.append(
                {
                    "name": record_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "visibility": visibility,
                    "components": self._parse_record_components(components),
                }
            )

        # Extract annotations used in the file
        annotation_pattern = r"@(\w+)(?:\([^)]*\))?"
        annotations = set()
        for match in re.finditer(annotation_pattern, content):
            annotations.add(match.group(1))
        structure.annotations = list(annotations)

        # Detect frameworks based on annotations and imports
        structure.framework = self._detect_framework(content, structure.annotations)

        # Count lambda expressions
        lambda_pattern = r"\([^)]*\)\s*->"
        structure.lambda_count = len(re.findall(lambda_pattern, content))

        # Count anonymous classes
        anonymous_pattern = r"new\s+[\w<>]+\s*\([^)]*\)\s*\{"
        structure.anonymous_classes_count = len(re.findall(anonymous_pattern, content))

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Java code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Class coupling
        - Inheritance depth indicators
        - Exception handling complexity

        Args:
            content: Java source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Calculate cyclomatic complexity
        complexity = 1

        decision_keywords = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\belse\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bdo\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\bthrow\b",
            r"\b&&\b",
            r"\|\|",
            r"\?",  # Logical operators and ternary
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        # Add complexity for enhanced for loops
        complexity += len(re.findall(r"for\s*\([^:]+:[^)]+\)", content))

        metrics.cyclomatic = complexity

        # Calculate cognitive complexity
        cognitive = 0
        nesting_level = 0
        max_nesting = 0

        lines = content.split("\n")
        for line in lines:
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Track nesting
            opening_braces = line.count("{")
            closing_braces = line.count("}")
            nesting_level += opening_braces - closing_braces
            max_nesting = max(max_nesting, nesting_level)

            # Control structures with nesting penalty
            control_patterns = [
                (r"\bif\b", 1),
                (r"\belse\s+if\b", 1),
                (r"\belse\b", 0),
                (r"\bfor\b", 1),
                (r"\bwhile\b", 1),
                (r"\bdo\b", 1),
                (r"\bswitch\b", 1),
                (r"\btry\b", 1),
                (r"\bcatch\b", 1),
            ]

            for pattern, weight in control_patterns:
                if re.search(pattern, line):
                    cognitive += weight * (1 + max(0, nesting_level - 1))

            # Exception handling adds complexity
            if re.search(r"\bthrow\s+new\b", line):
                cognitive += 2

            # Nested classes add complexity
            if re.search(r"\bclass\s+\w+\s*\{", line) and nesting_level > 1:
                cognitive += 3

        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting

        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = self._count_code_lines(content)
        metrics.comment_lines = self._count_comment_lines(content)
        metrics.comment_ratio = (
            metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
        )

        # Count classes and interfaces
        metrics.class_count = len(re.findall(r"\bclass\s+\w+", content))
        metrics.interface_count = len(re.findall(r"\binterface\s+\w+", content))
        metrics.enum_count = len(re.findall(r"\benum\s+\w+", content))
        metrics.record_count = len(re.findall(r"\brecord\s+\w+", content))

        # Count methods
        method_pattern = r"(?:public|private|protected|static|final|abstract|synchronized|native)\s+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{"
        metrics.method_count = len(re.findall(method_pattern, content))

        # Exception handling metrics
        metrics.try_blocks = len(re.findall(r"\btry\s*\{", content))
        metrics.catch_blocks = len(re.findall(r"\bcatch\s*\([^)]+\)", content))
        metrics.finally_blocks = len(re.findall(r"\bfinally\s*\{", content))
        # Count both method 'throws' declarations and explicit throw statements
        metrics.throws_declarations = len(re.findall(r"\bthrows\s+[\w.,\s]+", content))
        metrics.throws_declarations += len(re.findall(r"\bthrow\s+new\b", content))

        # Annotation metrics
        metrics.annotation_count = len(re.findall(r"@\w+", content))

        # Inheritance metrics
        metrics.extends_count = len(re.findall(r"\bextends\s+\w+", content))
        metrics.implements_count = len(re.findall(r"\bimplements\s+[\w,\s]+", content))

        # Lambda and stream metrics
        metrics.lambda_count = len(re.findall(r"\([^)]*\)\s*->", content))
        # Also count single-arg lambdas without parentheses: x -> x + 1
        metrics.lambda_count += len(re.findall(r"\b[A-Za-z_]\w*\s*->", content))
        metrics.stream_operations = len(
            re.findall(r"\.\s*(?:stream|filter|map|reduce|collect|forEach)\s*\(", content)
        )

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Java's verbosity
            inheritance_factor = 1 - (metrics.extends_count + metrics.implements_count) * 0.05
            exception_factor = 1 - (metrics.try_blocks * 0.02)

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * inheritance_factor
                + 10 * exception_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _categorize_java_import(self, module: str) -> str:
        """Categorize a Java import.

        Args:
            module: Import module path

        Returns:
            Category: 'jdk', 'javax', 'third_party', or 'local'
        """
        if module.startswith("java."):
            return "jdk"
        elif module.startswith("javax."):
            return "javax"
        elif module.startswith("com.") or module.startswith("org.") or module.startswith("net."):
            return "third_party"
        else:
            return "local"

    def _parse_implements_list(self, implements_str: str) -> List[str]:
        """Parse the implements clause into a list of interfaces.

        Args:
            implements_str: String containing comma-separated interfaces

        Returns:
            List of interface names
        """
        if not implements_str:
            return []

        # Handle generic types
        interfaces = []
        current = ""
        depth = 0

        for char in implements_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                if current.strip():
                    interfaces.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            interfaces.append(current.strip())

        return interfaces

    def _extract_class_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a class/interface/enum.

        Args:
            content: Full file content
            start_pos: Position after class declaration

        Returns:
            Class body content or None
        """
        # Find the opening brace
        brace_pos = content.find("{", start_pos)
        if brace_pos == -1:
            return None

        # Find matching closing brace
        brace_count = 1
        pos = brace_pos + 1

        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            return content[brace_pos + 1 : pos - 1]

        return None

    def _extract_methods(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract methods from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of method information
        """
        methods = []

        # Method pattern with modifiers
        method_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?(?:(static|final|abstract|synchronized|native)\s+)*(?:<[^>]+>\s+)?(?:([\w<>\[\]]+)\s+)?(\w+)\s*\(([^)]*)\)(?:\s+throws\s+([\w,\s]+))?"

        for match in re.finditer(method_pattern, class_body):
            return_type = match.group(3)
            method_name = match.group(4)

            # Filter out non-methods
            if method_name in [
                "if",
                "for",
                "while",
                "switch",
                "new",
                "return",
                "throw",
                "class",
                "interface",
            ]:
                continue

            # Check if it's a constructor (no return type, name matches class name)
            is_constructor = return_type is None or return_type == method_name

            methods.append(
                {
                    "name": method_name,
                    "visibility": match.group(1) or "package-private",
                    "modifiers": match.group(2).split() if match.group(2) else [],
                    "return_type": return_type if not is_constructor else None,
                    "parameters": self._parse_parameters(match.group(5)),
                    "throws": match.group(6).split(",") if match.group(6) else [],
                    "is_constructor": is_constructor,
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return methods

    def _extract_fields(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract fields from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of field information
        """
        fields = []

        # Field pattern
        field_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?(?:(static|final|volatile|transient)\s+)*([\w<>\[\]]+)\s+(\w+)\s*(?:=\s*([^;]+))?\s*;"

        for match in re.finditer(field_pattern, class_body):
            field_type = match.group(3)
            field_name = match.group(4)

            # Filter out method-like patterns
            if field_type in ["if", "for", "while", "return", "throw"]:
                continue

            fields.append(
                {
                    "name": field_name,
                    "type": field_type,
                    "visibility": match.group(1) or "package-private",
                    "modifiers": match.group(2).split() if match.group(2) else [],
                    "initial_value": match.group(5).strip() if match.group(5) else None,
                    "is_constant": "static" in (match.group(2) or "")
                    and "final" in (match.group(2) or ""),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return fields

    def _extract_inner_classes(self, class_body: str) -> List[str]:
        """Extract inner class names from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of inner class names
        """
        inner_classes = []

        # Inner class pattern
        inner_pattern = r"(?:static\s+)?(?:class|interface|enum)\s+(\w+)"

        for match in re.finditer(inner_pattern, class_body):
            inner_classes.append(match.group(1))

        return inner_classes

    def _extract_interface_methods(self, interface_body: str) -> List[Dict[str, Any]]:
        """Extract method signatures from interface body.

        Args:
            interface_body: Content of interface body

        Returns:
            List of method signatures
        """
        methods = []

        # Interface method pattern (can have default/static implementations)
        method_pattern = r"(?:^|\n)\s*(?:(default|static)\s+)?(?:<[^>]+>\s+)?(?:([\w<>\[\]]+)\s+)?(\w+)\s*\(([^)]*)\)"

        for match in re.finditer(method_pattern, interface_body):
            return_type = match.group(2)
            method_name = match.group(3)

            if method_name not in ["if", "for", "while", "switch"]:
                methods.append(
                    {
                        "name": method_name,
                        "return_type": return_type,
                        "parameters": self._parse_parameters(match.group(4)),
                        "is_default": match.group(1) == "default",
                        "is_static": match.group(1) == "static",
                    }
                )

        return methods

    def _extract_enum_values(self, enum_body: str) -> List[str]:
        """Extract enum constant values.

        Args:
            enum_body: Content of enum body

        Returns:
            List of enum constant names
        """
        values = []

        # Enum values are typically at the beginning, before any methods
        # Pattern: VALUE1, VALUE2(args), VALUE3
        enum_section = enum_body.split(";")[0] if ";" in enum_body else enum_body

        # Simple pattern for enum constants
        for match in re.finditer(r"\b([A-Z_][A-Z0-9_]*)\b", enum_section):
            values.append(match.group(1))

        return values

    def _parse_record_components(self, components_str: str) -> List[Dict[str, str]]:
        """Parse record components.

        Args:
            components_str: String containing record components

        Returns:
            List of component dictionaries
        """
        components = []

        if not components_str.strip():
            return components

        # Simple parsing - can be enhanced for annotations
        for component in components_str.split(","):
            component = component.strip()
            if component:
                parts = component.split()
                if len(parts) >= 2:
                    components.append({"type": parts[-2], "name": parts[-1]})

        return components

    def _parse_parameters(self, params_str: str) -> List[Dict[str, str]]:
        """Parse method parameters.

        Args:
            params_str: String containing method parameters

        Returns:
            List of parameter dictionaries
        """
        parameters = []

        if not params_str.strip():
            return parameters

        # Handle complex parameters with generics
        current_param = ""
        depth = 0

        for char in params_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                if current_param.strip():
                    parts = current_param.strip().split()
                    if len(parts) >= 2:
                        # Handle varargs
                        param_type = " ".join(parts[:-1])
                        param_name = parts[-1]
                        parameters.append(
                            {
                                "type": param_type,
                                "name": param_name,
                                "is_varargs": "..." in param_type,
                            }
                        )
                current_param = ""
                continue
            current_param += char

        # Add last parameter
        if current_param.strip():
            parts = current_param.strip().split()
            if len(parts) >= 2:
                param_type = " ".join(parts[:-1])
                param_name = parts[-1]
                parameters.append(
                    {"type": param_type, "name": param_name, "is_varargs": "..." in param_type}
                )

        return parameters

    def _detect_framework(self, content: str, annotations: List[str]) -> Optional[str]:
        """Detect which framework is being used.

        Args:
            content: Java source code
            annotations: List of annotations found

        Returns:
            Framework name or None
        """
        # Spring Framework
        spring_annotations = {
            "Controller",
            "Service",
            "Repository",
            "Component",
            "Autowired",
            "Bean",
            "Configuration",
            "RestController",
            "RequestMapping",
            "SpringBootApplication",
        }
        if spring_annotations.intersection(set(annotations)):
            return "Spring"

        # JUnit
        junit_annotations = {
            "Test",
            "Before",
            "After",
            "BeforeClass",
            "AfterClass",
            "BeforeEach",
            "AfterEach",
            "BeforeAll",
            "AfterAll",
        }
        if junit_annotations.intersection(set(annotations)):
            return "JUnit"

        # JAX-RS (REST)
        jaxrs_annotations = {"Path", "GET", "POST", "PUT", "DELETE", "Produces", "Consumes"}
        if jaxrs_annotations.intersection(set(annotations)):
            return "JAX-RS"

        # JPA/Hibernate
        jpa_annotations = {
            "Entity",
            "Table",
            "Id",
            "GeneratedValue",
            "Column",
            "OneToMany",
            "ManyToOne",
        }
        if jpa_annotations.intersection(set(annotations)):
            return "JPA/Hibernate"

        # Android
        if re.search(r"import\s+android\.", content):
            return "Android"

        # JavaFX
        if re.search(r"import\s+javafx\.", content):
            return "JavaFX"

        return None

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: Java source code

        Returns:
            Number of code lines
        """
        count = 0
        in_multiline_comment = False
        in_string = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Handle multiline comments
            if "/*" in line and not in_string:
                in_multiline_comment = True
                # Check if comment ends on same line
                if "*/" in line:
                    in_multiline_comment = False
                    # Count if there's code after comment
                    after_comment = line.split("*/")[1].strip()
                    if after_comment and not after_comment.startswith("//"):
                        count += 1
                continue

            if in_multiline_comment:
                if "*/" in line:
                    in_multiline_comment = False
                    # Count if there's code after comment
                    after_comment = line.split("*/")[1].strip()
                    if after_comment and not after_comment.startswith("//"):
                        count += 1
                continue

            # Skip empty lines and single-line comments
            if stripped and not stripped.startswith("//"):
                count += 1

        return count

    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines including Javadoc.

        Args:
            content: Java source code

        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_comment = False
        in_javadoc = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Single-line comments
            if stripped.startswith("//"):
                count += 1
                continue

            # Javadoc comments
            if stripped.startswith("/**"):
                in_javadoc = True
                count += 1
                if "*/" in line:
                    in_javadoc = False
                continue

            # Regular multi-line comments
            if "/*" in line and not in_javadoc:
                count += 1
                in_multiline_comment = True
                if "*/" in line:
                    in_multiline_comment = False
                continue

            if in_multiline_comment or in_javadoc:
                count += 1
                if "*/" in line:
                    in_multiline_comment = False
                    in_javadoc = False

        return count
