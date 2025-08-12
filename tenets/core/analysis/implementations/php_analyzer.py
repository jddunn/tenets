"""PHP code analyzer.

This module provides comprehensive analysis for PHP source files,
including support for modern PHP features, namespaces, and frameworks.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from ..base import LanguageAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)
from tenets.utils.logger import get_logger


class PhpAnalyzer(LanguageAnalyzer):
    """PHP code analyzer.

    Provides analysis for PHP files including:
    - Include/require analysis with variations
    - Namespace and use statement handling
    - Class, trait, and interface extraction
    - Function and method analysis with type hints
    - Property analysis with visibility
    - PHP 7+ features (typed properties, return types)
    - PHP 8+ features (attributes, union types, enums)
    - Framework detection (Laravel, Symfony, WordPress)
    - Composer dependency analysis

    Handles both procedural and object-oriented PHP code.
    """

    language_name = "php"
    file_extensions = [
        ".php",
        ".phtml",
        ".php3",
        ".php4",
        ".php5",
        ".php7",
        ".php8",
        ".phps",
        ".inc",
    ]

    def __init__(self):
        """Initialize the PHP analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract imports from PHP code.

        Handles:
        - use statements (classes, functions, constants)
        - include/require statements
        - include_once/require_once
        - Composer autoload
        - Namespace imports

        Args:
            content: PHP source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        # Track current namespace
        current_namespace = None

        for i, line in enumerate(lines, 1):
            # Skip comments
            if (
                line.strip().startswith("//")
                or line.strip().startswith("/*")
                or line.strip().startswith("*")
            ):
                continue

            # Namespace declaration
            namespace_match = re.match(r"^\s*namespace\s+([\w\\]+)\s*;", line)
            if namespace_match:
                current_namespace = namespace_match.group(1)
                continue

            # Use statements
            use_pattern = re.compile(
                r"^\s*use\s+((?:function|const)\s+)?([\w\\]+)(?:\s+as\s+(\w+))?\s*;"
            )
            match = use_pattern.match(line)
            if match:
                import_type = match.group(1).strip() if match.group(1) else "class"
                module = match.group(2)
                alias = match.group(3)

                imports.append(
                    ImportInfo(
                        module=module,
                        alias=alias,
                        line=i,
                        type=f"use_{import_type}",
                        is_relative=False,
                        namespace=current_namespace,
                        import_type=import_type,
                    )
                )
                continue

            # Group use statements (PHP 7+)
            group_use_pattern = re.compile(r"^\s*use\s+([\w\\]+)\\{([^}]+)}\s*;")
            match = group_use_pattern.match(line)
            if match:
                base_namespace = match.group(1)
                imports_list = match.group(2)

                for item in imports_list.split(","):
                    item = item.strip()
                    if " as " in item:
                        name, alias = item.split(" as ")
                        name = name.strip()
                        alias = alias.strip()
                    else:
                        name = item
                        alias = None

                    imports.append(
                        ImportInfo(
                            module=f"{base_namespace}\\{name}",
                            alias=alias,
                            line=i,
                            type="use_group",
                            is_relative=False,
                            namespace=current_namespace,
                        )
                    )
                continue

            # Include/require patterns
            include_patterns = [
                (r'include\s+[\'"]([^\'"]+)[\'"]', "include"),
                (r'include_once\s+[\'"]([^\'"]+)[\'"]', "include_once"),
                (r'require\s+[\'"]([^\'"]+)[\'"]', "require"),
                (r'require_once\s+[\'"]([^\'"]+)[\'"]', "require_once"),
                (r'include\s*\(?\s*[\'"]([^\'"]+)[\'"]\s*\)?', "include"),
                (r'require\s*\(?\s*[\'"]([^\'"]+)[\'"]\s*\)?', "require"),
            ]

            for pattern, include_type in include_patterns:
                match = re.search(pattern, line)
                if match:
                    file_path_str = match.group(1)
                    imports.append(
                        ImportInfo(
                            module=file_path_str,
                            line=i,
                            type=include_type,
                            is_relative=not file_path_str.startswith("/"),
                            is_file_include=True,
                        )
                    )
                    break

            # Dynamic includes with variables or path expressions
            dynamic_include = re.search(r"(?:include|require)(?:_once)?\s*\(?\s*\$\w+", line)
            dynamic_dir_include = re.search(
                r"(?:include|require)(?:_once)?\s*\(?\s*(?:__DIR__|dirname\s*\(\s*__FILE__\s*\))",
                line,
            )
            concat_include = re.search(r"(?:include|require)(?:_once)?[^;]*\.[^;]*;", line)
            if dynamic_include or dynamic_dir_include or concat_include:
                imports.append(
                    ImportInfo(
                        module="<dynamic>",
                        line=i,
                        type="dynamic_include",
                        is_relative=False,
                        is_dynamic=True,
                    )
                )

            # Composer autoload
            if "vendor/autoload.php" in line:
                imports.append(
                    ImportInfo(
                        module="composer_autoload",
                        line=i,
                        type="composer",
                        is_relative=False,
                        is_autoload=True,
                    )
                )

        # Check for composer.json dependencies
        if file_path.name.lower() == "composer.json":
            imports.extend(self._extract_composer_dependencies(content))

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract public members from PHP code.

        PHP doesn't have explicit exports, but public classes, functions,
        and constants are accessible from other files.

        Args:
            content: PHP source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported (public) symbols
        """
        exports = []

        # Extract namespace
        namespace_match = re.search(r"^\s*namespace\s+([\w\\]+)\s*;", content, re.MULTILINE)
        namespace = namespace_match.group(1) if namespace_match else None

        # Public classes
        class_pattern = r"(?:^|\n)\s*(?:(abstract|final)\s+)?class\s+(\w+)(?:\s+extends\s+([\w\\]+))?(?:\s+implements\s+([\w\\,\s]+))?"

        for match in re.finditer(class_pattern, content):
            modifiers = []
            if match.group(1):
                modifiers.append(match.group(1))

            exports.append(
                {
                    "name": match.group(2),
                    "type": "class",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "modifiers": modifiers,
                    "extends": match.group(3),
                    "implements": (
                        self._parse_implements_list(match.group(4)) if match.group(4) else []
                    ),
                }
            )

        # Interfaces
        interface_pattern = r"(?:^|\n)\s*interface\s+(\w+)(?:\s+extends\s+([\w\\,\s]+))?"

        for match in re.finditer(interface_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "interface",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "extends": (
                        self._parse_implements_list(match.group(2)) if match.group(2) else []
                    ),
                }
            )

        # Traits
        trait_pattern = r"(?:^|\n)\s*trait\s+(\w+)"

        for match in re.finditer(trait_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "trait",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        # Enums (PHP 8.1+)
        enum_pattern = r"(?:^|\n)\s*enum\s+(\w+)(?:\s*:\s*(\w+))?"

        for match in re.finditer(enum_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "enum",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "backed_type": match.group(2),
                }
            )

        # Global functions
        function_pattern = r"(?:^|\n)\s*function\s+(\w+)\s*\("

        # Track if we're inside a class
        class_ranges = []
        for match in re.finditer(r"(?:class|trait|interface)\s+\w+[^{]*\{", content):
            start = match.end()
            brace_count = 1
            pos = start

            while pos < len(content) and brace_count > 0:
                if content[pos] == "{":
                    brace_count += 1
                elif content[pos] == "}":
                    brace_count -= 1
                pos += 1

            class_ranges.append((start, pos))

        for match in re.finditer(function_pattern, content):
            func_pos = match.start()

            # Check if function is inside a class
            is_inside_class = any(start <= func_pos < end for start, end in class_ranges)

            if not is_inside_class:
                exports.append(
                    {
                        "name": match.group(1),
                        "type": "function",
                        "line": content[: match.start()].count("\n") + 1,
                        "namespace": namespace,
                    }
                )

        # Constants
        const_pattern = r"(?:^|\n)\s*const\s+(\w+)\s*="
        define_pattern = r'define\s*\(\s*[\'"](\w+)[\'"]'

        for match in re.finditer(const_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "constant",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        for match in re.finditer(define_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "constant",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "defined_with": "define",
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from PHP file.

        Extracts:
        - Namespace declaration
        - Classes with inheritance and traits
        - Interfaces with extension
        - Traits with composition
        - Enums (PHP 8.1+)
        - Functions with type hints
        - Properties with visibility and types
        - Methods with return types
        - PHP attributes/annotations

        Args:
            content: PHP source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Extract namespace
        namespace_match = re.search(r"^\s*namespace\s+([\w\\]+)\s*;", content, re.MULTILINE)
        if namespace_match:
            structure.namespace = namespace_match.group(1)

        # Extract classes
        class_pattern = r"(?:^|\n)\s*(?:(abstract|final)\s+)?class\s+(\w+)(?:\s+extends\s+([\w\\]+))?(?:\s+implements\s+([\w\\,\s]+))?"

        for match in re.finditer(class_pattern, content):
            class_name = match.group(2)
            modifiers = []
            if match.group(1):
                modifiers.append(match.group(1))

            extends = match.group(3)
            implements = self._parse_implements_list(match.group(4)) if match.group(4) else []

            # Find class body
            class_body = self._extract_block_body(content, match.end())

            # Extract class components
            methods = []
            properties = []
            traits_used = []
            constants = []

            if class_body:
                methods = self._extract_methods(class_body)
                properties = self._extract_properties(class_body)
                traits_used = self._extract_used_traits(class_body)
                constants = self._extract_class_constants(class_body)

            class_info = ClassInfo(
                name=class_name,
                line=content[: match.start()].count("\n") + 1,
                modifiers=modifiers,
                bases=[extends] if extends else [],
                interfaces=implements,
                methods=methods,
                properties=properties,
                traits_used=traits_used,
                constants=constants,
            )

            structure.classes.append(class_info)

        # Extract interfaces
        interface_pattern = r"(?:^|\n)\s*interface\s+(\w+)(?:\s+extends\s+([\w\\,\s]+))?"

        for match in re.finditer(interface_pattern, content):
            interface_name = match.group(1)
            extends = self._parse_implements_list(match.group(2)) if match.group(2) else []

            # Extract interface methods
            interface_body = self._extract_block_body(content, match.end())
            methods = self._extract_interface_methods(interface_body) if interface_body else []

            structure.interfaces.append(
                {
                    "name": interface_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "extends": extends,
                    "methods": methods,
                }
            )

        # Extract traits
        trait_pattern = r"(?:^|\n)\s*trait\s+(\w+)"

        for match in re.finditer(trait_pattern, content):
            trait_name = match.group(1)

            # Extract trait body
            trait_body = self._extract_block_body(content, match.end())
            methods = []
            properties = []
            traits_used = []

            if trait_body:
                methods = self._extract_methods(trait_body)
                properties = self._extract_properties(trait_body)
                traits_used = self._extract_used_traits(trait_body)

            structure.traits.append(
                {
                    "name": trait_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "methods": methods,
                    "properties": properties,
                    "uses": traits_used,
                }
            )

        # Extract enums (PHP 8.1+)
        enum_pattern = r"(?:^|\n)\s*enum\s+(\w+)(?:\s*:\s*(\w+))?"

        for match in re.finditer(enum_pattern, content):
            enum_name = match.group(1)
            backed_type = match.group(2)

            # Extract enum cases
            enum_body = self._extract_block_body(content, match.end())
            cases = self._extract_enum_cases(enum_body) if enum_body else []

            structure.enums.append(
                {
                    "name": enum_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "backed_type": backed_type,
                    "cases": cases,
                }
            )

        # Extract global functions
        structure.functions = self._extract_global_functions(content)

        # Extract global constants
        const_pattern = r"(?:^|\n)\s*const\s+(\w+)\s*="
        for match in re.finditer(const_pattern, content):
            structure.constants.append(match.group(1))

        define_pattern = r'define\s*\(\s*[\'"](\w+)[\'"]'
        for match in re.finditer(define_pattern, content):
            structure.constants.append(match.group(1))

        # Extract global variables
        global_var_pattern = r'\$GLOBALS\[[\'"](\w+)[\'"]\]'
        for match in re.finditer(global_var_pattern, content):
            structure.global_variables.append(f"${match.group(1)}")

        # Detect superglobals usage
        superglobals = [
            "$_GET",
            "$_POST",
            "$_SESSION",
            "$_COOKIE",
            "$_FILES",
            "$_SERVER",
            "$_ENV",
            "$_REQUEST",
        ]
        structure.superglobals_used = [sg for sg in superglobals if sg in content]

        # Detect framework
        structure.framework = self._detect_framework(content, file_path)

        # Check for test file
        structure.is_test_file = (
            "Test.php" in file_path.name
            or "test.php" in file_path.name.lower()
            or file_path.parts
            and "tests" in file_path.parts
        )

        # Count anonymous functions/closures
        structure.closure_count = len(
            re.findall(r"function\s*\([^)]*\)\s*(?:use\s*\([^)]*\))?\s*\{", content)
        )

        # Count arrow functions (PHP 7.4+)
        structure.arrow_function_count = len(re.findall(r"fn\s*\([^)]*\)\s*=>", content))

        # Count anonymous classes
        structure.anonymous_classes_count = len(
            re.findall(r"new\s+class(?:\s*\([^)]*\))?\s*(?:extends\s+[\w\\]+)?\s*(?:implements\s+[\w\\,\s]+)?\s*\{", content)
        )

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for PHP code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Nesting depth
        - Class coupling
        - PHP-specific metrics

        Args:
            content: PHP source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Calculate cyclomatic complexity
        complexity = 1

        decision_keywords = [
            r"\bif\b",
            r"\belseif\b",
            r"\belse\b",
            r"\bwhile\b",
            r"\bfor\b",
            r"\bforeach\b",
            r"\bdo\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\bfinally\b",
            r"\b\?\s*[^:]+\s*:",
            r"\b&&\b",
            r"\|\|",
            r"\band\b",
            r"\bor\b",
            r"\bxor\b",
            r"\?\?",  # Null coalescing operator
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        # Add complexity for match expressions (PHP 8+)
        complexity += len(re.findall(r"\bmatch\s*\(", content))

        metrics.cyclomatic = complexity

        # Calculate cognitive complexity
        cognitive = 0
        nesting_level = 0
        max_nesting = 0

        lines = content.split("\n")
        for line in lines:
            # Skip comments
            if (
                line.strip().startswith("//")
                or line.strip().startswith("/*")
                or line.strip().startswith("*")
            ):
                continue

            # Track nesting
            opening_braces = line.count("{")
            closing_braces = line.count("}")
            nesting_level += opening_braces - closing_braces
            max_nesting = max(max_nesting, nesting_level)

            # Control structures with nesting penalty
            control_patterns = [
                (r"\bif\b", 1),
                (r"\belseif\b", 1),
                (r"\belse\b", 0),
                (r"\bfor\b", 1),
                (r"\bforeach\b", 1),
                (r"\bwhile\b", 1),
                (r"\bswitch\b", 1),
                (r"\btry\b", 1),
                (r"\bcatch\b", 1),
                (r"\bmatch\b", 1),
            ]

            for pattern, weight in control_patterns:
                if re.search(pattern, line):
                    cognitive += weight * (1 + max(0, nesting_level - 1))

        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting

        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = self._count_code_lines(content)
        metrics.comment_lines = self._count_comment_lines(content)
        metrics.comment_ratio = (
            metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
        )

        # Count classes, interfaces, traits
        metrics.class_count = len(re.findall(r"\bclass\s+\w+", content))
        metrics.interface_count = len(re.findall(r"\binterface\s+\w+", content))
        metrics.trait_count = len(re.findall(r"\btrait\s+\w+", content))
        metrics.enum_count = len(re.findall(r"\benum\s+\w+", content))

        # Count functions/methods
        metrics.function_count = len(re.findall(r"\bfunction\s+\w+\s*\(", content))

        # Exception handling metrics
        metrics.try_blocks = len(re.findall(r"\btry\s*\{", content))
        metrics.catch_blocks = len(re.findall(r"\bcatch\s*\([^)]+\)", content))
        metrics.finally_blocks = len(re.findall(r"\bfinally\s*\{", content))
        metrics.throw_statements = len(re.findall(r"\bthrow\s+new\s+", content))

        # PHP-specific metrics
        metrics.global_usage = len(re.findall(r"\$GLOBALS\[", content))
        metrics.superglobal_usage = len(
            re.findall(r"\$_(?:GET|POST|SESSION|COOKIE|FILES|SERVER|ENV|REQUEST)\[", content)
        )
        metrics.eval_usage = len(re.findall(r"\beval\s*\(", content))
        metrics.dynamic_calls = len(re.findall(r"\$\w+\s*\(", content))  # Variable functions

        # Type hint metrics
        metrics.type_hints = len(
            re.findall(
                r":\s*(?:\?)?(?:int|string|bool|float|array|object|callable|iterable|mixed|void|self|parent|static|[\w\\]+)",
                content,
            )
        )
        metrics.nullable_types = len(
            re.findall(
                r"\?(?:int|string|bool|float|array|object|callable|iterable|mixed|[\w\\]+)", content
            )
        )
        metrics.union_types = len(re.findall(r":\s*[\w\\]+\|[\w\\]+", content))

        # Attribute/Annotation metrics
        metrics.attributes = len(re.findall(r"#\[[\w\\]+", content))
        metrics.doc_comments = len(re.findall(r"/\*\*", content))

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for PHP
            global_factor = 1 - (metrics.global_usage + metrics.superglobal_usage) * 0.01
            type_factor = (
                min(1.0, metrics.type_hints / metrics.function_count)
                if metrics.function_count > 0
                else 0
            )

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * global_factor
                + 10 * type_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _extract_composer_dependencies(self, content: str) -> List[ImportInfo]:
        """Extract dependencies from composer.json.

        Args:
            content: composer.json content

        Returns:
            List of ImportInfo objects for dependencies
        """
        imports = []

        try:
            import json

            composer_data = json.loads(content)

            # Extract require dependencies
            if "require" in composer_data:
                for package, version in composer_data["require"].items():
                    if not package.startswith("php") and not package.startswith("ext-"):
                        imports.append(
                            ImportInfo(
                                module=package,
                                version=version,
                                type="composer_require",
                                is_relative=False,
                                is_dependency=True,
                            )
                        )

            # Extract require-dev dependencies
            if "require-dev" in composer_data:
                for package, version in composer_data["require-dev"].items():
                    imports.append(
                        ImportInfo(
                            module=package,
                            version=version,
                            type="composer_require_dev",
                            is_relative=False,
                            is_dev_dependency=True,
                        )
                    )
        except Exception as e:
            self.logger.debug(f"Failed to parse composer.json: {e}")

        return imports

    def _parse_implements_list(self, implements_str: str) -> List[str]:
        """Parse implements/extends list.

        Args:
            implements_str: String with comma-separated interfaces

        Returns:
            List of interface names
        """
        if not implements_str:
            return []

        interfaces = []
        for interface in implements_str.split(","):
            interface = interface.strip()
            if interface:
                interfaces.append(interface)

        return interfaces

    def _extract_block_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a class/interface/trait block.

        Args:
            content: Source code
            start_pos: Position after declaration

        Returns:
            Block body content or None
        """
        # Find opening brace
        brace_pos = content.find("{", start_pos)
        if brace_pos == -1:
            return None

        # Find matching closing brace
        brace_count = 1
        pos = brace_pos + 1
        in_string = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            # Handle strings
            if not escape_next:
                if char in ['"', "'"]:
                    in_string = not in_string
                elif char == "\\":
                    escape_next = True
                elif not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
            else:
                escape_next = False

            pos += 1

        if brace_count == 0:
            return content[brace_pos + 1 : pos - 1]

        return None

    def _extract_methods(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract methods from class/trait body.

        Args:
            class_body: Content of class/trait body

        Returns:
            List of method information
        """
        methods = []

        # Method pattern with visibility and modifiers
        method_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?(?:(static|final|abstract)\s+)*function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([\?\w\|\\]+))?"

        for match in re.finditer(method_pattern, class_body):
            visibility = match.group(1) or "public"
            modifiers = match.group(2).split() if match.group(2) else []
            method_name = match.group(3)
            parameters = match.group(4)
            return_type = match.group(5)

            methods.append(
                {
                    "name": method_name,
                    "visibility": visibility,
                    "modifiers": modifiers,
                    "parameters": self._parse_parameters(parameters),
                    "return_type": return_type,
                    "line": class_body[: match.start()].count("\n") + 1,
                    "is_constructor": method_name == "__construct",
                    "is_destructor": method_name == "__destruct",
                    "is_magic": method_name.startswith("__"),
                }
            )

        return methods

    def _extract_properties(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract properties from class/trait body.

        Args:
            class_body: Content of class/trait body

        Returns:
            List of property information
        """
        properties = []

        # Property pattern with visibility and type (PHP 7.4+)
        property_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?(?:(static|readonly)\s+)?(?:([\?\w\|\\]+)\s+)?\$(\w+)(?:\s*=\s*([^;]+))?;"

        for match in re.finditer(property_pattern, class_body):
            visibility = match.group(1) or "public"
            modifiers = []
            if match.group(2):
                modifiers = match.group(2).split()

            properties.append(
                {
                    "name": match.group(4),
                    "visibility": visibility,
                    "modifiers": modifiers,
                    "type": match.group(3),
                    "default_value": match.group(5),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return properties

    def _extract_used_traits(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract used traits from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of used traits
        """
        traits = []

        # Simple use statement
        use_pattern = r"^\s*use\s+([\w\\,\s]+);"

        for match in re.finditer(use_pattern, class_body, re.MULTILINE):
            trait_list = match.group(1)
            for trait in trait_list.split(","):
                trait = trait.strip()
                if trait:
                    traits.append(
                        {"name": trait, "line": class_body[: match.start()].count("\n") + 1}
                    )

        # Trait with adaptations
        use_block_pattern = r"^\s*use\s+([\w\\,\s]+)\s*\{"

        for match in re.finditer(use_block_pattern, class_body, re.MULTILINE):
            trait_list = match.group(1)
            for trait in trait_list.split(","):
                trait = trait.strip()
                if trait:
                    traits.append(
                        {
                            "name": trait,
                            "line": class_body[: match.start()].count("\n") + 1,
                            "has_adaptations": True,
                        }
                    )

        return traits

    def _extract_class_constants(self, class_body: str) -> List[Dict[str, str]]:
        """Extract class constants.

        Args:
            class_body: Content of class body

        Returns:
            List of constant information
        """
        constants = []

        const_pattern = r"(?:^|\n)\s*(?:(public|private|protected)\s+)?const\s+(\w+)\s*="

        for match in re.finditer(const_pattern, class_body):
            visibility = match.group(1) or "public"
            constants.append(
                {
                    "name": match.group(2),
                    "visibility": visibility,
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return constants

    def _extract_interface_methods(self, interface_body: str) -> List[Dict[str, Any]]:
        """Extract method signatures from interface body.

        Args:
            interface_body: Content of interface body

        Returns:
            List of method signatures
        """
        methods = []

        # Interface method pattern
        method_pattern = (
            r"(?:^|\n)\s*(?:public\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([\?\w\|\\]+))?;"
        )

        for match in re.finditer(method_pattern, interface_body):
            methods.append(
                {
                    "name": match.group(1),
                    "parameters": self._parse_parameters(match.group(2)),
                    "return_type": match.group(3),
                }
            )

        return methods

    def _extract_enum_cases(self, enum_body: str) -> List[Dict[str, Any]]:
        """Extract enum cases.

        Args:
            enum_body: Content of enum body

        Returns:
            List of enum cases
        """
        cases = []

        # Enum case pattern
        case_pattern = r"^\s*case\s+(\w+)(?:\s*=\s*([^;]+))?;"

        for match in re.finditer(case_pattern, enum_body, re.MULTILINE):
            cases.append(
                {
                    "name": match.group(1),
                    "value": match.group(2),
                    "line": enum_body[: match.start()].count("\n") + 1,
                }
            )

        return cases

    def _extract_global_functions(self, content: str) -> List[FunctionInfo]:
        """Extract global functions (outside classes).

        Args:
            content: PHP source code

        Returns:
            List of FunctionInfo objects
        """
        functions = []

        # Find all class/trait/interface blocks to exclude
        block_ranges = []
        block_pattern = r"(?:class|trait|interface|enum)\s+\w+[^{]*\{"

        for match in re.finditer(block_pattern, content):
            start = match.end()
            brace_count = 1
            pos = start

            while pos < len(content) and brace_count > 0:
                if content[pos] == "{":
                    brace_count += 1
                elif content[pos] == "}":
                    brace_count -= 1
                pos += 1

            block_ranges.append((match.start(), pos))

        # Extract functions outside blocks
        function_pattern = r"(?:^|\n)\s*function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*([\?\w\|\\]+))?"

        for match in re.finditer(function_pattern, content):
            func_pos = match.start()

            # Check if function is outside all blocks
            is_global = not any(start <= func_pos < end for start, end in block_ranges)

            if is_global:
                functions.append(
                    FunctionInfo(
                        name=match.group(1),
                        line=content[: match.start()].count("\n") + 1,
                        parameters=self._parse_parameters(match.group(2)),
                        return_type=match.group(3),
                    )
                )

        return functions

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter information
        """
        parameters = []

        if not params_str.strip():
            return parameters

        # Split by comma, handling nested types
        params = []
        current_param = ""
        depth = 0

        for char in params_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                params.append(current_param.strip())
                current_param = ""
                continue
            current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        # Parse each parameter
        param_pattern = r"(?:([\?\w\|\\]+)\s+)?(?:&)?(?:\.\.\.)?\$(\w+)(?:\s*=\s*(.+))?"

        for param in params:
            match = re.match(param_pattern, param)
            if match:
                parameters.append(
                    {
                        "type": match.group(1),
                        "name": match.group(2),
                        "default": match.group(3),
                        "is_reference": "&" in param,
                        "is_variadic": "..." in param,
                    }
                )

        return parameters

    def _detect_framework(self, content: str, file_path: Path) -> Optional[str]:
        """Detect which framework is being used.

        Args:
            content: PHP source code
            file_path: Path to the file

        Returns:
            Framework name or None
        """
        # Symfony indicators (check first to avoid conflicts)
        symfony_indicators = [
            r"namespace\s+App\\Controller",
            r"use\s+Symfony\\",
            r"extends\s+AbstractController",
            r"#\[Route\(",
            r"@Route\(",
            r"Doctrine\\ORM",
        ]

        for pattern in symfony_indicators:
            if re.search(pattern, content):
                return "Symfony"

        # Laravel indicators
        laravel_indicators = [
            r"namespace\s+App\\",
            r"use\s+Illuminate\\",
            r"extends\s+Controller(?!\\)",  # More specific to avoid AbstractController
            r"extends\s+Model",
            r"class\s+\w+\s+extends\s+Migration",
            r"Route::",
            r"Eloquent",
            r"Blade",
        ]

        for pattern in laravel_indicators:
            if re.search(pattern, content):
                return "Laravel"

        # WordPress indicators
        wordpress_indicators = [
            r"add_action\s*\(",
            r"add_filter\s*\(",
            r"wp_enqueue_",
            r"get_option\s*\(",
            r"WP_Query",
            r"\$wpdb",
            r"Plugin Name:",
        ]

        for pattern in wordpress_indicators:
            if re.search(pattern, content):
                return "WordPress"

        # CodeIgniter indicators
        if re.search(r"extends\s+CI_Controller", content) or re.search(r"\$this->load->", content):
            return "CodeIgniter"

        # Yii indicators
        if re.search(r"namespace\s+app\\", content) or re.search(r"use\s+yii\\", content):
            return "Yii"

        # Composer project
        if "vendor/autoload.php" in content:
            return "Composer"

        return None

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: PHP source code

        Returns:
            Number of code lines
        """
        count = 0
        in_multiline_comment = False
        in_string = False
        string_delimiter = None

        for line in content.split("\n"):
            stripped = line.strip()

            # Handle multiline comments
            if "/*" in line and not in_string:
                in_multiline_comment = True
                if "*/" in line:
                    in_multiline_comment = False
                    after = line.split("*/")[1].strip()
                    if after and not after.startswith("//"):
                        count += 1
                continue

            if in_multiline_comment:
                if "*/" in line:
                    in_multiline_comment = False
                    after = line.split("*/")[1].strip()
                    if after and not after.startswith("//"):
                        count += 1
                continue

            # Skip empty lines and single-line comments
            if stripped and not stripped.startswith("//"):
                count += 1

        return count

    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines including PHPDoc.

        Args:
            content: PHP source code

        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_comment = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Single-line comments
            if stripped.startswith("//") or stripped.startswith("#"):
                count += 1
                continue

            # Multi-line comments and PHPDoc
            if "/*" in line:
                count += 1
                in_multiline_comment = True
                if "*/" in line:
                    in_multiline_comment = False
                continue

            if in_multiline_comment:
                count += 1
                if "*/" in line:
                    in_multiline_comment = False

        return count
