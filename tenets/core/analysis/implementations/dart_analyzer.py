"""Dart code analyzer with Flutter support.

This module provides comprehensive analysis for Dart source files,
including support for Flutter-specific patterns, null safety,
async programming, and modern Dart features.
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


class DartAnalyzer(LanguageAnalyzer):
    """Dart code analyzer with Flutter support.

    Provides comprehensive analysis for Dart files including:
    - Import and export directives
    - Part and library declarations
    - Classes with mixins and extensions
    - Null safety features (?, !, late)
    - Async/await, Future, and Stream handling
    - Flutter widgets and lifecycle methods
    - Factory and named constructors
    - Extension methods
    - Annotations and metadata
    - Generics and type parameters

    Supports Dart 2.x with null safety and Flutter framework patterns.
    """

    language_name = "dart"
    file_extensions = [".dart"]

    def __init__(self):
        """Initialize the Dart analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract import, export, part, and library directives from Dart code.

        Handles:
        - import statements: import 'package:flutter/material.dart';
        - export statements: export 'src/widget.dart';
        - part statements: part 'implementation.dart';
        - part of statements: part of 'library.dart';
        - library declarations: library my_library;
        - Conditional imports: import 'stub.dart' if (dart.library.io) 'io.dart';
        - Show/hide clauses: import 'dart:math' show Random hide PI;
        - Deferred imports: import 'big_lib.dart' deferred as big;

        Args:
            content: Dart source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//"):
                continue

            # Import statements - handle more complex patterns with show/hide
            # First, try to extract the basic import and parse show/hide separately
            basic_import_pattern = r"^\s*import\s+['\"]([^'\"]+)['\"](?:\s+if\s*\([^)]+\)\s*['\"][^'\"]+['\"]*)?(?:\s+deferred)?(?:\s+as\s+(\w+))?(.*?);"
            match = re.match(basic_import_pattern, line)
            if match:
                module_path = match.group(1)
                alias = match.group(2)
                show_hide_part = match.group(3) if match.group(3) else ""

                # Parse show/hide clauses
                show_symbols = []
                hide_symbols = []
                is_deferred = "deferred" in line

                # Extract show clause
                show_match = re.search(r"\bshow\s+([^;]+?)(?:\s+hide|$)", show_hide_part + " ")
                if show_match:
                    show_symbols = self._parse_symbols(show_match.group(1))

                # Extract hide clause
                hide_match = re.search(r"\bhide\s+([^;]+?)(?:\s+show|$)", show_hide_part + " ")
                if hide_match:
                    hide_symbols = self._parse_symbols(hide_match.group(1))

                # Determine import type
                import_type = "import"
                is_package = module_path.startswith("package:")
                is_dart_core = module_path.startswith("dart:")
                is_relative = module_path.startswith("../") or module_path.startswith("./")

                # Categorize the import
                category = self._categorize_import(module_path)

                imports.append(
                    ImportInfo(
                        module=module_path,
                        alias=alias,
                        line=i,
                        type=import_type,
                        is_relative=is_relative,
                        is_package=is_package,
                        is_dart_core=is_dart_core,
                        is_deferred=is_deferred,
                        category=category,
                        show_symbols=show_symbols if show_symbols else [],
                        hide_symbols=hide_symbols if hide_symbols else [],
                    )
                )

            # Export statements
            export_pattern = r"""
                ^\s*export\s+
                ['"]([^'"]+)['"]\s*
                (?:show\s+([^;]+))?\s*
                (?:hide\s+([^;]+))?\s*
                ;
            """
            match = re.match(export_pattern, line, re.VERBOSE)
            if match:
                module_path = match.group(1)
                show_clause = match.group(2)
                hide_clause = match.group(3)

                imports.append(
                    ImportInfo(
                        module=module_path,
                        line=i,
                        type="export",
                        is_relative=not module_path.startswith("package:")
                        and not module_path.startswith("dart:"),
                        show_symbols=self._parse_symbols(show_clause) if show_clause else [],
                        hide_symbols=self._parse_symbols(hide_clause) if hide_clause else [],
                        category=self._categorize_import(module_path),
                    )
                )

            # Part statements
            part_pattern = r"^\s*part\s+['\"]([^'\"]+)['\"]\s*;"
            match = re.match(part_pattern, line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="part",
                        is_relative=True,
                        is_part_file=True,
                    )
                )

            # Part of statements
            part_of_pattern = r"^\s*part\s+of\s+['\"]?([^'\";\s]+)['\"]?\s*;"
            match = re.match(part_of_pattern, line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="part_of",
                        is_relative=False,
                        is_library_part=True,
                    )
                )

            # Library declaration
            library_pattern = r"^\s*library\s+(\w+(?:\.\w+)*)\s*;"
            match = re.match(library_pattern, line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="library",
                        is_relative=False,
                        is_library_declaration=True,
                    )
                )

        # Handle conditional, multi-line imports like:
        # import 'stub.dart'
        #   if (dart.library.io) 'io_implementation.dart'
        #   if (dart.library.html) 'web_implementation.dart';
        cond_import_pattern = (
            r"import\s+['\"]([^'\"]+)['\"]\s*(?:\s*if\s*\([^)]+\)\s*['\"][^'\"]+['\"]\s*)+;"
        )
        for m in re.finditer(cond_import_pattern, content, re.MULTILINE):
            first_module = m.group(1)
            # Avoid duplicates if already added (e.g., if written in one line)
            if not any(imp.module == first_module and imp.type == "import" for imp in imports):
                imports.append(
                    ImportInfo(
                        module=first_module,
                        line=content[: m.start()].count("\n") + 1,
                        type="import",
                        is_relative=first_module.startswith("../") or first_module.startswith("./"),
                        is_package=first_module.startswith("package:"),
                        is_dart_core=first_module.startswith("dart:"),
                        category=self._categorize_import(first_module),
                        conditional=True,
                    )
                )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from Dart code.

        In Dart, exports include:
        - Public classes (not prefixed with _)
        - Public functions
        - Public variables and constants
        - Public typedefs
        - Public enums
        - Extension methods

        Args:
            content: Dart source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols
        """
        exports = []

        # Public classes (including abstract and mixins)
        class_pattern = r"^\s*(?:abstract\s+)?(?:final\s+)?(?:base\s+)?(?:interface\s+)?(?:mixin\s+)?class\s+([A-Z]\w*)"
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)

            modifiers = []
            match_str = match.group(0)
            if "abstract" in match_str:
                modifiers.append("abstract")
            if "final" in match_str:
                modifiers.append("final")
            if "base" in match_str:
                modifiers.append("base")
            if "interface" in match_str:
                modifiers.append("interface")
            if "mixin" in match_str:
                modifiers.append("mixin")

            exports.append(
                {
                    "name": class_name,
                    "type": "class",
                    "line": content[: match.start()].count("\n") + 1,
                    "modifiers": modifiers,
                    "is_public": True,
                }
            )

        # Mixins
        mixin_pattern = r"^\s*(?:base\s+)?mixin\s+([A-Z]\w*)"
        for match in re.finditer(mixin_pattern, content, re.MULTILINE):
            if not any(e["name"] == match.group(1) for e in exports):  # Avoid duplicates
                exports.append(
                    {
                        "name": match.group(1),
                        "type": "mixin",
                        "line": content[: match.start()].count("\n") + 1,
                        "is_public": True,
                    }
                )

        # Public functions (not starting with _), including async*, sync*
        func_pattern = r"^\s*(?:Future<?[^>]*>?\s+|Stream<?[^>]*>?\s+|void\s+|[\w<>]+\s+)?([a-z]\w*)\s*(?:<[^>]+>)?\s*\([^\{]*\)\s*(?:(?:async|sync)\s*\*|async)?\s*(?:=>|\{)"
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            if not func_name.startswith("_"):
                snippet = match.group(0)
                exports.append(
                    {
                        "name": func_name,
                        "type": "function",
                        "line": content[: match.start()].count("\n") + 1,
                        "is_public": True,
                        "is_async": ("async" in snippet),
                    }
                )

        # Public variables and constants
        var_pattern = r"^\s*(?:final\s+|const\s+|late\s+)?(?:static\s+)?(?:final\s+|const\s+)?(?:[\w<>?]+\s+)?([a-z]\w*)\s*(?:=|;)"
        for match in re.finditer(var_pattern, content, re.MULTILINE):
            var_name = match.group(1)
            if not var_name.startswith("_") and var_name not in [
                "if",
                "for",
                "while",
                "return",
                "class",
                "import",
            ]:
                var_type = "constant" if "const" in match.group(0) else "variable"
                exports.append(
                    {
                        "name": var_name,
                        "type": var_type,
                        "line": content[: match.start()].count("\n") + 1,
                        "is_public": True,
                        "is_final": "final" in match.group(0),
                        "is_late": "late" in match.group(0),
                    }
                )

        # Enums
        enum_pattern = r"^\s*enum\s+([A-Z]\w*)"
        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "enum",
                    "line": content[: match.start()].count("\n") + 1,
                    "is_public": True,
                }
            )

        # Typedefs
        typedef_pattern = r"^\s*typedef\s+([A-Z]\w*)"
        for match in re.finditer(typedef_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "typedef",
                    "line": content[: match.start()].count("\n") + 1,
                    "is_public": True,
                }
            )

        # Extension methods
        extension_pattern = r"^\s*extension\s+(?:([A-Z]\w*)\s+)?on\s+([A-Z]\w*)"
        for match in re.finditer(extension_pattern, content, re.MULTILINE):
            extension_name = match.group(1) or f"Extension on {match.group(2)}"
            exports.append(
                {
                    "name": extension_name,
                    "type": "extension",
                    "line": content[: match.start()].count("\n") + 1,
                    "on_type": match.group(2),
                    "is_public": True,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Dart file.

        Extracts:
        - Classes with inheritance, mixins, and interfaces
        - Constructors (default, named, factory)
        - Methods and getters/setters
        - Flutter widgets and lifecycle methods
        - Async functions and streams
        - Extension methods
        - Null safety features
        - Annotations

        Args:
            content: Dart source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Detect if it's a Flutter file
        structure.is_flutter = self._is_flutter_file(content)

        # Extract classes
        class_pattern = r"""
            ^\s*(?:@\w+(?:\([^)]*\))?\s*)*  # Annotations
            (?:(abstract)\s+)?
            (?:(final)\s+)?
            (?:(base)\s+)?
            (?:(interface)\s+)?
            (?:(mixin)\s+)?
            (?:(sealed)\s+)?
            class\s+(\w+)
            (?:<([^>\n{}]*?)>+)?  # Generics (tolerant of nested '>')
            (?:\s+extends\s+([^\{]+?))?
            (?:\s+with\s+([^\{]+?))?
            (?:\s+implements\s+([^\{]+?))?
            \s*\{
        """

        for match in re.finditer(class_pattern, content, re.VERBOSE | re.MULTILINE):
            class_name = match.group(7)

            # Extract class modifiers
            modifiers = []
            if match.group(1):
                modifiers.append("abstract")
            if match.group(2):
                modifiers.append("final")
            if match.group(3):
                modifiers.append("base")
            if match.group(4):
                modifiers.append("interface")
            if match.group(5):
                modifiers.append("mixin")
            if match.group(6):
                modifiers.append("sealed")

            # Parse inheritance
            extends = match.group(9).strip() if match.group(9) else None
            mixins = self._parse_type_list(match.group(10)) if match.group(10) else []
            implements = self._parse_type_list(match.group(11)) if match.group(11) else []

            # Check if it's a Flutter widget
            is_widget = False
            widget_type = None
            if extends:
                # Prefer concrete State<T> first to avoid misclassification
                if re.search(r"\bState<", extends):
                    is_widget = True
                    widget_type = "state"
                elif "StatelessWidget" in extends:
                    is_widget = True
                    widget_type = "stateless"
                elif "StatefulWidget" in extends:
                    is_widget = True
                    widget_type = "stateful"
                elif "InheritedWidget" in extends:
                    is_widget = True
                    widget_type = "inherited"

            # Extract class body
            class_body = self._extract_class_body(content, match.end())

            if class_body:
                # Extract constructors
                constructors = self._extract_constructors(class_body, class_name)

                # Extract methods
                methods = self._extract_methods(class_body)

                # Extract fields
                fields = self._extract_fields(class_body)

                # Extract getters/setters
                properties = self._extract_properties(class_body)
            else:
                constructors = []
                methods = []
                fields = []
                properties = []

            class_info = ClassInfo(
                name=class_name,
                line=content[: match.start()].count("\n") + 1,
                modifiers=modifiers,
                generics=match.group(8),
                bases=[extends] if extends else [],
                mixins=mixins,
                interfaces=implements,
                constructors=constructors,
                methods=methods,
                fields=fields,
                properties=properties,
                is_widget=is_widget,
                widget_type=widget_type,
                is_sealed="sealed" in modifiers,
            )

            # Balance generics angle brackets if regex captured incomplete nested generics
            if class_info.generics:
                try:
                    opens = class_info.generics.count("<")
                    closes = class_info.generics.count(">")
                    if opens > closes:
                        class_info.generics = class_info.generics + (">" * (opens - closes))
                except Exception:
                    pass

            structure.classes.append(class_info)

        # Fallback: capture classes with complex generic bounds that the primary regex may miss
        try:
            existing = {c.name for c in structure.classes}
            complex_class_pattern = r"""^\s*
                (?:(abstract|final|base|interface|mixin|sealed)\s+)*
                class\s+(\w+)\s*<([^\n{]+)>\s*
                (?:extends\s+([^\n{]+?))?\s*
                (?:with\s+([^\n{]+?))?\s*
                (?:implements\s+([^\n{]+?))?\s*\{
            """
            for m in re.finditer(complex_class_pattern, content, re.MULTILINE | re.VERBOSE):
                name = m.group(2)
                if name in existing:
                    continue
                modifiers_raw = m.group(1) or ""
                modifiers = [mod for mod in modifiers_raw.split() if mod]
                generics = m.group(3).strip()
                extends = m.group(4).strip() if m.group(4) else None
                mixins = self._parse_type_list(m.group(5)) if m.group(5) else []
                implements = self._parse_type_list(m.group(6)) if m.group(6) else []
                structure.classes.append(
                    ClassInfo(
                        name=name,
                        line=content[: m.start()].count("\n") + 1,
                        generics=generics,
                        bases=[extends] if extends else [],
                        mixins=mixins,
                        interfaces=implements,
                        constructors=[],
                        methods=[],
                        fields=[],
                        properties=[],
                        modifiers=modifiers,
                    )
                )
        except Exception:
            pass

        # Extract mixins (standalone)
        mixin_pattern = r"^\s*(?:base\s+)?mixin\s+(\w+)(?:<([^>]+)>)?(?:\s+on\s+([^{]+))?\s*\{"
        for match in re.finditer(mixin_pattern, content, re.MULTILINE):
            mixin_name = match.group(1)
            # Avoid duplicates with mixin classes
            if not any(c.name == mixin_name for c in structure.classes):
                structure.mixins.append(
                    {
                        "name": mixin_name,
                        "line": content[: match.start()].count("\n") + 1,
                        "generics": match.group(2),
                        "on_types": self._parse_type_list(match.group(3)) if match.group(3) else [],
                    }
                )

        # Extract top-level functions
        func_pattern = r"""
            ^\s*(?:@\w+(?:\([^)]*\))?\s*)*  # Annotations
            (?:(Future|Stream)(?:<[^>]+>)?\s+)?
            (?:(void|[\w<>?]+|\([^)]+\))\s+)?  # Return type or record type
            ([a-zA-Z_]\w*)\s*
            (?:<[^>]+>)?\s*  # Generic parameters
            \(([^)]*)\)\s*
            (?:(?:async|sync)\s*\*|async)?\s*  # async, async*, or sync*
            (?:=>|\{)
        """

        for match in re.finditer(func_pattern, content, re.VERBOSE | re.MULTILINE):
            func_name = match.group(3)
            # Skip if it's inside a class
            if not self._is_top_level(content, match.start()):
                continue

            return_type = match.group(1) or match.group(2)
            params = match.group(4)

            span = content[match.start() : match.end()]
            is_async = "async" in span
            is_generator = "*" in span

            func_info = FunctionInfo(
                name=func_name,
                line=content[: match.start()].count("\n") + 1,
                return_type=return_type,
                parameters=self._parse_parameters(params),
                is_async=is_async,
                is_generator=is_generator,
                is_private=func_name.startswith("_"),
            )

            structure.functions.append(func_info)

        # Extract enums (brace-aware, supports enhanced enums with methods)
        enum_head_pattern = r"^\s*enum\s+(\w+)(?:\s*<[^>]+>)?(?:\s+implements\s+[^\{]+)?\s*\{"
        for m in re.finditer(enum_head_pattern, content, re.MULTILINE):
            enum_name = m.group(1)
            enum_body = self._extract_block(content, m.end()) or ""
            if enum_body is None:
                continue
            # Determine the values section: up to first top-level ';' if present
            values_part = enum_body
            depth = 0
            cutoff = None
            for i, ch in enumerate(enum_body):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)
                elif ch == ";" and depth == 0:
                    cutoff = i
                    break
            if cutoff is not None:
                values_part = enum_body[:cutoff]
            values = self._parse_enum_values(values_part)
            structure.enums.append(
                {
                    "name": enum_name,
                    "line": content[: m.start()].count("\n") + 1,
                    "values": values,
                    "has_enhanced_features": ("(" in values_part) or (cutoff is not None),
                }
            )

        # Extract extensions
        extension_pattern = r"^\s*extension\s+(?:(\w+)\s+)?on\s+([^\{]+)\s*\{"
        for match in re.finditer(extension_pattern, content, re.MULTILINE):
            extension_name = match.group(1) or f"on {match.group(2)}"
            on_type = match.group(2).strip()

            structure.extensions.append(
                {
                    "name": extension_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "on_type": on_type,
                }
            )

        # Extract typedefs
        typedef_pattern = r"^\s*typedef\s+(\w+)(?:<[^>]+>)?\s*=\s*([^;]+);"
        for match in re.finditer(typedef_pattern, content, re.MULTILINE):
            structure.typedefs.append(
                {
                    "name": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                    "definition": match.group(2).strip(),
                }
            )

        # Count null safety features
        structure.nullable_types = len(re.findall(r"\w+\?(?:\s|,|\))", content))
        structure.null_assertions = len(re.findall(r"\w+!(?:\.|;|\s|\))", content))
        structure.late_variables = len(re.findall(r"\blate\s+", content))
        structure.null_aware_operators = len(re.findall(r"\?\?|\?\.", content))

        # Count async features
        structure.async_functions = len(re.findall(r"\basync\s*(?:\*)?\s*(?:=>|\{)", content))
        structure.await_expressions = len(re.findall(r"\bawait\s+", content))
        structure.future_count = len(re.findall(r"\bFuture(?:\s*<|[.(])", content))
        structure.stream_count = len(re.findall(r"\bStream(?:\s*<|[.(])", content))

        # Detect test file
        structure.is_test_file = (
            "_test.dart" in file_path.name or file_path.parts and "test" in file_path.parts
        )

        # Detect main function
        structure.has_main = bool(re.search(r"\bvoid\s+main\s*\(", content))

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Dart code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Null safety complexity
        - Async complexity
        - Flutter-specific complexity
        - Class hierarchy depth

        Args:
            content: Dart source code
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
            r"\b\?\s*[^:]+\s*:",  # Ternary operator
            r"\?\?",  # Null coalescing
            r"&&",
            r"\|\|",
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        metrics.cyclomatic = complexity

        # Calculate cognitive complexity
        cognitive = 0
        nesting_level = 0
        max_nesting = 0

        lines = content.split("\n")
        for line in lines:
            # Skip comments
            if line.strip().startswith("//"):
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

        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting

        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("//")])
        metrics.comment_lines = len([l for l in lines if l.strip().startswith("//")])
        metrics.comment_ratio = (
            metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
        )

        # Count classes and methods
        metrics.class_count = len(re.findall(r"\bclass\s+\w+", content))
        metrics.mixin_count = len(re.findall(r"\bmixin\s+\w+", content))
        metrics.method_count = len(
            re.findall(
                r"(?:^|\s)(?:Future|Stream|void|[\w<>]+)\s+\w+\s*\([^)]*\)\s*(?:async\s*)?(?:=>|\{)",
                content,
            )
        )

        # Null safety metrics
        metrics.nullable_types = len(re.findall(r"\w+\?(?:\s|,|\))", content))
        metrics.null_assertions = len(re.findall(r"\w+!(?:\.|;|\s|\))", content))
        metrics.late_keywords = len(re.findall(r"\blate\s+", content))
        metrics.null_aware_ops = len(re.findall(r"\?\?|\?\.|\?\.\?", content))
        metrics.required_keywords = len(re.findall(r"\brequired\s+", content))

        # Async metrics
        metrics.async_functions = len(re.findall(r"\basync\s*(?:\*)?\s*(?:=>|\{)", content))
        metrics.await_count = len(re.findall(r"\bawait\s+", content))
        metrics.future_count = len(re.findall(r"\bFuture(?:\s*<|[.(])", content))
        metrics.stream_count = len(re.findall(r"\bStream(?:\s*<|[.(])", content))
        metrics.completer_count = len(re.findall(r"\bCompleter<", content))

        # Flutter-specific metrics
        if self._is_flutter_file(content):
            metrics.widget_count = len(re.findall(r"\bWidget\b", content))
            metrics.build_methods = len(re.findall(r"\bWidget\s+build\s*\(", content))
            metrics.setstate_calls = len(re.findall(r"\bsetState\s*\(", content))
            metrics.stateful_widgets = len(re.findall(r"extends\s+StatefulWidget", content))
            metrics.stateless_widgets = len(re.findall(r"extends\s+StatelessWidget", content))
            metrics.inherited_widgets = len(re.findall(r"extends\s+InheritedWidget", content))

            # Flutter hooks and keys
            metrics.keys_used = len(
                re.findall(r"\bKey\s*\(|GlobalKey|ValueKey|ObjectKey|UniqueKey", content)
            )
            metrics.context_usage = len(re.findall(r"\bBuildContext\b", content))

        # Exception handling metrics
        metrics.try_blocks = len(re.findall(r"\btry\s*\{", content))
        metrics.catch_blocks = len(re.findall(r"\bcatch\s*\(", content))
        metrics.finally_blocks = len(re.findall(r"\bfinally\s*\{", content))
        metrics.throw_statements = len(re.findall(r"\bthrow\s+", content))
        metrics.rethrow_statements = len(re.findall(r"\brethrow\s*;", content))

        # Type system metrics
        metrics.generic_types = len(re.findall(r"<[\w\s,<>]+>", content))
        metrics.type_parameters = len(re.findall(r"<\w+(?:\s+extends\s+\w+)?>", content))
        metrics.dynamic_types = len(re.findall(r"\bdynamic\b", content))
        metrics.var_declarations = len(re.findall(r"\bvar\s+\w+", content))

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Dart
            null_safety_factor = 1 - (metrics.null_assertions * 0.01)
            async_factor = 1 - (metrics.async_functions * 0.01)
            flutter_factor = (
                1 - (metrics.setstate_calls * 0.02) if hasattr(metrics, "setstate_calls") else 1
            )
            type_factor = 1 + ((metrics.nullable_types - metrics.dynamic_types) * 0.001)

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * null_safety_factor
                + 5 * async_factor
                + 5 * flutter_factor
                + 5 * type_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _categorize_import(self, module_path: str) -> str:
        """Categorize a Dart import.

        Args:
            module_path: Import path

        Returns:
            Category string
        """
        if module_path.startswith("dart:"):
            core_lib = module_path[5:].split("/")[0]
            if core_lib in ["core", "collection", "math", "convert", "typed_data"]:
                return "dart_core"
            elif core_lib in ["async", "isolate"]:
                return "dart_async"
            elif core_lib in ["io", "html", "indexed_db", "web_gl", "web_audio"]:
                return "dart_io"
            else:
                return "dart_sdk"
        elif module_path.startswith("package:flutter/"):
            # Extract the flutter module part
            remaining = module_path[16:]  # Remove "package:flutter/"
            flutter_lib = remaining.split(".")[0].split("/")[0]  # Get first part before . or /
            if flutter_lib == "material":
                return "flutter_material"
            elif flutter_lib == "cupertino":
                return "flutter_cupertino"
            elif flutter_lib == "widgets":
                return "flutter_widgets"
            else:
                return "flutter"
        elif module_path.startswith("package:"):
            package_name = module_path[8:].split("/")[0]
            # Common packages
            if package_name in ["provider", "bloc", "riverpod", "get", "getx"]:
                return "state_management"
            elif package_name in ["dio", "http", "retrofit"]:
                return "networking"
            elif package_name in ["test", "flutter_test", "mockito"]:
                return "testing"
            else:
                return "third_party"
        else:
            return "local"

    def _is_flutter_file(self, content: str) -> bool:
        """Check if the file uses Flutter.

        Args:
            content: Dart source code

        Returns:
            True if it's a Flutter file
        """
        flutter_indicators = [
            r"import\s+['\"]package:flutter/",
            r"\bStatelessWidget\b",
            r"\bStatefulWidget\b",
            r"\bState<",
            r"\bBuildContext\b",
            r"\bWidget\s+build\s*\(",
            r"\bScaffold\b",
            r"\bContainer\b",
            r"\bColumn\b",
            r"\bRow\b",
        ]

        return any(re.search(pattern, content) for pattern in flutter_indicators)

    def _parse_symbols(self, symbols_str: str) -> List[str]:
        """Parse show/hide symbols.

        Args:
            symbols_str: Comma-separated symbols

        Returns:
            List of symbol names
        """
        symbols = []
        for symbol in symbols_str.split(","):
            symbol = symbol.strip()
            if symbol:
                symbols.append(symbol)
        return symbols

    def _parse_type_list(self, types_str: str) -> List[str]:
        """Parse comma-separated type list handling generics.

        Args:
            types_str: Type list string

        Returns:
            List of type names
        """
        types = []
        current = ""
        depth = 0

        for char in types_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                if current.strip():
                    types.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            types.append(current.strip())

        return types

    def _extract_class_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a class.

        Args:
            content: Source code
            start_pos: Position after class declaration

        Returns:
            Class body content or None
        """
        brace_count = 1
        pos = start_pos
        in_string = False
        in_multiline_string = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            # Handle string literals
            if not escape_next:
                if content[pos : pos + 3] in ['"""', "'''"]:
                    in_multiline_string = not in_multiline_string
                    pos += 2
                elif char in ['"', "'"] and not in_multiline_string:
                    in_string = not in_string
                elif char == "\\":
                    escape_next = True
                elif not in_string and not in_multiline_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
            else:
                escape_next = False

            pos += 1

        if brace_count == 0:
            return content[start_pos : pos - 1]

        return None

    def _extract_block(self, content: str, start_pos: int) -> Optional[str]:
        """Extract a balanced-brace block starting at start_pos (just after '{')."""
        brace_count = 1
        pos = start_pos
        in_string = False
        in_multiline_string = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            # Handle string literals
            if not escape_next:
                if content[pos : pos + 3] in ['"""', "'''"]:
                    in_multiline_string = not in_multiline_string
                    pos += 2
                elif char in ['"', "'"] and not in_multiline_string:
                    in_string = not in_string
                elif char == "\\":
                    escape_next = True
                elif not in_string and not in_multiline_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
            else:
                escape_next = False

            pos += 1

        if brace_count == 0:
            return content[start_pos : pos - 1]
        return None

    def _extract_constructors(self, class_body: str, class_name: str) -> List[Dict[str, Any]]:
        """Extract constructors from class body.

        Args:
            class_body: Class body content
            class_name: Name of the class

        Returns:
            List of constructor information
        """
        constructors = []

        # Default constructor
        default_pattern = rf"{class_name}\s*\(([^)]*)\)"
        for match in re.finditer(default_pattern, class_body):
            constructors.append(
                {
                    "type": "default",
                    "parameters": self._parse_parameters(match.group(1)),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        # Named constructors
        named_pattern = rf"{class_name}\.(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(named_pattern, class_body):
            constructors.append(
                {
                    "type": "named",
                    "name": match.group(1),
                    "parameters": self._parse_parameters(match.group(2)),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        # Factory constructors
        factory_pattern = rf"factory\s+{class_name}(?:\.(\w+))?\s*\(([^)]*)\)"
        for match in re.finditer(factory_pattern, class_body):
            constructors.append(
                {
                    "type": "factory",
                    "name": match.group(1),
                    "parameters": self._parse_parameters(match.group(2)),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        # Const constructors
        const_pattern = rf"const\s+{class_name}(?:\.(\w+))?\s*\(([^)]*)\)"
        for match in re.finditer(const_pattern, class_body):
            constructors.append(
                {
                    "type": "const",
                    "name": match.group(1),
                    "parameters": self._parse_parameters(match.group(2)),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return constructors

    def _extract_methods(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract methods from class body.

        Args:
            class_body: Class body content

        Returns:
            List of method information
        """
        methods = []

        method_pattern = r"""
            (?:@override\s+)?
            (?:(static)\s+)?
            (?:(Future|Stream)(?:<[^>]+>)?\s+)?
            (?:(void|[\w<>?]+)\s+)?
            ([a-zA-Z_]\w*)\s*
            (?:<[^>]+>)?\s*
            \(([^)]*)\)\s*
            (?:(?:async|sync)\s*\*|async)?\s*
            (?:=>|\{)
        """

        for match in re.finditer(method_pattern, class_body, re.VERBOSE):
            method_name = match.group(4)

            # Skip constructors
            if method_name[0].isupper():
                continue

            is_static = match.group(1) == "static"
            return_type = match.group(2) or match.group(3)
            params = match.group(5)
            span = class_body[match.start() : match.end()]
            is_async = "async" in span
            is_override = "@override" in class_body[max(0, match.start() - 50) : match.start()]

            methods.append(
                {
                    "name": method_name,
                    "is_static": is_static,
                    "return_type": return_type,
                    "parameters": self._parse_parameters(params),
                    "is_async": is_async,
                    "is_override": is_override,
                    "is_private": method_name.startswith("_"),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return methods

    def _extract_fields(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract fields from class body.

        Args:
            class_body: Class body content

        Returns:
            List of field information
        """
        fields = []

        field_pattern = r"""
            ^\s*(?:(static)\s+)?
            (?:(final|const|late)\s+)?
            (?:(final|const)\s+)?  # Can have both late final
            (?:([\w<>?]+)\s+)?
            ([a-zA-Z_]\w*)\s*
            (?:=\s*[^;]+)?;
        """

        for match in re.finditer(field_pattern, class_body, re.VERBOSE | re.MULTILINE):
            field_name = match.group(5)

            # Skip if it looks like a method call
            if "(" in match.group(0):
                continue

            is_static = match.group(1) == "static"
            modifier = match.group(2) or match.group(3)
            field_type = match.group(4)

            fields.append(
                {
                    "name": field_name,
                    "type": field_type,
                    "is_static": is_static,
                    "is_final": modifier == "final",
                    "is_const": modifier == "const",
                    "is_late": modifier == "late",
                    "is_private": field_name.startswith("_"),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return fields

    def _extract_properties(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract getters and setters from class body.

        Args:
            class_body: Class body content

        Returns:
            List of property information
        """
        properties = []

        # Getters
        getter_pattern = r"(?:([\w<>?]+)\s+)?get\s+(\w+)\s*(?:=>|\{)"
        for match in re.finditer(getter_pattern, class_body):
            properties.append(
                {
                    "name": match.group(2),
                    "type": "getter",
                    "return_type": match.group(1),
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        # Setters
        setter_pattern = r"(?:void\s+)?set\s+(\w+)\s*\("
        for match in re.finditer(setter_pattern, class_body):
            properties.append(
                {
                    "name": match.group(1),
                    "type": "setter",
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return properties

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse method parameters.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter dictionaries
        """
        if not params_str.strip():
            return []

        parameters = []
        params = self._split_parameters(params_str)

        for param in params:
            param = param.strip()
            if not param:
                continue

            param_dict = {}

            # Check for required
            if param.startswith("required "):
                param_dict["required"] = True
                param = param[9:]

            # Check for this.field initializers
            if param.startswith("this."):
                param_dict["is_field_initializer"] = True
                param_dict["name"] = param[5:].split(",")[0].split(")")[0]
            else:
                # Parse type and name
                # Handle nullable types, defaults, etc.
                parts = param.split("=")
                if len(parts) > 1:
                    param_dict["default"] = parts[1].strip()
                    param = parts[0].strip()

                # Extract type and name
                tokens = param.split()
                if len(tokens) >= 2:
                    param_dict["type"] = " ".join(tokens[:-1])
                    param_dict["name"] = tokens[-1]
                else:
                    param_dict["name"] = tokens[0] if tokens else param

            parameters.append(param_dict)

        return parameters

    def _split_parameters(self, params_str: str) -> List[str]:
        """Split parameters handling nested brackets and braces.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter strings
        """
        params = []
        current = ""
        depth = 0
        in_brackets = False

        for char in params_str:
            if char in "({[<":
                depth += 1
                if char == "[" or char == "{":
                    in_brackets = True
            elif char in ")}]>":
                depth -= 1
                if char == "]" or char == "}":
                    in_brackets = False
            elif char == "," and depth == 0 and not in_brackets:
                if current.strip():
                    params.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            params.append(current.strip())

        return params

    def _is_top_level(self, content: str, position: int) -> bool:
        """Check if a position is at top level (not inside a class).

        Args:
            content: Source code
            position: Position to check

        Returns:
            True if at top level
        """
        # Simple heuristic: count braces before position
        before = content[:position]
        brace_count = before.count("{") - before.count("}")
        return brace_count == 0

    def _parse_enum_values(self, enum_body: str) -> List[Dict[str, Any]]:
        """Parse enum values from enum body.

        Args:
            enum_body: Enum body content

        Returns:
            List of enum value dictionaries
        """
        values = []

        # Handle enhanced enums (Dart 2.17+)
        if "(" in enum_body:
            # Enhanced enum with constructors
            value_pattern = r"(\w+)(?:\([^)]*\))?"
            for match in re.finditer(value_pattern, enum_body):
                if match.group(1) not in ["const", "final"]:
                    values.append(
                        {
                            "name": match.group(1),
                            "has_args": "(" in match.group(0),
                        }
                    )
        else:
            # Simple enum
            for value in enum_body.split(","):
                value = value.strip()
                if value and not value.startswith("//"):
                    values.append({"name": value})

        return values
