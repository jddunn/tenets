"""Swift code analyzer with iOS/macOS and SwiftUI support.

This module provides comprehensive analysis for Swift source files,
including support for iOS/macOS development, SwiftUI, UIKit,
async/await, and modern Swift features.
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


class SwiftAnalyzer(LanguageAnalyzer):
    """Swift code analyzer with iOS/macOS and SwiftUI support.

    Provides comprehensive analysis for Swift files including:
    - Import statements
    - Classes, structs, enums, protocols
    - Extensions and protocol conformance
    - Optionals and optional chaining
    - Guard statements and if-let bindings
    - Async/await and actors
    - Property wrappers (@State, @Published, etc.)
    - Result builders (@ViewBuilder, etc.)
    - SwiftUI views and modifiers
    - UIKit components
    - Combine framework usage
    - Access control levels
    - Generics and associated types

    Supports Swift 5.x features and Apple platform development.
    """

    language_name = "swift"
    file_extensions = [".swift"]

    def __init__(self):
        """Initialize the Swift analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract import statements from Swift code.

        Handles:
        - import statements: import Foundation
        - Targeted imports: import struct Swift.Array
        - Conditional imports: @_exported import, @testable import
        - Module aliasing (limited in Swift)

        Args:
            content: Swift source code
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

            # Basic import
            import_pattern = r"^\s*(?:(@\w+)\s+)?import\s+(?:(struct|class|enum|protocol|func|var|let|typealias)\s+)?([.\w]+)"
            match = re.match(import_pattern, line)
            if match:
                attribute = match.group(1)
                import_kind = match.group(2)
                module = match.group(3)

                # Determine import type
                import_type = "import"
                is_testable = False
                is_exported = False

                if attribute:
                    if attribute == "@testable":
                        is_testable = True
                        import_type = "testable_import"
                    elif attribute == "@_exported":
                        is_exported = True
                        import_type = "exported_import"

                # Categorize the import
                category = self._categorize_import(module)
                is_apple_framework = self._is_apple_framework(module)

                imports.append(
                    ImportInfo(
                        module=module,
                        line=i,
                        type=import_type,
                        is_relative=False,
                        category=category,
                        is_apple_framework=is_apple_framework,
                        is_testable=is_testable,
                        is_exported=is_exported,
                        import_kind=import_kind,  # struct, class, etc.
                    )
                )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from Swift code.

        In Swift, exports are determined by access control:
        - public: Accessible from any module
        - open: Subclassable from any module (classes only)
        - internal: Default, accessible within module
        - fileprivate: Accessible within file
        - private: Accessible within scope

        Args:
            content: Swift source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols (public/open declarations)
        """
        exports = []

        # Classes (reference types)
        class_pattern = r"^\s*(?:(public|open|internal|fileprivate|private)\s+)?(?:(final|abstract)\s+)?class\s+(\w+)"
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access in ["public", "open"]:
                exports.append(
                    {
                        "name": match.group(3),
                        "type": "class",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                        "is_final": match.group(2) == "final",
                        "is_open": access == "open",
                    }
                )

        # Structs (value types)
        struct_pattern = r"^\s*(?:(public|internal|fileprivate|private)\s+)?struct\s+(\w+)"
        for match in re.finditer(struct_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                exports.append(
                    {
                        "name": match.group(2),
                        "type": "struct",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                    }
                )

        # Enums
        enum_pattern = (
            r"^\s*(?:(public|internal|fileprivate|private)\s+)?(?:(indirect)\s+)?enum\s+(\w+)"
        )
        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                exports.append(
                    {
                        "name": match.group(3),
                        "type": "enum",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                        "is_indirect": match.group(2) == "indirect",
                    }
                )

        # Protocols
        protocol_pattern = r"^\s*(?:(public|internal|fileprivate|private)\s+)?protocol\s+(\w+)"
        for match in re.finditer(protocol_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                exports.append(
                    {
                        "name": match.group(2),
                        "type": "protocol",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                    }
                )

        # Actors
        actor_pattern = r"^\s*(?:(public|internal|fileprivate|private)\s+)?actor\s+(\w+)"
        for match in re.finditer(actor_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                exports.append(
                    {
                        "name": match.group(2),
                        "type": "actor",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                    }
                )

        # Functions
        func_pattern = r"^\s*(?:(public|open|internal|fileprivate|private)\s+)?(?:((?:static|class|mutating|async|throws|rethrows)\s+)*)func\s+(\w+)"
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access in ["public", "open"]:
                modifier_string = match.group(2).strip() if match.group(2) else ""
                modifiers = modifier_string.split() if modifier_string else []
                # Scan ahead to include post-parameter modifiers (e.g., "async throws") before the function body
                ahead = content[match.end() :]
                brace_match = re.search(r"\{", ahead)
                sig_tail = ahead[: brace_match.start()] if brace_match else ahead[:200]
                is_async = ("async" in modifiers) or bool(re.search(r"\basync\b", sig_tail))
                is_throwing = ("throws" in modifiers or "rethrows" in modifiers) or bool(
                    re.search(r"\b(?:throws|rethrows)\b", sig_tail)
                )
                exports.append(
                    {
                        "name": match.group(3),
                        "type": "function",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                        "modifiers": modifiers,
                        "is_async": is_async,
                        "is_throwing": is_throwing,
                    }
                )

        # Properties
        prop_pattern = r"^\s*(?:(public|internal|fileprivate|private)\s+)?(?:(static|class|lazy|weak|unowned)\s+)?(let|var)\s+(\w+)"
        for match in re.finditer(prop_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                modifiers = match.group(2)
                prop_kind = match.group(3)
                prop_name = match.group(4)

                # Skip if it looks like a local variable
                if not self._is_likely_property(content, match.start()):
                    continue

                exports.append(
                    {
                        "name": prop_name,
                        "type": "property",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                        "is_constant": prop_kind == "let",
                        "is_variable": prop_kind == "var",
                        "modifier": modifiers,
                    }
                )

        # Type aliases
        typealias_pattern = r"^\s*(?:(public|internal|fileprivate|private)\s+)?typealias\s+(\w+)"
        for match in re.finditer(typealias_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                exports.append(
                    {
                        "name": match.group(2),
                        "type": "typealias",
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                    }
                )

        # Extensions (public extensions export their methods)
        extension_pattern = r"^\s*(?:(public|internal|fileprivate|private)\s+)?extension\s+(\w+)"
        for match in re.finditer(extension_pattern, content, re.MULTILINE):
            access = match.group(1) or "internal"
            if access == "public":
                exports.append(
                    {
                        "name": f"extension_{match.group(2)}",
                        "type": "extension",
                        "extended_type": match.group(2),
                        "line": content[: match.start()].count("\n") + 1,
                        "access_level": access,
                    }
                )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Swift file.

        Extracts:
        - Classes, structs, enums, protocols, actors
        - Methods and properties
        - Extensions and protocol conformance
        - SwiftUI views and modifiers
        - UIKit components
        - Async/await patterns
        - Property wrappers
        - Computed properties and property observers

        Args:
            content: Swift source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Detect platform
        structure.is_ios = self._is_ios_file(content)
        structure.is_swiftui = self._is_swiftui_file(content)
        structure.is_uikit = self._is_uikit_file(content)

        # Extract classes
        class_pattern = r"""
            ^\s*(?:(public|open|internal|fileprivate|private)\s+)?
            (?:(final)\s+)?
            class\s+(\w+)
            (?:<([^>]+)>)?  # Generic parameters
            (?:\s*:\s*([^{]+?))?  # Inheritance/conformance
            \s*\{
        """

        for match in re.finditer(class_pattern, content, re.VERBOSE | re.MULTILINE):
            access = match.group(1) or "internal"
            is_final = match.group(2) == "final"
            class_name = match.group(3)
            generics = match.group(4)
            inheritance = match.group(5)

            # Parse inheritance and protocol conformance
            superclass = None
            protocols = []
            if inheritance:
                inherited = self._parse_inheritance(inheritance)
                # First item might be superclass (for classes)
                if inherited and not self._is_protocol(inherited[0]):
                    superclass = inherited[0]
                    protocols = inherited[1:]
                else:
                    protocols = inherited

            # Extract class body
            class_body = self._extract_body(content, match.end())

            if class_body:
                methods = self._extract_methods(class_body)
                properties = self._extract_properties(class_body)
                nested_types = self._extract_nested_types(class_body)
            else:
                methods = []
                properties = []
                nested_types = []

            # Check for UIKit/SwiftUI types
            ui_type = None
            if structure.is_ios:
                if superclass:
                    if "ViewController" in superclass:
                        ui_type = "view_controller"
                    elif "View" in superclass and "UI" in superclass:
                        ui_type = "uiview"
                    elif "ViewModel" in class_name:
                        ui_type = "view_model"

            class_info = ClassInfo(
                name=class_name,
                line=content[: match.start()].count("\n") + 1,
                access_level=access,
                is_final=is_final,
                is_open=access == "open",
                generics=generics,
                superclass=superclass,
                protocols=protocols,
                methods=methods,
                properties=properties,
                nested_types=nested_types,
                ui_type=ui_type,
            )

            structure.classes.append(class_info)

        # Extract structs
        struct_pattern = r"""
            ^\s*(?:(public|internal|fileprivate|private)\s+)?
            struct\s+(\w+)
            (?:<([^>]+)>)?
            (?:\s*:\s*([^{]+?))?
            \s*\{
        """

        for match in re.finditer(struct_pattern, content, re.VERBOSE | re.MULTILINE):
            struct_name = match.group(2)
            struct_body = self._extract_body(content, match.end())

            # Check if it's a SwiftUI View
            is_swiftui_view = False
            if match.group(4):
                protocols = self._parse_inheritance(match.group(4))
                is_swiftui_view = "View" in protocols

            structure.structs.append(
                {
                    "name": struct_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "access_level": match.group(1) or "internal",
                    "generics": match.group(3),
                    "protocols": self._parse_inheritance(match.group(4)) if match.group(4) else [],
                    "methods": self._extract_methods(struct_body) if struct_body else [],
                    "properties": self._extract_properties(struct_body) if struct_body else [],
                    "is_swiftui_view": is_swiftui_view,
                }
            )

        # Extract enums
        enum_pattern = r"""
            ^\s*(?:(public|internal|fileprivate|private)\s+)?
            (?:(indirect)\s+)?
            enum\s+(\w+)
            (?:<([^>]+)>)?
            (?:\s*:\s*([^{]+?))?
            \s*\{
        """

        for match in re.finditer(enum_pattern, content, re.VERBOSE | re.MULTILINE):
            enum_name = match.group(3)
            enum_body = self._extract_body(content, match.end())

            # Parse raw value type or conformance
            raw_type = None
            protocols = []
            if match.group(5):
                inherited = self._parse_inheritance(match.group(5))
                # Check for raw value types
                if inherited and inherited[0] in ["Int", "String", "Double", "Float", "Character"]:
                    raw_type = inherited[0]
                    protocols = inherited[1:]
                else:
                    protocols = inherited

            structure.enums.append(
                {
                    "name": enum_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "access_level": match.group(1) or "internal",
                    "is_indirect": match.group(2) == "indirect",
                    "generics": match.group(4),
                    "raw_type": raw_type,
                    "protocols": protocols,
                    "cases": self._extract_enum_cases(enum_body) if enum_body else [],
                    "methods": self._extract_methods(enum_body) if enum_body else [],
                }
            )

        # Extract protocols
        protocol_pattern = r"""
            ^\s*(?:(public|internal|fileprivate|private)\s+)?
            protocol\s+(\w+)
            (?:\s*:\s*([^{]+?))?
            \s*\{
        """

        for match in re.finditer(protocol_pattern, content, re.VERBOSE | re.MULTILINE):
            protocol_body = self._extract_body(content, match.end())

            structure.protocols.append(
                {
                    "name": match.group(2),
                    "line": content[: match.start()].count("\n") + 1,
                    "access_level": match.group(1) or "internal",
                    "inherited_protocols": (
                        self._parse_inheritance(match.group(3)) if match.group(3) else []
                    ),
                    "requirements": (
                        self._extract_protocol_requirements(protocol_body) if protocol_body else []
                    ),
                }
            )

        # Extract actors
        actor_pattern = r"""
            ^\s*(?:(public|internal|fileprivate|private)\s+)?
            actor\s+(\w+)
            (?:\s*:\s*([^{]+?))?
            \s*\{
        """

        for match in re.finditer(actor_pattern, content, re.MULTILINE | re.VERBOSE):
            actor_body = self._extract_body(content, match.end())

            structure.actors.append(
                {
                    "name": match.group(2),
                    "line": content[: match.start()].count("\n") + 1,
                    "access_level": match.group(1) or "internal",
                    "protocols": self._parse_inheritance(match.group(3)) if match.group(3) else [],
                    "methods": self._extract_methods(actor_body) if actor_body else [],
                    "properties": self._extract_properties(actor_body) if actor_body else [],
                }
            )

        # Extract extensions
        extension_pattern = r"""
            ^\s*(?:(public|internal|fileprivate|private)\s+)?
            extension\s+(\w+)
            (?:\s*:\s*([^{]+?))?
            (?:\s+where\s+([^{]+?))?
            \s*\{
        """

        for match in re.finditer(extension_pattern, content, re.VERBOSE | re.MULTILINE):
            extension_body = self._extract_body(content, match.end())

            structure.extensions.append(
                {
                    "extended_type": match.group(2),
                    "line": content[: match.start()].count("\n") + 1,
                    "access_level": match.group(1) or "internal",
                    "protocols": self._parse_inheritance(match.group(3)) if match.group(3) else [],
                    "where_clause": match.group(4),
                    "methods": self._extract_methods(extension_body) if extension_body else [],
                    "properties": (
                        self._extract_properties(extension_body) if extension_body else []
                    ),
                }
            )

        # Extract global functions (including operators)
        func_pattern = r"""
            ^\s*(?:(public|internal|fileprivate|private)\s+)?
            (?:(static|class|mutating|async|throws|rethrows|prefix|postfix|infix)\s+)*
            func\s+(\w+|[+\-*/%=<>!&|^~?]+)  # Include operator symbols
            (?:<([^>]+)>)?  # Generic parameters
            \s*\([^)]*\)  # Parameters
            (?:\s*(?:async\s+)?(?:throws|rethrows))?  # Post-parameter modifiers
            (?:\s*->\s*([^{]+?))?  # Return type
            (?:\s+where\s+([^{]+?))?  # Where clause
            \s*\{
        """

        for match in re.finditer(func_pattern, content, re.VERBOSE | re.MULTILINE):
            # Simple check: if the function is not heavily indented, it's likely global
            line_start = content.rfind("\n", 0, match.start()) + 1
            line_content = content[line_start : match.start()]
            indent = len(line_content) - len(line_content.lstrip())

            # Functions with small indent (0-4 spaces) are likely global
            if indent <= 4:
                func_info = FunctionInfo(
                    name=match.group(3),
                    line=content[: match.start()].count("\n") + 1,
                    access_level=match.group(1) or "internal",
                    is_async="async" in content[match.start() : match.end()],
                    is_throwing="throws" in content[match.start() : match.end()],
                    generics=match.group(4),
                    return_type=match.group(5).strip() if match.group(5) else "Void",
                    where_clause=match.group(6),
                )
                structure.functions.append(func_info)

        # Count Swift-specific patterns
        structure.optional_count = len(re.findall(r"\w+\?(?:\s|,|\)|>)", content))
        structure.force_unwrap_count = len(re.findall(r"!(?:\.|,|\s|\))", content))
        structure.optional_chaining_count = len(re.findall(r"\?\.", content))
        structure.nil_coalescing_count = len(re.findall(r"\?\?", content))
        structure.guard_count = len(re.findall(r"\bguard\s+", content))
        structure.if_let_count = len(re.findall(r"\bif\s+let\s+", content))
        structure.guard_let_count = len(re.findall(r"\bguard\s+let\s+", content))

        # Count async/await
        structure.async_functions = len(
            re.findall(r"\basync\s+func\b|\bfunc\s+\w+[^{]*\basync\b", content)
        )
        structure.await_count = len(re.findall(r"\bawait\s+", content))

        # Count tasks more comprehensively
        task_patterns = [
            r"\bTask\s*\{",
            r"\bTask\.detached\s*\{",
            r"group\.addTask\s*\{",
        ]
        task_count = 0
        for pattern in task_patterns:
            task_count += len(re.findall(pattern, content))
        structure.task_count = task_count

        structure.actor_count = len(structure.actors)

        # Count property wrappers
        structure.property_wrappers = len(
            re.findall(
                r"@(?:State|StateObject|ObservedObject|Published|Binding|Environment|EnvironmentObject|AppStorage|SceneStorage|FocusState|GestureState)\b",
                content,
            )
        )

        # Count result builders
        structure.result_builders = len(
            re.findall(
                r"@(?:ViewBuilder|SceneBuilder|CommandsBuilder|ToolbarContentBuilder)\b", content
            )
        )

        # Count Combine usage
        structure.combine_publishers = len(
            re.findall(
                r"(?:Published|PassthroughSubject|CurrentValueSubject|AnyPublisher)", content
            )
        )
        structure.combine_operators = len(
            re.findall(
                r"\.(?:sink|map|filter|flatMap|combineLatest|merge|zip|debounce|throttle)\s*(?:\{|\()",
                content,
            )
        )

        # SwiftUI specific
        if structure.is_swiftui:
            structure.swiftui_views = len(
                [s for s in structure.structs if s.get("is_swiftui_view")]
            )
            structure.view_modifiers = len(
                re.findall(
                    r"\.(?:padding|frame|background|foregroundColor|font|cornerRadius|shadow|overlay|offset|opacity|scaleEffect|rotationEffect|animation|transition)\s*\(",
                    content,
                )
            )
            structure.body_count = len(re.findall(r"\bvar\s+body\s*:\s*some\s+View\s*\{", content))

        # Detect test file
        structure.is_test_file = (
            "Test" in file_path.name
            or file_path.name.endswith("Tests.swift")
            or any(part in ["Tests", "UITests", "test"] for part in file_path.parts)
        )

        # Detect main app entry
        structure.has_main = bool(
            re.search(r"@main\b", content)
            or re.search(r"@UIApplicationMain\b", content)
            or re.search(r"@NSApplicationMain\b", content)
        )

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Swift code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Optional handling complexity
        - Async/await complexity
        - SwiftUI/UIKit specific complexity
        - Protocol-oriented complexity

        Args:
            content: Swift source code
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
            r"\bswitch\b",
            r"\bcase\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\brepeat\b",
            r"\bguard\b",
            r"\bcatch\b",
            r"&&",
            r"\|\|",
            r"\?\?",  # Nil coalescing
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
                (r"\bswitch\b", 2),
                (r"\bfor\b", 1),
                (r"\bwhile\b", 1),
                (r"\brepeat\b", 1),
                (r"\bguard\b", 1),
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

        # Count types
        metrics.class_count = len(re.findall(r"\bclass\s+\w+", content))
        metrics.struct_count = len(re.findall(r"\bstruct\s+\w+", content))
        metrics.enum_count = len(re.findall(r"\benum\s+\w+", content))
        metrics.protocol_count = len(re.findall(r"\bprotocol\s+\w+", content))
        metrics.extension_count = len(re.findall(r"\bextension\s+\w+", content))
        metrics.actor_count = len(re.findall(r"\bactor\s+\w+", content))

        # Optional handling metrics
        # Count optional type declarations more comprehensively
        optional_type_patterns = [
            r"\w+\?\s*(?:[,\)\]>=\n]|$)",  # Type? (more permissive ending)
            r"\w+!\s*(?:[,\)\]>=\n]|$)",  # Type! (implicitly unwrapped)
            r":\s*\w+\?\s*(?:[,=\{\n]|$)",  # : Type?
            r":\s*\w+!\s*(?:[,=\{\n]|$)",  # : Type!
            r"let\s+\w+:\s*\w+\?",  # let variable: Type?
            r"var\s+\w+:\s*\w+\?",  # var variable: Type?
            r"let\s+\w+:\s*\w+!",  # let variable: Type!
            r"var\s+\w+:\s*\w+!",  # var variable: Type!
        ]
        optional_count = 0
        for pattern in optional_type_patterns:
            optional_count += len(re.findall(pattern, content))
        metrics.optional_types = optional_count

        metrics.force_unwraps = len(re.findall(r"!(?:\.|,|\s|\))", content))
        metrics.optional_chaining = len(re.findall(r"\?\.", content))
        metrics.nil_coalescing = len(re.findall(r"\?\?", content))
        metrics.guard_statements = len(re.findall(r"\bguard\s+", content))
        metrics.if_let_bindings = len(re.findall(r"\bif\s+let\s+", content))
        metrics.guard_let_bindings = len(re.findall(r"\bguard\s+let\s+", content))

        # Async/await metrics
        metrics.async_functions = len(
            re.findall(r"\basync\s+func\b|\bfunc\s+\w+[^{]*\basync\b", content)
        )
        metrics.await_calls = len(re.findall(r"\bawait\s+", content))

        # Task patterns - more comprehensive detection
        task_patterns = [
            r"\bTask\s*\{",  # Task { }
            r"\bTask\.detached\s*\{",  # Task.detached { }
            r"group\.addTask\s*\{",  # group.addTask { }
            r"\bwithTaskGroup\s*\(",  # withTaskGroup
            r"\bwithThrowingTaskGroup\s*\(",  # withThrowingTaskGroup
        ]
        task_count = 0
        for pattern in task_patterns:
            task_count += len(re.findall(pattern, content))
        metrics.task_count = task_count

        # Task groups specifically
        task_group_patterns = [
            r"\bwithTaskGroup\s*\(",
            r"\bwithThrowingTaskGroup\s*\(",
            r"group\.addTask\s*\{",
        ]
        task_groups = 0
        for pattern in task_group_patterns:
            task_groups += len(re.findall(pattern, content))
        metrics.task_groups = task_groups

        metrics.main_actor = len(re.findall(r"@MainActor\b", content))

        # Error handling
        metrics.do_blocks = len(re.findall(r"\bdo\s*\{", content))
        metrics.try_statements = len(re.findall(r"\btry[!?]?\s+", content))
        metrics.catch_blocks = len(re.findall(r"\bcatch\s+", content))
        metrics.throw_statements = len(re.findall(r"\bthrow\s+", content))
        metrics.defer_statements = len(re.findall(r"\bdefer\s*\{", content))

        # Closures and functional programming
        metrics.closure_count = len(re.findall(r"\{[^}]*(?:in\s+|\$0)[^}]*\}", content))
        metrics.trailing_closures = len(re.findall(r"\)\s*\{[^}]*(?:in\s+|\$0)", content))
        metrics.higher_order_functions = len(
            re.findall(r"\.(?:map|filter|reduce|flatMap|compactMap|forEach)\s*(?:\{|\()", content)
        )

        # Property wrappers (SwiftUI and others)
        metrics.state_wrappers = len(re.findall(r"@State\b", content))
        metrics.stateobject_wrappers = len(re.findall(r"@StateObject\b", content))
        metrics.observedobject_wrappers = len(re.findall(r"@ObservedObject\b", content))
        metrics.published_wrappers = len(re.findall(r"@Published\b", content))
        metrics.binding_wrappers = len(re.findall(r"@Binding\b", content))
        metrics.environment_wrappers = len(re.findall(r"@Environment(?:Object)?\b", content))

        # SwiftUI specific
        if self._is_swiftui_file(content):
            metrics.swiftui_views = len(re.findall(r":\s*(?:some\s+)?View\s*\{", content))
            metrics.view_body_count = len(
                re.findall(r"\bvar\s+body\s*:\s*some\s+View\s*\{", content)
            )
            metrics.view_modifiers = len(
                re.findall(
                    r"\.(?:padding|frame|background|foregroundColor|font|cornerRadius|shadow|overlay|offset|opacity|scaleEffect|rotationEffect|animation|transition)\s*\(",
                    content,
                )
            )
            metrics.geometryreader_usage = len(re.findall(r"\bGeometryReader\s*\{", content))
            metrics.foreach_usage = len(re.findall(r"\bForEach\s*(?:\(|<)", content))

        # UIKit specific
        if self._is_uikit_file(content):
            metrics.viewcontroller_count = len(re.findall(r":\s*UI\w*ViewController", content))
            metrics.view_lifecycle = len(
                re.findall(r"\boverride\s+func\s+(?:viewDid|viewWill)", content)
            )
            metrics.iboutlet_count = len(re.findall(r"@IBOutlet\b", content))
            metrics.ibaction_count = len(re.findall(r"@IBAction\b", content))
            metrics.delegation_count = len(re.findall(r"delegate\s*=\s*self", content))

        # Combine framework
        metrics.combine_publishers = len(
            re.findall(
                r"(?:Published|PassthroughSubject|CurrentValueSubject|AnyPublisher)", content
            )
        )
        metrics.combine_subscriptions = len(re.findall(r"\.sink\s*\{", content))

        # Combine operators
        combine_operator_patterns = [
            r"\.map\s*\{",
            r"\.filter\s*\{",
            r"\.flatMap\s*\{",
            r"\.debounce\s*\(",
            r"\.removeDuplicates\s*\(",
            r"\.delay\s*\(",
            r"\.throttle\s*\(",
            r"\.combineLatest\s*\(",
            r"\.merge\s*\(",
            r"\.zip\s*\(",
            r"\.retry\s*\(",
            r"\.catch\s*\{",
            r"\.replaceError\s*\(",
            r"\.switchToLatest\s*\(",
        ]
        combine_operators = 0
        for pattern in combine_operator_patterns:
            combine_operators += len(re.findall(pattern, content))
        metrics.combine_operators = combine_operators

        # Access control
        metrics.public_declarations = len(
            re.findall(r"\bpublic\s+(?:class|struct|enum|protocol|func|var|let)\b", content)
        )
        metrics.private_declarations = len(
            re.findall(r"\bprivate\s+(?:class|struct|enum|protocol|func|var|let)\b", content)
        )
        metrics.fileprivate_declarations = len(
            re.findall(r"\bfileprivate\s+(?:class|struct|enum|protocol|func|var|let)\b", content)
        )

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Swift
            optional_safety_factor = 1 - (metrics.force_unwraps * 0.02)
            async_factor = 1 - (metrics.await_calls * 0.001)
            guard_factor = 1 + (metrics.guard_statements * 0.005)
            protocol_factor = 1 + (metrics.protocol_count * 0.01)

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * optional_safety_factor
                + 5 * async_factor
                + 5 * guard_factor
                + 5 * protocol_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _categorize_import(self, module: str) -> str:
        """Categorize a Swift import.

        Args:
            module: Import module name

        Returns:
            Category string
        """
        if module == "Swift":
            return "swift_stdlib"
        elif module == "Foundation":
            return "foundation"
        elif module in ["UIKit", "AppKit"]:
            return "ui_framework"
        elif module == "SwiftUI":
            return "swiftui"
        elif module == "Combine":
            return "combine"
        elif module == "CoreData":
            return "coredata"
        elif module.startswith("Core"):
            return "core_framework"
        elif module in ["Network", "URLSession"]:
            return "networking"
        elif module in ["XCTest", "Testing"]:
            return "testing"
        elif self._is_apple_framework(module):
            return "apple_framework"
        else:
            return "third_party"

    def _is_apple_framework(self, module: str) -> bool:
        """Check if a module is an Apple framework.

        Args:
            module: Module name

        Returns:
            True if it's an Apple framework
        """
        # Core Apple frameworks that don't follow prefix patterns
        core_apple_frameworks = [
            "Foundation",
            "Swift",
            "Combine",
            "SwiftUI",
            "Testing",
            "Network",
            "Compression",
            "OSLog",
            "Observation",
        ]

        if module in core_apple_frameworks:
            return True

        apple_prefixes = [
            "UI",
            "NS",
            "CF",
            "CG",
            "CI",
            "CA",
            "AV",
            "ML",
            "AR",
            "VN",
            "Core",
            "Metal",
            "Vision",
            "Natural",
            "Speech",
            "Sound",
            "Photos",
            "Messages",
            "Maps",
            "Health",
            "Home",
            "Music",
            "Contacts",
            "Calendar",
            "Reminders",
            "Notes",
            "Safari",
            "WebKit",
            "JavaScriptCore",
            "CloudKit",
            "StoreKit",
            "GameKit",
            "PassKit",
            "WatchKit",
            "WidgetKit",
            "App",
            "Accessibility",
            "Accounts",
            "AdSupport",
            "AuthenticationServices",
        ]
        return any(module.startswith(prefix) for prefix in apple_prefixes)

    def _is_ios_file(self, content: str) -> bool:
        """Check if the file is iOS-related.

        Args:
            content: Swift source code

        Returns:
            True if it's an iOS file
        """
        ios_indicators = [
            r"import\s+UIKit",
            r"import\s+SwiftUI",
            r":\s*UI\w+",
            r"@UIApplicationMain",
            r"@main.*App\s*:",
        ]
        return any(re.search(pattern, content) for pattern in ios_indicators)

    def _is_swiftui_file(self, content: str) -> bool:
        """Check if the file uses SwiftUI.

        Args:
            content: Swift source code

        Returns:
            True if it uses SwiftUI
        """
        swiftui_indicators = [
            r"import\s+SwiftUI",
            r":\s*(?:some\s+)?View\b",
            r"@State\b",
            r"@Binding\b",
            r"@ObservedObject\b",
            r"@StateObject\b",
            r"@Published\b",
            r"var\s+body\s*:\s*some\s+View",
        ]
        return any(re.search(pattern, content) for pattern in swiftui_indicators)

    def _is_uikit_file(self, content: str) -> bool:
        """Check if the file uses UIKit.

        Args:
            content: Swift source code

        Returns:
            True if it uses UIKit
        """
        uikit_indicators = [
            r"import\s+UIKit",
            r":\s*UI\w*ViewController",
            r":\s*UIView\b",
            r"@IBOutlet\b",
            r"@IBAction\b",
            r"UITableView",
            r"UICollectionView",
        ]
        return any(re.search(pattern, content) for pattern in uikit_indicators)

    def _extract_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a type declaration.

        Args:
            content: Source code
            start_pos: Position after opening brace

        Returns:
            Body content or None
        """
        if start_pos >= len(content):
            return None

        brace_count = 1
        pos = start_pos
        in_string = False
        in_multiline_string = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            if not escape_next:
                if content[pos : pos + 3] == '"""':
                    in_multiline_string = not in_multiline_string
                    pos += 2
                elif char == '"' and not in_multiline_string:
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

    def _parse_inheritance(self, inheritance_str: str) -> List[str]:
        """Parse inheritance/conformance string.

        Args:
            inheritance_str: Inheritance string

        Returns:
            List of inherited types/protocols
        """
        if not inheritance_str:
            return []

        items = []
        current = ""
        depth = 0

        for char in inheritance_str:
            if char in "<([":
                depth += 1
            elif char in ">)]":
                depth -= 1
            elif char == "," and depth == 0:
                if current.strip():
                    items.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            items.append(current.strip())

        return items

    def _is_protocol(self, type_name: str) -> bool:
        """Check if a type name is likely a protocol.

        Args:
            type_name: Type name

        Returns:
            True if likely a protocol
        """
        # Common protocol patterns in Swift
        protocol_patterns = [
            r"Protocol$",
            r"Delegate$",
            r"DataSource$",
            r"able$",  # Codable, Comparable, etc.
            r"^Any\b",  # AnyObject, AnyHashable
        ]

        # Known Swift protocols
        known_protocols = {
            "Codable",
            "Encodable",
            "Decodable",
            "Equatable",
            "Hashable",
            "Comparable",
            "CustomStringConvertible",
            "Error",
            "View",
            "ObservableObject",
            "Identifiable",
            "RandomAccessCollection",
        }

        return type_name in known_protocols or any(
            re.search(pattern, type_name) for pattern in protocol_patterns
        )

    def _is_likely_property(self, content: str, position: int) -> bool:
        """Check if a declaration at position is likely a property.

        Args:
            content: Source code
            position: Position of declaration

        Returns:
            True if likely a property (not local variable)
        """
        # Check context before the position
        before = content[max(0, position - 100) : position]

        # If inside a function body, likely a local variable
        if re.search(r"\bfunc\s+\w+[^{]*\{[^}]*$", before):
            return False

        # If inside a closure, likely a local variable
        if re.search(r"\{[^}]*$", before) and not re.search(
            r"(?:class|struct|enum|protocol|extension)\s+\w+[^{]*\{[^}]*$", before
        ):
            return False

        return True

    def _extract_methods(self, body: str) -> List[Dict[str, Any]]:
        """Extract methods from type body."""
        # Implementation details...
        return []

    def _extract_properties(self, body: str) -> List[Dict[str, Any]]:
        """Extract properties from type body."""
        # Implementation details...
        return []

    def _extract_nested_types(self, body: str) -> List[Dict[str, Any]]:
        """Extract nested type declarations."""
        # Implementation details...
        return []

    def _extract_enum_cases(self, body: str) -> List[Dict[str, Any]]:
        """Extract enum cases."""
        # Implementation details...
        return []

    def _extract_protocol_requirements(self, body: str) -> List[Dict[str, Any]]:
        """Extract protocol requirements."""
        # Implementation details...
        return []

    def _is_inside_type(self, content: str, position: int) -> bool:
        """Check if position is inside a type declaration."""
        # Implementation details...
        return False
