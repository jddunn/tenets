"""C# code analyzer with Unity3D support.

This module provides comprehensive analysis for C# source files,
including support for modern C# features, .NET patterns, and Unity3D
specific constructs like MonoBehaviours, Coroutines, and Unity attributes.
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


class CSharpAnalyzer(LanguageAnalyzer):
    """C# code analyzer with Unity3D support.

    Provides comprehensive analysis for C# files including:
    - Using directives and namespace analysis
    - Class, interface, struct, enum, and record extraction
    - Property and event analysis
    - Async/await and Task-based patterns
    - LINQ query detection
    - Attribute processing
    - Unity3D specific patterns (MonoBehaviour, Coroutines, etc.)
    - .NET Framework/Core detection
    - Nullable reference types (C# 8+)
    - Pattern matching (C# 7+)

    Supports modern C# features and Unity3D development patterns.
    """

    language_name = "csharp"
    file_extensions = [".cs", ".csx"]

    def __init__(self):
        """Initialize the C# analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract using directives from C# code.

        Handles:
        - using statements: using System.Collections.Generic;
        - using static: using static System.Math;
        - using aliases: using Project = PC.MyCompany.Project;
        - global using (C# 10+): global using System.Text;
        - Unity-specific usings

        Args:
            content: C# source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports: List[ImportInfo] = []
        lines = content.splitlines()

        current_namespace: Optional[str] = None
        seen_code = False  # stop parsing usings after first non-using code element at top-level

        # Pre-compile patterns (hot path in large files)
        namespace_re = re.compile(r"^\s*namespace\s+([\w\.]+)")
        alias_re = re.compile(r"^\s*(?:(global)\s+)?using\s+([\w\.]+)\s*=\s*([^;]+?)\s*;")
        using_re = re.compile(r"^\s*(?:(global)\s+)?using\s+(?:(static)\s+)?([\w\.]+)\s*;")
        decl_re = re.compile(
            r"^\s*(?:public\s+)?(?:partial\s+)?(?:abstract\s+)?(?:sealed\s+)?(?:class|interface|struct|enum|delegate|record)\b"
        )

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            # Skip single-line comments
            if stripped.startswith("//"):
                continue

            # Namespace (track for nested usings)
            m = namespace_re.match(line)
            if m:
                current_namespace = m.group(1)
                # Don't treat namespace declaration itself as code for stopping further usings
                continue

            # Stop scanning after first real code (class/interface/etc.) at top-level
            if decl_re.match(line):
                seen_code = True
            if seen_code:
                # Still allow usings inside namespace blocks (indented) – C# allows that
                # Only break if this is a top-level code declaration and not inside a namespace context yet
                if current_namespace is None:
                    break

            # Using alias
            m = alias_re.match(line)
            if m:
                is_global = m.group(1) == "global"
                alias = m.group(2)
                target = m.group(3).strip()
                base_for_category = target.split("<", 1)[0].strip()
                category = self._categorize_import(base_for_category)
                is_unity = self._is_unity_import(base_for_category)
                imports.append(
                    ImportInfo(
                        module=target,
                        alias=alias,
                        line=i,
                        type="global_using_alias" if is_global else "using_alias",
                        is_relative=False,
                        category=category,
                        is_unity=is_unity,
                        namespace_context=current_namespace,
                    )
                )
                continue

            # Standard / static / global usings
            m = using_re.match(line)
            if m:
                is_global = m.group(1) == "global"
                is_static = m.group(2) == "static"
                ns = m.group(3)
                category = self._categorize_import(ns)
                is_unity = self._is_unity_import(ns)
                if is_global:
                    import_type = "global_using"
                elif is_static:
                    import_type = "using_static"
                else:
                    import_type = "using"
                imports.append(
                    ImportInfo(
                        module=ns,
                        line=i,
                        type=import_type,
                        is_relative=False,
                        category=category,
                        is_unity=is_unity,
                        namespace_context=current_namespace,
                    )
                )
                continue

        # .csproj dependency parsing
        if file_path.suffix.lower() == ".csproj":
            imports.extend(self._extract_csproj_dependencies(content))

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract public members from C# code.

        In C#, public members are accessible from other assemblies.
        This includes public classes, interfaces, structs, enums, delegates, etc.

        Args:
            content: C# source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported (public) symbols
        """
        exports = []

        # Extract namespace
        namespace_match = re.search(r"^\s*namespace\s+([\w\.]+)", content, re.MULTILINE)
        namespace = namespace_match.group(1) if namespace_match else ""

        # Public classes (including Unity MonoBehaviours)
        class_pattern = r"^\s*(?:public\s+)?(?:partial\s+)?(?:abstract\s+)?(?:sealed\s+)?(?:static\s+)?class\s+(\w+)(?:\s*:\s*([\w\.,\s]+))?"

        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            inheritance = match.group(2)

            modifiers = []
            if "abstract" in match.group(0):
                modifiers.append("abstract")
            if "sealed" in match.group(0):
                modifiers.append("sealed")
            if "static" in match.group(0):
                modifiers.append("static")
            if "partial" in match.group(0):
                modifiers.append("partial")

            # Check if it's a Unity component
            is_unity_component = False
            unity_base_class = None
            if inheritance:
                if "MonoBehaviour" in inheritance:
                    is_unity_component = True
                    unity_base_class = "MonoBehaviour"
                elif "ScriptableObject" in inheritance:
                    is_unity_component = True
                    unity_base_class = "ScriptableObject"
                elif "Editor" in inheritance:
                    is_unity_component = True
                    unity_base_class = "Editor"

            exports.append(
                {
                    "name": class_name,
                    "type": "class",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "modifiers": modifiers,
                    "inheritance": inheritance,
                    "is_unity_component": is_unity_component,
                    "unity_base_class": unity_base_class,
                }
            )

        # Public interfaces
        interface_pattern = r"^\s*(?:public\s+)?(?:partial\s+)?interface\s+(\w+)(?:<[^>]+>)?(?:\s*:\s*([\w\.,\s]+))?"

        for match in re.finditer(interface_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "interface",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "extends": match.group(2),
                }
            )

        # Public structs
        struct_pattern = r"^\s*(?:public\s+)?(?:readonly\s+)?(?:ref\s+)?struct\s+(\w+)"

        for match in re.finditer(struct_pattern, content, re.MULTILINE):
            modifiers = []
            if "readonly" in match.group(0):
                modifiers.append("readonly")
            if "ref" in match.group(0):
                modifiers.append("ref")

            exports.append(
                {
                    "name": match.group(1),
                    "type": "struct",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "modifiers": modifiers,
                }
            )

        # Public enums (support both 'enum' and 'enum class' styles)
        enum_pattern = r"^\s*(?:public\s+)?enum(?:\s+class)?\s+(\w+)(?:\s*:\s*([\w\.]+))?"

        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            enum_type = "enum_class" if "enum class" in match.group(0) else "enum"
            exports.append(
                {
                    "name": match.group(1),
                    "type": enum_type,
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "base_type": match.group(2),
                }
            )

        # Public delegates
        delegate_pattern = r"^\s*(?:public\s+)?delegate\s+(\w+)\s+(\w+(?:<[^>]+>)?)\s*\([^)]*\)"

        for match in re.finditer(delegate_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(2),
                    "type": "delegate",
                    "return_type": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        # Public records (C# 9+)
        record_pattern = r"^\s*(?:public\s+)?record\s+(?:class\s+|struct\s+)?(\w+)"

        for match in re.finditer(record_pattern, content, re.MULTILINE):
            record_type = "record_struct" if "struct" in match.group(0) else "record"
            exports.append(
                {
                    "name": match.group(1),
                    "type": record_type,
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from C# file.

        Extracts:
        - Namespace declarations
        - Classes with inheritance and interfaces
        - Properties with getters/setters
        - Methods including async methods
        - Events and delegates
        - Unity-specific components (MonoBehaviours, Coroutines)
        - LINQ queries
        - Attributes

        Args:
            content: C# source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Extract namespace
        namespace_match = re.search(r"^\s*namespace\s+([\w\.]+)", content, re.MULTILINE)
        if namespace_match:
            structure.namespace = namespace_match.group(1)

        # Detect if it's a Unity script
        structure.is_unity_script = self._is_unity_script(content)

        # Extract classes
        # Capture any stacked attribute blocks immediately preceding the class declaration in a named group
        # so we don't rely on a fragile backward scan that fails when the regex itself already consumed them.
        class_pattern = (
            r"(?:^|\n)\s*(?P<attr_block>(?:\[[^\]]+\]\s*)*)"
            r"(?:(?P<visibility>public|private|protected|internal)\s+)?"
            r"(?:(?P<partial>partial)\s+)?(?:(?P<abstract>abstract)\s+)?(?:(?P<sealed>sealed)\s+)?(?:(?P<static>static)\s+)?"
            r"class\s+(?P<class_name>\w+)(?:<(?P<generics>[^>]+)>)?(?:\s*:\s*(?P<inheritance>[\w\.,\s<>]+))?"
        )

        for match in re.finditer(class_pattern, content):
            attr_block = match.group("attr_block") or ""
            class_name = match.group("class_name") or ""
            generics = match.group("generics")
            inheritance = match.group("inheritance")

            # Prefer directly captured attribute block; fallback to legacy backward scan only if empty
            attributes = self._extract_attributes(attr_block) if attr_block else []
            if not attributes:
                # Legacy backward scan (kept for robustness in edge cases where regex miss might occur)
                start_line_index = content[: match.start()].count("\n")
                lines = content.splitlines()
                attr_lines: List[str] = []
                line_cursor = start_line_index - 1
                while line_cursor >= 0:
                    line_text = lines[line_cursor].strip()
                    if not line_text or not line_text.startswith("["):
                        break
                    attr_lines.insert(0, line_text)
                    line_cursor -= 1
                if attr_lines:
                    attributes = self._extract_attributes("\n".join(attr_lines))

            # Collect modifiers
            modifiers: List[str] = []
            for key in ["partial", "abstract", "sealed", "static"]:
                if match.group(key):
                    modifiers.append(match.group(key))

            visibility = match.group("visibility") or None

            # Parse inheritance
            bases = []
            interfaces = []
            is_monobehaviour = False
            is_scriptable_object = False

            if inheritance:
                for item in inheritance.split(","):
                    item = item.strip()
                    if item == "MonoBehaviour":
                        is_monobehaviour = True
                        bases.append(item)
                    elif item == "ScriptableObject":
                        is_scriptable_object = True
                        bases.append(item)
                    elif item.startswith("I"):  # Convention for interfaces
                        interfaces.append(item)
                    else:
                        bases.append(item)

            # Find class body
            class_body = self._extract_class_body(content, match.end())

            # Extract class components
            methods = []
            properties = []
            fields = []
            events = []
            unity_methods = []
            coroutines = []

            if class_body:
                methods = self._extract_methods(class_body)
                properties = self._extract_properties(class_body)
                fields = self._extract_fields(class_body)
                events = self._extract_events(class_body)

                if is_monobehaviour or is_scriptable_object:
                    unity_methods = self._extract_unity_methods(class_body)
                    coroutines = self._extract_coroutines(class_body)

            class_info = ClassInfo(
                name=class_name,
                line=content[: match.start()].count("\n") + 1,
                generics=generics,
                bases=bases,
                interfaces=interfaces,
                visibility=visibility,
                modifiers=modifiers,
                methods=methods,
                properties=properties,
                fields=fields,
                events=events,
                attributes=attributes,
                is_monobehaviour=is_monobehaviour,
                is_scriptable_object=is_scriptable_object,
                unity_methods=unity_methods,
                coroutines=coroutines,
            )

            structure.classes.append(class_info)

        # Extract interfaces
        interface_pattern = r"(?:^|\n)\s*(?:public\s+)?(?:partial\s+)?interface\s+(\w+)(?:<([^>]+)>)?(?:\s*:\s*([\w\.,\s<>]+))?"

        for match in re.finditer(interface_pattern, content):
            interface_name = match.group(1)
            generics = match.group(2)
            extends = match.group(3)

            # Extract interface methods
            interface_body = self._extract_class_body(content, match.end())
            methods = self._extract_interface_methods(interface_body) if interface_body else []

            structure.interfaces.append(
                {
                    "name": interface_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "generics": generics,
                    "extends": self._parse_interface_list(extends) if extends else [],
                    "methods": methods,
                }
            )

        # Extract structs
        struct_pattern = (
            r"(?:^|\n)\s*(?:public\s+)?(?:readonly\s+)?(?:ref\s+)?struct\s+(\w+)(?:<([^>]+)>)?"
        )

        for match in re.finditer(struct_pattern, content):
            struct_name = match.group(1)
            generics = match.group(2)

            modifiers = []
            if "readonly" in match.group(0):
                modifiers.append("readonly")
            if "ref" in match.group(0):
                modifiers.append("ref")

            structure.structs.append(
                {
                    "name": struct_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "generics": generics,
                    "modifiers": modifiers,
                }
            )

        # Extract enums
        enum_pattern = r"(?:^|\n)\s*(?:public\s+)?enum\s+(\w+)(?:\s*:\s*(\w+))?"

        for match in re.finditer(enum_pattern, content):
            enum_name = match.group(1)
            base_type = match.group(2)

            # Extract enum values
            enum_body = self._extract_class_body(content, match.end())
            values = self._extract_enum_values(enum_body) if enum_body else []

            structure.enums.append(
                {
                    "name": enum_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "base_type": base_type,
                    "values": values,
                }
            )

        # Extract delegates
        delegate_pattern = r"(?:^|\n)\s*(?:public\s+)?delegate\s+(\w+)\s+(\w+)\s*\(([^)]*)\)"

        for match in re.finditer(delegate_pattern, content):
            structure.delegates.append(
                {
                    "return_type": match.group(1),
                    "name": match.group(2),
                    "parameters": self._parse_parameters(match.group(3)),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Extract global functions (rare in C# but possible)
        structure.functions = self._extract_global_functions(content)

        # Extract LINQ queries
        structure.linq_queries = self._extract_linq_queries(content)

        # Count async methods
        structure.async_method_count = len(re.findall(r"\basync\s+(?:Task|ValueTask)", content))

        # Count lambda expressions
        structure.lambda_count = len(re.findall(r"=>\s*(?:\{|[^;{]+;)", content))

        # Detect framework
        structure.framework = self._detect_framework(content)

        # Check for test file
        structure.is_test_file = (
            "Test" in file_path.name
            or file_path.name.endswith("Tests.cs")
            or file_path.name.endswith("Test.cs")
            or any(part in ["Tests", "Test"] for part in file_path.parts)
        )

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for C# code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Unity-specific complexity (Coroutines, Update methods)
        - Async/await complexity
        - LINQ complexity
        - Exception handling complexity

        Args:
            content: C# source code
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
            r"\bforeach\b",
            r"\bwhile\b",
            r"\bdo\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\b&&\b",
            r"\|\|",
            r"\?\s*[^:]+\s*:",  # Ternary operator
            r"\?\?",  # Null coalescing operator
            r"\?\.(?!\s*\[)",  # Null conditional operator (not including ?.[])
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        # Add complexity for pattern matching (C# 7+)
        # "is" patterns
        complexity += len(re.findall(r"\bis\s+\w+\s+\w+", content))
        # Switch statements with when filters
        complexity += len(re.findall(r"\bswitch\s*\(.*\)\s*\{[\s\S]*?\bwhen\b", content))
        # Switch expressions with when clauses (=> and when)
        complexity += len(re.findall(r"\bswitch\s*\{[\s\S]*?=>[\s\S]*?\bwhen\b", content))

        metrics.cyclomatic = complexity

        # Calculate cognitive complexity
        cognitive = 0
        nesting_level = 0
        max_nesting = 0

        lines = content.splitlines()
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
                (r"\bforeach\b", 1),
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
            metrics.code_lines = self._count_code_lines(content)
            metrics.comment_lines = self._count_comment_lines(content)
            metrics.comment_ratio = (
                metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
            )

            # Count classes, interfaces, etc.
            metrics.class_count = len(re.findall(r"\bclass\s+\w+", content))
            metrics.interface_count = len(re.findall(r"\binterface\s+\w+", content))
            metrics.struct_count = len(re.findall(r"\bstruct\s+\w+", content))
            metrics.enum_count = len(re.findall(r"\benum\s+\w+", content))

            # Count methods
            metrics.method_count = len(
                re.findall(
                    r"(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:override\s+)?(?:virtual\s+)?(?:[\w<>\[\]]+)\s+\w+\s*\([^)]*\)\s*\{",
                    content,
                )
            )

            # Property metrics
            metrics.property_count = len(
                re.findall(
                    r"(?:public|private|protected|internal)\s+(?:static\s+)?(?:[\w<>\[\]]+)\s+\w+\s*\{\s*(?:get|set)",
                    content,
                )
            )
            metrics.auto_property_count = len(re.findall(r"\{\s*get;\s*(?:set;)?\s*\}", content))

            # Exception handling metrics
            metrics.try_blocks = len(re.findall(r"\btry\s*\{", content))
            metrics.catch_blocks = len(
                re.findall(r"\bcatch(?:\s+when\s*\([^)]*\))?\s*(?:\([^)]*\))?\s*\{", content)
            )
            metrics.finally_blocks = len(re.findall(r"\bfinally\s*\{", content))
            # Count both "throw;" and "throw new ..." forms
            metrics.throw_statements = len(re.findall(r"\bthrow\b", content))

            # Async/await metrics
            metrics.async_methods = len(re.findall(r"\basync\s+(?:Task|ValueTask)", content))
            metrics.await_statements = len(re.findall(r"\bawait\s+", content))

            # LINQ metrics
            metrics.linq_queries = len(re.findall(r"\bfrom\s+\w+\s+in\s+", content))
            metrics.linq_methods = len(
                re.findall(
                    r"\.\s*(?:Where|Select|OrderBy|GroupBy|Join|Any|All|First|Last|Single)\s*\(",
                    content,
                )
            )

            # Unity-specific metrics
            if self._is_unity_script(content):
                metrics.unity_components = len(
                    re.findall(r":\s*(?:MonoBehaviour|ScriptableObject)", content)
                )
                metrics.coroutines = len(re.findall(r"\bIEnumerator\s+\w+\s*\(", content))
                metrics.unity_methods = len(
                    re.findall(
                        r"\b(?:Start|Update|FixedUpdate|LateUpdate|OnEnable|OnDisable|Awake|OnDestroy|OnCollision(?:Enter|Exit|Stay)?|OnTrigger(?:Enter|Exit|Stay)?)\s*\(",
                        content,
                    )
                )
                metrics.serialize_fields = len(re.findall(r"\[SerializeField\]", content))
                metrics.unity_events = len(re.findall(r"\bUnityEvent(?:<[^>]+>)?\s+\w+", content))

            # Attribute metrics
            metrics.attribute_count = len(re.findall(r"\[[A-Z]\w*(?:\([^)]*\))?\]", content))

            # Nullable reference types (C# 8+): properties and locals/params with ? type, plus #nullable enable
            nullable_types = len(re.findall(r"[\w<>\[\]]+\?\s+\w+\s*[;=,)\}]", content))
            metrics.nullable_refs = nullable_types + len(re.findall(r"#nullable\s+enable", content))

            # Calculate maintainability index
            import math

            if metrics.code_lines > 0:
                # Adjusted for C#
                async_factor = 1 - (metrics.async_methods * 0.01)
                unity_factor = 1 - (getattr(metrics, "coroutines", 0) * 0.02)

                mi = (
                    171
                    - 5.2 * math.log(max(1, complexity))
                    - 0.23 * complexity
                    - 16.2 * math.log(metrics.code_lines)
                    + 10 * async_factor
                    + 10 * unity_factor
                )
                metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _categorize_import(self, namespace: str) -> str:
        """Categorize a C# import/using directive.

        Args:
            namespace: Namespace path

        Returns:
            Category: 'system', 'unity', 'third_party', or 'local'
        """
        if namespace.startswith("System"):
            return "system"
        elif namespace.startswith("Microsoft"):
            return "microsoft"
        elif (
            namespace.startswith("Unity")
            or namespace.startswith("UnityEngine")
            or namespace.startswith("UnityEditor")
        ):
            return "unity"
        elif namespace.startswith("TMPro"):
            return "unity_package"  # TextMeshPro
        elif "." in namespace:
            return "third_party"
        else:
            return "local"

    def _is_unity_import(self, namespace: str) -> bool:
        """Check if an import is Unity-related.

        Args:
            namespace: Namespace path

        Returns:
            True if it's a Unity import
        """
        unity_namespaces = [
            "UnityEngine",
            "UnityEditor",
            "Unity.",
            "TMPro",
            "Cinemachine",
            "UnityEngine.UI",
            "UnityEngine.Events",
            "UnityEngine.Rendering",
            "UnityEngine.InputSystem",
        ]
        return any(namespace.startswith(ns) for ns in unity_namespaces)

    def _is_unity_script(self, content: str) -> bool:
        """Check if the file is a Unity script.

        Args:
            content: C# source code

        Returns:
            True if it's a Unity script
        """
        unity_indicators = [
            r"using\s+UnityEngine",
            r":\s*MonoBehaviour",
            r":\s*ScriptableObject",
            r":\s*Editor",
            r"\[SerializeField\]",
            r"\[CreateAssetMenu",
            r"GameObject\s+",
            r"Transform\s+",
            r"Vector[23]\s+",
            r"Quaternion\s+",
        ]

        return any(re.search(pattern, content) for pattern in unity_indicators)

    def _extract_csproj_dependencies(self, content: str) -> List[ImportInfo]:
        """Extract dependencies from .csproj file.

        Args:
            content: .csproj XML content

        Returns:
            List of ImportInfo objects for dependencies
        """
        imports = []

        # Extract PackageReference elements
        package_pattern = r'<PackageReference\s+Include="([^"]+)"(?:\s+Version="([^"]+)")?'

        for match in re.finditer(package_pattern, content):
            package_name = match.group(1)
            version = match.group(2)

            imports.append(
                ImportInfo(
                    module=package_name,
                    version=version,
                    type="nuget_package",
                    is_relative=False,
                    is_dependency=True,
                )
            )

        # Extract ProjectReference elements
        project_pattern = r'<ProjectReference\s+Include="([^"]+)"'

        for match in re.finditer(project_pattern, content):
            project_path = match.group(1)

            imports.append(
                ImportInfo(
                    module=project_path,
                    type="project_reference",
                    is_relative=True,
                    is_project_reference=True,
                )
            )

        return imports

    def _parse_interface_list(self, interfaces_str: str) -> List[str]:
        """Parse a comma-separated list of interfaces.

        Args:
            interfaces_str: String with interfaces

        Returns:
            List of interface names
        """
        if not interfaces_str:
            return []

        interfaces = []
        current = ""
        depth = 0

        for char in interfaces_str:
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
        """Extract the body of a class/interface/struct.

        Args:
            content: Source code
            start_pos: Position after class declaration

        Returns:
            Class body content or None
        """
        brace_pos = content.find("{", start_pos)
        if brace_pos == -1:
            return None

        brace_count = 1
        pos = brace_pos + 1
        in_string = False
        in_char = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            if not escape_next:
                if char == '"' and not in_char:
                    in_string = not in_string
                elif char == "'" and not in_string:
                    in_char = not in_char
                elif char == "\\":
                    escape_next = True
                elif not in_string and not in_char:
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
        """Extract methods from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of method information
        """
        methods = []

        # Method pattern with modifiers
        method_pattern = r"(?:^|\n)\s*(?:\[[^\]]+\]\s*)*(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract|sealed|async|partial)\s+)*(?:([\w<>\[\]]+)\s+)?(\w+)\s*\(([^)]*)\)(?:\s*:\s*base\([^)]*\))?"

        for match in re.finditer(method_pattern, class_body):
            return_type = match.group(3)
            method_name = match.group(4)

            # Filter out false positives
            if method_name in ["if", "for", "while", "switch", "catch", "lock", "using", "fixed"]:
                continue

            # Check if it's a constructor
            is_constructor = return_type is None or return_type == method_name

            visibility = match.group(1) or "private"
            modifiers = match.group(2).split() if match.group(2) else []

            methods.append(
                {
                    "name": method_name,
                    "visibility": visibility,
                    "modifiers": modifiers,
                    "return_type": return_type if not is_constructor else None,
                    "parameters": self._parse_parameters(match.group(5)),
                    "is_constructor": is_constructor,
                    "is_async": "async" in modifiers,
                    "is_virtual": "virtual" in modifiers,
                    "is_override": "override" in modifiers,
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return methods

    def _extract_properties(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract properties from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of property information
        """
        properties = []

        # Property pattern
        property_pattern = r"(?:^|\n)\s*(?:\[[^\]]+\]\s*)*(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract)\s+)*([\w<>\[\]]+)\s+(\w+)\s*\{\s*(get|set|init)"

        for match in re.finditer(property_pattern, class_body):
            visibility = match.group(1) or "private"
            modifiers = match.group(2).split() if match.group(2) else []
            prop_type = match.group(3)
            prop_name = match.group(4)

            # Check for auto-property
            is_auto = bool(
                re.search(
                    rf"{re.escape(prop_name)}\s*\{{\s*get;\s*(?:set;|init;)?\s*\}}", class_body
                )
            )

            properties.append(
                {
                    "name": prop_name,
                    "type": prop_type,
                    "visibility": visibility,
                    "modifiers": modifiers,
                    "is_auto_property": is_auto,
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return properties

    def _extract_fields(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract fields from class body, including Unity-specific attributes.

        Args:
            class_body: Content of class body

        Returns:
            List of field information
        """
        fields = []

        # Field pattern (including Unity attributes)
        field_pattern = r"(?:^|\n)\s*(?:\[([^\]]+)\]\s*)*(?:(public|private|protected|internal)\s+)?(?:(static|readonly|const)\s+)*([\w<>\[\]]+)\s+(\w+)\s*(?:=\s*([^;]+))?\s*;"

        for match in re.finditer(field_pattern, class_body):
            attributes = match.group(1)
            visibility = match.group(2) or "private"
            modifiers = match.group(3).split() if match.group(3) else []
            field_type = match.group(4)
            field_name = match.group(5)
            initial_value = match.group(6)

            # Parse Unity attributes
            unity_attributes = []
            if attributes:
                if "SerializeField" in attributes:
                    unity_attributes.append("SerializeField")
                if "HideInInspector" in attributes:
                    unity_attributes.append("HideInInspector")
                if "Range" in attributes:
                    unity_attributes.append("Range")
                if "Header" in attributes:
                    unity_attributes.append("Header")
                if "Tooltip" in attributes:
                    unity_attributes.append("Tooltip")

            # Skip properties (they have get/set)
            if re.search(rf"{re.escape(field_name)}\s*\{{", class_body):
                continue

            fields.append(
                {
                    "name": field_name,
                    "type": field_type,
                    "visibility": visibility,
                    "modifiers": modifiers,
                    "initial_value": initial_value.strip() if initial_value else None,
                    "unity_attributes": unity_attributes,
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return fields

    def _extract_events(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract events from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of event information
        """
        events = []

        # Event pattern
        event_pattern = r"(?:^|\n)\s*(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override)\s+)*event\s+([\w<>]+)\s+(\w+)"

        for match in re.finditer(event_pattern, class_body):
            events.append(
                {
                    "name": match.group(4),
                    "type": match.group(3),
                    "visibility": match.group(1) or "private",
                    "modifiers": match.group(2).split() if match.group(2) else [],
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        # Unity Events
        unity_event_pattern = (
            r"(?:^|\n)\s*(?:(public|private|protected)\s+)?UnityEvent(?:<([^>]+)>)?\s+(\w+)"
        )

        for match in re.finditer(unity_event_pattern, class_body):
            events.append(
                {
                    "name": match.group(3),
                    "type": f"UnityEvent<{match.group(2)}>" if match.group(2) else "UnityEvent",
                    "visibility": match.group(1) or "public",
                    "is_unity_event": True,
                    "line": class_body[: match.start()].count("\n") + 1,
                }
            )

        return events

    def _extract_unity_methods(self, class_body: str) -> List[str]:
        """Extract Unity lifecycle methods.

        Args:
            class_body: Content of class body

        Returns:
            List of Unity method names
        """
        unity_methods = []

        unity_lifecycle = [
            "Awake",
            "Start",
            "OnEnable",
            "OnDisable",
            "OnDestroy",
            "Update",
            "FixedUpdate",
            "LateUpdate",
            "OnGUI",
            "OnRenderObject",
            "OnPreCull",
            "OnBecameVisible",
            "OnBecameInvisible",
            "OnWillRenderObject",
            "OnPreRender",
            "OnRenderImage",
            "OnPostRender",
            "OnDrawGizmos",
            "OnDrawGizmosSelected",
            "OnApplicationPause",
            "OnApplicationFocus",
            "OnApplicationQuit",
            "OnCollisionEnter",
            "OnCollisionStay",
            "OnCollisionExit",
            "OnCollisionEnter2D",
            "OnCollisionStay2D",
            "OnCollisionExit2D",
            "OnTriggerEnter",
            "OnTriggerStay",
            "OnTriggerExit",
            "OnTriggerEnter2D",
            "OnTriggerStay2D",
            "OnTriggerExit2D",
            "OnMouseDown",
            "OnMouseUp",
            "OnMouseEnter",
            "OnMouseExit",
            "OnMouseOver",
            "OnMouseDrag",
        ]

        for method in unity_lifecycle:
            if re.search(rf"\b{method}\s*\(", class_body):
                unity_methods.append(method)

        return unity_methods

    def _extract_coroutines(self, class_body: str) -> List[str]:
        """Extract Unity coroutines.

        Args:
            class_body: Content of class body

        Returns:
            List of coroutine names
        """
        coroutines = []

        coroutine_pattern = r"\bIEnumerator\s+(\w+)\s*\("

        for match in re.finditer(coroutine_pattern, class_body):
            coroutines.append(match.group(1))

        return coroutines

    def _extract_interface_methods(self, interface_body: str) -> List[Dict[str, Any]]:
        """Extract method signatures from interface body.

        Args:
            interface_body: Content of interface body

        Returns:
            List of method signatures
        """
        methods = []

        method_pattern = r"(?:^|\n)\s*([\w<>\[\]]+)\s+(\w+)\s*\(([^)]*)\)\s*;"

        for match in re.finditer(method_pattern, interface_body):
            methods.append(
                {
                    "return_type": match.group(1),
                    "name": match.group(2),
                    "parameters": self._parse_parameters(match.group(3)),
                }
            )

        # Property signatures in interfaces
        prop_pattern = r"(?:^|\n)\s*([\w<>\[\]]+)\s+(\w+)\s*\{\s*get;\s*(?:set;)?\s*\}"

        for match in re.finditer(prop_pattern, interface_body):
            methods.append(
                {
                    "type": "property",
                    "return_type": match.group(1),
                    "name": match.group(2),
                }
            )

        return methods

    def _extract_enum_values(self, enum_body: str) -> List[str]:
        """Extract enum values.

        Args:
            enum_body: Content of enum body

        Returns:
            List of enum value names
        """
        values = []

        # Simple pattern for enum values
        value_pattern = r"^\s*(\w+)(?:\s*=\s*[^,]+)?\s*,?"

        for match in re.finditer(value_pattern, enum_body, re.MULTILINE):
            value_name = match.group(1)
            if value_name:  # Filter out empty matches
                values.append(value_name)

        return values

    def _extract_global_functions(self, content: str) -> List[FunctionInfo]:
        """Extract top-level functions (rare in C# but possible in top-level programs).

        Args:
            content: C# source code

        Returns:
            List of FunctionInfo objects
        """
        functions = []

        # C# 9+ top-level programs might have functions
        # Look for functions outside of classes
        # This is a simplified approach
        if not re.search(r"\bclass\s+\w+", content) and not re.search(r"\bnamespace\s+", content):
            # Might be a top-level program
            func_pattern = (
                r"(?:static\s+)?(?:async\s+)?(?:[\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)\s*(?:\{|=>)"
            )

            for match in re.finditer(func_pattern, content):
                func_name = match.group(1)
                if func_name not in ["if", "for", "while", "switch"]:
                    functions.append(
                        FunctionInfo(
                            name=func_name,
                            line=content[: match.start()].count("\n") + 1,
                        )
                    )

        return functions

    def _extract_linq_queries(self, content: str) -> List[Dict[str, Any]]:
        """Extract LINQ queries.

        Args:
            content: C# source code

        Returns:
            List of LINQ query information
        """
        queries = []

        # Query syntax
        query_pattern = r"from\s+(\w+)\s+in\s+(\w+)(?:\s+where\s+[^;]+)?(?:\s+select\s+[^;]+)?"

        for match in re.finditer(query_pattern, content):
            queries.append(
                {
                    "type": "query_syntax",
                    "variable": match.group(1),
                    "source": match.group(2),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Method syntax (simplified)
        method_chains = re.findall(
            r"(\w+)\.(?:Where|Select|OrderBy|GroupBy|Join)\([^)]+\)", content
        )
        for chain in method_chains:
            queries.append(
                {
                    "type": "method_syntax",
                    "source": chain,
                }
            )

        return queries

    def _extract_attributes(self, attr_str: str) -> List[str]:
        """Extract attribute names from attribute string.

        Args:
            attr_str: Attribute string like "[Serializable, Obsolete]"

        Returns:
            List of attribute names
        """
        if not attr_str:
            return []

        # Normalize: collapse newlines, keep distinct blocks
        blocks = [b.strip() for b in re.findall(r"\[[^\]]+\]", attr_str)]
        names: List[str] = []
        for block in blocks:
            inner = block[1:-1].strip()  # remove [ ]
            # Split on commas that are not inside parentheses
            current = ""
            depth = 0
            for ch in inner:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                if ch == "," and depth == 0:
                    piece = current.strip()
                    if piece:
                        m = re.match(r"(\w+)", piece)
                        if m:
                            names.append(m.group(1))
                    current = ""
                else:
                    current += ch
            # Last piece
            piece = current.strip()
            if piece:
                m = re.match(r"(\w+)", piece)
                if m:
                    names.append(m.group(1))
        return names

    def _parse_parameters(self, params_str: str) -> List[Dict[str, str]]:
        """Parse method parameters.

        Args:
            params_str: Parameter string

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
            if char in "<([":
                depth += 1
            elif char in ">)]":
                depth -= 1
            elif char == "," and depth == 0:
                if current_param.strip():
                    param_dict = self._parse_single_parameter(current_param.strip())
                    if param_dict:
                        parameters.append(param_dict)
                current_param = ""
                continue
            current_param += char

        # Add last parameter
        if current_param.strip():
            param_dict = self._parse_single_parameter(current_param.strip())
            if param_dict:
                parameters.append(param_dict)

        return parameters

    def _parse_single_parameter(self, param_str: str) -> Optional[Dict[str, str]]:
        """Parse a single parameter.

        Args:
            param_str: Single parameter string

        Returns:
            Parameter dictionary or None
        """
        # Handle various parameter formats
        # Examples: "int x", "ref int x", "out string s", "int x = 5", "params int[] args"

        param_pattern = r"(?:(ref|out|in|params)\s+)?([\w<>\[\]\.]+)\s+(\w+)(?:\s*=\s*(.+))?"
        match = re.match(param_pattern, param_str)

        if match:
            return {
                "modifier": match.group(1),
                "type": match.group(2),
                "name": match.group(3),
                "default": match.group(4),
            }

        return None

    def _detect_framework(self, content: str) -> Optional[str]:
        """Detect which framework is being used.

        Args:
            content: C# source code

        Returns:
            Framework name or None
        """
        # Unity
        if self._is_unity_script(content):
            return "Unity"

        # ASP.NET Core
        if re.search(r"using\s+Microsoft\.AspNetCore", content):
            return "ASP.NET Core"

        # WPF
        if re.search(r"using\s+System\.Windows", content):
            return "WPF"

        # Xamarin
        if re.search(r"using\s+Xamarin", content):
            return "Xamarin"

        # MAUI
        if re.search(r"using\s+Microsoft\.Maui", content):
            return "MAUI"

        # Blazor
        if re.search(r"@page\s+", content) or re.search(
            r"using\s+Microsoft\.AspNetCore\.Components", content
        ):
            return "Blazor"

        # Console/Library
        if re.search(r"static\s+void\s+Main\s*\(", content):
            return "Console"

        return None

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: C# source code

        Returns:
            Number of code lines
        """
        count = 0
        in_multiline_comment = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Handle multiline comments
            if "/*" in line:
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
        """Count comment lines including XML documentation.

        Args:
            content: C# source code

        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_comment = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Single-line comments
            if stripped.startswith("//"):
                count += 1
                continue

            # XML documentation comments
            if stripped.startswith("///"):
                count += 1
                continue

            # Multi-line comments
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
