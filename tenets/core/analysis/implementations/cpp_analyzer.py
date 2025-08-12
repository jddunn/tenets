"""C/C++ code analyzer.

This module provides comprehensive analysis for C and C++ source files,
including headers, templates, and modern C++ features.
"""

import math
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


class CppAnalyzer(LanguageAnalyzer):
    """C/C++ code analyzer.

    Provides analysis for C and C++ files including:
    - Include directive analysis (system and local)
    - Class, struct, and union extraction
    - Template analysis
    - Function and method extraction
    - Namespace handling
    - Macro and preprocessor directive analysis
    - Modern C++ features (auto, lambdas, smart pointers)
    - STL usage detection
    - Memory management patterns

    Supports both C and C++ with appropriate feature detection.
    """

    language_name = "cpp"
    file_extensions = [".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".h++"]

    def __init__(self):
        """Initialize the C++ analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract includes from C/C++ code.

        Handles:
        - System includes: #include <iostream>
        - Local includes: #include "myheader.h"
        - Conditional includes with #ifdef
        - Include guards

        Args:
            content: C/C++ source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects representing includes
        """
        imports = []
        lines = content.split("\n")

        # Track preprocessor state
        ifdef_stack = []
        current_condition = True

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            # Handle preprocessor conditionals
            if stripped.startswith("#ifdef") or stripped.startswith("#ifndef"):
                condition = stripped.split()[1] if len(stripped.split()) > 1 else ""
                ifdef_stack.append(current_condition)
                # We'll track all includes regardless of conditionals for analysis
                continue
            elif stripped.startswith("#if"):
                ifdef_stack.append(current_condition)
                continue
            elif stripped.startswith("#else"):
                if ifdef_stack:
                    current_condition = not current_condition
                continue
            elif stripped.startswith("#elif"):
                continue
            elif stripped.startswith("#endif"):
                if ifdef_stack:
                    current_condition = ifdef_stack.pop()
                continue

            # System includes
            system_include = re.match(r"^\s*#\s*include\s*<([^>]+)>", line)
            if system_include:
                header = system_include.group(1)
                imports.append(
                    ImportInfo(
                        module=header,
                        line=i,
                        type="system",
                        is_relative=False,
                        is_stdlib=self._is_stdlib_header(header),
                        is_stl=self._is_stl_header(header),
                        conditional=len(ifdef_stack) > 0,
                    )
                )
                continue

            # Local includes
            local_include = re.match(r'^\s*#\s*include\s*"([^"]+)"', line)
            if local_include:
                header = local_include.group(1)
                imports.append(
                    ImportInfo(
                        module=header,
                        line=i,
                        type="local",
                        is_relative=True,
                        is_project_header=True,
                        conditional=len(ifdef_stack) > 0,
                    )
                )
                continue

        # Detect include guards
        self._detect_include_guards(content, imports)

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from C/C++ code.

        In C/C++, symbols are exported by default unless static.
        For headers, we extract declarations. For source files,
        we extract non-static definitions.

        Args:
            content: C/C++ source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols
        """
        exports = []
        is_header = file_path.suffix in [".h", ".hh", ".hpp", ".hxx", ".h++"]

        # Extract namespace if present
        namespace = self._extract_namespace(content)

        # Non-static functions
        func_pattern = r"^(?:template\s*<[^>]*>\s*)?(?!static)(?:(?:inline|extern|virtual|explicit|constexpr)\s+)*(?:[\w\s\*&:<>]+)\s+(\w+)\s*\([^)]*\)(?:\s*const)?(?:\s*noexcept)?(?:\s*override)?(?:\s*final)?(?:\s*=\s*0)?(?:\s*(?:\{|;))"

        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            # Filter out keywords
            if func_name not in [
                "if",
                "for",
                "while",
                "switch",
                "return",
                "delete",
                "new",
                "throw",
                "catch",
            ]:
                line_content = content[match.start() : match.end()]
                before_window = content[max(0, match.start() - 200) : match.start()]
                is_tmpl = (
                    ("template" in line_content)
                    or ("template" in before_window)
                    or self._is_template_function(content, match.start())
                )
                exports.append(
                    {
                        "name": func_name,
                        "type": "function",
                        "line": content[: match.start()].count("\n") + 1,
                        "namespace": namespace,
                        "is_inline": "inline" in line_content,
                        "is_virtual": "virtual" in line_content,
                        "is_pure_virtual": "= 0" in line_content,
                        "is_constexpr": "constexpr" in line_content,
                        "is_template": is_tmpl,
                    }
                )

        # Classes and structs (public by default in struct)
        class_pattern = r"\b(?:struct|(?<!enum\s)class)\s+(?:__declspec\([^)]+\)\s+)?(\w+)(?:\s*:\s*(?:public|private|protected)\s+[\w:]+)?(?:\s*\{|;)"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            is_struct = "struct" in match.group(0)
            # Find keyword position for accurate template check
            inner = match.group(0)
            kw = "struct" if "struct" in inner else "class"
            kw_pos = match.start() + inner.find(kw)

            exports.append(
                {
                    "name": class_name,
                    "type": "struct" if is_struct else "class",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                    "default_visibility": "public" if is_struct else "private",
                    "is_template": self._is_template_class(content, kw_pos),
                }
            )

        # Enums
        enum_pattern = r"\benum\s+(?:class\s+)?(\w+)(?:\s*:\s*\w+)?(?:\s*\{|;)"

        for match in re.finditer(enum_pattern, content):
            enum_name = match.group(1)
            is_enum_class = "enum class" in match.group(0)

            exports.append(
                {
                    "name": enum_name,
                    "type": "enum_class" if is_enum_class else "enum",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        # Unions
        union_pattern = r"\bunion\s+(\w+)(?:\s*\{|;)"

        for match in re.finditer(union_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "union",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        # Typedefs and using declarations
        typedef_pattern = r"\btypedef\s+.*?\s+(\w+)\s*;"

        for match in re.finditer(typedef_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "typedef",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        using_pattern = r"\busing\s+(\w+)\s*="

        for match in re.finditer(using_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "using_alias",
                    "line": content[: match.start()].count("\n") + 1,
                    "namespace": namespace,
                }
            )

        # Global variables (non-static)
        if not is_header:
            var_pattern = (
                r"^(?!static)(?:extern\s+)?(?:const\s+)?(?:[\w\s\*&:<>]+)\s+(\w+)\s*(?:=|;)"
            )

            for match in re.finditer(var_pattern, content, re.MULTILINE):
                var_name = match.group(1)
                if var_name not in [
                    "if",
                    "for",
                    "while",
                    "return",
                    "class",
                    "struct",
                    "enum",
                    "typedef",
                    "using",
                ]:
                    exports.append(
                        {
                            "name": var_name,
                            "type": "variable",
                            "line": content[: match.start()].count("\n") + 1,
                            "namespace": namespace,
                            "is_const": "const" in match.group(0),
                            "is_extern": "extern" in match.group(0),
                        }
                    )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from C/C++ file.

        Extracts:
        - Namespaces
        - Classes and structs with inheritance
        - Functions and methods
        - Templates
        - Macros and preprocessor directives
        - Global variables
        - Operator overloads

        Args:
            content: C/C++ source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Determine if it's C or C++
        is_cpp = self._is_cpp_file(file_path, content)
        structure.language_variant = "C++" if is_cpp else "C"

        # Extract namespaces (C++ only)
        if is_cpp:
            namespace_pattern = r"namespace\s+(\w+)\s*\{"
            for match in re.finditer(namespace_pattern, content):
                structure.namespaces.append(
                    {"name": match.group(1), "line": content[: match.start()].count("\n") + 1}
                )

        # Extract classes and structs
        class_pattern = r"(?:template\s*<[^>]+>\s*)?(?:struct|(?<!enum\s)class)\s+(\w+)(?:\s*:\s*((?:public|private|protected)\s+[\w:]+(?:\s*,\s*(?:public|private|protected)\s+[\w:]+)*))?"

        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            inheritance = match.group(2)

            # Parse inheritance
            bases = []
            if inheritance:
                for base in inheritance.split(","):
                    base = base.strip()
                    # Remove access specifier
                    base = re.sub(r"^(public|private|protected)\s+", "", base)
                    bases.append(base)

            # Find class body
            class_start = match.end()
            class_body = self._extract_class_body(content, class_start)

            # Extract methods and members
            methods = []
            fields = []

            if class_body:
                methods = self._extract_class_methods(class_body)
                fields = self._extract_class_fields(class_body)

            inner = match.group(0)
            kw = "struct" if "struct" in inner else "class"
            kw_pos = match.start() + inner.find(kw)
            class_info = ClassInfo(
                name=class_name,
                line=content[: match.start()].count("\n") + 1,
                bases=bases,
                methods=methods,
                fields=fields,
                is_struct="struct" in match.group(0),
                is_template=self._is_template_class(content, kw_pos),
            )

            structure.classes.append(class_info)

        # Extract standalone functions
        func_pattern = r"(?:template\s*<[^>]+>\s*)?(?:(?:inline|static|extern|virtual|explicit|constexpr)\s+)*(?:[\w\s\*&:<>]+)\s+(\w+)\s*\([^)]*\)(?:\s*const)?(?:\s*noexcept)?(?:\s*\{|;)"

        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)

            # Filter out keywords and methods
            if func_name in [
                "if",
                "for",
                "while",
                "switch",
                "return",
                "delete",
                "new",
                "throw",
                "catch",
            ]:
                continue

            # Check if it's inside a class (simple heuristic)
            if self._is_inside_class(content, match.start()):
                continue

            func_info = FunctionInfo(
                name=func_name,
                line=content[: match.start()].count("\n") + 1,
                is_static="static" in match.group(0),
                is_inline="inline" in match.group(0),
                is_constexpr="constexpr" in match.group(0),
                is_template="template" in content[max(0, match.start() - 100) : match.start()],
                is_exported="static" not in match.group(0),
            )

            structure.functions.append(func_info)

        # Extract templates
        template_pattern = r"template\s*<([^>]+)>\s*(?:class|struct|typename|function)\s+(\w+)"

        for match in re.finditer(template_pattern, content):
            structure.templates.append(
                {
                    "name": match.group(2),
                    "parameters": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Extract macros
        macro_pattern = r"^\s*#define\s+(\w+)(?:\([^)]*\))?"

        for match in re.finditer(macro_pattern, content, re.MULTILINE):
            macro_name = match.group(1)
            is_function_macro = "(" in match.group(0)

            structure.macros.append(
                {
                    "name": macro_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "is_function_macro": is_function_macro,
                }
            )

        # Extract global variables
        global_var_pattern = (
            r"^(?:static\s+)?(?:const\s+)?(?:[\w\s\*&:<>]+)\s+(\w+)\s*(?:=\s*[^;]+)?\s*;"
        )

        for match in re.finditer(global_var_pattern, content, re.MULTILINE):
            var_name = match.group(1)

            # Filter out function declarations and keywords
            if var_name in ["if", "for", "while", "return", "class", "struct", "enum", "typedef"]:
                continue

            if not self._is_inside_class(content, match.start()) and not self._is_inside_function(
                content, match.start()
            ):
                structure.variables.append(
                    {
                        "name": var_name,
                        "line": content[: match.start()].count("\n") + 1,
                        "type": "global",
                        "is_static": "static" in match.group(0),
                        "is_const": "const" in match.group(0),
                    }
                )

        # Extract unions
        union_pattern = r"union\s+(\w+)\s*\{"

        for match in re.finditer(union_pattern, content):
            structure.unions.append(
                {"name": match.group(1), "line": content[: match.start()].count("\n") + 1}
            )

        # Extract operator overloads
        operator_pattern = r"operator\s*(?:[\+\-\*\/\%\^\&\|\~\!\=\<\>\[\]\(\)]|\+\+|\-\-|\<\<|\>\>|\=\=|\!\=|\<\=|\>\=|\&\&|\|\||\+\=|\-\=|\*\=|\/\=|\%\=|\^\=|\&\=|\|\=|\<\<\=|\>\>\=|,|->\*?|new|delete)(?:\s*\[\])?"

        operator_count = len(re.findall(operator_pattern, content))
        structure.operator_overloads = operator_count

        # Detect STL usage (boolean for test compatibility)
        stl_types_found = self._detect_stl_usage(content)
        structure.uses_stl = bool(stl_types_found)
        structure.stl_types = stl_types_found  # Optionally keep the list for other uses

        # Detect smart pointers
        structure.smart_pointers = self._detect_smart_pointers(content)

        # Count lambda expressions
        lambda_pattern = r"\[[^\]]*\]\s*\([^)]*\)\s*(?:->[\w\s]+)?\s*\{"
        structure.lambda_count = len(re.findall(lambda_pattern, content))

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for C/C++ code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Preprocessor complexity
        - Template complexity
        - Memory management complexity

        Args:
            content: C/C++ source code
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
            r"\b&&\b",
            r"\|\|",
            r"\?",
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
            # Skip comments and preprocessor directives
            if (
                line.strip().startswith("//")
                or line.strip().startswith("/*")
                or line.strip().startswith("#")
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
                (r"\bfor\b", 1),
                (r"\bwhile\b", 1),
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

        # Count functions
        metrics.function_count = len(re.findall(r"[\w\s\*&:<>]+\s+\w+\s*\([^)]*\)\s*\{", content))

        # Count classes and structs
        metrics.class_count = len(re.findall(r"\b(?:class|struct)\s+\w+", content))

        # Template metrics
        metrics.template_count = len(re.findall(r"template\s*<", content))
        metrics.template_specializations = len(re.findall(r"template\s*<>", content))

        # Preprocessor metrics
        metrics.macro_count = len(re.findall(r"^\s*#define\s+", content, re.MULTILINE))
        metrics.ifdef_count = len(re.findall(r"^\s*#if(?:def|ndef)?\s+", content, re.MULTILINE))
        metrics.include_count = len(re.findall(r"^\s*#include\s+", content, re.MULTILINE))

        # Memory management metrics
        metrics.new_count = len(re.findall(r"\bnew\s+", content))
        # Count delete and delete[]
        metrics.delete_count = len(re.findall(r"\bdelete\s*(?:\[\])?", content))
        metrics.malloc_count = len(re.findall(r"\bmalloc\s*\(", content))
        metrics.free_count = len(re.findall(r"\bfree\s*\(", content))

        # Smart pointer usage (count both types and factory helpers)
        metrics.unique_ptr_count = len(re.findall(r"\bunique_ptr\s*<", content)) + len(
            re.findall(r"(?:\b[\w:]+::)?make_unique(?:\s*<[^>]+>)?\s*\(", content)
        )
        metrics.shared_ptr_count = len(re.findall(r"\bshared_ptr\s*<", content)) + len(
            re.findall(r"(?:\b[\w:]+::)?make_shared(?:\s*<[^>]+>)?\s*\(", content)
        )
        metrics.weak_ptr_count = len(re.findall(r"\bweak_ptr\s*<", content))

        # RAII indicators
        metrics.uses_raii = (
            metrics.unique_ptr_count > 0 or metrics.shared_ptr_count > 0 or "RAII" in content
        )

        # Calculate memory safety score
        manual_memory = (
            metrics.new_count + metrics.delete_count + metrics.malloc_count + metrics.free_count
        )
        smart_memory = metrics.unique_ptr_count + metrics.shared_ptr_count

        if manual_memory + smart_memory > 0:
            metrics.memory_safety_score = smart_memory / (manual_memory + smart_memory)
        else:
            metrics.memory_safety_score = 1.0

        # Calculate maintainability index
        if metrics.code_lines > 0:
            # Adjusted for C++ complexity
            template_factor = 1 - (metrics.template_count * 0.02)
            memory_factor = metrics.memory_safety_score

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * template_factor
                + 15 * memory_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _is_stdlib_header(self, header: str) -> bool:
        """Check if a header is from C standard library.

        Args:
            header: Header name

        Returns:
            True if it's a C stdlib header
        """
        c_stdlib = {
            "stdio.h",
            "stdlib.h",
            "string.h",
            "math.h",
            "time.h",
            "ctype.h",
            "limits.h",
            "float.h",
            "assert.h",
            "errno.h",
            "signal.h",
            "setjmp.h",
            "stdarg.h",
            "stddef.h",
            "locale.h",
            "wchar.h",
            "wctype.h",
            "stdint.h",
            "inttypes.h",
            "stdbool.h",
        }

        # C++ versions
        cpp_stdlib = {
            "cstdio",
            "cstdlib",
            "cstring",
            "cmath",
            "ctime",
            "cctype",
            "climits",
            "cfloat",
            "cassert",
            "cerrno",
            "csignal",
            "csetjmp",
            "cstdarg",
            "cstddef",
            "clocale",
            "cwchar",
            "cwctype",
            "cstdint",
            "cinttypes",
        }

        return header in c_stdlib or header in cpp_stdlib

    def _is_stl_header(self, header: str) -> bool:
        """Check if a header is from C++ STL.

        Args:
            header: Header name

        Returns:
            True if it's an STL header
        """
        stl_headers = {
            "iostream",
            "fstream",
            "sstream",
            "iomanip",
            "string",
            "vector",
            "list",
            "deque",
            "queue",
            "stack",
            "set",
            "map",
            "unordered_set",
            "unordered_map",
            "algorithm",
            "iterator",
            "functional",
            "numeric",
            "memory",
            "utility",
            "tuple",
            "array",
            "bitset",
            "regex",
            "thread",
            "mutex",
            "condition_variable",
            "future",
            "chrono",
            "random",
            "complex",
            "valarray",
            "exception",
            "stdexcept",
            "typeinfo",
            "type_traits",
        }

        return header in stl_headers

    def _detect_include_guards(self, content: str, imports: List[ImportInfo]) -> None:
        """Detect and mark include guard patterns.

        Args:
            content: Source code
            imports: List of imports to update
        """
        # Classic include guard pattern
        guard_pattern = r"^#ifndef\s+(\w+)\s*\n#define\s+\1"
        if re.search(guard_pattern, content, re.MULTILINE):
            for imp in imports:
                imp.has_include_guard = True

        # Modern pragma once
        if "#pragma once" in content:
            for imp in imports:
                imp.uses_pragma_once = True

    def _extract_namespace(self, content: str) -> Optional[str]:
        """Extract the primary namespace from the file.

        Args:
            content: Source code

        Returns:
            Namespace name or None
        """
        namespace_match = re.search(r"namespace\s+(\w+)\s*\{", content)
        return namespace_match.group(1) if namespace_match else None

    def _is_cpp_file(self, file_path: Path, content: str) -> bool:
        """Determine if the file is C++ (vs plain C).

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            True if it's a C++ file
        """
        # Check file extension
        cpp_extensions = {".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hxx", ".h++"}
        if file_path.suffix in cpp_extensions:
            return True

        # Check for C++ keywords
        cpp_keywords = [
            r"\bclass\b",
            r"\bnamespace\b",
            r"\btemplate\b",
            r"\bvirtual\b",
            r"\boverride\b",
            r"\bfinal\b",
            r"\bpublic:",
            r"\bprivate:",
            r"\bprotected:",
            r"\bstd::",
            r"\busing\s+namespace\b",
            r"\bnullptr\b",
            r"\bauto\b",
            r"\bdecltype\b",
            r"\bconstexpr\b",
        ]

        for keyword in cpp_keywords:
            if re.search(keyword, content):
                return True

        # Check for C++ headers
        cpp_headers = ["iostream", "string", "vector", "map", "algorithm"]
        for header in cpp_headers:
            if f"#include <{header}>" in content:
                return True

        return False

    def _extract_class_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a class/struct.

        Args:
            content: Source code
            start_pos: Position after class declaration

        Returns:
            Class body content or None
        """
        # Find opening brace
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

    def _extract_class_methods(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract methods from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of method information
        """
        methods = []
        current_visibility = "private"  # Default for class

        for line in class_body.split("\n"):
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Track visibility changes
            if re.match(r"^public:", line):
                current_visibility = "public"
                continue
            elif re.match(r"^private:", line):
                current_visibility = "private"
                continue
            elif re.match(r"^protected:", line):
                current_visibility = "protected"
                continue

            # Skip obvious field declarations only when no parentheses are present (to avoid skipping prototypes)
            if "(" not in line and re.match(
                r"^(?:static\s+)?(?:const\s+)?[\w:\s\*&<>]+\s+\w+(?:\s*\[[^\]]*\])?\s*(?:=\s*[^;]+)?\s*;\s*$",
                line,
            ):
                continue

            # Method patterns - try multiple patterns
            patterns = [
                r"^\s*(~?\w+)\s*\(\s*\)\s*;",  # ctor/dtor no params
                r"^\s*(?:virtual\s+|static\s+|inline\s+)*void\s+(~?\w+)\s*\([^)]*\)\s*(?:const\s+)?(?:override\s+)?(?:final\s+)?(?:=\s*(?:0|default|delete)\s*)?[{;]",
                r"^\s*(?:virtual\s+|static\s+|inline\s+)*(?:[\w:<>]+(?:\s*[\*&])?\s+)+(~?\w+)\s*\([^)]*\)\s*(?:const\s+)?(?:override\s+)?(?:final\s+)?(?:=\s*(?:0|default|delete)\s*)?[{;]",
                r"^\s*(?:explicit\s+)?(~?\w+)\s*\([^)]*\)\s*(?:[{;])",
            ]

            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    method_name = match.group(1)
                    if method_name not in [
                        "if",
                        "for",
                        "while",
                        "switch",
                        "return",
                        "int",
                        "float",
                        "double",
                        "char",
                        "bool",
                        "void",
                    ]:
                        methods.append(
                            {
                                "name": method_name,
                                "visibility": current_visibility,
                                "is_virtual": "virtual" in line,
                                "is_static": "static" in line,
                                "is_const": "const" in line,
                                "is_override": "override" in line,
                                "is_final": "final" in line,
                                "is_pure_virtual": "= 0" in line,
                                "is_default": "= default" in line,
                                "is_deleted": "= delete" in line,
                            }
                        )
                        break

        return methods

    def _is_template_function(self, content: str, func_pos: int) -> bool:
        """Check if a function is a template.

        Looks back a reasonable window and checks recent non-empty lines
        for a preceding template<...> declaration, ignoring comments and labels.
        """
        window = content[max(0, func_pos - 1000) : func_pos]
        # Quick substring check first
        if re.search(r"template\s*<[^>]*>", window):
            # Be a bit stricter by scanning the last few logical lines
            lines = [l.strip() for l in window.splitlines() if l.strip()]
            for l in reversed(lines[-10:]):
                if l.startswith("//"):
                    continue
                if l.startswith("template"):
                    return True
                # Stop if another declaration boundary is encountered
                if re.match(r"(?:class|struct|enum|namespace)\b", l):
                    break
        return False

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment, non-preprocessor lines."""
        code_lines = 0
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            if line.startswith("/*") or line.startswith("*") or line.endswith("*/"):
                continue
            code_lines += 1
        return code_lines

    def _count_comment_lines(self, content: str) -> int:
        """Count lines that are comments (//, /*, */)."""
        comment_lines = 0
        in_block = False
        for line in content.splitlines():
            l = line.strip()
            if in_block:
                comment_lines += 1
                if "*/" in l:
                    in_block = False
                continue
            if l.startswith("//"):
                comment_lines += 1
            elif l.startswith("/*"):
                comment_lines += 1
                if not "*/" in l:
                    in_block = True
        return comment_lines

    def _is_template_class(self, content: str, class_pos: int) -> bool:
        """Detect if a class is a template by looking for template<...> before the class."""
        window = content[max(0, class_pos - 1000) : class_pos]
        # Look for template<...> in the last few lines before class
        lines = [l.strip() for l in window.splitlines() if l.strip()]
        for l in reversed(lines[-10:]):
            if l.startswith("//"):
                continue
            if l.startswith("template"):
                return True
            if re.match(r"(?:class|struct|enum|namespace)\b", l):
                break
        return False

    def _extract_class_fields(self, class_body: str) -> list:
        """Extract field declarations from a class body."""
        fields = []
        for line in class_body.split("\n"):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # Skip method declarations
            if "(" in line and ")" in line:
                continue
            # Match field declarations
            m = re.match(
                r"^(?:static\s+)?(?:const\s+)?([\w:\s\*&<>]+)\s+(\w+)(?:\s*\[[^\]]*\])?\s*(?:=\s*[^;]+)?\s*;\s*$",
                line,
            )
            if m:
                type_str = m.group(1).strip()
                name = m.group(2)
                fields.append({"name": name, "type": type_str})
        return fields

    def _is_inside_class(self, content: str, pos: int) -> bool:
        """Heuristic to check if a position is inside a class body."""
        # Look for the nearest preceding 'class' or 'struct' and its '{', and the next '}'
        before = content[:pos]
        after = content[pos:]
        class_match = list(re.finditer(r"(class|struct)\s+\w+[^;{]*{", before))
        if not class_match:
            return False
        last_class = class_match[-1]
        open_brace = before.find("{", last_class.end() - 1)
        if open_brace == -1:
            return False
        # Find matching closing brace
        brace_count = 1
        i = open_brace + 1
        while i < len(content) and brace_count > 0:
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
            i += 1
        return open_brace < pos < i

    def _is_inside_function(self, content: str, pos: int) -> bool:
        """Heuristic to check if a position is inside a function body."""
        # Look for the nearest preceding ')' and '{', and the next '}'
        before = content[:pos]
        after = content[pos:]
        func_match = list(re.finditer(r"\)\s*{", before))
        if not func_match:
            return False
        last_func = func_match[-1]
        open_brace = before.find("{", last_func.start())
        if open_brace == -1:
            return False
        # Find matching closing brace
        brace_count = 1
        i = open_brace + 1
        while i < len(content) and brace_count > 0:
            if content[i] == "{":
                brace_count += 1
            elif content[i] == "}":
                brace_count -= 1
            i += 1
        return open_brace < pos < i

    def _detect_stl_usage(self, content: str) -> list:
        """Detect usage of STL headers/types."""
        stl_types = [
            "vector",
            "map",
            "set",
            "unordered_map",
            "unordered_set",
            "list",
            "deque",
            "queue",
            "stack",
            "array",
            "string",
            "tuple",
            "pair",
        ]
        found = set()
        for t in stl_types:
            if re.search(r"\bstd::" + t + r"\b", content):
                found.add(t)
        return list(found)

    def _detect_smart_pointers(self, content: str) -> list:
        """Detect usage of smart pointers."""
        smart_ptrs = ["unique_ptr", "shared_ptr", "weak_ptr"]
        found = set()
        for t in smart_ptrs:
            if re.search(r"\bstd::" + t + r"\b", content):
                found.add(t)
        return list(found)
