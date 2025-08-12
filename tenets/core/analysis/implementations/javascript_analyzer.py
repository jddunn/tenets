"""JavaScript and TypeScript code analyzer.

This module provides comprehensive analysis for JavaScript and TypeScript files,
including ES6+ features, JSX, CommonJS, and TypeScript-specific constructs.
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


class JavaScriptAnalyzer(LanguageAnalyzer):
    """JavaScript/TypeScript code analyzer.

    Provides analysis for JavaScript and TypeScript files including:
    - Import/export analysis (ES6 modules and CommonJS)
    - Function and class extraction (including arrow functions)
    - React component detection
    - TypeScript interface and type analysis
    - Complexity metrics for JS/TS code
    - Framework detection (React, Vue, Angular)
    - JSX/TSX support

    This analyzer uses regex-based parsing optimized for JavaScript's
    flexible syntax and various module systems.
    """

    language_name = "javascript"
    file_extensions = [".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"]

    def __init__(self):
        """Initialize the JavaScript analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract imports from JavaScript/TypeScript code.

        Handles multiple import styles:
        - ES6 imports: import x from 'module'
        - Named imports: import { x, y } from 'module'
        - Namespace imports: import * as x from 'module'
        - Side-effect imports: import 'module'
        - Dynamic imports: import('module')
        - CommonJS: require('module')
        - TypeScript type imports: import type { X } from 'module'

        Args:
            content: JavaScript/TypeScript source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        # ES6 import patterns
        es6_default = re.compile(r'^\s*import\s+(\w+)\s+from\s+[\'"`]([^\'"`]+)[\'"`]')
        es6_named = re.compile(r'^\s*import\s*\{([^}]+)\}\s*from\s+[\'"`]([^\'"`]+)[\'"`]')
        es6_namespace = re.compile(r'^\s*import\s*\*\s*as\s+(\w+)\s+from\s+[\'"`]([^\'"`]+)[\'"`]')
        es6_combined = re.compile(
            r'^\s*import\s+(\w+)\s*,\s*\{([^}]+)\}\s*from\s+[\'"`]([^\'"`]+)[\'"`]'
        )
        es6_side_effect = re.compile(r'^\s*import\s+[\'"`]([^\'"`]+)[\'"`]')

        # TypeScript type imports
        ts_type_import = re.compile(
            r'^\s*import\s+type\s+(?:\{([^}]+)\}|(\w+))\s+from\s+[\'"`]([^\'"`]+)[\'"`]'
        )

        # CommonJS patterns
        require_pattern = re.compile(
            r'(?:const|let|var)\s+(\w+)\s*=\s*require\s*\([\'"`]([^\'"`]+)[\'"`]\)'
        )
        require_destructure = re.compile(
            r'(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\s*\([\'"`]([^\'"`]+)[\'"`]\)'
        )

        # Dynamic import pattern
        dynamic_import = re.compile(r'import\s*\([\'"`]([^\'"`]+)[\'"`]\)')

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # ES6 default import
            match = es6_default.match(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(2),
                        alias=match.group(1),
                        line=i,
                        type="es6_default",
                        is_relative=match.group(2).startswith("."),
                        import_clause=match.group(1),
                    )
                )
                continue

            # ES6 named imports
            match = es6_named.match(line)
            if match:
                named_imports = match.group(1)
                module = match.group(2)

                # Parse individual named imports
                for name in named_imports.split(","):
                    name = name.strip()
                    if " as " in name:
                        original, alias = name.split(" as ")
                        imports.append(
                            ImportInfo(
                                module=module,
                                alias=alias.strip(),
                                line=i,
                                type="es6_named",
                                is_relative=module.startswith("."),
                                original_name=original.strip(),
                            )
                        )
                    else:
                        imports.append(
                            ImportInfo(
                                module=module,
                                alias=name,
                                line=i,
                                type="es6_named",
                                is_relative=module.startswith("."),
                                import_clause=name,
                            )
                        )
                continue

            # ES6 namespace import
            match = es6_namespace.match(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(2),
                        alias=match.group(1),
                        line=i,
                        type="es6_namespace",
                        is_relative=match.group(2).startswith("."),
                    )
                )
                continue

            # ES6 combined import (default + named)
            match = es6_combined.match(line)
            if match:
                module = match.group(3)
                # Default import
                imports.append(
                    ImportInfo(
                        module=module,
                        alias=match.group(1),
                        line=i,
                        type="es6_default",
                        is_relative=module.startswith("."),
                    )
                )
                # Named imports
                for name in match.group(2).split(","):
                    name = name.strip()
                    imports.append(
                        ImportInfo(
                            module=module,
                            alias=name,
                            line=i,
                            type="es6_named",
                            is_relative=module.startswith("."),
                        )
                    )
                continue

            # ES6 side-effect import
            match = es6_side_effect.match(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="es6_side_effect",
                        is_relative=match.group(1).startswith("."),
                    )
                )
                continue

            # TypeScript type imports
            match = ts_type_import.match(line)
            if match:
                module = match.group(3)
                if match.group(1):  # Named type imports
                    for name in match.group(1).split(","):
                        imports.append(
                            ImportInfo(
                                module=module,
                                alias=name.strip(),
                                line=i,
                                type="ts_type",
                                is_relative=module.startswith("."),
                            )
                        )
                else:  # Default type import
                    imports.append(
                        ImportInfo(
                            module=module,
                            alias=match.group(2),
                            line=i,
                            type="ts_type",
                            is_relative=module.startswith("."),
                        )
                    )
                continue

            # CommonJS require
            match = require_pattern.search(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(2),
                        alias=match.group(1),
                        line=i,
                        type="commonjs",
                        is_relative=match.group(2).startswith("."),
                    )
                )

            # CommonJS destructured require
            match = require_destructure.search(line)
            if match:
                module = match.group(2)
                for name in match.group(1).split(","):
                    name = name.strip()
                    imports.append(
                        ImportInfo(
                            module=module,
                            alias=name,
                            line=i,
                            type="commonjs_destructured",
                            is_relative=module.startswith("."),
                        )
                    )

            # Dynamic imports
            for match in dynamic_import.finditer(line):
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="dynamic",
                        is_relative=match.group(1).startswith("."),
                    )
                )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exports from JavaScript/TypeScript code.

        Handles multiple export styles:
        - ES6 default exports
        - ES6 named exports
        - ES6 export from
        - CommonJS module.exports
        - CommonJS exports.x
        - TypeScript type exports

        Args:
            content: JavaScript/TypeScript source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols with metadata
        """
        exports = []
        lines = content.split("\n")

        # ES6 export patterns
        export_default = re.compile(r"^\s*export\s+default\s+(.+)")
        export_named_declaration = re.compile(
            r"^\s*export\s+(?:async\s+)?(?:const|let|var|function|class|interface|type|enum)\s+(\w+)"
        )
        export_list = re.compile(r"^\s*export\s*\{([^}]+)\}")
        export_from = re.compile(r'^\s*export\s*\{([^}]+)\}\s*from\s+[\'"`]([^\'"`]+)[\'"`]')
        export_all = re.compile(
            r'^\s*export\s*\*\s*(?:as\s+(\w+)\s+)?from\s+[\'"`]([^\'"`]+)[\'"`]'
        )

        # TypeScript type exports
        ts_type_export = re.compile(r"^\s*export\s+type\s+(?:\{([^}]+)\}|(\w+))")

        # CommonJS patterns
        module_exports = re.compile(r"module\.exports\s*=\s*(.+)")
        exports_prop = re.compile(r"exports\.(\w+)\s*=")

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Default export
            match = export_default.match(line)
            if match:
                value = match.group(1).strip()
                export_type = (
                    "function"
                    if "function" in value
                    else "class" if "class" in value else "object" if "{" in value else "default"
                )

                exports.append(
                    {
                        "name": "default",
                        "type": export_type,
                        "line": i,
                        "value": value[:50] if len(value) > 50 else value,
                        "is_default": True,
                    }
                )
                continue

            # Named export declarations
            match = export_named_declaration.match(line)
            if match:
                name = match.group(1)
                export_type = (
                    "function"
                    if ("function" in line or re.search(r"^\s*export\s+async\s+function", line))
                    else (
                        "class"
                        if "class" in line
                        else (
                            "interface"
                            if "interface" in line
                            else (
                                "type"
                                if "type" in line
                                else "enum" if "enum" in line else "variable"
                            )
                        )
                    )
                )

                exports.append(
                    {
                        "name": name,
                        "type": export_type,
                        "line": i,
                        "is_const": "const" in line,
                        "is_async": "async" in line,
                    }
                )
                continue

            # Export list
            match = export_list.match(line)
            if match and "from" not in line:
                names = match.group(1)
                for name in names.split(","):
                    name = name.strip()
                    if " as " in name:
                        original, exported = name.split(" as ")
                        exports.append(
                            {
                                "name": exported.strip(),
                                "original_name": original.strip(),
                                "type": "named",
                                "line": i,
                            }
                        )
                    else:
                        exports.append({"name": name, "type": "named", "line": i})
                continue

            # Export from
            match = export_from.match(line)
            if match:
                names = match.group(1)
                module = match.group(2)
                for name in names.split(","):
                    name = name.strip()
                    exports.append(
                        {"name": name, "type": "re-export", "from_module": module, "line": i}
                    )
                continue

            # Export all from
            match = export_all.match(line)
            if match:
                exports.append(
                    {
                        "name": match.group(1) or "*",
                        "type": "re-export-all",
                        "from_module": match.group(2),
                        "line": i,
                    }
                )
                continue

            # TypeScript type exports
            match = ts_type_export.match(line)
            if match:
                if match.group(1):  # Export type list
                    for name in match.group(1).split(","):
                        exports.append({"name": name.strip(), "type": "type", "line": i})
                else:  # Single type export
                    exports.append({"name": match.group(2), "type": "type", "line": i})
                continue

            # CommonJS module.exports
            match = module_exports.search(line)
            if match:
                value = match.group(1).strip()
                exports.append(
                    {
                        "name": "module.exports",
                        "type": "commonjs",
                        "line": i,
                        "value": value[:50] if len(value) > 50 else value,
                    }
                )

            # CommonJS exports.property
            for match in exports_prop.finditer(line):
                exports.append({"name": match.group(1), "type": "commonjs_property", "line": i})

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from JavaScript/TypeScript file.

        Extracts:
        - Functions (regular, arrow, async, generator)
        - Classes (ES6 classes with inheritance)
        - Methods and properties
        - React components (class and functional)
        - TypeScript interfaces and types
        - Constants and variables

        Args:
            content: JavaScript/TypeScript source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()
        lines = content.split("\n")
        is_typescript = file_path.suffix in [".ts", ".tsx"]

        # Function patterns
        function_pattern = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*\(([^)]*)\)"
        )
        arrow_function = re.compile(
            r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*[^=]+)?\s*=>"
        )
        method_pattern = re.compile(r"^\s*(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*(?::\s*[^{]+)?\s*\{")

        # Class patterns
        class_pattern = re.compile(
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?"
        )

        # TypeScript patterns
        interface_pattern = re.compile(
            r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?"
        )
        type_pattern = re.compile(r"(?:export\s+)?type\s+(\w+)\s*=")
        enum_pattern = re.compile(r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)")

        # Variable patterns
        const_pattern = re.compile(r"(?:export\s+)?const\s+(\w+)\s*[=:]")
        let_var_pattern = re.compile(r"(?:export\s+)?(?:let|var)\s+(\w+)\s*[=:]")

        # React patterns
        react_component = re.compile(
            r"(?:export\s+)?(?:(?:const|let|var)\s+([A-Z][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|function\s+([A-Z][A-Za-z0-9_]*)\s*\(|class\s+([A-Z][A-Za-z0-9_]*)\b)",
            re.MULTILINE,
        )
        # Additional JSX arrow inline handlers
        jsx_arrow = re.compile(r"=>\s*\(")

        # Track current context
        current_class = None
        brace_depth = 0
        in_class = False

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Track brace depth for class context
            brace_depth += line.count("{") - line.count("}")
            if in_class and brace_depth == 0:
                in_class = False
                current_class = None

            # Functions
            match = function_pattern.search(line)
            if match and not in_class:
                func_info = FunctionInfo(
                    name=match.group(1),
                    line=i,
                    args=self._parse_js_params(match.group(2)),
                    is_async="async" in line,
                    is_generator="*" in line,
                    is_exported="export" in line,
                )
                structure.functions.append(func_info)
                continue

            # Arrow functions
            match = arrow_function.search(line)
            if match and not in_class:
                func_info = FunctionInfo(
                    name=match.group(1),
                    line=i,
                    args=self._parse_js_params(match.group(2)),
                    is_async="async" in line,
                    is_arrow=True,
                    is_exported="export" in line,
                )
                structure.functions.append(func_info)
                continue

            # Classes
            match = class_pattern.search(line)
            if match:
                class_info = ClassInfo(
                    name=match.group(1),
                    line=i,
                    bases=[match.group(2)] if match.group(2) else [],
                    interfaces=match.group(3).split(",") if match.group(3) else [],
                    is_abstract="abstract" in line,
                    is_exported="export" in line,
                )
                structure.classes.append(class_info)
                current_class = class_info
                in_class = True
                brace_depth = 1
                continue

            # Methods (inside classes)
            if in_class and current_class:
                # Match methods including private, getters/setters, and static
                method_match = re.match(r"^\s*(?:static\s+)?(?:async\s+)?(?:(get|set)\s+)?(#?\w+)\s*\(([^)]*)\)\s*\{", line)
                if method_match:
                    method_name = method_match.group(2)
                    if method_name not in ["if", "for", "while", "switch", "catch"]:
                        method_info = {
                            "name": method_name,  # preserve '#' for private
                            "line": i,
                            "args": self._parse_js_params(method_match.group(3)),
                            "is_async": "async" in line,
                            "is_static": line.lstrip().startswith("static "),
                            "is_private": method_name.startswith("#"),
                            "is_constructor": method_name == "constructor",
                        }
                        current_class.methods.append(method_info)

            # TypeScript interfaces
            if is_typescript:
                match = interface_pattern.search(line)
                if match:
                    structure.interfaces.append(
                        {
                            "name": match.group(1),
                            "line": i,
                            "extends": [e.strip() for e in match.group(2).split(",")] if match.group(2) else [],
                            "is_exported": "export" in line,
                        }
                    )
                    continue

                # TypeScript types
                match = type_pattern.search(line)
                if match:
                    structure.types.append(
                        {"name": match.group(1), "line": i, "is_exported": "export" in line}
                    )
                    continue

                # TypeScript enums
                match = enum_pattern.search(line)
                if match:
                    structure.enums.append(
                        {
                            "name": match.group(1),
                            "line": i,
                            "is_const": "const enum" in line,
                            "is_exported": "export" in line,
                        }
                    )
                    continue

            # React components: scan the whole content to catch multiline JSX and memo wrappers
            if i == 1:
                seen = set()
                for m in react_component.finditer(content):
                    comp = next((g for g in m.groups() if g), None)
                    if comp and comp[0].isupper() and comp not in seen:
                        frag = m.group(0).lstrip()
                        comp_type = "class" if frag.startswith("class ") else "functional"
                        line_no = content[: m.start()].count("\n") + 1
                        structure.components.append(
                            {
                                "name": comp,
                                "type": comp_type,
                                "line": line_no,
                                "is_exported": "export" in frag,
                            }
                        )
                        seen.add(comp)
            # Detect memoized components assigned from React.memo
            memo_assign = re.compile(r"(?:const|let|var)\s+([A-Z][A-Za-z0-9_]*)\s*=\s*React\.memo\s*\(")
            # corrected pattern (no quote after parenthesis)
            memo_assign = re.compile(r"(?:const|let|var)\s+([A-Z][A-Za-z0-9_]*)\s*=\s*React\.memo\s*\(")
            for m in memo_assign.finditer(content):
                comp_name = m.group(1)
                line_no = content[: m.start()].count("\n") + 1
                structure.components.append(
                    {"name": comp_name, "type": "functional", "line": line_no, "is_exported": False}
                )

            # Constants
            match = const_pattern.search(line)
            if match and not in_class:
                var_name = match.group(1)
                structure.variables.append(
                    {"name": var_name, "line": i, "type": "const", "is_exported": "export" in line}
                )
                if var_name.isupper():
                    structure.constants.append(var_name)

            # Variables (let/var)
            match = let_var_pattern.search(line)
            if match and not in_class:
                structure.variables.append(
                    {
                        "name": match.group(1),
                        "line": i,
                        "type": "let" if "let" in line else "var",
                        "is_exported": "export" in line,
                    }
                )

        # Detect framework
        structure.framework = self._detect_framework(content)

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for JavaScript/TypeScript code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Nesting depth
        - Function and class counts
        - Comment ratio

        Args:
            content: JavaScript/TypeScript source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Calculate cyclomatic complexity
        complexity = 1  # Base complexity

        # Decision point patterns
        decision_keywords = [
            r"\bif\b",
            r"\belse\s+if\b",
            r"\belse\b",
            r"\bwhile\b",
            r"\bfor\b",
            r"\bdo\b",
            r"\bswitch\b",
            r"\bcase\b",
            r"\bcatch\b",
            r"\bfinally\b",
            r"\?",  # Count ternary operators by '?'
            r"\|\|",
            r"&&",
            r"\?\?",  # Nullish coalescing
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
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Track nesting
            opening_braces = line.count("{")
            closing_braces = line.count("}")
            nesting_level += opening_braces - closing_braces
            max_nesting = max(max_nesting, nesting_level)

            # Add complexity for control structures with nesting penalty
            control_structures = [
                (r"\bif\b", 1),
                (r"\belse\s+if\b", 1),
                (r"\belse\b", 0),
                (r"\bfor\b", 1),
                (r"\bwhile\b", 1),
                (r"\bdo\b", 1),
                (r"\bswitch\b", 1),
                (r"\bcatch\b", 1),
            ]

            for pattern, base_score in control_structures:
                if re.search(pattern, line):
                    cognitive += base_score + max(0, nesting_level - 1)

            # Add complexity for nested ternary operators
            ternary_count = len(re.findall(r"\?", line))
            if ternary_count > 0:
                cognitive += ternary_count * (1 + max(0, nesting_level - 1))

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
        function_patterns = [
            r"function\s+\w+",
            r"=>",  # Arrow functions
            r"^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{",  # Methods
        ]

        function_count = 0
        for pattern in function_patterns:
            function_count += len(re.findall(pattern, content))
        metrics.function_count = function_count

        # Count classes
        metrics.class_count = len(re.findall(r"class\s+\w+", content))

        # Count interfaces (TypeScript)
        if file_path.suffix in [".ts", ".tsx"]:
            metrics.interface_count = len(re.findall(r"interface\s+\w+", content))
            metrics.type_count = len(re.findall(r"type\s+\w+\s*=", content))

        # Calculate maintainability index (simplified for JS)
        # MI = 171 - 5.2 * ln(CC) - 0.23 * CC - 16.2 * ln(LOC)
        import math

        if metrics.code_lines > 0:
            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _parse_js_params(self, params_str: str) -> List[str]:
        """Parse JavaScript function parameters.

        Handles default parameters, rest parameters, and TypeScript types.

        Args:
            params_str: Parameter string from function signature

        Returns:
            List of parameter names
        """
        if not params_str.strip():
            return []

        params = []
        depth = 0
        current_param = ""

        # Handle nested structures in default values
        for char in params_str:
            if char in "({[":
                depth += 1
            elif char in ")}]":
                depth -= 1
            elif char == "," and depth == 0:
                param = current_param.strip()
                if param:
                    # Extract parameter name
                    param_name = param.split("=")[0].split(":")[0].strip()
                    params.append(param_name)
                current_param = ""
                continue

            current_param += char

        # Add last parameter
        if current_param.strip():
            param_name = current_param.strip().split("=")[0].split(":")[0].strip()
            params.append(param_name)

        return params

    def _detect_framework(self, content: str) -> Optional[str]:
        """Detect which framework is being used.

        Args:
            content: JavaScript/TypeScript source code

        Returns:
            Framework name or None
        """
        # React indicators
        react_indicators = [
            r"import\s+.*React",
            r'from\s+[\'"]react[\'"]',
            r"\.jsx",
            r"useState",
            r"useEffect",
            r"componentDidMount",
            r"render\s*\(\s*\)",
        ]

        # Vue indicators
        vue_indicators = [
            r'from\s+[\'"]vue[\'"]',
            r"\.vue",
            r"createApp",
            r"defineComponent",
            r"ref\(",
            r"computed\(",
            r"mounted\s*\(\)",
        ]

        # Angular indicators
        angular_indicators = [
            r'from\s+[\'"]@angular',
            r"@Component",
            r"@Injectable",
            r"@NgModule",
            r"ngOnInit",
            r"constructor\s*\([^)]*private",
        ]

        # Svelte indicators
        svelte_indicators = [
            r"\.svelte",
            r"export\s+let\s+\w+",
            r"\$:",
            r"<script>",
            r'from\s+[\'"]svelte',
        ]

        # Check for frameworks
        for pattern in react_indicators:
            if re.search(pattern, content):
                return "React"

        for pattern in vue_indicators:
            if re.search(pattern, content):
                return "Vue"

        for pattern in angular_indicators:
            if re.search(pattern, content):
                return "Angular"

        for pattern in svelte_indicators:
            if re.search(pattern, content):
                return "Svelte"

        # Check for Node.js
        if re.search(r'require\s*\([\'"](?:fs|path|http|express)[\'"]', content):
            return "Node.js"

        return None

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: JavaScript/TypeScript source code

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
                # Check if comment ends on same line
                if "*/" in line:
                    in_multiline_comment = False
                    # Count the line if there's code after the comment
                    if line.split("*/")[1].strip():
                        count += 1
                continue

            if in_multiline_comment:
                if "*/" in line:
                    in_multiline_comment = False
                    # Count the line if there's code after the comment
                    if line.split("*/")[1].strip():
                        count += 1
                continue

            # Skip empty lines and single-line comments
            if stripped and not stripped.startswith("//"):
                count += 1

        return count

    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines in JavaScript/TypeScript code.

        Args:
            content: JavaScript/TypeScript source code

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

        # Count JSDoc comments
        jsdoc_pattern = r"/\*\*[\s\S]*?\*/"
        jsdoc_matches = re.findall(jsdoc_pattern, content)
        for match in jsdoc_matches:
            # Count lines in JSDoc comment
            count += match.count("\n")

        return count
