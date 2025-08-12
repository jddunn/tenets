"""Generic code analyzer for unsupported file types.

This module provides basic analysis capabilities for files that don't have
a specific language analyzer. It performs text-based analysis and pattern
matching to extract basic information.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import LanguageAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)
from tenets.utils.logger import get_logger


class GenericAnalyzer(LanguageAnalyzer):
    """Generic analyzer for unsupported file types.

    Provides basic analysis for text-based files including:
    - Line and character counting
    - Basic pattern matching for imports/includes
    - Simple complexity estimation
    - Keyword extraction
    - Configuration file parsing (JSON, YAML, XML, etc.)

    This analyzer serves as a fallback for files without specific
    language support and can handle various text formats.
    """

    language_name = "generic"
    file_extensions = []  # Accepts any extension

    def __init__(self):
        """Initialize the generic analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract potential imports/includes from generic text.

        Looks for common import patterns across various languages
        and configuration files.

        Args:
            content: File content
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with detected imports
        """
        imports = []
        lines = content.split("\n")

        # Common import/include patterns
        patterns = [
            # Include patterns (C-style, various scripting languages)
            (r"^\s*#include\s+<([^>]+)>", "include"),  # angle includes
            (r'^\s*#include\s+"([^"]+)"', "include"),  # quote includes
            (r"^\s*include\s+[\'\"]([^\'\"]+)[\'\"]", "include"),
            # CMake include()
            (r"^\s*include\s*\(\s*([^)\s]+)\s*\)", "include"),
            # Import patterns (various languages)
            (r'^\s*import\s+[\'"]([^\'"]+)[\'"]', "import"),  # import "module"
            (r"^\s*import\s+([A-Za-z_][\w\.]*)\b", "import"),  # import os
            (r'^\s*from\s+[\'"]([^\'"]+)[\'"]', "from"),  # from "mod"
            (r"^\s*from\s+([A-Za-z_][\w\.]*)\s+import\b", "from"),  # from pkg import X
            (r'^\s*require\s+[\'"]([^\'"]+)[\'"]', "require"),
            # PHP/Perl and JS style use statements
            (r"^\s*use\s+([\\\w:]+);?", "use"),  # use Data::Dumper; or use Foo\Bar;
            # Load/source patterns (shell scripts)
            (r'^\s*source\s+[\'"]?([^\'"]+)[\'"]?', "source"),
            (r'^\s*\.[ \t]+[\'"]?([^\'"]+)[\'"]?', "source"),
            # Configuration file references
            (r'[\'"]?(?:file|path|src|href|url)[\'"]?\s*[:=]\s*[\'"]([^\'"]+)[\'"]', "reference"),
        ]

        captured_modules: set[str] = set()

        for i, line in enumerate(lines, 1):
            # Skip comments (generic comment patterns) but keep C preprocessor includes
            if (
                line.strip().startswith("#") and not re.match(r"^\s*#include\b", line)
            ) or line.strip().startswith("//"):
                continue

            for pattern, import_type in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    module = match.group(1)
                    imports.append(
                        ImportInfo(
                            module=module,
                            line=i,
                            type=import_type,
                            is_relative=self._is_relative_path(module),
                        )
                    )
                    captured_modules.add(module)
                    break

            # Special case: 'use strict;' (JavaScript directive)
            if re.match(r"^\s*use\s+strict\s*;?\s*$", line):
                imports.append(ImportInfo(module="strict", line=i, type="use", is_relative=False))
                captured_modules.add("strict")

        # Special handling for specific file types
        if file_path.suffix.lower() in [".json", ".yaml", ".yml"]:
            imports.extend(self._extract_config_dependencies(content, file_path))

        # Detect standalone file references like config.yml in logs
        file_ref_pattern = re.compile(
            r"\b([\w./-]+\.(?:ya?ml|json|conf|cfg|ini|xml|toml|log|txt|sh))\b"
        )
        for i, line in enumerate(lines, 1):
            for m in file_ref_pattern.finditer(line):
                module = m.group(1)
                if module not in captured_modules:
                    imports.append(
                        ImportInfo(
                            module=module,
                            line=i,
                            type="reference",
                            is_relative=self._is_relative_path(module),
                        )
                    )
                    captured_modules.add(module)

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract potential exports from generic text.

        Looks for common export patterns and definitions.

        Args:
            content: File content
            file_path: Path to the file being analyzed

        Returns:
            List of potential exported symbols
        """
        exports = []

        # Common export/definition patterns
        patterns = [
            # Function-like definitions
            (r"^(?:function|def|func|sub|proc)\s+(\w+)", "function"),
            (r"^(\w+)\s*\(\)\s*\{", "function"),
            # Class-like definitions
            (r"^(?:class|struct|type|interface)\s+(\w+)", "class"),
            # Variable/constant definitions
            (r"^(?:export\s+)?(?:const|let|var|val)\s+(\w+)\s*=", "variable"),
            (r'^(\w+)\s*=\s*[\'"]?[^\'"\n]+[\'"]?', "assignment"),
            # Export statements
            (r"^export\s+(\w+)", "export"),
            (r"^module\.exports\.(\w+)", "export"),
        ]

        for pattern, export_type in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                name = match.group(1)
                exports.append(
                    {
                        "name": name,
                        "type": export_type,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        # For configuration files, extract top-level keys
        if file_path.suffix.lower() in [".json", ".yaml", ".yml", ".toml", ".ini"]:
            exports.extend(self._extract_config_keys(content, file_path))

        return exports

    def _extract_config_keys(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract top-level keys from common config formats without parsing libraries."""
        keys = []
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".json":
                # naive top-level key extractor
                for m in re.finditer(r"^\s*\"([A-Za-z0-9_\-\.]+)\"\s*:\s*", content, re.MULTILINE):
                    keys.append(
                        {
                            "name": m.group(1),
                            "type": "config_key",
                            "line": content[: m.start()].count("\n") + 1,
                        }
                    )
            elif suffix in [".yaml", ".yml"]:
                # YAML top-level keys: key: value at column 0
                for m in re.finditer(r"^(\w[\w\-\./]*)\s*:\s*", content, re.MULTILINE):
                    if m.start() == content.rfind("\n", 0, m.start()) + 1:  # ensure start of line
                        keys.append(
                            {
                                "name": m.group(1),
                                "type": "config_key",
                                "line": content[: m.start()].count("\n") + 1,
                            }
                        )
            elif suffix == ".toml":
                # TOML keys: key = value at top-level (ignore dotted tables)
                for m in re.finditer(r"^\s*([A-Za-z0-9_\-]+)\s*=\s*", content, re.MULTILINE):
                    keys.append(
                        {
                            "name": m.group(1),
                            "type": "config_key",
                            "line": content[: m.start()].count("\n") + 1,
                        }
                    )
            elif suffix == ".ini":
                # INI: capture both [sections] and keys inside sections
                in_section = False
                for i, line in enumerate(content.splitlines(), 1):
                    if re.match(r"^\s*\[.+\]", line):
                        in_section = True
                        keys.append(
                            {
                                "name": re.sub(r"^\s*\[|\]\s*$", "", line).strip(),
                                "type": "config_section",
                                "line": i,
                            }
                        )
                        continue
                    # Capture key=value lines regardless of being in a section
                    m = re.match(r"^\s*([A-Za-z0-9_\-\.]+)\s*=\s*", line)
                    if m:
                        keys.append({"name": m.group(1), "type": "config_key", "line": i})
        except Exception:
            pass
        return keys

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract basic structure from generic text.

        Attempts to identify structural elements using pattern matching
        and indentation analysis.

        Args:
            content: File content
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with detected elements
        """
        structure = CodeStructure()

        # Detect file type category
        file_type = self._detect_file_type(file_path)
        structure.file_type = file_type

        # Detect common YAML-based frameworks/configs
        try:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                # Initialize modules collection if not present
                if not hasattr(structure, "modules"):
                    structure.modules = []

                if self._is_docker_compose_file(file_path, content):
                    structure.framework = "docker-compose"
                    for svc in self._extract_compose_services(content):
                        structure.modules.append({"type": "service", **svc})
                elif self._looks_like_kubernetes_yaml(content):
                    structure.framework = "kubernetes"
                    for res in self._extract_k8s_resources(content):
                        structure.modules.append({"type": "resource", **res})
                else:
                    # Helm/Kustomize/GitHub Actions quick hints
                    name = file_path.name.lower()
                    if name == "chart.yaml":
                        structure.framework = "helm"
                    elif name == "values.yaml":
                        structure.framework = getattr(structure, "framework", None) or "helm"
                    elif name == "kustomization.yaml":
                        structure.framework = "kustomize"
                    elif ".github" in str(file_path).replace("\\", "/") and "/workflows/" in str(
                        file_path
                    ).replace("\\", "/"):
                        structure.framework = "github-actions"
        except Exception:
            # Never fail generic structure on heuristics
            pass

        # Extract functions (various patterns)
        function_patterns = [
            r"^(?:async\s+)?(?:function|def|func|sub|proc)\s+(\w+)",
            r"^(\w+)\s*\(\)\s*\{",
            r"^(\w+)\s*:\s*function",
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
        ]

        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                func_name = match.group(1)
                structure.functions.append(
                    FunctionInfo(name=func_name, line=content[: match.start()].count("\n") + 1)
                )

        # Extract classes/types
        class_patterns = [
            r"^(?:export\s+)?(?:class|struct|type|interface|enum)\s+(\w+)",
            r"^(\w+)\s*=\s*class\s*\{",
        ]

        for pattern in class_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                class_name = match.group(1)
                structure.classes.append(
                    ClassInfo(name=class_name, line=content[: match.start()].count("\n") + 1)
                )

        # Extract sections (markdown headers, etc.)
        if file_type in ["markdown", "documentation", "markup"]:
            section_pattern = r"^(#{1,6})\s+(.+)$"
            for match in re.finditer(section_pattern, content, re.MULTILINE):
                level = len(match.group(1))
                title = match.group(2)
                structure.sections.append(
                    {
                        "title": title,
                        "level": level,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        # Extract variables/constants
        var_patterns = [
            r"^(?:const|let|var|val)\s+(\w+)",
            r"^(\w+)\s*[:=]\s*[^=]",
            r"^export\s+(\w+)",
        ]

        for pattern in var_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                var_name = match.group(1)
                structure.variables.append(
                    {
                        "name": var_name,
                        "line": content[: match.start()].count("\n") + 1,
                        "type": "variable",
                    }
                )

        # Detect constants (UPPERCASE variables)
        const_pattern = r"^([A-Z][A-Z0-9_]*)\s*[:=]"
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            structure.constants.append(match.group(1))

        # Extract TODO/FIXME comments
        todo_pattern = r"(?:#|//|/\*|\*)\s*(TODO|FIXME|HACK|NOTE|XXX|BUG):\s*(.+)"
        for match in re.finditer(todo_pattern, content, re.IGNORECASE):
            structure.todos.append(
                {
                    "type": match.group(1).upper(),
                    "message": match.group(2).strip(),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Count blocks (based on indentation or braces)
        structure.block_count = content.count("{")
        structure.indent_levels = self._analyze_indentation(content)

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate basic complexity metrics for generic text.

        Provides simplified complexity estimation based on:
        - Line count and length
        - Nesting depth (indentation/braces)
        - Decision keywords
        - File type specific metrics

        Args:
            content: File content
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with basic metrics
        """
        metrics = ComplexityMetrics()

        # Basic line metrics
        lines = content.split("\n")
        # Trim leading/trailing empty lines for line count to match human expectations/tests
        start = 0
        end = len(lines)
        while start < end and lines[start].strip() == "":
            start += 1
        while end > start and lines[end - 1].strip() == "":
            end -= 1
        trimmed_lines = lines[start:end]

        # Preserve historical/test expectation: an entirely empty file counts as 1 line (logical line),
        # while code_lines will be 0. Non-empty (after trimming) counts actual trimmed lines.
        if not trimmed_lines:
            metrics.line_count = 1
        else:
            metrics.line_count = len(trimmed_lines)
        # Character count: count characters, and if file doesn't end with newline, count implicit final EOL
        metrics.character_count = len(content) + (0 if content.endswith("\n") else 1)

        # Count comment lines (generic patterns)
        comment_patterns = [
            r"^\s*#",  # Hash comments
            r"^\s*//",  # Double slash comments
            r"^\s*/\*",  # Block comment start
            r"^\s*\*",  # Block comment continuation
            r"^\s*<!--",  # HTML/XML comments
            r"^\s*;",  # Semicolon comments (INI, assembly)
            r"^\s*--",  # SQL/Lua comments
            r"^\s*%",  # LaTeX/MATLAB comments
        ]

        comment_lines = 0
        for line in trimmed_lines:
            if any(re.match(pattern, line) for pattern in comment_patterns):
                comment_lines += 1

        # Compute code lines as total lines minus comment lines (consistent with tests)
        # For empty file (line_count==1 but no trimmed lines), code_lines should be 0
        if not trimmed_lines:
            metrics.code_lines = 0
        else:
            metrics.code_lines = metrics.line_count - comment_lines

        metrics.comment_lines = comment_lines
        metrics.comment_ratio = comment_lines / metrics.line_count if metrics.line_count > 0 else 0

        # Estimate cyclomatic complexity (decision points)
        decision_keywords = [
            r"\bif\b",
            r"\belse\b",
            r"\belif\b",
            r"\belsif\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\bdo\b",
            r"\bcase\b",
            r"\bwhen\b",
            r"\btry\b",
            r"\bcatch\b",
            r"\bexcept\b",
            r"\bunless\b",
            r"\buntil\b",
            r"\bswitch\b",
            r"\b\?\s*[^:]+\s*:",
            r"\|\|",
            r"&&",
            r"\band\b",
            r"\bor\b",
        ]

        complexity = 1  # Base complexity
        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content, re.IGNORECASE))

        metrics.cyclomatic = min(complexity, 50)  # Cap at 50 for generic files

        # Estimate nesting depth
        max_depth = 0
        current_depth = 0

        for line in lines:
            # Track braces
            current_depth += line.count("{") - line.count("}")
            current_depth += line.count("(") - line.count(")")
            current_depth += line.count("[") - line.count("]")
            max_depth = max(max_depth, current_depth)

            # Reset if negative (mismatched brackets)
            if current_depth < 0:
                current_depth = 0

        # Also check indentation depth
        indent_depth = self._calculate_max_indent(lines)
        # Combine and cap at 10
        metrics.max_depth = min(max(max_depth, indent_depth), 10)

        # File type specific metrics
        file_type = self._detect_file_type(file_path)

        if file_type == "configuration":
            # For config files, count keys/sections
            metrics.key_count = len(re.findall(r"^\s*[\w\-\.]+\s*[:=]", content, re.MULTILINE))
            metrics.section_count = len(re.findall(r"^\s*\[[\w\-\.]+\]", content, re.MULTILINE))

        elif file_type == "markup":
            # For markup files, count tags
            metrics.tag_count = len(re.findall(r"<\w+", content))
            metrics.header_count = len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE))

        elif file_type == "data":
            # For data files, estimate structure
            if file_path.suffix.lower() == ".csv":
                lines_sample = lines[:10] if len(lines) > 10 else lines
                if lines_sample:
                    # Estimate columns
                    metrics.column_count = len(lines_sample[0].split(","))
                    metrics.row_count = len(lines) - 1  # Exclude header

        # Calculate a simple maintainability index
        if metrics.code_lines > 0:
            # Simplified calculation
            maintainability = 100

            # Penalize high complexity
            maintainability -= min(30, complexity * 0.5)

            # Penalize deep nesting
            maintainability -= min(20, metrics.max_depth * 2)

            # Reward comments
            maintainability += min(10, metrics.comment_ratio * 30)

            # Penalize very long files
            if metrics.line_count > 1000:
                maintainability -= 10
            elif metrics.line_count > 500:
                maintainability -= 5

            metrics.maintainability_index = max(0, min(100, maintainability))

        return metrics

    def _is_relative_path(self, path: str) -> bool:
        """Check if a path is relative.

        Args:
            path: Path string

        Returns:
            True if the path is relative
        """
        absolute_indicators = ["/", "\\", "http://", "https://", "ftp://", "file://"]
        return not any(path.startswith(ind) for ind in absolute_indicators)

    def _detect_file_type(self, file_path: Path) -> str:
        """Detect the general type of file.

        Args:
            file_path: Path to the file

        Returns:
            File type category string
        """
        extension = file_path.suffix.lower()
        name = file_path.name.lower()

        # Configuration files
        config_extensions = [
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
            ".properties",
            ".env",
        ]
        config_names = [
            "config",
            "settings",
            "preferences",
            ".env",
            "dockerfile",
            "makefile",
            "rakefile",
            "gulpfile",
            "gruntfile",
        ]

        if extension in config_extensions or any(n in name for n in config_names):
            return "configuration"

        # Markup/Documentation files
        markup_extensions = [
            ".md",
            ".markdown",
            ".rst",
            ".tex",
            ".html",
            ".xml",
            ".sgml",
            ".xhtml",
            ".svg",
        ]
        if extension in markup_extensions:
            return "markup"

        # Data files
        data_extensions = [".csv", ".tsv", ".dat", ".data", ".txt"]
        if extension in data_extensions:
            return "data"

        # Script files
        script_extensions = [".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd"]
        if extension in script_extensions:
            return "script"

        # Style files
        style_extensions = [".css", ".scss", ".sass", ".less", ".styl"]
        if extension in style_extensions:
            return "stylesheet"

        # Query files
        if extension in [".sql", ".graphql", ".gql"]:
            return "query"

        return "text"

    def _extract_config_dependencies(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract dependencies from configuration files.

        Args:
            content: Configuration file content
            file_path: Path to the file

        Returns:
            List of ImportInfo objects for dependencies
        """
        imports = []

        if file_path.suffix.lower() == ".json":
            # Look for dependency-like keys in JSON
            dep_patterns = [
                r'"(?:dependencies|devDependencies|peerDependencies|requires?)"\s*:\s*\{([^}]+)\}',
                r'"(?:import|include|require|extends?)"\s*:\s*"([^"]+)"',
            ]

            for pattern in dep_patterns:
                for match in re.finditer(pattern, content):
                    if "{" in match.group(1):
                        # Parse dependency object
                        deps = re.findall(r'"([^"]+)"\s*:\s*"[^"]+"', match.group(1))
                        for dep in deps:
                            imports.append(
                                ImportInfo(module=dep, type="dependency", is_relative=False)
                            )
                    else:
                        # Single dependency
                        imports.append(
                            ImportInfo(
                                module=match.group(1),
                                type="config_import",
                                is_relative=self._is_relative_path(match.group(1)),
                            )
                        )

        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            name_lower = file_path.name.lower()
            is_compose = self._is_docker_compose_file(file_path, content)
            looks_k8s = self._looks_like_kubernetes_yaml(content)

            # Common YAML references
            # Images (compose and k8s)
            for m in re.finditer(r"(?mi)^\s*image:\s*[\"\']?([^\s\"\']+)", content):
                imports.append(ImportInfo(module=m.group(1), type="image", is_relative=False))

            if is_compose:
                # depends_on services
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if re.match(r"^\s*depends_on\s*:\s*$", line):
                        j = i + 1
                        while j < len(lines) and re.match(r"^\s*-\s*", lines[j]):
                            svc = re.sub(r"^\s*-\s*", "", lines[j]).strip()
                            if svc:
                                imports.append(
                                    ImportInfo(module=svc, type="depends_on", is_relative=False)
                                )
                            j += 1
                # External compose file references via extends/include (compose v2)
                for m in re.finditer(r"(?mi)^\s*extends:\s*[\"\']?([^\s\"\']+)", content):
                    imports.append(
                        ImportInfo(
                            module=m.group(1),
                            type="extends",
                            is_relative=self._is_relative_path(m.group(1)),
                        )
                    )

            if looks_k8s:
                # ConfigMap and Secret references
                for m in re.finditer(
                    r"(?mis)(configMapRef|configMapKeyRef):\s*.*?\bname:\s*([\w.-]+)", content
                ):
                    imports.append(
                        ImportInfo(module=m.group(2), type="configmap", is_relative=False)
                    )
                for m in re.finditer(
                    r"(?mis)(secretRef|secretKeyRef):\s*.*?\bname:\s*([\w.-]+)", content
                ):
                    imports.append(ImportInfo(module=m.group(2), type="secret", is_relative=False))
                # Ingress hosts
                for m in re.finditer(r"(?mi)^\s*host:\s*([^\s#]+)", content):
                    imports.append(
                        ImportInfo(module=m.group(1), type="ingress_host", is_relative=False)
                    )
                # ServiceAccounts
                for m in re.finditer(r"(?mi)^\s*serviceAccountName:\s*([\w.-]+)", content):
                    imports.append(
                        ImportInfo(module=m.group(1), type="serviceaccount", is_relative=False)
                    )

        return imports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract basic structure from generic text.

        Attempts to identify structural elements using pattern matching
        and indentation analysis.

        Args:
            content: File content
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with detected elements
        """
        structure = CodeStructure()

        # Detect file type category
        file_type = self._detect_file_type(file_path)
        structure.file_type = file_type

        # Detect common YAML-based frameworks/configs
        try:
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                # Initialize modules collection if not present
                if not hasattr(structure, "modules"):
                    structure.modules = []

                if self._is_docker_compose_file(file_path, content):
                    structure.framework = "docker-compose"
                    for svc in self._extract_compose_services(content):
                        structure.modules.append({"type": "service", **svc})
                elif self._looks_like_kubernetes_yaml(content):
                    structure.framework = "kubernetes"
                    for res in self._extract_k8s_resources(content):
                        structure.modules.append({"type": "resource", **res})
                else:
                    # Helm/Kustomize/GitHub Actions quick hints
                    name = file_path.name.lower()
                    if name == "chart.yaml":
                        structure.framework = "helm"
                    elif name == "values.yaml":
                        structure.framework = getattr(structure, "framework", None) or "helm"
                    elif name == "kustomization.yaml":
                        structure.framework = "kustomize"
                    elif ".github" in str(file_path).replace("\\", "/") and "/workflows/" in str(
                        file_path
                    ).replace("\\", "/"):
                        structure.framework = "github-actions"
        except Exception:
            # Never fail generic structure on heuristics
            pass

        # Extract functions (various patterns)
        function_patterns = [
            r"^(?:async\s+)?(?:function|def|func|sub|proc)\s+(\w+)",
            r"^(\w+)\s*\(\)\s*\{",
            r"^(\w+)\s*:\s*function",
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
        ]

        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                func_name = match.group(1)
                structure.functions.append(
                    FunctionInfo(name=func_name, line=content[: match.start()].count("\n") + 1)
                )

        # Extract classes/types
        class_patterns = [
            r"^(?:export\s+)?(?:class|struct|type|interface|enum)\s+(\w+)",
            r"^(\w+)\s*=\s*class\s*\{",
        ]

        for pattern in class_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                class_name = match.group(1)
                structure.classes.append(
                    ClassInfo(name=class_name, line=content[: match.start()].count("\n") + 1)
                )

        # Extract sections (markdown headers, etc.)
        if file_type in ["markdown", "documentation", "markup"]:
            section_pattern = r"^(#{1,6})\s+(.+)$"
            for match in re.finditer(section_pattern, content, re.MULTILINE):
                level = len(match.group(1))
                title = match.group(2)
                structure.sections.append(
                    {
                        "title": title,
                        "level": level,
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        # Extract variables/constants
        var_patterns = [
            r"^(?:const|let|var|val)\s+(\w+)",
            r"^(\w+)\s*[:=]\s*[^=]",
            r"^export\s+(\w+)",
        ]

        for pattern in var_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                var_name = match.group(1)
                structure.variables.append(
                    {
                        "name": var_name,
                        "line": content[: match.start()].count("\n") + 1,
                        "type": "variable",
                    }
                )

        # Detect constants (UPPERCASE variables)
        const_pattern = r"^([A-Z][A-Z0-9_]*)\s*[:=]"
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            structure.constants.append(match.group(1))

        # Extract TODO/FIXME comments
        todo_pattern = r"(?:#|//|/\*|\*)\s*(TODO|FIXME|HACK|NOTE|XXX|BUG):\s*(.+)"
        for match in re.finditer(todo_pattern, content, re.IGNORECASE):
            structure.todos.append(
                {
                    "type": match.group(1).upper(),
                    "message": match.group(2).strip(),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Count blocks (based on indentation or braces)
        structure.block_count = content.count("{")
        structure.indent_levels = self._analyze_indentation(content)

        return structure

    def _is_docker_compose_file(self, file_path: Path, content: str) -> bool:
        name = file_path.name.lower()
        if name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
            return True
        return bool(re.search(r"(?mi)^\s*services\s*:\s*$", content))

    def _looks_like_kubernetes_yaml(self, content: str) -> bool:
        # Heuristic: presence of apiVersion and kind keys
        return bool(
            re.search(r"(?mi)^\s*apiVersion\s*:\s*\S+", content)
            and re.search(r"(?mi)^\s*kind\s*:\s*\S+", content)
        )

    def _extract_compose_services(self, content: str) -> List[Dict[str, Any]]:
        """Best-effort extraction of docker-compose services with images."""
        services: List[Dict[str, Any]] = []
        lines = content.splitlines()
        # Find the services: block
        try:
            svc_start = next(i for i, l in enumerate(lines) if re.match(r"^\s*services\s*:\s*$", l))
        except StopIteration:
            return services

        # Scan following indented blocks for service names at first indent level under services
        i = svc_start + 1
        while i < len(lines):
            line = lines[i]
            # Service key like "  web:" or with more spaces
            m = re.match(r"^(\s{2,})([A-Za-z0-9._-]+)\s*:\s*$", line)
            if m:
                base_indent = len(m.group(1))
                name = m.group(2)
                info: Dict[str, Any] = {"name": name}
                j = i + 1
                while j < len(lines):
                    ln = lines[j]
                    # Stop when indent less than or equal to base and a key starts
                    if re.match(r"^\s*$", ln):
                        j += 1
                        continue
                    cur_indent = len(ln) - len(ln.lstrip(" "))
                    if cur_indent <= base_indent:
                        break
                    # Capture common fields
                    img_m = re.match(r'^\s*image\s*:\s*"?([^"\s]+)"?', ln)
                    if img_m:
                        info["image"] = img_m.group(1)
                    port_m = re.match(r"^\s*ports\s*:\s*$", ln)
                    if port_m:
                        # count following list items
                        k = j + 1
                        ports = 0
                        while k < len(lines) and re.match(r"^\s*-\s*", lines[k]):
                            ports += 1
                            k += 1
                        if ports:
                            info["ports"] = ports
                    j += 1
                services.append(info)
                i = j
                continue
            # Break if we hit another top-level key
            if re.match(r"^\s*\w[^:]*\s*:\s*$", line) and not line.startswith("  "):
                break
            i += 1

        return services

    def _extract_k8s_resources(self, content: str) -> List[Dict[str, Any]]:
        """Extract Kubernetes resources (kind, name, images) from YAML (supports multi-doc)."""
        resources: List[Dict[str, Any]] = []
        docs = re.split(r"(?m)^---\s*$", content)
        for doc in docs:
            kind_m = re.search(r"(?mi)^\s*kind\s*:\s*([\w.-]+)", doc)
            if not kind_m:
                continue
            res: Dict[str, Any] = {"kind": kind_m.group(1)}
            name_m = re.search(r"(?mis)metadata\s*:\s*.*?\bname\s*:\s*([\w.-]+)", doc)
            if name_m:
                res["name"] = name_m.group(1)
            # collect images
            imgs = re.findall(r'(?mi)^\s*image\s*:\s*"?([^"\s]+)"?', doc)
            if imgs:
                res["images"] = imgs
            resources.append(res)
        return resources

    def _analyze_indentation(self, content: str) -> Dict[str, Any]:
        """Analyze indentation patterns in the file.

        Args:
            content: File content

        Returns:
            Dictionary with indentation analysis
        """
        lines = content.splitlines()
        indent_sizes: Dict[int, int] = {}
        tabs = 0
        spaces = 0
        max_indent = 0
        for ln in lines:
            if not ln.strip():
                continue
            leading = len(ln) - len(ln.lstrip(" \t"))
            if leading > max_indent:
                max_indent = leading
            if ln.startswith("\t"):
                tabs += 1
            elif ln.startswith(" "):
                spaces += 1
                count = len(ln) - len(ln.lstrip(" "))
                if count:
                    indent_sizes[count] = indent_sizes.get(count, 0) + 1
        style = "tabs" if tabs > spaces else "spaces"
        return {
            "style": style,
            "indent_char": "tab" if style == "tabs" else "space",
            "indent_sizes": indent_sizes,
            "max_level": self._calculate_max_indent(lines),
            "max_indent": max_indent,
        }

    def _calculate_max_indent(self, lines: List[str]) -> int:
        """Estimate maximum logical indentation level based on spaces/tabs."""
        # Determine common indent size (2 or 4), fallback 4
        sizes: Dict[int, int] = {}
        for ln in lines:
            if ln.startswith(" "):
                count = len(ln) - len(ln.lstrip(" "))
                if count:
                    sizes[count] = sizes.get(count, 0) + 1
        # Pick the most common divisor of 2 or 4
        indent_unit = 4
        if sizes:
            freq_pairs = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
            for size, _ in freq_pairs:
                if size % 2 == 0:
                    indent_unit = 2 if size % 2 == 0 and size % 4 != 0 else 4
                    break
        level = 0
        max_level = 0
        for ln in lines:
            if not ln.strip():
                continue
            if ln.startswith("\t"):
                # Treat tab as one level
                level = ln.count("\t")
            else:
                spaces = len(ln) - len(ln.lstrip(" "))
                level = spaces // indent_unit if indent_unit else 0
            if level > max_level:
                max_level = level
        return max_level
