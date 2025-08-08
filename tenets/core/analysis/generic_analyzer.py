"""Generic code analyzer for unsupported file types.

This module provides basic analysis capabilities for files that don't have
a specific language analyzer. It performs text-based analysis and pattern
matching to extract basic information.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import LanguageAnalyzer
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
            (r'^\s*#include\s+[<"]([^>"]+)[>"]', "include"),
            (r'^\s*include\s+[\'"]([^\'"]+)[\'"]', "include"),
            # Import patterns (various languages)
            (r'^\s*import\s+[\'"]([^\'"]+)[\'"]', "import"),
            (r'^\s*from\s+[\'"]([^\'"]+)[\'"]', "from"),
            (r'^\s*require\s+[\'"]([^\'"]+)[\'"]', "require"),
            (r'^\s*use\s+[\'"]([^\'"]+)[\'"]', "use"),
            # Load/source patterns (shell scripts)
            (r'^\s*source\s+[\'"]?([^\'"]+)[\'"]?', "source"),
            (r'^\s*\.\s+[\'"]?([^\'"]+)[\'"]?', "source"),
            # Configuration file references
            (r'[\'"]?(?:file|path|src|href|url)[\'"]?\s*[:=]\s*[\'"]([^\'"]+)[\'"]', "reference"),
        ]

        for i, line in enumerate(lines, 1):
            # Skip comments (generic comment patterns)
            if line.strip().startswith("#") or line.strip().startswith("//"):
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

        # Special handling for specific file types
        if file_path.suffix.lower() in [".json", ".yaml", ".yml"]:
            imports.extend(self._extract_config_dependencies(content, file_path))

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

        # Extract functions (various patterns)
        function_patterns = [
            r"^(?:async\s+)?(?:function|def|func|sub|proc)\s+(\w+)",
            r"^(\w+)\s*\(\)\s*\{",
            r"^(\w+)\s*:\s*function",
            r"^(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
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
        if file_type in ["markdown", "documentation"]:
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
        metrics.line_count = len(lines)
        metrics.character_count = len(content)

        # Count non-empty lines
        metrics.code_lines = len([l for l in lines if l.strip()])

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
        for line in lines:
            if any(re.match(pattern, line) for pattern in comment_patterns):
                comment_lines += 1

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
        metrics.max_depth = max(min(max_depth, 10), indent_depth)

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

        return imports

    def _extract_config_keys(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract top-level keys from configuration files.

        Args:
            content: Configuration file content
            file_path: Path to the file

        Returns:
            List of configuration keys
        """
        keys = []

        if file_path.suffix.lower() in [".json"]:
            # Extract JSON top-level keys
            key_pattern = r'^\s*"([^"]+)"\s*:'
            for match in re.finditer(key_pattern, content, re.MULTILINE):
                keys.append(
                    {
                        "name": match.group(1),
                        "type": "config_key",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            # Extract YAML top-level keys
            key_pattern = r"^([a-zA-Z_]\w*):"
            for match in re.finditer(key_pattern, content, re.MULTILINE):
                keys.append(
                    {
                        "name": match.group(1),
                        "type": "config_key",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        elif file_path.suffix.lower() in [".ini", ".cfg", ".conf"]:
            # Extract INI sections and keys
            section_pattern = r"^\[([^\]]+)\]"
            key_pattern = r"^([a-zA-Z_]\w*)\s*="

            for match in re.finditer(section_pattern, content, re.MULTILINE):
                keys.append(
                    {
                        "name": match.group(1),
                        "type": "config_section",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

            for match in re.finditer(key_pattern, content, re.MULTILINE):
                keys.append(
                    {
                        "name": match.group(1),
                        "type": "config_key",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        return keys

    def _analyze_indentation(self, content: str) -> Dict[str, Any]:
        """Analyze indentation patterns in the file.

        Args:
            content: File content

        Returns:
            Dictionary with indentation analysis
        """
        lines = content.split("\n")
        indent_counts = {"spaces": 0, "tabs": 0, "mixed": 0}
        indent_sizes = []

        for line in lines:
            if line and line[0] in [" ", "\t"]:
                # Count leading whitespace
                spaces = len(line) - len(line.lstrip(" "))
                tabs = len(line) - len(line.lstrip("\t"))

                if spaces > 0 and tabs > 0:
                    indent_counts["mixed"] += 1
                elif spaces > 0:
                    indent_counts["spaces"] += 1
                    indent_sizes.append(spaces)
                elif tabs > 0:
                    indent_counts["tabs"] += 1

        # Determine predominant style
        if indent_counts["mixed"] > 0:
            style = "mixed"
        elif indent_counts["tabs"] > indent_counts["spaces"]:
            style = "tabs"
        else:
            style = "spaces"

        # Calculate common indent size
        indent_size = 0
        if indent_sizes:
            from collections import Counter

            # Find most common indent difference
            diffs = []
            sorted_sizes = sorted(set(indent_sizes))
            for i in range(1, len(sorted_sizes)):
                diffs.append(sorted_sizes[i] - sorted_sizes[i - 1])
            if diffs:
                indent_size = Counter(diffs).most_common(1)[0][0]

        return {
            "style": style,
            "size": indent_size,
            "counts": indent_counts,
            "max_level": (
                max(indent_sizes) // indent_size if indent_size > 0 and indent_sizes else 0
            ),
        }

    def _calculate_max_indent(self, lines: List[str]) -> int:
        """Calculate maximum indentation level.

        Args:
            lines: List of file lines

        Returns:
            Maximum indentation level
        """
        max_indent = 0

        for line in lines:
            if line.strip():
                # Count leading spaces (treat tab as 4 spaces)
                indent = 0
                for char in line:
                    if char == " ":
                        indent += 1
                    elif char == "\t":
                        indent += 4
                    else:
                        break

                # Convert to indent level (assume 2 or 4 spaces per level)
                level = indent // 2 if indent < 20 else indent // 4
                max_indent = max(max_indent, level)

        return min(max_indent, 10)  # Cap at 10 for generic files
