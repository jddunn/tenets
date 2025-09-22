"""Shared utilities for summarization.

This module contains common utilities used across different summarizer
implementations to avoid code duplication.
"""

import ast
import re
from typing import Optional


class CodeDetector:
    """Unified code detection logic."""

    # Common code indicators across languages
    CODE_INDICATORS = [
        "def ",
        "class ",
        "function ",
        "import ",
        "from ",
        "const ",
        "var ",
        "let ",
        "return ",
        "if ",
        "else ",
        "for ",
        "while ",
        "switch ",
        "case ",
        "try ",
        "except ",
        "catch ",
        "finally ",
        "async ",
        "await ",
        "export ",
        "{",
        "}",
        "()",
        "=>",
        "[]",
        '"""',
        "'''",
        "//",
        "/*",
        "*/",
    ]

    # Language-specific indicators
    PYTHON_INDICATORS = [
        "def ",
        "class ",
        "import ",
        "from ",
        '"""',
        "'''",
        "@",
        "self.",
        "__init__",
    ]
    JS_INDICATORS = ["function ", "const ", "let ", "var ", "=>", "async ", "await ", "export "]
    JAVA_INDICATORS = ["public ", "private ", "protected ", "class ", "interface ", "extends "]

    @classmethod
    def looks_like_code(cls, text: str, threshold: int = 2) -> bool:
        """Check if text looks like code based on common indicators.

        Args:
            text: Text to check
            threshold: Minimum number of indicators to consider as code

        Returns:
            True if text appears to be code
        """
        if not text:
            return False

        # Check for obvious code patterns
        lines = text.split("\n")[:10]  # Check first 10 lines
        text_sample = "\n".join(lines)

        indicator_count = sum(1 for indicator in cls.CODE_INDICATORS if indicator in text_sample)
        return indicator_count >= threshold

    @classmethod
    def detect_language(cls, text: str, file_path: Optional[str] = None) -> Optional[str]:
        """Detect programming language from text content.

        Args:
            text: Code text
            file_path: Optional file path for extension-based detection

        Returns:
            Detected language or None
        """
        # First try file extension if available
        if file_path:
            ext = file_path.split(".")[-1].lower() if "." in file_path else ""
            ext_map = {
                "py": "python",
                "js": "javascript",
                "ts": "typescript",
                "java": "java",
                "cpp": "cpp",
                "c": "c",
                "cs": "csharp",
                "rb": "ruby",
                "go": "go",
                "rs": "rust",
                "php": "php",
            }
            if ext in ext_map:
                return ext_map[ext]

        # Fall back to content analysis
        text_lower = text.lower()

        # Python detection
        python_score = sum(1 for ind in cls.PYTHON_INDICATORS if ind in text)
        # JavaScript detection
        js_score = sum(1 for ind in cls.JS_INDICATORS if ind in text_lower)
        # Java detection
        java_score = sum(1 for ind in cls.JAVA_INDICATORS if ind in text)

        scores = {"python": python_score, "javascript": js_score, "java": java_score}

        if max(scores.values()) >= 2:
            return max(scores, key=scores.get)

        return None


class ImportParser:
    """Unified import parsing and detection."""

    # Import patterns for different languages
    IMPORT_PATTERNS = {
        "python": [
            re.compile(r"^import\s+(\S+)", re.MULTILINE),
            re.compile(r"^from\s+(\S+)\s+import", re.MULTILINE),
        ],
        "javascript": [
            re.compile(r'^import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE),
            re.compile(r'^const\s+.*\s+=\s+require\([\'"]([^\'"]+)[\'"]\)', re.MULTILINE),
            re.compile(r'^export\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE),
        ],
        "java": [re.compile(r"^import\s+([\w\.]+);", re.MULTILINE)],
        "csharp": [re.compile(r"^using\s+([\w\.]+);", re.MULTILINE)],
        "go": [
            re.compile(r'^import\s+"([^"]+)"', re.MULTILINE),
            re.compile(r"^import\s+\(([^)]+)\)", re.MULTILINE | re.DOTALL),
        ],
    }

    @classmethod
    def is_import_line(cls, line: str, language: str = "python") -> bool:
        """Check if a line is an import statement.

        Args:
            line: Line to check
            language: Programming language

        Returns:
            True if line is an import statement
        """
        line = line.strip()
        if not line:
            return False

        # Language-specific checks
        if language == "python":
            return line.startswith(("import ", "from ")) and not line.startswith("#")
        elif language in ["javascript", "typescript"]:
            return (
                line.startswith("import ")
                or "require(" in line
                or (line.startswith("export ") and "from" in line)
            )
        elif language == "java":
            return line.startswith("import ") and line.endswith(";")
        elif language == "csharp":
            return line.startswith("using ") and line.endswith(";")
        elif language == "go":
            return line.startswith("import ")
        elif language == "rust":
            return line.startswith("use ")
        elif language in ["c", "cpp", "c++"]:
            return line.startswith("#include")

        # Generic check
        return line.startswith(("import ", "from ", "using ", "#include", "use "))

    @classmethod
    def extract_imports(cls, text: str, language: str = "python") -> List[str]:
        """Extract all import statements from code.

        Args:
            text: Source code text
            language: Programming language

        Returns:
            List of import statements
        """
        imports = []

        if language in cls.IMPORT_PATTERNS:
            for pattern in cls.IMPORT_PATTERNS[language]:
                imports.extend(pattern.findall(text))
        else:
            # Fallback to line-by-line checking
            for line in text.split("\n"):
                if cls.is_import_line(line, language):
                    imports.append(line.strip())

        return imports

    @classmethod
    def summarize_imports(cls, imports: List[str], threshold: int = 5) -> str:
        """Summarize a list of imports.

        Args:
            imports: List of import statements
            threshold: Maximum imports before summarizing

        Returns:
            Summarized import text
        """
        if not imports:
            return ""

        if len(imports) <= threshold:
            return "\n".join(imports)

        # Group imports by type/source
        stdlib = []
        external = []
        local = []

        for imp in imports:
            imp_lower = imp.lower()
            # Simple heuristic - can be improved
            if imp.startswith(".") or imp.startswith("from ."):
                local.append(imp)
            elif any(
                std in imp_lower for std in ["os", "sys", "json", "math", "datetime", "collections"]
            ):
                stdlib.append(imp)
            else:
                external.append(imp)

        summary_parts = []
        if stdlib:
            summary_parts.append(f"# {len(stdlib)} stdlib imports")
            if len(stdlib) <= 2:
                summary_parts.extend(stdlib)
        if external:
            summary_parts.append(f"# {len(external)} external imports")
            if len(external) <= 2:
                summary_parts.extend(external)
        if local:
            summary_parts.append(f"# {len(local)} local imports")
            if len(local) <= 2:
                summary_parts.extend(local)

        return "\n".join(summary_parts)


class ASTParser:
    """Unified AST parsing for code structure extraction."""

    @staticmethod
    def extract_python_structure(code: str) -> Dict[str, List[Dict[str, str]]]:
        """Extract functions, classes, and docstrings from Python code.

        Args:
            code: Python source code

        Returns:
            Dictionary with 'functions' and 'classes' lists
        """
        structure = {"functions": [], "classes": [], "docstrings": []}

        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError):
            return structure

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "signature": ASTParser._get_function_signature(node),
                    "docstring": ast.get_docstring(node) or "",
                    "decorators": [d.id for d in node.decorator_list if hasattr(d, "id")],
                }
                structure["functions"].append(func_info)

            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [ASTParser._get_name(base) for base in node.bases],
                    "docstring": ast.get_docstring(node) or "",
                    "methods": [],
                }

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "signature": ASTParser._get_function_signature(item),
                            "docstring": ast.get_docstring(item) or "",
                        }
                        class_info["methods"].append(method_info)

                structure["classes"].append(class_info)

            elif isinstance(node, ast.Module):
                module_docstring = ast.get_docstring(node)
                if module_docstring:
                    structure["docstrings"].append(module_docstring)

        return structure

    @staticmethod
    def _get_function_signature(node: ast.FunctionDef) -> str:
        """Extract function signature from AST node.

        Args:
            node: AST function definition node

        Returns:
            Function signature string
        """
        args = []

        # Positional arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # Keyword-only arguments
        if node.args.kwonlyargs:
            args.append("*")
            for arg in node.args.kwonlyargs:
                args.append(arg.arg)

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # *args
        if node.args.vararg:
            args.insert(len(node.args.args), f"*{node.args.vararg.arg}")

        return f"def {node.name}({', '.join(args)})"

    @staticmethod
    def _get_name(node) -> str:
        """Get name from AST node safely.

        Args:
            node: AST node

        Returns:
            Node name or string representation
        """
        if hasattr(node, "id"):
            return node.id
        elif hasattr(node, "name"):
            return node.name
        elif isinstance(node, ast.Attribute):
            return f"{ASTParser._get_name(node.value)}.{node.attr}"
        else:
            return str(node)

    @staticmethod
    def extract_leading_comments(text: str, max_lines: int = 10) -> str:
        """Extract leading comments from code.

        Args:
            text: Source code text
            max_lines: Maximum lines to check

        Returns:
            Extracted comments
        """
        comments = []
        lines = text.split("\n")[:max_lines]

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                comments.append(stripped)
            elif stripped.startswith(('"""', "'''")):
                # Start of docstring
                docstring_lines = []
                in_docstring = True
                quote = stripped[:3]

                for rest_line in lines[lines.index(line) :]:
                    docstring_lines.append(rest_line)
                    if rest_line.strip().endswith(quote) and len(docstring_lines) > 1:
                        break

                comments.extend(docstring_lines)
                break
            elif stripped and not stripped.startswith(("import ", "from ")):
                # Hit actual code
                break

        return "\n".join(comments)


class TextTruncator:
    """Utilities for truncating text while preserving structure."""

    @staticmethod
    def smart_truncate(text: str, max_length: int, preserve_structure: bool = True) -> str:
        """Truncate text intelligently while trying to preserve structure.

        Args:
            text: Text to truncate
            max_length: Maximum character length
            preserve_structure: Whether to preserve code structure

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        if not preserve_structure:
            return text[:max_length] + "..."

        # Try to truncate at natural boundaries
        lines = text.split("\n")
        result = []
        current_length = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > max_length:
                # Check if we can at least add part of the line
                remaining = max_length - current_length
                if remaining > 20:  # Arbitrary min line length
                    result.append(line[: remaining - 3] + "...")
                else:
                    result.append("...")
                break
            result.append(line)
            current_length += line_length

        return "\n".join(result)

    @staticmethod
    def truncate_middle(text: str, max_length: int, context_ratio: float = 0.3) -> str:
        """Truncate middle of text while preserving beginning and end.

        Args:
            text: Text to truncate
            max_length: Maximum character length
            context_ratio: Ratio of text to keep at beginning/end

        Returns:
            Text with middle truncated
        """
        if len(text) <= max_length:
            return text

        # Calculate how much to keep from beginning and end
        keep_start = int(max_length * context_ratio)
        keep_end = int(max_length * context_ratio)

        # Ensure we have room for the ellipsis
        ellipsis = "\n...[truncated]...\n"
        keep_start = min(keep_start, (max_length - len(ellipsis)) // 2)
        keep_end = min(keep_end, (max_length - len(ellipsis)) // 2)

        return text[:keep_start] + ellipsis + text[-keep_end:]
