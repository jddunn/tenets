"""Ruby code analyzer.

This module provides comprehensive analysis for Ruby source files,
including support for Ruby's dynamic features, metaprogramming, and DSLs.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from .base import LanguageAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)
from tenets.utils.logger import get_logger


class RubyAnalyzer(LanguageAnalyzer):
    """Ruby code analyzer.

    Provides analysis for Ruby files including:
    - Require and gem dependency analysis
    - Class and module extraction with inheritance
    - Method analysis with visibility and metaprogramming
    - Block, proc, and lambda detection
    - DSL pattern recognition
    - Attribute accessors and metaprogramming
    - Ruby-specific patterns (symbols, instance variables)
    - Framework detection (Rails, Sinatra, RSpec)

    Handles Ruby's dynamic nature and metaprogramming features.
    """

    language_name = "ruby"
    file_extensions = [".rb", ".rake", ".gemspec", ".ru"]

    def __init__(self):
        """Initialize the Ruby analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract requires and gems from Ruby code.

        Handles:
        - require 'library'
        - require_relative 'file'
        - load 'file.rb'
        - gem 'gemname'
        - autoload :Module, 'file'
        - Bundler.require

        Args:
            content: Ruby source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        # Require patterns
        require_pattern = re.compile(r'^\s*require\s+[\'"]([^\'"]+)[\'"]')
        require_relative_pattern = re.compile(r'^\s*require_relative\s+[\'"]([^\'"]+)[\'"]')
        load_pattern = re.compile(r'^\s*load\s+[\'"]([^\'"]+)[\'"]')
        gem_pattern = re.compile(r'^\s*gem\s+[\'"]([^\'"]+)[\'"](?:,\s*[\'"]([^\'"]+)[\'"])?')
        autoload_pattern = re.compile(r'^\s*autoload\s+:(\w+),\s*[\'"]([^\'"]+)[\'"]')

        # Conditional require patterns
        require_if_pattern = re.compile(r'require\s+[\'"]([^\'"]+)[\'"]\s+if\s+')

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Standard require
            match = require_pattern.match(line)
            if match:
                module_name = match.group(1)
                imports.append(
                    ImportInfo(
                        module=module_name,
                        line=i,
                        type="require",
                        is_relative=False,
                        is_stdlib=self._is_stdlib_module(module_name),
                        is_gem=not self._is_stdlib_module(module_name)
                        and not module_name.startswith("."),
                    )
                )
                continue

            # Require relative
            match = require_relative_pattern.match(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="require_relative",
                        is_relative=True,
                        is_project_file=True,
                    )
                )
                continue

            # Load
            match = load_pattern.match(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="load",
                        is_relative=match.group(1).startswith("."),
                        reloads=True,  # load can reload files
                    )
                )
                continue

            # Gem
            match = gem_pattern.match(line)
            if match:
                gem_name = match.group(1)
                version = match.group(2) if match.group(2) else None
                imports.append(
                    ImportInfo(
                        module=gem_name,
                        line=i,
                        type="gem",
                        is_relative=False,
                        version=version,
                        is_gem=True,
                    )
                )
                continue

            # Autoload
            match = autoload_pattern.match(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(2),
                        alias=match.group(1),
                        line=i,
                        type="autoload",
                        is_relative=match.group(2).startswith("."),
                        lazy_load=True,
                    )
                )
                continue

            # Conditional requires
            match = require_if_pattern.search(line)
            if match:
                imports.append(
                    ImportInfo(
                        module=match.group(1),
                        line=i,
                        type="conditional_require",
                        is_relative=False,
                        conditional=True,
                    )
                )

            # Bundler.require
            if "Bundler.require" in line:
                imports.append(
                    ImportInfo(
                        module="Bundler",
                        line=i,
                        type="bundler_require",
                        is_relative=False,
                        loads_all_gems=True,
                    )
                )

        # Check for Gemfile dependencies if it's a Gemfile
        if file_path.name == "Gemfile":
            imports.extend(self._extract_gemfile_dependencies(content))

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract public methods and classes from Ruby code.

        In Ruby, everything is public by default unless specified otherwise.
        Module and class definitions are the primary exports.

        Args:
            content: Ruby source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols with metadata
        """
        exports = []

        # Classes
        class_pattern = r"^\s*class\s+(\w+)(?:\s*<\s*([\w:]+))?"
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            superclass = match.group(2) if match.group(2) else "Object"

            exports.append(
                {
                    "name": class_name,
                    "type": "class",
                    "line": content[: match.start()].count("\n") + 1,
                    "superclass": superclass,
                    "is_exception": "Error" in superclass or "Exception" in superclass,
                }
            )

        # Modules
        module_pattern = r"^\s*module\s+(\w+)"
        for match in re.finditer(module_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "module",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Top-level methods (become private methods of Object)
        # Track visibility for methods
        visibility = "public"
        class_context = None
        module_context = None

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Track class/module context
            class_match = re.match(r"^\s*class\s+(\w+)", line)
            if class_match:
                class_context = class_match.group(1)
                visibility = "public"  # Reset visibility in new class
                continue

            module_match = re.match(r"^\s*module\s+(\w+)", line)
            if module_match:
                module_context = module_match.group(1)
                visibility = "public"
                continue

            # Check for end of class/module
            if re.match(r"^\s*end\s*$", line):
                if class_context or module_context:
                    class_context = None
                    module_context = None
                    visibility = "public"
                continue

            # Track visibility changes
            if re.match(r"^\s*private\s*$", line):
                visibility = "private"
                continue
            elif re.match(r"^\s*protected\s*$", line):
                visibility = "protected"
                continue
            elif re.match(r"^\s*public\s*$", line):
                visibility = "public"
                continue

            # Methods
            method_match = re.match(r"^\s*def\s+(?:self\.)?(\w+(?:\?|!|=)?)", line)
            if method_match and visibility == "public":
                method_name = method_match.group(1)
                context = class_context or module_context or "global"

                exports.append(
                    {
                        "name": method_name,
                        "type": "method",
                        "line": i,
                        "context": context,
                        "is_class_method": "self." in line,
                        "is_predicate": method_name.endswith("?"),
                        "is_bang_method": method_name.endswith("!"),
                        "is_setter": method_name.endswith("="),
                    }
                )

        # Constants (UPPERCASE identifiers)
        const_pattern = r"^\s*([A-Z][A-Z0-9_]*)\s*="
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "constant",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Ruby file.

        Extracts:
        - Classes with inheritance and included modules
        - Modules with included/extended modules
        - Methods with visibility and type
        - Instance and class variables
        - Constants
        - Blocks, procs, and lambdas
        - Attribute accessors
        - Aliases

        Args:
            content: Ruby source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Extract classes with full information
        class_pattern = r"^\s*class\s+(\w+)(?:\s*<\s*([\w:]+))?"

        for match in re.finditer(class_pattern, content, re.MULTILINE):
            class_name = match.group(1)
            superclass = match.group(2)

            # Find class body
            class_start_line = content[: match.start()].count("\n") + 1
            class_body = self._extract_block_body(content, match.end(), "class")

            # Extract class components
            methods = []
            attributes = []
            included_modules = []
            extended_modules = []

            if class_body:
                methods = self._extract_methods(class_body)
                attributes = self._extract_attributes(class_body)
                included_modules = self._extract_included_modules(class_body)
                extended_modules = self._extract_extended_modules(class_body)

            class_info = ClassInfo(
                name=class_name,
                line=class_start_line,
                bases=[superclass] if superclass else [],
                methods=methods,
                attributes=attributes,
                included_modules=included_modules,
                extended_modules=extended_modules,
                is_singleton=class_name.startswith("<<"),
            )

            structure.classes.append(class_info)

        # Extract modules
        module_pattern = r"^\s*module\s+(\w+)"

        for match in re.finditer(module_pattern, content, re.MULTILINE):
            module_name = match.group(1)
            module_start_line = content[: match.start()].count("\n") + 1
            module_body = self._extract_block_body(content, match.end(), "module")

            methods = []
            included_modules = []
            extended_modules = []

            if module_body:
                methods = self._extract_methods(module_body)
                included_modules = self._extract_included_modules(module_body)
                extended_modules = self._extract_extended_modules(module_body)

            structure.modules.append(
                {
                    "name": module_name,
                    "line": module_start_line,
                    "methods": methods,
                    "included_modules": included_modules,
                    "extended_modules": extended_modules,
                }
            )

        # Extract standalone methods (outside classes/modules)
        structure.functions = self._extract_toplevel_methods(content)

        # Extract constants
        const_pattern = r"^\s*([A-Z][A-Z0-9_]*)\s*="
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            structure.constants.append(match.group(1))

        # Extract global variables
        global_var_pattern = r"\$\w+"
        global_vars = set(re.findall(global_var_pattern, content))
        structure.global_variables = list(global_vars)

        # Extract instance variables (class-level)
        ivar_pattern = r"@\w+"
        instance_vars = set(re.findall(ivar_pattern, content))
        structure.instance_variables = list(instance_vars)

        # Extract class variables
        cvar_pattern = r"@@\w+"
        class_vars = set(re.findall(cvar_pattern, content))
        structure.class_variables = list(class_vars)

        # Count blocks, procs, and lambdas
        structure.block_count = len(re.findall(r"\bdo\b|\{", content))
        structure.proc_count = len(re.findall(r"\bProc\.new\b|\bproc\b", content))
        structure.lambda_count = len(re.findall(r"\blambda\b|->|\bλ\b", content))

        # Detect Rails/framework patterns
        structure.framework = self._detect_framework(content, file_path)

        # Check for test file
        structure.is_test_file = (
            "_test.rb" in file_path.name
            or "_spec.rb" in file_path.name
            or file_path.parts
            and "test" in file_path.parts
            or file_path.parts
            and "spec" in file_path.parts
        )

        # Extract aliases
        alias_pattern = r"^\s*alias(?:_method)?\s+:?(\w+)\s+:?(\w+)"
        for match in re.finditer(alias_pattern, content, re.MULTILINE):
            structure.aliases.append(
                {
                    "new_name": match.group(1),
                    "original_name": match.group(2),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Ruby code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - ABC metrics (Assignment, Branch, Condition)
        - Method complexity
        - Metaprogramming complexity

        Args:
            content: Ruby source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Calculate cyclomatic complexity
        complexity = 1

        decision_keywords = [
            r"\bif\b",
            r"\bunless\b",
            r"\belsif\b",
            r"\belse\b",
            r"\bwhile\b",
            r"\buntil\b",
            r"\bfor\b",
            r"\bcase\b",
            r"\bwhen\b",
            r"\brescue\b",
            r"\b&&\b",
            r"\|\|",
            r"\band\b",
            r"\bor\b",
            r"\?.*:",  # Ternary operator
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        # Add complexity for iterators (they're essentially loops)
        iterator_methods = [
            r"\.each\b",
            r"\.map\b",
            r"\.select\b",
            r"\.reject\b",
            r"\.times\b",
            r"\.upto\b",
            r"\.downto\b",
        ]
        for iterator in iterator_methods:
            complexity += len(re.findall(iterator, content))

        metrics.cyclomatic = complexity

        # Calculate cognitive complexity
        cognitive = 0
        nesting_level = 0
        max_nesting = 0

        lines = content.split("\n")
        for line in lines:
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Track nesting
            if re.search(r"\b(if|unless|while|until|for|case|def|class|module|begin)\b", line):
                cognitive += 1 + nesting_level
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif re.search(r"\belsif\b", line):
                cognitive += 1 + nesting_level
            elif re.search(r"\brescue\b", line):
                cognitive += 1 + nesting_level
            elif re.search(r"\bend\b", line):
                nesting_level = max(0, nesting_level - 1)

            # Blocks add complexity
            if re.search(r"\bdo\b\s*\|", line) or re.search(r"\{\s*\|", line):
                cognitive += 1

        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting

        # Calculate ABC metrics
        abc_metrics = self._calculate_abc_metrics(content)
        metrics.abc_score = abc_metrics["score"]
        metrics.assignments = abc_metrics["assignments"]
        metrics.branches = abc_metrics["branches"]
        metrics.conditions = abc_metrics["conditions"]

        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = self._count_code_lines(content)
        metrics.comment_lines = self._count_comment_lines(content)
        metrics.comment_ratio = (
            metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
        )

        # Count methods and classes
        metrics.method_count = len(re.findall(r"^\s*def\s+", content, re.MULTILINE))
        metrics.class_count = len(re.findall(r"^\s*class\s+", content, re.MULTILINE))
        metrics.module_count = len(re.findall(r"^\s*module\s+", content, re.MULTILINE))

        # Metaprogramming metrics
        metaprogramming_methods = [
            "define_method",
            "method_missing",
            "const_missing",
            "class_eval",
            "instance_eval",
            "module_eval",
            "send",
            "__send__",
            "public_send",
            "define_singleton_method",
            "singleton_class",
        ]

        metaprogramming_count = 0
        for method in metaprogramming_methods:
            metaprogramming_count += len(re.findall(rf"\b{method}\b", content))

        metrics.metaprogramming_score = metaprogramming_count

        # Block metrics
        metrics.block_count = len(re.findall(r"\bdo\b|\{", content))
        metrics.proc_count = len(re.findall(r"\bProc\.new\b|\bproc\b", content))
        metrics.lambda_count = len(re.findall(r"\blambda\b|->|\bλ\b", content))

        # Test metrics
        if "_test.rb" in file_path.name or "_spec.rb" in file_path.name:
            metrics.test_count = len(re.findall(r'\b(?:test|it|describe|context)\s+[\'"]', content))
            metrics.assertion_count = len(re.findall(r"\bassert", content))
            metrics.expectation_count = len(re.findall(r"\bexpect\b|\bshould\b", content))

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Ruby's expressiveness
            metaprogramming_factor = 1 - (metaprogramming_count * 0.02)
            abc_factor = 1 - (metrics.abc_score / 100) if metrics.abc_score < 100 else 0

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * metaprogramming_factor
                + 10 * abc_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _is_stdlib_module(self, module: str) -> bool:
        """Check if a module is from Ruby standard library.

        Args:
            module: Module name

        Returns:
            True if it's a stdlib module
        """
        stdlib_modules = {
            "abbrev",
            "base64",
            "benchmark",
            "bigdecimal",
            "cgi",
            "coverage",
            "csv",
            "date",
            "dbm",
            "debug",
            "delegate",
            "digest",
            "drb",
            "english",
            "erb",
            "etc",
            "fcntl",
            "fiddle",
            "fileutils",
            "find",
            "forwardable",
            "gdbm",
            "getoptlong",
            "io/console",
            "io/nonblock",
            "io/wait",
            "ipaddr",
            "irb",
            "json",
            "logger",
            "matrix",
            "minitest",
            "monitor",
            "mutex_m",
            "net/ftp",
            "net/http",
            "net/imap",
            "net/pop",
            "net/smtp",
            "nkf",
            "objspace",
            "observer",
            "open-uri",
            "open3",
            "openssl",
            "optparse",
            "ostruct",
            "pathname",
            "pp",
            "prettyprint",
            "prime",
            "pstore",
            "psych",
            "pty",
            "racc",
            "rake",
            "rdoc",
            "readline",
            "reline",
            "resolv",
            "resolv-replace",
            "rexml",
            "rinda",
            "ripper",
            "rss",
            "rubygems",
            "scanf",
            "sdbm",
            "securerandom",
            "set",
            "shellwords",
            "singleton",
            "socket",
            "stringio",
            "strscan",
            "syslog",
            "tempfile",
            "time",
            "timeout",
            "tmpdir",
            "tracer",
            "tsort",
            "un",
            "uri",
            "weakref",
            "webrick",
            "win32ole",
            "yaml",
            "zlib",
        }

        # Check base module name (before any /)
        base_module = module.split("/")[0]
        return base_module in stdlib_modules

    def _extract_gemfile_dependencies(self, content: str) -> List[ImportInfo]:
        """Extract gem dependencies from Gemfile.

        Args:
            content: Gemfile content

        Returns:
            List of ImportInfo objects for gems
        """
        gems = []
        gem_pattern = r'^\s*gem\s+[\'"]([^\'"]+)[\'"](?:,\s*[\'"]([^\'"]+)[\'"])?'

        for match in re.finditer(gem_pattern, content, re.MULTILINE):
            gem_name = match.group(1)
            version = match.group(2) if match.group(2) else None

            gems.append(
                ImportInfo(
                    module=gem_name,
                    line=content[: match.start()].count("\n") + 1,
                    type="gemfile_dependency",
                    is_relative=False,
                    version=version,
                    is_gem=True,
                )
            )

        return gems

    def _extract_block_body(self, content: str, start_pos: int, block_type: str) -> Optional[str]:
        """Extract the body of a class/module block.

        Args:
            content: Source code
            start_pos: Position after class/module declaration
            block_type: 'class' or 'module'

        Returns:
            Block body content or None
        """
        # Ruby uses 'end' to close blocks
        block_depth = 1
        pos = start_pos

        while pos < len(content) and block_depth > 0:
            # Find next keyword that affects depth
            remaining = content[pos:]

            # Find next relevant keyword
            keyword_match = re.search(
                r"\b(class|module|def|if|unless|while|until|for|case|begin|do|end)\b", remaining
            )

            if not keyword_match:
                break

            keyword = keyword_match.group(1)
            pos += keyword_match.start()

            if keyword in [
                "class",
                "module",
                "def",
                "if",
                "unless",
                "while",
                "until",
                "for",
                "case",
                "begin",
                "do",
            ]:
                block_depth += 1
            elif keyword == "end":
                block_depth -= 1
                if block_depth == 0:
                    return content[start_pos:pos]

            pos += len(keyword)

        return None

    def _extract_methods(self, class_body: str) -> List[Dict[str, Any]]:
        """Extract methods from class/module body.

        Args:
            class_body: Content of class/module body

        Returns:
            List of method information
        """
        methods = []
        visibility = "public"

        lines = class_body.split("\n")
        for i, line in enumerate(lines):
            # Track visibility
            if re.match(r"^\s*private\s*$", line):
                visibility = "private"
                continue
            elif re.match(r"^\s*protected\s*$", line):
                visibility = "protected"
                continue
            elif re.match(r"^\s*public\s*$", line):
                visibility = "public"
                continue

            # Extract methods
            method_match = re.match(
                r"^\s*def\s+((?:self\.)?(?:\w+(?:\?|!|=)?)|(?:[\+\-\*\/\%\[\]]+|<<|>>|==|!=|<=|>=|<=>|\|\||&&))",
                line,
            )
            if method_match:
                method_name = method_match.group(1)

                methods.append(
                    {
                        "name": method_name,
                        "visibility": visibility,
                        "line": i + 1,
                        "is_class_method": method_name.startswith("self."),
                        "is_predicate": method_name.endswith("?"),
                        "is_bang_method": method_name.endswith("!"),
                        "is_setter": method_name.endswith("="),
                        "is_operator": bool(
                            re.match(r"^[\+\-\*\/\%\[\]<>=!&|]+$", method_name.replace("self.", ""))
                        ),
                    }
                )

        return methods

    def _extract_attributes(self, class_body: str) -> List[Dict[str, str]]:
        """Extract attribute accessors from class body.

        Args:
            class_body: Content of class body

        Returns:
            List of attribute information
        """
        attributes = []

        # attr_reader, attr_writer, attr_accessor
        attr_pattern = r"^\s*(attr_(?:reader|writer|accessor))\s+(.*?)$"

        for match in re.finditer(attr_pattern, class_body, re.MULTILINE):
            attr_type = match.group(1)
            attr_list = match.group(2)

            # Parse attribute names (can be symbols or strings)
            attr_names = re.findall(r"[:\'\"]?(\w+)[:\'\"]?", attr_list)

            for name in attr_names:
                attributes.append(
                    {
                        "name": name,
                        "type": attr_type,
                        "line": class_body[: match.start()].count("\n") + 1,
                    }
                )

        return attributes

    def _extract_included_modules(self, class_body: str) -> List[str]:
        """Extract included modules from class/module body.

        Args:
            class_body: Content of class/module body

        Returns:
            List of included module names
        """
        modules = []
        include_pattern = r"^\s*include\s+([\w:]+)"

        for match in re.finditer(include_pattern, class_body, re.MULTILINE):
            modules.append(match.group(1))

        return modules

    def _extract_extended_modules(self, class_body: str) -> List[str]:
        """Extract extended modules from class/module body.

        Args:
            class_body: Content of class/module body

        Returns:
            List of extended module names
        """
        modules = []
        extend_pattern = r"^\s*extend\s+([\w:]+)"

        for match in re.finditer(extend_pattern, class_body, re.MULTILINE):
            modules.append(match.group(1))

        return modules

    def _extract_toplevel_methods(self, content: str) -> List[FunctionInfo]:
        """Extract top-level methods (outside classes/modules).

        Args:
            content: Ruby source code

        Returns:
            List of FunctionInfo objects
        """
        methods = []
        in_class_or_module = False
        depth = 0

        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Track if we're inside a class or module
            if re.match(r"^\s*(class|module)\s+", line):
                in_class_or_module = True
                depth = 1
                continue

            if in_class_or_module:
                if re.search(
                    r"\b(class|module|def|if|unless|while|until|for|case|begin|do)\b", line
                ):
                    depth += 1
                elif re.search(r"\bend\b", line):
                    depth -= 1
                    if depth == 0:
                        in_class_or_module = False
                continue

            # Extract top-level methods
            method_match = re.match(r"^\s*def\s+(\w+(?:\?|!|=)?)", line)
            if method_match and not in_class_or_module:
                methods.append(
                    FunctionInfo(name=method_match.group(1), line=i + 1, is_toplevel=True)
                )

        return methods

    def _detect_framework(self, content: str, file_path: Path) -> Optional[str]:
        """Detect which framework is being used.

        Args:
            content: Ruby source code
            file_path: Path to the file

        Returns:
            Framework name or None
        """
        # Rails indicators
        rails_indicators = [
            r"class\s+\w+\s*<\s*(?:ApplicationController|ActiveRecord::Base|ActionController::Base)",
            r"Rails\.application",
            r"ActiveRecord",
            r"ActionController",
            r"ActionView",
            r"before_action",
            r"has_many",
            r"belongs_to",
            r"validates",
        ]

        for pattern in rails_indicators:
            if re.search(pattern, content):
                return "Rails"

        # Sinatra indicators
        if re.search(r'require\s+[\'"]sinatra[\'"]', content) or re.search(
            r'\b(?:get|post|put|delete|patch)\s+[\'"]/', content
        ):
            return "Sinatra"

        # RSpec indicators
        if re.search(r'require\s+[\'"]spec_helper[\'"]', content) or re.search(
            r'\bdescribe\s+[\'"]', content
        ):
            return "RSpec"

        # Minitest indicators
        if re.search(r'require\s+[\'"]minitest', content) or re.search(
            r"class\s+\w+\s*<\s*Minitest::Test", content
        ):
            return "Minitest"

        # Rake
        if file_path.suffix == ".rake" or "Rakefile" in file_path.name:
            return "Rake"

        return None

    def _calculate_abc_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate ABC (Assignment, Branch, Condition) metrics.

        Args:
            content: Ruby source code

        Returns:
            Dictionary with ABC metrics
        """
        assignments = 0
        branches = 0
        conditions = 0

        # Assignments
        assignment_patterns = [
            r"\w+\s*=\s*[^=]",  # Variable assignment
            r"\w+\s*\+=",  # Compound assignments
            r"\w+\s*-=",
            r"\w+\s*\*=",
            r"\w+\s*\/=",
            r"\w+\s*\|\|=",  # Or-equals
            r"\w+\s*&&=",  # And-equals
        ]

        for pattern in assignment_patterns:
            assignments += len(re.findall(pattern, content))

        # Branches (method calls)
        branch_patterns = [
            r"\.\w+\(",  # Method calls with parentheses
            r"\.\w+\s+[^=]",  # Method calls without parentheses
            r"\w+\(",  # Function calls
            r"super\b",  # Super calls
            r"yield\b",  # Yield statements
        ]

        for pattern in branch_patterns:
            branches += len(re.findall(pattern, content))

        # Conditions
        condition_patterns = [
            r"\bif\b",
            r"\bunless\b",
            r"\belsif\b",
            r"\bwhile\b",
            r"\buntil\b",
            r"\bcase\b",
            r"\bwhen\b",
            r"==",
            r"!=",
            r"<",
            r">",
            r"<=",
            r">=",
            r"=~",
            r"!~",  # Regex matches
            r"\?.*:",  # Ternary operator
            r"&&",
            r"\|\|",
            r"\band\b",
            r"\bor\b",
        ]

        for pattern in condition_patterns:
            conditions += len(re.findall(pattern, content))

        # Calculate ABC score (square root of sum of squares)
        import math

        score = math.sqrt(assignments**2 + branches**2 + conditions**2)

        return {
            "score": round(score, 2),
            "assignments": assignments,
            "branches": branches,
            "conditions": conditions,
        }

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: Ruby source code

        Returns:
            Number of code lines
        """
        count = 0
        in_multiline_comment = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Handle multiline comments (=begin...=end)
            if stripped == "=begin":
                in_multiline_comment = True
                continue
            elif stripped == "=end":
                in_multiline_comment = False
                continue

            if in_multiline_comment:
                continue

            # Skip empty lines and single-line comments
            if stripped and not stripped.startswith("#"):
                count += 1

        return count

    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines in Ruby code.

        Args:
            content: Ruby source code

        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_comment = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Single-line comments
            if stripped.startswith("#"):
                count += 1
                continue

            # Multi-line comments
            if stripped == "=begin":
                in_multiline_comment = True
                count += 1
                continue
            elif stripped == "=end":
                in_multiline_comment = False
                count += 1
                continue

            if in_multiline_comment:
                count += 1

        return count
