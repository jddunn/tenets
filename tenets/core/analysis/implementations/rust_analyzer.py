"""Rust code analyzer.

This module provides comprehensive analysis for Rust source files,
including support for Rust's ownership system, traits, and modern features.
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


class RustAnalyzer(LanguageAnalyzer):
    """Rust code analyzer.

    Provides comprehensive analysis for Rust files including:
    - Use statement and module analysis
    - Struct and enum extraction with generics
    - Trait definition and implementation
    - Function analysis with lifetimes and generics
    - Macro usage and definition
    - Ownership and borrowing patterns
    - Async/await support
    - Unsafe code detection
    - Cargo dependency analysis

    Handles Rust's unique features like ownership, lifetimes, and traits.
    """

    language_name = "rust"
    file_extensions = [".rs"]

    def __init__(self):
        """Initialize the Rust analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract imports from Rust code.

        Handles:
        - use statements with paths
        - use statements with glob imports
        - use statements with aliases
        - use statements with nested imports
        - extern crate declarations
        - mod declarations

        Args:
            content: Rust source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        # Support multi-line `use` statements by accumulating until semicolon
        use_acc: Optional[str] = None
        brace_depth = 0

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//"):
                continue

            # Accumulate multi-line use
            if use_acc is not None:
                use_acc += " " + line.strip()
                brace_depth += line.count("{") - line.count("}")
                if ";" in line and brace_depth <= 0:
                    # Completed statement
                    use_path = use_acc.rstrip(";").strip()
                    parsed_imports = self._parse_use_statement(use_path, i)
                    imports.extend(parsed_imports)
                    use_acc = None
                continue

            # Use statements (single-line start)
            use_pattern = re.compile(r"^\s*(?:pub\s+)?use\s+(.+);")
            start_use_pattern = re.compile(r"^\s*(?:pub\s+)?use\s+(.+)")
            match = use_pattern.match(line)
            if match:
                use_path = match.group(1).strip()
                parsed_imports = self._parse_use_statement(use_path, i)
                imports.extend(parsed_imports)
                continue
            # Start of multi-line use (no semicolon yet)
            start_match = start_use_pattern.match(line)
            if start_match and not line.strip().endswith(";"):
                use_acc = start_match.group(1).strip()
                brace_depth = line.count("{") - line.count("}")
                continue

            # Extern crate
            extern_pattern = re.compile(r"^\s*extern\s+crate\s+(\w+)(?:\s+as\s+(\w+))?;")
            match = extern_pattern.match(line)
            if match:
                crate_name = match.group(1)
                alias = match.group(2)

                imports.append(
                    ImportInfo(
                        module=crate_name,
                        alias=alias,
                        line=i,
                        type="extern_crate",
                        is_relative=False,
                        is_external=True,
                    )
                )
                continue

            # Mod declarations
            mod_pattern = re.compile(r"^\s*(?:pub\s+)?mod\s+(\w+);")
            match = mod_pattern.match(line)
            if match:
                module_name = match.group(1)

                imports.append(
                    ImportInfo(
                        module=module_name,
                        line=i,
                        type="mod",
                        is_relative=True,
                        is_module_declaration=True,
                    )
                )
                continue

        # Check for Cargo.toml dependencies
        if file_path.name == "Cargo.toml":
            imports.extend(self._extract_cargo_dependencies(content))

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract public items from Rust code.

        In Rust, items marked with 'pub' are exported.

        Args:
            content: Rust source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported (public) symbols
        """
        exports = []

        # Public functions
        pub_fn_pattern = r'^\s*pub\s+(?:(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?:extern\s+(?:"[^"]+"\s+)?)?)?fn\s+(\w+)'

        for match in re.finditer(pub_fn_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            line_content = content[match.start() : match.end()]

            exports.append(
                {
                    "name": func_name,
                    "type": "function",
                    "line": content[: match.start()].count("\n") + 1,
                    "is_async": "async" in line_content,
                    "is_unsafe": "unsafe" in line_content,
                    "is_const": "const fn" in line_content,
                    "is_extern": "extern" in line_content,
                }
            )

        # Public structs
        pub_struct_pattern = r"^\s*pub\s+struct\s+(\w+)"

        for match in re.finditer(pub_struct_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "struct",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Public enums
        pub_enum_pattern = r"^\s*pub\s+enum\s+(\w+)"

        for match in re.finditer(pub_enum_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "enum",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Public traits
        pub_trait_pattern = r"^\s*pub\s+(?:unsafe\s+)?trait\s+(\w+)"

        for match in re.finditer(pub_trait_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "trait",
                    "line": content[: match.start()].count("\n") + 1,
                    "is_unsafe": "unsafe trait" in match.group(0),
                }
            )

        # Public type aliases
        pub_type_pattern = r"^\s*pub\s+type\s+(\w+)"

        for match in re.finditer(pub_type_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "type_alias",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Public constants
        pub_const_pattern = r"^\s*pub\s+const\s+(\w+):"

        for match in re.finditer(pub_const_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "constant",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Public statics
        pub_static_pattern = r"^\s*pub\s+static\s+(?:mut\s+)?(\w+):"

        for match in re.finditer(pub_static_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "static",
                    "line": content[: match.start()].count("\n") + 1,
                    "is_mutable": "mut" in match.group(0),
                }
            )

        # Public modules
        pub_mod_pattern = r"^\s*pub\s+mod\s+(\w+)"

        for match in re.finditer(pub_mod_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "module",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Public macros (macro_rules!)
        pub_macro_pattern = r"#\[macro_export\]\s*\n\s*macro_rules!\s+(\w+)"

        for match in re.finditer(pub_macro_pattern, content):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "macro",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Rust file.

        Extracts:
        - Structs with fields and generics
        - Enums with variants
        - Traits with methods
        - Implementations (impl blocks)
        - Functions with signatures
        - Modules
        - Type aliases
        - Constants and statics
        - Macros

        Args:
            content: Rust source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Determine if it's a lib.rs or main.rs
        structure.is_library = file_path.name == "lib.rs"
        structure.is_binary = file_path.name == "main.rs"
        structure.is_test = file_path.name.endswith("_test.rs") or "tests" in file_path.parts

        # Extract structs
        struct_pattern = r"(?:^|\n)\s*(?:pub\s+)?struct\s+(\w+)(?:<([^>]+)>)?(?:\s*\(|;|\s*\{)?"

        for match in re.finditer(struct_pattern, content):
            struct_name = match.group(1)
            generics = match.group(2)

            # Determine struct type
            following_char = (
                content[match.end() : match.end() + 1] if match.end() < len(content) else ""
            )
            if following_char == "(":
                struct_type = "tuple"
            elif following_char == ";":
                struct_type = "unit"
            else:
                struct_type = "regular"

            # Extract struct body if regular struct
            fields = []
            if struct_type == "regular":
                struct_body = self._extract_block_body(content, match.end())
                if struct_body:
                    fields = self._extract_struct_fields(struct_body)

            class_info = ClassInfo(
                name=struct_name,
                line=content[: match.start()].count("\n") + 1,
                generics=generics,
                struct_type=struct_type,
                fields=fields,
                is_public="pub struct" in match.group(0),
            )

            structure.classes.append(class_info)  # Using classes for structs

        # Extract enums
        enum_pattern = r"(?:^|\n)\s*(?:pub\s+)?enum\s+(\w+)(?:<([^>]+)>)?"

        for match in re.finditer(enum_pattern, content):
            enum_name = match.group(1)
            generics = match.group(2)

            # Extract enum variants
            enum_body = self._extract_block_body(content, match.end())
            variants = self._extract_enum_variants(enum_body) if enum_body else []

            structure.enums.append(
                {
                    "name": enum_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "generics": generics,
                    "variants": variants,
                    "is_public": "pub enum" in match.group(0),
                }
            )

        # Extract traits
        trait_pattern = r"(?:^|\n)\s*(?:pub\s+)?(?:unsafe\s+)?trait\s+(\w+)(?:<([^>]+)>)?"

        for match in re.finditer(trait_pattern, content):
            trait_name = match.group(1)
            generics = match.group(2)

            # Extract trait methods
            trait_body = self._extract_block_body(content, match.end())
            methods = self._extract_trait_methods(trait_body) if trait_body else []

            structure.traits.append(
                {
                    "name": trait_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "generics": generics,
                    "methods": methods,
                    "is_unsafe": "unsafe trait" in match.group(0),
                    "is_public": "pub trait" in match.group(0),
                }
            )

        # Extract impl blocks
        impl_pattern = r"(?:^|\n)\s*(?:unsafe\s+)?impl(?:<([^>]+)>)?\s+(?:(\w+)(?:<[^>]+>)?\s+for\s+)?(\w+)(?:<[^>]+>)?"

        for match in re.finditer(impl_pattern, content):
            impl_generics = match.group(1)
            trait_name = match.group(2)
            type_name = match.group(3)

            # Extract impl methods
            impl_body = self._extract_block_body(content, match.end())
            methods = self._extract_impl_methods(impl_body) if impl_body else []

            structure.impl_blocks.append(
                {
                    "type": type_name,
                    "trait": trait_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "generics": impl_generics,
                    "methods": methods,
                    "is_unsafe": "unsafe impl" in match.group(0),
                }
            )

        # Extract functions
        fn_pattern = r'(?:^|\n)\s*(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?:extern\s+(?:"[^"]+"\s+)?)?fn\s+(\w+)(?:<([^>]+)>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]+))?'

        for match in re.finditer(fn_pattern, content):
            func_name = match.group(1)
            generics = match.group(2)
            params = match.group(3)
            return_type = match.group(4)

            # Skip if inside impl block
            if self._is_inside_impl(content, match.start()):
                continue

            func_info = FunctionInfo(
                name=func_name,
                line=content[: match.start()].count("\n") + 1,
                generics=generics,
                args=self._parse_rust_params(params),
                return_type=return_type.strip() if return_type else None,
                is_async="async fn" in match.group(0),
                is_unsafe="unsafe fn" in match.group(0),
                is_const="const fn" in match.group(0),
                is_extern="extern" in match.group(0),
                is_public="pub fn" in match.group(0),
            )

            structure.functions.append(func_info)

        # Extract modules
        mod_pattern = r"(?:^|\n)\s*(?:pub\s+)?mod\s+(\w+)\s*(?:;|\{)"

        for match in re.finditer(mod_pattern, content):
            module_name = match.group(1)
            is_inline = "{" in match.group(0)

            structure.modules.append(
                {
                    "name": module_name,
                    "line": content[: match.start()].count("\n") + 1,
                    "is_inline": is_inline,
                    "is_public": "pub mod" in match.group(0),
                }
            )

        # Extract type aliases
        type_alias_pattern = r"(?:^|\n)\s*(?:pub\s+)?type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);"

        for match in re.finditer(type_alias_pattern, content):
            structure.type_aliases.append(
                {
                    "name": match.group(1),
                    "generics": match.group(2),
                    "target": match.group(3).strip(),
                    "line": content[: match.start()].count("\n") + 1,
                    "is_public": "pub type" in match.group(0),
                }
            )

        # Extract constants
        const_pattern = r"(?:^|\n)\s*(?:pub\s+)?const\s+(\w+):\s*([^=]+)\s*="

        for match in re.finditer(const_pattern, content):
            structure.constants.append(match.group(1))

        # Extract statics
        static_pattern = r"(?:^|\n)\s*(?:pub\s+)?static\s+(?:mut\s+)?(\w+):"

        for match in re.finditer(static_pattern, content):
            structure.statics.append(
                {
                    "name": match.group(1),
                    "is_mutable": "mut" in match.group(0),
                    "is_public": "pub static" in match.group(0),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Extract macros
        macro_pattern = r"macro_rules!\s+(\w+)"

        for match in re.finditer(macro_pattern, content):
            structure.macros.append(
                {
                    "name": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                    "is_exported": "#[macro_export]"
                    in content[max(0, match.start() - 100) : match.start()],
                }
            )

        # Count unsafe blocks
        structure.unsafe_blocks = len(re.findall(r"\bunsafe\s*\{", content))

        # Count async functions
        structure.async_functions = len(re.findall(r"\basync\s+fn\b", content))

        # Also include await points at structure-level
        structure.await_points = len(re.findall(r"\.await", content))

        # Unsafe functions count at structure-level
        structure.unsafe_functions = len(re.findall(r"\bunsafe\s+fn\b", content))

        # Detect closures (lambdas) of the form `|...|` possibly with move/async before
        structure.lambda_count = len(re.findall(r"\|[^|]*\|\s*(?:->\s*[^\s{]+)?", content))

        # Detect test functions
        structure.test_functions = len(re.findall(r"#\[test\]", content))
        structure.bench_functions = len(re.findall(r"#\[bench\]", content))

        # Detect common derive macros
        derive_pattern = r"#\[derive\(([^)]+)\)\]"
        derives = []
        for match in re.finditer(derive_pattern, content):
            derive_list = match.group(1)
            for derive in derive_list.split(","):
                derives.append(derive.strip())
        structure.derives = list(set(derives))

        # Detect workspace/crate type
        if file_path.parent.name == "src":
            if file_path.parent.parent.joinpath("Cargo.toml").exists():
                structure.crate_type = "workspace_member"

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Rust code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Unsafe code metrics
        - Lifetime complexity
        - Generic complexity
        - Pattern matching complexity

        Args:
            content: Rust source code
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
            r"\bwhile\b",
            r"\bfor\b",
            r"\bloop\b",
            r"\bmatch\b",
            r"\b=>\b",  # Match arms
            r"\b&&\b",
            r"\|\|",
            r"\?",  # Try operator
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        # Add complexity for Result/Option handling
        complexity += len(re.findall(r"\.unwrap\(\)", content))
        complexity += len(re.findall(r"\.expect\(", content))

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
                (r"\bloop\b", 1),
                (r"\bmatch\b", 2),  # Match is more complex
                (r"\b=>\b", 0.5),  # Match arms
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

        # Count structures
        metrics.function_count = len(re.findall(r"\bfn\s+\w+", content))
        metrics.struct_count = len(re.findall(r"\bstruct\s+\w+", content))
        metrics.enum_count = len(re.findall(r"\benum\s+\w+", content))
        metrics.trait_count = len(re.findall(r"\btrait\s+\w+", content))
        metrics.impl_count = len(re.findall(r"\bimpl\b", content))

        # Unsafe metrics
        metrics.unsafe_blocks = len(re.findall(r"\bunsafe\s*\{", content))
        # Count only free (non-trait/non-impl) unsafe functions
        unsafe_fn_matches = list(re.finditer(r"\bunsafe\s+fn\b", content))
        # Build trait/impl spans
        spans: List[tuple[int, int]] = []
        for m in re.finditer(r"\btrait\b", content):
            span = self._find_block_span(content, m.end())
            if span:
                spans.append(span)
        for m in re.finditer(r"\bimpl\b", content):
            span = self._find_block_span(content, m.end())
            if span:
                spans.append(span)

        def _in_spans(idx: int) -> bool:
            for s, e in spans:
                if s <= idx < e:
                    return True
            return False

        metrics.unsafe_functions = sum(1 for m in unsafe_fn_matches if not _in_spans(m.start()))
        metrics.unsafe_traits = len(re.findall(r"\bunsafe\s+trait\b", content))
        metrics.unsafe_impl = len(re.findall(r"\bunsafe\s+impl\b", content))
        metrics.unsafe_score = (
            metrics.unsafe_blocks
            + metrics.unsafe_functions * 2
            + metrics.unsafe_traits * 3
            + metrics.unsafe_impl * 3
        )

        # Lifetime metrics
        metrics.lifetime_annotations = len(re.findall(r"'\w+", content))
        metrics.lifetime_bounds = len(re.findall(r"'\w+\s*:\s*'\w+", content))

        # Generic metrics
        metrics.generic_types = len(re.findall(r"<\s*(?:T|[A-Z])\s*(?:,\s*[A-Z]\s*)*>", content))
        metrics.trait_bounds = len(
            re.findall(r":\s*(?:Send|Sync|Clone|Copy|Debug|Display)", content)
        )

        # Async metrics
        metrics.async_functions = len(re.findall(r"\basync\s+fn\b", content))
        metrics.await_points = len(re.findall(r"\.await", content))

        # Error handling metrics
        metrics.result_types = len(re.findall(r"Result<", content))
        metrics.option_types = len(re.findall(r"Option<", content))
        metrics.unwrap_calls = len(re.findall(r"\.unwrap\(\)", content))
        metrics.expect_calls = len(re.findall(r"\.expect\(", content))
        metrics.question_marks = len(re.findall(r"\?(?:\s|;|\))", content))

        # Macro usage
        metrics.macro_invocations = len(re.findall(r"\w+!\s*(?:\[|\(|\{)", content))
        metrics.derive_macros = len(re.findall(r"#\[derive\(", content))

        # Test metrics
        if file_path.name.endswith("_test.rs") or "tests" in file_path.parts:
            metrics.test_count = len(re.findall(r"#\[test\]", content))
            metrics.bench_count = len(re.findall(r"#\[bench\]", content))
            metrics.assertion_count = len(re.findall(r"assert(?:_eq|_ne|!)?\!", content))

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Rust's safety features
            unsafe_factor = 1 - (metrics.unsafe_score * 0.02)
            error_handling_factor = min(
                1.0, metrics.question_marks / (metrics.unwrap_calls + metrics.expect_calls + 1)
            )

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 15 * unsafe_factor
                + 10 * error_handling_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _parse_use_statement(self, use_path: str, line_num: int) -> List[ImportInfo]:
        """Parse a use statement into ImportInfo objects.

        Args:
            use_path: The path in the use statement
            line_num: Line number of the use statement

        Returns:
            List of ImportInfo objects
        """
        imports = []

        # Handle grouped imports: use std::{io, fs};
        if "{" in use_path:
            base_path, group = use_path.split("{", 1)
            base_path = base_path.strip().rstrip("::")
            group = group.rstrip("}").strip()

            for item in group.split(","):
                item = item.strip()

                # Handle nested groups
                if "{" in item:
                    # Recursive handling of nested groups
                    nested_imports = self._parse_use_statement(f"{base_path}::{item}", line_num)
                    imports.extend(nested_imports)
                else:
                    # Handle aliases in group
                    if " as " in item:
                        name, alias = item.split(" as ")
                        full_path = f"{base_path}::{name.strip()}" if base_path else name.strip()
                        imports.append(
                            ImportInfo(
                                module=full_path,
                                alias=alias.strip(),
                                line=line_num,
                                type="use",
                                is_relative=self._is_relative_import(full_path),
                            )
                        )
                    else:
                        full_path = f"{base_path}::{item}" if base_path else item
                        imports.append(
                            ImportInfo(
                                module=full_path,
                                line=line_num,
                                type="use",
                                is_relative=self._is_relative_import(full_path),
                            )
                        )
        else:
            # Simple use statement
            if " as " in use_path:
                path, alias = use_path.split(" as ")
                imports.append(
                    ImportInfo(
                        module=path.strip(),
                        alias=alias.strip(),
                        line=line_num,
                        type="use",
                        is_relative=self._is_relative_import(path.strip()),
                    )
                )
            else:
                # Handle glob imports
                is_glob = use_path.endswith("*")
                imports.append(
                    ImportInfo(
                        module=use_path,
                        line=line_num,
                        type="use_glob" if is_glob else "use",
                        is_relative=self._is_relative_import(use_path),
                        is_glob=is_glob,
                    )
                )

        return imports

    def _is_relative_import(self, path: str) -> bool:
        """Check if an import path is relative.

        Args:
            path: Import path

        Returns:
            True if the import is relative
        """
        return path.startswith("self::") or path.startswith("super::") or path.startswith("crate::")

    def _extract_cargo_dependencies(self, content: str) -> List[ImportInfo]:
        """Extract dependencies from Cargo.toml.

        Args:
            content: Cargo.toml content

        Returns:
            List of ImportInfo objects for dependencies
        """
        imports = []

        # Simple TOML parsing for dependencies
        in_dependencies = False
        in_dev_dependencies = False

        for line in content.split("\n"):
            line = line.strip()

            if line == "[dependencies]":
                in_dependencies = True
                in_dev_dependencies = False
                continue
            elif line == "[dev-dependencies]":
                in_dependencies = False
                in_dev_dependencies = True
                continue
            elif line.startswith("["):
                in_dependencies = False
                in_dev_dependencies = False
                continue

            if in_dependencies or in_dev_dependencies:
                # Parse dependency line
                if "=" in line:
                    parts = line.split("=", 1)
                    crate_name = parts[0].strip()
                    version_info = parts[1].strip().strip('"')

                    imports.append(
                        ImportInfo(
                            module=crate_name,
                            version=version_info,
                            type=(
                                "cargo_dev_dependency"
                                if in_dev_dependencies
                                else "cargo_dependency"
                            ),
                            is_relative=False,
                            is_dev_dependency=in_dev_dependencies,
                        )
                    )

        return imports

    def _extract_block_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a block (struct, enum, trait, etc.).

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

        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            return content[brace_pos + 1 : pos - 1]

        return None

    def _find_block_span(self, content: str, start_pos: int) -> Optional[tuple[int, int]]:
        """Find the span (start_index_of_open_brace, end_index_after_closing_brace) for a block starting after a keyword.

        Looks for the next '{' after start_pos and returns the span covering the balanced block.
        """
        brace_pos = content.find("{", start_pos)
        if brace_pos == -1:
            return None
        depth = 1
        pos = brace_pos + 1
        while pos < len(content) and depth > 0:
            if content[pos] == "{":
                depth += 1
            elif content[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            return (brace_pos, pos)
        return None

    def _extract_struct_fields(self, struct_body: str) -> List[Dict[str, Any]]:
        """Extract fields from struct body.

        Args:
            struct_body: Content of struct body

        Returns:
            List of field information
        """
        fields = []

        # Field pattern
        field_pattern = r"^\s*(?:pub(?:\((?:crate|super|in\s+[^)]+)\))?\s+)?(\w+)\s*:\s*([^,\n]+)"

        for match in re.finditer(field_pattern, struct_body, re.MULTILINE):
            field_name = match.group(1)
            field_type = match.group(2).strip().rstrip(",")
            is_public = "pub" in match.group(0)

            fields.append(
                {
                    "name": field_name,
                    "type": field_type,
                    "is_public": is_public,
                    "line": struct_body[: match.start()].count("\n") + 1,
                }
            )

        return fields

    def _extract_enum_variants(self, enum_body: str) -> List[Dict[str, Any]]:
        """Extract variants from enum body.

        Args:
            enum_body: Content of enum body

        Returns:
            List of variant information
        """
        variants = []

        # Variant patterns (allow whitespace before payload)
        variant_pattern = r"^\s*(\w+)(?:\s*\(([^)]*)\)|\s*\{([^}]*)\})?(?:\s*=\s*(\d+))?"

        for match in re.finditer(variant_pattern, enum_body, re.MULTILINE):
            variant_name = match.group(1)
            tuple_data = match.group(2)
            struct_data = match.group(3)
            discriminant = match.group(4)

            if variant_name:  # Filter out empty matches
                # Determine type with robust fallback on the matched text
                text = match.group(0)
                if tuple_data is not None or ("(" in text and ")" in text):
                    variant_type = "tuple"
                elif struct_data is not None or ("{" in text and "}" in text):
                    variant_type = "struct"
                else:
                    variant_type = "unit"

                variants.append(
                    {
                        "name": variant_name,
                        "type": variant_type,
                        "discriminant": int(discriminant) if discriminant else None,
                        "line": enum_body[: match.start()].count("\n") + 1,
                    }
                )

        return variants

    def _extract_trait_methods(self, trait_body: str) -> List[Dict[str, Any]]:
        """Extract method signatures from trait body.

        Args:
            trait_body: Content of trait body

        Returns:
            List of method signatures
        """
        methods = []

        # Method signature pattern
        method_pattern = r"fn\s+(\w+)(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*->\s*([^;{]+))?"

        for match in re.finditer(method_pattern, trait_body):
            method_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            # Check if it has a default implementation
            has_body = "{" in trait_body[match.end() : match.end() + 10]

            methods.append(
                {
                    "name": method_name,
                    "parameters": self._parse_rust_params(params),
                    "return_type": return_type.strip() if return_type else None,
                    "has_default": has_body,
                    "line": trait_body[: match.start()].count("\n") + 1,
                }
            )

        return methods

    def _extract_impl_methods(self, impl_body: str) -> List[Dict[str, Any]]:
        """Extract methods from impl block.

        Args:
            impl_body: Content of impl block

        Returns:
            List of method information
        """
        methods = []

        # Method pattern
        method_pattern = r"(?:pub\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?fn\s+(\w+)(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]+))?"

        for match in re.finditer(method_pattern, impl_body):
            method_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            methods.append(
                {
                    "name": method_name,
                    "parameters": self._parse_rust_params(params),
                    "return_type": return_type.strip() if return_type else None,
                    "is_public": "pub fn" in match.group(0),
                    "is_async": "async fn" in match.group(0),
                    "is_unsafe": "unsafe fn" in match.group(0),
                    "is_const": "const fn" in match.group(0),
                    "line": impl_body[: match.start()].count("\n") + 1,
                }
            )

        return methods

    def _parse_rust_params(self, params_str: str) -> List[str]:
        """Parse Rust function parameters.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter descriptions
        """
        if not params_str.strip():
            return []

        params = []
        current_param = ""
        depth = 0

        for char in params_str:
            if char in "<([{":
                depth += 1
            elif char in ">)]}":
                depth -= 1
            elif char == "," and depth == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue

            current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        return params

    def _is_inside_impl(self, content: str, pos: int) -> bool:
        """Check if a position is inside an impl block.

        Args:
            content: Source code
            pos: Position to check

        Returns:
            True if inside an impl block
        """
        before = content[:pos]

        # Find impl blocks before position
        impl_matches = list(re.finditer(r"\bimpl\b[^{]*\{", before))
        if not impl_matches:
            return False

        # Check if we're inside the last impl
        last_impl = impl_matches[-1].end()
        between = before[last_impl:]

        open_braces = between.count("{")
        close_braces = between.count("}")

        return open_braces > close_braces

    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.

        Args:
            content: Rust source code

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
                # Count attribute lines as code
                if stripped.startswith("#[") or stripped.startswith("#!["):
                    count += 1
                else:
                    count += 1

        return count

    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines including doc comments.

        Args:
            content: Rust source code

        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_comment = False

        for line in content.split("\n"):
            stripped = line.strip()

            # Single-line comments (including doc comments)
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

        return count
