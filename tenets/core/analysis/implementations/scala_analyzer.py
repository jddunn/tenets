"""Scala code analyzer with functional programming support.

This module provides comprehensive analysis for Scala source files,
including support for object-oriented and functional programming paradigms,
pattern matching, implicits, and modern Scala 3 features.
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


class ScalaAnalyzer(LanguageAnalyzer):
    """Scala code analyzer with functional programming support.

    Provides comprehensive analysis for Scala files including:
    - Import statements with wildcards and renames
    - Package declarations and package objects
    - Classes, traits, objects, case classes
    - Implicit definitions and conversions
    - Pattern matching and case statements
    - For comprehensions and monadic operations
    - Higher-order functions and currying
    - Type parameters with variance annotations
    - Lazy vals and by-name parameters
    - Sealed traits and algebraic data types
    - Companion objects
    - Scala 3 features (given/using, extension methods, etc.)

    Supports both Scala 2.x and Scala 3.x syntax.
    """

    language_name = "scala"
    file_extensions = [".scala", ".sc"]

    def __init__(self):
        """Initialize the Scala analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract import statements from Scala code.

        Handles:
        - import statements: import scala.collection.mutable
        - Wildcard imports: import java.util._
        - Multiple imports: import java.util.{List, Map}
        - Renamed imports: import java.util.{List => JList}
        - Import all and hide: import java.util.{_, List => _}
        - Package declarations
        - Scala 3 given imports: import cats.implicits.given

        Args:
            content: Scala source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        # Track current package
        current_package = None

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("//"):
                continue

            # Package declaration
            package_pattern = r'^\s*package\s+([\w\.]+)'
            match = re.match(package_pattern, line)
            if match:
                current_package = match.group(1)
                imports.append(
                    ImportInfo(
                        module=current_package,
                        line=i,
                        type="package",
                        is_relative=False,
                        is_package_declaration=True,
                    )
                )
                continue

            # Import statements
            import_pattern = r'^\s*import\s+(.+?)(?:\s*//.*)?$'
            match = re.match(import_pattern, line)
            if match:
                import_str = match.group(1).strip()
                
                # Check for given imports (Scala 3)
                is_given_import = "given" in import_str
                if is_given_import:
                    import_str = import_str.replace(".given", "")
                    import_type = "given_import"
                else:
                    import_type = "import"

                # Handle multiple imports with braces
                if '{' in import_str and '}' in import_str:
                    # Extract base and items
                    base_match = re.match(r'([^{]+)\{([^}]+)\}', import_str)
                    if base_match:
                        base_path = base_match.group(1).strip()
                        items = base_match.group(2).strip()
                        
                        # Parse individual items
                        for item in items.split(','):
                            item = item.strip()
                            
                            # Handle renames (=> syntax)
                            if '=>' in item:
                                parts = item.split('=>')
                                original = parts[0].strip()
                                renamed = parts[1].strip()
                                
                                # Check if it's hiding (=> _)
                                if renamed == '_':
                                    continue  # Hidden import
                                else:
                                    imports.append(
                                        ImportInfo(
                                            module=f"{base_path}{original}",
                                            alias=renamed,
                                            line=i,
                                            type=import_type,
                                            is_relative=False,
                                            is_renamed=True,
                                            category=self._categorize_import(base_path),
                                        )
                                    )
                            elif item == '_':
                                # Wildcard import
                                imports.append(
                                    ImportInfo(
                                        module=f"{base_path}_",
                                        line=i,
                                        type=import_type,
                                        is_relative=False,
                                        is_wildcard=True,
                                        category=self._categorize_import(base_path),
                                    )
                                )
                            else:
                                # Regular import
                                imports.append(
                                    ImportInfo(
                                        module=f"{base_path}{item}",
                                        line=i,
                                        type=import_type,
                                        is_relative=False,
                                        category=self._categorize_import(base_path),
                                    )
                                )
                else:
                    # Single import (possibly with wildcard)
                    if import_str.endswith('._'):
                        # Wildcard import
                        base_path = import_str[:-2]
                        imports.append(
                            ImportInfo(
                                module=import_str,
                                line=i,
                                type=import_type,
                                is_relative=False,
                                is_wildcard=True,
                                category=self._categorize_import(base_path),
                            )
                        )
                    else:
                        # Regular single import
                        imports.append(
                            ImportInfo(
                                module=import_str,
                                line=i,
                                type=import_type,
                                is_relative=False,
                                is_given=is_given_import,
                                category=self._categorize_import(import_str),
                            )
                        )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from Scala code.

        In Scala, exports include:
        - Public classes and case classes
        - Public traits
        - Public objects (including companion objects)
        - Public defs (methods/functions)
        - Public vals and vars
        - Public type definitions
        - Implicit definitions
        - Given instances (Scala 3)

        Args:
            content: Scala source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols
        """
        exports = []

        # Classes and case classes
        class_pattern = r'^\s*(?:(sealed|abstract|final)\s+)?(?:(case)\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            modifiers = []
            if match.group(1):
                modifiers.append(match.group(1))
            if match.group(2):
                modifiers.append("case")
            
            exports.append(
                {
                    "name": match.group(3),
                    "type": "case_class" if "case" in modifiers else "class",
                    "line": content[:match.start()].count("\n") + 1,
                    "modifiers": modifiers,
                    "is_public": not match.group(3).startswith("_"),
                }
            )

        # Traits
        trait_pattern = r'^\s*(?:(sealed)\s+)?trait\s+(\w+)'
        for match in re.finditer(trait_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(2),
                    "type": "trait",
                    "line": content[:match.start()].count("\n") + 1,
                    "is_sealed": match.group(1) == "sealed",
                    "is_public": not match.group(2).startswith("_"),
                }
            )

        # Objects
        object_pattern = r'^\s*(?:(case)\s+)?object\s+(\w+)'
        for match in re.finditer(object_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(2),
                    "type": "case_object" if match.group(1) else "object",
                    "line": content[:match.start()].count("\n") + 1,
                    "is_public": not match.group(2).startswith("_"),
                }
            )

        # Functions/Methods
        def_pattern = r'^\s*(?:(override|implicit|inline|transparent)\s+)?(?:(private|protected)\s+)?def\s+(\w+)'
        for match in re.finditer(def_pattern, content, re.MULTILINE):
            visibility = match.group(2)
            if visibility != "private":
                func_name = match.group(3)
                exports.append(
                    {
                        "name": func_name,
                        "type": "function",
                        "line": content[:match.start()].count("\n") + 1,
                        "is_implicit": match.group(1) == "implicit",
                        "is_override": match.group(1) == "override",
                        "visibility": visibility or "public",
                        "is_public": not func_name.startswith("_") and visibility != "protected",
                    }
                )

        # Values and variables
        val_var_pattern = r'^\s*(?:(implicit|lazy)\s+)?(?:(private|protected)\s+)?(?:(val|var)\s+)(\w+)'
        for match in re.finditer(val_var_pattern, content, re.MULTILINE):
            visibility = match.group(2)
            if visibility != "private":
                var_name = match.group(4)
                exports.append(
                    {
                        "name": var_name,
                        "type": "variable" if match.group(3) == "var" else "value",
                        "line": content[:match.start()].count("\n") + 1,
                        "is_implicit": match.group(1) == "implicit",
                        "is_lazy": match.group(1) == "lazy",
                        "is_mutable": match.group(3) == "var",
                        "visibility": visibility or "public",
                        "is_public": not var_name.startswith("_") and visibility != "protected",
                    }
                )

        # Type definitions
        type_pattern = r'^\s*(?:(opaque)\s+)?type\s+(\w+)'
        for match in re.finditer(type_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(2),
                    "type": "type_alias",
                    "line": content[:match.start()].count("\n") + 1,
                    "is_opaque": match.group(1) == "opaque",
                    "is_public": True,
                }
            )

        # Enums (Scala 3)
        enum_pattern = r'^\s*enum\s+(\w+)'
        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "enum",
                    "line": content[:match.start()].count("\n") + 1,
                    "is_public": True,
                    "scala_version": 3,
                }
            )

        # Given instances (Scala 3)
        given_pattern = r'^\s*given\s+(?:(\w+)\s*:\s*)?(\w+)'
        for match in re.finditer(given_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1) or f"given_{match.group(2)}",
                    "type": "given",
                    "line": content[:match.start()].count("\n") + 1,
                    "given_type": match.group(2),
                    "is_public": True,
                    "scala_version": 3,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Scala file.

        Extracts:
        - Classes, traits, and objects
        - Case classes and algebraic data types
        - Methods with type parameters
        - Pattern matching constructs
        - For comprehensions
        - Implicit definitions
        - Companion objects
        - Extension methods (Scala 3)
        - Given/using (Scala 3)

        Args:
            content: Scala source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Detect Scala version (3.x has different syntax)
        structure.scala_version = 3 if self._is_scala3(content) else 2

        # Extract package
        package_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
        if package_match:
            structure.package = package_match.group(1)

        # Extract classes
        class_pattern = r'''
            ^\s*(?:(sealed|abstract|final)\s+)?
            (?:(case)\s+)?
            class\s+(\w+)
            (?:\[([^\]]+)\])?  # Type parameters
            (?:\s*\(([^)]*)\))?  # Primary constructor
            (?:\s+extends\s+([^{]+?))?
            (?:\s+with\s+([^{]+?))?
            (?:\s*\{|\s*$)
        '''
        
        for match in re.finditer(class_pattern, content, re.VERBOSE | re.MULTILINE):
            class_name = match.group(3)
            
            modifiers = []
            if match.group(1):
                modifiers.append(match.group(1))
            if match.group(2):
                modifiers.append("case")
            
            type_params = match.group(4)
            constructor_params = match.group(5)
            extends = match.group(6)
            with_traits = match.group(7)
            
            # Extract class body
            class_body = self._extract_body(content, match.end())
            
            if class_body:
                methods = self._extract_methods(class_body)
                fields = self._extract_fields(class_body)
            else:
                methods = []
                fields = []
            
            # Check for companion object
            companion_pattern = rf'^\s*object\s+{class_name}\s*\{{'
            has_companion = bool(re.search(companion_pattern, content, re.MULTILINE))
            
            class_info = ClassInfo(
                name=class_name,
                line=content[:match.start()].count("\n") + 1,
                modifiers=modifiers,
                type_parameters=type_params,
                constructor_params=self._parse_parameters(constructor_params) if constructor_params else [],
                bases=[extends.strip()] if extends else [],
                mixins=self._parse_with_traits(with_traits) if with_traits else [],
                methods=methods,
                fields=fields,
                has_companion=has_companion,
                is_case_class="case" in modifiers,
                is_sealed="sealed" in modifiers,
            )
            
            structure.classes.append(class_info)

        # Extract traits
        trait_pattern = r'''
            ^\s*(?:(sealed)\s+)?
            trait\s+(\w+)
            (?:\[([^\]]+)\])?  # Type parameters
            (?:\s+extends\s+([^{]+?))?
            (?:\s+with\s+([^{]+?))?
            (?:\s*\{|\s*$)
        '''
        
        for match in re.finditer(trait_pattern, content, re.VERBOSE | re.MULTILINE):
            trait_name = match.group(2)
            trait_body = self._extract_body(content, match.end())
            
            structure.traits.append(
                {
                    "name": trait_name,
                    "line": content[:match.start()].count("\n") + 1,
                    "is_sealed": match.group(1) == "sealed",
                    "type_parameters": match.group(3),
                    "extends": match.group(4).strip() if match.group(4) else None,
                    "with_traits": self._parse_with_traits(match.group(5)) if match.group(5) else [],
                    "methods": self._extract_methods(trait_body) if trait_body else [],
                }
            )

        # Extract objects (including package objects) with bodies
        object_pattern_with_body = r'^\s*(?:(case)\s+)?(?:(package)\s+)?object\s+(\w+)(?:\s+extends\s+([^\{\n]+?))?(?:\s+with\s+([^\{\n]+?))?\s*\{'
        for match in re.finditer(object_pattern_with_body, content, re.MULTILINE):
            object_name = match.group(3)
            object_body = self._extract_body(content, match.end())
            structure.objects.append(
                {
                    "name": object_name,
                    "line": content[:match.start()].count("\n") + 1,
                    "is_case_object": match.group(1) == "case",
                    "is_package_object": match.group(2) == "package",
                    "extends": match.group(4).strip() if match.group(4) else None,
                    "with_traits": self._parse_with_traits(match.group(5)) if match.group(5) else [],
                    "methods": self._extract_methods(object_body) if object_body else [],
                    "is_companion": any(c.name == object_name for c in structure.classes),
                }
            )

        # Body-less objects (e.g., case object Empty extends Something) - ensure not already captured
        object_pattern_no_body = r'^\s*(?:(case)\s+)?object\s+(\w+)(?:\s+extends\s+([^\{\n]+?))?(?:\s+with\s+([^\{\n]+?))?\s*(?:$|//|/\*)'
        for match in re.finditer(object_pattern_no_body, content, re.MULTILINE):
            name = match.group(2)
            if any(o["name"] == name for o in structure.objects):
                continue
            structure.objects.append(
                {
                    "name": name,
                    "line": content[:match.start()].count("\n") + 1,
                    "is_case_object": match.group(1) == "case",
                    "is_package_object": False,
                    "extends": match.group(3).strip() if match.group(3) else None,
                    "with_traits": self._parse_with_traits(match.group(4)) if match.group(4) else [],
                    "methods": [],
                    "is_companion": any(c.name == name for c in structure.classes),
                }
            )

        # Extract top-level functions
        func_pattern = r'''
            ^\s*(?:(implicit|inline|transparent)\s+)?
            def\s+(\w+)
            (?:\[([^\]]+)\])?  # Type parameters
            (\([^)]*\)(?:\s*\([^)]*\))*)  # Parameters (possibly curried)
            (?:\s*:\s*([^=\n{]+))?  # Return type
            \s*(?:=|{)
        '''
        
        for match in re.finditer(func_pattern, content, re.VERBOSE | re.MULTILINE):
            if not self._is_inside_class_or_object(content, match.start()):
                func_name = match.group(2)
                type_params = match.group(3)
                params = match.group(4)
                return_type = match.group(5)
                
                # Check if it's curried (multiple parameter lists)
                is_curried = params.count('(') > 1
                
                func_info = FunctionInfo(
                    name=func_name,
                    line=content[:match.start()].count("\n") + 1,
                    type_parameters=type_params,
                    parameters=self._parse_curried_parameters(params),
                    return_type=return_type.strip() if return_type else None,
                    is_implicit=match.group(1) == "implicit",
                    is_inline=match.group(1) == "inline",
                    is_curried=is_curried,
                )
                
                structure.functions.append(func_info)

        # Extract enums (Scala 3)
        enum_pattern = r'^\s*enum\s+(\w+)(?:\[([^\]]+)\])?\s*(?::\s*([^:{]+))?[\s:]*'
        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            # For Scala 3 enums with colon syntax, extract body manually
            if ':' in content[match.end()-2:match.end()]:
                enum_body = self._extract_indented_body(content, match.end())
            else:
                enum_body = self._extract_body(content, match.end())
            structure.enums.append(
                {
                    "name": match.group(1),
                    "line": content[:match.start()].count("\n") + 1,
                    "type_parameters": match.group(2),
                    "extends": match.group(3),
                    "cases": self._extract_enum_cases(enum_body) if enum_body else [],
                }
            )

        # Count pattern matching
        structure.match_expressions = len(re.findall(r'\bmatch\s*\{', content))
        structure.case_statements = len(re.findall(r'\bcase\s+', content))

        # Count for comprehensions
        structure.for_comprehensions = len(re.findall(r'\bfor\s*\{|\bfor\s*\(', content))
        structure.yield_expressions = len(re.findall(r'\byield\s+', content))

        # Count implicit definitions
        structure.implicit_defs = len(re.findall(r'\bimplicit\s+(?:def|val|class)', content))
        # Count implicit parameters in all contexts
        structure.implicit_params = (
            len(re.findall(r'\)\s*\(\s*implicit', content)) +  # Method implicit params
            len(re.findall(r'\(\s*implicit\s+', content))      # Constructor implicit params
        )

        # Count higher-order functions
        structure.lambda_expressions = len(re.findall(r'=>', content))
        structure.partial_functions = len(re.findall(r'\bPartialFunction\[', content))

        # Scala 3 specific
        if structure.scala_version == 3:
            structure.given_instances = len(re.findall(r'\bgiven\s+', content))
            structure.using_clauses = len(re.findall(r'\busing\s+', content))
            structure.extension_methods = len(re.findall(r'\bextension\s*\(', content))

        # Detect test file
        structure.is_test_file = (
            "test" in file_path.name.lower()
            or "spec" in file_path.name.lower()
            or any(part in ["test", "spec"] for part in file_path.parts)
        )

        # Detect main method/app
        structure.has_main = bool(re.search(r'def\s+main\s*\(|extends\s+App\b', content))

        return structure
    
    def _extract_indented_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract indented body content (for Scala 3 syntax).
        
        Args:
            content: Source code
            start_pos: Position after the colon
            
        Returns:
            Indented body content or None
        """
        lines = content[start_pos:].split('\n')
        if not lines:
            return None
            
        body_lines = []
        base_indent = None
        
        for line in lines:
            if not line.strip():
                continue
                
            indent = len(line) - len(line.lstrip())
            
            if base_indent is None:
                if line.strip():
                    base_indent = indent
                    body_lines.append(line)
            elif indent >= base_indent and line.strip():
                body_lines.append(line)
            elif line.strip():
                # Indentation decreased, end of body
                break
                
        return '\n'.join(body_lines) if body_lines else None

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Scala code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Pattern matching complexity
        - Functional programming complexity
        - Type complexity
        - Implicit complexity

        Args:
            content: Scala source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Calculate cyclomatic complexity
        complexity = 1

        decision_keywords = [
            r'\bif\b',
            r'\belse\s+if\b',
            r'\belse\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\bdo\b',
            r'\bmatch\b',
            r'\bcase\b',
            r'\btry\b',
            r'\bcatch\b',
            r'&&',
            r'\|\|',
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
                (r'\bif\b', 1),
                (r'\belse\s+if\b', 1),
                (r'\belse\b', 0),
                (r'\bfor\b', 1),
                (r'\bwhile\b', 1),
                (r'\bdo\b', 1),
                (r'\bmatch\b', 2),  # Pattern matching is more complex
                (r'\btry\b', 1),
            ]

            for pattern, weight in control_patterns:
                if re.search(pattern, line):
                    cognitive += weight * (1 + max(0, nesting_level - 1))

        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting

        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('//')])
        metrics.comment_lines = len([l for l in lines if l.strip().startswith('//')])
        metrics.comment_ratio = metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0

        # Count classes and traits
        metrics.class_count = len(re.findall(r'\bclass\s+\w+', content))
        metrics.trait_count = len(re.findall(r'\btrait\s+\w+', content))
        metrics.object_count = len(re.findall(r'\bobject\s+\w+', content))
        metrics.case_class_count = len(re.findall(r'\bcase\s+class\s+\w+', content))

        # Pattern matching metrics
        metrics.match_expressions = len(re.findall(r'\bmatch\s*\{', content))
        metrics.case_clauses = len(re.findall(r'\bcase\s+', content))
        metrics.pattern_guards = len(re.findall(r'\bcase\s+.*\s+if\s+', content))

        # Functional programming metrics
        # Count lambda expressions more accurately
        lambda_arrows = len(re.findall(r'=>', content))
        # Also count underscore lambdas like _ * 2, _ + _
        underscore_lambdas = len(re.findall(r'_\s*[+\-*/]|[+\-*/]\s*_', content))
        metrics.lambda_count = lambda_arrows + underscore_lambdas
        
        metrics.higher_order_functions = len(re.findall(r'\.(?:map|flatMap|filter|fold|reduce|collect|foreach|exists|forall|find|zip|groupBy|sortBy|distinct|take|drop)\s*\(', content))
        metrics.for_comprehensions = len(re.findall(r'\bfor\s*\{|\bfor\s*\(', content))
        metrics.partial_functions = len(re.findall(r'\bPartialFunction\[', content))

        # Type system metrics
        metrics.type_parameters = len(re.findall(r'\[[\w\s,:<>]+\]', content))
        metrics.variance_annotations = len(re.findall(r'[+-]\w+', content))
        # Count type aliases - both 'type X = Y' and abstract type members
        metrics.type_aliases = (
            len(re.findall(r'\btype\s+\w+\s*=', content)) +  # Concrete type aliases
            len(re.findall(r'\btype\s+\w+\b(?!\s*=)', content))  # Abstract type members
        )
        metrics.existential_types = len(re.findall(r'forSome\s*\{', content))

        # Implicit metrics
        metrics.implicit_defs = len(re.findall(r'\bimplicit\s+(?:def|val|var|class)', content))
        metrics.implicit_params = (
            len(re.findall(r'\)\s*\(\s*implicit', content)) +  # Method implicit params
            len(re.findall(r'\(\s*implicit\s+', content))      # Constructor implicit params
        )
        metrics.implicit_conversions = len(re.findall(r'implicit\s+def\s+\w+\([^)]*\):\s*\w+', content))

        # Concurrency metrics
        metrics.future_usage = len(re.findall(r'\bFuture\[|\bFuture\s*\{', content))
        metrics.actor_usage = len(re.findall(r'\bActor\b|\bActorRef\b', content))
        metrics.async_await = len(re.findall(r'\basync\s*\{|\bawait\b', content))

        # Collections metrics
        metrics.immutable_collections = len(re.findall(r'\b(?:List|Vector|Set|Map|Seq)(?:\[|\(|\.)', content))
        metrics.mutable_collections = len(re.findall(r'mutable\.(?:ListBuffer|ArrayBuffer|Set|Map)', content))

        # Exception handling
        metrics.try_blocks = len(re.findall(r'\btry\s*\{', content))
        metrics.catch_blocks = len(re.findall(r'\bcatch\s*\{', content))
        metrics.finally_blocks = len(re.findall(r'\bfinally\s*\{', content))
        metrics.throw_statements = len(re.findall(r'\bthrow\s+', content))
        metrics.option_usage = len(re.findall(r'\bOption\[|\bSome\(|\bNone\b', content))
        metrics.either_usage = len(re.findall(r'\bEither\[|\bLeft\(|\bRight\(', content))
        metrics.try_usage = len(re.findall(r'\bTry\[|\bSuccess\(|\bFailure\(', content))

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Scala
            functional_factor = 1 + (metrics.lambda_count * 0.001)
            implicit_factor = 1 - (metrics.implicit_conversions * 0.02)
            pattern_factor = 1 - (metrics.case_clauses * 0.001)
            type_safety_factor = 1 + (metrics.option_usage + metrics.either_usage) * 0.001

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * functional_factor
                + 5 * implicit_factor
                + 5 * pattern_factor
                + 5 * type_safety_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _categorize_import(self, module_path: str) -> str:
        """Categorize a Scala import.

        Args:
            module_path: Import path

        Returns:
            Category string
        """
        if module_path.startswith("scala."):
            if module_path.startswith("scala.collection"):
                return "scala_collections"
            elif module_path.startswith("scala.concurrent"):
                return "scala_concurrent"
            elif module_path.startswith("scala.util"):
                return "scala_util"
            else:
                return "scala_core"
        elif module_path.startswith("java."):
            return "java"
        elif module_path.startswith("akka."):
            return "akka"
        elif module_path.startswith("cats."):
            return "cats"
        elif module_path.startswith("zio."):
            return "zio"
        elif module_path.startswith("play."):
            return "play"
        elif module_path.startswith("slick."):
            return "slick"
        elif module_path.startswith("org.scalatest"):
            return "scalatest"
        elif module_path.startswith("org.specs2"):
            return "specs2"
        else:
            return "third_party"

    def _is_scala3(self, content: str) -> bool:
        """Check if the code uses Scala 3 syntax.

        Args:
            content: Scala source code

        Returns:
            True if Scala 3 syntax is detected
        """
        scala3_indicators = [
            r'\bgiven\s+',
            r'\busing\s+',
            r'\bextension\s*\(',
            r'\benum\s+\w+',
            r'\bthen\s*$',  # Scala 3 if-then
            r'\bdo\s*$',  # Scala 3 while-do
            r'\bopaque\s+type',
            r'\binfix\s+',
            r'\btransparent\s+',
            r'\bopen\s+class',
        ]

        return any(re.search(pattern, content, re.MULTILINE) for pattern in scala3_indicators)

    def _extract_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a class/trait/object.

        Args:
            content: Source code
            start_pos: Position after opening brace

        Returns:
            Body content or None
        """
        brace_count = 1
        pos = start_pos
        in_string = False
        in_multiline_string = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            if not escape_next:
                # Check for multiline strings
                if content[pos:pos+3] == '"""':
                    in_multiline_string = not in_multiline_string
                    pos += 2
                elif char == '"' and not in_multiline_string:
                    in_string = not in_string
                elif char == '\\':
                    escape_next = True
                elif not in_string and not in_multiline_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
            else:
                escape_next = False

            pos += 1

        if brace_count == 0:
            return content[start_pos:pos - 1]

        return None

    def _extract_methods(self, body: str) -> List[Dict[str, Any]]:
        """Extract methods from class/trait/object body.

        Args:
            body: Body content

        Returns:
            List of method information
        """
        methods = []

        # Match both block-bodied and expression-bodied methods (including symbolic and unary operators)
        # Method pattern (no mandatory blank line before). Captures symbolic, backticked, unary_ operators.
        method_pattern = r'''\s*
            (?:(override|private|protected|final|abstract|implicit)\s+)*
            def\s+((?:[\w$]+|`[^`]+`|::|[+\-*/<>=!&|:~]+|unary_[+\-!~]))
            (?:\[([^\]]+)\])?                    # Type parameters
            (\([^)]*\)(?:\s*\([^)]*\))*)?        # Parameter list(s) optional (operators without params)
            (?:\s*:\s*([^=\n{]+))?                # Return type
            \s*(?:=|\{)                            # Body start (expression or block)
        '''

        for match in re.finditer(method_pattern, body, re.MULTILINE | re.VERBOSE):
            modifiers_raw = match.group(1) or ""
            modifiers = [m for m in modifiers_raw.split() if m]
            method_name = match.group(2)
            type_params = match.group(3)
            params = match.group(4) or "()"  # Treat missing param list as empty
            return_type = match.group(5)

            methods.append(
                {
                    "name": method_name,
                    "modifiers": modifiers,
                    "type_parameters": type_params,
                    "parameters": self._parse_curried_parameters(params),
                    "return_type": return_type.strip() if return_type else None,
                    "is_abstract": "abstract" in modifiers,
                    "is_override": "override" in modifiers,
                    "is_implicit": "implicit" in modifiers,
                    "line": body[:match.start()].count("\n") + 1,
                }
            )

        # Fallback: capture '::' operator if not matched (some edge layouts may confuse verbose spacing)
        if not any(m["name"] == "::" for m in methods):
            op_pattern = r'^\s*def\s+::\s*(\([^)]*\))?(?:\s*:\s*([^=\n{]+))?\s*(?:=|\{)'
            for m in re.finditer(op_pattern, body, re.MULTILINE):
                params = m.group(1) or "()"
                return_type = m.group(2).strip() if m.group(2) else None
                methods.append(
                    {
                        "name": "::",
                        "modifiers": [],
                        "type_parameters": None,
                        "parameters": self._parse_curried_parameters(params),
                        "return_type": return_type,
                        "is_abstract": False,
                        "is_override": False,
                        "is_implicit": False,
                        "line": body[:m.start()].count("\n") + 1,
                    }
                )

        # Fallback explicit capture for '::' operator if still missing
        if not any(m.get("name") == "::" for m in methods):
            m = re.search(r'^\s*def\s+::\s*(\([^)]*\))(?:\s*:\s*([^=\n{]+))?\s*(?:=|\{)', body, re.MULTILINE)
            if m:
                params = m.group(1)
                return_type = m.group(2).strip() if m.group(2) else None
                methods.append(
                    {
                        "name": "::",
                        "modifiers": [],
                        "type_parameters": None,
                        "parameters": self._parse_curried_parameters(params),
                        "return_type": return_type,
                        "is_abstract": False,
                        "is_override": False,
                        "is_implicit": False,
                        "line": body[:m.start()].count('\n') + 1,
                    }
                )

        return methods


    def _extract_fields(self, body: str) -> List[Dict[str, Any]]:
        """Extract fields from class body.

        Args:
            body: Body content

        Returns:
            List of field information
        """
        fields = []

        field_pattern = r'''
            (?:^|\n)\s*
            (?:(private|protected|override|final|lazy|implicit)\s+)*
            (val|var)\s+
            (\w+)\s*:\s*([^=\n]+)
        '''

        for match in re.finditer(field_pattern, body, re.VERBOSE):
            modifiers = match.group(1).split() if match.group(1) else []
            field_type = match.group(2)
            field_name = match.group(3)
            type_annotation = match.group(4)

            fields.append(
                {
                    "name": field_name,
                    "type": type_annotation.strip(),
                    "is_mutable": field_type == "var",
                    "modifiers": modifiers,
                    "is_lazy": "lazy" in modifiers,
                    "is_implicit": "implicit" in modifiers,
                    "line": body[:match.start()].count("\n") + 1,
                }
            )

        return fields

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse constructor/method parameters.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter dictionaries
        """
        if not params_str:
            return []

        parameters = []
        params = self._split_parameters(params_str)

        for param in params:
            param = param.strip()
            if not param:
                continue

            param_dict = {}

            # Check for modifiers (val, var, implicit)
            if param.startswith("val "):
                param_dict["is_val"] = True
                param = param[4:]
            elif param.startswith("var "):
                param_dict["is_var"] = True
                param = param[4:]
            elif param.startswith("implicit "):
                param_dict["is_implicit"] = True
                param = param[9:]

            # Parse name and type
            if ":" in param:
                parts = param.split(":", 1)
                param_dict["name"] = parts[0].strip()
                
                # Check for default value
                if "=" in parts[1]:
                    type_parts = parts[1].split("=", 1)
                    param_dict["type"] = type_parts[0].strip()
                    param_dict["default"] = type_parts[1].strip()
                else:
                    param_dict["type"] = parts[1].strip()
            else:
                param_dict["name"] = param

            parameters.append(param_dict)

        return parameters

    def _parse_curried_parameters(self, params_str: str) -> List[List[Dict[str, Any]]]:
        """Parse curried parameters.

        Args:
            params_str: Curried parameter string

        Returns:
            List of parameter groups
        """
        groups = []
        current_group = ""
        paren_depth = 0

        for char in params_str:
            if char == '(':
                if paren_depth == 0 and current_group:
                    # Start of new group
                    groups.append(self._parse_parameters(current_group))
                    current_group = ""
                else:
                    current_group += char
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    # End of group
                    continue
                else:
                    current_group += char
            else:
                current_group += char

        if current_group:
            groups.append(self._parse_parameters(current_group))

        return groups if len(groups) > 1 else (groups[0] if groups else [])

    def _split_parameters(self, params_str: str) -> List[str]:
        """Split parameters handling nested types.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter strings
        """
        params = []
        current = ""
        depth = 0

        for char in params_str:
            if char in "[(<":
                depth += 1
            elif char in "])>":
                depth -= 1
            elif char == "," and depth == 0:
                if current.strip():
                    params.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            params.append(current.strip())

        return params

    def _parse_with_traits(self, traits_str: str) -> List[str]:
        """Parse 'with' traits handling type parameters.

        Args:
            traits_str: Traits string

        Returns:
            List of trait names
        """
        return self._split_parameters(traits_str.replace(" with ", ","))

    def _is_inside_class_or_object(self, content: str, position: int) -> bool:
        """Check if a position is inside a class or object.

        Args:
            content: Source code
            position: Position to check

        Returns:
            True if inside a class or object
        """
        before = content[:position]
        
        # Count class/object/trait openings and closings
        class_opens = len(re.findall(r'\b(?:class|trait|object)\s+\w+[^{]*\{', before))
        
        # Count matching closing braces (approximation)
        brace_count = before.count('{') - before.count('}')
        
        return brace_count > 0 and class_opens > 0

    def _extract_enum_cases(self, enum_body: str) -> List[Dict[str, Any]]:
        """Extract enum cases from Scala 3 enum body.

        Args:
            enum_body: Enum body content

        Returns:
            List of enum case information
        """
        cases = []

        case_pattern = r'case\s+(\w+)(?:\(([^)]*)\))?'
        
        for match in re.finditer(case_pattern, enum_body):
            cases.append(
                {
                    "name": match.group(1),
                    "parameters": self._parse_parameters(match.group(2)) if match.group(2) else [],
                }
            )

        return cases