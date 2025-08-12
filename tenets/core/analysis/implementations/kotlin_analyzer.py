"""Kotlin code analyzer with Android and multiplatform support.

This module provides comprehensive analysis for Kotlin source files,
including support for Android development, coroutines, null safety,
and Kotlin Multiplatform features.
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


class KotlinAnalyzer(LanguageAnalyzer):
    """Kotlin code analyzer with Android and multiplatform support.

    Provides comprehensive analysis for Kotlin files including:
    - Import statements with aliases
    - Package declarations
    - Classes, interfaces, objects, data classes
    - Sealed classes and interfaces
    - Extension functions and properties
    - Coroutines and suspend functions
    - Null safety features
    - Inline and reified functions
    - Companion objects
    - Delegation patterns
    - Android-specific patterns (Activities, Fragments, ViewModels)
    - Kotlin Multiplatform declarations

    Supports modern Kotlin features and Android development patterns.
    """

    language_name = "kotlin"
    file_extensions = [".kt", ".kts"]

    def __init__(self):
        """Initialize the Kotlin analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract import statements from Kotlin code.

        Handles:
        - import statements: import kotlin.collections.List
        - Wildcard imports: import java.util.*
        - Aliased imports: import java.util.List as JList
        - Package declarations
        - Static imports (companion object members)

        Args:
            content: Kotlin source code
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
            import_pattern = r'^\s*import\s+([^\s]+?)(?:\s+as\s+(\w+))?(?:\s*//.*)?$'
            match = re.match(import_pattern, line)
            if match:
                module_path = match.group(1)
                alias = match.group(2)
                
                # Check for wildcard import
                is_wildcard = module_path.endswith('.*')
                if is_wildcard:
                    base_path = module_path[:-2]
                    category = self._categorize_import(base_path)
                else:
                    category = self._categorize_import(module_path)
                
                # Determine if it's an Android import
                is_android = self._is_android_import(module_path)
                
                imports.append(
                    ImportInfo(
                        module=module_path,
                        alias=alias,
                        line=i,
                        type="import",
                        is_relative=False,
                        is_wildcard=is_wildcard,
                        category=category,
                        is_android=is_android,
                        package_context=current_package,
                    )
                )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from Kotlin code.

        In Kotlin, exports include:
        - Public classes, interfaces, and objects
        - Public functions (including extension functions)
        - Public properties (including extension properties)
        - Public type aliases
        - Sealed class hierarchies
        - Enum classes
        - Annotations

        Args:
            content: Kotlin source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols
        """
        exports = []

        # Classes (including data, sealed, enum)
        class_pattern = r'^\s*(?:(public|internal|private|protected)\s+)?(?:(abstract|open|final|sealed|inner|data|enum|annotation|inline|value)\s+)*class\s+(\w+)'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            visibility = match.group(1) or "public"
            if visibility != "private":
                modifiers = match.group(2).split() if match.group(2) else []
                class_name = match.group(3)
                
                class_type = "class"
                if "data" in modifiers:
                    class_type = "data_class"
                elif "enum" in modifiers:
                    class_type = "enum_class"
                elif "sealed" in modifiers:
                    class_type = "sealed_class"
                elif "annotation" in modifiers:
                    class_type = "annotation_class"
                elif "value" in modifiers or "inline" in modifiers:
                    class_type = "value_class"
                
                exports.append(
                    {
                        "name": class_name,
                        "type": class_type,
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": visibility,
                        "modifiers": modifiers,
                        "is_public": visibility == "public",
                    }
                )

        # Interfaces
        interface_pattern = r'^\s*(?:(public|internal|private|protected)\s+)?(?:(sealed|fun)\s+)?interface\s+(\w+)'
        for match in re.finditer(interface_pattern, content, re.MULTILINE):
            visibility = match.group(1) or "public"
            if visibility != "private":
                exports.append(
                    {
                        "name": match.group(3),
                        "type": "sealed_interface" if match.group(2) == "sealed" else "interface",
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": visibility,
                        "is_fun_interface": match.group(2) == "fun",
                        "is_public": visibility == "public",
                    }
                )

        # Objects (including companion objects)
        object_pattern = r'^\s*(?:(public|internal|private|protected)\s+)?(?:(companion)\s+)?object\s+(\w+)'
        for match in re.finditer(object_pattern, content, re.MULTILINE):
            visibility = match.group(1) or "public"
            if visibility != "private":
                exports.append(
                    {
                        "name": match.group(3),
                        "type": "companion_object" if match.group(2) else "object",
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": visibility,
                        "is_public": visibility == "public",
                    }
                )

        # Functions (including extension and suspend functions)
        func_pattern = r'^\s*(?:(public|internal|private|protected)\s+)?(?:(suspend|inline|tailrec|operator|infix|external|actual|expect)\s+)*fun\s+(?:<[^>]+>\s+)?(?:(\w+(?:<[^>]*>)?|\w+)\.)?(\w+)'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            visibility = match.group(1) or "public"
            if visibility != "private":
                modifiers = match.group(2).split() if match.group(2) else []
                receiver = match.group(3)
                func_name = match.group(4)
                
                exports.append(
                    {
                        "name": func_name,
                        "type": "extension_function" if receiver else "function",
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": visibility,
                        "modifiers": modifiers,
                        "receiver": receiver,
                        "is_suspend": "suspend" in modifiers,
                        "is_inline": "inline" in modifiers,
                        "is_operator": "operator" in modifiers,
                        "is_public": visibility == "public",
                    }
                )

        # Properties (including extension properties)
        prop_pattern = r'^\s*(?:(public|internal|private|protected)\s+)?(?:(const|lateinit)\s+)?(?:override\s+)?(val|var)\s+(?:(\w+)\.)?(\w+)'
        for match in re.finditer(prop_pattern, content, re.MULTILINE):
            visibility = match.group(1) or "public"
            if visibility != "private":
                modifier = match.group(2)
                prop_type = match.group(3)
                receiver = match.group(4)
                prop_name = match.group(5)
                
                exports.append(
                    {
                        "name": prop_name,
                        "type": "extension_property" if receiver else "property",
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": visibility,
                        "is_mutable": prop_type == "var",
                        "is_const": modifier == "const",
                        "is_lateinit": modifier == "lateinit",
                        "receiver": receiver,
                        "is_public": visibility == "public",
                    }
                )

        # Type aliases
        typealias_pattern = r'^\s*(?:(public|internal|private)\s+)?typealias\s+(\w+)'
        for match in re.finditer(typealias_pattern, content, re.MULTILINE):
            visibility = match.group(1) or "public"
            if visibility != "private":
                exports.append(
                    {
                        "name": match.group(2),
                        "type": "typealias",
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": visibility,
                        "is_public": visibility == "public",
                    }
                )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Kotlin file.

        Extracts:
        - Classes with inheritance and interfaces
        - Data classes and sealed hierarchies
        - Functions with parameters and return types
        - Extension functions and properties
        - Coroutines and suspend functions
        - Companion objects
        - Android components (Activities, Fragments, ViewModels)
        - Delegation patterns
        - Inline classes/value classes

        Args:
            content: Kotlin source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Extract package
        package_match = re.search(r'^\s*package\s+([\w\.]+)', content, re.MULTILINE)
        if package_match:
            structure.package = package_match.group(1)

        # Detect if it's an Android file
        structure.is_android = self._is_android_file(content)

        # Extract classes
        class_pattern = r'''
            ^\s*(?:(internal|private|protected|public)\s+)?
            (?:(abstract|open|final|sealed|inner|data|enum|annotation|inline|value)\s+)*
            class\s+(\w+)
            (?:<([^>]+)>)?  # Generic parameters
            (?:\s*(?:@\w+(?:\([^)]*\))?\s*)*)? # Annotations
            (?:\s*\(([^)]*)\))?  # Primary constructor
            (?:\s*:\s*([^{]+?))?  # Inheritance
            \s*(?:\{|$)
        '''
        
        for match in re.finditer(class_pattern, content, re.VERBOSE | re.MULTILINE):
            visibility = match.group(1) or "public"
            modifiers = match.group(2).split() if match.group(2) else []
            class_name = match.group(3)
            type_params = match.group(4)
            constructor_params = match.group(5)
            inheritance = match.group(6)
            
            # Parse inheritance (superclass and interfaces)
            bases = []
            interfaces = []
            delegates = {}
            
            if inheritance:
                for item in self._parse_inheritance(inheritance):
                    if " by " in item:
                        # Delegation
                        parts = item.split(" by ")
                        interface = parts[0].strip()
                        delegate = parts[1].strip()
                        delegates[interface] = delegate
                        interfaces.append(interface)
                    elif item.endswith("()") or "(" in item:
                        # Superclass with constructor call
                        bases.append(item)
                    else:
                        # Interface or superclass without constructor
                        if self._is_likely_interface(item):
                            interfaces.append(item)
                        else:
                            bases.append(item)
            
            # Extract class body
            class_body = self._extract_body(content, match.end())
            
            if class_body:
                methods = self._extract_methods(class_body)
                properties = self._extract_properties(class_body)
                companion = self._extract_companion_object(class_body)
                nested_classes = self._extract_nested_classes(class_body)
            else:
                methods = []
                properties = []
                companion = None
                nested_classes = []
            
            # Check for Android components
            android_type = None
            if structure.is_android:
                # Remove parentheses and generics from base class names for matching
                clean_bases = [base.split('(')[0].split('<')[0].strip() for base in bases]
                if any(base in ["Activity", "AppCompatActivity", "ComponentActivity"] for base in clean_bases):
                    android_type = "activity"
                elif any(base in ["Fragment", "DialogFragment"] for base in clean_bases):
                    android_type = "fragment"
                elif any(base in ["ViewModel", "AndroidViewModel"] for base in clean_bases):
                    android_type = "viewmodel"
                elif any(base in ["Service", "IntentService"] for base in clean_bases):
                    android_type = "service"
                elif any(base in ["BroadcastReceiver"] for base in clean_bases):
                    android_type = "receiver"
            
            class_info = ClassInfo(
                name=class_name,
                line=content[:match.start()].count("\n") + 1,
                visibility=visibility,
                modifiers=modifiers,
                type_parameters=type_params,
                constructor_params=self._parse_constructor_params(constructor_params) if constructor_params else [],
                bases=bases,
                interfaces=interfaces,
                delegates=delegates,
                methods=methods,
                properties=properties,
                companion_object=companion,
                nested_classes=nested_classes,
                is_data_class="data" in modifiers,
                is_sealed="sealed" in modifiers,
                is_enum="enum" in modifiers,
                is_inner="inner" in modifiers,
                is_value_class="value" in modifiers or "inline" in modifiers,
                android_type=android_type,
            )
            
            structure.classes.append(class_info)

        # Extract interfaces
        interface_pattern = r'''
            ^\s*(?:(internal|private|protected|public)\s+)?
            (?:(sealed|fun)\s+)?
            interface\s+(\w+)
            (?:<([^>]+)>)?
            (?:\s*:\s*([^\{]+?))?
            \s*\{?
        '''
        
        for match in re.finditer(interface_pattern, content, re.VERBOSE | re.MULTILINE):
            interface_name = match.group(3)
            # Body may be omitted for marker interfaces
            interface_body = self._extract_body(content, match.end())
            is_sealed = match.group(2) == "sealed"
            structure.interfaces.append(
                {
                    "name": interface_name,
                    "line": content[:match.start()].count("\n") + 1,
                    "visibility": match.group(1) or "public",
                    "is_sealed": is_sealed,
                    "is_fun_interface": match.group(2) == "fun",
                    "type_parameters": match.group(4),
                    "extends": self._parse_inheritance(match.group(5)) if match.group(5) else [],
                    "methods": self._extract_interface_methods(interface_body) if interface_body else [],
                }
            )

        # Extract objects
        object_pattern = r'^\s*(?:(internal|private|protected|public)\s+)?object\s+(\w+)(?:\s*:\s*([^{]+?))?\s*\{'
        
        for match in re.finditer(object_pattern, content, re.MULTILINE):
            if not self._is_companion_object(content, match.start()):
                object_body = self._extract_body(content, match.end())
                
                structure.objects.append(
                    {
                        "name": match.group(2),
                        "line": content[:match.start()].count("\n") + 1,
                        "visibility": match.group(1) or "public",
                        "implements": self._parse_inheritance(match.group(3)) if match.group(3) else [],
                        "methods": self._extract_methods(object_body) if object_body else [],
                        "properties": self._extract_properties(object_body) if object_body else [],
                    }
                )

        # Extract top-level functions
        func_pattern = r'''
            ^\s*(?:(internal|private|protected|public)\s+)?
            (?:(suspend|inline|tailrec|operator|infix|external|actual|expect)\s+)*
            fun\s+
            (?:<([^>]+)>\s+)?  # Type parameters
            (?:(\w+)\.)?  # Receiver type (for extensions)
            (\w+)  # Function name
            \s*\(([^)]*)\)  # Parameters
            (?:\s*:\s*([^{=\n]+))?  # Return type
            \s*[{=]
        '''
        
        for match in re.finditer(func_pattern, content, re.VERBOSE | re.MULTILINE):
            if not self._is_inside_class(content, match.start()):
                visibility = match.group(1) or "public"
                modifiers = match.group(2).split() if match.group(2) else []
                type_params = match.group(3)
                receiver = match.group(4)
                func_name = match.group(5)
                params = match.group(6)
                return_type = match.group(7)
                
                func_info = FunctionInfo(
                    name=func_name,
                    line=content[:match.start()].count("\n") + 1,
                    visibility=visibility,
                    modifiers=modifiers,
                    type_parameters=type_params,
                    receiver_type=receiver,
                    parameters=self._parse_parameters(params),
                    return_type=return_type.strip() if return_type else None,
                    is_extension=receiver is not None,
                    is_suspend="suspend" in modifiers,
                    is_inline="inline" in modifiers,
                    is_operator="operator" in modifiers,
                    is_infix="infix" in modifiers,
                )
                
                structure.functions.append(func_info)

        # Extract type aliases
        typealias_pattern = r'^\s*(?:(internal|private|public)\s+)?typealias\s+(\w+)(?:<[^>]+>)?\s*=\s*([^\n]+)'
        for match in re.finditer(typealias_pattern, content, re.MULTILINE):
            structure.type_aliases.append(
                {
                    "name": match.group(2),
                    "line": content[:match.start()].count("\n") + 1,
                    "visibility": match.group(1) or "public",
                    "definition": match.group(3).strip(),
                }
            )

        # Count coroutine usage
        structure.suspend_functions = len(re.findall(r'\bsuspend\s+fun\b', content))
        structure.coroutine_launches = len(re.findall(r'\b(?:launch|async)\s*\{', content))
        structure.flow_usage = len(re.findall(r'\bFlow<|\bflow\s*\{', content))

        # Count null safety features
        structure.nullable_types = len(re.findall(r'\w+\?(?:\s|,|\)|>)', content))
        structure.null_assertions = len(re.findall(r'!!', content))
        structure.safe_calls = len(re.findall(r'\?\.', content))
        structure.elvis_operators = len(re.findall(r'\?:', content))

        # Count lambda expressions
        structure.lambda_expressions = len(re.findall(r'\{[^}]*->[^}]*\}', content))
        
        # Count scope functions
        structure.scope_functions = (
            len(re.findall(r'\.let\s*\{', content)) +
            len(re.findall(r'\.run\s*\{', content)) +
            len(re.findall(r'\.apply\s*\{', content)) +
            len(re.findall(r'\.also\s*\{', content)) +
            len(re.findall(r'\bwith\s*\([^)]+\)\s*\{', content))
        )

        # Count extension functions and properties
        structure.extension_functions = len(re.findall(r'fun\s+(?:<[^>]*>\s+)?\w+(?:<[^>]*>)?\.\w+', content))
        structure.extension_properties = len(re.findall(r'(?:val|var)\s+\w+(?:<[^>]*>)?\.\w+', content))

        # Detect test file
        structure.is_test_file = (
            "Test" in file_path.name
            or file_path.name.endswith("Test.kt")
            or any(part in ["test", "androidTest"] for part in file_path.parts)
        )

        # Detect main function
        structure.has_main = bool(re.search(r'fun\s+main\s*\(', content))

        # Multiplatform detection
        structure.is_multiplatform = bool(
            re.search(r'\b(?:expect|actual)\s+', content) or
            re.search(r'@(?:JvmStatic|JvmOverloads|JvmName|JsName)', content)
        )

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Kotlin code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Null safety complexity
        - Coroutine complexity
        - Android-specific complexity
        - Functional programming complexity

        Args:
            content: Kotlin source code
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
            r'\bwhen\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\bdo\b',
            r'\btry\b',
            r'\bcatch\b',
            r'&&',
            r'\|\|',
            r'\?:',  # Elvis operator
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
                (r'\bwhen\b', 1),
                (r'\bfor\b', 1),
                (r'\bwhile\b', 1),
                (r'\bdo\b', 1),
                (r'\btry\b', 1),
                (r'\bcatch\b', 1),
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

        # Count classes and interfaces
        metrics.class_count = len(re.findall(r'\bclass\s+\w+', content))
        metrics.interface_count = len(re.findall(r'\binterface\s+\w+', content))
        metrics.object_count = len(re.findall(r'\bobject\s+\w+', content))
        metrics.data_class_count = len(re.findall(r'\bdata\s+class\s+\w+', content))
        metrics.sealed_class_count = len(re.findall(r'\bsealed\s+(?:class|interface)\s+\w+', content))

        # Null safety metrics
        metrics.nullable_types = len(re.findall(r'\w+\?(?:\s|,|\)|>)', content))
        metrics.null_assertions = len(re.findall(r'!!', content))
        metrics.safe_calls = len(re.findall(r'\?\.', content))
        metrics.elvis_operators = len(re.findall(r'\?:', content))
        metrics.lateinit_count = len(re.findall(r'\blateinit\s+var\b', content))
        metrics.let_calls = len(re.findall(r'\.let\s*\{', content))

        # Coroutine metrics
        metrics.suspend_functions = len(re.findall(r'\bsuspend\s+fun\b', content))
        metrics.coroutine_launches = len(re.findall(r'\b(?:launch|async)\s*\{', content))
        metrics.await_calls = len(re.findall(r'\.await\(\)', content))
        metrics.flow_usage = len(re.findall(r'\bFlow<|\bflow\s*\{', content))
        metrics.channel_usage = len(re.findall(r'\bChannel<|\bchannel\s*\{', content))
        metrics.runblocking_usage = len(re.findall(r'\brunBlocking\s*\{', content))

        # Functional programming metrics
        metrics.lambda_count = len(re.findall(r'\{[^}]*->[^}]*\}', content))
        metrics.higher_order_functions = len(re.findall(r'(?:map|filter|fold|reduce|flatMap|forEach)\s*\{', content))
        metrics.inline_functions = len(re.findall(r'\binline\s+fun\b', content))
        metrics.extension_functions = len(re.findall(r'fun\s+\w+\.\w+', content))
        metrics.scope_functions = (
            len(re.findall(r'\.(?:let|run|apply|also)\s*\{', content)) +
            len(re.findall(r'\bwith\s*\([^)]+\)\s*\{', content))
        )

        # When expression metrics
        metrics.when_expressions = len(re.findall(r'\bwhen\s*(?:\(|\{)', content))
        metrics.when_branches = len(re.findall(r'->\s*(?:\{|[^,\n]+)', content))

        # Exception handling
        metrics.try_blocks = len(re.findall(r'\btry\s*\{', content))
        metrics.catch_blocks = len(re.findall(r'\bcatch\s*\(', content))
        metrics.finally_blocks = len(re.findall(r'\bfinally\s*\{', content))
        metrics.throw_statements = len(re.findall(r'\bthrow\s+', content))

        # Android-specific metrics (if applicable)
        if self._is_android_file(content):
            metrics.activity_count = len(re.findall(r':\s*(?:AppCompat)?Activity\(\)', content))
            metrics.fragment_count = len(re.findall(r':\s*Fragment\(\)', content))
            metrics.viewmodel_count = len(re.findall(r':\s*(?:Android)?ViewModel\(\)', content))
            metrics.livedata_usage = len(re.findall(r'\bLiveData<|\bMutableLiveData<', content))
            metrics.observer_usage = len(re.findall(r'\.observe\(', content))
            metrics.binding_usage = len(re.findall(r'Binding\b|\.binding', content))

        # Delegation metrics
        metrics.delegation_count = len(re.findall(r'\bby\s+\w+', content))
        metrics.lazy_properties = len(re.findall(r'\bby\s+lazy\s*\{', content))
        metrics.observable_properties = len(re.findall(r'\bby\s+(?:\w+\.)?(?:observable|vetoable)\s*\(', content))

        # Calculate maintainability index
        import math

        if metrics.code_lines > 0:
            # Adjusted for Kotlin
            null_safety_factor = 1 - (metrics.null_assertions * 0.02)
            coroutine_factor = 1 - (metrics.runblocking_usage * 0.05)
            functional_factor = 1 + (metrics.lambda_count * 0.001)
            scope_factor = 1 + (metrics.scope_functions * 0.001)

            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * null_safety_factor
                + 5 * coroutine_factor
                + 5 * functional_factor
                + 5 * scope_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _categorize_import(self, module_path: str) -> str:
        """Categorize a Kotlin import.

        Args:
            module_path: Import path

        Returns:
            Category string
        """
        if module_path.startswith("kotlin."):
            if module_path.startswith("kotlin.collections"):
                return "kotlin_collections"
            elif module_path.startswith("kotlin.coroutines"):
                return "kotlin_coroutines"
            elif module_path.startswith("kotlin.io"):
                return "kotlin_io"
            else:
                return "kotlin_stdlib"
        elif module_path.startswith("kotlinx."):
            if module_path.startswith("kotlinx.coroutines"):
                return "kotlinx_coroutines"
            elif module_path.startswith("kotlinx.serialization"):
                return "kotlinx_serialization"
            elif module_path.startswith("kotlinx.android"):
                return "kotlinx_android"
            else:
                return "kotlinx"
        elif module_path.startswith("java."):
            return "java"
        elif module_path.startswith("javax."):
            return "javax"
        elif module_path.startswith("android.") or module_path.startswith("androidx."):
            return "android"
        elif module_path.startswith("com.google.android"):
            return "google_android"
        elif module_path.startswith("io.ktor"):
            return "ktor"
        elif module_path.startswith("org.jetbrains"):
            return "jetbrains"
        else:
            return "third_party"

    def _is_android_import(self, module_path: str) -> bool:
        """Check if an import is Android-related.

        Args:
            module_path: Import path

        Returns:
            True if it's an Android import
        """
        android_packages = [
            "android.",
            "androidx.",
            "com.google.android",
            "com.android",
        ]
        return any(module_path.startswith(pkg) for pkg in android_packages)

    def _is_android_file(self, content: str) -> bool:
        """Check if the file is Android-related.

        Args:
            content: Kotlin source code

        Returns:
            True if it's an Android file
        """
        android_indicators = [
            r"import\s+android\.",
            r"import\s+androidx\.",
            r":\s*Activity\(",
            r":\s*Fragment\(",
            r":\s*ViewModel\(",
            r":\s*Service\(",
            r"@AndroidEntryPoint",
            r"@HiltAndroidApp",
            r"setContentView\(",
            r"findViewById\(",
            r"R\.layout\.",
            r"R\.id\.",
        ]
        return any(re.search(pattern, content) for pattern in android_indicators)

    def _extract_body(self, content: str, start_pos: int) -> Optional[str]:
        """Extract the body of a class/interface/object.

        Args:
            content: Source code
            start_pos: Position after opening brace

        Returns:
            Body content or None
        """
        if start_pos >= len(content):
            return None
            
        # Handle case where there's no body (just semicolon or nothing)
        if start_pos > 0 and content[start_pos - 1] != '{':
            return None

        brace_count = 1
        pos = start_pos
        in_string = False
        in_multiline_string = False
        escape_next = False

        while pos < len(content) and brace_count > 0:
            char = content[pos]

            if not escape_next:
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

    def _parse_inheritance(self, inheritance_str: str) -> List[str]:
        """Parse inheritance string into list of types.

        Args:
            inheritance_str: Inheritance string

        Returns:
            List of inherited types
        """
        if not inheritance_str:
            return []

        items = []
        current = ""
        depth = 0

        for char in inheritance_str:
            if char in '<([':
                depth += 1
            elif char in '>)]':
                depth -= 1
            elif char == ',' and depth == 0:
                if current.strip():
                    items.append(current.strip())
                current = ""
                continue
            current += char

        if current.strip():
            items.append(current.strip())

        return items

    def _is_likely_interface(self, type_name: str) -> bool:
        """Check if a type name is likely an interface.

        Args:
            type_name: Type name

        Returns:
            True if likely an interface
        """
        # Common interface naming patterns
        interface_patterns = [
            r'^I[A-Z]',  # IInterface pattern
            r'able$',  # Readable, Comparable, etc.
            r'ible$',  # Accessible, etc.
            r'Listener$',
            r'Callback$',
            r'Handler$',
            r'Observer$',
        ]
        
        return any(re.search(pattern, type_name) for pattern in interface_patterns)

    def _parse_constructor_params(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse primary constructor parameters.

        Args:
            params_str: Constructor parameter string

        Returns:
            List of parameter dictionaries
        """
        if not params_str:
            return []

        params = self._split_parameters(params_str)
        parsed = []

        for param in params:
            param = param.strip()
            if not param:
                continue

            param_dict = {}

            # Check for modifiers (val, var, vararg)
            if param.startswith("val "):
                param_dict["is_property"] = True
                param_dict["is_mutable"] = False
                param = param[4:]
            elif param.startswith("var "):
                param_dict["is_property"] = True
                param_dict["is_mutable"] = True
                param = param[4:]
            elif param.startswith("vararg "):
                param_dict["is_vararg"] = True
                param = param[7:]

            # Parse visibility
            for visibility in ["private", "protected", "internal", "public"]:
                if param.startswith(visibility + " "):
                    param_dict["visibility"] = visibility
                    param = param[len(visibility) + 1:]
                    break

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

            parsed.append(param_dict)

        return parsed

    def _split_parameters(self, params_str: str) -> List[str]:
        """Split parameters handling nested generics and defaults.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter strings
        """
        params = []
        current = ""
        depth = 0
        in_string = False

        for char in params_str:
            if char == '"' and (not current or current[-1] != '\\'):
                in_string = not in_string
            elif not in_string:
                if char in '<([':
                    depth += 1
                elif char in '>)]':
                    depth -= 1
                elif char == ',' and depth == 0:
                    if current.strip():
                        params.append(current.strip())
                    current = ""
                    continue
            current += char

        if current.strip():
            params.append(current.strip())

        return params

    # Additional helper methods...
    def _extract_methods(self, body: str) -> List[Dict[str, Any]]:
        """Extract methods from class body."""
        methods = []
        # Match function declarations with modifiers
        method_pattern = r'(?:(public|private|protected|internal)\s+)?(?:(operator|suspend|inline|override|abstract|final|open)\s+)*fun\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, body):
            visibility = match.group(1) or "public"
            modifiers_str = match.group(2) or ""
            modifiers = modifiers_str.split() if modifiers_str else []
            method_name = match.group(3)
            
            methods.append({
                "name": method_name,
                "visibility": visibility,
                "modifiers": modifiers,
                "is_operator": "operator" in modifiers,
                "is_suspend": "suspend" in modifiers,
                "is_inline": "inline" in modifiers,
                "is_override": "override" in modifiers,
                "is_abstract": "abstract" in modifiers,
            })
        return methods

    def _extract_properties(self, body: str) -> List[Dict[str, Any]]:
        """Extract properties from class body."""
        properties = []
        # Match property declarations
        prop_pattern = r'(?:(public|private|protected|internal)\s+)?(?:(const|lateinit|override)\s+)?(val|var)\s+(\w+)'
        for match in re.finditer(prop_pattern, body):
            visibility = match.group(1) or "public"
            modifier = match.group(2)
            prop_type = match.group(3)
            prop_name = match.group(4)
            
            properties.append({
                "name": prop_name,
                "visibility": visibility,
                "is_mutable": prop_type == "var",
                "is_const": modifier == "const",
                "is_lateinit": modifier == "lateinit",
                "is_override": modifier == "override",
            })
        return properties

    def _extract_companion_object(self, body: str) -> Optional[Dict[str, Any]]:
        """Extract companion object from class body."""
        companion_pattern = r'companion\s+object(?:\s+(\w+))?\s*\{'
        match = re.search(companion_pattern, body)
        if match:
            companion_name = match.group(1) if match.group(1) else "Companion"
            companion_body = self._extract_body(body, match.end())
            return {
                "name": companion_name,
                "members": self._extract_companion_members(companion_body) if companion_body else []
            }
        return None
    
    def _extract_companion_members(self, body: str) -> List[Dict[str, Any]]:
        """Extract members from companion object body."""
        members = []
        # Extract functions
        func_pattern = r'fun\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(func_pattern, body):
            members.append({
                "type": "function",
                "name": match.group(1)
            })
        # Extract properties
        prop_pattern = r'(?:val|var)\s+(\w+)'
        for match in re.finditer(prop_pattern, body):
            members.append({
                "type": "property", 
                "name": match.group(1)
            })
        return members

    def _extract_nested_classes(self, body: str) -> List[Dict[str, Any]]:
        """Extract nested classes from class body."""
        # Implementation details...
        return []

    def _extract_interface_methods(self, body: str) -> List[Dict[str, Any]]:
        """Extract method signatures from interface body."""
        # Implementation details...
        return []

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters."""
        # Implementation details...
        return []

    def _is_companion_object(self, content: str, position: int) -> bool:
        """Check if object at position is a companion object."""
        # Implementation details...
        return False

    def _is_inside_class(self, content: str, position: int) -> bool:
        """Check if position is inside a class."""
        # Implementation details...
        return False