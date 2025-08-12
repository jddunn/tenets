"""GDScript code analyzer for Godot game development.

This module provides comprehensive analysis for GDScript source files,
including support for Godot-specific features like signals, exports,
node references, and engine lifecycle methods.
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


class GDScriptAnalyzer(LanguageAnalyzer):
    """GDScript code analyzer for Godot development.

    Provides comprehensive analysis for GDScript files including:
    - Preload and load statements
    - Class inheritance (extends)
    - Signal declarations and connections
    - Export variable declarations
    - Onready variables and node references
    - Godot lifecycle methods (_ready, _process, etc.)
    - Tool scripts and custom resources
    - Typed GDScript (static typing)
    - Inner classes
    - Setget properties
    - Remote and master/puppet keywords (networking)

    Supports Godot 3.x and 4.x GDScript syntax.
    """

    language_name = "gdscript"
    file_extensions = [".gd", ".tres", ".tscn"]  # .tres and .tscn can contain GDScript

    def __init__(self):
        """Initialize the GDScript analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract preload, load, and class references from GDScript code.

        Handles:
        - preload statements: preload("res://path/to/script.gd")
        - load statements: load("res://path/to/resource.tres")
        - const preloads: const MyClass = preload("res://MyClass.gd")
        - class_name declarations (Godot 3.1+)
        - Tool script declarations

        Args:
            content: GDScript source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            # Preload statements
            preload_pattern = r'(?:const\s+)?(\w+)?\s*=?\s*preload\s*\(\s*["\']([^"\']+)["\']\s*\)'
            # Use finditer to support multiple preloads on a single line and avoid overlapping matches
            for match in re.finditer(preload_pattern, line):
                const_name = match.group(1)
                resource_path = match.group(2)
                imports.append(
                    ImportInfo(
                        module=resource_path,
                        alias=const_name,
                        line=i,
                        type="preload",
                        is_relative=resource_path.startswith("res://") or resource_path.startswith("user://"),
                        is_resource=True,
                        resource_type=self._detect_resource_type(resource_path),
                    )
                )

            # Load statements (ensure we don't match the 'load' in 'preload')
            load_pattern = r'(?<!\w)load\s*\('
            for match in re.finditer(load_pattern, line):
                # Extract the actual path argument following this 'load('
                path_match = re.search(r'\(\s*["\']([^"\']+)["\']\s*\)', line[match.start():])
                if not path_match:
                    continue
                resource_path = path_match.group(1)
                imports.append(
                    ImportInfo(
                        module=resource_path,
                        line=i,
                        type="load",
                        is_relative=resource_path.startswith("res://") or resource_path.startswith("user://"),
                        is_runtime_load=True,
                        resource_type=self._detect_resource_type(resource_path),
                    )
                )

            # Class inheritance (extends)
            extends_pattern = r'^\s*extends\s+["\']?([^"\'\s]+)["\']?'
            match = re.match(extends_pattern, line)
            if match:
                parent_class = match.group(1)
                # Check if it's a path or class name
                is_path = "/" in parent_class or parent_class.endswith(".gd")
                imports.append(
                    ImportInfo(
                        module=parent_class,
                        line=i,
                        type="extends",
                        is_relative=is_path,
                        is_inheritance=True,
                        parent_type="script" if is_path else "class",
                    )
                )

            # Class_name declarations (for autoload/global classes)
            class_name_pattern = r'^\s*class_name\s+(\w+)(?:\s*,\s*["\']([^"\']+)["\'])?'
            match = re.match(class_name_pattern, line)
            if match:
                class_name = match.group(1)
                icon_path = match.group(2)
                if icon_path:
                    imports.append(
                        ImportInfo(
                            module=icon_path,
                            line=i,
                            type="icon",
                            is_relative=True,
                            is_resource=True,
                            associated_class=class_name,
                        )
                    )
        # Check for tool script declaration
        if re.search(r'^\s*tool\s*$', content, re.MULTILINE):
            imports.append(
                ImportInfo(
                    module="@tool",
                    line=1,
                    type="tool_mode",
                    is_relative=False,
                    is_editor_script=True,
                )
            )

        # Check for @tool annotation (Godot 4.x)
        if re.search(r'^\s*@tool\s*$', content, re.MULTILINE):
            imports.append(
                ImportInfo(
                    module="@tool",
                    line=1,
                    type="annotation",
                    is_relative=False,
                    is_editor_script=True,
                )
            )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from GDScript code.

        In GDScript, exports include:
        - class_name declarations (global classes)
        - export variables
        - signals
        - Public functions (by convention, non-underscore prefixed)

        Args:
            content: GDScript source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported symbols
        """
        exports = []

        # Extract class_name (makes class globally accessible)
        class_name_pattern = r'^\s*class_name\s+(\w+)(?:\s*,\s*["\']([^"\']+)["\'])?'
        match = re.search(class_name_pattern, content, re.MULTILINE)
        if match:
            exports.append(
                {
                    "name": match.group(1),
                    "type": "global_class",
                    "line": content[:match.start()].count("\n") + 1,
                    "icon": match.group(2),
                    "is_autoload_candidate": True,
                }
            )

        # Extract exported variables (Godot 3.x syntax)
        export_var_pattern = r'^\s*export(?:\s*\(([^)]*)\))?\s+(?:var\s+)?(\w+)'
        for match in re.finditer(export_var_pattern, content, re.MULTILINE):
            export_type = match.group(1)
            var_name = match.group(2)
            
            exports.append(
                {
                    "name": var_name,
                    "type": "export_var",
                    "line": content[:match.start()].count("\n") + 1,
                    "export_type": export_type,
                    "inspector_visible": True,
                }
            )

        # Extract exported variables (Godot 4.x syntax with @export)
        # Allow optional annotation arguments e.g., @export_range(0,1)
        export_annotation_pattern = r'^\s*@export(?:_([a-z_]+))?(?:\([^)]*\))?\s+(?:var\s+)?(\w+)'
        for match in re.finditer(export_annotation_pattern, content, re.MULTILINE):
            export_modifier = match.group(1)
            var_name = match.group(2)
            
            exports.append(
                {
                    "name": var_name,
                    "type": "export_var",
                    "line": content[:match.start()].count("\n") + 1,
                    "export_modifier": export_modifier,
                    "inspector_visible": True,
                    "godot_version": 4,
                }
            )

        # Extract signals
        signal_pattern = r'^\s*signal\s+(\w+)\s*(?:\(([^)]*)\))?'
        for match in re.finditer(signal_pattern, content, re.MULTILINE):
            signal_name = match.group(1)
            parameters = match.group(2)
            
            exports.append(
                {
                    "name": signal_name,
                    "type": "signal",
                    "line": content[:match.start()].count("\n") + 1,
                    "parameters": self._parse_signal_parameters(parameters),
                    "is_event": True,
                }
            )

        # Extract public functions (non-underscore prefixed)
        func_pattern = r'^\s*(?:static\s+)?func\s+([a-zA-Z]\w*)\s*\('
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            
            exports.append(
                {
                    "name": func_name,
                    "type": "function",
                    "line": content[:match.start()].count("\n") + 1,
                    "is_public": True,
                    "is_static": "static" in match.group(0),
                }
            )

        # Extract enums
        enum_pattern = r'^\s*enum\s+(\w+)\s*\{'
        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "enum",
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Extract constants (often used as exports in GDScript)
        const_pattern = r'^\s*const\s+([A-Z][A-Z0-9_]*)\s*='
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            exports.append(
                {
                    "name": match.group(1),
                    "type": "constant",
                    "line": content[:match.start()].count("\n") + 1,
                    "is_public": True,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from GDScript file.

        Extracts:
        - Class inheritance and structure
        - Inner classes
        - Functions with type hints
        - Godot lifecycle methods
        - Signals and their connections
        - Export variables
        - Onready variables
        - Node references
        - Setget properties
        - Enums and constants

        Args:
            content: GDScript source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Detect if it's a tool script
        structure.is_tool_script = bool(re.search(r'^\s*(?:@)?tool\s*$', content, re.MULTILINE))

        # Extract class name
        class_name_match = re.search(r'^\s*class_name\s+(\w+)', content, re.MULTILINE)
        if class_name_match:
            structure.class_name = class_name_match.group(1)

        # Extract parent class
        extends_match = re.search(r'^\s*extends\s+["\']?([^"\'\s]+)["\']?', content, re.MULTILINE)
        if extends_match:
            structure.parent_class = extends_match.group(1)

        # Detect Godot version (4.x uses @annotations)
        structure.godot_version = 4 if re.search(r'^\s*@(export|onready|tool)', content, re.MULTILINE) else 3

        # Extract main class info
        main_class = ClassInfo(
            name=getattr(structure, "class_name", None) or file_path.stem,
            line=1,
            bases=[getattr(structure, "parent_class", None)] if getattr(structure, "parent_class", None) else [],
        )

        # Extract functions
        func_pattern = r'^\s*(?:static\s+)?func\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)
            
            is_private = func_name.startswith("_")
            is_lifecycle = self._is_lifecycle_method(func_name)
            is_virtual = func_name.startswith("_") and not func_name.startswith("__")
            
            func_info = FunctionInfo(
                name=func_name,
                line=content[:match.start()].count("\n") + 1,
                parameters=self._parse_function_parameters(params),
                return_type=return_type.strip() if return_type else None,
                is_private=is_private,
                is_lifecycle=is_lifecycle,
                is_virtual=is_virtual,
                is_static="static" in content[match.start()-20:match.start()],
            )
            
            structure.functions.append(func_info)
            main_class.methods.append({
                "name": func_name,
                "visibility": "private" if is_private else "public",
                "is_lifecycle": is_lifecycle,
            })

        # Extract inner classes
        inner_class_pattern = r'^\s*class\s+(\w+)(?:\s+extends\s+([^:]+))?:'
        for match in re.finditer(inner_class_pattern, content, re.MULTILINE):
            inner_class = ClassInfo(
                name=match.group(1),
                line=content[:match.start()].count("\n") + 1,
                bases=[match.group(2).strip()] if match.group(2) else [],
                is_inner=True,
            )
            structure.classes.append(inner_class)

        # Add main class
        structure.classes.insert(0, main_class)

        # Extract signals
        signal_pattern = r'^\s*signal\s+(\w+)\s*(?:\(([^)]*)\))?'
        for match in re.finditer(signal_pattern, content, re.MULTILINE):
            structure.signals.append(
                {
                    "name": match.group(1),
                    "line": content[:match.start()].count("\n") + 1,
                    "parameters": self._parse_signal_parameters(match.group(2)),
                }
            )

        # Extract export variables
        # Godot 3.x
        export_pattern = r'^\s*export(?:\s*\(([^)]*)\))?\s+(?:var\s+)?(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^\n]+))?'
        for match in re.finditer(export_pattern, content, re.MULTILINE):
            structure.export_vars.append(
                {
                    "name": match.group(2),
                    "export_hint": match.group(1),
                    "type": match.group(3).strip() if match.group(3) else None,
                    "default": match.group(4).strip() if match.group(4) else None,
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Godot 4.x
        export_4_pattern = r'^\s*@export(?:_([a-z_]+))?(?:\([^)]*\))?\s+(?:var\s+)?(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^\n]+))?'
        for match in re.finditer(export_4_pattern, content, re.MULTILINE):
            structure.export_vars.append(
                {
                    "name": match.group(2),
                    "export_modifier": match.group(1),
                    "type": match.group(3).strip() if match.group(3) else None,
                    "default": match.group(4).strip() if match.group(4) else None,
                    "line": content[:match.start()].count("\n") + 1,
                    "godot_4": True,
                }
            )

        # Extract onready variables
        # Godot 3.x
        onready_pattern = r'^\s*onready\s+var\s+(\w+)(?:\s*:\s*([^=\n]+))?\s*=\s*([^\n]+)'
        for match in re.finditer(onready_pattern, content, re.MULTILINE):
            var_name = match.group(1)
            var_type = match.group(2)
            initialization = match.group(3)
            
            # Check if it's a node reference
            is_node_ref = bool(re.search(r'(?:\$|get_node)', initialization))
            node_path = self._extract_node_path(initialization)
            
            structure.onready_vars.append(
                {
                    "name": var_name,
                    "type": var_type.strip() if var_type else None,
                    "initialization": initialization.strip(),
                    "is_node_ref": is_node_ref,
                    "node_path": node_path,
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Godot 4.x
        onready_4_pattern = r'^\s*@onready\s+var\s+(\w+)(?:\s*:\s*([^=\n]+))?\s*=\s*([^\n]+)'
        for match in re.finditer(onready_4_pattern, content, re.MULTILINE):
            var_name = match.group(1)
            var_type = match.group(2)
            initialization = match.group(3)
            
            is_node_ref = bool(re.search(r'(?:\$|get_node)', initialization))
            node_path = self._extract_node_path(initialization)
            
            structure.onready_vars.append(
                {
                    "name": var_name,
                    "type": var_type.strip() if var_type else None,
                    "initialization": initialization.strip(),
                    "is_node_ref": is_node_ref,
                    "node_path": node_path,
                    "line": content[:match.start()].count("\n") + 1,
                    "godot_4": True,
                }
            )

        # Extract regular variables
        var_pattern = r'^\s*var\s+(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^\n]+))?'
        for match in re.finditer(var_pattern, content, re.MULTILINE):
            # Skip if it's an export or onready var
            line_start = content[:match.start()].rfind('\n') + 1
            line_content = content[line_start:match.end()]
            if 'export' in line_content or 'onready' in line_content or '@' in line_content:
                continue
                
            structure.variables.append(
                {
                    "name": match.group(1),
                    "type": match.group(2).strip() if match.group(2) else None,
                    "initial_value": match.group(3).strip() if match.group(3) else None,
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Extract constants
        const_pattern = r'^\s*const\s+(\w+)(?:\s*:\s*([^=\n]+))?\s*=\s*([^\n]+)'
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            structure.constants.append(
                {
                    "name": match.group(1),
                    "type": match.group(2).strip() if match.group(2) else None,
                    "value": match.group(3).strip(),
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Extract enums
        enum_pattern = r'^\s*enum\s+(\w+)\s*\{([^}]+)\}'
        for match in re.finditer(enum_pattern, content, re.MULTILINE):
            enum_name = match.group(1)
            enum_body = match.group(2)
            
            values = self._parse_enum_values(enum_body)
            
            structure.enums.append(
                {
                    "name": enum_name,
                    "values": values,
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Extract setget properties
        # Support optional setter/getter and missing entries: e.g., setget set_mana or setget , get_level
        setget_pattern = r'^\s*var\s+(\w+)(?:[^=\n]*=\s*[^\n]+)?\s+setget\s*(?:([A-Za-z_]\w*)\s*)?(?:,\s*([A-Za-z_]\w*)\s*)?'
        for match in re.finditer(setget_pattern, content, re.MULTILINE):
            structure.setget_properties.append(
                {
                    "name": match.group(1),
                    "setter": match.group(2) if match.group(2) else None,
                    "getter": match.group(3) if match.group(3) else None,
                    "line": content[:match.start()].count("\n") + 1,
                }
            )

        # Count node references
        structure.node_references = len(re.findall(r'\$["\']?[^"\'\s]+["\']?', content))
        structure.get_node_calls = len(re.findall(r'get_node\s*\(', content))

        # Count signal connections (method form and free function form)
        structure.connect_calls = len(
            re.findall(r'\.connect\s*\(|(?<!\.)\bconnect\s*\(', content)
        )
        structure.emit_signal_calls = len(re.findall(r'emit_signal\s*\(', content))

        # Detect if it's a custom resource
        structure.is_custom_resource = bool(
            structure.parent_class and "Resource" in structure.parent_class
        )

        # Detect if it's an editor plugin
        structure.is_editor_plugin = bool(
            structure.parent_class and "EditorPlugin" in structure.parent_class
        )

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for GDScript code.

        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity  
        - Godot-specific complexity (signals, exports, node references)
        - Nesting depth
        - Function count and complexity distribution

        Args:
            content: GDScript source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Calculate cyclomatic complexity
        complexity = 1

        decision_keywords = [
            r'\bif\b',
            r'\belif\b',
            r'\belse\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\bmatch\b',
            r'\bwhen\b',
            r'\band\b',
            r'\bor\b',
        ]

        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))

        # Each match-case branch contributes to complexity; count simple case labels (numbers, strings, or underscore)
        case_label_pattern = r'^\s*(?:_|-?\d+|\"[^\"\n]+\"|\'[^\'\n]+\')\s*:'
        complexity += len(re.findall(case_label_pattern, content, re.MULTILINE))

        # Inline lambda expressions (func(...) :) add decision/branching potential
        lambda_inline_pattern = r'func\s*\('  # named functions are 'func name(', lambdas are 'func(' directly
        complexity += len(re.findall(lambda_inline_pattern, content))

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

            # Track nesting by indentation (GDScript uses indentation)
            if line.strip():
                indent = len(line) - len(line.lstrip())
                # Assuming tab or 4 spaces as one level
                if '\t' in line[:indent]:
                    current_level = line[:indent].count('\t')
                else:
                    current_level = indent // 4
                
                max_nesting = max(max_nesting, current_level)

                # Control structures with nesting penalty
                control_patterns = [
                    (r'\bif\b', 1),
                    (r'\belif\b', 1),
                    (r'\belse\b', 0),
                    (r'\bfor\b', 1),
                    (r'\bwhile\b', 1),
                    (r'\bmatch\b', 1),
                ]

                for pattern, weight in control_patterns:
                    if re.search(pattern, line):
                        cognitive += weight * (1 + max(0, current_level))

        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting

        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        metrics.comment_lines = len([l for l in lines if l.strip().startswith('#')])
        metrics.comment_ratio = metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0

        # Count functions
        metrics.function_count = len(re.findall(r'\bfunc\s+\w+', content))
        
        # Count classes
        metrics.class_count = len(re.findall(r'\bclass\s+\w+', content))
        metrics.class_count += 1 if re.search(r'^\s*extends\s+', content, re.MULTILINE) else 0

        # Godot-specific metrics
        metrics.signal_count = len(re.findall(r'\bsignal\s+\w+', content))
        metrics.export_count = len(re.findall(r'(?:@)?export(?:_\w+)?(?:\([^)]*\))?\s+', content))
        metrics.onready_count = len(re.findall(r'(?:@)?onready\s+var', content))
        
        # Node reference metrics
        metrics.node_ref_count = len(re.findall(r'\$["\']?[^"\'\s]+["\']?', content))
        metrics.get_node_count = len(re.findall(r'get_node\s*\(', content))
        
        # Signal connection metrics
        metrics.connect_count = len(re.findall(r'\.connect\s*\(|(?<!\.)\bconnect\s*\(', content))
        metrics.emit_count = len(re.findall(r'emit_signal\s*\(', content))
        
        # Lifecycle method count
        lifecycle_methods = [
            '_ready', '_enter_tree', '_exit_tree', '_process', '_physics_process',
            '_input', '_unhandled_input', '_draw', '_gui_input', '_notification'
        ]
        metrics.lifecycle_count = sum(
            1 for method in lifecycle_methods 
            if re.search(rf'\bfunc\s+{method}\s*\(', content)
        )

        # RPC/Networking metrics
        metrics.rpc_count = len(re.findall(r'@rpc|rpc\(|rpc_unreliable\(|remotesync\s+func', content))
        
        # Type hints metrics
        metrics.typed_vars = len(re.findall(r'(?:var|const)\s+\w+\s*:\s*\w+', content))
        metrics.typed_funcs = len(re.findall(r'func\s+\w+\s*\([^)]*:\s*\w+[^)]*\)', content))
        metrics.return_types = len(re.findall(r'\)\s*->\s*\w+\s*:', content))

        # Calculate Godot-specific complexity score
        godot_complexity = (
            metrics.signal_count * 2 +
            metrics.export_count +
            metrics.onready_count +
            metrics.node_ref_count * 0.5 +
            metrics.connect_count * 2 +
            metrics.emit_count
        )

        # Calculate maintainability index
        import math
        
        if metrics.code_lines > 0:
            # Adjusted for GDScript
            godot_factor = 1 - (godot_complexity * 0.001)
            type_factor = 1 + (metrics.typed_vars + metrics.typed_funcs) * 0.001
            
            mi = (
                171
                - 5.2 * math.log(max(1, complexity))
                - 0.23 * complexity
                - 16.2 * math.log(metrics.code_lines)
                + 10 * godot_factor
                + 5 * type_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))

        return metrics

    def _detect_resource_type(self, path: str) -> str:
        """Detect the type of Godot resource from its path.

        Args:
            path: Resource path

        Returns:
            Resource type string
        """
        extension = Path(path).suffix.lower()
        
        resource_types = {
            '.gd': 'script',
            '.gdscript': 'script',
            '.tscn': 'scene',
            '.scn': 'scene',
            '.tres': 'resource',
            '.res': 'resource',
            '.png': 'texture',
            '.jpg': 'texture',
            '.jpeg': 'texture',
            '.svg': 'texture',
            '.ogg': 'audio',
            '.wav': 'audio',
            '.mp3': 'audio',
            '.glb': 'model',
            '.gltf': 'model',
            '.obj': 'model',
            '.dae': 'model',
            '.ttf': 'font',
            '.otf': 'font',
            '.woff': 'font',
            '.woff2': 'font',
        }
        
        return resource_types.get(extension, 'unknown')

    def _is_lifecycle_method(self, func_name: str) -> bool:
        """Check if a function is a Godot lifecycle method.

        Args:
            func_name: Function name

        Returns:
            True if it's a lifecycle method
        """
        lifecycle_methods = {
            '_ready', '_enter_tree', '_exit_tree', '_process', '_physics_process',
            '_input', '_unhandled_input', '_unhandled_key_input', '_gui_input',
            '_draw', '_notification', '_init',
            # Physics callbacks
            '_integrate_forces',
            # Area callbacks
            '_on_area_entered', '_on_area_exited',
            '_on_body_entered', '_on_body_exited',
            # Animation callbacks
            '_on_animation_finished', '_on_animation_started',
        }
        
        return func_name in lifecycle_methods or func_name.startswith('_on_')

    def _parse_signal_parameters(self, params_str: Optional[str]) -> List[str]:
        """Parse signal parameters.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter names
        """
        if not params_str:
            return []
        
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                # Handle typed parameters (param_name: type)
                if ':' in param:
                    param_name = param.split(':')[0].strip()
                else:
                    param_name = param
                params.append(param_name)
        
        return params

    def _parse_function_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters with type hints.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter dictionaries
        """
        if not params_str.strip():
            return []
        
        parameters = []
        params = self._split_parameters(params_str)
        
        for param in params:
            param = param.strip()
            if not param:
                continue
            
            # Parse parameter with optional type and default value
            # Format: name: Type = default
            param_dict = {}
            
            # Check for default value
            if '=' in param:
                param_part, default = param.split('=', 1)
                param_dict['default'] = default.strip()
            else:
                param_part = param
            
            # Check for type hint
            if ':' in param_part:
                name, type_hint = param_part.split(':', 1)
                param_dict['name'] = name.strip()
                param_dict['type'] = type_hint.strip()
            else:
                param_dict['name'] = param_part.strip()
            
            parameters.append(param_dict)
        
        return parameters

    def _split_parameters(self, params_str: str) -> List[str]:
        """Split parameters handling nested brackets.

        Args:
            params_str: Parameter string

        Returns:
            List of parameter strings
        """
        params = []
        current = ""
        depth = 0
        
        for char in params_str:
            if char in '([{':
                depth += 1
            elif char in ')]}':
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

    def _extract_node_path(self, initialization: str) -> Optional[str]:
        """Extract node path from initialization code.

        Args:
            initialization: Variable initialization code

        Returns:
            Node path if found
        """
        # Match $NodePath or $"Node Path"
        dollar_match = re.search(r'\$(["\']?)([^"\']+)\1', initialization)
        if dollar_match:
            return dollar_match.group(2)
        
        # Match get_node("NodePath")
        get_node_match = re.search(r'get_node\s*\(\s*["\']([^"\']+)["\']\s*\)', initialization)
        if get_node_match:
            return get_node_match.group(1)
        
        return None

    def _parse_enum_values(self, enum_body: str) -> List[Dict[str, Any]]:
        """Parse enum values from enum body.

        Args:
            enum_body: Content of enum body

        Returns:
            List of enum value dictionaries
        """
        values = []
        
        for item in enum_body.split(','):
            item = item.strip()
            if not item:
                continue
            
            # Handle VALUE = number format
            if '=' in item:
                name, value = item.split('=', 1)
                values.append({
                    "name": name.strip(),
                    "value": value.strip()
                })
            else:
                values.append({
                    "name": item,
                    "value": None  # Auto-assigned
                })
        
        return values