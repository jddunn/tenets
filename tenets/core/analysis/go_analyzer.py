"""Go language code analyzer.

This module provides comprehensive analysis for Go source files,
including package management, goroutines, channels, and Go-specific patterns.
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
    ClassInfo
)
from tenets.utils.logger import get_logger


class GoAnalyzer(LanguageAnalyzer):
    """Go code analyzer.
    
    Provides comprehensive analysis for Go files including:
    - Import analysis with vendored and internal imports
    - Function, method and interface extraction
    - Struct analysis with embedded types
    - Goroutine and channel detection
    - Error handling patterns
    - Defer statement tracking
    - Package-level analysis
    - Go module support
    
    Go's export mechanism is based on capitalization - identifiers
    starting with uppercase letters are exported.
    """
    
    language_name = "go"
    file_extensions = [".go"]
    
    def __init__(self):
        """Initialize the Go analyzer with logger."""
        self.logger = get_logger(__name__)
    
    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract imports from Go code.
        
        Handles:
        - Single imports: import "fmt"
        - Grouped imports: import ( "fmt" "strings" )
        - Aliased imports: import f "fmt"
        - Dot imports: import . "fmt"
        - Blank imports: import _ "database/sql"
        - Vendored imports
        - Internal packages
        
        Args:
            content: Go source code
            file_path: Path to the file being analyzed
            
        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        lines = content.split('\n')
        
        import_block = False
        import_block_start = 0
        
        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            # Single import statement
            single_import = re.match(r'^\s*import\s+"([^"]+)"', line)
            if single_import:
                imports.append(ImportInfo(
                    module=single_import.group(1),
                    line=i,
                    type="import",
                    is_relative=False,
                    is_vendored=self._is_vendored_import(single_import.group(1)),
                    is_internal='internal' in single_import.group(1)
                ))
                continue
            
            # Aliased single import
            aliased_import = re.match(r'^\s*import\s+(\w+)\s+"([^"]+)"', line)
            if aliased_import:
                imports.append(ImportInfo(
                    module=aliased_import.group(2),
                    alias=aliased_import.group(1),
                    line=i,
                    type="aliased",
                    is_relative=False,
                    is_vendored=self._is_vendored_import(aliased_import.group(2))
                ))
                continue
            
            # Dot import
            dot_import = re.match(r'^\s*import\s+\.\s+"([^"]+)"', line)
            if dot_import:
                imports.append(ImportInfo(
                    module=dot_import.group(1),
                    alias=".",
                    line=i,
                    type="dot_import",
                    is_relative=False
                ))
                continue
            
            # Blank import
            blank_import = re.match(r'^\s*import\s+_\s+"([^"]+)"', line)
            if blank_import:
                imports.append(ImportInfo(
                    module=blank_import.group(1),
                    alias="_",
                    line=i,
                    type="blank_import",
                    is_relative=False,
                    purpose="side_effects"
                ))
                continue
            
            # Import block start
            if re.match(r'^\s*import\s*\(', line):
                import_block = True
                import_block_start = i
                continue
            
            # Inside import block
            if import_block:
                # Check for end of import block
                if ')' in line:
                    import_block = False
                    continue
                
                # Standard import in block
                standard_import = re.match(r'^\s*"([^"]+)"', line)
                if standard_import:
                    module = standard_import.group(1)
                    imports.append(ImportInfo(
                        module=module,
                        line=i,
                        type="import",
                        is_relative=False,
                        is_vendored=self._is_vendored_import(module),
                        is_internal='internal' in module,
                        is_stdlib=self._is_stdlib_import(module)
                    ))
                    continue
                
                # Aliased import in block
                aliased_import = re.match(r'^\s*(\w+)\s+"([^"]+)"', line)
                if aliased_import:
                    module = aliased_import.group(2)
                    imports.append(ImportInfo(
                        module=module,
                        alias=aliased_import.group(1),
                        line=i,
                        type="aliased",
                        is_relative=False,
                        is_vendored=self._is_vendored_import(module)
                    ))
                    continue
                
                # Dot import in block
                dot_import = re.match(r'^\s*\.\s+"([^"]+)"', line)
                if dot_import:
                    imports.append(ImportInfo(
                        module=dot_import.group(1),
                        alias=".",
                        line=i,
                        type="dot_import",
                        is_relative=False
                    ))
                    continue
                
                # Blank import in block
                blank_import = re.match(r'^\s*_\s+"([^"]+)"', line)
                if blank_import:
                    imports.append(ImportInfo(
                        module=blank_import.group(1),
                        alias="_",
                        line=i,
                        type="blank_import",
                        is_relative=False,
                        purpose="side_effects"
                    ))
        
        # Categorize imports
        self._categorize_imports(imports)
        
        return imports
    
    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from Go code.
        
        In Go, exported identifiers start with an uppercase letter.
        This includes functions, types, constants, and variables.
        
        Args:
            content: Go source code
            file_path: Path to the file being analyzed
            
        Returns:
            List of exported symbols with metadata
        """
        exports = []
        
        # Extract package name
        package_match = re.search(r'^\s*package\s+(\w+)', content, re.MULTILINE)
        package_name = package_match.group(1) if package_match else 'unknown'
        
        # Exported functions
        func_pattern = r'^func\s+([A-Z][a-zA-Z0-9]*)\s*\(([^)]*)\)(?:\s*\(([^)]*)\))?(?:\s*([^{]+))?\s*\{'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            params = match.group(2)
            return_params = match.group(3)
            return_type = match.group(4)
            
            exports.append({
                'name': func_name,
                'type': 'function',
                'line': content[:match.start()].count('\n') + 1,
                'package': package_name,
                'signature': self._build_function_signature(func_name, params, return_params, return_type),
                'has_receiver': False
            })
        
        # Exported methods (with receivers)
        method_pattern = r'^func\s+\(([^)]+)\)\s+([A-Z][a-zA-Z0-9]*)\s*\(([^)]*)\)(?:\s*\(([^)]*)\))?(?:\s*([^{]+))?\s*\{'
        for match in re.finditer(method_pattern, content, re.MULTILINE):
            receiver = match.group(1)
            method_name = match.group(2)
            params = match.group(3)
            return_params = match.group(4)
            return_type = match.group(5)
            
            # Parse receiver type
            receiver_type = self._parse_receiver(receiver)
            
            exports.append({
                'name': method_name,
                'type': 'method',
                'line': content[:match.start()].count('\n') + 1,
                'receiver': receiver_type,
                'package': package_name,
                'signature': self._build_method_signature(receiver, method_name, params, return_params, return_type),
                'has_receiver': True
            })
        
        # Exported types (structs, interfaces, type aliases)
        type_pattern = r'^type\s+([A-Z][a-zA-Z0-9]*)\s+(.+?)(?:\n|\{)'
        for match in re.finditer(type_pattern, content, re.MULTILINE):
            type_name = match.group(1)
            type_def = match.group(2).strip()
            
            # Determine type kind
            if 'struct' in type_def:
                type_kind = 'struct'
            elif 'interface' in type_def:
                type_kind = 'interface'
            elif '=' in type_def:
                type_kind = 'alias'
            else:
                type_kind = 'type'
            
            exports.append({
                'name': type_name,
                'type': type_kind,
                'line': content[:match.start()].count('\n') + 1,
                'package': package_name,
                'definition': type_def[:50] if len(type_def) > 50 else type_def
            })
        
        # Exported constants
        const_pattern = r'^const\s+([A-Z][a-zA-Z0-9]*)\s*(?:[\w\s]+)?\s*='
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            exports.append({
                'name': match.group(1),
                'type': 'constant',
                'line': content[:match.start()].count('\n') + 1,
                'package': package_name
            })
        
        # Exported constant blocks
        const_block_pattern = r'^const\s*\((.*?)\)'
        for match in re.finditer(const_block_pattern, content, re.MULTILINE | re.DOTALL):
            block_content = match.group(1)
            for const_match in re.finditer(r'^\s*([A-Z][a-zA-Z0-9]*)', block_content, re.MULTILINE):
                exports.append({
                    'name': const_match.group(1),
                    'type': 'constant',
                    'line': content[:match.start()].count('\n') + block_content[:const_match.start()].count('\n') + 1,
                    'package': package_name,
                    'in_block': True
                })
        
        # Exported variables
        var_pattern = r'^var\s+([A-Z][a-zA-Z0-9]*)\s+'
        for match in re.finditer(var_pattern, content, re.MULTILINE):
            exports.append({
                'name': match.group(1),
                'type': 'variable',
                'line': content[:match.start()].count('\n') + 1,
                'package': package_name
            })
        
        return exports
    
    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from Go file.
        
        Extracts:
        - Package declaration
        - Functions and methods
        - Structs (treated as classes)
        - Interfaces
        - Type aliases
        - Constants and variables
        - Goroutines and channels
        - Init functions
        
        Args:
            content: Go source code
            file_path: Path to the file being analyzed
            
        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()
        
        # Extract package name
        package_match = re.search(r'^\s*package\s+(\w+)', content, re.MULTILINE)
        if package_match:
            structure.package = package_match.group(1)
            structure.is_main = structure.package == 'main'
        
        # Extract functions
        func_pattern = r'^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(([^)]*)\)(?:\s*\(([^)]*)\))?(?:\s*([^{]+))?'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            func_name = match.group(1)
            params = match.group(2)
            
            # Check for special functions
            is_init = func_name == 'init'
            is_main = func_name == 'main' and structure.is_main
            is_test = func_name.startswith('Test') or func_name.startswith('Benchmark')
            
            func_info = FunctionInfo(
                name=func_name,
                line=content[:match.start()].count('\n') + 1,
                args=self._parse_go_params(params),
                is_exported=func_name[0].isupper(),
                is_init=is_init,
                is_main=is_main,
                is_test=is_test
            )
            
            structure.functions.append(func_info)
        
        # Extract structs (as classes)
        struct_pattern = r'^type\s+(\w+)\s+struct\s*\{'
        for match in re.finditer(struct_pattern, content, re.MULTILINE):
            struct_name = match.group(1)
            
            # Find struct fields
            struct_start = match.end()
            brace_count = 1
            struct_end = struct_start
            
            for i, char in enumerate(content[struct_start:], struct_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        struct_end = i
                        break
            
            struct_content = content[struct_start:struct_end]
            fields = self._extract_struct_fields(struct_content)
            
            # Find methods for this struct
            methods = self._find_struct_methods(content, struct_name)
            
            class_info = ClassInfo(
                name=struct_name,
                line=content[:match.start()].count('\n') + 1,
                is_exported=struct_name[0].isupper(),
                fields=fields,
                methods=methods,
                embedded_types=self._find_embedded_types(struct_content)
            )
            
            structure.classes.append(class_info)
        
        # Extract interfaces
        interface_pattern = r'^type\s+(\w+)\s+interface\s*\{'
        for match in re.finditer(interface_pattern, content, re.MULTILINE):
            interface_name = match.group(1)
            
            # Find interface methods
            interface_start = match.end()
            brace_count = 1
            interface_end = interface_start
            
            for i, char in enumerate(content[interface_start:], interface_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        interface_end = i
                        break
            
            interface_content = content[interface_start:interface_end]
            methods = self._extract_interface_methods(interface_content)
            
            structure.interfaces.append({
                'name': interface_name,
                'line': content[:match.start()].count('\n') + 1,
                'is_exported': interface_name[0].isupper(),
                'methods': methods,
                'is_empty': len(methods) == 0  # Empty interface (interface{})
            })
        
        # Extract type aliases
        type_alias_pattern = r'^type\s+(\w+)\s*=\s*(.+)$'
        for match in re.finditer(type_alias_pattern, content, re.MULTILINE):
            structure.type_aliases.append({
                'name': match.group(1),
                'base_type': match.group(2).strip(),
                'line': content[:match.start()].count('\n') + 1,
                'is_exported': match.group(1)[0].isupper()
            })
        
        # Extract custom type definitions
        type_def_pattern = r'^type\s+(\w+)\s+(\w+)$'
        for match in re.finditer(type_def_pattern, content, re.MULTILINE):
            if not re.match(r'^type\s+\w+\s+(?:struct|interface)', content[match.start():]):
                structure.type_definitions.append({
                    'name': match.group(1),
                    'base_type': match.group(2),
                    'line': content[:match.start()].count('\n') + 1,
                    'is_exported': match.group(1)[0].isupper()
                })
        
        # Extract constants
        const_pattern = r'^const\s+(\w+)'
        for match in re.finditer(const_pattern, content, re.MULTILINE):
            const_name = match.group(1)
            structure.constants.append(const_name)
            structure.variables.append({
                'name': const_name,
                'line': content[:match.start()].count('\n') + 1,
                'type': 'constant',
                'is_exported': const_name[0].isupper()
            })
        
        # Extract variables
        var_pattern = r'^var\s+(\w+)'
        for match in re.finditer(var_pattern, content, re.MULTILINE):
            var_name = match.group(1)
            structure.variables.append({
                'name': var_name,
                'line': content[:match.start()].count('\n') + 1,
                'type': 'variable',
                'is_exported': var_name[0].isupper()
            })
        
        # Count goroutines
        goroutine_pattern = r'\bgo\s+(?:\w+\.)*\w+\s*\('
        structure.goroutines_count = len(re.findall(goroutine_pattern, content))
        
        # Count channels
        channel_pattern = r'(?:chan\s+\w+|<-chan\s+\w+|chan<-\s+\w+)'
        structure.channels_count = len(re.findall(channel_pattern, content))
        
        # Count defer statements
        defer_pattern = r'\bdefer\s+'
        structure.defer_count = len(re.findall(defer_pattern, content))
        
        # Detect test file
        structure.is_test_file = file_path.name.endswith('_test.go')
        
        return structure
    
    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for Go code.
        
        Calculates:
        - Cyclomatic complexity
        - Cognitive complexity
        - Error handling complexity
        - Concurrency complexity
        - Test coverage indicators
        
        Args:
            content: Go source code
            file_path: Path to the file being analyzed
            
        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()
        
        # Calculate cyclomatic complexity
        complexity = 1
        
        decision_keywords = [
            r'\bif\b', r'\belse\s+if\b', r'\belse\b', r'\bfor\b',
            r'\bswitch\b', r'\bcase\b', r'\bselect\b', r'\bdefault\b',
            r'\b&&\b', r'\|\|'
        ]
        
        for keyword in decision_keywords:
            complexity += len(re.findall(keyword, content))
        
        # Add complexity for range loops
        complexity += len(re.findall(r'\bfor\s+\w+\s*:=\s*range\b', content))
        
        metrics.cyclomatic = complexity
        
        # Calculate cognitive complexity
        cognitive = 0
        nesting_level = 0
        max_nesting = 0
        
        lines = content.split('\n')
        for line in lines:
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            # Track nesting
            opening_braces = line.count('{')
            closing_braces = line.count('}')
            nesting_level += opening_braces - closing_braces
            max_nesting = max(max_nesting, nesting_level)
            
            # Control structures with nesting penalty
            control_patterns = [
                (r'\bif\b', 1),
                (r'\bfor\b', 1),
                (r'\bswitch\b', 1),
                (r'\bselect\b', 2),  # Higher weight for select
                (r'\bcase\b', 0.5),
                (r'\belse\s+if\b', 1),
                (r'\belse\b', 0)
            ]
            
            for pattern, weight in control_patterns:
                if re.search(pattern, line):
                    cognitive += weight * (1 + max(0, nesting_level - 1))
            
            # Error handling complexity
            if 'err != nil' in line:
                cognitive += 1
                metrics.error_handling_count = getattr(metrics, 'error_handling_count', 0) + 1
            
            # Panic/recover complexity
            if re.search(r'\bpanic\b|\brecover\b', line):
                cognitive += 2
        
        metrics.cognitive = cognitive
        metrics.max_depth = max_nesting
        
        # Count code elements
        metrics.line_count = len(lines)
        metrics.code_lines = self._count_code_lines(content)
        metrics.comment_lines = self._count_comment_lines(content)
        metrics.comment_ratio = metrics.comment_lines / metrics.line_count if metrics.line_count > 0 else 0
        
        # Count functions and methods
        metrics.function_count = len(re.findall(r'^func\s+(?:\([^)]+\)\s+)?\w+\s*\(', content, re.MULTILINE))
        
        # Count structs and interfaces
        metrics.struct_count = len(re.findall(r'^type\s+\w+\s+struct\s*\{', content, re.MULTILINE))
        metrics.interface_count = len(re.findall(r'^type\s+\w+\s+interface\s*\{', content, re.MULTILINE))
        
        # Concurrency metrics
        metrics.goroutines_count = len(re.findall(r'\bgo\s+\w+', content))
        metrics.channels_count = len(re.findall(r'chan\s+\w+', content))
        metrics.select_statements = len(re.findall(r'\bselect\s*\{', content))
        metrics.mutex_usage = len(re.findall(r'sync\.(?:Mutex|RWMutex)', content))
        
        # Error handling metrics
        metrics.error_checks = len(re.findall(r'if\s+err\s*!=\s*nil', content))
        metrics.error_returns = len(re.findall(r'return\s+.*err', content))
        
        # Test metrics (if test file)
        if file_path.name.endswith('_test.go'):
            metrics.test_count = len(re.findall(r'^func\s+Test\w+\s*\(', content, re.MULTILINE))
            metrics.benchmark_count = len(re.findall(r'^func\s+Benchmark\w+\s*\(', content, re.MULTILINE))
            metrics.example_count = len(re.findall(r'^func\s+Example\w*\s*\(', content, re.MULTILINE))
        
        # Calculate maintainability index
        import math
        if metrics.code_lines > 0:
            # Adjusted for Go's error handling patterns
            error_factor = max(0, 1 - (metrics.error_checks / metrics.code_lines))
            mi = (171 
                  - 5.2 * math.log(max(1, complexity))
                  - 0.23 * complexity
                  - 16.2 * math.log(metrics.code_lines)
                  + 20 * error_factor)  # Bonus for proper error handling
            metrics.maintainability_index = max(0, min(100, mi))
        
        return metrics
    
    def _is_vendored_import(self, module: str) -> bool:
        """Check if an import is vendored.
        
        Args:
            module: Import module path
            
        Returns:
            True if the import is vendored
        """
        return 'vendor/' in module
    
    def _is_stdlib_import(self, module: str) -> bool:
        """Check if an import is from Go standard library.
        
        Args:
            module: Import module path
            
        Returns:
            True if the import is from stdlib
        """
        stdlib_packages = {
            'fmt', 'io', 'os', 'strings', 'bytes', 'errors', 'time',
            'math', 'sort', 'sync', 'context', 'net', 'http', 'json',
            'encoding', 'crypto', 'database', 'reflect', 'runtime',
            'testing', 'flag', 'log', 'path', 'filepath', 'regexp'
        }
        
        # Check if it's a known stdlib package
        package = module.split('/')[0]
        return package in stdlib_packages or not '.' in package
    
    def _categorize_imports(self, imports: List[ImportInfo]) -> None:
        """Categorize imports into stdlib, third-party, and local.
        
        Args:
            imports: List of ImportInfo objects to categorize
        """
        for imp in imports:
            if self._is_stdlib_import(imp.module):
                imp.category = 'stdlib'
            elif '.' in imp.module.split('/')[0]:  # Domain-based imports
                imp.category = 'third_party'
            else:
                imp.category = 'local'
    
    def _build_function_signature(self, name: str, params: str, 
                                 return_params: str, return_type: str) -> str:
        """Build a function signature string.
        
        Args:
            name: Function name
            params: Parameter string
            return_params: Return parameters (for multiple returns)
            return_type: Return type
            
        Returns:
            Complete function signature
        """
        signature = f"func {name}({params})"
        if return_params:
            signature += f" ({return_params})"
        elif return_type:
            signature += f" {return_type.strip()}"
        return signature
    
    def _build_method_signature(self, receiver: str, name: str, params: str,
                               return_params: str, return_type: str) -> str:
        """Build a method signature string.
        
        Args:
            receiver: Method receiver
            name: Method name
            params: Parameter string
            return_params: Return parameters
            return_type: Return type
            
        Returns:
            Complete method signature
        """
        signature = f"func ({receiver}) {name}({params})"
        if return_params:
            signature += f" ({return_params})"
        elif return_type:
            signature += f" {return_type.strip()}"
        return signature
    
    def _parse_receiver(self, receiver: str) -> str:
        """Parse receiver type from receiver string.
        
        Args:
            receiver: Receiver string like "r *Receiver"
            
        Returns:
            Receiver type
        """
        parts = receiver.strip().split()
        if len(parts) >= 2:
            return parts[-1].lstrip('*')
        return receiver
    
    def _parse_go_params(self, params: str) -> List[str]:
        """Parse Go function parameters.
        
        Args:
            params: Parameter string from function signature
            
        Returns:
            List of parameter names with types
        """
        if not params.strip():
            return []
        
        param_list = []
        # Handle complex parameter lists
        params = params.strip()
        
        # Simple parsing - can be enhanced for more complex cases
        for param in params.split(','):
            param = param.strip()
            if param:
                # Format: name type or just type
                parts = param.split()
                if len(parts) >= 2:
                    param_list.append(f"{parts[0]}: {' '.join(parts[1:])}")
                else:
                    param_list.append(param)
        
        return param_list
    
    def _extract_struct_fields(self, struct_content: str) -> List[Dict[str, Any]]:
        """Extract fields from struct definition.
        
        Args:
            struct_content: Content between struct braces
            
        Returns:
            List of field dictionaries
        """
        fields = []
        
        # Field pattern: name type `tags`
        field_pattern = r'^\s*(\w+)\s+([^`\n]+)(?:`([^`]+)`)?'
        
        for match in re.finditer(field_pattern, struct_content, re.MULTILINE):
            field_name = match.group(1)
            field_type = match.group(2).strip()
            tags = match.group(3) if match.group(3) else ""
            
            fields.append({
                'name': field_name,
                'type': field_type,
                'tags': tags,
                'is_exported': field_name[0].isupper(),
                'is_embedded': field_name == field_type
            })
        
        return fields
    
    def _find_embedded_types(self, struct_content: str) -> List[str]:
        """Find embedded types in struct.
        
        Args:
            struct_content: Content between struct braces
            
        Returns:
            List of embedded type names
        """
        embedded = []
        
        # Embedded types are fields without explicit names
        embedded_pattern = r'^\s*\*?([A-Z]\w+)\s*$'
        
        for match in re.finditer(embedded_pattern, struct_content, re.MULTILINE):
            embedded.append(match.group(1))
        
        return embedded
    
    def _find_struct_methods(self, content: str, struct_name: str) -> List[Dict[str, str]]:
        """Find all methods for a given struct.
        
        Args:
            content: Full file content
            struct_name: Name of the struct
            
        Returns:
            List of method information
        """
        methods = []
        
        # Method pattern with receiver
        method_pattern = rf'func\s+\([^)]*\*?{struct_name}[^)]*\)\s+(\w+)\s*\([^)]*\)'
        
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            methods.append({
                'name': method_name,
                'line': content[:match.start()].count('\n') + 1,
                'is_exported': method_name[0].isupper()
            })
        
        return methods
    
    def _extract_interface_methods(self, interface_content: str) -> List[Dict[str, str]]:
        """Extract method signatures from interface definition.
        
        Args:
            interface_content: Content between interface braces
            
        Returns:
            List of method signatures
        """
        methods = []
        
        # Method signature pattern
        method_pattern = r'^\s*(\w+)\s*\([^)]*\)(?:\s*(?:\([^)]*\)|[^(\n]+))?'
        
        for match in re.finditer(method_pattern, interface_content, re.MULTILINE):
            method_name = match.group(1)
            if method_name:  # Filter out empty matches
                methods.append({
                    'name': method_name,
                    'signature': match.group(0).strip(),
                    'is_exported': method_name[0].isupper()
                })
        
        return methods
    
    def _count_code_lines(self, content: str) -> int:
        """Count non-empty, non-comment lines of code.
        
        Args:
            content: Go source code
            
        Returns:
            Number of code lines
        """
        count = 0
        in_multiline_comment = False
        
        for line in content.split('\n'):
            stripped = line.strip()
            
            # Handle multiline comments
            if '/*' in line:
                in_multiline_comment = True
                # Check if comment ends on same line
                if '*/' in line:
                    in_multiline_comment = False
                    # Count if there's code after comment
                    after_comment = line.split('*/')[1].strip()
                    if after_comment and not after_comment.startswith('//'):
                        count += 1
                continue
            
            if in_multiline_comment:
                if '*/' in line:
                    in_multiline_comment = False
                    # Count if there's code after comment
                    after_comment = line.split('*/')[1].strip()
                    if after_comment and not after_comment.startswith('//'):
                        count += 1
                continue
            
            # Skip empty lines and single-line comments
            if stripped and not stripped.startswith('//'):
                count += 1
        
        return count
    
    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines in Go code.
        
        Args:
            content: Go source code
            
        Returns:
            Number of comment lines
        """
        count = 0
        in_multiline_comment = False
        
        for line in content.split('\n'):
            stripped = line.strip()
            
            # Single-line comments
            if stripped.startswith('//'):
                count += 1
                continue
            
            # Multi-line comments
            if '/*' in line:
                count += 1
                in_multiline_comment = True
                if '*/' in line:
                    in_multiline_comment = False
                continue
            
            if in_multiline_comment:
                count += 1
                if '*/' in line:
                    in_multiline_comment = False
        
        return count