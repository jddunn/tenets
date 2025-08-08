"""Base abstract class for language-specific code analyzers.

This module provides the abstract base class that all language-specific
analyzers must implement. It defines the common interface for extracting
imports, exports, structure, and calculating complexity metrics.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics
)


class LanguageAnalyzer(ABC):
    """Abstract base class for language-specific analyzers.
    
    Each language analyzer must implement this interface to provide
    language-specific analysis capabilities. This ensures a consistent
    API across all language analyzers while allowing for language-specific
    implementation details.
    
    Attributes:
        language_name: Name of the programming language
        file_extensions: List of file extensions this analyzer handles
    """
    
    language_name: str = "unknown"
    file_extensions: List[str] = []
    
    @abstractmethod
    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract import statements from source code.
        
        This method should identify and extract all import/include/require
        statements from the source code, including their type, location,
        and whether they are relative imports.
        
        Args:
            content: Source code content as string
            file_path: Path to the file being analyzed
            
        Returns:
            List of ImportInfo objects containing:
                - module: The imported module/package name
                - alias: Any alias assigned to the import
                - line: Line number of the import
                - type: Type of import (e.g., 'import', 'from', 'require')
                - is_relative: Whether this is a relative import
                - Additional language-specific fields
        
        Examples:
            Python: import os, from datetime import datetime
            JavaScript: import React from 'react', const fs = require('fs')
            Go: import "fmt", import _ "database/sql"
        """
        pass
    
    @abstractmethod
    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported symbols from source code.
        
        This method should identify all symbols (functions, classes, variables)
        that are exported from the module and available for use by other modules.
        
        Args:
            content: Source code content as string
            file_path: Path to the file being analyzed
            
        Returns:
            List of dictionaries containing:
                - name: Name of the exported symbol
                - type: Type of export (e.g., 'function', 'class', 'variable')
                - line: Line number where the export is defined
                - Additional language-specific metadata
        
        Examples:
            Python: __all__ = ['func1', 'Class1'], public functions/classes
            JavaScript: export default App, export { util1, util2 }
            Go: Capitalized functions/types are exported
        """
        pass
    
    @abstractmethod
    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract code structure from source file.
        
        This method should parse the source code and extract structural
        elements like classes, functions, methods, variables, constants,
        and other language-specific constructs.
        
        Args:
            content: Source code content as string
            file_path: Path to the file being analyzed
            
        Returns:
            CodeStructure object containing:
                - classes: List of ClassInfo objects
                - functions: List of FunctionInfo objects
                - variables: List of variable definitions
                - constants: List of constant definitions
                - interfaces: List of interface definitions (if applicable)
                - Additional language-specific structures
        
        Note:
            The depth of extraction depends on the language's parsing
            capabilities. AST-based parsing provides more detail than
            regex-based parsing.
        """
        pass
    
    @abstractmethod
    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for the source code.
        
        This method should calculate various complexity metrics including
        cyclomatic complexity, cognitive complexity, and other relevant
        metrics for understanding code complexity and maintainability.
        
        Args:
            content: Source code content as string
            file_path: Path to the file being analyzed
            
        Returns:
            ComplexityMetrics object containing:
                - cyclomatic: McCabe cyclomatic complexity
                - cognitive: Cognitive complexity score
                - halstead: Halstead complexity metrics (if calculated)
                - line_count: Total number of lines
                - function_count: Number of functions/methods
                - class_count: Number of classes
                - max_depth: Maximum nesting depth
                - maintainability_index: Maintainability index score
                - Additional language-specific metrics
        
        Complexity Calculation:
            Cyclomatic: Number of linearly independent paths
            Cognitive: Measure of how difficult code is to understand
            Halstead: Based on operators and operands count
        """
        pass
    
    def analyze(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Run complete analysis on source file.
        
        This method orchestrates all analysis methods to provide a complete
        analysis of the source file. It can be overridden by specific
        analyzers if they need custom orchestration logic.
        
        Args:
            content: Source code content as string
            file_path: Path to the file being analyzed
            
        Returns:
            Dictionary containing all analysis results:
                - imports: List of ImportInfo objects
                - exports: List of export dictionaries
                - structure: CodeStructure object
                - complexity: ComplexityMetrics object
                - Additional analysis results
        
        Note:
            Subclasses can override this method to add language-specific
            analysis steps or modify the analysis pipeline.
        """
        return {
            'imports': self.extract_imports(content, file_path),
            'exports': self.extract_exports(content, file_path),
            'structure': self.extract_structure(content, file_path),
            'complexity': self.calculate_complexity(content, file_path)
        }
    
    def supports_file(self, file_path: Path) -> bool:
        """Check if this analyzer supports the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this analyzer can handle the file, False otherwise
        """
        return file_path.suffix.lower() in self.file_extensions
    
    def get_language_info(self) -> Dict[str, Any]:
        """Get information about the language this analyzer supports.
        
        Returns:
            Dictionary containing:
                - name: Language name
                - extensions: Supported file extensions
                - features: List of supported analysis features
        """
        return {
            'name': self.language_name,
            'extensions': self.file_extensions,
            'features': [
                'imports',
                'exports', 
                'structure',
                'complexity'
            ]
        }