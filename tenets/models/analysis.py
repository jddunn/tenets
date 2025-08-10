"""Code analysis data models used by the analyzer system.

This module contains all data structures used by the code analysis subsystem,
including file analysis results, project metrics, and dependency graphs.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ImportInfo:
    """Information about an import statement in code.

    Represents imports across different languages (import, require, include, use).

    Attributes:
        module: The imported module/package name
        alias: Any alias assigned to the import
        line: Line number where import appears
        type: Type of import (import, from, require, include)
        is_relative: Whether this is a relative import
        level: Relative import level (Python), 0 for absolute
        from_module: Module specified in a 'from X import ...' statement
    """

    module: str
    alias: Optional[str] = None
    line: int = 0
    type: str = "import"
    is_relative: bool = False
    # Compatibility: some analyzers provide 'level' for Python relative imports
    level: int = 0
    # Additional metadata for 'from' imports
    from_module: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all import information
        """
        return asdict(self)


@dataclass
class ComplexityMetrics:
    """Code complexity metrics for analysis.

    Contains various complexity measurements used to assess code quality
    and maintainability.

    Attributes:
        cyclomatic: McCabe cyclomatic complexity
        cognitive: Cognitive complexity score
        halstead_volume: Halstead volume metric
        halstead_difficulty: Halstead difficulty metric
        maintainability_index: Maintainability index (0-100)
        line_count: Total number of lines
        function_count: Number of functions
        class_count: Number of classes
        max_depth: Maximum nesting depth
        comment_ratio: Ratio of comments to code
        code_lines: Number of actual code lines
        comment_lines: Number of comment lines
        character_count: Total number of characters
        key_count: Number of key/value pairs (for config files)
        section_count: Number of sections (for structured files)
        tag_count: Number of tags (for markup languages)
        header_count: Number of headers (for document files)
        column_count: Number of columns (for tabular data)
        row_count: Number of rows (for tabular data)
    """

    cyclomatic: int = 1
    cognitive: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    maintainability_index: float = 100.0
    line_count: int = 0
    function_count: int = 0
    class_count: int = 0
    max_depth: int = 0
    comment_ratio: float = 0.0
    code_lines: int = 0
    comment_lines: int = 0
    character_count: int = 0
    key_count: int = 0
    section_count: int = 0
    tag_count: int = 0
    header_count: int = 0
    column_count: int = 0
    row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all metrics
        """
        return asdict(self)


@dataclass
class FunctionInfo:
    """Information about a function or method.

    Represents functions, methods, procedures across languages.

    Attributes:
        name: Function/method name
        line_start: Starting line number
        line_end: Ending line number
        parameters: List of parameter names
        complexity: Cyclomatic complexity of the function
        line: Compatibility alias for line_start
        end_line: Compatibility alias for line_end
        is_toplevel: Whether function is top-level (for some analyzers)
        args: Argument strings with type hints (analyzer compatibility)
        decorators: Decorators applied to the function
        is_async: Whether the function is async
        docstring: Function docstring
        return_type: Return type annotation
    """

    name: str
    line_start: int = 0
    line_end: int = 0
    parameters: List[str] = field(default_factory=list)
    complexity: int = 1
    # Compatibility fields accepted by analyzers/tests
    line: int = 0
    end_line: int = 0
    is_toplevel: bool = False
    # Extended optional fields
    args: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    docstring: Optional[str] = None
    return_type: Optional[str] = None
    is_constructor: bool = False
    is_abstract: bool = False
    is_static: bool = False
    is_class: bool = False
    is_property: bool = False
    is_private: bool = False

    def __post_init__(self):
        # Map compatibility fields to canonical ones when provided
        if not self.line_start and self.line:
            self.line_start = self.line
        if not self.line_end and self.end_line:
            self.line_end = self.end_line

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing function information
        """
        return asdict(self)


@dataclass
class ClassInfo:
    """Information about a class or similar construct.

    Represents classes, structs, interfaces across languages.

    Attributes:
        name: Class/struct/interface name
        line_start: Starting line number
        line_end: Ending line number
        methods: List of methods in the class
        base_classes: List of base/parent class names
        line: Compatibility alias for line_start
        decorators: Decorator names applied to the class
        docstring: Class docstring
        is_abstract: Whether class is abstract
        metaclass: Metaclass name
        attributes: Collected class attributes
        end_line: Compatibility alias for line_end
        bases: Compatibility alias accepted by some analyzers/tests
    """

    name: str
    line_start: int = 0
    line_end: int = 0
    methods: List[FunctionInfo] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    # Compatibility alias used in some tests/analyzers
    line: int = 0
    # Extended optional fields used by analyzers
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False
    metaclass: Optional[str] = None
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    # Additional compatibility to accept `end_line` in constructor
    end_line: int = 0
    # Accept legacy/alternate parameter name for base classes
    bases: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.line_start and self.line:
            self.line_start = self.line
        if not self.line_end and self.end_line:
            self.line_end = self.end_line
        # Map compatibility alias `bases` -> `base_classes` and vice versa
        if not self.base_classes and self.bases:
            self.base_classes = list(self.bases)
        elif self.base_classes and not self.bases:
            self.bases = list(self.base_classes)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing class information with serialized methods
        """
        data = asdict(self)
        # Keep methods serialized
        data["methods"] = [m.to_dict() if hasattr(m, "to_dict") else m for m in self.methods]
        return data


@dataclass
class CodeStructure:
    """Represents the structure of a code file.

    Contains organized information about code elements found in a file.

    Attributes:
        classes: List of classes in the file
        functions: List of standalone functions
        imports: List of import statements
        file_type: Type of the file (e.g., script, module, package)
        sections: List of sections or blocks in the code
        variables: List of variables used
        constants: List of constants
        todos: List of TODO comments or annotations
        block_count: Total number of code blocks
        indent_levels: Indentation levels used in the code
        type_aliases: List of type alias definitions (Python 3.10+)
    """

    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    file_type: str = "text"
    sections: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    block_count: int = 0
    indent_levels: Dict[str, Any] = field(default_factory=dict)
    type_aliases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all structural information
        """
        base = {
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "imports": [i.to_dict() for i in self.imports],
        }
        # Include extended fields
        base.update(
            {
                "file_type": self.file_type,
                "sections": self.sections,
                "variables": self.variables,
                "constants": self.constants,
                "todos": self.todos,
                "block_count": self.block_count,
                "indent_levels": self.indent_levels,
                "type_aliases": self.type_aliases,
            }
        )
        return base


@dataclass
class FileAnalysis:
    """Complete analysis results for a single file.

    Contains all information extracted from analyzing a source code file,
    including structure, complexity, and metadata.

    Attributes:
        path: File path
        content: File content
        size: File size in bytes
        lines: Number of lines
        language: Programming language
        file_name: Name of the file
        file_extension: File extension
        last_modified: Last modification time
        hash: Content hash
        imports: List of imports
        exports: List of exports
        structure: Code structure information
        complexity: Complexity metrics
        classes: List of classes (convenience accessor)
        functions: List of functions (convenience accessor)
        keywords: Extracted keywords
        relevance_score: Relevance score for ranking
        quality_score: Code quality score
        error: Any error encountered during analysis
    """

    path: str
    content: str = ""
    size: int = 0
    lines: int = 0
    language: str = "unknown"
    file_name: str = ""
    file_extension: str = ""
    last_modified: Optional[datetime] = None
    hash: Optional[str] = None

    # Analysis results
    imports: List[ImportInfo] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)
    structure: Optional[CodeStructure] = None
    complexity: Optional[ComplexityMetrics] = None
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Git information
    git_info: Optional[Dict[str, Any]] = None

    # Ranking/scoring
    relevance_score: float = 0.0
    quality_score: float = 0.0

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all file analysis data
        """
        data = asdict(self)
        if self.last_modified:
            data["last_modified"] = self.last_modified.isoformat()
        if self.structure:
            data["structure"] = self.structure.to_dict()
        if self.complexity:
            data["complexity"] = self.complexity.to_dict()
        data["imports"] = [i.to_dict() for i in self.imports]
        data["classes"] = [c.to_dict() for c in self.classes]
        data["functions"] = [f.to_dict() for f in self.functions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileAnalysis":
        """Create FileAnalysis from dictionary.

        Args:
            data: Dictionary containing file analysis data

        Returns:
            FileAnalysis instance
        """
        # Handle datetime conversion
        if "last_modified" in data and isinstance(data["last_modified"], str):
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])

        # Handle nested objects
        if "imports" in data:
            data["imports"] = [
                ImportInfo(**i) if isinstance(i, dict) else i for i in data["imports"]
            ]
        if "classes" in data:
            data["classes"] = [
                ClassInfo(**c) if isinstance(c, dict) else c for c in data["classes"]
            ]
        if "functions" in data:
            data["functions"] = [
                FunctionInfo(**f) if isinstance(f, dict) else f for f in data["functions"]
            ]
        if "structure" in data and isinstance(data["structure"], dict):
            # Reconstruct CodeStructure
            structure_data = data["structure"]
            if "classes" in structure_data:
                structure_data["classes"] = [
                    ClassInfo(**c) if isinstance(c, dict) else c for c in structure_data["classes"]
                ]
            if "functions" in structure_data:
                structure_data["functions"] = [
                    FunctionInfo(**f) if isinstance(f, dict) else f
                    for f in structure_data["functions"]
                ]
            if "imports" in structure_data:
                structure_data["imports"] = [
                    ImportInfo(**i) if isinstance(i, dict) else i for i in structure_data["imports"]
                ]
            data["structure"] = CodeStructure(**structure_data)
        if "complexity" in data and isinstance(data["complexity"], dict):
            data["complexity"] = ComplexityMetrics(**data["complexity"])

        return cls(**data)


@dataclass
class DependencyGraph:
    """Represents project dependency graph.

    Tracks dependencies between files and modules in the project.

    Attributes:
        nodes: Dictionary of node ID to node data
        edges: List of edges (from_id, to_id, edge_data)
        cycles: List of detected dependency cycles
    """

    nodes: Dict[str, Any] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)

    def add_node(self, node_id: str, data: Any) -> None:
        """Add a node to the dependency graph.

        Args:
            node_id: Unique identifier for the node
            data: Node data (typically FileAnalysis)
        """
        self.nodes[node_id] = data

    def add_edge(self, from_id: str, to_id: str, import_info: Optional[ImportInfo] = None) -> None:
        """Add an edge representing a dependency.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            import_info: Optional import information
        """
        self.edges.append((from_id, to_id, import_info))

    def calculate_metrics(self) -> None:
        """Calculate graph metrics like centrality and cycles.

        Updates internal metrics based on current graph structure.
        """
        # This would calculate various graph metrics
        # For now, just detect simple cycles
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing graph structure
        """
        return {
            "nodes": list(self.nodes.keys()),
            "edges": [(e[0], e[1]) for e in self.edges],
            "cycles": self.cycles,
        }


@dataclass
class ProjectAnalysis:
    """Analysis results for an entire project.

    Aggregates file-level analysis into project-wide metrics and insights.

    Attributes:
        path: Project root path
        name: Project name
        files: List of analyzed files
        total_files: Total number of files
        analyzed_files: Number of successfully analyzed files
        failed_files: Number of files that failed analysis
        total_lines: Total lines of code
        total_code_lines: Total non-blank, non-comment lines
        total_comment_lines: Total comment lines
        average_complexity: Average cyclomatic complexity
        total_functions: Total number of functions
        total_classes: Total number of classes
        languages: Language distribution (language -> file count)
        language_distribution: Percentage distribution of languages
        frameworks: Detected frameworks
        project_type: Type of project (web, library, cli, etc.)
        dependency_graph: Project dependency graph
        summary: Project summary dictionary
    """

    path: str
    name: str
    files: List[FileAnalysis] = field(default_factory=list)
    total_files: int = 0
    analyzed_files: int = 0
    failed_files: int = 0

    # Aggregate metrics
    total_lines: int = 0
    total_code_lines: int = 0
    total_comment_lines: int = 0
    average_complexity: float = 0.0
    total_functions: int = 0
    total_classes: int = 0

    # Language info
    languages: Dict[str, int] = field(default_factory=dict)
    language_distribution: Dict[str, float] = field(default_factory=dict)

    # Project info
    frameworks: List[str] = field(default_factory=list)
    project_type: str = "unknown"
    dependency_graph: Optional[DependencyGraph] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all project analysis data
        """
        data = asdict(self)
        data["files"] = [f.to_dict() for f in self.files]
        if self.dependency_graph:
            data["dependency_graph"] = self.dependency_graph.to_dict()
        return data


@dataclass
class AnalysisReport:
    """Report generated from analysis results.

    Formatted output of analysis results for different consumers.

    Attributes:
        timestamp: When report was generated
        format: Report format (json, html, markdown, csv)
        content: Report content
        statistics: Analysis statistics
        output_path: Where report was saved (if applicable)
    """

    timestamp: datetime = field(default_factory=datetime.now)
    format: str = "json"
    content: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing report information
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "format": self.format,
            "statistics": self.statistics,
            "output_path": self.output_path,
        }
