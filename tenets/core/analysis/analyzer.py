"""Main code analyzer orchestrator for Tenets.

This module coordinates language-specific analyzers and provides a unified
interface for analyzing source code files. It handles analyzer selection,
caching, parallel processing, and fallback strategies.
"""

import os
import hashlib
import json
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Callable, Union
from datetime import datetime, timedelta

# Replacement imports after directory move
from .base import LanguageAnalyzer
from .python_analyzer import PythonAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer
from .go_analyzer import GoAnalyzer
from .java_analyzer import JavaAnalyzer
from .cpp_analyzer import CppAnalyzer
from .ruby_analyzer import RubyAnalyzer
from .php_analyzer import PhpAnalyzer
from .rust_analyzer import RustAnalyzer
from .generic_analyzer import GenericAnalyzer

from tenets.config import TenetsConfig
from tenets.models.analysis import (
    FileAnalysis,
    ProjectAnalysis,
    ComplexityMetrics,
    CodeStructure,
    ImportInfo,
    DependencyGraph,
    AnalysisReport,
)
from tenets.storage.cache import AnalysisCache
from tenets.utils.logger import get_logger


class CodeAnalyzer:
    """Main code analysis orchestrator.

    Coordinates language-specific analyzers and provides a unified interface
    for analyzing source code files. Handles caching, parallel processing,
    analyzer selection, and fallback strategies.

    Attributes:
        config: TenetsConfig instance for configuration
        logger: Logger instance for logging
        cache: AnalysisCache for caching analysis results
        analyzers: Dictionary mapping file extensions to analyzer instances
        stats: Analysis statistics and metrics
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the code analyzer.

        Args:
            config: Tenets configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize cache if enabled
        self.cache = None
        if config.cache.enabled:
            self.cache = AnalysisCache(config.cache.directory)
            self.logger.info(f"Cache initialized at {config.cache.directory}")

        # Initialize language analyzers
        self.analyzers = self._initialize_analyzers()

        # Thread pool for parallel analysis
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.scanner.workers)

        # Analysis statistics
        self.stats = {
            "files_analyzed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_time": 0,
            "languages": {},
        }

        self.logger.info(f"CodeAnalyzer initialized with {len(self.analyzers)} language analyzers")

    def _initialize_analyzers(self) -> Dict[str, LanguageAnalyzer]:
        """Initialize all language-specific analyzers.

        Returns:
            Dictionary mapping file extensions to analyzer instances
        """
        analyzers = {}

        # Initialize each language analyzer
        language_analyzers = [
            PythonAnalyzer(),
            JavaScriptAnalyzer(),
            GoAnalyzer(),
            JavaAnalyzer(),
            CppAnalyzer(),
            RubyAnalyzer(),
            PhpAnalyzer(),
            RustAnalyzer(),
        ]

        # Map extensions to analyzers
        for analyzer in language_analyzers:
            for ext in analyzer.file_extensions:
                analyzers[ext.lower()] = analyzer

        # Add generic analyzer as fallback
        generic = GenericAnalyzer()
        for ext in [
            ".txt",
            ".md",
            ".yml",
            ".yaml",
            ".json",
            ".xml",
            ".html",
            ".css",
            ".scss",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".dockerfile",
            ".makefile",
            ".cmake",
        ]:
            if ext not in analyzers:
                analyzers[ext] = generic

        self.logger.debug(f"Initialized analyzers for {len(analyzers)} file extensions")

        return analyzers

    def analyze_file(
        self,
        file_path: Path,
        deep: bool = False,
        extract_keywords: bool = True,
        use_cache: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> FileAnalysis:
        """Analyze a single file.

        Performs language-specific analysis on a file, extracting imports,
        structure, complexity metrics, and other relevant information.

        Args:
            file_path: Path to the file to analyze
            deep: Whether to perform deep analysis (AST parsing, etc.)
            extract_keywords: Whether to extract keywords from content
            use_cache: Whether to use cached results if available
            progress_callback: Optional callback for progress updates

        Returns:
            FileAnalysis object with complete analysis results

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file cannot be read
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check cache first
        if use_cache and self.cache:
            cache_key = self._get_cache_key(file_path)
            cached = self.cache.get(cache_key)

            if cached and self._is_cache_valid(cached, file_path):
                self.stats["cache_hits"] += 1
                self.logger.debug(f"Cache hit for {file_path}")

                if progress_callback:
                    progress_callback("cache_hit", file_path)

                return FileAnalysis.from_dict(cached["analysis"])
            else:
                self.stats["cache_misses"] += 1

        self.logger.debug(f"Analyzing file: {file_path}")

        try:
            # Read file content
            content = self._read_file_content(file_path)

            # Create base analysis
            analysis = FileAnalysis(
                path=str(file_path),
                content=content,
                size=file_path.stat().st_size,
                lines=content.count("\n") + 1,
                language=self._detect_language(file_path),
                file_name=file_path.name,
                file_extension=file_path.suffix,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                hash=self._calculate_file_hash(content),
            )

            # Get appropriate analyzer
            analyzer = self._get_analyzer(file_path)

            if analyzer and deep:
                try:
                    # Run language-specific analysis
                    self.logger.debug(f"Running {analyzer.language_name} analyzer on {file_path}")
                    analysis_results = analyzer.analyze(content, file_path)

                    # Update analysis object with results
                    analysis.imports = analysis_results.get("imports", [])
                    analysis.exports = analysis_results.get("exports", [])
                    analysis.structure = analysis_results.get("structure", CodeStructure())
                    analysis.complexity = analysis_results.get("complexity", ComplexityMetrics())

                    # Extract additional information
                    if analysis.structure:
                        analysis.classes = analysis.structure.classes
                        analysis.functions = analysis.structure.functions
                        analysis.modules = getattr(analysis.structure, "modules", [])

                except Exception as e:
                    self.logger.warning(f"Language-specific analysis failed for {file_path}: {e}")
                    analysis.error = str(e)
                    self.stats["errors"] += 1

            # Extract keywords if requested
            if extract_keywords:
                analysis.keywords = self._extract_keywords(content, analysis.language)

            # Add code quality metrics
            analysis.quality_score = self._calculate_quality_score(analysis)

            # Cache the result
            if use_cache and self.cache and not analysis.error:
                cache_data = {
                    "analysis": analysis.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                    "file_mtime": file_path.stat().st_mtime,
                    "file_size": file_path.stat().st_size,
                }
                self.cache.put(cache_key, cache_data)

            # Update statistics
            self.stats["files_analyzed"] += 1
            self.stats["languages"][analysis.language] = (
                self.stats["languages"].get(analysis.language, 0) + 1
            )

            if progress_callback:
                progress_callback("analyzed", file_path)

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            self.stats["errors"] += 1

            return FileAnalysis(
                path=str(file_path),
                error=str(e),
                file_name=file_path.name,
                file_extension=file_path.suffix,
            )

    def analyze_files(
        self,
        file_paths: List[Path],
        deep: bool = False,
        parallel: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> List[FileAnalysis]:
        """Analyze multiple files.

        Args:
            file_paths: List of file paths to analyze
            deep: Whether to perform deep analysis
            parallel: Whether to analyze files in parallel
            progress_callback: Optional callback for progress updates

        Returns:
            List of FileAnalysis objects
        """
        self.logger.info(f"Analyzing {len(file_paths)} files (parallel={parallel})")

        if parallel and len(file_paths) > 1:
            # Parallel analysis
            futures = []
            for file_path in file_paths:
                future = self._executor.submit(
                    self.analyze_file, file_path, deep=deep, progress_callback=progress_callback
                )
                futures.append((future, file_path))

            # Collect results
            results = []
            for future, file_path in futures:
                try:
                    result = future.result(timeout=self.config.scanner.timeout)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    self.logger.warning(f"Analysis timeout for {file_path}")
                    results.append(FileAnalysis(path=str(file_path), error="Analysis timeout"))
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {file_path}: {e}")
                    results.append(FileAnalysis(path=str(file_path), error=str(e)))

            return results
        else:
            # Sequential analysis
            results = []
            for i, file_path in enumerate(file_paths):
                result = self.analyze_file(file_path, deep=deep)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(file_paths))

            return results

    def analyze_project(
        self,
        project_path: Path,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        deep: bool = True,
        parallel: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> ProjectAnalysis:
        """Analyze an entire project.

        Args:
            project_path: Path to the project root
            patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_patterns: File patterns to exclude
            deep: Whether to perform deep analysis
            parallel: Whether to analyze files in parallel
            progress_callback: Optional callback for progress updates

        Returns:
            ProjectAnalysis object with complete project analysis
        """
        self.logger.info(f"Analyzing project: {project_path}")

        # Collect files to analyze
        files = self._collect_project_files(project_path, patterns, exclude_patterns)

        self.logger.info(f"Found {len(files)} files to analyze")

        # Analyze all files
        file_analyses = self.analyze_files(
            files, deep=deep, parallel=parallel, progress_callback=progress_callback
        )

        # Build project analysis
        project_analysis = ProjectAnalysis(
            path=str(project_path),
            name=project_path.name,
            files=file_analyses,
            total_files=len(file_analyses),
            analyzed_files=len([f for f in file_analyses if not f.error]),
            failed_files=len([f for f in file_analyses if f.error]),
        )

        # Calculate project-level metrics
        self._calculate_project_metrics(project_analysis)

        # Build dependency graph
        project_analysis.dependency_graph = self._build_dependency_graph(file_analyses)

        # Detect project type and framework
        project_analysis.project_type = self._detect_project_type(project_path, file_analyses)
        project_analysis.frameworks = self._detect_frameworks(file_analyses)

        # Generate summary
        project_analysis.summary = self._generate_project_summary(project_analysis)

        return project_analysis

    def generate_report(
        self,
        analysis: Union[FileAnalysis, ProjectAnalysis, List[FileAnalysis]],
        format: str = "json",
        output_path: Optional[Path] = None,
    ) -> AnalysisReport:
        """Generate an analysis report.

        Args:
            analysis: Analysis results to report on
            format: Report format ('json', 'html', 'markdown', 'csv')
            output_path: Optional path to save the report

        Returns:
            AnalysisReport object
        """
        self.logger.info(f"Generating {format} report")

        report = AnalysisReport(
            timestamp=datetime.now(), format=format, statistics=self.stats.copy()
        )

        # Generate report content based on format
        if format == "json":
            report.content = self._generate_json_report(analysis)
        elif format == "html":
            report.content = self._generate_html_report(analysis)
        elif format == "markdown":
            report.content = self._generate_markdown_report(analysis)
        elif format == "csv":
            report.content = self._generate_csv_report(analysis)
        else:
            raise ValueError(f"Unsupported report format: {format}")

        # Save report if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format in ["json", "csv"]:
                output_path.write_text(report.content)
            else:
                output_path.write_text(report.content, encoding="utf-8")

            self.logger.info(f"Report saved to {output_path}")
            report.output_path = str(output_path)

        return report

    def _get_analyzer(self, file_path: Path) -> Optional[LanguageAnalyzer]:
        """Get the appropriate analyzer for a file.

        Args:
            file_path: Path to the file

        Returns:
            LanguageAnalyzer instance or None
        """
        extension = file_path.suffix.lower()

        # Check direct mapping
        if extension in self.analyzers:
            return self.analyzers[extension]

        # Check alternative extensions
        if file_path.name == "Dockerfile":
            return self.analyzers.get(".dockerfile")
        elif file_path.name == "Makefile":
            return self.analyzers.get(".makefile")
        elif file_path.name == "CMakeLists.txt":
            return self.analyzers.get(".cmake")

        # Default to generic analyzer
        return GenericAnalyzer()

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file.

        Args:
            file_path: Path to the file

        Returns:
            Language name string
        """
        analyzer = self._get_analyzer(file_path)
        if analyzer:
            return analyzer.language_name

        # Fallback to extension-based detection
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".rb": "ruby",
            ".php": "php",
            ".rs": "rust",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "shell",
        }

        return extension_map.get(file_path.suffix.lower(), "unknown")

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding detection.

        Args:
            file_path: Path to the file

        Returns:
            File content as string

        Raises:
            UnicodeDecodeError: If file cannot be decoded
        """
        # Try different encodings
        encodings = [self.config.scanner.encoding, "utf-8", "latin-1", "cp1252", "utf-16"]

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        # If all encodings fail, read as binary and decode with errors='ignore'
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a file.

        Args:
            file_path: Path to the file

        Returns:
            Cache key string
        """
        # Include file path and modification time in key
        key_data = f"{file_path}:{file_path.stat().st_mtime}:{file_path.stat().st_size}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_cache_valid(self, cached_data: Dict, file_path: Path) -> bool:
        """Check if cached data is still valid.

        Args:
            cached_data: Cached analysis data
            file_path: Path to the file

        Returns:
            True if cache is valid, False otherwise
        """
        # Check file modification time
        if "file_mtime" in cached_data:
            if file_path.stat().st_mtime != cached_data["file_mtime"]:
                return False

        # Check file size
        if "file_size" in cached_data:
            if file_path.stat().st_size != cached_data["file_size"]:
                return False

        # Check cache age
        if "timestamp" in cached_data:
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            max_age = timedelta(hours=self.config.cache.max_age_hours)
            if datetime.now() - cache_time > max_age:
                return False

        return True

    def _calculate_file_hash(self, content: str) -> str:
        """Calculate hash of file content.

        Args:
            content: File content

        Returns:
            SHA256 hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _extract_keywords(self, content: str, language: str) -> List[str]:
        """Extract keywords from content.

        Args:
            content: File content
            language: Programming language

        Returns:
            List of keywords
        """
        import re
        from collections import Counter

        # Remove comments based on language
        if language == "python":
            content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r'""".*?"""', "", content, flags=re.DOTALL)
            content = re.sub(r"'''.*?'''", "", content, flags=re.DOTALL)
        elif language in ["javascript", "java", "cpp", "c", "go", "rust", "php"]:
            content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
            content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

        # Extract words
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", content)

        # Count frequency
        word_freq = Counter(words)

        # Filter common programming keywords
        common_keywords = {
            "if",
            "else",
            "elif",
            "for",
            "while",
            "do",
            "switch",
            "case",
            "def",
            "function",
            "func",
            "class",
            "struct",
            "interface",
            "return",
            "break",
            "continue",
            "pass",
            "yield",
            "import",
            "from",
            "export",
            "require",
            "include",
            "use",
            "public",
            "private",
            "protected",
            "static",
            "const",
            "let",
            "var",
            "true",
            "false",
            "null",
            "none",
            "undefined",
            "nil",
            "self",
            "this",
            "super",
            "try",
            "catch",
            "except",
            "finally",
            "throw",
            "throws",
            "new",
            "delete",
            "typeof",
            "instanceof",
            "and",
            "or",
            "not",
            "in",
            "is",
            "int",
            "float",
            "str",
            "bool",
            "string",
            "number",
            "boolean",
            "void",
        }

        # Get top keywords (excluding common ones)
        keywords = [
            word
            for word, count in word_freq.most_common(50)
            if word.lower() not in common_keywords and len(word) > 2
        ]

        return keywords[:20]  # Return top 20

    def _calculate_quality_score(self, analysis: FileAnalysis) -> float:
        """Calculate code quality score for a file.

        Args:
            analysis: FileAnalysis object

        Returns:
            Quality score between 0 and 100
        """
        if not analysis.complexity:
            return 50.0  # Default score if no complexity metrics

        score = 100.0

        # Penalize high complexity
        if analysis.complexity.cyclomatic:
            if analysis.complexity.cyclomatic > 10:
                score -= min(20, (analysis.complexity.cyclomatic - 10) * 2)

        if analysis.complexity.cognitive:
            if analysis.complexity.cognitive > 15:
                score -= min(20, (analysis.complexity.cognitive - 15) * 1.5)

        # Penalize deep nesting
        if hasattr(analysis.complexity, "max_depth"):
            if analysis.complexity.max_depth > 4:
                score -= min(10, (analysis.complexity.max_depth - 4) * 2)

        # Reward good comment ratio
        if hasattr(analysis.complexity, "comment_ratio"):
            if analysis.complexity.comment_ratio > 0.1:
                score += min(10, analysis.complexity.comment_ratio * 50)

        # Consider maintainability index
        if hasattr(analysis.complexity, "maintainability_index"):
            mi_factor = analysis.complexity.maintainability_index / 100
            score = score * 0.7 + analysis.complexity.maintainability_index * 0.3

        return max(0, min(100, score))

    def _collect_project_files(
        self,
        project_path: Path,
        patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Collect files to analyze in a project.

        Args:
            project_path: Project root path
            patterns: Include patterns
            exclude_patterns: Exclude patterns

        Returns:
            List of file paths to analyze
        """
        import fnmatch

        files = []

        # Default patterns if none provided
        if not patterns:
            patterns = ["*" + ext for ext in self.analyzers.keys()]

        # Default exclude patterns
        default_excludes = [
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "__pycache__",
            "node_modules",
            "vendor",
            ".git",
            ".svn",
            "dist",
            "build",
            "target",
            ".venv",
            "venv",
            "*.min.js",
            "*.min.css",
            "*.map",
        ]

        if exclude_patterns:
            exclude_patterns.extend(default_excludes)
        else:
            exclude_patterns = default_excludes

        # Walk the project directory
        for root, dirs, filenames in os.walk(project_path):
            root_path = Path(root)

            # Filter directories
            dirs[:] = [
                d
                for d in dirs
                if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)
            ]

            # Filter files
            for filename in filenames:
                file_path = root_path / filename

                # Check if file matches include patterns
                if any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
                    # Check if file doesn't match exclude patterns
                    if not any(fnmatch.fnmatch(filename, pattern) for pattern in exclude_patterns):
                        files.append(file_path)

        return files

    def _calculate_project_metrics(self, project: ProjectAnalysis) -> None:
        """Calculate aggregate metrics for a project.

        Args:
            project: ProjectAnalysis object to update
        """
        total_lines = 0
        total_code_lines = 0
        total_comment_lines = 0
        total_complexity = 0
        total_functions = 0
        total_classes = 0
        languages = {}

        for file_analysis in project.files:
            if file_analysis.error:
                continue

            total_lines += file_analysis.lines

            if file_analysis.complexity:
                if hasattr(file_analysis.complexity, "code_lines"):
                    total_code_lines += file_analysis.complexity.code_lines
                if hasattr(file_analysis.complexity, "comment_lines"):
                    total_comment_lines += file_analysis.complexity.comment_lines
                if file_analysis.complexity.cyclomatic:
                    total_complexity += file_analysis.complexity.cyclomatic
                if hasattr(file_analysis.complexity, "function_count"):
                    total_functions += file_analysis.complexity.function_count
                if hasattr(file_analysis.complexity, "class_count"):
                    total_classes += file_analysis.complexity.class_count

            # Count languages
            languages[file_analysis.language] = languages.get(file_analysis.language, 0) + 1

        # Update project metrics
        project.total_lines = total_lines
        project.total_code_lines = total_code_lines
        project.total_comment_lines = total_comment_lines
        project.average_complexity = total_complexity / len(project.files) if project.files else 0
        project.total_functions = total_functions
        project.total_classes = total_classes
        project.languages = languages

        # Calculate language distribution
        total_files = sum(languages.values())
        project.language_distribution = {
            lang: (count / total_files * 100) for lang, count in languages.items()
        }

    def _build_dependency_graph(self, file_analyses: List[FileAnalysis]) -> DependencyGraph:
        """Build dependency graph from file analyses.

        Args:
            file_analyses: List of FileAnalysis objects

        Returns:
            DependencyGraph object
        """
        graph = DependencyGraph()

        # Build file path to analysis mapping
        file_map = {analysis.path: analysis for analysis in file_analyses}

        # Add nodes for each file
        for analysis in file_analyses:
            graph.add_node(analysis.path, analysis)

        # Add edges based on imports
        for analysis in file_analyses:
            if not analysis.imports:
                continue

            for import_info in analysis.imports:
                # Try to resolve import to a file in the project
                target_file = self._resolve_import(import_info, analysis.path, file_map)
                if target_file:
                    graph.add_edge(analysis.path, target_file, import_info)

        # Calculate metrics
        graph.calculate_metrics()

        return graph

    def _resolve_import(
        self, import_info: ImportInfo, source_file: str, file_map: Dict[str, FileAnalysis]
    ) -> Optional[str]:
        """Resolve an import to a file in the project.

        Args:
            import_info: Import information
            source_file: Source file path
            file_map: Mapping of file paths to analyses

        Returns:
            Target file path or None
        """
        # This is a simplified implementation
        # Real implementation would need language-specific resolution

        source_path = Path(source_file)

        # Handle relative imports
        if import_info.is_relative:
            # Try to resolve relative to source file
            module_path = source_path.parent / import_info.module

            # Try different extensions
            for ext in [".py", ".js", ".ts", ".go", ".java", ".rb", ".php", ".rs"]:
                potential_path = module_path.with_suffix(ext)
                if str(potential_path) in file_map:
                    return str(potential_path)

        return None

    def _detect_project_type(self, project_path: Path, file_analyses: List[FileAnalysis]) -> str:
        """Detect the type of project.

        Args:
            project_path: Project root path
            file_analyses: List of file analyses

        Returns:
            Project type string
        """
        # Check for configuration files
        if (project_path / "package.json").exists():
            return "node"
        elif (project_path / "requirements.txt").exists() or (project_path / "setup.py").exists():
            return "python"
        elif (project_path / "go.mod").exists():
            return "go"
        elif (project_path / "pom.xml").exists() or (project_path / "build.gradle").exists():
            return "java"
        elif (project_path / "Cargo.toml").exists():
            return "rust"
        elif (project_path / "Gemfile").exists():
            return "ruby"
        elif (project_path / "composer.json").exists():
            return "php"
        elif (project_path / "CMakeLists.txt").exists():
            return "cmake"

        # Detect by dominant language
        languages = {}
        for analysis in file_analyses:
            languages[analysis.language] = languages.get(analysis.language, 0) + 1

        if languages:
            dominant_language = max(languages, key=languages.get)
            return dominant_language

        return "unknown"

    def _detect_frameworks(self, file_analyses: List[FileAnalysis]) -> List[str]:
        """Detect frameworks used in the project.

        Args:
            file_analyses: List of file analyses

        Returns:
            List of detected framework names
        """
        frameworks = set()

        for analysis in file_analyses:
            if hasattr(analysis.structure, "framework") and analysis.structure.framework:
                frameworks.add(analysis.structure.framework)

        return list(frameworks)

    def _generate_project_summary(self, project: ProjectAnalysis) -> Dict[str, Any]:
        """Generate a summary of the project analysis.

        Args:
            project: ProjectAnalysis object

        Returns:
            Summary dictionary
        """
        return {
            "total_files": project.total_files,
            "analyzed_files": project.analyzed_files,
            "failed_files": project.failed_files,
            "total_lines": project.total_lines,
            "total_code_lines": project.total_code_lines,
            "total_comment_lines": project.total_comment_lines,
            "average_complexity": round(project.average_complexity, 2),
            "total_functions": project.total_functions,
            "total_classes": project.total_classes,
            "languages": project.languages,
            "frameworks": project.frameworks,
            "project_type": project.project_type,
        }

    def _generate_json_report(self, analysis: Any) -> str:
        """Generate JSON report.

        Args:
            analysis: Analysis results

        Returns:
            JSON string
        """
        if isinstance(analysis, (FileAnalysis, ProjectAnalysis)):
            data = analysis.to_dict()
        elif isinstance(analysis, list):
            data = [a.to_dict() for a in analysis]
        else:
            data = analysis

        return json.dumps(data, indent=2, default=str)

    def _generate_html_report(self, analysis: Any) -> str:
        """Generate HTML report.

        Args:
            analysis: Analysis results

        Returns:
            HTML string
        """
        # Simplified HTML generation
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Code Analysis Report</h1>
        """

        if isinstance(analysis, ProjectAnalysis):
            html += f"""
            <h2>Project: {analysis.name}</h2>
            <p>Path: {analysis.path}</p>
            <h3>Summary</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Files</td><td class="metric">{analysis.total_files}</td></tr>
                <tr><td>Analyzed Files</td><td class="metric">{analysis.analyzed_files}</td></tr>
                <tr><td>Total Lines</td><td class="metric">{analysis.total_lines}</td></tr>
                <tr><td>Code Lines</td><td class="metric">{analysis.total_code_lines}</td></tr>
                <tr><td>Comment Lines</td><td class="metric">{analysis.total_comment_lines}</td></tr>
                <tr><td>Average Complexity</td><td class="metric">{analysis.average_complexity:.2f}</td></tr>
                <tr><td>Functions</td><td class="metric">{analysis.total_functions}</td></tr>
                <tr><td>Classes</td><td class="metric">{analysis.total_classes}</td></tr>
            </table>
            """

        html += """
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, analysis: Any) -> str:
        """Generate Markdown report.

        Args:
            analysis: Analysis results

        Returns:
            Markdown string
        """
        md = "# Code Analysis Report\n\n"

        if isinstance(analysis, ProjectAnalysis):
            md += f"## Project: {analysis.name}\n\n"
            md += f"**Path:** {analysis.path}\n\n"
            md += "### Summary\n\n"
            md += "| Metric | Value |\n"
            md += "|--------|-------|\n"
            md += f"| Total Files | {analysis.total_files} |\n"
            md += f"| Analyzed Files | {analysis.analyzed_files} |\n"
            md += f"| Total Lines | {analysis.total_lines} |\n"
            md += f"| Code Lines | {analysis.total_code_lines} |\n"
            md += f"| Comment Lines | {analysis.total_comment_lines} |\n"
            md += f"| Average Complexity | {analysis.average_complexity:.2f} |\n"
            md += f"| Functions | {analysis.total_functions} |\n"
            md += f"| Classes | {analysis.total_classes} |\n"
            md += "\n"

            if analysis.languages:
                md += "### Languages\n\n"
                for lang, count in analysis.languages.items():
                    md += f"- **{lang}**: {count} files\n"
                md += "\n"

            if analysis.frameworks:
                md += "### Frameworks Detected\n\n"
                for framework in analysis.frameworks:
                    md += f"- {framework}\n"
                md += "\n"

        return md

    def _generate_csv_report(self, analysis: Any) -> str:
        """Generate CSV report.

        Args:
            analysis: Analysis results

        Returns:
            CSV string
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        if isinstance(analysis, ProjectAnalysis):
            # Write headers
            writer.writerow(
                ["File", "Language", "Lines", "Complexity", "Functions", "Classes", "Quality Score"]
            )

            # Write file data
            for file_analysis in analysis.files:
                if file_analysis.error:
                    continue

                writer.writerow(
                    [
                        file_analysis.file_name,
                        file_analysis.language,
                        file_analysis.lines,
                        file_analysis.complexity.cyclomatic if file_analysis.complexity else 0,
                        len(file_analysis.functions) if file_analysis.functions else 0,
                        len(file_analysis.classes) if file_analysis.classes else 0,
                        (
                            file_analysis.quality_score
                            if hasattr(file_analysis, "quality_score")
                            else 0
                        ),
                    ]
                )

        return output.getvalue()

    def shutdown(self):
        """Shutdown the analyzer and clean up resources."""
        self._executor.shutdown(wait=True)

        if self.cache:
            self.cache.close()

        self.logger.info("CodeAnalyzer shutdown complete")
        self.logger.info(f"Analysis statistics: {self.stats}")
