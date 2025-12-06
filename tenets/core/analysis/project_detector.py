"""Project type detection and entry point discovery.

This module provides intelligent detection of project types, main entry points,
and project structure based on language analyzers and file patterns.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tenets.utils.logger import get_logger

# Import all language analyzers to access their metadata
from .implementations.cpp_analyzer import CppAnalyzer
from .implementations.csharp_analyzer import CSharpAnalyzer
from .implementations.css_analyzer import CSSAnalyzer
from .implementations.dart_analyzer import DartAnalyzer
from .implementations.gdscript_analyzer import GDScriptAnalyzer
from .implementations.go_analyzer import GoAnalyzer
from .implementations.html_analyzer import HTMLAnalyzer
from .implementations.java_analyzer import JavaAnalyzer
from .implementations.javascript_analyzer import JavaScriptAnalyzer
from .implementations.kotlin_analyzer import KotlinAnalyzer
from .implementations.php_analyzer import PhpAnalyzer
from .implementations.python_analyzer import PythonAnalyzer
from .implementations.ruby_analyzer import RubyAnalyzer
from .implementations.rust_analyzer import RustAnalyzer
from .implementations.scala_analyzer import ScalaAnalyzer
from .implementations.swift_analyzer import SwiftAnalyzer


class ProjectDetector:
    """Detects project type and structure using language analyzers.

    This class leverages the language-specific analyzers to detect project
    types and entry points, avoiding duplication of language-specific knowledge.
    """

    def __init__(self):
        """Initialize project detector with language analyzers."""
        self.logger = get_logger(__name__)

        # Initialize all language analyzers
        self.analyzers = [
            PythonAnalyzer(),
            JavaScriptAnalyzer(),
            JavaAnalyzer(),
            GoAnalyzer(),
            RustAnalyzer(),
            CppAnalyzer(),
            CSharpAnalyzer(),
            RubyAnalyzer(),
            PhpAnalyzer(),
            SwiftAnalyzer(),
            KotlinAnalyzer(),
            ScalaAnalyzer(),
            DartAnalyzer(),
            GDScriptAnalyzer(),
            HTMLAnalyzer(),
            CSSAnalyzer(),
        ]

        # Build dynamic mappings from analyzers
        self._build_mappings()

        # Additional framework patterns not tied to specific languages
        self.FRAMEWORK_PATTERNS = {
            "docker": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
            "kubernetes": ["k8s/", "kubernetes/", "deployment.yaml", "service.yaml"],
            "terraform": ["*.tf", "terraform.tfvars"],
            "ansible": ["ansible.cfg", "playbook.yml", "inventory"],
            "ci_cd": [".github/workflows/", ".gitlab-ci.yml", "Jenkinsfile", ".travis.yml"],
        }

    def _build_mappings(self):
        """Build entry points and indicators from language analyzers."""
        self.ENTRY_POINTS = {}
        self.PROJECT_INDICATORS = {}
        self.EXTENSION_TO_LANGUAGE = {}

        for analyzer in self.analyzers:
            lang = analyzer.language_name

            # Build entry points mapping
            if hasattr(analyzer, "entry_points") and analyzer.entry_points:
                self.ENTRY_POINTS[lang] = analyzer.entry_points

            # Build project indicators mapping
            if hasattr(analyzer, "project_indicators") and analyzer.project_indicators:
                for project_type, indicators in analyzer.project_indicators.items():
                    # Prefix with language for uniqueness
                    key = f"{lang}_{project_type}" if project_type != lang else project_type
                    self.PROJECT_INDICATORS[key] = indicators

            # Build extension to language mapping
            for ext in analyzer.file_extensions:
                self.EXTENSION_TO_LANGUAGE[ext] = lang

    def detect_project_type(self, path: Path) -> Dict[str, any]:
        """Detect project type and main entry points.

        Args:
            path: Root directory to analyze

        Returns:
            Dictionary containing:
                - type: Primary project type
                - languages: List of detected languages
                - frameworks: List of detected frameworks
                - entry_points: List of likely entry point files
                - confidence: Confidence score (0-1)
        """
        path = Path(path)
        if not path.exists():
            return {
                "type": "unknown",
                "languages": [],
                "frameworks": [],
                "entry_points": [],
                "confidence": 0.0,
            }

        # Collect all files
        all_files = []
        for ext in ["*.*", "Dockerfile", "Makefile", "Jenkinsfile"]:
            all_files.extend(path.rglob(ext))

        # Analyze file extensions to detect languages
        extensions = Counter()
        for file in all_files:
            if file.is_file():
                ext = file.suffix.lower()
                if ext:
                    extensions[ext] += 1

        # Determine primary languages based on extensions (aggregate per language for stability)
        language_counts = Counter()
        for ext, count in extensions.items():
            if ext in self.EXTENSION_TO_LANGUAGE:
                language_counts[self.EXTENSION_TO_LANGUAGE[ext]] += count

        analyzer_order = {analyzer.language_name: idx for idx, analyzer in enumerate(self.analyzers)}
        languages = [
            lang
            for lang, _ in sorted(
                language_counts.items(),
                key=lambda item: (
                    -item[1],
                    analyzer_order.get(item[0], len(analyzer_order)),
                    item[0],
                ),
            )
        ]

        # Detect frameworks based on indicators
        frameworks = []
        file_names = {f.name for f in all_files if f.is_file()}
        dir_names = {f.name for f in all_files if f.is_dir()}

        # Check language-specific project indicators
        for project_type, indicators in self.PROJECT_INDICATORS.items():
            for indicator in indicators:
                if indicator in file_names or indicator in dir_names:
                    frameworks.append(project_type)
                    break

        # Check general framework patterns
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if pattern.endswith("/"):
                    # Directory pattern
                    if pattern[:-1] in dir_names:
                        frameworks.append(framework)
                        break
                elif "*" in pattern:
                    # Glob pattern
                    if any(f.match(pattern) for f in all_files if f.is_file()):
                        frameworks.append(framework)
                        break
                else:
                    # File pattern
                    if pattern in file_names:
                        frameworks.append(framework)
                        break

        # Find entry points
        entry_points = self._find_entry_points(path, languages, file_names)

        # Determine primary project type
        project_type = self._determine_project_type(languages, frameworks)

        # Calculate confidence
        confidence = self._calculate_confidence(languages, frameworks, entry_points)

        return {
            "type": project_type,
            "languages": languages[:3],  # Top 3 languages
            "frameworks": list(set(frameworks))[:3],  # Top 3 unique frameworks
            "entry_points": entry_points[:5],  # Top 5 entry points
            "confidence": confidence,
        }

    def _find_entry_points(self, path: Path, languages: List[str], file_names: set) -> List[str]:
        """Find potential entry point files.

        Args:
            path: Project root directory
            languages: Detected languages
            file_names: Set of file names in project

        Returns:
            List of entry point file paths relative to project root
        """
        entry_points = []

        # Check language-specific entry points
        for lang in languages:
            if lang in self.ENTRY_POINTS:
                for entry_point in self.ENTRY_POINTS[lang]:
                    if entry_point in file_names:
                        # Find the actual path
                        for file_path in path.rglob(entry_point):
                            if file_path.is_file():
                                relative = file_path.relative_to(path)
                                entry_points.append(str(relative))
                                break

        # Special handling for package.json
        package_json = path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Check main field
                if "main" in data:
                    main_file = path / data["main"]
                    if main_file.exists():
                        entry_points.append(data["main"])

                # Check scripts for common entry points
                if "scripts" in data:
                    for script_name in ["start", "dev", "serve"]:
                        if script_name in data["scripts"]:
                            # Try to extract file from script command
                            script = data["scripts"][script_name]
                            parts = script.split()
                            for part in parts:
                                if part.endswith((".js", ".ts")):
                                    file = path / part
                                    if file.exists():
                                        entry_points.append(part)
                                    break
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                pass

        return list(dict.fromkeys(entry_points))  # Remove duplicates while preserving order

    def _determine_project_type(self, languages: List[str], frameworks: List[str]) -> str:
        """Determine the primary project type.

        Args:
            languages: List of detected languages
            frameworks: List of detected frameworks

        Returns:
            String describing the project type
        """
        if not languages:
            return "unknown"

        # Framework takes precedence
        if frameworks:
            framework = frameworks[0]
            # Clean up prefixed framework names
            if "_" in framework:
                parts = framework.split("_", 1)
                return parts[1]  # Return the framework part
            return framework

        # Fall back to primary language
        return languages[0]

    def _calculate_confidence(
        self, languages: List[str], frameworks: List[str], entry_points: List[str]
    ) -> float:
        """Calculate confidence score for detection.

        Args:
            languages: Detected languages
            frameworks: Detected frameworks
            entry_points: Detected entry points

        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0

        # Language detection confidence
        if languages:
            score += 0.3
            if len(languages) == 1:
                score += 0.1  # Single language project is more certain

        # Framework detection confidence
        if frameworks:
            score += 0.3
            if len(frameworks) == 1:
                score += 0.1  # Single framework is more certain

        # Entry point detection confidence
        if entry_points:
            score += 0.2
            if len(entry_points) >= 2:
                score += 0.1  # Multiple entry points found

        return min(score, 1.0)

    def find_main_file(self, path: Path) -> Optional[Path]:
        """Find the most likely main/entry file in a project.

        Args:
            path: Directory to search in

        Returns:
            Path to the main file, or None if not found
        """
        path = Path(path)
        if not path.is_dir():
            return None

        # Detect project info
        project_info = self.detect_project_type(path)

        # Use detected entry points
        if project_info["entry_points"]:
            main_file = path / project_info["entry_points"][0]
            if main_file.exists():
                return main_file

        # Fall back to language-specific patterns
        for lang in project_info["languages"]:
            if lang in self.ENTRY_POINTS:
                for entry_point in self.ENTRY_POINTS[lang]:
                    for file_path in path.rglob(entry_point):
                        if file_path.is_file():
                            return file_path

        return None
