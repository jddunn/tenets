"""Project type detection and entry point discovery.

This module provides intelligent detection of project types, main entry points,
and project structure based on file patterns and heuristics.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger


class ProjectDetector:
    """Detects project type and structure from file patterns."""
    
    # Entry point patterns by language/framework
    ENTRY_POINTS = {
        "python": [
            "__main__.py",
            "main.py",
            "app.py",
            "application.py",
            "run.py",
            "wsgi.py",
            "asgi.py",
            "manage.py",  # Django
            "setup.py",
            "pyproject.toml",
        ],
        "javascript": [
            "index.js",
            "index.ts",
            "main.js",
            "main.ts",
            "app.js",
            "app.ts",
            "server.js",
            "server.ts",
            "package.json",
        ],
        "web": [
            "index.html",
            "main.html",
            "home.html",
            "app.html",
        ],
        "java": [
            "Main.java",
            "Application.java",
            "App.java",
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
        ],
        "go": [
            "main.go",
            "go.mod",
            "go.sum",
        ],
        "rust": [
            "main.rs",
            "lib.rs",
            "Cargo.toml",
        ],
        "csharp": [
            "Program.cs",
            "Startup.cs",
            "*.csproj",
            "*.sln",
        ],
        "cpp": [
            "main.cpp",
            "main.cc",
            "main.cxx",
            "CMakeLists.txt",
            "Makefile",
        ],
        "ruby": [
            "main.rb",
            "app.rb",
            "application.rb",
            "config.ru",
            "Gemfile",
            "Rakefile",
        ],
        "php": [
            "index.php",
            "app.php",
            "composer.json",
        ],
    }
    
    # Project type indicators
    PROJECT_INDICATORS = {
        "python_package": ["setup.py", "pyproject.toml", "__init__.py"],
        "django": ["manage.py", "settings.py", "urls.py", "wsgi.py"],
        "flask": ["app.py", "application.py", "requirements.txt"],
        "fastapi": ["main.py", "app.py", "requirements.txt"],
        "node_package": ["package.json", "node_modules"],
        "react": ["package.json", "src/App.js", "src/App.tsx", "public/index.html"],
        "vue": ["package.json", "vue.config.js", "src/App.vue"],
        "angular": ["package.json", "angular.json", "src/app/app.module.ts"],
        "next": ["package.json", "next.config.js", "pages/"],
        "express": ["package.json", "server.js", "app.js"],
        "spring": ["pom.xml", "src/main/java/", "application.properties"],
        "maven": ["pom.xml"],
        "gradle": ["build.gradle", "build.gradle.kts"],
        "cargo": ["Cargo.toml", "src/main.rs", "src/lib.rs"],
        "go_module": ["go.mod", "go.sum"],
        "dotnet": ["*.csproj", "*.sln", "Program.cs"],
        "rails": ["Gemfile", "config.ru", "app/", "config/"],
        "laravel": ["composer.json", "artisan", "app/", "routes/"],
        "cmake": ["CMakeLists.txt"],
        "make": ["Makefile"],
    }
    
    # Language extensions
    LANGUAGE_EXTENSIONS = {
        "python": [".py", ".pyw", ".pyi"],
        "javascript": [".js", ".jsx", ".mjs", ".cjs"],
        "typescript": [".ts", ".tsx"],
        "java": [".java"],
        "go": [".go"],
        "rust": [".rs"],
        "cpp": [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h", ".hxx"],
        "c": [".c", ".h"],
        "csharp": [".cs"],
        "ruby": [".rb"],
        "php": [".php"],
        "swift": [".swift"],
        "kotlin": [".kt", ".kts"],
        "scala": [".scala"],
        "html": [".html", ".htm"],
        "css": [".css", ".scss", ".sass", ".less"],
        "sql": [".sql"],
        "yaml": [".yml", ".yaml"],
        "json": [".json"],
        "xml": [".xml"],
        "markdown": [".md", ".markdown"],
    }
    
    def __init__(self):
        """Initialize the project detector."""
        self.logger = get_logger(__name__)
    
    def detect_project(self, root_path: Path) -> Dict:
        """Detect project type and structure.
        
        Args:
            root_path: Root directory to analyze
            
        Returns:
            Dictionary containing:
                - type: Primary project type
                - languages: Languages used with percentages
                - frameworks: Detected frameworks
                - entry_points: Main entry point files
                - structure: Project structure information
        """
        root_path = Path(root_path)
        
        # Collect file statistics
        file_stats = self._collect_file_stats(root_path)
        
        # Detect languages
        languages = self._detect_languages(file_stats)
        
        # Detect frameworks and project types
        frameworks = self._detect_frameworks(root_path)
        
        # Find entry points
        entry_points = self._find_entry_points(root_path, languages, frameworks)
        
        # Determine primary project type
        project_type = self._determine_project_type(languages, frameworks, entry_points)
        
        # Analyze project structure
        structure = self._analyze_structure(root_path, project_type)
        
        return {
            "type": project_type,
            "languages": languages,
            "frameworks": frameworks,
            "entry_points": entry_points,
            "structure": structure,
            "root": str(root_path),
        }
    
    def _collect_file_stats(self, root_path: Path) -> Dict:
        """Collect statistics about files in the project."""
        stats = {
            "total_files": 0,
            "by_extension": Counter(),
            "by_directory": Counter(),
            "files": [],
        }
        
        # Common directories to skip
        skip_dirs = {
            ".git", "node_modules", "venv", ".venv", "env", 
            "target", "build", "dist", "__pycache__", ".pytest_cache"
        }
        
        for path in root_path.rglob("*"):
            # Skip hidden and ignored directories
            if any(part in skip_dirs for part in path.parts):
                continue
                
            if path.is_file():
                stats["total_files"] += 1
                stats["by_extension"][path.suffix.lower()] += 1
                stats["by_directory"][path.parent.name] += 1
                stats["files"].append(path)
        
        return stats
    
    def _detect_languages(self, file_stats: Dict) -> Dict[str, float]:
        """Detect programming languages used in the project."""
        language_counts = Counter()
        total_code_files = 0
        
        for ext, count in file_stats["by_extension"].items():
            for lang, extensions in self.LANGUAGE_EXTENSIONS.items():
                if ext in extensions:
                    language_counts[lang] += count
                    total_code_files += count
                    break
        
        # Calculate percentages
        languages = {}
        if total_code_files > 0:
            for lang, count in language_counts.most_common():
                percentage = (count / total_code_files) * 100
                if percentage >= 1.0:  # Only include if at least 1%
                    languages[lang] = round(percentage, 1)
        
        return languages
    
    def _detect_frameworks(self, root_path: Path) -> List[str]:
        """Detect frameworks and project types."""
        detected = []
        
        for framework, indicators in self.PROJECT_INDICATORS.items():
            for indicator in indicators:
                # Check if indicator exists
                if "*" in indicator:
                    # Handle glob patterns
                    if list(root_path.glob(indicator)):
                        detected.append(framework)
                        break
                else:
                    # Check exact path
                    check_path = root_path / indicator
                    if check_path.exists():
                        detected.append(framework)
                        break
        
        return detected
    
    def _find_entry_points(self, root_path: Path, languages: Dict, frameworks: List[str]) -> List[str]:
        """Find main entry points for the project."""
        entry_points = []
        
        # Check language-specific entry points
        for lang in languages:
            if lang in self.ENTRY_POINTS:
                for pattern in self.ENTRY_POINTS[lang]:
                    if "*" in pattern:
                        # Handle glob patterns
                        for match in root_path.glob(pattern):
                            entry_points.append(str(match.relative_to(root_path)))
                    else:
                        # Check exact path
                        check_path = root_path / pattern
                        if check_path.exists():
                            entry_points.append(pattern)
        
        # Special handling for package.json
        package_json = root_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    # Check for main field
                    if "main" in data:
                        entry_points.append(data["main"])
                    # Check scripts
                    if "scripts" in data:
                        if "start" in data["scripts"]:
                            # Parse start script for entry point
                            start_script = data["scripts"]["start"]
                            # Simple heuristic to extract file name
                            parts = start_script.split()
                            for part in parts:
                                if part.endswith((".js", ".ts")):
                                    entry_points.append(part)
            except Exception as e:
                self.logger.debug(f"Failed to parse package.json: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entry_points = []
        for ep in entry_points:
            if ep not in seen:
                seen.add(ep)
                unique_entry_points.append(ep)
        
        return unique_entry_points
    
    def _determine_project_type(self, languages: Dict, frameworks: List[str], entry_points: List[str]) -> str:
        """Determine the primary project type."""
        # Framework-based detection (highest priority)
        if "django" in frameworks:
            return "django_project"
        elif "flask" in frameworks or "fastapi" in frameworks:
            return "python_web_app"
        elif "react" in frameworks or "vue" in frameworks or "angular" in frameworks:
            return "frontend_spa"
        elif "next" in frameworks:
            return "fullstack_next"
        elif "express" in frameworks:
            return "node_backend"
        elif "spring" in frameworks:
            return "java_spring"
        elif "rails" in frameworks:
            return "ruby_rails"
        elif "laravel" in frameworks:
            return "php_laravel"
        elif "cargo" in frameworks:
            return "rust_project"
        elif "go_module" in frameworks:
            return "go_project"
        elif "dotnet" in frameworks:
            return "dotnet_project"
        
        # Language-based detection
        if languages:
            primary_lang = max(languages, key=languages.get)
            
            if primary_lang == "python":
                if "setup.py" in entry_points or "pyproject.toml" in entry_points:
                    return "python_package"
                else:
                    return "python_project"
            elif primary_lang == "javascript" or primary_lang == "typescript":
                if "package.json" in entry_points:
                    return "node_project"
                else:
                    return "javascript_project"
            elif primary_lang == "java":
                return "java_project"
            elif primary_lang == "go":
                return "go_project"
            elif primary_lang == "rust":
                return "rust_project"
            elif primary_lang == "cpp" or primary_lang == "c":
                return "cpp_project"
            elif primary_lang == "csharp":
                return "csharp_project"
            elif primary_lang == "ruby":
                return "ruby_project"
            elif primary_lang == "php":
                return "php_project"
            elif primary_lang == "html":
                return "static_website"
        
        return "unknown"
    
    def _analyze_structure(self, root_path: Path, project_type: str) -> Dict:
        """Analyze project structure based on type."""
        structure = {
            "directories": {},
            "key_files": [],
            "test_directories": [],
            "doc_directories": [],
        }
        
        # Common directory patterns
        common_dirs = {
            "src": "Source code",
            "lib": "Libraries",
            "app": "Application code",
            "api": "API endpoints",
            "components": "UI components",
            "pages": "Page components",
            "views": "View templates",
            "models": "Data models",
            "controllers": "Controllers",
            "services": "Service layer",
            "utils": "Utilities",
            "helpers": "Helper functions",
            "config": "Configuration",
            "public": "Public assets",
            "static": "Static files",
            "assets": "Asset files",
            "templates": "Templates",
            "tests": "Tests",
            "test": "Tests",
            "spec": "Specifications",
            "docs": "Documentation",
            "doc": "Documentation",
        }
        
        # Check for common directories
        for dir_name, description in common_dirs.items():
            dir_path = root_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                structure["directories"][dir_name] = description
                
                # Identify test directories
                if dir_name in ["tests", "test", "spec"]:
                    structure["test_directories"].append(dir_name)
                
                # Identify documentation directories
                if dir_name in ["docs", "doc", "documentation"]:
                    structure["doc_directories"].append(dir_name)
        
        # Identify key files based on project type
        key_patterns = {
            "python_package": ["setup.py", "pyproject.toml", "requirements.txt", "README.md"],
            "node_project": ["package.json", "package-lock.json", "yarn.lock", "README.md"],
            "java_project": ["pom.xml", "build.gradle", "README.md"],
            "go_project": ["go.mod", "go.sum", "README.md"],
            "rust_project": ["Cargo.toml", "Cargo.lock", "README.md"],
        }
        
        if project_type in key_patterns:
            for pattern in key_patterns[project_type]:
                file_path = root_path / pattern
                if file_path.exists():
                    structure["key_files"].append(pattern)
        
        return structure
    
    def find_dependencies_for_viz(self, root_path: Path, files: List[Path]) -> Dict[str, List[str]]:
        """Find dependencies optimized for visualization.
        
        Groups dependencies by module/package level rather than individual files.
        
        Args:
            root_path: Project root
            files: List of files to analyze
            
        Returns:
            Dictionary of module -> list of dependent modules
        """
        # Detect project info
        project_info = self.detect_project(root_path)
        
        # Group files by module/package
        modules = self._group_files_by_module(root_path, files, project_info)
        
        # Build module-level dependency graph
        module_deps = {}
        
        for module, module_files in modules.items():
            deps = set()
            
            for file_path in module_files:
                # Get file dependencies (this would come from the analyzer)
                # For now, we'll return the structure for the viz command to use
                pass
            
            if deps:
                module_deps[module] = sorted(deps)
        
        return module_deps
    
    def _group_files_by_module(self, root_path: Path, files: List[Path], project_info: Dict) -> Dict[str, List[Path]]:
        """Group files by module or package."""
        modules = {}
        
        for file_path in files:
            # Get relative path
            try:
                rel_path = file_path.relative_to(root_path)
            except ValueError:
                continue
            
            # Determine module name based on project type
            if project_info["type"].startswith("python"):
                # Python uses directory structure as modules
                module_parts = rel_path.parts[:-1]  # Exclude filename
                if module_parts:
                    module = ".".join(module_parts)
                else:
                    module = "root"
            elif project_info["type"].startswith("node") or "javascript" in project_info["type"]:
                # Node/JS often uses directory structure
                if len(rel_path.parts) > 1:
                    module = rel_path.parts[0]
                else:
                    module = "root"
            else:
                # Default to top-level directory
                if len(rel_path.parts) > 1:
                    module = rel_path.parts[0]
                else:
                    module = "root"
            
            if module not in modules:
                modules[module] = []
            modules[module].append(file_path)
        
        return modules