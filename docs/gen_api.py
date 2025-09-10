"""
Optimized API documentation generator with full auto-discovery and lazy loading.
Discovers all Python modules with docstrings and generates individual pages.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mkdocs_gen_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModuleDiscovery:
    """Discovers Python modules with documentation."""

    def __init__(self, package_root: str = "tenets"):
        self.package_root = Path(package_root)
        self.modules: List[Tuple[str, bool, str, int]] = []

    def has_docstring(self, filepath: Path) -> bool:
        """Check if a Python file has meaningful docstrings."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Quick check for any docstrings
            if '"""' not in content and "'''" not in content:
                return False

            tree = ast.parse(content, filename=str(filepath))

            # Check for module docstring
            if ast.get_docstring(tree):
                return True

            # Check for documented classes or functions
            doc_count = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node) and not node.name.startswith("_"):
                        doc_count += 1
                        if doc_count >= 2:  # At least 2 documented items
                            return True

            return doc_count > 0
        except Exception as e:
            logger.debug(f"Error checking {filepath}: {e}")
            return False

    def get_module_priority(self, module_name: str) -> int:
        """Assign priority for navigation ordering."""
        # Higher priority = appears first
        priority_map = {
            "tenets": 100,
            "tenets.api": 95,
            "tenets.config": 90,
            "tenets.core": 85,
            "tenets.core.distiller": 80,
            "tenets.core.analysis": 75,
            "tenets.core.ranking": 70,
            "tenets.core.instiller": 68,
            "tenets.core.git": 66,
            "tenets.cli": 65,
            "tenets.cli.commands": 60,
            "tenets.models": 55,
            "tenets.storage": 50,
            "tenets.utils": 45,
            "tenets.viz": 40,
        }

        # Check exact match first
        if module_name in priority_map:
            return priority_map[module_name]

        # Check prefix match for sub-modules
        for prefix, priority in sorted(priority_map.items(), key=lambda x: -len(x[0])):
            if module_name.startswith(prefix + "."):
                # Sub-modules get slightly lower priority than parent
                depth = module_name[len(prefix) :].count(".")
                return priority - (depth * 2) - 1

        return 0

    def discover(self) -> List[Tuple[str, bool, str, int]]:
        """Discover all modules with documentation."""
        if not self.package_root.exists():
            logger.error(f"Package root {self.package_root} not found")
            return []

        discovered = {}

        # Walk through all Python files
        for py_file in sorted(self.package_root.rglob("*.py")):
            # Skip test files and private modules
            if any(part.startswith(("test", "_test", "test_")) for part in py_file.parts):
                continue
            if "__pycache__" in str(py_file):
                continue
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            # Convert path to module name
            rel_path = py_file.relative_to(self.package_root.parent)
            parts = list(rel_path.parts[:-1])  # Remove filename

            if py_file.stem == "__init__":
                # Package module
                if len(parts) <= 1:  # Skip root __init__
                    continue
                module_name = ".".join(parts)
                is_package = True
            else:
                # Regular module
                parts.append(py_file.stem)
                module_name = ".".join(parts)
                is_package = False

            # Skip if no docstrings
            if not self.has_docstring(py_file):
                logger.debug(f"Skipping {module_name} - no significant docstrings")
                continue

            # Generate human-readable title
            title = self.create_title(module_name, is_package)
            priority = self.get_module_priority(module_name)

            discovered[module_name] = (module_name, is_package, title, priority)
            logger.info(f"Discovered: {module_name} (priority: {priority})")

        # Sort by priority then name
        self.modules = sorted(
            discovered.values(),
            key=lambda x: (-x[3], x[0]),  # Negative priority for descending order
        )

        return self.modules

    def create_title(self, module_name: str, is_package: bool) -> str:
        """Create a human-readable title from module name."""
        parts = module_name.split(".")

        # Special cases for better titles
        title_map = {
            "tenets": "Tenets Main Package",
            "tenets.api": "Public API",
            "tenets.config": "Configuration",
            "tenets.core": "Core Components",
            "tenets.core.analysis": "Code Analysis",
            "tenets.core.distiller": "Context Distiller",
            "tenets.core.ranking": "Ranking System",
            "tenets.core.instiller": "Tenet Instiller",
            "tenets.core.git": "Git Integration",
            "tenets.cli": "Command Line Interface",
            "tenets.cli.commands": "CLI Commands",
            "tenets.cli.app": "CLI Application",
            "tenets.models": "Data Models",
            "tenets.storage": "Storage & Caching",
            "tenets.utils": "Utilities",
            "tenets.viz": "Visualization",
        }

        if module_name in title_map:
            return title_map[module_name]

        # Generate title from last part
        last_part = parts[-1]

        # Handle special suffixes
        if last_part.endswith("_analyzer"):
            lang = last_part[:-9].title()
            return f"{lang} Analyzer"
        elif last_part.endswith("_parser"):
            return last_part[:-7].replace("_", " ").title() + " Parser"
        elif last_part.endswith("_formatter"):
            return last_part[:-10].replace("_", " ").title() + " Formatter"

        # Standard title generation
        title = last_part.replace("_", " ").title()

        if is_package:
            if title not in ["Index", "Init"]:
                title += " Package"

        return title


class APIDocGenerator:
    """Generates optimized API documentation pages."""

    def __init__(self, modules: List[Tuple[str, bool, str, int]]):
        self.modules = modules
        self.module_tree = self._build_tree()

    def _build_tree(self) -> Dict:
        """Build hierarchical tree of modules."""
        tree = {}

        for mod_name, is_pkg, title, priority in self.modules:
            parts = mod_name.split(".")
            current = tree

            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {
                        "_children": {},
                        "_info": None,
                        "_path": ".".join(parts[: i + 1]),
                    }

                if i == len(parts) - 1:
                    current[part]["_info"] = (mod_name, is_pkg, title, priority)

                current = current[part]["_children"]

        return tree

    def generate_module_page(self, mod_name: str, is_pkg: bool, title: str) -> None:
        """Generate a single module documentation page."""
        # Determine output path
        parts = mod_name.split(".")
        if is_pkg:
            doc_path = Path("api") / Path(*parts) / "index.md"
        else:
            doc_path = Path("api") / Path(*parts).with_suffix(".md")

        # Generate page content with optimized mkdocstrings options
        content = f"""# {title}

`{mod_name}`

::: {mod_name}
    options:
      # Display options
      show_source: false
      show_root_heading: false
      show_root_toc_entry: false
      show_object_full_path: false
      show_category_heading: true
      show_symbol_type_heading: true
      show_symbol_type_toc: false
      
      # Member options
      members_order: source
      group_by_category: true
      show_submodules: false
      inherited_members: false
      
      # Docstring options
      docstring_style: google
      docstring_section_style: table
      merge_init_into_class: true
      show_if_no_docstring: false
      
      # Signature options
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      unwrap_annotated: true
      signature_crossrefs: false
      
      # Filtering
      filters:
        - "!^_"
        - "!^test"
        - "!Test"
        - "!Mock"
      
      # Performance
      preload_modules: []
      load_external_modules: false
      heading_level: 2
"""

        logger.debug(f"Writing {doc_path}")
        with mkdocs_gen_files.open(doc_path.as_posix(), "w") as fd:
            fd.write(content)

        # Set edit path for GitHub edit button
        if is_pkg:
            src_path = Path("tenets") / Path(*parts[1:]) / "__init__.py"
        else:
            src_path = Path("tenets") / Path(*parts[1:]).with_suffix(".py")

        mkdocs_gen_files.set_edit_path(doc_path, src_path)

    def generate_index_page(self) -> None:
        """Generate the main API index page."""
        content = """# API Reference

Welcome to the Tenets API documentation. This reference covers all public modules, classes, and functions.

## 🚀 Quick Start

```python
from tenets import Tenets

# Initialize Tenets
t = Tenets()

# Distill context from your codebase
result = t.distill("implement user authentication")
print(f"Found {len(result.files)} relevant files")
print(result.content)

# Rank files without reading content
ranked = t.rank("fix payment bug", top=20)
for file in ranked:
    print(f"{file.path}: {file.relevance_score:.2f}")
```

## 📦 Package Structure

### Core Components

- **[Tenets](tenets/index.md)** - Main package interface and public API
- **[Config](tenets/config.md)** - Configuration management and settings
- **[Core](tenets/core/index.md)** - Core functionality and engines

### Analysis & Processing

- **[Analysis](tenets/core/analysis/index.md)** - Language-specific code analyzers
- **[Distiller](tenets/core/distiller/index.md)** - Context extraction and aggregation
- **[Ranking](tenets/core/ranking/index.md)** - Relevance ranking algorithms

### Command Line

- **[CLI](tenets/cli/index.md)** - Command-line interface framework
- **[Commands](tenets/cli/commands/index.md)** - Available CLI commands

### Data & Storage

- **[Models](tenets/models/index.md)** - Data models and structures
- **[Storage](tenets/storage/index.md)** - Caching and persistence layer

### Utilities

- **[Utils](tenets/utils/index.md)** - Helper functions and utilities
- **[Viz](tenets/viz/index.md)** - Visualization and diagram generation

## 📚 Common Tasks

### Context Extraction

```python
from tenets import Tenets

t = Tenets()

# Basic distillation
result = t.distill("implement OAuth2")

# With options
result = t.distill(
    "fix authentication bug",
    paths=["src/auth"],
    max_tokens=50000,
    include_tests=True
)
```

### File Ranking

```python
from tenets.core.ranking import RelevanceRanker

ranker = RelevanceRanker(algorithm="balanced")
files = ranker.rank(
    files=project_files,
    prompt_context=context,
    threshold=0.1
)
```

### Session Management

```python
from tenets import Tenets

t = Tenets()

# Create session
t.session_create("feature-xyz")

# Pin files
t.instill(
    session="feature-xyz",
    add_files=["src/core.py", "src/api.py"]
)

# Use session
result = t.distill("add validation", session="feature-xyz")
```

---

!!! tip "Performance Optimization"
    API documentation is generated with lazy loading enabled. Pages are loaded
    on-demand as you navigate, ensuring fast initial page loads.

!!! info "Documentation Coverage"
    This reference includes all public modules with docstrings. Private modules
    and those without documentation are excluded.
"""

        with mkdocs_gen_files.open("api/index.md", "w") as fd:
            fd.write(content)

    def generate_navigation(self) -> None:
        """Generate SUMMARY.md for literate-nav plugin."""
        nav_lines = []

        # Add header and overview
        nav_lines.append("# API Reference")
        nav_lines.append("")
        nav_lines.append("* [Overview](index.md)")

        # Simply iterate through all modules in order
        # Group by top-level package for better organization
        current_top = None

        for mod_name, is_pkg, title, priority in self.modules:
            parts = mod_name.split(".")
            top_level = parts[1] if len(parts) > 1 else parts[0]

            # Add spacing between top-level packages
            if current_top != top_level:
                if current_top is not None:
                    nav_lines.append("")
                current_top = top_level

            # Calculate indentation based on depth
            depth = len(parts) - 1
            indent = "  " * depth

            # Generate the path
            if is_pkg:
                path = "/".join(parts) + "/index.md"
            else:
                path = "/".join(parts) + ".md"

            # Add the navigation entry
            nav_lines.append(f"{indent}* [{title}]({path})")

        content = "\n".join(nav_lines)

        with mkdocs_gen_files.open("api/SUMMARY.md", "w") as fd:
            fd.write(content)

    def generate_all(self) -> None:
        """Generate all API documentation."""
        # Generate index
        self.generate_index_page()

        # Generate navigation
        self.generate_navigation()

        # Generate module pages
        for mod_name, is_pkg, title, _ in self.modules:
            try:
                self.generate_module_page(mod_name, is_pkg, title)
            except Exception as e:
                logger.error(f"Failed to generate {mod_name}: {e}")


def main():
    """Main entry point for API documentation generation."""
    logger.info("=" * 60)
    logger.info("Starting optimized API documentation generation")
    logger.info("=" * 60)

    # Discover modules
    discovery = ModuleDiscovery("tenets")
    modules = discovery.discover()

    logger.info(f"Discovered {len(modules)} modules with documentation")

    if not modules:
        logger.warning("No modules found with documentation")
        return

    # Generate documentation
    generator = APIDocGenerator(modules)
    generator.generate_all()

    logger.info("=" * 60)
    logger.info(f"API documentation generation complete!")
    logger.info(f"Generated {len(modules)} module pages")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
