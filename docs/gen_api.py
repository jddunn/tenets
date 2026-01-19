"""Generate API reference pages for mkdocs."""

from pathlib import Path

import mkdocs_gen_files

# Define the navigation structure
nav = mkdocs_gen_files.Nav()

# Get all Python source files
root_dir = Path("tenets")
src_paths = sorted(root_dir.rglob("*.py"))

# Build a mapping of packages to their modules
package_modules = {}
all_modules = []

# First pass: collect all modules and packages
for path in src_paths:
    # Skip test files, cache, and __pycache__
    if any(skip in str(path) for skip in ["__pycache__", "test_", "_test.py", "/tests/"]):
        continue

    module_path = path.relative_to(root_dir.parent).with_suffix("")
    parts = tuple(module_path.parts)

    # Skip __main__ files
    if parts[-1] == "__main__":
        continue

    # Store module info
    if parts[-1] != "__init__":
        all_modules.append((parts, path))

        # Track which modules belong to which packages
        if len(parts) > 1:
            package = parts[:-1]
            if package not in package_modules:
                package_modules[package] = []
            package_modules[package].append(parts[-1])

# Generate documentation for all modules
for parts, path in all_modules:
    doc_path = Path("api", *parts).with_suffix(".md")
    identifier = ".".join(parts)

    # Add to navigation
    nav[parts] = doc_path.relative_to("api").as_posix()

    # Generate the documentation page
    with mkdocs_gen_files.open(doc_path, "w") as fd:
        module_name = parts[-1]
        display_name = module_name.replace("_", " ").title()

        fd.write(f"# `{module_name}`\n\n")
        fd.write(f"**Full name:** `{identifier}`\n\n")
        fd.write(f"::: {identifier}\n")
        fd.write("    options:\n")
        fd.write("        show_source: true\n")
        fd.write("        show_root_heading: true\n")
        fd.write("        show_root_full_path: false\n")
        fd.write("        show_symbol_type_heading: true\n")
        fd.write("        show_symbol_type_toc: true\n")
        fd.write("        members_order: source\n")
        fd.write("        show_signature_annotations: true\n")
        fd.write("        separate_signature: true\n")
        fd.write("        line_length: 80\n")
        fd.write("        show_if_no_docstring: false\n")
        fd.write("        show_docstring_attributes: true\n")
        fd.write("        show_docstring_functions: true\n")
        fd.write("        show_docstring_classes: true\n")
        fd.write("        show_docstring_modules: true\n")
        fd.write("        show_docstring_description: true\n")
        fd.write("        merge_init_into_class: true\n")

    mkdocs_gen_files.set_edit_path(doc_path, path)

# Generate package index pages
processed_packages = set()
for path in src_paths:
    if "__init__.py" in str(path):
        module_path = path.relative_to(root_dir.parent).with_suffix("")
        parts = tuple(module_path.parts)[:-1]  # Remove __init__

        if parts and parts not in processed_packages:
            processed_packages.add(parts)

            doc_path = Path("api", *parts, "index.md")
            identifier = ".".join(parts)

            # Add to navigation
            nav[(*parts, "index")] = doc_path.relative_to("api").as_posix()

            with mkdocs_gen_files.open(doc_path, "w") as fd:
                package_name = parts[-1]

                fd.write(f"# `{identifier}` Package\n\n")

                # Add package docstring
                fd.write(f"::: {identifier}\n")
                fd.write("    options:\n")
                fd.write("        show_source: false\n")
                fd.write("        show_root_heading: false\n")
                fd.write("        show_root_full_path: false\n")
                fd.write("        show_submodules: false\n")
                fd.write("        members_order: source\n")
                fd.write("        show_if_no_docstring: true\n")

                # List direct subpackages
                subpackages = []
                for other_package in processed_packages:
                    if (
                        len(other_package) == len(parts) + 1
                        and other_package[: len(parts)] == parts
                    ):
                        subpackages.append(other_package[-1])

                if subpackages:
                    fd.write("\n## Subpackages\n\n")
                    for subpkg in sorted(subpackages):
                        display = subpkg.replace("_", " ").title()
                        fd.write(f"- [`{subpkg}`]({subpkg}/index.md) - {display} package\n")

                # List direct modules
                if parts in package_modules:
                    fd.write("\n## Modules\n\n")
                    for module in sorted(set(package_modules[parts])):
                        if module != "__init__":
                            display = module.replace("_", " ").title()
                            fd.write(f"- [`{module}`]({module}.md) - {display} module\n")

            mkdocs_gen_files.set_edit_path(doc_path, path)

# Create main tenets package index (special case)
with mkdocs_gen_files.open("api/tenets/index.md", "w") as fd:
    fd.write("""# `tenets` Package

Main package for Tenets - Context that feeds your prompts.

::: tenets
    options:
        show_source: false
        show_root_heading: false
        members_order: source
        show_if_no_docstring: true

## Main Subpackages

- [`cli`](cli/index.md) - Command-line interface
- [`core`](core/index.md) - Core functionality and algorithms
- [`models`](models/index.md) - Data models and structures
- [`storage`](storage/index.md) - Storage backends and persistence
- [`utils`](utils/index.md) - Utility functions and helpers
- [`viz`](viz/index.md) - Visualization and reporting tools

## Direct Modules

- [`config`](config.md) - Configuration management
""")

# Create the core package index with all its subpackages
with mkdocs_gen_files.open("api/tenets/core/index.md", "w") as fd:
    fd.write("""# `tenets.core` Package

Core functionality and algorithms for Tenets.

::: tenets.core
    options:
        show_source: false
        show_root_heading: false
        members_order: source

## Subpackages

- [`analysis`](analysis/index.md) - Code analysis engines for multiple languages
- [`distiller`](distiller/index.md) - Context distillation and aggregation
- [`examiner`](examiner/index.md) - Code examination and metrics
- [`git`](git/index.md) - Git integration and analysis
- [`instiller`](instiller/index.md) - Tenet and session injection
- [`momentum`](momentum/index.md) - Development momentum tracking
- [`nlp`](nlp/index.md) - Natural language processing utilities
- [`prompt`](prompt/index.md) - Prompt parsing and analysis
- [`ranking`](ranking/index.md) - File ranking and relevance scoring  
- [`reporting`](reporting/index.md) - Report generation
- [`session`](session/index.md) - Session state management
- [`summarizer`](summarizer/index.md) - Content summarization
""")

# Create CLI commands subpackage index
with mkdocs_gen_files.open("api/tenets/cli/commands/index.md", "w") as fd:
    fd.write("""# `tenets.cli.commands` Package

CLI command implementations.

::: tenets.cli.commands
    options:
        show_source: false
        show_root_heading: false

## Command Modules

- [`chronicle`](chronicle.md) - Git history analysis command
- [`config`](config.md) - Configuration management command
- [`distill`](distill.md) - Context distillation command
- [`examine`](examine.md) - Code examination command
- [`instill`](instill.md) - Session and tenet injection command
- [`momentum`](momentum.md) - Development momentum tracking command
- [`rank`](rank.md) - File ranking command
- [`session`](session.md) - Session management command
- [`system_instruction`](system_instruction.md) - System instruction command
- [`tenet`](tenet.md) - Tenet management command
- [`viz`](viz.md) - Visualization command

## Utility Modules

- [`_utils`](_utils.md) - Shared command utilities
""")

# Create the main API index
with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write("""# API Reference

Welcome to the Tenets API documentation. This section provides comprehensive documentation for all modules, classes, and functions in the Tenets package.

## Quick Navigation

### Core Packages

#### [`tenets`](tenets/index.md)
The main package containing core exports and initialization.

#### [`tenets.core`](tenets/core/index.md)
Core functionality including:
- [`analysis`](tenets/core/analysis/index.md) - Multi-language code analysis
- [`distiller`](tenets/core/distiller/index.md) - Context aggregation
- [`ranking`](tenets/core/ranking/index.md) - Relevance scoring
- [`nlp`](tenets/core/nlp/index.md) - NLP utilities
- [`prompt`](tenets/core/prompt/index.md) - Prompt analysis
- [`session`](tenets/core/session/index.md) - Session management

#### [`tenets.cli`](tenets/cli/index.md)
Command-line interface:
- [`app`](tenets/cli/app.md) - Main CLI application
- [`commands`](tenets/cli/commands/index.md) - Command implementations

#### [`tenets.models`](tenets/models/index.md)
Data models:
- [`analysis`](tenets/models/analysis.md) - Analysis results
- [`context`](tenets/models/context.md) - Context structures
- [`tenet`](tenets/models/tenet.md) - Tenet models
- [`summary`](tenets/models/summary.md) - Summary models

#### [`tenets.storage`](tenets/storage/index.md)
Storage backends:
- [`cache`](tenets/storage/cache.md) - Caching layer
- [`session_db`](tenets/storage/session_db.md) - Session database
- [`sqlite`](tenets/storage/sqlite.md) - SQLite backend

#### [`tenets.utils`](tenets/utils/index.md)
Utilities:
- [`scanner`](tenets/utils/scanner.md) - File scanning
- [`tokens`](tenets/utils/tokens.md) - Token counting
- [`logger`](tenets/utils/logger.md) - Logging utilities

#### [`tenets.viz`](tenets/viz/index.md)
Visualization:
- [`dependencies`](tenets/viz/dependencies.md) - Dependency graphs
- [`complexity`](tenets/viz/complexity.md) - Complexity visualization
- [`hotspots`](tenets/viz/hotspots.md) - Code hotspots

## Usage Examples

### Basic Usage

```python
from tenets import Tenets

# Initialize
tenets = Tenets()

# Build context
result = tenets.distill("implement user authentication")
print(result.content)
```

### Advanced Usage

```python
# Direct ranking
from tenets.core.ranking import RelevanceRanker

ranker = RelevanceRanker(algorithm="balanced")
files = ranker.rank(file_list, "add OAuth")

# Code analysis
from tenets.core.analysis import Analyzer

analyzer = Analyzer()
result = analyzer.analyze_file("app.py")

# Session management
from tenets.core.session import SessionManager

session = SessionManager()
session.create("feature-auth")
session.pin_file("auth.py")
```

## Finding Documentation

- **By Feature**: Browse packages above
- **By Name**: Use the search box
- **By Navigation**: Use the sidebar tree

!!! tip "Documentation Tips"
    - Each module page shows all classes and functions
    - Look for "Examples" sections in docstrings
    - Check return types and parameters for usage hints
""")

# Write the navigation summary
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
