"""Generate API reference pages and navigation with proper TOC.

This script uses mkdocs-gen-files and literate-nav for dynamic API documentation.
Configured to show classes, methods, and functions in the TOC sidebar.
"""

from pathlib import Path
import mkdocs_gen_files

# Navigation builder for literate-nav
nav = mkdocs_gen_files.Nav()

# Root of the project
root = Path(__file__).parent.parent
src = root / "tenets"

# Key modules to document with full submodules
MAIN_MODULES = [
    "tenets",
    "tenets.core",
    "tenets.core.analyzer",
    "tenets.core.distiller",
    "tenets.core.aggregator",
    "tenets.core.ranking",
    "tenets.cli",
    "tenets.cli.app",
    "tenets.cli.commands",
    "tenets.models",
    "tenets.storage",
    "tenets.utils",
    "tenets.viz",
    "tenets.config",
    "tenets.api",
]

# Track all generated modules for the index page
generated_modules = []

# Process all Python files in the tenets package
for path in sorted(src.rglob("*.py")):
    # Skip test files and cache
    if any(part in str(path) for part in ["__pycache__", "test", "_test", ".pyc"]):
        continue

    # Get module path relative to src
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    # Handle the module parts
    parts = tuple(module_path.parts)

    # Skip __main__ modules
    if parts[-1] == "__main__":
        continue

    # Handle __init__ files - they represent the package itself
    if parts[-1] == "__init__":
        if len(parts) == 1:
            # Root __init__.py - document as main package
            parts = ("index",)
            doc_path = Path("index.md")
            full_doc_path = Path("api", doc_path)
        else:
            # Package __init__.py - use parent name
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")

    # Build the identifier for mkdocstrings
    if parts == ("index",):
        # Main package
        identifier = "tenets"
        nav_parts = ("tenets",)
    else:
        identifier = ".".join(["tenets"] + list(parts))
        nav_parts = ("tenets",) + parts

    # Add to navigation
    nav[nav_parts] = doc_path.as_posix()

    # Track for index page
    generated_modules.append((identifier, doc_path.as_posix(), parts))

    # Generate the markdown content with mkdocstrings directive
    # Use different settings for main modules vs submodules
    is_main_module = identifier in MAIN_MODULES

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Write page title with full module path for clarity
        if parts == ("index",):
            print(f"# Tenets Package\n", file=fd)
        else:
            # Include module path for context (e.g., "core.ranking.strategies")
            title = " › ".join([p.replace("_", " ").title() for p in parts])
            print(f"# {title}\n", file=fd)
        print(f"`{identifier}`\n", file=fd)

        # Write mkdocstrings directive with enhanced options for TOC
        print(f"::: {identifier}", file=fd)
        print("    options:", file=fd)

        # Heading and TOC options
        print("        show_root_heading: true", file=fd)
        print("        show_root_toc_entry: true", file=fd)  # Show in TOC
        print("        show_object_full_path: false", file=fd)
        print("        show_symbol_type_heading: true", file=fd)  # Shows Class/Function
        print("        show_symbol_type_toc: true", file=fd)  # Include type in TOC

        # Content display options
        print("        show_source: false", file=fd)  # Don't show source by default
        print("        show_bases: true", file=fd)  # Show inheritance
        print("        show_submodules: " + ("true" if is_main_module else "false"), file=fd)

        # Member options
        print("        members: true", file=fd)  # Show all members
        print("        members_order: source", file=fd)
        print("        group_by_category: true", file=fd)
        print("        show_category_heading: true", file=fd)

        # Docstring options
        print("        show_if_no_docstring: false", file=fd)
        print("        docstring_style: google", file=fd)
        print("        docstring_section_style: table", file=fd)
        print("        merge_init_into_class: true", file=fd)

        # Signature options
        print("        separate_signature: true", file=fd)
        print("        show_signature_annotations: true", file=fd)
        print("        signature_crossrefs: true", file=fd)  # Enable cross-refs

        # Summary options - show summaries for better navigation
        print("        summary: true", file=fd)  # Enable summaries

        # Inheritance
        print("        inherited_members: false", file=fd)

        # Filters
        print("        filters:", file=fd)
        print('          - "!^_"', file=fd)
        print('          - "!^test"', file=fd)

        # Heading level - ensure we don't run out of levels
        print("        heading_level: 2", file=fd)

    # Set edit path for "edit on GitHub" link
    if full_doc_path.name == "index.md" and path.name == "__init__.py":
        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))
    else:
        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the literate-nav file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Create a comprehensive index page with explicit links and instructions
index_content = """# API Reference

Complete API documentation for the Tenets package with navigable table of contents.

!!! tip "Navigation"
    - Use the **left sidebar** to browse modules
    - Use the **right sidebar** (table of contents) to see classes, functions, and methods within each module
    - Click on any class or method name to jump directly to its documentation

## Quick Start

```python
from tenets import Tenets

# Initialize
tenets = Tenets()

# Build context for your prompt
result = tenets.distill("implement OAuth2 authentication")
print(result.content)
```

## Main Modules

Click on any module to view its complete API documentation with all classes and methods:

### Core Package

- **[tenets](index.md)** - Main package with `Tenets` class and public API
- **[tenets.api](api.md)** - Public API interface
- **[tenets.config](config.md)** - Configuration management

### Core Functionality

- **[tenets.core](core/index.md)** - Core engines and functionality
- **[tenets.core.analyzer](core/analyzer.md)** - `CodeAnalyzer` class for code analysis
- **[tenets.core.distiller](core/distiller.md)** - `Distiller` class for context distillation
- **[tenets.core.aggregator](core/aggregator.md)** - `Aggregator` class for content aggregation

### Ranking System

- **[tenets.core.ranking](core/ranking/index.md)** - Complete ranking system
- **[tenets.core.ranking.ranker](core/ranking/ranker.md)** - `RelevanceRanker` class
- **[tenets.core.ranking.strategies](core/ranking/strategies.md)** - Ranking strategies

### Command-Line Interface

- **[tenets.cli](cli/index.md)** - CLI package
- **[tenets.cli.app](cli/app.md)** - Main CLI application with Typer
- **[tenets.cli.commands](cli/commands/index.md)** - All CLI commands

### Data Models

- **[tenets.models](models/index.md)** - All data models
- **[tenets.models.file](models/file.md)** - `FileContext`, `FileMetadata` models
- **[tenets.models.context](models/context.md)** - `Context`, `ContextResult` models
- **[tenets.models.session](models/session.md)** - `Session` model
- **[tenets.models.config](models/config.md)** - Configuration models

### Storage & Utilities

- **[tenets.storage](storage/index.md)** - Storage and caching
- **[tenets.utils](utils/index.md)** - Utility functions
- **[tenets.viz](viz/index.md)** - Visualization tools

## How to Navigate

### Using the Table of Contents (Right Sidebar)

When viewing any module page, the right sidebar shows:

1. **Classes** - All classes defined in the module
   - Click to jump to the class documentation
   - Expand to see all methods and attributes

2. **Functions** - Module-level functions
   - Click to view function signature and documentation

3. **Attributes** - Module-level attributes and constants

### Example Navigation

1. Click on **[tenets.core.distiller](core/distiller.md)** to view the Distiller module
2. In the right sidebar, you'll see:
   - `Distiller` class
     - `__init__()` method
     - `distill()` method
     - `aggregate()` method
     - Other methods and attributes
3. Click on any method to jump directly to its documentation

## Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Tenets` | [tenets](index.md) | Main API class |
| `Distiller` | [tenets.core.distiller](core/distiller.md) | Context distillation |
| `CodeAnalyzer` | [tenets.core.analyzer](core/analyzer.md) | Code analysis |
| `RelevanceRanker` | [tenets.core.ranking.ranker](core/ranking/ranker.md) | File ranking |
| `Session` | [tenets.models.session](models/session.md) | Session management |
| `FileContext` | [tenets.models.file](models/file.md) | File representation |

## API Usage Examples

### Basic Usage

```python
from tenets import Tenets

# Initialize with default settings
tenets = Tenets()

# Or with custom configuration
tenets = Tenets(
    max_tokens=50000,
    algorithm="balanced"
)

# Distill context
result = tenets.distill("implement authentication")
```

### Advanced Ranking

```python
from tenets.core.ranking import RelevanceRanker

# Create ranker with specific algorithm
ranker = RelevanceRanker(algorithm="thorough")

# Rank files
ranked_files = ranker.rank(
    files=my_files,
    prompt="implement caching",
    threshold=0.1
)

# Get top files
top_files = ranked_files[:10]
```

### Using Models

```python
from tenets.models import FileContext, Session

# Create a file context
file = FileContext(
    path="src/main.py",
    content="...",
    metadata={"size": 1024}
)

# Create a session
session = Session(
    name="feature-dev",
    pinned_files=[file]
)
```

## Cross-References

All classes and functions are cross-referenced. You can click on any type annotation or class name to navigate to its documentation.

For example, when viewing a method that returns a `FileContext`, clicking on `FileContext` will take you to its documentation.
"""

with mkdocs_gen_files.open("api/index.md", "w") as index_file:
    index_file.write(index_content)
