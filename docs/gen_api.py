"""Generate API reference pages and navigation.

This script uses mkdocs-gen-files and literate-nav for dynamic API documentation.
Based on mkdocstrings best practices for 2024.
"""

from pathlib import Path
import mkdocs_gen_files

# Navigation builder for literate-nav
nav = mkdocs_gen_files.Nav()

# Root of the project
root = Path(__file__).parent.parent
src = root / "tenets"

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
    
    # Generate the markdown content with mkdocstrings directive
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Write page title
        title = parts[-1].replace("_", " ").title() if parts != ("index",) else "Tenets Package"
        print(f"# `{identifier}`\n", file=fd)
        
        # Write mkdocstrings directive
        print(f"::: {identifier}", file=fd)
        print("    options:", file=fd)
        print("        show_root_heading: true", file=fd)
        print("        show_root_full_path: false", file=fd)
        print("        show_source: true", file=fd)
        print("        show_object_full_path: false", file=fd)
        print("        show_category_heading: true", file=fd)
        print("        show_if_no_docstring: false", file=fd)
        print("        members_order: source", file=fd)
        print("        group_by_category: true", file=fd)
        print("        docstring_style: google", file=fd)
        print("        docstring_section_style: table", file=fd)
        print("        merge_init_into_class: true", file=fd)
        print("        separate_signature: true", file=fd)
        print("        show_signature_annotations: true", file=fd)
        print("        inherited_members: false", file=fd)
        print("        filters:", file=fd)
        print('          - "!^_"', file=fd)
        print('          - "!^test"', file=fd)
    
    # Set edit path for "edit on GitHub" link
    mkdocs_gen_files.set_edit_path(full_doc_path, Path("tenets") / path.relative_to(src))

# Write the literate-nav file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Also create a nice index page for the API section
index_content = """# API Reference

This section contains the complete API documentation for the Tenets package, automatically generated from source code docstrings.

## Quick Start

```python
from tenets import Tenets

# Initialize
tenets = Tenets()

# Build context for your prompt
result = tenets.distill("implement OAuth2 authentication")
print(result.content)
```

## Package Structure

The API documentation is organized to mirror the package structure:

- **tenets** - Main package with public API
- **tenets.core** - Core functionality and engines
- **tenets.cli** - Command-line interface
- **tenets.models** - Data models and structures
- **tenets.storage** - Storage and caching
- **tenets.utils** - Utility functions
- **tenets.viz** - Visualization tools

## Navigation

Use the sidebar to browse through modules, classes, and functions. Each page includes:

- Complete docstrings with descriptions
- Type annotations and signatures
- Parameter documentation
- Return value descriptions
- Usage examples where provided
- Links to source code

## Features

This documentation is generated using:
- **mkdocstrings** - Automatic documentation from docstrings
- **mkdocs-gen-files** - Dynamic page generation at build time
- **mkdocs-literate-nav** - Navigation structure from markdown

The documentation is automatically updated whenever the source code changes.
"""

with mkdocs_gen_files.open("api/index.md", "w") as index_file:
    index_file.write(index_content)