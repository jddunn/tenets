"""
Generate API documentation pages dynamically for all modules.
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
        if parts == ("index",):
            title = "Tenets Package"
        else:
            title = parts[-1].replace("_", " ").title()
            if len(parts) > 1:
                title = f"{'.'.join(parts)} - {title}"
        
        fd.write(f"# {title}\n\n")
        fd.write(f"`{identifier}`\n\n")
        
        # Write mkdocstrings directive with proper options
        fd.write(f"::: {identifier}\n")
        fd.write("    options:\n")
        fd.write("        show_source: false\n")
        fd.write("        show_root_heading: true\n")
        fd.write("        show_root_toc_entry: true\n")
        fd.write("        show_object_full_path: false\n")
        fd.write("        show_category_heading: true\n")
        fd.write("        show_symbol_type_heading: true\n")
        fd.write("        show_symbol_type_toc: true\n")
        fd.write("        members_order: source\n")
        fd.write("        group_by_category: true\n")
        fd.write("        members: true\n")
        fd.write("        docstring_style: google\n")
        fd.write("        docstring_section_style: table\n")
        fd.write("        merge_init_into_class: true\n")
        fd.write("        show_if_no_docstring: false\n")
        fd.write("        inherited_members: false\n")
        fd.write("        show_submodules: false\n")
        fd.write('        filters: ["!^_", "!^test"]\n')
        fd.write("        heading_level: 2\n")
        fd.write("        show_bases: true\n")
        fd.write("        show_signature: true\n")
        fd.write("        separate_signature: true\n")
        fd.write("        unwrap_annotated: true\n")
        fd.write("        signature_crossrefs: true\n")
    
    # Set edit path for "edit on GitHub" link
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the literate-nav file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Create a comprehensive index page with working links
index_content = """# API Reference

Complete API documentation for the Tenets package.

## Quick Navigation

### Core Components
- [**Tenets**](tenets/index.md) - Main package and API
- [**Config**](tenets/config.md) - Configuration system
- [**Core**](tenets/core/index.md) - Core functionality

### Command Line Interface
- [**CLI**](tenets/cli/index.md) - Command-line tools
- [**Commands**](tenets/cli/commands/index.md) - Available commands

### Subsystems
- [**Ranking**](tenets/core/ranking/index.md) - File ranking system
- [**Storage**](tenets/storage/index.md) - Caching and persistence
- [**Utils**](tenets/utils/index.md) - Utility functions
- [**Models**](tenets/models/index.md) - Data models
- [**Viz**](tenets/viz/index.md) - Visualization tools

## Key Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Tenets` | [tenets](tenets/index.md) | Main API class |
| `Distiller` | [tenets.core.distiller](tenets/core/distiller.md) | Context distillation |
| `CodeAnalyzer` | [tenets.core.analyzer](tenets/core/analyzer.md) | Code analysis |
| `RelevanceRanker` | [tenets.core.ranking.ranker](tenets/core/ranking/ranker.md) | File ranking |

## Usage Example

```python
from tenets import Tenets

# Initialize
t = Tenets()

# Distill context
result = t.distill("implement OAuth")
print(result.content)
```

## Navigation Tips

- Use the **left sidebar** to browse modules
- Use the **right sidebar** (table of contents) to see classes and methods within each module
- Click on any class or method name to jump to its documentation

---

!!! tip "Performance Note"
    API documentation pages are generated on-demand for better performance.
    Click on specific modules in the navigation to view their documentation.
"""

with mkdocs_gen_files.open("api/index.md", "w") as fd:
    fd.write(index_content)