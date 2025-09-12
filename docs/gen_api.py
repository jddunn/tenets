"""
Generate API reference documentation for MkDocs.

This script automatically discovers and documents all Python modules in the tenets package.
It creates a structured navigation and individual pages for each module with mkdocstrings.
"""

from pathlib import Path
import mkdocs_gen_files

# Root directories
root = Path(__file__).parent.parent
src = root / "tenets"

# Modules to skip (avoid problematic or test modules)
SKIP_PATTERNS = {
    "__pycache__",
    "test",
    "_test",
    "tests",
    ".pyc",
    "migrations",
    "examples",
    "benchmarks",
}

# Priority modules to document first (most important)
PRIORITY_MODULES = [
    "tenets",
    "tenets.api",
    "tenets.config",
    "tenets.core",
    "tenets.core.analyzer",
    "tenets.core.distiller",
    "tenets.core.ranking",
    "tenets.cli",
    "tenets.cli.app",
    "tenets.cli.commands",
    "tenets.models",
    "tenets.storage",
    "tenets.utils",
    "tenets.viz",
]

def should_skip(path: Path) -> bool:
    """Check if a path should be skipped."""
    path_str = str(path).lower()
    return any(pattern in path_str for pattern in SKIP_PATTERNS)

def get_module_priority(module_name: str) -> int:
    """Get priority for module (lower number = higher priority)."""
    try:
        return PRIORITY_MODULES.index(module_name)
    except ValueError:
        return len(PRIORITY_MODULES) + 1

def generate_module_page(path: Path, module_path: Path, doc_path: Path):
    """Generate documentation page for a single module."""
    # Get module identifier
    parts = tuple(module_path.parts)
    
    # Handle __init__.py files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        
    if not parts:
        return None, None
    
    identifier = ".".join(parts)
    
    # Create title
    if parts[-1] == "tenets" and len(parts) == 1:
        title = "Tenets - Main Package"
    else:
        title = parts[-1].replace("_", " ").title()
        if len(parts) > 1:
            title = f"{'.'.join(parts)} - {title}"
    
    # Generate page content with optimized mkdocstrings options
    content = f"""# {title}

`{identifier}`

::: {identifier}
    options:
        show_source: true
        show_root_heading: true
        show_root_full_path: false
        show_signature_annotations: true
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        show_category_heading: true
        show_if_no_docstring: false
        show_docstring_attributes: true
        show_docstring_functions: true
        show_docstring_classes: true
        show_docstring_modules: true
        show_docstring_description: true
        show_docstring_examples: true
        show_docstring_other_parameters: true
        show_docstring_parameters: true
        show_docstring_raises: true
        show_docstring_receives: true
        show_docstring_returns: true
        show_docstring_warns: true
        show_docstring_yields: true
        members_order: source
        group_by_category: true
        show_submodules: false
        docstring_style: google
        docstring_section_style: table
        merge_init_into_class: true
        inherited_members: false
        filters:
          - "!^_"
          - "!^test"
          - "!^Test"
"""
    
    return identifier, (doc_path, content, path.relative_to(root))

# Collect all modules
modules = []
for path in sorted(src.rglob("*.py")):
    if should_skip(path):
        continue
    
    module_path = path.relative_to(root).with_suffix("")
    doc_path = Path("api") / path.relative_to(root).with_suffix(".md")
    
    result = generate_module_page(path, module_path, doc_path)
    if result[0]:
        modules.append(result)

# Sort modules by priority and name
modules.sort(key=lambda x: (get_module_priority(x[0]), x[0]))

# Generate documentation files
nav_items = {}
for identifier, (doc_path, content, edit_path) in modules:
    full_doc_path = Path("api") / doc_path
    
    # Write the documentation file
    with mkdocs_gen_files.open(str(full_doc_path), "w") as fd:
        fd.write(content)
    
    # Set edit path
    mkdocs_gen_files.set_edit_path(full_doc_path, edit_path)
    
    # Track for navigation
    parts = identifier.split(".")
    nav_items[tuple(parts)] = str(doc_path).replace("\\", "/")

# Build navigation structure
def build_nav_tree(items):
    """Build hierarchical navigation from flat module list."""
    tree = {}
    for parts, path in items.items():
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = path
    return tree

def format_nav_tree(tree, indent=0):
    """Format navigation tree as markdown list."""
    lines = []
    for key, value in sorted(tree.items()):
        if key == "tenets" and indent == 0:
            # Special case for root package
            lines.append(f"* [**Tenets Package**]({value if isinstance(value, str) else 'tenets/index.md'})")
            if isinstance(value, dict):
                lines.extend(format_nav_tree(value, indent + 1))
        elif isinstance(value, dict):
            # It's a package with submodules
            title = key.replace("_", " ").title()
            # Check if package has its own index
            if "index.md" in str(value.values()):
                pkg_path = next((v for v in value.values() if "index.md" in v), "")
                lines.append(f"{'    ' * indent}* [**{title}**]({pkg_path})")
            else:
                lines.append(f"{'    ' * indent}* **{title}**")
            # Add submodules
            sub_items = {k: v for k, v in value.items() if k != key}
            if sub_items:
                lines.extend(format_nav_tree(sub_items, indent + 1))
        else:
            # It's a module
            title = key.replace("_", " ").title()
            lines.append(f"{'    ' * indent}* [{title}]({value})")
    return lines

# Generate main API index page
index_content = """# API Reference

Complete API documentation for the Tenets package, automatically generated from source code docstrings.

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

### Core Components
- **[Tenets](tenets/index.md)** - Main package exports and public API
- **[Core](tenets/core/index.md)** - Core engines for analysis, ranking, and distillation
- **[Config](tenets/config.md)** - Configuration management and settings

### Features
- **[CLI](tenets/cli/index.md)** - Command-line interface and commands
- **[Models](tenets/models/index.md)** - Data models and structures
- **[Storage](tenets/storage/index.md)** - Caching and persistence layer

### Utilities
- **[Utils](tenets/utils/index.md)** - Helper functions and utilities
- **[Viz](tenets/viz/index.md)** - Visualization and output formatting

## Documentation Features

Each module page includes:
- Complete class documentation with all methods and attributes
- Function signatures with type hints
- Detailed parameter descriptions and return types
- Usage examples from docstrings
- Links to source code

## Navigation

Use the sidebar to browse all available modules, or use the search feature to find specific classes or functions.
"""

with mkdocs_gen_files.open("api/index.md", "w") as f:
    f.write(index_content)

# Generate SUMMARY.md for literate-nav plugin
nav_tree = build_nav_tree(nav_items)
nav_lines = ["# API Reference\n", "* [Overview](index.md)"]
nav_lines.extend(format_nav_tree(nav_tree))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as f:
    f.write("\n".join(nav_lines))

print(f"Generated API documentation for {len(modules)} modules")