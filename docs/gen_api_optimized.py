"""
Optimized API documentation generator that creates separate pages per module
for better performance and lazy loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
import mkdocs_gen_files

logger = logging.getLogger(__name__)

# Core modules to document - explicitly list for control
MODULES_TO_DOCUMENT = [
    # Main entry points
    ("tenets", True, "Main Package"),
    ("tenets.api", False, "Public API"),
    ("tenets.config", False, "Configuration"),
    # Core functionality
    ("tenets.core", True, "Core Package"),
    ("tenets.core.analyzer", False, "Code Analyzer"),
    ("tenets.core.distiller", False, "Content Distiller"),
    ("tenets.core.ranking", True, "Ranking System"),
    ("tenets.core.ranking.ranker", False, "File Ranker"),
    ("tenets.core.ranking.strategies", False, "Ranking Strategies"),
    # CLI
    ("tenets.cli", True, "CLI Package"),
    ("tenets.cli.commands", True, "CLI Commands"),
    # Storage
    ("tenets.storage", True, "Storage Package"),
    ("tenets.storage.cache", False, "Cache System"),
    # Utils
    ("tenets.utils", True, "Utilities Package"),
]


def write_module_page(mod_name: str, is_pkg: bool, title: str) -> None:
    """Write a single module documentation page."""
    # Convert module name to path
    parts = mod_name.split(".")
    if is_pkg:
        doc_path = Path("api") / Path(*parts) / "index.md"
    else:
        doc_path = Path("api") / Path(*parts).with_suffix(".md")

    # Create minimal content for performance
    content = f"""# {title}

`{mod_name}`

::: {mod_name}
    options:
        show_source: false
        show_root_heading: false
        show_root_toc_entry: false
        show_object_full_path: false
        show_category_heading: true
        members_order: source
        group_by_category: true
        docstring_style: google
        merge_init_into_class: true
        show_if_no_docstring: false
        inherited_members: false
        show_submodules: false
        filters: ["!^_", "!^test"]
        heading_level: 2
        show_bases: false
        show_signature: true
        separate_signature: true
        unwrap_annotated: true
        signature_crossrefs: false
"""

    logger.info(f"Writing {doc_path}")
    with mkdocs_gen_files.open(doc_path.as_posix(), "w") as fd:
        fd.write(content)

    # Set edit path
    if is_pkg:
        src_path = Path("tenets") / Path(*parts[1:]) / "__init__.py"
    else:
        src_path = Path("tenets") / Path(*parts[1:]).with_suffix(".py")

    mkdocs_gen_files.set_edit_path(doc_path, src_path)


def write_index() -> None:
    """Write the main API index page."""
    content = """# API Reference

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

## Usage Example

```python
from tenets import Tenets

# Initialize
t = Tenets()

# Distill context
result = t.distill("implement OAuth")
print(result.content)
```

---

!!! tip "Performance Note"
    API documentation pages are generated on-demand for better performance.
    Click on specific modules in the navigation to view their documentation.
"""

    with mkdocs_gen_files.open("api/index.md", "w") as fd:
        fd.write(content)


def write_nav() -> None:
    """Write navigation file for literate-nav."""
    nav_parts = ["* [Overview](index.md)"]

    # Group modules by package
    current_package = None
    for mod_name, is_pkg, title in MODULES_TO_DOCUMENT:
        parts = mod_name.split(".")
        indent = "  " * (len(parts) - 1)

        if is_pkg:
            nav_parts.append(f"{indent}* [{title}]({'/'.join(parts)}/index.md)")
        else:
            filename = parts[-1] + ".md"
            nav_parts.append(f"{indent}* [{title}]({'/'.join(parts)}.md)")

    content = "# API Reference\n\n" + "\n".join(nav_parts)

    with mkdocs_gen_files.open("api/SUMMARY.md", "w") as fd:
        fd.write(content)


def main():
    """Main entry point."""
    logger.info("Generating optimized API documentation")

    # Write index
    write_index()

    # Write nav
    write_nav()

    # Write module pages
    for mod_name, is_pkg, title in MODULES_TO_DOCUMENT:
        try:
            write_module_page(mod_name, is_pkg, title)
        except Exception as e:
            logger.warning(f"Skipping {mod_name}: {e}")

    logger.info("API documentation generation complete")


if __name__ == "__main__":
    main()
