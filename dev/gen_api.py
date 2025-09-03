"""
Generate per-module API documentation pages for mkdocs using mkdocstrings.
Creates files under docs/api/ mirroring the tenets package structure so the
sidebar shows a nested tree.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Iterator, Tuple

import mkdocs_gen_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ROOT = Path(__file__).parent
API_DIR = Path("api")
PACKAGE_NAME = "tenets"

# Modules to skip (add any modules you want to exclude)
SKIP_MODULES = {
    "tenets.tests",
    "tenets.test",
    "tenets._internal",
    "tenets.examples",
}

# Private module/package prefixes to skip
PRIVATE_PREFIXES = ("_", "test_", "tests")


def should_document_module(module_name: str) -> bool:
    """Determine if a module should be documented."""
    # Skip if in skip list
    if module_name in SKIP_MODULES:
        return False

    # Skip if any part of the module path starts with private prefix
    parts = module_name.split(".")
    for part in parts:
        if any(part.startswith(prefix) for prefix in PRIVATE_PREFIXES):
            return False

    # Skip if it's a test module
    if any(x in module_name for x in ["test", "tests", "testing"]):
        return False

    return True


def iter_modules(package_name: str) -> Iterator[Tuple[str, bool]]:
    """
    Iterate through all modules in a package.

    Yields:
        Tuple of (module_name, is_package)
    """
    try:
        pkg = importlib.import_module(package_name)
    except ImportError as e:
        logger.warning(f"Could not import {package_name}: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error importing {package_name}: {e}")
        return

    if not hasattr(pkg, "__path__"):
        logger.debug(f"{package_name} is not a package (no __path__)")
        return

    prefix = pkg.__name__ + "."

    # Walk through the package
    for finder, mod_name, ispkg in pkgutil.walk_packages(
        pkg.__path__, prefix, onerror=lambda x: None
    ):
        if should_document_module(mod_name):
            yield mod_name, ispkg


def get_module_doc_path(mod_name: str, is_pkg: bool) -> Path:
    """Get the documentation file path for a module."""
    # Convert module name to path: tenets.core.ranking -> api/tenets/core/ranking
    rel_dir = API_DIR / Path(*mod_name.split("."))

    # Packages get index.md, modules get module_name.md
    if is_pkg:
        return rel_dir / "index.md"
    else:
        return rel_dir.with_suffix(".md")


def write_module_page(mod_name: str, is_pkg: bool) -> None:
    """Write the documentation page for a module."""
    out_path = get_module_doc_path(mod_name, is_pkg)

    # Create a nice title (last part of module name, capitalized)
    parts = mod_name.split(".")
    title = parts[-1].replace("_", " ").title()
    full_title = f"{title} {'Package' if is_pkg else 'Module'}"

    # Build the page content
    content = f"""---
title: {title}
---

# {full_title}

`{mod_name}`

::: {mod_name}
    options:
        show_source: true
        show_root_heading: false
        show_root_toc_entry: false
        show_object_full_path: false
        show_category_heading: true
        show_symbol_type_heading: true
        show_symbol_type_toc: true
        members_order: source
        group_by_category: true
        docstring_style: google
        docstring_section_style: table
        merge_init_into_class: true
        separate_signature: true
        line_length: 80
        show_signature_annotations: true
        signature_crossrefs: true
        summary: true
"""

    # Add filters for non-package modules to hide private members
    if not is_pkg:
        content += """        filters:
          - "!^_"
          - "!^test"
"""

    logger.info(f"Writing {out_path}")

    with mkdocs_gen_files.open(out_path.as_posix(), "w") as fd:
        fd.write(content)

    # Set edit path for the generated file
    # This allows "Edit" button to point to the source file
    if is_pkg:
        src_path = Path(PACKAGE_NAME) / Path(*parts[1:]) / "__init__.py"
    else:
        src_path = Path(PACKAGE_NAME) / Path(*parts[1:]).with_suffix(".py")

    if src_path.exists():
        mkdocs_gen_files.set_edit_path(out_path, src_path)


def write_pages_files() -> None:
    """Write .pages files for awesome-pages plugin (if using it)."""
    # Root .pages file for API section
    pages_yaml = """title: API Reference
arrange:
  - index.md
  - tenets
collapse_single_pages: false
"""

    with mkdocs_gen_files.open((API_DIR / ".pages").as_posix(), "w") as fd:
        fd.write(pages_yaml)

    # Package-level .pages file for better organization
    tenets_pages = """title: Tenets Package
arrange:
  - index.md
  - core
  - models
  - utils
  - parsers
  - cli
  - ...
"""

    with mkdocs_gen_files.open((API_DIR / "tenets" / ".pages").as_posix(), "w") as fd:
        fd.write(tenets_pages)


def write_root_index() -> None:
    """Write the root API index page."""
    index_content = """---
title: API Reference
description: Complete API documentation for the Tenets package
---

# API Reference

Welcome to the Tenets API reference documentation. This section contains detailed information about all modules, classes, functions, and other components of the Tenets package.

## Package Overview

The Tenets package provides intelligent code exploration and AI pair programming capabilities. It's organized into several key modules:

### Core Modules

- **[tenets](tenets/index.md)** - Main package initialization and exports
- **[tenets.core](tenets/core/index.md)** - Core functionality and main classes
- **[tenets.models](tenets/models/index.md)** - Data models and structures
- **[tenets.utils](tenets/utils/index.md)** - Utility functions and helpers
- **[tenets.parsers](tenets/parsers/index.md)** - Language parsers and analyzers
- **[tenets.cli](tenets/cli/index.md)** - Command-line interface

## Quick Start

### Installation

```bash
pip install tenets
```

### Basic Usage

```python
from tenets import Tenets

# Initialize Tenets
t = Tenets()

# Analyze a project
context = t.analyze('path/to/project')

# Get code context
code_context = context.get_context()
```

### Advanced Usage

```python
from tenets import Tenets, Config
from tenets.parsers import PythonParser

# Custom configuration
config = Config(
    max_depth=5,
    include_tests=False,
    parsers=[PythonParser()]
)

# Initialize with config
t = Tenets(config=config)

# Analyze with options
context = t.analyze(
    'path/to/project',
    include_patterns=['*.py', '*.js'],
    exclude_patterns=['*_test.py']
)
```

## Navigation

Use the sidebar to browse through the package structure. Each module and class has its own page with detailed documentation generated from docstrings.

## Contributing

For information on contributing to Tenets, please see the [Contributing Guide](../resources/contributing.md).

## Support

- [GitHub Issues](https://github.com/jddunn/tenets/issues)
- [Discord Community](https://discord.gg/DzNgXdYm)
- [Documentation](https://tenets.dev)
"""

    with mkdocs_gen_files.open((API_DIR / "index.md").as_posix(), "w") as fd:
        fd.write(index_content)


def write_nav_file() -> None:
    """Write SUMMARY.md for literate-nav plugin."""
    nav_content = """# API Reference

* [Overview](index.md)
* [tenets package](tenets/index.md)
    * [core](tenets/core/index.md)
    * [models](tenets/models/index.md)
    * [utils](tenets/utils/index.md)
    * [parsers](tenets/parsers/index.md)
    * [cli](tenets/cli/index.md)
"""

    with mkdocs_gen_files.open((API_DIR / "SUMMARY.md").as_posix(), "w") as fd:
        fd.write(nav_content)


def main():
    """Main entry point for the API documentation generator."""
    # Ensure the package root is in the Python path
    package_root = ROOT.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    logger.info(f"Generating API documentation for {PACKAGE_NAME}")
    logger.info(f"Package root: {package_root}")

    # Write root files
    write_root_index()
    write_nav_file()

    # Check if using awesome-pages plugin
    try:
        import mkdocs_awesome_pages_plugin

        write_pages_files()
        logger.info("Writing .pages files for awesome-pages plugin")
    except ImportError:
        logger.info("awesome-pages plugin not found, skipping .pages files")

    # Document the main package
    write_module_page(PACKAGE_NAME, True)

    # Document all submodules and subpackages
    documented_count = 0
    for name, is_pkg in iter_modules(PACKAGE_NAME):
        write_module_page(name, is_pkg)
        documented_count += 1

    logger.info(f"Generated documentation for {documented_count} modules/packages")


if __name__ == "__main__":
    main()
