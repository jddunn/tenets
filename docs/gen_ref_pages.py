"""Generate the API reference pages dynamically during MkDocs build.

This script is used by mkdocs-gen-files plugin to create virtual documentation
pages for each Python module in the tenets package.
"""

from pathlib import Path

import mkdocs_gen_files

# Get all Python files in the tenets package
root = Path(__file__).parent.parent
src = root / "tenets"

# Track what we've generated for nav
nav = mkdocs_gen_files.Nav()

# Skip these problematic modules
SKIP_MODULES = {
    "tenets.core.analysis.implementations",  # Skip the implementations package itself
    "tenets.tests",
    "tenets.test",
}

for path in sorted(src.rglob("*.py")):
    # Skip test files and private modules
    if "__pycache__" in str(path):
        continue
    if "test" in path.name.lower():
        continue

    # Get module path
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    # Get parts for module name
    parts = tuple(module_path.parts)

    # Skip if in skip list
    module_name = ".".join(parts)
    if module_name in SKIP_MODULES:
        continue

    # Skip if private module (starts with _)
    if any(part.startswith("_") and part != "__init__" for part in parts):
        continue

    # Handle __init__.py files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    # Skip if no parts left (root __init__.py)
    if not parts:
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Generate the page content
    identifier = ".".join(parts)

    # Generate mkdocstrings for all modules
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print(f"# {parts[-1].replace('_', ' ').title()}\n", file=fd)
        print(f"::: {identifier}", file=fd)
        print("    options:", file=fd)
        print("        show_source: false", file=fd)
        print("        show_root_heading: true", file=fd)
        print("        members_order: source", file=fd)
        print("        show_if_no_docstring: false", file=fd)
        print("        inherited_members: false", file=fd)
        print("        filters:", file=fd)
        print('          - "!^_"', file=fd)
        print('          - "!^test"', file=fd)

    # Set edit path to the source file
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write the navigation file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
