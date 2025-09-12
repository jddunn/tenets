"""Generate the API reference pages dynamically during MkDocs build.

This script is used by mkdocs-gen-files plugin to create virtual documentation
pages for each Python module in the tenets package.
"""

from pathlib import Path

import mkdocs_gen_files

# Get all Python files in the tenets package
root = Path(__file__).parent.parent
src = root / "tenets"

# Track what we've generated for nav (using dict for manual nav building)
nav_items = {}

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

    # Add to navigation dict - store relative path from api/ directory
    nav_items[parts] = str(doc_path).replace("\\", "/")

    # Generate the page content
    identifier = ".".join(parts)

    # Generate mkdocstrings for all modules
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Use safe title generation
        title = parts[-1].replace("_", " ").title()
        fd.write(f"# {identifier}\n\n")
        fd.write(f"::: {identifier}\n")
        fd.write("    options:\n")
        fd.write("        show_source: true\n")  # Enable source viewing
        fd.write("        show_root_heading: true\n")
        fd.write("        members_order: source\n")
        fd.write("        show_if_no_docstring: false\n")
        fd.write("        inherited_members: false\n")
        fd.write("        show_submodules: true\n")  # Show submodules
        fd.write("        filters:\n")
        fd.write('          - "!^_"\n')
        fd.write('          - "!^test"\n')

    # Set edit path to the source file
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


# Build navigation tree manually
def build_nav_tree(items):
    """Build a nested dict from flat module paths."""
    tree = {}
    for parts, path in items.items():
        current = tree
        for _i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # If there's a conflict (module and package with same name), convert to dict
                current[part] = {"__module__": current[part]}
            current = current[part]
        # Last part is the file
        if parts[-1] in current and isinstance(current[parts[-1]], dict):
            # Package index file
            current[parts[-1]]["__module__"] = path
        else:
            current[parts[-1]] = path
    return tree


def write_nav_markdown(tree, indent=0, parent_path=""):
    """Convert nav tree to clean markdown list with proper relative paths."""
    lines = []
    for key, value in sorted(tree.items()):
        if key == "__module__":
            continue  # Skip special module marker

        title = key.replace("_", " ").title()

        if isinstance(value, dict):
            # It's a package
            # Check if package has its own module
            if "__module__" in value:
                # Package with index - make it a link with relative path
                lines.append(f"{'    ' * indent}* [{title}]({value['__module__']})")
            else:
                # Package without index - just a header
                lines.append(f"{'    ' * indent}* **{title}**")

            # Add sub-items
            sub_items = {k: v for k, v in value.items() if k != "__module__"}
            if sub_items:
                lines.extend(write_nav_markdown(sub_items, indent + 1))
        else:
            # It's a module - create a link with relative path
            lines.append(f"{'    ' * indent}* [{title}]({value})")

    return lines


# Write the navigation file
nav_tree = build_nav_tree(nav_items)
nav_lines = write_nav_markdown(nav_tree)

# Write to the root SUMMARY.md for literate-nav to find
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.write("# API Reference\n\n")
    nav_file.write("This section contains the complete API documentation for the tenets package.\n\n")
    nav_file.write("## Modules\n\n")
    nav_file.write("\n".join(nav_lines))
