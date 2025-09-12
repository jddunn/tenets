"""Generate API reference pages for mkdocs."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Process all Python files in the tenets package
for path in sorted(Path("tenets").rglob("*.py")):
    # Skip test files and cache
    if "__pycache__" in str(path) or "test" in str(path):
        continue

    # Get the module path
    module_path = path.with_suffix("")
    doc_path = path.with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __main__ modules
    if parts[-1] == "__main__":
        continue

    # Handle __init__ files
    if parts[-1] == "__init__":
        # Skip package __init__ files to avoid duplication
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Generate the page with mkdocstrings
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        # Add a title for the page
        title = parts[-1].replace("_", " ").title()
        fd.write(f"# {title}\n\n")
        fd.write(f"::: {identifier}\n")

    # Set edit path for GitHub
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate literate-nav
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
