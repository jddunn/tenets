"""Generate API reference pages for mkdocs.

Standard approach based on mkdocstrings documentation.
"""

from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Process all Python files in the tenets package
for path in sorted(Path("tenets").rglob("*.py")):
    # Get the module path
    module_path = path.with_suffix("")
    doc_path = path.with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __main__ modules
    if parts[-1] == "__main__":
        continue

    # Handle __init__ files - they represent the package
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = Path(*parts[:-1], "index.md") if len(parts) > 1 else Path("index.md")
        full_doc_path = Path("api", doc_path)

    # Skip test files
    if "__pycache__" in str(path) or "test" in str(path):
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Generate the page with mkdocstrings
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"::: {identifier}", file=fd)

    # Set edit path for GitHub
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate literate-nav
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
