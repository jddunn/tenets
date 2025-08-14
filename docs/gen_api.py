"""
Generate per-module API documentation pages for mkdocs using mkdocstrings.
Creates files under docs/api/ mirroring the tenets package structure so the
sidebar shows a nested tree. Requires plugins: gen-files, awesome-pages, mkdocstrings.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import mkdocs_gen_files


def iter_modules(package_name: str):
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        return
    if not hasattr(pkg, "__path__"):
        return
    prefix = pkg.__name__ + "."
    for mod in pkgutil.walk_packages(pkg.__path__, prefix):
        yield mod.name, mod.ispkg


ROOT = Path(__file__).parent
API_DIR = Path("api")


def write_module_page(mod_name: str, is_pkg: bool):
    # Path like api/tenets/core/ranking/index.md for packages, module.md for leaf modules
    rel_dir = API_DIR / Path(*mod_name.split("."))
    if is_pkg:
        out_path = rel_dir / "index.md"
        title = mod_name
    else:
        out_path = rel_dir.with_suffix(".md")
        title = mod_name

        content = f"""---
title: {title}
hide:
    - toc
---

# {title}

::: {mod_name}
        options:
            show_source: false
            separate_signature: true
            members_order: source
            docstring_style: google
"""

    with mkdocs_gen_files.open(out_path.as_posix(), "w") as fd:
        fd.write(content)


def write_root_pages_file():
    # Root .pages to title the section
    pages_yaml = """title: API Reference
collapse_single_pages: true
"""
    with mkdocs_gen_files.open((API_DIR / ".pages").as_posix(), "w") as fd:
        fd.write(pages_yaml)

    # Root index
        index_md = """---
title: API Reference
hide:
    - toc
---

# API Reference

This section contains the full API reference generated from docstrings.

- Package: `tenets`
"""
    with mkdocs_gen_files.open((API_DIR / "index.md").as_posix(), "w") as fd:
        fd.write(index_md)


def main():
    # Ensure repo root on path
    sys.path.insert(0, str(ROOT.parent))

    write_root_pages_file()

    # Document top-level package and all subpackages/modules
    write_module_page("tenets", True)
    for name, is_pkg in iter_modules("tenets") or []:
        # Include all public modules and packages
        write_module_page(name, is_pkg)


if __name__ == "__main__":
    main()
