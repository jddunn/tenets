"""
Pytest configuration guardrails to prevent accidental collection outside the repo root
and avoid importing tests from the active virtual environment on Windows.

This helps when pytest is launched from within the venv directory by mistake.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def pytest_sessionstart(session):  # type: ignore[override]
    # Ensure repo root is on sys.path so local packages resolve first
    root = Path(__file__).parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # If running from inside a venv subfolder, change cwd to repo root
    try:
        cwd = Path.cwd()
        if any(part.lower() in {"venv", ".venv", "scripts", "lib", "lib64"} for part in cwd.parts):
            os.chdir(root)
    except Exception:
        # Best-effort; do not fail session start
        pass

    # Add ignore patterns to the session config to avoid collecting venv/site-packages
    ignore_dirs = {
        "venv",
        ".venv",
        "Lib",
        "lib",
        "Scripts",
        "bin",
        "Include",
        "include",
        "site-packages",
        "build",
        "dist",
    }
    config = session.config
    current_ignores = set(getattr(config, "ignore", []) or [])
    # Note: In newer pytest, config.addinivalue_line is the supported way
    for d in sorted(ignore_dirs):
        try:
            config.addinivalue_line("norecursedirs", d)  # type: ignore[attr-defined]
        except Exception:
            # If addinivalue_line isn't available or fails, continue silently
            pass
