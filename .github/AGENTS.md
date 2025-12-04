# AI Agent Instructions for Tenets

This document helps AI agents understand the Tenets codebase.

## Quick Reference

### What is Tenets?
MCP server, CLI, and Python library for intelligent code context. Helps AI assistants understand codebases.

### Key Entry Points

```bash
# CLI
tenets distill "description"    # Build code context
tenets examine                   # Analyze codebase

# MCP Server
tenets-mcp                       # Start MCP server
```

### Key Files

| File | Purpose |
|------|---------|
| `tenets/__init__.py` | Main `Tenets` class and public API |
| `tenets/mcp/server.py` | MCP server with tools/resources/prompts |
| `tenets/cli/app.py` | CLI commands |
| `tenets/core/ranking/` | File ranking algorithms |
| `tenets/models/context.py` | `ContextResult` data model |

### Python API

```python
from tenets import Tenets

# Initialize
t = Tenets(path="/project")

# Build context
result = t.distill("find authentication code")
print(result.markdown)

# Create session
t.session_create("my-session")
t.session_pin_file("src/auth.py", "my-session")

# Add guiding principles
t.tenet_add("Always validate user input", priority="high")
```

### MCP Tools

| Tool | Use For |
|------|---------|
| `distill` | Build context for coding tasks |
| `rank_files` | Preview relevance without content |
| `examine` | Codebase structure analysis |
| `session_create` | Start stateful session |
| `session_pin_file` | Pin important files |
| `tenet_add` | Add coding guidelines |

## Development

### Setup
```bash
pip install -e ".[dev,mcp]"
```

### Testing
```bash
pytest tests/
```

### Linting
```bash
ruff check tenets/
mypy tenets/
```

## Architecture

```
tenets/
├── __init__.py          # Tenets class
├── mcp/
│   └── server.py        # TenetsMCP class
├── core/
│   ├── nlp/             # Text processing
│   └── ranking/         # Relevance scoring
├── models/
│   └── context.py       # ContextResult
└── cli/
    └── app.py           # Typer CLI
```

## Contact

- GitHub: https://github.com/jddunn/tenets
- Email: team@manic.agency

