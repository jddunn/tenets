# API Reference

Welcome to the Tenets API documentation. This reference covers all public modules, classes, and functions.

## 🚀 Quick Start

```python
from tenets import Tenets

# Initialize Tenets
t = Tenets()

# Distill context from your codebase
result = t.distill("implement user authentication")
print(f"Found {len(result.files)} relevant files")
print(result.content)

# Rank files without reading content
ranked = t.rank("fix payment bug", top=20)
for file in ranked:
    print(f"{file.path}: {file.relevance_score:.2f}")
```

## 📦 Package Structure

### Core Components

- **[Tenets](tenets/index.md)** - Main package interface and public API
- **[Config](tenets/config.md)** - Configuration management and settings
- **[Core](tenets/core/index.md)** - Core functionality and engines

### Analysis & Processing

- **[Analysis](tenets/core/analysis/index.md)** - Language-specific code analyzers
- **[Distiller](tenets/core/distiller/index.md)** - Context extraction and aggregation
- **[Ranking](tenets/core/ranking/index.md)** - Relevance ranking algorithms

### Command Line

- **[CLI](tenets/cli/index.md)** - Command-line interface framework
- **[Commands](tenets/cli/commands/index.md)** - Available CLI commands

### Data & Storage

- **[Models](tenets/models/index.md)** - Data models and structures
- **[Storage](tenets/storage/index.md)** - Caching and persistence layer

### Utilities

- **[Utils](tenets/utils/index.md)** - Helper functions and utilities
- **[Viz](tenets/viz/index.md)** - Visualization and diagram generation

## 📚 Common Tasks

### Context Extraction

```python
from tenets import Tenets

t = Tenets()

# Basic distillation
result = t.distill("implement OAuth2")

# With options
result = t.distill(
    "fix authentication bug",
    paths=["src/auth"],
    max_tokens=50000,
    include_tests=True
)
```

### File Ranking

```python
from tenets.core.ranking import RelevanceRanker

ranker = RelevanceRanker(algorithm="balanced")
files = ranker.rank(
    files=project_files,
    prompt_context=context,
    threshold=0.1
)
```

### Session Management

```python
from tenets import Tenets

t = Tenets()

# Create session
t.session_create("feature-xyz")

# Pin files
t.instill(
    session="feature-xyz",
    add_files=["src/core.py", "src/api.py"]
)

# Use session
result = t.distill("add validation", session="feature-xyz")
```

---

!!! tip "Performance Optimization"
    API documentation is generated with lazy loading enabled. Pages are loaded
    on-demand as you navigate, ensuring fast initial page loads.

!!! info "Documentation Coverage"
    This reference includes all public modules with docstrings. Private modules
    and those without documentation are excluded.
