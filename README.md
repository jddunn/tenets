# **tenets**

<a href="https://tenets.dev"><img src="https://raw.githubusercontent.com/jddunn/tenets/master/docs/logos/tenets_dark_icon_transparent.png" alt="tenets logo" width="140" /></a>

**context that feeds your prompts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/tenets.svg)](https://pypi.org/project/tenets/)
[![MCP Server](https://img.shields.io/badge/MCP-Server-blue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMiAydjRNMTIgMTh2NE00LjkzIDQuOTNsLjgzIDIuODNNMTYuMjQgMTYuMjRsMi44My44M000LjkzIDE5LjA3bDIuODMtLjgzTTE2LjI0IDcuNzZsLjgzLTIuODNNMiAxMmg0TTE4IDEyaDQiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMyIgZmlsbD0id2hpdGUiLz48L3N2Zz4=)](https://tenets.dev/MCP/)
[![CI](https://github.com/jddunn/tenets/actions/workflows/ci.yml/badge.svg)](https://github.com/jddunn/tenets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jddunn/tenets/branch/master/graph/badge.svg)](https://codecov.io/gh/jddunn/tenets)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://tenets.dev/docs)

**tenets** is an **MCP server** for AI coding assistants: it finds, ranks, and aggregates the most relevant code, integrates natively with Cursor, Claude Desktop, Windsurf via the Model Context Protocol, and also ships a CLI/Python library plus interactive HTML reports with full JSON metadata for automation.

## What is tenets?

- **Finds** all relevant files automatically using NLP analysis
- **Ranks** them by importance using BM25, TF-IDF, ML embeddings, and git signals
- **Aggregates** them within your token budget with intelligent summarizing
- **Integrates** natively with AI assistants via Model Context Protocol (MCP)
- **Pins** critical files per session for guaranteed inclusion
- **Injects** your tenets (guiding principles) to maintain consistency across AI interactions
- **Transforms** content on demand (strip comments, condense whitespace, or force full raw context)

All processing runs locally - no API costs, no data leaving your machine, complete privacy.

## Installation

```bash
# Basic install - fully functional with BM25/TF-IDF ranking
pip install tenets

# With MCP server for AI assistant integration
pip install tenets[mcp]

# Optional extras
pip install tenets[light]  # Adds RAKE/YAKE keyword extraction algorithms
pip install tenets[viz]    # Adds visualization capabilities (graphs, charts)
pip install tenets[ml]     # Adds deep learning for semantic search (2GB+ download)
pip install tenets[all]    # Everything including all optional features
```

## MCP Quick Reference

- **Start the MCP server**
  ```bash
  pip install tenets[mcp]
  tenets-mcp
  ```
- **Cursor** (`~/.cursor/mcp.json`)
  ```json
  {
    "mcpServers": {
      "tenets": { "command": "tenets-mcp" }
    }
  }
  ```
- **Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`)
  ```json
  {
    "mcpServers": {
      "tenets": { "command": "tenets-mcp" }
    }
  }
  ```
- **Tools exposed**: `distill`, `rank`, `examine`, `session_*`, `tenet_*` (same surface as CLI).
- **Docs**: see `docs/MCP.md` for full endpoint/tool list, SSE/HTTP details, and IDE notes.

## MCP Server (AI Assistant Integration)

Tenets includes an MCP server for native integration with AI coding assistants:

```bash
# Start MCP server
pip install tenets[mcp]
tenets-mcp
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

**Cursor** (Settings → MCP Servers):
```json
{
  "tenets": {
    "command": "tenets-mcp"
  }
}
```

Once configured, ask your AI: *"Use tenets to find relevant files for implementing user authentication"*

See [MCP Documentation](docs/MCP.md) for full setup guide.

## Quick Start

### Three Ranking Modes

Tenets offers three modes that balance speed vs. accuracy for both `distill` and `rank` commands:

| Mode         | Speed       | Accuracy | Use Case                 | What It Does                                                 |
| ------------ | ----------- | -------- | ------------------------ | ------------------------------------------------------------ |
| **fast**     | Fastest     | Good     | Quick exploration        | Keyword & path matching, basic relevance                     |
| **balanced** | 1.5x slower | Better   | Most use cases (default) | BM25 scoring, keyword extraction, structure analysis         |
| **thorough** | 4x slower   | Best     | Complex refactoring      | ML semantic similarity, pattern detection, dependency graphs |

### Core Commands

#### `distill` - Build Context with Content

```bash
# Basic usage - finds and aggregates relevant files
tenets distill "implement OAuth2"  # Searches current directory by default

# Search specific directory
tenets distill "implement OAuth2" ./src

# Copy to clipboard (great for AI chats)
tenets distill "fix payment bug" --copy

# Generate interactive HTML report
tenets distill "analyze auth flow" --format html -o report.html

# Speed/accuracy trade-offs
tenets distill "debug issue" --mode fast       # <5s, keyword matching
tenets distill "refactor API" --mode thorough  # Semantic analysis

# ML-enhanced ranking (requires pip install tenets[ml])
tenets distill "fix auth bug" --ml              # Semantic embeddings
tenets distill "optimize queries" --ml --reranker  # Neural reranking (best accuracy)

# Transform content to save tokens
tenets distill "review code" --remove-comments --condense
```

#### `rank` - Preview Files Without Content

```bash
# See what files would be included (much faster than distill!)
tenets rank "implement payments" --top 20  # Searches current directory by default

# Understand WHY files are ranked
tenets rank "fix auth" --factors

# Tree view for structure understanding
tenets rank "add caching" --tree --scores

# ML-enhanced ranking for better accuracy
tenets rank "fix authentication" --ml           # Uses semantic embeddings
tenets rank "database optimization" --ml --reranker  # Cross-encoder reranking

# Export for automation
tenets rank "database migration" --format json | jq '.files[].path'

# Search specific directory
tenets rank "payment refactoring" ./src --top 10
```

### Sessions & Persistence

```bash
# Create a working session
tenets session create payment-feature

# Pin critical files for the session
tenets instill --session payment-feature --add-file src/core/payment.py

# Add guiding principles (tenets)
tenets tenet add "Always validate inputs" --priority critical
tenets instill --session payment-feature

# Build context using the session
tenets distill "add refund flow" --session payment-feature
```

### Other Commands

```bash
# Visualize architecture
tenets viz deps --output architecture.svg   # Dependency graph
tenets viz deps --format html -o deps.html  # Interactive HTML

# Track development patterns
tenets chronicle --since "last week"        # Git activity
tenets momentum --team                      # Sprint velocity

# Analyze codebase
tenets examine . --complexity --threshold 10  # Find complex code
```

## Configuration

Create `.tenets.yml` in your project:

```yaml
ranking:
  algorithm: balanced # fast | balanced | thorough
  threshold: 0.1
  use_git: true # Use git signals for relevance

context:
  max_tokens: 100000

output:
  format: markdown
  copy_on_distill: true # Auto-copy to clipboard

ignore:
  - vendor/
  - '*.generated.*'
```

## How It Works

### Code analysis intelligence

tenets employs a multi-layered approach optimized specifically for code understanding (but its core functionality could be applied to any field of document matching). It tokenizes `camelCase` and `snake_case` identifiers intelligently. Test files are excluded by default unless specifically mentioned in some way. Language-specific AST parsing for [15+ languages](./docs/supported-languages.md) is included.

### Multi-ranking NLP

Deterministic algorithms in `balanced` work reliably and quickly meant to be used by default. BM25 scoring prevents biasing of files which may use redundant patterns (test files with which might have "response" referenced over and over won't necessarily dominate searches for "response").

The default ranking factors consist of: BM25 scoring (25% - statistical relevance preventing repetition bias), keyword matching (20% - direct substring matching), path relevance (15%), TF-IDF similarity (10%), import centrality (10%), git signals (10% - recency 5%, frequency 5%), complexity relevance (5%), and type relevance (5%).

### Smart Summarization

When files exceed token budgets, tenets intelligently preserves:

- Function/class signatures
- Import statements
- Complex logic blocks
- Documentation and comments
- Recent changes

### ML / deep learning embeddings

Semantic understand can be had with ML features: `pip install tenets[ml]`. Enable with `--ml --reranker` flags or set `use_ml: true` and `use_reranker: true` in config.

In `thorough` mode, sentence-transformer embeddings are enabled, and _understand_ that `authenticate()` and `login()` are conceptually related for example, and that `payment` even has some crossover in relevancy (since these are typically associated together).

**Optional cross-encoder neural re-ranking** in this mode jointly evaluates query-document pairs with self-attention for superior accuracy.

A cross-encoder, for example, will correctly rank `"DEPRECATED: We no longer implement oauth2"` lower than `implement_authorization_flow()` for query `"implement oauth2"`, understanding the negative context despite keyword matches.

Since cross-encoders process document-query pairs together (O(n²) complexity), they're much slower than bi-encoders and only used for re-ranking top K results.

## Documentation

- **[Full Documentation](https://tenets.dev/docs)** - Complete guide and API reference
- **[CLI Reference](docs/CLI.md)** - All commands and options
- **[Configuration Guide](docs/CONFIG.md)** - Detailed configuration options
- **[Architecture Overview](docs/ARCHITECTURE.md)** - How tenets works internally

### Output Formats

```bash
# Markdown (default, optimized for AI)
tenets distill "implement OAuth2" --format markdown

# Interactive HTML with search, charts, copy buttons
tenets distill "review API" --format html -o report.html

# JSON for programmatic use
tenets distill "analyze" --format json | jq '.files[0]'

# XML optimized for Claude
tenets distill "debug issue" --format xml
```

## Python API

```python
from tenets import Tenets

# Initialize
tenets = Tenets()

# Basic usage
result = tenets.distill("implement user authentication")
print(f"Generated {result.token_count} tokens")

# Rank files without content
from tenets.core.ranking import RelevanceRanker
ranker = RelevanceRanker(algorithm="balanced")
ranked_files = ranker.rank(files, prompt_context, threshold=0.1)

for file in ranked_files[:10]:
    print(f"{file.path}: {file.relevance_score:.3f}")
```

## Supported Languages

Specialized analyzers for Python, JavaScript/TypeScript, Go, Java, C/C++, Ruby, PHP, Rust, and more. Configuration and documentation files are analyzed with smart heuristics for YAML, TOML, JSON, Markdown, etc.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**[Documentation](https://tenets.dev)** · **[MCP Guide](https://tenets.dev/MCP/)** · **[Privacy](https://tenets.dev/privacy/)** · **[Terms](https://tenets.dev/terms/)**

team@tenets.dev // team@manic.agency
