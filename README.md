# **tenets**

<a href="https://tenets.dev"><img src="https://raw.githubusercontent.com/jddunn/tenets/master/docs/logos/tenets_dark_icon_transparent.png" alt="tenets logo" width="140" /></a>

**context that feeds your prompts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/tenets.svg)](https://pypi.org/project/tenets/)
[![CI](https://github.com/jddunn/tenets/actions/workflows/ci.yml/badge.svg)](https://github.com/jddunn/tenets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jddunn/tenets/graph/badge.svg)](https://codecov.io/gh/jddunn/tenets)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://tenets.dev/docs)

**tenets** is a NLP CLI tool that automatically finds and builds the most relevant context from your codebase, using deterministic statistical algorithms as well as optional deep learning techniques.

Instead of manually copying files or searching for docs, tenets intelligently aggregates the most relevant files and metadata for interacting with AI assistants.

## What is tenets?

- **Finds** all relevant files automatically
- **Ranks** them by importance using multiple factors
- **Aggregates** them within your token budget
- **Formats** perfectly for any use case
- **Pins** critical files per session for guaranteed inclusion
- **Injects** your tenets (guiding principles) into session interactions automatically in prompts
- **Transforms** content on demand (strip comments, condense whitespace, or force full raw context)

### How It Works

tenets employs a multi-layered approach optimized specifically for code understanding (but its core functionality could be applied to any field of document matching). It tokenizes `camelCase` and `snake_case` identifiers intelligently. Test files are excluded by default unless specifically mentioned in some way. Language-specific AST parsing for [15+ languages](./docs/supported-languages.md) is included.

Deterministic algorithms in `balanced` work reliably and quickly meant to be used by default. BM25 scoring prevents files which may use redundant patterns (test files with `assert` etc.) from dominating results unfairly. 

The default ranking factors consist of: Keyword matching (20%), path relevance (15%), import centrality (10%), git signals (10%), and other code complexity metrics.

Optional semantic analysis captures relationships that keyword matching misses. Sentence-transformer embeddings *understand* that `authenticate()` and `login()` are conceptually related, and relevancy is judged accordingly.

We have additional optional cross-encoder neural re-ranking which jointly evaluates query-document pairs with self-attention. Since this is appending a document and query as a combined input (in O(n^2)), this is much slower than bi-encoders (in our ML-powered regular ranking), and as such is only used for re-ranking the top X results. But, with a cross-encoder, a query like `"implement oauth2"` will rank a document with the text: `"DEPRECATED: We no longer implement oauth2"` much lower compared a function that may be named `implement_authorization_flow()`, **even though** `authorization` in the method name doesn't exactly match `oauth2`.

tenets is able to scan, analyze, match, and summarize hundreds of files for relevant context in significantly under a minute typically, with a multi-tier caching system based on git and file metadata changes ensuring responsive performance, along with parallel and streaming file processing.

All processing runs locally - no API costs, no data leaving your machine, complete privacy.

## Installation

```bash
# Core features only - lightweight, no ML dependencies
pip install tenets

# Add specific features
pip install tenets[light]  # Adds keyword extraction & BM25/TF-IDF ranking
pip install tenets[viz]    # Adds visualization capabilities
pip install tenets[ml]     # Adds deep learning models (2GB+)
pip install tenets[all]    # Everything
```

**Python 3.13 Note:** Compatible but some ML features have limitations. Use Python 3.12 for full ML support.

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

## Documentation

- **[Full Documentation](https://tenets.dev/docs)** - Complete guide and API reference
- **[CLI Reference](docs/CLI.md)** - All commands and options
- **[Configuration Guide](docs/CONFIG.md)** - Detailed configuration options
- **[Architecture Overview](docs/ARCHITECTURE.md)** - How tenets works internally

### Smart Summarization

When files exceed token budgets, tenets intelligently preserves:

- Function/class signatures
- Import statements
- Complex logic blocks
- Documentation and comments
- Recent changes

For more details on the summarization system, see [Architecture Documentation](docs/ARCHITECTURE.md).

## Advanced Features

### Test File Handling

Tests are **excluded by default** for most prompts, **automatically included** when your prompt mentions testing:

```bash
# Tests excluded (better context):
tenets distill "explain auth flow"

# Tests included (detected by intent):
tenets distill "write tests for auth"
tenets distill "fix failing tests"

# Manual override:
tenets distill "review code" --include-tests
tenets distill "fix test_user.py" --exclude-tests
```

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

## Contact

team@tenets.dev // team@manic.agency
