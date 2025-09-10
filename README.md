# **tenets**

<a href="https://tenets.dev"><img src="https://raw.githubusercontent.com/jddunn/tenets/master/docs/logos/tenets_dark_icon_transparent.png" alt="tenets logo" width="140" /></a>

**context that feeds your prompts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tenets.svg)](https://pypi.org/project/tenets/)
[![CI](https://github.com/jddunn/tenets/actions/workflows/ci.yml/badge.svg)](https://github.com/jddunn/tenets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jddunn/tenets/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/jddunn/tenets)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://tenets.dev/docs)

**tenets** automatically finds and builds the most relevant context from your codebase. Instead of manually copying files or grepping with complex regex, tenets intelligently aggregates exactly what you need for debugging, building features, or coding with AI assistants. Works with any directory of files but best with Git repos.

## Core Features

Intelligent context aggregation that:

- **Finds** all relevant files automatically
- **Ranks** them by importance using multiple factors
- **Aggregates** them within your token budget
- **Formats** perfectly for any use case
- **Pins** critical files per session for guaranteed inclusion
- **Injects** your tenets (guiding principles) into session interactions automatically in prompts
- **Transforms** content on demand (strip comments, condense whitespace, or force full raw context)

## Installation

```bash
# Core features only - lightweight, no ML dependencies
pip install tenets

# Add specific features
pip install tenets[light]  # Adds keyword extraction & TF-IDF ranking
pip install tenets[viz]    # Adds visualization capabilities
pip install tenets[ml]     # Adds deep learning models (2GB+)
pip install tenets[all]    # Everything
```

**Python 3.13 Note:** Compatible but some ML features have limitations. Use Python 3.12 for full ML support.

## Quick Start

### Three Ranking Modes

Tenets offers three modes that balance speed vs. thoroughness for both `distill` and `rank` commands:

| Mode         | Speed | Thoroughness | Primary Algorithm | Use Case                 |
| ------------ | ----- | -------- | ---------------- | ------------------------ |
| **fast**     | ⚡⚡⚡ (100%) | Good     | Substring match   | Quick exploration, CI/CD |
| **balanced** | ⚡⚡ (150% - 1.5x slower) | Better   | BM25 ranking      | Most development (default) |
| **thorough** | ⚡ (400% - 4x slower) | Best     | BM25 + ML embeddings | Complex refactoring, semantic search |

**Fast mode**
- **Simple substring matching** - No regex or word boundary checking
- **No text processing** - No tokenization, abbreviation expansion, or stemming

**Balanced mode**
- **Builds BM25 corpus** - Creates inverted index for all files (one-time cost)
- **Uses word boundaries** - Regex matching for precise results
- **Processes text** - Handles abbreviations, compound words, plurals

**Thorough mode*
- **ML embeddings** - Semantic similarity (requires `tenets[ml]`)
- **Dual algorithms** - Uses both BM25 AND TF-IDF
- **Pattern recognition** - Detects code patterns and dependencies

⚠️ **Note**: Thorough mode finds semantically similar code, not just exact matches. Use Fast or Balanced if you need literal keyword matching.

**When to use each mode:**
- **Fast**: Most likely for programmatic tools, that would replace grepping
- **Balanced**: Day-to-day development / general usage (default, only 50% slower)
- **Thorough**: Major refactoring, in-depth reviews, architectural analysis

### Core Commands

#### `distill` - Build Context with Content

```bash
# Basic usage - finds and aggregates relevant files
tenets distill "implement OAuth2" ./src

# Copy to clipboard (great for AI chats)
tenets distill "fix payment bug" --copy --mode fast # Simple and fast matching

# Generate interactive HTML report
tenets distill "analyze auth flow" --format html -o report.html --mode balanced # Default mode with BM25 ranking

# Speed/accuracy trade-offs
tenets distill "debug tokenizing issue" --mode fast       # Skips corpus building, good for small codebases
tenets distill "refactor API" --mode thorough  # Most comprehensive analysis (needs ML packages for full features)

# Transform content to save tokens
tenets distill "aggregate any code that uses redux" --remove-comments --condense
```

#### `rank` - Preview Files Without Content

```bash
# See what files would be included (much faster than distill!)
tenets rank "implement payments" --top 20

# Understand WHY files are ranked
tenets rank "fix auth" --factors

# Tree view for structure understanding
tenets rank "add caching to file scanning based on timestamps" --tree --scores

# Export for automation
tenets rank "user database migration" --format json | jq '.files[].path'
```

**Tips**

- **Learn one to learn both**: Rank and distill commands are designed to be interchangeable, with the only difference between file context instead of just filepaths.

- **Be brutally curt**: Intentions and concepts are mapped (specifically in programming context), so the more you list the more matches (crossover you'll get; e.g. "fix, analyze, and debug this issue" is worse than "fix..").

- **Don't use many synonyms**: Sometimes in LLM interactions you mention multiple variations of a term, hoping to expand its context and connections. This is a static tool; more words in the prompt = lower processing speed, more crossover. Let tenets decide relationships, not you.

- **Planning**: Identify key functions, lines, variables, before making changes helps greatly. If you specify method names in the prompt, it will search for and aggregate all files that call it and smartly summarize the file's context around mentioned matches.

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
- **[CLI Reference](docs/cli.md)** - All commands and options
- **[Configuration Guide](docs/config.md)** - Detailed configuration options
- **[Architecture Overview](docs/architecture/index.md)** - How tenets works internally

## Advanced Features

### Smart Summarization

When files exceed token budgets, tenets intelligently preserves:

- Function/class signatures
- Import statements
- Complex logic blocks
- Documentation and comments
- Recent changes

For more details on the summarization system, see [Architecture Documentation](docs/architecture/index.md).

### Test File Handling

Tests are **excluded by default** for most prompts, **automatically included** when your prompt mentions testing:

```bash
# Tests excluded (better context):
tenets distill "explain auth flow"

# Tests included (detected by intent):
tenets distill "write tests for auth"
tenets distill "fix failing tests from these logs:.."

# Manual override:
tenets distill "" --include-tests
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

Specialized analyzers for Python, JavaScript/TypeScript, Go, Java, C/C++, Ruby, PHP, Rust, and more. Configuration and documentation files are analyzed with smart heuristics for YAML, TOML, JSON, Markdown, etc. See [supported languages](./docs/supported_languages.md) for the full list.

## Contributing

See [CONTRIBUTING.md](contributing.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---
