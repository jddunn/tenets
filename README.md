# **tenets**

<a href="https://tenets.dev"><img src="./docs/logos/tenets_dark_icon_transparent.png" alt="tenets logo" width="140" /></a>

**context that feeds your prompts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tenets.svg)](https://pypi.org/project/tenets/)
[![CI](https://github.com/jddunn/tenets/actions/workflows/ci.yml/badge.svg)](https://github.com/jddunn/tenets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jddunn/tenets/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/jddunn/tenets)
[![Test Results](https://img.shields.io/badge/tests-view%20results-blue)](https://app.codecov.io/gh/jddunn/tenets/tests)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://tenets.dev/docs)

See docs locally: [docs/](./docs/) • Live site: https://tenets.dev/docs

**tenets** automatically finds and builds the most relevant context from your codebase. Instead of manually copying files or searching for documentation, tenets intelligently aggregates exactly what you need - whether you're debugging, building features, or chatting with an AI assistant.

## What is tenets?

Think of it as intelligent context aggregation. You give it a prompt or query, and it:

- **Finds** all relevant files automatically
- **Ranks** them by importance using multiple factors
- **Aggregates** them within your token budget
- **Formats** perfectly for any use case
- **Pins** critical files per session for guaranteed inclusion priority
- **Transforms** content on demand (strip comments, condense whitespace, or force full raw context)

Plus powerful development intelligence:

- **Visualize** dependencies and architecture
- **Track** velocity and code evolution
- **Identify** hotspots and technical debt
- **Understand** team patterns and expertise

## Installation / Quick Start

### Python Compatibility

**Python 3.13 Note:** Tenets is compatible with Python 3.13, but some optional dependencies have compatibility issues:

- **YAKE keyword extraction** is automatically disabled on Python 3.13 due to an infinite loop bug
- **RAKE** is used as the primary keyword extraction method (fast, accurate, Python 3.13 compatible)
- For full ML features, consider using Python 3.12 until upstream dependencies are updated

### Pip

```bash
# Core features only - lightweight, no ML dependencies
pip install tenets

# Add specific features
pip install tenets[light]  # Adds RAKE, numpy, scikit-learn for keyword extraction & TF-IDF ranking
pip install tenets[viz]    # Adds visualization capabilities
pip install tenets[ml]     # Adds deep learning models (large dependencies, limited Python 3.13 support)

# Everything
pip install tenets[all]
```

### Install with Poetry

```bash
# Clone the repository
git clone https://github.com/jddunn/tenets.git
cd tenets

# Install with poetry
poetry install           # Core only

# Activate shell
poetry shell
```

### Getting Started

```bash
# Instead of manually finding and copying files...
tenets distill "implement OAuth2" ./src

# It automatically finds auth.py, user.py, config.yaml,
# related tests, dependency files — everything you actually need.
# (Git activity is used for relevance only; not shown in the output.)

# Copy result straight to your clipboard
tenets distill "implement OAuth2" ./src --copy

# Or write it to a file for inspection / sharing
tenets distill "implement OAuth2" ./src > context.md
# (equivalent: tenets distill "implement OAuth2" ./src -o context.md)

# Generate interactive HTML report
tenets distill "analyze authentication" --format html -o report.html

# NEW: See what files would be included WITHOUT the content
tenets rank "implement OAuth2" --top 10        # Show top 10 most relevant files
tenets rank "fix summarizing truncation bug" --tree --factors         # Tree view with ranking breakdown
tenets rank "review API" --format json -o ranked.json --mode thorough  # Export for automation, most accurate and slowest

# Choose your speed/accuracy trade-off
tenets distill "find summarizer truncation bug" --mode fast         # <5s, keyword matching
tenets distill "fix feature to reset user credentials" --mode balanced  # 10-30s, TF-IDF ranking (default)
tenets distill "refactor API" --mode thorough # semantic analysis

# Make copying the default (in .tenets.yml)
# output:\n#   copy_on_distill: true
```

## Why tenets?

When AI pair programming and working with large codebase, you usually have to do a lot of

- Manually searching for relevant files
- Copy-pasting code / docs over and over into multiple LLM prompts
- Gathering important context and dependencies structure

**tenets** uses intelligent NLP-based algorithms to solve this (no LLM API calls required!):

```bash
# Old way: Manual search and copy
$ grep -r "payment" .
$ cat payment.py api.py models.py  # Did I miss anything?
# Copy... paste... hope for the best

# With tenets: Automatic context building
$ tenets distill "fix payment processing bug"
# Finds: payment.py, API endpoints, models, config files, tests, error handlers
# (recent changes inform ranking) — all ranked by relevance
```

## How It Works

Tenets uses a sophisticated multi-stage pipeline to understand your code:

### 1. Intelligent Parsing

When you provide a prompt like "implement OAuth2 authentication", Tenets:

- Extracts key concepts using RAKE (Rapid Automatic Keyword Extraction) - fast & Python 3.13 compatible
- Falls back to TF-IDF and frequency-based extraction if needed
- Identifies intent (implementation/feature)
- Detects any file patterns or specific mentions
- Understands temporal context ("recent changes")

### 2. Code Analysis

For each file in your codebase, Tenets analyzes:

- **Structure**: Classes, functions, imports, exports
- **Dependencies**: What files import this, what it imports
- **Patterns**: Common code patterns (auth, API, database)
- **Metadata**: Language, size, complexity

### 3. Relevance Ranking

Files are scored using multiple factors:

- **Semantic Understanding** (25%): ML-based similarity to your prompt
- **Keyword Matching** (15%): Direct term matching
- **Statistical Relevance** (15%): TF-IDF scoring
- **Code Structure** (20%): Import centrality, path relevance
- **Git Signals** (15%): Recent changes, frequency (optional)
- **File Characteristics** (10%): Type, patterns

All factors are configurable - you can disable git integration, adjust weights, or add custom factors.

### 4. Context Optimization

Finally, Tenets:

- Selects files that score above threshold
- Fits them within token limits
- Summarizes large files if needed
- Formats for optimal LLM consumption

## Advanced Configuration

### Customizing Ranking Weights

Create `.tenets.yml` in your project:

```yaml
ranking:
  algorithm: ml # Use ML-based ranking
  threshold: 0.1 # Lower threshold for more files

  # Disable git factors for stable codebases
  use_git: false

  # Custom weights
  weights:
    semantic_similarity: 0.40 # Increase ML weight
    keyword_match: 0.20
    import_centrality: 0.20
    path_relevance: 0.20

  # Performance tuning
  cache:
    embeddings: true # Cache ML embeddings
    ttl_days: 30 # Longer cache lifetime
    max_size_mb: 2000 # More cache space

  ranking:
    workers: 8 # More parallel workers
    batch_size: 100 # Larger batches for ML
```

### Ranking files for relevance

Like repomix on steroids - smart filters, automatic relevance ranking, and configurable aggregation:

**File Ranking / Hierarchy**

The `rank` command exposes the powerful ranking engine without extracting file contents - perfect for understanding what's relevant before building context:

```bash
# See what files are most relevant to your query
tenets rank "implement payment gateway" --top 20

# Understand WHY files are ranked (see the scoring factors)
tenets rank "fix authentication" --factors
# Shows: semantic_similarity: 85%, keyword_match: 72%, import_centrality: 45%, etc.

# Export ranked files for automation or further processing
tenets rank "database migration" --format json | jq '.files[].path' | xargs git diff

# Tree view to understand project structure relevance
tenets rank "add caching" --tree --scores
📂 src/
  📄 cache_manager.py [0.892]
  📄 redis_client.py [0.834]
📂 src/api/
  📄 endpoints.py [0.756]
  📄 middleware.py [0.623]

# Use in scripts to analyze impact
for file in $(tenets rank "user authentication" --top 5 --no-scores); do
  echo "Analyzing $file..."
  # Your analysis here
done
```

**Why use `rank` instead of `distill`?**
- **Preview**: See what files would be included before generating full context
- **Performance**: Much faster - no file reading or content processing
- **Automation**: Export file lists for CI/CD, code review, or custom scripts  
- **Understanding**: See ranking factors to understand WHY files are relevant
- **Planning**: Identify key files before making changes

Tenets offers three ranking modes that balance speed vs. accuracy:

| Mode         | Speed          | Accuracy | Use Case                                | What It Does                                                                                                        |
| ------------ | -------------- | -------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **fast**     | Fastest         | Good     | Quick exploration, simple queries       | • Keyword & path matching<br>• Basic file type relevance<br>• No deep analysis                                      |
| **balanced** | Fast    | Better   | Most use cases (default)                | • TF-IDF corpus analysis<br>• BM25 relevance scoring<br>• Structure analysis<br>• Import/export tracking            |
| **thorough** | Slower | Best     | Complex refactoring, deep understanding | • Everything from balanced<br>• Semantic similarity (ML)<br>• Code pattern detection<br>• Dependency graph analysis |

**Performance Tips:**

- Start with `fast` for exploration, upgrade to `balanced/thorough` as needed
- Use `--stats` to see ranking performance metrics
- Combine with `--include/--exclude` to reduce file count before ranking
- Cache is shared across modes - second run is always faster

### Filtering and Targeting

```bash
# Distill by file types (include only Python & JS; exclude tests)
tenets distill "review API" --include "*.py,*.js" --exclude "test_*" --stats

# Smart aggregation with larger budget
tenets distill "understand auth flow" --max-tokens 50000 --mode balanced --stats

# Session-based for iterative work
# 1) Create a named session
tenets session create "new-feature"
# 2) Build broad context
tenets distill "design database schema" --session new-feature --stats
# 3) Narrow follow-up
tenets distill "add user model" --session new-feature --stats

# 4) Pin critical files to always include first
tenets instill --session new-feature --add-file src/core/database.py --add-folder src/core/migrations
tenets instill --session new-feature --list-pinned

# 5) Force raw content (no summarization) or shrink tokens
tenets distill "investigate slow queries" --session new-feature --full
tenets distill "summarize public API" --session new-feature --remove-comments --condense

# Working without explicit sessions (uses "default" session automatically)
tenets tenet add "Always validate inputs"
tenets instill  # Applies to default session
tenets distill "implement validation"  # Uses default session

# Save the default session with a meaningful name later
tenets session save validation-feature --delete-source

# Generate interactive HTML report with visualizations
tenets distill "analyze authentication flow" --format html -o report.html
# HTML reports include: search, copy buttons, export to JSON/Markdown, file charts
```

### Guiding Principles (Tenets)

Add persistent instructions that guide AI interactions:

```bash
# Add guiding principles (tenets)
tenets tenet add "Always use type hints in Python"
tenets tenet add "Follow RESTful conventions"
tenets tenet add "Include error handling"
tenets instill # Apply tenets to the session
```

**Injection Behavior:**
- **First Output**: Tenets are automatically injected on the first `distill` in any session
- **Unnamed Sessions**: Always receive tenets to establish context
- **Named Sessions**: Follow configured frequency after first injection
- **No Delay**: Immediate injection (previously waited for 5 operations)

### System Instruction (Global Prompt)

Set a single “system” instruction that can be auto-injected at the top of distilled context (once per session by default):

```bash
# Set and enable a system instruction (persisted)
tenets system-instruction set "You are a senior staff engineer. Prefer small, safe diffs; add tests; explain trade-offs." \
  --enable --position top --format markdown

# From a file
tenets system-instruction set --file prompts/system.md --enable

# Inspect / clear / dry-run
tenets system-instruction show
tenets system-instruction clear --yes
tenets system-instruction test --session my-feature
```

### Real-world flow: system + tenets + sessions

```bash
# 1) Create a working session
tenets session create payment-integration

# 2) Add guiding principles (tenets)
tenets tenet add "Always validate user input" --priority critical --category security
tenets tenet add "Use type hints in Python" --priority high --category style
tenets tenet add "Keep functions under 50 lines" --priority medium --category maintainability

# 3) Apply tenets for this session (smart injection cadence)
tenets instill --session payment-integration

# 4) Set a global system instruction
tenets system-instruction set "You are a precise coding assistant. Prefer incremental changes and defensive coding." --enable

# 5) Build context using the session (tenets + system prompt applied)
tenets distill "add OAuth2 refresh tokens" --session payment-integration --remove-comments --condense --stats

# 6) Pin critical files as you discover them
tenets instill --session payment-integration --add-file src/auth/service.py --add-folder src/auth/routes
tenets instill --session payment-integration --list-pinned

# 7) Iterate with narrower prompts leveraging the same 3
tenets distill "extract token rotation into helper" --session payment-integration
```

### Code Intelligence & Visualization

Understand your codebase at a glance:

```bash
# Dependency graphs with automatic project detection
tenets viz deps . --output architecture.svg  # Auto-detects project type
tenets viz deps . --level module --format html --output deps.html  # Interactive HTML
tenets viz deps . --level package --cluster-by package  # Package-level view
tenets viz deps . --layout circular --max-nodes 50  # Circular layout, top 50 nodes

# Multiple output formats (install viz extras: pip install tenets[viz])
tenets viz deps . --format svg --output arch.svg    # SVG with Graphviz
tenets viz deps . --format png --output arch.png    # PNG image
tenets viz deps . --format html --output deps.html  # Interactive D3.js/Plotly
tenets viz deps . --format dot --output graph.dot   # Graphviz DOT format
tenets viz deps . --format json --output data.json  # Raw JSON data

# Advanced filtering and visualization
tenets viz deps src/ --include "*.py" --exclude "*test*"  # Filter specific files
tenets viz deps . --layout shell --max-nodes 75           # Shell layout with node limit
tenets viz deps tenets/core --level module --cluster-by directory  # Focused subsystem view

# Complexity analysis
tenets viz complexity . --hotspots

# Git integration - automatic, no setup (used for relevance, not shown in output)
tenets chronicle --since "last week"
tenets viz contributors --active
```

### Developer Productivity

Track velocity, identify bottlenecks, measure progress:

```bash
# Sprint velocity
tenets momentum --team --since "sprint-start"

# Code ownership
tenets examine . --ownership

# Technical debt trends
tenets examine . --complexity-trend
```

### Flexible Configuration

Works instantly with smart defaults, fully configurable when needed:

```yaml
# .tenets.yml (optional)
context:
  ranking: balanced # fast | balanced | thorough | ml
  include_git: true # Use git signals for relevance (not shown in output)
  max_tokens: 100000
  # Transformations are enabled per-invocation (CLI flags):
  #   --full, --remove-comments, --condense

ignore:
  - vendor/
  - '*.generated.*'

output:
  format: markdown # markdown, json, xml
```

### Content Transformation Flags

Optimize token usage or force raw context inclusion when needed:

| Flag                | Purpose                                                                    | Notes                                                      |
| ------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `--full`            | Include full file contents (no summarization) until token budget exhausted | Good for audits; may reduce breadth                        |
| `--remove-comments` | Strip line & block comments (language-aware heuristics)                    | Safety: aborts if >60% of non-empty lines would be removed |
| `--condense`        | Collapse 3+ blank lines to 1, trim trailing spaces                         | Lossless for logic; pairs well with `--remove-comments`    |

Order: comments removed first, whitespace condensed second. Both affect token counting and packing decisions.

### Session Management

Sessions help organize your work and maintain context across commands:

**Default Session Behavior:**

- When no `--session` is specified, commands use a persistent "default" session
- The default session saves tenets, pinned files, and context just like named sessions
- You can save the default session with a meaningful name later

```bash
# Working without explicit sessions (uses "default" automatically)
tenets tenet add "Use dependency injection"
tenets instill --add-file src/core/container.py
tenets distill "refactor service layer"

# Later, save your work with a proper name
tenets session save dependency-refactor
tenets session save di-work --delete-source  # Save and clean up default
```

**Session Commands:**

```bash
# Session lifecycle
tenets session create my-feature    # Create new session
tenets session list                  # Show all sessions
tenets session resume my-feature     # Switch to session
tenets session exit                  # Mark current as inactive
tenets session delete my-feature     # Remove session

# Save sessions with new names
tenets session save production --from debug-session
tenets session save final --from default --delete-source
```

### Pinned Files (Session Persistence)

Guarantee critical files are prioritized for a session:

```bash
tenets instill --session auth-refactor --add-file src/auth/service.py
tenets instill --session auth-refactor --add-folder src/auth/controllers
tenets instill --session auth-refactor --list-pinned
tenets distill "add OAuth device flow" --session auth-refactor --remove-comments
```

Pinned files are stored in session metadata (SQLite) and automatically reloaded—no extra arguments needed on subsequent `distill` commands.

### Feature Sets Explained

| Feature Set        | Includes                                                       | Use When                                                      |
| ------------------ | -------------------------------------------------------------- | ------------------------------------------------------------- |
| **core** (default) | Basic file scanning, keyword matching, git integration         | You want fast, lightweight context building                   |
| **light**          | + numpy, scikit-learn, YAKE keyword extraction, TF-IDF ranking | Better keyword/TF‑IDF relevance without heavy ML              |
| **viz**            | + matplotlib, networkx, dependency graphs, complexity charts   | You want to visualize your codebase                           |
| **ml**             | + PyTorch, transformers, semantic search, embeddings           | You want state-of-the-art ranking (slower, 2GB+ dependencies) |
| **web**            | + FastAPI, web UI (coming soon)                                | You want to run tenets as a service                           |
| **all**            | Everything above                                               | You want all features                                         |

#### Makefile Shortcuts

Common tasks are wrapped in the Makefile:

```bash
make dev      # editable install with all + dev extras
make install  # core editable install
make test     # run full test suite with coverage
make build    # build sdist + wheel
```

### Working with AI Assistants

```bash
# Generate context for ChatGPT/Claude
tenets distill "explain authentication flow" > context.md
# Or copy straight to clipboard
tenets distill "explain authentication flow" --copy
# Paste context.md into your AI chat

# Add guiding principles
tenets tenet add "Always validate user input"
tenets tenet add "Use async/await for I/O operations"
# Apply to future contexts
tenets instill

# Interactive session for back-and-forth
tenets session create "debug-memory-leak"
# AI: "Show me the memory allocation code"
tenets distill "malloc|alloc" --include "*.c" --session debug-memory-leak
# AI: "I need to see the cleanup functions"
tenets distill "free|cleanup" --include "*.c" --session debug-memory-leak
```

### Common Usage Patterns

```bash
# Quick exploration with fast mode
tenets distill "database models" --mode fast --copy

# Standard development with balanced mode (default)
tenets distill "implement user registration" -o context.md

# Deep analysis with thorough mode
tenets distill "security vulnerabilities" --mode thorough --max-tokens 150000

# Generate interactive HTML report for sharing
tenets distill "API architecture review" --format html -o api-review.html
# Open in browser for search, charts, and export features

# Combine modes with sessions for iterative work
tenets session create oauth-impl
tenets distill "OAuth2 flow" --mode fast --session oauth-impl  # Quick overview
tenets distill "refresh token handling" --mode thorough --session oauth-impl  # Deep dive

# Export context in different formats
tenets distill "payment processing" --format json | jq '.files[].path'  # List files
tenets distill "error handling" --format xml > context.xml  # For Claude
tenets distill "test coverage" --format html -o report.html  # Interactive report
```

### Test File Handling

Tenets intelligently handles test files to improve context relevance:

**Default Behavior (Recommended):**

- Test files are **excluded by default** for most prompts
- Tests are **automatically included** when prompt mentions testing
- Improves context quality by focusing on production code

```bash
# These prompts exclude tests (better context for understanding):
tenets distill "explain authentication flow"
tenets distill "how does user registration work"
tenets distill "debug payment processing"

# These prompts automatically include tests (detected by intent):
tenets distill "write unit tests for auth module"
tenets distill "fix failing tests"
tenets distill "improve test coverage"
tenets distill "debug test_user_registration.py"
```

**Manual Override:**

```bash
# Force include tests even for non-test prompts
tenets distill "understand auth flow" --include-tests

# Force exclude tests even for test-related prompts
tenets distill "fix failing tests" --exclude-tests

# Traditional manual filtering (still works)
tenets distill "review code for payments" --exclude "test_*,*_test.py,tests/**"
```

**Configuration:**

```yaml
# .tenets.yml - customize test patterns for your project
scanner:
  exclude_tests_by_default: true # default
  test_patterns:
    - 'test_*.py' # Python
    - '*_test.py'
    - '*.test.js' # JavaScript
    - '*.spec.ts' # TypeScript
    - '*Test.java' # Java
  test_directories:
    - 'tests'
    - '__tests__'
    - 'spec'
```

### Output Formats

Tenets supports multiple output formats for different use cases:

```bash
# Default markdown format (optimized for AI assistants)
tenets distill "implement OAuth2" --format markdown

# XML format (optimized for Claude)
tenets distill "implement OAuth2" --format xml -o context.xml

# JSON for programmatic use
tenets distill "implement OAuth2" --format json | jq .files[0]

# Interactive HTML report with visualizations
tenets distill "implement OAuth2" --format html -o report.html
```

**HTML Reports** include:

- 🔍 Live search to filter files
- 📋 Copy buttons for individual files
- 📥 Export to JSON/Markdown
- 📊 Interactive charts (file distribution, token usage)
- 🔄 Expand/collapse for full file content
- 📱 Responsive design for any screen size

### Exploration & Analysis

```bash
# Instant project overview
tenets examine .

# What's been happening?
tenets chronicle --since "1 month" --summary

# Visualize your architecture with intelligent project detection
tenets viz deps .  # Auto-detects Python/JS/Java/Go/etc and finds entry points
tenets viz deps . --cluster-by directory  # Group by directories
tenets viz deps . --level module  # Module-level dependencies (aggregated)
tenets viz deps . --level package --output packages.svg  # Package architecture

# Find complex areas
tenets examine . --complexity --threshold 10
```

## Architecture Visualization

Generate beautiful, interactive dependency graphs to understand your codebase structure:

### Quick Examples

```bash
# Install visualization dependencies
pip install tenets[viz]

# Auto-detect project and generate dependency graph (ASCII by default)
tenets viz deps

# Generate SVG dependency graph
tenets viz deps --output architecture.svg

# Interactive HTML for exploration
tenets viz deps --format html --output interactive.html
# Open in browser for D3.js/Plotly interactive graph

# Different views for different needs
tenets viz deps --level file      # Detailed file-level dependencies
tenets viz deps --level module    # Module-level aggregation (recommended)
tenets viz deps --level package   # High-level package architecture

# Understand specific subsystems
tenets viz deps src/api --include "*.py" --exclude "*test*" --output api.svg
tenets viz deps frontend/ --include "*.js,*.jsx" --format html -o frontend.html
```

### Real-World Usage

```bash
# For documentation - clean package architecture
tenets viz deps --level package --format png --output docs/architecture.png

# For code review - module dependencies with clustering
tenets viz deps --level module --cluster-by directory --format html -o review.html

# For refactoring - find tightly coupled components
tenets viz deps --layout circular --format svg --output coupling.svg

# For large projects - limit to most connected files
tenets viz deps --max-nodes 100 --format html --output top100.html
```

### Features

- **Auto-Detection**: Automatically identifies Python, Node.js, Java, Go, Rust, etc.
- **Smart Aggregation**: File → Module → Package level views
- **Multiple Formats**: ASCII, SVG, PNG, HTML, DOT, JSON
- **Interactive HTML**: Explore dependencies with D3.js/Plotly
- **Pure Python**: No system dependencies, just `pip install`
- **Filtering**: Include/exclude patterns for focused analysis
- **Layouts**: Hierarchical, circular, shell, force-directed
- **Clustering**: Group by directory, module, or package

## Examination & Reports

Analyze complexity, hotspots, and ownership from the CLI, and export rich reports.

Basic terminal summary:

```bash
tenets examine .
```

Ownership and hotspots in terminal:

```bash
tenets examine . --ownership --hotspots --show-details
```

Generate an HTML report (opens well in a browser):

```bash
tenets examine . -f html -o code_report.html --ownership --hotspots --show-details
```

Other formats:

```bash
# JSON (for automation)
tenets examine . -f json -o report.json

# Markdown (for docs/PRs)
tenets examine . -f markdown -o report.md
```

Useful options:

- Filtering: `-i/--include "*.py,src/**/*.js"`, `-e/--exclude "tests/**,**/*.min.js"`
- Complexity threshold: `-t/--threshold 12`
- Depth limit: `--max-depth 6`
- Pick metrics explicitly: `-m cyclomatic -m cognitive`

Notes:

- If `-o/--output` is omitted with `-f html|markdown|json`, the file defaults to `examination_report.<format>` in the current directory.
- Sections shown include Complexity Analysis, Hotspot Analysis, Code Ownership, Overall Metrics, and a summary/health score when available.

## Advanced Features

### Multi-Factor Ranking

Choose the algorithm that fits your needs:

- **fast** - Quick keyword matching (always available)
- **balanced** - Keywords + structure + git (requires `light` extra)
- **thorough** - Deep analysis with AST + ML (requires `ml` extra)

### Smart Summarization

When files exceed token budgets, tenets intelligently preserves:

- Function/class signatures
- Import statements (with smart condensing - see below)
- Complex logic blocks
- Recent changes
- Documentation

#### Import Summarization

Tenets can automatically condense verbose import statements into concise summaries:

```python
# Instead of showing 20+ import lines:
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
# ... many more imports

# Tenets produces:
# Imports: 20 total
# Dependencies: numpy, pandas, requests, flask, sqlalchemy
# Local imports: 3
```

Control import summarization:
```bash
# Enabled by default (condenses imports when > 5 lines)
tenets distill "review code"

# Disable import summarization - show all imports verbatim
tenets distill "review code" --no-summarize-imports

# Configure via .tenets.yaml
summarizer:
  summarize_imports: true  # Enable/disable globally
  import_summary_threshold: 5  # Min imports to trigger summarization
```

#### Context-Aware Documentation Summarization

For documentation files (Markdown, config files, API docs), tenets provides intelligent context-aware summarization:

- **Smart Section Selection**: Automatically identifies and prioritizes sections containing references to your prompt keywords
- **Multi-level Relevance**: Uses direct keyword matches, semantic similarity, and contextual analysis
- **In-place Context Preservation**: Maintains relevant context sections within the summary for better understanding
- **Code Example Preservation**: Always preserves code snippets and configuration examples from relevant sections
- **Multi-format Support**: Works with Markdown, YAML, JSON, TOML, INI files, and more

```bash
# Documentation summarization automatically activates for docs files
tenets distill "configure OAuth2 authentication" docs/
# Returns focused summary with auth-related sections and code examples

# Works with API documentation
tenets distill "user management endpoints" api-docs.md
# Highlights relevant API endpoints and request/response examples

# Configuration file analysis
tenets distill "database connection settings" config/
# Shows relevant config sections with actual values and comments
```

**Benefits**:

- **Focused Context**: Get exactly the documentation sections you need
- **Preserved Examples**: Code snippets and configs stay intact
- **Intelligent Filtering**: Irrelevant sections are filtered out
- **Configurable**: Adjust search depth, confidence thresholds, and section limits

## 📚 Documentation

Core docs in `docs/`:

- [CLI Reference](docs/CLI.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Deep Dive](docs/DEVELOPMENT.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Testing Guide](docs/TESTING.md)

See `docs/CONFIG.md` for configuration details.

Full documentation is available at https://tenets.dev/docs.

### Building Documentation Locally

```bash
# Serve locally with live reload (includes API docs generation)
make docs
# Or directly:
mkdocs serve

# Quick dev mode (no API generation - faster for styling/content updates)
make docs-dev
# Or directly (skips API generation, uses dirty reload):
mkdocs serve --livereload --dirtyreload

# Build static site
make docs-build
# Or directly:
mkdocs build


  Fix test failures and suppress pytest ResourceWarningsFixed tenet command tests by correcting mock
  method signatures to use keyword arguments, adjusted instiller injection logic for adaptive mode to
  respect complexity thresholds on first distill, and suppressed coverage.py SQLite ResourceWarnings
  in pytest configuration.

# Deploy to GitHub Pages (requires permissions)
make docs-deploy
# Or directly:
mkdocs gh-deploy
```

**Note:** Use `make docs-dev` when working on documentation content, styling, or layout. It skips the API documentation generation step, making the server start much faster and reload more quickly. The `--dirtyreload` flag only rebuilds changed pages instead of the entire site.

## Configuration (quick start)

- Create a starter file: `tenets config init` (writes `.tenets.yml` at project root)
- Lower threshold to include more files:

```yaml
ranking:
  algorithm: fast
  threshold: 0.05
  use_tfidf: true
  use_stopwords: false
output:
  copy_on_distill: false # set true to always copy distilled context to clipboard
```

- Or set environment variables for one run:

```bash
TENETS_RANKING_THRESHOLD=0.05 TENETS_RANKING_ALGORITHM=fast tenets distill "implement OAuth2" .

# Copy directly (one-off)
tenets distill "implement OAuth2" --copy

# Or enable globally in config (.tenets.yml)
output:
  copy_on_distill: true
```

See the full guide: `docs/CONFIG.md`.

## Storage & Cache

Tenets stores all writable state in a user/project cache directory. This avoids writing into the installed package location (which may be read‑only under pip installs).

- Default cache directory: `${HOME}/.tenets/cache` (Windows: `%USERPROFILE%\.tenets\cache`)
- Main SQLite database: `${CACHE_DIR}/tenets.db`
- Subdirectories:
  - `${CACHE_DIR}/analysis/` – analysis cache (SQLite)
  - `${CACHE_DIR}/general/` – general cache (SQLite)

You can change the cache location via config or environment variables:

- Config file `.tenets.yml`:
  ```yaml
  cache:
    directory: /path/to/custom/cache
  ```
- Environment variable: `TENETS_CACHE_DIRECTORY=/path/to/custom/cache`

Sessions are persisted to the main SQLite database by default when a `TenetsConfig` is provided. The `SessionManager` uses an in‑memory mirror for speed and will write session metadata and context snapshots to `${CACHE_DIR}/tenets.db`.

## Supported languages and formats

The analyzer includes specialized parsers for many languages and formats. Files are routed based on extensions; HTML analyzer covers HTML and Vue SFCs. JSX/TSX are handled by the JavaScript analyzer.

### Languages with dedicated analyzers

| Language              | Analyzer           | Extensions                                                 |
| --------------------- | ------------------ | ---------------------------------------------------------- |
| Python                | PythonAnalyzer     | .py, .pyw, .pyi                                            |
| JavaScript/TypeScript | JavaScriptAnalyzer | .js, .jsx, .ts, .tsx, .mjs, .cjs                           |
| HTML + Vue SFC        | HTMLAnalyzer       | .html, .htm, .xhtml, .vue                                  |
| CSS/SCSS/Sass/Less    | CSSAnalyzer        | .css, .scss, .sass, .less, .styl, .stylus, .pcss, .postcss |
| Go                    | GoAnalyzer         | .go                                                        |
| Java                  | JavaAnalyzer       | .java                                                      |
| C/C++                 | CppAnalyzer        | .c, .cc, .cpp, .cxx, .c++, .h, .hh, .hpp, .hxx, .h++       |
| Ruby                  | RubyAnalyzer       | .rb, .rake, .gemspec, .ru                                  |
| PHP                   | PhpAnalyzer        | .php, .phtml, .inc, .php3, .php4, .php5, .phps             |
| Rust                  | RustAnalyzer       | .rs                                                        |
| Dart (Flutter aware)  | DartAnalyzer       | .dart                                                      |
| Kotlin                | KotlinAnalyzer     | .kt, .kts                                                  |
| Scala                 | ScalaAnalyzer      | .scala, .sc                                                |
| Swift                 | SwiftAnalyzer      | .swift                                                     |
| C#                    | CSharpAnalyzer     | .cs, .csx                                                  |
| GDScript (Godot)      | GDScriptAnalyzer   | .gd, .tres, .tscn                                          |

### Configuration, docs, and structured text (GenericAnalyzer)

These formats are analyzed with GenericAnalyzer. YAML files include heuristics for common ecosystems.

| Category                | Examples                                                                                      | Extensions / Filenames                           | Notes                                                                                                                        |
| ----------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| YAML (Compose/K8s/etc.) | docker-compose.yml, deployment.yaml, chart.yaml, kustomization.yaml, .github/workflows/\*.yml | .yaml, .yml                                      | Detects Docker Compose (services, images), Kubernetes (apiVersion/kind, images, refs), hints Helm, Kustomize, GitHub Actions |
| TOML                    | pyproject.toml, Cargo.toml                                                                    | .toml                                            | Extracts top-level keys                                                                                                      |
| INI/CFG/CONF            | app.ini, settings.cfg, nginx.conf, my.cnf                                                     | .ini, .cfg, .conf, .cnf                          | Sections and keys parsed                                                                                                     |
| Properties              | application.properties                                                                        | .properties, .props                              | Key/value parsing                                                                                                            |
| ENV files               | .env, .env.local, .env.production                                                             | .env, .env.\*                                    | Routed to Generic by filename                                                                                                |
| JSON/XML                | package.json, config.json, pom.xml                                                            | .json, .xml                                      | JSON deps/keys detected                                                                                                      |
| Markdown                | README.md, docs/\*.mdx                                                                        | .md, .markdown, .mdx, .mdown, .mkd, .mkdn, .mdwn | Sections and headings extracted                                                                                              |
| SQL                     | schema.sql, queries.sql                                                                       | .sql                                             | Basic metrics only                                                                                                           |
| Lock/HashiCorp          | yarn.lock, _.tf, _.tfvars, \*.hcl                                                             | .lock, .tf, .tfvars, .hcl                        | Routed to Generic                                                                                                            |
| Shell scripts           | build.sh, hooks/\*.bash                                                                       | .sh, .bash, .zsh, .fish                          | Routed to Generic                                                                                                            |
| Special files           | Dockerfile, Makefile, CMakeLists.txt, .gitignore, .editorconfig, .npmrc, .yarnrc, .nvmrc      | by name                                          | Routed to Generic by special-name handling                                                                                   |

Notes

- JSX/TSX are owned by JavaScriptAnalyzer; HTMLAnalyzer focuses on HTML and Vue SFCs.
- YAML heuristics set structure.framework (e.g., docker-compose, kubernetes) and populate modules with services/resources.
- Generic analyzer extracts imports/references from config where possible (images, depends_on, ConfigMaps/Secrets, etc.).

## Python API

Use Tenets programmatically in your Python projects:

```python
from tenets import Tenets
from pathlib import Path

# Initialize
tenets = Tenets()

# Basic context extraction (uses default session)
result = tenets.distill("implement user authentication")
print(f"Generated {result.token_count} tokens")
print(result.context[:500])  # First 500 chars

# Get ranked files without content (much faster!)
from tenets.core.ranking import RelevanceRanker
from tenets.core.prompt import PromptParser

# Parse prompt and get files
parser = PromptParser()
prompt_context = parser.parse("implement OAuth2")

# Scan and analyze
from tenets.core.scanner import FileScanner
from tenets.core.analysis.analyzer import CodeAnalyzer

scanner = FileScanner()
files = scanner.scan("./src")
analyzer = CodeAnalyzer()
analyzed = [analyzer.analyze_file(f) for f in files]

# Rank files
ranker = RelevanceRanker(algorithm="balanced")
ranked_files = ranker.rank(analyzed, prompt_context, threshold=0.1)

# Use the rankings
for file in ranked_files[:10]:
    print(f"{file.path}: {file.relevance_score:.3f}")
    if hasattr(file, 'relevance_factors'):
        for factor, score in file.relevance_factors.items():
            print(f"  - {factor}: {score:.2%}")

# Generate interactive HTML report
result = tenets.distill("review API design", format="html")
Path("api-review.html").write_text(result.context)

# Add guiding principles
tenets.add_tenet("Use type hints", priority="high")
tenets.add_tenet("Follow SOLID principles", category="architecture")
tenets.instill_tenets()

# Pin critical files
tenets.pin_file("src/core/auth.py")
tenets.pin_folder("src/api/")

# Named sessions for organized work
result = tenets.distill(
    "implement OAuth2",
    session_name="oauth-feature",
    mode="thorough",
    max_tokens=100000
)

# Custom configuration
from tenets.config import TenetsConfig

config = TenetsConfig(
    max_tokens=150000,
    ranking_algorithm="thorough",
    model="claude-3-opus"
)
tenets = Tenets(config)

# Analyze codebase
analysis = tenets.examine("./src")
print(f"Health Score: {analysis.health_score}")
print(f"Complexity Issues: {analysis.complexity.high_complexity_count}")
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

_Development sponsored by [manic.agency](https://manic.agency)_
