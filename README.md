# **tenets**

<a href="https://tenets.dev"><img src="./docs/logos/tenets_dark_icon_transparent.png" alt="tenets logo" width="140" /></a>

**context that feeds your prompts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tenets.svg)](https://pypi.org/project/tenets/)
[![CI](https://github.com/jddunn/tenets/actions/workflows/ci.yml/badge.svg)](https://github.com/jddunn/tenets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jddunn/tenets/branch/main/graph/badge.svg)](https://codecov.io/gh/jddunn/tenets)
[![Coverage Status](https://coveralls.io/repos/github/jddunn/tenets/badge.svg?branch=main)](https://coveralls.io/github/jddunn/tenets?branch=main)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.tenets.dev)

**tenets** automatically finds and builds the most relevant context from your codebase. Instead of manually copying files or searching for documentation, tenets intelligently aggregates exactly what you need - whether you're debugging, building features, or chatting with an AI assistant.

## What is tenets?

Think of it as intelligent context aggregation. You give it a prompt or query, and it:

- 🔍 **Finds** all relevant files automatically 
- 🎯 **Ranks** them by importance using multiple factors
- 📦 **Aggregates** them within your token budget
- 📋 **Formats** perfectly for any use case
- 📌 **Pins** critical files per session for guaranteed inclusion priority
- 🧹 **Transforms** content on demand (strip comments, condense whitespace, or force full raw context)

Plus powerful development intelligence:

- **Visualize** dependencies and architecture 
- **Track** velocity and code evolution  
- **Identify** hotspots and technical debt
- **Understand** team patterns and expertise

## Installation / Quick Start

### Pip

```bash
# Core features only - lightweight, no ML dependencies
pip install tenets

# Add specific features
pip install tenets[light]  # Adds numpy, scikit-learn for TF-IDF ranking
pip install tenets[viz]    # Adds visualization capabilities
pip install tenets[ml]     # Adds deep learning models (large dependencies)

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

## How it Works

1. **Scans** your codebase respecting .gitignore
2. **Analyzes** code structure, imports, and dependencies
3. **Ranks** files using multi-factor scoring:
   - Keyword matching (TF-IDF with `light` extra)
   - Import relationships
   - Git activity and recency (used for relevance; not shown in output)
   - Code complexity
   - Path relevance
   - Semantic similarity (with `ml` extra)
4. **Aggregates** intelligently within token limits
5. **Formats** for optimal consumption

*All processing is local. Your code never leaves your machine. By design tenets never offloads any logic to LLMs, using classical ML / NLP techniques when desired.*

## Key Features

### Intelligent Context Distillation

Like repomix on steroids - smart filters, automatic relevance ranking, and configurable aggregation:

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

### Code Intelligence & Visualization

Understand your codebase at a glance:

```bash
# Dependency graphs
tenets viz deps . --output architecture.svg

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
  ranking: balanced  # fast | balanced | thorough | ml
  include_git: true  # Use git signals for relevance (not shown in output)
  max_tokens: 100000
  # Transformations are enabled per-invocation (CLI flags):
  #   --full, --remove-comments, --condense

ignore:
  - vendor/
  - "*.generated.*"
  
output:
  format: markdown  # markdown, json, xml
```

### Content Transformation Flags

Optimize token usage or force raw context inclusion when needed:

| Flag | Purpose | Notes |
|------|---------|-------|
| `--full` | Include full file contents (no summarization) until token budget exhausted | Good for audits; may reduce breadth |
| `--remove-comments` | Strip line & block comments (language-aware heuristics) | Safety: aborts if >60% of non-empty lines would be removed |
| `--condense` | Collapse 3+ blank lines to 1, trim trailing spaces | Lossless for logic; pairs well with `--remove-comments` |

Order: comments removed first, whitespace condensed second. Both affect token counting and packing decisions.

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

| Feature Set | Includes | Use When |
|-------------|----------|----------|
| **core** (default) | Basic file scanning, keyword matching, git integration | You want fast, lightweight context building |
| **light** | + numpy, scikit-learn, YAKE keyword extraction, TF-IDF ranking | Better keyword/TF‑IDF relevance without heavy ML |
| **viz** | + matplotlib, networkx, dependency graphs, complexity charts | You want to visualize your codebase |
| **ml** | + PyTorch, transformers, semantic search, embeddings | You want state-of-the-art ranking (slower, 2GB+ dependencies) |
| **web** | + FastAPI, web UI (coming soon) | You want to run tenets as a service |
| **all** | Everything above | You want all features |

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

### Exploration & Analysis

```bash
# Instant project overview
tenets examine .

# What's been happening?
tenets chronicle --since "1 month" --summary

# Visualize your architecture
tenets viz deps . --cluster-by directory

# Find complex areas
tenets examine . --complexity --threshold 10
```

## Advanced Features

### Multi-Factor Ranking

Choose the algorithm that fits your needs:

- **fast** - Quick keyword matching (always available)
- **balanced** - Keywords + structure + git (requires `light` extra)
- **thorough** - Deep analysis with AST + ML (requires `ml` extra)

### Smart Summarization

When files exceed token budgets, tenets intelligently preserves:

- Function/class signatures
- Import statements
- Complex logic blocks
- Recent changes
- Documentation

## 📚 Documentation

Core docs in `docs/`:

- [CLI Reference](docs/CLI.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Deep Dive](docs/DEVELOPMENT.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Testing Guide](docs/TESTING.md)

See `docs/CONFIG.md` for configuration details.

Full documentation is available at https://docs.tenets.dev.

### Building Documentation Locally

```bash
# Serve locally with live reload
make docs
# Or directly:
mkdocs serve

# Build static site
make docs-build
# Or directly:
mkdocs build

# Deploy to GitHub Pages (requires permissions)
make docs-deploy
# Or directly:
mkdocs gh-deploy
```
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
  copy_on_distill: false  # set true to always copy distilled context to clipboard
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

| Language | Analyzer | Extensions |
|---|---|---|
| Python | PythonAnalyzer | .py, .pyw, .pyi |
| JavaScript/TypeScript | JavaScriptAnalyzer | .js, .jsx, .ts, .tsx, .mjs, .cjs |
| HTML + Vue SFC | HTMLAnalyzer | .html, .htm, .xhtml, .vue |
| CSS/SCSS/Sass/Less | CSSAnalyzer | .css, .scss, .sass, .less, .styl, .stylus, .pcss, .postcss |
| Go | GoAnalyzer | .go |
| Java | JavaAnalyzer | .java |
| C/C++ | CppAnalyzer | .c, .cc, .cpp, .cxx, .c++, .h, .hh, .hpp, .hxx, .h++ |
| Ruby | RubyAnalyzer | .rb, .rake, .gemspec, .ru |
| PHP | PhpAnalyzer | .php, .phtml, .inc, .php3, .php4, .php5, .phps |
| Rust | RustAnalyzer | .rs |
| Dart (Flutter aware) | DartAnalyzer | .dart |
| Kotlin | KotlinAnalyzer | .kt, .kts |
| Scala | ScalaAnalyzer | .scala, .sc |
| Swift | SwiftAnalyzer | .swift |
| C# | CSharpAnalyzer | .cs, .csx |
| GDScript (Godot) | GDScriptAnalyzer | .gd, .tres, .tscn |

### Configuration, docs, and structured text (GenericAnalyzer)

These formats are analyzed with GenericAnalyzer. YAML files include heuristics for common ecosystems.

| Category | Examples | Extensions / Filenames | Notes |
|---|---|---|---|
| YAML (Compose/K8s/etc.) | docker-compose.yml, deployment.yaml, chart.yaml, kustomization.yaml, .github/workflows/*.yml | .yaml, .yml | Detects Docker Compose (services, images), Kubernetes (apiVersion/kind, images, refs), hints Helm, Kustomize, GitHub Actions |
| TOML | pyproject.toml, Cargo.toml | .toml | Extracts top-level keys |
| INI/CFG/CONF | app.ini, settings.cfg, nginx.conf, my.cnf | .ini, .cfg, .conf, .cnf | Sections and keys parsed |
| Properties | application.properties | .properties, .props | Key/value parsing |
| ENV files | .env, .env.local, .env.production | .env, .env.* | Routed to Generic by filename |
| JSON/XML | package.json, config.json, pom.xml | .json, .xml | JSON deps/keys detected |
| Markdown | README.md, docs/*.mdx | .md, .markdown, .mdx, .mdown, .mkd, .mkdn, .mdwn | Sections and headings extracted |
| SQL | schema.sql, queries.sql | .sql | Basic metrics only |
| Lock/HashiCorp | yarn.lock, *.tf, *.tfvars, *.hcl | .lock, .tf, .tfvars, .hcl | Routed to Generic |
| Shell scripts | build.sh, hooks/*.bash | .sh, .bash, .zsh, .fish | Routed to Generic |
| Special files | Dockerfile, Makefile, CMakeLists.txt, .gitignore, .editorconfig, .npmrc, .yarnrc, .nvmrc | by name | Routed to Generic by special-name handling |

Notes
- JSX/TSX are owned by JavaScriptAnalyzer; HTMLAnalyzer focuses on HTML and Vue SFCs.
- YAML heuristics set structure.framework (e.g., docker-compose, kubernetes) and populate modules with services/resources.
- Generic analyzer extracts imports/references from config where possible (images, depends_on, ConfigMaps/Secrets, etc.).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Development sponsored by [manic.agency](https://manic.agency)*