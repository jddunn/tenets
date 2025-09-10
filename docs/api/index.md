---
title: API Reference
---

# API Reference

Complete API documentation for the Tenets package.

## Quick Links

- [Core Module](#core-module) - Main functionality
- [CLI Reference](#cli-reference) - Command-line interface
- [Models](#models) - Data structures
- [Utils](#utilities) - Helper functions

## Core Module

### Main Classes

#### `Tenets`
The main entry point for the Tenets library.

```python
from tenets import Tenets

t = Tenets(config=None)
```

**Methods:**
- `analyze(path, **kwargs)` - Analyze a codebase
- `get_context(**kwargs)` - Get code context
- `instill(path, **kwargs)` - Generate instilled context

#### `Config`
Configuration class for customizing Tenets behavior.

```python
from tenets import Config

config = Config(
    max_depth=5,
    include_tests=False,
    ranking_method="bm25"  # or "tfidf"
)
```

### Core Submodules

#### `tenets.core.analysis`
Code analysis and parsing functionality.

**Key Classes:**
- `Analyzer` - Main analysis orchestrator
- `FileAnalyzer` - Individual file analysis
- `SymbolExtractor` - Extract code symbols

#### `tenets.core.ranking`
Code ranking and relevance scoring.

**Key Classes:**
- `RankingEngine` - Main ranking interface
- `BM25Ranker` - BM25 ranking implementation
- `TFIDFRanker` - TF-IDF ranking implementation

#### `tenets.core.session`
Session management and state tracking.

**Key Classes:**
- `Session` - Session state manager
- `SessionConfig` - Session configuration

#### `tenets.core.instiller`
Context instillation and prompt generation.

**Key Classes:**
- `Instiller` - Main instillation engine
- `ContextBuilder` - Build context for prompts

#### `tenets.core.git`
Git repository integration.

**Key Classes:**
- `GitManager` - Git operations wrapper
- `GitAnalyzer` - Analyze git history

## CLI Reference

The Tenets CLI provides commands for code analysis and context generation.

### Basic Usage

```bash
# Analyze current directory
tenets analyze .

# Generate context
tenets instill . --output context.md

# With specific configuration
tenets analyze . --config tenets.yaml
```

### Commands

#### `analyze`
Analyze a codebase and generate insights.

```bash
tenets analyze <path> [OPTIONS]
```

**Options:**
- `--output, -o` - Output file path
- `--format` - Output format (json, yaml, markdown)
- `--max-depth` - Maximum directory depth
- `--include` - Include patterns
- `--exclude` - Exclude patterns

#### `instill`
Generate instilled context for AI prompts.

```bash
tenets instill <path> [OPTIONS]
```

**Options:**
- `--output, -o` - Output file path
- `--max-tokens` - Maximum token count
- `--ranking-method` - Ranking method (bm25, tfidf)
- `--top-k` - Number of top results

#### `config`
Manage Tenets configuration.

```bash
tenets config [OPTIONS]
```

**Options:**
- `--show` - Show current configuration
- `--init` - Initialize configuration file
- `--validate` - Validate configuration

## Models

### Data Structures

#### `CodeContext`
Represents analyzed code context.

**Attributes:**
- `files` - List of analyzed files
- `symbols` - Extracted symbols
- `dependencies` - Dependency graph
- `metrics` - Code metrics

#### `FileInfo`
Information about a single file.

**Attributes:**
- `path` - File path
- `language` - Programming language
- `content` - File content
- `symbols` - File symbols
- `imports` - Import statements

#### `Symbol`
Represents a code symbol (class, function, etc.).

**Attributes:**
- `name` - Symbol name
- `type` - Symbol type
- `line` - Line number
- `docstring` - Documentation
- `signature` - Function signature

## Utilities

### Helper Functions

#### File Operations
- `read_file(path)` - Read file content
- `write_file(path, content)` - Write file
- `find_files(pattern)` - Find files by pattern

#### Text Processing
- `tokenize(text)` - Tokenize text
- `extract_keywords(text)` - Extract keywords
- `summarize(text)` - Generate summary

#### Language Detection
- `detect_language(file)` - Detect programming language
- `get_parser(language)` - Get language parser

## Configuration

### Configuration File Format

```yaml
# tenets.yaml
version: 1

analysis:
  max_depth: 5
  include_tests: false
  follow_symlinks: false

ranking:
  method: bm25  # or tfidf
  top_k: 20
  min_score: 0.1

output:
  format: markdown
  max_tokens: 8000
  include_metadata: true

languages:
  - python
  - javascript
  - typescript
  - go
  - rust

exclude:
  - "**/node_modules/**"
  - "**/.venv/**"
  - "**/dist/**"
  - "**/build/**"
```

## Examples

### Basic Analysis

```python
from tenets import Tenets

# Initialize
t = Tenets()

# Analyze a project
context = t.analyze('path/to/project')

# Get ranked files
ranked = context.get_ranked_files(query="authentication")

# Generate context
instilled = t.instill(
    'path/to/project',
    max_tokens=4000,
    ranking_method="bm25"
)
```

### Custom Configuration

```python
from tenets import Tenets, Config

config = Config(
    max_depth=3,
    include_tests=False,
    languages=['python', 'javascript'],
    exclude_patterns=['**/test_*.py']
)

t = Tenets(config=config)
context = t.analyze('.')
```

### CLI Integration

```python
import subprocess
import json

# Run analysis via CLI
result = subprocess.run(
    ['tenets', 'analyze', '.', '--format', 'json'],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
```

## API Stability

The Tenets API follows semantic versioning. The current version is **0.1.0** (alpha).

- Public API methods are considered stable
- Internal methods (prefixed with `_`) may change
- Configuration format is stable
- CLI commands are stable

## Further Reading

- [Architecture Guide](../ARCHITECTURE.md) - System design and architecture
- [Development Guide](../development.md) - Contributing and development
- [CLI Documentation](../CLI.md) - Detailed CLI reference
- [Configuration Guide](../CONFIG.md) - Configuration options
