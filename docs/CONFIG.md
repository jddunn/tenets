# Configuration Guide

Comprehensive guide to configuring Tenets for optimal code context building.

## Overview

Tenets uses a hierarchical configuration system with multiple override levels:

**Precedence (lowest → highest):**
1. Default configuration (built-in)
2. Project file (`.tenets.yml` at repo root)
3. User file (`~/.config/tenets/config.yml` or `~/.tenets.yml`)
4. Environment variables (`TENETS_*`)
5. CLI flags (`--mode`, `--max-tokens`, etc.)
6. Programmatic overrides (`Tenets(config=...)`)

**Inspect configuration:**
```bash
tenets config show                # Full config
tenets config show --key ranking  # Specific section
tenets config show --format json  # JSON output
```

## Files and locations

Tenets searches these locations in order and uses the first it finds:
- ./\.tenets.yml
- ./\.tenets.yaml
- ./tenets.yml
- ./.config/tenets.yml
- ~/.config/tenets/config.yml
- ~/.tenets.yml

Create a starter file:

- tenets config init  # writes .tenets.yml in the current directory

## Complete Configuration Schema

All available configuration sections and their options:

```yaml
# ============= Core Settings =============
max_tokens: 100000          # Maximum tokens for context (default: 100000)
debug: false                # Enable debug logging
quiet: false                # Suppress non-essential output

# ============= File Scanning =============
scanner:
  respect_gitignore: true          # Honor .gitignore patterns
  follow_symlinks: false           # Follow symbolic links
  max_file_size: 5000000          # Max file size in bytes (5MB)
  max_files: 10000                # Maximum files to scan
  binary_check: true              # Skip binary files
  encoding: utf-8                 # File encoding
  workers: 4                      # Parallel scanning workers
  parallel_mode: auto             # auto | thread | process
  timeout: 5.0                    # Timeout per file (seconds)
  exclude_minified: true          # Skip minified files
  exclude_tests_by_default: true  # Skip test files unless explicit

  # Ignore patterns (in addition to .gitignore)
  additional_ignore_patterns:
    - '*.generated.*'
    - vendor/
    - node_modules/
    - '*.egg-info/'
    - __pycache__/
    - .pytest_cache/

  # Test file patterns
  test_patterns:
    - test_*.py
    - '*_test.py'
    - '*.test.js'
    - '*.spec.ts'

  # Test directories
  test_directories:
    - test
    - tests
    - __tests__
    - spec

# ============= Ranking System =============
ranking:
  algorithm: balanced             # fast | balanced | thorough | ml | custom
  threshold: 0.10                 # 0.0-1.0 (lower includes more files)
  text_similarity_algorithm: bm25 # bm25 (recommended) | tfidf (experimental)
  # Note: TF-IDF is available for experimentation but BM25 is significantly faster
  use_stopwords: false           # Filter common tokens
  use_embeddings: false          # Semantic similarity (requires ML)
  use_git: true                  # Include git signals
  use_ml: false                  # Machine learning features
  embedding_model: all-MiniLM-L6-v2  # Embedding model name
  workers: 2                     # Parallel ranking workers
  parallel_mode: auto            # thread | process | auto
  batch_size: 100               # Files per batch

  # Custom factor weights (0.0-1.0)
  custom_weights:
    keyword_match: 0.25
    path_relevance: 0.20
    import_graph: 0.20
    git_activity: 0.15
    file_type: 0.10
    complexity: 0.10

# ============= Summarization =============
summarizer:
  default_mode: auto             # auto | extractive | abstractive
  target_ratio: 0.3              # Target compression ratio
  enable_cache: true             # Cache summaries
  preserve_code_structure: true  # Keep code structure intact
  summarize_imports: true        # Condense import statements
  import_summary_threshold: 5    # Min imports to trigger summary
  max_cache_size: 100           # Max cached summaries
  quality_threshold: medium      # low | medium | high
  batch_size: 10                # Files per batch
  docstring_weight: 0.5         # Weight for docstrings
  include_all_signatures: true   # Include all function signatures

  # LLM settings (optional)
  llm_provider: null            # openai | anthropic | null
  llm_model: null               # Model name
  llm_temperature: 0.3          # Creativity (0.0-1.0)
  llm_max_tokens: 500           # Max tokens per summary
  enable_ml_strategies: false    # Use ML summarization

# ============= Tenet System =============
tenet:
  auto_instill: true              # Auto-apply tenets
  max_per_context: 5              # Max tenets per context
  reinforcement: true             # Reinforce important tenets
  injection_strategy: strategic   # strategic | sequential | random
  min_distance_between: 1000      # Min chars between injections
  prefer_natural_breaks: true     # Insert at natural boundaries
  storage_path: ~/.tenets/tenets  # Tenet storage location
  collections_enabled: true       # Enable tenet collections

  # Injection frequency
  injection_frequency: adaptive   # always | periodic | adaptive | manual
  injection_interval: 3           # For periodic mode
  session_complexity_threshold: 0.7  # Triggers adaptive injection
  min_session_length: 5           # Min prompts before injection

  # Advanced settings
  adaptive_injection: true        # Smart injection timing
  track_injection_history: true   # Track what was injected
  decay_rate: 0.1                # How fast tenets decay
  reinforcement_interval: 10      # Reinforce every N prompts
  session_aware: true            # Use session context
  session_memory_limit: 100      # Max session history
  persist_session_history: true   # Save session data

  # Priority settings
  priority_boost_critical: 2.0    # Boost for critical tenets
  priority_boost_high: 1.5       # Boost for high priority
  skip_low_priority_on_complex: true  # Skip low priority when complex

  # System instruction
  system_instruction: null        # Global system instruction
  system_instruction_enabled: false  # Enable system instruction
  system_instruction_position: top   # top | bottom
  system_instruction_format: markdown  # markdown | plain
  system_instruction_once_per_session: true  # Inject once per session

# ============= Caching =============
cache:
  enabled: true                  # Enable caching
  directory: ~/.tenets/cache     # Cache directory
  ttl_days: 7                   # Time to live (days)
  max_size_mb: 500              # Max cache size (MB)
  compression: false            # Compress cache data
  memory_cache_size: 1000       # In-memory cache entries
  max_age_hours: 24            # Max cache age (hours)

  # SQLite settings
  sqlite_pragmas:
    journal_mode: WAL
    synchronous: NORMAL
    cache_size: '-64000'
    temp_store: MEMORY

  # LLM cache
  llm_cache_enabled: true       # Cache LLM responses
  llm_cache_ttl_hours: 24      # LLM cache TTL

# ============= Output Formatting =============
output:
  default_format: markdown       # markdown | xml | json | html
  syntax_highlighting: true      # Enable syntax highlighting
  line_numbers: false           # Show line numbers
  max_line_length: 120          # Max line length
  include_metadata: true        # Include metadata
  compression_threshold: 10000  # Compress if larger (chars)
  summary_ratio: 0.25           # Summary compression ratio
  copy_on_distill: false        # Auto-copy to clipboard
  show_token_usage: true        # Show token counts
  show_cost_estimate: true      # Show LLM cost estimates

# ============= Git Integration =============
git:
  enabled: true                 # Use git information
  include_history: true         # Include commit history
  history_limit: 100           # Max commits to analyze
  include_blame: false         # Include git blame
  include_stats: true          # Include statistics

  # Ignore these authors
  ignore_authors:
    - dependabot[bot]
    - github-actions[bot]
    - renovate[bot]

  # Main branch names
  main_branches:
    - main
    - master
    - develop
    - trunk

# ============= NLP Settings =============
nlp:
  enabled: true                    # Enable NLP features
  stopwords_enabled: true          # Use stopwords
  code_stopword_set: minimal       # minimal | standard | aggressive
  prompt_stopword_set: aggressive  # minimal | standard | aggressive
  custom_stopword_files: []        # Custom stopword files

  # Tokenization
  tokenization_mode: auto          # auto | simple | advanced
  preserve_original_tokens: true   # Keep original tokens
  split_camelcase: true           # Split CamelCase
  split_snakecase: true           # Split snake_case
  min_token_length: 2             # Min token length

  # Keyword extraction
  keyword_extraction_method: auto  # auto | rake | yake | tfidf
  max_keywords: 30                # Max keywords to extract
  ngram_size: 3                  # N-gram size
  yake_dedup_threshold: 0.7      # YAKE deduplication

  # BM25 settings
  bm25_k1: 1.2                   # Term frequency saturation parameter
  bm25_b: 0.75                   # Length normalization parameter

  # TF-IDF settings (experimental - use only if needed)
  tfidf_use_sublinear: true      # Sublinear TF scaling
  tfidf_use_idf: true           # Use IDF
  tfidf_norm: l2                # Normalization

  # Embeddings
  embeddings_enabled: false       # Enable embeddings
  embeddings_model: all-MiniLM-L6-v2  # Model name
  embeddings_device: auto        # cpu | cuda | auto
  embeddings_cache: true         # Cache embeddings
  embeddings_batch_size: 32      # Batch size
  similarity_metric: cosine      # cosine | euclidean | manhattan
  similarity_threshold: 0.7      # Similarity threshold

  # Cache settings
  cache_embeddings_ttl_days: 30  # Embeddings cache TTL
  cache_tfidf_ttl_days: 7       # TF-IDF cache TTL (if using TF-IDF)
  cache_keywords_ttl_days: 7     # Keywords cache TTL

  # Performance
  multiprocessing_enabled: true   # Use multiprocessing
  multiprocessing_workers: null   # null = auto-detect
  multiprocessing_chunk_size: 100 # Chunk size

# ============= LLM Settings (Optional) =============
llm:
  enabled: false                # Enable LLM features
  provider: openai              # openai | anthropic | ollama
  fallback_providers:           # Fallback providers
    - anthropic
    - openrouter

  # API keys (use environment variables)
  api_keys:
    openai: ${OPENAI_API_KEY}
    anthropic: ${ANTHROPIC_API_KEY}
    openrouter: ${OPENROUTER_API_KEY}

  # API endpoints
  api_base_urls:
    openai: https://api.openai.com/v1
    anthropic: https://api.anthropic.com/v1
    openrouter: https://openrouter.ai/api/v1
    ollama: http://localhost:11434

  # Model selection
  models:
    default: gpt-4o-mini
    summarization: gpt-3.5-turbo
    analysis: gpt-4o
    embeddings: text-embedding-3-small
    code_generation: gpt-4o

  # Rate limits and costs
  max_cost_per_run: 0.1         # Max $ per run
  max_cost_per_day: 10.0        # Max $ per day
  max_tokens_per_request: 4000   # Max tokens per request
  max_context_length: 100000     # Max context length

  # Generation settings
  temperature: 0.3              # Creativity (0.0-1.0)
  top_p: 0.95                  # Nucleus sampling
  frequency_penalty: 0.0        # Frequency penalty
  presence_penalty: 0.0         # Presence penalty

  # Network settings
  requests_per_minute: 60       # Rate limit
  retry_on_error: true         # Retry failed requests
  max_retries: 3              # Max retry attempts
  retry_delay: 1.0            # Initial retry delay
  retry_backoff: 2.0          # Backoff multiplier
  timeout: 30                 # Request timeout (seconds)
  stream: false               # Stream responses

  # Logging and caching
  cache_responses: true        # Cache LLM responses
  cache_ttl_hours: 24         # Cache TTL (hours)
  log_requests: false         # Log requests
  log_responses: false        # Log responses

# ============= Custom Settings =============
custom: {}  # User-defined custom settings
```

### Key Configuration Notes

**Ranking:**
- `threshold`: Lower values (0.05-0.10) include more files, higher (0.20-0.30) for stricter matching
- `algorithm`:
  - `fast`: Quick keyword matching (~10ms/file)
  - `balanced`: Structural analysis + BM25 scoring (default)
  - `thorough`: Full analysis with relationships
  - `ml`: Machine learning with embeddings (requires extras)
- `custom_weights`: Fine-tune ranking factors (values 0.0-1.0)

**Scanner:**
- `respect_gitignore`: Always honors .gitignore patterns
- `exclude_tests_by_default`: Tests excluded unless `--include-tests` used
- `additional_ignore_patterns`: Added to built-in patterns

**Tenet System:**
- `auto_instill`: Automatically applies relevant tenets to context
- `injection_frequency`:
  - `always`: Every distill
  - `periodic`: Every N distills
  - `adaptive`: Based on complexity
  - `manual`: Only when explicitly called
- `system_instruction`: Global instruction added to all contexts

**Output:**
- `copy_on_distill`: Auto-copy result to clipboard
- `default_format`: Default output format (markdown recommended for LLMs)

**Performance:**
- `workers`: More workers = faster but more CPU/memory
- `cache.enabled`: Significantly speeds up repeated operations
- `ranking.batch_size`: Larger batches = more memory but faster

## Environment Variable Overrides

Any configuration option can be overridden via environment variables.

**Format:**
- Nested keys: `TENETS_<SECTION>_<KEY>=value`
- Top-level keys: `TENETS_<KEY>=value`
- Lists: Comma-separated values
- Booleans: `true` or `false` (case-insensitive)

**Common Examples:**
```bash
# Core settings
export TENETS_MAX_TOKENS=150000
export TENETS_DEBUG=true
export TENETS_QUIET=false

# Ranking configuration
export TENETS_RANKING_ALGORITHM=thorough
export TENETS_RANKING_THRESHOLD=0.05
export TENETS_RANKING_TEXT_SIMILARITY_ALGORITHM=tfidf  # Use BM25 instead (default)
export TENETS_RANKING_USE_EMBEDDINGS=true
export TENETS_RANKING_WORKERS=4

# Scanner settings
export TENETS_SCANNER_MAX_FILE_SIZE=10000000
export TENETS_SCANNER_RESPECT_GITIGNORE=true
export TENETS_SCANNER_EXCLUDE_TESTS_BY_DEFAULT=false

# Output settings
export TENETS_OUTPUT_DEFAULT_FORMAT=xml
export TENETS_OUTPUT_COPY_ON_DISTILL=true
export TENETS_OUTPUT_SHOW_TOKEN_USAGE=false

# Cache settings
export TENETS_CACHE_ENABLED=false
export TENETS_CACHE_DIRECTORY=/tmp/tenets-cache
export TENETS_CACHE_TTL_DAYS=14

# Git settings
export TENETS_GIT_ENABLED=false
export TENETS_GIT_HISTORY_LIMIT=50

# Tenet system
export TENETS_TENET_AUTO_INSTILL=false
export TENETS_TENET_MAX_PER_CONTEXT=10
export TENETS_TENET_INJECTION_FREQUENCY=periodic
export TENETS_TENET_INJECTION_INTERVAL=5

# System instruction
export TENETS_TENET_SYSTEM_INSTRUCTION="You are a senior engineer. Focus on security and performance."
export TENETS_TENET_SYSTEM_INSTRUCTION_ENABLED=true
```

**Usage Patterns:**
```bash
# One-time override
TENETS_RANKING_ALGORITHM=fast tenets distill "fix bug"

# Session-wide settings
export TENETS_RANKING_THRESHOLD=0.05
export TENETS_OUTPUT_COPY_ON_DISTILL=true
tenets distill "implement feature"  # Uses exported settings

# Verify configuration
tenets config show --key ranking
tenets config show --format json | jq '.ranking'
```

## CLI Flags and Programmatic Control

### CLI Flags

Command-line flags override configuration for that specific run:

```bash
# Core overrides
tenets distill "query" --max-tokens 50000
tenets distill "query" --format xml
tenets distill "query" --copy

# Ranking mode
tenets distill "query" --mode fast      # Quick analysis
tenets distill "query" --mode thorough  # Deep analysis
tenets distill "query" --mode ml        # With embeddings

# File filtering
tenets distill "query" --include "*.py" --exclude "test_*.py"
tenets distill "query" --include-tests  # Include test files

# Git control
tenets distill "query" --no-git  # Disable git signals

# Session management
tenets distill "query" --session feature-x

# Content optimization
tenets distill "query" --condense        # Aggressive compression
tenets distill "query" --remove-comments # Strip comments
tenets distill "query" --full            # No summarization
```

### Programmatic Configuration

**Basic usage with custom config:**
```python
from tenets import Tenets
from tenets.config import TenetsConfig

# Create custom configuration
config = TenetsConfig(
    max_tokens=150000,
    ranking={
        "algorithm": "thorough",
        "threshold": 0.05,
        "text_similarity_algorithm": "bm25",  # Recommended
        "use_embeddings": True,
        "workers": 4,
        "custom_weights": {
            "keyword_match": 0.30,
            "path_relevance": 0.25,
            "git_activity": 0.20,
        }
    },
    scanner={
        "respect_gitignore": True,
        "max_file_size": 10_000_000,
        "exclude_tests_by_default": False,
    },
    output={
        "default_format": "xml",
        "copy_on_distill": True,
    },
    tenet={
        "auto_instill": True,
        "max_per_context": 10,
        "system_instruction": "Focus on security and performance",
        "system_instruction_enabled": True,
    }
)

# Initialize with custom config
tenets = Tenets(config=config)

# Use it
result = tenets.distill(
    "implement caching layer",
    max_tokens=80000,  # Override config for this call
    mode="balanced",    # Override algorithm
)
```

**Load and modify existing config:**
```python
from tenets import Tenets
from tenets.config import TenetsConfig

# Load from file
config = TenetsConfig.from_file(".tenets.yml")

# Modify specific settings
config.ranking.algorithm = "fast"
config.ranking.threshold = 0.08
config.output.copy_on_distill = True

# Use modified config
tenets = Tenets(config=config)
```

**Runtime overrides:**
```python
# Config precedence: method args > instance config > file config
result = tenets.distill(
    prompt="add authentication",
    mode="thorough",        # Overrides config.ranking.algorithm
    max_tokens=100000,      # Overrides config.max_tokens
    format="json",          # Overrides config.output.default_format
    session_name="auth",    # Session-specific
    include_patterns=["*.py", "*.js"],
    exclude_patterns=["*.test.js"],
)
```

## Configuration Recipes

### For Different Use Cases

**Large Monorepo (millions of files):**
```yaml
max_tokens: 150000
scanner:
  max_files: 50000
  workers: 8
  parallel_mode: process
  exclude_tests_by_default: true
ranking:
  algorithm: fast
  threshold: 0.15
  workers: 4
  batch_size: 500
cache:
  enabled: true
  memory_cache_size: 5000
```

**Small Project (high precision):**
```yaml
max_tokens: 80000
ranking:
  algorithm: thorough
  threshold: 0.08
  text_similarity_algorithm: bm25  # Recommended over tfidf
  use_embeddings: true
  custom_weights:
    keyword_match: 0.35
    import_graph: 0.25
```

**Documentation-Heavy Project:**
```yaml
summarizer:
  docstring_weight: 0.8
  include_all_signatures: true
  preserve_code_structure: false
ranking:
  custom_weights:
    keyword_match: 0.20
    path_relevance: 0.30  # Prioritize doc paths
```

**Security-Focused Analysis:**
```yaml
tenet:
  system_instruction: |
    Focus on security implications.
    Flag any potential vulnerabilities.
    Suggest secure alternatives.
  system_instruction_enabled: true
  auto_instill: true
scanner:
  additional_ignore_patterns: []  # Don't skip anything
  exclude_tests_by_default: false
```

### Performance Tuning

**Maximum Speed (sacrifices precision):**
```yaml
ranking:
  algorithm: fast
  threshold: 0.05
  text_similarity_algorithm: bm25  # Default
  use_embeddings: false
  workers: 8
scanner:
  workers: 8
  timeout: 2.0
cache:
  enabled: true
  compression: false
```

**Maximum Precision (slower):**
```yaml
ranking:
  algorithm: thorough
  threshold: 0.20
  text_similarity_algorithm: bm25  # Recommended over tfidf
  use_embeddings: true
  use_git: true
  workers: 2
summarizer:
  quality_threshold: high
  enable_ml_strategies: true
```

**Memory-Constrained Environment:**
```yaml
scanner:
  max_files: 1000
  workers: 1
ranking:
  workers: 1
  batch_size: 50
cache:
  memory_cache_size: 100
  max_size_mb: 100
nlp:
  embeddings_batch_size: 8
  multiprocessing_enabled: false
```

### Common Workflows

**Bug Investigation:**
```yaml
ranking:
  algorithm: balanced
  threshold: 0.10
  custom_weights:
    git_activity: 0.30  # Recent changes matter
    complexity: 0.20    # Complex code = more bugs
git:
  include_history: true
  history_limit: 200
  include_blame: true
```

**New Feature Development:**
```yaml
ranking:
  algorithm: balanced
  threshold: 0.08
  custom_weights:
    import_graph: 0.30  # Dependencies matter
    path_relevance: 0.25 # Related modules
output:
  copy_on_distill: true
  show_token_usage: true
```

**Code Review Preparation:**
```yaml
summarizer:
  target_ratio: 0.5  # More detail
  preserve_code_structure: true
  include_all_signatures: true
output:
  syntax_highlighting: true
  line_numbers: true
  include_metadata: true
```

## Troubleshooting

### Common Issues and Solutions

**No files included in context:**
- Lower `ranking.threshold` (try 0.05)
- Use `--mode fast` for broader inclusion
- Increase `max_tokens` limit
- Check if files match `--include` patterns
- Verify files aren't in `.gitignore`
- Use `--include-tests` if analyzing test files

**Configuration not taking effect:**
```bash
# Check which config file is loaded
tenets config show | head -20

# Verify specific setting
tenets config show --key ranking.threshold

# Check config file location
ls -la .tenets.yml

# Test with explicit config
tenets --config ./my-config.yml distill "query"
```

**Environment variables not working:**
```bash
# Verify export (not just set)
export TENETS_RANKING_THRESHOLD=0.05  # Correct
TENETS_RANKING_THRESHOLD=0.05         # Wrong (not exported)

# Check if variable is set
echo $TENETS_RANKING_THRESHOLD

# Debug with explicit env
TENETS_DEBUG=true tenets config show
```

**Performance issues:**
- Reduce `scanner.max_files` and `scanner.max_file_size`
- Enable caching: `cache.enabled: true`
- Use `ranking.algorithm: fast`
- Reduce `ranking.workers` if CPU-constrained
- Exclude unnecessary paths with `additional_ignore_patterns`

**Token limit exceeded:**
- Increase `max_tokens` or use `--max-tokens`
- Enable `--condense` flag
- Use `--remove-comments`
- Increase `ranking.threshold` for stricter filtering
- Exclude test files: `scanner.exclude_tests_by_default: true`

**Cache issues:**
```bash
# Clear cache
rm -rf ~/.tenets/cache

# Disable cache temporarily
TENETS_CACHE_ENABLED=false tenets distill "query"

# Use custom cache location
export TENETS_CACHE_DIRECTORY=/tmp/tenets-cache
```

### Validation Commands

```bash
# Validate configuration syntax
tenets config validate

# Show effective configuration
tenets config show --format json | jq

# Test configuration with dry run
tenets distill "test query" --dry-run

# Check what files would be scanned
tenets examine . --dry-run

# Debug ranking process
TENETS_DEBUG=true tenets distill "query" 2>debug.log
```

## Advanced Topics

### Custom Ranking Strategies

Create a custom ranking strategy by combining weights:

```yaml
ranking:
  algorithm: custom
  custom_weights:
    keyword_match: 0.40    # Emphasize keyword relevance
    path_relevance: 0.15   # De-emphasize path matching
    import_graph: 0.15     # Moderate dependency weight
    git_activity: 0.10     # Low git signal weight
    file_type: 0.10        # File type matching
    complexity: 0.10       # Code complexity
```

### Multi-Environment Setup

Create environment-specific configs:

```bash
# Development
cp .tenets.yml .tenets.dev.yml
# Edit for dev settings

# Production analysis
cp .tenets.yml .tenets.prod.yml
# Edit for production settings

# Use specific config
tenets --config .tenets.dev.yml distill "query"
```

### Integration with CI/CD

```yaml
# .tenets.ci.yml - Optimized for CI
max_tokens: 50000
quiet: true
scanner:
  max_files: 5000
  workers: 2
ranking:
  algorithm: fast
  threshold: 0.10
cache:
  enabled: false  # Fresh analysis each run
output:
  default_format: json  # Machine-readable
```

## See Also

- [CLI Reference](cli.md) - Complete command documentation
- [API Reference](api/index.md) - Python API documentation
- [Architecture](architecture/index.md) - System design details
