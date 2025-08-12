# Configuration Guide

This guide explains how Tenets configuration works, where to put it, and how to override it via environment variables or code.

- Precedence (lowest → highest):
  1) Project file (.tenets.yml at repo root) or user file (~/.config/tenets/config.yml or ~/.tenets.yml)
  2) Environment variables (TENETS_*)
  3) Programmatic overrides (Tenets(config=...) or TenetsConfig(...))

Tenets loads configuration automatically on startup. You can inspect the effective config with:

- tenets config show
- tenets config show --key ranking
- tenets config show --format json

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

## YAML schema (supported keys)

Top-level keys and nested sections mirror the TenetsConfig dataclasses. Common options are below.

Minimal example (updated for consolidated ranking system):

```yaml
max_tokens: 100000

ranking:
  algorithm: balanced      # fast | balanced | thorough | ml | custom
  threshold: 0.10           # 0.0–1.0 (lower includes more files)
  use_tfidf: true           # Enable TF‑IDF weighting for keyword relevance
  use_stopwords: false      # Filter common tokens during TF‑IDF
  use_embeddings: false     # Semantic similarity (requires ML extras / --mode ml)
  embedding_model: all-MiniLM-L6-v2  # Model used when embeddings enabled
  workers: 2                # Parallel workers for ranking (>=2 recommended)
  parallel_mode: auto       # thread | process | auto (currently thread-based)
  custom_weights:           # Optional override weights (omit keys to use defaults)
    keyword_match: 0.25
    path_relevance: 0.20
    import_graph: 0.20
    git_activity: 0.15
    file_type: 0.10
    complexity: 0.10

scanner:
  respect_gitignore: true
  follow_symlinks: false
  max_file_size: 5000000   # bytes
  additional_ignore_patterns:
    - "*.generated.*"
    - vendor/

git:
  enabled: true

output:
  default_format: markdown  # markdown | xml | json
  copy_on_distill: false    # copy distilled context to clipboard automatically

cache:
  enabled: true
  ttl_days: 7
  max_size_mb: 500
  # directory: /custom/cache/path   # optional; default is ~/.tenets/cache

tenet:
  auto_instill: true
  max_per_context: 5
  reinforcement: true
```

Notes:
- ranking.threshold controls inclusion. For broader context, set 0.05–0.10.
- ranking.use_tfidf + ranking.use_stopwords affect keyword quality; disable TF‑IDF only for very small repos.
- ranking.use_embeddings adds a semantic factor (also available implicitly with --mode ml); requires installing ML extras.
- ranking.custom_weights lets you tune factor influence; unspecified factors fall back to internal strategy defaults.
- output.default_format affects default formatting; CLI -f/--format overrides it.
- output.copy_on_distill copies result automatically (same as --copy flag) when true.
- scanner.additional_ignore_patterns augments internal defaults; entries are simple fnmatch patterns.

## Environment variable overrides

All config keys can be overridden with TENETS_ variables. Structure:
- TENETS_<section>_<key>=value (for nested)
- TENETS_<key>=value (for top-level)

Common examples:
- TENETS_RANKING_THRESHOLD=0.05
- TENETS_RANKING_ALGORITHM=fast
- TENETS_RANKING_USE_TFIDF=true
- TENETS_RANKING_USE_STOPWORDS=true
- TENETS_RANKING_USE_EMBEDDINGS=true
- TENETS_RANKING_EMBEDDING_MODEL=all-MiniLM-L6-v2
- TENETS_RANKING_WORKERS=4
- TENETS_RANKING_PARALLEL_MODE=thread
- TENETS_SCANNER_RESPECT_GITIGNORE=true
- TENETS_SCANNER_FOLLOW_SYMLINKS=false
- TENETS_SCANNER_MAX_FILE_SIZE=8000000
- TENETS_CACHE_DIRECTORY=/path/to/cache
- TENETS_OUTPUT_DEFAULT_FORMAT=json
- TENETS_GIT_ENABLED=false
- TENETS_MAX_TOKENS=150000

Usage (Git Bash):
-- Single run: TENETS_RANKING_THRESHOLD=0.05 tenets distill "implement OAuth2" .
- Session export:
  - export TENETS_RANKING_THRESHOLD=0.05
  - export TENETS_RANKING_ALGORITHM=fast

Tip: Verify the override was picked up:
- tenets config show --key ranking

## CLI and programmatic control

CLI flags that affect behavior (override config for that run):
- --mode fast|balanced|thorough|ml  (ranking.algorithm)
- --max-tokens N                 (max_tokens)
- --include, --exclude           (scanner-like filtering at runtime)
- --no-git                       (git.enabled for the run)

Programmatic override (highest precedence) (new nested ranking config):

```python
from tenets import Tenets
from tenets.config import TenetsConfig

cfg = TenetsConfig(
  max_tokens=150000,
  ranking={
    "algorithm": "fast",
    "threshold": 0.05,
    "use_tfidf": True,
    "use_stopwords": False,
    "use_embeddings": False,
    "workers": 4,
  },
  scanner={"respect_gitignore": True},
)

ten = Tenets(config=cfg)
```

## Quick recipes

- Broader context (include more files):
```yaml
ranking:
  algorithm: fast
  threshold: 0.05
  use_tfidf: true
```

- Stricter context (only top matches):
```yaml
ranking:
  algorithm: balanced
  threshold: 0.25
  use_embeddings: false
```

- Disable git influence:
```yaml
git:
  enabled: false
```

- Move cache:
```yaml
cache:
  directory: /tmp/tenets-cache
```

## Troubleshooting

- No files included: lower ranking.threshold (e.g., 0.05), add --include "*.py", increase --max-tokens, enable fast mode, or temporarily raise workers.
- Not seeing config changes: ensure you’re editing the right .tenets.yml (repo root), then run tenets config show to confirm.
- Environment not applying: export variables in the same shell session where you run tenets.
