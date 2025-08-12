# Features

Tenets distills the most relevant slices of your codebase into high-signal context for LLMs – locally, fast, and privately.

## Core Pipeline

1. Scan: Respect .gitignore, configurable include/exclude globs
2. Analyze: Multi-language structural parsing & dependency extraction
3. Rank: Multi-factor (term frequency, structural weight, recency, path signals, optional TF‑IDF)
4. Aggregate: Token-aware packing with de-duplication and formatting
5. Optimize: Prunes low-value regions to fit budget

## Ranking Modes

| Mode | Purpose | Tradeoff |
|------|---------|----------|
| fast | Minimal analysis for speed | Lowest precision |
| balanced | Deeper structural weighting | Good default |
| thorough | Full pass w/ TF‑IDF + relationships | Highest precision |

Switch quickly:

```bash
TENETS_RANKING_ALGORITHM=thorough tenets make-context "trace memory leak"
```

## Clipboard & Output

```bash
# One-off copy
tenets make-context "refactor config loading" --copy

# Always copy
# tenets.toml
[output]
copy_on_distill = true
```

Python:

```python
ctx = Tenets().make_context(prompt="add rate limiting")
ctx.copy()  # same as CLI --copy
```

## Sessions & Persistence

Maintain evolving intent across multiple prompts:

```python
session = Tenets().create_session("ml-pipeline")
ctx1 = session.make_context("optimize training loop")
ctx2 = session.make_context("add early stopping")
```

Add durable guidance:

```bash
tenets tenet add "Prefer streaming responses for large payloads"
```

## Insights

```bash
# Complexity & hotspots
tenets analyze --complexity --hotspots
# Recent change heatmap
tenets track-changes --since 30d
```

## Dependency Graph

```bash
tenets viz deps --format ascii
```

## Supported Languages

See: [Supported Languages](supported-languages.md)

## Extensibility

* Register custom rankers
* Plug new language analyzers
* Environment-based overrides (`TENETS_*`)
* Embeddings (config keys present, future expansion)

## Safety & Privacy

* 100% local execution
* No outbound network calls by default
* Explicit paths & glob controls

## Next

* Explore configuration: [CONFIG.md](CONFIG.md)
* API surface: [api/](api/index.md)
* Architecture deep dive: [DEEP-DIVE.md](DEEP-DIVE.md)
