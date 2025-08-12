# Quick Start

Get productive with Tenets in under 60 seconds.

## 1. Install

```bash
pip install tenets
```

## 2. Generate Context (CLI)

```bash
tenets make-context "add optimistic locking to order updates"
```

Copy straight to your clipboard:

```bash
tenets make-context "refactor payment flow" --copy
```

Or enable auto-copy in `tenets.toml`:

```toml
[output]
copy_on_distill = true
```

## 3. Refine

Pin or force-include critical files:

```bash
tenets make-context "investigate cache stampede" --pin cache/*.py --pin config/settings.py
```

Exclude noise:

```bash
tenets make-context "debug webhook" --exclude "**/migrations/**" --exclude "**/tests/**"
```

## 4. Python API

```python
from tenets import Tenets

ctx = Tenets().make_context(
    prompt="implement bulk import",
    path="./",
    max_tokens=80_000,
)
print(ctx.token_count, "tokens")
ctx.copy()  # copies to clipboard (same behavior as --copy)
```

## 5. Sessions (Iterate)

```python
session = Tenets().create_session("checkout-fixes")
first = session.make_context("trace 500 errors in checkout")
second = session.make_context("add instrumentation around payment retries")
```

## 6. Visualization & Insight

```bash
# Complexity & hotspots
tenets analyze --complexity --hotspots

# Dependency graph (ASCII)
tenets viz deps --format ascii
```

## 7. Next

* See full CLI options: [CLI Reference](CLI.md)
* Tune ranking & tokens: [Configuration](CONFIG.md)
* Dive deeper: [Architecture](DEEP-DIVE.md)
