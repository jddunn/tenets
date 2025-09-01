# Quickstart

## Real-world flow: System instruction + Tenets + Sessions

```bash
# Create a working session
tenets session create auth-refresh

# Add guiding principles (tenets)
tenets tenet add "Prefer small, safe diffs" --priority high --category style
tenets tenet add "Always validate user input" --priority critical --category security

# Apply tenets for this session
tenets instill --session auth-refresh

# Set a global system instruction
tenets system-instruction set "You are a senior engineer. Add tests and document trade-offs." --enable

# Build context with transformations for token efficiency
tenets distill "add OAuth2 refresh tokens" --session auth-refresh --remove-comments --condense

# Pin files as you learn what matters
tenets instill --session auth-refresh --add-file src/auth/service.py --add-folder src/auth/routes
tenets instill --session auth-refresh --list-pinned
```

See also: CLI > System Instruction Commands, Tenet Commands, and Instill.
# Quick Start

Get productive with Tenets in under 60 seconds.

## 1. Install

```bash
pip install tenets
```

## 2. Generate Context (CLI)

```bash
tenets distill "add optimistic locking to order updates"
```

Copy straight to your clipboard:

```bash
tenets distill "refactor payment flow" --copy
```

Or enable auto-copy in `tenets.toml`:

```toml
[output]
copy_on_distill = true
```

## 3. Refine

Pin or force-include critical files:

```bash
# Build context for investigation
tenets distill "investigate cache stampede"

# Pin files are managed through instill command for sessions
tenets instill --add-file cache/*.py --add-file config/settings.py
```

Exclude noise:

```bash
tenets distill "debug webhook" --exclude "**/migrations/**,**/tests/**"
```

## 4. Python API

```python
from tenets import Tenets

tenets = Tenets()
result = tenets.distill(
    prompt="implement bulk import",
    max_tokens=80_000,
)
print(result.token_count, "tokens")
# Copy is done via CLI flag --copy or config setting
```

## 5. Sessions (Iterate)

```python
tenets = Tenets()
# Sessions are managed through distill parameters
first = tenets.distill("trace 500 errors in checkout", session_name="checkout-fixes")
second = tenets.distill("add instrumentation around payment retries", session_name="checkout-fixes")
```

## 6. Visualization & Insight

```bash
# Complexity & hotspots
tenets examine . --show-details --hotspots

# Dependency graph (Interactive HTML)
tenets viz deps --format html --output deps.html
```

## 7. Next

* See full CLI options: [CLI Reference](CLI.md)
* Tune ranking & tokens: [Configuration](CONFIG.md)
* Dive deeper: [Architecture](ARCHITECTURE.md)
