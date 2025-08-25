# Tenets Viz Deps Command Cheat Sheet

## Installation
```bash
pip install tenets[viz]  # Install visualization dependencies
```

## Basic Commands

### Simple Usage
```bash
tenets viz deps                     # Auto-detect project, show ASCII tree
tenets viz deps .                   # Analyze current directory
tenets viz deps src/                # Analyze specific directory
```

### Output Formats
```bash
tenets viz deps --format ascii      # Terminal tree (default)
tenets viz deps --format svg --output arch.svg     # Scalable vector graphics
tenets viz deps --format png --output arch.png     # PNG image
tenets viz deps --format html --output deps.html   # Interactive HTML
tenets viz deps --format dot --output graph.dot    # Graphviz DOT
tenets viz deps --format json --output data.json   # Raw JSON data
```

## Aggregation Levels

```bash
tenets viz deps --level file        # Individual file dependencies (detailed)
tenets viz deps --level module      # Module-level aggregation (recommended)
tenets viz deps --level package     # Package-level view (high-level)
```

## Clustering Options

```bash
tenets viz deps --cluster-by directory   # Group by directory structure
tenets viz deps --cluster-by module      # Group by module
tenets viz deps --cluster-by package     # Group by package
```

## Layout Algorithms

```bash
tenets viz deps --layout hierarchical   # Tree-like layout (default)
tenets viz deps --layout circular       # Circular/radial layout
tenets viz deps --layout shell          # Concentric circles
tenets viz deps --layout kamada         # Force-directed layout
```

## Filtering

```bash
# Include specific patterns
tenets viz deps --include "*.py"                    # Only Python files
tenets viz deps --include "*.js,*.jsx"              # JavaScript files
tenets viz deps --include "src/**/*.py"             # Python in src/

# Exclude patterns
tenets viz deps --exclude "*test*"                  # No test files
tenets viz deps --exclude "*.min.js,node_modules"   # Skip minified and deps

# Combined
tenets viz deps --include "*.py" --exclude "*test*"
```

## Node Limiting

```bash
tenets viz deps --max-nodes 50      # Show only top 50 most connected nodes
tenets viz deps --max-nodes 100     # Useful for large projects
```

## Real-World Examples

### For Documentation
```bash
# Clean architecture diagram for docs
tenets viz deps . --level package --format svg --output docs/architecture.svg

# Module overview with clustering
tenets viz deps . --level module --cluster-by directory --format png --output modules.png
```

### For Code Review
```bash
# Interactive exploration
tenets viz deps . --level module --format html --output review.html

# Focused on specific subsystem
tenets viz deps src/api --include "*.py" --format svg --output api_deps.svg
```

### For Refactoring
```bash
# Find circular dependencies
tenets viz deps . --layout circular --format html --output circular_deps.html

# Identify tightly coupled modules
tenets viz deps . --level module --layout circular --max-nodes 50 --output coupling.svg
```

### For Large Projects
```bash
# Top-level overview
tenets viz deps . --level package --max-nodes 20 --format svg --output overview.svg

# Most connected files
tenets viz deps . --max-nodes 100 --format html --output top100.html

# Specific subsystem deep dive
tenets viz deps backend/ --level module --cluster-by module --format html -o backend.html
```

## Project Type Auto-Detection

The command automatically detects:
- **Python**: Packages, Django, Flask, FastAPI
- **JavaScript/TypeScript**: Node.js, React, Vue, Angular
- **Java**: Maven, Gradle, Spring
- **Go**: Go modules
- **Rust**: Cargo projects
- **Ruby**: Rails, Gems
- **PHP**: Laravel, Composer
- And more...

## Tips

1. **Start Simple**: Use `tenets viz deps` first to see what's detected
2. **Use Levels**: Start with `--level package` for overview, drill down to `module` or `file`
3. **Interactive HTML**: Best for exploration, use `--format html`
4. **Filter Noise**: Use `--exclude "*test*,*mock*"` to focus on core code
5. **Save Time**: Use `--max-nodes` for large codebases
6. **Documentation**: SVG format scales well for docs
7. **Clustering**: Helps organize complex graphs visually

## Troubleshooting

```bash
# Check if dependencies are installed
python -c "import networkx, matplotlib, graphviz, plotly" 2>/dev/null && echo "All deps OK" || echo "Run: pip install tenets[viz]"

# Debug mode
TENETS_LOG_LEVEL=DEBUG tenets viz deps . 2>&1 | grep -E "(Detected|Found|Analyzing)"

# If graph is too large
tenets viz deps . --max-nodes 50 --level module  # Reduce nodes and aggregate
```

## Output Examples

### ASCII Tree (default)
```
Dependency Graph:
==================================================

main.py
  └─> utils.py
  └─> config.py
  └─> models.py

utils.py
  └─> config.py

models.py
  └─> utils.py
```

### What You Get
- **Project Info**: Auto-detected type, languages, frameworks
- **Entry Points**: Identified main files (main.py, index.js, etc.)
- **Dependency Graph**: Visual representation of code relationships
- **Multiple Views**: File, module, or package level perspectives