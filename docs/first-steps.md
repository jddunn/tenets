# First Steps

## Basic Usage

After installing Tenets, you can start analyzing your codebase immediately.

### Examine Your Codebase

```bash
# Analyze the current directory
tenets examine

# Analyze a specific path
tenets examine path/to/project
```

### Chronicle Git History

```bash
# Analyze recent commits
tenets chronicle --days 30

# Analyze specific author's commits
tenets chronicle --author "Your Name"
```

### Distill Insights

```bash
# Extract key insights from your codebase
tenets distill
```

### Visualize Relationships

```bash
# Generate an interactive visualization
tenets viz --output graph.html
```

## Configuration

See [Configuration Guide](CONFIG.md) for customizing Tenets behavior.

## Next Steps

- Read [Best Practices](best-practices.md) for optimal usage
- Check out [Examples](examples.md) for real-world scenarios
- Explore the [CLI Reference](CLI.md) for all available commands