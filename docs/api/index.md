# API Reference

Welcome to the Tenets API documentation. This section contains comprehensive documentation for all modules, classes, and functions in the Tenets package.

## Overview

The API documentation is automatically generated from the source code docstrings using mkdocstrings. Each module page includes:

- **Classes** - With all methods, attributes, and detailed descriptions
- **Functions** - Including parameters, return types, and usage examples
- **Type Hints** - Full type annotations for better IDE support
- **Examples** - Code snippets demonstrating usage where available

## Main Package Structure

### Core Package
**`tenets`** - The main package containing core exports and initialization

### Core Modules
**`tenets.core`** - Core functionality powering tenets
- Analysis engines for code understanding
- Ranking algorithms for relevance scoring  
- Context aggregation and management
- Session persistence and state management

### Command-Line Interface
**`tenets.cli`** - CLI application and commands
- Main application entry point
- Command implementations
- Terminal output formatting

### Data Models
**`tenets.models`** - Data structures and schemas
- File and context models
- Session and configuration models
- Analysis result structures

### Utilities
**`tenets.utils`** - Helper functions and utilities
- File operations and scanning
- Git integration helpers
- Token counting and management

## Navigation

The full module documentation is auto-generated during the build process. Use the sidebar navigation to browse through all available modules and their documentation.

!!! tip "Using the API"
    To use Tenets programmatically in your Python code:
    
    ```python
    from tenets import Tenets
    
    # Initialize
    tenets = Tenets()
    
    # Build context
    result = tenets.distill("implement user authentication")
    ```

!!! note "Auto-Generated Content"
    This documentation is generated directly from the source code. Any updates to docstrings in the codebase will be reflected here after rebuilding the documentation.