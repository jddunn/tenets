---
title: System Overview
---

# System Overview

<div class="loading-indicator">Loading optimized content...</div>

## What is Tenets?

Tenets is a sophisticated, local-first code intelligence platform that revolutionizes how developers interact with their codebases when working with AI assistants.

## Core Philosophy & Design Principles

### 1. Local-First Processing
All analysis, ranking, and context generation happens on the developer's machine. No code ever leaves the local environment. External API calls are only made for optional LLM-based summarization, and even then, only with explicit user consent.

### 2. Progressive Enhancement
The system provides value immediately with just Python installed, and scales up with optional dependencies. Core functionality works without any ML libraries, git integration works without any configuration, and advanced features gracefully degrade when dependencies are missing.

### 3. Intelligent Caching
Every expensive operation is cached at multiple levels - memory caches for hot data, SQLite for structured data, disk caches for analysis results, and specialized caches for embeddings. Cache invalidation is intelligent, using file modification times, git commits, and content hashes.

### 4. Configurable Intelligence
Every aspect of the ranking and analysis can be configured. Users can adjust factor weights, enable/disable features, add custom ranking functions, and tune performance parameters. The system adapts to different codebases and use cases.

### 5. Streaming Architecture
The system uses streaming and incremental processing wherever possible. Files are analyzed as they're discovered, rankings are computed in parallel, and results stream to the user as they become available.

## Complete System Architecture

```mermaid
graph TB
    subgraph "User Layer"
        CLI[CLI Interface]
        API[Python API]
        IDE[IDE Extensions]
    end

    subgraph "Command Processing"
        DISPATCHER[Command Dispatcher]
        DISTILL[Distill Command]
        RANK[Rank Command]
        EXAMINE[Examine Command]
        SESSION[Session Management]
    end

    subgraph "Intelligence Pipeline"
        PARSER[Prompt Parser]
        INTENT[Intent Detection]
        KEYWORDS[Keyword Extraction]
        SCANNER[File Scanner]
        ANALYZER[Code Analyzer]
        RANKER[Ranking Engine]
        BUILDER[Context Builder]
    end

    subgraph "Storage Layer"
        MEMORY[Memory Cache<br/>Hot Data]
        SQLITE[SQLite DB<br/>Sessions]
        DISK[Disk Cache<br/>Analysis]
        EMBED[Embedding Cache<br/>ML Vectors]
    end

    CLI --> DISPATCHER
    API --> DISPATCHER
    IDE --> DISPATCHER

    DISPATCHER --> DISTILL
    DISPATCHER --> RANK
    DISPATCHER --> EXAMINE
    DISPATCHER --> SESSION

    DISTILL --> PARSER
    PARSER --> INTENT
    INTENT --> KEYWORDS
    KEYWORDS --> SCANNER
    SCANNER --> ANALYZER
    ANALYZER --> RANKER
    RANKER --> BUILDER

    ANALYZER --> MEMORY
    RANKER --> SQLITE
    BUILDER --> DISK
    INTENT --> EMBED

    style CLI fill:#e1f5fe
    style BUILDER fill:#e8f5e8
    style ANALYZER fill:#fff3e0
    style RANKER fill:#fce4ec
```

### Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Input"
        PROMPT[User Prompt]
        FILES[Codebase Files]
        CONFIG[Configuration]
    end

    subgraph "Processing"
        DISCOVER[Discovery<br/>Find Files]
        ANALYZE[Analysis<br/>Extract Structure]
        RANK[Ranking<br/>Score Relevance]
        OPTIMIZE[Optimization<br/>Token Budget]
    end

    subgraph "Output"
        CONTEXT[Context]
        REPORT[Reports]
        VIZ[Visualizations]
    end

    PROMPT --> DISCOVER
    FILES --> DISCOVER
    CONFIG --> DISCOVER

    DISCOVER --> ANALYZE
    ANALYZE --> RANK
    RANK --> OPTIMIZE

    OPTIMIZE --> CONTEXT
    OPTIMIZE --> REPORT
    OPTIMIZE --> VIZ

    style PROMPT fill:#e3f2fd
    style CONTEXT fill:#e8f5e9
```

## System Components Overview

The Tenets system consists of several key architectural layers:

### 🔍 **Discovery Layer**
- File discovery and scanning
- Git repository analysis
- Language detection and classification

### 🧠 **Analysis Layer**
- Multi-language static analysis
- Code complexity measurement
- Pattern recognition and extraction

### 📊 **Intelligence Layer**
- Relevance ranking algorithms
- Machine learning pipelines
- Natural language processing

### 💾 **Storage Layer**
- Session management
- Caching architecture
- Data persistence

### 🔌 **Interface Layer**
- CLI commands
- API endpoints
- Configuration management

## Technology Stack

### Core Technologies

```mermaid
graph LR
    subgraph "Languages & Frameworks"
        PYTHON[Python 3.9+]
        TYPER[Typer CLI]
        PYDANTIC[Pydantic]
    end

    subgraph "Analysis Tools"
        AST[Python AST]
        TREE_SITTER[Tree-sitter]
        PYGMENTS[Pygments]
    end

    subgraph "ML/NLP Stack"
        SKLEARN[scikit-learn]
        TRANSFORMERS[Sentence Transformers]
        YAKE[YAKE]
        NUMPY[NumPy]
    end

    subgraph "Storage & Cache"
        SQLITE[SQLite]
        REDIS[Redis Optional]
        PICKLE[Pickle Cache]
    end

    PYTHON --> AST
    PYTHON --> SKLEARN
    TYPER --> SQLITE
    PYDANTIC --> PICKLE
```

### Key Components by Feature

| Component | Technology | Purpose |
|-----------|------------|---------|
| **CLI Framework** | Typer + Rich | Modern CLI with progress bars |
| **Configuration** | YAML + Pydantic | Type-safe configuration |
| **Code Parsing** | AST + Tree-sitter | Multi-language analysis |
| **Text Analysis** | BM25 (primary) | Probabilistic relevance scoring |
| **ML Models** | Sentence Transformers | Semantic similarity |
| **Keyword Extraction** | YAKE | Statistical extraction |
| **Git Integration** | GitPython | Version control mining |
| **Database** | SQLite | Session storage |
| **Caching** | LRU + Disk | Multi-level caching |
| **Visualization** | Graphviz + D3.js | Dependency graphs |

---

## 🔗 Related Documentation

- **[Core Systems →](core-systems.md)** - Detailed analysis engines and pipelines
- **[Data & Storage →](data-storage.md)** - Database design and caching
- **[Performance →](performance.md)** - Optimization and scalability

---

<div class="section-stats">
💡 This is a performance-optimized version. For the complete technical documentation, see the [original architecture file](../ARCHITECTURE-original.md).
</div>

<script>
// Mark this page as using architecture optimizations
document.body.classList.add('architecture-page');
</script>
