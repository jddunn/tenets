# System Overview and Core Philosophy

Tenets is a sophisticated, local-first code intelligence platform that revolutionizes how developers interact with their codebases when working with AI assistants. Unlike traditional code search tools, Tenets employs advanced multi-stage analysis combining NLP, ML, static code analysis, git history mining, and intelligent ranking to build optimal context.

## Core Architecture Principles

1. **Local-First Processing**: All analysis happens on the developer's machine. No code leaves the local environment. External API calls only for optional LLM-based summarization with explicit consent.

2. **Progressive Enhancement**: Provides value immediately with just Python installed, scales with optional dependencies. Core functionality works without ML libraries, git integration works without configuration.

3. **Intelligent Caching**: Every expensive operation is cached at multiple levels - memory caches for hot data, SQLite for structured data, disk caches for analysis results, specialized caches for embeddings.

4. **Configurable Intelligence**: Every aspect of ranking and analysis can be configured. Users can adjust factor weights, enable/disable features, add custom ranking functions.

5. **Streaming Architecture**: Uses streaming and incremental processing wherever possible. Files analyzed as discovered, rankings computed in parallel, results stream to user.

## Architecture Deep Dive

### Core System Design

Tenets employs a sophisticated multi-stage pipeline architecture that balances performance, accuracy, and extensibility. The system is built as a series of loosely coupled components that communicate through well-defined interfaces, allowing independent optimization and testing.

**Component Architecture**

The system consists of six primary components:

1. **PromptParser**: Advanced NLP pipeline that extracts intent, keywords, entities, temporal expressions, and external references. Uses hybrid approaches combining rule-based patterns with optional ML models. Handles GitHub/JIRA/Linear issue detection and automatic content fetching.

2. **FileScanner**: High-performance parallel file discovery with .gitignore respect and configurable filtering. Uses OS-optimized traversal algorithms and streams results to downstream components without blocking.

3. **CodeAnalyzer**: Language-specific analysis orchestrator managing 15+ language analyzers. Extracts AST structure, imports/exports, complexity metrics, and semantic information. Implements aggressive caching with content-hash keys.

4. **RelevanceRanker**: Multi-factor ranking engine computing 8+ signals in parallel including BM25/TF-IDF text similarity, keyword matching, import centrality, git activity, path relevance, and optional semantic similarity. Supports pluggable custom ranking functions.

5. **DependencyGraph**: Intelligent import resolution system building project-wide dependency trees. Uses two-phase symbol collection and parallel resolution to avoid O(n²) complexity on large codebases.

6. **ContextAggregator**: Token-aware content selection and summarization. Understands model-specific token limits, preserves critical code structure, and supports multiple output formats (Markdown, XML, JSON).

**Data Flow Architecture**

Data flows through the pipeline in a streaming fashion:

1. Prompt → Parser → Intent + Keywords + Entities
2. Scanner → File Paths → Analyzer Pool (parallel)
3. Analyzed Files → Corpus Builder → TF-IDF/BM25 Indices
4. Files + Indices → Ranker → Scored Files
5. Scored Files → Aggregator → Final Context

Each stage can process data as soon as it's available from the previous stage, maximizing throughput and minimizing latency.

**Caching Strategy**

Multi-level caching ensures expensive operations are rarely repeated:

- **Memory Cache**: Hot data with 5-minute TTL for active sessions
- **Analysis Cache**: File analysis results keyed by content hash (1-hour TTL)
- **Git Cache**: Commit data and blame info (24-hour TTL)
- **Embedding Cache**: ML embeddings permanent until file changes
- **External Content Cache**: GitHub/JIRA content with status-aware TTL

**Parallelization Strategy**

Tenets achieves high performance through aggressive parallelization:

- File discovery runs in parallel with configurable worker threads
- Each file analyzed independently in thread pool
- Ranking factors computed concurrently for all files
- Import resolution parallelized across modules
- Git operations batched and executed in parallel

The system dynamically adjusts parallelism based on file count and available resources, using ThreadPoolExecutor for CPU-bound tasks and async I/O for network operations.

### Key Design Decisions

**Why We Chose Streaming Over Batch Processing**
Rather than phase-based processing (scan all → analyze all → rank all), Tenets streams data through the pipeline. This keeps all CPU cores busy, reduces memory pressure, and delivers results faster.

**Why We Use Lightweight AST Parsing**
Full semantic analysis would be more accurate but 10-100x slower. We extract just enough structure (imports, functions, classes) to make intelligent ranking decisions. This 80/20 approach provides 90% of the value at 10% of the cost.

**Why We Prefer Heuristics Over Perfect Accuracy**
Perfect import resolution requires full type checking and can take minutes. Our heuristic approach is right 95% of the time but 100x faster. The ranking algorithm naturally handles uncertainty through multiple signals.

**Why Everything Is Incremental**
Full rebuilds are expensive and usually unnecessary. Every operation supports incremental updates - changed files trigger minimal re-analysis, the dependency graph updates locally, rankings adjust without full recalculation.

### Advanced Features

**Intent-Aware Ranking**
The system adjusts ranking weights based on detected intent. Debug tasks prioritize error handlers and logs, refactoring emphasizes complexity metrics, testing boosts test files. This context-aware approach improves relevance by 30-40%.

**External Content Integration**
Automatically detects and fetches content from GitHub issues, JIRA tickets, Linear tasks, and other platforms. Extracted content enhances context and improves ranking accuracy for issue-specific queries.

**Dependency Graph Intelligence**
Beyond simple import tracking, the system builds a complete dependency graph with:
- Import centrality via PageRank algorithm
- Circular dependency detection
- Module boundary identification
- Change impact analysis

**Multi-Algorithm Ranking Modes**
- **Fast Mode**: Keyword and path matching only (<5 seconds)
- **Balanced Mode**: Adds BM25 and structure analysis (default, 10-15 seconds)
- **Thorough Mode**: Includes ML embeddings and pattern analysis (20-30 seconds)
- **Custom Mode**: User-defined ranking functions and weights

**Performance Characteristics**

The system can analyze hundreds of complex files with full dependency graphing in under 30 seconds. For a typical 500-file project:
- Initial analysis: 10-15 seconds
- Cached analysis: 2-3 seconds
- Incremental updates: <1 second

Performance scales linearly with CPU cores and sub-linearly with file count due to intelligent sampling and early termination strategies.

### Scaling to Massive Codebases

For codebases with thousands of files, Tenets employs additional strategies:

- **Adaptive Sampling**: Analyzes a representative sample first, then expands
- **Priority Queues**: High-signal files (imports, recent changes) analyzed first
- **Early Termination**: Stops when confidence threshold reached
- **Distributed Caching**: Shared team caches for common dependencies
- **GPU Acceleration**: Optional CUDA support for embedding calculations

This architecture scales linearly with cores and sub-linearly with file count, maintaining sub-30-second response times even on codebases with 10,000+ files.