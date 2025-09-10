# System Overview and Core Philosophy

Tenets is a sophisticated, local-first code intelligence platform that revolutionizes how developers interact with their codebases when working with AI assistants. Unlike traditional code search tools, Tenets employs advanced multi-stage analysis combining NLP, ML, static code analysis, git history mining, and intelligent ranking to build optimal context.

## Core Architecture Principles

1. **Local-First Processing**: All analysis happens on the developer's machine. No code leaves the local environment. External API calls only for optional LLM-based summarization with explicit consent.

2. **Progressive Enhancement**: Provides value immediately with just Python installed, scales with optional dependencies. Core functionality works without ML libraries, git integration works without configuration.

3. **Intelligent Caching**: Every expensive operation is cached at multiple levels - memory caches for hot data, SQLite for structured data, disk caches for analysis results, specialized caches for embeddings.

4. **Configurable Intelligence**: Every aspect of ranking and analysis can be configured. Users can adjust factor weights, enable/disable features, add custom ranking functions.

5. **Streaming Architecture**: Uses streaming and incremental processing wherever possible. Files analyzed as discovered, rankings computed in parallel, results stream to user.