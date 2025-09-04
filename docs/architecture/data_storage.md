---
title: Data & Storage
description: How Tenets manages data persistence, caching, and storage
---

# Data & Storage Architecture

## Overview

Tenets uses a layered storage architecture to manage configuration, sessions, cache, and temporary data efficiently.

## Storage Architecture Overview

```mermaid
graph TB
    subgraph "Memory Cache - L1"
        LRU[LRU Cache<br/>1000 items<br/>Sub-ms access]
        HOT[Hot Data<br/>Recent queries<br/>Active sessions]
    end

    subgraph "SQLite Database - L2"
        SESSIONS[Sessions<br/>User state]
        CONFIG[Configuration<br/>Settings]
        RELATIONS[Dependencies<br/>File graphs]
    end

    subgraph "Disk Cache - L3"
        ANALYSIS[Analysis Results<br/>Parse trees]
        EMBEDDINGS[ML Embeddings<br/>Vectors]
        CONTENT[Processed Files<br/>Token counts]
    end

    subgraph "File System - L4"
        LOGS[Application Logs]
        EXPORTS[Exported Sessions]
        ARCHIVES[Historical Data]
    end

    LRU --> SESSIONS
    HOT --> CONFIG

    SESSIONS --> ANALYSIS
    CONFIG --> EMBEDDINGS
    RELATIONS --> CONTENT

    ANALYSIS --> LOGS
    EMBEDDINGS --> EXPORTS
    CONTENT --> ARCHIVES

    style LRU fill:#ffeb3b
    style SESSIONS fill:#4caf50
    style ANALYSIS fill:#2196f3
    style LOGS fill:#9e9e9e
```

## Storage Hierarchy

### 1. Configuration Storage

**Location**: Project root or user home directory

```
.tenets.yml          # Project configuration
~/.tenets/config     # Global user configuration
```

**Contents:**
- Ranking algorithms settings
- Output format preferences
- Ignore patterns
- Token limits

### 2. Session Storage

**Location**: `.tenets/sessions/` in project root

```
.tenets/
  sessions/
    default.json
    feature-auth.json
    bug-fix-123.json
```

**Session Data Structure:**
```json
{
  "id": "feature-auth",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T14:30:00Z",
  "pinned_files": [
    "src/auth/oauth.py",
    "src/auth/jwt.py"
  ],
  "tenets": [
    {
      "content": "Always validate JWT tokens",
      "priority": "critical"
    }
  ],
  "context": {
    "last_prompt": "implement OAuth2",
    "last_files": ["src/auth/*.py"]
  }
}
```

### 3. Cache Management

**Location**: `.tenets/cache/` or system temp directory

#### Cache Layers

1. **File Metadata Cache**
   - File sizes
   - Modification times
   - Language detection
   - Complexity scores

2. **Parse Tree Cache**
   - AST representations
   - Import graphs
   - Symbol tables
   - Documentation blocks

3. **Git History Cache**
   - Commit information
   - Author statistics
   - File change frequency
   - Branch information

4. **Ranking Cache**
   - Computed relevance scores
   - TF-IDF vectors
   - Semantic embeddings
   - Factor calculations

### 4. Temporary Storage

**Location**: System temp directory

Used for:
- Processing large files
- Intermediate computations
- Export generation
- Visualization rendering

## Session Management Architecture

### Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: tenets session create
    Created --> Active: First prompt
    Active --> Analyzing: Processing files
    Analyzing --> Active: Context built
    Active --> Updated: Subsequent prompts
    Updated --> Analyzing: Incremental update
    Active --> Exported: Save for sharing
    Exported --> Archived: Long-term storage
    Active --> [*]: Session ends
    Archived --> [*]

    note right of Active
        - Pinned files included
        - Tenets applied
        - Context cached
    end note

    note right of Updated
        - Only changed files
        - Incremental updates
        - Previous context referenced
    end note
```

### Session Storage Schema

```mermaid
graph TB
    subgraph "Session Tables"
        SESSIONS[sessions<br/>id, name, project, created]
        PROMPTS[prompts<br/>id, session_id, text]
        CONTEXTS[contexts<br/>id, prompt_id, content]
        FILES[pinned_files<br/>session_id, file_path]
        TENETS[tenets<br/>session_id, content, priority]
    end

    subgraph "Relationships"
        S_P[1:N Prompts per Session]
        P_C[1:1 Context per Prompt]
        S_F[1:N Pinned Files]
        S_T[1:N Tenets]
    end

    SESSIONS --> S_P
    S_P --> PROMPTS
    PROMPTS --> P_C
    P_C --> CONTEXTS
    SESSIONS --> S_F
    S_F --> FILES
    SESSIONS --> S_T
    S_T --> TENETS

    style SESSIONS fill:#e1f5fe
    style CONTEXTS fill:#e8f5e8
```

## Data Models

### FileInfo Model

```python
class FileInfo:
    path: str
    size: int
    language: str
    modified: datetime
    complexity: float
    imports: List[str]
    exports: List[str]
    relevance_score: float
```

### Context Model

```python
class Context:
    prompt: str
    files: List[FileInfo]
    token_count: int
    timestamp: datetime
    session_id: Optional[str]
    tenets: List[Tenet]
```

### Session Model

```python
class Session:
    id: str
    created_at: datetime
    updated_at: datetime
    pinned_files: List[str]
    tenets: List[Tenet]
    history: List[Context]
    config_overrides: Dict
```

## Persistence Strategies

### 1. Lazy Loading

- Load file contents only when needed
- Stream large files instead of loading entirely
- Defer expensive computations

### 2. Incremental Updates

- Update only changed files in cache
- Recompute scores only for affected files
- Maintain dirty flags for cache invalidation

### 3. Compression

- Compress cached parse trees
- Use efficient serialization formats
- Apply content deduplication

## Multi-Level Cache System

### Cache Architecture

```mermaid
graph LR
    subgraph "Cache Levels"
        L1[Memory Cache<br/>Hot data]
        L2[SQLite Cache<br/>Structured data]
        L3[Disk Cache<br/>Bulk storage]
    end

    subgraph "Cache Types"
        META[File Metadata<br/>Size, mtime, language]
        PARSE[Parse Trees<br/>AST, symbols]
        RANK[Rankings<br/>Scores, relevance]
        EMBED[Embeddings<br/>ML vectors]
    end

    subgraph "Performance"
        HIT[Cache Hit<br/><1ms]
        MISS[Cache Miss<br/>Recompute]
        WARM[Cache Warming<br/>Preload]
    end

    L1 --> META
    L2 --> PARSE
    L2 --> RANK
    L3 --> EMBED

    META --> HIT
    PARSE --> HIT
    RANK --> MISS
    EMBED --> WARM

    style HIT fill:#4caf50
    style MISS fill:#ff9800
```

### Cache Key Generation

```python
class CacheKeyGenerator:
    """Generate cache keys for different data types."""

    def file_metadata_key(self, path: str) -> str:
        return f"meta:{path}:{mtime}"

    def analysis_key(self, path: str, analyzer: str) -> str:
        return f"analysis:{analyzer}:{path}:{content_hash}"

    def ranking_key(self, prompt: str, algorithm: str) -> str:
        return f"rank:{algorithm}:{prompt_hash}:{git_commit}"

    def embedding_key(self, content: str, model: str) -> str:
        return f"embed:{model}:{content_hash}"
```

## Cache Invalidation

### Triggers

1. **File System Changes**
   - File modification
   - File deletion
   - New file creation

2. **Git Operations**
   - New commits
   - Branch switches
   - Merge operations

3. **Configuration Changes**
   - Algorithm updates
   - Threshold adjustments
   - Pattern modifications

4. **Time-based**
   - TTL expiration
   - Scheduled cleanup
   - Age-based eviction

### Invalidation Strategies

```mermaid
graph TD
    subgraph "Invalidation Triggers"
        MTIME[File Modified]
        HASH[Content Changed]
        GIT[Git Commit]
        DEP[Dependencies Changed]
        TTL[TTL Expired]
        MANUAL[User Refresh]
    end

    subgraph "Invalidation Actions"
        EVICT[Evict Entry]
        CASCADE[Cascade Delete]
        MARK[Mark Stale]
        REBUILD[Rebuild Cache]
    end

    subgraph "Rebuild Strategy"
        LAZY[Lazy - On demand]
        EAGER[Eager - Background]
        BATCH[Batch - Multiple files]
    end

    MTIME --> EVICT
    HASH --> CASCADE
    GIT --> CASCADE
    DEP --> CASCADE
    TTL --> MARK
    MANUAL --> REBUILD

    EVICT --> LAZY
    CASCADE --> EAGER
    MARK --> LAZY
    REBUILD --> BATCH

    style MTIME fill:#ffcdd2
    style CASCADE fill:#fff3e0
    style LAZY fill:#c5e1a5
```

### Cache Strategy Comparison

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **LRU** | Parse trees, hot data | Simple, effective | May evict important items |
| **TTL** | Git history, embeddings | Predictable freshness | Wasteful if unchanged |
| **Size-based** | File metadata | Memory bounded | May thrash on large files |
| **Content Hash** | Analysis results | Accurate | Hash computation overhead |
| **Hybrid** | Production default | Balanced approach | Complex implementation |

## Storage Optimization

### 1. Deduplication

- Identify duplicate content
- Store single copy with references
- Apply at file and block level

### 2. Compression

- Gzip for text content
- Binary serialization for structures
- Delta compression for versions

### 3. Indexing

- Build indices for fast lookup
- Maintain sorted structures
- Use bloom filters for existence checks

## Data Security

### 1. Sensitive Data Handling

- Never cache credentials
- Exclude sensitive patterns
- Sanitize output

### 2. Access Control

- Respect file system permissions
- Honor .gitignore patterns
- Apply project-specific rules

### 3. Cleanup

- Clear temporary files
- Sanitize memory
- Secure deletion when needed

## Storage Configuration

### Environment Variables

```bash
TENETS_CACHE_DIR=/custom/cache/path
TENETS_SESSION_DIR=/custom/session/path
TENETS_CACHE_TTL=3600
TENETS_MAX_CACHE_SIZE=1GB
```

### Configuration File

```yaml
storage:
  cache:
    enabled: true
    directory: .tenets/cache
    max_size: 1073741824  # 1GB in bytes
    ttl: 3600  # seconds
    compression: true

  sessions:
    directory: .tenets/sessions
    auto_save: true
    max_history: 100

  temp:
    cleanup_on_exit: true
    max_file_size: 104857600  # 100MB
```

## Performance Metrics

### Cache Hit Rates

- Target: >80% for repeated operations
- Monitor: File metadata, parse trees, git data
- Optimize: Adjust TTL and size limits

### Storage Usage

- Monitor: Disk space consumption
- Alert: When approaching limits
- Action: Automatic cleanup policies

### Access Patterns

- Track: Most accessed files
- Optimize: Preload frequently used data
- Adjust: Cache priorities based on usage
