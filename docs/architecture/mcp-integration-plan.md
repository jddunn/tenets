# Tenets MCP Integration: Investigation & Evolution Plan

> **Status**: Investigation Phase  
> **Date**: December 2024  
> **Vision**: Transform tenets into a first-class MCP (Model Context Protocol) server for AI-powered code intelligence

---

## Executive Summary

Tenets is positioned to become **the definitive MCP server for codebase intelligence**. Our existing NLP pipeline, multi-factor ranking system, session management, and local-first architecture align perfectly with MCP's design philosophy. This document outlines the comprehensive plan to evolve tenets from a CLI/API tool into a protocol-native context provider for AI agents and coding assistants.

### Why This Matters

AI coding assistants (Cursor, Windsurf, Claude Desktop, Codex CLI, etc.) are becoming the primary interface for software development. These tools need:
- **Intelligent context** beyond simple file reading
- **Ranked relevance** - not every file matters equally
- **Token optimization** - LLMs have limits
- **Session continuity** - coding is iterative
- **Guiding principles** - consistency across interactions

Tenets already solves these problems. MCP provides the universal protocol to expose these capabilities to **every** AI coding tool.

---

## Part 1: Current Architecture Analysis

### Existing Strengths for MCP

| Component | Current State | MCP Readiness |
|-----------|--------------|---------------|
| **Distiller** | Full NLP pipeline, ranking, aggregation | ⭐⭐⭐⭐⭐ Direct tool mapping |
| **Ranker** | BM25, TF-IDF, ML embeddings, git signals | ⭐⭐⭐⭐⭐ Core tool capability |
| **Session Manager** | SQLite persistence, context history | ⭐⭐⭐⭐ Resource + state management |
| **Tenet System** | Guiding principles injection | ⭐⭐⭐⭐⭐ Prompt templates |
| **Analyzers** | 15+ language AST parsers | ⭐⭐⭐⭐ Resource metadata |
| **Git Integration** | Chronicle, blame, momentum | ⭐⭐⭐⭐ Specialized tools |
| **Config System** | Hierarchical, env vars, YAML | ⭐⭐⭐⭐ MCP configuration |
| **Caching** | Multi-tier SQLite + memory | ⭐⭐⭐⭐⭐ Performance critical |

### Current Data Flow

```
User Prompt → PromptParser → Scanner → Analyzer → Ranker → Aggregator → Output
                   ↓             ↓         ↓          ↓
              Intent/Keywords  Files   Structure   Scores
                   ↓             ↓         ↓          ↓
                   └─────────────┴─────────┴──────────┘
                                    ↓
                            ContextResult
```

### Gap Analysis for MCP

| Requirement | Gap | Effort |
|-------------|-----|--------|
| JSON-RPC 2.0 transport | Not implemented | Medium |
| Tool definitions schema | Need to define | Low |
| Resource URI patterns | Need to design | Medium |
| Prompt templates | Partially exists (tenets) | Low |
| stdio/SSE/HTTP transport | Not implemented | Medium |
| Async operations | Partially async | Medium |
| Progress notifications | Not standardized | Low |

---

## Part 2: MCP Primitives Mapping

### Tools (Actions the AI can invoke)

Tools are the primary interface for AI agents to interact with tenets. Each maps to core functionality:

#### Core Context Tools

```python
@mcp.tool()
async def distill(
    prompt: str,
    path: str = ".",
    mode: Literal["fast", "balanced", "thorough"] = "balanced",
    max_tokens: int = 100000,
    format: Literal["markdown", "xml", "json"] = "markdown",
    include_tests: bool = False,
    session: Optional[str] = None,
) -> ContextResult:
    """
    Distill relevant context from a codebase for a given prompt.
    
    Returns ranked, aggregated code context optimized for LLM consumption.
    Uses multi-factor NLP ranking including BM25, keyword matching, 
    import centrality, and git signals.
    """
```

```python
@mcp.tool()
async def rank_files(
    prompt: str,
    path: str = ".",
    mode: Literal["fast", "balanced", "thorough", "ml"] = "balanced",
    top_n: int = 20,
    explain: bool = False,
) -> RankResult:
    """
    Rank files by relevance without retrieving content.
    
    Faster than distill - useful for understanding what files
    would be relevant before committing to full context retrieval.
    """
```

#### Analysis Tools

```python
@mcp.tool()
async def examine(
    path: str = ".",
    include_complexity: bool = True,
    include_hotspots: bool = True,
    include_ownership: bool = False,
) -> ExaminationResult:
    """
    Examine codebase structure, complexity, and quality metrics.
    
    Identifies maintenance hotspots, complexity outliers, and
    structural patterns in the codebase.
    """
```

```python
@mcp.tool()
async def chronicle(
    path: str = ".",
    since: str = "1 week",
    author: Optional[str] = None,
) -> ChronicleResult:
    """
    Analyze git history and development patterns.
    
    Returns commit activity, file churn, contributor patterns,
    and temporal development insights.
    """
```

```python
@mcp.tool()
async def momentum(
    path: str = ".",
    since: str = "last-month",
    team: bool = False,
) -> MomentumResult:
    """
    Track development velocity and team momentum.
    
    Provides sprint velocity metrics, contribution patterns,
    and development trend analysis.
    """
```

#### Session Management Tools

```python
@mcp.tool()
async def session_create(
    name: str,
    description: Optional[str] = None,
) -> SessionInfo:
    """Create a new development session for stateful context building."""
```

```python
@mcp.tool()
async def session_pin_file(
    session: str,
    file_path: str,
) -> bool:
    """Pin a file to a session for guaranteed inclusion in future distills."""
```

```python
@mcp.tool()
async def session_pin_folder(
    session: str,
    folder_path: str,
    patterns: Optional[List[str]] = None,
) -> int:
    """Pin all files in a folder (with optional filtering) to a session."""
```

#### Tenet Management Tools

```python
@mcp.tool()
async def tenet_add(
    content: str,
    priority: Literal["low", "medium", "high", "critical"] = "medium",
    category: Optional[str] = None,
    session: Optional[str] = None,
) -> Tenet:
    """
    Add a guiding principle (tenet) for consistent AI interactions.
    
    Tenets are strategically injected into context to maintain
    consistency and combat context drift in long conversations.
    """
```

```python
@mcp.tool()
async def tenet_instill(
    session: Optional[str] = None,
    force: bool = False,
) -> InstillResult:
    """Instill pending tenets, marking them active for injection."""
```

#### Visualization Tools

```python
@mcp.tool()
async def viz_dependencies(
    path: str = ".",
    format: Literal["svg", "png", "html", "json"] = "json",
    depth: int = 3,
) -> DependencyGraph:
    """
    Generate dependency graph visualization data.
    
    Maps import relationships and module dependencies
    for architecture understanding.
    """
```

```python
@mcp.tool()
async def viz_hotspots(
    path: str = ".",
    metric: Literal["complexity", "churn", "combined"] = "combined",
) -> HotspotData:
    """Identify and visualize maintenance hotspots in the codebase."""
```

---

### Resources (Data the AI can read)

Resources are read-only data exposed via URI patterns:

#### File Context Resources

```
tenets://context/{session_id}/latest
  → Latest generated context for a session

tenets://context/{session_id}/history
  → Context generation history for a session

tenets://files/{path}/analysis
  → Detailed analysis of a specific file (AST, complexity, imports)

tenets://files/{path}/summary  
  → Intelligent summary of a file preserving structure
```

#### Session State Resources

```
tenets://sessions/list
  → List of all sessions with metadata

tenets://sessions/{name}/state
  → Current session state including pinned files, tenets applied

tenets://sessions/{name}/context-history
  → Historical context results for the session
```

#### Codebase Intelligence Resources

```
tenets://analysis/structure
  → Project structure with language detection, file counts

tenets://analysis/dependencies
  → Full dependency graph in JSON format

tenets://analysis/complexity-report
  → Complexity metrics across the codebase

tenets://analysis/git-summary
  → Git history summary, recent activity, top contributors
```

#### Configuration Resources

```
tenets://config/current
  → Current configuration (read-only)

tenets://config/ranking-weights
  → Current ranking factor weights

tenets://tenets/list
  → All tenets with status and metadata

tenets://tenets/{id}
  → Specific tenet details
```

---

### Prompts (Reusable interaction templates)

Prompts define parameterized templates for common interactions:

#### Context Building Prompts

```python
@mcp.prompt()
def build_context_for_task(
    task_description: str,
    focus_areas: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> str:
    """
    Build optimal context for a development task.
    
    Analyzes the task, identifies relevant code areas,
    and generates comprehensive context with relevant
    tenets applied.
    """
    return f"""
Analyze this development task and build relevant context:

Task: {task_description}
{f"Focus Areas: {', '.join(focus_areas)}" if focus_areas else ""}
{f"Exclude: {', '.join(exclude_patterns)}" if exclude_patterns else ""}

Please use the tenets distill tool with appropriate parameters.
"""
```

#### Code Review Prompts

```python
@mcp.prompt()
def code_review_context(
    focus: Literal["security", "performance", "architecture", "general"] = "general",
    since: str = "last week",
) -> str:
    """Prepare context for code review with relevant changes and complexity analysis."""
```

#### Refactoring Prompts

```python
@mcp.prompt()
def refactoring_context(
    target: str,
    scope: Literal["file", "module", "system"] = "module",
) -> str:
    """Build context for refactoring with dependency impact analysis."""
```

#### Understanding Prompts

```python
@mcp.prompt()
def understand_codebase(
    depth: Literal["overview", "detailed", "deep"] = "overview",
    focus: Optional[str] = None,
) -> str:
    """Generate codebase understanding context with architecture insights."""
```

---

## Part 3: Architecture Evolution

### New Module Structure

```
tenets/
├── __init__.py              # Main API (unchanged)
├── cli/                     # CLI commands (unchanged)
├── core/                    # Core logic (unchanged)
├── models/                  # Data models (unchanged)
├── storage/                 # Storage layer (unchanged)
├── utils/                   # Utilities (unchanged)
├── mcp/                     # NEW: MCP Server Module
│   ├── __init__.py          # MCP server exports
│   ├── server.py            # FastMCP server implementation
│   ├── tools/               # Tool implementations
│   │   ├── __init__.py
│   │   ├── context.py       # distill, rank tools
│   │   ├── analysis.py      # examine, chronicle, momentum
│   │   ├── session.py       # session management tools
│   │   ├── tenets.py        # tenet management tools
│   │   └── viz.py           # visualization tools
│   ├── resources/           # Resource implementations
│   │   ├── __init__.py
│   │   ├── context.py       # Context resources
│   │   ├── session.py       # Session resources
│   │   ├── analysis.py      # Analysis resources
│   │   └── config.py        # Configuration resources
│   ├── prompts/             # Prompt templates
│   │   ├── __init__.py
│   │   ├── context.py       # Context building prompts
│   │   ├── review.py        # Code review prompts
│   │   └── refactoring.py   # Refactoring prompts
│   ├── transports/          # Transport implementations
│   │   ├── __init__.py
│   │   ├── stdio.py         # Local stdio transport
│   │   ├── sse.py           # SSE for web clients
│   │   └── http.py          # Streamable HTTP for remote
│   └── config.py            # MCP-specific configuration
└── api/                     # NEW: REST/GraphQL API (optional)
    ├── __init__.py
    ├── routes/
    └── schemas/
```

### Entry Points

```toml
# pyproject.toml additions
[project.scripts]
tenets = "tenets.cli.app:app"           # Existing CLI
tenets-mcp = "tenets.mcp.server:main"   # NEW: MCP Server

[project.entry-points."mcp.servers"]
tenets = "tenets.mcp.server:create_server"  # MCP discovery
```

### Configuration Schema

```yaml
# .tenets.yml additions
mcp:
  enabled: true
  transports:
    - type: stdio      # For local IDE integration
      enabled: true
    - type: sse        # For web-based clients  
      enabled: false
      port: 8080
    - type: http       # For remote deployment
      enabled: false
      port: 8081
      auth:
        type: bearer   # or "none", "oauth2"
        
  tools:
    distill: true
    rank: true
    examine: true
    chronicle: true
    momentum: true
    session: true
    tenet: true
    viz: true
    
  resources:
    context: true
    session: true
    analysis: true
    config: true
    
  prompts:
    enabled: true
    custom_templates_path: null  # Optional custom prompts
    
  security:
    sandbox_commands: true      # Sandbox tool execution
    max_file_size: 5_000_000    # Respect existing limits
    rate_limit:
      enabled: true
      requests_per_minute: 60
    audit_log:
      enabled: true
      path: ~/.tenets/mcp-audit.log
```

---

## Part 4: Implementation Strategy

### Phase 1: Core MCP Server (Weeks 1-2)

1. **Add MCP Python SDK dependency**
   ```toml
   dependencies = [
       ...
       "mcp[cli]>=1.0.0",
   ]
   ```

2. **Implement base server with stdio transport**
   - FastMCP wrapper for existing Tenets class
   - Basic tool definitions for distill and rank
   - stdio transport for local IDE integration

3. **Create initial tool implementations**
   - `distill` tool wrapping existing distiller
   - `rank_files` tool wrapping existing ranker
   - Basic error handling and progress reporting

### Phase 2: Full Tool Suite (Weeks 3-4)

1. **Analysis tools**
   - examine, chronicle, momentum wrappers
   - Dependency graph tools

2. **Session management tools**
   - Create, list, delete sessions
   - Pin file/folder operations
   - Session state management

3. **Tenet management tools**
   - Add, list, remove tenets
   - Instill operations
   - System instruction management

### Phase 3: Resources & Prompts (Weeks 5-6)

1. **Resource implementations**
   - Context resources with history
   - Session state resources
   - Analysis and config resources

2. **Prompt templates**
   - Context building prompts
   - Review and refactoring prompts
   - Custom prompt loading

3. **Transport expansion**
   - SSE transport for web integration
   - HTTP transport for remote deployment

### Phase 4: Production Hardening (Weeks 7-8)

1. **Security**
   - Input validation and sanitization
   - Rate limiting
   - Audit logging
   - Sandbox execution

2. **Performance**
   - Connection pooling
   - Request batching
   - Cache warming

3. **Observability**
   - Structured logging
   - Metrics collection
   - Health checks

### Phase 5: Ecosystem Integration (Weeks 9-10)

1. **IDE Integration Packages**
   - Cursor configuration templates
   - VS Code extension manifest
   - Claude Desktop config

2. **Documentation**
   - MCP integration guide
   - Tool reference
   - Configuration examples

3. **Deployment**
   - Docker image
   - Cloud deployment guides (Azure Functions, Cloud Run)
   - Homebrew formula update

---

## Part 5: Competitive Analysis

### Existing MCP Servers for Code

| Server | Focus | Gaps Tenets Fills |
|--------|-------|-------------------|
| **filesystem** | Raw file read/write | No intelligence, no ranking |
| **git** | Git operations | No context optimization |
| **github** | GitHub API | No local analysis |
| **postgres** | Database queries | Not code-focused |

### Tenets Differentiation

1. **Intelligent Context** - Not just files, but *ranked*, *optimized* context
2. **Token Awareness** - Built for LLM consumption limits
3. **Multi-Factor Ranking** - BM25 + TF-IDF + ML + Git signals
4. **Session Continuity** - Stateful workflows for iterative development
5. **Guiding Principles** - Tenets combat context drift
6. **Local-First Privacy** - No code leaves the machine
7. **Language Intelligence** - 15+ language AST parsers

---

## Part 6: Documentation Updates Required

### Architecture Docs to Update

1. **system-overview.md** - Add MCP as primary interface
2. **core-architecture.md** - Add MCP data flow diagrams
3. **cli-api.md** - Expand to CLI/API/MCP architecture
4. **roadmap.md** - Update with MCP milestones

### New Documentation

1. **mcp-server.md** - Complete MCP integration guide
2. **mcp-tools-reference.md** - Tool API documentation
3. **mcp-deployment.md** - Deployment and configuration
4. **mcp-security.md** - Security considerations

### README Updates

- Add MCP as headline feature
- Quick start for MCP usage
- Cursor/Claude Desktop configuration examples

---

## Part 7: Marketing & Product Positioning

### Value Proposition

> **"tenets: The intelligent context layer for AI coding assistants"**
>
> Stop fighting token limits. Stop manually copying files. Tenets automatically finds, ranks, and optimizes the exact context your AI assistant needs—with guiding principles that maintain consistency across every interaction.

### Key Messages

1. **For Individual Developers**
   - "Your AI assistant finally understands your codebase"
   - "Context that fits, ranked by relevance"
   - "Guiding principles that stick"

2. **For Teams**
   - "Consistent AI interactions across the team"
   - "Shared tenets, shared understanding"
   - "Local-first, privacy guaranteed"

3. **For Enterprises**
   - "Zero data egress - your code stays yours"
   - "Audit logging and compliance"
   - "Custom ML models and integrations"

### Integration Showcases

- **Cursor**: Native MCP integration walkthrough
- **Claude Desktop**: Configuration and usage guide
- **Windsurf/Codeium**: Integration patterns
- **Codex CLI**: Pipeline integration
- **Custom Agents**: SDK usage examples

---

## Part 8: Open Questions for Discussion

### Technical Questions

1. **Transport Priority**: Should stdio be the only transport initially, or include HTTP from the start for remote deployments?

2. **Tool Granularity**: Should `distill` be one tool or broken into `find_relevant`, `rank`, `aggregate`?

3. **Resource Caching**: How should MCP resource caching interact with existing tenets cache?

4. **Async Model**: Should all tools be async-first, or support both sync and async?

### Product Questions

1. **CLI vs MCP**: Should the CLI become a thin wrapper around MCP, or remain independent?

2. **Pricing/Tiers**: Should MCP features require pro/enterprise tiers?

3. **Custom Prompts**: How much prompt customization should be exposed via MCP?

4. **Enterprise Features**: What MCP-specific enterprise features are needed?

### Ecosystem Questions

1. **IDE Extensions**: Should we build Cursor/VS Code extensions, or rely on native MCP support?

2. **Marketplace**: Should tenets be listed in MCP server directories/marketplaces?

3. **Community**: How to encourage community-contributed tools/resources/prompts?

---

## Appendix A: MCP Protocol Reference

### JSON-RPC 2.0 Message Format

```json
// Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "distill",
    "arguments": {
      "prompt": "implement OAuth2",
      "mode": "balanced"
    }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": "...",
    "isError": false
  }
}
```

### Tool Definition Schema

```json
{
  "name": "distill",
  "description": "Distill relevant context from codebase",
  "inputSchema": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "Task or query to build context for"
      },
      "mode": {
        "type": "string",
        "enum": ["fast", "balanced", "thorough"],
        "default": "balanced"
      }
    },
    "required": ["prompt"]
  }
}
```

### Resource URI Patterns

```
tenets://context/{session}/latest    # Templated
tenets://sessions/list               # Static
tenets://files/{path}/analysis       # Dynamic path
```

---

## Appendix B: Dependency Additions

```toml
# Required for MCP
[project.optional-dependencies]
mcp = [
    "mcp[cli]>=1.0.0",        # Official MCP SDK
    "sse-starlette>=1.6.0",    # SSE transport
    "uvicorn>=0.23.0",         # ASGI server for HTTP
]

# Full installation
all = [
    "tenets[light,ml,viz,web,db,mcp]",
]
```

---

## Next Steps

1. [ ] Review and discuss this plan with stakeholders
2. [ ] Finalize tool/resource/prompt specifications
3. [ ] Create detailed technical design for Phase 1
4. [ ] Set up MCP SDK development environment
5. [ ] Begin implementation of core server

---

*This document represents the vision for tenets as an MCP server. All architectural changes should maintain backward compatibility with existing CLI and API interfaces while positioning MCP as the primary protocol for AI agent integration.*

