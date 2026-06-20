# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.2] - 2026-06-20

### Fixed
- fix(mcp): read watchdog state as one atomic snapshot (no reap-at-completion race) (41ad9bb)



## [0.13.1] - 2026-06-20

### Fixed
- fix(mcp): never reap the stdio server while a request is in flight (73aa078)



## [0.13.0] - 2026-06-20

### Added
- feat(mcp): self-terminate abandoned stdio servers (idle + orphan watchdog) (2a74287)



## [0.12.6] - 2026-06-19

### Performance
- perf(ranking): memoize per-module import resolution (kills O(N^2) import-graph build) (1ea0607)



## [0.12.5] - 2026-06-19

### Fixed
- fix(ranking): include resolved imports in _analyze_corpus memo key (c8b2266)



## [0.12.4] - 2026-06-19

### Performance
- perf(ranking): memoize prompt-independent _analyze_corpus (skips O(N^2) import-graph rebuild) (d9e40bf)



## [0.12.3] - 2026-06-19

### Fixed
- fix(ranking): correct BM25 corpus on file-emptied update + harden index (80f8815)



## [0.12.2] - 2026-06-19

### Fixed
- fix(cli): close sqlite connection in index status (+ CodeRabbit findings) (9d411bf)



## [0.12.1] - 2026-06-19

### Fixed
- Pin the MCP server's working directory to the project root (`CLAUDE_PROJECT_DIR` / `TENETS_PROJECT_ROOT`) so `.tenets.yml` is discovered reliably regardless of where the server is spawned.



## [0.12.0] - 2026-06-19

### Added
- `tenets index` command (`status` / `clear` / `build`) to inspect, warm, or wipe the persistent corpus index.
- Clean MCP logging: detailed logs route to `~/.tenets/logs/tenets-mcp.log` while stderr stays at WARNING+ (no more `INFO` surfacing as `[ERROR]` in MCP clients).



## [0.11.0] - 2026-06-19

### Added
- Persistent incremental BM25/TF-IDF corpus index: the lexical corpus is built once and reused instead of rebuilt on every `rank`/`distill`. Unchanged files are never re-tokenized (in-memory warm layer + `DiskCache`); warm corpus build drops ~280x. Indexed BM25 scoring is byte-identical to a fresh build; controlled by `cache.index_enabled` (default on).



## [0.10.0] - 2026-01-15

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.9.4] - 2025-01-15

### Added
- **MCP lazy loading / tool search**: New meta-tools for token-efficient tool discovery
  - `tenets_search_tools`: Search available tools by keyword or category
  - `tenets_get_tool_schema`: Get full schema for a tool on-demand
  - `TOOL_REGISTRY`: 11 discoverable tools with minimal metadata
  - ~80% reduction in initial token overhead (from ~15k to ~3k tokens)

### Fixed
- Improved MCP tool descriptions for better IDE agent discoverability with semantic triggers and error guidance
- Standardized parameter defaults across chronicle/momentum tools (both now use "1 week")
- Enhanced error messages with corrective retry patterns for all MCP tools
- Condensed tool metadata from 45+ lines to 15-25 lines for faster scanning in tool pickers

## [0.9.0] - 2025-12-15

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.8.1] - 2025-12-13

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.8.0] - 2025-12-13

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.7.6] - 2025-12-09

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.7.5] - 2025-12-08

### Fixed
- Disable colored output in MCP stdio mode

## [0.7.1] - 2025-12-06

### Fixed
- Resolved Python 3.14 import recursion in `tenets.core.__init__` (MCP tools now load on 3.14).
- Stabilized ML stubs for `sentence_transformers` on Python 3.13/3.14 in test fixtures.

## [0.7.0] - 2025-12-06

### Added
- MCP tools now support HTML distill output and include/exclude/test filters on rank.
- Expanded MCP test suite (79 tests) with Python 3.14-compatible mocks and integration flows.

### Changed
- README adds MCP quick reference with configs and tool list.


## [0.6.5] - 2025-12-06

### Changed
- Sync MCP-first docs and landing content; add MCP quick setup examples.


## [0.5.0] - 2025-12-04

### Added
- MCP server module with tenets-distill, tenets-rank, tenets-examine, tenets-chronicle tools
- MCP prompt templates for code exploration
- MCP resource providers for project metrics
- tenets-mcp CLI entry point

### Changed
- Documentation updated for MCP-first design
- pyproject.toml includes MCP optional dependencies

## [0.4.0] - 2025-09-22

### Added
- Neural reranker with cross-encoder models
- Rust and C++ import detection in language analyzers
- Initial public release with core distill/rank/examine/chronicle commands

## [0.3.2] - 2025-09-12

### Fixed
- CLI command syntax standardization for rank/distill
- README command examples corrected

## [0.3.1] - 2025-09-11

### Fixed
- Pattern key normalization
- Folder reference typos

## [0.1.0] - 2025-09-10

### Added
- Initial release
- Context distillation engine
- File relevance ranking with TF-IDF and BM25
- Code examination with complexity metrics
- Git chronicle for repository history analysis
- Tenet management system
