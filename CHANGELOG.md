# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
