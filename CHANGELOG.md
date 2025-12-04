# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2025-12-04

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.4.1] - 2025-12-04

### Changed
- ${COMMIT_TITLE} (${COMMI)



## [0.4.0] - 2025-12-04

### Added
- **MCP Server** - Full Model Context Protocol server implementation (`tenets-mcp`)
  - Native integration with Cursor, Claude Desktop, Windsurf, and other MCP clients
  - 13 MCP tools: distill, rank_files, examine, chronicle, momentum, session management, tenet management
  - 4 MCP resources: sessions list, session state, tenets list, config
  - 3 MCP prompts: build_context_for_task, code_review_context, understand_codebase
  - Support for stdio, SSE, and HTTP transports
- Privacy Policy and Terms of Service pages
- Enhanced GDPR cookie consent with 3-tier options (Accept All / Essential Only / Reject All)
- MCP installation option: `pip install tenets[mcp]`

### Changed
- Landing page now highlights MCP as headline feature
- Updated site description and SEO meta tags for MCP focus
- Improved architecture documentation with MCP-first design philosophy
- Updated roadmap to reflect MCP as foundational layer

### Documentation
- New comprehensive MCP documentation with IDE configuration guides
- Detailed MCP integration plan and architecture vision
- Updated API reference with MCP module documentation

## [0.3.3] - 2025-01-22

### Fixed
- Resolved NeuralReranker circular import issues for test compatibility
- Added Rust and C/C++ import detection support
- Updated documentation to reflect accurate ranking factor percentages
- Clarified package installation options (core vs [light] extras)

## [0.3.2] - 2025-01-21

### Fixed
- Corrected README command examples for rank and distill commands
- Fixed documentation site styles and navigation
- Improved API documentation generation with proper module/class links

## [0.2.0] - 2025-01-11

### Added
- Automated version bumping on direct push to master branch
- Version bump workflow documentation

### Fixed
- Standardized CLI command syntax for rank/distill commands
- Documentation links and PyPI badge references
- Markdown navigation generation
- Pattern key normalization and folder reference typos
- Release workflow automation with manual trigger support

## [0.1.1] - 2025-01-11

### Fixed
- Normalized pattern keys and fixed misspelled folder references
- Documentation navigation tree builder for module/package name conflicts
- Formatting issues in navigation builder

### Changed
- Updated package description

## [0.1.0] - 2025-01-11  [YANKED]

### Note
- Initial release was yanked from PyPI due to incorrect package metadata
- Re-released as v0.1.1 with corrections

### Added
- Initial public release of Tenets CLI and Python package
- Core command suite: chronicle, examine, distill, rank, instill
- Multiple output formats (JSON, Markdown, XML, HTML, Plain text)
- Intelligent code analysis for 20+ programming languages
- Gitignore-aware file traversal and smart filtering
- Context-aware document chunking and relevance ranking
- Momentum scores for tracking file activity
- Session management for incremental context building
- Configurable system instructions for LLM outputs
- Optional ML features for enhanced ranking (tenets[ml])
- Optional visualization features for code graphs (tenets[viz])
- Comprehensive test suite with ~70% coverage
- Full documentation at https://tenets.dev/docs
- GitHub Actions CI/CD pipeline
- PyPI package distribution
