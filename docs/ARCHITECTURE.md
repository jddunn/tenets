# Tenets Complete Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Philosophy & Design Principles](#core-philosophy--design-principles)
3. [Complete System Architecture](#complete-system-architecture)
4. [NLP/ML Pipeline Architecture](#nlpml-pipeline-architecture)
5. [File Discovery & Scanning System](#file-discovery--scanning-system)
6. [Code Analysis Engine](#code-analysis-engine)
7. [Relevance Ranking System](#relevance-ranking-system)
8. [Git Integration & Chronicle System](#git-integration--chronicle-system)
9. [Examination & Quality Analysis](#examination--quality-analysis)
10. [Momentum & Velocity Tracking](#momentum--velocity-tracking)
11. [Context Management & Optimization](#context-management--optimization)
12. [Session Management Architecture](#session-management-architecture)
13. [Storage & Caching Architecture](#storage--caching-architecture)
14. [Prompt Parsing & Understanding](#prompt-parsing--understanding)
15. [Output Generation & Formatting](#output-generation--formatting)
16. [Performance Architecture](#performance-architecture)
17. [Configuration System](#configuration-system)
18. [CLI & API Architecture](#cli--api-architecture)
19. [Visualization & Reporting](#visualization--reporting)
20. [Security & Privacy Architecture](#security--privacy-architecture)
21. [Extensibility & Plugin System](#extensibility--plugin-system)
22. [Deployment Architecture](#deployment-architecture)
23. [Testing & Quality Assurance](#testing--quality-assurance)
24. [Future Roadmap & Vision](#future-roadmap--vision)

## System Overview

### What is Tenets?

Tenets is a sophisticated, local-first code intelligence platform that revolutionizes how developers interact with their codebases when working with AI assistants. Unlike traditional code search tools or simple context builders, Tenets employs advanced multi-stage analysis combining natural language processing, machine learning, static code analysis, git history mining, and intelligent ranking to build optimal context for any given task.

The system operates entirely locally, ensuring complete privacy and security while delivering state-of-the-art code understanding capabilities. Every component is designed with performance in mind, utilizing aggressive caching, parallel processing, and incremental computation to handle codebases ranging from small projects to massive monorepos with millions of files.

### Core Architecture Principles

1. **Local-First Processing**: All analysis, ranking, and context generation happens on the developer's machine. No code ever leaves the local environment. External API calls are only made for optional LLM-based summarization, and even then, only with explicit user consent.

2. **Progressive Enhancement**: The system provides value immediately with just Python installed, and scales up with optional dependencies. Core functionality works without any ML libraries, git integration works without any configuration, and advanced features gracefully degrade when dependencies are missing.

3. **Intelligent Caching**: Every expensive operation is cached at multiple levels - memory caches for hot data, SQLite for structured data, disk caches for analysis results, and specialized caches for embeddings. Cache invalidation is intelligent, using file modification times, git commits, and content hashes.

4. **Configurable Intelligence**: Every aspect of the ranking and analysis can be configured. Users can adjust factor weights, enable/disable features, add custom ranking functions, and tune performance parameters. The system adapts to different codebases and use cases.

5. **Streaming Architecture**: The system uses streaming and incremental processing wherever possible. Files are analyzed as they're discovered, rankings are computed in parallel, and results stream to the user as they become available.

## Complete System Architecture

### High-Level Data Flow

```mermaid
graph TB
    subgraph "User Interaction Layer"
        CLI[CLI Interface<br/>typer]
        API[Python API<br/>Library]
        WebUI[Web UI<br/>Future]
        IDE[IDE Extensions]
    end

    subgraph "Command Orchestration"
        DISPATCHER[Command Dispatcher]
        DISTILL[Distill Command]
        EXAMINE[Examine Command]
        CHRONICLE[Chronicle Command]
        MOMENTUM[Momentum Command]
        SESSION[Session Management]
    end

    subgraph "Prompt Processing Layer"
        PARSER[Prompt Parser]
        INTENT[Intent Detection]
        KEYWORDS[Keyword Extraction]
        ENTITIES[Entity Extraction]

        subgraph "NLP Pipeline"
            TOKENIZER[Tokenizer]
            STOPWORDS[Stopwords]
            YAKE[YAKE Keywords]
            TFIDF[TF-IDF Analysis]
        end
    end

    subgraph "File Discovery & Analysis"
        SCANNER[File Scanner]
        GITIGNORE[.gitignore Parser]
        BINARY[Binary Detection]
        PARALLEL[Parallel Scanner]

        subgraph "Code Analysis Engine"
            PYTHON_ANALYZER[Python Analyzer]
            JS_ANALYZER[JavaScript Analyzer]
            GO_ANALYZER[Go Analyzer]
            JAVA_ANALYZER[Java Analyzer]
            GENERIC_ANALYZER[Generic Analyzer]
        end

        subgraph "AST & Structure"
            CLASSES[Class Extraction]
            FUNCTIONS[Function Extraction]
            IMPORTS[Import Analysis]
            EXPORTS[Export Analysis]
        end
    end

    subgraph "Intelligence & Ranking"
        subgraph "Ranking Engine"
            FAST[Fast Strategy]
            BALANCED[Balanced Strategy]
            THOROUGH[Thorough Strategy]
            ML[ML Strategy]
        end

        subgraph "Ranking Factors"
            SEMANTIC[Semantic Similarity<br/>25%]
            KEYWORD_MATCH[Keyword Matching<br/>15%]
            TFIDF_SIM[TF-IDF Similarity<br/>15%]
            IMPORT_CENT[Import Centrality<br/>10%]
            PATH_REL[Path Relevance<br/>10%]
            GIT_SIG[Git Signals<br/>15%]
        end

        subgraph "ML/NLP Pipeline"
            EMBEDDINGS[Local Embeddings]
            EMBED_CACHE[Embedding Cache]
            SIMILARITY[Similarity Computing]
        end
    end

    subgraph "Context Optimization"
        CONTEXT_BUILDER[Context Builder]
        TOKEN_COUNTER[Token Counter]
        SUMMARIZER[Summarizer]
        FORMATTER[Output Formatter]
    end

    subgraph "Storage & Persistence"
        SQLITE[SQLite Database<br/>Sessions]
        MEMORY[Memory Cache<br/>LRU]
        DISK[Disk Cache<br/>Analysis Results]
    end

    CLI --> DISPATCHER
    API --> DISPATCHER
    WebUI --> DISPATCHER
    IDE --> DISPATCHER

    DISPATCHER --> DISTILL
    DISPATCHER --> EXAMINE
    DISPATCHER --> CHRONICLE
    DISPATCHER --> MOMENTUM
    DISPATCHER --> SESSION

    DISTILL --> PARSER
    PARSER --> INTENT
    PARSER --> KEYWORDS
    PARSER --> ENTITIES

    INTENT --> TOKENIZER
    KEYWORDS --> YAKE
    ENTITIES --> TFIDF

    PARSER --> SCANNER
    SCANNER --> GITIGNORE
    SCANNER --> BINARY
    SCANNER --> PARALLEL

    SCANNER --> PYTHON_ANALYZER
    SCANNER --> JS_ANALYZER
    SCANNER --> GO_ANALYZER
    SCANNER --> JAVA_ANALYZER
    SCANNER --> GENERIC_ANALYZER

    PYTHON_ANALYZER --> CLASSES
    PYTHON_ANALYZER --> FUNCTIONS
    PYTHON_ANALYZER --> IMPORTS
    PYTHON_ANALYZER --> EXPORTS

    CLASSES --> FAST
    FUNCTIONS --> BALANCED
    IMPORTS --> THOROUGH
    EXPORTS --> ML

    FAST --> SEMANTIC
    BALANCED --> KEYWORD_MATCH
    THOROUGH --> TFIDF_SIM
    ML --> IMPORT_CENT

    SEMANTIC --> EMBEDDINGS
    EMBEDDINGS --> EMBED_CACHE
    EMBED_CACHE --> SIMILARITY

    SIMILARITY --> CONTEXT_BUILDER
    KEYWORD_MATCH --> CONTEXT_BUILDER
    TFIDF_SIM --> CONTEXT_BUILDER

    CONTEXT_BUILDER --> TOKEN_COUNTER
    CONTEXT_BUILDER --> SUMMARIZER
    CONTEXT_BUILDER --> FORMATTER

    FORMATTER --> SQLITE
    FORMATTER --> MEMORY
    FORMATTER --> DISK
```

### System Component Overview

```mermaid
graph LR
    subgraph "Core Components"
        NLP[NLP/ML Pipeline]
        SCAN[File Scanner]
        ANALYZE[Code Analyzer]
        RANK[Ranking Engine]
        CONTEXT[Context Builder]
    end

    subgraph "Analysis Tools"
        EXAMINE[Examine Tool]
        CHRONICLE[Chronicle Tool]
        MOMENTUM[Momentum Tool]
    end

    subgraph "Storage Systems"
        CACHE[Cache Manager]
        SESSION[Session Store]
        CONFIG[Configuration]
    end

    NLP --> RANK
    SCAN --> ANALYZE
    ANALYZE --> RANK
    RANK --> CONTEXT

    ANALYZE --> EXAMINE
    SCAN --> CHRONICLE
    CHRONICLE --> MOMENTUM

    RANK --> CACHE
    CONTEXT --> SESSION
    SESSION --> CONFIG
```

## NLP/ML Pipeline Architecture

### Pipeline Component Flow

```mermaid
graph TD
    subgraph "Input Processing"
        INPUT[Raw Text Input]
        PROMPT[User Prompt]
        CODE[Code Content]
    end

    subgraph "Tokenization Layer"
        CODE_TOK[Code Tokenizer<br/>camelCase, snake_case]
        TEXT_TOK[Text Tokenizer<br/>NLP processing]
    end

    subgraph "Keyword Extraction"
        YAKE_EXT[YAKE Extractor<br/>Statistical]
        TFIDF_EXT[TF-IDF Extractor<br/>Frequency-based]
        FREQ_EXT[Frequency Extractor<br/>Fallback]
    end

    subgraph "Stopword Management"
        CODE_STOP[Code Stopwords<br/>Minimal - 30 words]
        PROMPT_STOP[Prompt Stopwords<br/>Aggressive - 200+ words]
    end

    subgraph "Embedding Generation"
        LOCAL_EMB[Local Embeddings<br/>sentence-transformers]
        MODEL_SEL[Model Selection<br/>MiniLM, MPNet]
        FALLBACK[TF-IDF Fallback<br/>No ML required]
    end

    subgraph "Similarity Computing"
        COSINE[Cosine Similarity]
        EUCLIDEAN[Euclidean Distance]
        BATCH[Batch Processing]
    end

    subgraph "Caching System"
        MEM_CACHE[Memory Cache<br/>LRU 1000 items]
        DISK_CACHE[SQLite Cache<br/>30 day TTL]
    end

    INPUT --> CODE_TOK
    INPUT --> TEXT_TOK
    PROMPT --> TEXT_TOK
    CODE --> CODE_TOK

    CODE_TOK --> CODE_STOP
    TEXT_TOK --> PROMPT_STOP

    CODE_STOP --> YAKE_EXT
    PROMPT_STOP --> YAKE_EXT
    YAKE_EXT --> TFIDF_EXT
    TFIDF_EXT --> FREQ_EXT

    FREQ_EXT --> LOCAL_EMB
    LOCAL_EMB --> MODEL_SEL
    MODEL_SEL --> FALLBACK

    FALLBACK --> COSINE
    COSINE --> EUCLIDEAN
    EUCLIDEAN --> BATCH

    BATCH --> MEM_CACHE
    MEM_CACHE --> DISK_CACHE
```

### Embedding Model Architecture

```mermaid
graph LR
    subgraph "Model Options"
        MINI_L6[all-MiniLM-L6-v2<br/>90MB, Fast]
        MINI_L12[all-MiniLM-L12-v2<br/>120MB, Better]
        MPNET[all-mpnet-base-v2<br/>420MB, Best]
        QA_MINI[multi-qa-MiniLM<br/>Q&A Optimized]
    end

    subgraph "Processing Pipeline"
        BATCH_ENC[Batch Encoding]
        CHUNK[Document Chunking<br/>1000 chars, 100 overlap]
        VECTOR[Vector Operations<br/>NumPy optimized]
    end

    subgraph "Cache Strategy"
        KEY_GEN[Cache Key Generation<br/>model + content hash]
        WARM[Cache Warming]
        INVALID[Intelligent Invalidation]
    end

    MINI_L6 --> BATCH_ENC
    MINI_L12 --> BATCH_ENC
    MPNET --> BATCH_ENC
    QA_MINI --> BATCH_ENC

    BATCH_ENC --> CHUNK
    CHUNK --> VECTOR

    VECTOR --> KEY_GEN
    KEY_GEN --> WARM
    WARM --> INVALID
```

## File Discovery & Scanning System

### Scanner Architecture Flow

```mermaid
graph TD
    subgraph "Entry Points"
        ROOT[Project Root]
        PATHS[Specified Paths]
        PATTERNS[Include Patterns]
    end

    subgraph "Ignore System Hierarchy"
        CLI_IGNORE[CLI Arguments<br/>--exclude<br/>Highest Priority]
        TENETS_IGNORE[.tenetsignore<br/>Project-specific]
        GIT_IGNORE[.gitignore<br/>Version control]
        GLOBAL_IGNORE[Global Ignores<br/>~/.config/tenets/ignore<br/>Lowest Priority]
    end
    subgraph "Intelligent Test Exclusion"
        INTENT_DETECT[Intent Detection<br/>Test-related prompts?]
        CLI_OVERRIDE[CLI Override<br/>--include-tests / --exclude-tests]
        TEST_PATTERNS[Test Pattern Matching<br/>Multi-language support]
        TEST_DIRS[Test Directory Detection<br/>tests/, __tests__, spec/]
    end

    subgraph "Detection Systems"
        BINARY_DET[Binary Detection]
        EXT_CHECK[Extension Check]
        SIZE_CHECK[Size Check<br/>Max 10MB default]
        CONTENT_CHECK[Content Sampling<br/>Null byte detection]
        MAGIC_CHECK[Magic Number<br/>File signatures]
    end

    subgraph "Parallel Processing"
        WORK_QUEUE[Work Queue]
        PROCESS_POOL[Process Pool<br/>CPU-bound operations]
        THREAD_POOL[Thread Pool<br/>I/O operations]
        PROGRESS[Progress Tracking<br/>tqdm]
    end

    subgraph "Output"
        SCANNED_FILE[Scanned File Objects]
        METADATA[File Metadata]
        ANALYSIS_READY[Ready for Analysis]
    end

    ROOT --> CLI_IGNORE
    PATHS --> CLI_IGNORE
    PATTERNS --> CLI_IGNORE

    CLI_IGNORE --> TENETS_IGNORE
    TENETS_IGNORE --> GIT_IGNORE
    GIT_IGNORE --> GLOBAL_IGNORE

    GLOBAL_IGNORE --> BINARY_DET
    BINARY_DET --> EXT_CHECK
    EXT_CHECK --> SIZE_CHECK
    SIZE_CHECK --> CONTENT_CHECK
    CONTENT_CHECK --> MAGIC_CHECK

    MAGIC_CHECK --> WORK_QUEUE
    WORK_QUEUE --> PROCESS_POOL
    WORK_QUEUE --> THREAD_POOL
    PROCESS_POOL --> PROGRESS
    THREAD_POOL --> PROGRESS

    PROGRESS --> SCANNED_FILE
    SCANNED_FILE --> METADATA
    METADATA --> ANALYSIS_READY
```

### Binary Detection Strategy

```mermaid
flowchart TD
    FILE[Input File] --> EXT{Known Binary<br/>Extension?}
    EXT -->|Yes| BINARY[Mark as Binary]
    EXT -->|No| SIZE{Size > 10MB?}
    SIZE -->|Yes| SKIP[Skip File]
    SIZE -->|No| SAMPLE[Sample First 8KB]
    SAMPLE --> NULL{Contains<br/>Null Bytes?}
    NULL -->|Yes| BINARY
    NULL -->|No| RATIO[Calculate Text Ratio]
    RATIO --> THRESHOLD{Ratio > 95%<br/>Printable?}
    THRESHOLD -->|Yes| TEXT[Mark as Text]
    THRESHOLD -->|No| BINARY
    TEXT --> ANALYZE[Ready for Analysis]
    BINARY --> IGNORE[Skip Analysis]
    SKIP --> IGNORE
```

### Intelligent Test File Exclusion

Tenets implements intelligent test file handling to improve context relevance by automatically excluding or including test files based on the user's intent.

```mermaid
flowchart TD
    PROMPT[User Prompt] --> PARSE[Prompt Parsing]
    PARSE --> INTENT{Intent Detection<br/>Test-related?}

    INTENT -->|Yes| INCLUDE_TESTS[include_tests = True]
    INTENT -->|No| EXCLUDE_TESTS[include_tests = False]

    CLI_OVERRIDE{CLI Override?<br/>--include-tests<br/>--exclude-tests}
    CLI_OVERRIDE -->|--include-tests| FORCE_INCLUDE[include_tests = True]
    CLI_OVERRIDE -->|--exclude-tests| FORCE_EXCLUDE[include_tests = False]
    CLI_OVERRIDE -->|None| INTENT

    INCLUDE_TESTS --> SCAN_ALL[Scan All Files]
    EXCLUDE_TESTS --> TEST_FILTER[Apply Test Filters]
    FORCE_INCLUDE --> SCAN_ALL
    FORCE_EXCLUDE --> TEST_FILTER

    TEST_FILTER --> PATTERN_MATCH[Pattern Matching]
    PATTERN_MATCH --> DIR_MATCH[Directory Matching]

    subgraph "Test Patterns (Multi-language)"
        PY_PATTERNS["Python: test_*.py, *_test.py"]
        JS_PATTERNS["JavaScript: *.test.js, *.spec.js"]
        JAVA_PATTERNS["Java: *Test.java, *Tests.java"]
        GO_PATTERNS["Go: *_test.go"]
        GENERIC_PATTERNS["Generic: **/test/**, **/tests/**"]
    end

    subgraph "Test Directories"
        COMMON_DIRS["tests, __tests__, spec"]
        LANG_DIRS["unit_tests, integration_tests"]
        E2E_DIRS["e2e, e2e_tests, functional_tests"]
    end

    PATTERN_MATCH --> PY_PATTERNS
    PATTERN_MATCH --> JS_PATTERNS
    PATTERN_MATCH --> JAVA_PATTERNS
    PATTERN_MATCH --> GO_PATTERNS
    PATTERN_MATCH --> GENERIC_PATTERNS

    DIR_MATCH --> COMMON_DIRS
    DIR_MATCH --> LANG_DIRS
    DIR_MATCH --> E2E_DIRS

    PY_PATTERNS --> FILTERED_FILES[Filtered File List]
    JS_PATTERNS --> FILTERED_FILES
    JAVA_PATTERNS --> FILTERED_FILES
    GO_PATTERNS --> FILTERED_FILES
    GENERIC_PATTERNS --> FILTERED_FILES

    COMMON_DIRS --> FILTERED_FILES
    LANG_DIRS --> FILTERED_FILES
    E2E_DIRS --> FILTERED_FILES

    SCAN_ALL --> ANALYSIS[File Analysis]
    FILTERED_FILES --> ANALYSIS
```

**Intent Detection Patterns:**
- Test-related keywords: `test`, `tests`, `testing`, `unit`, `integration`, `spec`, `coverage`
- Test actions: `write tests`, `fix tests`, `run tests`, `test coverage`, `mock`
- Test files: `test_auth.py`, `auth.test.js`, `*Test.java`
- Test frameworks: `pytest`, `jest`, `mocha`, `junit`, `rspec`

**Benefits:**
- **Improved Relevance**: Non-test prompts get cleaner production code context
- **Automatic Intelligence**: Test prompts automatically include test files
- **Manual Override**: CLI flags provide full control when needed
- **Multi-language Support**: Recognizes test patterns across languages
- **Configuration**: Customizable patterns for project-specific conventions

## Code Analysis Engine

### Language Analyzer Architecture

```mermaid
graph TB
    subgraph "Base Analyzer Interface"
        BASE[LanguageAnalyzer<br/>Abstract Base Class]
        EXTRACT_IMP[extract_imports()]
        EXTRACT_EXP[extract_exports()]
        EXTRACT_CLS[extract_classes()]
        EXTRACT_FN[extract_functions()]
        CALC_COMP[calculate_complexity()]
        TRACE_DEP[trace_dependencies()]
    end

    subgraph "Language-Specific Analyzers"
        PYTHON[Python Analyzer<br/>Full AST parsing]
        JAVASCRIPT[JavaScript Analyzer<br/>ES6+ support]
        GOLANG[Go Analyzer<br/>Package detection]
        JAVA[Java Analyzer<br/>OOP patterns]
        RUST[Rust Analyzer<br/>Ownership patterns]
        GENERIC[Generic Analyzer<br/>Pattern-based fallback]
    end

    subgraph "Analysis Features"
        AST[AST Parsing]
        IMPORTS[Import Resolution]
        TYPES[Type Extraction]
        DOCS[Documentation Parsing]
        PATTERNS[Code Patterns]
        COMPLEXITY[Complexity Metrics]
    end

    BASE --> EXTRACT_IMP
    BASE --> EXTRACT_EXP
    BASE --> EXTRACT_CLS
    BASE --> EXTRACT_FN
    BASE --> CALC_COMP
    BASE --> TRACE_DEP

    BASE --> PYTHON
    BASE --> JAVASCRIPT
    BASE --> GOLANG
    BASE --> JAVA
    BASE --> RUST
    BASE --> GENERIC

    PYTHON --> AST
    PYTHON --> IMPORTS
    PYTHON --> TYPES
    PYTHON --> DOCS

    JAVASCRIPT --> PATTERNS
    GOLANG --> PATTERNS
    JAVA --> COMPLEXITY
    RUST --> COMPLEXITY
    GENERIC --> PATTERNS
```

### Python Analyzer Detail

```mermaid
graph LR
    subgraph "Python AST Analysis"
        AST_PARSE[AST Parser]
        NODE_VISIT[Node Visitor]
        SYMBOL_TABLE[Symbol Table]
    end

    subgraph "Code Structure"
        CLASSES[Class Definitions<br/>Inheritance chains]
        FUNCTIONS[Function Definitions<br/>Async detection]
        DECORATORS[Decorator Analysis]
        TYPE_HINTS[Type Hint Extraction]
    end

    subgraph "Import Analysis"
        ABS_IMP[Absolute Imports]
        REL_IMP[Relative Imports]
        STAR_IMP[Star Imports]
        IMPORT_GRAPH[Import Graph Building]
    end

    subgraph "Complexity Metrics"
        CYCLO[Cyclomatic Complexity<br/>+1 for if, for, while]
        COGNITIVE[Cognitive Complexity<br/>Nesting penalties]
        HALSTEAD[Halstead Metrics<br/>Operators/operands]
    end

    AST_PARSE --> NODE_VISIT
    NODE_VISIT --> SYMBOL_TABLE

    SYMBOL_TABLE --> CLASSES
    SYMBOL_TABLE --> FUNCTIONS
    SYMBOL_TABLE --> DECORATORS
    SYMBOL_TABLE --> TYPE_HINTS

    NODE_VISIT --> ABS_IMP
    NODE_VISIT --> REL_IMP
    NODE_VISIT --> STAR_IMP
    ABS_IMP --> IMPORT_GRAPH
    REL_IMP --> IMPORT_GRAPH
    STAR_IMP --> IMPORT_GRAPH

    SYMBOL_TABLE --> CYCLO
    SYMBOL_TABLE --> COGNITIVE
    SYMBOL_TABLE --> HALSTEAD
```

## Relevance Ranking System

### Multi-Factor Ranking Architecture

```mermaid
graph TD
    subgraph "Ranking Strategies"
        FAST[Fast Strategy<br/>~10ms/file<br/>Keyword + Path]
        BALANCED[Balanced Strategy<br/>~50ms/file<br/>TF-IDF + Structure]
        THOROUGH[Thorough Strategy<br/>~200ms/file<br/>Deep Analysis]
        ML_STRAT[ML Strategy<br/>~500ms/file<br/>Semantic Understanding]
    end

    subgraph "Semantic Understanding - 25%"
        SEM_SIM[Semantic Similarity<br/>ML-based understanding<br/>Local embeddings]
    end

    subgraph "Text Matching - 30%"
        KEY_MATCH[Keyword Matching<br/>15%<br/>Direct term hits]
        TFIDF_SIM[TF-IDF Similarity<br/>15%<br/>Statistical relevance]
    end

    subgraph "Code Structure - 20%"
        IMP_CENT[Import Centrality<br/>10%<br/>PageRank-style]
        PATH_REL[Path Relevance<br/>10%<br/>Directory structure]
    end

    subgraph "Git Signals - 15% (Optional)"
        GIT_REC[Git Recency<br/>5%<br/>Recent changes]
        GIT_FREQ[Git Frequency<br/>5%<br/>Change frequency]
        GIT_AUTH[Git Authors<br/>5%<br/>Author expertise]
    end

    subgraph "File Characteristics - 10%"
        FILE_TYPE[File Type<br/>5%<br/>Type relevance]
        CODE_PAT[Code Patterns<br/>5%<br/>Pattern matching]
    end

    subgraph "Scoring Engine"
        WEIGHTED[Weighted Combination]
        THRESHOLD[Threshold Filtering]
        NORMALIZED[Score Normalization]
        RANKED[Final Rankings]
    end

    FAST --> KEY_MATCH
    BALANCED --> TFIDF_SIM
    THOROUGH --> IMP_CENT
    ML_STRAT --> SEM_SIM

    SEM_SIM --> WEIGHTED
    KEY_MATCH --> WEIGHTED
    TFIDF_SIM --> WEIGHTED
    IMP_CENT --> WEIGHTED
    PATH_REL --> WEIGHTED
    GIT_REC --> WEIGHTED
    GIT_FREQ --> WEIGHTED
    GIT_AUTH --> WEIGHTED
    FILE_TYPE --> WEIGHTED
    CODE_PAT --> WEIGHTED

    WEIGHTED --> THRESHOLD
    THRESHOLD --> NORMALIZED
    NORMALIZED --> RANKED
```

### Factor Calculation Details

```mermaid
graph LR
    subgraph "Semantic Similarity Calculation"
        CHUNK[Chunk Long Files<br/>1000 chars, 100 overlap]
        EMBED[Generate Embeddings<br/>Local model]
        COSINE[Cosine Similarity]
        CACHE_SEM[Cache Results]
    end

    subgraph "Keyword Matching"
        FILENAME[Filename Match<br/>Weight: 0.4]
        IMPORT_M[Import Match<br/>Weight: 0.3]
        CLASS_FN[Class/Function Name<br/>Weight: 0.25]
        POSITION[Position Weight<br/>Early lines favored]
    end

    subgraph "Import Centrality"
        IN_EDGES[Incoming Edges<br/>Files importing this<br/>70% weight]
        OUT_EDGES[Outgoing Edges<br/>Files this imports<br/>30% weight]
        LOG_SCALE[Logarithmic Scaling<br/>High-degree nodes]
        NORMALIZE[Normalize 0-1]
    end

    subgraph "Git Signals"
        RECENCY[Recency Score<br/>Exponential decay<br/>30-day half-life]
        FREQUENCY[Frequency Score<br/>Log of commit count]
        EXPERTISE[Author Expertise<br/>Contribution volume]
        CHURN[Recent Churn<br/>Lines changed]
    end

    CHUNK --> EMBED
    EMBED --> COSINE
    COSINE --> CACHE_SEM

    FILENAME --> POSITION
    IMPORT_M --> POSITION
    CLASS_FN --> POSITION

    IN_EDGES --> LOG_SCALE
    OUT_EDGES --> LOG_SCALE
    LOG_SCALE --> NORMALIZE

    RECENCY --> EXPERTISE
    FREQUENCY --> EXPERTISE
    EXPERTISE --> CHURN
```

## Git Integration & Chronicle System

### Git Analysis Architecture

```mermaid
graph TD
    subgraph "Git Data Sources"
        COMMIT_LOG[Commit History]
        BLAME_DATA[Blame Information]
        BRANCH_INFO[Branch Analysis]
        MERGE_DATA[Merge Detection]
        CONFLICT_HIST[Conflict History]
    end

    subgraph "Chronicle Analysis"
        TEMPORAL[Temporal Analysis<br/>Activity patterns]
        CONTRIBUTORS[Contributor Tracking<br/>Author patterns]
        VELOCITY[Change Velocity<br/>Trend analysis]
        HOTSPOTS[Change Hotspots<br/>Problem areas]
    end

    subgraph "Metrics Calculation"
        BUS_FACTOR[Bus Factor<br/>Knowledge concentration]
        EXPERTISE[Author Expertise<br/>Domain knowledge]
        FRESHNESS[Code Freshness<br/>Age distribution]
        STABILITY[Change Stability<br/>Frequency patterns]
    end

    subgraph "Risk Assessment"
        KNOWLEDGE_RISK[Knowledge Risk<br/>Single points of failure]
        CHURN_RISK[Churn Risk<br/>High-change areas]
        COMPLEXITY_RISK[Complexity Risk<br/>Hard-to-maintain code]
        SUCCESSION[Succession Planning<br/>Knowledge transfer]
    end

    COMMIT_LOG --> TEMPORAL
    BLAME_DATA --> CONTRIBUTORS
    BRANCH_INFO --> VELOCITY
    MERGE_DATA --> HOTSPOTS
    CONFLICT_HIST --> HOTSPOTS

    CONTRIBUTORS --> BUS_FACTOR
    TEMPORAL --> EXPERTISE
    VELOCITY --> FRESHNESS
    HOTSPOTS --> STABILITY

    BUS_FACTOR --> KNOWLEDGE_RISK
    EXPERTISE --> CHURN_RISK
    FRESHNESS --> COMPLEXITY_RISK
    STABILITY --> SUCCESSION
```

### Chronicle Report Structure

```mermaid
graph LR
    subgraph "Executive Summary"
        HEALTH[Repository Health Score]
        KEY_METRICS[Key Metrics Dashboard]
        ALERTS[Risk Alerts]
    end

    subgraph "Activity Analysis"
        TIMELINE[Activity Timeline]
        PATTERNS[Change Patterns]
        TRENDS[Velocity Trends]
    end

    subgraph "Contributor Analysis"
        TEAM[Team Composition]
        EXPERTISE_MAP[Expertise Mapping]
        CONTRIBUTION[Contribution Patterns]
    end

    subgraph "Risk Assessment"
        RISKS[Identified Risks]
        RECOMMENDATIONS[Recommendations]
        ACTION_ITEMS[Action Items]
    end

    HEALTH --> TIMELINE
    KEY_METRICS --> PATTERNS
    ALERTS --> TRENDS

    TIMELINE --> TEAM
    PATTERNS --> EXPERTISE_MAP
    TRENDS --> CONTRIBUTION

    TEAM --> RISKS
    EXPERTISE_MAP --> RECOMMENDATIONS
    CONTRIBUTION --> ACTION_ITEMS
```

## Context Management & Optimization

### Context Building Pipeline

```mermaid
graph TD
    subgraph "Input Processing"
        RANKED_FILES[Ranked File Results]
        TOKEN_BUDGET[Available Token Budget]
        USER_PREFS[User Preferences]
    end

    subgraph "Selection Strategy"
        THRESHOLD[Score Threshold Filtering]
        TOP_N[Top-N Selection]
        DIVERSITY[Diversity Optimization]
        DEPENDENCIES[Dependency Inclusion]
    end

    subgraph "Token Management"
        MODEL_LIMITS[Model-Specific Limits<br/>4K, 8K, 16K, 32K, 100K]
        PROMPT_RESERVE[Prompt Token Reserve]
        RESPONSE_RESERVE[Response Token Reserve<br/>2K-4K]
        SAFETY_MARGIN[Safety Margin<br/>5% buffer]
    end

    subgraph "Content Optimization"
        SUMMARIZATION[Summarization Strategy]
        EXTRACTION[Key Component Extraction]
        COMPRESSION[Content Compression]
        FORMATTING[Output Formatting]
    end

    subgraph "Quality Assurance"
        COHERENCE[Context Coherence Check]
        COMPLETENESS[Completeness Validation]
        RELEVANCE[Relevance Verification]
        FINAL_OUTPUT[Final Context Output]
    end

    RANKED_FILES --> THRESHOLD
    TOKEN_BUDGET --> MODEL_LIMITS
    USER_PREFS --> TOP_N

    THRESHOLD --> TOP_N
    TOP_N --> DIVERSITY
    DIVERSITY --> DEPENDENCIES

    MODEL_LIMITS --> PROMPT_RESERVE
    PROMPT_RESERVE --> RESPONSE_RESERVE
    RESPONSE_RESERVE --> SAFETY_MARGIN

    DEPENDENCIES --> SUMMARIZATION
    SAFETY_MARGIN --> SUMMARIZATION
    SUMMARIZATION --> EXTRACTION
    EXTRACTION --> COMPRESSION
    COMPRESSION --> FORMATTING

    FORMATTING --> COHERENCE
    COHERENCE --> COMPLETENESS
    COMPLETENESS --> RELEVANCE
    RELEVANCE --> FINAL_OUTPUT
```

### Summarization Strategies

```mermaid
graph LR
    subgraph "Extraction Strategy"
        IMPORTS_EX[Imports/Exports<br/>Always included]
        SIGNATURES[Function/Class Signatures<br/>High priority]
        DOCSTRINGS[Docstrings/Comments<br/>Documentation]
        TYPES[Type Definitions<br/>Interface contracts]
    end

    subgraph "Compression Strategy"
        REDUNDANCY[Remove Redundancy<br/>Duplicate code]
        WHITESPACE[Normalize Whitespace<br/>Consistent formatting]
        COMMENTS[Condense Comments<br/>Key information only]
        BOILERPLATE[Remove Boilerplate<br/>Standard patterns]
    end

    subgraph "Semantic Strategy"
        MEANING[Preserve Meaning<br/>Core logic intact]
        CONTEXT[Maintain Context<br/>Relationship preservation]
        ABSTRACTIONS[Higher-level View<br/>Architectural overview]
        EXAMPLES[Key Examples<br/>Usage patterns]
    end

    subgraph "LLM Strategy (Optional)"
        EXTERNAL_API[External LLM API<br/>OpenAI/Anthropic]
        INTELLIGENT[Intelligent Summarization<br/>Context-aware]
        CONSENT[User Consent Required<br/>Privacy protection]
        FALLBACK[Fallback to Local<br/>If API unavailable]
    end

    IMPORTS_EX --> REDUNDANCY
    SIGNATURES --> WHITESPACE
    DOCSTRINGS --> COMMENTS
    TYPES --> BOILERPLATE

    REDUNDANCY --> MEANING
    WHITESPACE --> CONTEXT
    COMMENTS --> ABSTRACTIONS
    BOILERPLATE --> EXAMPLES

    MEANING --> EXTERNAL_API
    CONTEXT --> INTELLIGENT
    ABSTRACTIONS --> CONSENT
    EXAMPLES --> FALLBACK
```

## Session Management Architecture

### Session Lifecycle Flow

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> FirstPrompt: User provides initial prompt
    FirstPrompt --> Analyzing: Full codebase analysis
    Analyzing --> Active: Context built
    Active --> Interaction: Subsequent prompts
    Interaction --> Analyzing: Incremental updates
    Interaction --> Branching: Alternative exploration
    Branching --> Active: Branch selected
    Active --> Export: Save for sharing
    Export --> Archived: Long-term storage
    Archived --> [*]
    Active --> [*]: Session ends

    note right of FirstPrompt
        - Comprehensive analysis
        - All relevant files
        - Setup instructions
        - AI guidance
    end note

    note right of Interaction
        - Incremental updates only
        - Changed files highlighted
        - Previous context referenced
        - Minimal redundancy
    end note
```

### Session Storage Architecture

```mermaid
graph TB
    subgraph "Session Tables"
        SESSIONS[sessions<br/>id, name, project, created, updated]
        PROMPTS[prompts<br/>id, session_id, text, timestamp]
        CONTEXTS[contexts<br/>id, session_id, prompt_id, content]
        FILE_STATES[file_states<br/>session_id, file_path, state]
        AI_REQUESTS[ai_requests<br/>id, session_id, type, request]
    end

    subgraph "Relationships"
        SESSION_PROMPT[Session → Prompts<br/>One-to-Many]
        PROMPT_CONTEXT[Prompt → Context<br/>One-to-One]
        SESSION_FILES[Session → File States<br/>One-to-Many]
        SESSION_AI[Session → AI Requests<br/>One-to-Many]
    end

    subgraph "Operations"
        CREATE[Create Session]
        SAVE[Save State]
        RESTORE[Restore State]
        BRANCH[Branch Session]
        MERGE[Merge Sessions]
        EXPORT[Export Session]
    end

    SESSIONS --> SESSION_PROMPT
    SESSIONS --> SESSION_FILES
    SESSIONS --> SESSION_AI
    PROMPTS --> PROMPT_CONTEXT

    SESSION_PROMPT --> CREATE
    PROMPT_CONTEXT --> SAVE
    SESSION_FILES --> RESTORE
    SESSION_AI --> BRANCH
    CREATE --> MERGE
    SAVE --> EXPORT
```

## Storage & Caching Architecture

### Storage Hierarchy

```mermaid
graph TB
    subgraph "Memory Cache (Hottest)"
        LRU[LRU Cache<br/>1000 items default<br/>Sub-millisecond access]
        HOT_DATA[Frequently accessed data<br/>Recent analyses<br/>Active embeddings]
    end

    subgraph "SQLite Database (Structured)"
        SESSIONS_DB[Session Storage<br/>User interactions]
        CONFIG_DB[Configuration<br/>Settings & preferences]
        RELATIONS[Relationship data<br/>File dependencies]
        PERF[1-10ms access time]
    end

    subgraph "Disk Cache (Bulk)"
        ANALYSIS[Analysis Results<br/>File parsing cache]
        EMBEDDINGS[Embedding Cache<br/>ML vectors]
        FILE_CONTENT[File Content Cache<br/>Preprocessed data]
        BULK_PERF[10-100ms access time]
    end

    subgraph "File System (Cold)"
        LOGS[Application Logs<br/>Debugging information]
        EXPORTS[Exported Sessions<br/>Sharing & backup]
        ARCHIVES[Archived Data<br/>Historical sessions]
        COLD_PERF[100ms+ access time]
    end

    LRU --> SESSIONS_DB
    HOT_DATA --> CONFIG_DB

    SESSIONS_DB --> ANALYSIS
    CONFIG_DB --> EMBEDDINGS
    RELATIONS --> FILE_CONTENT

    ANALYSIS --> LOGS
    EMBEDDINGS --> EXPORTS
    FILE_CONTENT --> ARCHIVES
```

### Cache Invalidation Strategy

```mermaid
graph LR
    subgraph "Invalidation Triggers"
        FILE_MTIME[File Modification Time<br/>Filesystem change]
        CONTENT_HASH[Content Hash Change<br/>Actual content differs]
        GIT_COMMIT[Git Commit<br/>Version control change]
        DEP_CHANGE[Dependency Change<br/>Import graph update]
        TTL_EXPIRE[TTL Expiration<br/>Time-based cleanup]
        MANUAL[Manual Refresh<br/>User-initiated]
    end

    subgraph "Cache Levels Affected"
        MEMORY_INV[Memory Cache<br/>Immediate eviction]
        SQLITE_INV[SQLite Cache<br/>Mark as stale]
        DISK_INV[Disk Cache<br/>File removal]
        CASCADE[Cascade Invalidation<br/>Dependent entries]
    end

    subgraph "Rebuilding Strategy"
        LAZY[Lazy Rebuilding<br/>On-demand refresh]
        EAGER[Eager Rebuilding<br/>Background refresh]
        PARTIAL[Partial Rebuilding<br/>Incremental updates]
        BATCH[Batch Rebuilding<br/>Multiple files]
    end

    FILE_MTIME --> MEMORY_INV
    CONTENT_HASH --> SQLITE_INV
    GIT_COMMIT --> DISK_INV
    DEP_CHANGE --> CASCADE
    TTL_EXPIRE --> CASCADE
    MANUAL --> CASCADE

    MEMORY_INV --> LAZY
    SQLITE_INV --> EAGER
    DISK_INV --> PARTIAL
    CASCADE --> BATCH
```

## Performance Architecture

### Optimization Strategy Overview

```mermaid
graph TD
    subgraph "Parallel Processing"
        FILE_SCAN[File Scanning<br/>Process Pool<br/>CPU-bound operations]
        ANALYSIS[Code Analysis<br/>Thread Pool<br/>I/O operations]
        RANKING[Relevance Ranking<br/>Thread Pool<br/>Computation]
        EMBEDDING[Embedding Generation<br/>Batch Processing<br/>GPU if available]
    end

    subgraph "Streaming Architecture"
        INCREMENTAL[Incremental Discovery<br/>Stream files as found]
        PROGRESSIVE[Progressive Ranking<br/>Rank as analyzed]
        CHUNKED[Chunked Analysis<br/>Process in batches]
        STREAMING[Result Streaming<br/>First results quickly]
    end

    subgraph "Lazy Evaluation"
        DEFER[Defer Analysis<br/>Until needed]
        ON_DEMAND[On-demand Embeddings<br/>Generate when required]
        PROGRESSIVE_ENH[Progressive Enhancement<br/>Add features incrementally]
        JIT[Just-in-time Compilation<br/>Optimize hot paths]
    end

    subgraph "Memory Management"
        STREAMING_PROC[Streaming Processing<br/>Constant memory usage]
        GC[Incremental GC<br/>Prevent pauses]
        MMAP[Memory-mapped Files<br/>Large file handling]
        PRESSURE[Memory Pressure Monitor<br/>Adaptive behavior]
    end

    FILE_SCAN --> INCREMENTAL
    ANALYSIS --> PROGRESSIVE
    RANKING --> CHUNKED
    EMBEDDING --> STREAMING

    INCREMENTAL --> DEFER
    PROGRESSIVE --> ON_DEMAND
    CHUNKED --> PROGRESSIVE_ENH
    STREAMING --> JIT

    DEFER --> STREAMING_PROC
    ON_DEMAND --> GC
    PROGRESSIVE_ENH --> MMAP
    JIT --> PRESSURE
```

### Performance Benchmarks

#### File Analysis Performance

| Operation | Small (<10KB) | Medium (100KB) | Large (1MB) | Huge (10MB) |
|-----------|---------------|----------------|-------------|-------------|
| Read      | 0.5ms         | 2ms            | 10ms        | 100ms       |
| Tokenize  | 1ms           | 10ms           | 100ms       | 1s          |
| Parse AST | 5ms           | 50ms           | 500ms       | 5s          |
| Analyze   | 10ms          | 100ms          | 1s          | 10s         |
| Embed     | 50ms          | 100ms          | 200ms       | 500ms       |

#### System Performance

| Codebase | Files | Size | Full Analysis | Incremental | Memory |
|----------|-------|------|---------------|-------------|--------|
| Small    | 100   | 5MB  | 2s            | 100ms       | 50MB   |
| Medium   | 1K    | 50MB | 15s           | 500ms       | 200MB  |
| Large    | 10K   | 500MB| 2m            | 2s          | 800MB  |
| Huge     | 100K  | 5GB  | 20m           | 10s         | 3GB    |
| Monorepo | 1M    | 50GB | 3h            | 30s         | 8GB    |

## Configuration System

### Configuration Hierarchy

```mermaid
graph TB
    subgraph "Configuration Sources (Priority Order)"
        CLI[Command-line Arguments<br/>Highest Priority<br/>--algorithm, --exclude]
        ENV[Environment Variables<br/>TENETS_ALGORITHM=ml]
        PROJECT[Project Configuration<br/>.tenets.yml in project root]
        USER[User Configuration<br/>~/.config/tenets/config.yml]
        SYSTEM[System Defaults<br/>Built-in fallbacks<br/>Lowest Priority]
    end

    subgraph "Configuration Categories"
        RANKING_CONFIG[Ranking Configuration<br/>Algorithms, weights, factors]
        NLP_CONFIG[NLP Configuration<br/>Tokenization, stopwords]
        ML_CONFIG[ML Configuration<br/>Models, caching, devices]
        CACHE_CONFIG[Cache Configuration<br/>TTL, size limits, storage]
        SCANNER_CONFIG[Scanner Configuration<br/>Ignore patterns, limits]
        OUTPUT_CONFIG[Output Configuration<br/>Format, tokens, metadata]
    end

    subgraph "Dynamic Configuration"
        HOT_RELOAD[Hot Reload<br/>File change detection]
        API_UPDATE[Runtime API Updates<br/>Programmatic changes]
        VALIDATION[Configuration Validation<br/>Type checking, constraints]
        ROLLBACK[Error Rollback<br/>Revert on failure]
    end

    CLI --> RANKING_CONFIG
    ENV --> NLP_CONFIG
    PROJECT --> ML_CONFIG
    USER --> CACHE_CONFIG
    SYSTEM --> SCANNER_CONFIG

    RANKING_CONFIG --> HOT_RELOAD
    NLP_CONFIG --> API_UPDATE
    ML_CONFIG --> VALIDATION
    CACHE_CONFIG --> ROLLBACK
    SCANNER_CONFIG --> ROLLBACK
    OUTPUT_CONFIG --> ROLLBACK
```

### Complete Configuration Schema

```yaml
# .tenets.yml
version: 2

# Ranking configuration
ranking:
  algorithm: balanced  # fast|balanced|thorough|ml
  threshold: 0.1       # Minimum relevance score
  use_git: true        # Enable git signals
  use_ml: true         # Enable ML features

  # Factor weights (must sum to ~1.0)
  weights:
    semantic_similarity: 0.25
    keyword_match: 0.15
    tfidf_similarity: 0.15
    import_centrality: 0.10
    path_relevance: 0.10
    git_recency: 0.05
    git_frequency: 0.05
    git_authors: 0.05
    file_type: 0.05
    code_patterns: 0.05

  # Performance
  workers: 8           # Parallel workers
  batch_size: 100      # Batch size for ML

# NLP configuration
nlp:
  use_stopwords: true
  stopword_set: minimal  # minimal|aggressive|custom
  tokenizer: code        # code|text
  keyword_extractor: yake # yake|tfidf|frequency

# ML configuration
ml:
  model: all-MiniLM-L6-v2
  device: auto         # auto|cpu|cuda
  cache_embeddings: true
  embedding_dim: 384

# Cache configuration
cache:
  enabled: true
  directory: ~/.tenets/cache
  max_size_mb: 1000
  ttl_days: 7

  # SQLite pragmas
  sqlite_pragmas:
    journal_mode: WAL
    synchronous: NORMAL
    cache_size: -64000
    temp_store: MEMORY

# File scanning
scanner:
  respect_gitignore: true
  include_hidden: false
  follow_symlinks: false
  max_file_size_mb: 10
  binary_detection: true

  # Global ignores
  ignore_patterns:
    - "*.pyc"
    - "__pycache__"
    - "node_modules"
    - ".git"
    - ".venv"
    - "venv"
    - "*.egg-info"
    - "dist"
    - "build"

# Summarization configuration
summarizer:
  # Documentation context-aware summarization
  docs_context_aware: true           # Enable smart context-aware documentation summarization
  docs_show_in_place_context: true   # Preserve relevant context sections in-place within summaries
  docs_context_search_depth: 2       # 1=direct mentions, 2=semantic similarity, 3=deep analysis
  docs_context_min_confidence: 0.6   # Minimum confidence for context relevance (0.0-1.0)
  docs_context_max_sections: 10      # Maximum contextual sections to preserve per document
  docs_context_preserve_examples: true # Always preserve code examples and snippets

# Output configuration
output:
  format: markdown     # markdown|json|xml
  max_tokens: 100000
  include_metadata: true
  include_instructions: true
  copy_on_distill: false

# Session configuration
session:
  auto_save: true
  history_limit: 100
  branch_on_conflict: true

# Examination configuration
examination:
  complexity_threshold: 10
  duplication_threshold: 0.1
  min_test_coverage: 0.8

# Chronicle configuration
chronicle:
  include_merges: false
  max_commits: 1000
  analyze_patterns: true

# Momentum configuration
momentum:
  sprint_duration: 14
  velocity_window: 6
  include_weekends: false
```

## CLI & API Architecture

### Command Structure

```mermaid
graph TB
    subgraph "Main Commands"
        DISTILL[tenets distill<br/>Build optimal context]
        EXAMINE[tenets examine<br/>Code quality analysis]
        CHRONICLE[tenets chronicle<br/>Git history analysis]
        MOMENTUM[tenets momentum<br/>Velocity tracking]
        SESSION[tenets session<br/>Session management]
    end

    subgraph "Distill Options"
        PROMPT[--prompt TEXT<br/>Analysis prompt]
        ALGORITHM[--algorithm CHOICE<br/>fast|balanced|thorough|ml]
        MAX_TOKENS[--max-tokens INTEGER<br/>Token limit]
        FILTERS[--include/--exclude<br/>File patterns]
        SESSION_OPT[--session NAME<br/>Session context]
        COPY[--copy<br/>Copy to clipboard]
    end

    subgraph "Global Options"
        CONFIG[--config PATH<br/>Configuration file]
        VERBOSE[--verbose/-v<br/>Logging level]
        NO_CACHE[--no-cache<br/>Disable caching]
        WORKERS[--workers N<br/>Parallel processing]
    end

    subgraph "Output Formats"
        MARKDOWN[Markdown (default)<br/>Human-readable]
        JSON[JSON format<br/>Machine-readable]
        XML[XML format<br/>Structured data]
        RAW[Raw text<br/>Plain output]
    end

    DISTILL --> PROMPT
    DISTILL --> ALGORITHM
    DISTILL --> MAX_TOKENS
    DISTILL --> FILTERS
    DISTILL --> SESSION_OPT
    DISTILL --> COPY

    EXAMINE --> CONFIG
    CHRONICLE --> VERBOSE
    MOMENTUM --> NO_CACHE
    SESSION --> WORKERS

    COPY --> MARKDOWN
    SESSION_OPT --> JSON
    FILTERS --> XML
    WORKERS --> RAW
```

### Python API Design

```python
from tenets import Tenets

# Initialize
tenets = Tenets(path="./my-project")

# Simple usage
context = tenets.distill("implement OAuth2 authentication")

# Advanced usage
result = tenets.distill(
    prompt="refactor database layer",
    algorithm="ml",
    max_tokens=50000,
    filters=["*.py", "!test_*"]
)

# Session management
session = tenets.create_session("oauth-implementation")
context1 = session.distill("add OAuth2 support")
context2 = session.distill("add unit tests", incremental=True)

# Analysis tools
examination = tenets.examine()
chronicle = tenets.chronicle()
momentum = tenets.momentum()

# Configuration
tenets.configure(
    ranking_algorithm="thorough",
    use_ml=True,
    cache_ttl_days=30
)
```

## Security & Privacy Architecture

### Local-First Security Model

```mermaid
graph TB
    subgraph "Privacy Guarantees"
        LOCAL[All Processing Local<br/>No external API calls for analysis]
        NO_TELEMETRY[No Telemetry<br/>No usage tracking]
        NO_CLOUD[No Cloud Storage<br/>All data stays local]
        NO_PHONE_HOME[No Phone Home<br/>No automatic updates]
    end

    subgraph "Secret Detection"
        API_KEYS[API Key Detection<br/>Common patterns]
        PASSWORDS[Password Detection<br/>Credential patterns]
        TOKENS[Token Detection<br/>JWT, OAuth tokens]
        PRIVATE_KEYS[Private Key Detection<br/>RSA, SSH keys]
        CONNECTION_STRINGS[Connection Strings<br/>Database URLs]
        ENV_VARS[Environment Variables<br/>Sensitive values]
    end

    subgraph "Output Sanitization"
        REDACT[Redact Secrets<br/>Replace with placeholders]
        MASK_PII[Mask PII<br/>Personal information]
        CLEAN_PATHS[Clean File Paths<br/>Remove sensitive paths]
        REMOVE_URLS[Remove Internal URLs<br/>Private endpoints]
        ANONYMIZE[Anonymization Option<br/>Remove identifying info]
    end

    subgraph "Data Protection"
        ENCRYPTED_CACHE[Encrypted Cache<br/>Optional encryption at rest]
        SECURE_DELETE[Secure Deletion<br/>Overwrite sensitive data]
        ACCESS_CONTROL[File Access Control<br/>Respect permissions]
        AUDIT_LOG[Audit Logging<br/>Security events]
    end

    LOCAL --> API_KEYS
    NO_TELEMETRY --> PASSWORDS
    NO_CLOUD --> TOKENS
    NO_PHONE_HOME --> PRIVATE_KEYS

    API_KEYS --> REDACT
    PASSWORDS --> MASK_PII
    TOKENS --> CLEAN_PATHS
    PRIVATE_KEYS --> REMOVE_URLS
    CONNECTION_STRINGS --> ANONYMIZE
    ENV_VARS --> ANONYMIZE

    REDACT --> ENCRYPTED_CACHE
    MASK_PII --> SECURE_DELETE
    CLEAN_PATHS --> ACCESS_CONTROL
    REMOVE_URLS --> AUDIT_LOG
    ANONYMIZE --> AUDIT_LOG
```

### Secret Detection Patterns

```mermaid
graph LR
    subgraph "Detection Methods"
        REGEX[Regex Patterns<br/>Known formats]
        ENTROPY[Entropy Analysis<br/>Random strings]
        CONTEXT[Context Analysis<br/>Variable names]
        KEYWORDS[Keyword Detection<br/>password, secret, key]
    end

    subgraph "Secret Types"
        AWS[AWS Access Keys<br/>AKIA...]
        GITHUB[GitHub Tokens<br/>ghp_, gho_]
        JWT[JWT Tokens<br/>eyJ pattern]
        RSA[RSA Private Keys<br/>-----BEGIN RSA]
        DATABASE[Database URLs<br/>postgres://, mysql://]
        GENERIC[Generic Secrets<br/>High entropy strings]
    end

    subgraph "Response Actions"
        FLAG[Flag for Review<br/>Warn user]
        REDACT_AUTO[Auto Redaction<br/>Replace with [REDACTED]]
        EXCLUDE[Exclude File<br/>Skip entirely]
        LOG[Security Log<br/>Record detection]
    end

    REGEX --> AWS
    ENTROPY --> GITHUB
    CONTEXT --> JWT
    KEYWORDS --> RSA

    AWS --> FLAG
    GITHUB --> REDACT_AUTO
    JWT --> EXCLUDE
    RSA --> LOG
    DATABASE --> LOG
    GENERIC --> FLAG
```

## Testing & Quality Assurance

### Test Architecture

```mermaid
graph TB
    subgraph "Test Categories"
        UNIT[Unit Tests<br/>>90% coverage<br/>Fast, isolated]
        INTEGRATION[Integration Tests<br/>Component interaction<br/>Real workflows]
        E2E[End-to-End Tests<br/>Complete user journeys<br/>CLI to output]
        PERFORMANCE[Performance Tests<br/>Benchmark regression<br/>Memory usage]
    end

    subgraph "Test Structure"
        FIXTURES[Test Fixtures<br/>Sample codebases<br/>Known outputs]
        MOCKS[Mock Objects<br/>External dependencies<br/>Controlled behavior]
        HELPERS[Test Helpers<br/>Common operations<br/>Assertion utilities]
        FACTORIES[Data Factories<br/>Generate test data<br/>Realistic scenarios]
    end

    subgraph "Quality Metrics"
        COVERAGE[Code Coverage<br/>Line and branch coverage]
        COMPLEXITY[Complexity Limits<br/>Cyclomatic < 10]
        DUPLICATION[Duplication Check<br/>< 5% duplicate code]
        DOCUMENTATION[Documentation<br/>100% public API]
    end

    subgraph "Continuous Testing"
        PRE_COMMIT[Pre-commit Hooks<br/>Fast feedback]
        CI_PIPELINE[CI Pipeline<br/>Full test suite]
        NIGHTLY[Nightly Tests<br/>Extended scenarios]
        BENCHMARKS[Benchmark Tracking<br/>Performance trends]
    end

    UNIT --> FIXTURES
    INTEGRATION --> MOCKS
    E2E --> HELPERS
    PERFORMANCE --> FACTORIES

    FIXTURES --> COVERAGE
    MOCKS --> COMPLEXITY
    HELPERS --> DUPLICATION
    FACTORIES --> DOCUMENTATION

    COVERAGE --> PRE_COMMIT
    COMPLEXITY --> CI_PIPELINE
    DUPLICATION --> NIGHTLY
    DOCUMENTATION --> BENCHMARKS
```

### Test Coverage Requirements

```mermaid
graph LR
    subgraph "Coverage Targets"
        UNIT_COV[Unit Tests<br/>>90% coverage<br/>Critical paths 100%]
        INTEGRATION_COV[Integration Tests<br/>All major workflows<br/>Error scenarios]
        E2E_COV[E2E Tests<br/>Critical user journeys<br/>Happy paths]
        PERF_COV[Performance Tests<br/>Regression prevention<br/>Memory leak detection]
    end

    subgraph "Quality Gates"
        CODE_QUALITY[Code Quality<br/>Complexity < 10<br/>Function length < 50]
        DOCUMENTATION[Documentation<br/>100% public API<br/>Usage examples]
        SECURITY[Security Tests<br/>Secret detection<br/>Input validation]
        COMPATIBILITY[Compatibility<br/>Python 3.8+<br/>Multiple platforms]
    end

    UNIT_COV --> CODE_QUALITY
    INTEGRATION_COV --> DOCUMENTATION
    E2E_COV --> SECURITY
    PERF_COV --> COMPATIBILITY
```

## Future Roadmap & Vision

### Near Term (Q1 2025)

```mermaid
graph TB
    subgraph "Core Improvements"
        INCREMENTAL[Incremental Indexing<br/>Real-time updates<br/>Watch file changes]
        FASTER_EMBED[Faster Embeddings<br/>Model quantization<br/>ONNX optimization]
        LANGUAGE_SUP[Better Language Support<br/>30+ languages<br/>Language-specific patterns]
        IDE_PLUGINS[IDE Plugin Ecosystem<br/>VS Code, IntelliJ, Vim]
        CROSS_REPO[Cross-repository Analysis<br/>Monorepo support<br/>Dependency tracking]
    end

    subgraph "ML Enhancements"
        NEWER_MODELS[Newer Embedding Models<br/>Code-specific transformers<br/>Better accuracy]
        FINE_TUNING[Fine-tuning Pipeline<br/>Domain-specific models<br/>Custom training]
        MULTIMODAL[Multi-modal Understanding<br/>Diagrams, images<br/>Architecture docs]
        CODE_TRANSFORMERS[Code-specific Models<br/>Programming language aware<br/>Syntax understanding]
    end

    INCREMENTAL --> NEWER_MODELS
    FASTER_EMBED --> FINE_TUNING
    LANGUAGE_SUP --> MULTIMODAL
    IDE_PLUGINS --> CODE_TRANSFORMERS
    CROSS_REPO --> CODE_TRANSFORMERS
```

### Medium Term (Q2-Q3 2025)

```mermaid
graph TB
    subgraph "Platform Features"
        WEB_UI[Web UI<br/>Real-time collaboration<br/>Team workspaces]
        SHARED_CONTEXT[Shared Context Libraries<br/>Team knowledge base<br/>Best practices]
        KNOWLEDGE_GRAPHS[Knowledge Graphs<br/>Code relationships<br/>Semantic connections]
        AI_AGENTS[AI Agent Integration<br/>Autonomous assistance<br/>Proactive suggestions]
    end

    subgraph "Enterprise Features"
        SSO[SSO/SAML Support<br/>Enterprise authentication<br/>Role-based access]
        AUDIT[Audit Logging<br/>Compliance tracking<br/>Usage monitoring]
        COMPLIANCE[Compliance Modes<br/>GDPR, SOX, HIPAA<br/>Data governance]
        AIR_GAPPED[Air-gapped Deployment<br/>Offline operation<br/>Secure environments]
        CUSTOM_ML[Custom ML Models<br/>Private model training<br/>Domain expertise]
    end

    WEB_UI --> SSO
    SHARED_CONTEXT --> AUDIT
    KNOWLEDGE_GRAPHS --> COMPLIANCE
    AI_AGENTS --> AIR_GAPPED
    AI_AGENTS --> CUSTOM_ML
```

### Long Term (2026+)

```mermaid
graph TB
    subgraph "Vision Goals"
        AUTONOMOUS[Autonomous Code Understanding<br/>Self-improving analysis<br/>Minimal human input]
        PREDICTIVE[Predictive Development<br/>Anticipate needs<br/>Suggest improvements]
        UNIVERSAL[Universal Code Intelligence<br/>Any language, any domain<br/>Contextual understanding]
        INDUSTRY_STANDARD[Industry Standard<br/>AI pair programming<br/>Developer toolchain]
    end

    subgraph "Research Areas"
        GRAPH_NEURAL[Graph Neural Networks<br/>Code structure understanding<br/>Relationship modeling]
        REINFORCEMENT[Reinforcement Learning<br/>Ranking optimization<br/>Adaptive behavior]
        FEW_SHOT[Few-shot Learning<br/>New language support<br/>Rapid adaptation]
        EXPLAINABLE[Explainable AI<br/>Ranking transparency<br/>Decision reasoning]
        FEDERATED[Federated Learning<br/>Team knowledge sharing<br/>Privacy-preserving]
    end

    AUTONOMOUS --> GRAPH_NEURAL
    PREDICTIVE --> REINFORCEMENT
    UNIVERSAL --> FEW_SHOT
    INDUSTRY_STANDARD --> EXPLAINABLE
    INDUSTRY_STANDARD --> FEDERATED
```

### Technology Evolution

```mermaid
timeline
    title Tenets Technology Roadmap

    2025 Q1 : Core ML Pipeline
            : Local Embeddings
            : Multi-language Support
            : IDE Integrations

    2025 Q2 : Web Collaboration
            : Team Features
            : Enterprise Security
            : Performance Optimization

    2025 Q3 : Knowledge Graphs
            : AI Agent Integration
            : Custom Models
            : Advanced Analytics

    2026    : Autonomous Understanding
            : Predictive Intelligence
            : Graph Neural Networks
            : Industry Adoption

    2027+   : Universal Code Intelligence
            : Federated Learning
            : Next-gen AI Integration
            : Market Leadership
```

## Output Generation & Visualization

### Output Formatting System

The output formatting system in Tenets provides multiple format options to suit different use cases and integrations:

```mermaid
graph TB
    subgraph "Format Types"
        MARKDOWN[Markdown Format<br/>Human-readable]
        JSON[JSON Format<br/>Machine-parseable]
        XML[XML Format<br/>Structured data]
        HTML[HTML Format<br/>Interactive reports]
    end

    subgraph "HTML Report Features"
        INTERACTIVE[Interactive Elements<br/>Collapsible sections]
        VISUALS[Visualizations<br/>Charts & graphs]
        STYLING[Professional Styling<br/>Modern UI]
        RESPONSIVE[Responsive Design<br/>Mobile-friendly]
    end

    subgraph "Report Components"
        HEADER[Report Header<br/>Title & metadata]
        PROMPT_DISPLAY[Prompt Analysis<br/>Keywords & intent]
        STATS[Statistics Dashboard<br/>Metrics & KPIs]
        FILES[File Listings<br/>Code previews]
        GIT[Git Context<br/>Commits & contributors]
    end

    HTML --> INTERACTIVE
    HTML --> VISUALS
    HTML --> STYLING
    HTML --> RESPONSIVE

    INTERACTIVE --> HEADER
    VISUALS --> STATS
    STYLING --> FILES
    RESPONSIVE --> GIT
```

### HTML Report Generation

The HTML formatter leverages the reporting infrastructure to create rich, interactive reports:

#### Features:
- **Interactive Dashboard**: Collapsible sections, sortable tables, and filterable content
- **Visual Statistics**: Charts for file distribution, token usage, and relevance scores
- **Code Previews**: Syntax-highlighted code snippets with truncation for large files
- **Responsive Design**: Mobile-friendly layout that adapts to screen size
- **Professional Styling**: Modern UI with gradients, shadows, and animations
- **Git Integration**: Display of recent commits, contributors, and branch information

#### Architecture:

```python
class HTMLFormatter:
    """HTML report generation for distill command."""

    def format_html(self, aggregated, prompt_context, session):
        # Create HTML template with modern styling
        template = HTMLTemplate(theme="modern", include_charts=True)

        # Build report sections
        sections = [
            self._build_header(prompt_context, session),
            self._build_prompt_analysis(prompt_context),
            self._build_statistics(aggregated),
            self._build_file_cards(aggregated["included_files"]),
            self._build_git_context(aggregated.get("git_context"))
        ]

        # Generate final HTML with embedded styles and scripts
        return template.render(sections)
```

### Visualization Components

The visualization system provides rich visual representations of code analysis:

```mermaid
graph LR
    subgraph "Chart Types"
        BAR[Bar Charts<br/>File metrics]
        PIE[Pie Charts<br/>Language distribution]
        HEAT[Heatmaps<br/>Complexity visualization]
        GRAPH[Network Graphs<br/>Dependencies]
    end

    subgraph "Data Sources"
        METRICS[Code Metrics]
        RANKINGS[Relevance Rankings]
        TOKENS[Token Distribution]
        COVERAGE[Test Coverage]
    end

    METRICS --> BAR
    RANKINGS --> HEAT
    TOKENS --> PIE
    COVERAGE --> GRAPH
```

### Integration with Report Generator

The distill command now fully integrates with the report generation infrastructure:

1. **Shared Templates**: Uses the same HTML templates as the examine command
2. **Consistent Styling**: Unified visual design across all report types
3. **Reusable Components**: Shared visualization libraries and chart generators
4. **Export Options**: Support for PDF export via HTML rendering

### Usage Examples

```bash
# Generate HTML report for context
tenets distill "review API" --format html -o report.html

# Create interactive dashboard with verbose details
tenets distill "analyze security" --format html --verbose -o security_context.html

# Generate report with custom styling
tenets distill "refactor database" --format html --theme dark -o refactor.html
```

### Performance Optimizations

- **Lazy Loading**: Large code sections load on-demand
- **Virtual Scrolling**: Efficient rendering of long file lists
- **Minified Assets**: Compressed CSS and JavaScript
- **Inline Resources**: No external dependencies for offline viewing

## Conclusion

Tenets represents a fundamental shift in how developers interact with their codebases when working with AI. By combining sophisticated NLP/ML techniques with traditional code analysis, git mining, and intelligent caching, we've created a system that truly understands code in context.

The architecture is designed to be:

- **Performant**: Sub-second responses for most operations
- **Scalable**: From small projects to massive monorepos
- **Extensible**: Plugin system for custom logic
- **Private**: Everything runs locally
- **Intelligent**: State-of-the-art ML when available
- **Practical**: Works today, improves tomorrow

### Key Architectural Strengths

1. **Multi-Modal Intelligence**: Combines semantic understanding, structural analysis, and historical context
2. **Progressive Enhancement**: Works with minimal dependencies, scales with available resources
3. **Local-First Privacy**: Complete data sovereignty and security
4. **Configurable Ranking**: Every factor can be tuned for specific use cases
5. **Streaming Performance**: Results available as soon as possible
6. **Intelligent Caching**: Multiple cache levels with smart invalidation
7. **Extensible Design**: Plugin architecture for custom functionality

The future of code intelligence is local, intelligent, and developer-centric. Tenets embodies this vision while remaining practical and immediately useful for development teams of any size.
