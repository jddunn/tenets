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
9. [Context Management & Optimization](#context-management--optimization)
10. [Session Management Architecture](#session-management-architecture)
11. [Storage & Caching Architecture](#storage--caching-architecture)
12. [Performance Architecture](#performance-architecture)
13. [Configuration System](#configuration-system)
14. [CLI & API Architecture](#cli--api-architecture)
15. [Security & Privacy Architecture](#security--privacy-architecture)
16. [Testing & Quality Assurance](#testing--quality-assurance)
17. [Guiding Principles (Tenets) System](#guiding-principles-tenets-system)
18. [Output Generation & Visualization](#output-generation--visualization)
19. [Future Roadmap & Vision](#future-roadmap--vision)

## System Overview

Tenets is a sophisticated, local-first code intelligence platform that revolutionizes how developers interact with their codebases when working with AI assistants. Unlike traditional code search tools, Tenets employs advanced multi-stage analysis combining NLP, ML, static code analysis, git history mining, and intelligent ranking to build optimal context.

### Core Architecture Principles

1. **Local-First Processing**: All analysis happens on the developer's machine. No code leaves the local environment. External API calls only for optional LLM-based summarization with explicit consent.

2. **Progressive Enhancement**: Provides value immediately with just Python installed, scales with optional dependencies. Core functionality works without ML libraries, git integration works without configuration.

3. **Intelligent Caching**: Every expensive operation is cached at multiple levels - memory caches for hot data, SQLite for structured data, disk caches for analysis results, specialized caches for embeddings.

4. **Configurable Intelligence**: Every aspect of ranking and analysis can be configured. Users can adjust factor weights, enable/disable features, add custom ranking functions.

5. **Streaming Architecture**: Uses streaming and incremental processing wherever possible. Files analyzed as discovered, rankings computed in parallel, results stream to user.

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
            RAKE[RAKE Keywords]
            YAKE[YAKE Fallback]
            TFIDF[BM25/TF-IDF Analysis]
            BM25[BM25 Ranking]
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
            BM25_SIM[BM25 Similarity<br/>15%]
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
    KEYWORDS --> RAKE
    RAKE --> YAKE
    ENTITIES --> TFIDF
    ENTITIES --> BM25

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
    THOROUGH --> BM25_SIM
    ML --> IMPORT_CENT

    SEMANTIC --> EMBEDDINGS
    EMBEDDINGS --> EMBED_CACHE
    EMBED_CACHE --> SIMILARITY

    SIMILARITY --> CONTEXT_BUILDER
    KEYWORD_MATCH --> CONTEXT_BUILDER
    BM25_SIM --> CONTEXT_BUILDER

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

### Centralized NLP Components

```
tenets/core/nlp/
├── __init__.py          # Main NLP API exports
├── similarity.py        # Centralized similarity computations
├── keyword_extractor.py # Unified keyword extraction with SimpleRAKE
├── tokenizer.py        # Code and text tokenization
├── stopwords.py        # Stopword management with fallbacks
├── embeddings.py       # Embedding generation (ML optional)
├── ml_utils.py         # ML utility functions
├── bm25.py            # BM25 ranking algorithm (primary)
└── tfidf.py           # TF-IDF calculations (optional alternative)
```

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
        RAKE_EXT[RAKE Extractor<br/>Primary - Fast & Python 3.13 Compatible]
        YAKE_EXT[YAKE Extractor<br/>Secondary - Python < 3.13 Only]
        TFIDF_EXT[BM25/TF-IDF Extractor<br/>Frequency-based Fallback]
        FREQ_EXT[Frequency Extractor<br/>Final Fallback]
    end

    subgraph "Stopword Management"
        CODE_STOP[Code Stopwords<br/>Minimal - 30 words]
        PROMPT_STOP[Prompt Stopwords<br/>Aggressive - 200+ words]
    end

    subgraph "Embedding Generation"
        LOCAL_EMB[Local Embeddings<br/>sentence-transformers]
        MODEL_SEL[Model Selection<br/>MiniLM, MPNet]
        FALLBACK[BM25 Fallback<br/>No ML required]
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

    CODE_STOP --> RAKE_EXT
    PROMPT_STOP --> RAKE_EXT
    RAKE_EXT --> YAKE_EXT
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

### Keyword Extraction Algorithms Comparison

| Algorithm | Speed | Quality | Memory | Python 3.13 | Best For | Limitations |
|-----------|-------|----------|---------|-------------|----------|-------------|
| **RAKE** | Fast | Good | Low | ✅ Yes | Technical docs, Multi-word phrases | No semantic understanding |
| **SimpleRAKE** | Fast | Good | Minimal | ✅ Yes | No NLTK dependencies, Built-in | Basic tokenization only |
| **YAKE** | Moderate | Very Good | Low | ❌ No | Statistical analysis, Capital aware | Python 3.13 bug |
| **BM25** | Fast | Excellent | High | ✅ Yes | Primary ranking, Length variation | Needs corpus |
| **TF-IDF** | Fast | Good | Medium | ✅ Yes | Alternative to BM25 | Less effective for varying lengths |
| **Frequency** | Very Fast | Basic | Minimal | ✅ Yes | Fallback option | Very basic |

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

    subgraph "Minified & Build File Exclusion"
        MINIFIED_CHECK[Minified Detection<br/>*.min.js, *.bundle.js]
        BUILD_DIRS[Build Directories<br/>dist/, build/, out/]
        PROD_FILES[Production Files<br/>*.prod.js, *.compiled.js]
        NODE_MODULES[Dependencies<br/>node_modules/, vendor/]
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

    PATTERN_MATCH --> PY_PATTERNS
    PATTERN_MATCH --> JS_PATTERNS
    PATTERN_MATCH --> JAVA_PATTERNS
    PATTERN_MATCH --> GO_PATTERNS
    PATTERN_MATCH --> GENERIC_PATTERNS

    PY_PATTERNS --> FILTERED_FILES[Filtered File List]
    JS_PATTERNS --> FILTERED_FILES
    JAVA_PATTERNS --> FILTERED_FILES
    GO_PATTERNS --> FILTERED_FILES
    GENERIC_PATTERNS --> FILTERED_FILES

    SCAN_ALL --> ANALYSIS[File Analysis]
    FILTERED_FILES --> ANALYSIS
```

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

### Unified Ranking Architecture

```mermaid
graph TD
    subgraph "Ranking Strategies"
        FAST[Fast Strategy<br/>Fastest<br/>Keyword + Path Only]
        BALANCED[Balanced Strategy<br/>1.5x slower<br/>BM25 + Structure]
        THOROUGH[Thorough Strategy<br/>4x slower<br/>Full Analysis + ML]
        ML_STRAT[ML Strategy<br/>5x slower<br/>Semantic Embeddings]
    end

    subgraph "Text Analysis (40% in Balanced)"
        KEY_MATCH[Keyword Matching<br/>20%<br/>Direct term hits]
        BM25_SIM[BM25 Similarity<br/>20%<br/>Statistical relevance]
        BM25_SCORE[BM25 Score<br/>15%<br/>Probabilistic ranking]
    end

    subgraph "Code Structure Analysis (25% in Balanced)"
        PATH_REL[Path Relevance<br/>15%<br/>Directory structure]
        IMP_CENT[Import Centrality<br/>10%<br/>Dependency importance]
    end

    subgraph "File Characteristics (15% in Balanced)"
        COMPLEXITY_REL[Complexity Relevance<br/>5%<br/>Code complexity signals]
        FILE_TYPE[File Type Relevance<br/>5%<br/>Extension/type matching]
        CODE_PAT[Code Patterns<br/>5%<br/>AST pattern matching]
    end

    subgraph "Git Signals (10% in Balanced)"
        GIT_REC[Git Recency<br/>5%<br/>Recent changes]
        GIT_FREQ[Git Frequency<br/>5%<br/>Change frequency]
    end

    subgraph "ML Enhancement (Only in ML Strategy)"
        SEM_SIM[Semantic Similarity<br/>25%<br/>Embedding-based understanding]
        LOCAL_EMB[Local Embeddings<br/>sentence-transformers]
        EMBED_CACHE[Embedding Cache<br/>Performance optimization]
    end

    subgraph "Unified Pipeline"
        FILE_DISCOVERY[File Discovery<br/>Scanner + Filters]
        ANALYSIS[Code Analysis<br/>AST + Structure]
        RANKING[Multi-Factor Ranking<br/>Strategy-specific weights]
        AGGREGATION[Context Aggregation<br/>Token optimization]
    end

    FAST --> KEY_MATCH
    BALANCED --> BM25_SIM
    BALANCED --> BM25_SCORE
    THOROUGH --> IMP_CENT
    ML_STRAT --> SEM_SIM

    FILE_DISCOVERY --> ANALYSIS
    ANALYSIS --> RANKING
    RANKING --> AGGREGATION

    KEY_MATCH --> RANKING
    BM25_SIM --> RANKING
    BM25_SCORE --> RANKING
    PATH_REL --> RANKING
    IMP_CENT --> RANKING
    COMPLEXITY_REL --> RANKING
    FILE_TYPE --> RANKING
    CODE_PAT --> RANKING
    GIT_REC --> RANKING
    GIT_FREQ --> RANKING

    SEM_SIM --> LOCAL_EMB
    LOCAL_EMB --> EMBED_CACHE
    EMBED_CACHE --> RANKING
```

### Strategy Comparison

| Strategy | Speed | Accuracy | Use Cases | Factors Used |
|----------|-------|----------|-----------|--------------|
| **Fast** | Fastest | Basic | Quick file discovery | Keyword (60%), Path (30%), File type (10%) |
| **Balanced** | 1.5x slower | Good | **DEFAULT** Production usage | Keyword (20%), BM25 (35%), Structure (25%), Git (10%) |
| **Thorough** | 4x slower | High | Complex codebases | All balanced factors + enhanced analysis |
| **ML** | 5x slower | Highest | Semantic search | Embeddings (25%) + all thorough factors |

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
        IMPORTS_EX[Import Summarization<br/>Condenses when > threshold]
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
        SCANNER_CONFIG[Scanner Configuration<br/>Ignore patterns, limits<br/>Minified exclusion]
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
    bm25_similarity: 0.15
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
  keyword_extractor: rake # rake|yake|bm25|tfidf|frequency
  text_similarity: bm25   # bm25|tfidf

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
  exclude_minified: true
  exclude_tests: auto

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
  summarize_imports: true
  import_summary_threshold: 5
  docs_context_aware: true
  docs_show_in_place_context: true
  docs_context_search_depth: 2
  docs_context_min_confidence: 0.6
  docs_context_max_sections: 10
  docs_context_preserve_examples: true

# Output configuration
output:
  format: markdown     # markdown|json|xml|html
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

# Tenet configuration
tenet:
  auto_instill: true
  injection_frequency: adaptive
  max_per_context: 5
  system_instruction_enabled: true
```

## CLI & API Architecture

### Command Structure

```yaml
# Main Commands
tenets:
  distill:           # Build optimal context for prompts
    --copy           # Copy to clipboard
    --format         # Output format (markdown, xml, json, html)
    --max-tokens     # Token limit
    --exclude        # Exclude patterns
    --session        # Session name
    --algorithm      # Ranking algorithm

  examine:           # Code quality analysis
    --show-details   # Detailed metrics
    --hotspots       # Show maintenance hotspots
    --ownership      # Show code ownership
    --format         # Output format

  chronicle:         # Git history analysis
    --since          # Time range
    --author         # Filter by author
    --format         # Output format

  momentum:          # Velocity tracking (WIP)
    --team           # Team metrics
    --detailed       # Detailed breakdown

  session:           # Session management
    create           # Create new session
    list             # List sessions
    delete           # Delete session

  tenet:            # Manage guiding principles
    add             # Add new tenet
    list            # List tenets
    remove          # Remove tenet

  instill:          # Apply tenets and system instructions
    --dry-run       # Preview what would be applied
    --force         # Force application

  system-instruction: # Manage system instructions
    set             # Set instruction
    get             # Get current
    enable/disable  # Toggle
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

    subgraph "Output Sanitization (Roadmap)"
        REDACT[Redact Secrets<br/>**WIP** - Coming soon]
        MASK_PII[Mask PII<br/>**WIP** - Planned feature]
        CLEAN_PATHS[Clean File Paths<br/>Remove sensitive paths]
        REMOVE_URLS[Remove Internal URLs<br/>**WIP** - Under development]
        ANONYMIZE[Anonymization<br/>**WIP** - Future release]
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

### Secret Detection Patterns (Roadmap)

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
        UNIT[Unit Tests<br/>Target: >90% coverage<br/>Fast, isolated]
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

## Guiding Principles (Tenets) System

### Overview

The Guiding Principles system provides persistent, context-aware instructions to maintain consistency across AI interactions and combat context drift.

### Injection Strategy

```mermaid
graph TD
    subgraph "Injection Decision Engine"
        ANALYZER[Content Analyzer<br/>Structure & complexity]
        STRATEGY[Strategy Selector<br/>Top, distributed, contextual]
        INJECTOR[Smart Injector<br/>Natural break detection]
    end

    subgraph "Priority System"
        CRITICAL[Critical Principles<br/>Security, data integrity]
        HIGH[High Priority<br/>Architecture, performance]
        MEDIUM[Medium Priority<br/>Style, conventions]
        LOW[Low Priority<br/>Preferences, suggestions]
    end

    subgraph "Reinforcement"
        TOP_INJECTION[Top of Context<br/>Most visible]
        DISTRIBUTED[Throughout Content<br/>Natural sections]
        END_SUMMARY[End Reinforcement<br/>Key reminders]
    end

    ANALYZER --> STRATEGY
    STRATEGY --> INJECTOR

    CRITICAL --> TOP_INJECTION
    HIGH --> DISTRIBUTED
    MEDIUM --> DISTRIBUTED
    LOW --> END_SUMMARY
```

### Output Formats

**Markdown:**
```markdown
**🎯 Key Guiding Principle:** Always validate user input before processing
**📌 Important Guiding Principle:** Use async/await for all I/O operations
**💡 Guiding Principle:** Prefer composition over inheritance
```

**XML (Recommended by OpenAI):**
```xml
<guiding_principle priority="high" category="security">
  Always validate and sanitize user input
</guiding_principle>

<guiding_principles>
  <guiding_principle priority="critical">Maintain backward compatibility</guiding_principle>
  <guiding_principle priority="medium">Use descriptive variable names</guiding_principle>
</guiding_principles>
```

### Configuration

```yaml
tenet:
  auto_instill: true
  max_per_context: 5
  injection_strategy: strategic
  injection_frequency: adaptive  # 'always', 'periodic', 'adaptive', 'manual'
  injection_interval: 3          # For periodic mode
  min_session_length: 1          # First injection always happens
  system_instruction: "Prefer small, safe diffs and add tests"
  system_instruction_enabled: true
```

## Output Generation & Visualization

### Output Formatting System

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

### Visualization Components

```mermaid
graph LR
    subgraph "Project Detection"
        DETECTOR[Project Detector<br/>Auto-detects type]
        LANGUAGES[Language Analysis<br/>% distribution]
        FRAMEWORKS[Framework Detection<br/>Django, React, etc]
        ENTRYPOINTS[Entry Points<br/>main.py, index.js]
    end

    subgraph "Graph Generation"
        GRAPHGEN[Graph Generator<br/>Multiple formats]
        NETWORKX[NetworkX<br/>Graph algorithms]
        GRAPHVIZ[Graphviz<br/>DOT rendering]
        PLOTLY[Plotly<br/>Interactive HTML]
        D3JS[D3.js<br/>Web visualization]
    end

    subgraph "Dependency Visualization"
        FILE_DEPS[File-level<br/>Individual files]
        MODULE_DEPS[Module-level<br/>Aggregated modules]
        PACKAGE_DEPS[Package-level<br/>Top-level packages]
        CLUSTERING[Clustering<br/>Group by criteria]
    end

    subgraph "Output Formats"
        ASCII[ASCII Tree<br/>Terminal output]
        SVG[SVG<br/>Vector graphics]
        PNG[PNG/PDF<br/>Static images]
        HTML_INT[Interactive HTML<br/>D3.js/Plotly]
        DOT[DOT Format<br/>Graphviz source]
        JSON_OUT[JSON<br/>Raw data]
    end

    subgraph "Layout Algorithms"
        HIERARCHICAL[Hierarchical<br/>Tree layout]
        CIRCULAR[Circular<br/>Radial layout]
        SHELL[Shell<br/>Concentric circles]
        KAMADA[Kamada-Kawai<br/>Force-directed]
    end

    DETECTOR --> LANGUAGES
    DETECTOR --> FRAMEWORKS
    DETECTOR --> ENTRYPOINTS

    GRAPHGEN --> NETWORKX
    GRAPHGEN --> GRAPHVIZ
    GRAPHGEN --> PLOTLY
    GRAPHGEN --> D3JS

    FILE_DEPS --> MODULE_DEPS
    MODULE_DEPS --> PACKAGE_DEPS
    PACKAGE_DEPS --> CLUSTERING

    GRAPHGEN --> ASCII
    GRAPHGEN --> SVG
    GRAPHGEN --> PNG
    GRAPHGEN --> HTML_INT
    GRAPHGEN --> DOT
    GRAPHGEN --> JSON_OUT
```

## Future Roadmap & Vision

### Near Term

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

### Medium Term

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

### Long Term Vision

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

## Conclusion

Tenets combines sophisticated NLP/ML techniques with traditional code analysis, git mining, and intelligent caching to create a system that truly understands code in context. The architecture is designed for scalability, extensibility, and performance while maintaining complete privacy through local-first processing.

The future of code intelligence is local, intelligent, and developer-centric. Tenets embodies this vision while remaining practical and immediately useful for development teams of any size.