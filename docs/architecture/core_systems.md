---
title: Core Systems
description: Deep dive into Tenets core systems, algorithms, and processing pipelines
---

# Core Systems Architecture

## Overview

Tenets is built around a sophisticated, multi-layered architecture that transforms raw code into intelligent, contextual insights. At its heart, the system uses advanced ranking algorithms, comprehensive code analysis, and intelligent caching to provide developers with exactly the right context at exactly the right time.

## System Flow & Pipeline

The entire Tenets system follows a carefully orchestrated pipeline:

```mermaid
graph TB
    A[User Input] --> B[Prompt Parser]
    B --> C[Intent Detection]
    C --> D[File Discovery Engine]
    D --> E[Language-Specific Analysis]
    E --> F[Multi-Factor Ranking]
    F --> G[Token Optimization]
    G --> H[Context Assembly]
    H --> I[Output Formatting]

    subgraph "Analysis Pipeline"
        E1[Static Analysis]
        E2[Complexity Analysis]
        E3[Git History Analysis]
        E4[Dependency Analysis]
        E5[Pattern Recognition]
    end

    subgraph "Ranking Pipeline"
        F1[BM25 Scoring]
        F2[BM25 Relevance]
        F3[Semantic Similarity]
        F4[Git Signals]
        F5[File Characteristics]
        F6[Project Structure]
    end

    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E --> E5

    F --> F1
    F --> F2
    F --> F3
    F --> F4
    F --> F5
    F --> F6

    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fce4ec
```

## NLP/ML Pipeline Architecture

The NLP/ML pipeline powers Tenets' semantic understanding capabilities, enabling intelligent keyword extraction, similarity matching, and context-aware ranking.

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
        TFIDF_EXT[TF-IDF Extractor<br/>Alternative option]
        FREQ_EXT[Frequency Extractor<br/>Fallback]
    end

    subgraph "Embedding Generation"
        LOCAL_EMB[Local Embeddings<br/>sentence-transformers]
        MODEL_SEL[Model Selection<br/>MiniLM, MPNet]
        FALLBACK[BM25 Fallback<br/>No ML required]
    end

    subgraph "Similarity Computing"
        COSINE[Cosine Similarity]
        BATCH[Batch Processing]
        CACHE[Result Caching]
    end

    INPUT --> CODE_TOK
    INPUT --> TEXT_TOK
    PROMPT --> TEXT_TOK
    CODE --> CODE_TOK

    CODE_TOK --> YAKE_EXT
    TEXT_TOK --> YAKE_EXT
    YAKE_EXT --> TFIDF_EXT
    TFIDF_EXT --> FREQ_EXT

    FREQ_EXT --> LOCAL_EMB
    LOCAL_EMB --> MODEL_SEL
    MODEL_SEL --> FALLBACK

    FALLBACK --> COSINE
    COSINE --> BATCH
    BATCH --> CACHE

    style INPUT fill:#e1f5fe
    style CACHE fill:#e8f5e8
```

### Tokenization Strategy

Tenets uses specialized tokenizers for different content types:

#### Code Tokenizer
```python
class CodeTokenizer:
    """Handles programming language tokens."""

    def tokenize(self, text: str) -> List[str]:
        # Split on camelCase: 'getUserName' → ['get', 'User', 'Name']
        # Split on snake_case: 'get_user_name' → ['get', 'user', 'name']
        # Split on kebab-case: 'get-user-name' → ['get', 'user', 'name']
        # Preserve special tokens: '__init__', 'UTF-8'
        # Handle operators: '++', '==', '!='
```

#### Stopword Management
- **Code Stopwords**: Minimal set (~30 words) - 'function', 'class', 'return'
- **Prompt Stopwords**: Aggressive filtering (~200+ words) - common English words
- **Context-Aware**: Different stopword sets for different operations

### Embedding Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 90MB | Fastest | Good | Default, quick searches |
| **all-MiniLM-L12-v2** | 120MB | Fast | Better | Balanced performance |
| **all-mpnet-base-v2** | 420MB | Moderate | Best | Thorough mode |
| **multi-qa-MiniLM** | 90MB | Fast | Specialized | Q&A optimized tasks |

The system automatically selects models based on:
- Available memory
- Task complexity
- User configuration
- Performance requirements

## File Discovery & Scanning System

The file discovery system efficiently traverses codebases of any size, applying intelligent filtering and parallel processing.

### Scanner Architecture

```mermaid
graph TD
    subgraph "Entry Points"
        ROOT[Project Root]
        PATHS[Specified Paths]
        PATTERNS[Include Patterns]
    end

    subgraph "Ignore System Hierarchy"
        CLI[CLI Arguments<br/>Highest Priority]
        TENETS[.tenetsignore]
        GIT[.gitignore]
        GLOBAL[Global Ignores<br/>Lowest Priority]
    end

    subgraph "Detection Systems"
        BINARY[Binary Detection]
        SIZE[Size Check<br/>Max 10MB]
        CONTENT[Content Analysis]
    end

    subgraph "Parallel Processing"
        QUEUE[Work Queue]
        WORKERS[Thread Pool]
        PROGRESS[Progress Tracking]
    end

    ROOT --> CLI
    PATHS --> CLI
    PATTERNS --> CLI

    CLI --> TENETS
    TENETS --> GIT
    GIT --> GLOBAL

    GLOBAL --> BINARY
    BINARY --> SIZE
    SIZE --> CONTENT

    CONTENT --> QUEUE
    QUEUE --> WORKERS
    WORKERS --> PROGRESS

    style ROOT fill:#e1f5fe
    style PROGRESS fill:#e8f5e8
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
    RATIO --> THRESHOLD{>95%<br/>Printable?}
    THRESHOLD -->|Yes| TEXT[Mark as Text]
    THRESHOLD -->|No| BINARY
    TEXT --> ANALYZE[Ready for Analysis]
    BINARY --> IGNORE[Skip Analysis]
    SKIP --> IGNORE
```

## 1. Ranking System Deep Dive

The ranking system (`tenets.core.ranking`) is the intelligence core of Tenets, using multiple sophisticated algorithms to determine file relevance.

### Ranking Algorithms

#### Fast Algorithm (`FastRankingStrategy`)
- **Use Case**: Quick exploration, CI/CD, large codebases
- **Speed**: Sub-second on most projects (1.0x baseline)
- **Performance**: Baseline for all comparisons
- **Factors**: Keyword matching, path analysis, file size heuristics
- **Accuracy**: Good for obvious relevance matches

```python
# Fast algorithm priorities:
1. Exact keyword matches in filename/path
2. File extension relevance to prompt
3. Directory importance (src/ > tests/)
4. Recent modification time
5. File size (prefer moderate sizes)
```

#### Balanced Algorithm (`BalancedRankingStrategy`) - **Default**
- **Use Case**: Daily development, general context building
- **Speed**: 1-3 seconds on typical projects (~6.4x slower than Fast)
- **Performance**: 540% slower than Fast mode
- **Factors**: BM25, keywords, basic git signals, structural analysis
- **Accuracy**: High for most development tasks

```python
# Balanced algorithm combines:
- BM25 relevance scoring (35% weight)
- BM25 document ranking (30% weight)
- Git activity signals (15% weight)
- File structure analysis (15% weight)
```

#### Thorough Algorithm (`ThoroughRankingStrategy`)
- **Use Case**: Complex refactoring, architecture reviews, debugging
- **Speed**: 3-10 seconds depending on codebase (~13.3x slower than Fast, ~2.1x slower than Balanced)
- **Performance**: 1230% slower than Fast mode
- **Factors**: ML semantic similarity, dependency graphs, pattern analysis
- **Accuracy**: Highest, finds subtle relevance connections

```python
# Thorough algorithm includes:
- Semantic similarity via embeddings
- Complex dependency analysis
- Code pattern recognition
- Cross-reference analysis
- Advanced git history mining
```

### Multi-Factor Ranking Architecture

The ranking system combines multiple signals to determine file relevance:

```mermaid
graph TD
    subgraph "Ranking Strategies"
        FAST[Fast Strategy<br/>1.0x speed<br/>Keywords + Path]
        BALANCED[Balanced Strategy<br/>6.4x slower<br/>BM25 + Structure]
        THOROUGH[Thorough Strategy<br/>13.3x slower<br/>Deep Analysis + ML]
    end

    subgraph "Core Ranking Factors"
        SEM[Semantic Similarity<br/>25% weight]
        TEXT[Text Matching<br/>30% weight]
        STRUCT[Code Structure<br/>20% weight]
        GIT[Git Signals<br/>15% weight]
        FILE[File Characteristics<br/>10% weight]
    end

    subgraph "Scoring Engine"
        COMBINE[Weighted Combination]
        NORMALIZE[Score Normalization]
        FILTER[Threshold Filtering]
        RANK[Final Rankings]
    end

    FAST --> TEXT
    BALANCED --> STRUCT
    THOROUGH --> SEM

    SEM --> COMBINE
    TEXT --> COMBINE
    STRUCT --> COMBINE
    GIT --> COMBINE
    FILE --> COMBINE

    COMBINE --> NORMALIZE
    NORMALIZE --> FILTER
    FILTER --> RANK

    style FAST fill:#90caf9
    style BALANCED fill:#a5d6a7
    style THOROUGH fill:#ffab91
```

### Ranking Factors Explained

#### 1. **BM25 (Best Matching 25) - Primary Ranking Algorithm**
```python
class BM25Calculator:
    def score(self, query, document):
        """
        BM25 is the primary probabilistic ranking function used
        throughout Tenets for document relevance scoring.

        Key features:
        - Document length normalization (parameter b)
        - Term frequency saturation (parameter k1)
        - Significantly faster than alternatives
        - Industry standard in search engines
        """
```

**Use Case**: Primary ranking algorithm with 35% weight in balanced mode, handling varying document lengths and preventing term repetition over-weighting.

#### 2. **TF-IDF (Term Frequency-Inverse Document Frequency) - Optional**
```python
class TFIDFCalculator:
    def calculate_relevance(self, document, query_terms, corpus):
        """
        TF-IDF is available as an alternative ranking method
        for experimentation. Not recommended for production use.

        TF = (term frequency in doc) / (total terms in doc)
        IDF = log(total docs / docs containing term)
        TF-IDF = TF × IDF

        Note: BM25 provides superior performance and accuracy.
        """
```

**Status**: Available for experimentation but not recommended. Use BM25 for production workloads.

#### 3. **Git Activity Signals**
```python
class GitRankingFactor:
    def calculate_git_signals(self, file_path):
        return {
            'recent_commits': recent_commit_count,      # Files changed recently
            'commit_frequency': historical_changes,     # Frequently modified files
            'author_diversity': unique_contributors,    # Files many people work on
            'recency_score': days_since_last_change,   # How fresh is the file
            'blame_distribution': line_ownership,      # Code ownership patterns
        }
```

**Why This Matters**: Recently changed files are more likely relevant to current work. Files with many contributors often contain core logic.

#### 4. **Structural Analysis Factors**
```python
class StructuralRankingFactor:
    def analyze_structure(self, file_analysis):
        return {
            'complexity_score': cyclomatic_complexity,   # Code complexity
            'import_centrality': incoming_references,    # How many files import this
            'export_richness': outgoing_dependencies,    # What this file provides
            'directory_importance': path_significance,    # /src vs /tests importance
            'file_role': detected_file_type,            # Config, model, util, etc.
        }
```

### Ranking Strategy Selection

The system intelligently selects strategies based on context:

```python
def select_strategy(self, context: PromptContext, config: TenetsConfig) -> RankingStrategy:
    """
    Auto-select the best strategy based on:
    - Codebase size
    - Available dependencies
    - User preferences
    - Time constraints
    """
    if config.ranking.algorithm == "auto":
        if file_count > 10000:
            return FastRankingStrategy()
        elif has_ml_dependencies and context.complexity_level == "high":
            return ThoroughRankingStrategy()
        else:
            return BalancedRankingStrategy()
```

## 2. Analysis Engine Architecture

The analysis engine (`tenets.core.analysis`) provides deep code understanding through language-specific parsers and cross-language analysis.

### Language Analyzer System

#### Analyzer Architecture
```python
class LanguageAnalyzer(ABC):
    """Base class for all language-specific analyzers"""

    @abstractmethod
    def analyze_structure(self, content: str) -> CodeStructure:
        """Extract functions, classes, imports, etc."""

    @abstractmethod
    def calculate_complexity(self, content: str) -> ComplexityMetrics:
        """Calculate cyclomatic complexity, maintainability index"""

    @abstractmethod
    def extract_dependencies(self, content: str) -> List[ImportInfo]:
        """Find imports, requires, includes"""
```

#### Supported Languages & Features

| Language | Parser | Complexity | Dependencies | AST Analysis | Special Features |
|----------|--------|------------|--------------|--------------|------------------|
| **Python** | Full AST | ✅ CC, MI | ✅ imports, from | ✅ Full | Decorators, async/await |
| **JavaScript/TypeScript** | Full AST | ✅ CC, MI | ✅ import/require | ✅ Full | React components, Node.js |
| **Java** | Full AST | ✅ CC, MI | ✅ import, package | ✅ Full | Annotations, Spring |
| **C#** | Full AST | ✅ CC, MI | ✅ using, namespace | ✅ Full | Attributes, LINQ |
| **Go** | Full AST | ✅ CC, MI | ✅ import, package | ✅ Full | Goroutines, interfaces |
| **Rust** | Full AST | ✅ CC, MI | ✅ use, extern | ✅ Full | Traits, lifetimes |
| **C++** | Regex+ | ✅ Basic | ✅ #include | Partial | Headers, namespaces |
| **Ruby** | Full AST | ✅ CC, MI | ✅ require, gem | ✅ Full | Gems, Rails detection |
| **PHP** | Full AST | ✅ CC, MI | ✅ require, use | ✅ Full | Composer, namespaces |
| **Kotlin** | Full AST | ✅ CC, MI | ✅ import, package | ✅ Full | Coroutines, DSLs |

*CC = Cyclomatic Complexity, MI = Maintainability Index*

### Complexity Analysis Deep Dive

#### Cyclomatic Complexity
```python
def calculate_cyclomatic_complexity(self, node):
    """
    Measures the number of linearly independent paths through code.

    Decision points that increase complexity:
    - if/elif statements
    - for/while loops
    - try/catch blocks
    - switch/case statements
    - logical operators (&&, ||)
    - ternary operators
    """
    complexity = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity
```

**Complexity Levels:**
- 1-10: Simple, low risk
- 11-20: Moderate complexity
- 21-50: High complexity, refactor recommended
- 50+: Very high risk, immediate attention needed

#### Maintainability Index
```python
def calculate_maintainability_index(self, metrics):
    """
    MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)

    Where:
    - HV = Halstead Volume (operators + operands)
    - CC = Cyclomatic Complexity
    - LOC = Lines of Code

    Scale: 0-100 (higher is more maintainable)
    """
```

**MI Interpretation:**
- 85-100: Excellent maintainability
- 70-85: Good maintainability
- 50-70: Moderate maintainability
- 25-50: Below average, needs attention
- 0-25: Difficult to maintain

### Ranking Factor Details

| Factor Category | Weight | Components | Description |
|-----------------|--------|------------|-------------|
| **Semantic Similarity** | 25% | Embedding cosine similarity, contextual relevance | ML-based understanding of semantic meaning |
| **Text Matching** | 35% | Keyword matches (20%), BM25 (15%) | Direct term matching and probabilistic relevance |
| **Code Structure** | 20% | Import centrality (10%), path relevance (10%) | File importance in codebase architecture |
| **Git Signals** | 15% | Recency (5%), frequency (5%), authors (5%) | Version control activity indicators |
| **File Characteristics** | 10% | File type (5%), code patterns (5%) | Language and pattern-based relevance |

### Dependency Analysis

#### Import Graph Construction
```python
class DependencyAnalyzer:
    def build_dependency_graph(self, project_files):
        """
        Builds a directed graph of file dependencies:

        1. Extract all imports/includes from each file
        2. Resolve import paths to actual files
        3. Create edges between dependent files
        4. Calculate centrality metrics
        5. Identify circular dependencies
        """

        graph = nx.DiGraph()

        for file in project_files:
            imports = self.extract_imports(file)
            for imp in imports:
                target = self.resolve_import_path(imp, file)
                if target:
                    graph.add_edge(file.path, target.path)

        return self.calculate_centrality_metrics(graph)
```

#### Dependency Metrics

The system calculates various centrality metrics to understand file importance:

```mermaid
graph LR
    subgraph "Centrality Metrics"
        IN[In-Degree<br/>Files importing this]
        OUT[Out-Degree<br/>Files imported]
        BETWEEN[Betweenness<br/>Bridge importance]
        PAGE[PageRank<br/>Link analysis]
    end

    subgraph "Analysis"
        CRITICAL[Critical Files<br/>High in-degree]
        HUBS[Hub Files<br/>High out-degree]
        BRIDGES[Bridge Files<br/>High betweenness]
        CENTRAL[Central Files<br/>High PageRank]
    end

    IN --> CRITICAL
    OUT --> HUBS
    BETWEEN --> BRIDGES
    PAGE --> CENTRAL

    style CRITICAL fill:#ffcdd2
    style CENTRAL fill:#c5e1a5
```

- **In-Degree**: Number of files that import this file (dependency count)
- **Out-Degree**: Number of files this file imports (dependency scope)
- **Betweenness**: How often this file appears in shortest paths between other files
- **PageRank**: Importance based on the importance of files that import it
- **Clustering Coefficient**: How connected this file's dependencies are to each other

## 3. Summarization System

When files exceed token limits, Tenets uses intelligent summarization to preserve the most important content.

### Summarization Strategies

#### 1. **Structural Preservation**
```python
class StructuralSummarizer:
    def summarize(self, content, target_size):
        """
        Preserves code structure while reducing content:

        Priority Order:
        1. Function/class signatures (always keep)
        2. Public API definitions
        3. Complex logic blocks
        4. Important comments/docstrings
        5. Configuration/constants
        6. Simple variable assignments (truncate)
        7. Repetitive code (remove)
        """
```

#### 2. **Semantic Summarization** (ML mode)
```python
class SemanticSummarizer:
    def summarize(self, content, context, target_size):
        """
        Uses ML to understand semantic importance:

        1. Split content into logical segments
        2. Generate embeddings for each segment
        3. Calculate relevance to user prompt
        4. Rank segments by importance
        5. Select top segments that fit token budget
        6. Maintain code coherence
        """
```

#### 3. **Contextual Summarization**
```python
class ContextualSummarizer:
    def summarize_for_prompt(self, file_content, prompt_context):
        """
        Tailors summarization to specific prompt needs:

        - "debug bug": Keep error handling, logging
        - "add feature": Keep interfaces, extension points
        - "refactor": Keep complex logic, dependencies
        - "security": Keep authentication, validation
        """
```

## 4. Context Optimization & Token Management

Tenets intelligently manages token budgets to maximize relevant context while staying within model limits.

### Token Budget Allocation

```mermaid
graph TB
    subgraph "Token Budget"
        TOTAL[Total Token Limit]
        PROMPT[Prompt Reservation<br/>~2K tokens]
        RESPONSE[Response Buffer<br/>~4K tokens]
        CONTEXT[Available for Context]
    end

    subgraph "Optimization Strategy"
        RANK[Rank Files]
        SELECT[Select Top N]
        MEASURE[Measure Tokens]
        SUMMARIZE[Summarize if Needed]
    end

    subgraph "Content Priority"
        CRITICAL[Critical Files<br/>Always included]
        HIGH[High Relevance<br/>Full content]
        MEDIUM[Medium Relevance<br/>Summarized]
        LOW[Low Relevance<br/>Signatures only]
    end

    TOTAL --> PROMPT
    TOTAL --> RESPONSE
    TOTAL --> CONTEXT

    CONTEXT --> RANK
    RANK --> SELECT
    SELECT --> MEASURE
    MEASURE --> SUMMARIZE

    SUMMARIZE --> CRITICAL
    SUMMARIZE --> HIGH
    SUMMARIZE --> MEDIUM
    SUMMARIZE --> LOW

    style TOTAL fill:#e1f5fe
    style CRITICAL fill:#ffcdd2
```

## 5. Caching & Performance Architecture

Tenets uses sophisticated multi-level caching to provide instant responses after initial analysis.

### Cache Hierarchy

```mermaid
graph TB
    A[Request] --> B[Memory Cache]
    B -->|Miss| C[SQLite Session DB]
    C -->|Miss| D[File System Cache]
    D -->|Miss| E[Analysis Pipeline]

    E --> F[Store in File Cache]
    F --> G[Store in SQLite]
    G --> H[Store in Memory]
    H --> I[Return Results]

    subgraph "Cache Types"
        J[Hot Data - Memory]
        K[Structured Data - SQLite]
        L[Analysis Results - Disk]
        M[Embeddings - Specialized]
    end

    style B fill:#ffeb3b
    style C fill:#4caf50
    style D fill:#2196f3
    style E fill:#ff5722
```

### Cache Invalidation Strategy

```python
class CacheManager:
    def should_invalidate(self, file_path, cache_entry):
        """
        Smart cache invalidation based on:

        1. File modification time (mtime)
        2. Content hash changes (for accuracy)
        3. Git commit changes (for git-based signals)
        4. Configuration changes (for ranking params)
        5. Dependency changes (for import analysis)
        """

        reasons = []

        if file_path.stat().st_mtime > cache_entry.mtime:
            reasons.append("file_modified")

        if self.calculate_content_hash(file_path) != cache_entry.content_hash:
            reasons.append("content_changed")

        if self.get_git_commit_hash() != cache_entry.git_hash:
            reasons.append("git_changed")

        return len(reasons) > 0, reasons
```

### Prompt Parsing & Understanding

```mermaid
graph LR
    subgraph "Prompt Analysis"
        INPUT[User Prompt]
        INTENT[Intent Detection]
        KEYWORDS[Keyword Extraction]
        CONTEXT[Context Building]
    end

    subgraph "Intent Types"
        DEBUG[Debug/Fix<br/>Error handling focus]
        FEATURE[Feature/Add<br/>New functionality]
        REFACTOR[Refactor/Optimize<br/>Code improvement]
        EXPLORE[Explore/Understand<br/>Learning focus]
    end

    subgraph "Extraction Methods"
        YAKE[YAKE Algorithm<br/>Statistical extraction]
        BM25[BM25<br/>Probabilistic relevance]
        NER[Named Entity<br/>Code entities]
    end

    INPUT --> INTENT
    INTENT --> KEYWORDS
    KEYWORDS --> CONTEXT

    INTENT --> DEBUG
    INTENT --> FEATURE
    INTENT --> REFACTOR
    INTENT --> EXPLORE

    KEYWORDS --> YAKE
    KEYWORDS --> BM25
    KEYWORDS --> NER
```

### Performance Optimizations

#### Parallel Processing
```python
def analyze_files_parallel(self, files, max_workers=None):
    """
    Parallel analysis with intelligent work distribution:

    - Small files: Batch process in single thread
    - Large files: Individual threads
    - I/O bound: Higher thread count
    - CPU bound: Thread count = CPU cores
    """

    with ThreadPoolExecutor(max_workers=self.optimal_worker_count()) as executor:
        future_to_file = {
            executor.submit(self.analyze_file, f): f
            for f in files
        }
```

#### Memory Management
```python
class MemoryManager:
    def manage_analysis_memory(self):
        """
        Prevents memory bloat during large project analysis:

        1. Stream file processing (don't load all at once)
        2. Release analysis objects after ranking
        3. Use generators for large result sets
        4. Monitor memory usage and trigger cleanup
        """
```

## 5. Session Management & State

Sessions provide persistent context and configuration for project work.

### Session Architecture
```python
class SessionManager:
    def create_session(self, name, project_path):
        """
        Session contains:
        - Pinned files (guaranteed inclusion)
        - Custom ranking weights
        - Project-specific configuration
        - Analysis cache keys
        - Tenet definitions (guiding principles)
        """

    def merge_contexts(self, session_name, new_context):
        """
        Intelligently merge new analysis with existing session:

        1. Preserve pinned files
        2. Update file rankings
        3. Maintain tenet priorities
        4. Merge analysis results
        5. Update cache references
        """
```

### Tenets System
```python
class TenetManager:
    """
    Manages 'tenets' - persistent principles that guide context generation

    Examples:
    - "Always include error handling examples"
    - "Prioritize async/await patterns"
    - "Include security considerations"
    - "Focus on performance implications"
    """

    def apply_tenets(self, context_result, active_tenets):
        """
        Inject tenets into generated context:

        1. Analyze context for tenet relevance
        2. Select most applicable tenets
        3. Format tenets appropriately
        4. Insert at strategic positions
        5. Ensure token budget compliance
        """
```

## Integration & Extension Points

### Custom Analyzers
```python
class MyCustomAnalyzer(LanguageAnalyzer):
    def analyze_structure(self, content: str) -> CodeStructure:
        # Implement custom parsing logic
        pass

    def get_language_patterns(self) -> Dict[str, str]:
        return {
            'function_def': r'def\s+(\w+)\s*\(',
            'class_def': r'class\s+(\w+)',
            'import': r'import\s+(.+)',
        }
```

### Custom Ranking Factors
```python
class MyRankingFactor(RankingFactor):
    def calculate(self, file_analysis, prompt_context):
        """
        Add custom ranking logic:
        - Domain-specific patterns
        - Company coding standards
        - Architecture preferences
        - Performance considerations
        """
        return relevance_score
```

### Plugin System
```python
class TenetsPlugin:
    def register_analyzer(self, language: str, analyzer_class):
        """Register custom language analyzer"""

    def register_ranking_factor(self, factor_name: str, factor_class):
        """Register custom ranking factor"""

    def register_output_formatter(self, format_name: str, formatter_class):
        """Register custom output format"""
```

## Performance Characteristics

### Typical Performance (on modern hardware)

| Operation | Small Project (<1K files) | Medium Project (1K-10K files) | Large Project (10K+ files) |
|-----------|---------------------------|-------------------------------|----------------------------|
| **Fast Mode** | <1s (1.0x baseline) | 1-3s | 3-8s |
| **Balanced Mode** | 1-2s (~6.4x slower) | 3-8s | 10-30s |
| **Thorough Mode** | 2-5s (~13.3x slower) | 8-20s | 30s-2m |
| **Cache Hit** | <100ms | <200ms | <500ms |

### Relative Performance

| Mode | Speed Multiplier | Percentage Slower |
|------|-----------------|-------------------|
| **Fast** | 1.0x | 0% (baseline) |
| **Balanced** | 6.4x | 540% |
| **Thorough** | 13.3x | 1230% |

### Memory Usage

```python
# Typical memory footprint:
base_memory = 50_000_000      # ~50MB base
per_file_overhead = 5_000     # ~5KB per analyzed file
embedding_cache = 200_000_000 # ~200MB for ML embeddings (when used)
```

### Scalability Limits

- **Files**: Tested up to 100K+ files
- **File Size**: Handles up to 10MB individual files
- **Concurrency**: Scales to available CPU cores
- **Memory**: Degrades gracefully with limited RAM

## Language Analyzer Architecture

Tenets provides specialized analyzers for each programming language, extracting structure, complexity, and relationships.

### Analyzer System Overview

```mermaid
graph TB
    subgraph "Base Analyzer Interface"
        BASE[LanguageAnalyzer<br/>Abstract Base]
        EXTRACT[extract_structure()]
        COMPLEX[calculate_complexity()]
        DEPS[trace_dependencies()]
    end

    subgraph "Language-Specific Analyzers"
        PYTHON[Python<br/>Full AST]
        JS[JavaScript<br/>ES6+ support]
        GO[Go<br/>Package detection]
        JAVA[Java<br/>OOP patterns]
        RUST[Rust<br/>Ownership]
        GENERIC[Generic<br/>Pattern-based]
    end

    subgraph "Analysis Output"
        CLASSES[Classes & Methods]
        FUNCTIONS[Functions & Signatures]
        IMPORTS[Import Graph]
        METRICS[Complexity Metrics]
    end

    BASE --> EXTRACT
    BASE --> COMPLEX
    BASE --> DEPS

    BASE --> PYTHON
    BASE --> JS
    BASE --> GO
    BASE --> JAVA
    BASE --> RUST
    BASE --> GENERIC

    PYTHON --> CLASSES
    PYTHON --> FUNCTIONS
    PYTHON --> IMPORTS
    PYTHON --> METRICS

    style BASE fill:#e1f5fe
    style METRICS fill:#e8f5e8
```

### Python Analyzer Deep Dive

```mermaid
graph LR
    subgraph "AST Analysis"
        PARSE[AST Parser]
        VISIT[Node Visitor]
        TABLE[Symbol Table]
    end

    subgraph "Structure Extraction"
        CLASSES[Classes<br/>Inheritance]
        FUNCTIONS[Functions<br/>Async/Sync]
        DECORATORS[Decorators]
        TYPES[Type Hints]
    end

    subgraph "Import Analysis"
        ABS[Absolute Imports]
        REL[Relative Imports]
        GRAPH[Import Graph]
    end

    subgraph "Complexity"
        CYCLO[Cyclomatic]
        COGNITIVE[Cognitive]
        HALSTEAD[Halstead]
    end

    PARSE --> VISIT
    VISIT --> TABLE

    TABLE --> CLASSES
    TABLE --> FUNCTIONS
    TABLE --> DECORATORS
    TABLE --> TYPES

    VISIT --> ABS
    VISIT --> REL
    ABS --> GRAPH
    REL --> GRAPH

    TABLE --> CYCLO
    TABLE --> COGNITIVE
    TABLE --> HALSTEAD
```

### Language Support Matrix

| Language | AST Support | Complexity Analysis | Import Resolution | Special Features |
|----------|------------|-------------------|-------------------|------------------|
| **Python** | ✅ Full | ✅ CC, MI, Halstead | ✅ Complete | Decorators, async/await, type hints |
| **JavaScript/TypeScript** | ✅ Full | ✅ CC, MI | ✅ ES6+ modules | React components, JSX, Node.js |
| **Java** | ✅ Full | ✅ CC, MI | ✅ Package/class | Annotations, Spring framework |
| **C#** | ✅ Full | ✅ CC, MI | ✅ Namespace/using | Attributes, LINQ, async |
| **Go** | ✅ Full | ✅ CC, MI | ✅ Package/import | Goroutines, interfaces |
| **Rust** | ✅ Full | ✅ CC, MI | ✅ Use/extern | Traits, lifetimes, macros |
| **Ruby** | ✅ Full | ✅ CC, MI | ✅ Require/gem | Gems, Rails detection |
| **PHP** | ✅ Full | ✅ CC, MI | ✅ Require/use | Composer, namespaces |
| **C/C++** | 🟨 Regex+ | ✅ Basic | ✅ #include | Headers, templates |
| **Others** | 🟨 Pattern | 🟨 Basic | 🟨 Pattern-based | Extensible patterns |

*CC = Cyclomatic Complexity, MI = Maintainability Index*

This architecture provides the foundation for intelligent, fast, and scalable code intelligence that adapts to any project size and developer workflow.
