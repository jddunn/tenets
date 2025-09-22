# Relevance Ranking System

## Unified Ranking Architecture

```mermaid
graph TD
    subgraph "Ranking Strategies"
        FAST[Fast Strategy<br/>Fastest<br/>Keyword + Path Only]
        BALANCED[Balanced Strategy<br/>1.5x slower<br/>BM25 + Structure]
        THOROUGH[Thorough Strategy<br/>4x slower<br/>Full Analysis + ML]
        ML_STRAT[ML Strategy<br/>5x slower<br/>Semantic Embeddings]
    end

    subgraph "Text Analysis (55% in Balanced)"
        KEY_MATCH[Keyword Matching<br/>20%<br/>Direct substring matching]
        BM25_SCORE[BM25 Score<br/>25%<br/>Statistical relevance preventing repetition bias]
        TFIDF_SIM[TF-IDF Similarity<br/>10%<br/>Term frequency analysis]
    end

    subgraph "Code Structure Analysis (25% in Balanced)"
        PATH_REL[Path Relevance<br/>15%<br/>Directory structure]
        IMP_CENT[Import Centrality<br/>10%<br/>Dependency importance]
    end

    subgraph "File Characteristics (10% in Balanced)"
        COMPLEXITY_REL[Complexity Relevance<br/>5%<br/>Code complexity signals]
        TYPE_REL[Type Relevance<br/>5%<br/>Extension/type matching]
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
    BALANCED --> BM25_SCORE
    BALANCED --> TFIDF_SIM
    THOROUGH --> IMP_CENT
    ML_STRAT --> SEM_SIM

    FILE_DISCOVERY --> ANALYSIS
    ANALYSIS --> RANKING
    RANKING --> AGGREGATION

    KEY_MATCH --> RANKING
    BM25_SCORE --> RANKING
    TFIDF_SIM --> RANKING
    PATH_REL --> RANKING
    IMP_CENT --> RANKING
    COMPLEXITY_REL --> RANKING
    TYPE_REL --> RANKING
    CODE_PAT --> RANKING
    GIT_REC --> RANKING
    GIT_FREQ --> RANKING

    SEM_SIM --> LOCAL_EMB
    LOCAL_EMB --> EMBED_CACHE
    EMBED_CACHE --> RANKING
```

## Strategy Comparison

| Strategy | Speed | Accuracy | Use Cases | Factors Used |
|----------|-------|----------|-----------|--------------|
| **Fast** | Fastest | Basic | Quick file discovery | Keyword (60%), Path (30%), File type (10%) |
| **Balanced** | 1.5x slower | Good | **DEFAULT** Production usage | BM25 (25%), Keyword (20%), Path (15%), TF-IDF (10%), Import (10%), Git (10%), Complexity (5%), Type (5%) |
| **Thorough** | 4x slower | High | Complex codebases | All balanced factors + enhanced analysis |
| **ML** | 5x slower | Highest | Semantic search | Embeddings (25%) + all thorough factors |

## Factor Calculation Details

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
