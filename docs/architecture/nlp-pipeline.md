# NLP/ML Pipeline Architecture

## Centralized NLP Components

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

## Pipeline Component Flow

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

## Keyword Extraction Algorithms Comparison

| Algorithm | Speed | Quality | Memory | Python 3.13 | Best For | Limitations |
|-----------|-------|----------|---------|-------------|----------|-------------|
| **RAKE** | Fast | Good | Low | ✅ Yes | Technical docs, Multi-word phrases | No semantic understanding |
| **SimpleRAKE** | Fast | Good | Minimal | ✅ Yes | No NLTK dependencies, Built-in | Basic tokenization only |
| **YAKE** | Moderate | Very Good | Low | ❌ No | Statistical analysis, Capital aware | Python 3.13 bug |
| **BM25** | Fast | Excellent | High | ✅ Yes | Primary ranking, Length variation | Needs corpus |
| **TF-IDF** | Fast | Good | Medium | ✅ Yes | Alternative to BM25 | Less effective for varying lengths |
| **Frequency** | Very Fast | Basic | Minimal | ✅ Yes | Fallback option | Very basic |

## Embedding Model Architecture

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