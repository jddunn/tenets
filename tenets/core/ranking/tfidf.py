"""TF-IDF and BM25 calculators for relevance ranking.

This module implements Term Frequency-Inverse Document Frequency (TF-IDF) and
BM25 algorithms optimized for code search and ranking. It provides efficient
document indexing, term weighting, and similarity calculation with support for
code-aware tokenization and optional stopword filtering.

The implementation is designed for:
- Fast incremental document addition
- Memory-efficient sparse vector representation
- Code-aware tokenization (camelCase, snake_case, etc.)
- Optional stopword filtering for programming contexts
- Cached IDF values for performance
- Both TF-IDF and BM25 scoring algorithms
"""

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger


class TFIDFCalculator:
    """TF-IDF calculator with code-aware tokenization.

    Implements Term Frequency-Inverse Document Frequency scoring optimized for
    code search. Uses vector space model with cosine similarity for ranking.

    Key features:
    - Code-aware tokenization (preserves camelCase, snake_case patterns)
    - Optional stopword filtering (programming-specific stopwords)
    - Sublinear TF scaling to reduce impact of very frequent terms
    - L2 normalization for cosine similarity
    - Efficient sparse vector representation
    - Incremental document addition with IDF cache updates

    Attributes:
        use_stopwords: Whether to filter stopwords
        stopwords: Set of stopwords to filter
        document_count: Total number of documents in corpus
        document_frequency: Term -> document count mapping
        document_vectors: Document ID -> TF-IDF vector mapping
        document_norms: Document ID -> L2 norm for normalization
        idf_cache: Cached IDF values for terms
        vocabulary: Set of all unique terms in corpus
    """

    def __init__(self, use_stopwords: bool = False):
        """Initialize TF-IDF calculator.

        Args:
            use_stopwords: Whether to filter common programming stopwords.
                          Default False to preserve all terms for code contexts.
        """
        self.logger = get_logger(__name__)
        self.use_stopwords = use_stopwords
        self.stopwords: Set[str] = set()

        if self.use_stopwords:
            self.stopwords = self._load_stopwords()

        # Core data structures
        self.document_count = 0
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.document_vectors: Dict[str, Dict[str, float]] = {}
        self.document_norms: Dict[str, float] = {}
        self.idf_cache: Dict[str, float] = {}
        self.vocabulary: Set[str] = set()

        # Tokenization patterns
        self.token_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
        self.camel_case_pattern = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)")
        self.snake_case_pattern = re.compile(r"[a-z]+|[A-Z]+")

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from stopwords.txt file.

        Returns:
            Set of lowercase stopword strings.
        """
        stopwords = set()

        # Find stopwords.txt in same directory as this file
        stopwords_path = Path(__file__).parent / "stopwords.txt"

        try:
            if stopwords_path.exists():
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            stopwords.add(line.lower())
                self.logger.debug(f"Loaded {len(stopwords)} stopwords")
            else:
                self.logger.warning(f"Stopwords file not found at {stopwords_path}")
        except Exception as e:
            self.logger.error(f"Failed to load stopwords: {e}")

        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with code-aware processing.

        Performs intelligent tokenization for code:
        - Splits camelCase: "getUserName" -> ["get", "user", "name", "getusername"]
        - Splits snake_case: "user_name" -> ["user", "name", "user_name"]
        - Preserves original tokens alongside splits for better matching
        - Filters single characters except important ones (i, a, x, y, z)
        - Optionally filters stopwords
        - Converts to lowercase for case-insensitive matching

        Args:
            text: Input text to tokenize

        Returns:
            List of normalized tokens
        """
        if not text:
            return []

        tokens: List[str] = []
        raw_tokens = self.token_pattern.findall(text)

        for token in raw_tokens:
            # Skip single chars except important ones
            if len(token) == 1 and token.lower() not in {"i", "a", "x", "y", "z"}:
                continue

            # Process camelCase and PascalCase
            if any(c.isupper() for c in token) and not token.isupper():
                # Split on case boundaries
                parts = self.camel_case_pattern.findall(token)
                # Add individual parts
                for part in parts:
                    if len(part) > 1:
                        tokens.append(part.lower())
                # Also keep original token for exact matching
                tokens.append(token.lower())

            # Process snake_case
            elif "_" in token:
                parts = token.split("_")
                # Add individual parts
                for part in parts:
                    if part and len(part) > 1:
                        tokens.append(part.lower())
                # Keep original for exact matching
                tokens.append(token.lower())

            else:
                # Regular token
                tokens.append(token.lower())

        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        # Filter stopwords if enabled
        if self.use_stopwords and self.stopwords:
            unique_tokens = [t for t in unique_tokens if t not in self.stopwords]

        return unique_tokens

    def compute_tf(self, tokens: List[str], use_sublinear: bool = True) -> Dict[str, float]:
        """Compute term frequency with optional sublinear scaling.

        Args:
            tokens: List of tokens from document
            use_sublinear: Use log scaling (1 + log(tf)) to reduce impact of
                          very frequent terms. Default True.

        Returns:
            Dictionary mapping terms to TF scores
        """
        if not tokens:
            return {}

        tf_raw = Counter(tokens)

        if use_sublinear:
            # Sublinear TF: 1 + log(count)
            # Reduces impact of terms that appear many times
            return {term: 1.0 + math.log(count) for term, count in tf_raw.items()}
        else:
            # Normalized TF: count / total
            total = len(tokens)
            return {term: count / total for term, count in tf_raw.items()}

    def compute_idf(self, term: str) -> float:
        """Compute inverse document frequency for a term.

        IDF = log((N + 1) / (df + 1)) where:
        - N = total documents
        - df = documents containing term

        Uses smoothing to avoid division by zero and reduce impact of
        terms appearing in all documents.

        Args:
            term: Term to compute IDF for

        Returns:
            IDF value
        """
        if term in self.idf_cache:
            return self.idf_cache[term]

        if self.document_count == 0:
            return 0.0

        # Use smoothed IDF to handle edge cases
        df = self.document_frequency.get(term, 0)
        idf = math.log((self.document_count + 1) / (df + 1))

        self.idf_cache[term] = idf
        return idf

    def add_document(self, doc_id: str, text: str) -> Dict[str, float]:
        """Add document to corpus and compute TF-IDF vector.

        Args:
            doc_id: Unique document identifier
            text: Document text content

        Returns:
            TF-IDF vector for the document
        """
        # Tokenize document
        tokens = self.tokenize(text)

        if not tokens:
            self.document_vectors[doc_id] = {}
            self.document_norms[doc_id] = 0.0
            return {}

        # Update corpus statistics
        self.document_count += 1
        unique_terms = set(tokens)

        for term in unique_terms:
            self.document_frequency[term] += 1
            self.vocabulary.add(term)

        # Compute TF scores
        tf_scores = self.compute_tf(tokens)

        # Compute TF-IDF vector
        tfidf_vector = {}
        for term, tf in tf_scores.items():
            idf = self.compute_idf(term)
            tfidf_vector[term] = tf * idf

        # L2 normalization for cosine similarity
        norm = math.sqrt(sum(score**2 for score in tfidf_vector.values()))

        if norm > 0:
            tfidf_vector = {term: score / norm for term, score in tfidf_vector.items()}
            self.document_norms[doc_id] = norm
        else:
            self.document_norms[doc_id] = 0.0

        self.document_vectors[doc_id] = tfidf_vector

        # Clear IDF cache since document frequencies changed
        self.idf_cache.clear()

        return tfidf_vector

    def compute_similarity(self, query_text: str, doc_id: str) -> float:
        """Compute cosine similarity between query and document.

        Args:
            query_text: Query text
            doc_id: Document identifier

        Returns:
            Cosine similarity score (0-1)
        """
        # Get document vector
        doc_vector = self.document_vectors.get(doc_id, {})
        if not doc_vector:
            return 0.0

        # Process query
        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return 0.0

        # Compute query TF-IDF vector
        query_tf = self.compute_tf(query_tokens)
        query_vector = {}

        for term, tf in query_tf.items():
            if term in self.vocabulary:
                idf = self.compute_idf(term)
                query_vector[term] = tf * idf

        # Normalize query vector
        query_norm = math.sqrt(sum(score**2 for score in query_vector.values()))
        if query_norm > 0:
            query_vector = {term: score / query_norm for term, score in query_vector.items()}
        else:
            return 0.0

        # Compute dot product (cosine similarity)
        similarity = 0.0
        for term, query_score in query_vector.items():
            if term in doc_vector:
                similarity += query_score * doc_vector[term]

        return max(0.0, min(1.0, similarity))

    def get_top_terms(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get top TF-IDF terms for a document.

        Useful for understanding why a document ranks highly.

        Args:
            doc_id: Document identifier
            n: Number of top terms to return

        Returns:
            List of (term, tfidf_score) tuples sorted by score
        """
        doc_vector = self.document_vectors.get(doc_id, {})
        if not doc_vector:
            return []

        sorted_terms = sorted(doc_vector.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:n]

    def build_corpus(self, documents: List[Tuple[str, str]]) -> None:
        """Build TF-IDF corpus from multiple documents.

        Batch processing method for efficiency when adding many documents.

        Args:
            documents: List of (doc_id, text) tuples
        """
        self.logger.info(f"Building TF-IDF corpus from {len(documents)} documents")

        for doc_id, text in documents:
            self.add_document(doc_id, text)

        self.logger.info(
            f"Corpus built: {self.document_count} documents, "
            f"{len(self.vocabulary)} unique terms"
        )


class BM25Calculator:
    """BM25 ranking algorithm implementation.

    BM25 (Best Matching 25) is a probabilistic ranking function that often
    outperforms TF-IDF for information retrieval. It includes document length
    normalization and term saturation.

    The algorithm uses:
    - Term frequency saturation (diminishing returns for repeated terms)
    - Document length normalization (penalizes very long documents)
    - Tunable parameters k1 and b for different corpus characteristics

    Attributes:
        k1: Term frequency saturation parameter (default 1.2)
        b: Length normalization parameter (default 0.75)
        use_stopwords: Whether to filter stopwords
        stopwords: Set of stopwords to filter
        document_count: Total documents in corpus
        document_frequency: Term -> document count mapping
        document_lengths: Document ID -> length mapping
        average_doc_length: Average document length in corpus
        vocabulary: Set of all unique terms
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75, use_stopwords: bool = False):
        """Initialize BM25 calculator.

        Args:
            k1: Controls term frequency saturation. Higher values mean
                less saturation. Range: [1.2, 2.0] typically.
            b: Controls length normalization. 0 = no normalization,
               1 = full normalization. Range: [0, 1].
            use_stopwords: Whether to filter stopwords
        """
        self.logger = get_logger(__name__)
        self.k1 = k1
        self.b = b
        self.use_stopwords = use_stopwords

        # Reuse tokenizer from TF-IDF
        self._tfidf = TFIDFCalculator(use_stopwords=use_stopwords)
        self.stopwords = self._tfidf.stopwords

        # Core data structures
        self.document_count = 0
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.document_tokens: Dict[str, List[str]] = {}
        self.average_doc_length = 0.0
        self.vocabulary: Set[str] = set()
        self.idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using TF-IDF tokenizer.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return self._tfidf.tokenize(text)

    def add_document(self, doc_id: str, text: str) -> None:
        """Add document to BM25 corpus.

        Args:
            doc_id: Unique document identifier
            text: Document text content
        """
        tokens = self.tokenize(text)

        if not tokens:
            self.document_lengths[doc_id] = 0
            self.document_tokens[doc_id] = []
            return

        # Update corpus statistics
        self.document_count += 1
        self.document_lengths[doc_id] = len(tokens)
        self.document_tokens[doc_id] = tokens

        # Update document frequency
        unique_terms = set(tokens)
        for term in unique_terms:
            self.document_frequency[term] += 1
            self.vocabulary.add(term)

        # Update average document length
        total_length = sum(self.document_lengths.values())
        self.average_doc_length = total_length / max(1, self.document_count)

        # Clear IDF cache
        self.idf_cache.clear()

    def compute_idf(self, term: str) -> float:
        """Compute IDF component for BM25.

        BM25 IDF = log((N - df + 0.5) / (df + 0.5))

        Args:
            term: Term to compute IDF for

        Returns:
            IDF value
        """
        if term in self.idf_cache:
            return self.idf_cache[term]

        df = self.document_frequency.get(term, 0)
        idf = math.log((self.document_count - df + 0.5) / (df + 0.5))

        self.idf_cache[term] = idf
        return idf

    def score_document(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document.

        Args:
            query_tokens: Tokenized query
            doc_id: Document identifier

        Returns:
            BM25 score
        """
        if doc_id not in self.document_tokens:
            return 0.0

        doc_tokens = self.document_tokens[doc_id]
        if not doc_tokens:
            return 0.0

        doc_length = self.document_lengths[doc_id]

        # Count term frequencies in document
        doc_tf = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.vocabulary:
                continue

            # IDF component
            idf = self.compute_idf(term)

            # Term frequency component with saturation
            tf = doc_tf.get(term, 0)

            # Length normalization factor
            norm_factor = 1 - self.b + self.b * (doc_length / self.average_doc_length)

            # BM25 formula
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)

            score += idf * tf_component

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search documents using BM25 ranking.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Score all documents
        scores = []
        for doc_id in self.document_tokens:
            score = self.score_document(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def build_corpus(self, documents: List[Tuple[str, str]]) -> None:
        """Build BM25 corpus from multiple documents.

        Args:
            documents: List of (doc_id, text) tuples
        """
        self.logger.info(f"Building BM25 corpus from {len(documents)} documents")

        for doc_id, text in documents:
            self.add_document(doc_id, text)

        self.logger.info(
            f"BM25 corpus built: {self.document_count} documents, "
            f"{len(self.vocabulary)} unique terms, "
            f"avg doc length: {self.average_doc_length:.1f}"
        )
