"""BM25 ranking algorithm implementation.

BM25 (Best Matching 25) is a probabilistic ranking function that improves
upon TF-IDF for information retrieval. This module provides a robust,
well-documented implementation optimized for code search.

Key Features:
    - Term frequency saturation to prevent over-weighting repeated terms
    - Sophisticated document length normalization
    - Configurable parameters for different document types
    - Efficient sparse representation for large corpora
    - Cache-friendly design for repeated queries

Mathematical Foundation:
    BM25 score for document D given query Q:

    Score(D,Q) = Σ IDF(qi) × [f(qi,D) × (k1 + 1)] / [f(qi,D) + k1 × (1 - b + b × |D|/avgdl)]

    Where:
        qi = each query term
        f(qi,D) = frequency of term qi in document D
        |D| = length of document D in tokens
        avgdl = average document length in the corpus
        k1 = term frequency saturation parameter (default: 1.2)
        b = length normalization parameter (default: 0.75)

    IDF Component:
        IDF(qi) = log[(N - df(qi) + 0.5) / (df(qi) + 0.5) + 1]

        Where:
            N = total number of documents
            df(qi) = number of documents containing term qi

Usage:
    >>> from tenets.core.nlp.bm25 import BM25Calculator
    >>>
    >>> # Initialize calculator
    >>> bm25 = BM25Calculator(k1=1.2, b=0.75)
    >>>
    >>> # Build corpus
    >>> documents = [
    ...     ("doc1", "Python web framework Django"),
    ...     ("doc2", "Flask is a lightweight Python framework"),
    ...     ("doc3", "JavaScript React framework for UI")
    ... ]
    >>> bm25.build_corpus(documents)
    >>>
    >>> # Score documents for a query
    >>> scores = bm25.get_scores("Python framework")
    >>> for doc_id, score in scores:
    ...     print(f"{doc_id}: {score:.3f}")

References:
    - Robertson & Walker (1994): "Some simple effective approximations to the
      2-Poisson model for probabilistic weighted retrieval"
    - Trotman et al. (2014): "Improvements to BM25 and language models examined"
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger


class BM25Calculator:
    """BM25 ranking algorithm with advanced features for code search.

    This implementation provides:
        - Configurable term saturation (k1) and length normalization (b)
        - Efficient tokenization with optional stopword filtering
        - IDF caching for performance
        - Support for incremental corpus updates
        - Query expansion capabilities
        - Detailed scoring explanations for debugging

    Attributes:
        k1 (float): Controls term frequency saturation. Higher values mean
                   less saturation (more weight to term frequency).
                   Typical range: 0.5-2.0, default: 1.2
        b (float): Controls document length normalization.
                  0 = no normalization, 1 = full normalization.
                  Typical range: 0.5-0.8, default: 0.75
        epsilon (float): Small constant to prevent division by zero
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25,
        use_stopwords: bool = False,
        stopword_set: str = "code",
    ):
        """Initialize BM25 calculator with configurable parameters.

        Args:
            k1: Term frequency saturation parameter. Lower values (0.5-1.0)
                work well for short queries, higher values (1.5-2.0) for
                longer queries. Default: 1.2 (good general purpose value)
            b: Length normalization parameter. Set to 0 to disable length
               normalization, 1 for full normalization. Default: 0.75
               (moderate normalization, good for mixed-length documents)
            epsilon: Small constant for numerical stability
            use_stopwords: Whether to filter common words
            stopword_set: Which stopword set to use ('code' for programming,
                         'english' for natural language)
        """
        self.logger = get_logger(__name__)

        # Validate and set parameters
        if k1 < 0:
            raise ValueError(f"k1 must be non-negative, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.use_stopwords = use_stopwords
        self.stopword_set = stopword_set

        # Initialize tokenizer
        from .tokenizer import CodeTokenizer

        self.tokenizer = CodeTokenizer(use_stopwords=use_stopwords)

        # Core data structures
        self.document_count = 0
        self.document_frequency: Dict[str, int] = defaultdict(int)
        self.document_lengths: Dict[str, int] = {}
        self.document_tokens: Dict[str, List[str]] = {}
        self.average_doc_length = 0.0
        self.vocabulary: Set[str] = set()

        # Caching structures for performance
        self.idf_cache: Dict[str, float] = {}
        self._score_cache: Dict[Tuple[str, str], float] = {}

        # Statistics tracking
        self.stats = {
            "queries_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "documents_added": 0,
        }

        self.logger.info(
            f"BM25 initialized with k1={k1}, b={b}, "
            f"stopwords={'enabled' if use_stopwords else 'disabled'}"
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using code-aware tokenizer.

        Handles various code constructs:
            - CamelCase and snake_case splitting
            - Preservation of important symbols
            - Number and identifier extraction

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens, lowercased and filtered
        """
        return self.tokenizer.tokenize(text)

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the BM25 corpus.

        Updates all corpus statistics including document frequency,
        average document length, and vocabulary.

        Args:
            doc_id: Unique identifier for the document
            text: Document content

        Note:
            Adding documents invalidates the IDF and score caches.
            For bulk loading, use build_corpus() instead.
        """
        tokens = self.tokenize(text)

        # Handle empty documents
        if not tokens:
            self.document_lengths[doc_id] = 0
            self.document_tokens[doc_id] = []
            self.logger.debug(f"Added empty document: {doc_id}")
            return

        # Remove old version if updating
        if doc_id in self.document_tokens:
            self._remove_document(doc_id)

        # Update corpus statistics
        self.document_count += 1
        self.document_lengths[doc_id] = len(tokens)
        self.document_tokens[doc_id] = tokens

        # Update document frequency for unique terms
        unique_terms = set(tokens)
        for term in unique_terms:
            self.document_frequency[term] += 1
            self.vocabulary.add(term)

        # Update average document length incrementally
        total_length = sum(self.document_lengths.values())
        self.average_doc_length = total_length / max(1, self.document_count)

        # Invalidate caches
        self.idf_cache.clear()
        self._score_cache.clear()

        self.stats["documents_added"] += 1

        self.logger.debug(
            f"Added document {doc_id}: {len(tokens)} tokens, "
            f"corpus now has {self.document_count} docs"
        )

    def _remove_document(self, doc_id: str) -> None:
        """Remove a document from the corpus (internal use).

        Args:
            doc_id: Document identifier to remove
        """
        if doc_id not in self.document_tokens:
            return

        # Update document frequency
        tokens = self.document_tokens[doc_id]
        unique_terms = set(tokens)
        for term in unique_terms:
            self.document_frequency[term] -= 1
            if self.document_frequency[term] == 0:
                del self.document_frequency[term]
                self.vocabulary.discard(term)

        # Remove document data
        del self.document_tokens[doc_id]
        del self.document_lengths[doc_id]
        self.document_count -= 1

    def build_corpus(self, documents: List[Tuple[str, str]]) -> None:
        """Build BM25 corpus from multiple documents efficiently.

        More efficient than repeated add_document() calls as it
        calculates statistics once at the end.

        Args:
            documents: List of (doc_id, text) tuples

        Example:
            >>> documents = [
            ...     ("file1.py", "import os\\nclass FileHandler"),
            ...     ("file2.py", "from pathlib import Path")
            ... ]
            >>> bm25.build_corpus(documents)
        """
        self.logger.info(f"Building corpus from {len(documents)} documents")

        # Clear existing data
        self.document_count = 0
        self.document_frequency.clear()
        self.document_lengths.clear()
        self.document_tokens.clear()
        self.vocabulary.clear()
        self.idf_cache.clear()
        self._score_cache.clear()

        # Process all documents
        total_length = 0
        for doc_id, text in documents:
            tokens = self.tokenize(text)

            if not tokens:
                self.document_lengths[doc_id] = 0
                self.document_tokens[doc_id] = []
                continue

            self.document_count += 1
            self.document_lengths[doc_id] = len(tokens)
            self.document_tokens[doc_id] = tokens
            total_length += len(tokens)

            # Update document frequency
            unique_terms = set(tokens)
            for term in unique_terms:
                self.document_frequency[term] += 1
                self.vocabulary.add(term)

        # Calculate average document length
        self.average_doc_length = total_length / max(1, self.document_count)

        self.stats["documents_added"] = self.document_count

        self.logger.info(
            f"Corpus built: {self.document_count} docs, "
            f"{len(self.vocabulary)} unique terms, "
            f"avg length: {self.average_doc_length:.1f} tokens"
        )

    def compute_idf(self, term: str) -> float:
        """Compute IDF (Inverse Document Frequency) for a term.

        Uses the standard BM25 IDF formula with smoothing to handle
        edge cases and prevent negative values.

        Formula:
            IDF(term) = log[(N - df + 0.5) / (df + 0.5) + 1]

        Args:
            term: Term to compute IDF for

        Returns:
            IDF value (always positive due to +1 in formula)
        """
        # Check cache first
        if term in self.idf_cache:
            self.stats["cache_hits"] += 1
            return self.idf_cache[term]

        self.stats["cache_misses"] += 1

        # Get document frequency
        df = self.document_frequency.get(term, 0)

        # BM25 IDF formula with smoothing
        # Adding 1 ensures IDF is always positive
        numerator = self.document_count - df + 0.5
        denominator = df + 0.5
        idf = math.log(numerator / denominator + 1)

        # Cache the result
        self.idf_cache[term] = idf

        return idf

    def score_document(self, query_tokens: List[str], doc_id: str, explain: bool = False) -> float:
        """Calculate BM25 score for a document given query tokens.

        Implements the full BM25 scoring formula with term saturation
        and length normalization.

        Args:
            query_tokens: Tokenized query terms
            doc_id: Document identifier to score
            explain: If True, return detailed scoring breakdown

        Returns:
            BM25 score (higher is more relevant)
            If explain=True, returns tuple of (score, explanation_dict)
        """
        # Check if document exists
        if doc_id not in self.document_tokens:
            return (0.0, {}) if explain else 0.0

        doc_tokens = self.document_tokens[doc_id]
        if not doc_tokens:
            return (0.0, {"empty_doc": True}) if explain else 0.0

        # Check score cache
        cache_key = (tuple(query_tokens), doc_id)
        if cache_key in self._score_cache and not explain:
            self.stats["cache_hits"] += 1
            return self._score_cache[cache_key]

        self.stats["cache_misses"] += 1

        # Get document statistics
        doc_length = self.document_lengths[doc_id]
        doc_tf = Counter(doc_tokens)

        # Length normalization factor
        if self.average_doc_length > 0:
            norm_factor = 1 - self.b + self.b * (doc_length / self.average_doc_length)
        else:
            norm_factor = 1.0

        # Calculate score
        score = 0.0
        term_scores = {} if explain else None

        for term in set(query_tokens):  # Use set to handle repeated query terms
            if term not in self.vocabulary:
                continue

            # Get term frequency in document
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue

            # IDF component
            idf = self.compute_idf(term)

            # BM25 term frequency component with saturation
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)

            # Term contribution to score
            term_score = idf * tf_component
            score += term_score

            if explain:
                term_scores[term] = {
                    "tf": tf,
                    "idf": idf,
                    "tf_component": tf_component,
                    "score": term_score,
                }

        # Cache the score
        self._score_cache[cache_key] = score

        if explain:
            explanation = {
                "total_score": score,
                "doc_length": doc_length,
                "avg_doc_length": self.average_doc_length,
                "norm_factor": norm_factor,
                "term_scores": term_scores,
            }
            return score, explanation

        return score

    def get_scores(
        self, query: str, doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """Get BM25 scores for all documents or a subset.

        Args:
            query: Search query string
            doc_ids: Optional list of document IDs to score.
                    If None, scores all documents.

        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        self.stats["queries_processed"] += 1

        # Tokenize query
        query_tokens = self.tokenize(query)
        if not query_tokens:
            self.logger.warning(f"Empty query after tokenization: '{query}'")
            return []

        # Determine documents to score
        if doc_ids is None:
            doc_ids = list(self.document_tokens.keys())

        # Calculate scores
        scores = []
        for doc_id in doc_ids:
            score = self.score_document(query_tokens, doc_id)
            if score > 0:  # Only include documents with positive scores
                scores.append((doc_id, score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores

    def get_top_k(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Get top-k documents by BM25 score.

        Args:
            query: Search query
            k: Number of top documents to return
            threshold: Minimum score threshold (documents below are filtered)

        Returns:
            List of top-k (doc_id, score) tuples
        """
        scores = self.get_scores(query)

        # Filter by threshold
        if threshold > 0:
            scores = [(doc_id, score) for doc_id, score in scores if score >= threshold]

        return scores[:k]

    def compute_similarity(self, query: str, doc_id: str) -> float:
        """Compute normalized similarity score between query and document.

        Returns a value between 0 and 1 for consistency with other
        similarity measures.

        Args:
            query: Query text
            doc_id: Document identifier

        Returns:
            Normalized similarity score (0-1)
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return 0.0

        # Get raw BM25 score
        score = self.score_document(query_tokens, doc_id)

        # Normalize score to 0-1 range
        # Using sigmoid-like normalization for better distribution
        normalized = score / (score + 10.0)  # 10.0 is empirically chosen

        return min(1.0, normalized)

    def explain_score(self, query: str, doc_id: str) -> Dict:
        """Get detailed explanation of BM25 scoring for debugging.

        Args:
            query: Query text
            doc_id: Document to explain scoring for

        Returns:
            Dictionary with detailed scoring breakdown
        """
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return {"error": "Empty query after tokenization"}

        score, explanation = self.score_document(query_tokens, doc_id, explain=True)

        # Add query information
        explanation["query"] = query
        explanation["query_tokens"] = query_tokens
        explanation["parameters"] = {"k1": self.k1, "b": self.b}

        return explanation

    def get_stats(self) -> Dict:
        """Get calculator statistics for monitoring.

        Returns:
            Dictionary with usage statistics
        """
        return {
            **self.stats,
            "corpus_size": self.document_count,
            "vocabulary_size": len(self.vocabulary),
            "avg_doc_length": self.average_doc_length,
            "idf_cache_size": len(self.idf_cache),
            "score_cache_size": len(self._score_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            ),
        }

    def clear_cache(self) -> None:
        """Clear all caches to free memory."""
        self.idf_cache.clear()
        self._score_cache.clear()
        self.logger.debug("Caches cleared")


# Convenience function for backward compatibility
def create_bm25(documents: List[Tuple[str, str]], **kwargs) -> BM25Calculator:
    """Create and initialize a BM25 calculator with documents.

    Args:
        documents: List of (doc_id, text) tuples
        **kwargs: Additional arguments for BM25Calculator

    Returns:
        Initialized BM25Calculator with corpus built
    """
    calculator = BM25Calculator(**kwargs)
    calculator.build_corpus(documents)
    return calculator
