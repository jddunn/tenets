"""TF-IDF calculator for relevance ranking.

This module provides TF-IDF text similarity as an optional fallback
to the primary BM25 ranking algorithm. The TF-IDF implementation
reuses centralized logic from keyword_extractor.
"""

from typing import Dict, List, Set, Tuple

from tenets.utils.logger import get_logger


class TFIDFCalculator:
    """TF-IDF calculator for ranking.

    Simplified wrapper around NLP TFIDFCalculator to maintain
    existing ranking API while using centralized logic.
    """

    def __init__(self, use_stopwords: bool = False):
        """Initialize TF-IDF calculator.

        Args:
            use_stopwords: Whether to filter stopwords (uses 'code' set)
        """
        self.logger = get_logger(__name__)
        self.use_stopwords = use_stopwords

        # Use centralized NLP TF-IDF calculator
        from tenets.core.nlp.keyword_extractor import TFIDFCalculator as NLPTFIDFCalculator

        self._calculator = NLPTFIDFCalculator(
            use_stopwords=use_stopwords,
            stopword_set="code",  # Use minimal stopwords for code/code-search
        )

        # Expose a mutable stopword set expected by tests; we'll additionally
        # filter tokens against this set in tokenize() when enabled
        if use_stopwords:
            try:
                from tenets.core.nlp.stopwords import StopwordManager

                sw = StopwordManager().get_set("code")
                self.stopwords: Set[str] = set(sw.words) if sw else set()
            except Exception:
                self.stopwords = set()
        else:
            self.stopwords = set()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLP tokenizer.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        tokens = self._calculator.tokenize(text)
        if self.use_stopwords and self.stopwords:
            sw = self.stopwords
            tokens = [t for t in tokens if t not in sw and t.lower() not in sw]
        return tokens

    def add_document(self, doc_id: str, text: str) -> Dict[str, float]:
        """Add document to corpus.

        Args:
            doc_id: Document identifier
            text: Document content

        Returns:
            TF-IDF vector for document
        """
        # Invalidate IDF cache before/after adding a document to reflect corpus change
        try:
            if hasattr(self._calculator, "idf_cache"):
                self._calculator.idf_cache = {}
        except Exception:
            pass
        result = self._calculator.add_document(doc_id, text)
        try:
            if hasattr(self._calculator, "idf_cache"):
                self._calculator.idf_cache = {}
        except Exception:
            pass
        return result

    # Expose core TF/IDF computations used by tests
    def compute_tf(self, tokens: List[str], use_sublinear: bool = True) -> Dict[str, float]:
        return self._calculator.compute_tf(tokens, use_sublinear=use_sublinear)

    def compute_idf(self, term: str) -> float:
        return self._calculator.compute_idf(term)

    def compute_similarity(self, query_text: str, doc_id: str) -> float:
        """Compute similarity between query and document.

        Args:
            query_text: Query text
            doc_id: Document identifier

        Returns:
            Cosine similarity score (0-1)
        """
        return self._calculator.compute_similarity(query_text, doc_id)

    def get_top_terms(self, doc_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return the top-n TF-IDF terms for a given document.

        Args:
            doc_id: Document identifier
            n: Maximum number of terms to return

        Returns:
            List of (term, score) sorted by score descending
        """
        vec = self._calculator.document_vectors.get(doc_id, {})
        if not vec:
            return []
        # Already normalized; just sort and take top-n
        return sorted(vec.items(), key=lambda x: x[1], reverse=True)[:n]

    def build_corpus(self, documents: List[Tuple[str, str]]) -> None:
        """Build corpus from documents.

        Args:
            documents: List of (doc_id, text) tuples
        """
        self._calculator.build_corpus(documents)

    @property
    def document_vectors(self) -> Dict[str, Dict[str, float]]:
        """Get document vectors."""
        return self._calculator.document_vectors

    @property
    def document_norms(self) -> Dict[str, float]:
        """Get document vector norms."""
        return getattr(self._calculator, "document_norms", {})

    @property
    def vocabulary(self) -> set:
        """Get vocabulary."""
        return self._calculator.vocabulary

    # Properties that tests expect to be mutable on the calculator
    @property
    def document_count(self) -> int:
        return getattr(self._calculator, "document_count", 0)

    @document_count.setter
    def document_count(self, value: int) -> None:
        self._calculator.document_count = value

    @property
    def document_frequency(self) -> Dict[str, int]:
        return getattr(self._calculator, "document_frequency", {})

    @document_frequency.setter
    def document_frequency(self, value: Dict[str, int]) -> None:
        self._calculator.document_frequency = value

    @property
    def idf_cache(self) -> Dict[str, float]:
        return getattr(self._calculator, "idf_cache", {})

    @idf_cache.setter
    def idf_cache(self, value: Dict[str, float]) -> None:
        self._calculator.idf_cache = value

