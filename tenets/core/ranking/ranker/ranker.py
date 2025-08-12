"""Relevance ranking system for Tenets.

This module implements sophisticated multi-factor ranking algorithms to score
and sort files by relevance to a given prompt. It supports multiple ranking
strategies from simple keyword matching to advanced ML-based semantic similarity.

The ranking system considers:
- Keyword relevance (TF-IDF, BM25)
- Code structure and imports
- Git activity and recency
- File path relevance
- Semantic similarity (with ML features)
- Custom scoring factors

Rankings can be customized through configuration or by implementing custom
ranking algorithms.
"""

import re
import math
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import concurrent.futures
from abc import ABC, abstractmethod
from enum import Enum
import string
import os

from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext
from tenets.utils.logger import get_logger


class RankingAlgorithm(Enum):
    """Available ranking algorithms."""

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    ML = "ml"
    CUSTOM = "custom"


@dataclass
class RankingFactors:
    """Individual ranking factors for a file.

    Each factor contributes to the final relevance score based on
    configured weights.

    Attributes:
        keyword_match: Score based on keyword presence and frequency
        tfidf_similarity: TF-IDF similarity score
        bm25_score: BM25 relevance score
        path_relevance: Score based on file path matching
        import_centrality: How central this file is in import graph
        git_recency: Score based on recent git activity
        git_frequency: Score based on change frequency
        complexity_relevance: Relevance based on code complexity
        semantic_similarity: ML-based semantic similarity
        type_relevance: Relevance based on file type
        custom_scores: Dictionary of custom scoring factors
    """

    keyword_match: float = 0.0
    tfidf_similarity: float = 0.0
    bm25_score: float = 0.0
    path_relevance: float = 0.0
    import_centrality: float = 0.0
    git_recency: float = 0.0
    git_frequency: float = 0.0
    complexity_relevance: float = 0.0
    semantic_similarity: float = 0.0
    type_relevance: float = 0.0
    custom_scores: Dict[str, float] = field(default_factory=dict)

    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted score from factors.

        Args:
            weights: Dictionary of factor weights

        Returns:
            Weighted relevance score
        """
        score = 0.0

        # Standard factors
        factor_values = {
            "keyword_match": self.keyword_match,
            "tfidf_similarity": self.tfidf_similarity,
            "bm25_score": self.bm25_score,
            "path_relevance": self.path_relevance,
            "import_centrality": self.import_centrality,
            "git_recency": self.git_recency,
            "git_frequency": self.git_frequency,
            "complexity_relevance": self.complexity_relevance,
            "semantic_similarity": self.semantic_similarity,
            "type_relevance": self.type_relevance,
        }

        for factor, value in factor_values.items():
            if factor in weights:
                score += value * weights[factor]

        # Custom factors
        for custom_factor, value in self.custom_scores.items():
            if custom_factor in weights:
                score += value * weights[custom_factor]

        return min(1.0, max(0.0, score))  # Clamp to [0, 1]


@dataclass
class RankedFile:
    """A file with its relevance ranking.

    Attributes:
        analysis: FileAnalysis object
        score: Overall relevance score (0-1)
        factors: Breakdown of ranking factors
        explanation: Human-readable explanation of ranking
    """

    analysis: FileAnalysis
    score: float
    factors: RankingFactors
    explanation: str = ""

    def __lt__(self, other):
        """Compare by score for sorting."""
        return self.score < other.score


class TFIDFCalculator:
    """TF-IDF calculator with optional stopwords and code-aware normalization.

    Implements Term Frequency-Inverse Document Frequency scoring optimized for
    code search. Uses vector space model with cosine similarity for ranking.

    Key features:
    - Optional stopword filtering (off by default for code contexts)
    - Code-aware tokenization (preserves camelCase, snake_case, etc.)
    - Document frequency tracking for IDF calculation
    - Efficient vector representation and similarity computation
    - Sublinear TF scaling and L2 normalization
    """

    def __init__(self, use_stopwords: bool = False):
        """Initialize TF-IDF calculator.

        Args:
            use_stopwords: Whether to filter stopwords (default False for code)
        """
        self.logger = get_logger(__name__)
        self.use_stopwords = use_stopwords
        self.stopwords = set()

        # Load stopwords if requested
        if self.use_stopwords:
            self.stopwords = self._load_stopwords()

        # Core TF-IDF data structures
        self.document_count = 0
        self.document_frequency = defaultdict(int)  # term -> number of docs containing term
        self.document_vectors = {}  # doc_id -> {term: tf_idf_weight}
        self.document_norms = {}  # doc_id -> L2 norm for normalization
        self.idf_cache = {}  # term -> idf value
        self.vocabulary = set()  # all unique terms

        # Token extraction patterns for code
        self.token_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")

        # Patterns for extracting meaningful code tokens
        self.camel_case_pattern = re.compile(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)")
        self.snake_case_pattern = re.compile(r"[a-z]+|[A-Z]+")

    def _load_stopwords(self) -> Set[str]:
        """Load stopwords from file (../stopwords.txt relative to this file).

        Returns:
            Set of stopword strings in lowercase
        """
        stopwords = set()

        # Get path to stopwords file (one level up from this file)
        current_dir = Path(__file__).parent
        stopwords_path = current_dir.parent / "stopwords.txt"

        try:
            if stopwords_path.exists():
                with open(stopwords_path, "r", encoding="utf-8") as f:
                    stopwords = {line.strip().lower() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(stopwords)} stopwords from {stopwords_path}")
            else:
                self.logger.warning(f"Stopwords file not found at {stopwords_path}")
        except Exception as e:
            self.logger.error(f"Failed to load stopwords: {e}")

        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms for TF-IDF calculation.

        Performs code-aware tokenization:
        - Splits camelCase and PascalCase into components
        - Preserves snake_case as single tokens
        - Extracts alphanumeric identifiers
        - Optionally filters stopwords
        - Converts to lowercase for matching

        Args:
            text: Input text to tokenize

        Returns:
            List of normalized tokens
        """
        if not text:
            return []

        tokens = []

        # Extract all word-like tokens
        raw_tokens = self.token_pattern.findall(text)

        for token in raw_tokens:
            # Skip single character tokens except important ones
            if len(token) == 1 and token not in {"i", "a", "x", "y", "z"}:
                continue

            # Split camelCase and PascalCase
            if any(c.isupper() for c in token) and not token.isupper():
                # Handle camelCase/PascalCase
                parts = self.camel_case_pattern.findall(token)
                tokens.extend([p.lower() for p in parts if len(p) > 1])
                # Also keep the original token
                tokens.append(token.lower())
            elif "_" in token:
                # Handle snake_case - split but also keep original
                parts = token.split("_")
                tokens.extend([p.lower() for p in parts if p])
                tokens.append(token.lower())
            else:
                # Regular token
                tokens.append(token.lower())

        # Filter stopwords if enabled
        if self.use_stopwords and self.stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def compute_tf(self, tokens: List[str], use_sublinear: bool = True) -> Dict[str, float]:
        """Compute term frequency for a document.

        Uses sublinear TF scaling by default (1 + log(tf)) to reduce
        the impact of very frequent terms within a document.

        Args:
            tokens: List of tokens in the document
            use_sublinear: Whether to use logarithmic scaling

        Returns:
            Dictionary mapping terms to their TF scores
        """
        tf = Counter(tokens)

        if use_sublinear:
            # Sublinear scaling: 1 + log(tf)
            # Reduces impact of terms that appear many times
            tf_scaled = {}
            for term, count in tf.items():
                tf_scaled[term] = 1.0 + math.log(count) if count > 0 else 0.0
            return tf_scaled
        else:
            # Raw frequency normalized by document length
            total = len(tokens)
            if total == 0:
                return {}
            return {term: count / total for term, count in tf.items()}

    def compute_idf(self, term: str) -> float:
        """Compute inverse document frequency for a term.

        IDF = log(N / df) where:
        - N is total number of documents
        - df is number of documents containing the term

        Uses smoothing (+1) to avoid division by zero and reduce impact
        of terms that appear in all documents.

        Args:
            term: Term to compute IDF for

        Returns:
            IDF value for the term
        """
        # Check cache first
        if term in self.idf_cache:
            return self.idf_cache[term]

        if self.document_count == 0:
            return 0.0

        # Handle single document case - use a small positive value
        # to ensure TF-IDF scores are non-zero for normalization
        if self.document_count == 1:
            idf = 1.0  # Fixed value for single document
        else:
            # Smoothed IDF: log((N + 1) / (df + 1))
            # Prevents division by zero and handles new terms gracefully
            df = self.document_frequency.get(term, 0)
            idf = math.log((self.document_count + 1) / (df + 1))

        # Cache the result
        self.idf_cache[term] = idf

        return idf

    def add_document(self, doc_id: str, text: str) -> Dict[str, float]:
        """Add a document to the TF-IDF corpus and compute its vector.

        Updates document frequency counts and computes TF-IDF vector
        for the new document.

        Args:
            doc_id: Unique identifier for the document
            text: Document text content

        Returns:
            TF-IDF vector for the document
        """
        # Tokenize the document
        tokens = self.tokenize(text)

        if not tokens:
            self.document_vectors[doc_id] = {}
            self.document_norms[doc_id] = 0.0
            return {}

        # Update document count
        self.document_count += 1

        # Update document frequency for each unique term
        unique_terms = set(tokens)
        for term in unique_terms:
            self.document_frequency[term] += 1
            self.vocabulary.add(term)

        # Compute TF for this document
        tf_scores = self.compute_tf(tokens)

        # Compute TF-IDF vector
        tfidf_vector = {}
        for term, tf in tf_scores.items():
            idf = self.compute_idf(term)
            tfidf_vector[term] = tf * idf

        # Compute L2 norm for normalization
        norm = math.sqrt(sum(score**2 for score in tfidf_vector.values()))

        # Store normalized vector
        if norm > 0:
            tfidf_vector = {term: score / norm for term, score in tfidf_vector.items()}
            self.document_norms[doc_id] = norm
        else:
            self.document_norms[doc_id] = 0.0

        self.document_vectors[doc_id] = tfidf_vector

        # Clear IDF cache since document frequencies changed
        # (do this after computing the vector to ensure cache consistency)
        self.idf_cache.clear()

        return tfidf_vector

    def compute_similarity(self, query_text: str, doc_id: str) -> float:
        """Compute cosine similarity between query and document.

        Uses dot product of normalized TF-IDF vectors to compute
        cosine similarity in range [0, 1].

        Args:
            query_text: Query text
            doc_id: Document identifier

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Get document vector
        doc_vector = self.document_vectors.get(doc_id, {})
        if not doc_vector:
            return 0.0

        # Compute query vector
        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return 0.0

        query_tf = self.compute_tf(query_tokens)
        query_vector = {}

        for term, tf in query_tf.items():
            # Only consider terms that exist in corpus
            if term in self.vocabulary:
                idf = self.compute_idf(term)
                query_vector[term] = tf * idf

        # Normalize query vector
        query_norm = math.sqrt(sum(score**2 for score in query_vector.values()))
        if query_norm > 0:
            query_vector = {term: score / query_norm for term, score in query_vector.items()}
        else:
            return 0.0

        # Compute dot product (cosine similarity since vectors are normalized)
        similarity = 0.0
        for term, query_score in query_vector.items():
            if term in doc_vector:
                similarity += query_score * doc_vector[term]

        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

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

        # Sort by TF-IDF score
        sorted_terms = sorted(doc_vector.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:n]

    def build_corpus(self, documents: List[Tuple[str, str]]) -> None:
        """Build TF-IDF corpus from multiple documents.

        Batch processing for efficiency when adding many documents.

        Args:
            documents: List of (doc_id, text) tuples
        """
        self.logger.info(f"Building TF-IDF corpus from {len(documents)} documents")

        for doc_id, text in documents:
            self.add_document(doc_id, text)

        self.logger.info(f"Corpus built with {len(self.vocabulary)} unique terms")


class RankingStrategy(ABC):
    """Abstract base class for ranking strategies.

    Each strategy implements a different approach to calculating
    relevance scores for files.
    """

    @abstractmethod
    def rank_file(
        self, file: FileAnalysis, prompt_context: PromptContext, corpus_stats: Dict[str, Any]
    ) -> RankingFactors:
        """Calculate ranking factors for a file.

        Args:
            file: File to rank
            prompt_context: Parsed prompt information
            corpus_stats: Statistics about the entire codebase

        Returns:
            RankingFactors with calculated scores
        """
        pass

    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """Get factor weights for this strategy.

        Returns:
            Dictionary of factor weights
        """
        pass


class FastRankingStrategy(RankingStrategy):
    """Fast keyword-based ranking.

    This strategy uses simple keyword matching and path analysis
    for quick relevance scoring. Suitable for large codebases where
    speed is critical.
    """

    def rank_file(
        self, file: FileAnalysis, prompt_context: PromptContext, corpus_stats: Dict[str, Any]
    ) -> RankingFactors:
        """Fast ranking based on keywords and paths.

        Args:
            file: File to rank
            prompt_context: Parsed prompt information
            corpus_stats: Statistics about the entire codebase

        Returns:
            RankingFactors with keyword and path scores
        """
        factors = RankingFactors()

        # Keyword matching
        if file.content and prompt_context.keywords:
            content_lower = file.content.lower()
            keyword_hits = 0
            keyword_density = 0

            for keyword in prompt_context.keywords:
                keyword_lower = keyword.lower()
                count = content_lower.count(keyword_lower)
                if count > 0:
                    keyword_hits += 1
                    keyword_density += count / len(file.content.split())

            factors.keyword_match = min(1.0, keyword_hits / len(prompt_context.keywords))
            factors.keyword_match *= 1 + min(0.5, keyword_density)  # Boost for density

        # Path relevance
        path_lower = file.path.lower()
        path_score = 0.0

        for keyword in prompt_context.keywords:
            if keyword.lower() in path_lower:
                path_score += 0.3

        # Bonus for important paths
        important_paths = ["main", "index", "app", "core", "api", "handler", "service"]
        for important in important_paths:
            if important in path_lower:
                path_score += 0.2
                break

        factors.path_relevance = min(1.0, path_score)

        # File type relevance
        factors.type_relevance = self._calculate_type_relevance(file, prompt_context)

        return factors

    def get_weights(self) -> Dict[str, float]:
        """Get weights for fast ranking.

        Returns:
            Factor weights emphasizing keywords and paths
        """
        return {"keyword_match": 0.6, "path_relevance": 0.3, "type_relevance": 0.1}

    def _calculate_type_relevance(self, file: FileAnalysis, prompt_context: PromptContext) -> float:
        """Calculate relevance based on file type.

        Args:
            file: File to analyze
            prompt_context: Prompt context

        Returns:
            Type relevance score
        """
        path_lower = file.path.lower()
        task_type = prompt_context.task_type

        if task_type == "test":
            if "test" in path_lower or "spec" in path_lower:
                return 1.0
            return 0.3
        elif task_type == "debug":
            if "error" in path_lower or "exception" in path_lower or "log" in path_lower:
                return 0.8
            return 0.5
        elif task_type == "feature":
            if "test" in path_lower:
                return 0.2  # Tests less relevant for new features
            return 0.6
        elif task_type == "refactor":
            if file.complexity and file.complexity.cyclomatic > 10:
                return 0.8  # Complex files more relevant for refactoring
            return 0.5
        else:
            return 0.5  # Neutral


class BalancedRankingStrategy(RankingStrategy):
    """Balanced multi-factor ranking.

    This strategy combines multiple ranking factors including keywords,
    structure, git history, and code relationships for comprehensive
    relevance scoring.
    """

    def __init__(self):
        """Initialize balanced ranking strategy."""
        self.logger = get_logger(__name__)
        self._tfidf_calculator = None
        self._import_graph = None

    def rank_file(
        self, file: FileAnalysis, prompt_context: PromptContext, corpus_stats: Dict[str, Any]
    ) -> RankingFactors:
        """Balanced ranking using multiple factors.

        Args:
            file: File to rank
            prompt_context: Parsed prompt information
            corpus_stats: Statistics about the entire codebase

        Returns:
            RankingFactors with multiple scores calculated
        """
        factors = RankingFactors()

        # Enhanced keyword matching with position weighting
        factors.keyword_match = self._calculate_keyword_score(file, prompt_context.keywords)

        # TF-IDF similarity using our custom implementation
        if corpus_stats.get("tfidf_calculator"):
            tfidf_calc = corpus_stats["tfidf_calculator"]
            if file.path in tfidf_calc.document_vectors:
                factors.tfidf_similarity = tfidf_calc.compute_similarity(
                    prompt_context.text, file.path
                )

        # Path structure analysis
        factors.path_relevance = self._analyze_path_structure(file.path, prompt_context)

        # Import centrality (how many files depend on this)
        if corpus_stats.get("import_graph"):
            factors.import_centrality = self._calculate_import_centrality(
                file, corpus_stats["import_graph"]
            )

        # Git activity scores (support optional git_info field on FileAnalysis)
        git_info = getattr(file, "git_info", None)
        if git_info:
            factors.git_recency = self._calculate_git_recency(git_info)
            factors.git_frequency = self._calculate_git_frequency(git_info)

        # Complexity relevance
        if file.complexity:
            factors.complexity_relevance = self._calculate_complexity_relevance(
                file.complexity, prompt_context
            )

        # File type relevance
        factors.type_relevance = self._calculate_type_relevance(file, prompt_context)

        return factors

    def get_weights(self) -> Dict[str, float]:
        """Get weights for balanced ranking.

        Returns:
            Balanced factor weights
        """
        return {
            "keyword_match": 0.25,
            "tfidf_similarity": 0.20,
            "path_relevance": 0.15,
            "import_centrality": 0.15,
            "git_recency": 0.10,
            "git_frequency": 0.05,
            "complexity_relevance": 0.05,
            "type_relevance": 0.05,
        }

    def _calculate_keyword_score(self, file: FileAnalysis, keywords: List[str]) -> float:
        """Calculate sophisticated keyword relevance score.

        Args:
            file: File to analyze
            keywords: List of keywords from prompt

        Returns:
            Keyword relevance score
        """
        if not keywords or not file.content:
            return 0.0

        score = 0.0
        content_lower = file.content.lower()
        content_lines = content_lower.split("\n")

        def _safe_name(obj: Any) -> str:
            try:
                # For unittest.mock.Mock, check _mock_name first as it's more reliable
                n = getattr(obj, "_mock_name", None)
                if isinstance(n, str):
                    return n
                # Fallback to name attribute for regular objects
                n = getattr(obj, "name", None)
                if isinstance(n, str):
                    return n
            except Exception:
                return ""
            return ""

        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_score = 0.0

            # Check filename (highest weight)
            try:
                if keyword_lower in Path(file.path).name.lower():
                    keyword_score += 0.4
            except Exception:
                pass

            # Check imports (high weight)
            for imp in file.imports:
                try:
                    module = getattr(imp, "module", "") or ""
                    if isinstance(module, str) and keyword_lower in module.lower():
                        keyword_score += 0.3
                        break
                except Exception:
                    continue

            # Check class/function names (medium weight)
            if file.classes:
                for cls in file.classes:
                    name = _safe_name(cls)
                    if name and keyword_lower in name.lower():
                        keyword_score += 0.3
                        break

            if file.functions:
                for func in file.functions:
                    name = _safe_name(func)
                    if name and keyword_lower in name.lower():
                        keyword_score += 0.35
                        break

            # Check content with position weighting
            occurrences = 0
            position_weight_sum = 0.0

            for i, line in enumerate(content_lines):
                if keyword_lower in line:
                    occurrences += 1
                    # Earlier lines have higher weight
                    position_weight = 1.0 - (i / max(1, len(content_lines))) * 0.5
                    position_weight_sum += position_weight

            if occurrences > 0:
                # Logarithmic scaling for frequency
                import math as _math

                freq_score = _math.log(1 + occurrences) / _math.log(10)
                keyword_score += min(0.3, freq_score * (position_weight_sum / max(1, occurrences)))

            score += keyword_score

        return min(1.0, score / len(keywords))

    def _analyze_path_structure(self, file_path: str, prompt_context: PromptContext) -> float:
        """Analyze file path for structural relevance.

        Args:
            file_path: Path to the file
            prompt_context: Prompt context

        Returns:
            Path relevance score
        """
        path = Path(file_path)
        path_parts = [p.lower() for p in path.parts]
        score = 0.0

        # Check for keyword matches in path
        for keyword in prompt_context.keywords:
            keyword_lower = keyword.lower()
            for part in path_parts:
                if keyword_lower in part:
                    score += 0.3
                    break

        # Architecture-relevant paths
        architecture_indicators = {
            "api": 0.2,
            "controller": 0.2,
            "service": 0.2,
            "model": 0.2,
            "view": 0.15,
            "handler": 0.2,
            "manager": 0.15,
            "repository": 0.15,
            "dao": 0.15,
            "util": 0.1,
            "helper": 0.1,
            "config": 0.15,
            "core": 0.25,
            "main": 0.25,
            "index": 0.25,
            "app": 0.25,
        }

        for indicator, weight in architecture_indicators.items():
            if any(indicator in part for part in path_parts):
                score += weight
                break

        # Penalize test files unless looking for tests
        if prompt_context.task_type != "test":
            if any("test" in part or "spec" in part for part in path_parts):
                score *= 0.5

        # Penalize deeply nested files
        depth_penalty = max(0, len(path_parts) - 3) * 0.05
        score -= depth_penalty

        return max(0.0, min(1.0, score))

    def _calculate_import_centrality(
        self, file: FileAnalysis, import_graph: Dict[str, Set[str]]
    ) -> float:
        """Calculate how central a file is in the import graph.

        Args:
            file: File to analyze
            import_graph: Dictionary mapping files to their importers

        Returns:
            Import centrality score
        """
        # Count how many files import this file
        file_path = file.path
        importers = import_graph.get(file_path, set())

        if not importers:
            return 0.0

        # Normalize by total files
        total_files = len(import_graph)
        centrality = len(importers) / max(1, total_files)

        # Apply logarithmic scaling for very central files
        if centrality > 0.1:
            centrality = 0.1 + math.log(centrality / 0.1) / 10

        return min(1.0, centrality * 2)  # Scale up

    def _calculate_git_recency(self, git_info: Dict[str, Any]) -> float:
        """Calculate score based on how recently file was modified.

        Args:
            git_info: Git information for the file

        Returns:
            Git recency score
        """
        if not git_info or "last_modified" not in git_info:
            return 0.5  # Neutral if no git info

        try:
            last_modified = datetime.fromisoformat(git_info["last_modified"])
            days_ago = (datetime.now() - last_modified).days

            if days_ago <= 1:
                return 1.0
            elif days_ago <= 7:
                return 0.8
            elif days_ago <= 30:
                return 0.6
            elif days_ago <= 90:
                return 0.4
            elif days_ago <= 365:
                return 0.2
            else:
                return 0.1

        except Exception:
            return 0.5

    def _calculate_git_frequency(self, git_info: Dict[str, Any]) -> float:
        """Calculate score based on how frequently file changes.

        Args:
            git_info: Git information for the file

        Returns:
            Git frequency score
        """
        if not git_info or "commit_count" not in git_info:
            return 0.5

        commit_count = git_info.get("commit_count", 0)

        # Logarithmic scaling
        if commit_count == 0:
            return 0.0
        elif commit_count <= 5:
            return 0.3
        elif commit_count <= 20:
            return 0.5
        elif commit_count <= 50:
            return 0.7
        else:
            return min(1.0, 0.7 + math.log(commit_count / 50) / 10)

    def _calculate_complexity_relevance(
        self, complexity: Any, prompt_context: PromptContext
    ) -> float:
        """Calculate relevance based on code complexity.

        Args:
            complexity: Complexity metrics
            prompt_context: Prompt context

        Returns:
            Complexity relevance score
        """
        if not complexity or not hasattr(complexity, "cyclomatic"):
            return 0.5

        cyclomatic = complexity.cyclomatic
        task_type = prompt_context.task_type

        if task_type == "refactor":
            # High complexity files are very relevant for refactoring
            if cyclomatic > 20:
                return 1.0
            elif cyclomatic > 10:
                return 0.8
            elif cyclomatic > 5:
                return 0.6
            else:
                return 0.3

        elif task_type == "debug":
            # Complex files more likely to have bugs
            if cyclomatic > 15:
                return 0.8
            elif cyclomatic > 10:
                return 0.6
            else:
                return 0.4

        else:
            # Neutral relevance for other tasks
            return 0.5

    def _calculate_type_relevance(self, file: FileAnalysis, prompt_context: PromptContext) -> float:
        """Calculate relevance based on file type and task.

        Args:
            file: File to analyze
            prompt_context: Prompt context

        Returns:
            Type relevance score
        """
        path_lower = file.path.lower()
        task_type = prompt_context.task_type

        # Task-specific relevance
        if task_type == "test":
            if "test" in path_lower or "spec" in path_lower:
                return 1.0
            elif any(test_kw in prompt_context.keywords for test_kw in ["test", "testing", "spec"]):
                return 0.7
            return 0.3

        elif task_type == "debug":
            if any(debug_kw in path_lower for debug_kw in ["error", "exception", "log", "debug"]):
                return 0.9
            return 0.5

        elif task_type == "feature":
            if "test" in path_lower:
                return 0.3  # Tests less relevant for new features
            elif any(
                impl_kw in path_lower for impl_kw in ["impl", "service", "handler", "controller"]
            ):
                return 0.8
            return 0.6

        elif task_type == "refactor":
            # All non-test files potentially relevant
            if "test" not in path_lower:
                return 0.7
            return 0.4

        else:
            return 0.5


class ThoroughRankingStrategy(RankingStrategy):
    """Thorough deep analysis ranking.

    This strategy performs comprehensive analysis including AST parsing,
    semantic similarity (if ML features available), and detailed code
    pattern matching for the most accurate relevance scoring.
    """

    def __init__(self):
        """Initialize thorough ranking strategy."""
        self.logger = get_logger(__name__)
        self._ml_model = None
        self._load_ml_model()

    def _load_ml_model(self):
        """Load ML model for semantic similarity if available."""
        try:
            # Import through package for easier patching in tests
            from tenets.core.ranking.ranker import SentenceTransformer  # type: ignore

            if SentenceTransformer is not None:
                self._ml_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.logger.info("ML model loaded for semantic ranking")
        except ImportError:
            self.logger.debug("ML features not available - install with: pip install tenets[ml]")

    def rank_file(
        self, file: FileAnalysis, prompt_context: PromptContext, corpus_stats: Dict[str, Any]
    ) -> RankingFactors:
        """Thorough ranking with deep analysis.

        Args:
            file: File to rank
            prompt_context: Parsed prompt information
            corpus_stats: Statistics about the entire codebase

        Returns:
            RankingFactors with comprehensive scores
        """
        # Start with balanced ranking
        balanced = BalancedRankingStrategy()
        factors = balanced.rank_file(file, prompt_context, corpus_stats)

        # Add semantic similarity if ML available
        if self._ml_model and file.content:
            factors.semantic_similarity = self._calculate_semantic_similarity(
                file.content, prompt_context.text
            )

        # Deep code pattern analysis
        pattern_scores = self._analyze_code_patterns(file, prompt_context)
        factors.custom_scores.update(pattern_scores)

        # AST-based analysis for supported languages
        if file.structure:
            ast_scores = self._analyze_ast_relevance(file, prompt_context)
            factors.custom_scores.update(ast_scores)

        # Dependency depth analysis
        if corpus_stats.get("dependency_tree"):
            factors.custom_scores["dependency_depth"] = self._calculate_dependency_depth(
                file, corpus_stats["dependency_tree"]
            )

        return factors

    def get_weights(self) -> Dict[str, float]:
        """Get weights for thorough ranking.

        Returns:
            Comprehensive factor weights
        """
        weights = {
            "keyword_match": 0.15,
            "tfidf_similarity": 0.10,
            "semantic_similarity": 0.25,  # High weight if available
            "path_relevance": 0.10,
            "import_centrality": 0.10,
            "git_recency": 0.05,
            "git_frequency": 0.05,
            "complexity_relevance": 0.05,
            "type_relevance": 0.05,
            "code_patterns": 0.05,
            "ast_relevance": 0.05,
        }

        # Adjust weights if ML not available
        if not self._ml_model:
            weights["semantic_similarity"] = 0.0
            weights["keyword_match"] = 0.25
            weights["tfidf_similarity"] = 0.20

        return weights

    def _calculate_semantic_similarity(self, file_content: str, prompt_text: str) -> float:
        """Calculate semantic similarity using ML model.

        Args:
            file_content: Content of the file
            prompt_text: Prompt text

        Returns:
            Semantic similarity score
        """
        if not self._ml_model:
            return 0.0

        try:
            # Truncate content if too long
            max_length = 512  # Model's max sequence length
            if len(file_content) > max_length * 4:
                # Take beginning and end
                file_content = (
                    file_content[: max_length * 2] + " ... " + file_content[-max_length * 2 :]
                )

            # Encode texts
            file_embedding = self._ml_model.encode(file_content, convert_to_tensor=True)
            prompt_embedding = self._ml_model.encode(prompt_text, convert_to_tensor=True)

            # Calculate cosine similarity via package-level symbol for patching
            from tenets.core.ranking.ranker import cosine_similarity  # type: ignore

            similarity = cosine_similarity(
                file_embedding.unsqueeze(0), prompt_embedding.unsqueeze(0)
            ).item()

            # Scale to 0-1 range (similarity is already -1 to 1, but usually positive)
            return max(0.0, similarity)

        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0

    def _analyze_code_patterns(
        self, file: FileAnalysis, prompt_context: PromptContext
    ) -> Dict[str, float]:
        """Analyze code patterns for relevance.

        Args:
            file: File to analyze
            prompt_context: Prompt context

        Returns:
            Dictionary of pattern scores
        """
        scores = {}
        content = file.content or ""

        # Authentication patterns
        if any(kw in prompt_context.keywords for kw in ["auth", "login", "oauth", "jwt", "token"]):
            auth_patterns = [
                r"\bauth[a-z]*\b",
                r"\blogin\b",
                r"\blogout\b",
                r"\btoken\b",
                r"\bjwt\b",
                r"\boauth\b",
                r"\bsession\b",
                r"\bcredential",
                r"\bpassword\b",
            ]

            pattern_count = sum(
                len(re.findall(pattern, content, re.IGNORECASE)) for pattern in auth_patterns
            )

            scores["auth_patterns"] = min(1.0, pattern_count / 20)

        # API patterns
        if any(kw in prompt_context.keywords for kw in ["api", "rest", "endpoint", "route"]):
            api_patterns = [
                r"@(app|router)\.(get|post|put|delete|patch)",
                r"\bapi[/_]",
                r"\bendpoint\b",
                r"\broute[rs]?\b",
                r'(GET|POST|PUT|DELETE|PATCH)\s*[\'"]/',
            ]

            pattern_count = sum(
                len(re.findall(pattern, content, re.IGNORECASE)) for pattern in api_patterns
            )

            scores["api_patterns"] = min(1.0, pattern_count / 15)

        # Database patterns
        if any(kw in prompt_context.keywords for kw in ["database", "db", "sql", "query", "model"]):
            db_patterns = [
                r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b",
                r"\.query\(",
                r"\.find\(",
                r"\.save\(",
                r"\.create\(",
                r"\bmodel[s]?\b",
                r"\bschema\b",
                r"\btable[s]?\b",
            ]

            pattern_count = sum(
                len(re.findall(pattern, content, re.IGNORECASE)) for pattern in db_patterns
            )

            scores["db_patterns"] = min(1.0, pattern_count / 20)

        return scores

    def _analyze_ast_relevance(
        self, file: FileAnalysis, prompt_context: PromptContext
    ) -> Dict[str, float]:
        """Analyze AST structure for relevance.

        Args:
            file: File with structure information
            prompt_context: Prompt context

        Returns:
            Dictionary of AST-based scores
        """
        scores = {}

        if not file.structure:
            return scores

        # Analyze class relevance
        if file.structure.classes:
            class_relevance = 0.0
            for cls in file.structure.classes:
                for keyword in prompt_context.keywords:
                    cls_name = getattr(cls, "name", None) or getattr(cls, "_mock_name", "")
                    if isinstance(cls_name, str) and keyword.lower() in cls_name.lower():
                        class_relevance += 0.5
                        break

            scores["class_relevance"] = min(1.0, class_relevance)

        # Analyze function relevance
        if file.structure.functions:
            function_relevance = 0.0
            for func in file.structure.functions:
                for keyword in prompt_context.keywords:
                    func_name = getattr(func, "name", None) or getattr(func, "_mock_name", "")
                    if isinstance(func_name, str) and keyword.lower() in func_name.lower():
                        function_relevance += 0.3
                        break

            scores["function_relevance"] = min(1.0, function_relevance)

        # Analyze complexity distribution
        if file.structure.functions:
            complex_functions = sum(
                1
                for func in file.structure.functions
                if hasattr(func, "complexity") and func.complexity > 10
            )

            if prompt_context.task_type == "refactor":
                scores["complexity_distribution"] = min(1.0, complex_functions / 5)

        return scores

    def _calculate_dependency_depth(
        self, file: FileAnalysis, dependency_tree: Dict[str, Any]
    ) -> float:
        """Calculate file's depth in dependency tree.

        Args:
            file: File to analyze
            dependency_tree: Project dependency tree

        Returns:
            Dependency depth score
        """
        # Files at the root of dependency tree are often more important
        depth = dependency_tree.get(file.path, {}).get("depth", -1)

        if depth == -1:
            return 0.5  # Unknown
        elif depth == 0:
            return 1.0  # Root level - very important
        elif depth == 1:
            return 0.8
        elif depth == 2:
            return 0.6
        elif depth == 3:
            return 0.4
        else:
            return 0.2  # Deep dependency


class RelevanceRanker:
    """Main relevance ranking system.

    Orchestrates the ranking process, managing different strategies and
    coordinating the analysis of files to produce ranked results.

    Attributes:
        config: TenetsConfig instance
        logger: Logger instance
        strategies: Dictionary of available ranking strategies
        _corpus_analyzer: Analyzer for corpus-wide statistics
        _custom_rankers: List of custom ranking functions
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the relevance ranker.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize ranking strategies
        self.strategies = {
            RankingAlgorithm.FAST: FastRankingStrategy(),
            RankingAlgorithm.BALANCED: BalancedRankingStrategy(),
            RankingAlgorithm.THOROUGH: ThoroughRankingStrategy(),
        }

        # Custom rankers
        self._custom_rankers = []

        # Thread pool for parallel ranking
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.ranking.workers)

        self.logger.info(f"RelevanceRanker initialized with {len(self.strategies)} strategies")

    def rank_files(
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        algorithm: str = "balanced",
        parallel: bool = True,
    ) -> List[FileAnalysis]:
        """Rank files by relevance to prompt.

        This is the main entry point for ranking files. It analyzes the corpus,
        applies the selected ranking strategy, and returns files sorted by
        relevance.

        Args:
            files: List of files to rank
            prompt_context: Parsed prompt information
            algorithm: Ranking algorithm to use
            parallel: Whether to rank files in parallel

        Returns:
            List of FileAnalysis objects sorted by relevance (highest first)

        Raises:
            ValueError: If algorithm is not recognized
        """
        if not files:
            return []

        self.logger.info(f"Ranking {len(files)} files using {algorithm} algorithm")

        # Get strategy
        strategy = self._get_strategy(algorithm)
        if not strategy:
            raise ValueError(f"Unknown ranking algorithm: {algorithm}")

        # Analyze corpus for statistics (including TF-IDF setup)
        corpus_stats = self._analyze_corpus(files, prompt_context)

        # Rank files
        ranked_files = []

        if parallel and len(files) > 10:
            # Parallel ranking for large sets
            futures = []

            for file in files:
                future = self._executor.submit(
                    self._rank_single_file, file, prompt_context, corpus_stats, strategy
                )
                futures.append((file, future))

            # Collect results
            for file, future in futures:
                try:
                    ranked_file = future.result(timeout=5.0)
                    ranked_files.append(ranked_file)
                except Exception as e:
                    self.logger.warning(f"Failed to rank {file.path}: {e}")
                    # Add with zero score
                    ranked_files.append(
                        RankedFile(
                            analysis=file,
                            score=0.0,
                            factors=RankingFactors(),
                            explanation=f"Ranking failed: {str(e)}",
                        )
                    )
        else:
            # Sequential ranking with error isolation per file (match parallel behavior)
            for file in files:
                try:
                    ranked_file = self._rank_single_file(
                        file, prompt_context, corpus_stats, strategy
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to rank {file.path}: {e}")
                    ranked_file = RankedFile(
                        analysis=file,
                        score=0.0,
                        factors=RankingFactors(),
                        explanation=f"Ranking failed: {str(e)}",
                    )
                ranked_files.append(ranked_file)

        # Apply custom rankers
        for custom_ranker in self._custom_rankers:
            ranked_files = custom_ranker(ranked_files, prompt_context)

        # Sort by score
        ranked_files.sort(reverse=True)

        # Filter by threshold
        threshold = self.config.ranking.threshold
        filtered_files = [rf.analysis for rf in ranked_files if rf.score >= threshold]

        # Update relevance scores in FileAnalysis objects
        for i, rf in enumerate(ranked_files):
            if rf.score >= threshold:
                rf.analysis.relevance_score = rf.score
                rf.analysis.relevance_rank = i + 1

        self.logger.info(
            f"Ranking complete: {len(filtered_files)}/{len(files)} files "
            f"above threshold ({threshold})"
        )

        return filtered_files

    def _get_strategy(self, algorithm: str) -> Optional[RankingStrategy]:
        """Get ranking strategy by name.

        Args:
            algorithm: Algorithm name

        Returns:
            RankingStrategy instance or None
        """
        # Try enum first
        try:
            algo_enum = RankingAlgorithm(algorithm)
            return self.strategies.get(algo_enum)
        except ValueError:
            pass

        # Try direct lookup
        for key, strategy in self.strategies.items():
            if key.value == algorithm:
                return strategy

        return None

    def _rank_single_file(
        self,
        file: FileAnalysis,
        prompt_context: PromptContext,
        corpus_stats: Dict[str, Any],
        strategy: RankingStrategy,
    ) -> RankedFile:
        """Rank a single file.

        Args:
            file: File to rank
            prompt_context: Prompt context
            corpus_stats: Corpus statistics
            strategy: Ranking strategy to use

        Returns:
            RankedFile with score and factors
        """
        # Calculate ranking factors
        factors = strategy.rank_file(file, prompt_context, corpus_stats)

        # Get weights
        weights = strategy.get_weights()

        # Calculate final score
        score = factors.get_weighted_score(weights)

        # Generate explanation
        explanation = self._generate_explanation(factors, weights)

        return RankedFile(analysis=file, score=score, factors=factors, explanation=explanation)

    def _analyze_corpus(
        self, files: List[FileAnalysis], prompt_context: PromptContext
    ) -> Dict[str, Any]:
        """Analyze the corpus of files for statistics including TF-IDF.

        Args:
            files: All files in the corpus
            prompt_context: Prompt context for configuration

        Returns:
            Dictionary of corpus-wide statistics
        """
        stats = {
            "total_files": len(files),
            "languages": Counter(),
            "file_sizes": [],
            "import_graph": defaultdict(set),
            "dependency_tree": {},
        }

        # Initialize TF-IDF calculator
        # Check if we should use stopwords (could be from config or prompt context)
        use_stopwords = getattr(self.config, "use_tfidf_stopwords", False)
        tfidf_calc = TFIDFCalculator(use_stopwords=use_stopwords)

        # Build TF-IDF corpus
        documents = []
        for file in files:
            if file.content:
                documents.append((file.path, file.content))

        tfidf_calc.build_corpus(documents)
        stats["tfidf_calculator"] = tfidf_calc

        # Analyze each file
        for file in files:
            # Language distribution
            stats["languages"][file.language] += 1

            # File sizes
            stats["file_sizes"].append(file.size)

            # Build import graph
            for imp in file.imports:
                if hasattr(imp, "module"):
                    # Try to resolve import to file
                    imported_file = self._resolve_import(imp.module, file.path, files)
                    if imported_file:
                        stats["import_graph"][imported_file].add(file.path)

        # Calculate additional statistics
        stats["avg_file_size"] = (
            sum(stats["file_sizes"]) / len(stats["file_sizes"]) if stats["file_sizes"] else 0
        )
        stats["total_imports"] = sum(len(importers) for importers in stats["import_graph"].values())

        return stats

    def _resolve_import(
        self, module_name: str, from_file: str, all_files: List[FileAnalysis]
    ) -> Optional[str]:
        """Resolve an import to a file path.

        Args:
            module_name: Name of imported module
            from_file: File doing the importing
            all_files: All files in the project

        Returns:
            Resolved file path or None
        """
        # Simple resolution - would be enhanced for real use
        # Look for file with matching name
        module_parts = module_name.split(".")

        for file in all_files:
            file_path = Path(file.path)
            file_stem = file_path.stem

            # Check if file name matches module
            if file_stem == module_parts[-1]:
                return file.path

            # Check if path contains module structure
            if all(part in str(file_path) for part in module_parts):
                return file.path

        return None

    def _generate_explanation(self, factors: RankingFactors, weights: Dict[str, float]) -> str:
        """Generate human-readable explanation of ranking.

        Args:
            factors: Ranking factors
            weights: Factor weights

        Returns:
            Explanation string
        """
        explanations = []

        # Sort factors by contribution to score
        factor_contributions = []

        for factor_name, weight in weights.items():
            if hasattr(factors, factor_name):
                value = getattr(factors, factor_name)
                contribution = value * weight
                if contribution > 0.01:  # Only include significant factors
                    factor_contributions.append((factor_name, value, contribution))

        factor_contributions.sort(key=lambda x: x[2], reverse=True)

        # Generate explanations for top factors
        for factor_name, value, contribution in factor_contributions[:3]:
            if factor_name == "keyword_match":
                explanations.append(f"Strong keyword match ({value:.2f})")
            elif factor_name == "tfidf_similarity":
                explanations.append(f"High TF-IDF similarity ({value:.2f})")
            elif factor_name == "semantic_similarity":
                explanations.append(f"High semantic similarity ({value:.2f})")
            elif factor_name == "import_centrality":
                explanations.append(f"Central to import graph ({value:.2f})")
            elif factor_name == "git_recency":
                explanations.append(f"Recently modified ({value:.2f})")
            elif factor_name == "path_relevance":
                explanations.append(f"Relevant path structure ({value:.2f})")

        return "; ".join(explanations) if explanations else "Low relevance"

    def register_custom_ranker(
        self, ranker_func: Callable[[List[RankedFile], PromptContext], List[RankedFile]]
    ):
        """Register a custom ranking function.

        Custom rankers are applied after the main ranking strategy and can
        adjust scores based on project-specific logic.

        Args:
            ranker_func: Function that takes ranked files and returns modified list
        """
        self._custom_rankers.append(ranker_func)
        self.logger.info(f"Registered custom ranker: {ranker_func.__name__}")

    def shutdown(self):
        """Shutdown the ranker and clean up resources."""
        self._executor.shutdown(wait=True)
        self.logger.info("RelevanceRanker shutdown complete")
