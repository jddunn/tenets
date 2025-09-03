"""Summarization strategies with NLP integration.

This module provides various summarization strategies that leverage the
centralized NLP components for improved text processing and analysis.

Strategies:
- ExtractiveStrategy: Selects important sentences using NLP keyword extraction
- CompressiveStrategy: Removes redundancy using NLP tokenization
- TextRankStrategy: Graph-based ranking with NLP preprocessing
- TransformerStrategy: Neural summarization (requires ML)
- NLPEnhancedStrategy: Advanced strategy using all NLP features
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple

from tenets.utils.logger import get_logger

# Import centralized NLP components
try:
    from tenets.core.nlp.embeddings import create_embedding_model
    from tenets.core.nlp.keyword_extractor import KeywordExtractor, TFIDFCalculator
    from tenets.core.nlp.similarity import SemanticSimilarity
    from tenets.core.nlp.stopwords import StopwordManager
    from tenets.core.nlp.tokenizer import TextTokenizer

    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

# Try ML imports
try:
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SummarizationStrategy(ABC):
    """Abstract base class for summarization strategies."""

    name: str = "base"
    description: str = "Base summarization strategy"
    requires_ml: bool = False

    @abstractmethod
    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Summarize text.

        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Summarized text
        """
        pass


class ExtractiveStrategy(SummarizationStrategy):
    """Extractive summarization using NLP components.

    Selects the most important sentences based on keyword density,
    position, and optionally semantic similarity. Uses centralized
    NLP components for improved sentence scoring.
    """

    name = "extractive"
    description = "Extract important sentences using NLP analysis"
    requires_ml = False

    def __init__(self, use_nlp: bool = True):
        """Initialize extractive strategy.

        Args:
            use_nlp: Whether to use NLP components for enhanced extraction
        """
        self.logger = get_logger(__name__)
        self.use_nlp = use_nlp and NLP_AVAILABLE

        if self.use_nlp:
            # Initialize NLP components
            self.keyword_extractor = KeywordExtractor(
                use_stopwords=True,
                stopword_set="prompt",  # Use aggressive stopwords for summarization
            )
            self.tokenizer = TextTokenizer(use_stopwords=True)
            self.logger.info("ExtractiveStrategy using NLP components")
        else:
            self.logger.info("ExtractiveStrategy using basic extraction")

    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Extract important sentences to create summary.

        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return text

        # Score sentences
        if self.use_nlp:
            scores = self._score_sentences_nlp(sentences, text)
        else:
            scores = self._score_sentences_basic(sentences)

        # Select top sentences
        target_length = int(len(text) * target_ratio)
        if max_length:
            target_length = min(target_length, max_length)

        selected = self._select_sentences(sentences, scores, target_length, min_length)

        return " ".join(selected)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting (could use NLTK if available)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _score_sentences_nlp(self, sentences: List[str], full_text: str) -> List[float]:
        """Score sentences using NLP components.

        Args:
            sentences: List of sentences
            full_text: Full text for context

        Returns:
            List of scores
        """
        # Extract keywords from full text
        keywords = self.keyword_extractor.extract(full_text, max_keywords=20, include_scores=True)

        keyword_dict = dict(keywords) if keywords else {}

        scores = []
        for i, sentence in enumerate(sentences):
            score = 0.0

            # Position score (earlier sentences more important)
            position_score = 1.0 - (i / len(sentences)) * 0.5
            score += position_score * 0.3

            # Keyword density score
            sentence_tokens = self.tokenizer.tokenize(sentence)
            if sentence_tokens:
                keyword_score = sum(keyword_dict.get(token, 0) for token in sentence_tokens) / len(
                    sentence_tokens
                )
                score += keyword_score * 0.5

            # Length score (prefer medium-length sentences)
            length_score = min(1.0, len(sentence) / 100)
            score += length_score * 0.2

            scores.append(score)

        return scores

    def _score_sentences_basic(self, sentences: List[str]) -> List[float]:
        """Score sentences using basic heuristics.

        Args:
            sentences: List of sentences

        Returns:
            List of scores
        """
        scores = []

        # Simple word frequency
        words = []
        for sentence in sentences:
            words.extend(sentence.lower().split())

        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1

        for i, sentence in enumerate(sentences):
            score = 0.0

            # Position score
            position_score = 1.0 - (i / len(sentences)) * 0.5
            score += position_score * 0.3

            # Word frequency score
            sentence_words = sentence.lower().split()
            if sentence_words:
                freq_score = sum(word_freq.get(word, 0) for word in sentence_words) / len(
                    sentence_words
                )
                score += freq_score * 0.5

            # Length score
            length_score = min(1.0, len(sentence) / 100)
            score += length_score * 0.2

            scores.append(score)

        return scores

    def _select_sentences(
        self,
        sentences: List[str],
        scores: List[float],
        target_length: int,
        min_length: Optional[int],
    ) -> List[str]:
        """Select top sentences to meet target length.

        Args:
            sentences: List of sentences
            scores: Sentence scores
            target_length: Target total length
            min_length: Minimum length

        Returns:
            Selected sentences in original order
        """
        # Sort by score
        sentence_scores = list(zip(sentences, scores, range(len(sentences))))
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        selected = []
        selected_indices = []
        current_length = 0

        for sentence, score, idx in sentence_scores:
            sentence_length = len(sentence)

            if current_length + sentence_length <= target_length:
                selected.append(sentence)
                selected_indices.append(idx)
                current_length += sentence_length
            elif min_length and current_length < min_length:
                # Add anyway to meet minimum
                selected.append(sentence)
                selected_indices.append(idx)
                current_length += sentence_length
            else:
                break

        # Sort back to original order
        selected_indices.sort()
        return [sentences[i] for i in selected_indices]


class CompressiveStrategy(SummarizationStrategy):
    """Compressive summarization using NLP tokenization.

    Removes redundant words and phrases while maintaining meaning.
    Uses NLP tokenizer for better word processing.
    """

    name = "compressive"
    description = "Remove redundancy using NLP tokenization"
    requires_ml = False

    def __init__(self, use_nlp: bool = True):
        """Initialize compressive strategy.

        Args:
            use_nlp: Whether to use NLP components
        """
        self.logger = get_logger(__name__)
        self.use_nlp = use_nlp and NLP_AVAILABLE

        if self.use_nlp:
            self.tokenizer = TextTokenizer(use_stopwords=True)
            self.stopword_manager = StopwordManager()
            self.stopwords = self.stopword_manager.get_set("prompt")
            self.logger.info("CompressiveStrategy using NLP components")

    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Compress text by removing redundancy.

        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Compressed text
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        compressed = []
        seen_concepts = set()
        current_length = 0
        target_length = int(len(text) * target_ratio)

        if max_length:
            target_length = min(target_length, max_length)

        for sentence in sentences:
            # Compress sentence
            if self.use_nlp:
                compressed_sent = self._compress_sentence_nlp(sentence, seen_concepts)
            else:
                compressed_sent = self._compress_sentence_basic(sentence, seen_concepts)

            if compressed_sent:
                compressed.append(compressed_sent)
                current_length += len(compressed_sent)

                # Update seen concepts
                if self.use_nlp:
                    tokens = self.tokenizer.tokenize(compressed_sent)
                    seen_concepts.update(tokens)
                else:
                    words = compressed_sent.lower().split()
                    seen_concepts.update(words)

                if current_length >= target_length:
                    break

        result = " ".join(compressed)

        # Check minimum length
        if min_length and len(result) < min_length:
            # Add more sentences
            for sentence in sentences[len(compressed) :]:
                compressed.append(sentence)
                if len(" ".join(compressed)) >= min_length:
                    break
            result = " ".join(compressed)

        return result

    def _compress_sentence_nlp(self, sentence: str, seen_concepts: Set[str]) -> str:
        """Compress sentence using NLP components.

        Args:
            sentence: Input sentence
            seen_concepts: Already seen concepts

        Returns:
            Compressed sentence
        """
        tokens = self.tokenizer.tokenize(sentence)

        # Remove redundant tokens
        important_tokens = []
        for token in tokens:
            if token not in seen_concepts or len(important_tokens) < 3:
                important_tokens.append(token)

        if not important_tokens:
            return ""

        # Reconstruct sentence (simplified)
        return " ".join(important_tokens)

    def _compress_sentence_basic(self, sentence: str, seen_concepts: Set[str]) -> str:
        """Compress sentence using basic method.

        Args:
            sentence: Input sentence
            seen_concepts: Already seen concepts

        Returns:
            Compressed sentence
        """
        words = sentence.split()

        # Remove common words and redundancy
        important_words = []
        for word in words:
            word_lower = word.lower()
            if (word_lower not in seen_concepts or len(important_words) < 3) and len(word) > 2:
                important_words.append(word)

        if not important_words:
            return ""

        return " ".join(important_words)


class TextRankStrategy(SummarizationStrategy):
    """TextRank summarization with NLP preprocessing.

    Graph-based ranking algorithm that uses NLP components for
    better text preprocessing and similarity computation.
    """

    name = "textrank"
    description = "Graph-based summarization with NLP preprocessing"
    requires_ml = True  # Requires sklearn

    def __init__(self, use_nlp: bool = True):
        """Initialize TextRank strategy.

        Args:
            use_nlp: Whether to use NLP components
        """
        self.logger = get_logger(__name__)
        self.use_nlp = use_nlp and NLP_AVAILABLE and SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "TextRank requires scikit-learn. Install with: pip install scikit-learn"
            )

        if self.use_nlp:
            self.tfidf_calc = TFIDFCalculator(use_stopwords=True)
            self.logger.info("TextRankStrategy using NLP components")

    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Summarize using TextRank algorithm.

        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            TextRank summary
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return text

        # Build similarity matrix
        if self.use_nlp:
            similarity_matrix = self._build_similarity_matrix_nlp(sentences)
        else:
            similarity_matrix = self._build_similarity_matrix_sklearn(sentences)

        # Calculate scores using PageRank-style algorithm
        scores = self._calculate_scores(similarity_matrix)

        # Select top sentences
        target_length = int(len(text) * target_ratio)
        if max_length:
            target_length = min(target_length, max_length)

        ranked_sentences = sorted(
            zip(sentences, scores, range(len(sentences))), key=lambda x: x[1], reverse=True
        )

        selected = []
        selected_indices = []
        current_length = 0

        for sentence, score, idx in ranked_sentences:
            if current_length + len(sentence) <= target_length:
                selected.append(sentence)
                selected_indices.append(idx)
                current_length += len(sentence)
            elif min_length and current_length < min_length:
                selected.append(sentence)
                selected_indices.append(idx)
                current_length += len(sentence)
            else:
                break

        # Sort back to original order
        selected_indices.sort()
        return " ".join([sentences[i] for i in selected_indices])

    def _build_similarity_matrix_nlp(self, sentences: List[str]) -> np.ndarray:
        """Build similarity matrix using NLP components.

        Args:
            sentences: List of sentences

        Returns:
            Similarity matrix
        """
        n = len(sentences)
        matrix = np.zeros((n, n))

        # Add sentences to TF-IDF calculator
        for i, sentence in enumerate(sentences):
            self.tfidf_calc.add_document(str(i), sentence)

        # Calculate similarities
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.tfidf_calc.compute_similarity(sentences[i], str(j))
                matrix[i][j] = similarity
                matrix[j][i] = similarity

        return matrix

    def _build_similarity_matrix_sklearn(self, sentences: List[str]) -> np.ndarray:
        """Build similarity matrix using sklearn.

        Args:
            sentences: List of sentences

        Returns:
            Similarity matrix
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        return cosine_similarity(tfidf_matrix)

    def _calculate_scores(self, similarity_matrix: np.ndarray) -> List[float]:
        """Calculate TextRank scores.

        Args:
            similarity_matrix: Sentence similarity matrix

        Returns:
            List of scores
        """
        n = len(similarity_matrix)
        scores = np.ones(n) / n
        damping = 0.85

        # Power iteration
        for _ in range(100):
            new_scores = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        new_scores[i] += similarity_matrix[j][i] * scores[j]

            new_scores = damping * new_scores + (1 - damping) / n

            # Check convergence
            if np.allclose(scores, new_scores, atol=1e-4):
                break

            scores = new_scores

        return scores.tolist()


class TransformerStrategy(SummarizationStrategy):
    """Transformer-based neural summarization.

    Uses pre-trained transformer models for high-quality
    abstractive summarization.
    """

    name = "transformer"
    description = "Neural summarization using transformers"
    requires_ml = True

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """Initialize transformer strategy.

        Args:
            model_name: HuggingFace model name
        """
        self.logger = get_logger(__name__)

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available. Install with: pip install transformers")

        self.model_name = model_name
        self.summarizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the transformer model."""
        try:
            self.summarizer = pipeline("summarization", model=self.model_name)
            self.logger.info(f"Loaded transformer model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load transformer model: {e}")
            raise

    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Summarize using transformer model.

        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Neural summary
        """
        if not self.summarizer:
            raise RuntimeError("Transformer model not initialized")

        # Calculate target lengths
        target_max = int(len(text) * target_ratio)
        if max_length:
            target_max = min(target_max, max_length)

        target_min = min_length or int(target_max * 0.5)

        # Adjust for model tokens (roughly 1 token = 4 chars)
        max_tokens = min(target_max // 4, 512)
        min_tokens = target_min // 4

        try:
            result = self.summarizer(
                text, max_length=max_tokens, min_length=min_tokens, do_sample=False
            )

            return result[0]["summary_text"]

        except Exception as e:
            self.logger.error(f"Transformer summarization failed: {e}")
            # Fallback to extractive
            extractive = ExtractiveStrategy()
            return extractive.summarize(text, target_ratio, max_length, min_length)


class NLPEnhancedStrategy(SummarizationStrategy):
    """Advanced summarization using all NLP features.

    Combines multiple NLP components for advanced extractive
    summarization with semantic understanding.
    """

    name = "nlp_enhanced"
    description = "Advanced summarization with full NLP integration"
    requires_ml = True  # Requires embeddings

    def __init__(self):
        """Initialize NLP-enhanced strategy."""
        self.logger = get_logger(__name__)

        if not NLP_AVAILABLE:
            raise ImportError("NLP components not available")

        # Initialize all NLP components
        self.keyword_extractor = KeywordExtractor(
            use_yake=True, use_stopwords=True, stopword_set="prompt"
        )
        self.tokenizer = TextTokenizer(use_stopwords=True)
        self.tfidf_calc = TFIDFCalculator(use_stopwords=True)

        # Try to initialize embeddings for semantic similarity
        try:
            self.embedding_model = create_embedding_model()
            self.semantic_sim = SemanticSimilarity(self.embedding_model)
            self.use_embeddings = True
            self.logger.info("NLPEnhancedStrategy using embeddings")
        except Exception as e:
            self.logger.warning(f"Embeddings not available: {e}")
            self.use_embeddings = False

    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> str:
        """Summarize using comprehensive NLP analysis.

        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            NLP-enhanced summary
        """
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        # Extract key concepts
        keywords = self.keyword_extractor.extract(text, max_keywords=20, include_scores=True)
        keyword_dict = dict(keywords) if keywords else {}

        # Score sentences with multiple factors
        scores = []
        for i, sentence in enumerate(sentences):
            score = 0.0

            # 1. Keyword relevance (30%)
            tokens = self.tokenizer.tokenize(sentence)
            if tokens:
                keyword_score = sum(keyword_dict.get(t, 0) for t in tokens) / len(tokens)
                score += keyword_score * 0.3

            # 2. Position importance (20%)
            if i == 0:  # First sentence
                score += 0.2
            elif i == len(sentences) - 1:  # Last sentence
                score += 0.1
            else:
                score += (1.0 - i / len(sentences)) * 0.1

            # 3. TF-IDF relevance (25%)
            self.tfidf_calc.add_document(f"sent_{i}", sentence)

            # 4. Semantic similarity to document (25% if available)
            if self.use_embeddings:
                try:
                    doc_sim = self.semantic_sim.compute(sentence, text)
                    score += doc_sim * 0.25
                except Exception:
                    pass

            scores.append(score)

        # Add TF-IDF scores
        for i, sentence in enumerate(sentences):
            tfidf_score = self.tfidf_calc.compute_similarity(text, f"sent_{i}")
            scores[i] += tfidf_score * 0.25

        # Select diverse sentences (avoid redundancy)
        target_length = int(len(text) * target_ratio)
        if max_length:
            target_length = min(target_length, max_length)

        selected = self._select_diverse_sentences(sentences, scores, target_length, min_length)

        return " ".join(selected)

    def _select_diverse_sentences(
        self,
        sentences: List[str],
        scores: List[float],
        target_length: int,
        min_length: Optional[int],
    ) -> List[str]:
        """Select diverse high-scoring sentences.

        Args:
            sentences: List of sentences
            scores: Sentence scores
            target_length: Target total length
            min_length: Minimum length

        Returns:
            Selected sentences in original order
        """
        # Sort by score
        sentence_data = list(zip(sentences, scores, range(len(sentences))))
        sentence_data.sort(key=lambda x: x[1], reverse=True)

        selected = []
        selected_indices = []
        selected_tokens = set()
        current_length = 0

        for sentence, score, idx in sentence_data:
            # Check diversity (avoid redundancy)
            tokens = set(self.tokenizer.tokenize(sentence))

            # Calculate overlap with already selected
            if selected_tokens:
                overlap = len(tokens & selected_tokens) / len(tokens)
                if overlap > 0.7:  # Skip if too similar
                    continue

            if current_length + len(sentence) <= target_length:
                selected.append(sentence)
                selected_indices.append(idx)
                selected_tokens.update(tokens)
                current_length += len(sentence)
            elif min_length and current_length < min_length:
                selected.append(sentence)
                selected_indices.append(idx)
                selected_tokens.update(tokens)
                current_length += len(sentence)
            else:
                break

        # Sort back to original order
        selected_indices.sort()
        return [sentences[i] for i in selected_indices]
