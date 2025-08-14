"""Ranking factors and scored file models.

This module defines the data structures for ranking factors and scored files.
It provides a comprehensive set of factors that contribute to relevance scoring,
along with utilities for calculating weighted scores and generating explanations.

The ranking system uses multiple orthogonal factors to determine file relevance,
allowing for flexible and accurate scoring across different use cases.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.models.analysis import FileAnalysis
from tenets.utils.logger import get_logger


class FactorWeight(Enum):
    """Standard weight presets for ranking factors.

    These presets provide balanced weights for different use cases.
    Can be overridden with custom weights in configuration.
    """

    # Individual factor weights for fine-grained control
    KEYWORD_MATCH = 0.25
    TFIDF_SIMILARITY = 0.20
    BM25_SCORE = 0.15
    PATH_RELEVANCE = 0.15
    IMPORT_CENTRALITY = 0.10
    GIT_RECENCY = 0.05
    GIT_FREQUENCY = 0.05
    COMPLEXITY_RELEVANCE = 0.05
    SEMANTIC_SIMILARITY = 0.25  # When ML is available
    TYPE_RELEVANCE = 0.10
    CODE_PATTERNS = 0.10
    AST_RELEVANCE = 0.10
    DEPENDENCY_DEPTH = 0.05


@dataclass
class RankingFactors:
    """Comprehensive ranking factors for a file.

    Each factor represents a different dimension of relevance. The final
    relevance score is computed as a weighted sum of these factors.

    Factors are grouped into categories:
    - Text-based: keyword_match, tfidf_similarity, bm25_score
    - Structure-based: path_relevance, import_centrality, dependency_depth
    - Git-based: git_recency, git_frequency, git_author_relevance
    - Complexity-based: complexity_relevance, maintainability_score
    - Semantic: semantic_similarity (requires ML)
    - Pattern-based: code_patterns, ast_relevance
    - Custom: custom_scores for project-specific factors

    Attributes:
        keyword_match: Direct keyword matching score (0-1)
        tfidf_similarity: TF-IDF cosine similarity score (0-1)
        bm25_score: BM25 relevance score (0-1)
        path_relevance: File path relevance to query (0-1)
        import_centrality: How central file is in import graph (0-1)
        git_recency: How recently file was modified (0-1)
        git_frequency: How frequently file changes (0-1)
        git_author_relevance: Relevance based on commit authors (0-1)
        complexity_relevance: Relevance based on code complexity (0-1)
        maintainability_score: Code maintainability score (0-1)
        semantic_similarity: ML-based semantic similarity (0-1)
        type_relevance: Relevance based on file type (0-1)
        code_patterns: Pattern matching score (0-1)
        ast_relevance: AST structure relevance (0-1)
        dependency_depth: Dependency tree depth score (0-1)
        test_coverage: Test coverage relevance (0-1)
        documentation_score: Documentation quality score (0-1)
        custom_scores: Dictionary of custom factor scores
        metadata: Additional metadata about factor calculation
    """

    # Text-based factors
    keyword_match: float = 0.0
    tfidf_similarity: float = 0.0
    bm25_score: float = 0.0

    # Structure-based factors
    path_relevance: float = 0.0
    import_centrality: float = 0.0
    dependency_depth: float = 0.0

    # Git-based factors
    git_recency: float = 0.0
    git_frequency: float = 0.0
    git_author_relevance: float = 0.0

    # Complexity-based factors
    complexity_relevance: float = 0.0
    maintainability_score: float = 0.0

    # Semantic factors (ML)
    semantic_similarity: float = 0.0

    # Pattern-based factors
    type_relevance: float = 0.0
    code_patterns: float = 0.0
    ast_relevance: float = 0.0

    # Quality factors
    test_coverage: float = 0.0
    documentation_score: float = 0.0

    # Custom and metadata
    custom_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_weighted_score(self, weights: Dict[str, float], normalize: bool = True) -> float:
        """Calculate weighted relevance score.

        Args:
            weights: Dictionary mapping factor names to weights
            normalize: Whether to normalize final score to [0, 1]

        Returns:
            Weighted relevance score
        """
        score = 0.0
        total_weight = 0.0

        # Map attribute names to values
        factor_values = {
            "keyword_match": self.keyword_match,
            "tfidf_similarity": self.tfidf_similarity,
            "bm25_score": self.bm25_score,
            "path_relevance": self.path_relevance,
            "import_centrality": self.import_centrality,
            "dependency_depth": self.dependency_depth,
            "git_recency": self.git_recency,
            "git_frequency": self.git_frequency,
            "git_author_relevance": self.git_author_relevance,
            "complexity_relevance": self.complexity_relevance,
            "maintainability_score": self.maintainability_score,
            "semantic_similarity": self.semantic_similarity,
            "type_relevance": self.type_relevance,
            "code_patterns": self.code_patterns,
            "ast_relevance": self.ast_relevance,
            "test_coverage": self.test_coverage,
            "documentation_score": self.documentation_score,
        }

        # Add standard factors
        for factor_name, factor_value in factor_values.items():
            if factor_name in weights:
                weight = weights[factor_name]
                score += factor_value * weight
                total_weight += weight

        # Add custom factors
        for custom_name, custom_value in self.custom_scores.items():
            if custom_name in weights:
                weight = weights[custom_name]
                score += custom_value * weight
                total_weight += weight

        # Normalize if requested and weights exist
        if normalize and total_weight > 0:
            score = score / total_weight

        return max(0.0, min(1.0, score))

    def get_top_factors(
        self, weights: Dict[str, float], n: int = 5
    ) -> List[Tuple[str, float, float]]:
        """Get the top contributing factors.

        Args:
            weights: Factor weights
            n: Number of top factors to return

        Returns:
            List of (factor_name, value, contribution) tuples
        """
        contributions = []

        # Calculate contributions for all factors
        factor_values = {
            "keyword_match": self.keyword_match,
            "tfidf_similarity": self.tfidf_similarity,
            "bm25_score": self.bm25_score,
            "path_relevance": self.path_relevance,
            "import_centrality": self.import_centrality,
            "dependency_depth": self.dependency_depth,
            "git_recency": self.git_recency,
            "git_frequency": self.git_frequency,
            "git_author_relevance": self.git_author_relevance,
            "complexity_relevance": self.complexity_relevance,
            "maintainability_score": self.maintainability_score,
            "semantic_similarity": self.semantic_similarity,
            "type_relevance": self.type_relevance,
            "code_patterns": self.code_patterns,
            "ast_relevance": self.ast_relevance,
            "test_coverage": self.test_coverage,
            "documentation_score": self.documentation_score,
        }

        for factor_name, factor_value in factor_values.items():
            if factor_name in weights and factor_value > 0:
                contribution = factor_value * weights[factor_name]
                contributions.append((factor_name, factor_value, contribution))

        # Add custom factors
        for custom_name, custom_value in self.custom_scores.items():
            if custom_name in weights and custom_value > 0:
                contribution = custom_value * weights[custom_name]
                contributions.append((custom_name, custom_value, contribution))

        # Sort by contribution
        contributions.sort(key=lambda x: x[2], reverse=True)

        return contributions[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert factors to dictionary representation.

        Returns:
            Dictionary with all factor values
        """
        return {
            "keyword_match": self.keyword_match,
            "tfidf_similarity": self.tfidf_similarity,
            "bm25_score": self.bm25_score,
            "path_relevance": self.path_relevance,
            "import_centrality": self.import_centrality,
            "dependency_depth": self.dependency_depth,
            "git_recency": self.git_recency,
            "git_frequency": self.git_frequency,
            "git_author_relevance": self.git_author_relevance,
            "complexity_relevance": self.complexity_relevance,
            "maintainability_score": self.maintainability_score,
            "semantic_similarity": self.semantic_similarity,
            "type_relevance": self.type_relevance,
            "code_patterns": self.code_patterns,
            "ast_relevance": self.ast_relevance,
            "test_coverage": self.test_coverage,
            "documentation_score": self.documentation_score,
            "custom_scores": self.custom_scores,
            "metadata": self.metadata,
        }


@dataclass
class RankedFile:
    """A file with its relevance ranking.

    Combines a FileAnalysis with ranking scores and metadata.
    Provides utilities for comparison, explanation generation,
    and result formatting.

    Attributes:
        analysis: The FileAnalysis object
        score: Overall relevance score (0-1)
        factors: Detailed ranking factors
        explanation: Human-readable ranking explanation
        confidence: Confidence in the ranking (0-1)
        rank: Position in ranked list (1-based)
        metadata: Additional ranking metadata
    """

    analysis: FileAnalysis
    score: float
    factors: RankingFactors
    explanation: str = ""
    confidence: float = 1.0
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "RankedFile") -> bool:
        """Compare by score for sorting."""
        return self.score < other.score

    def __le__(self, other: "RankedFile") -> bool:
        """Compare by score for sorting."""
        return self.score <= other.score

    def __gt__(self, other: "RankedFile") -> bool:
        """Compare by score for sorting."""
        return self.score > other.score

    def __ge__(self, other: "RankedFile") -> bool:
        """Compare by score for sorting."""
        return self.score >= other.score

    def __eq__(self, other: object) -> bool:
        """Check equality by file path and score."""
        if not isinstance(other, RankedFile):
            return False
        return self.analysis.path == other.analysis.path and abs(self.score - other.score) < 0.001

    @property
    def path(self) -> str:
        """Get file path."""
        return self.analysis.path

    @property
    def file_name(self) -> str:
        """Get file name."""
        return Path(self.analysis.path).name

    @property
    def language(self) -> str:
        """Get file language."""
        return self.analysis.language

    def generate_explanation(self, weights: Dict[str, float], verbose: bool = False) -> str:
        """Generate human-readable explanation of ranking.

        Args:
            weights: Factor weights used for ranking
            verbose: Include detailed factor breakdown

        Returns:
            Explanation string
        """
        if self.explanation and not verbose:
            return self.explanation

        # Get top contributing factors
        top_factors = self.factors.get_top_factors(weights, n=3)

        if not top_factors:
            return "Low relevance (no significant factors)"

        # Build explanation
        explanations = []

        for factor_name, value, contribution in top_factors:
            # Generate human-readable factor description
            if factor_name == "keyword_match":
                explanations.append(f"Strong keyword match ({value:.2f})")
            elif factor_name == "tfidf_similarity":
                explanations.append(f"High TF-IDF similarity ({value:.2f})")
            elif factor_name == "bm25_score":
                explanations.append(f"High BM25 relevance ({value:.2f})")
            elif factor_name == "semantic_similarity":
                explanations.append(f"High semantic similarity ({value:.2f})")
            elif factor_name == "path_relevance":
                explanations.append(f"Relevant file path ({value:.2f})")
            elif factor_name == "import_centrality":
                explanations.append(f"Central to import graph ({value:.2f})")
            elif factor_name == "git_recency":
                explanations.append(f"Recently modified ({value:.2f})")
            elif factor_name == "git_frequency":
                explanations.append(f"Frequently changed ({value:.2f})")
            elif factor_name == "complexity_relevance":
                explanations.append(f"Relevant complexity ({value:.2f})")
            elif factor_name == "code_patterns":
                explanations.append(f"Matching code patterns ({value:.2f})")
            elif factor_name == "type_relevance":
                explanations.append(f"Relevant file type ({value:.2f})")
            else:
                explanations.append(f"{factor_name.replace('_', ' ').title()} ({value:.2f})")

        if verbose:
            # Add confidence and rank info
            if self.rank:
                explanations.append(f"Rank: #{self.rank}")
            explanations.append(f"Confidence: {self.confidence:.2f}")

        explanation = "; ".join(explanations)
        self.explanation = explanation

        return explanation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all ranking information
        """
        return {
            "path": self.analysis.path,
            "score": self.score,
            "rank": self.rank,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "factors": self.factors.to_dict(),
            "metadata": self.metadata,
            "file_info": {
                "name": self.file_name,
                "language": self.language,
                "size": self.analysis.size,
                "lines": self.analysis.lines,
            },
        }


class RankingExplainer:
    """Utility class for generating ranking explanations.

    Provides detailed explanations of why files ranked the way they did,
    useful for debugging and understanding ranking behavior.
    """

    def __init__(self):
        """Initialize the explainer."""
        self.logger = get_logger(__name__)

    def explain_ranking(
        self,
        ranked_files: List[RankedFile],
        weights: Dict[str, float],
        top_n: int = 10,
        include_factors: bool = True,
    ) -> str:
        """Generate comprehensive ranking explanation.

        Args:
            ranked_files: List of ranked files
            weights: Factor weights used
            top_n: Number of top files to explain
            include_factors: Include factor breakdown

        Returns:
            Formatted explanation string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RANKING EXPLANATION")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        lines.append(f"Total files ranked: {len(ranked_files)}")
        if ranked_files:
            lines.append(f"Score range: {ranked_files[0].score:.3f} - {ranked_files[-1].score:.3f}")
            avg_score = sum(f.score for f in ranked_files) / len(ranked_files)
            lines.append(f"Average score: {avg_score:.3f}")
        lines.append("")

        # Weight configuration
        lines.append("Factor Weights:")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for factor, weight in sorted_weights:
            if weight > 0:
                lines.append(f"  {factor:25s}: {weight:.2f}")
        lines.append("")

        # Top files explanation
        lines.append(f"Top {min(top_n, len(ranked_files))} Files:")
        lines.append("-" * 80)

        for i, ranked_file in enumerate(ranked_files[:top_n], 1):
            lines.append(f"\n{i}. {ranked_file.path}")
            lines.append(f"   Score: {ranked_file.score:.3f}")
            lines.append(f"   {ranked_file.generate_explanation(weights, verbose=False)}")

            if include_factors:
                lines.append("   Factor Breakdown:")
                top_factors = ranked_file.factors.get_top_factors(weights, n=5)
                for factor_name, value, contribution in top_factors:
                    lines.append(
                        f"     - {factor_name:20s}: {value:.3f} × {weights.get(factor_name, 0):.2f} = {contribution:.3f}"
                    )

        return "\n".join(lines)

    def compare_rankings(
        self,
        rankings1: List[RankedFile],
        rankings2: List[RankedFile],
        labels: Tuple[str, str] = ("Ranking 1", "Ranking 2"),
    ) -> str:
        """Compare two different rankings.

        Useful for understanding how different algorithms or weights
        affect ranking results.

        Args:
            rankings1: First ranking
            rankings2: Second ranking
            labels: Labels for the two rankings

        Returns:
            Comparison report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("RANKING COMPARISON")
        lines.append("=" * 80)
        lines.append("")

        # Create path to rank mappings
        rank1_map = {r.path: i + 1 for i, r in enumerate(rankings1)}
        rank2_map = {r.path: i + 1 for i, r in enumerate(rankings2)}

        # Find differences
        all_paths = set(rank1_map.keys()) | set(rank2_map.keys())

        differences = []
        for path in all_paths:
            rank1 = rank1_map.get(path, len(rankings1) + 1)
            rank2 = rank2_map.get(path, len(rankings2) + 1)
            diff = abs(rank1 - rank2)
            differences.append((path, rank1, rank2, diff))

        # Sort by difference
        differences.sort(key=lambda x: x[3], reverse=True)

        # Report
        lines.append(f"{labels[0]}: {len(rankings1)} files")
        lines.append(f"{labels[1]}: {len(rankings2)} files")
        lines.append("")

        lines.append("Largest Rank Differences:")
        lines.append("-" * 80)

        for path, rank1, rank2, diff in differences[:10]:
            if diff > 0:
                direction = "↑" if rank2 < rank1 else "↓"
                lines.append(
                    f"{Path(path).name:30s}: #{rank1:3d} → #{rank2:3d} ({direction}{diff:3d})"
                )

        return "\n".join(lines)
