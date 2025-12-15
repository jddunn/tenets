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

    Supports multiple output formats:
    - text: Human-readable text for CLI output
    - json: Structured JSON for programmatic use
    - markdown: Formatted markdown for documentation
    """

    # Human-readable factor names
    FACTOR_NAMES = {
        "keyword_match": "Keyword Match",
        "tfidf_similarity": "TF-IDF Similarity",
        "bm25_score": "BM25 Score",
        "path_relevance": "Path Relevance",
        "import_centrality": "Import Centrality",
        "dependency_depth": "Dependency Depth",
        "git_recency": "Git Recency",
        "git_frequency": "Git Change Frequency",
        "git_author_relevance": "Author Relevance",
        "complexity_relevance": "Complexity Score",
        "maintainability_score": "Maintainability",
        "semantic_similarity": "Semantic Similarity",
        "type_relevance": "File Type Relevance",
        "code_patterns": "Code Pattern Match",
        "ast_relevance": "AST Structure Match",
        "test_coverage": "Test Coverage",
        "documentation_score": "Documentation Quality",
    }

    # Factor descriptions for help text
    FACTOR_DESCRIPTIONS = {
        "keyword_match": "Direct keyword matching between query and file content",
        "tfidf_similarity": "Term frequency-inverse document frequency similarity",
        "bm25_score": "BM25 probabilistic relevance score",
        "path_relevance": "How well the file path matches query terms",
        "import_centrality": "How central the file is in the import graph",
        "dependency_depth": "Position in the dependency hierarchy",
        "git_recency": "How recently the file was modified",
        "git_frequency": "How frequently the file changes over time",
        "git_author_relevance": "Relevance based on commit authors",
        "complexity_relevance": "Cyclomatic complexity relevance to query",
        "maintainability_score": "Code maintainability index",
        "semantic_similarity": "ML-based semantic similarity score",
        "type_relevance": "Relevance based on file extension/type",
        "code_patterns": "Pattern matching for code structures",
        "ast_relevance": "Abstract syntax tree structural relevance",
        "test_coverage": "Test file relevance",
        "documentation_score": "Documentation quality score",
    }

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

    def explain_file_ranking(
        self,
        ranked_file: RankedFile,
        weights: Dict[str, float],
        format: str = "text",
    ) -> Dict[str, Any]:
        """Generate detailed explanation for a single file's ranking.

        Provides comprehensive breakdown of why a file received its score,
        useful for debugging relevance issues.

        Args:
            ranked_file: The ranked file to explain
            weights: Factor weights used for ranking
            format: Output format ('text', 'json', 'markdown')

        Returns:
            Dictionary with explanation data and formatted output
        """
        factors = ranked_file.factors
        all_factors = factors.to_dict()

        # Calculate contribution for each factor
        contributions = []
        total_weight = sum(weights.values())

        for factor_name, value in all_factors.items():
            if factor_name in ("custom_scores", "metadata"):
                continue

            weight = weights.get(factor_name, 0)
            contribution = value * weight
            normalized = contribution / total_weight if total_weight > 0 else 0

            if value > 0 or weight > 0:
                contributions.append({
                    "factor": factor_name,
                    "display_name": self.FACTOR_NAMES.get(factor_name, factor_name),
                    "description": self.FACTOR_DESCRIPTIONS.get(factor_name, ""),
                    "value": round(value, 4),
                    "weight": round(weight, 4),
                    "contribution": round(contribution, 4),
                    "normalized_contribution": round(normalized, 4),
                    "percentage": round(normalized * 100, 2) if total_weight > 0 else 0,
                })

        # Sort by contribution
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        # Build result
        result = {
            "file": ranked_file.path,
            "score": round(ranked_file.score, 4),
            "rank": ranked_file.rank,
            "confidence": round(ranked_file.confidence, 4),
            "factors": contributions,
            "top_contributors": [c for c in contributions[:5] if c["contribution"] > 0],
            "zero_factors": [c["factor"] for c in contributions if c["value"] == 0],
            "summary": ranked_file.generate_explanation(weights, verbose=True),
        }

        # Format output based on requested format
        if format == "markdown":
            result["formatted"] = self._format_markdown(result)
        elif format == "text":
            result["formatted"] = self._format_text(result)
        else:
            result["formatted"] = None  # JSON format doesn't need extra formatting

        return result

    def _format_text(self, result: Dict[str, Any]) -> str:
        """Format explanation as plain text."""
        lines = [
            f"File: {result['file']}",
            f"Score: {result['score']:.4f}",
            f"Rank: #{result['rank']}" if result['rank'] else "Rank: N/A",
            "",
            "Factor Breakdown:",
            "-" * 60,
        ]

        for f in result["factors"]:
            if f["contribution"] > 0:
                lines.append(
                    f"  {f['display_name']:25s} "
                    f"{f['value']:.3f} × {f['weight']:.2f} = "
                    f"{f['contribution']:.4f} ({f['percentage']:.1f}%)"
                )

        if result["zero_factors"]:
            lines.extend([
                "",
                "Zero-value factors:",
                f"  {', '.join(result['zero_factors'][:5])}",
            ])

        return "\n".join(lines)

    def _format_markdown(self, result: Dict[str, Any]) -> str:
        """Format explanation as markdown."""
        lines = [
            f"## File: `{result['file']}`",
            "",
            f"**Score:** {result['score']:.4f}",
            f"**Rank:** #{result['rank']}" if result['rank'] else "**Rank:** N/A",
            "",
            "### Factor Breakdown",
            "",
            "| Factor | Value | Weight | Contribution | % |",
            "|--------|-------|--------|--------------|---|",
        ]

        for f in result["factors"]:
            if f["contribution"] > 0:
                lines.append(
                    f"| {f['display_name']} | {f['value']:.3f} | "
                    f"{f['weight']:.2f} | {f['contribution']:.4f} | {f['percentage']:.1f}% |"
                )

        return "\n".join(lines)

    def debug_ranking(
        self,
        ranked_files: List[RankedFile],
        weights: Dict[str, float],
        query: str = "",
    ) -> Dict[str, Any]:
        """Generate comprehensive debug information for ranking results.

        Provides detailed analysis of the ranking process, useful for
        understanding and improving ranking quality.

        Args:
            ranked_files: List of ranked files
            weights: Factor weights used
            query: Original query string

        Returns:
            Dictionary with comprehensive debug information
        """
        if not ranked_files:
            return {
                "error": "No files to analyze",
                "query": query,
            }

        # Calculate statistics
        scores = [rf.score for rf in ranked_files]
        avg_score = sum(scores) / len(scores)
        score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)

        # Analyze factor usage
        factor_usage = {}
        for rf in ranked_files:
            factors_dict = rf.factors.to_dict()
            for factor, value in factors_dict.items():
                if factor in ("custom_scores", "metadata"):
                    continue
                if factor not in factor_usage:
                    factor_usage[factor] = {
                        "count_nonzero": 0,
                        "sum": 0,
                        "max": 0,
                        "values": [],
                    }
                if value > 0:
                    factor_usage[factor]["count_nonzero"] += 1
                    factor_usage[factor]["sum"] += value
                    factor_usage[factor]["max"] = max(factor_usage[factor]["max"], value)
                    factor_usage[factor]["values"].append(value)

        # Calculate factor averages
        for factor, stats in factor_usage.items():
            if stats["values"]:
                stats["avg"] = sum(stats["values"]) / len(stats["values"])
            else:
                stats["avg"] = 0
            del stats["values"]  # Remove raw values from output

        # Identify problematic patterns
        issues = []

        # Check for factors with zero contribution
        for factor, stats in factor_usage.items():
            weight = weights.get(factor, 0)
            if weight > 0 and stats["count_nonzero"] == 0:
                issues.append({
                    "type": "unused_factor",
                    "factor": factor,
                    "message": f"Factor '{factor}' has weight {weight} but all values are 0",
                })

        # Check for score clustering
        if score_variance < 0.01:
            issues.append({
                "type": "low_variance",
                "message": "Score variance is very low - files may not be well differentiated",
                "variance": score_variance,
            })

        # Check for threshold issues
        threshold_count = sum(1 for s in scores if s > 0.5)
        if threshold_count == 0:
            issues.append({
                "type": "low_scores",
                "message": "No files scored above 0.5 - query may be too specific",
            })
        elif threshold_count == len(scores):
            issues.append({
                "type": "high_scores",
                "message": "All files scored above 0.5 - query may be too broad",
            })

        return {
            "query": query,
            "total_files": len(ranked_files),
            "statistics": {
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
                "avg_score": round(avg_score, 4),
                "variance": round(score_variance, 4),
                "median_score": round(sorted(scores)[len(scores) // 2], 4),
            },
            "weights": {k: round(v, 4) for k, v in weights.items() if v > 0},
            "factor_usage": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in factor_usage.items()
            },
            "issues": issues,
            "top_files": [
                {
                    "path": rf.path,
                    "score": round(rf.score, 4),
                    "top_factor": rf.factors.get_top_factors(weights, n=1)[0][0]
                    if rf.factors.get_top_factors(weights, n=1) else None,
                }
                for rf in ranked_files[:10]
            ],
        }
