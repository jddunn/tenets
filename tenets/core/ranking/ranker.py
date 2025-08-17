"""Main relevance ranking orchestrator.

This module provides the main RelevanceRanker class that coordinates different
ranking strategies, manages corpus analysis, and produces ranked results. It
supports multiple algorithms, parallel processing, and custom ranking extensions.

The ranker is designed to be efficient, scalable, and extensible while providing
high-quality relevance scoring for code search and context generation.
"""

import concurrent.futures
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext
from tenets.utils.logger import get_logger

from ..nlp.tfidf import BM25Calculator, TFIDFCalculator
from .factors import RankedFile, RankingExplainer, RankingFactors
from .strategies import (
    BalancedRankingStrategy,
    FastRankingStrategy,
    MLRankingStrategy,
    RankingStrategy,
    ThoroughRankingStrategy,
)

# Optional symbol for tests to patch ML model class
try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# Provide a module-level cosine_similarity function for tests to patch
def cosine_similarity(a, b):  # pragma: no cover - simple fallback
    try:
        def to_vec(x):
            try:
                if hasattr(x, "detach"):
                    x = x.detach()
                if hasattr(x, "flatten"):
                    x = x.flatten()
                if hasattr(x, "tolist"):
                    x = x.tolist()
            except Exception:
                pass

            def flatten(seq):
                for item in seq:
                    if isinstance(item, (list, tuple)):
                        yield from flatten(item)
                    else:
                        try:
                            yield float(item)
                        except Exception:
                            yield 0.0

            if isinstance(x, (list, tuple)):
                return list(flatten(x))
            try:
                return [float(x)]
            except Exception:
                return [0.0]

        va = to_vec(a)
        vb = to_vec(b)
        n = min(len(va), len(vb))
        if n == 0:
            return 0.0
        va = va[:n]
        vb = vb[:n]
        dot = sum(va[i] * vb[i] for i in range(n))
        import math as _m
        norm_a = _m.sqrt(sum(v * v for v in va)) or 1.0
        norm_b = _m.sqrt(sum(v * v for v in vb)) or 1.0
        return float(dot / (norm_a * norm_b))
    except Exception:
        return 0.0


class RankingAlgorithm(Enum):
    """Available ranking algorithms.

    Each algorithm provides different trade-offs between speed and accuracy.
    """

    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    ML = "ml"
    CUSTOM = "custom"


@dataclass
class RankingStats:
    """Statistics from ranking operation.

    Tracks performance metrics and diagnostic information about the
    ranking process for monitoring and optimization.

    Attributes:
        total_files: Total number of files processed
        files_ranked: Number of files successfully ranked
        files_failed: Number of files that failed ranking
        time_elapsed: Total time in seconds
        algorithm_used: Which algorithm was used
        threshold_applied: Relevance threshold used
        files_above_threshold: Number of files above threshold
        average_score: Average relevance score
        max_score: Maximum relevance score
        min_score: Minimum relevance score
        corpus_stats: Dictionary of corpus statistics
    """

    total_files: int = 0
    files_ranked: int = 0
    files_failed: int = 0
    time_elapsed: float = 0.0
    algorithm_used: str = ""
    threshold_applied: float = 0.0
    files_above_threshold: int = 0
    average_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0
    corpus_stats: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all statistics
        """
        return {
            "total_files": self.total_files,
            "files_ranked": self.files_ranked,
            "files_failed": self.files_failed,
            "time_elapsed": self.time_elapsed,
            "algorithm_used": self.algorithm_used,
            "threshold_applied": self.threshold_applied,
            "files_above_threshold": self.files_above_threshold,
            "average_score": self.average_score,
            "max_score": self.max_score,
            "min_score": self.min_score,
            "corpus_stats": self.corpus_stats,
        }


class RelevanceRanker:
    """Main relevance ranking system.

    Orchestrates the ranking process by analyzing the corpus, selecting
    appropriate strategies, and producing ranked results. Supports multiple
    algorithms, parallel processing, and custom ranking extensions.

    The ranker follows a multi-stage process:
    1. Corpus analysis (TF-IDF, import graph, statistics)
    2. Strategy selection based on algorithm
    3. Parallel factor calculation
    4. Score aggregation and weighting
    5. Filtering and sorting

    Attributes:
        config: TenetsConfig instance
        logger: Logger instance
        strategies: Available ranking strategies
        custom_rankers: Custom ranking functions
        executor: Thread pool for parallel processing
        stats: Latest ranking statistics
        cache: Internal cache for optimizations
    """

    def __init__(
        self,
        config: TenetsConfig,
        algorithm: Optional[str] = None,
        use_stopwords: Optional[bool] = None,
    ):
        """Initialize the relevance ranker.

        Args:
            config: Tenets configuration
            algorithm: Override default algorithm
            use_stopwords: Override stopword filtering setting
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Determine algorithm
        algo_str = algorithm or config.ranking.algorithm
        try:
            self.algorithm = RankingAlgorithm(algo_str)
        except ValueError:
            self.logger.warning(f"Unknown algorithm '{algo_str}', using balanced")
            self.algorithm = RankingAlgorithm.BALANCED

        # Stopword configuration
        self.use_stopwords = (
            use_stopwords if use_stopwords is not None else config.ranking.use_stopwords
        )

        # Initialize strategies
        self.strategies: Dict[RankingAlgorithm, RankingStrategy] = {
            RankingAlgorithm.FAST: FastRankingStrategy(),
            RankingAlgorithm.BALANCED: BalancedRankingStrategy(),
            RankingAlgorithm.THOROUGH: ThoroughRankingStrategy(),
            RankingAlgorithm.ML: MLRankingStrategy(),
        }

        # Custom rankers list (keep public and test-expected private alias)
        self.custom_rankers: List[Callable] = []
        self._custom_rankers: List[Callable] = self.custom_rankers

        # Thread pool for parallel ranking
        max_workers = int(config.ranking.workers or 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        # Backwards-compat alias expected by some tests
        self._executor = self.executor

        # Statistics and cache
        self.stats = RankingStats()
        self.cache = {}

        # ML model (loaded lazily)
        self._ml_model = None

        # Optional ML embedding model placeholder for tests that patch it
        # Also expose module-level symbol on instance for convenience
        self.SentenceTransformer = SentenceTransformer

        self.logger.info(
            f"RelevanceRanker initialized: algorithm={self.algorithm.value}, "
            f"workers={max_workers}, use_stopwords={self.use_stopwords}"
        )

    def rank_files(
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        algorithm: Optional[str] = None,
        parallel: bool = True,
        explain: bool = False,
    ) -> List[FileAnalysis]:
        """Rank files by relevance to prompt.

        This is the main entry point for ranking files. It analyzes the corpus,
        applies the selected ranking strategy, and returns files sorted by
        relevance above the configured threshold.

        Args:
            files: List of files to rank
            prompt_context: Parsed prompt information
            algorithm: Override algorithm for this ranking
            parallel: Whether to rank files in parallel
            explain: Whether to generate ranking explanations

        Returns:
            List of FileAnalysis objects sorted by relevance (highest first)
            and filtered by threshold

        Raises:
            ValueError: If algorithm is invalid
        """
        if not files:
            return []

        start_time = time.time()

        # Reset statistics
        self.stats = RankingStats(
            total_files=len(files),
            algorithm_used=algorithm or self.algorithm.value,
            threshold_applied=self.config.ranking.threshold,
        )

        self.logger.info(f"Ranking {len(files)} files using {self.stats.algorithm_used} algorithm")

        # Select strategy
        if algorithm:
            try:
                algo_enum = RankingAlgorithm(algorithm)
                strategy = self.strategies.get(algo_enum)
            except ValueError:
                raise ValueError(f"Unknown ranking algorithm: {algorithm}")
        else:
            strategy = self._get_strategy(self.algorithm.value)

        if not strategy:
            raise ValueError(f"No strategy for algorithm: {self.algorithm}")

        # Analyze corpus
        corpus_stats = self._analyze_corpus(files, prompt_context)
        self.stats.corpus_stats = corpus_stats

        # Rank files
        ranked_files = self._rank_with_strategy(
            files, prompt_context, corpus_stats, strategy, parallel
        )

        # Apply custom rankers
        for custom_ranker in self.custom_rankers:
            try:
                ranked_files = custom_ranker(ranked_files, prompt_context)
            except Exception as e:
                self.logger.warning(f"Custom ranker failed: {e}")

        # Sort by score
        ranked_files.sort(reverse=True)

    # Filter by threshold and update statistics
        threshold = self.config.ranking.threshold
        filtered_files = []
        scores = []

        for i, rf in enumerate(ranked_files):
            scores.append(rf.score)

            if rf.score >= threshold:
                # Update FileAnalysis with ranking info
                rf.analysis.relevance_score = rf.score
                rf.analysis.relevance_rank = i + 1

                # Generate explanation if requested
                if explain:
                    rf.explanation = rf.generate_explanation(strategy.get_weights(), verbose=True)

                filtered_files.append(rf.analysis)

        # Update statistics
        self.stats.files_ranked = len(ranked_files)
        self.stats.files_above_threshold = len(filtered_files)
        self.stats.time_elapsed = time.time() - start_time

        if scores:
            self.stats.average_score = sum(scores) / len(scores)
            self.stats.max_score = max(scores)
            self.stats.min_score = min(scores)

        # If nothing passed threshold, fall back to returning top 1-3 files
        if not filtered_files and ranked_files:
            top_k = min(3, len(ranked_files))
            fallback = [rf.analysis for rf in ranked_files[:top_k]]
            for i, a in enumerate(fallback, 1):
                a.relevance_score = ranked_files[i-1].score
                a.relevance_rank = i
            filtered_files = fallback

        self.logger.info(
            f"Ranking complete: {len(filtered_files)}/{len(files)} files "
            f"above threshold ({threshold:.2f}) in {self.stats.time_elapsed:.2f}s"
        )

        # Generate explanation report if requested
        if explain and ranked_files:
            explainer = RankingExplainer()
            explanation = explainer.explain_ranking(ranked_files[:20], strategy.get_weights())
            self.logger.info(f"Ranking Explanation:\n{explanation}")

        return filtered_files

    def _get_strategy(self, algorithm: Optional[str] = None) -> RankingStrategy:
        """Return the strategy instance for the given algorithm string.

        If algorithm is None, use the current instance algorithm.
        """
        algo = algorithm or self.algorithm.value
        try:
            algo_enum = RankingAlgorithm(algo)
        except ValueError:
            algo_enum = self.algorithm
        return self.strategies.get(algo_enum)

    def _rank_with_strategy(
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        corpus_stats: Dict[str, Any],
        strategy: RankingStrategy,
        parallel: bool,
    ) -> List[RankedFile]:
        """Rank files using a specific strategy.

        Args:
            files: Files to rank
            prompt_context: Prompt context
            corpus_stats: Corpus statistics
            strategy: Ranking strategy to use
            parallel: Whether to use parallel processing

        Returns:
            List of RankedFile objects
        """
        ranked_files = []
        weights = strategy.get_weights()

        if parallel and len(files) > 10:
            # Parallel ranking
            futures = []

            for file in files:
                future = self.executor.submit(
                    self._rank_single_file, file, prompt_context, corpus_stats, strategy, weights
                )
                futures.append((file, future))

            # Collect results
            for file, future in futures:
                try:
                    ranked_file = future.result(timeout=5.0)
                    if ranked_file:
                        ranked_files.append(ranked_file)
                        self.stats.files_ranked += 1
                except Exception as e:
                    self.logger.warning(f"Failed to rank {file.path}: {e}")
                    self.stats.files_failed += 1
                    # Add with zero score
                    ranked_files.append(
                        RankedFile(
                            analysis=file,
                            score=0.0,
                            factors=RankingFactors(),
                            explanation=f"Ranking failed: {e!s}",
                        )
                    )
        else:
            # Sequential ranking
            for file in files:
                try:
                    ranked_file = self._rank_single_file(
                        file, prompt_context, corpus_stats, strategy, weights
                    )
                    if ranked_file:
                        ranked_files.append(ranked_file)
                        self.stats.files_ranked += 1
                except Exception as e:
                    self.logger.warning(f"Failed to rank {file.path}: {e}")
                    self.stats.files_failed += 1
                    ranked_files.append(
                        RankedFile(
                            analysis=file,
                            score=0.0,
                            factors=RankingFactors(),
                            explanation=f"Ranking failed: {e!s}",
                        )
                    )

        return ranked_files

    def _generate_explanation(self, factors: RankingFactors, weights: Dict[str, float]) -> str:
        """Generate a short human-readable explanation for a file's score.

        Matches tests expecting mentions like "keyword match" and "TF-IDF" when
        those factors contribute meaningfully. Falls back to "Low relevance" when
        there are no significant contributions.
        """
        if not weights:
            return "Low relevance"

        # Map factor keys to friendly names
        friendly = {
            "keyword_match": "Keyword match",
            "tfidf_similarity": "TF-IDF similarity",
            "semantic_similarity": "Semantic similarity",
            "path_relevance": "Path relevance",
            "import_centrality": "Import centrality",
            "bm25_score": "BM25 score",
            "code_patterns": "Code patterns",
            "ast_relevance": "AST relevance",
        }

        # Gather top contributing non-zero weighted factors
        contributions: List[tuple[str, float]] = []
        for key, w in weights.items():
            val = getattr(factors, key, 0.0)
            if w > 0 and val > 0:
                contributions.append((key, val * w))

        if not contributions:
            return "Low relevance"

        contributions.sort(key=lambda x: x[1], reverse=True)
        top = [k for k, _ in contributions[:3]]

        parts = []
        for k in top:
            name = friendly.get(k, k.replace("_", " ").title())
            val = getattr(factors, k, 0.0)
            parts.append(f"{name}: {val:.2f}")

        return "; ".join(parts)

    def _rank_single_file(
        self,
        file: FileAnalysis,
        prompt_context: PromptContext,
        corpus_stats: Dict[str, Any],
        strategy: RankingStrategy,
        weights: Dict[str, float],
    ) -> RankedFile:
        """Rank a single file.

        Args:
            file: File to rank
            prompt_context: Prompt context
            corpus_stats: Corpus statistics
            strategy: Ranking strategy
            weights: Factor weights

        Returns:
            RankedFile object
        """
        # Calculate ranking factors
        factors = strategy.rank_file(file, prompt_context, corpus_stats)

        # Calculate weighted score
        score = factors.get_weighted_score(weights)

        # Generate basic explanation
        top_factors = factors.get_top_factors(weights, n=3)
        explanations = []
        for factor_name, value, contribution in top_factors:
            explanations.append(f"{factor_name}: {value:.2f}")
        explanation = "; ".join(explanations) if explanations else "Low relevance"

        return RankedFile(analysis=file, score=score, factors=factors, explanation=explanation)

    def _analyze_corpus(
        self, files: List[FileAnalysis], prompt_context: PromptContext
    ) -> Dict[str, Any]:
        """Analyze the corpus for statistics.

        Builds TF-IDF index, import graph, and other corpus-wide statistics
        that are used by ranking strategies.

        Args:
            files: All files in corpus
            prompt_context: Prompt context

        Returns:
            Dictionary of corpus statistics
        """
        self.logger.debug(f"Analyzing corpus of {len(files)} files")

        stats = {
            "total_files": len(files),
            "languages": Counter(),
            "file_sizes": [],
            "import_graph": defaultdict(set),
            "dependency_tree": {},
        }

        # Initialize TF-IDF calculator
        # Honor ad-hoc test flag `use_tfidf_stopwords` if present on config
        use_sw = self.use_stopwords
        try:
            if hasattr(self.config, "use_tfidf_stopwords"):
                use_sw = bool(getattr(self.config, "use_tfidf_stopwords"))
        except Exception:
            pass

        tfidf_calc = TFIDFCalculator(use_stopwords=use_sw)

        # Initialize BM25 calculator
        bm25_calc = BM25Calculator(use_stopwords=use_sw)

        # Build corpus and collect statistics
        documents = []

        for file in files:
            # Language distribution
            stats["languages"][file.language] += 1

            # File sizes
            stats["file_sizes"].append(file.size)

            # Add to TF-IDF/BM25 corpus
            if file.content:
                documents.append((file.path, file.content))

            # Build import graph
            for imp in file.imports:
                if hasattr(imp, "module"):
                    module = imp.module
                    # Try to resolve import to file
                    imported_file = self._resolve_import(module, file.path, files)
                    if imported_file:
                        stats["import_graph"][imported_file].add(file.path)

        # Build TF-IDF corpus
        if documents:
            tfidf_calc.build_corpus(documents)
            stats["tfidf_calculator"] = tfidf_calc

            # Build BM25 corpus
            bm25_calc.build_corpus(documents)
            stats["bm25_calculator"] = bm25_calc

        # Calculate additional statistics
        if stats["file_sizes"]:
            stats["avg_file_size"] = sum(stats["file_sizes"]) / len(stats["file_sizes"])
            stats["total_size"] = sum(stats["file_sizes"])
        else:
            stats["avg_file_size"] = 0
            stats["total_size"] = 0

        stats["total_imports"] = sum(len(importers) for importers in stats["import_graph"].values())

        # Build dependency tree (simplified)
        stats["dependency_tree"] = self._build_dependency_tree(files, stats["import_graph"])

        self.logger.debug(
            f"Corpus analysis complete: {len(stats['languages'])} languages, "
            f"{stats['total_imports']} imports"
        )

        return stats

    def _resolve_import(
        self, module_name: str, from_file: str, all_files: List[FileAnalysis]
    ) -> Optional[str]:
        """Resolve an import to a file path.

        Attempts to match import statements to actual files in the project.
        Uses heuristics based on module name and file paths.

        Args:
            module_name: Name of imported module
            from_file: File doing the importing
            all_files: All files in project

        Returns:
            Resolved file path or None
        """
        if not module_name:
            return None

        # Split module into parts
        module_parts = module_name.split(".")

        # Try different resolution strategies
        for file in all_files:
            file_path = Path(file.path)
            file_stem = file_path.stem

            # Direct filename match
            if file_stem == module_parts[-1]:
                return file.path

            # Check if path contains module structure
            path_parts = [p.lower() for p in file_path.parts]
            module_parts_lower = [p.lower() for p in module_parts]

            # Check if all module parts appear in path
            if all(part in str(file_path).lower() for part in module_parts_lower):
                return file.path

            # Check relative imports
            if module_name.startswith("."):
                # Relative import - resolve based on from_file location
                from_dir = Path(from_file).parent
                relative_parts = module_parts[1:]  # Skip the '.'

                # Try to find matching file in relative location
                if file_stem in relative_parts and str(from_dir) in str(file_path):
                    return file.path

        return None

    def _build_dependency_tree(
        self, files: List[FileAnalysis], import_graph: Dict[str, Set[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Build dependency tree from import graph.

        Creates a tree structure showing dependency depth for each file.
        Files with no dependencies are at depth 0, files that only depend
        on depth-0 files are at depth 1, etc.

        Args:
            files: All files
            import_graph: Import dependency graph

        Returns:
            Dictionary mapping file paths to dependency info
        """
        dependency_tree = {}

        # Initialize all files
        file_paths = {f.path for f in files}

        # Find root files (no dependencies)
        root_files = set()
        for file_path in file_paths:
            # Check if file imports anything
            imports_others = False
            for imported, importers in import_graph.items():
                if file_path in importers:
                    imports_others = True
                    break

            if not imports_others:
                root_files.add(file_path)
                dependency_tree[file_path] = {"depth": 0, "dependencies": set()}

        # Build tree iteratively
        current_depth = 0
        max_iterations = 10  # Prevent infinite loops

        while current_depth < max_iterations:
            found_new = False
            current_depth += 1

            for file_path in file_paths:
                if file_path in dependency_tree:
                    continue  # Already processed

                # Check if all dependencies are resolved
                dependencies = import_graph.get(file_path, set())
                if all(dep in dependency_tree for dep in dependencies):
                    # Calculate depth as max dependency depth + 1
                    if dependencies:
                        max_dep_depth = max(dependency_tree[dep]["depth"] for dep in dependencies)
                        depth = max_dep_depth + 1
                    else:
                        depth = 0

                    dependency_tree[file_path] = {"depth": depth, "dependencies": dependencies}
                    found_new = True

            if not found_new:
                break  # No new files resolved

        # Mark unresolved files
        for file_path in file_paths:
            if file_path not in dependency_tree:
                dependency_tree[file_path] = {
                    "depth": -1,  # Circular or unresolved
                    "dependencies": import_graph.get(file_path, set()),
                }

        return dependency_tree

    def register_custom_ranker(
        self, ranker_func: Callable[[List[RankedFile], PromptContext], List[RankedFile]]
    ):
        """Register a custom ranking function.

        Custom rankers are applied after the main ranking strategy and can
        adjust scores based on project-specific logic.

        Args:
            ranker_func: Function that takes ranked files and returns modified list

        Example:
            >>> def boost_tests(ranked_files, prompt_context):
            ...     if 'test' in prompt_context.text:
            ...         for rf in ranked_files:
            ...             if 'test' in rf.path:
            ...                 rf.score *= 1.5
            ...     return ranked_files
            >>> ranker.register_custom_ranker(boost_tests)
        """
        self.custom_rankers.append(ranker_func)
        # Keep alias updated
        self._custom_rankers = self.custom_rankers
        self.logger.info(f"Registered custom ranker: {ranker_func.__name__}")

    def get_ranking_explanation(self, ranked_files: List[RankedFile], top_n: int = 10) -> str:
        """Get detailed explanation of ranking results.

        Args:
            ranked_files: List of ranked files
            top_n: Number of top files to explain

        Returns:
            Formatted explanation string
        """
        explainer = RankingExplainer()
        strategy = self.strategies.get(self.algorithm)
        weights = strategy.get_weights() if strategy else {}

        return explainer.explain_ranking(ranked_files[:top_n], weights, top_n=top_n)

    def get_stats(self) -> RankingStats:
        """Get latest ranking statistics.

        Returns:
            RankingStats object
        """
        return self.stats

    def shutdown(self):
        """Shutdown the ranker and clean up resources."""
        self.executor.shutdown(wait=True)
        self.logger.info("RelevanceRanker shutdown complete")


# Convenience function for creating ranker
def create_ranker(
    config: Optional[TenetsConfig] = None, algorithm: str = "balanced", use_stopwords: bool = False
) -> RelevanceRanker:
    """Create a configured relevance ranker.

    Args:
        config: Configuration (uses default if None)
        algorithm: Ranking algorithm to use
        use_stopwords: Whether to filter stopwords

    Returns:
        Configured RelevanceRanker instance
    """
    if config is None:
        config = TenetsConfig()

    return RelevanceRanker(config, algorithm=algorithm, use_stopwords=use_stopwords)
