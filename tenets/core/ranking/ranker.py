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

from ..nlp.bm25 import BM25Calculator
from ..nlp.tfidf import TFIDFCalculator
from .factors import RankedFile, RankingExplainer, RankingFactors
from .strategies import (
    BalancedRankingStrategy,
    FastRankingStrategy,
    RankingStrategy,
    ThoroughRankingStrategy,
)

# Optional symbol for tests to patch ML model class - lazy loaded when needed
SentenceTransformer = None  # Will be imported lazily when ML ranking is used


# Import cosine similarity from the central module
try:
    from tenets.core.nlp.similarity import cosine_similarity
except ImportError:
    # Fallback for tests or when similarity module not available
    def cosine_similarity(a, b):  # pragma: no cover - simple fallback
        try:
            import math

            # Simple dot product cosine similarity for lists
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
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

        # ML configuration - skip for fast mode, force enable for Thorough mode
        if algo_str == "fast":
            self.use_ml = False  # Fast mode never uses ML
        elif algo_str == "thorough":
            self.use_ml = True  # Thorough ALWAYS uses ML regardless of config
            self.logger.info("Thorough mode selected - force enabling ML features")
        else:
            self.use_ml = (
                config.ranking.use_ml if config and hasattr(config.ranking, "use_ml") else False
            )

        # Initialize strategies lazily to avoid loading unnecessary models
        self._strategies_cache: Dict[RankingAlgorithm, RankingStrategy] = {}
        self.strategies = self._strategies_cache  # Alias for compatibility

        # Pre-populate core strategies for tests that expect them
        # These are lightweight and don't load ML models until actually used
        self._init_core_strategies()

        # Custom rankers list (keep public and test-expected private alias)
        self.custom_rankers: List[Callable] = []
        self._custom_rankers: List[Callable] = self.custom_rankers

        # Thread pool for parallel ranking (lazy initialization to avoid Windows issues)
        from tenets.utils.multiprocessing import get_ranking_workers, log_worker_info

        max_workers = get_ranking_workers(config)
        self.max_workers = max_workers  # Store for logging
        self._executor_instance = None  # Will be created lazily
        # Backwards-compat alias expected by some tests
        self._executor = None

        # Statistics and cache
        self.stats = RankingStats()
        self.cache = {}

        # ML model (loaded lazily)
        self._ml_model = None

        # Optional ML embedding model placeholder for tests that patch it
        # Also expose module-level symbol on instance for convenience
        self.SentenceTransformer = SentenceTransformer

        # Log worker configuration
        log_worker_info(self.logger, "RelevanceRanker", max_workers)

        # Log initialization details (skip ML status for fast mode)
        if algo_str == "fast":
            self.logger.info(
                f"RelevanceRanker initialized: algorithm={self.algorithm.value}, "
                f"use_stopwords={self.use_stopwords}"
            )
        else:
            self.logger.info(
                f"RelevanceRanker initialized: algorithm={self.algorithm.value}, "
                f"use_stopwords={self.use_stopwords}, use_ml={self.use_ml}"
            )

    @property
    def executor(self):
        """Lazy initialization of ThreadPoolExecutor to avoid Windows import issues."""
        if self._executor_instance is None:
            try:
                self._executor_instance = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                )
                self._executor = self._executor_instance  # Backwards-compat alias
            except Exception as e:
                # If there's an actual issue, fall back to sequential processing
                self.logger.warning(
                    f"Failed to initialize ThreadPoolExecutor: {e}. "
                    "Falling back to sequential processing."
                )
                self._executor_instance = None  # Force sequential processing
                self._executor = None
        return self._executor_instance

    def _init_core_strategies(self):
        """Initialize core ranking strategies.

        Pre-populates the strategies cache with lightweight strategy instances.
        These don't load heavy ML models until actually used.
        """
        # Only initialize lightweight strategies at startup, pass config
        self._strategies_cache[RankingAlgorithm.FAST] = FastRankingStrategy(self.config)
        self._strategies_cache[RankingAlgorithm.BALANCED] = BalancedRankingStrategy(self.config)
        # DON'T initialize THOROUGH here - it will be created lazily when needed
        # to avoid loading ML models unnecessarily

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

        # Try to initialize executor if parallel is requested
        if parallel and self.executor is None:
            self.logger.info("Parallel processing not available, using sequential ranking")
            parallel = False

        # Log the actual algorithm being used (not the default)
        actual_algo = algorithm or self.algorithm.value
        self.logger.info(
            f"Ranking {len(files)} files using {actual_algo} algorithm "
            f"(parallel={parallel}, workers={self.max_workers if parallel else 1})"
        )

        # Select strategy
        actual_algorithm = algorithm if algorithm else self.algorithm.value
        self.logger.info(f"Selecting strategy for algorithm: {actual_algorithm}")

        if algorithm:
            try:
                strategy = self._get_strategy(algorithm)
                self.logger.info(f"Using {strategy.__class__.__name__} for {algorithm} mode")
            except ValueError:
                raise ValueError(f"Unknown ranking algorithm: {algorithm}")
        else:
            strategy = self._get_strategy(self.algorithm.value)
            self.logger.info(
                f"Using default {strategy.__class__.__name__} for {self.algorithm.value} mode"
            )

        if not strategy:
            raise ValueError(f"No strategy for algorithm: {self.algorithm}")

        # Pre-filter files based on mode for appropriate workload
        files_to_rank = files
        if len(files) > 100:
            if actual_algorithm == "fast":
                # Fast mode: moderate pre-filtering for better coverage
                limit = 100
                self.logger.info(f"Fast mode: Pre-filtering {len(files)} files for speed")
            elif actual_algorithm == "balanced":
                # Balanced mode: moderate pre-filtering
                limit = 200
                self.logger.info(f"Balanced mode: Pre-filtering {len(files)} files")
            else:
                # Thorough mode: process all files
                limit = len(files)

            if limit < len(files):
                priority_files = []
                other_files = []
                keywords_lower = [kw.lower() for kw in prompt_context.keywords]

                for f in files:
                    file_path_lower = f.path.lower()
                    # Check if any keyword appears in the file path
                    if any(kw in file_path_lower for kw in keywords_lower):
                        priority_files.append(f)
                    else:
                        other_files.append(f)

                # Take priority files plus others up to limit
                files_to_rank = priority_files + other_files[: max(limit - len(priority_files), 0)]
                self.logger.info(
                    f"Reduced to {len(files_to_rank)} files ({len(priority_files)} priority)"
                )

        # Analyze corpus - pass the actual algorithm being used
        corpus_stats = self._analyze_corpus(files_to_rank, prompt_context, actual_algorithm)
        self.stats.corpus_stats = corpus_stats

        # Rank files
        ranked_files = self._rank_with_strategy(
            files_to_rank, prompt_context, corpus_stats, strategy, parallel
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
                a.relevance_score = ranked_files[i - 1].score
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
        Strategies are created lazily on first use to avoid loading unnecessary models.
        """
        algo = algorithm or self.algorithm.value
        try:
            algo_enum = RankingAlgorithm(algo)
        except ValueError:
            # If algorithm was explicitly provided and is invalid, raise error
            if algorithm:
                raise ValueError(f"Unknown ranking algorithm: {algorithm}")
            # Otherwise fall back to the instance algorithm
            algo_enum = self.algorithm

        # Lazy creation of strategies with config
        if algo_enum not in self._strategies_cache:
            if algo_enum == RankingAlgorithm.FAST:
                self._strategies_cache[algo_enum] = FastRankingStrategy(self.config)
            elif algo_enum == RankingAlgorithm.BALANCED:
                self._strategies_cache[algo_enum] = BalancedRankingStrategy(self.config)
            elif algo_enum == RankingAlgorithm.THOROUGH or algo_enum == RankingAlgorithm.ML:
                self._strategies_cache[algo_enum] = ThoroughRankingStrategy(self.config)

        return self._strategies_cache.get(algo_enum)

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

        if parallel and len(files) > 10 and self.executor is not None:
            # Parallel ranking (only if executor is available)
            self.logger.info(
                f"Using parallel ranking with {self.max_workers} workers for {len(files)} files"
            )
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
            self.logger.info(
                f"Using sequential ranking for {len(files)} files "
                f"(parallel={parallel}, threshold for parallel: >10 files)"
            )
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
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        algorithm: Optional[str] = None,
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

        # Skip expensive corpus building for fast mode
        actual_algo = algorithm or self.algorithm.value
        if actual_algo == "fast" or actual_algo == RankingAlgorithm.FAST.value:
            self.logger.info("Fast mode: Skipping corpus building for speed")
            # Only collect basic statistics for fast mode
            for file in files:
                stats["languages"][file.language] += 1
                stats["file_sizes"].append(file.size)
        else:
            # Initialize text similarity calculator based on config
            # Honor ad-hoc test flag `use_tfidf_stopwords` if present on config
            use_sw = self.use_stopwords
            try:
                if hasattr(self.config, "use_tfidf_stopwords"):
                    use_sw = bool(self.config.use_tfidf_stopwords)
            except Exception:
                pass

            # Choose text similarity algorithm from config
            text_sim_algo = getattr(self.config.ranking, "text_similarity_algorithm", "bm25")

            # Initialize the calculators we need
            if text_sim_algo == "tfidf":
                # Use TF-IDF only if explicitly configured
                tfidf_calc = TFIDFCalculator(use_stopwords=use_sw)
                bm25_calc = None
                self.logger.info(f"Using TF-IDF for text similarity ({actual_algo} mode)")
            else:
                # Use BM25 (default for all modes including thorough)
                bm25_calc = BM25Calculator(use_stopwords=use_sw)
                tfidf_calc = None  # Don't build TF-IDF corpus when not needed
                self.logger.info(f"Using BM25 for text similarity ({actual_algo} mode)")

            # Build corpus and collect statistics
            documents = []

            for file in files:
                # Language distribution
                stats["languages"][file.language] += 1

                # File sizes
                stats["file_sizes"].append(file.size)

                # Add to TF-IDF/BM25 corpus
                if file.content:
                    # Limit document size for performance based on mode
                    if actual_algo == "balanced":
                        # Balanced: respect configured content limit for corpus building
                        limit = (
                            getattr(self.config.ranking, "ranking_content_scan_limit_balanced", -1)
                            if self.config
                            else -1
                        )
                        content = (
                            file.content[:limit]
                            if limit > 0 and len(file.content) > limit
                            else file.content
                        )
                    else:
                        # Thorough: use full content
                        content = file.content
                    documents.append((file.path, content))

                # Build import graph
                for imp in file.imports:
                    if hasattr(imp, "module"):
                        module = imp.module
                        # Try to resolve import to file
                        imported_file = self._resolve_import(module, file.path, files)
                        if imported_file:
                            stats["import_graph"][imported_file].add(file.path)

            # Build text similarity corpus
            if documents:
                if tfidf_calc:
                    tfidf_calc.build_corpus(documents)
                    stats["tfidf_calculator"] = tfidf_calc

                if bm25_calc:
                    bm25_calc.build_corpus(documents)
                    stats["bm25_calculator"] = bm25_calc

                    # Pre-compute BM25 scores for all documents to avoid O(n²) complexity
                    query_terms = " ".join(prompt_context.keywords)
                    all_scores = bm25_calc.get_scores(query_terms)
                    # Store as dict for O(1) lookup
                    stats["bm25_scores"] = {doc_id: score for doc_id, score in all_scores}
                    self.logger.debug(
                        f"Pre-computed BM25 scores for {len(stats['bm25_scores'])} documents"
                    )

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
        if self._executor_instance is not None:
            self._executor_instance.shutdown(wait=True)
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
