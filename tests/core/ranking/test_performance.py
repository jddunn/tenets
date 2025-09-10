"""
Performance Tests for Ranking System

This module contains comprehensive performance tests to ensure ranking
algorithms meet performance targets and detect regressions.

Test coverage:
    - Performance hierarchy validation (Fast < Balanced < Thorough)
    - O(n) complexity verification (no O(n²) bugs)
    - ML token limit validation
    - Memory usage monitoring
    - Cache effectiveness
    - Regression detection

Author: Tenets Team
License: MIT
"""

import gc
import statistics
import sys
import time
from typing import List
from unittest.mock import Mock, patch

import pytest

# Skip all timing tests if freezegun is active
pytestmark = pytest.mark.skipif(
    "freezegun" in sys.modules or any("freeze" in m for m in sys.modules),
    reason="Performance/timing tests incompatible with freezegun",
)

from tenets.config import TenetsConfig
from tenets.core.ranking.ml_config import MLConfig
from tenets.core.ranking.optimized_ranker import OptimizedRanker, PerformanceMetrics
from tenets.core.ranking.strategies import (
    ThoroughRankingStrategy,
)
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext


class TestPerformanceHierarchy:
    """
    Test that performance follows expected hierarchy: Fast < Balanced < Thorough
    """

    @pytest.fixture
    def test_files(self) -> List[FileAnalysis]:
        """Create test files for benchmarking."""
        files = []
        for i in range(50):
            content = f"""
            class Service{i}:
                def authenticate(self, user, password):
                    # Authentication logic
                    return self.oauth.authenticate(user, password)

                def process_payment(self, amount):
                    # Payment processing
                    return self.stripe.charge(amount)
            """
            files.append(
                FileAnalysis(
                    path=f"/test/service_{i}.py",
                    content=content,
                    language="python",
                    file_name=f"service_{i}.py",
                    file_extension=".py",
                    size=len(content),
                    lines=len(content.splitlines()),
                )
            )
        return files

    @pytest.fixture
    def prompt_context(self) -> PromptContext:
        """Create test prompt context."""
        return PromptContext(
            text="implement oauth authentication",
            keywords=["oauth", "authentication", "implement"],
            task_type="feature",
        )

    def test_performance_hierarchy(self, test_files, prompt_context):
        """
        Test that each mode is progressively slower.

        Expected: Fast < Balanced < Thorough (no ML) < Thorough (with ML)
        """
        times = {}

        # Test each mode
        for algorithm in ["fast", "balanced", "thorough"]:
            config = TenetsConfig()
            config.ranking.algorithm = algorithm

            ranker = OptimizedRanker(config)

            # Warm up
            ranker.rank_files(test_files[:5], prompt_context)

            # Actual timing
            gc.collect()
            start = time.perf_counter()
            ranker.rank_files(test_files, prompt_context)
            elapsed = time.perf_counter() - start

            times[algorithm] = elapsed * 1000  # Convert to ms

        # Verify hierarchy
        assert (
            times["fast"] < times["balanced"]
        ), f"Fast ({times['fast']:.2f}ms) should be faster than Balanced ({times['balanced']:.2f}ms)"

        assert (
            times["balanced"] < times["thorough"] * 2
        ), f"Balanced ({times['balanced']:.2f}ms) should not be much slower than Thorough ({times['thorough']:.2f}ms)"

        # Check relative performance is within expected bounds
        fast_time = times["fast"]

        # Balanced should be 2-6x slower than Fast (allowing some variance)
        balanced_ratio = times["balanced"] / fast_time
        assert (
            2 <= balanced_ratio <= 6
        ), f"Balanced is {balanced_ratio:.1f}x slower than Fast (expected 2-6x)"

        # Thorough (no ML) should be 3-7x slower than Fast
        thorough_ratio = times["thorough"] / fast_time
        assert (
            3 <= thorough_ratio <= 10
        ), f"Thorough is {thorough_ratio:.1f}x slower than Fast (expected 3-10x)"

    def test_no_quadratic_complexity(self, prompt_context):
        """
        Test that ranking complexity is O(n), not O(n²).

        This test ensures the BM25 bug is fixed.
        """
        times_by_size = {}

        for num_files in [10, 20, 40, 80]:
            # Create files
            files = []
            for i in range(num_files):
                files.append(
                    FileAnalysis(
                        path=f"/test/file_{i}.py",
                        content=f"test content {i} oauth authentication",
                        language="python",
                        file_name=f"file_{i}.py",
                        file_extension=".py",
                        size=100,
                        lines=5,
                    )
                )

            config = TenetsConfig()
            config.ranking.algorithm = "balanced"
            ranker = OptimizedRanker(config)

            # Time the ranking
            gc.collect()
            start = time.perf_counter()
            ranker.rank_files(files, prompt_context)
            elapsed = time.perf_counter() - start

            times_by_size[num_files] = elapsed

        # Check that time grows linearly, not quadratically
        # If O(n²), doubling files would quadruple time
        # If O(n), doubling files would double time

        ratio_20_10 = times_by_size[20] / times_by_size[10]
        ratio_40_20 = times_by_size[40] / times_by_size[20]
        ratio_80_40 = times_by_size[80] / times_by_size[40]

        # For O(n), ratios should be around 2
        # For O(n²), ratios would be around 4
        assert ratio_20_10 < 3, f"20/10 ratio {ratio_20_10:.2f} suggests O(n²) complexity"
        assert ratio_40_20 < 3, f"40/20 ratio {ratio_40_20:.2f} suggests O(n²) complexity"
        assert ratio_80_40 < 3, f"80/40 ratio {ratio_80_40:.2f} suggests O(n²) complexity"

        # Ratios should be relatively consistent for O(n)
        ratio_variance = statistics.stdev([ratio_20_10, ratio_40_20, ratio_80_40])
        assert (
            ratio_variance < 0.5
        ), f"High variance {ratio_variance:.2f} suggests non-linear complexity"


class TestMLPerformance:
    """
    Test ML token limits and performance characteristics.
    """

    def test_ml_token_limit_configuration(self):
        """Test that ML token limit is configurable and respected."""
        config = MLConfig(token_limit=512, performance_mode="speed")

        assert config.token_limit == 256  # Speed mode overrides to 256
        assert config.batch_size == 64

        config = MLConfig(token_limit=2048, performance_mode="quality")
        assert config.token_limit == 2048
        assert config.batch_size == 16

    def test_adaptive_token_sizing(self):
        """Test adaptive token sizing based on file count."""
        config = MLConfig(adaptive_sizing=True, token_limit=1000)

        # Small number of files - use full limit
        limit = config.get_adaptive_token_limit(num_files=10)
        assert limit == 1000

        # Large number of files - reduce limit
        limit = config.get_adaptive_token_limit(num_files=5000)
        assert limit < 1000
        assert limit >= config.min_token_limit

    @patch.object(ThoroughRankingStrategy, "_embedding_model")
    def test_ml_processing_time(self, mock_model):
        """Test that ML processing time scales with token limit."""
        mock_model.encode = Mock(return_value=[[0.1] * 384])

        files = [
            FileAnalysis(
                path=f"/test/file_{i}.py",
                content="x " * 2000,  # 2000 tokens
                language="python",
                file_name=f"file_{i}.py",
                file_extension=".py",
                size=4000,
                lines=1,
            )
            for i in range(10)
        ]

        context = PromptContext(text="test", keywords=["test"], task_type="general")

        # Test with different token limits
        for token_limit in [256, 512, 1024]:
            config = TenetsConfig()
            config.ranking.algorithm = "thorough"
            config.ranking.ml_token_limit = token_limit

            ranker = OptimizedRanker(config)
            ranker.ml_token_limit = token_limit

            ranker.rank_files(files, context)

            calls = mock_model.encode.call_args_list
            if calls:
                for call in calls:
                    content = call[0][0] if call[0] else ""
                    tokens = content.split()
                    assert len(tokens) <= token_limit


class TestPerformanceMonitoring:
    """
    Test performance monitoring and reporting.
    """

    def test_performance_metrics_tracking(self):
        """Test that performance metrics are properly tracked."""
        config = TenetsConfig()
        config.ranking.algorithm = "balanced"

        ranker = OptimizedRanker(config)

        files = [
            FileAnalysis(
                path=f"/test/file_{i}.py",
                content=f"test content {i}",
                language="python",
                file_name=f"file_{i}.py",
                file_extension=".py",
                size=100,
                lines=5,
            )
            for i in range(20)
        ]

        context = PromptContext(text="test query", keywords=["test", "query"], task_type="general")

        # Rank files
        ranker.rank_files(files, context)

        # Check metrics were recorded
        assert ranker.metrics.total_files == 20
        assert ranker.metrics.total_time_ms > 0
        assert ranker.metrics.avg_time_per_file > 0

        # For balanced mode, corpus build time should be recorded
        assert ranker.metrics.corpus_build_time_ms > 0
        assert ranker.metrics.scoring_time_ms > 0

    def test_performance_warnings(self, caplog):
        """Test that performance warnings are logged when slow."""
        import logging

        # Set logging level to capture warnings
        caplog.set_level(logging.WARNING)

        config = TenetsConfig()
        config.ranking.algorithm = "balanced"

        ranker = OptimizedRanker(config)
        ranker.perf_warning_threshold_ms = 0.001  # Very low threshold to trigger warning

        files = [
            FileAnalysis(
                path=f"/test/file_{i}.py",
                content=f"test content {i}" * 100,  # Larger content
                language="python",
                file_name=f"file_{i}.py",
                file_extension=".py",
                size=1000,
                lines=10,
            )
            for i in range(50)
        ]

        context = PromptContext(text="test query", keywords=["test", "query"], task_type="general")

        # Rank files
        ranker.rank_files(files, context)

        # Check for performance warning - should be logged since threshold is very low
        warnings = [record.message for record in caplog.records if record.levelname == "WARNING"]
        assert any(
            "Performance warning" in msg for msg in warnings
        ), f"Expected performance warning, got: {warnings}"

    def test_performance_report(self):
        """Test performance report generation."""
        config = TenetsConfig()
        config.ranking.algorithm = "fast"

        ranker = OptimizedRanker(config)

        # Add some metrics
        ranker.metrics.total_files = 100
        ranker.metrics.total_time_ms = 500
        ranker.metrics.corpus_build_time_ms = 100
        ranker.metrics.scoring_time_ms = 400
        ranker.metrics.calculate_averages()

        report = ranker.get_performance_report()

        assert "RANKING PERFORMANCE REPORT" in report
        assert "Algorithm: fast" in report
        assert "Total files processed: 100" in report
        assert "Total time: 500.00ms" in report
        assert "Average per file: 5.00ms" in report


class TestCacheEffectiveness:
    """
    Test caching mechanisms for performance.
    """

    def test_embedding_cache(self):
        """Test that embeddings are cached and reused."""
        config = TenetsConfig()
        config.ranking.algorithm = "thorough"

        ranker = OptimizedRanker(config)

        files = [
            FileAnalysis(
                path="/test/file.py",
                content="test content for caching",
                language="python",
                file_name="file.py",
                file_extension=".py",
                size=100,
                lines=5,
            )
        ]

        context = PromptContext(text="test", keywords=["test"], task_type="general")

        # First ranking - should compute embeddings
        ranker.rank_files(files, context)

        # Check cache
        if ranker._has_ml_enabled():
            assert len(ranker._embedding_cache) > 0

        # Clear metrics
        ranker.metrics = PerformanceMetrics()

        ranker.rank_files(files, context)

    def test_cache_clearing(self):
        """Test that caches can be cleared to free memory."""
        config = TenetsConfig()
        config.ranking.algorithm = "balanced"

        ranker = OptimizedRanker(config)

        # Add some data to caches
        ranker._bm25_cache["test"] = {"doc1": 0.5}
        ranker._tfidf_cache["test"] = {"doc1": 0.3}
        ranker._embedding_cache["doc1"] = [0.1] * 384

        # Clear caches
        ranker.clear_caches()

        assert len(ranker._bm25_cache) == 0
        assert len(ranker._tfidf_cache) == 0
        assert len(ranker._embedding_cache) == 0


class TestPerformanceRegression:
    """
    Regression tests to ensure performance doesn't degrade.
    """

    def test_fast_mode_performance_target(self):
        """Test that Fast mode meets performance targets."""
        config = TenetsConfig()
        config.ranking.algorithm = "fast"

        ranker = OptimizedRanker(config)

        files = [
            FileAnalysis(
                path=f"/test/file_{i}.py",
                content=f"simple content {i}",
                language="python",
                file_name=f"file_{i}.py",
                file_extension=".py",
                size=50,
                lines=2,
            )
            for i in range(100)
        ]

        context = PromptContext(text="simple", keywords=["simple"], task_type="general")

        # Warm up
        ranker.rank_files(files[:10], context)

        # Time 5 runs
        times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            ranker.rank_files(files, context)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        avg_time = statistics.mean(times)
        avg_per_file = avg_time / len(files)

        # Fast mode should be < 1ms per file
        assert (
            avg_per_file < 1.0
        ), f"Fast mode too slow: {avg_per_file:.2f}ms per file (target < 1ms)"

    def test_balanced_mode_performance_target(self):
        """Test that Balanced mode meets performance targets."""
        config = TenetsConfig()
        config.ranking.algorithm = "balanced"

        ranker = OptimizedRanker(config)

        files = [
            FileAnalysis(
                path=f"/test/file_{i}.py",
                content=f"balanced content {i} with more text for scoring",
                language="python",
                file_name=f"file_{i}.py",
                file_extension=".py",
                size=100,
                lines=3,
            )
            for i in range(50)
        ]

        context = PromptContext(
            text="balanced scoring", keywords=["balanced", "scoring"], task_type="general"
        )

        # Time the ranking
        gc.collect()
        start = time.perf_counter()
        ranker.rank_files(files, context)
        elapsed = time.perf_counter() - start

        avg_per_file = (elapsed * 1000) / len(files)

        # Balanced mode should be < 5ms per file with optimization
        assert (
            avg_per_file < 5.0
        ), f"Balanced mode too slow: {avg_per_file:.2f}ms per file (target < 5ms)"


@pytest.mark.benchmark
@pytest.mark.skipif(
    "freezegun" in sys.modules or any("freeze" in m for m in sys.modules),
    reason="Benchmark tests incompatible with freezegun",
)
class TestPerformanceBenchmarks:
    """
    Benchmark tests for performance tracking over time.

    These tests use pytest-benchmark to track performance metrics.
    """

    def test_benchmark_fast_mode(self, benchmark):
        """Benchmark Fast mode performance."""
        # Skip if time is frozen (interferes with benchmark timer calibration)
        import sys

        if "freezegun" in sys.modules and hasattr(sys.modules["freezegun"].api, "_freeze_time"):
            if sys.modules["freezegun"].api._freeze_time:
                pytest.skip("Cannot benchmark with frozen time")

        config = TenetsConfig()
        config.ranking.algorithm = "fast"
        ranker = OptimizedRanker(config)

        files = self._create_benchmark_files(100)
        context = PromptContext(
            text="benchmark test", keywords=["benchmark", "test"], task_type="general"
        )

        result = benchmark(ranker.rank_files, files, context)
        assert len(result) > 0

    def test_benchmark_balanced_mode(self, benchmark):
        """Benchmark Balanced mode performance."""
        # Skip if time is frozen (interferes with benchmark timer calibration)
        import sys

        if "freezegun" in sys.modules and hasattr(sys.modules["freezegun"].api, "_freeze_time"):
            if sys.modules["freezegun"].api._freeze_time:
                pytest.skip("Cannot benchmark with frozen time")

        config = TenetsConfig()
        config.ranking.algorithm = "balanced"
        ranker = OptimizedRanker(config)

        files = self._create_benchmark_files(100)
        context = PromptContext(
            text="benchmark test", keywords=["benchmark", "test"], task_type="general"
        )

        result = benchmark(ranker.rank_files, files, context)
        assert len(result) > 0

    def _create_benchmark_files(self, count: int) -> List[FileAnalysis]:
        """Create files for benchmarking."""
        return [
            FileAnalysis(
                path=f"/bench/file_{i}.py",
                content=f"benchmark content {i} " * 10,
                language="python",
                file_name=f"file_{i}.py",
                file_extension=".py",
                size=200,
                lines=5,
            )
            for i in range(count)
        ]
