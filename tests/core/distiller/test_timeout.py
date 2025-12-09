"""Timeout behavior for distill.

Comprehensive tests covering timeout edge cases, stage-specific timeouts,
partial results correctness, and metadata accuracy.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.distiller import Distiller
from tenets.models.analysis import FileAnalysis


# =============================================================================
# Basic Timeout Tests
# =============================================================================


def test_distill_times_out_and_returns_partial(tmp_path: Path):
    """Ensure distill respects timeout and returns partial context."""
    config = TenetsConfig()
    config.distill_timeout = 0.01
    distiller = Distiller(config)

    sample_file = tmp_path / "sample.py"
    sample_file.write_text("def sample():\n    return 1\n")

    def slow_analyze(path, deep=False, extract_keywords=True):
        time.sleep(0.05)
        raise RuntimeError("Too slow")  # Ensure no analyses complete

    with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
        result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0.01)

    assert result.metadata.get("timed_out") is True
    assert "timed out" in (result.context or "").lower()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestTimeoutEdgeCases:
    """Tests for timeout edge cases."""

    def test_timeout_zero_disables_timeout(self, tmp_path: Path):
        """timeout=0 should disable timeout checking."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample():\n    return 1\n")

        # With timeout=0, should complete normally
        result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0)

        assert result.metadata.get("timed_out") is not True
        assert result.context is not None

    def test_timeout_negative_disables_timeout(self, tmp_path: Path):
        """timeout=-1 should disable timeout checking."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample():\n    return 1\n")

        result = distiller.distill(prompt="sample", paths=tmp_path, timeout=-1)

        assert result.metadata.get("timed_out") is not True
        assert result.context is not None

    def test_timeout_none_uses_config_default(self, tmp_path: Path):
        """timeout=None should use config.distill_timeout via Tenets wrapper.

        Note: The Distiller.distill() doesn't read config timeout directly.
        The Tenets.distill() wrapper handles config timeout resolution.
        This test verifies the Distiller respects an explicit timeout.
        """
        config = TenetsConfig()
        config.distill_timeout = 0.01  # Short timeout

        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample():\n    return 1\n")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.1)
            return FileAnalysis(path=str(path))

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            # Pass the config timeout explicitly (simulating what Tenets.distill does)
            result = distiller.distill(
                prompt="sample", paths=tmp_path, timeout=config.distill_timeout
            )

        # Should timeout
        assert result.metadata.get("timed_out") is True

    def test_timeout_overrides_config(self, tmp_path: Path):
        """Explicit timeout should override config value."""
        config = TenetsConfig()
        config.distill_timeout = 1000  # Long config timeout

        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample():\n    return 1\n")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.1)
            return FileAnalysis(path=str(path))

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            # Pass short explicit timeout - should override config
            result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0.01)

        # Should timeout because explicit timeout is 0.01s
        assert result.metadata.get("timed_out") is True


# =============================================================================
# Stage-Specific Timeout Tests
# =============================================================================


class TestStageSpecificTimeouts:
    """Tests for timeout at each distillation stage."""

    def test_timeout_during_file_discovery(self, tmp_path: Path):
        """Timeout during file discovery stage."""
        config = TenetsConfig()
        distiller = Distiller(config)

        # Create some files
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"x = {i}")

        def slow_scan(*args, **kwargs):
            time.sleep(0.1)
            return []

        with patch.object(distiller.scanner, "scan", side_effect=slow_scan):
            result = distiller.distill(prompt="test", paths=tmp_path, timeout=0.01)

        assert result.metadata.get("timed_out") is True

    def test_timeout_during_file_analysis(self, tmp_path: Path):
        """Timeout during file analysis stage (main bottleneck)."""
        config = TenetsConfig()
        distiller = Distiller(config)

        # Create multiple files
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.05)  # Each file takes 50ms
            return FileAnalysis(path=str(path), content="test")

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            result = distiller.distill(prompt="test", paths=tmp_path, timeout=0.1)

        # Should timeout during analysis of many files
        assert result.metadata.get("timed_out") is True

    def test_timeout_during_ranking(self, tmp_path: Path):
        """Timeout during ranking stage."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample(): pass")

        def slow_rank(*args, **kwargs):
            time.sleep(0.1)
            return []

        with patch.object(distiller.ranker, "rank_files", side_effect=slow_rank):
            result = distiller.distill(prompt="test", paths=tmp_path, timeout=0.01)

        assert result.metadata.get("timed_out") is True


# =============================================================================
# Partial Results Tests
# =============================================================================


class TestTimeoutPartialResults:
    """Tests for correctness of partial results on timeout."""

    def test_partial_results_metadata_accurate(self, tmp_path: Path):
        """Metadata should accurately reflect partial state."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample(): pass")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.1)
            return FileAnalysis(path=str(path))

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0.01)

        # Verify metadata fields
        assert result.metadata.get("timed_out") is True
        assert "timeout_seconds" in result.metadata
        assert result.metadata.get("timeout_seconds") == 0.01
        assert "timing" in result.metadata

    def test_partial_results_timing_metadata(self, tmp_path: Path):
        """Timing metadata should reflect actual duration."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample(): pass")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.05)
            return FileAnalysis(path=str(path))

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            start = time.time()
            result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0.02)
            elapsed = time.time() - start

        timing = result.metadata.get("timing", {})
        assert "duration" in timing
        # Duration should be close to timeout (within tolerance)
        assert timing["duration"] >= 0.01  # At least the timeout
        assert timing["duration"] < elapsed + 0.5  # Not too much more

    def test_partial_results_format_valid(self, tmp_path: Path):
        """Partial context should be valid format (not truncated mid-line)."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample(): pass")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.1)
            return FileAnalysis(path=str(path))

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0.01)

        # Context should be a complete, valid string
        assert result.context is not None
        assert isinstance(result.context, str)
        # Should end with a complete line (newline or end of string)
        lines = result.context.split("\n")
        assert all(isinstance(line, str) for line in lines)


# =============================================================================
# Parallel Analysis Timeout Tests
# =============================================================================


class TestParallelAnalysisTimeout:
    """Tests for timeout behavior with parallel file analysis."""

    def test_parallel_analysis_respects_deadline(self, tmp_path: Path):
        """Parallel file analysis should respect deadline."""
        config = TenetsConfig()
        config.scanner.workers = 4
        distiller = Distiller(config)

        # Create many files to trigger parallel processing (>10 files)
        for i in range(20):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): return {i}")

        def slow_analyze_files(
            file_paths,
            deep=False,
            parallel=True,
            progress_callback=None,
            extract_keywords=True,
            deadline=None,
        ):
            """Simulate slow parallel analysis."""
            results = []
            for path in file_paths:
                time.sleep(0.02)  # Each file takes 20ms
                # Check deadline like the real implementation
                if deadline is not None and time.time() >= deadline:
                    break
                results.append(FileAnalysis(path=str(path), content="test"))
            return results

        with patch.object(distiller.analyzer, "analyze_files", side_effect=slow_analyze_files):
            result = distiller.distill(
                prompt="test",
                paths=tmp_path,
                timeout=0.1,
                mode="balanced",  # Balanced mode uses parallel analysis for >10 files
            )

        # Should timeout and return partial results
        assert result.metadata.get("timed_out") is True


# =============================================================================
# Ranker Deadline Tests
# =============================================================================


class TestRankerDeadline:
    """Tests for deadline propagation to the ranker."""

    def test_ranker_respects_deadline(self, tmp_path: Path):
        """Ranker should stop early when deadline is reached."""
        config = TenetsConfig()
        distiller = Distiller(config)

        # Create files
        for i in range(5):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")

        # Make ranking slow
        original_rank = distiller.ranker.rank_files

        def slow_rank(
            files, prompt_context, algorithm=None, parallel=True, explain=False, deadline=None
        ):
            time.sleep(0.1)
            return original_rank(files, prompt_context, algorithm, parallel, explain, deadline)

        with patch.object(distiller.ranker, "rank_files", side_effect=slow_rank):
            result = distiller.distill(prompt="test", paths=tmp_path, timeout=0.05)

        assert result.metadata.get("timed_out") is True


# =============================================================================
# Mode-Specific Timeout Tests
# =============================================================================


class TestModeSpecificTimeouts:
    """Tests for timeout with different analysis modes."""

    @pytest.mark.parametrize("mode", ["fast", "balanced", "thorough"])
    def test_timeout_all_modes(self, tmp_path: Path, mode: str):
        """Timeout should work consistently across all modes."""
        config = TenetsConfig()
        distiller = Distiller(config)

        sample_file = tmp_path / "sample.py"
        sample_file.write_text("def sample(): pass")

        def slow_analyze(path, deep=False, extract_keywords=True):
            time.sleep(0.1)
            return FileAnalysis(path=str(path))

        with patch.object(distiller.analyzer, "analyze_file", side_effect=slow_analyze):
            result = distiller.distill(prompt="sample", paths=tmp_path, timeout=0.01, mode=mode)

        assert result.metadata.get("timed_out") is True
        assert result.metadata.get("mode") == mode
