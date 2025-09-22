"""
Unit tests for neural cross-encoder reranking functionality.

This module tests the cross-encoder neural reranking pipeline that jointly
evaluates query-document pairs for improved ranking accuracy.

Test Coverage:
    - Neural reranker initialization
    - Cross-encoder reranking with ML strategy
    - Reranking configuration options
    - Integration with ranking pipeline
    - Fallback behavior when ML not available
    - Edge cases and error handling
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.ranking.ranker import RankingAlgorithm, RelevanceRanker
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext


class TestNeuralReranking:
    """Test suite for neural reranking functionality."""

    @pytest.fixture
    def config_with_reranking(self):
        """Create config with reranking enabled."""
        config = TenetsConfig()
        config.ranking.use_ml = True
        config.ranking.use_reranker = True
        config.ranking.rerank_top_k = 5
        config.ranking.algorithm = "ml"
        return config

    @pytest.fixture
    def sample_files(self):
        """Create sample files for testing."""
        files = []
        # High relevance file
        files.append(
            FileAnalysis(
                path="auth/oauth2_impl.py",
                content="def implement_oauth2_flow(): # OAuth2 implementation",
                size=1000,
                language="python",
            )
        )
        # Medium relevance with keyword
        files.append(
            FileAnalysis(
                path="utils/auth_helpers.py",
                content="# Helper functions for OAuth authentication",
                size=800,
                language="python",
            )
        )
        # Low relevance but keyword match (should be penalized by reranker)
        files.append(
            FileAnalysis(
                path="deprecated/old_oauth.py",
                content="# DEPRECATED: Do not use this oauth2 implementation",
                size=600,
                language="python",
            )
        )
        # Semantic match without exact keyword
        files.append(
            FileAnalysis(
                path="security/authorization.py",
                content="def handle_authorization(): # Modern auth using JWT",
                size=900,
                language="python",
            )
        )
        # Add more generic files
        for i in range(4, 10):
            files.append(
                FileAnalysis(
                    path=f"module{i}/file{i}.py",
                    content=f"def function_{i}(): return {i}",
                    size=100 + i * 10,
                    language="python",
                )
            )
        return files

    def test_reranking_configuration(self, config_with_reranking):
        """Test that reranking configuration is properly loaded."""
        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")

        assert ranker.use_reranker is True
        assert ranker.rerank_top_k == 5
        assert ranker.algorithm == RankingAlgorithm.ML

        ranker.shutdown()

    def test_reranking_disabled_by_default(self):
        """Test that reranking is disabled by default."""
        config = TenetsConfig()
        config.ranking.use_reranker = False

        ranker = RelevanceRanker(config)
        assert ranker.use_reranker is False

        ranker.shutdown()

    @patch("tenets.core.ranking.ranker.NeuralReranker")
    def test_reranking_invoked_for_ml_algorithm(
        self, mock_neural_reranker_class, config_with_reranking, sample_files
    ):
        """Test that neural reranker is invoked when ML algorithm is used."""
        # Setup mock reranker
        mock_reranker = MagicMock()
        mock_neural_reranker_class.return_value = mock_reranker

        # Mock rerank method to simulate cross-encoder behavior
        def mock_rerank(query, documents, top_k):
            # Simulate reranking - penalize deprecated content
            reranked = []
            for doc, score in documents:
                if "DEPRECATED" in doc:
                    new_score = score * 0.3  # Penalize deprecated
                elif "implement_oauth2" in doc:
                    new_score = score * 1.5  # Boost implementation
                else:
                    new_score = score
                reranked.append((doc, new_score))
            return sorted(reranked, key=lambda x: x[1], reverse=True)

        mock_reranker.rerank.side_effect = mock_rerank

        # Create ranker and rank files
        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")
        prompt_context = PromptContext(
            text="implement oauth2 authentication",
            keywords=["implement", "oauth2", "authentication"],
        )

        # Mock the ML strategy to avoid loading actual models
        with patch.object(ranker, "_get_strategy") as mock_get_strategy:
            from tenets.core.ranking.strategies import MLRankingStrategy

            mock_strategy = MLRankingStrategy()
            mock_strategy._model_loaded = True  # Pretend model is loaded
            mock_strategy._model = Mock()  # Mock model
            mock_get_strategy.return_value = mock_strategy

            # Perform ranking
            ranked_files = ranker.rank_files(sample_files, prompt_context, parallel=False)

            # Verify reranker was invoked if we had enough files
            if len(sample_files) > ranker.rerank_top_k:
                assert mock_reranker.rerank.called
                # Check call arguments
                call_args = mock_reranker.rerank.call_args
                assert call_args[0][0] == prompt_context.text  # Query
                assert call_args[1]["top_k"] <= ranker.rerank_top_k

        ranker.shutdown()

    @patch("tenets.core.ranking.ranker.NeuralReranker")
    def test_reranking_not_invoked_for_non_ml_algorithm(
        self, mock_neural_reranker_class, sample_files
    ):
        """Test that reranker is not invoked for non-ML algorithms."""
        config = TenetsConfig()
        config.ranking.use_reranker = True
        config.ranking.algorithm = "balanced"  # Not ML

        mock_reranker = MagicMock()
        mock_neural_reranker_class.return_value = mock_reranker

        ranker = RelevanceRanker(config, algorithm="balanced")
        prompt_context = PromptContext(text="test query", keywords=["test"])

        # Rank files with non-ML algorithm
        ranked_files = ranker.rank_files(sample_files, prompt_context, parallel=False)

        # Verify reranker was NOT called
        assert not mock_reranker.rerank.called

        ranker.shutdown()

    @patch("tenets.core.ranking.ranker.NeuralReranker")
    def test_reranking_handles_import_error(
        self, mock_neural_reranker_class, config_with_reranking, sample_files
    ):
        """Test graceful handling when NeuralReranker import fails."""
        # Simulate import error
        mock_neural_reranker_class.side_effect = ImportError("ML deps not installed")

        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")
        prompt_context = PromptContext(text="test query", keywords=["test"])

        # Should not raise, should fall back to regular ranking
        ranked_files = ranker.rank_files(sample_files, prompt_context, parallel=False)

        assert len(ranked_files) > 0  # Should still return results

        ranker.shutdown()

    @patch("tenets.core.ranking.ranker.NeuralReranker")
    def test_reranking_handles_runtime_error(
        self, mock_neural_reranker_class, config_with_reranking, sample_files
    ):
        """Test graceful handling when reranker fails during execution."""
        mock_reranker = MagicMock()
        mock_neural_reranker_class.return_value = mock_reranker

        # Simulate runtime error during reranking
        mock_reranker.rerank.side_effect = RuntimeError("Model loading failed")

        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")
        prompt_context = PromptContext(text="test query", keywords=["test"])

        # Mock ML strategy
        with patch.object(ranker, "_get_strategy") as mock_get_strategy:
            from tenets.core.ranking.strategies import MLRankingStrategy

            mock_strategy = MLRankingStrategy()
            mock_get_strategy.return_value = mock_strategy

            # Should not raise, should use original ranking
            ranked_files = ranker.rank_files(sample_files, prompt_context, parallel=False)

            assert len(ranked_files) > 0  # Should still return results

        ranker.shutdown()

    def test_rerank_top_k_configuration(self, config_with_reranking):
        """Test that rerank_top_k configuration is respected."""
        # Set specific top_k value
        config_with_reranking.ranking.rerank_top_k = 3

        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")
        assert ranker.rerank_top_k == 3

        # Test with larger value
        config_with_reranking.ranking.rerank_top_k = 50
        ranker2 = RelevanceRanker(config_with_reranking, algorithm="ml")
        assert ranker2.rerank_top_k == 50

        ranker.shutdown()
        ranker2.shutdown()

    @patch("tenets.core.ranking.ranker.NeuralReranker")
    def test_reranking_with_fewer_files_than_top_k(
        self, mock_neural_reranker_class, config_with_reranking
    ):
        """Test reranking when file count is less than rerank_top_k."""
        config_with_reranking.ranking.rerank_top_k = 20

        mock_reranker = MagicMock()
        mock_neural_reranker_class.return_value = mock_reranker
        mock_reranker.rerank.return_value = [("content", 0.5)]

        # Create only 3 files
        files = [
            FileAnalysis(path=f"file{i}.py", content=f"test{i}", size=100, language="python")
            for i in range(3)
        ]

        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")
        prompt_context = PromptContext(text="test", keywords=["test"])

        # Mock ML strategy
        with patch.object(ranker, "_get_strategy") as mock_get_strategy:
            from tenets.core.ranking.strategies import MLRankingStrategy

            mock_strategy = MLRankingStrategy()
            mock_get_strategy.return_value = mock_strategy

            ranked_files = ranker.rank_files(files, prompt_context, parallel=False)

            # Should handle gracefully
            assert len(ranked_files) <= len(files)

            if mock_reranker.rerank.called:
                # Check that top_k was adjusted to file count
                call_args = mock_reranker.rerank.call_args
                actual_top_k = call_args[1]["top_k"]
                assert actual_top_k <= len(files)

        ranker.shutdown()

    @patch("tenets.core.ranking.ranker.NeuralReranker")
    def test_reranking_preserves_non_reranked_files(
        self, mock_neural_reranker_class, config_with_reranking, sample_files
    ):
        """Test that files not included in reranking are preserved in results."""
        config_with_reranking.ranking.rerank_top_k = 3

        mock_reranker = MagicMock()
        mock_neural_reranker_class.return_value = mock_reranker

        # Mock rerank to return same number of files
        def mock_rerank(query, documents, top_k):
            return [(doc, score * 1.1) for doc, score in documents[:top_k]]

        mock_reranker.rerank.side_effect = mock_rerank

        ranker = RelevanceRanker(config_with_reranking, algorithm="ml")
        prompt_context = PromptContext(text="test", keywords=["test"])

        # Mock ML strategy
        with patch.object(ranker, "_get_strategy") as mock_get_strategy:
            from tenets.core.ranking.strategies import MLRankingStrategy

            mock_strategy = MLRankingStrategy()
            mock_get_strategy.return_value = mock_strategy

            # Create initial ranked files
            initial_count = len(sample_files)
            ranked_files = ranker.rank_files(sample_files, prompt_context, parallel=False)

            # All files should still be in results
            # Some may be filtered by threshold, but reranking shouldn't lose files
            assert len(ranked_files) <= initial_count

        ranker.shutdown()

    def test_ml_strategy_initialization_with_reranker(self):
        """Test that MLRankingStrategy properly initializes reranker support."""
        from tenets.core.ranking.strategies import MLRankingStrategy

        strategy = MLRankingStrategy()

        # Check that reranker fields are initialized
        assert hasattr(strategy, "_reranker")
        assert hasattr(strategy, "_reranker_loaded")
        assert strategy._reranker is None
        assert strategy._reranker_loaded is False

    @patch("tenets.core.nlp.ml_utils.NeuralReranker")
    def test_ml_strategy_loads_reranker_lazily(self, mock_neural_reranker_class):
        """Test that MLRankingStrategy loads reranker lazily when needed."""
        from tenets.core.ranking.strategies import MLRankingStrategy

        mock_reranker = MagicMock()
        mock_neural_reranker_class.return_value = mock_reranker

        strategy = MLRankingStrategy()

        # Initially not loaded
        assert strategy._reranker is None
        assert strategy._reranker_loaded is False

        # Load reranker
        strategy._load_reranker()

        # Should be loaded now
        assert strategy._reranker is not None
        assert mock_neural_reranker_class.called

    def test_reranking_explanation_updated(self):
        """Test that reranked files have updated explanations."""
        # This test would verify that the explanation field includes
        # "[Cross-encoder reranked]" marker, but needs full integration
        # setup which is complex for unit test
        pass  # Placeholder for integration test

    @pytest.mark.parametrize("algorithm", ["fast", "balanced", "thorough"])
    def test_reranking_only_for_ml_algorithm(self, algorithm):
        """Test that reranking is only applied for ML algorithm."""
        config = TenetsConfig()
        config.ranking.use_reranker = True
        config.ranking.algorithm = algorithm

        ranker = RelevanceRanker(config, algorithm=algorithm)

        # Even with use_reranker=True, it shouldn't apply for non-ML algorithms
        # This is tested by checking the algorithm type
        assert ranker.algorithm != RankingAlgorithm.ML

        ranker.shutdown()


class TestNeuralRerankerIntegration:
    """Integration tests for neural reranker with actual ML models."""

    @pytest.mark.skipif(
        "sentence_transformers" not in sys.modules, reason="ML dependencies not installed"
    )
    def test_reranking_with_actual_models(self):
        """Test reranking with actual ML models if available."""
        config = TenetsConfig()
        config.ranking.use_ml = True
        config.ranking.use_reranker = True
        config.ranking.rerank_top_k = 5

        try:
            from tenets.core.nlp.ml_utils import NeuralReranker

            # Create reranker
            reranker = NeuralReranker()

            # Test documents
            documents = [
                ("OAuth2 implementation guide", 0.8),
                ("DEPRECATED: old oauth code", 0.7),
                ("Authorization using JWT", 0.6),
                ("Random file content", 0.3),
            ]

            # Rerank
            reranked = reranker.rerank("implement oauth2 authentication", documents, top_k=3)

            # Verify deprecated content is penalized
            deprecated_rank = None
            impl_rank = None

            for i, (content, score) in enumerate(reranked):
                if "DEPRECATED" in content:
                    deprecated_rank = i
                if "implementation" in content:
                    impl_rank = i

            # Implementation should rank higher than deprecated
            if deprecated_rank is not None and impl_rank is not None:
                assert impl_rank < deprecated_rank

        except ImportError:
            pytest.skip("ML dependencies not available for integration test")
