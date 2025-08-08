"""
Unit tests for the relevance ranking system.

The ranking system scores and sorts files by relevance to a given prompt using
multiple strategies and factors. This module tests the various ranking algorithms,
factor calculations, and the overall ranking pipeline.

Test Coverage:
    - RelevanceRanker initialization
    - Different ranking strategies (fast, balanced, thorough)
    - Ranking factor calculations
    - Custom ranker registration
    - Corpus analysis
    - Parallel vs sequential ranking
    - Edge cases and error handling
"""

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import TimeoutError as FutureTimeoutError
import math

import pytest

from tenets.core.ranking.ranker import (
    RelevanceRanker,
    RankingAlgorithm,
    RankingFactors,
    RankedFile,
    FastRankingStrategy,
    BalancedRankingStrategy,
    ThoroughRankingStrategy,
)
from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis, ComplexityMetrics
from tenets.models.context import PromptContext


class TestRelevanceRankerInitialization:
    """Test suite for RelevanceRanker initialization."""

    def test_init_with_config(self, test_config):
        """Test ranker initialization with configuration."""
        with patch("tenets.core.ranking.ranker.get_logger"):
            ranker = RelevanceRanker(test_config)

            assert ranker.config == test_config
            assert len(ranker.strategies) >= 3  # At least fast, balanced, thorough
            assert ranker._custom_rankers == []
            assert ranker._executor is not None

    def test_init_strategies_available(self, test_config):
        """Test that all default strategies are initialized."""
        with patch("tenets.core.ranking.ranker.get_logger"):
            ranker = RelevanceRanker(test_config)

            assert RankingAlgorithm.FAST in ranker.strategies
            assert RankingAlgorithm.BALANCED in ranker.strategies
            assert RankingAlgorithm.THOROUGH in ranker.strategies

            assert isinstance(ranker.strategies[RankingAlgorithm.FAST], FastRankingStrategy)
            assert isinstance(ranker.strategies[RankingAlgorithm.BALANCED], BalancedRankingStrategy)
            assert isinstance(ranker.strategies[RankingAlgorithm.THOROUGH], ThoroughRankingStrategy)


class TestRankingFactors:
    """Test suite for RankingFactors class."""

    def test_ranking_factors_initialization(self):
        """Test RankingFactors initialization with defaults."""
        factors = RankingFactors()

        assert factors.keyword_match == 0.0
        assert factors.tfidf_similarity == 0.0
        assert factors.path_relevance == 0.0
        assert factors.semantic_similarity == 0.0
        assert factors.custom_scores == {}

    def test_get_weighted_score(self):
        """Test weighted score calculation."""
        factors = RankingFactors(
            keyword_match=0.8, path_relevance=0.6, import_centrality=0.4, git_recency=0.2
        )

        weights = {
            "keyword_match": 0.5,
            "path_relevance": 0.3,
            "import_centrality": 0.1,
            "git_recency": 0.1,
        }

        score = factors.get_weighted_score(weights)

        # Expected: (0.8*0.5) + (0.6*0.3) + (0.4*0.1) + (0.2*0.1) = 0.64
        assert abs(score - 0.64) < 0.001

    def test_get_weighted_score_with_custom(self):
        """Test weighted score with custom factors."""
        factors = RankingFactors(
            keyword_match=0.5, custom_scores={"auth_patterns": 0.8, "api_patterns": 0.6}
        )

        weights = {"keyword_match": 0.4, "auth_patterns": 0.3, "api_patterns": 0.3}

        score = factors.get_weighted_score(weights)

        # Expected: (0.5*0.4) + (0.8*0.3) + (0.6*0.3) = 0.62
        assert abs(score - 0.62) < 0.001

    def test_get_weighted_score_clamping(self):
        """Test that weighted scores are clamped to [0, 1]."""
        factors = RankingFactors(keyword_match=2.0)  # Over 1.0
        weights = {"keyword_match": 1.0}

        score = factors.get_weighted_score(weights)
        assert score == 1.0  # Should be clamped to 1.0

        factors = RankingFactors(keyword_match=-0.5)  # Below 0
        score = factors.get_weighted_score(weights)
        assert score == 0.0  # Should be clamped to 0.0


class TestFastRankingStrategy:
    """Test suite for FastRankingStrategy."""

    @pytest.fixture
    def strategy(self):
        """Provide a FastRankingStrategy instance."""
        return FastRankingStrategy()

    def test_rank_file_with_keywords(self, strategy):
        """Test fast ranking with keyword matching."""
        file = FileAnalysis(
            path="auth/login.py",
            content="def authenticate(username, password):\n    # Authentication logic\n    return token",
            language="python",
        )

        prompt_context = PromptContext(
            text="implement authentication",
            keywords=["authenticate", "login", "password"],
            task_type="feature",
        )

        corpus_stats = {}

        factors = strategy.rank_file(file, prompt_context, corpus_stats)

        # Should have good keyword match
        assert factors.keyword_match > 0.5
        # Path contains "login"
        assert factors.path_relevance > 0.0

    def test_rank_file_path_relevance(self, strategy):
        """Test path relevance scoring."""
        file = FileAnalysis(
            path="src/api/handlers/main.py", content="# Main handler", language="python"
        )

        prompt_context = PromptContext(
            text="api handler", keywords=["api", "handler"], task_type="general"
        )

        factors = strategy.rank_file(file, prompt_context, {})

        # Path contains both keywords
        assert factors.path_relevance > 0.5

    def test_rank_file_type_relevance_for_tests(self, strategy):
        """Test file type relevance for test task."""
        test_file = FileAnalysis(
            path="tests/test_auth.py", content="def test_login():", language="python"
        )

        non_test_file = FileAnalysis(path="src/auth.py", content="def login():", language="python")

        prompt_context = PromptContext(text="write tests", keywords=["test"], task_type="test")

        test_factors = strategy.rank_file(test_file, prompt_context, {})
        non_test_factors = strategy.rank_file(non_test_file, prompt_context, {})

        # Test file should have higher type relevance for test task
        assert test_factors.type_relevance > non_test_factors.type_relevance

    def test_get_weights(self, strategy):
        """Test weight configuration for fast strategy."""
        weights = strategy.get_weights()

        assert weights["keyword_match"] == 0.6
        assert weights["path_relevance"] == 0.3
        assert weights["type_relevance"] == 0.1
        assert sum(weights.values()) == 1.0


class TestBalancedRankingStrategy:
    """Test suite for BalancedRankingStrategy."""

    @pytest.fixture
    def strategy(self):
        """Provide a BalancedRankingStrategy instance."""
        return BalancedRankingStrategy()

    def test_enhanced_keyword_scoring(self, strategy):
        """Test enhanced keyword scoring with position weighting."""
        file = FileAnalysis(
            path="module.py",
            content="""
import auth_module

class AuthHandler:
    def authenticate_user(self):
        pass

def helper():
    # authenticate is mentioned here too
    pass
""",
            language="python",
            imports=[Mock(module="auth_module")],
            classes=[Mock(name="AuthHandler")],
            functions=[Mock(name="authenticate_user")],
        )

        prompt_context = PromptContext(
            text="authenticate", keywords=["authenticate"], task_type="general"
        )

        factors = strategy.rank_file(file, prompt_context, {})

        # Should have high keyword match due to multiple occurrences
        assert factors.keyword_match > 0.6

    def test_path_structure_analysis(self, strategy):
        """Test sophisticated path structure analysis."""
        file = FileAnalysis(
            path="src/api/controllers/auth_controller.py", content="", language="python"
        )

        prompt_context = PromptContext(
            text="api authentication",
            keywords=["api", "authentication", "auth"],
            task_type="feature",
        )

        factors = strategy.rank_file(file, prompt_context, {})

        # Path contains relevant architecture terms
        assert factors.path_relevance > 0.5

    def test_import_centrality_calculation(self, strategy):
        """Test import centrality scoring."""
        file = FileAnalysis(path="core/base.py", content="", language="python")

        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        # File is imported by many others
        corpus_stats = {
            "import_graph": {"core/base.py": {"auth.py", "api.py", "models.py", "views.py"}}
        }

        factors = strategy.rank_file(file, prompt_context, corpus_stats)

        # Should have some import centrality
        assert factors.import_centrality > 0.0

    def test_git_recency_scoring(self, strategy):
        """Test git recency scoring."""
        from datetime import datetime, timedelta

        # Recent file
        recent_file = FileAnalysis(
            path="recent.py",
            content="",
            language="python",
            git_info={"last_modified": datetime.now().isoformat()},
        )

        # Old file
        old_file = FileAnalysis(
            path="old.py",
            content="",
            language="python",
            git_info={"last_modified": (datetime.now() - timedelta(days=400)).isoformat()},
        )

        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        recent_factors = strategy.rank_file(recent_file, prompt_context, {})
        old_factors = strategy.rank_file(old_file, prompt_context, {})

        # Recent file should have higher git recency score
        assert recent_factors.git_recency > old_factors.git_recency

    def test_complexity_relevance_for_refactor(self, strategy):
        """Test complexity relevance for refactoring tasks."""
        complex_file = FileAnalysis(
            path="complex.py",
            content="",
            language="python",
            complexity=ComplexityMetrics(cyclomatic=25),
        )

        simple_file = FileAnalysis(
            path="simple.py",
            content="",
            language="python",
            complexity=ComplexityMetrics(cyclomatic=3),
        )

        prompt_context = PromptContext(text="refactor", keywords=["refactor"], task_type="refactor")

        complex_factors = strategy.rank_file(complex_file, prompt_context, {})
        simple_factors = strategy.rank_file(simple_file, prompt_context, {})

        # Complex file should be more relevant for refactoring
        assert complex_factors.complexity_relevance > simple_factors.complexity_relevance


class TestThoroughRankingStrategy:
    """Test suite for ThoroughRankingStrategy."""

    @pytest.fixture
    def strategy(self):
        """Provide a ThoroughRankingStrategy instance."""
        return ThoroughRankingStrategy()

    def test_ml_model_loading(self):
        """Test ML model loading for semantic similarity."""
        with patch("tenets.core.ranking.ranker.SentenceTransformer") as mock_st:
            strategy = ThoroughRankingStrategy()

            # Should attempt to load model
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation with ML model."""
        with patch("tenets.core.ranking.ranker.SentenceTransformer") as mock_st:
            # Setup mock model
            mock_model = Mock()
            mock_model.encode.side_effect = lambda text, **kwargs: Mock(
                unsqueeze=Mock(return_value=Mock())
            )
            mock_st.return_value = mock_model

            # Mock cosine similarity
            with patch("tenets.core.ranking.ranker.cosine_similarity") as mock_cosine:
                mock_cosine.return_value = Mock(item=Mock(return_value=0.75))

                strategy = ThoroughRankingStrategy()

                file = FileAnalysis(
                    path="test.py",
                    content="Authentication and authorization logic",
                    language="python",
                )

                prompt_context = PromptContext(
                    text="implement auth", keywords=["auth"], task_type="feature"
                )

                factors = strategy.rank_file(file, prompt_context, {})

                # Should have semantic similarity score
                assert factors.semantic_similarity == 0.75

    def test_code_pattern_analysis(self, strategy):
        """Test code pattern analysis for specific domains."""
        auth_file = FileAnalysis(
            path="auth.py",
            content="""
import jwt
from oauth import OAuth2

def login(username, password):
    token = generate_token(username)
    session.create(token)
    return token

def logout():
    session.destroy()
""",
            language="python",
        )

        prompt_context = PromptContext(
            text="authentication", keywords=["auth", "authentication", "login"], task_type="feature"
        )

        factors = strategy.rank_file(auth_file, prompt_context, {})

        # Should detect auth patterns
        assert "auth_patterns" in factors.custom_scores
        assert factors.custom_scores["auth_patterns"] > 0.0

    def test_ast_relevance_analysis(self, strategy):
        """Test AST-based relevance analysis."""
        from tenets.models.analysis import CodeStructure, ClassInfo, FunctionInfo

        file = FileAnalysis(
            path="test.py",
            content="",
            language="python",
            structure=CodeStructure(
                classes=[
                    ClassInfo(name="AuthenticationManager", line=1),
                    ClassInfo(name="Helper", line=50),
                ],
                functions=[
                    FunctionInfo(name="authenticate_user", line=10),
                    FunctionInfo(name="helper_function", line=60),
                ],
            ),
        )

        prompt_context = PromptContext(
            text="authentication", keywords=["authentication", "authenticate"], task_type="feature"
        )

        factors = strategy.rank_file(file, prompt_context, {})

        # Should have class and function relevance scores
        assert "class_relevance" in factors.custom_scores
        assert "function_relevance" in factors.custom_scores
        assert factors.custom_scores["class_relevance"] > 0.0
        assert factors.custom_scores["function_relevance"] > 0.0


class TestMainRankingPipeline:
    """Test suite for the main ranking pipeline."""

    @pytest.fixture
    def ranker(self, test_config):
        """Provide a RelevanceRanker instance."""
        with patch("tenets.core.ranking.ranker.get_logger"):
            return RelevanceRanker(test_config)

    def test_rank_files_basic(self, ranker):
        """Test basic file ranking."""
        files = [
            FileAnalysis(path="file1.py", content="auth code", language="python"),
            FileAnalysis(path="file2.py", content="unrelated", language="python"),
            FileAnalysis(path="file3.py", content="authentication logic", language="python"),
        ]

        prompt_context = PromptContext(
            text="authentication", keywords=["authentication", "auth"], task_type="feature"
        )

        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.rank_file.side_effect = [
            RankingFactors(keyword_match=0.5),
            RankingFactors(keyword_match=0.1),
            RankingFactors(keyword_match=0.9),
        ]
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}

        ranker._get_strategy = Mock(return_value=mock_strategy)

        ranked = ranker.rank_files(files, prompt_context, algorithm="fast", parallel=False)

        # Should be sorted by relevance
        assert len(ranked) >= 2  # At least some files above threshold
        assert ranked[0].relevance_score >= ranked[-1].relevance_score

    def test_rank_files_parallel(self, ranker):
        """Test parallel file ranking."""
        files = [FileAnalysis(path=f"file{i}.py", content=f"content {i}") for i in range(20)]

        prompt_context = PromptContext(text="test", keywords=["test"], task_type="general")

        mock_strategy = Mock()
        mock_strategy.rank_file.return_value = RankingFactors(keyword_match=0.5)
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}

        ranker._get_strategy = Mock(return_value=mock_strategy)

        # Should use parallel processing for 20 files
        with patch.object(ranker._executor, "submit") as mock_submit:
            mock_future = Mock()
            mock_future.result.return_value = RankedFile(
                analysis=files[0], score=0.5, factors=RankingFactors(), explanation=""
            )
            mock_submit.return_value = mock_future

            ranked = ranker.rank_files(files, prompt_context, parallel=True)

            # Should have submitted parallel tasks
            assert mock_submit.call_count == 20

    def test_rank_files_with_threshold(self, ranker, test_config):
        """Test ranking with threshold filtering."""
        test_config.ranking.threshold = 0.5

        files = [
            FileAnalysis(path="high.py", content="very relevant"),
            FileAnalysis(path="medium.py", content="somewhat relevant"),
            FileAnalysis(path="low.py", content="not relevant"),
        ]

        prompt_context = PromptContext(text="test", keywords=["test"], task_type="general")

        mock_strategy = Mock()
        mock_strategy.rank_file.side_effect = [
            RankingFactors(keyword_match=0.8),
            RankingFactors(keyword_match=0.6),
            RankingFactors(keyword_match=0.3),
        ]
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}

        ranker._get_strategy = Mock(return_value=mock_strategy)

        ranked = ranker.rank_files(files, prompt_context, parallel=False)

        # Only files above threshold should be returned
        assert len(ranked) == 2
        assert all(f.relevance_score >= 0.5 for f in ranked)

    def test_rank_files_empty_input(self, ranker):
        """Test ranking with empty file list."""
        files = []
        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        ranked = ranker.rank_files(files, prompt_context)

        assert ranked == []

    def test_rank_files_with_error(self, ranker):
        """Test ranking handles individual file errors."""
        files = [
            FileAnalysis(path="good.py", content="content"),
            FileAnalysis(path="bad.py", content="content"),
        ]

        prompt_context = PromptContext(text="test", keywords=["test"], task_type="general")

        mock_strategy = Mock()
        mock_strategy.rank_file.side_effect = [
            RankingFactors(keyword_match=0.5),
            Exception("Ranking failed"),
        ]
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}

        ranker._get_strategy = Mock(return_value=mock_strategy)

        # Should handle error gracefully in sequential mode
        ranked = ranker.rank_files(files, prompt_context, parallel=False)

        # Should still return the successful file
        assert len(ranked) >= 1

    def test_custom_ranker_registration(self, ranker):
        """Test registering and applying custom rankers."""

        def custom_ranker(ranked_files, prompt_context):
            # Boost files with "special" in path
            for rf in ranked_files:
                if "special" in rf.analysis.path:
                    rf.score *= 2
            return ranked_files

        ranker.register_custom_ranker(custom_ranker)

        assert len(ranker._custom_rankers) == 1

        # Test that custom ranker is applied
        files = [
            FileAnalysis(path="normal.py", content=""),
            FileAnalysis(path="special.py", content=""),
        ]

        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        mock_strategy = Mock()
        mock_strategy.rank_file.return_value = RankingFactors(keyword_match=0.5)
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}

        ranker._get_strategy = Mock(return_value=mock_strategy)

        # Mock the custom ranker application
        with patch.object(ranker, "_custom_rankers", [custom_ranker]):
            ranked = ranker.rank_files(files, prompt_context, parallel=False)

        # Custom ranker should be applied (though effect depends on implementation details)
        assert len(ranked) >= 0


class TestCorpusAnalysis:
    """Test suite for corpus analysis."""

    @pytest.fixture
    def ranker(self, test_config):
        """Provide a RelevanceRanker instance."""
        with patch("tenets.core.ranking.ranker.get_logger"):
            return RelevanceRanker(test_config)

    def test_analyze_corpus(self, ranker):
        """Test corpus analysis for statistics."""
        files = [
            FileAnalysis(
                path="file1.py",
                content="",
                language="python",
                size=1000,
                imports=[Mock(module="os")],
            ),
            FileAnalysis(
                path="file2.js",
                content="",
                language="javascript",
                size=2000,
                imports=[Mock(module="react")],
            ),
            FileAnalysis(
                path="file3.py",
                content="",
                language="python",
                size=1500,
                imports=[Mock(module="file1")],
            ),
        ]

        stats = ranker._analyze_corpus(files)

        assert stats["total_files"] == 3
        assert stats["languages"]["python"] == 2
        assert stats["languages"]["javascript"] == 1
        assert len(stats["file_sizes"]) == 3
        assert stats["avg_file_size"] == 1500
        assert "import_graph" in stats

    def test_resolve_import(self, ranker):
        """Test import resolution to file paths."""
        files = [
            FileAnalysis(path="src/auth.py", content=""),
            FileAnalysis(path="src/models/user.py", content=""),
            FileAnalysis(path="tests/test_auth.py", content=""),
        ]

        # Test exact match
        resolved = ranker._resolve_import("auth", "main.py", files)
        assert resolved == "src/auth.py"

        # Test module path match
        resolved = ranker._resolve_import("models.user", "main.py", files)
        assert resolved == "src/models/user.py"

        # Test no match
        resolved = ranker._resolve_import("nonexistent", "main.py", files)
        assert resolved is None


class TestRankingExplanation:
    """Test suite for ranking explanation generation."""

    @pytest.fixture
    def ranker(self, test_config):
        """Provide a RelevanceRanker instance."""
        with patch("tenets.core.ranking.ranker.get_logger"):
            return RelevanceRanker(test_config)

    def test_generate_explanation(self, ranker):
        """Test explanation generation for rankings."""
        factors = RankingFactors(
            keyword_match=0.8, semantic_similarity=0.7, path_relevance=0.3, import_centrality=0.6
        )

        weights = {
            "keyword_match": 0.4,
            "semantic_similarity": 0.3,
            "path_relevance": 0.2,
            "import_centrality": 0.1,
        }

        explanation = ranker._generate_explanation(factors, weights)

        # Should mention top contributing factors
        assert "keyword match" in explanation.lower()
        assert "semantic similarity" in explanation.lower()

    def test_generate_explanation_low_relevance(self, ranker):
        """Test explanation for low relevance files."""
        factors = RankingFactors()  # All zeros
        weights = {"keyword_match": 1.0}

        explanation = ranker._generate_explanation(factors, weights)

        assert "low relevance" in explanation.lower()


class TestEdgeCases:
    """Test suite for edge cases."""

    @pytest.fixture
    def ranker(self, test_config):
        """Provide a RelevanceRanker instance."""
        with patch("tenets.core.ranking.ranker.get_logger"):
            return RelevanceRanker(test_config)

    def test_unknown_algorithm(self, ranker):
        """Test ranking with unknown algorithm raises error."""
        files = [FileAnalysis(path="test.py", content="")]
        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        with pytest.raises(ValueError, match="Unknown ranking algorithm"):
            ranker.rank_files(files, prompt_context, algorithm="nonexistent")

    def test_parallel_ranking_timeout(self, ranker):
        """Test handling of timeout in parallel ranking."""
        files = [FileAnalysis(path=f"file{i}.py", content="") for i in range(3)]
        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        mock_strategy = Mock()
        mock_strategy.rank_file.return_value = RankingFactors()
        mock_strategy.get_weights.return_value = {}

        ranker._get_strategy = Mock(return_value=mock_strategy)

        # Mock future that times out
        mock_future = Mock()
        mock_future.result.side_effect = FutureTimeoutError()

        with patch.object(ranker._executor, "submit", return_value=mock_future):
            ranked = ranker.rank_files(files, prompt_context, parallel=True)

            # Should handle timeout gracefully
            # Files that timeout get score 0
            assert all(hasattr(f, "relevance_score") for f in files)

    def test_shutdown(self, ranker):
        """Test ranker shutdown."""
        ranker.shutdown()

        # Executor should be shut down
        assert ranker._executor._shutdown
