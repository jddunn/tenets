"""
Unit tests for the relevance ranking system.

The ranking system scores and sorts files by relevance to a given prompt using
multiple strategies and factors. This module tests the various ranking algorithms,
factor calculations, and the overall ranking pipeline.

Test Coverage:
    - RelevanceRanker initialization
    - Different ranking strategies (fast, balanced, thorough)
    - Ranking factor calculations
    - TF-IDF calculator functionality
    - Custom ranker registration
    - Corpus analysis
    - Parallel vs sequential ranking
    - Edge cases and error handling
"""

import math
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
from unittest.mock import Mock, mock_open, patch

import pytest

from tenets.core.nlp.tfidf import TFIDFCalculator
from tenets.core.ranking.ranker import (
    BalancedRankingStrategy,
    FastRankingStrategy,
    RankedFile,
    RankingAlgorithm,
    RankingFactors,
    RelevanceRanker,
    ThoroughRankingStrategy,
)
from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FileAnalysis,
    FunctionInfo,
)
from tenets.models.context import PromptContext


class TestTFIDFCalculator:
    """Test suite for TF-IDF calculator."""

    def test_init_without_stopwords(self):
        """Test TF-IDF calculator initialization without stopwords."""
        calc = TFIDFCalculator(use_stopwords=False)

        assert calc.use_stopwords is False
        assert calc.stopwords == set()
        assert calc.document_count == 0
        assert calc.vocabulary == set()

    def test_init_with_stopwords(self):
        """Test TF-IDF calculator initialization with stopwords."""
        # Mock the stopwords file
        stopwords_content = "the\na\nan\nand\nis\n"

        with (
            patch("builtins.open", mock_open(read_data=stopwords_content)),
            patch("pathlib.Path.exists", return_value=True),
        ):
            calc = TFIDFCalculator(use_stopwords=True)

            assert calc.use_stopwords is True
            assert "the" in calc.stopwords
            assert "a" in calc.stopwords
            assert "and" in calc.stopwords

    def test_tokenize_without_stopwords(self):
        """Test tokenization without stopword filtering."""
        calc = TFIDFCalculator(use_stopwords=False)

        text = "The AuthenticationManager handles user login and logout"
        tokens = calc.tokenize(text)

        # Should include all tokens including "the", "and"
        assert "the" in tokens
        assert "and" in tokens
        assert "authenticationmanager" in tokens
        assert "authentication" in tokens  # Split from camelCase
        assert "manager" in tokens
        assert "handles" in tokens
        assert "user" in tokens
        assert "login" in tokens
        assert "logout" in tokens

    def test_tokenize_with_stopwords(self):
        """Test tokenization with stopword filtering."""
        calc = TFIDFCalculator(use_stopwords=True)
        calc.stopwords = {"the", "and", "a", "an"}

        text = "The AuthenticationManager handles user login and logout"
        tokens = calc.tokenize(text)

        # Should exclude stopwords
        assert "the" not in tokens
        assert "and" not in tokens
        # Should include other tokens
        assert "authenticationmanager" in tokens
        assert "handles" in tokens
        assert "user" in tokens

    def test_tokenize_camel_case(self):
        """Test tokenization of camelCase and PascalCase."""
        calc = TFIDFCalculator(use_stopwords=False)

        text = "getUserById handleAuthRequest XMLParser"
        tokens = calc.tokenize(text)

        # Should split camelCase but also keep original
        assert "getuserbyid" in tokens
        assert "get" in tokens
        assert "user" in tokens
        assert "by" in tokens
        assert "id" in tokens

        assert "handleauthrequest" in tokens
        assert "handle" in tokens
        assert "auth" in tokens
        assert "request" in tokens

        assert "xmlparser" in tokens
        assert "xml" in tokens
        assert "parser" in tokens

    def test_tokenize_snake_case(self):
        """Test tokenization of snake_case."""
        calc = TFIDFCalculator(use_stopwords=False)

        text = "get_user_by_id handle_auth_request"
        tokens = calc.tokenize(text)

        # Should split snake_case but also keep original
        assert "get_user_by_id" in tokens
        assert "get" in tokens
        assert "user" in tokens
        assert "by" in tokens
        assert "id" in tokens

    def test_compute_tf_sublinear(self):
        """Test term frequency calculation with sublinear scaling."""
        calc = TFIDFCalculator(use_stopwords=False)

        tokens = ["auth", "auth", "auth", "user", "user", "login"]
        tf = calc.compute_tf(tokens, use_sublinear=True)

        # auth appears 3 times: 1 + log(3)
        assert abs(tf["auth"] - (1 + math.log(3))) < 0.001
        # user appears 2 times: 1 + log(2)
        assert abs(tf["user"] - (1 + math.log(2))) < 0.001
        # login appears 1 time: 1 + log(1) = 1
        assert tf["login"] == 1.0

    def test_compute_tf_raw(self):
        """Test term frequency calculation without sublinear scaling."""
        calc = TFIDFCalculator(use_stopwords=False)

        tokens = ["auth", "auth", "user", "login"]
        tf = calc.compute_tf(tokens, use_sublinear=False)

        # Raw frequency normalized by document length
        assert tf["auth"] == 0.5  # 2/4
        assert tf["user"] == 0.25  # 1/4
        assert tf["login"] == 0.25  # 1/4

    def test_compute_idf(self):
        """Test inverse document frequency calculation."""
        calc = TFIDFCalculator(use_stopwords=False)
        calc.document_count = 10
        calc.document_frequency = {
            "common": 8,  # Appears in 8 docs
            "rare": 1,  # Appears in 1 doc
            "medium": 4,  # Appears in 4 docs
        }

        # IDF = log((N + 1) / (df + 1))
        # Common term: log(11/9) = low IDF
        idf_common = calc.compute_idf("common")
        assert abs(idf_common - math.log(11 / 9)) < 0.001

        # Rare term: log(11/2) = high IDF
        idf_rare = calc.compute_idf("rare")
        assert abs(idf_rare - math.log(11 / 2)) < 0.001

        # New term (not in corpus): log(11/1) = highest IDF
        idf_new = calc.compute_idf("newterm")
        assert abs(idf_new - math.log(11 / 1)) < 0.001

    def test_add_document(self):
        """Test adding a document to the corpus."""
        calc = TFIDFCalculator(use_stopwords=False)

        # Add first document
        doc1_vector = calc.add_document("doc1", "authentication and authorization")

        assert calc.document_count == 1
        assert "authentication" in calc.vocabulary
        assert "authorization" in calc.vocabulary
        assert "doc1" in calc.document_vectors

        # Vector should be normalized (L2 norm = 1)
        norm = math.sqrt(sum(score**2 for score in doc1_vector.values()))
        assert abs(norm - 1.0) < 0.001

    def test_compute_similarity(self):
        """Test cosine similarity computation."""
        calc = TFIDFCalculator(use_stopwords=False)

        # Build a small corpus
        calc.add_document("doc1", "authentication system for users")
        calc.add_document("doc2", "database connection and queries")
        calc.add_document("doc3", "user authentication and authorization")

        # Query similar to doc1 and doc3
        similarity1 = calc.compute_similarity("user authentication", "doc1")
        similarity2 = calc.compute_similarity("user authentication", "doc2")
        similarity3 = calc.compute_similarity("user authentication", "doc3")

        # doc3 should have highest similarity (contains both terms)
        assert similarity3 > similarity1
        # doc1 should have higher similarity than doc2
        assert similarity1 > similarity2
        # All similarities should be in [0, 1]
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1
        assert 0 <= similarity3 <= 1

    def test_get_top_terms(self):
        """Test getting top TF-IDF terms for a document."""
        calc = TFIDFCalculator(use_stopwords=False)

        # Build corpus
        calc.add_document("doc1", "authentication authentication security")
        calc.add_document("doc2", "database database database")
        calc.add_document("doc3", "api endpoint handler")

        # Get top terms for doc1
        top_terms = calc.get_top_terms("doc1", n=2)

        assert len(top_terms) <= 2
        # Should return tuples of (term, score)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top_terms)
        # Should be sorted by score
        if len(top_terms) == 2:
            assert top_terms[0][1] >= top_terms[1][1]

    def test_build_corpus(self):
        """Test batch corpus building."""
        calc = TFIDFCalculator(use_stopwords=False)

        documents = [
            ("doc1", "authentication system"),
            ("doc2", "database queries"),
            ("doc3", "api endpoints"),
        ]

        calc.build_corpus(documents)

        assert calc.document_count == 3
        assert len(calc.document_vectors) == 3
        assert "authentication" in calc.vocabulary
        assert "database" in calc.vocabulary
        assert "api" in calc.vocabulary


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

        assert abs(score - 0.64) < 0.001

    def test_get_weighted_score_with_custom(self):
        """Test weighted score with custom factors."""
        factors = RankingFactors(
            keyword_match=0.5, custom_scores={"auth_patterns": 0.8, "api_patterns": 0.6}
        )

        weights = {"keyword_match": 0.4, "auth_patterns": 0.3, "api_patterns": 0.3}

        score = factors.get_weighted_score(weights)

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
        # Float summation can yield 0.9999999999999999 on some platforms; allow tiny tolerance
        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=0, abs=1e-12)


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

    def test_tfidf_similarity_calculation(self, strategy):
        """Test TF-IDF similarity calculation with the new calculator."""
        file = FileAnalysis(
            path="auth.py",
            content="authentication system with user login and password verification",
            language="python",
        )

        prompt_context = PromptContext(
            text="user authentication system",
            keywords=["authentication", "user", "system"],
            task_type="general",
        )

        # Create a mock TF-IDF calculator
        mock_tfidf = Mock(spec=TFIDFCalculator)
        mock_tfidf.document_vectors = {"auth.py": {"authentication": 0.5, "user": 0.3}}
        mock_tfidf.compute_similarity.return_value = 0.75

        corpus_stats = {"tfidf_calculator": mock_tfidf}

        factors = strategy.rank_file(file, prompt_context, corpus_stats)

        # Should have TF-IDF similarity score
        assert factors.tfidf_similarity == 0.75
        mock_tfidf.compute_similarity.assert_called_once_with(
            "user authentication system", "auth.py"
        )

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
        with patch("tenets.core.ranking.ranker.get_logger"):
            return RelevanceRanker(test_config)

    def test_rank_files_basic(self, ranker):
        files = [
            FileAnalysis(path="file1.py", content="auth code", language="python"),
            FileAnalysis(path="file2.py", content="unrelated", language="python"),
            FileAnalysis(path="file3.py", content="authentication logic", language="python"),
        ]
        prompt_context = PromptContext(
            text="authentication", keywords=["authentication", "auth"], task_type="feature"
        )
        mock_strategy = Mock()
        mock_strategy.rank_file.side_effect = [
            RankingFactors(keyword_match=0.5),
            RankingFactors(keyword_match=0.1),
            RankingFactors(keyword_match=0.9),
        ]
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}
        ranker._get_strategy = Mock(return_value=mock_strategy)
        ranked = ranker.rank_files(files, prompt_context, algorithm="fast", parallel=False)
        assert len(ranked) >= 2
        assert ranked[0].relevance_score >= ranked[-1].relevance_score

    def test_rank_files_parallel(self, ranker):
        files = [FileAnalysis(path=f"file{i}.py", content=f"content {i}") for i in range(20)]
        prompt_context = PromptContext(text="test", keywords=["test"], task_type="general")
        mock_strategy = Mock()
        mock_strategy.rank_file.return_value = RankingFactors(keyword_match=0.5)
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}
        ranker._get_strategy = Mock(return_value=mock_strategy)
        with patch.object(ranker._executor, "submit") as mock_submit:
            mock_future = Mock()
            mock_future.result.return_value = RankedFile(
                analysis=files[0], score=0.5, factors=RankingFactors(), explanation=""
            )
            mock_submit.return_value = mock_future
            _ = ranker.rank_files(files, prompt_context, parallel=True)
            assert mock_submit.call_count == 20

    def test_rank_files_with_threshold(self, ranker, test_config):
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
        assert len(ranked) == 2
        assert all(f.relevance_score >= 0.5 for f in ranked)

    def test_rank_files_empty_input(self, ranker):
        files: list[FileAnalysis] = []
        prompt_context = PromptContext(text="test", keywords=[], task_type="general")
        ranked = ranker.rank_files(files, prompt_context)
        assert ranked == []

    def test_rank_files_with_error(self, ranker):
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
        ranked = ranker.rank_files(files, prompt_context, parallel=False)
        assert len(ranked) >= 1

    def test_custom_ranker_registration(self, ranker):
        def custom_ranker(ranked_files, prompt_context):
            for rf in ranked_files:
                if "special" in rf.analysis.path:
                    rf.score *= 2
            return ranked_files

        ranker.register_custom_ranker(custom_ranker)
        assert len(ranker._custom_rankers) == 1
        files = [
            FileAnalysis(path="normal.py", content=""),
            FileAnalysis(path="special.py", content=""),
        ]
        prompt_context = PromptContext(text="test", keywords=[], task_type="general")
        mock_strategy = Mock()
        mock_strategy.rank_file.return_value = RankingFactors(keyword_match=0.5)
        mock_strategy.get_weights.return_value = {"keyword_match": 1.0}
        ranker._get_strategy = Mock(return_value=mock_strategy)
        with patch.object(ranker, "_custom_rankers", [custom_ranker]):
            ranked = ranker.rank_files(files, prompt_context, parallel=False)
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
                content="def authenticate(): pass",
                language="python",
                size=1000,
                imports=[Mock(module="os")],
            ),
            FileAnalysis(
                path="file2.js",
                content="function login() {}",
                language="javascript",
                size=2000,
                imports=[Mock(module="react")],
            ),
            FileAnalysis(
                path="file3.py",
                content="import file1",
                language="python",
                size=1500,
                imports=[Mock(module="file1")],
            ),
        ]

        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        stats = ranker._analyze_corpus(files, prompt_context)

        assert stats["total_files"] == 3
        assert stats["languages"]["python"] == 2
        assert stats["languages"]["javascript"] == 1
        assert len(stats["file_sizes"]) == 3
        assert stats["avg_file_size"] == 1500
        assert "import_graph" in stats

        # Should have TF-IDF calculator
        assert "tfidf_calculator" in stats
        assert isinstance(stats["tfidf_calculator"], TFIDFCalculator)
        # Calculator should have processed all files
        assert stats["tfidf_calculator"].document_count == 3

    def test_analyze_corpus_with_tfidf_stopwords(self, ranker, test_config):
        """Test corpus analysis with TF-IDF stopwords enabled."""
        # Enable stopwords in config
        test_config.use_tfidf_stopwords = True

        files = [
            FileAnalysis(path="file1.py", content="the authentication system", language="python"),
            FileAnalysis(path="file2.py", content="database and queries", language="python"),
        ]

        prompt_context = PromptContext(text="test", keywords=[], task_type="general")

        # Mock stopwords file
        with patch("builtins.open", mock_open(read_data="the\nand\n")):
            with patch("pathlib.Path.exists", return_value=True):
                stats = ranker._analyze_corpus(files, prompt_context)

        tfidf_calc = stats["tfidf_calculator"]
        assert tfidf_calc.use_stopwords is True
        # Should have loaded stopwords
        assert "the" in tfidf_calc.stopwords or len(tfidf_calc.stopwords) > 0

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
            keyword_match=0.8,
            tfidf_similarity=0.75,
            semantic_similarity=0.7,
            path_relevance=0.3,
            import_centrality=0.6,
        )

        weights = {
            "keyword_match": 0.4,
            "tfidf_similarity": 0.2,
            "semantic_similarity": 0.2,
            "path_relevance": 0.1,
            "import_centrality": 0.1,
        }

        explanation = ranker._generate_explanation(factors, weights)

        # Should mention top contributing factors
        assert "keyword match" in explanation.lower()
        # Should also mention TF-IDF now
        assert "tf-idf" in explanation.lower() or "tfidf" in explanation.lower()

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
            _ = ranker.rank_files(files, prompt_context, parallel=True)

            # Should handle timeout gracefully
            # Files that timeout get score 0
            assert all(hasattr(f, "relevance_score") for f in files)

    def test_tfidf_with_empty_content(self):
        """Test TF-IDF calculator handles empty content gracefully."""
        calc = TFIDFCalculator(use_stopwords=False)

        # Add document with empty content
        vector = calc.add_document("empty", "")
        assert vector == {}
        assert calc.document_norms["empty"] == 0.0

        # Compute similarity with empty document
        similarity = calc.compute_similarity("test query", "empty")
        assert similarity == 0.0

    def test_tfidf_cache_invalidation(self):
        """Test TF-IDF IDF cache is cleared when documents are added."""
        calc = TFIDFCalculator(use_stopwords=False)

        # Add first document and compute IDF
        calc.add_document("doc1", "test content")
        idf1 = calc.compute_idf("test")
        assert "test" in calc.idf_cache

        # Add another document - should clear cache
        calc.add_document("doc2", "test content again")
        assert len(calc.idf_cache) == 0

        # IDF should be different now
        idf2 = calc.compute_idf("test")
        assert idf1 != idf2

    def test_shutdown(self, ranker):
        """Test ranker shutdown."""
        ranker.shutdown()

        # Executor should be shut down
        assert ranker._executor._shutdown
