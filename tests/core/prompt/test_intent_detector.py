"""Unit tests for intent detection system.

Tests pattern-based and ML-enhanced intent detection with
comprehensive coverage of all intent types and confidence scoring.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tenets.core.prompt.intent_detector import (
    HybridIntentDetector,
    Intent,
    PatternBasedDetector,
    SemanticIntentDetector,
)


class TestIntent:
    """Test Intent dataclass."""

    def test_intent_creation(self):
        """Test creating an intent."""
        intent = Intent(
            type="implement",
            confidence=0.85,
            evidence=["add", "create", "new feature"],
            keywords=["authentication", "oauth2"],
            metadata={"score": 3.5},
            source="pattern",
        )

        assert intent.type == "implement"
        assert intent.confidence == 0.85
        assert intent.evidence == ["add", "create", "new feature"]
        assert intent.keywords == ["authentication", "oauth2"]
        assert intent.metadata["score"] == 3.5
        assert intent.source == "pattern"

    def test_intent_to_dict(self):
        """Test converting intent to dictionary."""
        intent = Intent(
            type="debug",
            confidence=0.9,
            evidence=["fix", "bug"],
            keywords=["error", "crash"],
            metadata={"priority": "high"},
            source="ml",
        )

        result = intent.to_dict()

        assert result["type"] == "debug"
        assert result["confidence"] == 0.9
        assert result["evidence"] == ["fix", "bug"]
        assert result["keywords"] == ["error", "crash"]
        assert result["metadata"]["priority"] == "high"
        assert result["source"] == "ml"


class TestPatternBasedDetector:
    """Test pattern-based intent detection."""

    @pytest.fixture
    def detector(self):
        """Create pattern detector instance."""
        return PatternBasedDetector()

    def test_load_default_patterns(self, detector):
        """Test loading default intent patterns."""
        expected_intents = [
            "implement",
            "debug",
            "understand",
            "refactor",
            "test",
            "document",
            "review",
            "optimize",
            "integrate",
            "migrate",
            "configure",
            "analyze",
        ]

        for intent in expected_intents:
            assert intent in detector.patterns
            assert "patterns" in detector.patterns[intent]
            assert "keywords" in detector.patterns[intent]
            assert "examples" in detector.patterns[intent]
            assert "weight" in detector.patterns[intent]

    def test_load_patterns_from_file(self, tmp_path):
        """Test loading patterns from JSON file."""
        patterns_file = tmp_path / "patterns.json"
        patterns = {
            "custom_intent": {
                "patterns": [r"\bcustom\b"],
                "keywords": ["custom"],
                "examples": ["custom task"],
                "weight": 1.5,
            }
        }
        patterns_file.write_text(json.dumps(patterns))

        detector = PatternBasedDetector(patterns_file)

        assert "custom_intent" in detector.patterns
        assert detector.patterns["custom_intent"]["weight"] == 1.5

    def test_compile_patterns(self, detector):
        """Test pattern compilation."""
        assert "implement" in detector.compiled_patterns
        assert "debug" in detector.compiled_patterns

        # Check patterns are compiled with weights
        for intent_type, patterns in detector.compiled_patterns.items():
            for pattern, weight in patterns:
                assert hasattr(pattern, "match")  # Is a compiled regex
                assert isinstance(weight, float)

    def test_detect_implement_intent(self, detector):
        """Test detecting implementation intent."""
        texts = [
            "implement user authentication",
            "add new feature for data export",
            "create REST API endpoints",
            "build notification system",
            "develop payment integration",
            "make a new dashboard",
            "write code for user management",
        ]

        for text in texts:
            intents = detector.detect(text)

            # Should detect implement intent
            implement_intents = [i for i in intents if i.type == "implement"]
            assert len(implement_intents) > 0

            intent = implement_intents[0]
            assert intent.confidence > 0
            assert intent.source == "pattern"
            assert len(intent.evidence) > 0

    def test_detect_debug_intent(self, detector):
        """Test detecting debug intent."""
        texts = [
            "fix authentication bug",
            "debug memory leak issue",
            "resolve database connection error",
            "troubleshoot API timeout",
            "investigate application crash",
            "the login is not working",
            "broken payment processing",
            "solve the problem with uploads",
            "application fails to start",
        ]

        for text in texts:
            intents = detector.detect(text)

            debug_intents = [i for i in intents if i.type == "debug"]
            assert len(debug_intents) > 0

            intent = debug_intents[0]
            # Debug has higher weight (1.2)
            assert intent.metadata["score"] > 0

    def test_detect_understand_intent(self, detector):
        """Test detecting understand intent."""
        texts = [
            "explain how the caching works",
            "understand authentication flow",
            "show me the data pipeline",
            "how does the algorithm work",
            "what is the architecture",
            "describe the system design",
            "where is the configuration stored",
        ]

        for text in texts:
            intents = detector.detect(text)

            understand_intents = [i for i in intents if i.type == "understand"]
            assert len(understand_intents) > 0

    def test_detect_refactor_intent(self, detector):
        """Test detecting refactor intent."""
        texts = [
            "refactor authentication module",
            "clean up database queries",
            "improve code organization",
            "modernize legacy components",
            "restructure project layout",
            "optimize and simplify the code",
            "reorganize the file structure",
            "update old implementations",
        ]

        for text in texts:
            intents = detector.detect(text)

            refactor_intents = [i for i in intents if i.type == "refactor"]
            assert len(refactor_intents) > 0

    def test_detect_test_intent(self, detector):
        """Test detecting test intent."""
        texts = [
            "write unit tests for auth",
            "add integration test coverage",
            "create end-to-end test suite",
            "improve test coverage",
            "setup testing with pytest",
            "add specs for the new feature",
            "test the API endpoints",
        ]

        for text in texts:
            intents = detector.detect(text)

            test_intents = [i for i in intents if i.type == "test"]
            assert len(test_intents) > 0

    def test_detect_document_intent(self, detector):
        """Test detecting document intent."""
        texts = [
            "document the API endpoints",
            "add documentation for the module",
            "write README file",
            "create user guide",
            "add code comments",
            "describe the functions",
        ]

        for text in texts:
            intents = detector.detect(text)

            doc_intents = [i for i in intents if i.type == "document"]
            assert len(doc_intents) > 0

    def test_detect_review_intent(self, detector):
        """Test detecting review intent."""
        texts = [
            "review the pull request",
            "code review for auth module",
            "check the implementation",
            "audit the security",
            "inspect the changes",
            "analyze the code quality",
        ]

        for text in texts:
            intents = detector.detect(text)

            review_intents = [i for i in intents if i.type == "review"]
            assert len(review_intents) > 0

    def test_detect_optimize_intent(self, detector):
        """Test detecting optimize intent."""
        texts = [
            "optimize database performance",
            "improve application speed",
            "reduce memory usage",
            "fix performance bottleneck",
            "make the queries faster",
            "profile the application",
            "enhance efficiency",
            "reduce latency",
        ]

        for text in texts:
            intents = detector.detect(text)

            optimize_intents = [i for i in intents if i.type == "optimize"]
            assert len(optimize_intents) > 0

            # Optimize has higher weight (1.1)
            intent = optimize_intents[0]
            assert intent.metadata.get("weight") == detector.patterns["optimize"]["weight"]

    def test_detect_integrate_intent(self, detector):
        """Test detecting integrate intent."""
        texts = [
            "integrate with Stripe API",
            "connect to external service",
            "add OAuth provider",
            "implement webhook handler",
            "interface with third-party library",
        ]

        for text in texts:
            intents = detector.detect(text)

            integrate_intents = [i for i in intents if i.type == "integrate"]
            assert len(integrate_intents) > 0

    def test_detect_migrate_intent(self, detector):
        """Test detecting migrate intent."""
        texts = [
            "migrate to new framework",
            "upgrade database schema",
            "port to Python 3",
            "transfer data to new system",
            "move to version 2.0",
        ]

        for text in texts:
            intents = detector.detect(text)

            migrate_intents = [i for i in intents if i.type == "migrate"]
            assert len(migrate_intents) > 0

    def test_detect_configure_intent(self, detector):
        """Test detecting configure intent."""
        texts = [
            "configure CI/CD pipeline",
            "setup development environment",
            "update config files",
            "deploy to production",
            "change settings",
        ]

        for text in texts:
            intents = detector.detect(text)

            config_intents = [i for i in intents if i.type == "configure"]
            assert len(config_intents) > 0

    def test_detect_analyze_intent(self, detector):
        """Test detecting analyze intent."""
        texts = [
            "analyze performance metrics",
            "examine error logs",
            "investigate data patterns",
            "study the usage statistics",
            "explore the dataset",
        ]

        for text in texts:
            intents = detector.detect(text)

            analyze_intents = [i for i in intents if i.type == "analyze"]
            assert len(analyze_intents) > 0

    def test_detect_multiple_intents(self, detector):
        """Test detecting multiple intents in one text."""
        text = "refactor the authentication module and add unit tests"

        intents = detector.detect(text)

        intent_types = [i.type for i in intents]
        assert "refactor" in intent_types
        assert "test" in intent_types

        # Both should have reasonable confidence
        for intent in intents:
            assert intent.confidence > 0

    def test_detect_with_keywords(self, detector):
        """Test that keywords boost confidence."""
        text = "implement create build new feature functionality capability"

        intents = detector.detect(text)

        implement_intents = [i for i in intents if i.type == "implement"]
        assert len(implement_intents) > 0

        intent = implement_intents[0]
        assert len(intent.keywords) > 0
        assert intent.metadata["keyword_matches"] > 0
        assert intent.metadata["pattern_matches"] > 0

    def test_confidence_normalization(self, detector):
        """Test that confidence is normalized to 0-1 range."""
        text = "implement add create build develop make write code new feature functionality capability"

        intents = detector.detect(text)

        for intent in intents:
            assert 0 <= intent.confidence <= 1

    def test_empty_text_detection(self, detector):
        """Test detecting intent from empty text."""
        intents = detector.detect("")

        assert len(intents) == 0

    def test_no_match_detection(self, detector):
        """Test text with no matching patterns."""
        intents = detector.detect("the cat sat on the mat")

        # Might match some general words but with low confidence
        if intents:
            for intent in intents:
                assert intent.confidence < 0.5


class TestSemanticIntentDetector:
    """Test ML-based semantic intent detection."""

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", False)
    def test_no_ml_available(self):
        """Test when ML features are not available."""
        detector = SemanticIntentDetector()

        assert detector.model is None
        assert detector.similarity_calculator is None

        intents = detector.detect("implement authentication")
        assert intents == []

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    @patch("tenets.core.prompt.intent_detector.SemanticSimilarity")
    def test_with_ml_available(self, mock_similarity_class, mock_create_model):
        """Test when ML features are available."""
        # Mock model and similarity calculator
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        mock_similarity = MagicMock()
        mock_similarity_class.return_value = mock_similarity

        detector = SemanticIntentDetector()

        assert detector.model is not None
        assert detector.similarity_calculator is not None
        mock_create_model.assert_called_once_with(model_name="all-MiniLM-L6-v2")

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    @patch("tenets.core.prompt.intent_detector.SemanticSimilarity")
    def test_detect_with_ml(self, mock_similarity_class, mock_create_model):
        """Test semantic intent detection."""
        # Setup mocks
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        mock_similarity = MagicMock()
        mock_similarity_class.return_value = mock_similarity

        # Mock similarity scores for each example
        mock_similarity.compute.side_effect = [
            0.85,  # High similarity with first implement example
            0.6,
            0.5,
            0.4,
            0.3,  # Lower similarities with other implement examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with debug examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with understand examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with refactor examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with test examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with document examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with optimize examples
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,  # Low similarities with integrate examples
        ]

        detector = SemanticIntentDetector()

        intents = detector.detect("build user authentication system", threshold=0.6)

        assert len(intents) >= 1
        intent = intents[0]
        assert intent.type == "implement"
        assert intent.confidence == 0.85
        assert intent.source == "ml"
        assert "best_match" in intent.metadata
        assert intent.metadata["max_similarity"] == 0.85

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    @patch("tenets.core.prompt.intent_detector.SemanticSimilarity")
    def test_detect_below_threshold(self, mock_similarity_class, mock_create_model):
        """Test detection with similarities below threshold."""
        # Setup mocks
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        mock_similarity = MagicMock()
        mock_similarity_class.return_value = mock_similarity

        # All similarities below threshold
        mock_similarity.compute.return_value = 0.3

        detector = SemanticIntentDetector()

        intents = detector.detect("random unrelated text", threshold=0.6)

        assert len(intents) == 0

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    @patch("tenets.core.prompt.intent_detector.SemanticSimilarity")
    def test_detect_multiple_above_threshold(self, mock_similarity_class, mock_create_model):
        """Test when multiple intents are above threshold."""
        # Setup mocks
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        mock_similarity = MagicMock()
        mock_similarity_class.return_value = mock_similarity

        # Different similarities for different intent types
        similarities = []
        # Implement examples - high similarity
        similarities.extend([0.75, 0.7, 0.65, 0.6, 0.6])
        # Debug examples - also high
        similarities.extend([0.8, 0.75, 0.7, 0.65, 0.6])
        # Others - low
        for _ in range(30):  # Remaining examples
            similarities.append(0.3)

        mock_similarity.compute.side_effect = similarities

        detector = SemanticIntentDetector()

        intents = detector.detect("fix and implement authentication", threshold=0.6)

        # Should detect both implement and debug
        assert len(intents) >= 2
        intent_types = [i.type for i in intents]
        assert "implement" in intent_types
        assert "debug" in intent_types

    def test_get_intent_examples(self):
        """Test that intent examples are properly defined."""
        detector = SemanticIntentDetector()

        expected_intents = [
            "implement",
            "debug",
            "understand",
            "refactor",
            "test",
            "document",
            "optimize",
            "integrate",
        ]

        for intent in expected_intents:
            assert intent in detector.intent_examples
            assert len(detector.intent_examples[intent]) >= 3

            # Check examples are strings
            for example in detector.intent_examples[intent]:
                assert isinstance(example, str)
                assert len(example) > 0

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    def test_model_initialization_failure(self, mock_create_model):
        """Test handling model initialization failure."""
        mock_create_model.side_effect = Exception("Model not found")

        detector = SemanticIntentDetector()

        assert detector.model is None
        assert detector.similarity_calculator is None

        # Should return empty list when detection is attempted
        intents = detector.detect("test text")
        assert intents == []


class TestHybridIntentDetector:
    """Test hybrid intent detection system."""

    @pytest.fixture
    def detector(self):
        """Create hybrid detector with ML disabled."""
        return HybridIntentDetector(use_ml=False)

    def test_initialization_no_ml(self, detector):
        """Test initialization without ML."""
        assert detector.pattern_detector is not None
        assert detector.semantic_detector is None
        assert detector.keyword_extractor is not None

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    @patch("tenets.core.prompt.intent_detector.SemanticSimilarity")
    def test_initialization_with_ml(self, mock_similarity_class, mock_create_model):
        """Test initialization with ML enabled."""
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        mock_similarity_class.return_value = MagicMock()

        detector = HybridIntentDetector(use_ml=True)

        assert detector.pattern_detector is not None
        assert detector.semantic_detector is not None

    def test_detect_single_intent(self, detector):
        """Test detecting single primary intent."""
        text = "implement OAuth2 authentication for the API"

        intent = detector.detect(text)

        assert intent is not None
        assert intent.type == "implement"
        assert intent.confidence > 0
        assert len(intent.keywords) > 0
        assert intent.source in ["pattern", "combined", "default"]

    def test_detect_debug_intent(self, detector):
        """Test detecting debug intent."""
        text = "fix the memory leak bug in the application"

        intent = detector.detect(text)

        assert intent.type == "debug"
        assert intent.confidence > 0
        assert "fix" in intent.evidence or "bug" in intent.evidence

    def test_detect_with_low_confidence(self, detector):
        """Test detection with low confidence text."""
        text = "maybe something about stuff"

        intent = detector.detect(text, min_confidence=0.1)

        assert intent is not None
        # Should default to understand if no clear intent
        if intent.metadata.get("default"):
            assert intent.type == "understand"

    def test_detect_default_intent(self, detector):
        """Test defaulting to understand intent."""
        text = "the cat sat on the mat"

        intent = detector.detect(text, min_confidence=0.8)

        assert intent is not None
        assert intent.type == "understand"
        assert intent.metadata.get("default") is True
        assert intent.source == "default"
        assert intent.confidence == 0.5  # Default confidence

    def test_detect_multiple_intents(self, detector):
        """Test detecting multiple intents."""
        text = "refactor the code, add tests, and fix the bug"

        intents = detector.detect_multiple(text, max_intents=3)

        assert len(intents) <= 3

        # Should find multiple intent types
        intent_types = [i.type for i in intents]
        assert len(set(intent_types)) >= 2

        # Should be sorted by confidence
        for i in range(len(intents) - 1):
            assert intents[i].confidence >= intents[i + 1].confidence

    def test_detect_multiple_with_limit(self, detector):
        """Test max_intents limit in detect_multiple."""
        text = "implement feature, debug issue, test code, document API, review PR"

        intents = detector.detect_multiple(text, max_intents=2)

        assert len(intents) <= 2

    def test_combine_intents_weighted(self, detector):
        """Test combining intents with weighted method."""
        intents = [
            Intent("implement", 0.8, ["add"], [], {}, "pattern"),
            Intent("implement", 0.9, ["create"], [], {}, "ml"),
            Intent("debug", 0.7, ["fix"], [], {}, "pattern"),
        ]

        combined = detector._combine_intents(intents, ["test", "keyword"], "weighted", 0.6, 0.4)

        assert len(combined) == 2  # implement and debug

        # Check implement intent (has two detections)
        impl = [i for i in combined if i.type == "implement"][0]
        assert impl.source == "combined"
        assert impl.metadata["num_detections"] == 2
        assert len(impl.keywords) > 0

        # Confidence should be weighted average
        expected_conf = (0.8 * 0.6 + 0.9 * 0.4) / (0.6 + 0.4)
        assert abs(impl.confidence - expected_conf) < 0.01

    def test_combine_intents_max(self, detector):
        """Test combining intents with max method."""
        intents = [
            Intent("implement", 0.7, ["add"], [], {}, "pattern"),
            Intent("implement", 0.9, ["create"], [], {}, "ml"),
            Intent("debug", 0.6, ["fix"], [], {}, "pattern"),
        ]

        combined = detector._combine_intents(intents, [], "max", 0.6, 0.4)

        assert len(combined) == 2  # implement and debug

        # Implement should have max confidence
        impl = [i for i in combined if i.type == "implement"][0]
        assert impl.confidence == 0.9
        assert impl.metadata["num_detections"] == 2

    def test_combine_intents_vote(self, detector):
        """Test combining intents with vote method."""
        intents = [
            Intent("debug", 0.7, ["fix"], [], {}, "pattern"),
            Intent("debug", 0.8, ["resolve"], [], {}, "ml"),
            Intent("debug", 0.6, ["debug"], [], {}, "pattern"),
        ]

        combined = detector._combine_intents(intents, [], "vote", 0.6, 0.4)

        assert len(combined) == 1
        assert combined[0].type == "debug"
        assert combined[0].metadata["votes"] == 3
        # Average confidence
        assert combined[0].confidence == pytest.approx(0.7, rel=0.1)

    @patch("tenets.core.prompt.intent_detector.ML_AVAILABLE", True)
    @patch("tenets.core.prompt.intent_detector.create_embedding_model")
    @patch("tenets.core.prompt.intent_detector.SemanticSimilarity")
    def test_detect_with_both_detectors(self, mock_similarity_class, mock_create_model):
        """Test detection using both pattern and ML detectors."""
        # Setup ML mocks
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        mock_similarity = MagicMock()
        mock_similarity_class.return_value = mock_similarity
        mock_similarity.compute.return_value = 0.75

        detector = HybridIntentDetector(use_ml=True)

        text = "implement new feature for user authentication"

        intent = detector.detect(text)

        assert intent is not None
        assert intent.type == "implement"
        # Should combine results from both
        if intent.source == "combined":
            assert "sources" in intent.metadata or "num_detections" in intent.metadata

    def test_get_intent_context(self, detector):
        """Test getting intent context."""
        intent = Intent(
            "implement",
            0.85,
            ["add", "create"],
            ["feature", "authentication"],
            {"score": 3},
            "pattern",
        )

        context = detector.get_intent_context(intent)

        assert context["type"] == "implement"
        assert context["confidence"] == 0.85
        assert context["is_high_confidence"] is True
        assert context["is_medium_confidence"] is False
        assert context["is_low_confidence"] is False
        assert context["keywords"] == ["feature", "authentication"]
        assert context["evidence"] == ["add", "create"]
        assert context["task_type"] == "feature"
        assert len(context["examples"]) > 0
        assert len(context["related_keywords"]) > 0

    def test_get_intent_context_confidence_levels(self, detector):
        """Test intent context confidence levels."""
        # High confidence (>= 0.7)
        intent_high = Intent("debug", 0.75, [], [], {}, "pattern")
        context_high = detector.get_intent_context(intent_high)
        assert context_high["is_high_confidence"] is True
        assert context_high["is_medium_confidence"] is False

        # Medium confidence (0.4 - 0.7)
        intent_med = Intent("debug", 0.5, [], [], {}, "pattern")
        context_med = detector.get_intent_context(intent_med)
        assert context_med["is_high_confidence"] is False
        assert context_med["is_medium_confidence"] is True

        # Low confidence (< 0.4)
        intent_low = Intent("debug", 0.3, [], [], {}, "pattern")
        context_low = detector.get_intent_context(intent_low)
        assert context_low["is_high_confidence"] is False
        assert context_low["is_low_confidence"] is True

    def test_task_type_mapping(self, detector):
        """Test intent to task type mapping."""
        mappings = [
            ("implement", "feature"),
            ("debug", "debug"),
            ("understand", "understand"),
            ("refactor", "refactor"),
            ("test", "test"),
            ("document", "document"),
            ("review", "review"),
            ("optimize", "optimize"),
            ("integrate", "feature"),
            ("migrate", "refactor"),
            ("configure", "configuration"),
            ("analyze", "analysis"),
        ]

        for intent_type, expected_task in mappings:
            intent = Intent(intent_type, 0.8, [], [], {}, "pattern")
            context = detector.get_intent_context(intent)
            assert context["task_type"] == expected_task

    def test_detect_with_custom_weights(self, detector):
        """Test detection with custom pattern/ML weights."""
        text = "implement authentication system"

        # High pattern weight
        intent = detector.detect(text, combine_method="weighted", pattern_weight=0.9, ml_weight=0.1)

        assert intent.type == "implement"

    def test_detect_empty_text(self, detector):
        """Test detecting from empty text."""
        intent = detector.detect("")

        assert intent is not None
        assert intent.type == "understand"  # Default
        assert intent.metadata.get("default") is True

    def test_detect_multiple_empty_text(self, detector):
        """Test detecting multiple intents from empty text."""
        intents = detector.detect_multiple("", max_intents=3)

        # Should return default understand intent
        assert len(intents) == 1
        assert intents[0].type == "understand"
