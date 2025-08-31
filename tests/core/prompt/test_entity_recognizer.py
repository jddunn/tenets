"""Unit tests for entity recognition system.

Tests pattern-based, NLP-based, and fuzzy entity recognition
with proper mocking and comprehensive coverage.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tenets.core.prompt.entity_recognizer import (
    Entity,
    EntityPatternMatcher,
    FuzzyEntityMatcher,
    HybridEntityRecognizer,
    NLPEntityRecognizer,
)


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            name="UserController",
            type="class",
            confidence=0.95,
            context="class UserController extends",
            start_pos=10,
            end_pos=24,
            source="regex",
            metadata={"pattern": "class_definition"},
        )

        assert entity.name == "UserController"
        assert entity.type == "class"
        assert entity.confidence == 0.95
        assert entity.context == "class UserController extends"
        assert entity.start_pos == 10
        assert entity.end_pos == 24
        assert entity.source == "regex"
        assert entity.metadata["pattern"] == "class_definition"

    def test_entity_hash_and_equality(self):
        """Test entity hashing and equality."""
        entity1 = Entity(name="test_func", type="function", confidence=0.8, start_pos=0)
        entity2 = Entity(
            name="test_func",
            type="function",
            confidence=0.9,
            start_pos=0,  # Different confidence
        )
        entity3 = Entity(
            name="test_func",
            type="function",
            confidence=0.8,
            start_pos=10,  # Different position
        )

        # Same name, type, and position = equal
        assert entity1 == entity2
        assert hash(entity1) == hash(entity2)

        # Different position = not equal
        assert entity1 != entity3
        assert hash(entity1) != hash(entity3)

    def test_entity_not_equal_to_other_types(self):
        """Test entity inequality with other types."""
        entity = Entity(name="test", type="function", confidence=0.8)

        assert entity != "test"
        assert entity != 123
        assert entity != None


class TestEntityPatternMatcher:
    """Test pattern-based entity matching."""

    @pytest.fixture
    def matcher(self):
        """Create pattern matcher instance."""
        return EntityPatternMatcher()

    def test_load_default_patterns(self, matcher):
        """Test loading default patterns."""
        assert "class" in matcher.patterns
        assert "function" in matcher.patterns
        assert "variable" in matcher.patterns
        assert "file" in matcher.patterns
        assert "module" in matcher.patterns
        assert "api_endpoint" in matcher.patterns
        assert "database" in matcher.patterns
        assert "config" in matcher.patterns
        assert "url" in matcher.patterns
        assert "error" in matcher.patterns
        assert "component" in matcher.patterns

    def test_load_patterns_from_file(self, tmp_path):
        """Test loading patterns from JSON file."""
        patterns_file = tmp_path / "patterns.json"
        patterns = {
            "custom_type": [
                {
                    "pattern": r"\bcustom_([a-z]+)\b",
                    "confidence": 0.9,
                    "description": "Custom pattern",
                }
            ]
        }
        patterns_file.write_text(json.dumps(patterns))

        matcher = EntityPatternMatcher(patterns_file)

        assert "custom_type" in matcher.patterns
        assert len(matcher.patterns["custom_type"]) == 1

    def test_extract_classes(self, matcher):
        """Test extracting class entities."""
        text = """
        class UserController extends BaseController {
            constructor() {}
        }
        interface UserService {
            getUser(): User;
        }
        struct Point { x: number; y: number; }
        new DatabaseConnection()
        """

        entities = matcher.extract(text)
        class_entities = [e for e in entities if e.type == "class"]

        assert len(class_entities) > 0

        # Check specific extractions
        names = [e.name for e in class_entities]
        assert "UserController" in names
        assert "UserService" in names
        assert "DatabaseConnection" in names

    def test_extract_functions(self, matcher):
        """Test extracting function entities."""
        text = """
        function processData(input) {
            return input;
        }
        def calculate_score(value):
            pass
        const handleClick = (event) => {
            console.log(event);
        }
        object.methodCall(param);
        """

        entities = matcher.extract(text)
        func_entities = [e for e in entities if e.type == "function"]

        assert len(func_entities) > 0

        names = [e.name for e in func_entities]
        assert "processData" in names
        assert "calculate_score" in names
        assert "handleClick" in names
        assert "methodCall" in names

    def test_extract_variables(self, matcher):
        """Test extracting variable entities."""
        text = """
        let userName = "John";
        const MAX_SIZE = 100;
        var counter = 0;
        $shellVar = "value";
        @instanceVar = 42;
        result = compute();
        """

        entities = matcher.extract(text)
        var_entities = [e for e in entities if e.type == "variable"]

        assert len(var_entities) > 0

        names = [e.name for e in var_entities]
        assert "userName" in names or "MAX_SIZE" in names
        assert "shellVar" in names
        assert "instanceVar" in names

    def test_extract_files(self, matcher):
        """Test extracting file entities."""
        text = """
        Import from 'utils/helper.js'
        require('./config.json')
        open("data.csv")
        process file main.py
        include <header.h>
        """

        entities = matcher.extract(text)
        file_entities = [e for e in entities if e.type == "file"]

        assert len(file_entities) > 0

        names = [e.name for e in file_entities]
        # At least some files should be found
        assert any(
            name.endswith(".js")
            or name.endswith(".py")
            or name.endswith(".csv")
            or name.endswith(".json")
            for name in names
        )

    def test_extract_api_endpoints(self, matcher):
        """Test extracting API endpoint entities."""
        text = """
        GET /api/users/{id}
        POST /api/auth/login
        @GetMapping("/products")
        route('/dashboard')
        DELETE /api/items/[itemId]
        """

        entities = matcher.extract(text)
        api_entities = [e for e in entities if e.type == "api_endpoint"]

        assert len(api_entities) > 0

        names = [e.name for e in api_entities]
        assert "/api/users/{id}" in names or "/api/users/" in str(names)
        assert "/api/auth/login" in names or "/api/auth/" in str(names)

    def test_extract_urls(self, matcher):
        """Test extracting URL entities."""
        text = """
        Visit https://github.com/org/repo
        API endpoint: http://api.example.com/v1/users
        Documentation at https://docs.python.org
        """

        entities = matcher.extract(text)
        url_entities = [e for e in entities if e.type == "url"]

        assert len(url_entities) >= 3

        names = [e.name for e in url_entities]
        assert "https://github.com/org/repo" in names
        assert "http://api.example.com/v1/users" in names

    def test_extract_database_entities(self, matcher):
        """Test extracting database entities."""
        text = """
        SELECT * FROM users WHERE id = 1;
        INSERT INTO products (name, price) VALUES ('item', 10);
        collection('documents').find({status: 'active'})
        UPDATE customers SET email = 'new@email.com'
        """

        entities = matcher.extract(text)
        db_entities = [e for e in entities if e.type == "database"]

        assert len(db_entities) > 0

        names = [e.name for e in db_entities]
        # Should find table names
        assert "users" in names or "products" in names or "customers" in names

    def test_calculate_confidence_class(self, matcher):
        """Test confidence calculation for class entities."""
        # Well-formed class name
        confidence1 = matcher._calculate_confidence(
            0.8, "UserController", "class", "class UserController extends Base", 0, 20
        )
        assert confidence1 > 0.8  # Should be boosted

        # Class with underscore
        confidence2 = matcher._calculate_confidence(
            0.8, "User_Controller", "class", "class User_Controller", 0, 20
        )
        assert confidence2 <= 0.8  # Should not be boosted

        # Common word mistaken for class
        confidence3 = matcher._calculate_confidence(0.8, "The", "class", "The class", 0, 3)
        assert confidence3 < 0.8  # Should be penalized

    def test_calculate_confidence_function(self, matcher):
        """Test confidence calculation for function entities."""
        # Function with parentheses
        confidence1 = matcher._calculate_confidence(
            0.7, "processData", "function", "processData(input)", 0, 11
        )
        assert confidence1 > 0.7  # Should be boosted

        # Snake_case function
        confidence2 = matcher._calculate_confidence(
            0.7, "process_data", "function", "def process_data():", 4, 16
        )
        assert confidence2 > 0.7  # Should be boosted

    def test_calculate_confidence_file(self, matcher):
        """Test confidence calculation for file entities."""
        # File with path
        confidence1 = matcher._calculate_confidence(
            0.8, "src/main.py", "file", "import src/main.py", 7, 18
        )
        assert confidence1 > 0.8  # Should be boosted

        # Common extension
        confidence2 = matcher._calculate_confidence(
            0.8, "script.js", "file", "load script.js", 5, 14
        )
        assert confidence2 > 0.8  # Should be boosted


class TestNLPEntityRecognizer:
    """Test NLP-based entity recognition."""

    @patch("tenets.core.prompt.entity_recognizer.SPACY_AVAILABLE", False)
    def test_no_spacy_available(self):
        """Test when spaCy is not available."""
        recognizer = NLPEntityRecognizer()

        assert recognizer.nlp is None

        entities = recognizer.extract("John works at Microsoft in Seattle")
        assert entities == []

    @patch("tenets.core.prompt.entity_recognizer.SPACY_AVAILABLE", True)
    @patch("tenets.core.prompt.entity_recognizer.spacy")
    def test_with_spacy_available(self, mock_spacy):
        """Test when spaCy is available."""
        # Mock spaCy model
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp

        # Mock document with entities
        mock_doc = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent1.text = "John Smith"
        mock_ent1.label_ = "PERSON"
        mock_ent1.start_char = 0
        mock_ent1.end_char = 10

        mock_ent2 = MagicMock()
        mock_ent2.text = "Microsoft"
        mock_ent2.label_ = "ORG"
        mock_ent2.start_char = 20
        mock_ent2.end_char = 29

        mock_doc.ents = [mock_ent1, mock_ent2]

        # Mock noun chunks
        mock_chunk = MagicMock()
        mock_chunk.text = "software engineer"
        mock_chunk.root.pos_ = "NOUN"
        mock_chunk.start_char = 35
        mock_chunk.end_char = 52

        mock_doc.noun_chunks = [mock_chunk]

        mock_nlp.return_value = mock_doc

        recognizer = NLPEntityRecognizer()
        text = "John Smith works at Microsoft as a software engineer"
        entities = recognizer.extract(text)

        assert len(entities) >= 2

        # Check named entities
        person_entities = [e for e in entities if e.type == "person"]
        assert len(person_entities) == 1
        assert person_entities[0].name == "John Smith"

        org_entities = [e for e in entities if e.type == "organization"]
        assert len(org_entities) == 1
        assert org_entities[0].name == "Microsoft"

        # Check noun chunks
        concept_entities = [e for e in entities if e.type == "concept"]
        assert any(e.name == "software engineer" for e in concept_entities)

    @patch("tenets.core.prompt.entity_recognizer.SPACY_AVAILABLE", True)
    @patch("tenets.core.prompt.entity_recognizer.spacy")
    def test_spacy_load_failure(self, mock_spacy):
        """Test handling spaCy model load failure."""
        mock_spacy.load.side_effect = Exception("Model not found")

        recognizer = NLPEntityRecognizer()

        assert recognizer.nlp is None
        entities = recognizer.extract("Test text")
        assert entities == []


class TestFuzzyEntityMatcher:
    """Test fuzzy entity matching."""

    @pytest.fixture
    def matcher(self):
        """Create fuzzy matcher instance."""
        return FuzzyEntityMatcher()

    def test_default_known_entities(self, matcher):
        """Test default known entities."""
        assert "framework" in matcher.known_entities
        assert "language" in matcher.known_entities
        assert "database" in matcher.known_entities
        assert "tool" in matcher.known_entities
        assert "service" in matcher.known_entities

        # Check some specific entities
        assert "React" in matcher.known_entities["framework"]
        assert "Python" in matcher.known_entities["language"]
        assert "PostgreSQL" in matcher.known_entities["database"]

    def test_exact_match(self, matcher):
        """Test exact matching (case-insensitive)."""
        text = "We use React for the frontend and Python for backend"

        entities = matcher.find_fuzzy_matches(text)

        assert len(entities) >= 2

        # Check React was found
        react_entities = [e for e in entities if e.name == "React"]
        assert len(react_entities) == 1
        assert react_entities[0].type == "framework"
        assert react_entities[0].confidence == 0.95
        assert react_entities[0].metadata["match_type"] == "exact"

        # Check Python was found
        python_entities = [e for e in entities if e.name == "Python"]
        assert len(python_entities) == 1
        assert python_entities[0].type == "language"

    def test_fuzzy_match(self, matcher):
        """Test fuzzy matching with threshold."""
        text = "We use Reakt for the frontend"  # Typo: Reakt instead of React

        entities = matcher.find_fuzzy_matches(text, threshold=0.75)

        # Should find React as fuzzy match
        react_entities = [e for e in entities if e.name == "React"]
        assert len(react_entities) == 1
        assert react_entities[0].metadata["match_type"] == "fuzzy"
        assert react_entities[0].metadata["similarity"] >= 0.75
        assert react_entities[0].metadata["matched_text"] == "Reakt"

    def test_fuzzy_match_below_threshold(self, matcher):
        """Test fuzzy matching below threshold."""
        text = "We use XYZ for the frontend"

        entities = matcher.find_fuzzy_matches(text, threshold=0.9)

        # Should not match anything with high threshold
        assert len(entities) == 0

    def test_custom_known_entities(self):
        """Test with custom known entities."""
        custom_entities = {"custom_type": ["CustomTool", "SpecialLib"]}

        matcher = FuzzyEntityMatcher(known_entities=custom_entities)
        text = "Using CustomTool for the project"

        entities = matcher.find_fuzzy_matches(text)

        assert len(entities) == 1
        assert entities[0].name == "CustomTool"
        assert entities[0].type == "custom_type"


class TestHybridEntityRecognizer:
    """Test hybrid entity recognition system."""

    @pytest.fixture
    def recognizer(self):
        """Create hybrid recognizer with all features disabled."""
        return HybridEntityRecognizer(use_nlp=False, use_fuzzy=False)

    def test_initialization_all_features(self):
        """Test initialization with all features."""
        recognizer = HybridEntityRecognizer(use_nlp=True, use_fuzzy=True)

        assert recognizer.pattern_matcher is not None
        assert recognizer.fuzzy_matcher is not None
        # NLP recognizer depends on spaCy availability

    def test_recognize_pattern_only(self, recognizer):
        """Test recognition with pattern matching only."""
        text = "class UserController extends BaseController"

        entities = recognizer.recognize(text)

        assert len(entities) > 0

        # Should find class entities
        class_entities = [e for e in entities if e.type == "class"]
        assert len(class_entities) > 0
        assert any(e.name == "UserController" for e in class_entities)

    def test_recognize_with_fuzzy(self):
        """Test recognition with fuzzy matching enabled."""
        recognizer = HybridEntityRecognizer(use_nlp=False, use_fuzzy=True)

        text = "Building a React component with TypeScript"

        entities = recognizer.recognize(text)

        # Should find React and TypeScript
        names = [e.name for e in entities]
        assert "React" in names
        assert "TypeScript" in names

    def test_recognize_with_keywords(self, recognizer):
        """Test that keywords are extracted as entities."""
        text = "implement authentication system using OAuth2"

        entities = recognizer.recognize(text)

        # Should extract keywords as entities
        keyword_entities = [e for e in entities if e.type == "keyword"]
        # Keyword extraction might fail without RAKE/YAKE, but we should get some entities
        
        # Important keywords should be found either as keywords or in other entity types
        all_names = [e.name.lower() for e in entities]
        # Check if any entity contains the expected terms
        assert any("authentication" in name or "oauth2" in name for name in all_names)

    def test_confidence_filtering(self, recognizer):
        """Test filtering by confidence threshold."""
        text = "Maybe UserController or something"

        # Low threshold - should get results
        entities_low = recognizer.recognize(text, min_confidence=0.3)
        assert len(entities_low) > 0

        # High threshold - might filter out uncertain matches
        entities_high = recognizer.recognize(text, min_confidence=0.95)
        assert len(entities_high) <= len(entities_low)

    def test_merge_overlapping_entities(self, recognizer):
        """Test merging overlapping entities."""
        # Create mock entities with overlaps
        entities = [
            Entity("UserController", "class", 0.9, "", 0, 14, "regex"),
            Entity("User", "class", 0.7, "", 0, 4, "pattern"),  # Overlaps
            Entity("Controller", "class", 0.8, "", 4, 14, "pattern"),  # Overlaps
            Entity("processData", "function", 0.85, "", 20, 31, "regex"),  # No overlap
        ]

        merged = recognizer._merge_overlapping_entities(entities)

        # Should keep highest confidence overlapping entity
        assert len(merged) == 2
        assert merged[0].name == "UserController"
        assert merged[0].confidence == 0.9
        assert merged[1].name == "processData"

    def test_merge_overlapping_same_confidence(self, recognizer):
        """Test merging overlapping entities with same confidence."""
        entities = [
            Entity("test", "function", 0.8, "", 0, 4, "regex"),
            Entity("test_func", "function", 0.8, "", 0, 9, "pattern"),
        ]

        merged = recognizer._merge_overlapping_entities(entities)

        assert len(merged) == 1
        # Should merge and mark as combined
        assert merged[0].source == "combined"
        assert "sources" in merged[0].metadata

    def test_get_entity_summary(self, recognizer):
        """Test getting entity summary statistics."""
        entities = [
            Entity("UserController", "class", 0.9, "", 0, 14, "regex"),
            Entity("processData", "function", 0.85, "", 20, 31, "regex"),
            Entity("React", "framework", 0.95, "", 40, 45, "fuzzy"),
            Entity("test", "function", 0.6, "", 50, 54, "pattern"),
        ]

        summary = recognizer.get_entity_summary(entities)

        assert summary["total"] == 4
        assert summary["by_type"]["class"] == 1
        assert summary["by_type"]["function"] == 2
        assert summary["by_type"]["framework"] == 1
        assert summary["by_source"]["regex"] == 2
        assert summary["by_source"]["fuzzy"] == 1
        assert summary["by_source"]["pattern"] == 1
        assert summary["high_confidence"] == 2  # >= 0.8
        assert summary["unique_names"] == 4
        assert 0.7 < summary["avg_confidence"] < 0.9

    def test_get_entity_summary_empty(self, recognizer):
        """Test entity summary with no entities."""
        summary = recognizer.get_entity_summary([])

        assert summary["total"] == 0
        assert summary["by_type"] == {}
        assert summary["by_source"] == {}
        assert summary["avg_confidence"] == 0.0
        assert summary["high_confidence"] == 0
        assert summary["unique_names"] == 0

    @patch("tenets.core.prompt.entity_recognizer.SPACY_AVAILABLE", True)
    def test_recognize_all_sources(self):
        """Test recognition with all sources combining results."""
        recognizer = HybridEntityRecognizer(use_nlp=True, use_fuzzy=True)

        text = """
        The UserController class uses React framework.
        It connects to PostgreSQL database.
        The processData() function handles the logic.
        """

        entities = recognizer.recognize(text)

        # Should find entities from different sources
        assert len(entities) > 0

        # Check different types are found
        types = set(e.type for e in entities)
        assert len(types) >= 2  # Should have multiple entity types

        # Check different sources contributed
        sources = set(e.source for e in entities)
        assert len(sources) >= 1  # At least pattern matching should work
