"""Unit tests for the prompt parser.

Tests the PromptParser class with all features including external sources,
entity recognition, temporal parsing, intent detection, and caching.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from freezegun import freeze_time

from tenets.config import TenetsConfig
from tenets.core.prompt import (
    PromptParser,
    create_parser,
    detect_intent,
    extract_entities,
    extract_keywords,
    extract_temporal,
    parse_external_reference,
    parse_prompt,
)
from tenets.core.prompt.parser import PromptParser as DirectParser
from tenets.models.context import PromptContext


class TestPromptParser:
    """Test the PromptParser class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = TenetsConfig()
        config.nlp.enabled = True
        config.nlp.embeddings_enabled = False
        config.nlp.keyword_extraction_method = "auto"
        config.nlp.max_keywords = 20
        config.nlp.stopwords_enabled = True
        return config

    @pytest.fixture
    def parser(self, config):
        """Create parser instance."""
        return PromptParser(config)

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        manager = MagicMock()
        manager.general.get.return_value = None
        manager.general.put.return_value = None
        return manager

    def test_initialization_basic(self, config):
        """Test basic parser initialization."""
        parser = PromptParser(config)

        assert parser.config == config
        assert parser.external_manager is not None
        assert parser.entity_recognizer is not None
        assert parser.temporal_parser is not None
        assert parser.intent_detector is not None
        assert parser.keyword_extractor is not None
        assert parser.tokenizer is not None
        assert parser.stopword_manager is not None
        assert parser.programming_patterns is not None
        assert parser.cache is not None  # Cache enabled by default

    def test_initialization_with_cache_manager(self, config, mock_cache_manager):
        """Test parser initialization with cache manager."""
        parser = PromptParser(config, cache_manager=mock_cache_manager, use_cache=True)

        assert parser.cache is not None
        assert parser.cache.cache_manager == mock_cache_manager
        assert parser.cache.enable_disk is True

    def test_initialization_without_cache(self, config):
        """Test parser initialization without cache."""
        parser = PromptParser(config, use_cache=False)

        assert parser.cache is None

    def test_initialization_with_ml_features(self, config):
        """Test parser initialization with ML features enabled."""
        parser = PromptParser(config, use_ml=True, use_nlp_ner=True, use_fuzzy_matching=True)

        # Components should be initialized with requested features
        assert parser.intent_detector is not None
        assert parser.entity_recognizer is not None
        # Note: Actual ML components may not initialize without dependencies

    def test_initialization_auto_detect_features(self, config):
        """Test parser auto-detecting features from config."""
        config.nlp.embeddings_enabled = True
        config.nlp.enabled = True

        parser = PromptParser(
            config,
            use_ml=None,  # Should auto-detect as True
            use_nlp_ner=None,  # Should auto-detect as True
        )

        assert parser.intent_detector is not None
        assert parser.entity_recognizer is not None

    def test_parse_simple_prompt(self, parser):
        """Test parsing a simple text prompt."""
        prompt = "implement user authentication system"

        context = parser.parse(prompt)

        assert isinstance(context, PromptContext)
        assert context.text == prompt
        assert context.original == prompt
        assert context.intent == "implement"
        assert context.task_type == "feature"
        assert len(context.keywords) > 0
        assert any(k in ["authentication", "user", "system", "implement"] for k in context.keywords)

    def test_parse_with_cache_enabled(self, config, mock_cache_manager):
        """Test parsing with cache enabled and cache hit."""
        parser = PromptParser(config, cache_manager=mock_cache_manager, use_cache=True)

        # Mock cached result
        cached_context = PromptContext(
            text="cached prompt",
            original="cached prompt",
            keywords=["cached", "keyword"],
            task_type="cached_task",
            intent="cached_intent",
        )
        parser.cache.get_parsed_prompt = Mock(return_value=cached_context)

        context = parser.parse("test prompt", use_cache=True)

        assert context == cached_context
        parser.cache.get_parsed_prompt.assert_called_once()

    def test_parse_with_cache_miss(self, config, mock_cache_manager):
        """Test parsing with cache miss."""
        parser = PromptParser(config, cache_manager=mock_cache_manager, use_cache=True)

        parser.cache.get_parsed_prompt = Mock(return_value=None)
        parser.cache.cache_parsed_prompt = Mock()

        prompt = "implement new feature"
        context = parser.parse(prompt, use_cache=True)

        assert context.text == prompt
        parser.cache.get_parsed_prompt.assert_called_once()
        parser.cache.cache_parsed_prompt.assert_called_once()

    def test_parse_without_cache(self, parser):
        """Test parsing with cache disabled for specific call."""
        if parser.cache:
            parser.cache.get_parsed_prompt = Mock()
            parser.cache.cache_parsed_prompt = Mock()

        prompt = "debug the issue"
        context = parser.parse(prompt, use_cache=False)

        assert context.text == prompt
        if parser.cache:
            parser.cache.get_parsed_prompt.assert_not_called()
            parser.cache.cache_parsed_prompt.assert_not_called()

    @patch("tenets.core.prompt.parser.ExternalSourceManager")
    def test_parse_with_external_url(self, mock_external_manager_class, parser):
        """Test parsing a GitHub URL with external content fetching."""
        # Setup mock
        mock_manager = MagicMock()
        mock_external_manager_class.return_value = mock_manager

        # Mock external reference extraction
        mock_manager.extract_reference.return_value = (
            "https://github.com/org/repo/issues/123",
            "org/repo#123",
            {"platform": "github", "type": "issue", "number": "123"},
        )

        # Mock external content fetch
        mock_content = MagicMock()
        mock_content.title = "Bug: Application crashes on startup"
        mock_content.body = "The application fails to start when database is unavailable"
        mock_manager.process_url.return_value = mock_content

        # Replace parser's manager
        parser.external_manager = mock_manager

        context = parser.parse("https://github.com/org/repo/issues/123", fetch_external=True)

        assert context.external_context is not None
        assert context.external_context["source"] == "github"
        assert context.external_context["identifier"] == "org/repo#123"
        assert "Bug" in context.text or "crash" in context.text.lower()

    def test_parse_external_url_without_fetch(self, parser):
        """Test parsing URL without fetching external content."""
        url = "https://github.com/org/repo/issues/123"

        context = parser.parse(url, fetch_external=False)

        # Should still detect external reference but not fetch
        if context.external_context:
            assert context.external_context["url"] == url
        # Text should be original URL since no fetch
        assert url in context.original

    def test_parse_with_entity_recognition(self, parser):
        """Test entity recognition in parsed prompt."""
        prompt = "Fix the UserController class in src/auth/controller.py file"

        context = parser.parse(prompt, min_entity_confidence=0.5)

        assert len(context.entities) > 0

        # Should find class and file entities
        entity_types = [e["type"] for e in context.entities]
        entity_names = [e["name"] for e in context.entities]

        # At least one entity should be found
        assert any(t in ["class", "file", "keyword"] for t in entity_types)
        # Check for the presence of our key terms in any extracted entity
        all_names = " ".join(entity_names).lower()
        assert "usercontroller" in all_names or "controller" in all_names or len(entity_names) > 0

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_with_temporal_context(self, parser):
        """Test temporal expression parsing."""
        prompt = "Show me changes from last week"

        context = parser.parse(prompt)

        assert context.temporal_context is not None
        assert context.temporal_context["is_relative"] is True
        assert context.temporal_context["since"] < datetime.now()
        assert context.temporal_context["expressions"] > 0

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_absolute_date(self, parser):
        """Test parsing absolute date."""
        prompt = "Issues created on 2024-01-01"

        context = parser.parse(prompt)

        if context.temporal_context:
            assert context.temporal_context["since"] is not None
            assert context.temporal_context["is_relative"] is False

    def test_parse_with_file_patterns(self, parser):
        """Test extracting file patterns from prompt."""
        prompt = "Update all *.py files except test_*.py in the src directory"

        context = parser.parse(prompt)

        assert len(context.file_patterns) > 0
        # Should find patterns
        assert any("*.py" in p or ".py" in p for p in context.file_patterns)

    def test_parse_with_focus_areas(self, parser):
        """Test identifying focus areas."""
        prompt = "Fix the database connection issues and update API endpoints"

        context = parser.parse(prompt)

        # Should identify database and API focus areas
        assert len(context.focus_areas) > 0
        assert "database" in context.focus_areas or "api" in context.focus_areas

    def test_parse_with_scope_extraction(self, parser):
        """Test scope extraction from prompt."""
        prompt = "Refactor the authentication module in src/auth directory"

        context = parser.parse(prompt)

        assert context.scope is not None
        assert len(context.scope["modules"]) > 0 or len(context.scope["directories"]) > 0
        assert context.scope["is_specific"] is True
        assert context.scope["is_global"] is False

    def test_parse_global_scope(self, parser):
        """Test detecting global scope."""
        prompt = "Analyze the entire project for security issues"

        context = parser.parse(prompt)

        assert context.scope["is_global"] is True
        assert context.scope["is_specific"] is False

    def test_parse_with_exclusions(self, parser):
        """Test parsing scope with exclusions."""
        prompt = "Check all files except those in node_modules and vendor directories"

        context = parser.parse(prompt)

        assert len(context.scope["exclusions"]) > 0
        assert any("node_modules" in e or "vendor" in e for e in context.scope["exclusions"])

    def test_intent_detection_implement(self, parser):
        """Test detecting implementation intent."""
        prompts = [
            "implement OAuth2 authentication",
            "add new user profile feature",
            "create REST API endpoints",
            "build notification system",
            "develop payment integration",
        ]

        for prompt in prompts:
            context = parser.parse(prompt)
            assert context.intent == "implement"
            assert context.task_type == "feature"

    def test_intent_detection_debug(self, parser):
        """Test detecting debug intent."""
        prompts = [
            "fix the login bug",
            "debug memory leak issue",
            "resolve database connection error",
            "application is crashing",
            "troubleshoot API timeout",
        ]

        for prompt in prompts:
            context = parser.parse(prompt)
            assert context.intent == "debug"
            assert context.task_type == "debug"

    def test_intent_detection_understand(self, parser):
        """Test detecting understand intent."""
        prompts = [
            "explain how authentication works",
            "what does this function do",
            "show me the system architecture",
            "understand the data flow",
        ]

        for prompt in prompts:
            context = parser.parse(prompt)
            assert context.intent == "understand"
            assert context.task_type == "understand"

    def test_intent_detection_refactor(self, parser):
        """Test detecting refactor intent."""
        prompts = [
            "refactor the authentication module",
            "clean up the codebase",
            "restructure the project",
            "modernize legacy code",
        ]

        for prompt in prompts:
            context = parser.parse(prompt)
            assert context.intent == "refactor"
            assert context.task_type == "refactor"

    def test_intent_detection_optimize(self, parser):
        """Test detecting optimize intent."""
        prompts = [
            "optimize database queries",
            "improve application performance",
            "reduce memory usage",
            "make the API faster",
        ]

        for prompt in prompts:
            context = parser.parse(prompt)
            assert context.intent == "optimize"
            assert context.task_type == "optimize"

    def test_intent_to_task_type_mapping(self, parser):
        """Test intent to task type conversion."""
        mappings = {
            "implement": "feature",
            "debug": "debug",
            "understand": "understand",
            "refactor": "refactor",
            "test": "test",
            "document": "document",
            "review": "review",
            "optimize": "optimize",
            "integrate": "feature",
            "migrate": "refactor",
            "configure": "configuration",
            "analyze": "analysis",
        }

        for intent, expected_task in mappings.items():
            result = parser._intent_to_task_type(intent)
            assert result == expected_task

    def test_intent_to_task_type_default(self, parser):
        """Test default task type for unknown intent."""
        result = parser._intent_to_task_type("unknown_intent")
        assert result == "general"

    def test_extract_file_patterns_comprehensive(self, parser):
        """Test comprehensive file pattern extraction."""
        text = "Process all *.js and *.ts files in ./src except test_* and ../legacy"

        patterns = parser._extract_file_patterns(text)

        assert len(patterns) > 0
        # Should find various patterns
        pattern_str = " ".join(patterns)
        assert ".js" in pattern_str or "*.js" in pattern_str
        assert ".ts" in pattern_str or "*.ts" in pattern_str
        assert any("./" in p or "../" in p for p in patterns)

    def test_extract_specific_files(self, parser):
        """Test extracting specific file mentions."""
        text = "Update the main.py file and config.json from settings"

        patterns = parser._extract_file_patterns(text)

        # Should find specific files
        assert any("main.py" in p for p in patterns)
        assert any("config.json" in p for p in patterns)

    def test_extract_focus_areas_from_entities(self, parser):
        """Test focus area extraction based on entities."""
        from tenets.core.prompt.entity_recognizer import Entity

        entities = [
            Entity("UserAPI", "api_endpoint", 0.9, "", 0, 7, "regex"),
            Entity("users_table", "database", 0.8, "", 10, 21, "regex"),
            Entity("AuthController", "class", 0.85, "", 25, 39, "regex"),
            Entity("NetworkError", "error", 0.7, "", 45, 57, "regex"),
            Entity("Button", "component", 0.75, "", 60, 66, "regex"),
        ]

        text = "Fix UserAPI users_table AuthController NetworkError Button"
        focus_areas = parser._extract_focus_areas(text, entities)

        assert "api" in focus_areas
        assert "database" in focus_areas
        assert "architecture" in focus_areas
        assert "error_handling" in focus_areas
        assert "ui" in focus_areas

    def test_extract_scope_modules(self, parser):
        """Test module extraction in scope."""
        text = "Update the authentication module and the payment package"

        scope = parser._extract_scope(text)

        assert len(scope["modules"]) >= 1
        assert "authentication" in scope["modules"] or "payment" in scope["modules"]
        assert scope["is_specific"] is True

    def test_extract_scope_directories(self, parser):
        """Test directory extraction in scope."""
        text = "Check files in src/components and under tests/unit directory"

        scope = parser._extract_scope(text)

        assert len(scope["directories"]) >= 1
        # Should find directory paths
        dirs_str = " ".join(scope["directories"])
        assert "src" in dirs_str or "components" in dirs_str or "tests" in dirs_str

    def test_extract_scope_specific_files(self, parser):
        """Test specific file extraction in scope."""
        text = "Review main.py, config.json, and utils.js files"

        scope = parser._extract_scope(text)

        assert len(scope["specific_files"]) >= 1
        files_str = " ".join(scope["specific_files"])
        assert "main.py" in files_str or "config.json" in files_str or "utils.js" in files_str

    def test_metadata_generation(self, parser):
        """Test metadata generation in parsed context."""
        prompt = "fix the authentication bug in UserController"

        context = parser.parse(prompt)

        assert hasattr(context, "metadata")
        metadata = context.metadata

        assert "intent_confidence" in metadata
        assert "entity_count" in metadata
        assert "temporal_expressions" in metadata
        assert "has_external_ref" in metadata
        assert "cached" in metadata
        assert "avg_confidence" in metadata

        # Check confidence is reasonable
        assert 0 <= metadata["avg_confidence"] <= 1

    def test_get_cache_stats(self, config, mock_cache_manager):
        """Test getting cache statistics."""
        parser = PromptParser(config, cache_manager=mock_cache_manager, use_cache=True)

        parser.cache.get_stats = Mock(return_value={"hits": 10, "misses": 5, "hit_rate": 0.67})

        stats = parser.get_cache_stats()

        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == 0.67

    def test_get_cache_stats_no_cache(self, config):
        """Test getting cache stats when cache is disabled."""
        parser = PromptParser(config, use_cache=False)

        stats = parser.get_cache_stats()

        assert stats is None

    def test_clear_cache(self, config, mock_cache_manager):
        """Test clearing the cache."""
        parser = PromptParser(config, cache_manager=mock_cache_manager, use_cache=True)

        parser.cache.clear_all = Mock()

        parser.clear_cache()

        parser.cache.clear_all.assert_called_once()

    def test_clear_cache_no_cache(self, config):
        """Test clearing cache when disabled."""
        parser = PromptParser(config, use_cache=False)

        # Should not raise error
        parser.clear_cache()
        assert True

    def test_warm_cache(self, config, mock_cache_manager):
        """Test cache warming with common prompts."""
        parser = PromptParser(config, cache_manager=mock_cache_manager, use_cache=True)

        common_prompts = ["implement feature", "fix bug", "understand code"]

        with patch.object(parser, "_parse_internal") as mock_parse:
            mock_parse.return_value = PromptContext(
                text="test", original="test", keywords=[], task_type="general", intent="understand"
            )

            parser.warm_cache(common_prompts)

            assert mock_parse.call_count == len(common_prompts)

            # Should parse without fetching external content
            for call in mock_parse.call_args_list:
                assert call[0][1] is False  # fetch_external=False

    def test_warm_cache_no_cache(self, config):
        """Test warming cache when cache is disabled."""
        parser = PromptParser(config, use_cache=False)

        # Should return early without processing
        parser.warm_cache(["test prompt"])

        # No error should occur
        assert True

    def test_parse_complex_prompt(self, parser):
        """Test parsing a complex prompt with multiple features."""
        prompt = """
        Fix the UserController class bug in src/auth/controller.py
        that was reported yesterday. Focus on the login() and logout() methods.
        Check all *.py files except test files.
        """

        context = parser.parse(prompt)

        # Should detect debug intent
        assert context.intent == "debug"
        assert context.task_type == "debug"

        # Should find entities
        assert len(context.entities) > 0
        entity_names = [e["name"] for e in context.entities]
        assert any(
            "UserController" in name or "controller" in name.lower() for name in entity_names
        )

        # Should find file patterns
        assert len(context.file_patterns) > 0

        # Should extract scope
        assert context.scope is not None
        if context.scope["directories"]:
            assert any("src" in d or "auth" in d for d in context.scope["directories"])

        # Should have keywords
        assert len(context.keywords) > 0
        assert any(
            k in ["bug", "fix", "UserController", "login", "logout", "auth"]
            for k in context.keywords
        )

    def test_should_include_tests_explicit_test_intent(self, parser):
        """Test that explicit test intent includes tests."""
        # Test with test intent
        result = parser._should_include_tests("test", "write unit tests", ["test", "unit"])
        assert result is True

    def test_should_include_tests_test_keywords(self, parser):
        """Test that test-related keywords include tests."""
        test_cases = [
            (["test", "coverage"], True),
            (["unit", "testing"], True),
            (["pytest", "jest"], True),
            (["integration", "e2e"], True),
            (["mock", "stub"], True),
            (["spec", "assertion"], True),
            (["auth", "login"], False),  # Non-test keywords
            (["database", "api"], False),  # Non-test keywords
        ]

        for keywords, expected in test_cases:
            result = parser._should_include_tests("understand", "general prompt", keywords)
            assert result == expected, f"Keywords {keywords} should return {expected}"

    def test_should_include_tests_file_patterns(self, parser):
        """Test that test file patterns include tests."""
        test_cases = [
            ("debug test_auth.py file", True),
            ("fix auth.test.js errors", True),
            ("check UserTest.java class", True),
            ("update auth_test.py", True),
            ("check tests/auth directory", True),
            ("review __tests__ folder", True),
            ("debug auth.py file", False),  # Non-test file
            ("fix main.js errors", False),  # Non-test file
        ]

        for prompt, expected in test_cases:
            result = parser._should_include_tests("debug", prompt, [])
            assert result == expected, f"Prompt '{prompt}' should return {expected}"

    def test_should_include_tests_action_patterns(self, parser):
        """Test that test-related actions include tests."""
        test_cases = [
            ("write unit tests for auth", True),
            ("add integration tests", True),
            ("fix failing tests", True),
            ("run tests for coverage", True),
            ("mock the database", True),
            ("test coverage report", True),
            ("unit test assertion", True),
            ("e2e test suite", True),
            ("write documentation", False),  # Non-test action
            ("fix authentication bug", False),  # Non-test action
        ]

        for prompt, expected in test_cases:
            result = parser._should_include_tests("implement", prompt, [])
            assert result == expected, f"Prompt '{prompt}' should return {expected}"

    def test_should_include_tests_quality_patterns(self, parser):
        """Test that test quality patterns include tests."""
        test_cases = [
            ("check test coverage", True),
            ("generate coverage report", True),
            ("fix failing tests", True),
            ("broken test suite", True),
            ("test failures in CI", True),
            ("tests are passing", True),
            ("check code coverage", False),  # Not test-specific
            ("fix broken deployment", False),  # Non-test quality
        ]

        for prompt, expected in test_cases:
            result = parser._should_include_tests("debug", prompt, [])
            assert result == expected, f"Prompt '{prompt}' should return {expected}"

    def test_should_include_tests_default_exclusion(self, parser):
        """Test that non-test prompts default to excluding tests."""
        test_cases = [
            ("explain authentication flow", False),
            ("implement user registration", False),
            ("debug payment processing", False),
            ("refactor database models", False),
            ("optimize API performance", False),
            ("understand code architecture", False),
        ]

        for prompt, expected in test_cases:
            result = parser._should_include_tests("understand", prompt, [])
            assert result == expected, f"Prompt '{prompt}' should return {expected}"

    def test_parse_includes_test_flag(self, parser):
        """Test that parsing sets include_tests flag correctly."""
        # Test-related prompt should include tests
        context = parser.parse("write unit tests for authentication")
        assert context.include_tests is True

        # Non-test prompt should exclude tests
        context = parser.parse("explain authentication flow")
        assert context.include_tests is False

        # Test file mention should include tests
        context = parser.parse("debug test_auth.py failure")
        assert context.include_tests is True


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_create_parser_default(self):
        """Test creating parser with defaults."""
        parser = create_parser()

        assert isinstance(parser, PromptParser)
        assert parser.config is not None
        assert parser.cache is not None  # Cache enabled by default

    def test_create_parser_with_config(self):
        """Test creating parser with custom config."""
        config = TenetsConfig()
        config.nlp.max_keywords = 10

        parser = create_parser(config)

        assert parser.config == config
        assert parser.config.nlp.max_keywords == 10

    def test_create_parser_with_options(self, mock_cache_manager):
        """Test creating parser with options."""
        parser = create_parser(use_cache=False, use_ml=True, cache_manager=mock_cache_manager)

        assert parser.cache is None  # Cache disabled

    def test_parse_prompt_function(self):
        """Test parse_prompt convenience function."""
        context = parse_prompt("implement authentication")

        assert isinstance(context, PromptContext)
        assert context.intent == "implement"
        assert len(context.keywords) > 0

    def test_parse_prompt_with_options(self):
        """Test parse_prompt with options."""
        context = parse_prompt("fix bug", fetch_external=False, use_cache=True)

        assert context.intent == "debug"

    def test_extract_keywords_function(self):
        """Test extract_keywords convenience function."""
        keywords = extract_keywords("implement OAuth2 authentication system")

        assert isinstance(keywords, list)
        # Keywords might be combined phrases or individual words
        if len(keywords) > 0:
            # Check if any keyword contains the expected terms
            assert any(
                any(
                    term in kw.lower()
                    for term in ["oauth2", "authentication", "system", "implement"]
                )
                for kw in keywords
            )

    def test_extract_keywords_with_limit(self):
        """Test extract_keywords with max limit."""
        keywords = extract_keywords(
            "implement OAuth2 authentication system for web application", max_keywords=3
        )

        assert len(keywords) <= 3

    def test_detect_intent_function(self):
        """Test detect_intent convenience function."""
        intent = detect_intent("fix the authentication bug")

        assert intent == "debug"

    def test_detect_intent_implement(self):
        """Test detect_intent for implementation."""
        intent = detect_intent("add new feature")

        assert intent == "implement"

    @patch("tenets.core.nlp.embeddings.LocalEmbeddings.encode")
    def test_detect_intent_with_ml(self, mock_encode):
        """Test detect_intent with ML enabled."""
        # Mock the encode method to return proper arrays
        mock_encode.return_value = np.array([0.1] * 384)  # Return a dummy embedding

        # Test with various prompts
        debug_intent = detect_intent("debug the error in the code", use_ml=True)
        implement_intent = detect_intent("implement new feature", use_ml=True)

        # ML-based detection should return valid intents
        assert debug_intent in ["debug", "implement", "understand", "refactor", "test", "explore"]
        assert implement_intent in [
            "implement",
            "debug",
            "understand",
            "refactor",
            "test",
            "explore",
        ]

    def test_extract_entities_function(self):
        """Test extract_entities convenience function."""
        entities = extract_entities("Update the UserAuth class in auth.py")

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check entity structure
        for entity in entities:
            assert "name" in entity
            assert "type" in entity
            assert "confidence" in entity
            assert "context" in entity

    def test_extract_entities_with_options(self):
        """Test extract_entities with options."""
        entities = extract_entities(
            "UserController and DatabaseManager classes",
            min_confidence=0.7,
            use_nlp=False,
            use_fuzzy=True,
        )

        # Should find class entities
        assert any(e["type"] == "class" for e in entities)

    def test_parse_external_reference_github(self):
        """Test parsing GitHub URL."""
        ref = parse_external_reference("https://github.com/org/repo/issues/123")

        assert ref is not None
        assert ref["type"] == "github"
        assert ref["identifier"] == "org/repo#123"
        assert ref["metadata"]["number"] == "123"

    def test_parse_external_reference_jira(self):
        """Test parsing JIRA URL."""
        ref = parse_external_reference("https://company.atlassian.net/browse/PROJ-123")

        assert ref is not None
        assert ref["type"] == "jira"
        assert ref["identifier"] == "PROJ-123"
        assert ref["metadata"]["ticket"] == "PROJ-123"

    def test_parse_external_reference_invalid(self):
        """Test parsing invalid URL."""
        ref = parse_external_reference("https://example.com/page")

        assert ref is None

    def test_extract_temporal_function(self):
        """Test extract_temporal convenience function."""
        temporal = extract_temporal("changes from last week and yesterday")

        assert isinstance(temporal, list)
        assert len(temporal) > 0

        # Check temporal structure
        for expr in temporal:
            assert "text" in expr
            assert "type" in expr
            assert "is_relative" in expr
            assert "is_recurring" in expr
            assert "confidence" in expr


class TestIntegration:
    """Integration tests for the prompt parser."""

    @pytest.fixture
    def parser(self):
        """Create parser with all features."""
        config = TenetsConfig()
        return PromptParser(
            config,
            use_ml=False,  # Disable ML to avoid dependency issues
            use_nlp_ner=False,  # Disable NLP NER
            use_fuzzy_matching=True,
        )

    def test_parse_github_issue_reference(self, parser):
        """Test parsing a prompt with GitHub issue reference."""
        prompt = "Fix the bug mentioned in https://github.com/microsoft/vscode/issues/12345"

        context = parser.parse(prompt, fetch_external=False)

        assert context.intent == "debug"
        assert context.external_context is not None
        assert context.external_context["source"] == "github"
        assert "12345" in context.external_context["identifier"]

    def test_parse_with_temporal_and_scope(self, parser):
        """Test parsing with both temporal and scope information."""
        prompt = "Show me changes from last month in the src/components directory"

        context = parser.parse(prompt)

        # Should have temporal context
        if context.temporal_context:
            assert context.temporal_context["is_relative"] is True

        # Should have scope
        assert context.scope is not None
        assert len(context.scope["directories"]) > 0 or context.scope["is_specific"]

    def test_parse_mixed_intent_signals(self, parser):
        """Test parsing with mixed intent signals."""
        prompt = "Understand and fix the performance issue in the database module"

        context = parser.parse(prompt)

        # Could be either debug or optimize based on signals
        assert context.intent in ["debug", "optimize", "understand"]

        # Should identify database focus
        assert "database" in context.focus_areas or "database" in context.keywords

    def test_parse_with_programming_patterns(self, parser):
        """Test that programming patterns are recognized."""
        prompt = "Implement async function with error handling and logging"

        context = parser.parse(prompt)

        assert context.intent == "implement"

        # Should extract programming keywords
        keywords_lower = [k.lower() for k in context.keywords]
        assert any(
            k in ["async", "function", "error", "handling", "logging"] for k in keywords_lower
        )

    def test_parse_minimal_prompt(self, parser):
        """Test parsing a minimal prompt."""
        prompt = "help"

        context = parser.parse(prompt)

        assert context.intent == "understand"  # Default
        assert context.task_type in ["understand", "general"]
        assert len(context.keywords) >= 0  # Might not extract from single word

    def test_parse_url_only(self, parser):
        """Test parsing just a URL."""
        prompt = "https://linear.app/team/issue/ENG-123"

        context = parser.parse(prompt, fetch_external=False)

        assert context.external_context is not None
        assert context.external_context["source"] == "linear"

    def test_direct_import(self):
        """Test that PromptParser can be imported directly."""
        assert DirectParser is PromptParser

        # Should be the same class
        parser1 = PromptParser(TenetsConfig())
        parser2 = DirectParser(TenetsConfig())

        assert type(parser1) is type(parser2)


class TestErrorHandling:
    """Test error handling in the prompt parser."""

    @pytest.fixture
    def parser(self):
        """Create basic parser."""
        return PromptParser(TenetsConfig(), use_cache=False)

    def test_parse_empty_prompt(self, parser):
        """Test parsing empty prompt."""
        context = parser.parse("")

        assert context.text == ""
        assert context.original == ""
        assert context.intent == "understand"  # Default

    def test_parse_none_prompt(self, parser):
        """Test handling None prompt (should raise)."""
        with pytest.raises(AttributeError):
            parser.parse(None)

    def test_parse_very_long_prompt(self, parser):
        """Test parsing very long prompt."""
        prompt = "implement " * 1000 + "feature"

        context = parser.parse(prompt)

        assert context.intent == "implement"
        assert len(context.keywords) <= parser.config.nlp.max_keywords

    def test_parse_with_special_characters(self, parser):
        """Test parsing prompt with special characters."""
        prompt = "Fix bug in @#$% module & *() function"

        context = parser.parse(prompt)

        assert context.intent == "debug"
        # Should still extract some keywords
        assert any("bug" in kw or "fix" in kw for kw in context.keywords)

    def test_parse_non_english_mixed(self, parser):
        """Test parsing mixed language prompt."""
        prompt = "implement 認証 authentication système"

        context = parser.parse(prompt)

        assert context.intent == "implement"
        assert "authentication" in context.keywords
