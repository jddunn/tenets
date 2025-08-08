"""Tests for the PromptParser module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import os

from tenets.core.prompt.parser import (
    PromptParser, IntentType, ParsedEntity, 
    TemporalContext, ExternalReference
)
from tenets.models.context import PromptContext
from tenets.config import TenetsConfig


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def parser(config):
    """Create a PromptParser instance."""
    return PromptParser(config)


class TestPromptParser:
    """Test suite for PromptParser."""
    
    def test_initialization(self, config):
        """Test parser initialization."""
        parser = PromptParser(config)
        
        assert parser.config == config
        assert parser._intent_patterns is not None
        assert parser._entity_patterns is not None
        assert parser._temporal_patterns is not None
        
    def test_parse_simple_prompt(self, parser):
        """Test parsing a simple prompt."""
        prompt = "implement OAuth2 authentication for the API"
        context = parser.parse(prompt)
        
        assert isinstance(context, PromptContext)
        assert context.text == prompt
        assert context.original == prompt
        assert context.task_type == 'feature'
        assert context.intent == IntentType.IMPLEMENT.value
        assert 'oauth2' in [k.lower() for k in context.keywords]
        assert 'api' in [k.lower() for k in context.keywords]
        
    def test_detect_intent_implement(self, parser):
        """Test intent detection for implementation."""
        prompts = [
            "implement new feature",
            "add authentication",
            "create user model",
            "build REST API",
            "develop dashboard"
        ]
        
        for prompt in prompts:
            intent = parser._detect_intent(prompt)
            assert intent == IntentType.IMPLEMENT
            
    def test_detect_intent_debug(self, parser):
        """Test intent detection for debugging."""
        prompts = [
            "fix login bug",
            "debug memory leak",
            "solve authentication issue",
            "troubleshoot API errors",
            "users getting 401 errors"
        ]
        
        for prompt in prompts:
            intent = parser._detect_intent(prompt)
            assert intent == IntentType.DEBUG
            
    def test_detect_intent_understand(self, parser):
        """Test intent detection for understanding."""
        prompts = [
            "how does authentication work",
            "explain the payment flow",
            "what does this function do",
            "show me the API structure",
            "understand caching mechanism"
        ]
        
        for prompt in prompts:
            intent = parser._detect_intent(prompt)
            assert intent == IntentType.UNDERSTAND
            
    def test_detect_intent_refactor(self, parser):
        """Test intent detection for refactoring."""
        prompts = [
            "refactor authentication module",
            "restructure database schema",
            "improve code organization",
            "clean up API endpoints",
            "optimize query performance"
        ]
        
        for prompt in prompts:
            intent = parser._detect_intent(prompt)
            assert intent in [IntentType.REFACTOR, IntentType.OPTIMIZE]
            
    def test_extract_entities(self, parser):
        """Test entity extraction."""
        prompt = """
        Update the User class in auth.py to add a validate_password method.
        Also modify the login function to use the new validation.
        """
        
        entities = parser._extract_entities(prompt)
        
        assert len(entities) > 0
        
        # Check for class entity
        class_entities = [e for e in entities if e.type == 'class']
        assert any(e.name == 'User' for e in class_entities)
        
        # Check for function entity
        func_entities = [e for e in entities if e.type == 'function']
        assert any('validate_password' in e.name for e in func_entities)
        
        # Check for file entity
        file_entities = [e for e in entities if e.type == 'file']
        assert any('auth.py' in e.name for e in file_entities)
        
    def test_extract_keywords_with_yake(self, parser):
        """Test keyword extraction with YAKE if available."""
        prompt = """
        Implement OAuth2 authentication with JWT tokens for the REST API.
        Support refresh tokens and integrate with Redis for session caching.
        """
        
        keywords = parser._extract_keywords(prompt)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        
        # Should include important technical terms
        keywords_lower = [k.lower() for k in keywords]
        assert any('oauth' in k for k in keywords_lower)
        assert any('jwt' in k or 'token' in k for k in keywords_lower)
        
    def test_extract_programming_keywords(self, parser):
        """Test extraction of programming-specific keywords."""
        prompt = """
        Set up Docker containers with Kubernetes orchestration.
        Use PostgreSQL for the database and Redis for caching.
        Implement GraphQL API with authentication.
        """
        
        keywords = parser._extract_programming_keywords(prompt)
        
        assert 'docker' in keywords
        assert 'kubernetes' in keywords
        assert 'postgres' in keywords
        assert 'redis' in keywords
        assert 'graphql' in keywords
        assert 'auth' in keywords
        
    def test_extract_file_patterns(self, parser):
        """Test file pattern extraction."""
        prompt = """
        Review all *.py files in the src/ directory.
        Exclude test_*.py files and anything in __pycache__.
        Focus on .js and .tsx files in the frontend folder.
        """
        
        patterns = parser._extract_file_patterns(prompt)
        
        assert '*.py' in patterns or '.py' in patterns
        assert any('src/' in p for p in patterns)
        assert '.js' in patterns
        assert '.tsx' in patterns
        
    def test_extract_focus_areas(self, parser):
        """Test focus area extraction."""
        prompt = """
        Fix authentication issues in the API endpoints.
        The database queries are too slow and need optimization.
        Also review security vulnerabilities in user input handling.
        """
        
        entities = parser._extract_entities(prompt)
        focus_areas = parser._extract_focus_areas(prompt, entities)
        
        assert 'authentication' in focus_areas
        assert 'api' in focus_areas
        assert 'database' in focus_areas
        assert 'security' in focus_areas
        assert 'performance' in focus_areas
        
    def test_extract_temporal_context_relative(self, parser):
        """Test temporal context extraction with relative timeframes."""
        test_cases = [
            ("changes from yesterday", "yesterday", 1),
            ("recent modifications", "recent", 7),
            ("what changed last week", "last week", 7),
            ("updates from last month", "last month", 30),
        ]
        
        for prompt, expected_timeframe, expected_days in test_cases:
            temporal = parser._extract_temporal_context(prompt)
            
            assert temporal is not None
            assert temporal.timeframe == expected_timeframe
            assert temporal.is_relative == True
            assert temporal.since is not None
            
            # Check approximate time delta
            delta = datetime.now() - temporal.since
            assert delta.days <= expected_days + 1
            
    def test_extract_temporal_context_absolute(self, parser):
        """Test temporal context extraction with absolute dates."""
        prompt = "show changes since 2024-01-15"
        
        temporal = parser._extract_temporal_context(prompt)
        
        assert temporal is not None
        assert temporal.timeframe == "2024-01-15"
        assert temporal.is_relative == False
        assert temporal.since.year == 2024
        assert temporal.since.month == 1
        assert temporal.since.day == 15
        
    def test_extract_scope(self, parser):
        """Test scope extraction."""
        prompt = """
        Refactor the authentication module in the src/auth directory.
        Focus on login.py and session.py files.
        Exclude anything in tests/ and ignore *.pyc files.
        """
        
        scope = parser._extract_scope(prompt)
        
        assert 'authentication' in scope['modules'] or 'auth' in scope['modules']
        assert any('auth' in d for d in scope['directories'])
        assert 'login.py' in scope['specific_files']
        assert 'session.py' in scope['specific_files']
        assert any('test' in e for e in scope['exclusions'])
        assert scope['is_specific'] == True
        
    def test_detect_external_reference_github(self, parser):
        """Test GitHub URL detection."""
        prompt = "implement the feature described in https://github.com/owner/repo/issues/123"
        
        ref = parser._detect_external_reference(prompt)
        
        assert ref is not None
        assert ref.type == 'github'
        assert ref.identifier == 'owner/repo#123'
        assert ref.metadata['owner'] == 'owner'
        assert ref.metadata['repo'] == 'repo'
        assert ref.metadata['number'] == '123'
        
    def test_detect_external_reference_jira(self, parser):
        """Test JIRA URL detection."""
        prompt = "fix the bug in https://company.atlassian.net/browse/PROJ-456"
        
        ref = parser._detect_external_reference(prompt)
        
        assert ref is not None
        assert ref.type == 'jira'
        assert ref.identifier == 'PROJ-456'
        assert ref.metadata['ticket'] == 'PROJ-456'
        
    @patch('requests.get')
    def test_fetch_external_content_github(self, mock_get, parser):
        """Test fetching content from GitHub."""
        # Mock GitHub API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'title': 'Test Issue',
            'body': 'Issue description',
            'user': {'login': 'testuser'}
        }
        mock_get.return_value = mock_response
        
        ref = ExternalReference(
            type='github',
            url='https://github.com/owner/repo/issues/123',
            identifier='owner/repo#123',
            metadata={
                'owner': 'owner',
                'repo': 'repo',
                'number': '123',
                'type': 'issues'
            }
        )
        
        content = parser._fetch_external_content(ref)
        
        assert 'Title: Test Issue' in content
        assert 'Description: Issue description' in content
        mock_get.assert_called()
        
    def test_parse_with_external_url(self, parser):
        """Test parsing a prompt that is a URL."""
        with patch.object(parser, '_fetch_external_content') as mock_fetch:
            mock_fetch.return_value = "Fetched content: implement new feature"
            
            prompt = "https://github.com/owner/repo/issues/123"
            context = parser.parse(prompt)
            
            assert context.original == prompt
            assert context.text == "Fetched content: implement new feature"
            assert context.external_context is not None
            assert context.external_context['source'] == 'github'
            
    def test_intent_to_task_type_mapping(self, parser):
        """Test intent to task type conversion."""
        mappings = [
            (IntentType.IMPLEMENT, 'feature'),
            (IntentType.DEBUG, 'debug'),
            (IntentType.UNDERSTAND, 'understand'),
            (IntentType.REFACTOR, 'refactor'),
            (IntentType.TEST, 'test'),
            (IntentType.DOCUMENT, 'document'),
            (IntentType.REVIEW, 'review'),
            (IntentType.OPTIMIZE, 'optimize'),
            (IntentType.INTEGRATE, 'feature'),
            (IntentType.MIGRATE, 'refactor'),
        ]
        
        for intent, expected_task in mappings:
            task_type = parser._intent_to_task_type(intent)
            assert task_type == expected_task
            
    def test_simple_keyword_extraction_fallback(self, parser):
        """Test keyword extraction without YAKE."""
        # Mock YAKE not being available
        parser._keyword_extractor = None
        
        prompt = "implement user authentication with password hashing"
        keywords = parser._simple_keyword_extraction(prompt)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert 'authentication' in keywords
        assert 'password' in keywords
        assert 'hashing' in keywords
        
        # Common words should be filtered
        assert 'with' not in keywords
        assert 'the' not in keywords
        
    def test_enhance_with_context(self, parser):
        """Test enhancing prompt context with additional info."""
        context = PromptContext(
            text="original prompt",
            keywords=['keyword1'],
            focus_areas=['area1']
        )
        
        additional_info = {
            'keywords': ['keyword2', 'keyword3'],
            'focus_areas': ['area2'],
            'metadata': {'source': 'test'}
        }
        
        enhanced = parser.enhance_with_context(context, additional_info)
        
        assert 'keyword1' in enhanced.keywords
        assert 'keyword2' in enhanced.keywords
        assert 'keyword3' in enhanced.keywords
        assert 'area1' in enhanced.focus_areas
        assert 'area2' in enhanced.focus_areas
        assert enhanced.metadata['source'] == 'test'
        
    def test_complex_prompt_parsing(self, parser):
        """Test parsing a complex, multi-faceted prompt."""
        prompt = """
        Fix the authentication bug in https://github.com/myapp/backend/issues/456
        that's causing users to get 401 errors when accessing /api/users endpoint.
        The issue started happening yesterday after deploying the OAuth2 changes.
        Check the User class in src/auth/models.py and the login function in auth.py.
        Focus on the JWT token validation and Redis session caching.
        Exclude test files and anything in the vendor directory.
        """
        
        context = parser.parse(prompt)
        
        # Check intent and task type
        assert context.task_type == 'debug'
        assert context.intent == IntentType.DEBUG.value
        
        # Check external context
        assert context.external_context is not None
        assert context.external_context['source'] == 'github'
        
        # Check entities
        assert any(e.name == 'User' and e.type == 'class' for e in context.entities)
        assert any('auth.py' in e.name for e in context.entities if e.type == 'file')
        
        # Check keywords
        keywords_lower = [k.lower() for k in context.keywords]
        assert any('auth' in k for k in keywords_lower)
        assert any('jwt' in k or 'token' in k for k in keywords_lower)
        assert any('redis' in k for k in keywords_lower)
        
        # Check temporal context
        assert context.temporal_context is not None
        assert context.temporal_context.timeframe == 'yesterday'
        
        # Check scope
        assert context.scope['is_specific'] == True
        assert any('vendor' in e for e in context.scope['exclusions'])
        
    def test_url_parsing_edge_cases(self, parser):
        """Test URL parsing edge cases."""
        # No URL in prompt
        ref = parser._detect_external_reference("no url here")
        assert ref is None
        
        # Generic URL
        ref = parser._detect_external_reference("check https://example.com/page")
        assert ref is not None
        assert ref.type == 'url'
        assert ref.identifier == 'https://example.com/page'
        
        # GitLab URL
        ref = parser._detect_external_reference(
            "see https://gitlab.com/group/project/-/merge_requests/789"
        )
        assert ref is not None
        assert ref.type == 'gitlab'
        assert ref.metadata['number'] == '789'
        
    def test_error_handling_in_external_fetch(self, parser):
        """Test error handling when fetching external content."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Network error")
            
            ref = ExternalReference(
                type='url',
                url='https://example.com',
                identifier='test'
            )
            
            content = parser._fetch_external_content(ref)
            
            # Should handle error gracefully
            assert content is not None
            assert len(content) > 0