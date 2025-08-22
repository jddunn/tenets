"""Tests for documentation-specific keyword extraction in prompt parser."""

import pytest

from tenets.config import TenetsConfig
from tenets.core.prompt.parser import PromptParser


class TestDocumentationKeywordExtraction:
    """Test documentation-specific keyword extraction functionality."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return TenetsConfig()

    @pytest.fixture
    def parser(self, config):
        """Create a prompt parser instance."""
        return PromptParser(config, use_cache=False, use_ml=False, use_nlp_ner=False)

    def test_api_keyword_extraction(self, parser):
        """Test extraction of API-related keywords."""
        text = "How do I configure the REST API endpoint for user authentication?"
        keywords = parser._extract_documentation_keywords(text)
        
        assert "api" in keywords
        assert "endpoint" in keywords
        assert "authentication" in keywords
        assert "auth" in keywords

    def test_configuration_keyword_extraction(self, parser):
        """Test extraction of configuration-related keywords."""
        text = "Update the database configuration settings in the config file"
        keywords = parser._extract_documentation_keywords(text)
        
        assert "database" in keywords
        assert "configuration" in keywords
        assert "config" in keywords
        assert "setting" in keywords

    def test_setup_keyword_extraction(self, parser):
        """Test extraction of setup and installation keywords."""
        text = "Installation guide for Docker deployment setup requirements"
        keywords = parser._extract_documentation_keywords(text)
        
        assert "installation" in keywords
        assert "docker" in keywords
        assert "deployment" in keywords
        assert "setup" in keywords
        assert "requirement" in keywords

    def test_documentation_structure_keywords(self, parser):
        """Test extraction of documentation structure keywords."""
        text = "Create a tutorial guide with examples and troubleshooting FAQ"
        keywords = parser._extract_documentation_keywords(text)
        
        assert "tutorial" in keywords
        assert "guide" in keywords
        assert "example" in keywords
        assert "troubleshoot" in keywords
        assert "faq" in keywords

    def test_technology_keyword_extraction(self, parser):
        """Test extraction of technology and tool names."""
        text = "Configure React app with TypeScript using npm and docker"
        keywords = parser._extract_technology_keywords(text)
        
        assert "react" in keywords
        assert "typescript" in keywords
        assert "npm" in keywords
        assert "docker" in keywords

    def test_format_keyword_extraction(self, parser):
        """Test extraction of file format keywords."""
        text = "Parse JSON data from YAML config files and markdown documentation"
        keywords = parser._extract_format_keywords(text)
        
        assert "json" in keywords
        assert "yaml" in keywords
        assert "markdown" in keywords

    def test_format_with_extensions(self, parser):
        """Test extraction of file extensions."""
        text = "Update the config.json and settings.yaml files"
        keywords = parser._extract_format_keywords(text)
        
        assert "json" in keywords
        assert "yaml" in keywords

    def test_mixed_keyword_extraction(self, parser):
        """Test extraction from text with multiple keyword types."""
        text = """
        Set up the REST API authentication using JWT tokens in the configuration.
        Deploy with Docker and test the endpoints with Python requests.
        Document the installation process in markdown format.
        """
        keywords = parser._extract_documentation_keywords(text)
        
        # Should extract from all categories
        api_keywords = {"api", "authentication", "auth", "jwt", "token", "endpoint"}
        config_keywords = {"configuration", "config"}
        tech_keywords = {"docker", "python"}
        format_keywords = {"markdown"}
        
        found_api = sum(1 for k in api_keywords if k in keywords)
        found_config = sum(1 for k in config_keywords if k in keywords)
        found_tech = sum(1 for k in tech_keywords if k in keywords)
        found_format = sum(1 for k in format_keywords if k in keywords)
        
        assert found_api > 0
        assert found_config > 0
        assert found_tech > 0
        assert found_format > 0

    def test_keyword_deduplication(self, parser):
        """Test that duplicate keywords are removed."""
        text = "API endpoint configuration for API authentication and endpoint security"
        keywords = parser._extract_documentation_keywords(text)
        
        # Should only have one instance of each keyword
        assert keywords.count("api") <= 1
        assert keywords.count("endpoint") <= 1
        assert keywords.count("configuration") <= 1

    def test_keyword_length_limit(self, parser):
        """Test that keywords are limited to reasonable length."""
        text = """
        Configure the REST API authentication endpoint with JWT tokens.
        Set up database configuration for PostgreSQL connection.
        Deploy using Docker containers with Kubernetes orchestration.
        Document installation process with troubleshooting guide.
        Implement monitoring with logging and analytics tracking.
        """
        keywords = parser._extract_documentation_keywords(text)
        
        # Should be limited to 15 keywords as specified in the method
        assert len(keywords) <= 15

    def test_empty_and_short_keyword_filtering(self, parser):
        """Test that empty and very short keywords are filtered out."""
        text = "a of in API the configuration to and setup"
        keywords = parser._extract_documentation_keywords(text)
        
        # Should filter out stopwords and very short words
        assert "a" not in keywords
        assert "of" not in keywords
        assert "in" not in keywords
        assert "the" not in keywords
        assert "to" not in keywords
        assert "and" not in keywords
        
        # Should keep meaningful keywords
        assert "api" in keywords
        assert "configuration" in keywords
        assert "setup" in keywords

    def test_case_preservation(self, parser):
        """Test that original case is preserved for keywords."""
        text = "Configure the API endpoint for user Authentication"
        keywords = parser._extract_documentation_keywords(text)
        
        # Should preserve original case when possible
        assert any(k in ["api", "API"] for k in keywords)
        assert any(k in ["authentication", "Authentication"] for k in keywords)

    def test_programming_concept_extraction(self, parser):
        """Test extraction of programming concepts in documentation."""
        text = "Document the function interface and class module dependencies"
        keywords = parser._extract_documentation_keywords(text)
        
        assert "function" in keywords
        assert "interface" in keywords
        assert "class" in keywords
        assert "module" in keywords
        assert "dependency" in keywords

    def test_usage_keyword_extraction(self, parser):
        """Test extraction of usage and operational keywords."""
        text = "Command line usage examples with logging and performance optimization"
        keywords = parser._extract_documentation_keywords(text)
        
        assert "command" in keywords
        assert "usage" in keywords
        assert "example" in keywords
        assert "logging" in keywords
        assert "performance" in keywords
        assert "optimization" in keywords

    def test_integration_with_main_parsing(self, parser):
        """Test that documentation keywords are integrated into main parsing."""
        prompt = "How to configure API authentication with JWT tokens in the documentation"
        
        context = parser.parse(prompt)
        
        # Should include both regular keywords and documentation-specific ones
        keywords = context.keywords
        
        # Should have documentation-specific keywords
        doc_keywords = {"api", "authentication", "auth", "jwt", "token", "configuration", "config"}
        found_doc_keywords = sum(1 for k in doc_keywords if k in keywords)
        
        assert found_doc_keywords > 0

    def test_no_keywords_in_text(self, parser):
        """Test behavior when no documentation keywords are found."""
        text = "random text without any specific technical terms"
        keywords = parser._extract_documentation_keywords(text)
        
        # Should return empty list or very few keywords
        assert len(keywords) <= 2  # Might catch "text" or similar generic terms