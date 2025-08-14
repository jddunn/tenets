"""Prompt parsing and understanding system.

This package provides intelligent prompt analysis using centralized NLP components
to extract intent, keywords, entities, and context from user queries. The parser
supports various input formats including plain text, URLs (GitHub issues, JIRA
tickets), and structured queries.

The prompt parser leverages the centralized NLP package for all text processing:
- Keyword extraction via nlp.keyword_extractor
- Tokenization via nlp.tokenizer
- Stopword filtering via nlp.stopwords

Main components:
- PromptParser: Main parser for analyzing prompts
- PromptContext: Structured context extracted from prompts
- IntentType: Types of user intent (implement, debug, test, etc.)
- ParsedEntity: Entities extracted from prompts
- TemporalContext: Time-based context from prompts
- ExternalReference: External URLs and references

Example usage:
    >>> from tenets.core.prompt import PromptParser
    >>> from tenets.config import TenetsConfig
    >>>
    >>> # Create parser with config
    >>> config = TenetsConfig()
    >>> parser = PromptParser(config)
    >>>
    >>> # Parse a prompt
    >>> context = parser.parse("implement OAuth2 authentication for the API")
    >>> print(f"Task type: {context.task_type}")
    >>> print(f"Keywords: {context.keywords}")
    >>> print(f"Intent: {context.intent}")
    >>>
    >>> # Parse from GitHub issue
    >>> context = parser.parse("https://github.com/org/repo/issues/123")
    >>> print(f"External source: {context.external_context['source']}")
"""

from typing import List, Optional, Dict, Any

# Import main classes
from .parser import (
    PromptParser,
    IntentType,
    ParsedEntity,
    TemporalContext,
    ExternalReference,
)

# Import from models (these are typically in tenets.models.context)
try:
    from tenets.models.context import PromptContext, TaskType
except ImportError:
    # Models might not be available in some test scenarios
    PromptContext = None
    TaskType = None

# Version info
__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Main parser
    "PromptParser",
    "create_parser",
    # Data classes
    "IntentType",
    "ParsedEntity", 
    "TemporalContext",
    "ExternalReference",
    # Context (from models)
    "PromptContext",
    "TaskType",
    # Utilities
    "parse_prompt",
    "extract_keywords",
    "detect_intent",
]


def create_parser(config=None) -> PromptParser:
    """Create a configured prompt parser.
    
    Convenience function to quickly create a parser with sensible defaults.
    Uses centralized NLP components for all text processing.
    
    Args:
        config: Optional TenetsConfig instance
        
    Returns:
        Configured PromptParser instance
        
    Example:
        >>> from tenets.core.prompt import create_parser
        >>> parser = create_parser()
        >>> context = parser.parse("add user authentication")
    """
    if config is None:
        from tenets.config import TenetsConfig
        config = TenetsConfig()
    
    return PromptParser(config)


def parse_prompt(prompt: str, config=None) -> Any:  # Returns PromptContext
    """Parse a prompt without managing parser instances.
    
    Convenience function for one-off prompt parsing. Uses centralized
    NLP components including keyword extraction and tokenization.
    
    Args:
        prompt: The prompt text or URL to parse
        config: Optional TenetsConfig instance
        
    Returns:
        PromptContext with extracted information
        
    Example:
        >>> from tenets.core.prompt import parse_prompt
        >>> context = parse_prompt("implement caching layer")
        >>> print(f"Keywords: {context.keywords}")
    """
    parser = create_parser(config)
    return parser.parse(prompt)


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    """Extract keywords from text using NLP components.
    
    Uses the centralized NLP keyword extractor with YAKE/TF-IDF/frequency
    fallback chain for robust keyword extraction.
    
    Args:
        text: Input text
        max_keywords: Maximum keywords to extract
        
    Returns:
        List of extracted keywords
        
    Example:
        >>> from tenets.core.prompt import extract_keywords
        >>> keywords = extract_keywords("implement OAuth2 authentication")
        >>> print(keywords)  # ['oauth2', 'authentication', 'implement']
    """
    from tenets.core.nlp.keyword_extractor import KeywordExtractor
    
    extractor = KeywordExtractor(
        use_stopwords=True,
        stopword_set='prompt'  # Use aggressive stopwords for prompts
    )
    
    return extractor.extract(text, max_keywords=max_keywords, include_scores=False)


def detect_intent(prompt: str) -> str:
    """Detect the primary intent from a prompt.
    
    Analyzes prompt text to determine user intent (implement, debug,
    understand, refactor, test, etc.).
    
    Args:
        prompt: The prompt text
        
    Returns:
        Intent type string
        
    Example:
        >>> from tenets.core.prompt import detect_intent
        >>> intent = detect_intent("fix the authentication bug")
        >>> print(intent)  # 'debug'
    """
    from tenets.config import TenetsConfig
    
    parser = PromptParser(TenetsConfig())
    context = parser.parse(prompt)
    return context.intent


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Extract named entities from text.
    
    Identifies classes, functions, files, modules, and other
    programming entities mentioned in the text.
    
    Args:
        text: Input text
        
    Returns:
        List of entity dictionaries with name, type, and confidence
        
    Example:
        >>> from tenets.core.prompt import extract_entities
        >>> entities = extract_entities("update the UserAuth class in auth.py")
        >>> for entity in entities:
        ...     print(f"{entity['type']}: {entity['name']}")
    """
    from tenets.config import TenetsConfig
    
    parser = PromptParser(TenetsConfig())
    # Use private method for entity extraction
    entities = parser._extract_entities(text)
    
    return [
        {
            "name": e.name,
            "type": e.type,
            "confidence": e.confidence,
            "context": e.context
        }
        for e in entities
    ]


def parse_external_reference(url: str) -> Optional[Dict[str, Any]]:
    """Parse an external reference URL.
    
    Extracts information from GitHub issues, JIRA tickets, GitLab MRs,
    and other external references.
    
    Args:
        url: URL to parse
        
    Returns:
        Dictionary with reference information or None
        
    Example:
        >>> from tenets.core.prompt import parse_external_reference
        >>> ref = parse_external_reference("https://github.com/org/repo/issues/123")
        >>> print(ref['type'])  # 'github'
        >>> print(ref['identifier'])  # 'org/repo#123'
    """
    from tenets.config import TenetsConfig
    
    parser = PromptParser(TenetsConfig())
    ref = parser._detect_external_reference(url)
    
    if ref:
        return {
            "type": ref.type,
            "url": ref.url,
            "identifier": ref.identifier,
            "metadata": ref.metadata
        }
    
    return None