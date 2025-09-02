"""Prompt parsing and understanding system.

This package provides intelligent prompt analysis to extract intent, keywords,
entities, temporal context, and external references from user queries. The parser
supports various input formats including plain text, URLs (GitHub issues, JIRA
tickets, Linear, Notion, etc.), and structured queries.

Core Features:
- Intent detection (implement, debug, test, refactor, etc.)
- Keyword extraction using multiple algorithms (YAKE, TF-IDF, frequency)
- Entity recognition (classes, functions, files, APIs, databases)
- Temporal parsing (dates, ranges, recurring patterns)
- External source integration (GitHub, GitLab, JIRA, Linear, Asana, Notion)
- Intelligent caching with TTL management
- Programming pattern recognition
- Scope and focus area detection

The parser leverages centralized NLP components for:
- Keyword extraction via nlp.keyword_extractor
- Tokenization via nlp.tokenizer
- Stopword filtering via nlp.stopwords
- Programming patterns via nlp.programming_patterns

Example:
    >>> from tenets.core.prompt import PromptParser
    >>> from tenets.config import TenetsConfig
    >>>
    >>> # Create parser with config
    >>> config = TenetsConfig()
    >>> parser = PromptParser(config)
    >>>
    >>> # Parse a prompt
    >>> context = parser.parse("implement OAuth2 authentication for the API")
    >>> print(f"Intent: {context.intent}")
    >>> print(f"Keywords: {context.keywords}")
    >>> print(f"Task type: {context.task_type}")
    >>>
    >>> # Parse from GitHub issue
    >>> context = parser.parse("https://github.com/org/repo/issues/123")
    >>> print(f"External source: {context.external_context['source']}")
    >>> print(f"Issue title: {context.text}")
"""

from typing import Any, Dict, List, Optional

# Import component classes
from tenets.utils.external_sources import (
    AsanaHandler,
    ExternalContent,
    ExternalSourceHandler,
    ExternalSourceManager,
    GitHubHandler,
    GitLabHandler,
    JiraHandler,
    LinearHandler,
    NotionHandler,
)

from .cache import (
    CacheEntry,
    PromptCache,
)
from .entity_recognizer import (
    Entity,
    EntityPatternMatcher,
    FuzzyEntityMatcher,
    HybridEntityRecognizer,
    NLPEntityRecognizer,
)
from .intent_detector import (
    HybridIntentDetector,
    Intent,
    PatternBasedDetector,
    SemanticIntentDetector,
)

# Import temporal parser classes
from .temporal_parser import (
    TemporalExpression,
    TemporalParser,
    TemporalPatternMatcher,
)

# Import main parser classes
from .parser import (
    PromptParser,
)

# Import from models
try:
    from tenets.models.context import PromptContext, TaskType
except ImportError:
    # Models might not be available in test scenarios
    PromptContext = None
    TaskType = None

# Version info
__version__ = "0.2.0"

# Public API exports
__all__ = [
    # Main parser
    "PromptParser",
    "create_parser",
    # Data classes
    # Context (from models)
    "PromptContext",
    "TaskType",
    # External sources
    "ExternalContent",
    "ExternalSourceHandler",
    "ExternalSourceManager",
    "GitHubHandler",
    "GitLabHandler",
    "JiraHandler",
    "LinearHandler",
    "AsanaHandler",
    "NotionHandler",
    # Entity recognition
    "Entity",
    "EntityPatternMatcher",
    "NLPEntityRecognizer",
    "FuzzyEntityMatcher",
    "HybridEntityRecognizer",
    # Temporal parsing
    "TemporalExpression",
    "TemporalPatternMatcher",
    "TemporalParser",
    # Intent detection
    "Intent",
    "PatternBasedDetector",
    "SemanticIntentDetector",
    "HybridIntentDetector",
    # Caching
    "CacheEntry",
    "PromptCache",
    # Convenience functions
    "parse_prompt",
    "extract_keywords",
    "detect_intent",
    "extract_entities",
    "parse_external_reference",
    "extract_temporal",
]


def create_parser(
    config=None, use_cache: bool = True, use_ml: bool = None, cache_manager=None
) -> PromptParser:
    """Create a configured prompt parser.

    Convenience function to quickly create a parser with sensible defaults.

    Args:
        config: Optional TenetsConfig instance (creates default if None)
        use_cache: Whether to enable caching (default: True)
        use_ml: Whether to use ML features (None = auto-detect from config)
        cache_manager: Optional cache manager for persistence

    Uses centralized NLP components for all text processing.

    Returns:
        Configured PromptParser instance

    Example:
        >>> parser = create_parser()
        >>> context = parser.parse("add user authentication")
        >>> print(context.intent)
    """
    if config is None:
        from tenets.config import TenetsConfig

        config = TenetsConfig()

    return PromptParser(config, cache_manager=cache_manager, use_cache=use_cache, use_ml=use_ml)


def parse_prompt(
    prompt: str, config=None, fetch_external: bool = True, use_cache: bool = False
) -> Any:  # Returns PromptContext
    """Parse a prompt without managing parser instances.

    Convenience function for one-off prompt parsing. Uses centralized
    NLP components including keyword extraction and tokenization.

    Args:
        prompt: The prompt text or URL to parse
        config: Optional TenetsConfig instance
        fetch_external: Whether to fetch external content (default: True)
        use_cache: Whether to use caching (default: False for one-off)

    Returns:
        PromptContext with extracted information

    Example:
        >>> context = parse_prompt("implement caching layer")
        >>> print(f"Keywords: {context.keywords}")
        >>> print(f"Intent: {context.intent}")
    """
    parser = create_parser(config, use_cache=use_cache)
    return parser.parse(prompt, fetch_external=fetch_external)


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    """Extract keywords from text using NLP components.

    Uses the centralized keyword extractor with YAKE/TF-IDF/frequency
    fallback chain for robust keyword extraction.

    Args:
        text: Input text to analyze
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of extracted keywords

    Example:
        >>> keywords = extract_keywords("implement OAuth2 authentication")
        >>> print(keywords)  # ['oauth2', 'authentication', 'implement']
    """
    from tenets.core.nlp.keyword_extractor import KeywordExtractor

    extractor = KeywordExtractor(
        use_stopwords=True,
        stopword_set="prompt",  # Use aggressive stopwords for prompts
    )

    return extractor.extract(text, max_keywords=max_keywords, include_scores=False)


def detect_intent(prompt: str, use_ml: bool = False) -> str:
    """Analyzes prompt text to determine user intent.

    Args:
        prompt: The prompt text to analyze
        use_ml: Whether to use ML-based detection (requires ML dependencies)

    Returns:
        Intent type string (implement, debug, understand, etc.)

    Example:
        >>> intent = detect_intent("fix the authentication bug")
        >>> print(intent)  # 'debug'
    """
    from tenets.config import TenetsConfig

    parser = PromptParser(TenetsConfig(), use_ml=use_ml, use_cache=False)
    context = parser.parse(prompt, use_cache=False)

    return context.intent


def extract_entities(
    text: str, min_confidence: float = 0.5, use_nlp: bool = False, use_fuzzy: bool = True
) -> List[Dict[str, Any]]:
    """Extract named entities from text.

    Identifies classes, functions, files, modules, and other
    programming entities mentioned in the text.

    Args:
        text: Input text to analyze
        min_confidence: Minimum confidence threshold
        use_nlp: Whether to use NLP-based NER (requires spaCy)
        use_fuzzy: Whether to use fuzzy matching

    Returns:
        List of entity dictionaries with name, type, and confidence

    Example:
        >>> entities = extract_entities("update the UserAuth class in auth.py")
        >>> for entity in entities:
        ...     print(f"{entity['type']}: {entity['name']}")
    """
    from .entity_recognizer import HybridEntityRecognizer

    recognizer = HybridEntityRecognizer(use_nlp=use_nlp, use_fuzzy=use_fuzzy)

    entities = recognizer.recognize(text, min_confidence=min_confidence)

    return [
        {"name": e.name, "type": e.type, "confidence": e.confidence, "context": e.context}
        for e in entities
    ]


def parse_external_reference(url: str) -> Optional[Dict[str, Any]]:
    """Parse an external reference URL.

    Extracts information from GitHub issues, JIRA tickets, GitLab MRs,
    Linear issues, Asana tasks, Notion pages, and other external references.

    Args:
        url: URL to parse

    Returns:
        Dictionary with reference information or None if not recognized

    Example:
        >>> ref = parse_external_reference("https://github.com/org/repo/issues/123")
        >>> print(ref['type'])  # 'github'
        >>> print(ref['identifier'])  # 'org/repo#123'
    """
    from tenets.utils.external_sources import ExternalSourceManager

    manager = ExternalSourceManager()
    result = manager.extract_reference(url)

    if result:
        url, identifier, metadata = result
        return {
            "type": metadata.get("platform", metadata.get("type", "unknown")),
            "url": url,
            "identifier": identifier,
            "metadata": metadata,
        }

    return None


def extract_temporal(text: str) -> List[Dict[str, Any]]:
    """Extract temporal expressions from text.

    Identifies dates, time ranges, relative dates, and recurring patterns.

    Args:
        text: Input text to analyze

    Returns:
        List of temporal expression dictionaries

    Example:
        >>> temporal = extract_temporal("changes from last week")
        >>> for expr in temporal:
        ...     print(f"{expr['text']}: {expr['type']}")
    """
    from .temporal_parser import TemporalParser

    parser = TemporalParser()
    expressions = parser.parse(text)

    return [
        {
            "text": e.text,
            "type": e.type,
            "start_date": e.start_date.isoformat() if e.start_date else None,
            "end_date": e.end_date.isoformat() if e.end_date else None,
            "is_relative": e.is_relative,
            "is_recurring": e.is_recurring,
            "confidence": e.confidence,
        }
        for e in expressions
    ]
