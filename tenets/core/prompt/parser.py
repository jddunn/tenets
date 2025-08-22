"""Prompt parsing and understanding system with modular components.

This module analyzes user prompts to extract intent, keywords, entities,
temporal context, and external references using a comprehensive set of
specialized components and NLP techniques.
"""

import re
from typing import Any, Dict, List, Optional, Set

from tenets.config import TenetsConfig

# Import centralized NLP components
from tenets.core.nlp.keyword_extractor import KeywordExtractor
from tenets.core.nlp.programming_patterns import get_programming_patterns
from tenets.core.nlp.stopwords import StopwordManager
from tenets.core.nlp.tokenizer import CodeTokenizer, TextTokenizer
from tenets.core.prompt.cache import PromptCache
from tenets.core.prompt.entity_recognizer import Entity, HybridEntityRecognizer
from tenets.core.prompt.intent_detector import HybridIntentDetector
from tenets.core.prompt.temporal_parser import TemporalParser
from tenets.models.context import PromptContext

# Import modular components
from tenets.utils.external_sources import ExternalSourceManager
from tenets.utils.logger import get_logger

# Optional storage support
try:
    from tenets.storage.cache import CacheManager

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = None


class PromptParser:
    """Comprehensive prompt parser with modular components and caching.

    This parser provides advanced prompt analysis using:
    - External source fetching (GitHub, GitLab, JIRA, Linear, etc.)
    - Hybrid entity recognition (regex + NLP + fuzzy matching)
    - Advanced temporal parsing (dates, ranges, recurring patterns)
    - ML-enhanced intent detection (pattern + semantic similarity)
    - Intelligent caching with TTL management

    The parser integrates with centralized NLP components for:
    - Keyword extraction (YAKE, TF-IDF, frequency-based)
    - Tokenization (text and code aware)
    - Stopword filtering (context-specific sets)
    - Programming pattern recognition

    Attributes:
        config: TenetsConfig instance for configuration
        logger: Logger instance for debugging
        cache: PromptCache for caching results (optional)
        external_manager: Manager for external source fetching
        entity_recognizer: Hybrid entity recognition system
        temporal_parser: Temporal expression parser
        intent_detector: Hybrid intent detection system
        keyword_extractor: Centralized keyword extraction
        tokenizer: Text tokenizer for general text
        code_tokenizer: Code-aware tokenizer
        stopword_manager: Stopword management system
        programming_patterns: Programming pattern recognition

    Example:
        >>> from tenets.config import TenetsConfig
        >>> from tenets.core.prompt import PromptParser
        >>>
        >>> config = TenetsConfig()
        >>> parser = PromptParser(config)
        >>>
        >>> # Parse a simple prompt
        >>> context = parser.parse("implement OAuth2 authentication")
        >>> print(f"Intent: {context.intent}")
        >>> print(f"Keywords: {context.keywords}")
        >>>
        >>> # Parse with external reference
        >>> context = parser.parse("https://github.com/org/repo/issues/123")
        >>> print(f"External source: {context.external_context['source']}")
    """

    def __init__(
        self,
        config: TenetsConfig,
        cache_manager: Optional[Any] = None,
        use_cache: bool = True,
        use_ml: bool = None,
        use_nlp_ner: bool = None,
        use_fuzzy_matching: bool = True,
    ):
        """Initialize the prompt parser with configurable components.

        Args:
            config: Tenets configuration object
            cache_manager: Optional CacheManager instance for persistence
            use_cache: Whether to enable caching (default: True)
            use_ml: Whether to use ML features for intent detection
                   (None = auto-detect from config.nlp.embeddings_enabled)
            use_nlp_ner: Whether to use NLP-based named entity recognition
                        (None = auto-detect from config.nlp.enabled)
            use_fuzzy_matching: Whether to use fuzzy entity matching (default: True)

        Note:
            ML features require additional dependencies (transformers, torch).
            NLP NER requires spaCy with a language model installed.
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Determine feature flags from config if not specified
        if use_ml is None:
            use_ml = config.nlp.embeddings_enabled
        if use_nlp_ner is None:
            use_nlp_ner = config.nlp.enabled

        # Initialize optional cache
        self.cache = None
        if use_cache:
            self.cache = PromptCache(
                cache_manager=cache_manager,
                enable_memory_cache=True,
                enable_disk_cache=cache_manager is not None,
                memory_cache_size=100,
            )

        # Initialize all components
        self._init_components(
            cache_manager=cache_manager,
            use_ml=use_ml,
            use_nlp_ner=use_nlp_ner,
            use_fuzzy_matching=use_fuzzy_matching,
        )

        # Initialize extraction patterns
        self._init_patterns()

        self.logger.info(
            f"PromptParser initialized (cache={use_cache}, ml={use_ml}, "
            f"nlp_ner={use_nlp_ner}, fuzzy={use_fuzzy_matching})"
        )

    def _init_components(
        self,
        cache_manager: Optional[Any],
        use_ml: bool,
        use_nlp_ner: bool,
        use_fuzzy_matching: bool,
    ):
        """Initialize all parser components.

        Args:
            cache_manager: Optional cache manager for external sources
            use_ml: Whether to enable ML features
            use_nlp_ner: Whether to enable NLP NER
            use_fuzzy_matching: Whether to enable fuzzy matching
        """
        # External source manager for GitHub, JIRA, etc.
        self.external_manager = ExternalSourceManager(cache_manager)

        # Entity recognizer with configurable features
        self.entity_recognizer = HybridEntityRecognizer(
            use_nlp=use_nlp_ner,
            use_fuzzy=use_fuzzy_matching,
            patterns_file=None,  # Use default patterns
            spacy_model="en_core_web_sm",
        )

        # Temporal parser for dates and time expressions
        self.temporal_parser = TemporalParser(patterns_file=None)

        # Intent detector with optional ML
        self.intent_detector = HybridIntentDetector(
            use_ml=use_ml,
            patterns_file=None,
            model_name=self.config.nlp.embeddings_model,
        )

        # NLP components from centralized package
        self.keyword_extractor = KeywordExtractor(
            use_yake=self.config.nlp.keyword_extraction_method in ["auto", "yake"],
            language="en",
            use_stopwords=self.config.nlp.stopwords_enabled,
            stopword_set="prompt",  # Use prompt-specific stopwords
        )

        self.tokenizer = TextTokenizer(use_stopwords=True)
        self.code_tokenizer = CodeTokenizer(use_stopwords=False)
        self.stopword_manager = StopwordManager()

        # Programming patterns for code-specific extraction
        self.programming_patterns = get_programming_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for scope and file extraction."""
        # File pattern indicators
        self._file_pattern_indicators = [
            r"\*\.\w+",  # *.py, *.js
            r"test_\*",  # test_*
            r"\*_test",  # *_test
            r"\.\./",  # ../
            r"\./",  # ./
            r"~/",  # ~/
        ]

    def parse(
        self,
        prompt: str,
        use_cache: bool = True,
        fetch_external: bool = True,
        min_entity_confidence: float = 0.5,
        min_intent_confidence: float = 0.3,
    ) -> PromptContext:
        """Parse a prompt to extract structured context.

        This method performs comprehensive analysis including:
        - External reference detection and fetching
        - Intent detection with confidence scoring
        - Keyword extraction using multiple algorithms
        - Entity recognition (classes, functions, files, etc.)
        - Temporal expression parsing
        - File pattern extraction
        - Focus area identification
        - Scope analysis

        Args:
            prompt: The user's prompt text or URL to parse
            use_cache: Whether to use cached results if available (default: True)
            fetch_external: Whether to fetch content from external URLs (default: True)
            min_entity_confidence: Minimum confidence threshold for entity recognition (default: 0.5)
            min_intent_confidence: Minimum confidence threshold for intent detection (default: 0.3)

        Returns:
            PromptContext: Structured context with all extracted information

        Example:
            >>> context = parser.parse(
            ...     "fix the authentication bug in UserController",
            ...     min_entity_confidence=0.7
            ... )
            >>> print(context.intent)  # 'debug'
            >>> print(context.entities)  # [{'name': 'UserController', 'type': 'class', ...}]
        """
        # Validate prompt early; tests expect AttributeError for None
        if prompt is None:
            raise AttributeError("prompt must be a string, not None")
        self.logger.debug(f"Parsing prompt: {str(prompt)[:100]}...")

        # Check cache first if enabled
        if use_cache and self.cache:
            cached = self.cache.get_parsed_prompt(prompt)
            if cached:
                self.logger.info("Using cached prompt parsing result")
                return cached

        # Perform actual parsing
        result = self._parse_internal(
            prompt,
            fetch_external,
            min_entity_confidence,
            min_intent_confidence,
        )

        # Cache the result if caching is enabled
        if use_cache and self.cache:
            avg_confidence = result.metadata.get("avg_confidence", 0.7)
            self.cache.cache_parsed_prompt(prompt, result, metadata={"confidence": avg_confidence})

        return result

    def _parse_internal(
        self,
        prompt: str,
        fetch_external: bool,
        min_entity_confidence: float,
        min_intent_confidence: float,
    ) -> PromptContext:
        """Internal parsing logic.

        Args:
            prompt: Prompt text to parse
            fetch_external: Whether to fetch external content
            min_entity_confidence: Minimum entity confidence
            min_intent_confidence: Minimum intent confidence

        Returns:
            PromptContext with extracted information
        """
        # 1. Check for external references (GitHub, JIRA, etc.)
        external_content = None
        external_context = None

        external_ref = self.external_manager.extract_reference(prompt)
        if external_ref:
            url, identifier, metadata = external_ref

            if fetch_external:
                # Try to fetch content with caching
                if self.cache:
                    external_content = self.cache.get_external_content(url)

                if not external_content:
                    external_content = self.external_manager.process_url(url)
                    if external_content and self.cache:
                        self.cache.cache_external_content(url, external_content, metadata=metadata)

            external_context = {
                "source": metadata.get("platform", "unknown"),
                "url": url,
                "identifier": identifier,
                "metadata": metadata,
            }

            # Use fetched content if available
            if external_content:
                prompt_text = f"{external_content.title}\n{external_content.body}"
            else:
                prompt_text = prompt
        else:
            prompt_text = prompt

        # 2. Detect intent (implement, debug, understand, etc.)
        intent_result = None
        if self.cache:
            intent_result = self.cache.get_intent(prompt_text)

        if not intent_result:
            intent_result = self.intent_detector.detect(
                prompt_text, min_confidence=min_intent_confidence
            )
            if self.cache and intent_result:
                self.cache.cache_intent(
                    prompt_text, intent_result, confidence=intent_result.confidence
                )

        # Prefer detector result, but add light heuristics to disambiguate
        intent = intent_result.type if intent_result else "understand"
        lower_text = prompt_text.lower()
        # Ensure prompts asking to explain/what/show/understand map to 'understand'
        if any(kw in lower_text for kw in ["explain", "what does", "show me", "understand"]):
            intent = "understand"
        # Ensure performance optimization phrasing maps to 'optimize' (not 'refactor')
        if (
            "optimize" in lower_text
            or ("improve" in lower_text and "performance" in lower_text)
            or ("reduce" in lower_text and "memory" in lower_text)
            or ("make" in lower_text and "faster" in lower_text)
        ):
            intent = "optimize"
        task_type = self._intent_to_task_type(intent)

        # 3. Extract keywords using multiple algorithms
        keywords = self.keyword_extractor.extract(
            prompt_text,
            max_keywords=self.config.nlp.max_keywords,
        )

        # Add programming-specific keywords
        prog_keywords = self.programming_patterns.extract_programming_keywords(prompt_text)
        keywords = list(set(keywords + prog_keywords))[: self.config.nlp.max_keywords]

        # Add documentation-specific keywords for better context-aware summarization
        doc_keywords = self._extract_documentation_keywords(prompt_text)
        keywords = list(set(keywords + doc_keywords))[: self.config.nlp.max_keywords]

        # 4. Recognize entities (classes, functions, files, etc.)
        entities_list = None
        if self.cache:
            entities_list = self.cache.get_entities(prompt_text)

        if not entities_list:
            entities_list = self.entity_recognizer.recognize(
                prompt_text, merge_overlapping=True, min_confidence=min_entity_confidence
            )
            if self.cache and entities_list:
                avg_confidence = (
                    sum(e.confidence for e in entities_list) / len(entities_list)
                    if entities_list
                    else 0
                )
                self.cache.cache_entities(prompt_text, entities_list, confidence=avg_confidence)

        # Convert entities to expected format
        entities = []
        for entity in entities_list:
            entities.append(
                {
                    "name": entity.name,
                    "type": entity.type,
                    "confidence": entity.confidence,
                    "context": entity.context,
                }
            )

        # 5. Parse temporal expressions (dates, time ranges, etc.)
        temporal_expressions = self.temporal_parser.parse(prompt_text)
        temporal_context = None

        if temporal_expressions:
            # Get overall temporal context
            temporal_info = self.temporal_parser.get_temporal_context(temporal_expressions)

            # Convert to expected format
            first_expr = temporal_expressions[0]
            temporal_context = {
                "timeframe": temporal_info.get("timeframe"),
                "since": first_expr.start_date,
                "until": first_expr.end_date,
                "is_relative": first_expr.is_relative,
                "is_recurring": any(e.is_recurring for e in temporal_expressions),
                "expressions": len(temporal_expressions),
            }

        # 6. Extract file patterns
        file_patterns = self._extract_file_patterns(prompt)

        # 7. Extract focus areas based on content
        focus_areas = self._extract_focus_areas(prompt_text, entities_list)

        # 8. Extract scope information
        scope = self._extract_scope(prompt)

        # Build comprehensive metadata
        metadata = {
            "intent_confidence": intent_result.confidence if intent_result else 0,
            "entity_count": len(entities),
            "temporal_expressions": len(temporal_expressions) if temporal_expressions else 0,
            "has_external_ref": external_ref is not None,
            "cached": False,  # This result is fresh
        }

        if intent_result:
            metadata["intent_evidence"] = intent_result.evidence[:3]
            metadata["intent_source"] = intent_result.source

        # Calculate average confidence across all components
        confidences = []
        if intent_result:
            confidences.append(intent_result.confidence)
        if entities_list:
            confidences.extend([e.confidence for e in entities_list])
        metadata["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.5

        # Determine if tests should be included based on intent and content
        include_tests = self._should_include_tests(intent, prompt_text, keywords)

        # Build final prompt context
        context = PromptContext(
            text=prompt_text,
            original=prompt,
            keywords=keywords,
            task_type=task_type,
            intent=intent,
            entities=entities,
            file_patterns=file_patterns,
            focus_areas=focus_areas,
            temporal_context=temporal_context,
            scope=scope,
            external_context=external_context,
            metadata=metadata,
            include_tests=include_tests,
        )

        self.logger.info(
            f"Parsing complete: task={task_type}, intent={intent}, "
            f"keywords={len(keywords)}, entities={len(entities)}, "
            f"temporal={temporal_context is not None}, external={external_context is not None}"
        )

        return context

    def _intent_to_task_type(self, intent: str) -> str:
        """Convert intent to task type.

        Args:
            intent: Intent string

        Returns:
            Corresponding task type
        """
        intent_to_task = {
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
        return intent_to_task.get(intent, "general")

    def _should_include_tests(self, intent: str, prompt_text: str, keywords: List[str]) -> bool:
        """Determine if test files should be included based on prompt analysis.

        Tests are included if:
        1. Intent is explicitly 'test'
        2. Prompt contains test-related keywords
        3. Prompt mentions writing, modifying, or checking tests
        4. Prompt asks about test coverage or test failures

        Args:
            intent: Detected intent (test, implement, debug, etc.)
            prompt_text: The full prompt text
            keywords: Extracted keywords

        Returns:
            True if test files should be included, False otherwise
        """
        # Explicit test intent - always include tests
        if intent == "test":
            return True

        # Check for test-related keywords in extracted keywords
        test_keywords = {
            "test",
            "tests",
            "testing",
            "unit",
            "integration",
            "e2e",
            "end-to-end",
            "spec",
            "specs",
            "coverage",
            "jest",
            "pytest",
            "mocha",
            "jasmine",
            "junit",
            "testng",
            "rspec",
            "tdd",
            "bdd",
            "assertion",
            "mock",
            "stub",
        }

        if any(kw.lower() in test_keywords for kw in keywords):
            return True

        # Check for test-related patterns in prompt text
        lower_prompt = prompt_text.lower()

        # Test file patterns
        test_file_patterns = [
            r"\btest_\w+\.py\b",  # test_auth.py
            r"\w+_test\.py\b",  # auth_test.py
            r"\b\w+\.test\.\w+\b",  # auth.test.js
            r"\b\w+\.spec\.\w+\b",  # auth.spec.js
            r"\btests?/\w+",  # tests/auth
            r"\b__tests__/\w+",  # __tests__/auth
        ]

        if any(re.search(pattern, lower_prompt) for pattern in test_file_patterns):
            return True

        # Test action patterns - looking for explicit test-related actions
        test_action_patterns = [
            r"\b(?:write|add|create|implement|build)\s+(?:unit\s+)?tests?\b",
            r"\b(?:test|testing)\s+(?:the|this|that|coverage)\b",
            r"\b(?:fix|debug|update|modify|check|review)\s+(?:the\s+)?tests?\b",
            r"\b(?:test|check)\s+(?:coverage|failures?|errors?)\b",
            r"\b(?:run|execute)\s+(?:the\s+)?tests?\b",
            r"\bmock\s+(?:the|this|that)\b",
            r"\bunit\s+test\b",
            r"\bintegration\s+test\b",
            r"\be2e\s+test\b",
            r"\bend-to-end\s+test\b",
            r"\btest\s+suite\b",
            r"\btest\s+cases?\b",
            r"\bassertions?\b.*\b(?:fail|pass|error)\b",
        ]

        if any(re.search(pattern, lower_prompt) for pattern in test_action_patterns):
            return True

        # Test quality/coverage patterns
        test_quality_patterns = [
            r"\btest\s+coverage\b",
            r"\bcoverage\s+report\b",
            r"\bfailing\s+tests?\b",
            r"\btest\s+failures?\b",
            r"\bbroken\s+tests?\b",
            r"\btest\s+(?:pass|fail)\b",
        ]

        if any(re.search(pattern, lower_prompt) for pattern in test_quality_patterns):
            return True

        # Default: exclude tests for non-test-related prompts
        return False

    def _extract_file_patterns(self, text: str) -> List[str]:
        """Extract file patterns from text.

        Args:
            text: Text to analyze

        Returns:
            List of file patterns found
        """
        patterns = []

        # Look for explicit file patterns
        for indicator in self._file_pattern_indicators:
            matches = re.findall(indicator, text)
            patterns.extend(matches)

        # Look for file extensions
        ext_pattern = r"\*?\.\w{2,4}\b"
        extensions = re.findall(ext_pattern, text)
        patterns.extend(extensions)

        # Look for specific file mentions (both 'file X' and 'X file')
        file_mentions = re.findall(r"\b(?:file|in|from)\s+([a-zA-Z0-9_\-/]+\.\w{2,4})", text)
        trailing_file_mentions = re.findall(r"\b([a-zA-Z0-9_\-/]+\.\w{2,4})\s+file\b", text)
        file_mentions.extend(trailing_file_mentions)
        patterns.extend(file_mentions)

        # Also capture standalone filenames like 'config.json' even without nearby qualifiers
        standalone_files = re.findall(
            r"\b([a-zA-Z0-9_\-./]+\.(?:json|ya?ml|toml|ini|conf|cfg|txt|md|py|js|ts|tsx|jsx|java|rb|go|rs|php|c|cpp|h|hpp))\b",
            text,
        )
        patterns.extend(standalone_files)

        return list(set(patterns))  # Deduplicate

    def _extract_focus_areas(self, text: str, entities: List[Entity]) -> List[str]:
        """Extract focus areas from text and entities.

        Args:
            text: Text to analyze
            entities: Recognized entities

        Returns:
            List of focus areas identified
        """
        focus_areas = set()
        text_lower = text.lower()

        # Use programming patterns to identify focus areas
        pattern_categories = self.programming_patterns.get_pattern_categories()

        for category in pattern_categories:
            keywords = self.programming_patterns.get_category_keywords(category)
            # Check if any category keywords appear in text
            if any(kw.lower() in text_lower for kw in keywords):
                focus_areas.add(category)

        # Add areas based on entity types
        if entities:
            entity_types = set(e.type for e in entities)
            if "api_endpoint" in entity_types:
                focus_areas.add("api")
            if "database" in entity_types:
                focus_areas.add("database")
            if "class" in entity_types or "module" in entity_types:
                focus_areas.add("architecture")
            if "error" in entity_types:
                focus_areas.add("error_handling")
            if "component" in entity_types:
                focus_areas.add("ui")

        return list(focus_areas)

    def _extract_scope(self, text: str) -> Dict[str, Any]:
        """Extract scope indicators from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with scope information
        """
        scope = {
            "modules": [],
            "directories": [],
            "specific_files": [],
            "exclusions": [],
            "is_global": False,
            "is_specific": False,
        }

        # Module/package references
        module_patterns = [
            r"\b(?:in|for|of)\s+(?:the\s+)?([a-z][a-z0-9_]*)\s+(?:module|package|component)\b",
            r"\b(?:the\s+)?([a-z][a-z0-9_]*(?:ication|ization)?)\s+(?:module|package|component)\b",
        ]

        modules: Set[str] = set()
        for pat in module_patterns:
            for m in re.findall(pat, text, re.IGNORECASE):
                modules.add(m)
        scope["modules"] = list(modules)

        # Directory references
        dir_patterns = [
            r"(?:in|under|within)\s+(?:the\s+)?([a-zA-Z0-9_\-./]+)(?:\s+directory)?",
            r"\b([a-zA-Z0-9_\-./]+/[a-zA-Z0-9_\-./]*)\b",
        ]

        directories = set()
        for pattern in dir_patterns:
            for match in re.findall(pattern, text):
                # Filter out URLs and other false positives
                if not match.startswith("http") and "/" in match:
                    directories.add(match)
        scope["directories"] = list(directories)

        # Specific file references
        file_pattern = r"\b([a-zA-Z0-9_\-/]+\.(?:py|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|rb|php))\b"
        files = re.findall(file_pattern, text)
        scope["specific_files"] = list(set(files))

        # Exclusion patterns (capture common directories like node_modules/vendor as well)
        exclude_pattern = (
            r"(?:except|exclude|not|ignore)\s+(?:anything\s+in\s+)?([a-zA-Z0-9_\-/*]+/?)"
        )
        exclusions = set(re.findall(exclude_pattern, text, re.IGNORECASE))
        # Add explicit common exclusions if mentioned anywhere
        for common in ["node_modules", "vendor"]:
            if re.search(rf"\b{common}\b", text, re.IGNORECASE):
                exclusions.add(common)
        scope["exclusions"] = list(exclusions)

        # Determine scope type
        if any(
            word in text.lower() for word in ["entire", "whole", "all", "everything", "project"]
        ):
            scope["is_global"] = True
        elif scope["modules"] or scope["directories"] or scope["specific_files"]:
            scope["is_specific"] = True

        return scope

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics or None if cache is disabled

        Example:
            >>> stats = parser.get_cache_stats()
            >>> if stats:
            ...     print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        """
        if self.cache:
            return self.cache.get_stats()
        return None

    def clear_cache(self) -> None:
        """Clear all cached data.

        This removes all cached parsing results, external content,
        entities, and intents from both memory and disk cache.

        Example:
            >>> parser.clear_cache()
            >>> print("Cache cleared")
        """
        if self.cache:
            self.cache.clear_all()
            self.logger.info("Cleared prompt parser cache")

    def warm_cache(self, common_prompts: List[str]) -> None:
        """Pre-warm cache with common prompts.

        This method pre-parses a list of common prompts to populate
        the cache, improving performance for frequently used queries.

        Args:
            common_prompts: List of common prompts to pre-parse

        Example:
            >>> common = [
            ...     "implement authentication",
            ...     "fix bug",
            ...     "understand architecture"
            ... ]
            >>> parser.warm_cache(common)
        """
        if not self.cache:
            return

        self.logger.info(f"Pre-warming cache with {len(common_prompts)} prompts")

        for prompt in common_prompts:
            # Parse without using cache to generate fresh results
            # Use positional args to match tests that assert on call args
            _ = self._parse_internal(
                prompt,
                False,  # fetch_external
                0.5,  # min_entity_confidence
                0.3,  # min_intent_confidence
            )

        self.logger.info("Cache pre-warming complete")

    def _extract_documentation_keywords(self, text: str) -> List[str]:
        """Extract documentation-specific keywords for context-aware summarization.

        This method identifies keywords that are particularly relevant for documentation
        files, including API endpoints, configuration parameters, installation steps,
        and other documentation-specific concepts.

        Args:
            text: The prompt text to analyze

        Returns:
            List of documentation-specific keywords
        """
        doc_keywords = []
        text_lower = text.lower()

        # API and endpoint related keywords
        api_patterns = [
            r"\b(api|endpoint|route|url|uri|path)\b",
            r"\b(get|post|put|delete|patch|head|options)\b",
            r"\b(request|response|payload|parameter|header|body)\b",
            r"\b(authentication|auth|token|key|secret|oauth|jwt)\b",
            r"\b(rate.?limit|quota|throttle)\b",
            r"\b(webhook|callback|event|notification)\b",
        ]

        # Configuration and setup keywords
        config_patterns = [
            r"\b(config|configuration|setting|option|parameter|env|environment)\b",
            r"\b(install|installation|setup|deployment|deploy)\b",
            r"\b(requirement|dependency|prerequisite|version)\b",
            r"\b(database|db|connection|credential)\b",
            r"\b(server|host|port|domain|certificate|ssl|tls)\b",
            r"\b(docker|container|image|volume|network)\b",
        ]

        # Documentation structure keywords
        structure_patterns = [
            r"\b(tutorial|guide|walkthrough|example|demo)\b",
            r"\b(getting.?started|quick.?start|introduction|overview)\b",
            r"\b(troubleshoot|faq|help|support|issue|problem)\b",
            r"\b(changelog|release.?note|migration|upgrade)\b",
            r"\b(readme|documentation|doc|manual)\b",
        ]

        # Programming concepts in documentation
        programming_patterns = [
            r"\b(function|method|class|interface|module|package)\b",
            r"\b(variable|constant|property|attribute|field)\b",
            r"\b(import|include|require|export|dependency)\b",
            r"\b(library|framework|sdk|plugin|extension)\b",
            r"\b(debug|test|unit.?test|integration.?test)\b",
        ]

        # Usage and operational keywords
        usage_patterns = [
            r"\b(usage|how.?to|example|snippet|sample)\b",
            r"\b(command|cli|script|tool|utility)\b",
            r"\b(log|logging|monitor|metric|analytics)\b",
            r"\b(backup|restore|migration|sync)\b",
            r"\b(performance|optimization|cache|memory)\b",
        ]

        # Extract matches from all pattern groups
        all_patterns = (
            api_patterns
            + config_patterns
            + structure_patterns
            + programming_patterns
            + usage_patterns
        )

        for pattern in all_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle groups in regex
                    doc_keywords.extend([m for m in match if m])
                else:
                    doc_keywords.append(match)

        # Add specific technology and tool names commonly found in documentation
        tech_keywords = self._extract_technology_keywords(text)
        doc_keywords.extend(tech_keywords)

        # Add file extension and format keywords
        format_keywords = self._extract_format_keywords(text)
        doc_keywords.extend(format_keywords)

        # Remove duplicates and common stopwords, keep original case for some keywords
        unique_keywords = []
        seen = set()

        for keyword in doc_keywords:
            keyword_clean = keyword.strip().lower()
            if (
                keyword_clean
                and len(keyword_clean) > 2
                and keyword_clean not in seen
                and not keyword_clean.isdigit()
                and keyword_clean not in {"the", "and", "but", "for", "are", "with", "this", "that"}
            ):
                unique_keywords.append(keyword.strip())
                seen.add(keyword_clean)

        return unique_keywords[:15]  # Limit to top 15 documentation keywords

    def _extract_technology_keywords(self, text: str) -> List[str]:
        """Extract technology and tool names from text."""
        tech_keywords = []

        # Common technologies mentioned in documentation
        technologies = [
            # Programming languages
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c++",
            "c#",
            "php",
            "ruby",
            "swift",
            "kotlin",
            "scala",
            "clojure",
            "elixir",
            "dart",
            # Web frameworks
            "react",
            "vue",
            "angular",
            "svelte",
            "next.js",
            "nuxt",
            "express",
            "django",
            "flask",
            "fastapi",
            "spring",
            "rails",
            "laravel",
            "gin",
            # Databases
            "postgresql",
            "mysql",
            "mongodb",
            "redis",
            "elasticsearch",
            "sqlite",
            "cassandra",
            "dynamodb",
            "firebase",
            "supabase",
            # Cloud platforms
            "aws",
            "azure",
            "gcp",
            "heroku",
            "vercel",
            "netlify",
            "digitalocean",
            # DevOps tools
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "jenkins",
            "github",
            "gitlab",
            "circleci",
            "travisci",
            "nginx",
            "apache",
            # Package managers
            "npm",
            "yarn",
            "pip",
            "conda",
            "composer",
            "maven",
            "gradle",
            "cargo",
            # Testing frameworks
            "jest",
            "pytest",
            "junit",
            "mocha",
            "cypress",
            "selenium",
            "playwright",
        ]

        text_lower = text.lower()
        for tech in technologies:
            if re.search(rf"\b{re.escape(tech)}\b", text_lower):
                tech_keywords.append(tech)

        return tech_keywords

    def _extract_format_keywords(self, text: str) -> List[str]:
        """Extract file format and data format keywords."""
        format_keywords = []

        # File extensions and formats
        formats = [
            "json",
            "yaml",
            "xml",
            "csv",
            "markdown",
            "html",
            "css",
            "scss",
            "toml",
            "ini",
            "conf",
            "env",
            "dockerfile",
            "makefile",
            "sql",
            "graphql",
            "proto",
            "avro",
            "parquet",
        ]

        text_lower = text.lower()
        for fmt in formats:
            if re.search(rf"\b{re.escape(fmt)}\b", text_lower):
                format_keywords.append(fmt)

        # Also check for file extensions with dots
        ext_pattern = r"\.(json|yaml|yml|xml|csv|md|html|css|js|ts|py|java|go|rs)\b"
        extensions = re.findall(ext_pattern, text_lower)
        format_keywords.extend(extensions)

        return format_keywords
