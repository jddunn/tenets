"""Prompt parsing and understanding system using centralized NLP components.

This module analyzes user prompts to extract intent, keywords, and context
using the centralized NLP package for all text processing operations.
No more duplicate pattern matching - uses centralized programming patterns.
"""

import base64
import importlib
import importlib.util
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

# Import centralized NLP components
from tenets.core.nlp.keyword_extractor import KeywordExtractor
from tenets.core.nlp.tokenizer import TextTokenizer, CodeTokenizer
from tenets.core.nlp.stopwords import StopwordManager
from tenets.core.nlp.programming_patterns import get_programming_patterns

from tenets.config import TenetsConfig
from tenets.models.context import PromptContext, TaskType
from tenets.utils.logger import get_logger

# Optional dependencies
requests = None
if importlib.util.find_spec("requests"):
    try:
        requests = importlib.import_module("requests")
    except Exception:
        requests = None

BeautifulSoup = None
if importlib.util.find_spec("bs4"):
    try:
        bs4_mod = importlib.import_module("bs4")
        BeautifulSoup = getattr(bs4_mod, "BeautifulSoup", None)
    except Exception:
        BeautifulSoup = None


class IntentType(Enum):
    """Types of user intent detected in prompts."""

    IMPLEMENT = "implement"
    DEBUG = "debug"
    UNDERSTAND = "understand"
    REFACTOR = "refactor"
    TEST = "test"
    DOCUMENT = "document"
    REVIEW = "review"
    OPTIMIZE = "optimize"
    INTEGRATE = "integrate"
    MIGRATE = "migrate"


@dataclass
class ParsedEntity:
    """An entity extracted from the prompt."""

    name: str
    type: str
    confidence: float
    context: str = ""


@dataclass
class TemporalContext:
    """Temporal context extracted from prompt."""

    timeframe: str
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    is_relative: bool = True


@dataclass
class ExternalReference:
    """External reference found in prompt."""

    type: str
    url: str
    identifier: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptParser:
    """Parses and analyzes prompts using centralized NLP components.

    This parser leverages the NLP package for all text processing:
    - Keyword extraction via KeywordExtractor
    - Tokenization via TextTokenizer
    - Stopword filtering via StopwordManager
    - Programming patterns via ProgrammingPatterns

    No more duplicate pattern matching logic!

    Attributes:
        config: TenetsConfig instance
        logger: Logger instance
        keyword_extractor: Centralized keyword extraction
        tokenizer: Centralized text tokenizer
        stopword_manager: Centralized stopword management
        programming_patterns: Centralized programming patterns
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the prompt parser with NLP components.

        Args:
            config: Tenets configuration with NLP settings
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize NLP components
        self.keyword_extractor = KeywordExtractor(
            use_yake=config.nlp.keyword_extraction_method in ["auto", "yake"],
            language="en",
            use_stopwords=config.nlp.stopwords_enabled,
            stopword_set="prompt",  # Use aggressive stopwords for prompts
        )

        self.tokenizer = TextTokenizer(use_stopwords=True)
        self.code_tokenizer = CodeTokenizer(use_stopwords=False)
        self.stopword_manager = StopwordManager()

        # Load centralized programming patterns
        self.programming_patterns = get_programming_patterns()

        # Initialize patterns for intent detection
        self._init_patterns()

        self.logger.info(
            "PromptParser initialized with centralized NLP components and programming patterns"
        )

    def _init_patterns(self):
        """Initialize regex patterns for extraction (non-programming patterns only)."""
        # Intent detection patterns
        self._intent_patterns = {
            IntentType.IMPLEMENT: [
                r"\b(?:implement|add|create|build|develop|make|write)\b",
                r"\b(?:new|feature|functionality|capability)\b",
            ],
            IntentType.DEBUG: [
                r"\b(?:debug|fix|solve|resolve|troubleshoot|investigate)\b",
                r"\b(?:bug|issue|problem|error|exception|crash|fail)\b",
                r"\b(?:not working|broken|wrong|incorrect)\b",
            ],
            IntentType.UNDERSTAND: [
                r"\b(?:understand|explain|how|what|where|why|show)\b",
                r"\b(?:works?|does|flow|architecture|structure)\b",
            ],
            IntentType.REFACTOR: [
                r"\b(?:refactor|restructure|reorganize|improve|clean)\b",
                r"\b(?:optimize|simplify|modernize|update)\b",
            ],
            IntentType.TEST: [
                r"\b(?:test|testing|unit test|integration test|e2e)\b",
                r"\b(?:coverage|spec|suite|assertion)\b",
            ],
            IntentType.DOCUMENT: [
                r"\b(?:document|documentation|docs|readme|comment)\b",
                r"\b(?:describe|annotate|explain)\b",
            ],
            IntentType.REVIEW: [
                r"\b(?:review|check|audit|inspect|analyze)\b",
                r"\b(?:code review|pr review|pull request)\b",
            ],
            IntentType.OPTIMIZE: [
                r"\b(?:optimize|performance|speed|faster|slower)\b",
                r"\b(?:memory|cpu|resource|efficient)\b",
            ],
            IntentType.INTEGRATE: [
                r"\b(?:integrate|connect|interface|api|webhook)\b",
                r"\b(?:third.party|external|service)\b",
            ],
            IntentType.MIGRATE: [
                r"\b(?:migrate|upgrade|port|move|transfer)\b",
                r"\b(?:version|legacy|old|new)\b",
            ],
        }

        # Entity extraction patterns (non-programming)
        self._entity_patterns = {
            "class": r"\b(?:class|model|entity|object)\s+([A-Z][a-zA-Z0-9_]*)|\b([A-Z][a-zA-Z0-9_]*)\s+class\b",
            "function": r"\b(?:function|method|func|def)\s+([a-z_][a-zA-Z0-9_]*)",
            "file": r"\b([a-zA-Z0-9_\-/]*[a-zA-Z0-9_\-]+\.(?:py|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|rb|php|cs|json|yaml|yml|toml|ini))\b",
            "module": r"\b(?:module|package|library)\s+([a-z][a-z0-9_]*)",
            "variable": r"\b(?:variable|var|const|let)\s+([a-z_][a-zA-Z0-9_]*)",
            "api_endpoint": r"(?:GET|POST|PUT|DELETE|PATCH)\s+([/a-zA-Z0-9_\-{}]+)",
            "url": r"https?://[^\s]+",
            "config_key": r"(?:config|setting|env|environment)\s+([A-Z_][A-Z0-9_]*)",
        }

        # Temporal expression patterns
        self._temporal_patterns = {
            "relative": {
                "recent": timedelta(days=7),
                "recently": timedelta(days=7),
                "yesterday": timedelta(days=1),
                "today": timedelta(hours=24),
                "last week": timedelta(weeks=1),
                "last month": timedelta(days=30),
                "last year": timedelta(days=365),
            },
            "absolute": r"\b(\d{4}-\d{2}-\d{2})\b",
            "indicators": [
                "recent",
                "recently",
                "new",
                "newly",
                "latest",
                "current",
                "yesterday",
                "today",
                "now",
                "just",
                "changed",
                "modified",
                "updated",
                "last",
                "past",
                "previous",
                "ago",
                "since",
            ],
        }

        # File pattern indicators
        self._file_pattern_indicators = [
            r"\*\.\w+",  # *.py, *.js
            r"test_\*",  # test_*
            r"\*_test",  # *_test
            r"\.\./",  # ../
            r"\./",  # ./
            r"~/",  # ~/
        ]

    def parse(self, prompt: str) -> PromptContext:
        """Parse a prompt into structured context using NLP components.

        Uses centralized NLP components for all text processing.

        Args:
            prompt: The user's prompt text or URL

        Returns:
            PromptContext with extracted information
        """
        self.logger.debug(f"Parsing prompt with NLP: {prompt[:100]}...")

        # Check if prompt is a URL
        external_ref = self._detect_external_reference(prompt)
        if external_ref:
            fetched = self._fetch_external_content(external_ref)
            prompt_text = fetched if fetched and not fetched.startswith("Content from ") else prompt
            external_context = {
                "source": external_ref.type,
                "url": external_ref.url,
                "identifier": external_ref.identifier,
                "metadata": external_ref.metadata,
            }
        else:
            prompt_text = prompt
            external_context = None

        # Build analysis texts
        original_text = prompt
        combined_text = (
            f"{prompt_text}\n\n{original_text}" if prompt_text != original_text else prompt_text
        )

        # Detect intent and task type
        intent = self._detect_intent(original_text)
        task_type = self._intent_to_task_type(intent)

        # Extract keywords using centralized NLP
        keywords = self._extract_keywords_nlp(combined_text)

        # Extract entities
        entities = self._extract_entities(original_text)

        # Extract file patterns
        file_patterns = self._extract_file_patterns(original_text)

        # Extract focus areas
        focus_areas = self._extract_focus_areas(original_text, entities)

        # Detect temporal context
        temporal = self._extract_temporal_context(original_text)

        # Extract scope indicators
        scope = self._extract_scope(original_text)

        # Build prompt context
        context = PromptContext(
            text=prompt_text,
            original=prompt,
            keywords=keywords,
            task_type=task_type,
            intent=intent.value,
            entities=entities,
            file_patterns=file_patterns,
            focus_areas=focus_areas,
            temporal_context=temporal,
            scope=scope,
            external_context=external_context,
        )

        self.logger.info(
            f"NLP-parsed prompt: task_type={task_type}, "
            f"keywords={len(keywords)}, entities={len(entities)}"
        )

        return context

    def _extract_keywords_nlp(self, text: str) -> List[str]:
        """Extract keywords using centralized NLP components.

        No more duplicate logic - uses centralized keyword extractor
        and programming patterns.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Use centralized keyword extractor
        keywords = self.keyword_extractor.extract(
            text, max_keywords=self.config.nlp.max_keywords, include_scores=False
        )

        # Add programming-specific keywords using centralized patterns
        prog_keywords = self.programming_patterns.extract_programming_keywords(text)

        # Combine and deduplicate
        all_keywords = list(keywords) + prog_keywords

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in all_keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)

        return unique_keywords[: self.config.nlp.max_keywords]

    def _detect_intent(self, text: str) -> IntentType:
        """Detect the primary intent of the prompt.

        Args:
            text: Prompt text

        Returns:
            Detected IntentType
        """
        text_lower = text.lower()
        intent_scores = {}

        # Score each intent type
        for intent_type, patterns in self._intent_patterns.items():
            score = 0
            matched_words = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                score += len(matches)
                matched_words.extend(matches)
            intent_scores[intent_type] = score
            if score > 0:
                self.logger.debug(f"{intent_type}: score={score}, matches={matched_words}")

        # Get highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0]

        # Default based on common keywords
        if any(word in text_lower for word in ["add", "implement", "create", "build"]):
            return IntentType.IMPLEMENT
        elif any(word in text_lower for word in ["fix", "bug", "error", "broken"]):
            return IntentType.DEBUG
        elif any(word in text_lower for word in ["how", "what", "explain", "understand"]):
            return IntentType.UNDERSTAND
        elif any(word in text_lower for word in ["test", "spec", "coverage"]):
            return IntentType.TEST
        else:
            return IntentType.UNDERSTAND  # Default

    def _intent_to_task_type(self, intent: IntentType) -> str:
        """Convert intent to task type."""
        intent_to_task = {
            IntentType.IMPLEMENT: "feature",
            IntentType.DEBUG: "debug",
            IntentType.UNDERSTAND: "understand",
            IntentType.REFACTOR: "refactor",
            IntentType.TEST: "test",
            IntentType.DOCUMENT: "document",
            IntentType.REVIEW: "review",
            IntentType.OPTIMIZE: "optimize",
            IntentType.INTEGRATE: "feature",
            IntentType.MIGRATE: "refactor",
        }
        return intent_to_task.get(intent, "general")

    def _detect_external_reference(self, text: str) -> Optional[ExternalReference]:
        """Detect if text contains an external reference."""
        url_match = re.search(r"https?://[^\s]+", text)
        if not url_match:
            return None

        url = url_match.group(0)
        parsed = urlparse(url)

        # GitHub issue/PR
        if "github.com" in parsed.netloc:
            match = re.search(r"github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)", url)
            if match:
                return ExternalReference(
                    type="github",
                    url=url,
                    identifier=f"{match.group(1)}/{match.group(2)}#{match.group(4)}",
                    metadata={
                        "owner": match.group(1),
                        "repo": match.group(2),
                        "type": match.group(3),
                        "number": match.group(4),
                    },
                )

        # JIRA ticket
        if re.search(r"atlassian\.net|jira", parsed.netloc):
            match = re.search(r"/browse/([A-Z]+-\d+)", url)
            if match:
                return ExternalReference(
                    type="jira",
                    url=url,
                    identifier=match.group(1),
                    metadata={"ticket": match.group(1)},
                )

        # Generic URL
        return ExternalReference(
            type="url", url=url, identifier=url, metadata={"domain": parsed.netloc}
        )

    def _fetch_external_content(self, ref: ExternalReference) -> str:
        """Fetch content from external reference."""
        if not requests:
            self.logger.warning("requests library not available for external content fetching")
            return f"Content from {ref.url}"

        try:
            if ref.type == "github":
                owner = ref.metadata.get("owner")
                repo = ref.metadata.get("repo")
                number = ref.metadata.get("number")
                issue_type = ref.metadata.get("type", "issues")

                api_url = f"https://api.github.com/repos/{owner}/{repo}/{issue_type}/{number}"
                headers = {"Accept": "application/vnd.github.v3+json"}

                github_token = os.environ.get("GITHUB_TOKEN")
                if github_token:
                    headers["Authorization"] = f"token {github_token}"

                response = requests.get(api_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    content = f"Title: {data.get('title', '')}\nDescription: {data.get('body', '')}"
                    return content

        except Exception as e:
            self.logger.error(f"Failed to fetch external content: {e}")

        return f"Content from {ref.url}"

    def _extract_entities(self, text: str) -> List[ParsedEntity]:
        """Extract entities from text."""
        entities = []

        for entity_type, pattern in self._entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get entity name from first non-empty group
                entity_name = None
                if match.groups():
                    for group in match.groups():
                        if group:
                            entity_name = group
                            break
                if not entity_name:
                    entity_name = match.group(0)

                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                entities.append(
                    ParsedEntity(
                        name=entity_name,
                        type=entity_type,
                        confidence=0.8,
                        context=context,
                    )
                )

        return entities

    def _extract_file_patterns(self, text: str) -> List[str]:
        """Extract file patterns from text."""
        patterns = []

        # Look for explicit file patterns
        for indicator in self._file_pattern_indicators:
            matches = re.findall(indicator, text)
            patterns.extend(matches)

        # Look for file extensions
        ext_pattern = r"\*?\.\w{2,4}\b"
        extensions = re.findall(ext_pattern, text)
        patterns.extend(extensions)

        # Look for specific file mentions
        file_mentions = re.findall(r"\b(?:file|in|from)\s+([a-zA-Z0-9_\-/]+\.\w{2,4})", text)
        patterns.extend(file_mentions)

        return list(set(patterns))  # Deduplicate

    def _extract_focus_areas(self, text: str, entities: List[ParsedEntity]) -> List[str]:
        """Extract focus areas from text using centralized patterns."""
        focus_areas = set()
        text_lower = text.lower()

        # Use programming patterns to identify focus areas
        pattern_categories = self.programming_patterns.get_pattern_categories()

        for category in pattern_categories:
            keywords = self.programming_patterns.get_category_keywords(category)
            # Check if any category keywords appear in text
            if any(kw.lower() in text_lower for kw in keywords):
                focus_areas.add(category)

        # Add areas based on entities
        entity_types = set(e.type for e in entities)
        if "api_endpoint" in entity_types:
            focus_areas.add("api")
        if "class" in entity_types or "module" in entity_types:
            focus_areas.add("architecture")

        return list(focus_areas)

    def _extract_temporal_context(self, text: str) -> Optional[TemporalContext]:
        """Extract temporal context from text."""
        text_lower = text.lower()

        # Check for absolute dates with "since" indicator
        since_pattern = r"(?:since|from)\s+(\d{4}-\d{2}-\d{2})"
        since_match = re.search(since_pattern, text)
        if since_match:
            try:
                date = datetime.strptime(since_match.group(1), "%Y-%m-%d")
                return TemporalContext(
                    timeframe=since_match.group(1),
                    since=date,
                    until=datetime.now(),
                    is_relative=False,
                )
            except ValueError:
                pass

        # Check for relative timeframes
        for timeframe, delta in self._temporal_patterns["relative"].items():
            if timeframe in text_lower:
                now = datetime.now()
                return TemporalContext(
                    timeframe=timeframe, since=now - delta, until=now, is_relative=True
                )

        # Check for temporal indicators
        has_temporal = any(
            indicator in text_lower for indicator in self._temporal_patterns["indicators"]
        )

        if has_temporal:
            return TemporalContext(
                timeframe="recent",
                since=datetime.now() - timedelta(days=7),
                until=datetime.now(),
                is_relative=True,
            )

        return None

    def _extract_scope(self, text: str) -> Dict[str, Any]:
        """Extract scope indicators from text."""
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
                directories.add(match)
        scope["directories"] = list(directories)

        # Specific file references
        file_pattern = r"\b([a-zA-Z0-9_\-/]+\.(?:py|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|rb|php))\b"
        files = re.findall(file_pattern, text)
        scope["specific_files"] = list(set(files))

        # Exclusion patterns
        exclude_pattern = (
            r"(?:except|exclude|not|ignore)\s+(?:anything\s+in\s+)?([a-zA-Z0-9_\-/*]+/?)"
        )
        exclusions = set(re.findall(exclude_pattern, text, re.IGNORECASE))
        scope["exclusions"] = list(exclusions)

        # Determine scope type
        if any(
            word in text.lower() for word in ["entire", "whole", "all", "everything", "project"]
        ):
            scope["is_global"] = True
        elif scope["modules"] or scope["directories"] or scope["specific_files"]:
            scope["is_specific"] = True

        return scope

    def enhance_with_context(
        self, prompt_context: PromptContext, additional_info: Dict[str, Any]
    ) -> PromptContext:
        """Enhance prompt context with additional information."""
        # Add additional keywords
        if "keywords" in additional_info:
            prompt_context.keywords.extend(additional_info["keywords"])
            prompt_context.keywords = list(set(prompt_context.keywords))[
                : self.config.nlp.max_keywords
            ]

        # Add focus areas
        if "focus_areas" in additional_info:
            prompt_context.focus_areas.extend(additional_info["focus_areas"])
            prompt_context.focus_areas = list(set(prompt_context.focus_areas))

        # Update metadata
        if "metadata" in additional_info:
            if not hasattr(prompt_context, "metadata"):
                prompt_context.metadata = {}
            prompt_context.metadata.update(additional_info["metadata"])

        return prompt_context
