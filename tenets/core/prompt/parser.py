"""Prompt parsing and understanding system.

This module analyzes user prompts to extract intent, keywords, and context
that guides the relevance ranking and file selection process. It supports
various input formats including plain text, URLs, and structured queries.

The parser identifies:
- Task type (feature, debug, test, refactor, etc.)
- Key concepts and entities
- File patterns and specific paths
- External context (GitHub issues, JIRA tickets)
- Temporal indicators (recent, new, changed)
- Scope indicators (specific modules, areas)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timedelta
from collections import Counter
from enum import Enum

from tenets.config import TenetsConfig
from tenets.models.context import PromptContext, TaskType
from tenets.utils.logger import get_logger


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
    """An entity extracted from the prompt.

    Attributes:
        name: Entity name
        type: Entity type (class, function, module, etc.)
        confidence: Confidence score (0-1)
        context: Surrounding context
    """

    name: str
    type: str
    confidence: float
    context: str = ""


@dataclass
class TemporalContext:
    """Temporal context extracted from prompt.

    Attributes:
        timeframe: Relative timeframe (e.g., "recent", "yesterday")
        since: Datetime to look back from
        until: Datetime to look up to
        is_relative: Whether timeframe is relative or absolute
    """

    timeframe: str
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    is_relative: bool = True


@dataclass
class ExternalReference:
    """External reference found in prompt.

    Attributes:
        type: Type of reference (github, jira, url, etc.)
        url: Full URL
        identifier: Extracted identifier (issue number, ticket ID)
        metadata: Additional metadata
    """

    type: str
    url: str
    identifier: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptParser:
    """Parses and analyzes prompts to extract structured information.

    This parser uses multiple techniques including:
    - Pattern matching for common programming concepts
    - NLP techniques for keyword extraction
    - URL parsing for external references
    - Temporal expression parsing
    - Intent classification

    Attributes:
        config: TenetsConfig instance
        logger: Logger instance
        _keyword_extractor: Keyword extraction system
        _intent_patterns: Patterns for intent detection
        _entity_patterns: Patterns for entity extraction
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the prompt parser.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize extraction systems
        self._init_patterns()
        self._init_keyword_extractor()

        self.logger.info("PromptParser initialized")

    def _init_patterns(self):
        """Initialize regex patterns for extraction."""

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

        # Entity extraction patterns
        self._entity_patterns = {
            "class": r"\b(?:class|model|entity|object)\s+([A-Z][a-zA-Z0-9_]*)",
            "function": r"\b(?:function|method|func|def)\s+([a-z_][a-zA-Z0-9_]*)",
            "module": r"\b(?:module|package|library)\s+([a-z][a-z0-9_]*)",
            "variable": r"\b(?:variable|var|const|let)\s+([a-z_][a-zA-Z0-9_]*)",
            "file": r"(?:file|\.py|\.js|\.java|\.cpp|\.go|\.rs)\s*[:\s]?\s*([a-zA-Z0-9_/\-\.]+)",
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

    def _init_keyword_extractor(self):
        """Initialize keyword extraction system."""
        try:
            # Try to use YAKE if available
            import yake

            self._keyword_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # Max ngram size
                dedupLim=0.7,
                dedupFunc="seqm",
                windowsSize=1,
                top=20,
                features=None,
            )
            self.logger.debug("YAKE keyword extractor initialized")
        except ImportError:
            # Fallback to simple extraction
            self._keyword_extractor = None
            self.logger.debug("YAKE not available, using simple keyword extraction")

    def parse(self, prompt: str) -> PromptContext:
        """Parse a prompt into structured context.

        This is the main entry point for prompt parsing. It analyzes the
        input text and extracts all relevant information to guide the
        file ranking and selection process.

        Args:
            prompt: The user's prompt text or URL

        Returns:
            PromptContext with extracted information

        Example:
            >>> parser = PromptParser(config)
            >>> context = parser.parse("implement OAuth2 authentication for the API")
            >>> print(context.task_type)  # "feature"
            >>> print(context.keywords)  # ["oauth2", "authentication", "api"]
        """
        self.logger.debug(f"Parsing prompt: {prompt[:100]}...")

        # Check if prompt is a URL
        external_ref = self._detect_external_reference(prompt)
        if external_ref:
            # Fetch and parse external content
            prompt_text = self._fetch_external_content(external_ref)
            external_context = {
                "source": external_ref.type,
                "url": external_ref.url,
                "identifier": external_ref.identifier,
                "metadata": external_ref.metadata,
            }
        else:
            prompt_text = prompt
            external_context = None

        # Detect intent and task type
        intent = self._detect_intent(prompt_text)
        task_type = self._intent_to_task_type(intent)

        # Extract keywords
        keywords = self._extract_keywords(prompt_text)

        # Extract entities
        entities = self._extract_entities(prompt_text)

        # Extract file patterns
        file_patterns = self._extract_file_patterns(prompt_text)

        # Extract focus areas
        focus_areas = self._extract_focus_areas(prompt_text, entities)

        # Detect temporal context
        temporal = self._extract_temporal_context(prompt_text)

        # Extract scope indicators
        scope = self._extract_scope(prompt_text)

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
            f"Parsed prompt: task_type={task_type}, "
            f"keywords={len(keywords)}, entities={len(entities)}"
        )

        return context

    def _detect_external_reference(self, text: str) -> Optional[ExternalReference]:
        """Detect if text contains an external reference.

        Args:
            text: Input text

        Returns:
            ExternalReference if found, None otherwise
        """
        # Check for URLs
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

        # GitLab issue/MR
        if "gitlab" in parsed.netloc:
            match = re.search(
                r"gitlab\.[^/]+/([^/]+)/([^/]+)/(-/)?(?:issues|merge_requests)/(\d+)", url
            )
            if match:
                return ExternalReference(
                    type="gitlab",
                    url=url,
                    identifier=f"{match.group(1)}/{match.group(2)}#{match.group(4)}",
                    metadata={
                        "group": match.group(1),
                        "project": match.group(2),
                        "number": match.group(4),
                    },
                )

        # Generic URL
        return ExternalReference(
            type="url", url=url, identifier=url, metadata={"domain": parsed.netloc}
        )

    def _fetch_external_content(self, ref: ExternalReference) -> str:
        """Fetch content from external reference.

        Args:
            ref: External reference

        Returns:
            Fetched content text
        """
        import requests
        import base64

        content_parts = []

        try:
            if ref.type == "github":
                # GitHub API
                owner = ref.metadata.get("owner")
                repo = ref.metadata.get("repo")
                number = ref.metadata.get("number")
                issue_type = ref.metadata.get("type", "issues")

                # Get issue/PR content
                api_url = f"https://api.github.com/repos/{owner}/{repo}/{issue_type}/{number}"
                headers = {"Accept": "application/vnd.github.v3+json"}

                # Add auth token if available
                github_token = os.environ.get("GITHUB_TOKEN")
                if github_token:
                    headers["Authorization"] = f"token {github_token}"

                response = requests.get(api_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    content_parts.append(f"Title: {data.get('title', '')}")
                    content_parts.append(f"Description: {data.get('body', '')}")

                    # Get comments
                    comments_url = f"{api_url}/comments"
                    comments_response = requests.get(comments_url, headers=headers, timeout=10)
                    if comments_response.status_code == 200:
                        comments = comments_response.json()
                        for comment in comments[:5]:  # First 5 comments
                            content_parts.append(
                                f"Comment by {comment.get('user', {}).get('login', 'unknown')}: {comment.get('body', '')}"
                            )

            elif ref.type == "jira":
                # JIRA API
                ticket = ref.metadata.get("ticket")
                jira_domain = os.environ.get("JIRA_DOMAIN", "your-domain.atlassian.net")
                jira_email = os.environ.get("JIRA_EMAIL")
                jira_token = os.environ.get("JIRA_API_TOKEN")

                if jira_email and jira_token:
                    api_url = f"https://{jira_domain}/rest/api/3/issue/{ticket}"
                    auth = base64.b64encode(f"{jira_email}:{jira_token}".encode()).decode()
                    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

                    response = requests.get(api_url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        fields = data.get("fields", {})
                        content_parts.append(f"Summary: {fields.get('summary', '')}")
                        content_parts.append(f"Description: {fields.get('description', '')}")

                        # Get comments
                        comments = fields.get("comment", {}).get("comments", [])
                        for comment in comments[:5]:
                            body = comment.get("body", "")
                            author = comment.get("author", {}).get("displayName", "unknown")
                            content_parts.append(f"Comment by {author}: {body}")

            elif ref.type == "gitlab":
                # GitLab API
                group = ref.metadata.get("group")
                project = ref.metadata.get("project")
                number = ref.metadata.get("number")

                gitlab_domain = os.environ.get("GITLAB_DOMAIN", "gitlab.com")
                gitlab_token = os.environ.get("GITLAB_TOKEN")

                api_url = (
                    f"https://{gitlab_domain}/api/v4/projects/{group}%2F{project}/issues/{number}"
                )
                headers = {"Accept": "application/json"}

                if gitlab_token:
                    headers["PRIVATE-TOKEN"] = gitlab_token

                response = requests.get(api_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    content_parts.append(f"Title: {data.get('title', '')}")
                    content_parts.append(f"Description: {data.get('description', '')}")

                    # Get notes (comments)
                    notes_url = f"{api_url}/notes"
                    notes_response = requests.get(notes_url, headers=headers, timeout=10)
                    if notes_response.status_code == 200:
                        notes = notes_response.json()
                        for note in notes[:5]:
                            if not note.get("system"):  # Skip system notes
                                content_parts.append(
                                    f"Comment by {note.get('author', {}).get('name', 'unknown')}: {note.get('body', '')}"
                                )

            else:
                # Generic URL - try to fetch content
                response = requests.get(ref.url, timeout=10)
                if response.status_code == 200:
                    # Try to extract text content
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(response.text, "html.parser")

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Get text
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = " ".join(chunk for chunk in chunks if chunk)

                    content_parts.append(text[:5000])  # First 5000 chars

        except requests.RequestException as e:
            self.logger.warning(f"Failed to fetch content from {ref.url}: {e}")
            content_parts.append(f"Failed to fetch content from {ref.url}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching content: {e}")
            content_parts.append(f"Error fetching content: {str(e)}")

        # Join all content
        full_content = "\n\n".join(content_parts)

        # If we got nothing, return the URL itself
        if not full_content.strip():
            return f"Content from {ref.url}"

        return full_content

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
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            intent_scores[intent_type] = score

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
        """Convert intent to task type.

        Args:
            intent: Detected intent

        Returns:
            Task type string
        """
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

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        keywords = []

        if self._keyword_extractor:
            # Use YAKE
            try:
                extracted = self._keyword_extractor.extract_keywords(text)
                keywords = [kw[0] for kw in extracted[:20]]  # Top 20 keywords
            except Exception as e:
                self.logger.warning(f"YAKE extraction failed: {e}")
                keywords = self._simple_keyword_extraction(text)
        else:
            # Use simple extraction
            keywords = self._simple_keyword_extraction(text)

        # Add programming-specific keywords
        prog_keywords = self._extract_programming_keywords(text)
        keywords.extend(prog_keywords)

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)

        return unique_keywords[:30]  # Limit to 30 keywords

    def _simple_keyword_extraction(self, text: str) -> List[str]:
        """Simple keyword extraction without YAKE.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Remove common words
        stopwords = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "with",
            "to",
            "for",
            "of",
            "as",
            "by",
            "that",
            "this",
            "it",
            "from",
            "be",
            "are",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "mine",
            "yours",
            "please",
            "want",
            "needs",
            "like",
            "help",
            "use",
            "using",
            "implement",
            "add",
            "create",
            "make",
            "build",
            "get",
            "set",
            "update",
            "delete",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", text)

        # Count frequencies
        word_freq = Counter(word.lower() for word in words)

        # Filter and sort
        keywords = [
            word
            for word, count in word_freq.most_common(50)
            if word not in stopwords and len(word) > 2
        ]

        return keywords[:20]

    def _extract_programming_keywords(self, text: str) -> List[str]:
        """Extract programming-specific keywords.

        Args:
            text: Input text

        Returns:
            List of programming keywords
        """
        keywords = []

        # Common programming concepts
        prog_patterns = {
            "oauth": r"\boauth2?\b",
            "jwt": r"\bjwt\b",
            "api": r"\bapi\b",
            "rest": r"\brest(?:ful)?\b",
            "graphql": r"\bgraphql\b",
            "database": r"\b(?:database|db)\b",
            "cache": r"\bcach(?:e|ing)\b",
            "queue": r"\bqueue\b",
            "async": r"\basync(?:hronous)?\b",
            "auth": r"\bauth(?:entication|orization)?\b",
            "crud": r"\bcrud\b",
            "mvc": r"\bmvc\b",
            "docker": r"\bdocker\b",
            "kubernetes": r"\bkubernetes|k8s\b",
            "microservice": r"\bmicroservice\b",
            "serverless": r"\bserverless\b",
            "webhook": r"\bwebhook\b",
            "websocket": r"\bwebsocket\b",
            "redis": r"\bredis\b",
            "postgres": r"\bpostgres(?:ql)?\b",
            "mongodb": r"\bmongodb?\b",
            "elasticsearch": r"\belasticsearch\b",
        }

        text_lower = text.lower()
        for keyword, pattern in prog_patterns.items():
            if re.search(pattern, text_lower):
                keywords.append(keyword)

        # Framework/library names
        frameworks = [
            "react",
            "vue",
            "angular",
            "django",
            "flask",
            "fastapi",
            "express",
            "spring",
            "rails",
            "laravel",
            "symfony",
            "tensorflow",
            "pytorch",
            "scikit-learn",
            "pandas",
            "numpy",
            "jest",
            "pytest",
            "mocha",
            "jasmine",
            "unittest",
        ]

        for framework in frameworks:
            if framework in text_lower:
                keywords.append(framework)

        return keywords

    def _extract_entities(self, text: str) -> List[ParsedEntity]:
        """Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of ParsedEntity objects
        """
        entities = []

        for entity_type, pattern in self._entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(1) if match.groups() else match.group(0)

                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]

                entities.append(
                    ParsedEntity(
                        name=entity_name,
                        type=entity_type,
                        confidence=0.8,  # Fixed confidence for pattern matches
                        context=context,
                    )
                )

        return entities

    def _extract_file_patterns(self, text: str) -> List[str]:
        """Extract file patterns from text.

        Args:
            text: Input text

        Returns:
            List of file patterns
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

        # Look for specific file mentions
        file_mentions = re.findall(r"\b(?:file|in|from)\s+([a-zA-Z0-9_\-/]+\.\w{2,4})", text)
        patterns.extend(file_mentions)

        # Look for directory patterns
        dir_patterns = re.findall(r"(?:in|from|under)\s+([a-zA-Z0-9_\-/]+/)(?:\s|$)", text)
        patterns.extend(dir_patterns)

        return list(set(patterns))  # Deduplicate

    def _extract_focus_areas(self, text: str, entities: List[ParsedEntity]) -> List[str]:
        """Extract focus areas from text.

        Args:
            text: Input text
            entities: Extracted entities

        Returns:
            List of focus areas
        """
        focus_areas = []

        # Common focus area indicators
        area_patterns = {
            "authentication": r"\b(?:auth|login|logout|session|token|oauth|jwt)\b",
            "api": r"\b(?:api|endpoint|route|rest|graphql|webhook)\b",
            "database": r"\b(?:database|db|sql|query|model|schema|migration)\b",
            "frontend": r"\b(?:ui|ux|frontend|react|vue|angular|component|view)\b",
            "backend": r"\b(?:backend|server|service|controller|handler)\b",
            "testing": r"\b(?:test|spec|coverage|unit|integration|e2e)\b",
            "security": r"\b(?:security|vulnerability|encryption|csrf|xss|sql.injection)\b",
            "performance": r"\b(?:performance|optimization|speed|cache|slow|fast)\b",
            "deployment": r"\b(?:deploy|deployment|ci|cd|docker|kubernetes|aws|azure)\b",
            "configuration": r"\b(?:config|configuration|settings|environment|env)\b",
        }

        text_lower = text.lower()
        for area, pattern in area_patterns.items():
            if re.search(pattern, text_lower):
                focus_areas.append(area)

        # Add areas based on entities
        entity_types = set(e.type for e in entities)
        if "api_endpoint" in entity_types:
            focus_areas.append("api")
        if "class" in entity_types or "module" in entity_types:
            focus_areas.append("architecture")

        return list(set(focus_areas))  # Deduplicate

    def _extract_temporal_context(self, text: str) -> Optional[TemporalContext]:
        """Extract temporal context from text.

        Args:
            text: Input text

        Returns:
            TemporalContext if found, None otherwise
        """
        text_lower = text.lower()

        # Check for temporal indicators
        has_temporal = any(
            indicator in text_lower for indicator in self._temporal_patterns["indicators"]
        )

        if not has_temporal:
            return None

        # Check for relative timeframes
        for timeframe, delta in self._temporal_patterns["relative"].items():
            if timeframe in text_lower:
                now = datetime.now()
                return TemporalContext(
                    timeframe=timeframe, since=now - delta, until=now, is_relative=True
                )

        # Check for absolute dates
        date_match = re.search(self._temporal_patterns["absolute"], text)
        if date_match:
            try:
                date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                return TemporalContext(
                    timeframe=date_match.group(1),
                    since=date,
                    until=datetime.now(),
                    is_relative=False,
                )
            except ValueError:
                pass

        # Generic recent context
        if has_temporal:
            return TemporalContext(
                timeframe="recent",
                since=datetime.now() - timedelta(days=7),
                until=datetime.now(),
                is_relative=True,
            )

        return None

    def _extract_scope(self, text: str) -> Dict[str, Any]:
        """Extract scope indicators from text.

        Args:
            text: Input text

        Returns:
            Dictionary of scope information
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
        module_pattern = (
            r"\b(?:in|for|of)\s+(?:the\s+)?([a-z][a-z0-9_]*)\s+(?:module|package|component)"
        )
        modules = re.findall(module_pattern, text, re.IGNORECASE)
        scope["modules"] = list(set(modules))

        # Directory references
        dir_pattern = r"(?:in|under|within)\s+([a-zA-Z0-9_\-/]+/)"
        directories = re.findall(dir_pattern, text)
        scope["directories"] = list(set(directories))

        # Specific file references
        file_pattern = r"\b([a-zA-Z0-9_\-/]+\.\w{2,4})\b"
        files = re.findall(file_pattern, text)
        scope["specific_files"] = list(set(files))

        # Exclusion patterns
        exclude_pattern = r"(?:except|exclude|not|ignore)\s+([a-zA-Z0-9_\-/*]+)"
        exclusions = re.findall(exclude_pattern, text, re.IGNORECASE)
        scope["exclusions"] = list(set(exclusions))

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
        """Enhance prompt context with additional information.

        This can be used to add information from external sources or
        previous interactions.

        Args:
            prompt_context: Original prompt context
            additional_info: Additional information to add

        Returns:
            Enhanced PromptContext
        """
        # Add additional keywords
        if "keywords" in additional_info:
            prompt_context.keywords.extend(additional_info["keywords"])
            prompt_context.keywords = list(set(prompt_context.keywords))[:30]

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
