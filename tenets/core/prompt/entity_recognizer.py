"""Hybrid entity recognition system.

Combines fast regex-based extraction with optional NLP-based NER for
improved accuracy. Includes confidence scoring and fuzzy matching.
"""

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenets.core.nlp.keyword_extractor import KeywordExtractor
from tenets.utils.logger import get_logger

# Try to import spaCy for NER
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


@dataclass
class Entity:
    """Recognized entity with confidence and context."""

    name: str
    type: str
    confidence: float
    context: str = ""
    start_pos: int = -1
    end_pos: int = -1
    source: str = "regex"  # 'regex', 'ner', 'fuzzy', 'combined'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Make entity hashable for deduplication."""
        return hash((self.name, self.type, self.start_pos))

    def __eq__(self, other):
        """Equality based on name, type, and position."""
        if not isinstance(other, Entity):
            return False
        return (
            self.name == other.name
            and self.type == other.type
            and self.start_pos == other.start_pos
        )


class EntityPatternMatcher:
    """Regex-based entity pattern matching."""

    def __init__(self, patterns_file: Optional[Path] = None):
        """Initialize with entity patterns.

        Args:
            patterns_file: Path to entity patterns JSON file
        """
        self.logger = get_logger(__name__)
        self.patterns = self._load_patterns(patterns_file)
        self.compiled_patterns = self._compile_patterns()

    def _load_patterns(self, patterns_file: Optional[Path]) -> Dict[str, List[Dict]]:
        """Load entity patterns from JSON file."""
        if patterns_file is None:
            patterns_file = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "patterns"
                / "entity_patterns.json"
            )

        if patterns_file.exists():
            try:
                with open(patterns_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load entity patterns: {e}")

        # Return default patterns if file not found
        return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict[str, List[Dict]]:
        """Get comprehensive default entity patterns."""
        return {
            "class": [
                {
                    "pattern": r"\b(?:class|interface|trait|enum|struct)\s+([A-Z][a-zA-Z0-9_]*)",
                    "confidence": 0.95,
                    "description": "Class/interface definition",
                },
                {
                    "pattern": r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)(?:\s+class|\s+interface)?",
                    "confidence": 0.85,
                    "description": "PascalCase class name",
                },
                {
                    "pattern": r"\bnew\s+([A-Z][a-zA-Z0-9_]*)\s*\(",
                    "confidence": 0.9,
                    "description": "Class instantiation",
                },
                {
                    "pattern": r"extends\s+([A-Z][a-zA-Z0-9_]*)",
                    "confidence": 0.9,
                    "description": "Class inheritance",
                },
                {
                    "pattern": r"implements\s+([A-Z][a-zA-Z0-9_]*)",
                    "confidence": 0.9,
                    "description": "Interface implementation",
                },
            ],
            "function": [
                {
                    "pattern": r"\b(?:function|method|def|fn|func|procedure|sub)\s+([a-z_][a-zA-Z0-9_]*)",
                    "confidence": 0.95,
                    "description": "Function definition",
                },
                {
                    "pattern": r"\b([a-z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:\{|=>|->)",
                    "confidence": 0.8,
                    "description": "Function with body",
                },
                {
                    "pattern": r"\.([a-z_][a-zA-Z0-9_]*)\s*\(",
                    "confidence": 0.85,
                    "description": "Method call",
                },
                {
                    "pattern": r"\b(?:async|static|public|private|protected)\s+([a-z_][a-zA-Z0-9_]*)\s*\(",
                    "confidence": 0.9,
                    "description": "Modified function",
                },
            ],
            "variable": [
                {
                    "pattern": r"\b(?:var|let|const|val|my|our)\s+([a-z_][a-zA-Z0-9_]*)",
                    "confidence": 0.9,
                    "description": "Variable declaration",
                },
                {
                    "pattern": r"\b([a-z_][a-zA-Z0-9_]*)\s*(?:=|:=|<-)\s*",
                    "confidence": 0.7,
                    "description": "Variable assignment",
                },
                {
                    "pattern": r"\$([a-zA-Z_][a-zA-Z0-9_]*)",
                    "confidence": 0.85,
                    "description": "Shell/PHP variable",
                },
                {
                    "pattern": r"@([a-zA-Z_][a-zA-Z0-9_]*)",
                    "confidence": 0.8,
                    "description": "Instance variable",
                },
            ],
            "constant": [
                {
                    "pattern": r"\b([A-Z_][A-Z0-9_]*)\b",
                    "confidence": 0.75,
                    "description": "SCREAMING_SNAKE_CASE constant",
                },
                {
                    "pattern": r"\bconst\s+([A-Z_][A-Z0-9_]*)",
                    "confidence": 0.95,
                    "description": "Const declaration",
                },
                {
                    "pattern": r"#define\s+([A-Z_][A-Z0-9_]*)",
                    "confidence": 0.95,
                    "description": "C/C++ macro",
                },
            ],
            "file": [
                {
                    "pattern": r"\b([a-zA-Z0-9_\-/]*[a-zA-Z0-9_\-]+\.(?:py|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|rb|php|cs|swift|kt|scala|dart|r|m|mm|sql|sh|bash|zsh|fish|ps1|psm1|psd1|yml|yaml|json|xml|toml|ini|cfg|conf|env|gitignore|dockerignore|md|rst|txt|csv|tsv|html|htm|css|scss|sass|less|vue|svelte))\b",
                    "confidence": 0.95,
                    "description": "File with extension",
                },
                {
                    "pattern": r"(?:from|import|require|include)\s+['\"]([^'\"]+)['\"]",
                    "confidence": 0.9,
                    "description": "Imported file",
                },
                {
                    "pattern": r"(?:open|read|write|load)\s*\(['\"]([^'\"]+)['\"]",
                    "confidence": 0.85,
                    "description": "File operation",
                },
            ],
            "module": [
                {
                    "pattern": r"(?:import|from)\s+([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)",
                    "confidence": 0.9,
                    "description": "Python/JS module",
                },
                {
                    "pattern": r"(?:use|using)\s+([A-Z][a-zA-Z0-9_]*(?:::[A-Z][a-zA-Z0-9_]*)*)",
                    "confidence": 0.9,
                    "description": "Rust/C++ module",
                },
                {
                    "pattern": r"package\s+([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)",
                    "confidence": 0.95,
                    "description": "Package declaration",
                },
                {
                    "pattern": r"namespace\s+([A-Z][a-zA-Z0-9_]*(?:\\[A-Z][a-zA-Z0-9_]*)*)",
                    "confidence": 0.95,
                    "description": "Namespace declaration",
                },
            ],
            "api_endpoint": [
                {
                    "pattern": r"(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+([/a-zA-Z0-9_\-{}\[\]:]+)",
                    "confidence": 0.95,
                    "description": "REST endpoint",
                },
                {
                    "pattern": r"@(?:Get|Post|Put|Delete|Patch|RequestMapping)\s*\(['\"]([^'\"]+)['\"]",
                    "confidence": 0.95,
                    "description": "Annotated endpoint",
                },
                {
                    "pattern": r"(?:route|path)\s*\(['\"]([/a-zA-Z0-9_\-{}\[\]:]+)['\"]",
                    "confidence": 0.9,
                    "description": "Route definition",
                },
                {"pattern": r"/api/[a-zA-Z0-9_\-/]+", "confidence": 0.8, "description": "API path"},
            ],
            "database": [
                {
                    "pattern": r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+(?:FROM|INTO|TABLE)?\s*([a-zA-Z_][a-zA-Z0-9_]*)",
                    "confidence": 0.9,
                    "description": "SQL table",
                },
                {
                    "pattern": r"(?:collection|table|model)\s*\(['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
                    "confidence": 0.85,
                    "description": "Database collection/table",
                },
                {
                    "pattern": r"\.(?:find|insert|update|delete|save)\s*\(\s*\{['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]",
                    "confidence": 0.8,
                    "description": "NoSQL operation",
                },
            ],
            "config": [
                {
                    "pattern": r"(?:process\.env\.|os\.environ\[?['\"]?)([A-Z_][A-Z0-9_]*)",
                    "confidence": 0.9,
                    "description": "Environment variable",
                },
                {
                    "pattern": r"(?:config|settings|options)\[?['\"]([a-zA-Z_][a-zA-Z0-9_\.]*)['\"]?\]?",
                    "confidence": 0.75,
                    "description": "Config key",
                },
                {
                    "pattern": r"--([a-z\-]+)(?:\s+|=)([^\s]+)?",
                    "confidence": 0.85,
                    "description": "CLI argument",
                },
            ],
            "url": [
                {
                    "pattern": r"https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=]+",
                    "confidence": 0.95,
                    "description": "HTTP(S) URL",
                },
                {
                    "pattern": r"(?:ftp|ssh|git|ws|wss)://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=]+",
                    "confidence": 0.95,
                    "description": "Protocol URL",
                },
                {
                    "pattern": r"[a-zA-Z0-9\-]+\.(?:com|org|net|io|dev|app|co|ai|ml|cloud|tech)\b",
                    "confidence": 0.8,
                    "description": "Domain name",
                },
            ],
            "error": [
                {
                    "pattern": r"(?:Error|Exception|Fault|Failure):\s*([A-Za-z0-9_]+(?:Error|Exception)?)",
                    "confidence": 0.9,
                    "description": "Error type",
                },
                {
                    "pattern": r"([A-Z][a-zA-Z0-9]*(?:Error|Exception|Fault|Failure))\b",
                    "confidence": 0.85,
                    "description": "Exception class",
                },
                {
                    "pattern": r"(?:throw|raise|panic)\s+(?:new\s+)?([A-Z][a-zA-Z0-9]*)",
                    "confidence": 0.9,
                    "description": "Thrown exception",
                },
            ],
            "component": [
                {
                    "pattern": r"<([A-Z][a-zA-Z0-9]*)(?:\s+[^>]*)?>",
                    "confidence": 0.9,
                    "description": "React/Vue component",
                },
                {
                    "pattern": r"@Component\s*\(\s*\{[^}]*selector:\s*['\"]([a-z\-]+)['\"]",
                    "confidence": 0.95,
                    "description": "Angular component",
                },
                {
                    "pattern": r"(?:Widget|Component|View|Controller)\s+([A-Z][a-zA-Z0-9]*)",
                    "confidence": 0.85,
                    "description": "UI component",
                },
            ],
        }

    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float, str]]]:
        """Compile regex patterns for efficiency."""
        compiled = {}

        for entity_type, patterns in self.patterns.items():
            compiled[entity_type] = []
            for pattern_info in patterns:
                try:
                    regex = re.compile(
                        pattern_info["pattern"], re.IGNORECASE if entity_type == "file" else 0
                    )
                    confidence = pattern_info.get("confidence", 0.5)
                    description = pattern_info.get("description", "")
                    compiled[entity_type].append((regex, confidence, description))
                except re.error as e:
                    self.logger.warning(
                        f"Invalid regex pattern for {entity_type}: {pattern_info['pattern']} - {e}"
                    )

        return compiled

    def extract(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        entities = []

        for entity_type, patterns in self.compiled_patterns.items():
            for pattern, base_confidence, description in patterns:
                for match in pattern.finditer(text):
                    # Get entity name from first non-empty group
                    entity_name = None
                    if match.groups():
                        for group in match.groups():
                            if group:
                                entity_name = group
                                break
                    else:
                        entity_name = match.group(0)

                    if not entity_name:
                        continue

                    # Calculate confidence based on context
                    confidence = self._calculate_confidence(
                        base_confidence, entity_name, entity_type, text, match.start(), match.end()
                    )

                    # Get surrounding context
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]

                    entity = Entity(
                        name=entity_name,
                        type=entity_type,
                        confidence=confidence,
                        context=context,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        source="regex",
                        metadata={"pattern_description": description},
                    )

                    entities.append(entity)

        return entities

    def _calculate_confidence(
        self,
        base_confidence: float,
        entity_name: str,
        entity_type: str,
        text: str,
        start_pos: int,
        end_pos: int,
    ) -> float:
        """Calculate dynamic confidence score.

        Args:
            base_confidence: Base confidence from pattern
            entity_name: Extracted entity name
            entity_type: Type of entity
            text: Full text
            start_pos: Start position in text
            end_pos: End position in text

        Returns:
            Adjusted confidence score
        """
        confidence = base_confidence

        # Boost confidence for certain patterns
        if entity_type == "class":
            # Boost if follows naming conventions
            if entity_name[0].isupper() and "_" not in entity_name:
                confidence += 0.05
            # Boost if mentioned multiple times
            if text.count(entity_name) > 1:
                confidence += 0.05

        elif entity_type == "function":
            # Boost if follows camelCase or snake_case
            if "_" in entity_name or (
                entity_name[0].islower() and any(c.isupper() for c in entity_name[1:])
            ):
                confidence += 0.05
            # Boost if has parentheses nearby
            if "(" in text[end_pos : end_pos + 2]:
                confidence += 0.1

        elif entity_type == "file":
            # Boost if path looks valid
            if "/" in entity_name or "\\" in entity_name:
                confidence += 0.05
            # Boost for common file extensions
            common_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs", ".rb"}
            if any(entity_name.endswith(ext) for ext in common_extensions):
                confidence += 0.05

        elif entity_type == "constant":
            # Boost if all uppercase
            if entity_name.isupper():
                confidence += 0.1

        # Penalize if looks like a common word
        common_words = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "which",
            "when",
            "where",
        }
        if entity_name.lower() in common_words:
            confidence -= 0.3

        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, confidence))


class NLPEntityRecognizer:
    """NLP-based named entity recognition using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize NLP entity recognizer.

        Args:
            model_name: spaCy model to use
        """
        self.logger = get_logger(__name__)
        self.nlp = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                self.logger.info(f"Loaded spaCy model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load spaCy model {model_name}: {e}")
                self.logger.info("Install with: python -m spacy download en_core_web_sm")

    def extract(self, text: str) -> List[Entity]:
        """Extract entities using NLP.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        if not self.nlp:
            return []

        entities = []
        doc = self.nlp(text)

        # Map spaCy entity types to our types
        type_mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",
            "DATE": "date",
            "TIME": "time",
            "MONEY": "money",
            "PERCENT": "percent",
            "PRODUCT": "product",
            "EVENT": "event",
            "WORK_OF_ART": "project",
            "LAW": "regulation",
            "LANGUAGE": "language",
            "FAC": "facility",
        }

        # Extract named entities
        for ent in doc.ents:
            entity_type = type_mapping.get(ent.label_, "other")

            entity = Entity(
                name=ent.text,
                type=entity_type,
                confidence=0.8,  # spaCy entities are generally reliable
                context=text[max(0, ent.start_char - 50) : min(len(text), ent.end_char + 50)],
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                source="ner",
                metadata={"spacy_label": ent.label_},
            )
            entities.append(entity)

        # Also extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            # Filter out common/short chunks
            if len(chunk.text) > 3 and chunk.root.pos_ in ["NOUN", "PROPN"]:
                entity = Entity(
                    name=chunk.text,
                    type="concept",
                    confidence=0.6,
                    context=text[
                        max(0, chunk.start_char - 50) : min(len(text), chunk.end_char + 50)
                    ],
                    start_pos=chunk.start_char,
                    end_pos=chunk.end_char,
                    source="ner",
                    metadata={"chunk_type": "noun_chunk"},
                )
                entities.append(entity)

        return entities


class FuzzyEntityMatcher:
    """Fuzzy matching for entity recognition."""

    def __init__(self, known_entities: Optional[Dict[str, List[str]]] = None):
        """Initialize fuzzy matcher.

        Args:
            known_entities: Dictionary of entity type -> list of known entity names
        """
        self.logger = get_logger(__name__)
        self.known_entities = known_entities or self._get_default_known_entities()

    def _get_default_known_entities(self) -> Dict[str, List[str]]:
        """Get default known entities for matching."""
        return {
            "framework": [
                "React",
                "Vue",
                "Angular",
                "Svelte",
                "Next.js",
                "Nuxt",
                "Gatsby",
                "Django",
                "Flask",
                "FastAPI",
                "Express",
                "Nest.js",
                "Koa",
                "Spring",
                "Rails",
                "Laravel",
                "Symfony",
                "ASP.NET",
                "TensorFlow",
                "PyTorch",
                "Keras",
                "scikit-learn",
            ],
            "language": [
                "Python",
                "JavaScript",
                "TypeScript",
                "Java",
                "C++",
                "C#",
                "Go",
                "Rust",
                "Ruby",
                "PHP",
                "Swift",
                "Kotlin",
                "Scala",
                "Dart",
                "R",
                "Julia",
                "MATLAB",
                "Perl",
                "Lua",
                "Haskell",
                "Clojure",
            ],
            "database": [
                "PostgreSQL",
                "MySQL",
                "MongoDB",
                "Redis",
                "Elasticsearch",
                "Cassandra",
                "DynamoDB",
                "SQLite",
                "MariaDB",
                "Oracle",
                "SQL Server",
                "CouchDB",
                "Neo4j",
                "InfluxDB",
                "RabbitMQ",
            ],
            "tool": [
                "Docker",
                "Kubernetes",
                "Jenkins",
                "GitHub Actions",
                "GitLab CI",
                "CircleCI",
                "Travis CI",
                "Terraform",
                "Ansible",
                "Puppet",
                "Git",
                "npm",
                "yarn",
                "pip",
                "Maven",
                "Gradle",
                "webpack",
                "Babel",
                "ESLint",
                "Prettier",
                "Jest",
                "Mocha",
                "Cypress",
            ],
            "service": [
                "AWS",
                "Azure",
                "Google Cloud",
                "Heroku",
                "Netlify",
                "Vercel",
                "DigitalOcean",
                "Cloudflare",
                "Firebase",
                "Supabase",
                "Stripe",
                "PayPal",
                "Twilio",
                "SendGrid",
                "Auth0",
                "Okta",
            ],
        }

    def find_fuzzy_matches(self, text: str, threshold: float = 0.8) -> List[Entity]:
        """Find fuzzy matches for known entities.

        Args:
            text: Text to search in
            threshold: Similarity threshold (0-1)

        Returns:
            List of matched entities
        """
        entities = []
        text_lower = text.lower()

        for entity_type, known_names in self.known_entities.items():
            for known_name in known_names:
                known_lower = known_name.lower()

                # Check for exact match first (case-insensitive)
                if known_lower in text_lower:
                    # Find position
                    pos = text_lower.find(known_lower)
                    entity = Entity(
                        name=known_name,
                        type=entity_type,
                        confidence=0.95,
                        context=text[max(0, pos - 50) : min(len(text), pos + len(known_name) + 50)],
                        start_pos=pos,
                        end_pos=pos + len(known_name),
                        source="fuzzy",
                        metadata={"match_type": "exact"},
                    )
                    entities.append(entity)
                    continue

                # Check for fuzzy match in words
                words = re.findall(r"\b\w+\b", text)
                for i, word in enumerate(words):
                    similarity = SequenceMatcher(None, word.lower(), known_lower).ratio()

                    if similarity >= threshold:
                        # Find position in original text
                        word_pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
                        match = word_pattern.search(text)

                        if match:
                            entity = Entity(
                                name=known_name,
                                type=entity_type,
                                confidence=similarity * 0.9,  # Slightly lower than exact match
                                context=text[
                                    max(0, match.start() - 50) : min(len(text), match.end() + 50)
                                ],
                                start_pos=match.start(),
                                end_pos=match.end(),
                                source="fuzzy",
                                metadata={
                                    "match_type": "fuzzy",
                                    "similarity": similarity,
                                    "matched_text": word,
                                },
                            )
                            entities.append(entity)

        return entities


class HybridEntityRecognizer:
    """Main entity recognizer combining all approaches."""

    def __init__(
        self,
        use_nlp: bool = True,
        use_fuzzy: bool = True,
        patterns_file: Optional[Path] = None,
        spacy_model: str = "en_core_web_sm",
        known_entities: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize hybrid entity recognizer.

        Args:
            use_nlp: Whether to use NLP-based NER
            use_fuzzy: Whether to use fuzzy matching
            patterns_file: Path to entity patterns JSON
            spacy_model: spaCy model name
            known_entities: Known entities for fuzzy matching
        """
        self.logger = get_logger(__name__)

        # Initialize components
        self.pattern_matcher = EntityPatternMatcher(patterns_file)

        self.nlp_recognizer = None
        if use_nlp and SPACY_AVAILABLE:
            self.nlp_recognizer = NLPEntityRecognizer(spacy_model)

        self.fuzzy_matcher = None
        if use_fuzzy:
            self.fuzzy_matcher = FuzzyEntityMatcher(known_entities)

        self.keyword_extractor = KeywordExtractor(use_stopwords=True, stopword_set="prompt")

    def recognize(
        self, text: str, merge_overlapping: bool = True, min_confidence: float = 0.5
    ) -> List[Entity]:
        """Recognize entities using all available methods.

        Args:
            text: Text to extract entities from
            merge_overlapping: Whether to merge overlapping entities
            min_confidence: Minimum confidence threshold

        Returns:
            List of recognized entities
        """
        all_entities = []

        # 1. Regex-based extraction (fastest)
        regex_entities = self.pattern_matcher.extract(text)
        all_entities.extend(regex_entities)
        self.logger.debug(f"Regex extraction found {len(regex_entities)} entities")

        # 2. NLP-based NER (if available)
        if self.nlp_recognizer:
            nlp_entities = self.nlp_recognizer.extract(text)
            all_entities.extend(nlp_entities)
            self.logger.debug(f"NLP extraction found {len(nlp_entities)} entities")

        # 3. Fuzzy matching (if enabled)
        if self.fuzzy_matcher:
            fuzzy_entities = self.fuzzy_matcher.find_fuzzy_matches(text)
            all_entities.extend(fuzzy_entities)
            self.logger.debug(f"Fuzzy matching found {len(fuzzy_entities)} entities")

        # 4. Extract keywords as potential entities
        keywords = self.keyword_extractor.extract(text, max_keywords=20)
        for keyword in keywords:
            # Check if keyword is already covered
            if not any(keyword.lower() in e.name.lower() for e in all_entities):
                # Find keyword position in text
                keyword_lower = keyword.lower()
                text_lower = text.lower()
                pos = text_lower.find(keyword_lower)

                if pos >= 0:
                    entity = Entity(
                        name=keyword,
                        type="keyword",
                        confidence=0.6,
                        context=text[max(0, pos - 50) : min(len(text), pos + len(keyword) + 50)],
                        start_pos=pos,
                        end_pos=pos + len(keyword),
                        source="keyword",
                        metadata={"extraction_method": "keyword"},
                    )
                    all_entities.append(entity)

        # Filter by confidence
        filtered_entities = [e for e in all_entities if e.confidence >= min_confidence]

        # Merge overlapping entities if requested
        if merge_overlapping:
            filtered_entities = self._merge_overlapping_entities(filtered_entities)

        # Sort by position and confidence
        filtered_entities.sort(key=lambda e: (e.start_pos, -e.confidence))

        return filtered_entities

    def _merge_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities, keeping highest confidence.

        Args:
            entities: List of entities to merge

        Returns:
            List of merged entities
        """
        if not entities:
            return []

        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: (e.start_pos, -e.confidence))

        merged = []
        current = sorted_entities[0]

        for entity in sorted_entities[1:]:
            # Check for overlap
            if entity.start_pos < current.end_pos:
                # Overlapping - keep the one with higher confidence
                if entity.confidence > current.confidence:
                    current = entity
                elif entity.confidence == current.confidence:
                    # Same confidence - merge information
                    if entity.source != current.source:
                        current = Entity(
                            name=current.name,
                            type=current.type,
                            confidence=current.confidence,
                            context=current.context,
                            start_pos=current.start_pos,
                            end_pos=max(current.end_pos, entity.end_pos),
                            source="combined",
                            metadata={
                                "sources": [current.source, entity.source],
                                "merged_with": entity.name,
                            },
                        )
            else:
                # No overlap - add current and move to next
                merged.append(current)
                current = entity

        # Don't forget the last entity
        merged.append(current)

        return merged

    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, Any]:
        """Get summary statistics about recognized entities.

        Args:
            entities: List of entities

        Returns:
            Summary dictionary
        """
        summary = {
            "total": len(entities),
            "by_type": {},
            "by_source": {},
            "avg_confidence": 0.0,
            "high_confidence": 0,
            "unique_names": set(),
        }

        for entity in entities:
            # Count by type
            summary["by_type"][entity.type] = summary["by_type"].get(entity.type, 0) + 1

            # Count by source
            summary["by_source"][entity.source] = summary["by_source"].get(entity.source, 0) + 1

            # Track unique names
            summary["unique_names"].add(entity.name.lower())

            # Count high confidence
            if entity.confidence >= 0.8:
                summary["high_confidence"] += 1

        # Calculate average confidence
        if entities:
            summary["avg_confidence"] = sum(e.confidence for e in entities) / len(entities)

        # Convert set to count
        summary["unique_names"] = len(summary["unique_names"])

        return summary
