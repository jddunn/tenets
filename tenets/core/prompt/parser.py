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
from tenets.core.prompt.normalizer import normalize_list, EntityNormalizer
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
    """Comprehensive prompt parser with modular components and caching."""

    def __init__(
        self,
        config: TenetsConfig,
        cache_manager: Optional[Any] = None,
        use_cache: bool = True,
        use_ml: bool = None,
        use_nlp_ner: bool = None,
        use_fuzzy_matching: bool = True,
    ):
        self.config = config
        self.logger = get_logger(__name__)

        if use_ml is None:
            use_ml = config.nlp.embeddings_enabled
        if use_nlp_ner is None:
            use_nlp_ner = config.nlp.enabled

        self.cache = None
        if use_cache:
            self.cache = PromptCache(
                cache_manager=cache_manager,
                enable_memory_cache=True,
                enable_disk_cache=cache_manager is not None,
                memory_cache_size=100,
            )

        self._init_components(
            cache_manager=cache_manager,
            use_ml=use_ml,
            use_nlp_ner=use_nlp_ner,
            use_fuzzy_matching=use_fuzzy_matching,
        )
        self._init_patterns()

    def _init_components(
        self,
        cache_manager: Optional[Any],
        use_ml: bool,
        use_nlp_ner: bool,
        use_fuzzy_matching: bool,
    ) -> None:
        self.external_manager = ExternalSourceManager(cache_manager)
        self.entity_recognizer = HybridEntityRecognizer(
            use_nlp=use_nlp_ner,
            use_fuzzy=use_fuzzy_matching,
            patterns_file=None,
            spacy_model="en_core_web_sm",
        )
        self.temporal_parser = TemporalParser(patterns_file=None)
        self.intent_detector = HybridIntentDetector(
            use_ml=use_ml,
            patterns_file=None,
            model_name=self.config.nlp.embeddings_model,
        )
        self.keyword_extractor = KeywordExtractor(
            use_yake=self.config.nlp.keyword_extraction_method in ["auto", "yake"],
            language="en",
            use_stopwords=self.config.nlp.stopwords_enabled,
            stopword_set="prompt",
        )
        self.tokenizer = TextTokenizer(use_stopwords=True)
        self.code_tokenizer = CodeTokenizer(use_stopwords=False)
        self.stopword_manager = StopwordManager()
        self.programming_patterns = get_programming_patterns()

    def _init_patterns(self) -> None:
        self._file_pattern_indicators = [
            r"\*\.\w+",
            r"test_\*",
            r"\*_test",
            r"\.\./",
            r"\./",
            r"~/",
        ]

    def parse(
        self,
        prompt: str,
        use_cache: bool = True,
        fetch_external: bool = True,
        min_entity_confidence: float = 0.5,
        min_intent_confidence: float = 0.3,
    ) -> PromptContext:
        if prompt is None:
            raise AttributeError("prompt must be a string, not None")
        self.logger.debug(f"Parsing prompt: {str(prompt)[:100]}...")

        if use_cache and self.cache:
            cached = self.cache.get_parsed_prompt(prompt)
            if cached:
                self.logger.info("Using cached prompt parsing result")
                return cached

        result = self._parse_internal(
            prompt,
            fetch_external,
            min_entity_confidence,
            min_intent_confidence,
        )

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
        external_content = None
        external_context = None

        external_ref = self.external_manager.extract_reference(prompt)
        if external_ref:
            url, identifier, metadata = external_ref
            if fetch_external:
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
            prompt_text = (
                f"{external_content.title}\n{external_content.body}" if external_content else prompt
            )
        else:
            prompt_text = prompt

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

        intent = intent_result.type if intent_result else "understand"
        lower_text = prompt_text.lower()
        if any(kw in lower_text for kw in ["explain", "what does", "show me", "understand"]):
            intent = "understand"
        if (
            "optimize" in lower_text
            or ("improve" in lower_text and "performance" in lower_text)
            or ("reduce" in lower_text and "memory" in lower_text)
            or ("make" in lower_text and "faster" in lower_text)
        ):
            intent = "optimize"
        task_type = self._intent_to_task_type(intent)

        # 3. Keywords
        raw_keywords = self.keyword_extractor.extract(
            prompt_text, max_keywords=self.config.nlp.max_keywords
        )
        prog_keywords = self.programming_patterns.extract_programming_keywords(prompt_text)
        keywords = list(set(raw_keywords + prog_keywords))[: self.config.nlp.max_keywords]
        doc_keywords = self._extract_documentation_keywords(prompt_text)
        keywords = list(set(keywords + doc_keywords))[: self.config.nlp.max_keywords]
        keywords, keyword_norm_meta = normalize_list(keywords)

        # 4. Entities
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

        entities: List[Dict[str, Any]] = []
        ent_normalizer = EntityNormalizer()
        entity_variation_counts: Dict[str, int] = {}
        for entity in entities_list or []:
            norm_res = ent_normalizer.normalize(entity.name)
            entities.append(
                {
                    "name": norm_res.canonical,
                    "type": entity.type,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "original": entity.name,
                    "normalization": {"steps": norm_res.steps, "variants": norm_res.variants},
                }
            )
            entity_variation_counts[norm_res.canonical] = len(norm_res.variants)

        # 5. Temporal
        temporal_expressions = self.temporal_parser.parse(prompt_text)
        temporal_context = None
        if temporal_expressions:
            temporal_info = self.temporal_parser.get_temporal_context(temporal_expressions)
            first_expr = temporal_expressions[0]
            temporal_context = {
                "timeframe": temporal_info.get("timeframe"),
                "since": first_expr.start_date,
                "until": first_expr.end_date,
                "is_relative": first_expr.is_relative,
                "is_recurring": any(e.is_recurring for e in temporal_expressions),
                "expressions": len(temporal_expressions),
            }

        # 6-8. Patterns, focus, scope
        file_patterns = self._extract_file_patterns(prompt)
        focus_areas = self._extract_focus_areas(prompt_text, entities_list or [])
        scope = self._extract_scope(prompt)

        metadata: Dict[str, Any] = {
            "intent_confidence": intent_result.confidence if intent_result else 0,
            "entity_count": len(entities),
            "temporal_expressions": len(temporal_expressions) if temporal_expressions else 0,
            "has_external_ref": external_ref is not None,
            "cached": False,
        }
        if intent_result:
            metadata["intent_evidence"] = intent_result.evidence[:3]
            metadata["intent_source"] = intent_result.source
        metadata["nlp_normalization"] = {
            "keywords": {
                "total": len(keywords),
                "original_total": len(raw_keywords),
                "normalized": keyword_norm_meta,
            },
            "entities": {"total": len(entities), "variation_counts": entity_variation_counts},
        }
        confidences: List[float] = []
        if intent_result:
            confidences.append(intent_result.confidence)
        if entities_list:
            confidences.extend([e.confidence for e in entities_list])
        metadata["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.5

        include_tests = self._should_include_tests(intent, prompt_text, keywords)
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
        if intent == "test":
            return True
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
        lower_prompt = prompt_text.lower()
        test_file_patterns = [
            r"\btest_\w+\.py\b",
            r"\w+_test\.py\b",
            r"\b\w+\.test\.\w+\b",
            r"\b\w+\.spec\.\w+\b",
            r"\btests?/",
            r"\b__tests__\b",
            r"\w+Test\.\w+",
            r"Test\w+\.\w+",
        ]
        if any(re.search(pattern, prompt_text, re.IGNORECASE) for pattern in test_file_patterns):
            return True
        test_action_patterns = [
            r"\b(?:write|add|create|implement|build)\s+(?:unit\s+|integration\s+|e2e\s+)?tests?\b",
            r"\b(?:test|testing)\s+(?:the|this|that|coverage)\b",
            r"\b(?:fix|debug|update|modify|check|review)\s+(?:the\s+|failing\s+)?tests?\b",
            r"\b(?:test|check)\s+(?:coverage|failures?|errors?)\b",
            r"\b(?:run|execute)\s+(?:the\s+)?tests?\b",
            r"\bmock\s+(?:the|this|that)\b",
            r"\bunit\s+tests?\b",
            r"\bintegration\s+tests?\b",
            r"\be2e\s+tests?\b",
            r"\bend-to-end\s+tests?\b",
            r"\btest\s+suite\b",
            r"\btest\s+cases?\b",
            r"\bassertions?\b.*\b(?:fail|pass|error)\b",
            r"\btest\s+coverage\b",
        ]
        if any(re.search(pattern, lower_prompt) for pattern in test_action_patterns):
            return True
        test_quality_patterns = [
            r"\btest\s+coverage\b",
            r"\bcoverage\s+report\b",
            r"\bfailing\s+tests?\b",
            r"\btest\s+failures?\b",
            r"\bbroken\s+tests?\b",
            r"\btests?\s+(?:are\s+)?(?:pass|fail|passing|failing)\b",
            r"\btest\s+(?:pass|fail)\b",
        ]
        if any(re.search(pattern, lower_prompt) for pattern in test_quality_patterns):
            return True
        return False

    def _extract_file_patterns(self, text: str) -> List[str]:
        patterns: List[str] = []
        for indicator in self._file_pattern_indicators:
            matches = re.findall(indicator, text)
            patterns.extend(matches)
        ext_pattern = r"\*?\.\w{2,4}\b"
        extensions = re.findall(ext_pattern, text)
        patterns.extend(extensions)
        file_mentions = re.findall(r"\b(?:file|in|from)\s+([a-zA-Z0-9_\-/]+\.\w{2,4})", text)
        trailing_file_mentions = re.findall(r"\b([a-zA-Z0-9_\-/]+\.\w{2,4})\s+file\b", text)
        file_mentions.extend(trailing_file_mentions)
        patterns.extend(file_mentions)
        standalone_files = re.findall(
            r"\b([a-zA-Z0-9_\-./]+\.(?:json|ya?ml|toml|ini|conf|cfg|txt|md|py|js|ts|tsx|jsx|java|rb|go|rs|php|c|cpp|h|hpp))\b",
            text,
        )
        patterns.extend(standalone_files)
        return list(set(patterns))

    def _extract_focus_areas(self, text: str, entities: List[Entity]) -> List[str]:
        focus_areas: Set[str] = set()
        text_lower = text.lower()
        pattern_categories = self.programming_patterns.get_pattern_categories()
        for category in pattern_categories:
            keywords = self.programming_patterns.get_category_keywords(category)
            if any(kw.lower() in text_lower for kw in keywords):
                focus_areas.add(category)
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
        scope: Dict[str, Any] = {
            "modules": [],
            "directories": [],
            "specific_files": [],
            "exclusions": [],
            "is_global": False,
            "is_specific": False,
        }
        module_patterns = [
            r"\b(?:in|for|of)\s+(?:the\s+)?([a-z][a-z0-9_]*)\s+(?:module|package|component)\b",
            r"\b(?:the\s+)?([a-z][a-z0-9_]*(?:ication|ization)?)\s+(?:module|package|component)\b",
        ]
        modules: Set[str] = set()
        for pat in module_patterns:
            for m in re.findall(pat, text, re.IGNORECASE):
                modules.add(m)
        scope["modules"] = list(modules)
        dir_patterns = [
            r"(?:in|under|within)\s+(?:the\s+)?([a-zA-Z0-9_\-./]+)(?:\s+directory)?",
            r"\b([a-zA-Z0-9_\-./]+/[a-zA-Z0-9_\-./]*)\b",
        ]
        directories = set()
        for pattern in dir_patterns:
            for match in re.findall(pattern, text):
                if not match.startswith("http") and "/" in match:
                    directories.add(match)
        scope["directories"] = list(directories)
        file_pattern = r"\b([a-zA-Z0-9_\-/]+\.(?:py|js|ts|tsx|jsx|java|cpp|c|h|hpp|go|rs|rb|php))\b"
        files = re.findall(file_pattern, text)
        scope["specific_files"] = list(set(files))
        exclude_pattern = r"(?:except|exclude|not|ignore)\s+(?:anything\s+in\s+)?([a-zA-Z0-9_\-/*]+/?)"
        exclusions = set(re.findall(exclude_pattern, text, re.IGNORECASE))
        for common in ["node_modules", "vendor"]:
            if re.search(rf"\b{common}\b", text, re.IGNORECASE):
                exclusions.add(common)
        scope["exclusions"] = list(exclusions)
        if any(word in text.lower() for word in ["entire", "whole", "all", "everything", "project"]):
            scope["is_global"] = True
        elif scope["modules"] or scope["directories"] or scope["specific_files"]:
            scope["is_specific"] = True
        return scope

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        if self.cache:
            return self.cache.get_stats()
        return None

    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear_all()
            self.logger.info("Cleared prompt parser cache")

    def warm_cache(self, common_prompts: List[str]) -> None:
        if not self.cache:
            return
        self.logger.info(f"Pre-warming cache with {len(common_prompts)} prompts")
        for prompt in common_prompts:
            _ = self._parse_internal(prompt, False, 0.5, 0.3)
        self.logger.info("Cache pre-warming complete")

    def _extract_documentation_keywords(self, text: str) -> List[str]:
        doc_keywords: List[str] = []
        text_lower = text.lower()
        api_patterns = [
            r"\b(api|endpoint|route|url|uri|path)\b",
            r"\b(get|post|put|delete|patch|head|options)\b",
            r"\b(request|response|payload|parameter|header|body)\b",
            r"\b(authentication|auth|token|key|secret|oauth|jwt)\b",
            r"\b(rate.?limit|quota|throttle)\b",
            r"\b(webhook|callback|event|notification)\b",
        ]
        config_patterns = [
            r"\b(config|configuration|settings?|options?|parameters?|env|environment)\b",
            r"\b(install|installation|setup|deployment|deploy)\b",
            r"\b(requirements?|dependenc(?:y|ies)|prerequisites?|versions?)\b",
            r"\b(database|db|connection|credential)\b",
            r"\b(server|host|port|domain|certificate|ssl|tls)\b",
            r"\b(docker|container|image|volume|network)\b",
        ]
        structure_patterns = [
            r"\b(tutorial|guide|walkthrough|examples?|demo)\b",
            r"\b(getting.?started|quick.?start|introduction|overview)\b",
            r"\b(troubleshoot(?:ing)?|faq|help|support|issue|problem)\b",
            r"\b(changelog|release.?note|migration|upgrade)\b",
            r"\b(readme|documentation|doc|manual)\b",
        ]
        programming_patterns = [
            r"\b(functions?|methods?|class(?:es)?|interfaces?|modules?|packages?)\b",
            r"\b(variables?|constants?|propert(?:y|ies)|attributes?|fields?)\b",
            r"\b(imports?|includes?|requires?|exports?|dependenc(?:y|ies))\b",
            r"\b(library|framework|sdk|plugin|extension)\b",
            r"\b(debug|test|unit.?test|integration.?test)\b",
        ]
        usage_patterns = [
            r"\b(usage|how.?to|examples?|snippets?|samples?)\b",
            r"\b(command|cli|script|tool|utility)\b",
            r"\b(log|logging|monitor|metric|analytics)\b",
            r"\b(backup|restore|migration|sync)\b",
            r"\b(performance|optimization|cache|memory)\b",
        ]
        all_patterns = api_patterns + config_patterns + structure_patterns + programming_patterns + usage_patterns
        for pattern in all_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    doc_keywords.extend([m for m in match if m])
                else:
                    doc_keywords.append(match)
        doc_keywords.extend(self._extract_technology_keywords(text))
        doc_keywords.extend(self._extract_format_keywords(text))
        normalized_keywords: List[str] = []
        for keyword in doc_keywords:
            kw_lower = keyword.lower()
            if kw_lower in ["settings", "setting"]:
                normalized_keywords.extend(["setting", "settings"])
            elif kw_lower in ["requirements", "requirement"]:
                normalized_keywords.extend(["requirement", "requirements"])
            elif kw_lower in ["examples", "example"]:
                normalized_keywords.extend(["example", "examples"])
            elif kw_lower in ["dependencies", "dependency"]:
                normalized_keywords.extend(["dependency", "dependencies"])
            elif kw_lower in ["functions", "function"]:
                normalized_keywords.extend(["function", "functions"])
            elif kw_lower in ["modules", "module"]:
                normalized_keywords.extend(["module", "modules"])
            elif kw_lower in ["interfaces", "interface"]:
                normalized_keywords.extend(["interface", "interfaces"])
            elif kw_lower in ["classes", "class"]:
                normalized_keywords.extend(["class", "classes"])
            elif kw_lower == "authentication":
                normalized_keywords.extend(["authentication", "auth"])
            elif kw_lower == "configuration":
                normalized_keywords.extend(["configuration", "config"])
            elif kw_lower == "troubleshooting":
                normalized_keywords.extend(["troubleshooting", "troubleshoot"])
            else:
                normalized_keywords.append(keyword)
        doc_keywords.extend(normalized_keywords)
        unique_keywords: List[str] = []
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
        return unique_keywords[:15]

    def _extract_technology_keywords(self, text: str) -> List[str]:
        tech_keywords: List[str] = []
        technologies = [
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
            "aws",
            "azure",
            "gcp",
            "heroku",
            "vercel",
            "netlify",
            "digitalocean",
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
            "npm",
            "yarn",
            "pip",
            "conda",
            "composer",
            "maven",
            "gradle",
            "cargo",
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
        format_keywords: List[str] = []
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
        ext_pattern = r"\.(json|yaml|yml|xml|csv|md|html|css|js|ts|py|java|go|rs)\b"
        extensions = re.findall(ext_pattern, text_lower)
        format_keywords.extend(extensions)
        return format_keywords
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
        raw_keywords = self.keyword_extractor.extract(
            prompt_text,
            max_keywords=self.config.nlp.max_keywords,
        )

        # Add programming-specific keywords
        prog_keywords = self.programming_patterns.extract_programming_keywords(prompt_text)
        keywords = list(set(raw_keywords + prog_keywords))[: self.config.nlp.max_keywords]

        # Add documentation-specific keywords for better context-aware summarization
        doc_keywords = self._extract_documentation_keywords(prompt_text)
        keywords = list(set(keywords + doc_keywords))[: self.config.nlp.max_keywords]

        # Normalize keywords and collect normalization metadata
        normalized_keywords, keyword_norm_meta = normalize_list(keywords)
        keywords = normalized_keywords

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
        ent_normalizer = EntityNormalizer()
        entity_variation_counts = {}
        for entity in entities_list or []:
            norm_res = ent_normalizer.normalize(entity.name)
            entities.append(
                {
                    "name": norm_res.canonical,
                    "type": entity.type,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "original": entity.name,
                    "normalization": {
                        "steps": norm_res.steps,
                        "variants": norm_res.variants,
                    },
                }
            )
            entity_variation_counts[norm_res.canonical] = len(norm_res.variants)

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
        focus_areas = self._extract_focus_areas(prompt_text, entities_list or [])

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

        # Add NLP normalization details for explainability
        metadata["nlp_normalization"] = {
            "keywords": {
                "total": len(keywords),
                "original_total": len(raw_keywords),
                "normalized": keyword_norm_meta,
            },
            "entities": {
                "total": len(entities),
                "variation_counts": entity_variation_counts,
            },
        }

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

        # Test file patterns - check in original case for proper file names
        test_file_patterns = [
            r"\btest_\w+\.py\b",  # test_auth.py
            r"\w+_test\.py\b",  # auth_test.py
            r"\b\w+\.test\.\w+\b",  # auth.test.js
            r"\b\w+\.spec\.\w+\b",  # auth.spec.js
            r"\btests?/",  # tests/ or test/
            r"\b__tests__\b",  # __tests__
            r"\w+Test\.\w+",  # UserTest.java, AuthTest.php
            r"Test\w+\.\w+",  # TestUser.java
        ]

        # Check patterns in both original case and lowercase
        if any(re.search(pattern, prompt_text, re.IGNORECASE) for pattern in test_file_patterns):
            return True

        # Test action patterns - looking for explicit test-related actions
        test_action_patterns = [
            r"\b(?:write|add|create|implement|build)\s+(?:unit\s+|integration\s+|e2e\s+)?tests?\b",
            r"\b(?:test|testing)\s+(?:the|this|that|coverage)\b",
            r"\b(?:fix|debug|update|modify|check|review)\s+(?:the\s+|failing\s+)?tests?\b",
            r"\b(?:test|check)\s+(?:coverage|failures?|errors?)\b",
            r"\b(?:run|execute)\s+(?:the\s+)?tests?\b",
            r"\bmock\s+(?:the|this|that)\b",
            r"\bunit\s+tests?\b",
            r"\bintegration\s+tests?\b",
            r"\be2e\s+tests?\b",
            r"\bend-to-end\s+tests?\b",
            r"\btest\s+suite\b",
            r"\btest\s+cases?\b",
            r"\bassertions?\b.*\b(?:fail|pass|error)\b",
            r"\btest\s+coverage\b",
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
            r"\btests?\s+(?:are\s+)?(?:pass|fail|passing|failing)\b",
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
            r"\b(config|configuration|settings?|options?|parameters?|env|environment)\b",
            r"\b(install|installation|setup|deployment|deploy)\b",
            r"\b(requirements?|dependenc(?:y|ies)|prerequisites?|versions?)\b",
            r"\b(database|db|connection|credential)\b",
            r"\b(server|host|port|domain|certificate|ssl|tls)\b",
            r"\b(docker|container|image|volume|network)\b",
        ]

        # Documentation structure keywords
        structure_patterns = [
            r"\b(tutorial|guide|walkthrough|examples?|demo)\b",
            r"\b(getting.?started|quick.?start|introduction|overview)\b",
            r"\b(troubleshoot(?:ing)?|faq|help|support|issue|problem)\b",
            r"\b(changelog|release.?note|migration|upgrade)\b",
            r"\b(readme|documentation|doc|manual)\b",
        ]

        # Programming concepts in documentation
        programming_patterns = [
            r"\b(functions?|methods?|class(?:es)?|interfaces?|modules?|packages?)\b",
            r"\b(variables?|constants?|propert(?:y|ies)|attributes?|fields?)\b",
            r"\b(imports?|includes?|requires?|exports?|dependenc(?:y|ies))\b",
            r"\b(library|framework|sdk|plugin|extension)\b",
            r"\b(debug|test|unit.?test|integration.?test)\b",
        ]

        # Usage and operational keywords
        usage_patterns = [
            r"\b(usage|how.?to|examples?|snippets?|samples?)\b",
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
        
        # Process and normalize keywords
        normalized_keywords = []
        for keyword in doc_keywords:
            kw_lower = keyword.lower()
            # Normalize to base form and add variations
            if kw_lower in ["settings", "setting"]:
                normalized_keywords.extend(["setting", "settings"])
            elif kw_lower in ["requirements", "requirement"]:
                normalized_keywords.extend(["requirement", "requirements"])
            elif kw_lower in ["examples", "example"]:
                normalized_keywords.extend(["example", "examples"]) 
            elif kw_lower in ["dependencies", "dependency"]:
                normalized_keywords.extend(["dependency", "dependencies"])
            elif kw_lower in ["functions", "function"]:
                normalized_keywords.extend(["function", "functions"])
            elif kw_lower in ["modules", "module"]:
                normalized_keywords.extend(["module", "modules"])
            elif kw_lower in ["interfaces", "interface"]:
                normalized_keywords.extend(["interface", "interfaces"])
            elif kw_lower in ["classes", "class"]:
                normalized_keywords.extend(["class", "classes"])
            elif kw_lower == "authentication":
                normalized_keywords.extend(["authentication", "auth"])
            elif kw_lower == "configuration":
                normalized_keywords.extend(["configuration", "config"])
            elif kw_lower == "troubleshooting":
                normalized_keywords.extend(["troubleshooting", "troubleshoot"])
            else:
                normalized_keywords.append(keyword)
        doc_keywords.extend(normalized_keywords)

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
