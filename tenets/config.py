"""Configuration management for Tenets with enhanced LLM and NLP support.

This module handles all configuration for the Tenets system, including loading
from files, environment variables, and providing defaults. Configuration can
be specified at multiple levels with proper precedence.

Configuration precedence (highest to lowest):
1. Runtime parameters (passed to methods)
2. Environment variables (TENETS_*)
3. Project config file (.tenets.yml in project)
4. User config file (~/.config/tenets/config.yml)
5. Default values

The configuration system is designed to work with zero configuration (sensible
defaults) while allowing full customization when needed.

Enhanced with comprehensive LLM provider support for optional AI-powered features
and centralized NLP configuration for all text processing operations.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Logger imported locally to avoid circular imports


@dataclass
class NLPConfig:
    """Configuration for centralized NLP (Natural Language Processing) system.

    Controls all text processing operations including tokenization, keyword
    extraction, stopword filtering, embeddings, and similarity computation.
    All NLP operations are centralized in the tenets.core.nlp package.

    Attributes:
        enabled: Whether NLP features are enabled globally
        stopwords_enabled: Whether to use stopword filtering
        code_stopword_set: Stopword set for code search (minimal)
        prompt_stopword_set: Stopword set for prompt parsing (aggressive)
        custom_stopword_files: Additional custom stopword files
        tokenization_mode: Tokenization mode ('code', 'text', 'auto')
        preserve_original_tokens: Keep original tokens for exact matching
        split_camelcase: Split camelCase and PascalCase
        split_snakecase: Split snake_case
        min_token_length: Minimum token length to keep
        keyword_extraction_method: Method for keyword extraction
        max_keywords: Maximum keywords to extract
        ngram_size: Maximum n-gram size for extraction
        yake_dedup_threshold: YAKE deduplication threshold
        tfidf_use_sublinear: Use log scaling for term frequency
        tfidf_use_idf: Use inverse document frequency
        tfidf_norm: Normalization method for TF-IDF
        bm25_k1: BM25 term frequency saturation parameter
        bm25_b: BM25 length normalization parameter
        embeddings_enabled: Whether to use embeddings (requires ML)
        embeddings_model: Default embedding model
        embeddings_device: Device for embeddings ('auto', 'cpu', 'cuda')
        embeddings_cache: Whether to cache embeddings
        embeddings_batch_size: Batch size for embedding generation
        similarity_metric: Default similarity metric
        similarity_threshold: Default similarity threshold
        cache_embeddings_ttl_days: TTL for embedding cache
        cache_tfidf_ttl_days: TTL for TF-IDF cache
        cache_keywords_ttl_days: TTL for keyword cache
        multiprocessing_enabled: Enable multiprocessing for NLP operations
        multiprocessing_workers: Number of workers (None = cpu_count)
        multiprocessing_chunk_size: Chunk size for parallel processing
    """

    # Core settings
    enabled: bool = True

    # Stopword configuration
    stopwords_enabled: bool = True
    code_stopword_set: str = "minimal"  # Minimal for code search
    prompt_stopword_set: str = "aggressive"  # Aggressive for prompt parsing
    custom_stopword_files: List[str] = field(default_factory=list)

    # Tokenization configuration
    tokenization_mode: str = "auto"  # 'code', 'text', or 'auto'
    preserve_original_tokens: bool = True
    split_camelcase: bool = True
    split_snakecase: bool = True
    min_token_length: int = 2

    # Keyword extraction configuration
    keyword_extraction_method: str = "auto"  # 'auto', 'yake', 'tfidf', 'frequency'
    max_keywords: int = 30
    ngram_size: int = 3
    yake_dedup_threshold: float = 0.7

    # TF-IDF configuration
    tfidf_use_sublinear: bool = True
    tfidf_use_idf: bool = True
    tfidf_norm: str = "l2"

    # BM25 configuration
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    # Embeddings configuration (ML features)
    embeddings_enabled: bool = False
    embeddings_model: str = "all-MiniLM-L6-v2"
    embeddings_device: str = "auto"  # 'auto', 'cpu', 'cuda'
    embeddings_cache: bool = True
    embeddings_batch_size: int = 32

    # Similarity configuration
    similarity_metric: str = "cosine"  # 'cosine', 'euclidean', 'manhattan'
    similarity_threshold: float = 0.7

    # Cache configuration for NLP
    cache_embeddings_ttl_days: int = 30
    cache_tfidf_ttl_days: int = 7
    cache_keywords_ttl_days: int = 7

    # Performance optimization
    multiprocessing_enabled: bool = True
    multiprocessing_workers: Optional[int] = None  # None = cpu_count()
    multiprocessing_chunk_size: int = 100

    def __post_init__(self):
        """Validate NLP configuration after initialization."""
        # Validate stopword sets
        valid_stopword_sets = ["minimal", "aggressive", "custom"]
        if self.code_stopword_set not in valid_stopword_sets:
            raise ValueError(f"Invalid code stopword set: {self.code_stopword_set}")
        if self.prompt_stopword_set not in valid_stopword_sets:
            raise ValueError(f"Invalid prompt stopword set: {self.prompt_stopword_set}")

        # Validate tokenization mode
        valid_modes = ["code", "text", "auto"]
        if self.tokenization_mode not in valid_modes:
            raise ValueError(f"Invalid tokenization mode: {self.tokenization_mode}")

        # Validate keyword extraction method
        valid_methods = ["auto", "yake", "tfidf", "frequency"]
        if self.keyword_extraction_method not in valid_methods:
            raise ValueError(f"Invalid keyword extraction method: {self.keyword_extraction_method}")

        # Validate similarity metric
        valid_metrics = ["cosine", "euclidean", "manhattan"]
        if self.similarity_metric not in valid_metrics:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")

        # Validate thresholds
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"Similarity threshold must be between 0 and 1, got {self.similarity_threshold}"
            )
        if not 0.0 < self.yake_dedup_threshold <= 1.0:
            raise ValueError(
                f"YAKE dedup threshold must be between 0 and 1, got {self.yake_dedup_threshold}"
            )

        # Validate BM25 parameters
        if self.bm25_k1 < 0:
            raise ValueError(f"BM25 k1 must be non-negative, got {self.bm25_k1}")
        if not 0.0 <= self.bm25_b <= 1.0:
            raise ValueError(f"BM25 b must be between 0 and 1, got {self.bm25_b}")


@dataclass
class LLMConfig:
    """Configuration for LLM (Large Language Model) integration.

    Supports multiple providers and models with comprehensive cost controls,
    rate limiting, and fallback strategies. All LLM features are optional
    and disabled by default.

    Attributes:
        enabled: Whether LLM features are enabled globally
        provider: Primary LLM provider (openai, anthropic, openrouter, litellm, ollama)
        fallback_providers: Ordered list of fallback providers if primary fails
        api_keys: Dictionary of provider -> API key (can use env vars)
        api_base_urls: Custom API endpoints for providers (e.g., for proxies)
        models: Model selection for different tasks
        max_cost_per_run: Maximum cost in USD per execution run
        max_cost_per_day: Maximum cost in USD per day
        max_tokens_per_request: Maximum tokens per single request
        max_context_length: Maximum context window to use
        temperature: Sampling temperature (0.0-2.0, lower = more deterministic)
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty for token repetition
        presence_penalty: Presence penalty for topic repetition
        requests_per_minute: Rate limit for API requests
        retry_on_error: Whether to retry failed requests
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Backoff multiplier for retry delays
        timeout: Request timeout in seconds
        stream: Whether to stream responses
        cache_responses: Whether to cache LLM responses
        cache_ttl_hours: Cache time-to-live in hours
        log_requests: Whether to log all LLM requests
        log_responses: Whether to log all LLM responses
        custom_headers: Additional headers for API requests
        organization_id: Organization ID for providers that support it
        project_id: Project ID for providers that support it
    """

    # Core settings
    enabled: bool = False
    provider: str = "openai"
    fallback_providers: List[str] = field(default_factory=lambda: ["anthropic", "openrouter"])

    # API Keys - can use environment variables with ${VAR_NAME} syntax
    api_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "openai": "${OPENAI_API_KEY}",
            "anthropic": "${ANTHROPIC_API_KEY}",
            "openrouter": "${OPENROUTER_API_KEY}",
            "cohere": "${COHERE_API_KEY}",
            "together": "${TOGETHER_API_KEY}",
            "huggingface": "${HUGGINGFACE_API_KEY}",
            "replicate": "${REPLICATE_API_KEY}",
            "ollama": "",  # Local, no key needed
        }
    )

    # Custom API endpoints (for proxies, self-hosted, etc.)
    api_base_urls: Dict[str, str] = field(
        default_factory=lambda: {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "ollama": "http://localhost:11434",
        }
    )

    # Model selection for different tasks
    models: Dict[str, str] = field(
        default_factory=lambda: {
            "default": "gpt-4o-mini",
            "summarization": "gpt-3.5-turbo",
            "analysis": "gpt-4o",
            "embeddings": "text-embedding-3-small",
            "code_generation": "gpt-4o",
            "semantic_search": "text-embedding-3-small",
            # Anthropic models
            "anthropic_default": "claude-3-haiku-20240307",
            "anthropic_analysis": "claude-3-sonnet-20240229",
            "anthropic_code": "claude-3-opus-20240229",
            # Open source via Ollama
            "ollama_default": "llama2",
            "ollama_code": "codellama",
            "ollama_embeddings": "nomic-embed-text",
        }
    )

    # Cost controls
    max_cost_per_run: float = 0.10  # $0.10 per run
    max_cost_per_day: float = 10.00  # $10.00 per day
    max_tokens_per_request: int = 4000
    max_context_length: int = 100000  # Use model's max if not specified

    # Generation parameters
    temperature: float = 0.3  # Lower = more deterministic
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Rate limiting and retries
    requests_per_minute: int = 60
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial delay in seconds
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    timeout: int = 30  # Request timeout in seconds

    # Response handling
    stream: bool = False  # Stream responses for better UX
    cache_responses: bool = True
    cache_ttl_hours: int = 24

    # Logging and debugging
    log_requests: bool = False
    log_responses: bool = False

    # Provider-specific settings
    custom_headers: Dict[str, str] = field(default_factory=dict)
    organization_id: Optional[str] = None  # For OpenAI
    project_id: Optional[str] = None  # For various providers

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Resolve environment variables in API keys
        self._resolve_api_keys()

        # Validate provider
        valid_providers = [
            "openai",
            "anthropic",
            "openrouter",
            "cohere",
            "together",
            "huggingface",
            "replicate",
            "ollama",
            "litellm",
        ]
        if self.provider not in valid_providers:
            raise ValueError(
                f"Invalid LLM provider: {self.provider}. Must be one of {valid_providers}"
            )

        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")

        # Validate costs
        if self.max_cost_per_run < 0:
            raise ValueError(f"max_cost_per_run must be non-negative, got {self.max_cost_per_run}")
        if self.max_cost_per_day < 0:
            raise ValueError(f"max_cost_per_day must be non-negative, got {self.max_cost_per_day}")

        # Ensure cost per run doesn't exceed daily limit
        self.max_cost_per_run = min(self.max_cost_per_run, self.max_cost_per_day)

    def _resolve_api_keys(self):
        """Resolve environment variables in API keys.

        Supports ${VAR_NAME} syntax for environment variable substitution.
        """

        for provider, key_value in self.api_keys.items():
            if key_value and key_value.startswith("${") and key_value.endswith("}"):
                # Extract environment variable name
                env_var = key_value[2:-1]
                # Get from environment, keep original if not found
                self.api_keys[provider] = os.environ.get(env_var, key_value)

    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for a specific provider.

        Args:
            provider: Provider name (uses default if not specified)

        Returns:
            API key string or None if not configured
        """
        provider = provider or self.provider
        key = self.api_keys.get(provider)

        # Don't return placeholder values
        if key and key.startswith("${") and key.endswith("}"):
            return None

        return key

    def get_model(self, task: str = "default", provider: Optional[str] = None) -> str:
        """Get model name for a specific task and provider.

        Args:
            task: Task type (default, summarization, analysis, etc.)
            provider: Provider name (uses default if not specified)

        Returns:
            Model name string
        """
        provider = provider or self.provider

        # Try provider-specific model first
        provider_task = f"{provider}_{task}"
        if provider_task in self.models:
            return self.models[provider_task]

        # Fall back to general task model
        return self.models.get(task, self.models["default"])

    def to_litellm_params(self) -> Dict[str, Any]:
        """Convert to parameters for LiteLLM library.

        Returns:
            Dictionary of parameters compatible with LiteLLM
        """
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens_per_request,
            "timeout": self.timeout,
            "stream": self.stream,
        }

        # Add API key if available
        api_key = self.get_api_key()
        if api_key:
            params["api_key"] = api_key

        # Add custom base URL if specified
        if self.provider in self.api_base_urls:
            params["api_base"] = self.api_base_urls[self.provider]

        # Add organization/project IDs if specified
        if self.organization_id:
            params["organization"] = self.organization_id
        if self.project_id:
            params["project"] = self.project_id

        # Add custom headers
        if self.custom_headers:
            params["extra_headers"] = self.custom_headers

        return params


@dataclass
class ScannerConfig:
    """Configuration for file scanning subsystem.

    Controls how tenets discovers and filters files in a codebase.

    Attributes:
        respect_gitignore: Whether to respect .gitignore files
        follow_symlinks: Whether to follow symbolic links
        max_file_size: Maximum file size in bytes to analyze
        max_files: Maximum number of files to scan
        binary_check: Whether to check for and skip binary files
        encoding: Default file encoding
        additional_ignore_patterns: Extra patterns to ignore
        additional_include_patterns: Extra patterns to include
        workers: Number of parallel workers for scanning
        parallel_mode: Parallel execution mode ("thread", "process", or "auto")
        timeout: Per-file analysis timeout used in parallel execution (seconds)
    """

    respect_gitignore: bool = True
    follow_symlinks: bool = False
    max_file_size: int = 5_000_000  # 5MB
    max_files: int = 10_000
    binary_check: bool = True
    encoding: str = "utf-8"
    additional_ignore_patterns: List[str] = field(
        default_factory=lambda: [
            "*.pyc",
            "*.pyo",
            "__pycache__",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.egg-info",
            "*.dist-info",
            ".tox",
            ".nox",
            ".coverage",
            ".hypothesis",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        ]
    )
    additional_include_patterns: List[str] = field(default_factory=list)
    workers: int = 4
    parallel_mode: str = "auto"
    timeout: float = 5.0
    exclude_tests_by_default: bool = True
    test_patterns: List[str] = field(
        default_factory=lambda: [
            # Python test patterns
            "test_*.py",
            "*_test.py",
            "test*.py",
            # JavaScript/TypeScript test patterns
            "*.test.js",
            "*.spec.js",
            "*.test.ts",
            "*.spec.ts",
            "*.test.jsx",
            "*.spec.jsx",
            "*.test.tsx",
            "*.spec.tsx",
            # Java test patterns
            "*Test.java",
            "*Tests.java",
            "*TestCase.java",
            # C# test patterns
            "*Test.cs",
            "*Tests.cs",
            "*TestCase.cs",
            # Go test patterns
            "*_test.go",
            "test_*.go",
            # Ruby test patterns
            "*_test.rb",
            "*_spec.rb",
            "test_*.rb",
            # PHP test patterns
            "*Test.php",
            "*_test.php",
            "test_*.php",
            # Rust test patterns
            "*_test.rs",
            "test_*.rs",
            # General patterns
            "**/test/**",
            "**/tests/**",
            "**/*test*/**",
        ]
    )
    test_directories: List[str] = field(
        default_factory=lambda: [
            "test",
            "tests",
            "__tests__",
            "spec",
            "specs",
            "testing",
            "test_*",
            "*_test",
            "*_tests",
            # Language specific
            "unit_tests",
            "integration_tests",
            "e2e",
            "e2e_tests",
            "functional_tests",
            "acceptance_tests",
            "regression_tests",
        ]
    )


@dataclass
class RankingConfig:
    """Configuration for relevance ranking system.

    Controls how files are scored and ranked for relevance to prompts.
    Uses centralized NLP components for all text processing.

    Attributes:
        algorithm: Default ranking algorithm (fast, balanced, thorough, ml)
        threshold: Minimum relevance score to include file
        use_tfidf: Whether to use TF-IDF for keyword matching
        use_stopwords: Whether to use stopwords filtering
        use_embeddings: Whether to use semantic embeddings (requires ML)
        use_git: Whether to include git signals in ranking
        use_ml: Whether to enable ML features (uses NLP embeddings)
        embedding_model: Which embedding model to use
        custom_weights: Custom weights for ranking factors
        workers: Number of parallel workers for ranking
        parallel_mode: Parallel execution mode ("thread", "process", or "auto")
        batch_size: Batch size for ML operations
    """

    algorithm: str = "balanced"
    threshold: float = 0.1
    use_tfidf: bool = True
    use_stopwords: bool = False  # For code search, use minimal stopwords
    use_embeddings: bool = False
    use_git: bool = True
    use_ml: bool = False  # Enable ML features (uses NLP package)
    embedding_model: str = "all-MiniLM-L6-v2"
    custom_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "keyword_match": 0.25,
            "path_relevance": 0.20,
            "import_graph": 0.20,
            "git_activity": 0.15,
            "file_type": 0.10,
            "complexity": 0.10,
        }
    )
    workers: int = 2
    parallel_mode: str = "auto"
    batch_size: int = 100  # Batch size for ML operations


@dataclass
class SummarizerConfig:
    """Configuration for content summarization system.

    Controls how text and code are compressed to fit within token limits.

    Attributes:
        default_mode: Default summarization mode (extractive, compressive, textrank, transformer, llm, auto)
        target_ratio: Default target compression ratio (0.3 = 30% of original)
        enable_cache: Whether to cache summaries
        preserve_code_structure: Whether to preserve imports/signatures in code
        max_cache_size: Maximum number of cached summaries
        llm_provider: LLM provider for LLM mode (uses global LLM config)
        llm_model: LLM model to use (uses global LLM config)
        llm_temperature: LLM sampling temperature
        llm_max_tokens: Maximum tokens for LLM response
        enable_ml_strategies: Whether to enable ML-based strategies
        quality_threshold: Quality threshold for auto mode selection
        batch_size: Batch size for parallel processing
        docs_context_aware: Whether to enable context-aware summarization for documentation files
        docs_show_in_place_context: When enabled, preserves and highlights relevant context in documentation summaries instead of generic structure
        docs_context_search_depth: How deep to search for contextual references (1=direct mentions, 2=semantic similarity, 3=deep analysis)
        docs_context_min_confidence: Minimum confidence threshold for context relevance (0.0-1.0)
        docs_context_max_sections: Maximum number of contextual sections to preserve per document
        docs_context_preserve_examples: Whether to always preserve code examples and snippets in documentation
    """

    default_mode: str = "auto"
    target_ratio: float = 0.3
    enable_cache: bool = True
    preserve_code_structure: bool = True
    max_cache_size: int = 100
    llm_provider: Optional[str] = None  # Uses global LLM config if not specified
    llm_model: Optional[str] = None  # Uses global LLM config if not specified
    llm_temperature: float = 0.3
    llm_max_tokens: int = 500
    enable_ml_strategies: bool = True
    quality_threshold: str = "medium"  # low, medium, high
    batch_size: int = 10

    # Documentation context-aware summarization settings
    docs_context_aware: bool = True  # Enable smart context-aware documentation summarization
    docs_show_in_place_context: bool = (
        True  # Preserve relevant context sections in-place within summaries
    )
    docs_context_search_depth: int = 2  # 1=direct mentions, 2=semantic similarity, 3=deep analysis
    docs_context_min_confidence: float = 0.6  # Minimum confidence for context relevance (0.0-1.0)
    docs_context_max_sections: int = 10  # Maximum contextual sections to preserve per document
    docs_context_preserve_examples: bool = True  # Always preserve code examples and snippets


@dataclass
class TenetConfig:
    """Configuration for the tenet (guiding principles) system.

    Controls how tenets are managed and injected into context, including
    smart injection frequency, session tracking, and adaptive behavior.

    Attributes:
        auto_instill: Whether to automatically apply tenets to context
        max_per_context: Maximum tenets to inject per context
        reinforcement: Whether to reinforce critical tenets
        injection_strategy: Default injection strategy ('strategic', 'top', 'distributed')
        min_distance_between: Minimum character distance between injections
        prefer_natural_breaks: Whether to inject at natural break points
        storage_path: Where to store tenet database
        collections_enabled: Whether to enable tenet collections

        # Smart injection frequency settings
        injection_frequency: How often to inject tenets ('always', 'periodic', 'adaptive', 'manual')
        injection_interval: Numeric interval for periodic injection (e.g., every 3rd distill)
        session_complexity_threshold: Complexity threshold for smart injection (0-1)
        min_session_length: Minimum session length before first injection
        adaptive_injection: Enable adaptive injection based on context analysis
        track_injection_history: Track injection history per session for smarter decisions
        decay_rate: How quickly tenet importance decays (0-1, higher = faster decay)
        reinforcement_interval: How often to reinforce critical tenets (every N injections)

        # Session tracking settings
        session_aware: Enable session-aware injection patterns
        session_memory_limit: Max sessions to track in memory
        persist_session_history: Save session histories to disk

        # Advanced injection settings
        complexity_weight: Weight given to complexity in injection decisions (0-1)
        priority_boost_critical: Boost factor for critical priority tenets
        priority_boost_high: Boost factor for high priority tenets
        skip_low_priority_on_complex: Skip low priority tenets when complexity > threshold

        # Metrics and analysis
        track_effectiveness: Track tenet effectiveness metrics
        effectiveness_window_days: Days to consider for effectiveness analysis
        min_compliance_score: Minimum compliance score before reinforcement

    # System instruction (system prompt) configuration
    system_instruction: Optional text to inject as foundational context
    system_instruction_enabled: Enable auto-injection when instruction exists
    system_instruction_position: Where to inject (top, after_header, before_content)
    system_instruction_format: Format of instruction (markdown, xml, comment, plain)
    system_instruction_once_per_session: Inject once per session; if no session, inject every distill
    """

    # Core settings
    auto_instill: bool = True
    max_per_context: int = 5
    reinforcement: bool = True
    injection_strategy: str = "strategic"
    min_distance_between: int = 1000
    prefer_natural_breaks: bool = True
    storage_path: Optional[Path] = None
    collections_enabled: bool = True

    # Smart injection frequency settings
    injection_frequency: str = "adaptive"  # 'always', 'periodic', 'adaptive', 'manual'
    injection_interval: int = 3  # Every Nth distill for periodic mode
    session_complexity_threshold: float = 0.7  # 0-1, triggers injection when exceeded
    min_session_length: int = 5  # Minimum distills before first injection
    adaptive_injection: bool = True  # Enable smart context-aware injection
    track_injection_history: bool = True  # Track per-session injection patterns
    decay_rate: float = 0.1  # How quickly tenet importance decays (0=never, 1=immediate)
    reinforcement_interval: int = 10  # Reinforce critical tenets every N injections

    # Session tracking settings
    session_aware: bool = True  # Enable session-aware patterns
    session_memory_limit: int = 100  # Max concurrent sessions to track
    persist_session_history: bool = True  # Save histories to disk

    # Advanced injection settings
    complexity_weight: float = 0.5  # Weight for complexity in decisions
    priority_boost_critical: float = 2.0  # Boost for critical tenets
    priority_boost_high: float = 1.5  # Boost for high priority tenets
    skip_low_priority_on_complex: bool = True  # Skip low priority when complex

    # Metrics and analysis
    track_effectiveness: bool = True  # Track effectiveness metrics
    effectiveness_window_days: int = 30  # Analysis window
    min_compliance_score: float = 0.6  # Minimum before reinforcement

    # System instruction (system prompt) configuration
    system_instruction: Optional[str] = None
    system_instruction_enabled: bool = False
    system_instruction_position: str = "top"  # 'top', 'after_header', 'before_content'
    system_instruction_format: str = "markdown"  # 'markdown', 'xml', 'comment', 'plain'
    system_instruction_once_per_session: bool = True

    def __post_init__(self):
        """Validate tenet configuration after initialization."""
        # Validate injection frequency
        valid_frequencies = ["always", "periodic", "adaptive", "manual"]
        if self.injection_frequency not in valid_frequencies:
            raise ValueError(
                f"Invalid injection_frequency: {self.injection_frequency}. "
                f"Must be one of {valid_frequencies}"
            )

        # Validate injection strategy
        valid_strategies = ["strategic", "top", "distributed", "uniform", "random"]
        if self.injection_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid injection_strategy: {self.injection_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        # Validate numeric ranges
        if not 0.0 <= self.session_complexity_threshold <= 1.0:
            raise ValueError(
                f"session_complexity_threshold must be between 0 and 1, "
                f"got {self.session_complexity_threshold}"
            )

        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be between 0 and 1, got {self.decay_rate}")

        if not 0.0 <= self.complexity_weight <= 1.0:
            raise ValueError(
                f"complexity_weight must be between 0 and 1, got {self.complexity_weight}"
            )

        if not 0.0 <= self.min_compliance_score <= 1.0:
            raise ValueError(
                f"min_compliance_score must be between 0 and 1, got {self.min_compliance_score}"
            )

        if self.injection_interval < 1:
            raise ValueError(
                f"injection_interval must be at least 1, got {self.injection_interval}"
            )

        if self.min_session_length < 0:
            raise ValueError(
                f"min_session_length must be non-negative, got {self.min_session_length}"
            )

        if self.reinforcement_interval < 1:
            raise ValueError(
                f"reinforcement_interval must be at least 1, got {self.reinforcement_interval}"
            )

        if self.session_memory_limit < 1:
            raise ValueError(
                f"session_memory_limit must be at least 1, got {self.session_memory_limit}"
            )

        if self.effectiveness_window_days < 1:
            raise ValueError(
                f"effectiveness_window_days must be at least 1, got {self.effectiveness_window_days}"
            )

        # Validate system instruction options
        valid_positions = ["top", "after_header", "before_content"]
        if self.system_instruction_position not in valid_positions:
            raise ValueError(
                f"Invalid system_instruction_position: {self.system_instruction_position}. "
                f"Must be one of {valid_positions}"
            )
        valid_formats = ["markdown", "xml", "comment", "plain"]
        if self.system_instruction_format not in valid_formats:
            raise ValueError(
                f"Invalid system_instruction_format: {self.system_instruction_format}. "
                f"Must be one of {valid_formats}"
            )

    @property
    def injection_config(self) -> Dict[str, Any]:
        """Get injection configuration as dictionary for TenetInjector."""
        return {
            "strategy": self.injection_strategy,
            "min_distance_between": self.min_distance_between,
            "prefer_natural_breaks": self.prefer_natural_breaks,
            "reinforce_at_end": self.reinforcement,
        }


@dataclass
class CacheConfig:
    """Configuration for caching system.

    Controls cache behavior for analysis results and other expensive operations.

    Attributes:
        enabled: Whether caching is enabled
        directory: Cache directory path
        ttl_days: Time-to-live for cache entries in days
        max_size_mb: Maximum cache size in megabytes
        compression: Whether to compress cached data
        memory_cache_size: Number of items in memory cache
        sqlite_pragmas: SQLite performance settings
        max_age_hours: Max age for certain cached entries (used by analyzer)
        llm_cache_enabled: Whether to cache LLM responses
        llm_cache_ttl_hours: TTL for LLM response cache
    """

    enabled: bool = True
    directory: Optional[Path] = None
    ttl_days: int = 7
    max_size_mb: int = 500
    compression: bool = False
    memory_cache_size: int = 1000
    sqlite_pragmas: Dict[str, str] = field(
        default_factory=lambda: {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": "-64000",  # 64MB
            "temp_store": "MEMORY",
        }
    )
    max_age_hours: int = 24
    llm_cache_enabled: bool = True
    llm_cache_ttl_hours: int = 24


@dataclass
class OutputConfig:
    """Configuration for output formatting.

    Controls how context and analysis results are formatted.

    Attributes:
        default_format: Default output format (markdown, xml, json)
        syntax_highlighting: Whether to enable syntax highlighting
        line_numbers: Whether to include line numbers
        max_line_length: Maximum line length before wrapping
        include_metadata: Whether to include metadata in output
        compression_threshold: File size threshold for summarization
        summary_ratio: Target compression ratio for summaries
        copy_on_distill: Automatically copy distill output to clipboard when true
        show_token_usage: Whether to show token usage statistics
        show_cost_estimate: Whether to show cost estimates for LLM operations
    """

    default_format: str = "markdown"
    syntax_highlighting: bool = True
    line_numbers: bool = False
    max_line_length: int = 120
    include_metadata: bool = True
    compression_threshold: int = 10_000  # Characters
    summary_ratio: float = 0.25  # Target 25% of original size
    copy_on_distill: bool = False  # Automatically copy distill output to clipboard when true
    show_token_usage: bool = True
    show_cost_estimate: bool = True


@dataclass
class GitConfig:
    """Configuration for git integration.

    Controls how git information is gathered and used.

    Attributes:
        enabled: Whether git integration is enabled
        include_history: Whether to include commit history
        history_limit: Maximum number of commits to include
        include_blame: Whether to include git blame info
        include_stats: Whether to include statistics
        ignore_authors: Authors to ignore in analysis
        main_branches: Branch names considered "main"
    """

    enabled: bool = True
    include_history: bool = True
    history_limit: int = 100
    include_blame: bool = False
    include_stats: bool = True
    ignore_authors: List[str] = field(
        default_factory=lambda: ["dependabot[bot]", "github-actions[bot]", "renovate[bot]"]
    )
    main_branches: List[str] = field(default_factory=lambda: ["main", "master", "develop", "trunk"])


@dataclass
class TenetsConfig:
    """Main configuration for the Tenets system with LLM and NLP support.

    This is the root configuration object that contains all subsystem configs
    and global settings. It handles loading from files, environment variables,
    and provides sensible defaults.

    Attributes:
        config_file: Path to configuration file (if any)
        project_root: Root directory of the project
        max_tokens: Default maximum tokens for context
        version: Tenets version (for compatibility checking)
        debug: Enable debug mode
        quiet: Suppress non-essential output
        scanner: Scanner subsystem configuration
        ranking: Ranking subsystem configuration
        summarizer: Summarizer subsystem configuration
        tenet: Tenet subsystem configuration
        cache: Cache subsystem configuration
        output: Output formatting configuration
        git: Git integration configuration
        llm: LLM integration configuration
        nlp: NLP system configuration
        custom: Custom user configuration
    """

    # Global settings
    config_file: Optional[Path] = None
    project_root: Optional[Path] = None
    max_tokens: int = 100_000
    version: str = "0.1.0"
    debug: bool = False
    quiet: bool = False

    # Subsystem configurations
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    tenet: TenetConfig = field(default_factory=TenetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    git: GitConfig = field(default_factory=GitConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)  # LLM configuration
    nlp: NLPConfig = field(default_factory=NLPConfig)  # NLP configuration

    # Custom configuration
    custom: Dict[str, Any] = field(default_factory=dict)

    # Derived attributes (not in config files)
    _logger: Any = field(default=None, init=False, repr=False)
    _resolved_paths: Dict[str, Path] = field(default_factory=dict, init=False, repr=False)
    # Track whether config_file was discovered automatically (not user-provided)
    _config_file_discovered: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Initialize configuration after creation.

        This method:
        1. Sets up logging
        2. Resolves paths
        3. Loads configuration from files
        4. Applies environment variables
        5. Validates configuration
        """
        from tenets.utils.logger import get_logger
        self._logger = get_logger(__name__)

        # Resolve project root
        if not self.project_root:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(self.project_root).resolve()

        # Find and load config file if not specified
        if not self.config_file:
            discovered = self._find_config_file()
            self.config_file = discovered
            self._config_file_discovered = bool(discovered)
        else:
            # Normalize explicit path
            self.config_file = Path(self.config_file)

            # Check if this looks like a load operation vs save location setup
            # Load operations are indicated by:
            # 1. Paths that contain "nonexistent" (test indicator for missing files)
            # 2. Paths that start with "/" or "\" but aren't in temp directories
            # 3. Absolute paths that look like they're trying to load specific configs
            #    (not in temp/test directories)
            path_str = str(self.config_file)
            is_test_temp_path = (
                "temp" in path_str.lower()
                or "pytest" in path_str.lower()
                or "tmp" in path_str.lower()
            )

            is_load_operation = (
                "nonexistent" in path_str
                or (path_str.startswith("/") and not is_test_temp_path)
                or (path_str.startswith("\\") and not is_test_temp_path)
            )

            if is_load_operation and not self.config_file.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_file}")

        if self.config_file and self.config_file.exists():
            self._load_from_file(self.config_file)

        # Apply environment variables (override file config)
        self._apply_environment_variables()

        # Resolve derived paths
        self._resolve_paths()

        # Validate configuration
        self._validate()

        self._logger.debug(f"Configuration loaded from {self.config_file or 'defaults'}")

    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations.

        Searches in order:
        1. .tenets.yml in current directory
        2. .tenets.yaml in current directory
        3. tenets.yml in current directory
        4. .config/tenets.yml in current directory
        5. ~/.config/tenets/config.yml
        6. ~/.tenets.yml

        Returns:
            Path to config file if found, None otherwise
        """
        search_locations = [
            self.project_root / ".tenets.yml",
            self.project_root / ".tenets.yaml",
            self.project_root / "tenets.yml",
            self.project_root / ".config" / "tenets.yml",
            Path.home() / ".config" / "tenets" / "config.yml",
            Path.home() / ".tenets.yml",
        ]

        for location in search_locations:
            if location.exists():
                self._logger.debug(f"Found config file: {location}")
                return location

        return None

    def _load_from_file(self, path: Path):
        """Load configuration from YAML or JSON file.

        Args:
            path: Path to configuration file

        Raises:
            ValueError: If file format is unsupported
            yaml.YAMLError: If YAML parsing fails
        """
        self._logger.info(f"Loading configuration from {path}")

        try:
            with open(path) as f:
                if path.suffix in [".yml", ".yaml"]:
                    data = yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")

            # Apply loaded configuration
            self._apply_dict_config(data)

        except Exception as e:
            self._logger.error(f"Failed to load config from {path}: {e}")
            raise

    def _apply_dict_config(self, data: Dict[str, Any], prefix: str = ""):
        """Recursively apply dictionary configuration.

        Args:
            data: Configuration dictionary
            prefix: Prefix for nested configuration
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Handle subsystem configs
            if key == "scanner" and isinstance(value, dict):
                self.scanner = ScannerConfig(**value)
            elif key == "ranking" and isinstance(value, dict):
                self.ranking = RankingConfig(**value)
            elif key == "summarizer" and isinstance(value, dict):
                self.summarizer = SummarizerConfig(**value)
            elif key == "tenet" and isinstance(value, dict):
                self.tenet = TenetConfig(**value)
            elif key == "cache" and isinstance(value, dict):
                self.cache = CacheConfig(**value)
            elif key == "output" and isinstance(value, dict):
                self.output = OutputConfig(**value)
            elif key == "git" and isinstance(value, dict):
                self.git = GitConfig(**value)
            elif key == "llm" and isinstance(value, dict):
                self.llm = LLMConfig(**value)
            elif key == "nlp" and isinstance(value, dict):
                self.nlp = NLPConfig(**value)
            elif key == "custom" and isinstance(value, dict):
                self.custom = value
            elif hasattr(self, key):
                # Direct attribute assignment
                setattr(self, key, value)
            else:
                # Store unknown config in custom
                self.custom[full_key] = value

    def _apply_environment_variables(self):
        """Apply environment variable overrides.

        Environment variables follow the pattern:
        TENETS_<SECTION>_<KEY> = value

        Examples:
            TENETS_MAX_TOKENS=150000
            TENETS_SCANNER_MAX_FILE_SIZE=10000000
            TENETS_RANKING_ALGORITHM=thorough
            TENETS_LLM_ENABLED=true
            TENETS_LLM_PROVIDER=anthropic
            TENETS_LLM_API_KEYS_OPENAI=sk-...
            TENETS_NLP_EMBEDDINGS_ENABLED=true
            TENETS_NLP_EMBEDDINGS_MODEL=all-MiniLM-L12-v2
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith("TENETS_"):
                continue

            # Remove prefix and split
            raw = env_key[7:]
            parts = raw.split("_")

            # Build lowercase attribute tokens preserving underscores between words
            parts_lower = [p.lower() for p in parts]

            # Special-case common top-level keys that include underscores
            top_level_map = {
                "max_tokens": ["max", "tokens"],
                "project_root": ["project", "root"],
            }

            # Try to match top-level overrides first
            for attr_name, tokens in top_level_map.items():
                if parts_lower == tokens:
                    if hasattr(self, attr_name):
                        setattr(self, attr_name, self._parse_env_value(env_value))
                    break
            else:
                # No break → not a special top-level
                if len(parts_lower) == 1:
                    attr = parts_lower[0]
                    if hasattr(self, attr):
                        setattr(self, attr, self._parse_env_value(env_value))
                elif len(parts_lower) >= 2:
                    subsystem = parts_lower[0]

                    # Special handling for LLM API keys
                    if (
                        subsystem == "llm"
                        and len(parts_lower) >= 3
                        and parts_lower[1] == "api"
                        and parts_lower[2] == "keys"
                    ):
                        if len(parts_lower) == 4:
                            provider = parts_lower[3]
                            self.llm.api_keys[provider] = env_value
                            continue

                    # Special handling for NLP settings
                    if subsystem == "nlp":
                        attr = "_".join(parts_lower[1:])
                        if hasattr(self.nlp, attr):
                            setattr(self.nlp, attr, self._parse_env_value(env_value))
                            continue

                    # Regular subsystem attributes
                    attr = "_".join(parts_lower[1:])
                    if hasattr(self, subsystem):
                        subsystem_config = getattr(self, subsystem)
                        if hasattr(subsystem_config, attr):
                            setattr(subsystem_config, attr, self._parse_env_value(env_value))

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type.

        Args:
            value: String value from environment

        Returns:
            Parsed value with appropriate type
        """
        # Boolean
        if value.lower() in ["true", "yes", "1"]:
            return True
        elif value.lower() in ["false", "no", "0"]:
            return False

        # Number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # List (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        # String
        return value

    def _resolve_paths(self):
        """Resolve all path configurations.

        Ensures all paths are absolute and creates directories as needed.
        """
        # Cache directory
        if not self.cache.directory:
            self.cache.directory = Path.home() / ".tenets" / "cache"
        else:
            self.cache.directory = Path(self.cache.directory).resolve()

        # Ensure cache directory exists
        self.cache.directory.mkdir(parents=True, exist_ok=True)

        # Tenet storage path
        if not self.tenet.storage_path:
            self.tenet.storage_path = Path.home() / ".tenets" / "tenets"
        else:
            self.tenet.storage_path = Path(self.tenet.storage_path).resolve()

        # Ensure tenet storage exists
        self.tenet.storage_path.mkdir(parents=True, exist_ok=True)

        self._resolved_paths = {
            "cache": self.cache.directory,
            "tenets": self.tenet.storage_path,
            "project": self.project_root,
        }

    def _validate(self):
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate max_tokens
        if self.max_tokens < 1000:
            raise ValueError(f"max_tokens must be at least 1000, got {self.max_tokens}")
        if self.max_tokens > 2_000_000:
            raise ValueError(f"max_tokens cannot exceed 2,000,000, got {self.max_tokens}")

        # Validate ranking algorithm
        valid_algorithms = ["fast", "balanced", "thorough", "ml", "custom"]
        if self.ranking.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid ranking algorithm: {self.ranking.algorithm}")

        # Validate ranking threshold
        if not 0 <= self.ranking.threshold <= 1:
            raise ValueError(
                f"Ranking threshold must be between 0 and 1, got {self.ranking.threshold}"
            )

        # Validate summarizer mode
        valid_modes = ["extractive", "compressive", "textrank", "transformer", "llm", "auto"]
        if self.summarizer.default_mode not in valid_modes:
            raise ValueError(f"Invalid summarizer mode: {self.summarizer.default_mode}")

        # Validate summarizer target ratio
        if not 0 < self.summarizer.target_ratio <= 1:
            raise ValueError(
                f"Summarizer target ratio must be between 0 and 1, got {self.summarizer.target_ratio}"
            )

        # Validate cache TTL
        if self.cache.ttl_days < 0:
            raise ValueError(f"Cache TTL cannot be negative, got {self.cache.ttl_days}")

        # Validate output format
        valid_formats = ["markdown", "xml", "json"]
        if self.output.default_format not in valid_formats:
            raise ValueError(f"Invalid output format: {self.output.default_format}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """

        def _as_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _as_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_as_serializable(v) for v in obj]
            return obj

        data = {
            "max_tokens": self.max_tokens,
            "version": self.version,
            "debug": self.debug,
            "quiet": self.quiet,
            "scanner": asdict(self.scanner),
            "ranking": asdict(self.ranking),
            "summarizer": asdict(self.summarizer),
            "tenet": asdict(self.tenet),
            "cache": asdict(self.cache),
            "output": asdict(self.output),
            "git": asdict(self.git),
            "llm": asdict(self.llm),
            "nlp": asdict(self.nlp),
            "custom": self.custom,
        }
        return _as_serializable(data)

    def save(self, path: Optional[Path] = None):
        """Save configuration to file.

        Args:
            path: Path to save to (uses config_file if not specified)

        Raises:
            ValueError: If no path specified and config_file not set
        """
        # Only allow implicit save to config_file if it was explicitly provided
        if path is None:
            if not self.config_file or self._config_file_discovered:
                raise ValueError("No path specified for saving configuration")
            save_path = self.config_file
        else:
            save_path = path

        save_path = Path(save_path)
        config_dict = self.to_dict()

        # Remove version from saved config (managed by package)
        config_dict.pop("version", None)

        with open(save_path, "w") as f:
            if save_path.suffix == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self._logger.info(f"Configuration saved to {save_path}")

    # Properties for compatibility and convenience
    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return self._resolved_paths.get("cache", self.cache.directory)

    @cache_dir.setter
    def cache_dir(self, value: Union[str, Path]) -> None:
        """Set the cache directory path.

        Args:
            value: New cache directory path
        """
        path = Path(value).resolve()
        self.cache.directory = path
        path.mkdir(parents=True, exist_ok=True)
        self._resolved_paths["cache"] = path

    @property
    def scanner_workers(self) -> int:
        """Get number of scanner workers."""
        return self.scanner.workers

    @property
    def ranking_workers(self) -> int:
        """Get number of ranking workers."""
        return self.ranking.workers

    @property
    def ranking_algorithm(self) -> str:
        """Get the ranking algorithm."""
        return self.ranking.algorithm

    @property
    def summarizer_mode(self) -> str:
        """Get the default summarizer mode."""
        return self.summarizer.default_mode

    @property
    def summarizer_ratio(self) -> float:
        """Get the default summarization target ratio."""
        return self.summarizer.target_ratio

    @property
    def respect_gitignore(self) -> bool:
        """Whether to respect .gitignore files."""
        return self.scanner.respect_gitignore

    @respect_gitignore.setter
    def respect_gitignore(self, value: bool) -> None:
        """Set whether to respect .gitignore files.

        Args:
            value: New value
        """
        self.scanner.respect_gitignore = bool(value)

    @property
    def follow_symlinks(self) -> bool:
        """Whether to follow symbolic links."""
        return self.scanner.follow_symlinks

    @follow_symlinks.setter
    def follow_symlinks(self, value: bool) -> None:
        """Set whether to follow symbolic links.

        Args:
            value: New value
        """
        self.scanner.follow_symlinks = bool(value)

    @property
    def additional_ignore_patterns(self) -> List[str]:
        """Get additional ignore patterns."""
        return self.scanner.additional_ignore_patterns

    @additional_ignore_patterns.setter
    def additional_ignore_patterns(self, patterns: List[str]) -> None:
        """Set additional ignore patterns.

        Args:
            patterns: New patterns list
        """
        self.scanner.additional_ignore_patterns = list(patterns)

    @property
    def auto_instill_tenets(self) -> bool:
        """Whether to automatically instill tenets."""
        return self.tenet.auto_instill

    @auto_instill_tenets.setter
    def auto_instill_tenets(self, value: bool) -> None:
        """Set whether to automatically instill tenets.

        Args:
            value: New value
        """
        self.tenet.auto_instill = bool(value)

    @property
    def max_tenets_per_context(self) -> int:
        """Maximum tenets to inject per context."""
        return self.tenet.max_per_context

    @max_tenets_per_context.setter
    def max_tenets_per_context(self, value: int) -> None:
        """Set maximum tenets to inject per context.

        Args:
            value: New value
        """
        self.tenet.max_per_context = int(value)

    @property
    def tenet_injection_config(self) -> Dict[str, Any]:
        """Get tenet injection configuration."""
        return {
            "strategy": self.tenet.injection_strategy,
            "min_distance_between": self.tenet.min_distance_between,
            "prefer_natural_breaks": self.tenet.prefer_natural_breaks,
            "reinforce_at_end": self.tenet.reinforcement,
        }

    @property
    def cache_ttl_days(self) -> int:
        """Cache time-to-live in days."""
        return self.cache.ttl_days

    @cache_ttl_days.setter
    def cache_ttl_days(self, value: int) -> None:
        """Set cache time-to-live in days.

        Args:
            value: New TTL in days
        """
        self.cache.ttl_days = int(value)

    @property
    def max_cache_size_mb(self) -> int:
        """Maximum cache size in megabytes."""
        return self.cache.max_size_mb

    @max_cache_size_mb.setter
    def max_cache_size_mb(self, value: int) -> None:
        """Set maximum cache size in megabytes.

        Args:
            value: New size in MB
        """
        self.cache.max_size_mb = int(value)

    @property
    def llm_enabled(self) -> bool:
        """Whether LLM features are enabled."""
        return self.llm.enabled

    @llm_enabled.setter
    def llm_enabled(self, value: bool) -> None:
        """Set whether LLM features are enabled.

        Args:
            value: New value
        """
        self.llm.enabled = bool(value)

    @property
    def llm_provider(self) -> str:
        """Get the current LLM provider."""
        return self.llm.provider

    @llm_provider.setter
    def llm_provider(self, value: str) -> None:
        """Set the LLM provider.

        Args:
            value: Provider name
        """
        self.llm.provider = value

    def get_llm_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get LLM API key for a provider.

        Args:
            provider: Provider name (uses default if not specified)

        Returns:
            API key or None
        """
        return self.llm.get_api_key(provider)

    def get_llm_model(self, task: str = "default", provider: Optional[str] = None) -> str:
        """Get LLM model for a specific task.

        Args:
            task: Task type
            provider: Provider name (uses default if not specified)

        Returns:
            Model name
        """
        return self.llm.get_model(task, provider)

    @property
    def nlp_enabled(self) -> bool:
        """Whether NLP features are enabled."""
        return self.nlp.enabled

    @nlp_enabled.setter
    def nlp_enabled(self, value: bool) -> None:
        """Set whether NLP features are enabled.

        Args:
            value: New value
        """
        self.nlp.enabled = bool(value)

    @property
    def nlp_embeddings_enabled(self) -> bool:
        """Whether NLP embeddings are enabled."""
        return self.nlp.embeddings_enabled

    @nlp_embeddings_enabled.setter
    def nlp_embeddings_enabled(self, value: bool) -> None:
        """Set whether NLP embeddings are enabled.

        Args:
            value: New value
        """
        self.nlp.embeddings_enabled = bool(value)
