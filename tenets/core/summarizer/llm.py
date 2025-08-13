"""LLM-based summarization strategies.

This module provides integration with Large Language Models (LLMs) for
high-quality summarization. Supports OpenAI, Anthropic, and OpenRouter APIs.

NOTE: These strategies incur API costs. Use with caution and appropriate
rate limiting. Always check pricing before using in production.
"""

import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from tenets.utils.logger import get_logger


class LLMProvider(Enum):
    """Supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"  # For local models like Ollama


@dataclass
class LLMConfig:
    """Configuration for LLM summarization.
    
    Attributes:
        provider: LLM provider to use
        model: Model name/ID
        api_key: API key (if not in environment)
        base_url: Base URL for API (for custom endpoints)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        system_prompt: System prompt template
        user_prompt: User prompt template
        retry_attempts: Number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
    """
    
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 500
    system_prompt: str = """You are an expert at summarizing code and technical documentation. 
Your summaries are concise, accurate, and preserve critical technical details."""
    user_prompt: str = """Summarize the following text to approximately {target_percent}% of its original length. 
Focus on the most important information and maintain technical accuracy.

Text to summarize:
{text}

Summary:"""
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment.
        
        Returns:
            API key or None
        """
        if self.api_key:
            return self.api_key
            
        # Check environment variables
        env_vars = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
        }
        
        env_var = env_vars.get(self.provider)
        if env_var:
            return os.getenv(env_var)
            
        return None


class LLMSummarizer:
    """Base class for LLM-based summarization.
    
    Provides common functionality for different LLM providers.
    Handles API calls, retries, and error handling.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM summarizer.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        self.logger = get_logger(__name__)
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the API client for the provider."""
        api_key = self.config.get_api_key()
        
        if not api_key and self.config.provider != LLMProvider.LOCAL:
            self.logger.warning(
                f"No API key found for {self.config.provider.value}. "
                f"Set {self.config.provider.value.upper()}_API_KEY environment variable."
            )
            return
            
        if self.config.provider == LLMProvider.OPENAI:
            self._init_openai(api_key)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic(api_key)
        elif self.config.provider == LLMProvider.OPENROUTER:
            self._init_openrouter(api_key)
        elif self.config.provider == LLMProvider.LOCAL:
            self._init_local()
            
    def _init_openai(self, api_key: str):
        """Initialize OpenAI client."""
        try:
            import openai
            
            if hasattr(openai, 'OpenAI'):
                # New client style (>= 1.0)
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url
                )
            else:
                # Legacy style
                openai.api_key = api_key
                if self.config.base_url:
                    openai.api_base = self.config.base_url
                self.client = openai
                
            self.logger.info(f"Initialized OpenAI client with model {self.config.model}")
        except ImportError:
            self.logger.error("OpenAI library not installed. Run: pip install openai")
            
    def _init_anthropic(self, api_key: str):
        """Initialize Anthropic client."""
        try:
            import anthropic
            
            self.client = anthropic.Anthropic(
                api_key=api_key,
                base_url=self.config.base_url
            )
            self.logger.info(f"Initialized Anthropic client with model {self.config.model}")
        except ImportError:
            self.logger.error("Anthropic library not installed. Run: pip install anthropic")
            
    def _init_openrouter(self, api_key: str):
        """Initialize OpenRouter client."""
        # OpenRouter uses OpenAI-compatible API
        try:
            import openai
            
            if hasattr(openai, 'OpenAI'):
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url=self.config.base_url or "https://openrouter.ai/api/v1"
                )
            else:
                openai.api_key = api_key
                openai.api_base = self.config.base_url or "https://openrouter.ai/api/v1"
                self.client = openai
                
            self.logger.info(f"Initialized OpenRouter client with model {self.config.model}")
        except ImportError:
            self.logger.error("OpenAI library not installed. Run: pip install openai")
            
    def _init_local(self):
        """Initialize local model client (e.g., Ollama)."""
        # Could use requests or specific local model library
        self.logger.info("Local model mode - implement based on your setup")
        
    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Summarize text using LLM.
        
        Args:
            text: Text to summarize
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length
            custom_prompt: Custom prompt override
            
        Returns:
            Summarized text
            
        Raises:
            RuntimeError: If API call fails after retries
        """
        if not self.client:
            raise RuntimeError(f"No client initialized for {self.config.provider.value}")
            
        # Prepare prompt
        target_percent = int(target_ratio * 100)
        
        if custom_prompt:
            user_prompt = custom_prompt.format(
                text=text,
                target_percent=target_percent,
                max_length=max_length,
                min_length=min_length
            )
        else:
            user_prompt = self.config.user_prompt.format(
                text=text,
                target_percent=target_percent
            )
            
        # Add length constraints to prompt if specified
        if max_length:
            user_prompt += f"\nMaximum length: {max_length} characters"
        if min_length:
            user_prompt += f"\nMinimum length: {min_length} characters"
            
        # Make API call with retries
        for attempt in range(self.config.retry_attempts):
            try:
                summary = self._call_api(user_prompt)
                
                # Validate length constraints
                if max_length and len(summary) > max_length:
                    summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
                elif min_length and len(summary) < min_length:
                    # Request longer summary
                    user_prompt += f"\n\nThe summary is too short. Please provide more detail."
                    continue
                    
                return summary
                
            except Exception as e:
                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e}"
                )
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to summarize after {self.config.retry_attempts} attempts: {e}")
                    
        return text[:max_length] if max_length else text  # Fallback
        
    def _call_api(self, user_prompt: str) -> str:
        """Make API call to LLM provider.
        
        Args:
            user_prompt: User prompt
            
        Returns:
            Generated summary
        """
        if self.config.provider == LLMProvider.OPENAI:
            return self._call_openai(user_prompt)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(user_prompt)
        elif self.config.provider == LLMProvider.OPENROUTER:
            return self._call_openrouter(user_prompt)
        elif self.config.provider == LLMProvider.LOCAL:
            return self._call_local(user_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
            
    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API."""
        try:
            if hasattr(self.client, 'chat'):
                # New client style
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                return response.choices[0].message.content
            else:
                # Legacy style
                response = self.client.ChatCompletion.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
            
    def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                system=self.config.system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
            
    def _call_openrouter(self, user_prompt: str) -> str:
        """Call OpenRouter API (OpenAI-compatible)."""
        return self._call_openai(user_prompt)  # Same as OpenAI
        
    def _call_local(self, user_prompt: str) -> str:
        """Call local model."""
        # Implement based on your local setup (e.g., Ollama)
        raise NotImplementedError("Local model support not implemented")
        
    def estimate_cost(self, text: str) -> Dict[str, float]:
        """Estimate cost of summarization.
        
        Args:
            text: Text to summarize
            
        Returns:
            Dictionary with cost estimates
        """
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(text) // 4
        output_tokens = int(input_tokens * 0.3)  # Assume 30% compression
        
        # Pricing per 1K tokens (as of 2024)
        pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }
        
        model_pricing = pricing.get(self.config.model, {"input": 0.001, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }


class LLMSummaryStrategy:
    """LLM-based summarization strategy for use with Summarizer.
    
    Wraps LLMSummarizer to match the SummarizationStrategy interface.
    
    WARNING: This strategy incurs API costs. Always estimate costs before use.
    """
    
    name = "llm"
    description = "High-quality summarization using Large Language Models (costs $)"
    requires_ml = False  # But requires API access
    
    def __init__(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
    model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        """Initialize LLM strategy.
        
        Args:
            provider: LLM provider name or enum
            model: Model to use
            api_key: API key (if not in environment)
        """
        self.logger = get_logger(__name__)
        
        # Convert string to enum if needed
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
            
        # Create config
        config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key
        )
        
        # Initialize summarizer
        self.summarizer = LLMSummarizer(config)
        
        # Warn about costs
        self.logger.warning(
            f"LLM summarization enabled with {provider.value}/{model}. "
            f"This will incur API costs. Use estimate_cost() to check pricing."
        )
        
    def summarize(
        self,
        text: str,
        target_ratio: float = 0.3,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None
    ) -> str:
        """Summarize text using LLM.
        
        Args:
            text: Input text
            target_ratio: Target compression ratio
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            LLM-generated summary
        """
        return self.summarizer.summarize(
            text,
            target_ratio=target_ratio,
            max_length=max_length,
            min_length=min_length
        )
        
    def estimate_cost(self, text: str) -> Dict[str, float]:
        """Estimate cost for summarizing text.
        
        Args:
            text: Text to summarize
            
        Returns:
            Cost estimate dictionary
        """
        return self.summarizer.estimate_cost(text)


def create_llm_summarizer(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMSummaryStrategy:
    """Create an LLM summarizer with defaults.
    
    Args:
        provider: Provider name (openai, anthropic, openrouter)
        model: Model name (uses provider default if None)
        api_key: API key (uses environment if None)
        
    Returns:
        Configured LLMSummaryStrategy
        
    Example:
    >>> summarizer = create_llm_summarizer("openai", "gpt-4o-mini")
        >>> summary = summarizer.summarize(long_text, target_ratio=0.2)
    """
    # Default models for each provider
    default_models = {
    "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
    "openrouter": "openai/gpt-4o-mini",
        "local": "llama2"
    }
    
    if model is None:
        model = default_models.get(provider.lower(), "gpt-4o-mini")
        
    return LLMSummaryStrategy(
        provider=provider,
        model=model,
        api_key=api_key
    )