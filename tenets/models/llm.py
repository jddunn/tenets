"""LLM pricing and limits.

Centralized reference for model pricing and token limits. Values are
best-effort and can be overridden at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Dict as DictType
from typing import List, Optional

from tenets.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float


@dataclass(frozen=True)
class ModelLimits:
    max_context: int
    max_output: int


# Best-effort defaults; adjust as needed without strict guarantees.
_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "gpt-4o-mini": ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006),
    "gpt-4.1": ModelPricing(input_per_1k=0.005, output_per_1k=0.015),
    "gpt-4": ModelPricing(input_per_1k=0.03, output_per_1k=0.06),
    "gpt-3.5-turbo": ModelPricing(input_per_1k=0.001, output_per_1k=0.002),
    # Anthropic
    "claude-3-opus": ModelPricing(input_per_1k=0.015, output_per_1k=0.075),
    "claude-3-5-sonnet": ModelPricing(input_per_1k=0.003, output_per_1k=0.015),
    "claude-3-haiku": ModelPricing(input_per_1k=0.00025, output_per_1k=0.00125),
}

_LIMITS: Dict[str, ModelLimits] = {
    "gpt-4o": ModelLimits(max_context=128_000, max_output=4_096),
    "gpt-4o-mini": ModelLimits(max_context=128_000, max_output=4_096),
    "gpt-4.1": ModelLimits(max_context=128_000, max_output=4_096),
    "gpt-4": ModelLimits(max_context=8_192, max_output=2_048),
    "gpt-3.5-turbo": ModelLimits(max_context=16_385, max_output=1_024),
    "claude-3-opus": ModelLimits(max_context=200_000, max_output=4_096),
    "claude-3-5-sonnet": ModelLimits(max_context=200_000, max_output=4_096),
    "claude-3-haiku": ModelLimits(max_context=200_000, max_output=4_096),
}


def get_model_pricing(model: Optional[str]) -> ModelPricing:
    """Return pricing for a model or a conservative default.

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-opus"). If None or
            unknown, returns a zero-cost placeholder suitable for dry runs.

    Returns:
        ModelPricing: Pricing per 1K input and output tokens.
    """
    if not model:
        return ModelPricing(0.0, 0.0)
    return _PRICING.get(model, ModelPricing(0.0, 0.0))


def get_model_limits(model: Optional[str]) -> ModelLimits:
    """Return token limits for a model or a conservative default.

    Args:
        model: Model name. If None or unknown, returns a safe default budget.

    Returns:
        ModelLimits: Maximum context and output tokens.
    """
    if not model:
        return ModelLimits(max_context=100_000, max_output=4_096)
    return _LIMITS.get(model, ModelLimits(max_context=100_000, max_output=4_096))


def estimate_cost(input_tokens: int, output_tokens: int, model: Optional[str]) -> Dict[str, float]:
    """Estimate API cost for a given token usage and model.

    Args:
        input_tokens: Number of prompt/input tokens.
        output_tokens: Expected number of completion/output tokens.
        model: Target model name used to look up pricing.

    Returns:
        dict: Cost breakdown with keys: input_tokens, output_tokens,
        input_cost, output_cost, total_cost.
    """
    p = get_model_pricing(model)
    input_cost = (input_tokens / 1000.0) * p.input_per_1k
    output_cost = (output_tokens / 1000.0) * p.output_per_1k
    total = input_cost + output_cost
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }


def _infer_provider(model_name: str) -> str:
    name = model_name.lower()
    if name.startswith("gpt-"):
        return "OpenAI"
    if name.startswith("claude-"):
        return "Anthropic"
    return "Unknown"


def list_supported_models() -> List[DictType[str, object]]:
    """List known models with provider, limits, and pricing.

    Returns:
        A list of dicts: name, provider, context_tokens, input_price, output_price
    """
    out: List[DictType[str, object]] = []
    for name, pricing in _PRICING.items():
        limits = _LIMITS.get(name, get_model_limits(None))
        out.append(
            {
                "name": name,
                "provider": _infer_provider(name),
                "context_tokens": limits.max_context,
                "input_price": pricing.input_per_1k,
                "output_price": pricing.output_per_1k,
            }
        )
    # Sort by provider then name
    out.sort(key=lambda m: (m["provider"], m["name"]))
    return out


# Back-compat for CLI display
SUPPORTED_MODELS: List[DictType[str, object]] = list_supported_models()
