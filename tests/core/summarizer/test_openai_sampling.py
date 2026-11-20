"""The request fields the summarizer sends each OpenAI model."""

from tenets.core.summarizer.llm import REASONING_HEADROOM, _openai_sampling
from tenets.models.llm import get_model_limits, get_model_pricing


def test_gpt6_luna_runs_without_a_reasoning_step_and_keeps_temperature() -> None:
    assert _openai_sampling("gpt-6-luna", 0.3, 500) == {
        "reasoning_effort": "none",
        "temperature": 0.3,
        "max_completion_tokens": 500,
    }


def test_gpt62_gets_a_completion_cap_and_no_temperature() -> None:
    assert _openai_sampling("gpt-6.2-sol", 0.3, 500) == {
        "reasoning_effort": "low",
        "max_completion_tokens": 500 + REASONING_HEADROOM,
    }


def test_an_openrouter_slug_is_read_by_its_model_name() -> None:
    assert "max_tokens" not in _openai_sampling("openai/gpt-6.2-sol", 0.3, 500)


def test_a_chat_model_keeps_max_tokens_and_temperature() -> None:
    assert _openai_sampling("gpt-4o-mini", 0.3, 500) == {"temperature": 0.3, "max_tokens": 500}


def test_an_openrouter_slug_gets_its_model_pricing_and_limits() -> None:
    assert get_model_pricing("openai/gpt-6.2-sol") == get_model_pricing("gpt-6.2-sol")
    assert get_model_pricing("openai/gpt-6.2-sol").input_per_1k == 0.002
    assert get_model_limits("openai/gpt-6-luna") == get_model_limits("gpt-6-luna")
