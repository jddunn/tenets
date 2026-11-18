"""The request fields the summarizer sends each OpenAI model."""

from tenets.core.summarizer.llm import _openai_sampling


def test_gpt6_luna_runs_without_a_reasoning_step_and_keeps_temperature():
    assert _openai_sampling("gpt-6-luna", 0.3, 500) == {
        "reasoning_effort": "none",
        "temperature": 0.3,
        "max_completion_tokens": 500,
    }


def test_gpt62_gets_a_completion_cap_and_no_temperature():
    assert _openai_sampling("gpt-6.2-sol", 0.3, 500) == {
        "reasoning_effort": "low",
        "max_completion_tokens": 500,
    }


def test_an_openrouter_slug_is_read_by_its_model_name():
    assert "max_tokens" not in _openai_sampling("openai/gpt-6.2-sol", 0.3, 500)


def test_a_chat_model_keeps_max_tokens_and_temperature():
    assert _openai_sampling("gpt-4o-mini", 0.3, 500) == {"temperature": 0.3, "max_tokens": 500}
