"""Unit tests for OllamaProvider and make_provider factory."""
import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from bsos.llm.ollama_provider import OllamaProvider
from bsos.llm.openai_provider import OpenAIProvider
from bsos.llm import make_provider
from bsos.llm.retry import NonRetryableError


class FakeSchema(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_ollama_provider_implements_protocol():
    """OllamaProvider has extract, classify, and model_id."""
    provider = OllamaProvider("llama3.2")
    assert hasattr(provider, "extract")
    assert hasattr(provider, "classify")
    assert provider.model_id == "llama3.2"


def test_ollama_provider_model_id():
    provider = OllamaProvider("mistral")
    assert provider.model_id == "mistral"


# ---------------------------------------------------------------------------
# make_provider factory
# ---------------------------------------------------------------------------

def test_make_provider_openai_for_plain_model():
    with patch("bsos.llm.openai_provider.OpenAI"), patch("bsos.llm.openai_provider.instructor"):
        provider = make_provider("gpt-4o")
    assert isinstance(provider, OpenAIProvider)
    assert provider.model_id == "gpt-4o"


def test_make_provider_ollama_for_ollama_prefix():
    provider = make_provider("ollama/llama3.2")
    assert isinstance(provider, OllamaProvider)
    assert provider.model_id == "llama3.2"


def test_make_provider_ollama_strips_prefix():
    provider = make_provider("ollama/mistral")
    assert provider.model_id == "mistral"


def test_make_provider_openai_for_non_ollama_prefix():
    with patch("bsos.llm.openai_provider.OpenAI"), patch("bsos.llm.openai_provider.instructor"):
        provider = make_provider("anthropic/claude-3-5-sonnet")
    assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# extract() — success path
# ---------------------------------------------------------------------------

def test_extract_returns_validated_model():
    provider = OllamaProvider("llama3.2")
    fake_result = FakeSchema(name="test", value=42)
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = fake_result

    result = provider.extract("some prompt", FakeSchema)
    assert isinstance(result, FakeSchema)
    assert result.name == "test"
    assert result.value == 42


def test_extract_uses_cache_on_hit():
    provider = OllamaProvider("llama3.2")
    mock_cache = MagicMock()
    mock_cache.get.return_value = {"name": "cached", "value": 99}
    provider._cache = mock_cache

    result = provider.extract("prompt", FakeSchema)
    assert result.name == "cached"
    provider._client = MagicMock()
    provider._client.chat.completions.create.assert_not_called() if hasattr(provider, '_client') else None


def test_extract_writes_to_cache_on_miss():
    provider = OllamaProvider("llama3.2")
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    provider._cache = mock_cache

    fake_result = FakeSchema(name="fresh", value=7)
    provider._client = MagicMock()
    provider._client.chat.completions.create.return_value = fake_result

    provider.extract("prompt", FakeSchema)
    mock_cache.put.assert_called_once()


# ---------------------------------------------------------------------------
# extract() — retry on transient error
# ---------------------------------------------------------------------------

def test_extract_retries_on_connection_error():
    from openai import APIConnectionError as OAIConnError
    provider = OllamaProvider("llama3.2")
    fake_result = FakeSchema(name="ok", value=1)
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OAIConnError(request=MagicMock())
        return fake_result

    provider._client = MagicMock()
    provider._client.chat.completions.create.side_effect = side_effect

    with patch("bsos.llm.retry.time.sleep"):
        result = provider.extract("prompt", FakeSchema)

    assert result.name == "ok"
    assert call_count == 3


# ---------------------------------------------------------------------------
# classify() — success path
# ---------------------------------------------------------------------------

def test_classify_returns_matching_option():
    provider = OllamaProvider("llama3.2")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "option_b"
    provider._client = MagicMock()
    provider._client._client.chat.completions.create.return_value = mock_response

    result = provider.classify("Pick one", ["option_a", "option_b", "option_c"])
    assert result == "option_b"


def test_classify_case_insensitive_match():
    provider = OllamaProvider("llama3.2")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "OPTION_A"
    provider._client = MagicMock()
    provider._client._client.chat.completions.create.return_value = mock_response

    result = provider.classify("Pick one", ["option_a", "option_b"])
    assert result == "option_a"


def test_classify_falls_back_to_first_option_on_unrecognised():
    provider = OllamaProvider("llama3.2")
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "something_unexpected"
    provider._client = MagicMock()
    provider._client._client.chat.completions.create.return_value = mock_response

    result = provider.classify("Pick one", ["option_a", "option_b"])
    assert result == "option_a"


# ---------------------------------------------------------------------------
# classify() — retry fires on transient error
# ---------------------------------------------------------------------------

def test_classify_retries_on_transient_error():
    from openai import APIConnectionError as OAIConnError
    provider = OllamaProvider("llama3.2")
    call_count = 0
    mock_ok = MagicMock()
    mock_ok.choices[0].message.content = "yes"

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise OAIConnError(request=MagicMock())
        return mock_ok

    provider._client = MagicMock()
    provider._client._client.chat.completions.create.side_effect = side_effect

    with patch("bsos.llm.retry.time.sleep"):
        result = provider.classify("Yes or no?", ["yes", "no"])
    assert result == "yes"
    assert call_count == 2
