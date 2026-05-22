import tempfile
import os
from bsos.llm.cache import LLMResponseCache, prompt_hash


def test_cache_miss_returns_none():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMResponseCache(os.path.join(d, "test.db"))
        assert cache.get("gpt-4", "hello") is None


def test_cache_put_and_get():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMResponseCache(os.path.join(d, "test.db"))
        cache.put("gpt-4", "hello", {"answer": 42})
        result = cache.get("gpt-4", "hello")
        assert result == {"answer": 42}


def test_cache_keyed_by_model():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMResponseCache(os.path.join(d, "test.db"))
        cache.put("gpt-4", "hello", {"model": "gpt4"})
        cache.put("gpt-3.5", "hello", {"model": "gpt35"})
        assert cache.get("gpt-4", "hello") == {"model": "gpt4"}
        assert cache.get("gpt-3.5", "hello") == {"model": "gpt35"}


def test_cache_put_replaces_existing():
    with tempfile.TemporaryDirectory() as d:
        cache = LLMResponseCache(os.path.join(d, "test.db"))
        cache.put("gpt-4", "hello", {"v": 1})
        cache.put("gpt-4", "hello", {"v": 2})
        assert cache.get("gpt-4", "hello") == {"v": 2}


def test_prompt_hash_is_deterministic():
    assert prompt_hash("test prompt") == prompt_hash("test prompt")


def test_prompt_hash_differs_for_different_prompts():
    assert prompt_hash("prompt a") != prompt_hash("prompt b")
