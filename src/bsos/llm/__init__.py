"""LLM provider factory."""
from bsos.llm.cache import LLMResponseCache
from bsos.llm.openai_provider import OpenAIProvider
from bsos.llm.ollama_provider import OllamaProvider, DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from bsos.llm.anthropic_provider import AnthropicProvider


def make_provider(
    model_id: str,
    *,
    cache: LLMResponseCache | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    ollama_base_url: str = DEFAULT_BASE_URL,
) -> OpenAIProvider | OllamaProvider | AnthropicProvider:
    """Return the appropriate LLMProvider for model_id.

    Routing:
    - 'ollama/<model>' → OllamaProvider (local, no key needed)
    - 'claude-*'       → AnthropicProvider (reads ANTHROPIC_API_KEY)
    - anything else    → OpenAIProvider (reads OPENAI_API_KEY)
    """
    if model_id.startswith("ollama/"):
        ollama_model = model_id[len("ollama/"):]
        return OllamaProvider(ollama_model, base_url=ollama_base_url, timeout=timeout, cache=cache)
    if model_id.startswith("claude-"):
        return AnthropicProvider(model_id, timeout=timeout, cache=cache)
    return OpenAIProvider(model_id, timeout=timeout, cache=cache)
