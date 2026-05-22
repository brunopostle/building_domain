"""Ollama LLMProvider implementation with JSON schema fallback."""
from pydantic import BaseModel


class OllamaProvider:
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url
        self._schema_enforced: bool | None = None

    @property
    def model_id(self) -> str:
        return self._model

    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel:
        raise NotImplementedError

    def classify(self, prompt: str, options: list[str]) -> str:
        raise NotImplementedError
