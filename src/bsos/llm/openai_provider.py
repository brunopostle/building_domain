"""OpenAI-compatible LLMProvider implementation."""
from pydantic import BaseModel


class OpenAIProvider:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key

    @property
    def model_id(self) -> str:
        return self._model

    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel:
        raise NotImplementedError

    def classify(self, prompt: str, options: list[str]) -> str:
        raise NotImplementedError
