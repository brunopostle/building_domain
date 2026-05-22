"""FakeLLMProvider dispatch table for tests."""
from pydantic import BaseModel


class FakeLLMProvider:
    """Test double for LLMProvider. Register (prompt_fragment, schema) → response pairs."""

    def __init__(self, responses: dict | None = None):
        self._responses: dict = responses or {}
        self._model = "fake-model"

    @property
    def model_id(self) -> str:
        return self._model

    def register(self, prompt_fragment: str, schema: type[BaseModel], response: BaseModel) -> None:
        self._responses[(prompt_fragment, schema)] = response

    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel:
        for (fragment, s), response in self._responses.items():
            if fragment in prompt and s is schema:
                return response
        raise ValueError(f"No fake response registered for prompt fragment in: {prompt!r}")

    def classify(self, prompt: str, options: list[str]) -> str:
        return options[0]
