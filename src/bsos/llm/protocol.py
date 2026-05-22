"""LLMProvider protocol."""
from typing import Protocol
from pydantic import BaseModel


class LLMProvider(Protocol):
    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel: ...
    def classify(self, prompt: str, options: list[str]) -> str: ...

    @property
    def model_id(self) -> str: ...
