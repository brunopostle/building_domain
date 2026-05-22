"""FakeLLMProvider for integration tests.

Dispatch table key: (type[BaseModel], entity_name: str)
- entity_name is REQUIRED; raises ValueError if None (catches pipeline code that omits it)
- Returns minimal empty response when key is absent (entity not in fixture set)
"""
from typing import Any
from pydantic import BaseModel


_EMPTY_RESPONSES: dict[type, BaseModel] = {}


def _minimal_response(schema: type[BaseModel]) -> BaseModel:
    """Return a minimal valid empty response for any schema."""
    data: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        ann = field.annotation
        origin = getattr(ann, "__origin__", None)
        if origin is list:
            data[name] = []
        elif ann is float or ann == "float":
            data[name] = 0.5
        elif ann is bool or ann == "bool":
            data[name] = False
        elif ann is int or ann == "int":
            data[name] = 0
        elif ann is str or ann == "str":
            data[name] = ""
        else:
            data[name] = None
    try:
        return schema.model_validate(data)
    except Exception:
        return schema.model_construct(**data)


class FakeLLMProvider:
    """Test double for LLMProvider.

    Dispatch table: dict[(type[BaseModel], entity_name)] → BaseModel instance
    """

    def __init__(self, responses: dict[tuple[type[BaseModel], str], BaseModel] | None = None):
        self._responses: dict[tuple[type[BaseModel], str], BaseModel] = responses or {}
        self._model = "fake-model"

    @property
    def model_id(self) -> str:
        return self._model

    def register(self, schema: type[BaseModel], entity_name: str, response: BaseModel) -> None:
        self._responses[(schema, entity_name)] = response

    def extract(self, prompt: str, schema: type[BaseModel], *, entity_name: str | None = None) -> BaseModel:
        if entity_name is None:
            raise ValueError(
                "FakeLLMProvider requires entity_name to be set on every extract() call. "
                "This catches pipeline code that omits the parameter."
            )
        key = (schema, entity_name)
        if key in self._responses:
            return self._responses[key]
        return _minimal_response(schema)

    def classify(self, prompt: str, options: list[str]) -> str:
        return options[0]
