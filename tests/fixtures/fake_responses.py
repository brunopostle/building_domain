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


def build_standard_fixture() -> "FakeLLMProvider":
    """Return a fully-configured FakeLLMProvider for standard pipeline tests.

    Entities in the fixture knowledge base:
      Roof (component), Precipitation (component), Formwork (activity),
      Concrete pouring (activity), Roof membrane (material),
      Roof covering (material — near-duplicate of Roof, merged in Pass 2).

    Pass 3 response highlights:
      - Roof protects_from Precipitation  (main roof assertion)
      - Roof requires Roof membrane       (structural dependency)
      - Concrete pouring depends_on Formwork  (process relation)
      - Concrete pouring conflicts_with Formwork  (contradictory pair)
    """
    from bsos.pipeline.schemas import (
        AssertionExtractionResponse, ConceptDiscoveryResponse, ConceptExpansionResponse,
        DiscoveredConcept, ExtractedAssertion,
    )

    p = FakeLLMProvider()

    # ------------------------------------------------------------------
    # Pass 1: bootstrap discovery
    # ------------------------------------------------------------------
    p.register(
        ConceptDiscoveryResponse,
        "__bootstrap__",
        ConceptDiscoveryResponse(concepts=[
            DiscoveredConcept(name="Roof", entity_type="component",
                              description="Top external enclosure of a building"),
            DiscoveredConcept(name="Precipitation", entity_type="component",
                              description="Rain, snow, or hail acting on the building"),
            DiscoveredConcept(name="Formwork", entity_type="activity",
                              description="Temporary mould into which concrete is poured"),
            DiscoveredConcept(name="Concrete pouring", entity_type="activity",
                              description="Placing wet concrete into formwork"),
        ]),
    )

    # Pass 1: expansions
    p.register(
        ConceptExpansionResponse,
        "Roof",
        ConceptExpansionResponse(sub_concepts=[
            DiscoveredConcept(name="Roof membrane", entity_type="material",
                              description="Waterproof layer applied to roof deck"),
            DiscoveredConcept(name="Roof covering", entity_type="material",
                              description="Outer protective layer of a roof"),  # near-dup of Roof
        ]),
    )
    # Other expansions return empty (FakeLLMProvider fallback handles this)

    # ------------------------------------------------------------------
    # Pass 3: relationship extraction (same response for all 3 framings)
    # ------------------------------------------------------------------
    p.register(
        AssertionExtractionResponse,
        "Roof",
        AssertionExtractionResponse(assertions=[
            ExtractedAssertion(
                predicate="protects_from",
                object_name="Precipitation",
                knowledge_origin="physical",
                confidence=0.92,
                rationale="Roof is primary weather barrier",
            ),
            ExtractedAssertion(
                predicate="requires",
                object_name="Roof membrane",
                knowledge_origin="engineering",
                confidence=0.85,
                rationale="Membrane provides waterproofing",
            ),
        ]),
    )

    p.register(
        AssertionExtractionResponse,
        "Concrete pouring",
        AssertionExtractionResponse(assertions=[
            ExtractedAssertion(
                predicate="depends_on",
                object_name="Formwork",
                knowledge_origin="engineering",
                confidence=0.95,
                rationale="Formwork must be in place before concrete is poured",
            ),
            # Contradictory pair: same entity pair, opposing predicates
            ExtractedAssertion(
                predicate="conflicts_with",
                object_name="Formwork",
                knowledge_origin="engineering",
                confidence=0.20,
                rationale="Contradictory low-confidence extraction — for conflict detection testing",
            ),
        ]),
    )

    return p
