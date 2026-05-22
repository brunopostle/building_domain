"""Pydantic schemas for LLM structured output — separate from persistence models.

Each pass uses a distinct schema class so FakeLLMProvider can dispatch on type[BaseModel].
"""
from typing import Literal
from pydantic import BaseModel


class DiscoveredConcept(BaseModel):
    name: str
    entity_type: Literal["component", "system", "space", "material", "activity"]
    description: str = ""


class ConceptDiscoveryResponse(BaseModel):
    """Pass 1 bootstrap / free-text seed discovery."""
    concepts: list[DiscoveredConcept]


class ConceptExpansionResponse(BaseModel):
    """Pass 1 sub-concept expansion for one top-level concept."""
    sub_concepts: list[DiscoveredConcept]
