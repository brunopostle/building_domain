"""Pydantic schemas for LLM structured output — separate from persistence models.

Each pass uses a distinct schema class so FakeLLMProvider can dispatch on type[BaseModel].
"""
from typing import Literal
from pydantic import BaseModel, Field


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


# Pass 3 schemas

PredicateLiteral = Literal[
    "requires", "depends_on", "protects_from", "unsuitable_for",
    "improves", "conflicts_with", "contains", "connects_to", "supports",
]

KnowledgeOriginLiteral = Literal["physical", "engineering", "architectural", "cultural"]


class ExtractedAssertion(BaseModel):
    """One relationship extracted from a single LLM framing."""
    predicate: PredicateLiteral
    object_name: str = Field(description="Name of the related building concept")
    knowledge_origin: KnowledgeOriginLiteral = "engineering"
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    applicability: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = ""


class AssertionExtractionResponse(BaseModel):
    """Pass 3 extraction result for one entity framing."""
    assertions: list[ExtractedAssertion]


# Pass 4 schemas

class ExtractedSpatialRelation(BaseModel):
    """One spatial/topological relationship extracted for a single entity."""
    relation: str = Field(description="Spatial relation type (from vocabulary or free-text if unknown)")
    object_name: str = Field(description="Name of the related building entity")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    knowledge_origin: KnowledgeOriginLiteral = "architectural"
    rationale: str = ""


class SpatialRelationExtractionResponse(BaseModel):
    """Pass 4 extraction result for one entity."""
    spatial_relations: list[ExtractedSpatialRelation]
