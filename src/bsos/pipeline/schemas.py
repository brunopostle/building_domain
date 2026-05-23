"""Pydantic schemas for LLM structured output — separate from persistence models.

Each pass uses a distinct schema class so FakeLLMProvider can dispatch on type[BaseModel].
"""
from typing import Literal
from pydantic import BaseModel, Field


class DiscoveredConcept(BaseModel):
    name: str
    entity_type: Literal["component", "system", "space", "material", "activity", "ifc_class"]
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

KnowledgeOriginLiteral = Literal["physical", "engineering", "architectural", "cultural", "schema"]


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


# Pass 5 schemas

class ExtractedProcessRelation(BaseModel):
    """One temporal ordering relationship between two activities."""
    predecessor_name: str = Field(description="Activity that must happen first")
    successor_name: str = Field(description="Activity that follows")
    hard_constraint: bool = Field(default=True, description="True if ordering is physically required")
    rationale: str = Field(description="Why this ordering is required (must be non-empty)")


class ProcessRelationExtractionResponse(BaseModel):
    """Pass 5 extraction result for one entity."""
    process_relations: list[ExtractedProcessRelation]


# Pass 6 schemas

ConstraintTypeLiteral = Literal["must", "must_not"]


class ExtractedConstraint(BaseModel):
    """One binary design rule extracted for a single entity."""
    rule: str = Field(description="The constraint rule text")
    constraint_type: ConstraintTypeLiteral = Field(description="'must' or 'must_not'")
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    knowledge_origin: KnowledgeOriginLiteral = "engineering"
    rationale: str = ""


class ConstraintExtractionResponse(BaseModel):
    """Pass 6 extraction result for one entity."""
    constraints: list[ExtractedConstraint]


# Pass 7 schemas

class ExtractedAntiPattern(BaseModel):
    """One failure condition or pathological configuration extracted for an entity."""
    name: str = Field(description="Short descriptive name for this anti-pattern")
    conditions: list[str] = Field(default_factory=list, description="Conditions that lead to this failure")
    consequences: list[str] = Field(default_factory=list, description="Resulting failures or harms")
    mitigations: list[str] = Field(default_factory=list, description="Ways to avoid or recover from this")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    knowledge_origin: KnowledgeOriginLiteral = "engineering"
    rationale: str = ""


class AntiPatternExtractionResponse(BaseModel):
    """Pass 7 extraction result for one entity."""
    anti_patterns: list[ExtractedAntiPattern]


# Pass 8 schemas

class ExtractedPattern(BaseModel):
    """One Alexander-style architectural pattern extracted for an entity."""
    name: str = Field(description="Short descriptive name for the pattern")
    context: list[str] = Field(default_factory=list, description="Situations where the pattern applies")
    problem: str = Field(description="The recurring problem this pattern addresses")
    force_descriptions: list[str] = Field(default_factory=list, description="Competing forces at play (free-text)")
    solution: str = Field(description="The pattern solution")
    consequences: list[str] = Field(default_factory=list, description="Results of applying the pattern")
    emergent_properties: list[str] = Field(default_factory=list, description="Properties that emerge from the pattern")
    related_pattern_names: list[str] = Field(default_factory=list, description="Names of related patterns (free-text)")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    knowledge_origin: KnowledgeOriginLiteral = "architectural"
    rationale: str = ""


class PatternExtractionResponse(BaseModel):
    """Pass 8 extraction result for one entity."""
    patterns: list[ExtractedPattern]


# Pass 9 schemas

ForceDirectionLiteral = Literal["increase", "decrease"]


class ExtractedForce(BaseModel):
    """One design pressure extracted for an entity."""
    name: str = Field(description="Force name as a directional pressure (must contain a direction qualifier)")
    direction: ForceDirectionLiteral = Field(description="'increase' or 'decrease'")
    affects: list[str] = Field(default_factory=list, description="Names of entities this force acts upon")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    knowledge_origin: KnowledgeOriginLiteral = "engineering"
    rationale: str = ""


class ForceExtractionResponse(BaseModel):
    """Pass 9 extraction result for one entity."""
    forces: list[ExtractedForce]


# Pass 11 schemas

class AdversarialFinding(BaseModel):
    """One adversarial finding for an extracted item."""
    item_id: str = Field(description="UUID of the assertion, constraint, pattern, or antipattern")
    item_type: Literal["assertion", "constraint", "pattern", "antipattern"]
    finding_type: Literal["exception", "context_limitation", "potential_error", "scope_restriction"]
    detail: str = Field(description="Specific, concrete description of the finding")
    suggested_action: Literal["add_exception", "add_condition", "flag_for_review", "deprecate"]
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class AdversarialValidationResponse(BaseModel):
    """Pass 11 adversarial review result for a batch of assertions."""
    findings: list[AdversarialFinding]
