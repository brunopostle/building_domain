from typing import Literal
from pydantic import BaseModel

EntityType = Literal["component", "system", "space", "material", "activity"]

CORE_PREDICATES: frozenset[str] = frozenset({
    "requires",
    "depends_on",
    "protects_from",
    "unsuitable_for",
    "improves",
    "conflicts_with",
    "contains",
    "connects_to",
    "supports",
})

SPATIAL_RELATION_TYPES: list[str] = [
    "adjacent_to",
    "contains",
    "connects_to",
    "accessible_from",
    "above",
    "below",
    "enclosed_by",
]


class PredicateDefinition(BaseModel):
    predicate: str
    meaning: str
    allowed_subject_types: list[EntityType]
    allowed_object_types: list[EntityType]
    directional: bool


PREDICATE_REGISTRY: dict[str, PredicateDefinition] = {
    "requires": PredicateDefinition(
        predicate="requires",
        meaning="functional dependency — subject cannot operate correctly without object",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "depends_on": PredicateDefinition(
        predicate="depends_on",
        meaning="existence/process dependency — subject's existence or function depends on object",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "protects_from": PredicateDefinition(
        predicate="protects_from",
        meaning="shielding — subject shields from object (a hazard or condition)",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "unsuitable_for": PredicateDefinition(
        predicate="unsuitable_for",
        meaning="incompatibility — subject material or component is unsuitable for object use/context",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "improves": PredicateDefinition(
        predicate="improves",
        meaning="performance enhancement — subject improves object quality or performance",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "conflicts_with": PredicateDefinition(
        predicate="conflicts_with",
        meaning="design incompatibility — subject and object cannot coexist without trade-off",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=False,
    ),
    "contains": PredicateDefinition(
        predicate="contains",
        meaning="compositional/material — subject physically contains object (not spatial containment)",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=True,
    ),
    "connects_to": PredicateDefinition(
        predicate="connects_to",
        meaning="functional connectivity — subject connects to object functionally (not spatial topology)",
        allowed_subject_types=[],
        allowed_object_types=[],
        directional=False,
    ),
    "supports": PredicateDefinition(
        predicate="supports",
        meaning="structural load transfer — subject transmits loads from or to object",
        allowed_subject_types=["component", "system"],
        allowed_object_types=["component", "system"],
        directional=True,
    ),
}
