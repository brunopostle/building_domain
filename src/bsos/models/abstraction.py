from typing import Literal
from pydantic import model_validator
from bsos.models.base import ProvenanceMixin


class AbstractionNode(ProvenanceMixin):
    # knowledge_origin is NOT a stored field — excluded from persistence.
    # Always query the abstraction_node_effective_origins SQLite view for origin.
    # Overridden here to None so it is never required on construction.
    knowledge_origin: Literal["physical", "engineering", "cultural", "architectural"] | None = None  # type: ignore[assignment]

    id: str
    statement: str
    child_ids: list[str]
    abstraction_rationale: str

    @model_validator(mode="after")
    def _origin_not_used(self) -> "AbstractionNode":
        if self.knowledge_origin is not None:
            raise ValueError(
                "AbstractionNode.knowledge_origin must not be set — "
                "query abstraction_node_effective_origins view instead"
            )
        return self
