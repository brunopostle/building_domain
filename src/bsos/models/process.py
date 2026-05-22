from pydantic import model_validator
from bsos.models.base import ProvenanceMixin


class ProcessRelation(ProvenanceMixin):
    id: str
    predecessor_id: str
    successor_id: str
    hard_constraint: bool

    @model_validator(mode="after")
    def _rationale_required(self) -> "ProcessRelation":
        if not self.rationale:
            raise ValueError("ProcessRelation.rationale must always be populated")
        return self
