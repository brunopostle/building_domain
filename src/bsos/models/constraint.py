from typing import Literal
from bsos.models.base import ProvenanceMixin


class Constraint(ProvenanceMixin):
    id: str
    subject_id: str
    rule: str
    constraint_type: Literal["must", "must_not"]
    conditions: list[str] = []
    exceptions: list[str] = []
