from typing import Annotated
from pydantic import Field
from bsos.models.base import ProvenanceMixin


class Assertion(ProvenanceMixin):
    id: str

    subject_id: str
    predicate: str
    object_id: str

    subject_type: str
    object_type: str

    conditions: list[str] = []
    exceptions: list[str] = []
    applicability: list[str] = []

    cross_prompt_consistency: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    prompt_framing_count: int | None = None
