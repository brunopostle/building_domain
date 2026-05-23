from datetime import datetime
from typing import Annotated, Literal
from pydantic import BaseModel, Field


class BaseProvenanceMixin(BaseModel):
    source_model: str
    source_prompt: str | None
    created_at: datetime
    extraction_run_id: str | None = None


class ProvenanceMixin(BaseProvenanceMixin):
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    status: Literal["proposed", "accepted", "deprecated", "conflicted"] = "proposed"
    knowledge_origin: Literal["physical", "engineering", "cultural", "architectural", "schema"]
    rationale: str | None = None
    conflict_evaluated_at: datetime | None = None
