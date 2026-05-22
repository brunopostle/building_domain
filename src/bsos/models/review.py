from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class ReviewDecision(BaseModel):
    id: str
    item_id: str
    item_type: str
    decision: Literal["accept", "reject", "map_to", "defer"]
    mapped_to: str | None = None
    rationale: str | None = None
    reviewer: str
    created_at: datetime
