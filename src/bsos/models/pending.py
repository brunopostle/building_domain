from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class PendingPredicate(BaseModel):
    id: str
    value: str
    vocabulary_type: Literal["predicate", "spatial_relation"]
    occurrence_count: int = 1
    first_seen_at: datetime
    last_seen_at: datetime
    flagged_for_review: bool = False
