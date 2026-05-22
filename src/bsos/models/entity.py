from typing import Literal
from bsos.models.base import BaseProvenanceMixin


class Entity(BaseProvenanceMixin):
    id: str
    name: str
    entity_type: Literal["component", "system", "space", "material", "activity"]
    description: str = ""
    status: Literal["proposed", "accepted", "deprecated"] = "proposed"
    is_entrance: bool = False
