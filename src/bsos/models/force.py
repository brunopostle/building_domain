from enum import Enum
from bsos.models.base import ProvenanceMixin


class ForceDirection(str, Enum):
    increase = "increase"
    decrease = "decrease"


class Force(ProvenanceMixin):
    id: str
    name: str
    direction: ForceDirection
    affects: list[str]  # Entity UUIDs
