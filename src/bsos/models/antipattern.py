from bsos.models.base import ProvenanceMixin


class AntiPattern(ProvenanceMixin):
    id: str
    name: str

    conditions: list[str]
    consequences: list[str]
    mitigations: list[str]
