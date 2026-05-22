from bsos.models.base import ProvenanceMixin


class Pattern(ProvenanceMixin):
    id: str
    name: str

    context: list[str]

    problem: str
    force_descriptions: list[str]
    force_ids: list[str] = []

    solution: str

    consequences: list[str]

    related_pattern_names: list[str]
    related_pattern_ids: list[str] = []

    emergent_properties: list[str]
