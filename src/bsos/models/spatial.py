from bsos.models.base import ProvenanceMixin


class SpatialRelation(ProvenanceMixin):
    id: str
    subject_id: str
    relation: str
    object_id: str
