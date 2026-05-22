"""Repository for AbstractionNode records.

knowledge_origin is NOT stored on AbstractionNodeRow — it is computed via the
abstraction_node_effective_origins SQLite view. This repository overrides
_knowledge_origin_stored = False to prevent callers from using the base
filter_by_knowledge_origin() method directly.
"""
from sqlmodel import select, text
from bsos.models.abstraction import AbstractionNode
from bsos.persistence.models import AbstractionNodeRow
from bsos.persistence.repos.base import BaseRepository


class AbstractionNodeRepository(BaseRepository[AbstractionNode, AbstractionNodeRow]):
    _extraction_model = AbstractionNode
    _persistence_model = AbstractionNodeRow
    _json_list_fields = frozenset({"child_ids"})
    _knowledge_origin_stored = False

    def list_by_child(self, assertion_id: str) -> list[AbstractionNode]:
        """Return all AbstractionNodes whose child_ids contain the given assertion UUID."""
        rows = self._session.exec(
            select(AbstractionNodeRow).where(
                AbstractionNodeRow.id.in_(  # type: ignore[attr-defined]
                    self._session.exec(
                        text(
                            "SELECT an.id FROM abstraction_nodes an, "
                            "json_each(an.child_ids) c WHERE c.value = :aid"
                        ).bindparams(aid=assertion_id)
                    ).scalars()
                )
            )
        ).all()
        return [self.from_persistence(r) for r in rows]

    def list_proposed(self) -> list[AbstractionNode]:
        rows = self._session.exec(
            select(AbstractionNodeRow).where(AbstractionNodeRow.status == "proposed")
        ).all()
        return [self.from_persistence(r) for r in rows]

    def count_proposed(self) -> int:
        from sqlalchemy import func
        return self._session.exec(
            select(func.count()).where(AbstractionNodeRow.status == "proposed")  # type: ignore[call-overload]
        ).one()

    def get_effective_origins(self, abstraction_node_id: str) -> dict[str, int]:
        """Return {knowledge_origin: count} from the effective_origins view."""
        rows = self._session.exec(
            text(
                "SELECT knowledge_origin, origin_count "
                "FROM abstraction_node_effective_origins "
                "WHERE abstraction_node_id = :id"
            ).bindparams(id=abstraction_node_id)
        ).all()
        return {row[0]: row[1] for row in rows}

    def get_majority_origin(self, abstraction_node_id: str) -> str | None:
        """Return the majority knowledge_origin for the given AbstractionNode."""
        origins = self.get_effective_origins(abstraction_node_id)
        if not origins:
            return None
        return max(origins, key=origins.__getitem__)
