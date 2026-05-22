from sqlmodel import select
from bsos.models.entity import Entity
from bsos.persistence.models import EntityRow, EntityAliasRow
from bsos.persistence.repos.base import BaseRepository


class EntityRepository(BaseRepository[Entity, EntityRow]):
    _extraction_model = Entity
    _persistence_model = EntityRow

    def get_by_name(self, name: str) -> Entity | None:
        row = self._session.exec(
            select(EntityRow).where(EntityRow.name == name)
        ).first()
        return self.from_persistence(row) if row else None

    def get_by_name_or_alias(self, name: str) -> Entity | None:
        """Case-insensitive lookup against name and entity_aliases."""
        row = self._session.exec(
            select(EntityRow).where(EntityRow.name.ilike(name))  # type: ignore[attr-defined]
        ).first()
        if row:
            return self.from_persistence(row)
        alias_row = self._session.exec(
            select(EntityAliasRow).where(EntityAliasRow.alias.ilike(name))  # type: ignore[attr-defined]
        ).first()
        if alias_row:
            entity_row = self._session.get(EntityRow, alias_row.entity_id)
            return self.from_persistence(entity_row) if entity_row else None
        return None

    def add_alias(self, entity_id: str, alias: str) -> None:
        row = EntityAliasRow(entity_id=entity_id, alias=alias)
        self._session.add(row)
        self._session.flush()

    def get_aliases(self, entity_id: str) -> list[str]:
        rows = self._session.exec(
            select(EntityAliasRow).where(EntityAliasRow.entity_id == entity_id)
        ).all()
        return [r.alias for r in rows]
