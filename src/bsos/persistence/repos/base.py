"""BaseRepository: generic to_persistence / from_persistence via model_dump."""
from __future__ import annotations
import json
from typing import Generic, TypeVar, Type
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, select

P = TypeVar("P", bound=BaseModel)   # Pydantic extraction model
R = TypeVar("R", bound=SQLModel)    # SQLModel persistence row

# Fields whose values are JSON-encoded lists in the database.
# Subclasses declare their own JSON_LIST_FIELDS to override.
JSON_LIST_FIELDS: frozenset[str] = frozenset()


class BaseRepository(Generic[P, R]):
    _extraction_model: Type[P]
    _persistence_model: Type[R]
    _json_list_fields: frozenset[str] = JSON_LIST_FIELDS
    _knowledge_origin_stored: bool = True

    def __init__(self, session: Session):
        self._session = session

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_persistence(self, obj: P) -> R:
        data = obj.model_dump(mode="python")
        for field in self._json_list_fields:
            if field in data:
                data[field] = json.dumps(data[field])
        return self._persistence_model(**data)

    def from_persistence(self, row: R) -> P:
        data = {c.name: getattr(row, c.name) for c in row.__table__.columns}
        for field in self._json_list_fields:
            if field in data and isinstance(data[field], str):
                data[field] = json.loads(data[field])
        return self._extraction_model.model_validate(data)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, obj: P) -> None:
        row = self.to_persistence(obj)
        self._session.add(row)
        self._session.flush()

    def get(self, id: str) -> P | None:
        row = self._session.get(self._persistence_model, id)
        return self.from_persistence(row) if row else None

    def list(self) -> list[P]:
        rows = self._session.exec(select(self._persistence_model)).all()
        return [self.from_persistence(r) for r in rows]

    def filter_by_knowledge_origin(self, origin: str) -> list[P]:
        if not self._knowledge_origin_stored:
            raise NotImplementedError(
                f"{type(self).__name__} does not store knowledge_origin directly — "
                "use the abstraction_node_effective_origins view"
            )
        rows = self._session.exec(
            select(self._persistence_model).where(
                self._persistence_model.knowledge_origin == origin  # type: ignore[attr-defined]
            )
        ).all()
        return [self.from_persistence(r) for r in rows]
