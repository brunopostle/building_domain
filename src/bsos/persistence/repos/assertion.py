from sqlmodel import select
from bsos.models.assertion import Assertion
from bsos.persistence.models import AssertionRow
from bsos.persistence.repos.base import BaseRepository


class AssertionRepository(BaseRepository[Assertion, AssertionRow]):
    _extraction_model = Assertion
    _persistence_model = AssertionRow
    _json_list_fields = frozenset({"conditions", "exceptions", "applicability"})

    def list_by_subject(self, subject_id: str) -> list[Assertion]:
        rows = self._session.exec(
            select(AssertionRow).where(AssertionRow.subject_id == subject_id)
        ).all()
        return [self.from_persistence(r) for r in rows]

    def list_by_predicate(self, predicate: str) -> list[Assertion]:
        rows = self._session.exec(
            select(AssertionRow).where(AssertionRow.predicate == predicate)
        ).all()
        return [self.from_persistence(r) for r in rows]

    def list_pending_conflict_evaluation(self) -> list[Assertion]:
        rows = self._session.exec(
            select(AssertionRow).where(AssertionRow.conflict_evaluated_at.is_(None))  # type: ignore[attr-defined]
        ).all()
        return [self.from_persistence(r) for r in rows]
