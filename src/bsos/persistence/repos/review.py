"""Repositories for ReviewDecision, ConflictPair, and ProvenanceLog."""
from datetime import datetime
from sqlmodel import select
from bsos.models.review import ReviewDecision
from bsos.persistence.models import ReviewDecisionRow, ConflictPairRow, ProvenanceLogRow
from bsos.persistence.repos.base import BaseRepository


class ReviewDecisionRepository(BaseRepository[ReviewDecision, ReviewDecisionRow]):
    _extraction_model = ReviewDecision
    _persistence_model = ReviewDecisionRow
    _json_list_fields = frozenset()

    def list_by_item(self, item_id: str) -> list[ReviewDecision]:
        rows = self._session.exec(
            select(ReviewDecisionRow).where(ReviewDecisionRow.item_id == item_id)
        ).all()
        return [self.from_persistence(r) for r in rows]

    def has_decision(self, item_id: str) -> bool:
        row = self._session.exec(
            select(ReviewDecisionRow).where(ReviewDecisionRow.item_id == item_id)
        ).first()
        return row is not None


class ConflictPair:
    """Lightweight dataclass — not a Pydantic extraction model."""
    def __init__(
        self,
        id: str,
        item_a_id: str,
        item_a_type: str,
        item_b_id: str,
        item_b_type: str,
        detected_at: datetime,
        classification: str,
    ) -> None:
        self.id = id
        self.item_a_id = item_a_id
        self.item_a_type = item_a_type
        self.item_b_id = item_b_id
        self.item_b_type = item_b_type
        self.detected_at = detected_at
        self.classification = classification


class ConflictPairRepository:
    def __init__(self, session) -> None:
        self._session = session

    def add(self, pair: ConflictPair) -> None:
        row = ConflictPairRow(
            id=pair.id,
            item_a_id=pair.item_a_id,
            item_a_type=pair.item_a_type,
            item_b_id=pair.item_b_id,
            item_b_type=pair.item_b_type,
            detected_at=pair.detected_at,
            classification=pair.classification,
        )
        self._session.add(row)
        self._session.flush()

    def get(self, id: str) -> ConflictPair | None:
        row = self._session.get(ConflictPairRow, id)
        return self._from_row(row) if row else None

    def find_existing(self, item_a_id: str, item_b_id: str) -> ConflictPair | None:
        """Check if a pair already exists (order-insensitive)."""
        row = self._session.exec(
            select(ConflictPairRow).where(
                (
                    (ConflictPairRow.item_a_id == item_a_id) & (ConflictPairRow.item_b_id == item_b_id)
                ) | (
                    (ConflictPairRow.item_a_id == item_b_id) & (ConflictPairRow.item_b_id == item_a_id)
                )
            )
        ).first()
        return self._from_row(row) if row else None

    def list_for_item(self, item_id: str) -> list[ConflictPair]:
        rows = self._session.exec(
            select(ConflictPairRow).where(
                (ConflictPairRow.item_a_id == item_id) | (ConflictPairRow.item_b_id == item_id)
            )
        ).all()
        return [self._from_row(r) for r in rows]

    def count_conflicted_items(self) -> int:
        """Count distinct item IDs involved in conflict pairs (approximate queue depth)."""
        from sqlalchemy import text
        result = self._session.exec(
            text(
                "SELECT COUNT(DISTINCT id) FROM ("
                "  SELECT item_a_id AS id FROM conflict_pairs"
                "  UNION SELECT item_b_id AS id FROM conflict_pairs"
                ")"
            )
        ).one()
        return result[0] if result else 0

    @staticmethod
    def _from_row(row: ConflictPairRow) -> ConflictPair:
        return ConflictPair(
            id=row.id,
            item_a_id=row.item_a_id,
            item_a_type=row.item_a_type,
            item_b_id=row.item_b_id,
            item_b_type=row.item_b_type,
            detected_at=row.detected_at,
            classification=row.classification,
        )


class ProvenanceLogEntry:
    """Lightweight dataclass for provenance log entries."""
    def __init__(
        self,
        id: str,
        item_id: str,
        item_type: str,
        old_status: str | None,
        new_status: str,
        changed_at: datetime,
        changed_by: str,
    ) -> None:
        self.id = id
        self.item_id = item_id
        self.item_type = item_type
        self.old_status = old_status
        self.new_status = new_status
        self.changed_at = changed_at
        self.changed_by = changed_by


class ProvenanceLogRepository:
    def __init__(self, session) -> None:
        self._session = session

    def add(self, entry: ProvenanceLogEntry) -> None:
        row = ProvenanceLogRow(
            id=entry.id,
            item_id=entry.item_id,
            item_type=entry.item_type,
            old_status=entry.old_status,
            new_status=entry.new_status,
            changed_at=entry.changed_at,
            changed_by=entry.changed_by,
        )
        self._session.add(row)
        self._session.flush()

    def list_for_item(self, item_id: str) -> list[ProvenanceLogEntry]:
        rows = self._session.exec(
            select(ProvenanceLogRow)
            .where(ProvenanceLogRow.item_id == item_id)
            .order_by(ProvenanceLogRow.changed_at)  # type: ignore[arg-type]
        ).all()
        return [self._from_row(r) for r in rows]

    @staticmethod
    def _from_row(row: ProvenanceLogRow) -> ProvenanceLogEntry:
        return ProvenanceLogEntry(
            id=row.id,
            item_id=row.item_id,
            item_type=row.item_type,
            old_status=row.old_status,
            new_status=row.new_status,
            changed_at=row.changed_at,
            changed_by=row.changed_by,
        )
