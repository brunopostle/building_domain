"""Repository helpers for pending vocabulary review tables."""
from datetime import datetime, timezone

from sqlmodel import Session, select

from bsos.persistence.models import PendingPredicateRow, PendingSpatialRelationTypeRow


def upsert_pending_predicate(session: Session, value: str) -> None:
    now = datetime.now(timezone.utc)
    row = session.exec(
        select(PendingPredicateRow).where(PendingPredicateRow.value == value)
    ).first()
    if row:
        row.occurrence_count += 1
        row.last_seen_at = now
    else:
        session.add(PendingPredicateRow(
            value=value,
            vocabulary_type="predicate",
            occurrence_count=1,
            first_seen_at=now,
            last_seen_at=now,
        ))
    session.flush()


def upsert_pending_spatial_relation_type(session: Session, value: str) -> None:
    now = datetime.now(timezone.utc)
    row = session.exec(
        select(PendingSpatialRelationTypeRow).where(PendingSpatialRelationTypeRow.value == value)
    ).first()
    if row:
        row.occurrence_count += 1
        row.last_seen_at = now
    else:
        session.add(PendingSpatialRelationTypeRow(
            value=value,
            vocabulary_type="spatial_relation",
            occurrence_count=1,
            first_seen_at=now,
            last_seen_at=now,
        ))
    session.flush()
