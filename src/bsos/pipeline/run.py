"""ExtractionRun lifecycle — create/complete rows in extraction_runs table."""
import json
import uuid
from datetime import datetime, timezone

import structlog
from sqlmodel import Session, func, select

from bsos.persistence.models import EntityRow, AssertionRow, ExtractionRunRow

log = structlog.get_logger()


def start_run(session: Session, models: list[str], passes: list[str], seed: str | None) -> str:
    run_id = str(uuid.uuid4())
    entity_count = session.exec(select(func.count(EntityRow.id))).one()
    assertion_count = session.exec(select(func.count(AssertionRow.id))).one()

    row = ExtractionRunRow(
        id=run_id,
        started_at=datetime.now(timezone.utc),
        models=json.dumps(models),
        passes=json.dumps(passes),
        seed=seed,
        entity_count_before=entity_count,
        assertion_count_before=assertion_count,
    )
    session.add(row)
    session.commit()
    log.info("extraction_run_started", run_id=run_id, models=models, passes=passes)
    return run_id


def complete_run(session: Session, run_id: str) -> None:
    row = session.get(ExtractionRunRow, run_id)
    if row is None:
        return
    row.completed_at = datetime.now(timezone.utc)
    row.entity_count_after = session.exec(select(func.count(EntityRow.id))).one()
    row.assertion_count_after = session.exec(select(func.count(AssertionRow.id))).one()
    session.commit()
    log.info("extraction_run_completed", run_id=run_id,
             entity_count_after=row.entity_count_after)
