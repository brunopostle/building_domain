"""Integration tests for bsos doctor's process_relation cycle-conflict check
(building_domain-8lp): cycle-conflicted process_relations have no
conflict_pairs row by design, and previously produced no doctor warning at
all, leaving 432 rows permanently stuck and invisible."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from bsos.cli.doctor import app
from bsos.normalization.conflict_detection import _run_cycle_detection
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, PendingForceRefRow, ProcessRelationRow

NOW = datetime.now(timezone.utc)
runner = CliRunner()


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    eng = create_db_engine(str(path))
    create_views(eng)
    return str(path), eng


def _make_entity(session: Session, name: str) -> EntityRow:
    row = EntityRow(
        id=str(uuid.uuid4()),
        name=name,
        entity_type="activity",
        status="accepted",
        source_model="test",
        created_at=NOW,
    )
    session.add(row)
    return row


def _make_pr(session: Session, pred_id: str, succ_id: str) -> ProcessRelationRow:
    row = ProcessRelationRow(
        id=str(uuid.uuid4()),
        predecessor_id=pred_id,
        successor_id=succ_id,
        hard_constraint=True,
        source_model="test",
        created_at=NOW,
        confidence=0.9,
        status="proposed",
        knowledge_origin="extracted",
        rationale="test",
    )
    session.add(row)
    return row


def test_doctor_ok_with_no_cyclic_conflicts(db_path):
    path, _eng = db_path
    result = runner.invoke(app, ["--db", path])
    assert "no cycle-conflicted process_relations pending review" in result.output


def test_doctor_flags_cycle_conflicted_process_relations(db_path):
    path, eng = db_path
    with Session(eng) as session:
        e1 = _make_entity(session, "e1")
        e2 = _make_entity(session, "e2")
        e3 = _make_entity(session, "e3")
        _make_pr(session, e1.id, e2.id)
        _make_pr(session, e2.id, e3.id)
        _make_pr(session, e3.id, e1.id)
        session.commit()

    _run_cycle_detection(eng)

    result = runner.invoke(app, ["--db", path])
    assert "3 process_relation(s) conflicted by cycle detection" in result.output
    assert "bsos review pending --type process_relation" in result.output
    assert result.exit_code == 1


def test_doctor_flags_validation_failure_pending_force_refs(db_path):
    path, eng = db_path
    with Session(eng) as session:
        session.add(PendingForceRefRow(
            description="Wind Load Downward",
            failure_type="validation_failure",
            created_at=NOW,
        ))
        session.commit()

    result = runner.invoke(app, ["--db", path])
    assert "1 pending_force_ref(s) with failure_type=validation_failure" in result.output
    assert "review-pending --type force" not in result.output
    assert result.exit_code == 1


def test_doctor_fix_clears_validation_failure_pending_force_refs(db_path):
    path, eng = db_path
    with Session(eng) as session:
        session.add(PendingForceRefRow(
            description="Wind Load Downward",
            failure_type="validation_failure",
            created_at=NOW,
        ))
        session.commit()

    result = runner.invoke(app, ["--db", path, "--fix"])
    assert "1 pending_force_ref(s) cleared" in result.output

    with Session(eng) as session:
        remaining = session.exec(select(PendingForceRefRow)).all()
    assert remaining == []
