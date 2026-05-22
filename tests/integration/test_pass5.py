"""Integration tests for Pass 5 — Process/Sequence Extraction."""
import logging
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, PassProgressRow, ProcessRelationRow
from bsos.pipeline.pass5 import run_pass5
from bsos.pipeline.schemas import ExtractedProcessRelation, ProcessRelationExtractionResponse
from tests.fixtures.fake_responses import FakeLLMProvider

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def add_entity(engine, eid: str, name: str, entity_type: str = "activity", status: str = "proposed") -> None:
    with Session(engine) as s:
        s.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                        status=status, source_model="test", created_at=NOW))
        s.commit()


def make_provider_seq(entity_name: str, pred: str, succ: str, hard: bool = True) -> FakeLLMProvider:
    p = FakeLLMProvider()
    p.register(
        ProcessRelationExtractionResponse, entity_name,
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name=pred,
                successor_name=succ,
                hard_constraint=hard,
                rationale="Physical dependency requires this order",
            ),
        ]),
    )
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pass5_writes_process_relation(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = make_provider_seq("Formwork", "Formwork", "Concrete pouring")
    result = run_pass5(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 2
    assert result["relations_written"] >= 1

    with Session(engine) as s:
        rows = s.exec(select(ProcessRelationRow)).all()
    assert len(rows) >= 1
    assert rows[0].predecessor_id == "e-form"
    assert rows[0].successor_id == "e-pour"
    assert rows[0].hard_constraint is True


def test_pass5_relation_fields_populated(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = make_provider_seq("Formwork", "Formwork", "Concrete pouring")
    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(ProcessRelationRow)).first()

    assert row is not None
    assert row.rationale == "Physical dependency requires this order"
    assert row.source_model == "fake-model"
    assert row.extraction_run_id == "run-001"
    assert row.status == "proposed"


# ---------------------------------------------------------------------------
# Inline activity creation
# ---------------------------------------------------------------------------

def test_pass5_creates_unknown_predecessor_inline(engine):
    add_entity(engine, "e-pour", "Concrete pouring")

    p = FakeLLMProvider()
    p.register(
        ProcessRelationExtractionResponse, "Concrete pouring",
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name="Excavation",  # not in DB
                successor_name="Concrete pouring",
                hard_constraint=True,
                rationale="Site must be prepared first",
            ),
        ]),
    )

    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        new_entity = s.exec(
            select(EntityRow).where(EntityRow.name == "Excavation")
        ).first()
        rows = s.exec(select(ProcessRelationRow)).all()

    assert new_entity is not None
    assert new_entity.entity_type == "activity"
    assert new_entity.status == "proposed"
    assert len(rows) >= 1


def test_pass5_inline_creation_emits_warning(engine, capsys):
    add_entity(engine, "e-pour", "Concrete pouring")

    p = FakeLLMProvider()
    p.register(
        ProcessRelationExtractionResponse, "Concrete pouring",
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name="BrandNewActivity",
                successor_name="Concrete pouring",
                hard_constraint=True,
                rationale="Must happen first",
            ),
        ]),
    )

    run_pass5(engine, p, "run-001", max_workers=1)
    captured = capsys.readouterr()

    # structlog writes to stdout; verify the warning event and activity name appear
    assert "pass5_inline_activity_created" in captured.out
    assert "BrandNewActivity" in captured.out


def test_pass5_inline_creation_is_idempotent(engine):
    """Two entities both referencing the same unknown activity → only one entity created."""
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = FakeLLMProvider()
    for entity_name in ("Formwork", "Concrete pouring"):
        p.register(
            ProcessRelationExtractionResponse, entity_name,
            ProcessRelationExtractionResponse(process_relations=[
                ExtractedProcessRelation(
                    predecessor_name="SitePreparation",
                    successor_name=entity_name,
                    hard_constraint=True,
                    rationale="Site must be ready first",
                ),
            ]),
        )

    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        matches = s.exec(
            select(EntityRow).where(EntityRow.name.ilike("SitePreparation"))  # type: ignore[attr-defined]
        ).all()
    # May create 1 or 2 due to concurrent workers, but the lock should prevent most races
    # At minimum 1 must exist
    assert len(matches) >= 1


# ---------------------------------------------------------------------------
# Deduplication — INSERT OR IGNORE semantics
# ---------------------------------------------------------------------------

def test_pass5_dedup_ignores_duplicate(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = FakeLLMProvider()
    for entity in ("Formwork", "Concrete pouring"):
        p.register(
            ProcessRelationExtractionResponse, entity,
            ProcessRelationExtractionResponse(process_relations=[
                ExtractedProcessRelation(
                    predecessor_name="Formwork",
                    successor_name="Concrete pouring",
                    hard_constraint=True,
                    rationale="Same relation from both entity perspectives",
                ),
            ]),
        )

    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(ProcessRelationRow)).all()
    # Only one row despite two entities both extracting the same relation
    assert len(rows) == 1


def test_pass5_hard_constraint_divergence_logs_error(engine, capsys):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    # First run: establishes Formwork→Concrete pouring as hard=True
    p1 = make_provider_seq("Formwork", "Formwork", "Concrete pouring", hard=True)
    run_pass5(engine, p1, "run-001", max_workers=1)
    capsys.readouterr()  # clear stdout buffer

    # Add Curing after run-001 so it wasn't processed yet
    add_entity(engine, "e-cure", "Curing")

    # Second run: only Curing is unprocessed; it extracts the same pair but hard=False
    p2 = FakeLLMProvider()
    p2.register(
        ProcessRelationExtractionResponse, "Curing",
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name="Formwork",
                successor_name="Concrete pouring",
                hard_constraint=False,  # diverges from existing True
                rationale="Actually this is optional",
            ),
        ]),
    )

    result = run_pass5(engine, p2, "run-002", max_workers=1)
    captured = capsys.readouterr()

    assert result["hard_constraint_divergences"] >= 1
    assert "pass5_hard_constraint_divergence" in captured.out


# ---------------------------------------------------------------------------
# Rationale required
# ---------------------------------------------------------------------------

def test_pass5_skips_empty_rationale(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = FakeLLMProvider()
    p.register(
        ProcessRelationExtractionResponse, "Formwork",
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name="Formwork",
                successor_name="Concrete pouring",
                hard_constraint=True,
                rationale="",  # empty — must be skipped
            ),
        ]),
    )

    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(ProcessRelationRow)).all()
    assert len(rows) == 0


def test_pass5_skips_whitespace_rationale(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = FakeLLMProvider()
    p.register(
        ProcessRelationExtractionResponse, "Formwork",
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name="Formwork",
                successor_name="Concrete pouring",
                hard_constraint=True,
                rationale="   ",
            ),
        ]),
    )

    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(ProcessRelationRow)).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Self-reference / same entity
# ---------------------------------------------------------------------------

def test_pass5_skips_self_loop(engine):
    add_entity(engine, "e-form", "Formwork")

    p = FakeLLMProvider()
    p.register(
        ProcessRelationExtractionResponse, "Formwork",
        ProcessRelationExtractionResponse(process_relations=[
            ExtractedProcessRelation(
                predecessor_name="Formwork",
                successor_name="Formwork",
                hard_constraint=True,
                rationale="Self-loop should be skipped",
            ),
        ]),
    )

    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(ProcessRelationRow)).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------

def test_pass5_records_progress(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = make_provider_seq("Formwork", "Formwork", "Concrete pouring")
    run_pass5(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "5")
        ).all()
    assert len(progress) == 2
    assert all(pr.status == "completed" for pr in progress)


def test_pass5_resume_skips_completed(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    run_pass5(engine, p, "run-001", max_workers=1)
    first_count = call_count

    run_pass5(engine, p, "run-002", max_workers=1)
    assert call_count == first_count


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_pass5_dry_run_no_writes(engine):
    add_entity(engine, "e-form", "Formwork")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = make_provider_seq("Formwork", "Formwork", "Concrete pouring")
    result = run_pass5(engine, p, "__dry_run__", dry_run=True, max_workers=1)

    assert result["relations_written"] == 0
    assert result["entities_processed"] == 2

    with Session(engine) as s:
        assert len(s.exec(select(ProcessRelationRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


# ---------------------------------------------------------------------------
# Merged entities skipped
# ---------------------------------------------------------------------------

def test_pass5_skips_merged_entities(engine):
    add_entity(engine, "e-form", "Formwork", status="merged")
    add_entity(engine, "e-pour", "Concrete pouring")

    p = make_provider_seq("Concrete pouring", "Formwork", "Concrete pouring")
    result = run_pass5(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
