"""Integration tests for Pass 4 — Spatial Relation Extraction."""
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    EntityRow, PassProgressRow, PendingSpatialRelationTypeRow, SpatialRelationRow,
)
from bsos.pipeline.pass4 import run_pass4
from bsos.pipeline.schemas import ExtractedSpatialRelation, SpatialRelationExtractionResponse
from tests.fixtures.fake_responses import FakeLLMProvider

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def add_entity(engine, eid: str, name: str, entity_type: str = "component", status: str = "proposed") -> None:
    with Session(engine) as s:
        s.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                        status=status, source_model="test", created_at=NOW))
        s.commit()


def make_provider_adjacent(subject_name: str, object_name: str) -> FakeLLMProvider:
    p = FakeLLMProvider()
    p.register(
        SpatialRelationExtractionResponse, subject_name,
        SpatialRelationExtractionResponse(spatial_relations=[
            ExtractedSpatialRelation(
                relation="adjacent_to",
                object_name=object_name,
                confidence=0.85,
                knowledge_origin="architectural",
                rationale="They share a boundary",
            ),
        ]),
    )
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pass4_writes_spatial_relation(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    p = make_provider_adjacent("Wall", "Floor")
    result = run_pass4(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 2
    assert result["relations_written"] >= 1

    with Session(engine) as s:
        rows = s.exec(select(SpatialRelationRow)).all()
    assert len(rows) >= 1
    assert any(r.relation == "adjacent_to" and r.object_id == "e-floor" for r in rows)


def test_pass4_relation_fields_populated(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    p = make_provider_adjacent("Wall", "Floor")
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(SpatialRelationRow)).first()

    assert row is not None
    assert row.subject_id == "e-wall"
    assert row.relation == "adjacent_to"
    assert row.object_id == "e-floor"
    assert row.source_model == "fake-model"
    assert row.extraction_run_id == "run-001"
    assert row.confidence == pytest.approx(0.85)
    assert row.knowledge_origin == "architectural"


# ---------------------------------------------------------------------------
# Self-reference skip
# ---------------------------------------------------------------------------

def test_pass4_skips_self_reference(engine):
    add_entity(engine, "e-wall", "Wall")

    p = FakeLLMProvider()
    p.register(
        SpatialRelationExtractionResponse, "Wall",
        SpatialRelationExtractionResponse(spatial_relations=[
            ExtractedSpatialRelation(relation="adjacent_to", object_name="Wall", confidence=0.5),
        ]),
    )
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(SpatialRelationRow)).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Unresolved object name — skip, no row written
# ---------------------------------------------------------------------------

def test_pass4_skips_unresolved_object(engine):
    add_entity(engine, "e-wall", "Wall")

    p = FakeLLMProvider()
    p.register(
        SpatialRelationExtractionResponse, "Wall",
        SpatialRelationExtractionResponse(spatial_relations=[
            ExtractedSpatialRelation(relation="adjacent_to", object_name="Nonexistent", confidence=0.5),
        ]),
    )
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(SpatialRelationRow)).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Unknown relation type → pending_spatial_relation_types AND row written
# ---------------------------------------------------------------------------

def test_pass4_writes_unknown_relation_to_pending(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    p = FakeLLMProvider()
    p.register(
        SpatialRelationExtractionResponse, "Wall",
        SpatialRelationExtractionResponse(spatial_relations=[
            ExtractedSpatialRelation(
                relation="structurally_supports",
                object_name="Floor",
                confidence=0.7,
            ),
        ]),
    )
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        pending = s.exec(select(PendingSpatialRelationTypeRow)).all()
        rows = s.exec(select(SpatialRelationRow)).all()

    assert any(p.value == "structurally_supports" for p in pending)
    assert len(rows) >= 1
    assert rows[0].relation == "structurally_supports"


def test_pass4_pending_occurrence_count_increments(engine):
    """Two entities in the same run both using the same unknown type → count=2."""
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        SpatialRelationExtractionResponse, "Wall",
        SpatialRelationExtractionResponse(spatial_relations=[
            ExtractedSpatialRelation(relation="custom_type", object_name="Floor", confidence=0.5),
        ]),
    )
    p.register(
        SpatialRelationExtractionResponse, "Roof",
        SpatialRelationExtractionResponse(spatial_relations=[
            ExtractedSpatialRelation(relation="custom_type", object_name="Floor", confidence=0.5),
        ]),
    )
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(
            select(PendingSpatialRelationTypeRow).where(
                PendingSpatialRelationTypeRow.value == "custom_type"
            )
        ).first()
    assert row is not None
    assert row.occurrence_count == 2


def test_pass4_known_relation_not_in_pending(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    p = make_provider_adjacent("Wall", "Floor")
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        pending = s.exec(select(PendingSpatialRelationTypeRow)).all()
    assert len(pending) == 0


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------

def test_pass4_records_progress(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    p = make_provider_adjacent("Wall", "Floor")
    run_pass4(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "4")
        ).all()
    assert len(progress) == 2
    assert all(pr.status == "completed" for pr in progress)


def test_pass4_resume_skips_completed(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    p.register(
        SpatialRelationExtractionResponse, "Wall",
        SpatialRelationExtractionResponse(spatial_relations=[]),
    )

    run_pass4(engine, p, "run-001", max_workers=1)
    first_count = call_count

    run_pass4(engine, p, "run-002", max_workers=1)
    assert call_count == first_count


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_pass4_dry_run_no_writes(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")

    p = make_provider_adjacent("Wall", "Floor")
    result = run_pass4(engine, p, "__dry_run__", dry_run=True, max_workers=1)

    assert result["relations_written"] == 0
    assert result["entities_processed"] == 2

    with Session(engine) as s:
        assert len(s.exec(select(SpatialRelationRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


# ---------------------------------------------------------------------------
# Merged entities skipped
# ---------------------------------------------------------------------------

def test_pass4_skips_merged_entities(engine):
    add_entity(engine, "e-wall", "Wall", status="merged")
    add_entity(engine, "e-floor", "Floor")

    p = make_provider_adjacent("Floor", "Wall")
    result = run_pass4(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
