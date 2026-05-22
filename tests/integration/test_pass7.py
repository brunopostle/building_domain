"""Integration tests for Pass 7 — Anti-Pattern Extraction."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import AntiPatternRow, EntityRow, PassProgressRow
from bsos.pipeline.pass7 import run_pass7
from bsos.pipeline.schemas import AntiPatternExtractionResponse, ExtractedAntiPattern
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


def make_provider_ap(entity_name: str, ap_name: str) -> FakeLLMProvider:
    p = FakeLLMProvider()
    p.register(
        AntiPatternExtractionResponse, entity_name,
        AntiPatternExtractionResponse(anti_patterns=[
            ExtractedAntiPattern(
                name=ap_name,
                conditions=["Poor workmanship", "No inspection"],
                consequences=["Water ingress", "Structural deterioration"],
                mitigations=["Follow BS 8000 standards", "Inspect at completion"],
                confidence=0.88,
                knowledge_origin="engineering",
                rationale="Common failure mode in practice",
            ),
        ]),
    )
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pass7_writes_anti_pattern(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_ap("Roof", "Ponding water failure")
    result = run_pass7(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
    assert result["anti_patterns_written"] == 1

    with Session(engine) as s:
        rows = s.exec(select(AntiPatternRow)).all()
    assert len(rows) == 1
    assert rows[0].name == "Ponding water failure"
    assert rows[0].subject_id == "e-roof"


def test_pass7_subject_id_links_to_entity(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_ap("Wall", "Thermal bridging")
    run_pass7(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(AntiPatternRow)).first()
    assert row.subject_id == "e-wall"


def test_pass7_conditions_consequences_mitigations_as_json(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_ap("Roof", "Ponding water failure")
    run_pass7(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(AntiPatternRow)).first()

    assert json.loads(row.conditions) == ["Poor workmanship", "No inspection"]
    assert json.loads(row.consequences) == ["Water ingress", "Structural deterioration"]
    assert json.loads(row.mitigations) == ["Follow BS 8000 standards", "Inspect at completion"]


def test_pass7_provenance_fields_populated(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_ap("Roof", "Ponding water failure")
    run_pass7(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(AntiPatternRow)).first()

    assert row.source_model == "fake-model"
    assert row.extraction_run_id == "run-001"
    assert row.confidence == pytest.approx(0.88)
    assert row.knowledge_origin == "engineering"
    assert row.rationale == "Common failure mode in practice"
    assert row.status == "proposed"


def test_pass7_multiple_anti_patterns_per_entity(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        AntiPatternExtractionResponse, "Roof",
        AntiPatternExtractionResponse(anti_patterns=[
            ExtractedAntiPattern(name="Ponding", conditions=[], consequences=[], mitigations=[],
                                 confidence=0.8, knowledge_origin="physical"),
            ExtractedAntiPattern(name="Thermal bridging at eaves", conditions=[], consequences=[],
                                 mitigations=[], confidence=0.75, knowledge_origin="engineering"),
        ]),
    )
    result = run_pass7(engine, p, "run-001", max_workers=1)

    assert result["anti_patterns_written"] == 2
    with Session(engine) as s:
        rows = s.exec(select(AntiPatternRow)).all()
    assert len(rows) == 2
    names = {r.name for r in rows}
    assert "Ponding" in names
    assert "Thermal bridging at eaves" in names


# ---------------------------------------------------------------------------
# Empty / invalid responses
# ---------------------------------------------------------------------------

def test_pass7_skips_entity_on_empty_response(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_ap("Roof", "Ponding")
    result = run_pass7(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 2
    assert result["anti_patterns_written"] == 1


def test_pass7_skips_empty_name(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        AntiPatternExtractionResponse, "Roof",
        AntiPatternExtractionResponse(anti_patterns=[
            ExtractedAntiPattern(name="", conditions=[], consequences=[], mitigations=[],
                                 confidence=0.5, knowledge_origin="physical"),
        ]),
    )
    run_pass7(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(AntiPatternRow)).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------

def test_pass7_records_progress(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_ap("Roof", "Ponding")
    run_pass7(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "7")
        ).all()
    assert len(progress) == 2
    assert all(pr.status == "completed" for pr in progress)


def test_pass7_resume_skips_completed(engine):
    add_entity(engine, "e-roof", "Roof")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    run_pass7(engine, p, "run-001", max_workers=1)
    first_count = call_count

    run_pass7(engine, p, "run-002", max_workers=1)
    assert call_count == first_count


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_pass7_dry_run_no_writes(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_ap("Roof", "Ponding")
    result = run_pass7(engine, p, "__dry_run__", dry_run=True, max_workers=1)

    assert result["anti_patterns_written"] == 0
    assert result["entities_processed"] == 1

    with Session(engine) as s:
        assert len(s.exec(select(AntiPatternRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


# ---------------------------------------------------------------------------
# Merged entities skipped
# ---------------------------------------------------------------------------

def test_pass7_skips_merged_entities(engine):
    add_entity(engine, "e-roof", "Roof", status="merged")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_ap("Wall", "Damp penetration")
    result = run_pass7(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
