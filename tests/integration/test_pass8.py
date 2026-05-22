"""Integration tests for Pass 8 — Pattern Extraction."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, PassProgressRow, PatternRow
from bsos.pipeline.pass8 import run_pass8
from bsos.pipeline.schemas import ExtractedPattern, PatternExtractionResponse
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


def make_provider_pattern(entity_name: str, pattern_name: str) -> FakeLLMProvider:
    p = FakeLLMProvider()
    p.register(
        PatternExtractionResponse, entity_name,
        PatternExtractionResponse(patterns=[
            ExtractedPattern(
                name=pattern_name,
                context=["Urban housing", "Multi-storey buildings"],
                problem="Rainwater accumulates on flat roofs causing leaks",
                force_descriptions=["Drainage vs. green roof retention", "Cost vs. longevity"],
                solution="Provide a minimum 1:80 fall towards outlets with overflow drainage",
                consequences=["Reduced leak risk", "Increased structural load at drain points"],
                emergent_properties=["Predictable water management"],
                related_pattern_names=["Parapet Wall Detailing", "Roof Garden"],
                confidence=0.85,
                knowledge_origin="engineering",
                rationale="Fundamental roof design principle",
            ),
        ]),
    )
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pass8_writes_pattern(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    result = run_pass8(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
    assert result["patterns_written"] == 1

    with Session(engine) as s:
        rows = s.exec(select(PatternRow)).all()
    assert len(rows) == 1
    assert rows[0].name == "Positive Drainage"
    assert rows[0].subject_id == "e-roof"


def test_pass8_subject_id_links_to_entity(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_pattern("Wall", "Cavity Wall")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()
    assert row.subject_id == "e-wall"


def test_pass8_force_ids_starts_empty(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()
    assert json.loads(row.force_ids) == []


def test_pass8_related_pattern_ids_starts_empty(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()
    assert json.loads(row.related_pattern_ids) == []


def test_pass8_force_descriptions_stored(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()
    assert json.loads(row.force_descriptions) == [
        "Drainage vs. green roof retention",
        "Cost vs. longevity",
    ]


def test_pass8_related_pattern_names_stored(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()
    assert json.loads(row.related_pattern_names) == ["Parapet Wall Detailing", "Roof Garden"]


def test_pass8_all_json_list_fields(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()

    assert json.loads(row.context) == ["Urban housing", "Multi-storey buildings"]
    assert json.loads(row.consequences) == ["Reduced leak risk", "Increased structural load at drain points"]
    assert json.loads(row.emergent_properties) == ["Predictable water management"]


def test_pass8_scalar_fields_populated(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(PatternRow)).first()

    assert row.problem == "Rainwater accumulates on flat roofs causing leaks"
    assert row.solution == "Provide a minimum 1:80 fall towards outlets with overflow drainage"
    assert row.confidence == pytest.approx(0.85)
    assert row.knowledge_origin == "engineering"
    assert row.source_model == "fake-model"
    assert row.extraction_run_id == "run-001"
    assert row.status == "proposed"


def test_pass8_multiple_patterns_per_entity(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        PatternExtractionResponse, "Roof",
        PatternExtractionResponse(patterns=[
            ExtractedPattern(name="Positive Drainage", context=[], problem="Ponding",
                             force_descriptions=[], solution="Add falls",
                             consequences=[], confidence=0.8, knowledge_origin="engineering"),
            ExtractedPattern(name="Warm Roof Construction", context=[], problem="Condensation",
                             force_descriptions=[], solution="Insulation above deck",
                             consequences=[], confidence=0.75, knowledge_origin="physical"),
        ]),
    )
    result = run_pass8(engine, p, "run-001", max_workers=1)

    assert result["patterns_written"] == 2
    with Session(engine) as s:
        rows = s.exec(select(PatternRow)).all()
    assert {r.name for r in rows} == {"Positive Drainage", "Warm Roof Construction"}


# ---------------------------------------------------------------------------
# Skips invalid patterns
# ---------------------------------------------------------------------------

def test_pass8_skips_empty_name(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        PatternExtractionResponse, "Roof",
        PatternExtractionResponse(patterns=[
            ExtractedPattern(name="", context=[], problem="A problem",
                             force_descriptions=[], solution="A solution",
                             consequences=[], confidence=0.5, knowledge_origin="physical"),
        ]),
    )
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        assert len(s.exec(select(PatternRow)).all()) == 0


def test_pass8_skips_empty_problem(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        PatternExtractionResponse, "Roof",
        PatternExtractionResponse(patterns=[
            ExtractedPattern(name="Some Pattern", context=[], problem="",
                             force_descriptions=[], solution="A solution",
                             consequences=[], confidence=0.5, knowledge_origin="physical"),
        ]),
    )
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        assert len(s.exec(select(PatternRow)).all()) == 0


def test_pass8_skips_empty_solution(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        PatternExtractionResponse, "Roof",
        PatternExtractionResponse(patterns=[
            ExtractedPattern(name="Some Pattern", context=[], problem="A problem",
                             force_descriptions=[], solution="",
                             consequences=[], confidence=0.5, knowledge_origin="physical"),
        ]),
    )
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        assert len(s.exec(select(PatternRow)).all()) == 0


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------

def test_pass8_records_progress(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_pattern("Roof", "Positive Drainage")
    run_pass8(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "8")
        ).all()
    assert len(progress) == 2
    assert all(pr.status == "completed" for pr in progress)


def test_pass8_resume_skips_completed(engine):
    add_entity(engine, "e-roof", "Roof")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    run_pass8(engine, p, "run-001", max_workers=1)
    first_count = call_count

    run_pass8(engine, p, "run-002", max_workers=1)
    assert call_count == first_count


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_pass8_dry_run_no_writes(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_pattern("Roof", "Positive Drainage")
    result = run_pass8(engine, p, "__dry_run__", dry_run=True, max_workers=1)

    assert result["patterns_written"] == 0
    assert result["entities_processed"] == 1

    with Session(engine) as s:
        assert len(s.exec(select(PatternRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


# ---------------------------------------------------------------------------
# Merged entities skipped
# ---------------------------------------------------------------------------

def test_pass8_skips_merged_entities(engine):
    add_entity(engine, "e-roof", "Roof", status="merged")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_pattern("Wall", "Cavity Wall")
    result = run_pass8(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
