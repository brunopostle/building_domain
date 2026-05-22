"""Integration tests for Pass 6 — Constraint Extraction."""
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import ConstraintRow, EntityRow, PassProgressRow
from bsos.pipeline.pass6 import run_pass6
from bsos.pipeline.schemas import ConstraintExtractionResponse, ExtractedConstraint
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


def make_provider_must(entity_name: str, rule: str) -> FakeLLMProvider:
    p = FakeLLMProvider()
    p.register(
        ConstraintExtractionResponse, entity_name,
        ConstraintExtractionResponse(constraints=[
            ExtractedConstraint(
                rule=rule,
                constraint_type="must",
                confidence=0.9,
                knowledge_origin="physical",
                rationale="Physical law requires this",
            ),
        ]),
    )
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pass6_writes_constraint(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_must("Roof", "Roof must have a drainage path")
    result = run_pass6(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1
    assert result["constraints_written"] == 1

    with Session(engine) as s:
        rows = s.exec(select(ConstraintRow)).all()
    assert len(rows) == 1
    assert rows[0].subject_id == "e-roof"
    assert rows[0].rule == "Roof must have a drainage path"
    assert rows[0].constraint_type == "must"


def test_pass6_constraint_fields_populated(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        ConstraintExtractionResponse, "Roof",
        ConstraintExtractionResponse(constraints=[
            ExtractedConstraint(
                rule="Roof must not have standing water",
                constraint_type="must_not",
                conditions=["flat roof"],
                exceptions=["green roof with retention layer"],
                confidence=0.85,
                knowledge_origin="engineering",
                rationale="Standing water causes structural degradation",
            ),
        ]),
    )
    run_pass6(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(ConstraintRow)).first()

    assert row is not None
    assert row.constraint_type == "must_not"
    assert row.confidence == pytest.approx(0.85)
    assert row.knowledge_origin == "engineering"
    assert row.rationale == "Standing water causes structural degradation"
    assert row.source_model == "fake-model"
    assert row.extraction_run_id == "run-001"
    # conditions and exceptions stored as JSON strings
    import json
    assert json.loads(row.conditions) == ["flat roof"]
    assert json.loads(row.exceptions) == ["green roof with retention layer"]


def test_pass6_must_not_constraint_type(engine):
    add_entity(engine, "e-elec", "Electrical panel")

    p = FakeLLMProvider()
    p.register(
        ConstraintExtractionResponse, "Electrical panel",
        ConstraintExtractionResponse(constraints=[
            ExtractedConstraint(
                rule="Electrical panel must not be installed in wet areas",
                constraint_type="must_not",
                confidence=0.95,
                knowledge_origin="engineering",
            ),
        ]),
    )
    run_pass6(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(ConstraintRow)).all()
    assert len(rows) == 1
    assert rows[0].constraint_type == "must_not"


def test_pass6_multiple_constraints_per_entity(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        ConstraintExtractionResponse, "Roof",
        ConstraintExtractionResponse(constraints=[
            ExtractedConstraint(rule="Roof must have drainage", constraint_type="must",
                                confidence=0.9, knowledge_origin="physical"),
            ExtractedConstraint(rule="Roof must not have gaps at perimeter",
                                constraint_type="must_not", confidence=0.85,
                                knowledge_origin="engineering"),
        ]),
    )
    result = run_pass6(engine, p, "run-001", max_workers=1)

    assert result["constraints_written"] == 2

    with Session(engine) as s:
        rows = s.exec(select(ConstraintRow)).all()
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Empty response
# ---------------------------------------------------------------------------

def test_pass6_skips_entity_on_empty_response(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wall", "Wall")

    # Roof has constraints; Wall returns empty
    p = make_provider_must("Roof", "Roof must have drainage")
    result = run_pass6(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 2
    assert result["constraints_written"] == 1  # only Roof's constraint


def test_pass6_skips_empty_rule(engine):
    add_entity(engine, "e-roof", "Roof")

    p = FakeLLMProvider()
    p.register(
        ConstraintExtractionResponse, "Roof",
        ConstraintExtractionResponse(constraints=[
            ExtractedConstraint(rule="", constraint_type="must", confidence=0.5,
                                knowledge_origin="physical"),
        ]),
    )
    run_pass6(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(ConstraintRow)).all()
    assert len(rows) == 0


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------

def test_pass6_records_progress(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_must("Roof", "Roof must have drainage")
    run_pass6(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "6")
        ).all()
    assert len(progress) == 2
    assert all(pr.status == "completed" for pr in progress)


def test_pass6_resume_skips_completed(engine):
    add_entity(engine, "e-roof", "Roof")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    run_pass6(engine, p, "run-001", max_workers=1)
    first_count = call_count

    run_pass6(engine, p, "run-002", max_workers=1)
    assert call_count == first_count


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_pass6_dry_run_no_writes(engine):
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_must("Roof", "Roof must have drainage")
    result = run_pass6(engine, p, "__dry_run__", dry_run=True, max_workers=1)

    assert result["constraints_written"] == 0
    assert result["entities_processed"] == 1

    with Session(engine) as s:
        assert len(s.exec(select(ConstraintRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


# ---------------------------------------------------------------------------
# Merged entities skipped
# ---------------------------------------------------------------------------

def test_pass6_skips_merged_entities(engine):
    add_entity(engine, "e-roof", "Roof", status="merged")
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_must("Wall", "Wall must be fire-rated in stairwells")
    result = run_pass6(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1


# ---------------------------------------------------------------------------
# Prompt contains calibration examples
# ---------------------------------------------------------------------------

def test_pass6_prompt_contains_calibration_examples(engine):
    """Verify the prompt includes the required calibration examples from the spec."""
    add_entity(engine, "e-roof", "Roof")

    captured_prompts = []

    class CapturingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            captured_prompts.append(prompt)
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CapturingProvider()
    run_pass6(engine, p, "run-001", max_workers=1)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "drainage" in prompt.lower()          # constraint calibration example
    assert "structural support" in prompt.lower() # assertion calibration example
