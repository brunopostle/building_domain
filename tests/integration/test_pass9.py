"""Integration tests for Pass 9 — Force Extraction with direction validation."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    EntityRow, ForceRow, PassProgressRow,
    PendingEntityRefRow, PendingForceRefRow,
)
from bsos.pipeline.pass9 import run_pass9, _validate_direction, INCREASE_QUALIFIERS, DECREASE_QUALIFIERS
from bsos.pipeline.schemas import ExtractedForce, ForceExtractionResponse
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


def make_provider_force(entity_name: str, force_name: str, direction: str,
                        affects: list[str] | None = None) -> FakeLLMProvider:
    p = FakeLLMProvider()
    p.register(
        ForceExtractionResponse, entity_name,
        ForceExtractionResponse(forces=[
            ExtractedForce(
                name=force_name,
                direction=direction,
                affects=affects or [],
                confidence=0.8,
                knowledge_origin="engineering",
                rationale="Design pressure identified in practice",
            ),
        ]),
    )
    return p


# ---------------------------------------------------------------------------
# Direction validation unit tests
# ---------------------------------------------------------------------------

def test_validate_direction_increase_passes():
    for qualifier in INCREASE_QUALIFIERS:
        assert _validate_direction(f"Pressure for {qualifier} thermal performance", "increase")


def test_validate_direction_decrease_passes():
    for qualifier in DECREASE_QUALIFIERS:
        assert _validate_direction(f"Force to {qualifier} heat loss", "decrease")


def test_validate_direction_increase_fails_without_qualifier():
    assert not _validate_direction("Thermal performance pressure", "increase")


def test_validate_direction_decrease_fails_without_qualifier():
    assert not _validate_direction("Heat loss pressure", "decrease")


def test_validate_direction_case_insensitive():
    assert _validate_direction("IMPROVED thermal performance", "increase")
    assert _validate_direction("REDUCED heat loss", "decrease")


def test_validate_direction_substring_match():
    assert _validate_direction("Maximized daylight", "increase")   # contains "maximized"
    assert _validate_direction("Minimized cold bridging", "decrease")  # "minimized" — not in list
    # "minimised" and "minimized" are both in DECREASE_QUALIFIERS
    assert _validate_direction("Minimised cold bridging", "decrease")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_pass9_writes_force(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-ins", "Insulation")

    p = make_provider_force("Wall", "Improved thermal performance", "increase", ["Insulation"])
    result = run_pass9(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 2
    assert result["forces_written"] >= 1

    with Session(engine) as s:
        rows = s.exec(select(ForceRow)).all()
    assert len(rows) >= 1
    assert rows[0].name == "Improved thermal performance"
    assert rows[0].direction == "increase"


def test_pass9_force_fields_populated(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Reduced heat loss", "decrease")
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(ForceRow)).first()

    assert row.source_model == "fake-model"
    assert row.extraction_run_id == "run-001"
    assert row.confidence == pytest.approx(0.8)
    assert row.knowledge_origin == "engineering"
    assert row.status == "proposed"


def test_pass9_affects_resolved_to_ids(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-ins", "Insulation")
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_force("Wall", "Improved thermal performance", "increase",
                            affects=["Insulation", "Roof"])
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(ForceRow).where(ForceRow.name == "Improved thermal performance")).first()

    affects_ids = json.loads(row.affects)
    assert "e-ins" in affects_ids
    assert "e-roof" in affects_ids


def test_pass9_empty_affects_stored_as_empty_list(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Improved airtightness", "increase", affects=[])
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        row = s.exec(select(ForceRow)).first()
    assert json.loads(row.affects) == []


# ---------------------------------------------------------------------------
# Direction validation failure
# ---------------------------------------------------------------------------

def test_pass9_validation_failure_skips_force_row(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Thermal performance pressure", "increase")  # no qualifier
    result = run_pass9(engine, p, "run-001", max_workers=1)

    assert result["validation_failures"] >= 1
    with Session(engine) as s:
        assert len(s.exec(select(ForceRow)).all()) == 0


def test_pass9_validation_failure_writes_pending_force_ref(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Thermal pressure", "increase")  # no qualifier
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        pending = s.exec(select(PendingForceRefRow)).all()

    assert len(pending) >= 1
    assert pending[0].failure_type == "validation_failure"
    assert pending[0].description == "Thermal pressure"


def test_pass9_validation_failure_does_not_block_valid_forces(engine):
    """An entity with one invalid and one valid force: only valid one is stored."""
    add_entity(engine, "e-wall", "Wall")

    p = FakeLLMProvider()
    p.register(
        ForceExtractionResponse, "Wall",
        ForceExtractionResponse(forces=[
            ExtractedForce(name="Bad force no qualifier", direction="increase",
                           affects=[], confidence=0.5, knowledge_origin="engineering"),
            ExtractedForce(name="Improved airtightness", direction="increase",
                           affects=[], confidence=0.8, knowledge_origin="engineering"),
        ]),
    )
    result = run_pass9(engine, p, "run-001", max_workers=1)

    assert result["forces_written"] == 1
    assert result["validation_failures"] == 1

    with Session(engine) as s:
        rows = s.exec(select(ForceRow)).all()
    assert len(rows) == 1
    assert rows[0].name == "Improved airtightness"


# ---------------------------------------------------------------------------
# Unresolved affects → pending_entity_refs
# ---------------------------------------------------------------------------

def test_pass9_unresolved_affects_written_to_pending_entity_refs(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Improved airtightness", "increase",
                            affects=["UnknownMaterial"])
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        force_row = s.exec(select(ForceRow)).first()
        pending = s.exec(select(PendingEntityRefRow)).all()

    assert force_row is not None
    assert len(pending) == 1
    assert pending[0].entity_name == "UnknownMaterial"
    assert pending[0].source_force_id == force_row.id


def test_pass9_partially_resolved_affects(engine):
    """Some affects resolve, some don't — resolved IDs stored, unresolved go to pending."""
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-ins", "Insulation")

    p = make_provider_force("Wall", "Improved thermal performance", "increase",
                            affects=["Insulation", "PhantomMaterial"])
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        force_row = s.exec(select(ForceRow)).first()
        pending = s.exec(select(PendingEntityRefRow)).all()

    affects_ids = json.loads(force_row.affects)
    assert "e-ins" in affects_ids
    assert len(pending) == 1
    assert pending[0].entity_name == "PhantomMaterial"


def test_pass9_unresolved_ref_does_not_prevent_force_write(engine):
    """Force row is still written even when all affects are unresolved."""
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Reduced heat loss", "decrease",
                            affects=["Ghost1", "Ghost2"])
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        force_rows = s.exec(select(ForceRow)).all()
        pending = s.exec(select(PendingEntityRefRow)).all()

    assert len(force_rows) == 1
    assert json.loads(force_rows[0].affects) == []
    assert len(pending) == 2


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------

def test_pass9_records_progress(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_force("Wall", "Improved airtightness", "increase")
    run_pass9(engine, p, "run-001", max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "9")
        ).all()
    assert len(progress) == 2
    assert all(pr.status == "completed" for pr in progress)


def test_pass9_resume_skips_completed(engine):
    add_entity(engine, "e-wall", "Wall")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    run_pass9(engine, p, "run-001", max_workers=1)
    first_count = call_count

    run_pass9(engine, p, "run-002", max_workers=1)
    assert call_count == first_count


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_pass9_dry_run_no_writes(engine):
    add_entity(engine, "e-wall", "Wall")

    p = make_provider_force("Wall", "Improved airtightness", "increase")
    result = run_pass9(engine, p, "__dry_run__", dry_run=True, max_workers=1)

    assert result["forces_written"] == 0
    assert result["entities_processed"] == 1

    with Session(engine) as s:
        assert len(s.exec(select(ForceRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


# ---------------------------------------------------------------------------
# Merged entities skipped
# ---------------------------------------------------------------------------

def test_pass9_skips_merged_entities(engine):
    add_entity(engine, "e-wall", "Wall", status="merged")
    add_entity(engine, "e-roof", "Roof")

    p = make_provider_force("Roof", "Improved weather resistance", "increase")
    result = run_pass9(engine, p, "run-001", max_workers=1)

    assert result["entities_processed"] == 1


# ---------------------------------------------------------------------------
# Prompt contains qualifier terms
# ---------------------------------------------------------------------------

def test_pass9_prompt_contains_qualifier_terms(engine):
    add_entity(engine, "e-wall", "Wall")

    captured_prompts = []

    class CapturingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            captured_prompts.append(prompt)
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CapturingProvider()
    run_pass9(engine, p, "run-001", max_workers=1)

    assert len(captured_prompts) >= 1
    prompt = captured_prompts[0]
    # Spot-check a few required qualifier words from each direction
    assert "improved" in prompt.lower()
    assert "reduced" in prompt.lower()
    assert "increase" in prompt.lower()
    assert "decrease" in prompt.lower()
