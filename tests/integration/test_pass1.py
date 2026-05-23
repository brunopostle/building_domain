"""Integration tests for Pass 1 — Concept Discovery pipeline."""
import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, PassProgressRow
from bsos.pipeline.pass1 import run_pass1
from bsos.pipeline.schemas import (
    ConceptDiscoveryResponse, ConceptExpansionResponse, DiscoveredConcept,
)
from tests.fixtures.fake_responses import FakeLLMProvider


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def provider():
    p = FakeLLMProvider()
    p.register(
        ConceptDiscoveryResponse,
        "__bootstrap__",
        ConceptDiscoveryResponse(concepts=[
            DiscoveredConcept(name="Roof", entity_type="component", description="Top covering"),
            DiscoveredConcept(name="HVAC", entity_type="system", description="Climate control"),
        ]),
    )
    p.register(
        ConceptExpansionResponse,
        "Roof",
        ConceptExpansionResponse(sub_concepts=[
            DiscoveredConcept(name="Roof membrane", entity_type="material", description="Waterproof layer"),
            DiscoveredConcept(name="Roof insulation", entity_type="material", description="Thermal layer"),
        ]),
    )
    # HVAC expansion returns empty (FakeLLMProvider fallback)
    return p


def test_pass1_writes_entities(session, provider):
    run_pass1(session, provider, run_id="run-001")

    rows = session.exec(select(EntityRow)).all()
    names = {r.name for r in rows}
    assert "Roof" in names
    assert "HVAC" in names
    assert "Roof membrane" in names
    assert "Roof insulation" in names


def test_pass1_deduplication(session, provider):
    run_pass1(session, provider, run_id="run-001")

    rows = session.exec(select(EntityRow)).all()
    # No duplicate names (case-insensitive)
    lower_names = [r.name.lower() for r in rows]
    assert len(lower_names) == len(set(lower_names))


def test_pass1_records_progress(session, provider):
    run_pass1(session, provider, run_id="run-001")

    progress = session.exec(
        select(PassProgressRow).where(PassProgressRow.pass_number == "1")
    ).all()
    assert len(progress) == 1
    assert progress[0].status == "completed"
    assert progress[0].model == "fake-model"


def test_pass1_idempotent(session, provider):
    run_pass1(session, provider, run_id="run-001")
    run_pass1(session, provider, run_id="run-002")

    rows = session.exec(select(EntityRow)).all()
    names = [r.name.lower() for r in rows]
    # No duplicates even after two runs
    assert len(names) == len(set(names))


def test_pass1_dry_run_no_db_writes(session, provider):
    concepts = run_pass1(session, provider, run_id="__dry_run__", dry_run=True)

    assert len(concepts) > 0
    rows = session.exec(select(EntityRow)).all()
    assert len(rows) == 0
    progress = session.exec(select(PassProgressRow)).all()
    assert len(progress) == 0


def test_pass1_with_domain_seed(session):
    p = FakeLLMProvider()
    seed = "hospital building"
    entity_name = seed[:40].strip()
    p.register(
        ConceptDiscoveryResponse,
        entity_name,
        ConceptDiscoveryResponse(concepts=[
            DiscoveredConcept(name="Operating theatre", entity_type="space", description="Surgical room"),
        ]),
    )
    run_pass1(session, p, run_id="run-001", seed=seed)

    rows = session.exec(select(EntityRow)).all()
    names = {r.name for r in rows}
    assert "Operating theatre" in names


def test_pass1_seed_file_skips_discovery(session):
    """When seed_is_file_contents=True, skip LLM discovery; expand each listed name."""
    p = FakeLLMProvider()
    p.register(
        ConceptExpansionResponse,
        "Beam",
        ConceptExpansionResponse(sub_concepts=[
            DiscoveredConcept(name="Steel beam", entity_type="component", description="Metal structural member"),
        ]),
    )
    run_pass1(session, p, run_id="run-001", seed="Beam\nColumn\n", seed_is_file_contents=True)

    rows = session.exec(select(EntityRow)).all()
    names = {r.name for r in rows}
    assert "Beam" in names
    assert "Column" in names
    assert "Steel beam" in names


def test_pass1_returns_concepts(session, provider):
    concepts = run_pass1(session, provider, run_id="run-001")
    assert isinstance(concepts, list)
    assert len(concepts) >= 2


def test_pass1_entity_type_stored(session, provider):
    run_pass1(session, provider, run_id="run-001")

    row = session.exec(select(EntityRow).where(EntityRow.name == "HVAC")).one_or_none()
    assert row is not None
    assert row.entity_type == "system"


def test_pass1_source_model_stored(session, provider):
    run_pass1(session, provider, run_id="run-001")

    rows = session.exec(select(EntityRow)).all()
    assert all(r.source_model == "fake-model" for r in rows)


def test_pass1_apl_patterns_merged(session, provider):
    """APL pattern names are added as 'space' entities alongside bootstrap concepts."""
    apl = ["Light On Two Sides Of Every Room", "South Facing Outdoors", "Entrance Transition"]
    run_pass1(session, provider, run_id="run-001", apl_patterns=apl)

    rows = session.exec(select(EntityRow)).all()
    names = {r.name for r in rows}
    for pattern in apl:
        assert pattern in names

    # APL entities have entity_type "space"
    for row in rows:
        if row.name in apl:
            assert row.entity_type == "space"
            assert row.source_prompt == "apl_pattern_seed"


def test_pass1_apl_patterns_deduplicated(session, provider):
    """APL patterns with same name as a bootstrap concept are not double-added."""
    # 'Roof' is already in the bootstrap fixture
    apl = ["Roof", "South Facing Outdoors"]
    run_pass1(session, provider, run_id="run-001", apl_patterns=apl)

    rows = session.exec(select(EntityRow)).all()
    lower_names = [r.name.lower() for r in rows]
    assert lower_names.count("roof") == 1
