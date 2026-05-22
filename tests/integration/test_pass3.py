"""Integration tests for Pass 3 — Relationship Extraction."""
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import AssertionRow, EntityRow, PassProgressRow
from bsos.pipeline.pass3 import run_pass3, _group_assertions
from bsos.pipeline.schemas import (
    AssertionExtractionResponse, ExtractedAssertion,
)
from tests.fixtures.fake_responses import FakeLLMProvider

NOW = datetime.now(timezone.utc)
DIM = 8


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Predefined vectors: A and B are very similar (cos_sim ~0.99), C is orthogonal
VEC_A = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_B = _unit([0.99, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_C = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

TEXT_VECTORS = {
    "Roof requires Waterproof membrane": VEC_A,
    "Roof depends_on Waterproof membrane": VEC_B,
    "Roof contains Insulation": VEC_C,
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    return np.array(
        [TEXT_VECTORS.get(t, VEC_C) for t in texts],
        dtype=np.float32,
    )


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def add_entity(engine, eid: str, name: str, entity_type: str = "component") -> None:
    with Session(engine) as s:
        s.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                        source_model="test", created_at=NOW))
        s.commit()


def make_provider() -> FakeLLMProvider:
    p = FakeLLMProvider()
    # Framing 1: Roof requires Waterproof membrane
    p.register(
        AssertionExtractionResponse, "Roof",
        AssertionExtractionResponse(assertions=[
            ExtractedAssertion(predicate="requires", object_name="Waterproof membrane",
                               confidence=0.9, rationale="keeps water out"),
        ]),
    )
    return p


def test_pass3_writes_assertions(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wmem", "Waterproof membrane", "material")

    p = make_provider()
    result = run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)

    assert result["entities_processed"] == 2
    with Session(engine) as s:
        rows = s.exec(select(AssertionRow)).all()
    assert len(rows) >= 1
    pred_objs = {(r.predicate, r.object_id) for r in rows}
    assert ("requires", "e-wmem") in pred_objs


def test_pass3_skips_self_reference(engine):
    add_entity(engine, "e1", "Roof")

    p = FakeLLMProvider()
    p.register(
        AssertionExtractionResponse, "Roof",
        AssertionExtractionResponse(assertions=[
            ExtractedAssertion(predicate="requires", object_name="Roof", confidence=0.5),
        ]),
    )
    run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(AssertionRow)).all()
    assert len(rows) == 0


def test_pass3_skips_unresolved_object(engine):
    add_entity(engine, "e1", "Roof")

    p = FakeLLMProvider()
    p.register(
        AssertionExtractionResponse, "Roof",
        AssertionExtractionResponse(assertions=[
            ExtractedAssertion(predicate="requires", object_name="Unicorn",
                               confidence=0.5),
        ]),
    )
    run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(AssertionRow)).all()
    assert len(rows) == 0


def test_pass3_records_progress(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wmem", "Waterproof membrane", "material")

    p = make_provider()
    run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)

    with Session(engine) as s:
        progress = s.exec(
            select(PassProgressRow).where(PassProgressRow.pass_number == "3")
        ).all()
    assert len(progress) == 2  # one per entity
    assert all(pr.status == "completed" for pr in progress)


def test_pass3_resume_skips_completed(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wmem", "Waterproof membrane", "material")

    call_count = 0

    class CountingProvider(FakeLLMProvider):
        def extract(self, prompt, schema, *, entity_name=None):
            nonlocal call_count
            call_count += 1
            return super().extract(prompt, schema, entity_name=entity_name)

    p = CountingProvider()
    p.register(
        AssertionExtractionResponse, "Roof",
        AssertionExtractionResponse(assertions=[]),
    )

    run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)
    first_count = call_count

    # Second run — all entities already completed
    run_pass3(engine, p, "run-002", _embedder=fake_embedder, max_workers=1)
    assert call_count == first_count  # no additional LLM calls


def test_pass3_dry_run_no_writes(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wmem", "Waterproof membrane", "material")

    p = make_provider()
    result = run_pass3(engine, p, "__dry_run__", _embedder=fake_embedder,
                       dry_run=True, max_workers=1)

    assert result["assertions_written"] == 0
    assert result["entities_processed"] == 2

    with Session(engine) as s:
        assert len(s.exec(select(AssertionRow)).all()) == 0
        assert len(s.exec(select(PassProgressRow)).all()) == 0


def test_pass3_skips_merged_entities(engine):
    with Session(engine) as s:
        s.add(EntityRow(id="e1", name="Roof", entity_type="component",
                        status="merged", source_model="test", created_at=NOW))
        s.add(EntityRow(id="e2", name="Waterproof membrane", entity_type="material",
                        source_model="test", created_at=NOW))
        s.commit()

    p = make_provider()
    result = run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)

    # Only 1 active entity
    assert result["entities_processed"] == 1


def test_pass3_stores_knowledge_origin(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-wmem", "Waterproof membrane", "material")

    p = FakeLLMProvider()
    p.register(
        AssertionExtractionResponse, "Roof",
        AssertionExtractionResponse(assertions=[
            ExtractedAssertion(predicate="requires", object_name="Waterproof membrane",
                               knowledge_origin="physical", confidence=0.8),
        ]),
    )
    run_pass3(engine, p, "run-001", _embedder=fake_embedder, max_workers=1)

    with Session(engine) as s:
        rows = s.exec(select(AssertionRow)).all()
    assert any(r.knowledge_origin == "physical" for r in rows)


# ---------------------------------------------------------------------------
# Unit tests for _group_assertions
# ---------------------------------------------------------------------------

def test_group_assertions_singleton():
    framing_lists = [
        [ExtractedAssertion(predicate="requires", object_name="Membrane")],
        [],
        [],
    ]
    groups = _group_assertions(framing_lists, "Roof", fake_embedder)
    assert len(groups) == 1
    assert groups[0]["consistency"] is None
    assert groups[0]["framing_count"] == 1


def test_group_assertions_matches_similar():
    """Assertions with VEC_A and VEC_B should be grouped (cosine ~0.99 > 0.70)."""
    framing_lists = [
        [ExtractedAssertion(predicate="requires", object_name="Waterproof membrane")],
        [ExtractedAssertion(predicate="depends_on", object_name="Waterproof membrane")],
        [ExtractedAssertion(predicate="contains", object_name="Insulation")],
    ]
    groups = _group_assertions(framing_lists, "Roof", fake_embedder)
    # "requires Waterproof membrane" (VEC_A) and "depends_on Waterproof membrane" (VEC_B) match
    # "contains Insulation" (VEC_C) is orthogonal → singleton
    assert len(groups) == 2
    multi = [g for g in groups if g["framing_count"] > 1]
    assert len(multi) == 1
    assert multi[0]["consistency"] is not None
    assert multi[0]["consistency"] > 0.70


def test_group_assertions_empty_framings():
    groups = _group_assertions([[], [], []], "Roof", fake_embedder)
    assert groups == []


def test_group_assertions_consistency_stored():
    framing_lists = [
        [ExtractedAssertion(predicate="requires", object_name="Waterproof membrane")],
        [ExtractedAssertion(predicate="depends_on", object_name="Waterproof membrane")],
        [],
    ]
    groups = _group_assertions(framing_lists, "Roof", fake_embedder)
    matched = [g for g in groups if g["framing_count"] == 2]
    assert len(matched) == 1
    assert 0.0 < matched[0]["consistency"] <= 1.0
