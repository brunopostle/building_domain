"""Integration tests for Pass 2 — Entity Deduplication."""
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AssertionRow, EmbeddingRow, EntityAliasRow, EntityRow, PassProgressRow,
)
from bsos.pipeline.pass2 import run_pass2

NOW = datetime.now(timezone.utc)
DIM = 8


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Two near-identical directions (cosine distance ~0.02) and one orthogonal direction
ROOF_VEC = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ROOF_DUP_VEC = _unit([0.99, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # cos_dist ~0.01
WALL_VEC = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # cos_dist 1.0 from Roof


def make_fake_embedder(name_to_vec: dict[str, np.ndarray]):
    def embedder(texts: list[str]) -> np.ndarray:
        return np.array([name_to_vec[t] for t in texts], dtype=np.float32)
    return embedder


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


def add_entity(session: Session, eid: str, name: str, entity_type: str = "component") -> EntityRow:
    row = EntityRow(
        id=eid,
        name=name,
        entity_type=entity_type,
        source_model="test",
        created_at=NOW,
    )
    session.add(row)
    return row


def add_assertion(session: Session, aid: str, subject_id: str, object_id: str) -> AssertionRow:
    row = AssertionRow(
        id=aid,
        subject_id=subject_id,
        predicate="requires",
        object_id=object_id,
        subject_type="component",
        object_type="component",
        source_model="test",
        created_at=NOW,
        confidence=0.9,
        knowledge_origin="physical",
    )
    session.add(row)
    return row


def test_pass2_no_duplicates(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Wall")
    session.commit()

    embedder = make_fake_embedder({"Roof": ROOF_VEC, "Wall": WALL_VEC})
    result = run_pass2(session, "run-001", _embedder=embedder)

    assert result["clusters_found"] == 0
    assert result["entities_merged"] == 0
    rows = session.exec(select(EntityRow)).all()
    assert all(r.status != "merged" for r in rows)


def test_pass2_merges_near_duplicates(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "roof covering")
    add_entity(session, "e3", "Wall")
    session.commit()

    embedder = make_fake_embedder({
        "Roof": ROOF_VEC,
        "roof covering": ROOF_DUP_VEC,
        "Wall": WALL_VEC,
    })
    result = run_pass2(session, "run-001", _embedder=embedder)

    assert result["clusters_found"] == 1
    assert result["entities_merged"] == 1

    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).all()
    assert len(merged) == 1


def test_pass2_canonical_election_by_assertion_count(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "roof covering")
    add_entity(session, "e3", "Wall")
    # e2 has more assertions → should be canonical
    add_assertion(session, "a1", "e2", "e3")
    add_assertion(session, "a2", "e2", "e3")
    session.commit()

    embedder = make_fake_embedder({
        "Roof": ROOF_VEC,
        "roof covering": ROOF_DUP_VEC,
        "Wall": WALL_VEC,
    })
    run_pass2(session, "run-001", _embedder=embedder)

    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).one()
    assert merged.name == "Roof"  # fewer assertions → merged


def test_pass2_updates_assertion_fks(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "roof covering")
    add_entity(session, "e3", "Wall")
    add_assertion(session, "a1", "e2", "e3")  # subject points to duplicate
    add_assertion(session, "a2", "e3", "e2")  # object points to duplicate
    session.commit()

    embedder = make_fake_embedder({
        "Roof": ROOF_VEC,
        "roof covering": ROOF_DUP_VEC,
        "Wall": WALL_VEC,
    })
    run_pass2(session, "run-001", _embedder=embedder)

    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).one()
    canonical_id = next(
        e.id for e in session.exec(select(EntityRow)).all()
        if e.id != merged.id and e.name in ("Roof", "roof covering")
    )

    for row in session.exec(select(AssertionRow)).all():
        assert row.subject_id != merged.id
        assert row.object_id != merged.id
        assert row.subject_id in (canonical_id, "e3")
        assert row.object_id in (canonical_id, "e3")


def test_pass2_adds_alias(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "roof covering")
    session.commit()

    embedder = make_fake_embedder({"Roof": ROOF_VEC, "roof covering": ROOF_DUP_VEC})
    run_pass2(session, "run-001", _embedder=embedder)

    aliases = session.exec(select(EntityAliasRow)).all()
    assert len(aliases) == 1
    alias_names = {a.alias for a in aliases}
    # The merged entity's name becomes an alias on the canonical
    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).one()
    assert merged.name in alias_names


def test_pass2_dry_run_no_writes(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "roof covering")
    session.commit()

    embedder = make_fake_embedder({"Roof": ROOF_VEC, "roof covering": ROOF_DUP_VEC})
    result = run_pass2(session, "__dry_run__", _embedder=embedder, dry_run=True)

    assert result["clusters_found"] == 1
    assert result["entities_merged"] == 1

    rows = session.exec(select(EntityRow)).all()
    assert all(r.status != "merged" for r in rows)
    aliases = session.exec(select(EntityAliasRow)).all()
    assert len(aliases) == 0
    progress = session.exec(select(PassProgressRow)).all()
    assert len(progress) == 0


def test_pass2_records_progress(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Wall")
    session.commit()

    embedder = make_fake_embedder({"Roof": ROOF_VEC, "Wall": WALL_VEC})
    run_pass2(session, "run-001", _embedder=embedder)

    progress = session.exec(
        select(PassProgressRow).where(PassProgressRow.pass_number == "2")
    ).all()
    assert len(progress) == 1
    assert progress[0].status == "completed"


def test_pass2_caches_embeddings(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Wall")
    session.commit()

    call_count = 0

    def counting_embedder(texts: list[str]) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.array(
            [ROOF_VEC if t == "Roof" else WALL_VEC for t in texts], dtype=np.float32
        )

    run_pass2(session, "run-001", _embedder=counting_embedder)
    first_count = call_count

    run_pass2(session, "run-002", _embedder=counting_embedder)
    # Second run should not call embedder again (cached)
    assert call_count == first_count

    cached = session.exec(select(EmbeddingRow)).all()
    assert len(cached) == 2


def test_pass2_idempotent(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "roof covering")
    add_entity(session, "e3", "Wall")
    session.commit()

    embedder = make_fake_embedder({
        "Roof": ROOF_VEC,
        "roof covering": ROOF_DUP_VEC,
        "Wall": WALL_VEC,
    })
    run_pass2(session, "run-001", _embedder=embedder)
    # Second run must not crash or produce extra merges
    run_pass2(session, "run-002", _embedder=embedder)

    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).all()
    assert len(merged) == 1


def test_pass2_skips_already_merged(session):
    add_entity(session, "e1", "Roof")
    dup = add_entity(session, "e2", "roof covering")
    dup.status = "merged"
    session.commit()

    embedder = make_fake_embedder({"Roof": ROOF_VEC})
    result = run_pass2(session, "run-001", _embedder=embedder)

    assert result["entities_merged"] == 0


def test_pass2_single_entity_records_progress(session):
    add_entity(session, "e1", "Roof")
    session.commit()

    embedder = make_fake_embedder({"Roof": ROOF_VEC})
    run_pass2(session, "run-001", _embedder=embedder)

    progress = session.exec(
        select(PassProgressRow).where(PassProgressRow.pass_number == "2")
    ).all()
    assert len(progress) == 1
