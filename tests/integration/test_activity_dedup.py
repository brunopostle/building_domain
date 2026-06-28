"""Integration tests for the activity-dedup step (building_domain-e9k).

Pass 5 mints near-duplicate ``activity`` entities because
``_get_or_create_activity`` matches existing activities by exact name only.
``run_activity_dedup`` folds the wording variants into one canonical via
embedding clustering, repointing the process_relations FKs through
``merge_entity``.
"""
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, ProcessRelationRow
from bsos.pipeline.pass5 import run_activity_dedup

NOW = datetime.now(timezone.utc)


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Two near-identical directions (cosine distance ~0.01) and one orthogonal.
DECK_VEC = _unit([1.0, 0.0, 0.0, 0.0])
DECK_DUP_VEC = _unit([0.99, 0.14, 0.0, 0.0])   # cos_dist ~0.01 from DECK_VEC
WALL_VEC = _unit([0.0, 1.0, 0.0, 0.0])          # cos_dist 1.0 from DECK_VEC


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


def add_activity(session: Session, eid: str, name: str, created: datetime = NOW,
                 entity_type: str = "activity") -> None:
    session.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                          status="proposed", source_model="test", created_at=created))


def add_proc(session: Session, pred: str, succ: str) -> None:
    session.add(ProcessRelationRow(
        id=str(uuid.uuid4()), predecessor_id=pred, successor_id=succ,
        hard_constraint=True, source_model="test", source_prompt="p",
        created_at=NOW, confidence=0.8, status="proposed",
        knowledge_origin="engineering", rationale="ordering",
    ))


def test_no_merge_when_all_distinct(session):
    add_activity(session, "a1", "Install Roof Deck")
    add_activity(session, "a2", "Build Wall")
    session.commit()

    embedder = make_fake_embedder({"Install Roof Deck": DECK_VEC, "Build Wall": WALL_VEC})
    result = run_activity_dedup(session, "run-1", _embedder=embedder)

    assert result["clusters_found"] == 0
    assert result["entities_merged"] == 0
    assert all(r.status != "merged" for r in session.exec(select(EntityRow)).all())


def test_merges_wording_variants(session):
    add_activity(session, "a1", "Roof Decking Installation")
    add_activity(session, "a2", "Install Roof Decking")
    add_activity(session, "a3", "Build Wall")
    session.commit()

    embedder = make_fake_embedder({
        "Roof Decking Installation": DECK_VEC,
        "Install Roof Decking": DECK_DUP_VEC,
        "Build Wall": WALL_VEC,
    })
    result = run_activity_dedup(session, "run-1", _embedder=embedder)

    assert result["clusters_found"] == 1
    assert result["entities_merged"] == 1
    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).all()
    assert len(merged) == 1


def test_only_activities_are_scoped(session):
    # A near-duplicate component pair must be left untouched.
    add_activity(session, "c1", "Roof Decking Installation", entity_type="component")
    add_activity(session, "c2", "Install Roof Decking", entity_type="component")
    session.commit()

    embedder = make_fake_embedder({
        "Roof Decking Installation": DECK_VEC,
        "Install Roof Decking": DECK_DUP_VEC,
    })
    result = run_activity_dedup(session, "run-1", _embedder=embedder)

    assert result["clusters_found"] == 0
    assert all(r.status != "merged" for r in session.exec(select(EntityRow)).all())


def test_process_relation_fks_repointed(session):
    # a1/a2 are duplicates; a2 carries the sequencing edge. After merge the edge
    # must point at the surviving canonical, not the merged-away duplicate.
    add_activity(session, "a1", "Roof Decking Installation")
    add_activity(session, "a2", "Install Roof Decking")
    add_activity(session, "wall", "Build Wall")
    add_proc(session, "wall", "a2")   # Build Wall -> Install Roof Decking
    session.commit()

    embedder = make_fake_embedder({
        "Roof Decking Installation": DECK_VEC,
        "Install Roof Decking": DECK_DUP_VEC,
        "Build Wall": WALL_VEC,
    })
    run_activity_dedup(session, "run-1", _embedder=embedder)

    canonical = session.exec(
        select(EntityRow).where(EntityRow.entity_type == "activity",
                                EntityRow.status != "merged",
                                EntityRow.name != "Build Wall")
    ).one()
    procs = session.exec(select(ProcessRelationRow)).all()
    assert len(procs) == 1
    assert procs[0].successor_id == canonical.id


def test_canonical_election_prefers_process_degree(session):
    # a1 has the sequencing edge, a2 does not -> a1 must win even though a2 is older.
    older = NOW - timedelta(days=1)
    add_activity(session, "a1", "Roof Decking Installation", created=NOW)
    add_activity(session, "a2", "Install Roof Decking", created=older)
    add_activity(session, "wall", "Build Wall")
    add_proc(session, "wall", "a1")
    session.commit()

    embedder = make_fake_embedder({
        "Roof Decking Installation": DECK_VEC,
        "Install Roof Decking": DECK_DUP_VEC,
        "Build Wall": WALL_VEC,
    })
    run_activity_dedup(session, "run-1", _embedder=embedder)

    merged = session.exec(select(EntityRow).where(EntityRow.status == "merged")).one()
    assert merged.name == "Install Roof Decking"  # a2 folded into a1


def test_dry_run_makes_no_changes(session):
    add_activity(session, "a1", "Roof Decking Installation")
    add_activity(session, "a2", "Install Roof Decking")
    session.commit()

    embedder = make_fake_embedder({
        "Roof Decking Installation": DECK_VEC,
        "Install Roof Decking": DECK_DUP_VEC,
    })
    result = run_activity_dedup(session, "run-1", _embedder=embedder, dry_run=True)

    assert result["clusters_found"] == 1
    assert result["entities_merged"] == 1
    assert all(r.status != "merged" for r in session.exec(select(EntityRow)).all())
