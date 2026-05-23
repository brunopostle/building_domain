"""Integration tests for the abstraction_node_effective_origins SQLite view.

Covers:
  - View creation via create_views / verify_views
  - get_effective_origins: correct knowledge_origin aggregation from child assertions
  - get_majority_origin: returns dominant origin (tie-breaking not guaranteed)
  - list_by_child: finds AbstractionNodes containing a given assertion UUID
  - count_proposed: counts only proposed nodes
  - View is absent when create_views has not been called (negative case)
  - Node with no matching assertions returns empty origins
"""
import json
import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import Session

from bsos.persistence.database import create_db_engine, create_views, verify_views
from bsos.persistence.models import AbstractionNodeRow, AssertionRow, EntityRow
from bsos.persistence.repos.abstraction import AbstractionNodeRepository

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    eng = create_db_engine(str(tmp_path / "test.db"))
    create_views(eng)
    return eng


@pytest.fixture
def engine_no_views(tmp_path):
    """Engine without views — for negative-case tests."""
    return create_db_engine(str(tmp_path / "no_views.db"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_entity(session, name: str = "roof") -> str:
    eid = str(uuid.uuid4())
    session.add(EntityRow(
        id=eid, name=name, entity_type="component", status="accepted",
        source_model="test", created_at=NOW,
    ))
    return eid


def _add_assertion(session, subject_id: str, object_id: str, origin: str = "physical") -> str:
    aid = str(uuid.uuid4())
    session.add(AssertionRow(
        id=aid, subject_id=subject_id, predicate="requires", object_id=object_id,
        subject_type="component", object_type="component",
        source_model="test", created_at=NOW,
        confidence=0.9, status="proposed", knowledge_origin=origin,
    ))
    return aid


def _add_node(session, child_ids: list[str], status: str = "proposed") -> str:
    nid = str(uuid.uuid4())
    session.add(AbstractionNodeRow(
        id=nid,
        statement="test abstraction",
        child_ids=json.dumps(child_ids),
        abstraction_rationale="test rationale",
        source_model="test",
        created_at=NOW,
        confidence=0.8,
        status=status,
    ))
    return nid


# ---------------------------------------------------------------------------
# View creation
# ---------------------------------------------------------------------------

class TestViewCreation:
    def test_verify_views_returns_true_after_create(self, engine):
        assert verify_views(engine) is True

    def test_verify_views_returns_false_without_create(self, engine_no_views):
        assert verify_views(engine_no_views) is False

    def test_create_views_is_idempotent(self, engine):
        """Calling create_views twice must not raise (uses CREATE VIEW IF NOT EXISTS)."""
        create_views(engine)  # second call
        assert verify_views(engine) is True


# ---------------------------------------------------------------------------
# get_effective_origins
# ---------------------------------------------------------------------------

class TestGetEffectiveOrigins:
    def test_single_origin(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s, "roof"); e2 = _add_entity(s, "rain")
            a1 = _add_assertion(s, e1, e2, origin="physical")
            a2 = _add_assertion(s, e1, e2, origin="physical")
            nid = _add_node(s, [a1, a2])
            s.commit()

            repo = AbstractionNodeRepository(s)
            origins = repo.get_effective_origins(nid)

        assert origins == {"physical": 2}

    def test_mixed_origins(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            a1 = _add_assertion(s, e1, e2, origin="physical")
            a2 = _add_assertion(s, e1, e2, origin="physical")
            a3 = _add_assertion(s, e1, e2, origin="engineering")
            nid = _add_node(s, [a1, a2, a3])
            s.commit()

            repo = AbstractionNodeRepository(s)
            origins = repo.get_effective_origins(nid)

        assert origins["physical"] == 2
        assert origins["engineering"] == 1

    def test_all_four_origins(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            aids = [_add_assertion(s, e1, e2, origin=o)
                    for o in ("physical", "engineering", "architectural", "cultural")]
            nid = _add_node(s, aids)
            s.commit()

            repo = AbstractionNodeRepository(s)
            origins = repo.get_effective_origins(nid)

        assert set(origins.keys()) == {"physical", "engineering", "architectural", "cultural"}
        assert all(v == 1 for v in origins.values())

    def test_no_matching_assertions_returns_empty(self, engine):
        """Node whose child_ids reference non-existent assertions → empty dict."""
        with Session(engine) as s:
            nid = _add_node(s, [str(uuid.uuid4()), str(uuid.uuid4())])
            s.commit()

            repo = AbstractionNodeRepository(s)
            origins = repo.get_effective_origins(nid)

        assert origins == {}

    def test_nonexistent_node_returns_empty(self, engine):
        with Session(engine) as s:
            repo = AbstractionNodeRepository(s)
            origins = repo.get_effective_origins(str(uuid.uuid4()))
        assert origins == {}

    def test_view_isolated_per_node(self, engine):
        """Origins for one node must not bleed into another node's query."""
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            a_phys = _add_assertion(s, e1, e2, origin="physical")
            a_eng = _add_assertion(s, e1, e2, origin="engineering")
            nid1 = _add_node(s, [a_phys])
            nid2 = _add_node(s, [a_eng])
            s.commit()

            repo = AbstractionNodeRepository(s)
            assert repo.get_effective_origins(nid1) == {"physical": 1}
            assert repo.get_effective_origins(nid2) == {"engineering": 1}


# ---------------------------------------------------------------------------
# get_majority_origin
# ---------------------------------------------------------------------------

class TestGetMajorityOrigin:
    def test_clear_majority(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            aids = [_add_assertion(s, e1, e2, origin="physical") for _ in range(3)]
            aids.append(_add_assertion(s, e1, e2, origin="engineering"))
            nid = _add_node(s, aids)
            s.commit()

            repo = AbstractionNodeRepository(s)
            majority = repo.get_majority_origin(nid)

        assert majority == "physical"

    def test_all_same_origin(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            aids = [_add_assertion(s, e1, e2, origin="architectural") for _ in range(5)]
            nid = _add_node(s, aids)
            s.commit()

            repo = AbstractionNodeRepository(s)
            assert repo.get_majority_origin(nid) == "architectural"

    def test_no_assertions_returns_none(self, engine):
        with Session(engine) as s:
            nid = _add_node(s, [str(uuid.uuid4())])
            s.commit()

            repo = AbstractionNodeRepository(s)
            assert repo.get_majority_origin(nid) is None


# ---------------------------------------------------------------------------
# list_by_child
# ---------------------------------------------------------------------------

class TestListByChild:
    def test_finds_node_containing_assertion(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            a1 = _add_assertion(s, e1, e2)
            a2 = _add_assertion(s, e1, e2)
            nid = _add_node(s, [a1, a2])
            s.commit()

            repo = AbstractionNodeRepository(s)
            result = repo.list_by_child(a1)

        assert len(result) == 1
        assert result[0].id == nid

    def test_returns_empty_when_not_child(self, engine):
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            a1 = _add_assertion(s, e1, e2)
            a2 = _add_assertion(s, e1, e2)
            _add_node(s, [a1])
            s.commit()

            repo = AbstractionNodeRepository(s)
            result = repo.list_by_child(a2)

        assert result == []

    def test_returns_multiple_nodes_sharing_child(self, engine):
        """An assertion can be a child of more than one abstraction node."""
        with Session(engine) as s:
            e1 = _add_entity(s); e2 = _add_entity(s, "obj")
            a1 = _add_assertion(s, e1, e2)
            nid1 = _add_node(s, [a1])
            nid2 = _add_node(s, [a1])
            s.commit()

            repo = AbstractionNodeRepository(s)
            result = repo.list_by_child(a1)

        assert len(result) == 2
        ids = {n.id for n in result}
        assert nid1 in ids and nid2 in ids


# ---------------------------------------------------------------------------
# count_proposed
# ---------------------------------------------------------------------------

class TestCountProposed:
    def test_counts_only_proposed(self, engine):
        with Session(engine) as s:
            _add_node(s, [], status="proposed")
            _add_node(s, [], status="proposed")
            _add_node(s, [], status="accepted")
            _add_node(s, [], status="deprecated")
            s.commit()

            repo = AbstractionNodeRepository(s)
            assert repo.count_proposed() == 2

    def test_empty_returns_zero(self, engine):
        with Session(engine) as s:
            repo = AbstractionNodeRepository(s)
            assert repo.count_proposed() == 0
