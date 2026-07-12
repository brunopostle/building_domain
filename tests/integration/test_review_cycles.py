"""Integration tests for cycle-conflicted process_relation review.

Sub-task 3 cycle detection (conflict_detection._run_cycle_detection) marks
every edge in a strongly-connected component conflicted directly, with no
conflict_pairs row (a cycle is N-shaped, not pair-shaped). Before
building_domain-8lp, `bsos review pending` had no path to these rows at all.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import Session

from bsos.cli.review import _cyclic_groups, _review_cycles
from bsos.normalization.conflict_detection import _run_cycle_detection
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import ConflictPairRow, EntityRow, ProcessRelationRow

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    db_path = tmp_path / "test.db"
    eng = create_db_engine(str(db_path))
    create_views(eng)
    return eng


def _make_entity(session: Session, name: str = "entity_a") -> EntityRow:
    row = EntityRow(
        id=str(uuid.uuid4()),
        name=name,
        entity_type="activity",
        status="accepted",
        source_model="test",
        created_at=NOW,
    )
    session.add(row)
    return row


def _make_pr(session: Session, pred_id: str, succ_id: str, source_model: str = "test") -> ProcessRelationRow:
    row = ProcessRelationRow(
        id=str(uuid.uuid4()),
        predecessor_id=pred_id,
        successor_id=succ_id,
        hard_constraint=True,
        source_model=source_model,
        created_at=NOW,
        confidence=0.9,
        status="proposed",
        knowledge_origin="extracted",
        rationale="test",
    )
    session.add(row)
    return row


class TestCyclicGroups:
    def test_groups_cyclic_edges_together(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = _make_pr(session, e1.id, e2.id)
            r2 = _make_pr(session, e2.id, e3.id)
            r3 = _make_pr(session, e3.id, e1.id)
            session.commit()
            ids = {r1.id, r2.id, r3.id}

        _run_cycle_detection(engine)

        with Session(engine) as session:
            groups = _cyclic_groups(session)

        assert len(groups) == 1
        assert {r.id for r in groups[0]} == ids

    def test_pair_shaped_conflicts_excluded(self, engine):
        """process_relations with a conflict_pairs row (Sub-task 2 divergence)
        must not show up in cycle groups — they're already reviewable via
        `bsos review pending --type conflict`."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            r1 = _make_pr(session, e1.id, e2.id, source_model="model-a")
            r2 = _make_pr(session, e1.id, e2.id, source_model="model-b")
            r1.status = "conflicted"
            r2.status = "conflicted"
            session.add(ConflictPairRow(
                id=str(uuid.uuid4()),
                item_a_id=r1.id,
                item_a_type="process_relation",
                item_b_id=r2.id,
                item_b_type="process_relation",
                detected_at=NOW,
                classification="contradictory",
            ))
            session.commit()

        with Session(engine) as session:
            groups = _cyclic_groups(session)

        assert groups == []

    def test_empty_when_no_conflicts(self, engine):
        with Session(engine) as session:
            groups = _cyclic_groups(session)
        assert groups == []


class TestReviewCycles:
    def test_stats_reports_counts(self, engine, capsys):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            _make_pr(session, e1.id, e2.id)
            _make_pr(session, e2.id, e3.id)
            _make_pr(session, e3.id, e1.id)
            session.commit()

        _run_cycle_detection(engine)

        with Session(engine) as session:
            reviewed = _review_cycles(session, limit=20, stats=True)

        assert reviewed == 0
        out = capsys.readouterr().out
        assert "3 edge(s) in 1 cycle(s)" in out

    def test_break_deprecates_selected_and_accepts_rest(self, engine, monkeypatch):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = _make_pr(session, e1.id, e2.id)
            r2 = _make_pr(session, e2.id, e3.id)
            r3 = _make_pr(session, e3.id, e1.id)
            session.commit()
            ids_in_order = [r1.id, r2.id, r3.id]

        _run_cycle_detection(engine)

        import typer
        monkeypatch.setattr(typer, "prompt", lambda *a, **k: "break=0")

        with Session(engine) as session:
            reviewed = _review_cycles(session, limit=20, stats=False)
            assert reviewed == 1

            groups = _cyclic_groups(session)
            # cycle is resolved: no edges remain conflicted-without-pair
            assert groups == []

            rows = [session.get(ProcessRelationRow, i) for i in ids_in_order]
            statuses = {r.status for r in rows}
            assert statuses == {"accepted", "deprecated"}
            deprecated = [r for r in rows if r.status == "deprecated"]
            assert len(deprecated) == 1

    def test_keep_all_accepts_every_edge(self, engine, monkeypatch):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = _make_pr(session, e1.id, e2.id)
            r2 = _make_pr(session, e2.id, e3.id)
            r3 = _make_pr(session, e3.id, e1.id)
            session.commit()
            ids = {r1.id, r2.id, r3.id}

        _run_cycle_detection(engine)

        import typer
        monkeypatch.setattr(typer, "prompt", lambda *a, **k: "keep-all")

        with Session(engine) as session:
            reviewed = _review_cycles(session, limit=20, stats=False)
            assert reviewed == 1

        with Session(engine) as session:
            for rid in ids:
                assert session.get(ProcessRelationRow, rid).status == "accepted"

    def test_defer_leaves_rows_conflicted(self, engine, monkeypatch):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            _make_pr(session, e1.id, e2.id)
            _make_pr(session, e2.id, e3.id)
            _make_pr(session, e3.id, e1.id)
            session.commit()

        _run_cycle_detection(engine)

        import typer
        monkeypatch.setattr(typer, "prompt", lambda *a, **k: "defer")

        with Session(engine) as session:
            reviewed = _review_cycles(session, limit=20, stats=False)
            assert reviewed == 0
            groups = _cyclic_groups(session)
            assert len(groups) == 1
