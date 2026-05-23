"""Integration tests for conflict detection — bsos validate --conflicts."""
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.normalization.conflict_detection import (
    CONFLICT_QUEUE_CAP,
    SIMILARITY_THRESHOLD,
    _cascade_abstraction_nodes,
    _run_conflict_detection,
    _run_cycle_detection,
    _run_process_relation_divergence,
    run_conflict_detection,
)
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AbstractionNodeRow,
    AssertionRow,
    ConflictPairRow,
    ConstraintRow,
    EntityRow,
    ProcessRelationRow,
    ProvenanceLogRow,
    ReviewDecisionRow,
)

NOW = datetime.now(timezone.utc)
DIM = 8


# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / float(np.linalg.norm(a))


# High-similarity pair (cos_sim ≈ 0.99 → above SIMILARITY_THRESHOLD)
VEC_A = _unit([1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_B = _unit([0.99, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Low-similarity vector (cos_sim ≈ 0.0 → below SIMILARITY_THRESHOLD)
VEC_LOW = _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

_PREDICATE_VEC: dict[str, np.ndarray] = {
    "high_a": VEC_A,
    "high_b": VEC_B,
    "low_c": VEC_LOW,
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    default = np.zeros(DIM, dtype=np.float32)
    default[7] = 1.0
    return np.array([_PREDICATE_VEC.get(t.split(" | ")[0], default) for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake LLM provider
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, classification: str = "contradictory", model_id: str = "fake-llm"):
        self._classification = classification
        self._model_id = model_id
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def extract(self, prompt: str, schema, **kwargs):
        self.calls.append(prompt)
        from bsos.normalization.conflict_detection import _ConflictClassification
        return _ConflictClassification(
            classification=self._classification,
            rationale="test rationale",
        )

    def classify(self, prompt: str, options: list[str]) -> str:
        return self._classification


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------

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
        entity_type="space",
        status="accepted",
        source_model="test",
        created_at=NOW,
    )
    session.add(row)
    return row


def _make_assertion(
    session: Session,
    predicate: str,
    subject_id: str,
    object_id: str,
    status: str = "proposed",
    conflict_evaluated_at=None,
) -> AssertionRow:
    row = AssertionRow(
        id=str(uuid.uuid4()),
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        subject_type="space",
        object_type="space",
        source_model="test",
        created_at=NOW,
        confidence=0.9,
        status=status,
        knowledge_origin="extracted",
        conflict_evaluated_at=conflict_evaluated_at,
    )
    session.add(row)
    return row


# ---------------------------------------------------------------------------
# Sub-task 1: Assertion conflict detection
# ---------------------------------------------------------------------------

class TestAssertionConflictDetection:

    def test_contradictory_pair_marked_conflicted(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "high_a", e1.id, e2.id)
            b = _make_assertion(session, "high_b", e1.id, e2.id)
            session.commit()
            a_id, b_id = a.id, b.id

        provider = FakeProvider("contradictory")
        result = _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        assert result["conflicts_found"] >= 1

        with Session(engine) as session:
            ra = session.get(AssertionRow, a_id)
            rb = session.get(AssertionRow, b_id)
            assert ra.status == "conflicted"
            assert rb.status == "conflicted"

            pair = session.exec(
                select(ConflictPairRow).where(
                    ((ConflictPairRow.item_a_id == a_id) & (ConflictPairRow.item_b_id == b_id))
                    | ((ConflictPairRow.item_a_id == b_id) & (ConflictPairRow.item_b_id == a_id))
                )
            ).first()
            assert pair is not None
            assert pair.classification == "contradictory"

            prov = session.exec(
                select(ProvenanceLogRow).where(ProvenanceLogRow.item_id == a_id)
            ).first()
            assert prov is not None
            assert prov.new_status == "conflicted"

    def test_duplicate_pair_writes_conflict_pair_but_no_status_change(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "high_a", e1.id, e2.id)
            b = _make_assertion(session, "high_b", e1.id, e2.id)
            session.commit()
            a_id, b_id = a.id, b.id

        provider = FakeProvider("duplicate")
        _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        with Session(engine) as session:
            ra = session.get(AssertionRow, a_id)
            rb = session.get(AssertionRow, b_id)
            # Status unchanged — only contradictory triggers status='conflicted'
            assert ra.status == "proposed"
            assert rb.status == "proposed"

            pair = session.exec(select(ConflictPairRow)).first()
            assert pair is not None
            assert pair.classification == "duplicate"

    def test_low_similarity_pair_skipped(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            _make_assertion(session, "high_a", e1.id, e2.id)
            _make_assertion(session, "low_c", e1.id, e2.id)
            session.commit()

        provider = FakeProvider("contradictory")
        result = _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        # No LLM calls because similarity below threshold
        assert result["llm_calls"] == 0
        assert result["conflicts_found"] == 0

    def test_limit_stops_early(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            _make_assertion(session, "high_a", e1.id, e2.id)
            _make_assertion(session, "high_b", e1.id, e2.id)
            _make_assertion(session, "high_b", e1.id, e3.id)
            session.commit()

        provider = FakeProvider("contradictory")
        result = _run_conflict_detection(engine, provider, fake_embedder, limit=1)

        assert result["llm_calls"] <= 1

    def test_already_evaluated_skipped(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            _make_assertion(session, "high_a", e1.id, e2.id, conflict_evaluated_at=NOW)
            _make_assertion(session, "high_b", e1.id, e2.id, conflict_evaluated_at=NOW)
            session.commit()

        provider = FakeProvider("contradictory")
        result = _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        assert result["llm_calls"] == 0

    def test_existing_conflict_pair_reuses_without_reclassification(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "high_a", e1.id, e2.id)
            b = _make_assertion(session, "high_b", e1.id, e2.id)
            session.add(ConflictPairRow(
                id=str(uuid.uuid4()),
                item_a_id=a.id,
                item_a_type="assertion",
                item_b_id=b.id,
                item_b_type="assertion",
                detected_at=NOW,
                classification="contradictory",
            ))
            session.commit()
            a_id, b_id = a.id, b.id

        provider = FakeProvider("complementary")  # Would say complementary if called
        _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        with Session(engine) as session:
            # Status changes still applied from existing contradictory pair
            ra = session.get(AssertionRow, a_id)
            rb = session.get(AssertionRow, b_id)
            assert ra.status == "conflicted"
            assert rb.status == "conflicted"
        # But provider was NOT called (existing pair reused)
        assert len(provider.calls) == 0

    def test_conflict_evaluated_at_stamped(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "high_a", e1.id, e2.id)
            session.commit()
            a_id = a.id

        provider = FakeProvider("unrelated")
        _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        with Session(engine) as session:
            ra = session.get(AssertionRow, a_id)
            assert ra.conflict_evaluated_at is not None


# ---------------------------------------------------------------------------
# Sub-task 2: ProcessRelation divergence
# ---------------------------------------------------------------------------

class TestProcessRelationDivergence:

    def _make_pr(self, session: Session, pred_id: str, succ_id: str,
                 hard: bool, source_model: str, status: str = "proposed") -> ProcessRelationRow:
        row = ProcessRelationRow(
            id=str(uuid.uuid4()),
            predecessor_id=pred_id,
            successor_id=succ_id,
            hard_constraint=hard,
            source_model=source_model,
            created_at=NOW,
            confidence=0.9,
            status=status,
            knowledge_origin="extracted",
            rationale="test",
        )
        session.add(row)
        return row

    def test_disagreement_marks_conflicted(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            r1 = self._make_pr(session, e1.id, e2.id, hard=True, source_model="model_a")
            r2 = self._make_pr(session, e1.id, e2.id, hard=False, source_model="model_b")
            session.commit()
            r1_id, r2_id = r1.id, r2.id

        result = _run_process_relation_divergence(engine)

        assert result["divergences_found"] == 1
        with Session(engine) as session:
            for rid in (r1_id, r2_id):
                row = session.get(ProcessRelationRow, rid)
                assert row.status == "conflicted"
            review = session.exec(select(ReviewDecisionRow)).first()
            assert review is not None
            assert review.decision == "defer"

    def test_agreement_not_flagged(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            self._make_pr(session, e1.id, e2.id, hard=True, source_model="model_a")
            self._make_pr(session, e1.id, e2.id, hard=True, source_model="model_b")
            session.commit()

        result = _run_process_relation_divergence(engine)

        assert result["divergences_found"] == 0

    def test_single_source_not_flagged(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            self._make_pr(session, e1.id, e2.id, hard=True, source_model="model_a")
            session.commit()

        result = _run_process_relation_divergence(engine)

        assert result["divergences_found"] == 0


# ---------------------------------------------------------------------------
# Sub-task 3: Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:

    def _make_pr(self, session: Session, pred_id: str, succ_id: str) -> ProcessRelationRow:
        row = ProcessRelationRow(
            id=str(uuid.uuid4()),
            predecessor_id=pred_id,
            successor_id=succ_id,
            hard_constraint=True,
            source_model="test",
            created_at=NOW,
            confidence=0.9,
            status="proposed",
            knowledge_origin="extracted",
            rationale="test",
        )
        session.add(row)
        return row

    def test_cycle_detected_and_marked(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = self._make_pr(session, e1.id, e2.id)
            r2 = self._make_pr(session, e2.id, e3.id)
            r3 = self._make_pr(session, e3.id, e1.id)  # closes cycle
            session.commit()
            ids = {r1.id, r2.id, r3.id}

        result = _run_cycle_detection(engine)

        assert result["cycles_found"] >= 1
        assert result["cyclic_edges_marked"] == 3

        with Session(engine) as session:
            for rid in ids:
                row = session.get(ProcessRelationRow, rid)
                assert row.status == "conflicted"

    def test_acyclic_graph_untouched(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = self._make_pr(session, e1.id, e2.id)
            r2 = self._make_pr(session, e2.id, e3.id)
            session.commit()
            ids = {r1.id, r2.id}

        result = _run_cycle_detection(engine)

        assert result["cycles_found"] == 0
        assert result["cyclic_edges_marked"] == 0

        with Session(engine) as session:
            for rid in ids:
                row = session.get(ProcessRelationRow, rid)
                assert row.status == "proposed"

    def test_empty_graph(self, engine):
        result = _run_cycle_detection(engine)
        assert result["cycles_found"] == 0


# ---------------------------------------------------------------------------
# Sub-task 4: AbstractionNode cascade
# ---------------------------------------------------------------------------

class TestAbstractionNodeCascade:

    def _make_abstraction(
        self, session: Session, child_ids: list[str], status: str = "proposed"
    ) -> AbstractionNodeRow:
        row = AbstractionNodeRow(
            id=str(uuid.uuid4()),
            statement="test abstraction",
            child_ids=json.dumps(child_ids),
            abstraction_rationale="test",
            source_model="test",
            created_at=NOW,
            confidence=0.9,
            status=status,
        )
        session.add(row)
        return row

    def test_majority_conflicted_children_cascade(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a1 = _make_assertion(session, "p1", e1.id, e2.id, status="conflicted")
            a2 = _make_assertion(session, "p2", e1.id, e2.id, status="conflicted")
            a3 = _make_assertion(session, "p3", e1.id, e2.id, status="proposed")
            node = self._make_abstraction(session, [a1.id, a2.id, a3.id])
            session.commit()
            node_id = node.id
            a1_id, a2_id = a1.id, a2.id

        result = _cascade_abstraction_nodes(engine, {a1_id, a2_id})

        assert result["abstraction_nodes_conflicted"] == 1
        with Session(engine) as session:
            n = session.get(AbstractionNodeRow, node_id)
            assert n.status == "conflicted"

    def test_minority_conflicted_children_no_cascade(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a1 = _make_assertion(session, "p1", e1.id, e2.id, status="conflicted")
            a2 = _make_assertion(session, "p2", e1.id, e2.id, status="proposed")
            a3 = _make_assertion(session, "p3", e1.id, e2.id, status="proposed")
            node = self._make_abstraction(session, [a1.id, a2.id, a3.id])
            session.commit()
            node_id = node.id
            a1_id = a1.id

        result = _cascade_abstraction_nodes(engine, {a1_id})

        assert result["abstraction_nodes_conflicted"] == 0
        with Session(engine) as session:
            n = session.get(AbstractionNodeRow, node_id)
            assert n.status == "proposed"

    def test_already_conflicted_node_skipped(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a1 = _make_assertion(session, "p1", e1.id, e2.id, status="conflicted")
            a2 = _make_assertion(session, "p2", e1.id, e2.id, status="conflicted")
            node = self._make_abstraction(session, [a1.id, a2.id], status="conflicted")
            session.commit()
            node_id = node.id
            a1_id, a2_id = a1.id, a2.id

        result = _cascade_abstraction_nodes(engine, {a1_id, a2_id})

        # Re-evaluated count should be 0 (already conflicted, skipped)
        assert result["abstraction_nodes_conflicted"] == 0


# ---------------------------------------------------------------------------
# Dry-run integration
# ---------------------------------------------------------------------------

class TestDryRun:

    def test_dry_run_no_writes(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            _make_assertion(session, "high_a", e1.id, e2.id)
            session.commit()

        provider = FakeProvider("contradictory")
        result = run_conflict_detection(
            engine, provider, _embedder=fake_embedder, dry_run=True
        )

        assert result["dry_run"] is True
        assert "unevaluated_assertions" in result

        with Session(engine) as session:
            pairs = session.exec(select(ConflictPairRow)).all()
            assert len(pairs) == 0
