"""Integration tests for conflict detection — bsos validate --conflicts."""
import json
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.normalization.conflict_detection import (
    CONFLICT_QUEUE_CAP,
    SIMILARITY_THRESHOLD,
    _cascade_abstraction_nodes,
    _item_text,
    _run_conflict_detection,
    _run_cycle_detection,
    _run_process_relation_divergence,
    _assertion_pair_shares_entities,
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

    def vec_for(text: str) -> np.ndarray:
        # _item_text now prefixes "subj predicate obj" ahead of the raw
        # predicate token, so match by containment rather than exact
        # first-token equality (still robust to the added entity-name context).
        for predicate, vec in _PREDICATE_VEC.items():
            if predicate in text:
                return vec
        return default

    return np.array([vec_for(t) for t in texts], dtype=np.float32)


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


class ConcurrencyTrackingProvider:
    """Records peak in-flight extract() calls to prove classification runs in parallel."""

    def __init__(self, classification: str = "unrelated", delay: float = 0.05):
        self._classification = classification
        self._delay = delay
        self.model_id = "fake-llm"
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0
        self.calls = 0

    def extract(self, prompt: str, schema, **kwargs):
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
            self.calls += 1
        time.sleep(self._delay)
        with self._lock:
            self._current -= 1
        from bsos.normalization.conflict_detection import _ConflictClassification
        return _ConflictClassification(classification=self._classification, rationale="test")


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
# _item_text — entity-name-aware text used for both the embedding
# pre-filter and the LLM classification prompt
# ---------------------------------------------------------------------------

class TestAssertionPairSharesEntities:

    def test_true_for_same_pair_same_direction(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "supports", e1.id, e2.id)
            b = _make_assertion(session, "connects_to", e1.id, e2.id)
            session.commit()
            assert _assertion_pair_shares_entities(a, b) is True

    def test_true_for_reversed_pair(self, engine):
        """Same two entities, reversed relationship direction — a genuine
        candidate for contradiction, must remain eligible for LLM review."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "depends_on", e1.id, e2.id)
            b = _make_assertion(session, "depends_on", e2.id, e1.id)
            session.commit()
            assert _assertion_pair_shares_entities(a, b) is True

    def test_false_for_same_subject_different_object(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            a = _make_assertion(session, "supports", e1.id, e2.id)
            b = _make_assertion(session, "supports", e1.id, e3.id)
            session.commit()
            assert _assertion_pair_shares_entities(a, b) is False

    def test_false_for_chain_sharing_only_one_entity(self, engine):
        """A's object is B's subject, but the pairs don't otherwise match —
        two different facts about a chain, not a contradiction candidate."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            a = _make_assertion(session, "depends_on", e1.id, e2.id)
            b = _make_assertion(session, "depends_on", e2.id, e3.id)
            session.commit()
            assert _assertion_pair_shares_entities(a, b) is False

    def test_true_for_non_assertion_rows(self, engine):
        """Non-assertion pairs are unaffected by this check — still eligible
        for LLM classification exactly as before."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            c = ConstraintRow(id="c1", subject_id=e1.id, rule="must drain",
                              constraint_type="must", confidence=0.9,
                              knowledge_origin="engineering", source_model="test",
                              created_at=NOW)
            session.add(c)
            session.commit()
            assert _assertion_pair_shares_entities(c, c) is True


class TestItemText:

    def test_assertion_text_includes_entity_names_not_just_predicate(self, engine):
        """Two assertions sharing a predicate but concerning different entities
        must not collapse to the same 'predicate | rationale' text — that was
        the root cause of same-predicate-different-entity pairs being pulled
        into the similarity pre-filter and misclassified as contradictory."""
        with Session(engine) as session:
            wall = _make_entity(session, "Wall")
            window = _make_entity(session, "Window")
            door = _make_entity(session, "Door")
            frame = _make_entity(session, "Frame")
            a = _make_assertion(session, "connects_to", wall.id, window.id)
            a.rationale = "Windows are set into wall openings."
            b = _make_assertion(session, "connects_to", door.id, frame.id)
            b.rationale = "Doors are hung from their frames."
            session.commit()
            names = {
                wall.id: "Wall", window.id: "Window", door.id: "Door", frame.id: "Frame",
            }
            text_a = _item_text(a, names)
            text_b = _item_text(b, names)

        assert text_a != text_b
        assert "Wall" in text_a and "Window" in text_a
        assert "Door" in text_b and "Frame" in text_b

    def test_assertion_text_falls_back_to_raw_id_without_names(self, engine):
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            a = _make_assertion(session, "requires", e1.id, e2.id)
            session.commit()
            e1_id, e2_id = e1.id, e2.id
            text = _item_text(a)  # no names dict supplied

        assert e1_id in text
        assert e2_id in text
        assert "requires" in text


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

    def test_same_subject_predicate_different_object_skips_llm_and_is_complementary(self, engine):
        """One-to-many relationships (same subject+predicate, different
        object, e.g. 'Wall supports Window' / 'Wall supports Door') must
        never be flagged contradictory, and must be resolved without an LLM
        call — that shape is exactly the false-positive pattern that
        survived the entity-name-aware prompt fix."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            a = _make_assertion(session, "high_a", e1.id, e2.id)
            b = _make_assertion(session, "high_a", e1.id, e3.id)
            session.commit()
            a_id, b_id = a.id, b.id

        provider = FakeProvider("contradictory")  # would misclassify if ever reached
        result = _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        assert provider.calls == []
        assert result["conflicts_found"] == 0

        with Session(engine) as session:
            ra = session.get(AssertionRow, a_id)
            rb = session.get(AssertionRow, b_id)
            assert ra.status != "conflicted"
            assert rb.status != "conflicted"

            pair = session.exec(
                select(ConflictPairRow).where(
                    ((ConflictPairRow.item_a_id == a_id) & (ConflictPairRow.item_b_id == b_id))
                    | ((ConflictPairRow.item_a_id == b_id) & (ConflictPairRow.item_b_id == a_id))
                )
            ).first()
            assert pair is not None
            assert pair.classification == "complementary"

    def test_chain_sharing_one_entity_skips_llm_and_is_complementary(self, engine):
        """A chain (A's object is B's subject, no full pair overlap) —
        e.g. 'IfcPropertyAbstraction depends_on IfcSimpleProperty' and
        'IfcSimpleProperty depends_on IfcProperty' — survived the narrower
        same-subject-same-predicate rule; the broadened same-entity-pair
        rule must catch it too."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            a = _make_assertion(session, "high_a", e1.id, e2.id)
            b = _make_assertion(session, "high_b", e2.id, e3.id)
            session.commit()
            a_id, b_id = a.id, b.id

        provider = FakeProvider("contradictory")  # would misclassify if ever reached
        result = _run_conflict_detection(engine, provider, fake_embedder, limit=None)

        assert provider.calls == []
        assert result["conflicts_found"] == 0

        with Session(engine) as session:
            pair = session.exec(
                select(ConflictPairRow).where(
                    ((ConflictPairRow.item_a_id == a_id) & (ConflictPairRow.item_b_id == b_id))
                    | ((ConflictPairRow.item_a_id == b_id) & (ConflictPairRow.item_b_id == a_id))
                )
            ).first()
            assert pair is not None
            assert pair.classification == "complementary"

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

    def test_embeddings_are_cached_across_runs(self, engine):
        """Second sweep should only re-embed new/changed items, not the whole corpus."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            _make_assertion(session, "high_a", e1.id, e2.id)
            _make_assertion(session, "low_c", e1.id, e2.id)
            session.commit()

        calls: list[list[str]] = []

        def counting_embedder(texts: list[str]) -> np.ndarray:
            calls.append(list(texts))
            return fake_embedder(texts)

        provider = FakeProvider("unrelated")

        # First sweep embeds both existing items and populates the cache.
        _run_conflict_detection(engine, provider, counting_embedder, limit=None)
        assert len(calls) == 1
        assert any("high_a" in t for t in calls[0])
        assert any("low_c" in t for t in calls[0])
        assert len(calls[0]) == 2

        # Add a new assertion and reopen the sweep: only the new item's text
        # should reach the embedder, the other two must come from the cache.
        with Session(engine) as session:
            e3 = _make_entity(session, "e3")
            e1 = session.exec(select(EntityRow).where(EntityRow.name == "e1")).first()
            _make_assertion(session, "high_b", e1.id, e3.id)
            session.commit()

        _run_conflict_detection(engine, provider, counting_embedder, limit=None)

        assert len(calls) == 2
        assert len(calls[1]) == 1
        assert "high_b" in calls[1][0]

    def test_llm_classification_runs_concurrently(self, engine):
        """Multiple queued pairs should overlap on the thread pool, not run one at a time."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            # 4 mutually-similar assertions -> 6 candidate pairs to classify.
            for _ in range(4):
                _make_assertion(session, "high_a", e1.id, e2.id)
            session.commit()

        provider = ConcurrencyTrackingProvider(delay=0.05)
        _run_conflict_detection(engine, provider, fake_embedder, limit=None, workers=4)

        assert provider.calls == 6
        assert provider.max_concurrent >= 2


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

    def _make_pr(
        self, session: Session, pred_id: str, succ_id: str, subject_id: str | None = None,
    ) -> ProcessRelationRow:
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
            subject_id=subject_id,
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

    def test_cross_context_reversed_edges_not_flagged(self, engine):
        """building_domain-eue repro: two unrelated subjects assert opposite
        orderings for the same pair of generic activities (e.g. "Concrete
        Curing" vs "Waterproofing" for two different components). Each
        ordering is locally true; the union is not a real contradiction."""
        with Session(engine) as session:
            curing = _make_entity(session, "Concrete Curing")
            waterproofing = _make_entity(session, "Waterproofing")
            column = _make_entity(session, "Column")
            slab = _make_entity(session, "Slab")
            r1 = self._make_pr(session, curing.id, waterproofing.id, subject_id=column.id)
            r2 = self._make_pr(session, waterproofing.id, curing.id, subject_id=slab.id)
            session.commit()
            ids = {r1.id, r2.id}

        result = _run_cycle_detection(engine)

        assert result["cyclic_edges_marked"] == 0
        with Session(engine) as session:
            for rid in ids:
                assert session.get(ProcessRelationRow, rid).status == "proposed"

    def test_same_subject_cycle_still_flagged(self, engine):
        """Partitioning must not hide a real cycle asserted within one context."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            subject = _make_entity(session, "subject")
            r1 = self._make_pr(session, e1.id, e2.id, subject_id=subject.id)
            r2 = self._make_pr(session, e2.id, e3.id, subject_id=subject.id)
            r3 = self._make_pr(session, e3.id, e1.id, subject_id=subject.id)
            session.commit()
            ids = {r1.id, r2.id, r3.id}

        result = _run_cycle_detection(engine)

        assert result["cyclic_edges_marked"] == 3
        with Session(engine) as session:
            for rid in ids:
                assert session.get(ProcessRelationRow, rid).status == "conflicted"

    def test_universal_cycle_still_flagged(self, engine):
        """subject_id=NULL edges ("universal") are still checked for cycles
        among themselves — only cross-context unions are exempted."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = self._make_pr(session, e1.id, e2.id)
            r2 = self._make_pr(session, e2.id, e3.id)
            r3 = self._make_pr(session, e3.id, e1.id)
            session.commit()
            ids = {r1.id, r2.id, r3.id}

        result = _run_cycle_detection(engine)

        assert result["cyclic_edges_marked"] == 3
        with Session(engine) as session:
            for rid in ids:
                assert session.get(ProcessRelationRow, rid).status == "conflicted"

    def test_resolved_cycle_not_reflagged(self, engine):
        """A prior keep-all ReviewDecisionRow for this exact edge set means
        cycle detection must not flip the (accepted) edges back to conflicted."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = self._make_pr(session, e1.id, e2.id)
            r2 = self._make_pr(session, e2.id, e3.id)
            r3 = self._make_pr(session, e3.id, e1.id)
            r1.status = r2.status = r3.status = "accepted"
            ids = {r1.id, r2.id, r3.id}
            session.commit()

        from bsos.normalization.conflict_detection import cycle_hash

        with Session(engine) as session:
            session.add(ReviewDecisionRow(
                id=str(uuid.uuid4()),
                item_id=cycle_hash(ids),
                item_type="process_relation_cycle",
                decision="keep-all",
                rationale="cycle review: not actually a conflict, all edges correct",
                reviewer="human",
                created_at=NOW,
            ))
            session.commit()

        result = _run_cycle_detection(engine)

        assert result["cyclic_edges_marked"] == 0
        assert result["cycles_already_resolved"] == 1
        with Session(engine) as session:
            for rid in ids:
                assert session.get(ProcessRelationRow, rid).status == "accepted"

    def test_resolved_cycle_reflagged_when_edges_change(self, engine):
        """Adding a new edge into a previously-resolved cycle changes its
        identity hash, so it's treated as a fresh, unreviewed cycle."""
        with Session(engine) as session:
            e1 = _make_entity(session, "e1")
            e2 = _make_entity(session, "e2")
            e3 = _make_entity(session, "e3")
            r1 = self._make_pr(session, e1.id, e2.id)
            r2 = self._make_pr(session, e2.id, e3.id)
            r3 = self._make_pr(session, e3.id, e1.id)
            r1.status = r2.status = r3.status = "accepted"
            ids = {r1.id, r2.id, r3.id}
            session.commit()

        from bsos.normalization.conflict_detection import cycle_hash

        with Session(engine) as session:
            session.add(ReviewDecisionRow(
                id=str(uuid.uuid4()),
                item_id=cycle_hash(ids),
                item_type="process_relation_cycle",
                decision="keep-all",
                rationale="cycle review: not actually a conflict, all edges correct",
                reviewer="human",
                created_at=NOW,
            ))
            session.commit()

        with Session(engine) as session:
            e4 = _make_entity(session, "e4")
            e1 = session.exec(select(EntityRow).where(EntityRow.name == "e1")).first()
            r4 = self._make_pr(session, e4.id, e1.id)
            e2 = session.exec(select(EntityRow).where(EntityRow.name == "e2")).first()
            r5 = self._make_pr(session, e2.id, e4.id)  # e2 -> e4 -> e1 -> e2 grows the SCC
            session.commit()

        result = _run_cycle_detection(engine)

        assert result["cycles_already_resolved"] == 0
        assert result["cyclic_edges_marked"] == 5


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
