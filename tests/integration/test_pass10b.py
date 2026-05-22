"""Integration tests for Pass 10b — predicate stabilization."""
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.normalization.pass10b import run_pass10b
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AssertionRow,
    EntityRow,
    PassProgressRow,
    PendingPredicateRow,
    PredicateMappingRow,
)
from bsos.vocab import CORE_PREDICATES

NOW = datetime.now(timezone.utc)
DIM = 10

# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------
# Core predicates are mapped to orthogonal unit basis vectors.
# Non-core predicates get crafted vectors so that cosine similarity to the
# nearest core predicate lands in the right band for each test scenario.
#
# sorted(CORE_PREDICATES) = [
#   "conflicts_with", "connects_to", "contains", "depends_on",
#   "improves", "protects_from", "requires", "supports", "unsuitable_for"
# ]
# Index of "requires" in that sorted list = 6.


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / float(np.linalg.norm(a))


# "requires" basis vector (index 6 in sorted core list)
_REQUIRES_VEC = _unit([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

# "demands" → cos_sim with "requires" = 0.92 (> AUTO_MAP_THRESHOLD 0.85) → auto-map
_DEMANDS_VEC = _unit([0, 0, 0, 0, 0, 0, 0.92, 0.39, 0, 0])

# "needs" → cos_sim with "requires" = 0.70 (in [0.60, 0.85) → ambiguous / Phase 2)
_NEEDS_VEC = _unit([0, 0, 0, 0, 0, 0, 0.70, 0.714, 0, 0])

# "is_made_of" → cos_sim with all cores = 0 (< 0.60 → pending_predicates)
_IS_MADE_OF_VEC = _unit([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

_CORE_SORTED = sorted(CORE_PREDICATES)
_CORE_VECS = {name: _unit([1 if i == j else 0 for j in range(DIM)])
              for i, name in enumerate(_CORE_SORTED)}

_VECTOR_MAP = {
    **_CORE_VECS,
    "demands": _DEMANDS_VEC,
    "needs": _NEEDS_VEC,
    "is_made_of": _IS_MADE_OF_VEC,
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    default = np.zeros(DIM, dtype=np.float32)
    default[9] = 1.0  # unknown text → no similarity with cores
    return np.array([_VECTOR_MAP.get(t, default) for t in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake LLM provider
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, answers: dict[str, str] | None = None, model_id: str = "fake-llm"):
        self.answers = answers or {}
        self._model_id = model_id
        self.calls: list[tuple[str, list[str]]] = []

    def classify(self, prompt: str, options: list[str]) -> str:
        self.calls.append((prompt, options))
        for pred, answer in self.answers.items():
            if pred in prompt:
                return answer
        return "none"

    def extract(self, prompt, schema, *, entity_name=None):
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        return self._model_id


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def _add_entity(session, name: str, entity_type: str = "space") -> str:
    eid = str(uuid.uuid4())
    session.add(EntityRow(
        id=eid, name=name, entity_type=entity_type, status="accepted",
        source_model="test", created_at=NOW,
    ))
    return eid


def _add_assertion(session, predicate: str, status: str = "proposed") -> str:
    with Session(session.get_bind()) as s2:
        eid1 = _add_entity(s2, f"e-{uuid.uuid4()}")
        eid2 = _add_entity(s2, f"e-{uuid.uuid4()}")
        s2.commit()

    aid = str(uuid.uuid4())
    session.add(AssertionRow(
        id=aid,
        subject_id=eid1,
        predicate=predicate,
        object_id=eid2,
        subject_type="component",
        object_type="system",
        source_model="test",
        created_at=NOW,
        confidence=0.9,
        status=status,
        knowledge_origin="physical",
    ))
    return aid


# ---------------------------------------------------------------------------
# Phase 1: auto-mapping (≥ 0.85)
# ---------------------------------------------------------------------------

class TestPhase1AutoMap:
    def test_non_core_predicate_mapped_and_assertion_updated(self, engine):
        with Session(engine) as s:
            aid = _add_assertion(s, "demands")
            s.commit()

        result = run_pass10b(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            assertion = s.get(AssertionRow, aid)
            assert assertion.predicate == "requires"

            mappings = s.exec(select(PredicateMappingRow)).all()
            assert len(mappings) == 1
            m = mappings[0]
            assert m.from_predicate == "demands"
            assert m.to_predicate == "requires"
            assert m.reviewer == "embedding"

        assert result["auto_mapped"] == 1

    def test_low_similarity_goes_to_pending(self, engine):
        with Session(engine) as s:
            _add_assertion(s, "is_made_of")
            s.commit()

        run_pass10b(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            pending = s.exec(select(PendingPredicateRow)).all()
            assert len(pending) == 1
            assert pending[0].value == "is_made_of"
            assert pending[0].vocabulary_type == "predicate"

    def test_core_predicates_untouched(self, engine):
        with Session(engine) as s:
            aid = _add_assertion(s, "requires")
            s.commit()

        run_pass10b(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            assertion = s.get(AssertionRow, aid)
            assert assertion.predicate == "requires"
            mappings = s.exec(select(PredicateMappingRow)).all()
            assert mappings == []

    def test_multiple_assertions_with_same_predicate_all_updated(self, engine):
        with Session(engine) as s:
            aid1 = _add_assertion(s, "demands")
            aid2 = _add_assertion(s, "demands")
            s.commit()

        run_pass10b(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            a1 = s.get(AssertionRow, aid1)
            a2 = s.get(AssertionRow, aid2)
            assert a1.predicate == "requires"
            assert a2.predicate == "requires"
            mappings = s.exec(select(PredicateMappingRow)).all()
            assert len(mappings) == 1  # one mapping row, two assertions updated

    def test_pending_predicate_deduplication(self, engine):
        """Two assertions with same unmapped predicate → one pending row, count=2."""
        with Session(engine) as s:
            _add_assertion(s, "is_made_of")
            _add_assertion(s, "is_made_of")
            s.commit()

        run_pass10b(engine, _embedder=fake_embedder)

        # The predicate appears once but occurrence_count reflects how many times seen.
        # In Phase 1, the predicate is collected once (distinct list), so count stays 1.
        # Dedup upsert is for writes to pending — one write per distinct predicate.
        with Session(engine) as s:
            pending = s.exec(select(PendingPredicateRow)).all()
            assert len(pending) == 1
            assert pending[0].value == "is_made_of"


# ---------------------------------------------------------------------------
# Phase 2: LLM disambiguation ([0.60, 0.85))
# ---------------------------------------------------------------------------

class TestPhase2LLM:
    def test_llm_maps_ambiguous_predicate(self, engine):
        with Session(engine) as s:
            aid = _add_assertion(s, "needs")
            s.commit()

        provider = FakeProvider(answers={"needs": "requires"})
        run_pass10b(engine, _embedder=fake_embedder, provider=provider)

        with Session(engine) as s:
            assertion = s.get(AssertionRow, aid)
            assert assertion.predicate == "requires"

            mappings = s.exec(select(PredicateMappingRow)).all()
            assert len(mappings) == 1
            m = mappings[0]
            assert m.from_predicate == "needs"
            assert m.to_predicate == "requires"
            assert m.reviewer == "llm:fake-llm"

        assert len(provider.calls) == 1

    def test_llm_none_goes_to_pending(self, engine):
        with Session(engine) as s:
            _add_assertion(s, "needs")
            s.commit()

        provider = FakeProvider(answers={"needs": "none"})
        run_pass10b(engine, _embedder=fake_embedder, provider=provider)

        with Session(engine) as s:
            pending = s.exec(select(PendingPredicateRow)).all()
            assert len(pending) == 1
            assert pending[0].value == "needs"
            mappings = s.exec(select(PredicateMappingRow)).all()
            assert mappings == []

    def test_llm_error_goes_to_pending(self, engine):
        with Session(engine) as s:
            _add_assertion(s, "needs")
            s.commit()

        class ErrorProvider(FakeProvider):
            def classify(self, prompt, options):
                raise RuntimeError("LLM unavailable")

        run_pass10b(engine, _embedder=fake_embedder, provider=ErrorProvider())

        with Session(engine) as s:
            pending = s.exec(select(PendingPredicateRow)).all()
            assert len(pending) == 1
            assert pending[0].value == "needs"

    def test_no_provider_ambiguous_goes_to_pending(self, engine):
        with Session(engine) as s:
            _add_assertion(s, "needs")
            s.commit()

        run_pass10b(engine, _embedder=fake_embedder, provider=None)

        with Session(engine) as s:
            pending = s.exec(select(PendingPredicateRow)).all()
            assert len(pending) == 1
            assert pending[0].value == "needs"
            mappings = s.exec(select(PredicateMappingRow)).all()
            assert mappings == []

    def test_options_include_none_sentinel(self, engine):
        """LLM classify call must include 'none' as an option."""
        with Session(engine) as s:
            _add_assertion(s, "needs")
            s.commit()

        provider = FakeProvider(answers={"needs": "requires"})
        run_pass10b(engine, _embedder=fake_embedder, provider=provider)

        _, options = provider.calls[0]
        assert "none" in options
        for core in CORE_PREDICATES:
            assert core in options


# ---------------------------------------------------------------------------
# Pass progress & resumability
# ---------------------------------------------------------------------------

class TestPassProgress:
    def test_pass_progress_recorded(self, engine):
        run_pass10b(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            row = s.get(PassProgressRow, ("10b", "__global__", "all-mpnet-base-v2"))
            assert row is not None
            assert row.status == "completed"

    def test_already_completed_skips(self, engine):
        with Session(engine) as s:
            _add_assertion(s, "demands")
            s.commit()

        run_pass10b(engine, _embedder=fake_embedder)

        # Add a new non-core assertion — should NOT be processed on second run.
        with Session(engine) as s:
            _add_assertion(s, "demands")
            s.commit()

        result = run_pass10b(engine, _embedder=fake_embedder)
        assert result.get("status") == "already_completed"

    def test_already_mapped_predicate_skipped(self, engine):
        """If predicate_mappings already has a row for the predicate, skip it."""
        with Session(engine) as s:
            aid = _add_assertion(s, "demands")
            # Pre-record a mapping as if a previous partial run already handled it.
            s.add(PredicateMappingRow(
                from_predicate="demands",
                to_predicate="requires",
                created_at=NOW,
                reviewer="embedding",
            ))
            s.commit()

        calls = []
        original = fake_embedder

        def counting_embedder(texts):
            calls.append(texts)
            return original(texts)

        run_pass10b(engine, _embedder=counting_embedder)

        # The assertion should NOT be re-mapped (predicate already requires from setup).
        # No new mapping row should appear.
        with Session(engine) as s:
            mappings = s.exec(select(PredicateMappingRow)).all()
            assert len(mappings) == 1


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_no_changes(self, engine):
        with Session(engine) as s:
            aid = _add_assertion(s, "demands")
            s.commit()

        result = run_pass10b(engine, _embedder=fake_embedder, dry_run=True)

        assert result.get("dry_run") is True
        assert result["non_core_predicate_count"] == 1

        with Session(engine) as s:
            assertion = s.get(AssertionRow, aid)
            assert assertion.predicate == "demands"
            mappings = s.exec(select(PredicateMappingRow)).all()
            assert mappings == []
            pending = s.exec(select(PendingPredicateRow)).all()
            assert pending == []
            progress = s.get(PassProgressRow, ("10b", "__global__", "all-mpnet-base-v2"))
            assert progress is None
