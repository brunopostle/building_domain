"""Integration tests for Pass 10c — abstraction synthesis."""
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.normalization.pass10c import (
    ABSTRACTION_QUEUE_CAP,
    MIN_CLUSTER_SIZE,
    run_pass10c,
)
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import AbstractionNodeRow, AssertionRow, EntityRow, PassProgressRow
from bsos.persistence.repos.abstraction import AbstractionNodeRepository

NOW = datetime.now(timezone.utc)
DIM = 8

# ---------------------------------------------------------------------------
# Fake embedder — deterministic unit vectors for cluster control
# ---------------------------------------------------------------------------
#
# Cluster A: 3 assertions with nearly identical vectors → within threshold → one cluster
# Cluster B: 3 assertions orthogonal to A and to each other → separate cluster per pair
# Singleton: 1 assertion far from everything → won't form a qualifying cluster
#
# We use a label-based dispatch so tests can set assertion vectors by ID.

_VECTOR_REGISTRY: dict[str, np.ndarray] = {}


def _register(key: str, vec: list[float]) -> np.ndarray:
    a = np.array(vec, dtype=np.float32)
    a = a / float(np.linalg.norm(a))
    _VECTOR_REGISTRY[key] = a
    return a


# Three vectors tightly clustered (cos_sim ≈ 0.99)
VEC_A1 = _register("A1", [1.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_A2 = _register("A2", [0.99, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_A3 = _register("A3", [0.98, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Orthogonal vectors — each will be its own singleton "cluster"
VEC_B1 = _register("B1", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_B2 = _register("B2", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _default_vec(seed: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(seed)) % (2**31))
    v = rng.random(DIM).astype(np.float32) + 0.001
    return v / float(np.linalg.norm(v))


# Map from assertion "rationale" tag → vector so tests can control clustering.
_RATIONALE_TO_VEC: dict[str, np.ndarray] = {
    "cluster:A1": VEC_A1,
    "cluster:A2": VEC_A2,
    "cluster:A3": VEC_A3,
    "cluster:B1": VEC_B1,
    "cluster:B2": VEC_B2,
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    result = []
    for t in texts:
        for key, vec in _RATIONALE_TO_VEC.items():
            if key in t:
                result.append(vec)
                break
        else:
            result.append(_default_vec(t))
    return np.array(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------

class FakeProviderA:
    """Always returns a successful synthesis."""
    model_id = "model-a"
    calls: list[str] = []

    def __init__(self):
        self.calls = []

    def extract(self, prompt: str, schema):
        self.calls.append(prompt)
        return schema(
            statement="Roofs protect from weather",
            abstraction_rationale="All source assertions describe protective functions",
            confidence=0.85,
        )

    def classify(self, prompt, options):
        raise NotImplementedError


class FakeProviderB:
    """Adversarial validator — answer is configurable."""
    model_id = "model-b"

    def __init__(self, verdict: str = "no"):
        self.verdict = verdict
        self.calls: list[str] = []

    def classify(self, prompt: str, options: list[str]) -> str:
        self.calls.append(prompt)
        return self.verdict

    def extract(self, prompt, schema, *, entity_name=None):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def _add_entity(session, name: str) -> str:
    eid = str(uuid.uuid4())
    session.add(EntityRow(
        id=eid, name=name, entity_type="component", status="accepted",
        source_model="test", created_at=NOW,
    ))
    return eid


def _add_assertion(
    session,
    subject_id: str,
    object_id: str,
    predicate: str = "requires",
    rationale: str = "",
    status: str = "proposed",
) -> str:
    aid = str(uuid.uuid4())
    session.add(AssertionRow(
        id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id,
        subject_type="component", object_type="system",
        source_model="test", created_at=NOW,
        confidence=0.9, status=status, knowledge_origin="physical",
        rationale=rationale,
    ))
    return aid


def _make_cluster(session, n: int, cluster_tag: str, extra_entity: bool = True) -> tuple[str, list[str]]:
    """Add n assertions with the same subject and the given cluster tag in rationale.
    Returns (subject_id, [assertion_ids]).
    """
    subj = _add_entity(session, f"subj-{uuid.uuid4()}")
    aids = []
    for i in range(n):
        obj = _add_entity(session, f"obj-{uuid.uuid4()}")
        aid = _add_assertion(session, subj, obj, rationale=f"cluster:{cluster_tag}{i+1}")
        aids.append(aid)
    return subj, aids


# ---------------------------------------------------------------------------
# Basic cluster synthesis
# ---------------------------------------------------------------------------

class TestClusterSynthesis:
    def test_qualifying_cluster_creates_node(self, engine):
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            obj1 = _add_entity(s, "rain")
            obj2 = _add_entity(s, "wind")
            obj3 = _add_entity(s, "snow")
            aid1 = _add_assertion(s, subj, obj1, rationale="cluster:A1")
            aid2 = _add_assertion(s, subj, obj2, rationale="cluster:A2")
            aid3 = _add_assertion(s, subj, obj3, rationale="cluster:A3")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        assert result["nodes_created"] == 1
        assert result["clusters_processed"] >= 1

        with Session(engine) as s:
            nodes = s.exec(select(AbstractionNodeRow)).all()
            assert len(nodes) == 1
            node = nodes[0]
            assert node.status == "proposed"
            child_ids = json.loads(node.child_ids)
            assert set(child_ids) == {aid1, aid2, aid3}

    def test_cluster_below_min_size_skipped(self, engine):
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            obj1 = _add_entity(s, "rain")
            obj2 = _add_entity(s, "wind")
            aid1 = _add_assertion(s, subj, obj1, rationale="cluster:A1")
            aid2 = _add_assertion(s, subj, obj2, rationale="cluster:A2")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        assert result["nodes_created"] == 0

        with Session(engine) as s:
            nodes = s.exec(select(AbstractionNodeRow)).all()
            assert len(nodes) == 0

    def test_orthogonal_vectors_not_clustered(self, engine):
        """Orthogonal vectors produce singleton clusters below min_cluster_size."""
        with Session(engine) as s:
            subj = _add_entity(s, "wall")
            obj1 = _add_entity(s, "e1")
            obj2 = _add_entity(s, "e2")
            obj3 = _add_entity(s, "e3")
            # B1, B2 are orthogonal; a third orthogonal vector from _default_vec
            _add_assertion(s, subj, obj1, rationale="cluster:B1")
            _add_assertion(s, subj, obj2, rationale="cluster:B2")
            _add_assertion(s, subj, obj3, rationale="cluster:unique-xyz")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        assert result["nodes_created"] == 0

    def test_only_accepted_and_proposed_assertions_processed(self, engine):
        """Deprecated assertions should not be included."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            obj1 = _add_entity(s, "rain")
            obj2 = _add_entity(s, "wind")
            obj3 = _add_entity(s, "snow")
            _add_assertion(s, subj, obj1, rationale="cluster:A1", status="accepted")
            _add_assertion(s, subj, obj2, rationale="cluster:A2", status="proposed")
            _add_assertion(s, subj, obj3, rationale="cluster:A3", status="deprecated")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        # Only 2 non-deprecated, below min_cluster_size=3
        assert result["nodes_created"] == 0


# ---------------------------------------------------------------------------
# Adversarial validation
# ---------------------------------------------------------------------------

class TestAdversarialValidation:
    def test_llm_b_no_creates_node(self, engine):
        """LLM-B says 'no' (no new info) → node is created."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        pa = FakeProviderA()
        pb = FakeProviderB(verdict="no")
        result = run_pass10c(engine, pa, pb, _embedder=fake_embedder)

        assert result["nodes_created"] == 1
        assert len(pb.calls) == 1
        _, options = pa.calls, pb.calls[0]
        assert "yes" in pb.calls[0]

    def test_llm_b_yes_discards_node(self, engine):
        """LLM-B says 'yes' (introduces new info) → node discarded."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        pa = FakeProviderA()
        pb = FakeProviderB(verdict="yes")
        result = run_pass10c(engine, pa, pb, _embedder=fake_embedder)

        assert result["nodes_created"] == 0

        with Session(engine) as s:
            nodes = s.exec(select(AbstractionNodeRow)).all()
            assert nodes == []

    def test_same_model_skips_validation(self, engine):
        """If provider_b has same model_id as provider_a, validation is skipped."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        pa = FakeProviderA()
        # Same model_id as pa
        pb = FakeProviderB(verdict="yes")
        pb.model_id = pa.model_id

        result = run_pass10c(engine, pa, pb, _embedder=fake_embedder)

        # Validation skipped → node created despite pb.verdict="yes"
        assert result["nodes_created"] == 1
        assert pb.calls == []

    def test_no_provider_b_skips_validation(self, engine):
        """No provider_b → no validation, node always created."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), provider_b=None, _embedder=fake_embedder)
        assert result["nodes_created"] == 1

    def test_llm_b_error_still_creates_node(self, engine):
        """If adversarial validate raises, fall through and create node."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        class ErrorProviderB(FakeProviderB):
            def classify(self, prompt, options):
                raise RuntimeError("unavailable")

        result = run_pass10c(engine, FakeProviderA(), ErrorProviderB(), _embedder=fake_embedder)
        assert result["nodes_created"] == 1


# ---------------------------------------------------------------------------
# Queue cap
# ---------------------------------------------------------------------------

class TestQueueCap:
    def test_stops_when_cap_reached(self, engine):
        """If proposed nodes ≥ ABSTRACTION_QUEUE_CAP, stop and return cap_reached=True."""
        # Pre-fill the table with CAP proposed nodes.
        with Session(engine) as s:
            for i in range(ABSTRACTION_QUEUE_CAP):
                s.add(AbstractionNodeRow(
                    id=str(uuid.uuid4()),
                    statement=f"stmt {i}",
                    child_ids=json.dumps([]),
                    abstraction_rationale="prefilled",
                    source_model="pre",
                    created_at=NOW,
                    confidence=0.8,
                    status="proposed",
                ))
            # Add a qualifying cluster.
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        assert result["cap_reached"] is True
        assert result["nodes_created"] == 0

    def test_cap_not_reached_when_below(self, engine):
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)
        assert result["cap_reached"] is False


# ---------------------------------------------------------------------------
# Pass progress & resumability
# ---------------------------------------------------------------------------

class TestPassProgress:
    def test_pass_progress_recorded(self, engine):
        run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        with Session(engine) as s:
            row = s.get(PassProgressRow, ("10c", "__global__", "all-mpnet-base-v2"))
            assert row is not None
            assert row.status == "completed"

    def test_already_completed_skips(self, engine):
        run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)

        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder)
        assert result.get("status") == "already_completed"

    def test_llm_a_error_no_node_created(self, engine):
        """Synthesis LLM error → cluster skipped, no node."""
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        class ErrorProviderA(FakeProviderA):
            def extract(self, prompt, schema):
                raise RuntimeError("LLM down")

        result = run_pass10c(engine, ErrorProviderA(), _embedder=fake_embedder)
        assert result["nodes_created"] == 0

        with Session(engine) as s:
            nodes = s.exec(select(AbstractionNodeRow)).all()
            assert nodes == []


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_no_changes(self, engine):
        with Session(engine) as s:
            subj = _add_entity(s, "roof")
            for i in range(3):
                obj = _add_entity(s, f"e{i}")
                _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder, dry_run=True)

        assert result.get("dry_run") is True
        assert result["eligible_subject_groups"] == 1

        with Session(engine) as s:
            nodes = s.exec(select(AbstractionNodeRow)).all()
            assert nodes == []
            progress = s.get(PassProgressRow, ("10c", "__global__", "all-mpnet-base-v2"))
            assert progress is None

    def test_dry_run_counts_eligible_groups(self, engine):
        with Session(engine) as s:
            # Two eligible groups (3 assertions each)
            for _ in range(2):
                subj = _add_entity(s, f"subj-{uuid.uuid4()}")
                for i in range(3):
                    obj = _add_entity(s, f"e{i}-{uuid.uuid4()}")
                    _add_assertion(s, subj, obj, rationale=f"cluster:A{i+1}")
            # One ineligible group (2 assertions)
            subj2 = _add_entity(s, "small-subj")
            for i in range(2):
                obj = _add_entity(s, f"small-e{i}")
                _add_assertion(s, subj2, obj, rationale=f"cluster:B{i+1}")
            s.commit()

        result = run_pass10c(engine, FakeProviderA(), _embedder=fake_embedder, dry_run=True)
        assert result["eligible_subject_groups"] == 2
