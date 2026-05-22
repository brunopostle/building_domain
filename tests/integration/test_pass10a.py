"""Integration tests for Pass 10a — ref resolution."""
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.config import get_config
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    EntityRow,
    ForceRow,
    PassProgressRow,
    PatternRow,
    PendingEntityRefRow,
    PendingForceRefRow,
    PendingPatternRefRow,
)
from bsos.normalization.pass10a import run_pass10a

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fake embedder: returns deterministic unit vectors keyed by text content
# ---------------------------------------------------------------------------

DIM = 8


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Similar pair (cosine > 0.85): "increased daylight" ≈ "increased daylighting"
VEC_DAYLIGHT = _unit([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_DAYLIGHTING = _unit([0.98, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Dissimilar: "reduced heat gain" — orthogonal to the daylight group
VEC_HEAT = _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Pattern names
VEC_LIGHT_TWO_SIDES = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_LIGHT_TWO_SIDES_VARIANT = _unit([0.0, 0.97, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
VEC_SOUTH_COURTYARD = _unit([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

VECTOR_MAP = {
    "increased daylighting": VEC_DAYLIGHT,
    "increased daylight": VEC_DAYLIGHT,
    "increased daylighting force": VEC_DAYLIGHTING,
    "reduced heat gain": VEC_HEAT,
    "light on two sides": VEC_LIGHT_TWO_SIDES,
    "light on two sides of room": VEC_LIGHT_TWO_SIDES_VARIANT,
    "south-facing courtyard": VEC_SOUTH_COURTYARD,
}


def fake_embedder(texts: list[str]) -> np.ndarray:
    default = np.zeros(DIM, dtype=np.float32)
    default[7] = 1.0
    return np.array(
        [VECTOR_MAP.get(t.lower().strip(), default) for t in texts],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Fixtures
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


def _add_force(session, name: str, direction: str = "increase", affects: list[str] | None = None) -> str:
    fid = str(uuid.uuid4())
    session.add(ForceRow(
        id=fid, name=name, direction=direction,
        affects=json.dumps(affects or []),
        source_model="test", created_at=NOW,
        confidence=0.9, status="proposed", knowledge_origin="physical",
    ))
    return fid


def _add_pattern(
    session,
    name: str,
    force_descriptions: list[str] | None = None,
    related_pattern_names: list[str] | None = None,
) -> str:
    pid = str(uuid.uuid4())
    session.add(PatternRow(
        id=pid, name=name, problem="problem", solution="solution",
        force_descriptions=json.dumps(force_descriptions or []),
        force_ids=json.dumps([]),
        related_pattern_names=json.dumps(related_pattern_names or []),
        related_pattern_ids=json.dumps([]),
        context=json.dumps([]),
        consequences=json.dumps([]),
        emergent_properties=json.dumps([]),
        source_model="test", created_at=NOW,
        confidence=0.9, status="proposed", knowledge_origin="architectural",
    ))
    return pid


# ---------------------------------------------------------------------------
# Tests: force_descriptions resolution
# ---------------------------------------------------------------------------

class TestForceDescriptionResolution:
    def test_exact_match_resolved(self, engine):
        with Session(engine) as s:
            force_id = _add_force(s, "increased daylighting")
            pattern_id = _add_pattern(s, "p1", force_descriptions=["increased daylighting"])
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            pattern = s.get(PatternRow, pattern_id)
            assert json.loads(pattern.force_descriptions) == []
            assert force_id in json.loads(pattern.force_ids)

        assert result["force_descriptions_resolved"] == 1
        assert result["force_descriptions_unresolved"] == 0

    def test_similarity_match_resolved(self, engine):
        """Description similar but not identical to force name → resolved via embedding."""
        with Session(engine) as s:
            force_id = _add_force(s, "increased daylighting")
            # "increased daylighting force" has VEC_DAYLIGHTING which is close to VEC_DAYLIGHT
            pattern_id = _add_pattern(s, "p1", force_descriptions=["increased daylighting force"])
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            pattern = s.get(PatternRow, pattern_id)
            assert json.loads(pattern.force_descriptions) == []
            assert force_id in json.loads(pattern.force_ids)

        assert result["force_descriptions_resolved"] == 1

    def test_unresolved_written_to_pending_force_refs(self, engine):
        """Description with no similar force → written to pending_force_refs."""
        with Session(engine) as s:
            _add_force(s, "increased daylighting")  # orthogonal to "reduced heat gain"
            pattern_id = _add_pattern(s, "p1", force_descriptions=["reduced heat gain"])
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            pattern = s.get(PatternRow, pattern_id)
            assert json.loads(pattern.force_descriptions) == []  # always cleared
            assert json.loads(pattern.force_ids) == []           # no match
            pending = s.exec(select(PendingForceRefRow)).all()
            assert len(pending) == 1
            assert pending[0].description == "reduced heat gain"
            assert pending[0].pattern_id == pattern_id
            assert pending[0].failure_type == "unresolved_ref"

        assert result["force_descriptions_unresolved"] == 1

    def test_already_empty_descriptions_skipped(self, engine):
        """Pattern with force_descriptions=[] is a resumable skip."""
        with Session(engine) as s:
            force_id = _add_force(s, "increased daylighting")
            # Pattern already resolved (force_descriptions cleared, force_ids populated)
            pid = str(uuid.uuid4())
            s.add(PatternRow(
                id=pid, name="p1", problem="p", solution="s",
                force_descriptions=json.dumps([]),
                force_ids=json.dumps([force_id]),
                related_pattern_names=json.dumps([]),
                related_pattern_ids=json.dumps([]),
                context=json.dumps([]),
                consequences=json.dumps([]),
                emergent_properties=json.dumps([]),
                source_model="test", created_at=NOW,
                confidence=0.9, status="proposed", knowledge_origin="architectural",
            ))
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        # force_ids should be unchanged
        with Session(engine) as s:
            pattern = s.get(PatternRow, pid)
            assert force_id in json.loads(pattern.force_ids)

        assert result["force_descriptions_resolved"] == 0


# ---------------------------------------------------------------------------
# Tests: related_pattern_names resolution
# ---------------------------------------------------------------------------

class TestRelatedPatternNamesResolution:
    def test_exact_match_resolved(self, engine):
        with Session(engine) as s:
            pid_target = _add_pattern(s, "light on two sides")
            pid_source = _add_pattern(
                s, "south courtyard",
                related_pattern_names=["light on two sides"],
            )
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            p = s.get(PatternRow, pid_source)
            assert json.loads(p.related_pattern_names) == []
            assert pid_target in json.loads(p.related_pattern_ids)

        assert result["pattern_names_resolved"] == 1

    def test_similarity_match_resolved(self, engine):
        """Pattern name close but not identical → resolved via embedding."""
        with Session(engine) as s:
            pid_target = _add_pattern(s, "light on two sides")
            pid_source = _add_pattern(
                s, "courtyard",
                related_pattern_names=["light on two sides of room"],
            )
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            p = s.get(PatternRow, pid_source)
            assert json.loads(p.related_pattern_names) == []
            assert pid_target in json.loads(p.related_pattern_ids)

    def test_unresolved_written_to_pending_pattern_refs(self, engine):
        with Session(engine) as s:
            _add_pattern(s, "light on two sides")
            pid_source = _add_pattern(
                s, "courtyard",
                related_pattern_names=["south-facing courtyard"],  # orthogonal vector
            )
            s.commit()

        run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            p = s.get(PatternRow, pid_source)
            assert json.loads(p.related_pattern_names) == []
            refs = s.exec(select(PendingPatternRefRow)).all()
            assert len(refs) == 1
            assert refs[0].pattern_name == "south-facing courtyard"
            assert refs[0].source_pattern_id == pid_source

    def test_self_not_matched(self, engine):
        """A pattern's own name should not resolve to itself."""
        with Session(engine) as s:
            pid = _add_pattern(
                s, "light on two sides",
                related_pattern_names=["light on two sides"],  # self-reference
            )
            s.commit()

        run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            p = s.get(PatternRow, pid)
            # Self should be excluded from candidates → unresolved
            assert json.loads(p.related_pattern_names) == []
            related = json.loads(p.related_pattern_ids)
            assert pid not in related


# ---------------------------------------------------------------------------
# Tests: entity ref resolution
# ---------------------------------------------------------------------------

class TestEntityRefResolution:
    def test_pending_ref_resolved_when_entity_exists(self, engine):
        with Session(engine) as s:
            eid = _add_entity(s, "roof")
            fid = _add_force(s, "increased daylighting")
            s.add(PendingEntityRefRow(
                entity_name="roof",
                source_force_id=fid,
                created_at=NOW,
            ))
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            force = s.get(ForceRow, fid)
            assert eid in json.loads(force.affects)
            remaining = s.exec(select(PendingEntityRefRow)).all()
            assert len(remaining) == 0

        assert result["entity_refs_resolved"] == 1
        assert result["entity_refs_remaining"] == 0

    def test_unresolvable_ref_stays_in_table(self, engine):
        with Session(engine) as s:
            fid = _add_force(s, "increased daylighting")
            s.add(PendingEntityRefRow(
                entity_name="unknown_entity_xyz",
                source_force_id=fid,
                created_at=NOW,
            ))
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            remaining = s.exec(select(PendingEntityRefRow)).all()
            assert len(remaining) == 1

        assert result["entity_refs_remaining"] == 1

    def test_no_duplicate_uuid_added_to_affects(self, engine):
        """Resolving the same entity twice should not produce duplicates."""
        with Session(engine) as s:
            eid = _add_entity(s, "roof")
            fid = _add_force(s, "increased daylighting", affects=[eid])
            s.add(PendingEntityRefRow(entity_name="roof", source_force_id=fid, created_at=NOW))
            s.commit()

        run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            force = s.get(ForceRow, fid)
            affects = json.loads(force.affects)
            assert affects.count(eid) == 1


# ---------------------------------------------------------------------------
# Tests: pass progress and config
# ---------------------------------------------------------------------------

class TestPassProgressAndConfig:
    def test_pass_progress_recorded(self, engine):
        run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            row = s.get(PassProgressRow, ("10a", "__global__", "all-mpnet-base-v2"))
            assert row is not None
            assert row.status == "completed"

    def test_config_flag_set(self, engine):
        run_pass10a(engine, _embedder=fake_embedder)

        with Session(engine) as s:
            val = get_config(s, "passes_3_9_refs_resolved")
            assert val == "1"

    def test_already_completed_is_skipped(self, engine):
        """Running pass10a twice: second run returns early without re-processing."""
        with Session(engine) as s:
            _add_force(s, "increased daylighting")
            _add_pattern(s, "p1", force_descriptions=["increased daylighting"])
            s.commit()

        run_pass10a(engine, _embedder=fake_embedder)

        # Add a new unresolved description — should NOT be processed on second run
        with Session(engine) as s:
            _add_pattern(s, "p2", force_descriptions=["increased daylighting"])
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder)
        assert result.get("status") == "already_completed"

    def test_dry_run_makes_no_changes(self, engine):
        with Session(engine) as s:
            _add_force(s, "increased daylighting")
            pid = _add_pattern(s, "p1", force_descriptions=["increased daylighting"])
            s.commit()

        result = run_pass10a(engine, _embedder=fake_embedder, dry_run=True)

        assert result.get("dry_run") is True
        assert result["patterns_with_force_descriptions"] == 1

        with Session(engine) as s:
            pattern = s.get(PatternRow, pid)
            # force_descriptions must be unchanged
            assert json.loads(pattern.force_descriptions) == ["increased daylighting"]
