"""Section 16.3 — Topology reasoning evaluation test.

Planted-violation harness: builds a realistic multi-space building graph,
introduces deliberate accessibility violations, runs bsos validate topology,
and asserts 100% recall (all planted violations detected).

Gate: set BSOS_E2E=1 to run.
"""
import os
import pytest
from datetime import datetime, timezone
from typer.testing import CliRunner
from sqlmodel import Session

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, SpatialRelationRow

pytestmark = pytest.mark.skipif(
    not os.environ.get("BSOS_E2E"),
    reason="Set BSOS_E2E=1 to run end-to-end evaluation tests",
)

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_db(tmp_path):
    db = tmp_path / "eval.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _engine(db_path):
    return create_db_engine(db_path)


def _space(eid, name, *, is_entrance=False):
    return EntityRow(
        id=eid,
        name=name,
        entity_type="space",
        status="accepted",
        is_entrance=is_entrance,
        source_model="eval_harness",
        created_at=NOW,
    )


def _edge(eid, subject_id, object_id):
    """accessible_from(subject, object) — subject is accessible from object."""
    return SpatialRelationRow(
        id=eid,
        subject_id=subject_id,
        relation="accessible_from",
        object_id=object_id,
        confidence=1.0,
        knowledge_origin="architectural",
        source_model="eval_harness",
        created_at=NOW,
    )


def _run_topology(db):
    return runner.invoke(app, ["validate", "topology", "--db", db])


def _detected_violations(result_output, violation_names):
    """Return which planted violation names appear in the command output."""
    return {name for name in violation_names if name in result_output}


# ---------------------------------------------------------------------------
# Scenario 1: Isolated island — spaces with zero edges
# ---------------------------------------------------------------------------

def test_eval_isolated_spaces(tmp_path):
    """All spaces with no accessible_from edges are detected."""
    db = _init_db(tmp_path)
    eng = _engine(db)

    planted = {"Storage Room", "Plant Room"}

    with Session(eng) as s:
        s.add(_space("e-entrance", "Main Entrance", is_entrance=True))
        s.add(_space("e-lobby", "Lobby"))
        s.add(_space("e-storage", "Storage Room"))   # violation: no edge
        s.add(_space("e-plant", "Plant Room"))        # violation: no edge
        s.add(_edge("r1", "e-lobby", "e-entrance"))
        s.commit()

    result = _run_topology(db)
    assert result.exit_code != 0, "Validator should exit non-zero on violations"

    detected = _detected_violations(result.output, planted)
    missed = planted - detected
    assert not missed, f"Missed violations: {missed}\nOutput:\n{result.output}"


# ---------------------------------------------------------------------------
# Scenario 2: Severed bridge — spaces cut off when a link is absent
# ---------------------------------------------------------------------------

def test_eval_severed_bridge(tmp_path):
    """Spaces downstream of a missing link are all detected."""
    db = _init_db(tmp_path)
    eng = _engine(db)

    # Layout: entrance → lobby → corridor → office_wing → office_a, office_b
    # Violation: corridor → office_wing edge is intentionally absent,
    # making office_wing, office_a, office_b unreachable.
    planted = {"Office Wing", "Office A", "Office B"}

    with Session(eng) as s:
        s.add(_space("e-entrance", "Main Entrance", is_entrance=True))
        s.add(_space("e-lobby", "Lobby"))
        s.add(_space("e-corridor", "Corridor"))
        s.add(_space("e-wing", "Office Wing"))   # violation: no edge from corridor
        s.add(_space("e-oa", "Office A"))
        s.add(_space("e-ob", "Office B"))

        s.add(_edge("r1", "e-lobby", "e-entrance"))
        s.add(_edge("r2", "e-corridor", "e-lobby"))
        # deliberately no edge: e-wing accessible_from e-corridor
        s.add(_edge("r3", "e-oa", "e-wing"))
        s.add(_edge("r4", "e-ob", "e-wing"))
        s.commit()

    result = _run_topology(db)
    assert result.exit_code != 0

    detected = _detected_violations(result.output, planted)
    missed = planted - detected
    assert not missed, f"Missed violations: {missed}\nOutput:\n{result.output}"


# ---------------------------------------------------------------------------
# Scenario 3: Mixed — reachable spaces alongside violations; no false positives
# ---------------------------------------------------------------------------

def test_eval_no_false_positives(tmp_path):
    """Correctly reachable spaces are NOT reported as violations."""
    db = _init_db(tmp_path)
    eng = _engine(db)

    reachable = {"Reception", "Open Plan", "Meeting Room"}

    with Session(eng) as s:
        s.add(_space("e-entrance", "Main Entrance", is_entrance=True))
        s.add(_space("e-reception", "Reception"))
        s.add(_space("e-open", "Open Plan"))
        s.add(_space("e-meeting", "Meeting Room"))

        s.add(_edge("r1", "e-reception", "e-entrance"))
        s.add(_edge("r2", "e-open", "e-reception"))
        s.add(_edge("r3", "e-meeting", "e-open"))
        s.commit()

    result = _run_topology(db)
    assert result.exit_code == 0, f"Should pass — no violations\nOutput:\n{result.output}"

    false_positives = _detected_violations(result.output, reachable)
    assert not false_positives, f"False positives reported: {false_positives}"


# ---------------------------------------------------------------------------
# Scenario 4: Multi-entrance building with violations in one wing
# ---------------------------------------------------------------------------

def test_eval_multi_entrance_partial_violation(tmp_path):
    """With two entrances, only the genuinely unreachable space is flagged."""
    db = _init_db(tmp_path)
    eng = _engine(db)

    planted = {"Basement Store"}
    reachable_ok = {"North Lobby", "South Lobby", "Stairwell"}

    with Session(eng) as s:
        s.add(_space("e-north", "North Entrance", is_entrance=True))
        s.add(_space("e-south", "South Entrance", is_entrance=True))
        s.add(_space("e-nlobby", "North Lobby"))
        s.add(_space("e-slobby", "South Lobby"))
        s.add(_space("e-stair", "Stairwell"))
        s.add(_space("e-basement", "Basement Store"))   # violation

        s.add(_edge("r1", "e-nlobby", "e-north"))
        s.add(_edge("r2", "e-slobby", "e-south"))
        s.add(_edge("r3", "e-stair", "e-nlobby"))
        s.add(_edge("r4", "e-stair", "e-slobby"))  # reachable from both
        # no edge to e-basement
        s.commit()

    result = _run_topology(db)
    assert result.exit_code != 0

    detected = _detected_violations(result.output, planted)
    missed = planted - detected
    assert not missed, f"Missed violations: {missed}\nOutput:\n{result.output}"

    false_positives = _detected_violations(result.output, reachable_ok)
    assert not false_positives, f"False positives: {false_positives}"


# ---------------------------------------------------------------------------
# Scenario 5: Full building — high-volume planted violation recall
# ---------------------------------------------------------------------------

def test_eval_high_volume_recall(tmp_path):
    """All violations in a 20-space building are detected (recall = 100%)."""
    db = _init_db(tmp_path)
    eng = _engine(db)

    with Session(eng) as s:
        s.add(_space("e-ent", "Main Entrance", is_entrance=True))

        # Reachable spine: entrance → atrium → corridor_a → corridor_b
        s.add(_space("e-atrium", "Atrium"))
        s.add(_space("e-ca", "Corridor A"))
        s.add(_space("e-cb", "Corridor B"))
        s.add(_edge("r-ea", "e-atrium", "e-ent"))
        s.add(_edge("r-ac", "e-ca", "e-atrium"))
        s.add(_edge("r-cb", "e-cb", "e-ca"))

        # 10 reachable rooms off corridor_a / corridor_b
        reachable_ids = set()
        for i in range(1, 6):
            rid = f"e-ra{i}"
            rname = f"Room A{i}"
            s.add(_space(rid, rname))
            s.add(_edge(f"r-ra{i}", rid, "e-ca"))
            reachable_ids.add(rid)

        for i in range(1, 6):
            rid = f"e-rb{i}"
            rname = f"Room B{i}"
            s.add(_space(rid, rname))
            s.add(_edge(f"r-rb{i}", rid, "e-cb"))
            reachable_ids.add(rid)

        # 5 planted violation spaces — no edges, no path from entrance
        planted_names = set()
        for i in range(1, 6):
            vid = f"e-v{i}"
            vname = f"Isolated Zone {i}"
            s.add(_space(vid, vname))
            planted_names.add(vname)

        s.commit()

    result = _run_topology(db)
    assert result.exit_code != 0

    detected = _detected_violations(result.output, planted_names)
    missed = planted_names - detected
    recall = len(detected) / len(planted_names)

    assert recall == 1.0, (
        f"Recall {recall:.0%} — missed violations: {missed}\nOutput:\n{result.output}"
    )
