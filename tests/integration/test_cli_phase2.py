"""Integration tests for Phase 2 CLI extensions."""
import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner
from sqlmodel import Session, select

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AntiPatternRow, ConstraintRow, EntityRow, ForceRow,
    PatternRow, PendingPredicateRow, PendingSpatialRelationTypeRow,
    ProcessRelationRow, SpatialRelationRow,
)

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)


def _init_db(tmp_path):
    db = tmp_path / "t.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _engine(db_path):
    return create_db_engine(db_path)


def _add_entity(engine, eid, name, entity_type="component", status="proposed", is_entrance=False):
    with Session(engine) as s:
        s.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                        status=status, source_model="test", created_at=NOW,
                        is_entrance=is_entrance))
        s.commit()


# ---------------------------------------------------------------------------
# bsos query --type constraint
# ---------------------------------------------------------------------------

def test_query_constraint_text(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    with Session(eng) as s:
        s.add(ConstraintRow(id="c1", subject_id="e-roof", rule="Roof must have drainage",
                            constraint_type="must", confidence=0.9,
                            knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["query", "Roof", "--type", "constraint", "--db", db,
                                 "--include-proposed"])
    assert result.exit_code == 0, result.output
    assert "Roof must have drainage" in result.output
    assert "must" in result.output.lower()


def test_query_constraint_json(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    with Session(eng) as s:
        s.add(ConstraintRow(id="c1", subject_id="e-roof", rule="Drainage required",
                            constraint_type="must", confidence=0.85,
                            knowledge_origin="physical", source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["query", "Roof", "--type", "constraint", "--json", "--db", db,
                                 "--include-proposed"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["entity"] == "Roof"
    assert len(data["constraints"]) == 1
    assert data["constraints"][0]["rule"] == "Drainage required"


def test_query_constraint_entity_not_found(tmp_path):
    db = _init_db(tmp_path)
    result = runner.invoke(app, ["query", "NoSuch", "--type", "constraint", "--db", db])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# bsos query --type antipattern
# ---------------------------------------------------------------------------

def test_query_antipattern_text(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-wall", "Wall")
    with Session(eng) as s:
        s.add(AntiPatternRow(id="ap1", subject_id="e-wall", name="Thermal bridging",
                             conditions=json.dumps(["Uninsulated junction"]),
                             consequences=json.dumps(["Heat loss"]),
                             mitigations=json.dumps(["Add insulation"]),
                             confidence=0.8, knowledge_origin="engineering",
                             source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["query", "Wall", "--type", "antipattern", "--db", db,
                                 "--include-proposed"])
    assert result.exit_code == 0, result.output
    assert "Thermal bridging" in result.output
    assert "Heat loss" in result.output


# ---------------------------------------------------------------------------
# bsos query --type force
# ---------------------------------------------------------------------------

def test_query_force_text(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-ins", "Insulation")
    with Session(eng) as s:
        s.add(ForceRow(id="f1", name="Reduced heat loss", direction="decrease",
                       affects=json.dumps(["e-ins"]), confidence=0.85,
                       knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["query", "Insulation", "--type", "force", "--db", db,
                                 "--include-proposed"])
    assert result.exit_code == 0, result.output
    assert "Reduced heat loss" in result.output
    assert "decrease" in result.output.lower()


# ---------------------------------------------------------------------------
# bsos query --type spatial
# ---------------------------------------------------------------------------

def test_query_spatial_text(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-wall", "Wall")
    _add_entity(eng, "e-floor", "Floor")
    with Session(eng) as s:
        s.add(SpatialRelationRow(id="sr1", subject_id="e-wall", relation="sits_on",
                                 object_id="e-floor", confidence=0.9,
                                 knowledge_origin="architectural",
                                 source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["query", "Wall", "--type", "spatial", "--db", db,
                                 "--include-proposed"])
    assert result.exit_code == 0, result.output
    assert "sits_on" in result.output
    assert "Floor" in result.output


# ---------------------------------------------------------------------------
# bsos query --type process
# ---------------------------------------------------------------------------

def test_query_process_text(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-frame", "Framing")
    _add_entity(eng, "e-ins", "Insulation")
    with Session(eng) as s:
        s.add(ProcessRelationRow(id="r1", predecessor_id="e-frame", successor_id="e-ins",
                                 hard_constraint=True, rationale="Frame before insulate",
                                 confidence=0.9, knowledge_origin="engineering",
                                 source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["query", "Insulation", "--type", "process", "--db", db])
    assert result.exit_code == 0, result.output
    assert "Framing" in result.output
    assert "Insulation" in result.output


# ---------------------------------------------------------------------------
# bsos status — Phase 2 rows
# ---------------------------------------------------------------------------

def test_status_shows_phase2_rows(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-wall", "Wall")
    with Session(eng) as s:
        s.add(ConstraintRow(id="c1", subject_id="e-wall", rule="Rule A",
                            constraint_type="must", confidence=0.8,
                            knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.add(ForceRow(id="f1", name="Improved strength", direction="increase",
                       affects="[]", confidence=0.7, knowledge_origin="engineering",
                       source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["status", "--db", db])
    assert result.exit_code == 0, result.output
    assert "constraints" in result.output
    assert "forces" in result.output


def test_status_shows_pending_counts(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    with Session(eng) as s:
        s.add(PendingPredicateRow(value="enables", occurrence_count=3,
                                  first_seen_at=NOW, last_seen_at=NOW))
        s.add(PendingSpatialRelationTypeRow(value="wraps_around", occurrence_count=2,
                                            first_seen_at=NOW, last_seen_at=NOW))
        s.commit()

    result = runner.invoke(app, ["status", "--db", db])
    assert result.exit_code == 0, result.output
    assert "Pending predicates awaiting review" in result.output
    assert "Pending spatial relation types awaiting review" in result.output


def test_status_json_includes_phase2(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-wall", "Wall")
    with Session(eng) as s:
        s.add(AntiPatternRow(id="ap1", subject_id="e-wall", name="Cold bridge",
                             conditions="[]", consequences="[]", mitigations="[]",
                             confidence=0.7, knowledge_origin="engineering",
                             source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["status", "--db", db, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "phase2" in data
    assert data["phase2"]["antipatterns"] == 1
    assert "pending" in data


# ---------------------------------------------------------------------------
# bsos review pending --stats
# ---------------------------------------------------------------------------

def test_review_pending_stats(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    with Session(eng) as s:
        s.add(PendingPredicateRow(value="enables", occurrence_count=10,
                                  first_seen_at=NOW, last_seen_at=NOW))
        s.commit()

    result = runner.invoke(app, ["review", "pending", "--stats", "--db", db])
    assert result.exit_code == 0, result.output
    assert "threshold" in result.output.lower()
    assert "predicates" in result.output.lower()


# ---------------------------------------------------------------------------
# bsos validate topology
# ---------------------------------------------------------------------------

def test_validate_topology_all_reachable(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-entry", "Entry", entity_type="space", is_entrance=True)
    _add_entity(eng, "e-hall", "Hall", entity_type="space")
    with Session(eng) as s:
        s.add(SpatialRelationRow(id="sr1", subject_id="e-hall", relation="accessible_from",
                                 object_id="e-entry", confidence=0.9,
                                 knowledge_origin="architectural",
                                 source_model="test", created_at=NOW))
        s.commit()

    result = runner.invoke(app, ["validate", "topology", "--db", db])
    assert result.exit_code == 0, result.output
    assert "reachable" in result.output.lower()


def test_validate_topology_detects_unreachable(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-entry", "Entry", entity_type="space", is_entrance=True)
    _add_entity(eng, "e-basement", "Basement", entity_type="space")
    # No accessible_from edge to Basement

    result = runner.invoke(app, ["validate", "topology", "--db", db])
    assert result.exit_code != 0
    assert "Basement" in result.output


def test_validate_topology_warns_no_entrances(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-hall", "Hall", entity_type="space")

    result = runner.invoke(app, ["validate", "topology", "--db", db])
    assert result.exit_code == 0
    assert "entrance" in result.stderr.lower()


def test_validate_topology_no_spaces(tmp_path):
    db = _init_db(tmp_path)
    result = runner.invoke(app, ["validate", "topology", "--db", db])
    assert result.exit_code == 0
    assert "No space" in result.output


# ---------------------------------------------------------------------------
# bsos curate set-entrance
# ---------------------------------------------------------------------------

def test_curate_set_entrance(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-lobby", "Lobby", entity_type="space")

    result = runner.invoke(app, ["curate", "set-entrance", "Lobby", "--db", db])
    assert result.exit_code == 0, result.output
    assert "Lobby" in result.output

    with Session(eng) as s:
        row = s.get(EntityRow, "e-lobby")
    assert row.is_entrance is True


def test_curate_set_entrance_unset(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-lobby", "Lobby", entity_type="space", is_entrance=True)

    result = runner.invoke(app, ["curate", "set-entrance", "Lobby", "--unset", "--db", db])
    assert result.exit_code == 0, result.output

    with Session(eng) as s:
        row = s.get(EntityRow, "e-lobby")
    assert row.is_entrance is False


def test_curate_set_entrance_entity_not_found(tmp_path):
    db = _init_db(tmp_path)
    result = runner.invoke(app, ["curate", "set-entrance", "NoSuch", "--db", db])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# bsos curate add / list / export / verify
# ---------------------------------------------------------------------------

def _add_assertion(engine, aid, subject_id, predicate, object_id,
                   subject_type="component", object_type="component",
                   source_model="gpt-4", status="accepted"):
    from bsos.persistence.models import AssertionRow
    with Session(engine) as s:
        s.add(AssertionRow(
            id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id,
            subject_type=subject_type, object_type=object_type,
            confidence=0.9, status=status, knowledge_origin="engineering",
            source_model=source_model, created_at=NOW,
        ))
        s.commit()


def test_curate_add_basic(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")

    result = runner.invoke(app, ["curate", "add", "Roof", "requires", "Drain", "--db", db])
    assert result.exit_code == 0, result.output
    assert "Roof" in result.output
    assert "requires" in result.output
    assert "Drain" in result.output

    from bsos.persistence.models import AssertionRow
    with Session(eng) as s:
        rows = s.exec(select(AssertionRow).where(AssertionRow.source_model == "human")).all()
    assert len(rows) == 1
    assert rows[0].predicate == "requires"
    assert rows[0].status == "accepted"


def test_curate_add_unknown_predicate(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")

    result = runner.invoke(app, ["curate", "add", "Roof", "no_such_predicate", "Drain", "--db", db])
    assert result.exit_code != 0


def test_curate_add_missing_entity(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")

    result = runner.invoke(app, ["curate", "add", "Roof", "requires", "NoSuch", "--db", db])
    assert result.exit_code != 0


def test_curate_add_with_conditions_exceptions(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")

    result = runner.invoke(app, [
        "curate", "add", "Roof", "requires", "Drain",
        "--condition", "in wet climate",
        "--exception", "green roof",
        "--db", db,
    ])
    assert result.exit_code == 0, result.output

    from bsos.persistence.models import AssertionRow
    with Session(eng) as s:
        rows = s.exec(select(AssertionRow).where(AssertionRow.source_model == "human")).all()
    assert len(rows) == 1
    import json
    assert "in wet climate" in json.loads(rows[0].conditions)
    assert "green roof" in json.loads(rows[0].exceptions)


def test_curate_list_empty(tmp_path):
    db = _init_db(tmp_path)
    result = runner.invoke(app, ["curate", "list", "--db", db])
    assert result.exit_code == 0
    assert "No ground-truth" in result.output


def test_curate_list_shows_human_assertions(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")
    _add_assertion(eng, "a-human", "e-roof", "requires", "e-drain", source_model="human")
    _add_assertion(eng, "a-model", "e-roof", "requires", "e-drain", source_model="gpt-4")

    result = runner.invoke(app, ["curate", "list", "--db", db])
    assert result.exit_code == 0
    assert "1 ground-truth" in result.output
    assert "Roof" in result.output


def test_curate_list_filter_by_entity(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")
    _add_entity(eng, "e-wall", "Wall")
    _add_entity(eng, "e-insul", "Insulation")
    _add_assertion(eng, "a1", "e-roof", "requires", "e-drain", source_model="human")
    _add_assertion(eng, "a2", "e-wall", "requires", "e-insul", source_model="human")

    result = runner.invoke(app, ["curate", "list", "--entity", "Roof", "--db", db])
    assert result.exit_code == 0
    assert "Roof" in result.output
    assert "Wall" not in result.output


def test_curate_export_json(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")
    _add_assertion(eng, "a-human", "e-roof", "requires", "e-drain", source_model="human")

    result = runner.invoke(app, ["curate", "export", "--db", db])
    assert result.exit_code == 0
    import json
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["subject"] == "Roof"
    assert data[0]["predicate"] == "requires"
    assert data[0]["object"] == "Drain"


def test_curate_export_to_file(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")
    _add_assertion(eng, "a-human", "e-roof", "requires", "e-drain", source_model="human")

    out = str(tmp_path / "export.json")
    result = runner.invoke(app, ["curate", "export", "--output", out, "--db", db])
    assert result.exit_code == 0
    import json
    with open(out) as f:
        data = json.load(f)
    assert len(data) == 1


def test_curate_verify_no_ground_truth(tmp_path):
    db = _init_db(tmp_path)
    result = runner.invoke(app, ["curate", "verify", "--db", db])
    assert result.exit_code == 0
    assert "No ground-truth" in result.output


def test_curate_verify_exact_match(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")
    _add_assertion(eng, "a-human", "e-roof", "requires", "e-drain", source_model="human")
    _add_assertion(eng, "a-model", "e-roof", "requires", "e-drain", source_model="gpt-4", status="accepted")

    result = runner.invoke(app, ["curate", "verify", "--db", db])
    assert result.exit_code == 0
    assert "1/1" in result.output
    assert "PASS" in result.output


def test_curate_verify_below_target(tmp_path):
    db = _init_db(tmp_path)
    eng = _engine(db)
    _add_entity(eng, "e-roof", "Roof")
    _add_entity(eng, "e-drain", "Drain")
    _add_assertion(eng, "a-human", "e-roof", "requires", "e-drain", source_model="human")
    # No corpus assertions — coverage = 0%

    result = runner.invoke(app, ["curate", "verify", "--db", db])
    assert result.exit_code != 0
    assert "WARNING" in result.output or "0/" in result.output
