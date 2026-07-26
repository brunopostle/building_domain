"""Integration tests for `bsos export` / `bsos import` round-tripping."""
import json
from datetime import datetime, timezone

from typer.testing import CliRunner
from sqlmodel import Session, select

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine
from bsos.persistence.models import (
    AbstractionNodeRow, AntiPatternRow, AssertionRow, ConstraintRow,
    EntityAliasRow, EntityRow, ForceRow, PatternRow,
)

runner = CliRunner(mix_stderr=False)
NOW = datetime.now(timezone.utc)


def _init_db(tmp_path, name="t.db"):
    db = tmp_path / name
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _engine(db_path):
    return create_db_engine(db_path)


def _seed_abstraction_node(engine):
    """Two entities, an assertion linking them, and an abstraction node over it."""
    with Session(engine) as s:
        s.add(EntityRow(id="e-wall", name="Wall", entity_type="component",
                         status="accepted", source_model="test", created_at=NOW))
        s.add(EntityRow(id="e-window", name="Window", entity_type="component",
                         status="accepted", source_model="test", created_at=NOW))
        s.add(AssertionRow(
            id="a-1", subject_id="e-window", predicate="requires",
            object_id="e-wall", subject_type="component", object_type="component",
            source_model="test", created_at=NOW, confidence=0.9,
            status="accepted", knowledge_origin="test",
        ))
        s.add(AbstractionNodeRow(
            id="n-1", statement="Windows require walls",
            child_ids=json.dumps(["a-1"]),
            abstraction_rationale="test", source_model="test",
            created_at=NOW, confidence=0.8, status="accepted",
        ))
        s.commit()


def test_export_abstraction_node_includes_child_ids(tmp_path):
    db = _init_db(tmp_path)
    _seed_abstraction_node(_engine(db))
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db, "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    node = data["abstraction_nodes"][0]
    assert node["child_ids"] == ["a-1"]
    assert "child_count" not in node


def test_import_round_trips_abstraction_node_child_ids(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    _seed_abstraction_node(_engine(db1))
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    assert "dropped" not in result.output.lower()

    with Session(_engine(db2)) as s:
        node = s.exec(select(AbstractionNodeRow)).one()
        assert json.loads(node.child_ids) == ["a-1"]


def test_import_drops_child_ids_for_missing_assertions(tmp_path):
    db2 = _init_db(tmp_path, "t2.db")
    data = {
        "abstraction_nodes": [{
            "id": "n-1",
            "statement": "Orphaned abstraction",
            "child_ids": ["missing-assertion"],
            "abstraction_rationale": "test",
            "confidence": 0.8,
            "status": "accepted",
            "created_at": None,
        }],
    }
    src = tmp_path / "orphan.json"
    src.write_text(json.dumps(data))

    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(src), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    assert "1 abstraction node child reference" in result.stderr

    with Session(_engine(db2)) as s:
        node = s.exec(select(AbstractionNodeRow)).one()
        assert json.loads(node.child_ids) == []


# ---------------------------------------------------------------------------
# entities.is_entrance
# ---------------------------------------------------------------------------

def test_import_round_trips_entity_is_entrance(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    with Session(_engine(db1)) as s:
        s.add(EntityRow(id="e-lobby", name="Lobby Door", entity_type="space",
                         status="accepted", is_entrance=True,
                         source_model="test", created_at=NOW))
        s.commit()
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["entities"][0]["is_entrance"] is True

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    with Session(_engine(db2)) as s:
        entity = s.exec(select(EntityRow)).one()
        assert entity.is_entrance is True


# ---------------------------------------------------------------------------
# entity_aliases
# ---------------------------------------------------------------------------

def test_import_round_trips_entity_aliases(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    with Session(_engine(db1)) as s:
        s.add(EntityRow(id="e-wall", name="Wall", entity_type="component",
                         status="accepted", source_model="test", created_at=NOW))
        s.add(EntityAliasRow(entity_id="e-wall", alias="Masonry Wall"))
        s.commit()
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["entity_aliases"] == [{"entity": "Wall", "alias": "Masonry Wall"}]

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    with Session(_engine(db2)) as s:
        alias = s.exec(select(EntityAliasRow)).one()
        entity = s.get(EntityRow, alias.entity_id)
        assert entity.name == "Wall"
        assert alias.alias == "Masonry Wall"

    # Re-importing the same snapshot must not duplicate the alias row.
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    with Session(_engine(db2)) as s:
        assert len(s.exec(select(EntityAliasRow)).all()) == 1


# ---------------------------------------------------------------------------
# forces.affects
# ---------------------------------------------------------------------------

def test_import_round_trips_force_affects(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    with Session(_engine(db1)) as s:
        s.add(EntityRow(id="e-wall", name="Wall", entity_type="component",
                         status="accepted", source_model="test", created_at=NOW))
        s.add(ForceRow(id="f-1", name="Thermal bridging", direction="minimize",
                        affects=json.dumps(["e-wall"]), source_model="test",
                        created_at=NOW, confidence=0.7, status="accepted",
                        knowledge_origin="test"))
        s.commit()
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["forces"][0]["affects"] == ["Wall"]

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    assert "dropped" not in result.stderr.lower()
    with Session(_engine(db2)) as s:
        force = s.exec(select(ForceRow)).one()
        assert json.loads(force.affects) == ["e-wall"]


def test_import_drops_force_affects_for_missing_entity(tmp_path):
    db2 = _init_db(tmp_path, "t2.db")
    data = {
        "forces": [{
            "id": "f-1", "name": "Orphan force", "direction": "minimize",
            "affects": ["Nonexistent Entity"], "rationale": "", "confidence": 0.7,
            "knowledge_origin": "test", "status": "accepted", "created_at": None,
        }],
    }
    src = tmp_path / "orphan.json"
    src.write_text(json.dumps(data))
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(src), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    assert "1 force 'affects' reference" in result.stderr
    with Session(_engine(db2)) as s:
        force = s.exec(select(ForceRow)).one()
        assert json.loads(force.affects) == []


# ---------------------------------------------------------------------------
# patterns.force_ids / related_pattern_ids
# ---------------------------------------------------------------------------

def test_import_round_trips_pattern_force_and_related_ids(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    with Session(_engine(db1)) as s:
        s.add(ForceRow(id="f-1", name="Daylighting", direction="maximize",
                        affects=json.dumps([]), source_model="test",
                        created_at=NOW, confidence=0.7, status="accepted",
                        knowledge_origin="test"))
        s.add(PatternRow(id="p-1", name="Daylight Core", problem="p", solution="s",
                          force_ids=json.dumps(["f-1"]),
                          related_pattern_ids=json.dumps(["p-2"]),
                          source_model="test", created_at=NOW, confidence=0.7,
                          status="accepted", knowledge_origin="test"))
        s.add(PatternRow(id="p-2", name="Front-to-Back Flow", problem="p", solution="s",
                          related_pattern_ids=json.dumps(["p-1"]),
                          source_model="test", created_at=NOW, confidence=0.7,
                          status="accepted", knowledge_origin="test"))
        s.commit()
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output
    patterns_by_id = {p["id"]: p for p in json.loads(out.read_text())["patterns"]}
    assert patterns_by_id["p-1"]["force_ids"] == ["f-1"]
    assert patterns_by_id["p-1"]["related_pattern_ids"] == ["p-2"]
    assert patterns_by_id["p-2"]["related_pattern_ids"] == ["p-1"]

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    assert "dropped" not in result.stderr.lower()
    with Session(_engine(db2)) as s:
        p1 = s.get(PatternRow, "p-1")
        p2 = s.get(PatternRow, "p-2")
        assert json.loads(p1.force_ids) == ["f-1"]
        assert json.loads(p1.related_pattern_ids) == ["p-2"]
        assert json.loads(p2.related_pattern_ids) == ["p-1"]


def test_import_drops_pattern_refs_for_missing_force_and_pattern(tmp_path):
    db2 = _init_db(tmp_path, "t2.db")
    data = {
        "patterns": [{
            "id": "p-1", "name": "Orphan pattern", "problem": "p", "solution": "s",
            "force_ids": ["missing-force"], "related_pattern_ids": ["missing-pattern"],
            "confidence": 0.7, "knowledge_origin": "test", "status": "accepted",
            "created_at": None,
        }],
    }
    src = tmp_path / "orphan.json"
    src.write_text(json.dumps(data))
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(src), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    assert "dropped 1 force reference" in result.stderr
    assert "1 related-pattern reference" in result.stderr
    with Session(_engine(db2)) as s:
        pattern = s.exec(select(PatternRow)).one()
        assert json.loads(pattern.force_ids) == []
        assert json.loads(pattern.related_pattern_ids) == []


# ---------------------------------------------------------------------------
# constraints.rationale / antipatterns.rationale
# ---------------------------------------------------------------------------

def test_import_round_trips_constraint_rationale(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    with Session(_engine(db1)) as s:
        s.add(EntityRow(id="e-wall", name="Wall", entity_type="component",
                         status="accepted", source_model="test", created_at=NOW))
        s.add(ConstraintRow(
            id="c-1", subject_id="e-wall", rule="Must be load-bearing",
            constraint_type="structural", source_model="test", created_at=NOW,
            confidence=0.9, status="accepted", knowledge_origin="test",
            rationale="Because the roof loads onto it directly.",
        ))
        s.commit()
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["constraints"][0]["rationale"] == (
        "Because the roof loads onto it directly."
    )

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    with Session(_engine(db2)) as s:
        constraint = s.exec(select(ConstraintRow)).one()
        assert constraint.rationale == "Because the roof loads onto it directly."


def test_import_round_trips_antipattern_rationale(tmp_path):
    db1 = _init_db(tmp_path, "t1.db")
    with Session(_engine(db1)) as s:
        s.add(EntityRow(id="e-wall", name="Wall", entity_type="component",
                         status="accepted", source_model="test", created_at=NOW))
        s.add(AntiPatternRow(
            id="ap-1", name="Thermal bridge at wall junction", subject_id="e-wall",
            source_model="test", created_at=NOW, confidence=0.9, status="accepted",
            knowledge_origin="test", rationale="Uninsulated junction leaks heat.",
        ))
        s.commit()
    out = tmp_path / "export.json"
    result = runner.invoke(app, ["export", "--db", db1, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["antipatterns"][0]["rationale"] == (
        "Uninsulated junction leaks heat."
    )

    db2 = _init_db(tmp_path, "t2.db")
    result = runner.invoke(app, [
        "import", "--db", db2, "-i", str(out), "--force", "--skip-index",
    ])
    assert result.exit_code == 0, result.output
    with Session(_engine(db2)) as s:
        antipattern = s.exec(select(AntiPatternRow)).one()
        assert antipattern.rationale == "Uninsulated junction leaks heat."
