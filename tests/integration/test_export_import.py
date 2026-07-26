"""Integration tests for `bsos export` / `bsos import` round-tripping."""
import json
from datetime import datetime, timezone

from typer.testing import CliRunner
from sqlmodel import Session, select

from bsos.cli.main import app
from bsos.persistence.database import create_db_engine
from bsos.persistence.models import (
    AbstractionNodeRow, AssertionRow, EntityRow,
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
