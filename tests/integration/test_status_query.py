"""Integration tests for bsos status and bsos query commands."""
import json
from datetime import datetime, timezone
from typer.testing import CliRunner
from sqlmodel import Session
from bsos.cli.main import app
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, AssertionRow

runner = CliRunner()
NOW = datetime.now(timezone.utc)


def _init_db(tmp_path):
    db = tmp_path / "t.db"
    runner.invoke(app, ["init", "--db", str(db), "--no-gitignore"])
    return str(db)


def _add_fixture_data(db_path: str, tmp_path):
    engine = create_db_engine(db_path)
    with Session(engine) as session:
        session.add(EntityRow(
            id="e-roof", name="roof", entity_type="component", status="accepted",
            source_model="test", source_prompt="p", created_at=NOW,
        ))
        session.add(EntityRow(
            id="e-precip", name="precipitation", entity_type="system", status="accepted",
            source_model="test", source_prompt="p", created_at=NOW,
        ))
        session.add(AssertionRow(
            id="a1", subject_id="e-roof", predicate="protects_from", object_id="e-precip",
            subject_type="component", object_type="system",
            confidence=0.95, status="accepted", knowledge_origin="physical",
            source_model="test", source_prompt="p", created_at=NOW,
            conditions="[]", exceptions="[]", applicability="[]",
        ))
        session.add(AssertionRow(
            id="a2", subject_id="e-roof", predicate="requires", object_id="e-precip",
            subject_type="component", object_type="system",
            confidence=0.80, status="proposed", knowledge_origin="engineering",
            source_model="test", source_prompt="p", created_at=NOW,
            conditions='["wet climates"]', exceptions="[]", applicability="[]",
        ))
        session.commit()


def test_status_shows_counts(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["status", "--db", db])
    assert result.exit_code == 0, result.output
    assert "entities" in result.output
    assert "assertions" in result.output


def test_status_json(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["status", "--db", db, "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["entities"]["accepted"] == 2
    assert data["assertions"]["accepted"] == 1
    assert data["assertions"]["proposed"] == 1


def test_query_accepted_only(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["query", "roof", "--db", db])
    assert result.exit_code == 0, result.output
    assert "protects_from" in result.output
    assert "requires" not in result.output  # proposed, excluded by default


def test_query_include_proposed(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["query", "roof", "--db", db, "--include-proposed"])
    assert result.exit_code == 0
    assert "protects_from" in result.output
    assert "requires" in result.output


def test_query_json(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["query", "roof", "--db", db, "--json", "--include-proposed"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 2
    # sorted by confidence DESC
    assert data[0]["confidence"] >= data[1]["confidence"]


def test_query_sorted_by_confidence(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["query", "roof", "--db", db, "--json", "--include-proposed"])
    data = json.loads(result.output)
    confidences = [r["confidence"] for r in data]
    assert confidences == sorted(confidences, reverse=True)


def test_query_entity_not_found(tmp_path):
    db = _init_db(tmp_path)
    result = runner.invoke(app, ["query", "nonexistent", "--db", db])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_query_case_insensitive(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, ["query", "ROOF", "--db", db])
    assert result.exit_code == 0
    assert "protects_from" in result.output


def test_query_min_confidence(tmp_path):
    db = _init_db(tmp_path)
    _add_fixture_data(db, tmp_path)
    result = runner.invoke(app, [
        "query", "roof", "--db", db, "--min-confidence", "0.90", "--include-proposed"
    ])
    assert result.exit_code == 0
    data_check = runner.invoke(app, [
        "query", "roof", "--db", db, "--json", "--min-confidence", "0.90", "--include-proposed"
    ])
    data = json.loads(data_check.output)
    assert all(r["confidence"] >= 0.90 for r in data)
