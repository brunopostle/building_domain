"""Integration tests for persistence layer against a real SQLite database."""
import tempfile
import os
from datetime import datetime, timezone
import pytest
from sqlmodel import Session
from bsos.persistence.database import create_db_engine, create_views, verify_views
from bsos.persistence.repos.entity import EntityRepository
from bsos.persistence.repos.assertion import AssertionRepository
from bsos.models.entity import Entity
from bsos.models.assertion import Assertion

NOW = datetime.now(timezone.utc)
BASE_PROV = dict(source_model="test", source_prompt="p", created_at=NOW)
PROV = dict(**BASE_PROV, confidence=0.9, knowledge_origin="physical")


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def make_entity(**kwargs) -> Entity:
    defaults = dict(id="e1", name="roof", entity_type="component", **BASE_PROV)
    defaults.update(kwargs)
    return Entity(**defaults)


def make_assertion(**kwargs) -> Assertion:
    defaults = dict(
        id="a1", subject_id="e1", predicate="requires", object_id="e2",
        subject_type="component", object_type="component", **PROV,
    )
    defaults.update(kwargs)
    return Assertion(**defaults)


def test_entity_roundtrip(session):
    repo = EntityRepository(session)
    e = make_entity()
    repo.add(e)
    session.commit()
    result = repo.get("e1")
    assert result is not None
    assert result.name == "roof"
    assert result.entity_type == "component"
    assert result.status == "proposed"


def test_entity_get_by_name(session):
    repo = EntityRepository(session)
    repo.add(make_entity())
    session.commit()
    result = repo.get_by_name("roof")
    assert result is not None
    assert result.id == "e1"


def test_entity_alias(session):
    repo = EntityRepository(session)
    repo.add(make_entity())
    repo.add_alias("e1", "rooftop")
    session.commit()
    result = repo.get_by_name_or_alias("rooftop")
    assert result is not None
    assert result.name == "roof"
    aliases = repo.get_aliases("e1")
    assert "rooftop" in aliases


def test_entity_get_by_name_or_alias_case_insensitive(session):
    repo = EntityRepository(session)
    repo.add(make_entity())
    session.commit()
    result = repo.get_by_name_or_alias("ROOF")
    assert result is not None


def test_assertion_roundtrip_with_json_lists(session):
    # Add referenced entities first to satisfy any future FK constraints
    e_repo = EntityRepository(session)
    e_repo.add(make_entity(id="e1", name="roof"))
    e_repo.add(make_entity(id="e2", name="precipitation"))
    session.commit()

    repo = AssertionRepository(session)
    a = make_assertion(
        conditions=["in wet climates"],
        exceptions=["covered walkways"],
        applicability=["residential"],
    )
    repo.add(a)
    session.commit()

    result = repo.get("a1")
    assert result is not None
    assert result.conditions == ["in wet climates"]
    assert result.exceptions == ["covered walkways"]
    assert result.applicability == ["residential"]


def test_assertion_list_by_subject(session):
    e_repo = EntityRepository(session)
    e_repo.add(make_entity(id="e1", name="roof"))
    e_repo.add(make_entity(id="e2", name="structure"))
    session.commit()

    repo = AssertionRepository(session)
    repo.add(make_assertion(id="a1", object_id="e2", predicate="requires"))
    repo.add(make_assertion(id="a2", object_id="e2", predicate="protects_from"))
    session.commit()

    results = repo.list_by_subject("e1")
    assert len(results) == 2


def test_views_created(engine):
    assert verify_views(engine) is True


def test_wal_mode(engine):
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("PRAGMA journal_mode")).fetchone()
    assert result[0] == "wal"
