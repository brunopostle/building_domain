"""Tests for context-aware filtering and propose_assertion MCP tools."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import AssertionRow, EntityRow
from bsos.mcp_server.server import (
    get_requirements_tool,
    get_dependencies_tool,
    propose_assertion_tool,
)

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def add_entity(engine, eid, name, entity_type="component"):
    with Session(engine) as s:
        s.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                        source_model="test", created_at=NOW))
        s.commit()


def add_assertion(engine, aid, subject_id, predicate, object_id,
                  applicability=None, confidence=0.8):
    with Session(engine) as s:
        s.add(AssertionRow(
            id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id,
            subject_type="component", object_type="component",
            confidence=confidence, knowledge_origin="engineering", status="proposed",
            applicability=json.dumps(applicability or []),
            source_model="test", created_at=NOW,
        ))
        s.commit()


# ---------------------------------------------------------------------------
# Context filtering on get_requirements
# ---------------------------------------------------------------------------

def test_context_filter_no_context_returns_all(engine):
    add_entity(engine, "e-room", "Room")
    add_entity(engine, "e-light", "Natural Light")
    add_entity(engine, "e-heat", "Heating")
    add_assertion(engine, "a1", "e-room", "requires", "e-light",
                  applicability=["residential"])
    add_assertion(engine, "a2", "e-room", "requires", "e-heat",
                  applicability=["healthcare"])

    with Session(engine) as s:
        result = get_requirements_tool(s, "Room")
    assert len(result["assertions"]) == 2


def test_context_filter_matches_applicability(engine):
    add_entity(engine, "e-room", "Room")
    add_entity(engine, "e-light", "Natural Light")
    add_entity(engine, "e-heat", "Heating")
    add_assertion(engine, "a1", "e-room", "requires", "e-light",
                  applicability=["residential"])
    add_assertion(engine, "a2", "e-room", "requires", "e-heat",
                  applicability=["healthcare"])

    with Session(engine) as s:
        result = get_requirements_tool(s, "Room", context="residential")
    assert len(result["assertions"]) == 1
    assert result["assertions"][0]["object"] == "Natural Light"


def test_context_filter_empty_applicability_always_included(engine):
    add_entity(engine, "e-room", "Room")
    add_entity(engine, "e-air", "Ventilation")
    add_entity(engine, "e-heat", "Heating")
    add_assertion(engine, "a1", "e-room", "requires", "e-air",
                  applicability=[])  # universal
    add_assertion(engine, "a2", "e-room", "requires", "e-heat",
                  applicability=["healthcare"])

    with Session(engine) as s:
        result = get_requirements_tool(s, "Room", context="residential")
    # universal row included, healthcare-only row excluded
    assert len(result["assertions"]) == 1
    assert result["assertions"][0]["object"] == "Ventilation"


def test_context_filter_case_insensitive(engine):
    add_entity(engine, "e-room", "Room")
    add_entity(engine, "e-light", "Natural Light")
    add_assertion(engine, "a1", "e-room", "requires", "e-light",
                  applicability=["Residential"])

    with Session(engine) as s:
        result = get_requirements_tool(s, "Room", context="RESIDENTIAL")
    assert len(result["assertions"]) == 1


def test_context_filter_substring_match(engine):
    add_entity(engine, "e-room", "Room")
    add_entity(engine, "e-light", "Natural Light")
    add_assertion(engine, "a1", "e-room", "requires", "e-light",
                  applicability=["hot_climate_residential"])

    with Session(engine) as s:
        result = get_requirements_tool(s, "Room", context="residential")
    assert len(result["assertions"]) == 1


def test_context_filter_on_dependencies(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-struct", "Structure")
    add_assertion(engine, "a1", "e-roof", "depends_on", "e-struct",
                  applicability=["commercial"])

    with Session(engine) as s:
        result = get_dependencies_tool(s, "Roof", context="residential")
    assert len(result["assertions"]) == 0

    with Session(engine) as s:
        result = get_dependencies_tool(s, "Roof", context="commercial")
    assert len(result["assertions"]) == 1


# ---------------------------------------------------------------------------
# propose_assertion
# ---------------------------------------------------------------------------

def test_propose_assertion_creates_proposed_row(engine):
    add_entity(engine, "e-room", "Room")
    add_entity(engine, "e-light", "Natural Light")

    with Session(engine) as s:
        result = propose_assertion_tool(
            s, "Room", "requires", "Natural Light",
            rationale="Rooms need light for habitability",
            confidence=0.85,
        )

    assert result["status"] == "proposed"
    assert "assertion_id" in result
    assert result["assertion"]["subject"] == "Room"
    assert result["assertion"]["predicate"] == "requires"
    assert result["assertion"]["object"] == "Natural Light"
    assert result["assertion"]["confidence"] == 0.85
    assert result["assertion"]["status"] == "proposed"


def test_propose_assertion_persisted_in_db(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-found", "Foundation")

    with Session(engine) as s:
        result = propose_assertion_tool(
            s, "Wall", "depends_on", "Foundation",
            rationale="Walls sit on foundations",
        )
    assertion_id = result["assertion_id"]

    with Session(engine) as s:
        row = s.get(AssertionRow, assertion_id)
    assert row is not None
    assert row.predicate == "depends_on"
    assert row.status == "proposed"
    assert row.source_model == "mcp_agent"
    assert row.rationale == "Walls sit on foundations"


def test_propose_assertion_subject_not_found(engine):
    add_entity(engine, "e-found", "Foundation")

    with Session(engine) as s:
        result = propose_assertion_tool(
            s, "NonExistent", "depends_on", "Foundation", rationale="test"
        )
    assert result["error"] == "subject_not_found"


def test_propose_assertion_object_not_found(engine):
    add_entity(engine, "e-wall", "Wall")

    with Session(engine) as s:
        result = propose_assertion_tool(
            s, "Wall", "depends_on", "NonExistent", rationale="test"
        )
    assert result["error"] == "object_not_found"


def test_propose_assertion_default_confidence_and_origin(engine):
    add_entity(engine, "e-roof", "Roof")
    add_entity(engine, "e-struct", "Structure")

    with Session(engine) as s:
        result = propose_assertion_tool(
            s, "Roof", "requires", "Structure", rationale="Roofs need support"
        )

    assert result["assertion"]["confidence"] == 0.7
    assert result["assertion"]["knowledge_origin"] == "architectural"
