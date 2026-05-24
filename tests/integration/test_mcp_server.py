"""Integration tests for the BSOS MCP server tools."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session

import numpy as np

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import AssertionRow, EntityAliasRow, EntityRow, EmbeddingRow
from bsos.mcp_server.server import (
    create_server,
    get_dependencies_tool,
    get_requirements_tool,
    resolve_entity,
    search_entities_tool,
    SEARCH_EMBEDDING_MODEL,
)

NOW = datetime.now(timezone.utc)


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


def add_entity(session: Session, eid: str, name: str, entity_type: str = "component") -> EntityRow:
    row = EntityRow(id=eid, name=name, entity_type=entity_type,
                    source_model="test", created_at=NOW)
    session.add(row)
    return row


def add_assertion(
    session: Session, aid: str, subject_id: str, predicate: str, object_id: str,
    confidence: float = 0.8, knowledge_origin: str = "engineering",
) -> AssertionRow:
    row = AssertionRow(
        id=aid, subject_id=subject_id, predicate=predicate, object_id=object_id,
        subject_type="component", object_type="component",
        confidence=confidence, knowledge_origin=knowledge_origin, status="proposed",
        source_model="test", created_at=NOW,
    )
    session.add(row)
    return row


# ---------------------------------------------------------------------------
# resolve_entity
# ---------------------------------------------------------------------------

def test_resolve_entity_by_name(session):
    add_entity(session, "e1", "Roof")
    session.commit()
    assert resolve_entity(session, "Roof") is not None
    assert resolve_entity(session, "roof") is not None  # case-insensitive


def test_resolve_entity_by_alias(session):
    add_entity(session, "e1", "Roof")
    session.add(EntityAliasRow(entity_id="e1", alias="rooftop"))
    session.commit()
    result = resolve_entity(session, "rooftop")
    assert result is not None
    assert result.name == "Roof"


def test_resolve_entity_not_found(session):
    session.commit()
    assert resolve_entity(session, "Unicorn") is None


def test_resolve_entity_skips_merged(session):
    row = add_entity(session, "e1", "Roof")
    row.status = "merged"
    session.commit()
    assert resolve_entity(session, "Roof") is None


# ---------------------------------------------------------------------------
# get_requirements_tool
# ---------------------------------------------------------------------------

def test_get_requirements_returns_assertions(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Waterproof membrane", "material")
    add_assertion(session, "a1", "e1", "requires", "e2")
    session.commit()

    result = get_requirements_tool(session, "Roof")
    assert result["entity"] == "Roof"
    assert len(result["assertions"]) == 1
    a = result["assertions"][0]
    assert a["subject"] == "Roof"
    assert a["predicate"] == "requires"
    assert a["object"] == "Waterproof membrane"


def test_get_requirements_includes_depends_on(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Structure", "component")
    add_assertion(session, "a1", "e1", "depends_on", "e2")
    session.commit()

    result = get_requirements_tool(session, "Roof")
    assert len(result["assertions"]) == 1
    assert result["assertions"][0]["predicate"] == "depends_on"


def test_get_requirements_excludes_other_predicates(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Rain", "component")
    add_assertion(session, "a1", "e1", "protects_from", "e2")
    session.commit()

    result = get_requirements_tool(session, "Roof")
    assert len(result["assertions"]) == 0


def test_get_requirements_entity_not_found(session):
    session.commit()
    result = get_requirements_tool(session, "Unicorn")
    assert result["error"] == "entity_not_found"
    assert result["query"] == "Unicorn"


def test_get_requirements_only_where_subject(session):
    """Only return assertions where entity is subject, not object."""
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Structure")
    add_assertion(session, "a1", "e2", "requires", "e1")  # e1 is object here
    session.commit()

    result = get_requirements_tool(session, "Roof")
    assert len(result["assertions"]) == 0


# ---------------------------------------------------------------------------
# get_dependencies_tool
# ---------------------------------------------------------------------------

def test_get_dependencies_as_subject(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Structure")
    add_assertion(session, "a1", "e1", "depends_on", "e2")
    session.commit()

    result = get_dependencies_tool(session, "Roof")
    assert result["entity"] == "Roof"
    assert len(result["assertions"]) == 1


def test_get_dependencies_as_object(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Foundation")
    add_assertion(session, "a1", "e2", "depends_on", "e1")
    session.commit()

    result = get_dependencies_tool(session, "Roof")
    assert len(result["assertions"]) == 1


def test_get_dependencies_both_directions(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Structure")
    add_entity(session, "e3", "Waterproof membrane", "material")
    add_assertion(session, "a1", "e1", "depends_on", "e2")
    add_assertion(session, "a2", "e3", "depends_on", "e1")
    session.commit()

    result = get_dependencies_tool(session, "Roof")
    assert len(result["assertions"]) == 2


def test_get_dependencies_only_depends_on_predicate(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Structure")
    add_assertion(session, "a1", "e1", "requires", "e2")  # not depends_on
    session.commit()

    result = get_dependencies_tool(session, "Roof")
    assert len(result["assertions"]) == 0


def test_get_dependencies_entity_not_found(session):
    session.commit()
    result = get_dependencies_tool(session, "Unicorn")
    assert result["error"] == "entity_not_found"


# ---------------------------------------------------------------------------
# JSON list fields decoded correctly
# ---------------------------------------------------------------------------

def test_assertion_conditions_decoded(session):
    add_entity(session, "e1", "Roof")
    add_entity(session, "e2", "Insulation", "material")
    row = add_assertion(session, "a1", "e1", "requires", "e2")
    row.conditions = '["in cold climates", "above 1000m elevation"]'
    session.commit()

    result = get_requirements_tool(session, "Roof")
    assert result["assertions"][0]["conditions"] == ["in cold climates", "above 1000m elevation"]


# ---------------------------------------------------------------------------
# create_server (smoke test — just verify it returns a FastMCP instance)
# ---------------------------------------------------------------------------

def test_create_server_returns_mcp(engine, tmp_path):
    from mcp.server.fastmcp import FastMCP
    db_path = str(tmp_path / "test.db")
    create_db_engine(db_path)  # ensure DB file exists
    server = create_server(db_path)
    assert isinstance(server, FastMCP)


def test_create_server_has_expected_tools(engine, tmp_path):
    db_path = str(tmp_path / "test.db")
    create_db_engine(db_path)
    server = create_server(db_path)
    tool_names = {t.name for t in server._tool_manager.list_tools()}
    assert "get_requirements" in tool_names
    assert "get_dependencies" in tool_names
    assert "search_entities" in tool_names


# ---------------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------------

DIM = 4  # tiny vectors for test speed


def _make_vec(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def _stub_embedder(texts: list[str]) -> np.ndarray:
    """Return a fixed vector per query text for deterministic tests."""
    vectors = {
        "entrance hall": [1.0, 0.0, 0.0, 0.0],
        "foyer":         [0.9, 0.1, 0.0, 0.0],
        "roof":          [0.0, 0.0, 0.0, 1.0],
    }
    default = [0.25, 0.25, 0.25, 0.25]
    result = np.array([vectors.get(t.lower(), default) for t in texts], dtype=np.float32)
    # Normalise rows
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.where(norms == 0, 1, norms)


class _StubEmbedder:
    def encode(self, texts):
        return _stub_embedder(texts)


def _add_embedding(session, entity_id, vector_values):
    vec = np.array(vector_values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    session.add(EmbeddingRow(
        item_type="entity",
        item_id=entity_id,
        model=SEARCH_EMBEDDING_MODEL,
        dim=DIM,
        content_hash="test",
        vector=vec.tobytes(),
    ))


def test_search_entities_returns_ranked_results(session):
    add_entity(session, "e-foyer", "Foyer")
    add_entity(session, "e-roof", "Roof")
    add_entity(session, "e-beam", "Beam")
    _add_embedding(session, "e-foyer", [0.9, 0.1, 0.0, 0.0])
    _add_embedding(session, "e-roof",  [0.0, 0.0, 0.0, 1.0])
    _add_embedding(session, "e-beam",  [0.0, 0.0, 1.0, 0.0])
    session.commit()

    result = search_entities_tool(session, "entrance hall", _embedder=_StubEmbedder())
    assert result["query"] == "entrance hall"
    names = [r["name"] for r in result["results"]]
    assert names[0] == "Foyer"
    assert "Roof" in names or "Beam" in names


def test_search_entities_respects_max_results(session):
    for i in range(5):
        add_entity(session, f"e{i}", f"Entity{i}")
        _add_embedding(session, f"e{i}", [1.0, float(i) * 0.1, 0.0, 0.0])
    session.commit()

    result = search_entities_tool(session, "entrance hall", max_results=2, _embedder=_StubEmbedder())
    assert len(result["results"]) <= 2


def test_search_entities_skips_merged(session):
    add_entity(session, "e-merged", "OldName")
    session.get(EntityRow, "e-merged").status = "merged"
    _add_embedding(session, "e-merged", [1.0, 0.0, 0.0, 0.0])
    add_entity(session, "e-active", "ActiveName")
    _add_embedding(session, "e-active", [0.9, 0.1, 0.0, 0.0])
    session.commit()

    result = search_entities_tool(session, "entrance hall", _embedder=_StubEmbedder())
    names = [r["name"] for r in result["results"]]
    assert "OldName" not in names
    assert "ActiveName" in names


def test_search_entities_min_score_filters(session):
    add_entity(session, "e-close", "CloseMatch")
    add_entity(session, "e-far", "FarMatch")
    _add_embedding(session, "e-close", [1.0, 0.0, 0.0, 0.0])
    _add_embedding(session, "e-far",   [0.0, 0.0, 1.0, 0.0])
    session.commit()

    result = search_entities_tool(
        session, "entrance hall", min_score=0.8, _embedder=_StubEmbedder()
    )
    names = [r["name"] for r in result["results"]]
    assert "CloseMatch" in names
    assert "FarMatch" not in names
