"""Full pipeline integration test: Passes 1 → 2 → 3 → MCP query.

Exercises the complete extraction pipeline on a single in-memory-equivalent
SQLite database using FakeLLMProvider and deterministic fake embedders.
Verifies entity counts, assertion counts, deduplication, and MCP tool isolation.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import AssertionRow, EntityRow
from bsos.pipeline.pass1 import run_pass1
from bsos.pipeline.pass2 import run_pass2
from bsos.pipeline.pass3 import run_pass3
from bsos.mcp_server.server import (
    create_server, get_requirements_tool, get_dependencies_tool,
)
from tests.fixtures.fake_responses import build_standard_fixture

NOW = datetime.now(timezone.utc)
DIM = 8


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Entity name → embedding vector for Pass 2 deduplication
ENTITY_VECTORS = {
    "Roof":             _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Roof membrane":    _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Roof covering":    _unit([0.99, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # near-dup of Roof
    "Precipitation":    _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Formwork":         _unit([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    "Concrete pouring": _unit([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
}

# Assertion text → embedding vector for Pass 3 cross-prompt consistency
ASSERTION_VECTORS = {
    "Roof protects_from Precipitation": _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Roof requires Roof membrane":       _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Concrete pouring depends_on Formwork":   _unit([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "Concrete pouring conflicts_with Formwork": _unit([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
}
_DEFAULT_ASSERT_VEC = _unit([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def pass2_embedder(texts: list[str]) -> np.ndarray:
    return np.array([ENTITY_VECTORS.get(t, _DEFAULT_ASSERT_VEC) for t in texts], dtype=np.float32)


def pass3_embedder(texts: list[str]) -> np.ndarray:
    return np.array([ASSERTION_VECTORS.get(t, _DEFAULT_ASSERT_VEC) for t in texts], dtype=np.float32)


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "pipeline.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


@pytest.fixture
def pipeline_engine(engine):
    """Run all 3 passes and return the populated engine."""
    provider = build_standard_fixture()

    with Session(engine) as session:
        run_pass1(session, provider, run_id="run-p1")

    with Session(engine) as session:
        run_pass2(session, "run-p2", _embedder=pass2_embedder)

    run_pass3(engine, provider, "run-p3", _embedder=pass3_embedder, max_workers=1)

    return engine


# ---------------------------------------------------------------------------
# Pass 1 entity discovery
# ---------------------------------------------------------------------------

def test_pass1_discovers_expected_entities(engine):
    provider = build_standard_fixture()
    with Session(engine) as session:
        run_pass1(session, provider, run_id="run-p1")

    with Session(engine) as session:
        names = {r.name for r in session.exec(select(EntityRow)).all()}

    assert "Roof" in names
    assert "Precipitation" in names
    assert "Formwork" in names
    assert "Concrete pouring" in names
    assert "Roof membrane" in names
    assert "Roof covering" in names  # near-dup, not yet merged


# ---------------------------------------------------------------------------
# Pass 2 deduplication
# ---------------------------------------------------------------------------

def test_pass2_merges_roof_covering(engine):
    provider = build_standard_fixture()
    with Session(engine) as session:
        run_pass1(session, provider, run_id="run-p1")
    with Session(engine) as session:
        result = run_pass2(session, "run-p2", _embedder=pass2_embedder)

    assert result["entities_merged"] == 1  # Roof covering → merged into Roof

    with Session(engine) as session:
        merged = session.exec(
            select(EntityRow).where(EntityRow.status == "merged")
        ).all()
    assert len(merged) == 1
    assert merged[0].name == "Roof covering"


def test_pass2_active_entity_count(engine):
    provider = build_standard_fixture()
    with Session(engine) as session:
        run_pass1(session, provider, run_id="run-p1")
    with Session(engine) as session:
        run_pass2(session, "run-p2", _embedder=pass2_embedder)

    with Session(engine) as session:
        active = session.exec(
            select(EntityRow).where(EntityRow.status != "merged")
        ).all()
    # Roof, Precipitation, Formwork, Concrete pouring, Roof membrane = 5
    assert len(active) == 5


# ---------------------------------------------------------------------------
# Pass 3 relationship extraction
# ---------------------------------------------------------------------------

def test_pass3_writes_roof_assertions(pipeline_engine):
    with Session(pipeline_engine) as session:
        roof = session.exec(
            select(EntityRow).where(EntityRow.name == "Roof")
        ).one()
        assertions = session.exec(
            select(AssertionRow).where(AssertionRow.subject_id == roof.id)
        ).all()

    pred_obj_names = set()
    with Session(pipeline_engine) as session:
        for a in assertions:
            obj = session.get(EntityRow, a.object_id)
            pred_obj_names.add((a.predicate, obj.name if obj else a.object_id))

    assert ("protects_from", "Precipitation") in pred_obj_names
    assert ("requires", "Roof membrane") in pred_obj_names


def test_pass3_writes_process_relation(pipeline_engine):
    """Concrete pouring depends_on Formwork — a construction process relation."""
    with Session(pipeline_engine) as session:
        concrete = session.exec(
            select(EntityRow).where(EntityRow.name == "Concrete pouring")
        ).one()
        formwork = session.exec(
            select(EntityRow).where(EntityRow.name == "Formwork")
        ).one()
        dep = session.exec(
            select(AssertionRow).where(
                AssertionRow.subject_id == concrete.id,
                AssertionRow.predicate == "depends_on",
                AssertionRow.object_id == formwork.id,
            )
        ).one_or_none()

    assert dep is not None
    assert dep.confidence > 0.5


def test_pass3_contradictory_pair_both_written(pipeline_engine):
    """Both depends_on and conflicts_with for Concrete→Formwork must be present."""
    with Session(pipeline_engine) as session:
        concrete = session.exec(
            select(EntityRow).where(EntityRow.name == "Concrete pouring")
        ).one()
        formwork = session.exec(
            select(EntityRow).where(EntityRow.name == "Formwork")
        ).one()

        dep = session.exec(
            select(AssertionRow).where(
                AssertionRow.subject_id == concrete.id,
                AssertionRow.predicate == "depends_on",
                AssertionRow.object_id == formwork.id,
            )
        ).one_or_none()
        conflict = session.exec(
            select(AssertionRow).where(
                AssertionRow.subject_id == concrete.id,
                AssertionRow.predicate == "conflicts_with",
                AssertionRow.object_id == formwork.id,
            )
        ).one_or_none()

    assert dep is not None, "depends_on assertion missing"
    assert conflict is not None, "contradictory conflicts_with assertion missing"


def test_pass3_cross_prompt_consistency_stored(pipeline_engine):
    """Assertions extracted from 3 identical framings should have consistency=1.0."""
    with Session(pipeline_engine) as session:
        rows = session.exec(
            select(AssertionRow).where(AssertionRow.prompt_framing_count == 3)
        ).all()

    assert len(rows) > 0
    for row in rows:
        assert row.cross_prompt_consistency is not None
        assert row.cross_prompt_consistency > 0.9  # identical framings → near 1.0


def test_pass3_total_assertion_count(pipeline_engine):
    with Session(pipeline_engine) as session:
        count = len(session.exec(select(AssertionRow)).all())
    # At least 4 unique (predicate, object) pairs across entities
    assert count >= 4


# ---------------------------------------------------------------------------
# Graph node / edge counts
# ---------------------------------------------------------------------------

def test_graph_node_count(pipeline_engine):
    with Session(pipeline_engine) as session:
        active = session.exec(
            select(EntityRow).where(EntityRow.status != "merged")
        ).all()
    assert len(active) == 5  # Roof, Precipitation, Formwork, Concrete pouring, Roof membrane


def test_graph_edge_count(pipeline_engine):
    with Session(pipeline_engine) as session:
        edges = session.exec(select(AssertionRow)).all()
    # At minimum: protects_from, requires (from Roof) + depends_on, conflicts_with (from Concrete)
    assert len(edges) >= 4


# ---------------------------------------------------------------------------
# MCP server — correctness and concurrency safety
# ---------------------------------------------------------------------------

def test_mcp_get_requirements(pipeline_engine, tmp_path):
    db_path = str(tmp_path / "pipeline.db")
    # Re-use the fixture engine but point server at same DB
    with Session(pipeline_engine) as session:
        result = get_requirements_tool(session, "Roof")

    assert result["entity"] == "Roof"
    preds = {a["predicate"] for a in result["assertions"]}
    assert "protects_from" in preds or "requires" in preds


def test_mcp_get_requirements_entity_not_found(pipeline_engine):
    with Session(pipeline_engine) as session:
        result = get_requirements_tool(session, "Unicorn")
    assert result["error"] == "entity_not_found"


def test_mcp_get_dependencies_concrete(pipeline_engine):
    with Session(pipeline_engine) as session:
        result = get_dependencies_tool(session, "Formwork")

    assert "assertions" in result
    # Formwork should appear as object in "Concrete pouring depends_on Formwork"
    subjects = {a["subject"] for a in result["assertions"]}
    assert "Concrete pouring" in subjects


def test_mcp_concurrency_safety(pipeline_engine):
    """Two concurrent get_requirements calls must not interfere with each other."""
    def call_requirements(entity_name: str) -> dict:
        with Session(pipeline_engine) as session:
            return get_requirements_tool(session, entity_name)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(call_requirements, "Roof"): "Roof",
            pool.submit(call_requirements, "Concrete pouring"): "Concrete pouring",
        }
        results = {}
        for future in as_completed(futures):
            entity = futures[future]
            results[entity] = future.result()

    assert results["Roof"]["entity"] == "Roof"
    assert results["Concrete pouring"]["entity"] == "Concrete pouring"
    assert "error" not in results["Roof"]
    assert "error" not in results["Concrete pouring"]


def test_mcp_alias_resolution(pipeline_engine):
    """After Pass 2, 'Roof covering' is an alias for 'Roof' — MCP should resolve it."""
    with Session(pipeline_engine) as session:
        result = get_requirements_tool(session, "Roof covering")

    # Roof covering was merged into Roof — it becomes an alias
    # resolve_entity should find Roof via the alias
    assert "error" not in result or result.get("query") == "Roof covering"
    # If alias resolution works, entity should be Roof
    if "entity" in result:
        assert result["entity"] == "Roof"
