"""Integration tests for `bsos seed-ifc-classes` and the schema enumerator."""
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.ifc_schema_seed import (
    SCHEMA_SOURCE,
    iter_ifc_classes,
)
from bsos.persistence.models import EntityRow

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    eng = create_db_engine(str(tmp_path / "test.db"))
    create_views(eng)
    return eng


def _run_seed(engine, **kwargs):
    """Drive the CLI callback against a given engine, bypassing path resolution."""
    from bsos.cli import seed_ifc_classes as mod

    def fake_open_db(db):
        return engine, Session(engine)

    orig = mod.open_db
    mod.open_db = fake_open_db
    try:
        mod.seed_ifc_classes(db=None, **kwargs)
    finally:
        mod.open_db = orig


# --- enumerator ---------------------------------------------------------------

def test_iter_returns_real_entity_classes_only():
    classes = iter_ifc_classes(["IFC4"])
    names = {c.name for c in classes}
    # real entity classes present
    assert "IfcWall" in names
    assert "IfcSpace" in names
    assert "IfcDoor" in names
    # defined types / enums / selects excluded
    assert "IfcLengthMeasure" not in names
    assert "IfcWallTypeEnum" not in names
    # all names are canonical IFC names
    assert all(n.startswith("Ifc") for n in names)


def test_supertype_and_abstract_captured():
    by_name = {c.name: c for c in iter_ifc_classes(["IFC4"])}
    wall = by_name["IfcWall"]
    assert wall.supertype == "IfcBuildingElement"
    assert wall.is_abstract is False
    assert "IFC entity class" in wall.description


def test_union_across_schemas_records_both():
    classes = iter_ifc_classes(["IFC4", "IFC4X3"])
    by_name = {c.name: c for c in classes}
    # IfcWall exists in both; recorded once with both schemas
    assert by_name["IfcWall"].schemas == ("IFC4", "IFC4X3")
    # IFC4X3-only infrastructure class present
    assert "IfcAlignment" in by_name


# --- CLI seeding --------------------------------------------------------------

def test_seed_creates_canonical_entities(engine):
    _run_seed(engine, schema=["IFC4"], force=False)
    with Session(engine) as s:
        rows = s.exec(
            select(EntityRow).where(EntityRow.entity_type == "ifc_class")
        ).all()
        assert len(rows) == len(iter_ifc_classes(["IFC4"]))
        assert all(r.source_model == SCHEMA_SOURCE for r in rows)
        assert all(r.status == "accepted" for r in rows)
        names = {r.name for r in rows}
        assert "IfcWall" in names


def test_seed_is_idempotent(engine):
    _run_seed(engine, schema=["IFC4"], force=False)
    _run_seed(engine, schema=["IFC4"], force=False)
    with Session(engine) as s:
        rows = s.exec(
            select(EntityRow).where(EntityRow.entity_type == "ifc_class")
        ).all()
        assert len(rows) == len(iter_ifc_classes(["IFC4"]))


def test_seed_adopts_matching_llm_entity_in_place(engine):
    # a valid prose entity minted by the LLM, with the canonical name
    with Session(engine) as s:
        s.add(EntityRow(
            id="llm-wall",
            name="IfcWall",
            entity_type="ifc_class",
            description="hallucinated prose",
            status="proposed",
            source_model="claude-haiku",
            created_at=NOW,
        ))
        s.commit()

    _run_seed(engine, schema=["IFC4"], force=False)

    with Session(engine) as s:
        walls = s.exec(
            select(EntityRow).where(
                EntityRow.name == "IfcWall",
                EntityRow.status != "merged",
            )
        ).all()
        # adopted in place — same row id preserved, no duplicate created
        assert len(walls) == 1
        assert walls[0].id == "llm-wall"
        assert walls[0].source_model == SCHEMA_SOURCE
        assert walls[0].status == "accepted"
        assert "IFC entity class" in walls[0].description
