"""Integration tests for get_ifc_psets MCP tool."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import Session

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import EntityRow, IFCPropertySetRow
from bsos.mcp_server.server import get_ifc_psets_tool

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"
    eng = create_db_engine(str(db))
    create_views(eng)
    return eng


def add_entity(engine, eid, name, entity_type="component", status="proposed"):
    with Session(engine) as s:
        s.add(EntityRow(id=eid, name=name, entity_type=entity_type,
                        status=status, source_model="test", created_at=NOW))
        s.commit()


def add_pset_row(engine, entity_id, ifc_class, pset_name, property_name,
                 value_type="IfcLabel", description="desc", rationale="rationale"):
    with Session(engine) as s:
        s.add(IFCPropertySetRow(
            id=str(uuid.uuid4()),
            entity_id=entity_id,
            ifc_class=ifc_class,
            pset_name=pset_name,
            property_name=property_name,
            value_type=value_type,
            description=description,
            rationale=rationale,
        ))
        s.commit()


def test_returns_psets_for_known_entity(engine):
    add_entity(engine, "e-wall", "Wall")
    add_pset_row(engine, "e-wall", "IfcWall", "Pset_WallCommon", "IsExternal",
                 "IfcBoolean", "External flag", "energy compliance")

    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "Wall")

    assert result["entity"] == "Wall"
    assert result["entity_type"] == "component"
    assert result["source"] == "curated"
    assert len(result["ifc_mappings"]) == 1
    mapping = result["ifc_mappings"][0]
    assert mapping["ifc_class"] == "IfcWall"
    assert len(mapping["properties"]) == 1
    prop = mapping["properties"][0]
    assert prop["pset_name"] == "Pset_WallCommon"
    assert prop["property_name"] == "IsExternal"
    assert prop["value_type"] == "IfcBoolean"


def test_groups_properties_by_ifc_class(engine):
    add_entity(engine, "e-space", "Office", entity_type="space")
    add_pset_row(engine, "e-space", "IfcSpace", "Pset_SpaceCommon", "OccupancyType")
    add_pset_row(engine, "e-space", "IfcSpace", "Pset_SpaceCommon", "GrossFloorArea",
                 "IfcAreaMeasure", "area", "area schedule")
    add_pset_row(engine, "e-space", "IfcSpace", "Pset_LightingDesign", "AmbientIlluminance",
                 "IfcIlluminanceMeasure", "illuminance", "lighting check")

    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "Office")

    assert result["source"] == "curated"
    assert len(result["ifc_mappings"]) == 1  # all grouped under IfcSpace
    props = result["ifc_mappings"][0]["properties"]
    pset_names = {p["pset_name"] for p in props}
    assert "Pset_SpaceCommon" in pset_names
    assert "Pset_LightingDesign" in pset_names
    assert len(props) == 3


def test_falls_back_to_entity_type_defaults(engine):
    add_entity(engine, "e-custom", "Custom Space", entity_type="space")
    # No pset rows added

    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "Custom Space")

    assert result["entity"] == "Custom Space"
    assert result["source"] == "entity_type_default"
    assert len(result["ifc_mappings"]) > 0
    assert result["ifc_mappings"][0]["ifc_class"] == "IfcSpace"


def test_returns_note_when_no_defaults_either(engine):
    add_entity(engine, "e-activity", "Bricklaying", entity_type="activity")
    # No pset rows, no defaults for activity type

    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "Bricklaying")

    assert result["entity"] == "Bricklaying"
    assert result["ifc_mappings"] == []
    assert "note" in result


def test_entity_not_found(engine):
    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "Nonexistent Entity XYZ")

    assert result["error"] == "entity_not_found"


def test_case_insensitive_lookup(engine):
    add_entity(engine, "e-door", "Door")
    add_pset_row(engine, "e-door", "IfcDoor", "Pset_DoorCommon", "FireRating")

    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "door")

    assert result["entity"] == "Door"
    assert len(result["ifc_mappings"]) == 1


def test_merged_entity_not_resolved(engine):
    with Session(engine) as s:
        s.add(EntityRow(id="e-merged", name="OldWall", entity_type="component",
                        status="merged", source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_ifc_psets_tool(s, "OldWall")

    assert result["error"] == "entity_not_found"
