"""Integration tests for Phase 2 MCP server tools."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AntiPatternRow, ConstraintRow, EntityRow, ForceRow,
    PatternRow, ProcessRelationRow, SpatialRelationRow,
)
from bsos.mcp_server.server import (
    get_constraints_tool, get_failure_modes_tool, get_forces_tool,
    get_patterns_tool, get_process_sequence_tool, get_spatial_relations_tool,
)

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


# ---------------------------------------------------------------------------
# get_constraints
# ---------------------------------------------------------------------------

def test_get_constraints_returns_rules(engine):
    add_entity(engine, "e-roof", "Roof")
    with Session(engine) as s:
        s.add(ConstraintRow(
            id="c1", subject_id="e-roof",
            rule="Roof must have a drainage path",
            constraint_type="must",
            confidence=0.9, knowledge_origin="engineering",
            source_model="test", created_at=NOW,
        ))
        s.commit()

    with Session(engine) as s:
        result = get_constraints_tool(s, "Roof")

    assert result["entity"] == "Roof"
    assert len(result["constraints"]) == 1
    assert result["constraints"][0]["rule"] == "Roof must have a drainage path"
    assert result["constraints"][0]["constraint_type"] == "must"


def test_get_constraints_entity_not_found(engine):
    with Session(engine) as s:
        result = get_constraints_tool(s, "NoSuchEntity")
    assert result["error"] == "entity_not_found"


def test_get_constraints_sorted_by_confidence_then_origin(engine):
    add_entity(engine, "e-wall", "Wall")
    with Session(engine) as s:
        s.add(ConstraintRow(id="c1", subject_id="e-wall", rule="Rule A",
                            constraint_type="must", confidence=0.5,
                            knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.add(ConstraintRow(id="c2", subject_id="e-wall", rule="Rule B",
                            constraint_type="must", confidence=0.9,
                            knowledge_origin="cultural", source_model="test", created_at=NOW))
        s.add(ConstraintRow(id="c3", subject_id="e-wall", rule="Rule C",
                            constraint_type="must", confidence=0.9,
                            knowledge_origin="physical", source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_constraints_tool(s, "Wall")

    names = [c["rule"] for c in result["constraints"]]
    # highest confidence first; among equal confidence, physical before cultural
    assert names[0] == "Rule C"
    assert names[1] == "Rule B"
    assert names[2] == "Rule A"


def test_get_constraints_min_confidence_filter(engine):
    add_entity(engine, "e-wall", "Wall")
    with Session(engine) as s:
        s.add(ConstraintRow(id="c1", subject_id="e-wall", rule="Low",
                            constraint_type="must", confidence=0.3,
                            knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.add(ConstraintRow(id="c2", subject_id="e-wall", rule="High",
                            constraint_type="must", confidence=0.8,
                            knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_constraints_tool(s, "Wall", min_confidence=0.5)

    assert len(result["constraints"]) == 1
    assert result["constraints"][0]["rule"] == "High"


def test_get_constraints_max_results(engine):
    add_entity(engine, "e-wall", "Wall")
    with Session(engine) as s:
        for i in range(5):
            s.add(ConstraintRow(id=f"c{i}", subject_id="e-wall", rule=f"Rule {i}",
                                constraint_type="must", confidence=0.7,
                                knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_constraints_tool(s, "Wall", max_results=3)

    assert len(result["constraints"]) == 3


def test_get_constraints_only_matching_entity(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-roof", "Roof")
    with Session(engine) as s:
        s.add(ConstraintRow(id="c1", subject_id="e-roof", rule="Roof rule",
                            constraint_type="must", confidence=0.8,
                            knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_constraints_tool(s, "Wall")

    assert result["constraints"] == []


# ---------------------------------------------------------------------------
# get_failure_modes
# ---------------------------------------------------------------------------

def test_get_failure_modes_returns_antipatterns(engine):
    add_entity(engine, "e-wall", "Wall")
    with Session(engine) as s:
        s.add(AntiPatternRow(
            id="ap1", subject_id="e-wall",
            name="Thermal bridging",
            conditions=json.dumps(["Uninsulated junction"]),
            consequences=json.dumps(["Heat loss", "Condensation"]),
            mitigations=json.dumps(["Insulate junction"]),
            confidence=0.85, knowledge_origin="engineering",
            source_model="test", created_at=NOW,
        ))
        s.commit()

    with Session(engine) as s:
        result = get_failure_modes_tool(s, "Wall")

    assert result["entity"] == "Wall"
    assert len(result["failure_modes"]) == 1
    fm = result["failure_modes"][0]
    assert fm["name"] == "Thermal bridging"
    assert "Heat loss" in fm["consequences"]
    assert "Insulate junction" in fm["mitigations"]


def test_get_failure_modes_entity_not_found(engine):
    with Session(engine) as s:
        result = get_failure_modes_tool(s, "Ghost")
    assert result["error"] == "entity_not_found"


def test_get_failure_modes_only_matching_entity(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-roof", "Roof")
    with Session(engine) as s:
        s.add(AntiPatternRow(id="ap1", subject_id="e-roof", name="Roof failure",
                             conditions="[]", consequences="[]", mitigations="[]",
                             confidence=0.7, knowledge_origin="engineering",
                             source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_failure_modes_tool(s, "Wall")

    assert result["failure_modes"] == []


# ---------------------------------------------------------------------------
# get_patterns
# ---------------------------------------------------------------------------

def test_get_patterns_force_ids_resolved(engine):
    add_entity(engine, "e-roof", "Roof")
    with Session(engine) as s:
        s.add(ForceRow(id="f1", name="Reduced heat loss", direction="decrease",
                       affects="[]", confidence=0.8, knowledge_origin="engineering",
                       source_model="test", created_at=NOW))
        s.add(PatternRow(
            id="p1", subject_id="e-roof", name="Warm Roof",
            context="[]", problem="Condensation", solution="Insulation above deck",
            force_descriptions=json.dumps(["Heat loss concern"]),
            force_ids=json.dumps(["f1"]),
            consequences="[]", related_pattern_names="[]",
            related_pattern_ids="[]", emergent_properties="[]",
            confidence=0.85, knowledge_origin="engineering",
            source_model="test", created_at=NOW,
        ))
        s.commit()

    with Session(engine) as s:
        result = get_patterns_tool(s, "Roof")

    assert result["entity"] == "Roof"
    assert len(result["patterns"]) == 1
    pat = result["patterns"][0]
    assert pat["name"] == "Warm Roof"
    assert "Reduced heat loss" in pat["forces"]
    assert "forces_warning" not in pat


def test_get_patterns_falls_back_to_descriptions_when_no_force_ids(engine):
    add_entity(engine, "e-roof", "Roof")
    with Session(engine) as s:
        s.add(PatternRow(
            id="p1", subject_id="e-roof", name="Positive Drainage",
            context="[]", problem="Ponding", solution="Add falls",
            force_descriptions=json.dumps(["Drainage vs retention"]),
            force_ids="[]",
            consequences="[]", related_pattern_names="[]",
            related_pattern_ids="[]", emergent_properties="[]",
            confidence=0.8, knowledge_origin="engineering",
            source_model="test", created_at=NOW,
        ))
        s.commit()

    with Session(engine) as s:
        result = get_patterns_tool(s, "Roof")

    pat = result["patterns"][0]
    assert pat["forces"] == ["Drainage vs retention"]
    assert "forces_warning" in pat


def test_get_patterns_entity_not_found(engine):
    with Session(engine) as s:
        result = get_patterns_tool(s, "NoSuch")
    assert result["error"] == "entity_not_found"


# ---------------------------------------------------------------------------
# get_forces
# ---------------------------------------------------------------------------

def test_get_forces_returns_forces_affecting_entity(engine):
    add_entity(engine, "e-ins", "Insulation")
    with Session(engine) as s:
        s.add(ForceRow(id="f1", name="Reduced heat loss", direction="decrease",
                       affects=json.dumps(["e-ins"]),
                       confidence=0.8, knowledge_origin="engineering",
                       source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_forces_tool(s, "Insulation")

    assert result["entity"] == "Insulation"
    assert len(result["forces"]) == 1
    assert result["forces"][0]["name"] == "Reduced heat loss"
    assert result["forces"][0]["direction"] == "decrease"


def test_get_forces_entity_not_in_affects_not_returned(engine):
    add_entity(engine, "e-ins", "Insulation")
    add_entity(engine, "e-wall", "Wall")
    with Session(engine) as s:
        s.add(ForceRow(id="f1", name="Improved strength", direction="increase",
                       affects=json.dumps(["e-wall"]),
                       confidence=0.8, knowledge_origin="engineering",
                       source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_forces_tool(s, "Insulation")

    assert result["forces"] == []


def test_get_forces_entity_not_found(engine):
    with Session(engine) as s:
        result = get_forces_tool(s, "Ghost")
    assert result["error"] == "entity_not_found"


def test_get_forces_min_confidence_filter(engine):
    add_entity(engine, "e-ins", "Insulation")
    with Session(engine) as s:
        s.add(ForceRow(id="f1", name="Low confidence force", direction="increase",
                       affects=json.dumps(["e-ins"]), confidence=0.2,
                       knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.add(ForceRow(id="f2", name="High confidence force", direction="decrease",
                       affects=json.dumps(["e-ins"]), confidence=0.9,
                       knowledge_origin="engineering", source_model="test", created_at=NOW))
        s.commit()

    with Session(engine) as s:
        result = get_forces_tool(s, "Insulation", min_confidence=0.5)

    assert len(result["forces"]) == 1
    assert result["forces"][0]["name"] == "High confidence force"


# ---------------------------------------------------------------------------
# get_spatial_relations
# ---------------------------------------------------------------------------

def test_get_spatial_relations_as_subject(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")
    with Session(engine) as s:
        s.add(SpatialRelationRow(
            id="sr1", subject_id="e-wall", relation="sits_on", object_id="e-floor",
            confidence=0.9, knowledge_origin="architectural",
            source_model="test", created_at=NOW,
        ))
        s.commit()

    with Session(engine) as s:
        result = get_spatial_relations_tool(s, "Wall")

    assert result["entity"] == "Wall"
    assert len(result["spatial_relations"]) == 1
    sr = result["spatial_relations"][0]
    assert sr["subject"] == "Wall"
    assert sr["relation"] == "sits_on"
    assert sr["object"] == "Floor"


def test_get_spatial_relations_as_object(engine):
    add_entity(engine, "e-wall", "Wall")
    add_entity(engine, "e-floor", "Floor")
    with Session(engine) as s:
        s.add(SpatialRelationRow(
            id="sr1", subject_id="e-wall", relation="sits_on", object_id="e-floor",
            confidence=0.9, knowledge_origin="architectural",
            source_model="test", created_at=NOW,
        ))
        s.commit()

    with Session(engine) as s:
        result = get_spatial_relations_tool(s, "Floor")

    assert len(result["spatial_relations"]) == 1
    assert result["spatial_relations"][0]["subject"] == "Wall"


def test_get_spatial_relations_entity_not_found(engine):
    with Session(engine) as s:
        result = get_spatial_relations_tool(s, "Ghost")
    assert result["error"] == "entity_not_found"


# ---------------------------------------------------------------------------
# get_process_sequence
# ---------------------------------------------------------------------------

def _add_process_relation(engine, rid, pred_id, succ_id):
    with Session(engine) as s:
        s.add(ProcessRelationRow(
            id=rid, predecessor_id=pred_id, successor_id=succ_id,
            hard_constraint=True, rationale="Required ordering",
            confidence=0.9, knowledge_origin="engineering",
            source_model="test", created_at=NOW,
        ))
        s.commit()


def test_get_process_sequence_dag_ordering(engine):
    add_entity(engine, "e-frame", "Framing")
    add_entity(engine, "e-ins", "Insulation")
    add_entity(engine, "e-plaster", "Plastering")

    _add_process_relation(engine, "r1", "e-frame", "e-ins")
    _add_process_relation(engine, "r2", "e-ins", "e-plaster")

    with Session(engine) as s:
        result = get_process_sequence_tool(s, "Insulation")

    assert result["has_cycle"] is False
    assert result["truncated"] is False
    seq = result["sequence"]
    assert seq.index("Framing") < seq.index("Insulation")
    assert seq.index("Insulation") < seq.index("Plastering")


def test_get_process_sequence_entity_not_found(engine):
    with Session(engine) as s:
        result = get_process_sequence_tool(s, "Ghost")
    assert result["error"] == "entity_not_found"


def test_get_process_sequence_entity_not_in_graph(engine):
    add_entity(engine, "e-wall", "Wall")

    with Session(engine) as s:
        result = get_process_sequence_tool(s, "Wall")

    assert result["sequence"] == ["Wall"]
    assert result["has_cycle"] is False


def test_get_process_sequence_cycle_detected(engine):
    add_entity(engine, "e-a", "ActivityA")
    add_entity(engine, "e-b", "ActivityB")
    add_entity(engine, "e-c", "ActivityC")

    _add_process_relation(engine, "r1", "e-a", "e-b")
    _add_process_relation(engine, "r2", "e-b", "e-c")
    _add_process_relation(engine, "r3", "e-c", "e-a")  # creates cycle

    with Session(engine) as s:
        result = get_process_sequence_tool(s, "ActivityA")

    assert result["has_cycle"] is True
    assert "cycle_description" in result
    assert "ActivityA" in result["sequence"] or "ActivityA" in result["cycle_description"]


def test_get_process_sequence_max_depth_truncation(engine):
    # Build a long linear chain and query from the middle
    ids = [f"e-{i}" for i in range(10)]
    names = [f"Activity{i}" for i in range(10)]
    for eid, name in zip(ids, names):
        add_entity(engine, eid, name)
    for i in range(9):
        _add_process_relation(engine, f"r{i}", ids[i], ids[i + 1])

    with Session(engine) as s:
        result = get_process_sequence_tool(s, "Activity5", max_depth=2)

    assert result["truncated"] is True


def test_get_process_sequence_case_insensitive(engine):
    add_entity(engine, "e-frame", "Framing")
    add_entity(engine, "e-ins", "Insulation")
    _add_process_relation(engine, "r1", "e-frame", "e-ins")

    with Session(engine) as s:
        result = get_process_sequence_tool(s, "framing")

    assert result["entity"] == "Framing"
    assert "Framing" in result["sequence"]
