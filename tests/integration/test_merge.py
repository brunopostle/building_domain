"""Integration tests for the shared FK-aware entity merge utility."""
import json
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.merge import merge_entity
from bsos.persistence.models import (
    AntiPatternRow,
    AssertionRow,
    ConstraintRow,
    EntityAliasRow,
    EntityRow,
    ForceRow,
    IFCPropertySetRow,
    PatternRow,
    ProcessRelationRow,
    SpatialRelationRow,
)

NOW = datetime.now(timezone.utc)


@pytest.fixture
def session(tmp_path):
    eng = create_db_engine(str(tmp_path / "test.db"))
    create_views(eng)
    with Session(eng) as s:
        yield s


def _entity(s, eid, name, etype="ifc_class"):
    s.add(EntityRow(id=eid, name=name, entity_type=etype, source_model="test", created_at=NOW))


def _assertion(s, aid, subj, obj, pred="requires", subj_type="ifc_class", obj_type="component"):
    s.add(AssertionRow(
        id=aid, subject_id=subj, predicate=pred, object_id=obj,
        subject_type=subj_type, object_type=obj_type,
        source_model="test", created_at=NOW, confidence=0.9, knowledge_origin="physical",
    ))


# canonical = "C" (schema seed), duplicate = "D" (prose variant)
def _seed_pair(s):
    _entity(s, "C", "IfcWall", "ifc_class")
    _entity(s, "D", "IFC Wall (Gallery Perimeter)", "ifc_class")
    _entity(s, "X", "Mortar", "component")


def test_merge_repoints_assertion_subject_and_object(session):
    _seed_pair(session)
    _assertion(session, "a1", "D", "X")   # subject is dup
    _assertion(session, "a2", "X", "D", obj_type="ifc_class")   # object is dup
    session.commit()

    counts = merge_entity(session, "C", "D")
    session.commit()

    rows = session.exec(select(AssertionRow)).all()
    assert all(r.subject_id != "D" and r.object_id != "D" for r in rows)
    a1 = session.get(AssertionRow, "a1")
    assert a1.subject_id == "C" and a1.subject_type == "ifc_class"
    a2 = session.get(AssertionRow, "a2")
    assert a2.object_id == "C" and a2.object_type == "ifc_class"
    assert counts["repointed"] == 2


def test_merge_dedups_colliding_assertions(session):
    _seed_pair(session)
    _assertion(session, "a1", "C", "X")  # already on canonical
    _assertion(session, "a2", "D", "X")  # collides after repoint
    session.commit()

    counts = merge_entity(session, "C", "D")
    session.commit()

    remaining = session.exec(select(AssertionRow)).all()
    assert len(remaining) == 1
    assert remaining[0].id == "a1"
    assert counts["deleted"] == 1


def test_merge_drops_self_loop_assertion(session):
    _seed_pair(session)
    _assertion(session, "a1", "C", "D", obj_type="ifc_class")  # C requires D -> self-loop
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    assert session.exec(select(AssertionRow)).all() == []


def test_merge_repoints_constraints_patterns_antipatterns(session):
    _seed_pair(session)
    session.add(ConstraintRow(id="c1", subject_id="D", rule="r", constraint_type="t",
                              source_model="test", created_at=NOW, confidence=0.9,
                              knowledge_origin="physical"))
    session.add(PatternRow(id="p1", name="P", subject_id="D", problem="x", solution="y",
                           source_model="test", created_at=NOW, confidence=0.9,
                           knowledge_origin="physical"))
    session.add(AntiPatternRow(id="ap1", name="AP", subject_id="D",
                               source_model="test", created_at=NOW, confidence=0.9,
                               knowledge_origin="physical"))
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    assert session.get(ConstraintRow, "c1").subject_id == "C"
    assert session.get(PatternRow, "p1").subject_id == "C"
    assert session.get(AntiPatternRow, "ap1").subject_id == "C"


def test_merge_repoints_spatial_relations_with_dedup(session):
    _seed_pair(session)
    session.add(SpatialRelationRow(id="s1", subject_id="C", relation="adjacent_to",
                                   object_id="X", source_model="test", created_at=NOW,
                                   confidence=0.9, knowledge_origin="physical"))
    session.add(SpatialRelationRow(id="s2", subject_id="D", relation="adjacent_to",
                                   object_id="X", source_model="test", created_at=NOW,
                                   confidence=0.9, knowledge_origin="physical"))  # collides
    session.add(SpatialRelationRow(id="s3", subject_id="D", relation="connects_to",
                                   object_id="X", source_model="test", created_at=NOW,
                                   confidence=0.9, knowledge_origin="physical"))  # repoints
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    rows = session.exec(select(SpatialRelationRow)).all()
    ids = {r.id for r in rows}
    assert ids == {"s1", "s3"}
    assert all(r.subject_id != "D" for r in rows)


def test_merge_repoints_process_relations_respecting_unique(session):
    _seed_pair(session)
    session.add(ProcessRelationRow(id="pr1", predecessor_id="C", successor_id="X",
                                   hard_constraint=True, source_model="test", created_at=NOW,
                                   confidence=0.9, knowledge_origin="physical", rationale="r"))
    session.add(ProcessRelationRow(id="pr2", predecessor_id="D", successor_id="X",
                                   hard_constraint=True, source_model="test", created_at=NOW,
                                   confidence=0.9, knowledge_origin="physical", rationale="r"))  # collides
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    rows = session.exec(select(ProcessRelationRow)).all()
    assert {r.id for r in rows} == {"pr1"}


def test_merge_repoints_forces_affects(session):
    _seed_pair(session)
    session.add(ForceRow(id="f1", name="Gravity", direction="down",
                         affects=json.dumps(["X", "D"]), source_model="test",
                         created_at=NOW, confidence=0.9, knowledge_origin="physical"))
    # affects already containing canonical -> dedup, not duplicate entry
    session.add(ForceRow(id="f2", name="Wind", direction="lateral",
                         affects=json.dumps(["C", "D"]), source_model="test",
                         created_at=NOW, confidence=0.9, knowledge_origin="physical"))
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    assert json.loads(session.get(ForceRow, "f1").affects) == ["X", "C"]
    assert json.loads(session.get(ForceRow, "f2").affects) == ["C"]


def test_merge_repoints_ifc_psets(session):
    _seed_pair(session)
    session.add(IFCPropertySetRow(id="ip1", entity_id="D", ifc_class="IfcWall",
                                  pset_name="Pset_WallCommon", property_name="IsExternal",
                                  value_type="IfcBoolean", description="d"))
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    assert session.get(IFCPropertySetRow, "ip1").entity_id == "C"


def test_merge_adds_alias_and_marks_merged(session):
    _seed_pair(session)
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    assert session.get(EntityRow, "D").status == "merged"
    aliases = session.exec(select(EntityAliasRow)).all()
    assert any(a.entity_id == "C" and a.alias == "IFC Wall (Gallery Perimeter)" for a in aliases)


def test_merge_repoints_existing_aliases_on_duplicate(session):
    _seed_pair(session)
    session.add(EntityAliasRow(entity_id="D", alias="some old alias"))
    session.commit()

    merge_entity(session, "C", "D")
    session.commit()

    alias = session.exec(
        select(EntityAliasRow).where(EntityAliasRow.alias == "some old alias")
    ).one()
    assert alias.entity_id == "C"


def test_merge_rejects_self_merge(session):
    _seed_pair(session)
    session.commit()
    with pytest.raises(ValueError):
        merge_entity(session, "C", "C")


def test_merge_rejects_missing_entity(session):
    _seed_pair(session)
    session.commit()
    with pytest.raises(ValueError):
        merge_entity(session, "C", "nonexistent")


def test_merge_preserves_types_when_update_types_false(session):
    _seed_pair(session)
    _assertion(session, "a1", "D", "X", subj_type="space")
    session.commit()

    merge_entity(session, "C", "D", update_types=False)
    session.commit()

    assert session.get(AssertionRow, "a1").subject_type == "space"
