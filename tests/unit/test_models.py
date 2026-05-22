from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from bsos.models import (
    Entity, Assertion, Pattern, Force, ForceDirection, AntiPattern,
    ProcessRelation, SpatialRelation, AbstractionNode, ReviewDecision,
    Constraint, PendingPredicate,
)

NOW = datetime.now(timezone.utc)
BASE_PROV = dict(source_model="test-model", source_prompt="test prompt", created_at=NOW)
PROV = dict(**BASE_PROV, confidence=0.9, knowledge_origin="physical")


def test_entity_basic():
    e = Entity(id="abc", name="roof", entity_type="component", **BASE_PROV)
    assert e.status == "proposed"
    assert e.is_entrance is False


def test_entity_no_conflicted_status():
    with pytest.raises(ValidationError):
        Entity(id="abc", name="roof", entity_type="component", status="conflicted", **BASE_PROV)


def test_assertion_basic():
    a = Assertion(
        id="a1", subject_id="e1", predicate="requires", object_id="e2",
        subject_type="component", object_type="component", **PROV,
    )
    assert a.conditions == []
    assert a.cross_prompt_consistency is None


def test_assertion_confidence_bounds():
    with pytest.raises(ValidationError):
        Assertion(
            id="a1", subject_id="e1", predicate="requires", object_id="e2",
            subject_type="component", object_type="component",
            confidence=1.5, **{k: v for k, v in PROV.items() if k != "confidence"},
        )


def test_process_relation_requires_rationale():
    with pytest.raises(ValidationError):
        ProcessRelation(
            id="p1", predecessor_id="e1", successor_id="e2",
            hard_constraint=True, rationale=None, **PROV,
        )


def test_process_relation_with_rationale():
    pr = ProcessRelation(
        id="p1", predecessor_id="e1", successor_id="e2",
        hard_constraint=True,
        rationale="waterproofing must precede finishes to prevent moisture damage",
        **PROV,
    )
    assert pr.hard_constraint is True


def test_force_direction_enum():
    f = Force(id="f1", name="increased daylight", direction=ForceDirection.increase,
              affects=["e1"], **PROV)
    assert f.direction == ForceDirection.increase


def test_abstraction_node_rejects_knowledge_origin():
    with pytest.raises(ValidationError):
        AbstractionNode(
            id="an1", statement="roofs protect buildings",
            child_ids=["a1", "a2"], abstraction_rationale="common theme",
            knowledge_origin="physical", **BASE_PROV,
            confidence=0.9, status="proposed",
            conflict_evaluated_at=None,
        )


def test_abstraction_node_without_knowledge_origin():
    an = AbstractionNode(
        id="an1", statement="roofs protect buildings",
        child_ids=["a1", "a2"], abstraction_rationale="common theme",
        **BASE_PROV, confidence=0.9, status="proposed",
    )
    assert an.knowledge_origin is None


def test_pending_predicate():
    pp = PendingPredicate(
        id="pp1", value="resists", vocabulary_type="predicate",
        first_seen_at=NOW, last_seen_at=NOW,
    )
    assert pp.occurrence_count == 1
    assert pp.flagged_for_review is False
