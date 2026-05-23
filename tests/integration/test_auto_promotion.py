"""Integration tests for auto-promotion logic (Sections 7, 7.1)."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, select

from bsos.normalization.auto_promotion import (
    MULTI_CA_MIN_CONFIDENCE,
    MULTI_CA_MIN_MODELS,
    MULTI_PE_MIN_CONFIDENCE,
    MULTI_PE_MIN_MODELS,
    SINGLE_CPC_THRESHOLD,
    SINGLE_MIN_FRAMINGS,
    run_auto_promotion,
)
from bsos.persistence.database import create_db_engine, create_views
from bsos.persistence.models import (
    AssertionRow,
    ConstraintRow,
    EntityRow,
    ProvenanceLogRow,
    ReviewDecisionRow,
)

NOW = datetime.now(timezone.utc)


@pytest.fixture
def engine(tmp_path):
    db_path = tmp_path / "test.db"
    eng = create_db_engine(str(db_path))
    create_views(eng)
    return eng


def _entity(session: Session, name: str = "room") -> EntityRow:
    row = EntityRow(
        id=str(uuid.uuid4()),
        name=name,
        entity_type="space",
        status="accepted",
        source_model="test",
        created_at=NOW,
    )
    session.add(row)
    return row


def _assertion(
    session: Session,
    subject_id: str,
    object_id: str,
    predicate: str = "has_property",
    knowledge_origin: str = "physical",
    source_model: str = "model-a",
    confidence: float = 0.9,
    status: str = "proposed",
    conflict_evaluated_at=NOW,
    cross_prompt_consistency: float | None = None,
    prompt_framing_count: int | None = None,
) -> AssertionRow:
    row = AssertionRow(
        id=str(uuid.uuid4()),
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        subject_type="space",
        object_type="space",
        source_model=source_model,
        created_at=NOW,
        confidence=confidence,
        status=status,
        knowledge_origin=knowledge_origin,
        conflict_evaluated_at=conflict_evaluated_at,
        cross_prompt_consistency=cross_prompt_consistency,
        prompt_framing_count=prompt_framing_count,
    )
    session.add(row)
    return row


def _constraint(
    session: Session,
    subject_id: str,
    rule: str = "must have window",
    constraint_type: str = "spatial",
    knowledge_origin: str = "physical",
    source_model: str = "model-a",
    confidence: float = 0.85,
    status: str = "proposed",
    conflict_evaluated_at=NOW,
) -> ConstraintRow:
    row = ConstraintRow(
        id=str(uuid.uuid4()),
        subject_id=subject_id,
        rule=rule,
        constraint_type=constraint_type,
        source_model=source_model,
        created_at=NOW,
        confidence=confidence,
        status=status,
        knowledge_origin=knowledge_origin,
        conflict_evaluated_at=conflict_evaluated_at,
    )
    session.add(row)
    return row


# ---------------------------------------------------------------------------
# Single-model path
# ---------------------------------------------------------------------------

class TestSingleModelPromotion:
    def test_promotes_physical_assertion_with_sufficient_cpc(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            a = _assertion(
                session, e.id, e.id,
                knowledge_origin="physical",
                cross_prompt_consistency=SINGLE_CPC_THRESHOLD,
                prompt_framing_count=SINGLE_MIN_FRAMINGS,
            )
            session.commit()
            a_id = a.id
            source_model = a.source_model

        result = run_auto_promotion(engine)

        assert result["single_model_promoted"] == 1
        assert result["total_promoted"] >= 1
        with Session(engine) as session:
            row = session.get(AssertionRow, a_id)
            assert row.status == "accepted"
            decision = session.exec(
                select(ReviewDecisionRow).where(ReviewDecisionRow.item_id == a_id)
            ).first()
            assert decision is not None
            assert decision.decision == "accept"
            assert decision.reviewer == source_model
            prov = session.exec(
                select(ProvenanceLogRow).where(ProvenanceLogRow.item_id == a_id)
            ).first()
            assert prov is not None
            assert prov.new_status == "accepted"
            assert prov.old_status == "proposed"

    def test_does_not_promote_below_cpc_threshold(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            a = _assertion(
                session, e.id, e.id,
                knowledge_origin="physical",
                cross_prompt_consistency=SINGLE_CPC_THRESHOLD - 0.01,
                prompt_framing_count=SINGLE_MIN_FRAMINGS,
            )
            session.commit()
            a_id = a.id

        run_auto_promotion(engine)

        with Session(engine) as session:
            row = session.get(AssertionRow, a_id)
            assert row.status == "proposed"

    def test_does_not_promote_below_framing_count(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            a = _assertion(
                session, e.id, e.id,
                knowledge_origin="engineering",
                cross_prompt_consistency=SINGLE_CPC_THRESHOLD,
                prompt_framing_count=SINGLE_MIN_FRAMINGS - 1,
            )
            session.commit()
            a_id = a.id

        run_auto_promotion(engine)

        with Session(engine) as session:
            row = session.get(AssertionRow, a_id)
            assert row.status == "proposed"

    def test_does_not_promote_cultural_assertion(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            _assertion(
                session, e.id, e.id,
                knowledge_origin="cultural",
                cross_prompt_consistency=0.99,
                prompt_framing_count=10,
            )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["single_model_promoted"] == 0

    def test_does_not_promote_without_conflict_evaluated_at(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            a = _assertion(
                session, e.id, e.id,
                knowledge_origin="physical",
                cross_prompt_consistency=0.95,
                prompt_framing_count=5,
                conflict_evaluated_at=None,
            )
            session.commit()
            a_id = a.id

        run_auto_promotion(engine)

        with Session(engine) as session:
            row = session.get(AssertionRow, a_id)
            assert row.status == "proposed"

    def test_does_not_promote_already_conflicted(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            a = _assertion(
                session, e.id, e.id,
                knowledge_origin="physical",
                cross_prompt_consistency=0.95,
                prompt_framing_count=5,
                status="conflicted",
            )
            session.commit()
            a_id = a.id

        run_auto_promotion(engine)

        with Session(engine) as session:
            row = session.get(AssertionRow, a_id)
            assert row.status == "conflicted"

    def test_promotes_engineering_origin(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            _assertion(
                session, e.id, e.id,
                knowledge_origin="engineering",
                cross_prompt_consistency=SINGLE_CPC_THRESHOLD,
                prompt_framing_count=SINGLE_MIN_FRAMINGS,
            )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["single_model_promoted"] == 1


# ---------------------------------------------------------------------------
# Multi-model path
# ---------------------------------------------------------------------------

class TestMultiModelPromotion:
    def test_promotes_physical_assertion_two_models(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            a1 = _assertion(session, e.id, e.id, source_model="model-a", confidence=0.85)
            a2 = _assertion(session, e.id, e.id, source_model="model-b", confidence=0.85)
            session.commit()
            a1_id, a2_id = a1.id, a2.id

        result = run_auto_promotion(engine)
        assert result["multi_model_promoted"] == 2

        with Session(engine) as session:
            for aid in [a1_id, a2_id]:
                row = session.get(AssertionRow, aid)
                assert row.status == "accepted"
                decision = session.exec(
                    select(ReviewDecisionRow).where(ReviewDecisionRow.item_id == aid)
                ).first()
                assert decision.reviewer == "consensus"

    def test_does_not_promote_single_model_multi_item(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            _assertion(session, e.id, e.id, source_model="model-a", confidence=0.85)
            _assertion(session, e.id, e.id, source_model="model-a", confidence=0.85)
            session.commit()

        result = run_auto_promotion(engine)
        assert result["multi_model_promoted"] == 0

    def test_does_not_promote_physical_below_confidence(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            _assertion(
                session, e.id, e.id, source_model="model-a",
                confidence=MULTI_PE_MIN_CONFIDENCE - 0.05,
            )
            _assertion(
                session, e.id, e.id, source_model="model-b",
                confidence=MULTI_PE_MIN_CONFIDENCE - 0.05,
            )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["multi_model_promoted"] == 0

    def test_does_not_promote_cultural_with_only_two_models(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            _assertion(
                session, e.id, e.id, source_model="model-a",
                knowledge_origin="cultural", confidence=0.95,
            )
            _assertion(
                session, e.id, e.id, source_model="model-b",
                knowledge_origin="cultural", confidence=0.95,
            )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["multi_model_promoted"] == 0

    def test_promotes_cultural_with_three_models_high_confidence(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            for model in ("model-a", "model-b", "model-c"):
                _assertion(
                    session, e.id, e.id, source_model=model,
                    knowledge_origin="cultural",
                    confidence=MULTI_CA_MIN_CONFIDENCE,
                )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["multi_model_promoted"] == 3

    def test_promotes_multi_model_constraint(self, engine):
        with Session(engine) as session:
            e = _entity(session)
            session.flush()
            c1 = _constraint(session, e.id, source_model="model-a", confidence=0.85)
            c2 = _constraint(session, e.id, source_model="model-b", confidence=0.85)
            session.commit()
            c1_id, c2_id = c1.id, c2.id

        result = run_auto_promotion(engine)
        assert result["multi_model_promoted"] == 2

        with Session(engine) as session:
            for cid in [c1_id, c2_id]:
                row = session.get(ConstraintRow, cid)
                assert row.status == "accepted"


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------

class TestAutoPromoteConfig:
    def test_skips_when_disabled(self, engine):
        from bsos.config import set_config
        with Session(engine) as session:
            set_config(session, "auto_promote_enabled", "0")
            e = _entity(session)
            session.flush()
            _assertion(
                session, e.id, e.id,
                knowledge_origin="physical",
                cross_prompt_consistency=0.99,
                prompt_framing_count=10,
            )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["skipped"] is True
        assert result["total_promoted"] == 0

    def test_runs_when_enabled_explicitly(self, engine):
        from bsos.config import set_config
        with Session(engine) as session:
            set_config(session, "auto_promote_enabled", "1")
            e = _entity(session)
            session.flush()
            _assertion(
                session, e.id, e.id,
                knowledge_origin="physical",
                cross_prompt_consistency=SINGLE_CPC_THRESHOLD,
                prompt_framing_count=SINGLE_MIN_FRAMINGS,
            )
            session.commit()

        result = run_auto_promotion(engine)
        assert result["skipped"] is False
        assert result["single_model_promoted"] == 1
