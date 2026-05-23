"""Auto-promotion logic for the single-model and multi-model paths (Sections 7, 7.1).

Prerequisite before calling run_auto_promotion(): run_conflict_detection() must have stamped
conflict_evaluated_at on all candidate items. Items without that stamp are skipped.

Single-model path (Section 7.1):
  - AssertionRow only (only type with cross_prompt_consistency / prompt_framing_count)
  - physical/engineering: cpc >= 0.85 across >= 3 framings → auto-accept
  - cultural/architectural: skip (require human acceptance)

Multi-model path (Section 7):
  - All promotable types, grouped by natural key
  - physical/engineering: >= 2 distinct models agree + mean confidence >= 0.80 → auto-accept
  - cultural/architectural: >= 3 distinct models agree + mean confidence >= 0.90 → auto-accept

Config key auto_promote_enabled='0' disables all promotion.
"""
import uuid
from datetime import datetime, timezone
from statistics import mean
from typing import Callable

import structlog
from sqlmodel import Session, select

from bsos.config import get_config
from bsos.persistence.models import (
    AntiPatternRow,
    AssertionRow,
    ConstraintRow,
    ForceRow,
    PatternRow,
    ProvenanceLogRow,
    ReviewDecisionRow,
    SpatialRelationRow,
)

log = structlog.get_logger()

PHYSICAL_ENGINEERING = frozenset({"physical", "engineering"})

# Single-model thresholds (Section 7.1)
SINGLE_CPC_THRESHOLD = 0.85
SINGLE_MIN_FRAMINGS = 3

# Multi-model thresholds (Section 7)
MULTI_PE_MIN_MODELS = 2
MULTI_PE_MIN_CONFIDENCE = 0.80
MULTI_CA_MIN_MODELS = 3
MULTI_CA_MIN_CONFIDENCE = 0.90

# (model_class, natural-key lambda, item_type label)
_MULTI_MODEL_TARGETS: list[tuple] = [
    (AssertionRow, lambda r: (r.subject_id, r.predicate, r.object_id), "assertion"),
    (ConstraintRow, lambda r: (r.subject_id, r.rule, r.constraint_type), "constraint"),
    (PatternRow, lambda r: (r.name,), "pattern"),
    (ForceRow, lambda r: (r.name,), "force"),
    (AntiPatternRow, lambda r: (r.name, r.subject_id or ""), "antipattern"),
    (SpatialRelationRow, lambda r: (r.subject_id, r.relation, r.object_id), "spatial_relation"),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_decision(
    session: Session, item_id: str, item_type: str, reviewer: str, rationale: str
) -> None:
    session.add(ReviewDecisionRow(
        id=str(uuid.uuid4()),
        item_id=item_id,
        item_type=item_type,
        decision="accept",
        mapped_to=None,
        rationale=rationale,
        reviewer=reviewer,
        created_at=_now(),
    ))


def _write_provenance(
    session: Session, item_id: str, item_type: str, old_status: str, changed_by: str
) -> None:
    session.add(ProvenanceLogRow(
        id=str(uuid.uuid4()),
        item_id=item_id,
        item_type=item_type,
        old_status=old_status,
        new_status="accepted",
        changed_at=_now(),
        changed_by=changed_by,
    ))


def _accept(
    session: Session, row, item_type: str, reviewer: str, rationale: str
) -> None:
    old = row.status
    row.status = "accepted"
    _write_decision(session, row.id, item_type, reviewer, rationale)
    _write_provenance(session, row.id, item_type, old, reviewer)


def _run_single_model_promotion(session: Session) -> int:
    """Promote AssertionRows with sufficient cross-prompt consistency (Section 7.1)."""
    promoted = 0
    candidates = session.exec(
        select(AssertionRow).where(
            AssertionRow.status == "proposed",
            AssertionRow.conflict_evaluated_at.isnot(None),  # type: ignore[attr-defined]
        )
    ).all()

    for row in candidates:
        if row.knowledge_origin not in PHYSICAL_ENGINEERING:
            continue
        if row.cross_prompt_consistency is None or row.prompt_framing_count is None:
            continue
        if row.cross_prompt_consistency < SINGLE_CPC_THRESHOLD:
            continue
        if row.prompt_framing_count < SINGLE_MIN_FRAMINGS:
            continue
        _accept(
            session, row, "assertion", row.source_model,
            f"single-model: cpc={row.cross_prompt_consistency:.3f} framings={row.prompt_framing_count}",
        )
        promoted += 1
        log.info(
            "auto_promoted_single",
            item_id=row.id,
            cpc=row.cross_prompt_consistency,
            framings=row.prompt_framing_count,
        )

    return promoted


def _run_multi_model_promotion(
    session: Session,
    model_class,
    key_fn: Callable,
    item_type: str,
) -> int:
    """Promote items where >= N distinct source_models agree on the same natural key (Section 7)."""
    promoted = 0

    candidates = session.exec(
        select(model_class).where(  # type: ignore[call-overload]
            model_class.status == "proposed",  # type: ignore[attr-defined]
            model_class.conflict_evaluated_at.isnot(None),  # type: ignore[attr-defined]
        )
    ).all()

    groups: dict[tuple, list] = {}
    for row in candidates:
        groups.setdefault(key_fn(row), []).append(row)

    for _key, rows in groups.items():
        models = {r.source_model for r in rows}
        if len(models) < 2:
            continue

        is_physical = any(r.knowledge_origin in PHYSICAL_ENGINEERING for r in rows)
        avg_conf = mean(r.confidence for r in rows)

        if is_physical:
            if len(models) < MULTI_PE_MIN_MODELS or avg_conf < MULTI_PE_MIN_CONFIDENCE:
                continue
        else:
            if len(models) < MULTI_CA_MIN_MODELS or avg_conf < MULTI_CA_MIN_CONFIDENCE:
                continue

        rationale = f"multi-model: {len(models)} models agree, mean_confidence={avg_conf:.3f}"
        for row in rows:
            _accept(session, row, item_type, "consensus", rationale)
            promoted += 1
        log.info(
            "auto_promoted_multi",
            item_type=item_type,
            count=len(rows),
            models=sorted(models),
            avg_confidence=avg_conf,
        )

    return promoted


def run_auto_promotion(engine) -> dict:
    """Run auto-promotion for single-model and multi-model paths.

    Call after run_conflict_detection() so conflict_evaluated_at is populated.
    Returns a result dict with promoted counts.
    """
    with Session(engine) as session:
        if get_config(session, "auto_promote_enabled") == "0":
            log.info("auto_promotion_skipped", reason="auto_promote_enabled=0")
            return {
                "skipped": True,
                "single_model_promoted": 0,
                "multi_model_promoted": 0,
                "total_promoted": 0,
            }

    with Session(engine) as session:
        single_promoted = _run_single_model_promotion(session)
        session.commit()

    multi_promoted = 0
    for model_class, key_fn, type_label in _MULTI_MODEL_TARGETS:
        with Session(engine) as session:
            n = _run_multi_model_promotion(session, model_class, key_fn, type_label)
            session.commit()
        multi_promoted += n

    result = {
        "skipped": False,
        "single_model_promoted": single_promoted,
        "multi_model_promoted": multi_promoted,
        "total_promoted": single_promoted + multi_promoted,
    }
    log.info("auto_promotion_complete", **result)
    return result
